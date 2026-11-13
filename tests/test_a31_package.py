"""A31's RunPod package (tools/colab_chat/a31_jobs.py, a31_select.py, a31_gate.py; lane PROTOCOL.md section 50). Claude Opus 5.5,
13 Nov 2026. No GPU, no network: the jobs file against its rendering and S71's spec, the selection rule's table, the gate's
comparison, and jobs_vm.py's promotion reading the decisions the rule writes."""
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import a31_gate as G  # noqa: E402
import a31_jobs as J  # noqa: E402
import a31_select as S  # noqa: E402


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def test_jobs_file_is_its_rendering():
    assert (ROOT / "tools" / "colab_chat" / "a31_cue.json").read_text() == json.dumps(J.render(), indent=1) + "\n"


def test_specs_differ_from_s71_only_where_registered():
    base = flat(J.S71)
    allowed = {"train.seed", "data.twin_examples", "train.twin_loss.groups", "train.twin_loss.margin", "train.twin_loss.lambda",
               "train.twin_loss.mu", "export.formats"}
    names = set()
    for j in J.render()["jobs"]:
        if "spec" not in j:
            continue
        names.add(j["name"])
        f = flat(j["spec"])
        changed = {k for k in set(base) | set(f) if base.get(k) != f.get(k)}
        assert changed <= allowed, (j["name"], changed - allowed)
        seed = f["train.seed"]
        assert seed == (J.DEV_SEED if j["name"].startswith("D") else int(j["name"].split("_")[1]))
        tl = j["spec"]["train"].get("twin_loss")
        kind = j["name"].split("_")[0]
        if kind in ("D0", "P"):
            assert tl is None
        else:
            assert tl["margin"] == J.MARGIN and tl["groups"] == J.GROUPS and tl["lambda"] in J.GRID
            assert tl["mu"] == (0.0 if kind in ("Dm", "M") else tl["lambda"])
            assert (j.get("promoted_by") is not None) == (kind in ("M", "N"))
        assert j["after"] in ("gate", ["gate", "select"])
    # every arm at both seeds, every candidate
    for sd in J.SEEDS:
        assert f"P_{sd}" in names
        for x in J.GRID:
            assert {f"M_{sd}_{J.lname(x)}", f"N_{sd}_{J.lname(x)}"} <= names
    assert J.SEEDS == (1, 50) and abs(J.SEEDS[1] - J.SEEDS[0]) > 2


def test_every_model_is_scored():
    jobs = {j["name"]: j for j in J.render()["jobs"]}
    for n, j in jobs.items():
        if "spec" in j:
            assert jobs[f"score_{n}"]["after"] == n and jobs[f"score_{n}"]["light"]


def table(vals, both, sel):
    val = {"D0": 4.0}
    dev = {"D0": {"both_tok": 0.1, "sel": 0.0, "n": 10}}
    for i, x in enumerate(S.GRID):
        for p in ("Dm", "Dn"):
            val[f"{p}_{S.lname(x)}"] = vals[i]
            dev[f"{p}_{S.lname(x)}"] = {"both_tok": both[i], "sel": sel[i], "n": 10}
    return val, dev


@pytest.mark.parametrize("vals,both,sel,m,n", [
    ((4.0, 4.05, 4.07), (0.2, 0.3, 0.4), (0.1, 0.3, 0.2), 1.0, 0.3),    # all within 1.02 x 4.0
    ((4.0, 4.05, 4.2), (0.2, 0.3, 0.4), (0.1, 0.3, 0.5), 0.3, 0.3),     # 1.0 ineligible
    ((4.1, 4.2, 4.3), (0.5, 0.3, 0.4), (0.5, 0.3, 0.4), 0.1, 0.1),      # none eligible: the smallest
    ((4.0, 4.0, 4.0), (0.3, 0.3, 0.2), (0.2, 0.2, 0.1), 0.1, 0.1),      # ties to the smaller
])
def test_selection_rule(vals, both, sel, m, n):
    val, dev = table(vals, both, sel)
    out = S.decide(val, dev)
    assert out["arms"]["M"]["chosen"] == m and out["arms"]["N"]["chosen"] == n
    d = out["decisions"]["a31"]
    assert sum(d.values()) == 2 and d[f"M_{S.lname(m)}"] and d[f"N_{S.lname(n)}"]


def test_promotion_reads_the_decisions(tmp_path, monkeypatch):
    val, dev = table((4.0, 4.0, 4.0), (0.1, 0.2, 0.3), (0.3, 0.2, 0.1))
    (tmp_path / "decisions.json").write_text(json.dumps(S.decide(val, dev)))
    monkeypatch.setenv("JOBS_CHAT_DIR", str(tmp_path))
    import importlib
    import jobs_vm
    importlib.reload(jobs_vm)
    jobs = J.render()["jobs"]
    got = {}
    for j in jobs:
        if j.get("promoted_by"):
            j = dict(j, promoted_by=dict(j["promoted_by"], files=[str(tmp_path / "decisions.json")]))
            got[j["name"]] = jobs_vm.promotion(j)
    assert got[f"M_1_{S.lname(1.0)}"] is True and got[f"M_50_{S.lname(1.0)}"] is True
    assert got[f"N_1_{S.lname(0.1)}"] is True and got[f"N_50_{S.lname(0.1)}"] is True
    assert sum(got.values()) == 4 and len(got) == 12


def ev(loss, tl, da, db, dn):
    return {"step": 1, "loss": loss, "twin": {"loss": tl, "d_a": da, "d_b": db, "d_n": dn}}


def test_gate_comparison():
    ct = ev(2.5, 0.8, 0.30, 0.20, 0.10)
    assert G.compare(ct, ev(2.505, 0.801, 0.31, 0.21, 0.11), dict(ct)) == []
    assert G.compare(ct, ev(2.6, 0.8, 0.3, 0.2, 0.1), dict(ct))                    # SFT loss off by 4%
    assert G.compare(ct, ev(2.5, 0.9, 0.3, 0.2, 0.1), dict(ct))                    # twin loss off
    assert G.compare(ct, ev(2.5, 0.8, 0.4, 0.2, 0.1), dict(ct))                    # d_a off
    assert G.compare(ct, ev(2.5, 0.8, 0.3, 0.2, 0.1), dict(ct, loss=2.5000002))    # SFT batch changed by the twin loss
    assert G.compare(None, ct, ct) == ["a run failed"]


def test_gate_spec_is_a_one_step_twin_run():
    g = J.gate_spec()
    assert g["train"]["steps"] == 1 and g["train"]["batch"] == 2 and g["train"]["twin_loss"]["groups"] == 2
    assert g["train"]["twin_loss"]["mu"] > 0 and g["data"]["twin_examples"].endswith("train_twins.jsonl")


import a31_early as E  # noqa: E402


def test_every_training_job_has_its_early_readouts():
    jobs = {j["name"]: j for j in J.render()["jobs"]}
    for n, j in jobs.items():
        if "spec" not in j:
            continue
        assert j["snapshot_steps"] == [128, 256, 512]
        assert all(s % j["spec"]["train"]["checkpoint_every"] == 0 for s in j["snapshot_steps"])
        for s in J.EARLY:
            e = jobs[f"early_{n}_{s}"]
            assert e["rung_of"] == {"job": n, "step": s} and e["light"] and "early.json" in e["outputs"]
            ref = e["cmd"][e["cmd"].index("--ref") + 1]
            assert ref == ("D0" if n.startswith("D") else "P_" + n.split("_")[1])
            # the early requests only: never the final test sets
            assert "a31_readout_req.jsonl" not in " ".join(e["cmd"])


def step_ev(step, loss, twin=None):
    e = {"event": "step", "step": step, "loss": loss}
    if twin:
        e["twin"] = twin
    return e


def test_health_rule():
    tw = {"loss": 0.5, "d_a": 0.1, "d_b": 0.2, "d_n": 0.05, "active": 0.9}
    ev = [step_ev(s, 2.0, tw) for s in range(1, 257)] + [{"event": "eval", "step": 256, "val_loss": 4.5}]
    ref = [{"event": "eval", "step": 256, "val_loss": 4.0}]
    assert E.health(ev, ref, 256)[0]                                   # 12.5% above
    assert not E.health(ev, [{"event": "eval", "step": 256, "val_loss": 4.2}], 256)[0]   # 7% above
    assert not E.health(ev, ref, 128)[0]                               # the loss rule reads the 25% cut-off only
    bad = ev[:10] + [step_ev(11, float("nan"), tw)]
    assert E.health(bad, ref, 128) == (True, "non-finite loss")
    tm = E.train_metrics(ev, 256)
    assert tm["step"] == 256 and abs(tm["twin_active_mean32"] - 0.9) < 1e-9


def test_early_measures():
    def r(lp, n):
        return {"sum": lp, "logprobs": [lp / n] * n}
    recs = [r(-2, 2), r(-6, 2), r(-4, 4), r(-12, 4), r(-3, 2)] * 2
    m = E.measures(recs, 2)
    assert m["n"] == 2 and m["both_tok"] == 1.0 and abs(m["nui"] - 0.5) < 1e-9 and abs(m["sel"] - (m["dec"] - 0.5)) < 1e-9
    line = E.summary_line("M_1_0p3", 256, {"early": {"dev": m}, "train": {"sft_loss_mean32": 2.0}, "val_loss": 4.1})
    assert line.startswith("M_1_0p3 step 256; dev: both/tok 1.000")
