"""A34's package (the lane's cue_selective/a34/REGISTRATION.md): the jobs file, the specs against A31's, the floor rule, the
reference builder and the gate's floor checks. Claude Opus 5.5, 15 Nov 2026. CPU only, seconds."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import a31_jobs as A31  # noqa: E402
import a34_gate as GATE  # noqa: E402
import a34_jobs as J  # noqa: E402
import a34_ref as REF  # noqa: E402
import a34_select as SEL  # noqa: E402


def jobs():
    return {j["name"]: j for j in J.render()["jobs"]}


def test_jobs_file_is_its_rendering():
    assert json.loads((ROOT / "tools" / "colab_chat" / "a34_floor.json").read_text()) == J.render()
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "a34_jobs.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_reruns_are_a31s_specs_exactly():
    js = jobs()
    a31 = {j["name"]: j for j in A31.render()["jobs"]}
    assert js["D0r"]["spec"] == a31["D0"]["spec"]
    assert js["P_1r"]["spec"] == a31["P_1"]["spec"] and js["P_50r"]["spec"] == a31["P_50"]["spec"]
    assert js["M_1r"]["spec"] == a31["M_1_0p3"]["spec"]


def test_floor_specs_add_only_the_floor():
    js = jobs()
    a31 = {j["name"]: j for j in A31.render()["jobs"]}
    for f in J.FLOOR_GRID:
        for sd in J.SEEDS:
            for arm, base in (("F", f"M_{sd}_0p3"), ("NF", f"N_{sd}_0p3")):
                s = json.loads(json.dumps(js[f"{arm}_{sd}_{J.lname(f)}"]["spec"]))
                assert s["train"]["twin_loss"].pop("floor") == f
                assert s["data"].pop("twin_floor_ref") == f"{J.C}/ref_P_{sd}r/floor_ref.jsonl"
                assert s == a31[base]["spec"], (arm, sd, f)
        s = json.loads(json.dumps(js[f"Fd_{J.lname(f)}"]["spec"]))
        assert s["train"]["twin_loss"].pop("floor") == f and s["data"].pop("twin_floor_ref") == f"{J.C}/ref_D0r/floor_ref.jsonl"
        assert s == a31["Dm_0p3"]["spec"]


def test_order_dependencies_and_promotion():
    js = jobs()
    train = [n for n, j in js.items() if "spec" in j]
    assert train[:7] == ["D0r", "P_1r", "Fd_0p3", "Fd_1p0", "Fd_3p0", "P_50r", "M_1r"]
    for r in ("D0r", "P_1r", "P_50r"):
        assert set(J.REF_STEPS) | set(J.EARLY) <= set(js[r]["snapshot_steps"])
        assert js[f"ref_{r}"]["after"] == [r, "ref_start"] and js[f"ref_{r}"]["light"]
    for f in J.FLOOR_GRID:
        assert js[f"Fd_{J.lname(f)}"]["after"] == ["gate", "ref_D0r"]
        for sd in J.SEEDS:
            for arm in ("F", "NF"):
                j = js[f"{arm}_{sd}_{J.lname(f)}"]
                assert j["after"] == ["gate", "select", f"ref_P_{sd}r"]
                assert j["promoted_by"]["key"] == f"{arm}_{J.lname(f)}" and j["promoted_by"]["round"] == "a34"
                assert js[f"score_{arm}_{sd}_{J.lname(f)}"]["after"] == [f"{arm}_{sd}_{J.lname(f)}", "select"]
    sel = js["select"]
    assert sel["after_terminal"] and {"D0r", "score_D0r", "ref_D0r"} <= set(sel["after"])
    assert {f"early_Fd_{J.lname(f)}_{s}" for f in J.FLOOR_GRID for s in J.EARLY} <= set(sel["after"])


def test_no_sweep_or_reference_job_reads_the_final_requests():
    js = jobs()
    for n, j in js.items():
        if n.startswith(("Fd_", "D0r", "early_", "ref_", "score_Fd_", "select", "gate")) or n == "score_D0r":
            assert J.FINAL_REQ not in json.dumps(j), n
    for n in ("P_1r", "P_50r", "M_1r"):
        assert js[f"score_{n}"]["after"] == [n, "select"]


def test_early_health_references():
    js = jobs()
    ref = {n: j["cmd"][j["cmd"].index("--ref") + 1] for n, j in js.items() if n.startswith("early_")}
    assert ref["early_D0r_256"] == "D0r" and ref["early_Fd_1p0_256"] == "D0r" and ref["early_M_1r_256"] == "P_1r"
    assert ref["early_F_50_1p0_256"] == "P_50r" and ref["early_NF_1_3p0_256"] == "P_1r" and ref["early_P_50r_256"] == "P_50r"


def test_gate_spec():
    g = J.gate_spec()
    assert g["train"]["twin_loss"]["floor"] == 1.0 and g["data"]["twin_floor_ref"].endswith("a34_gate_ref.jsonl")
    s = json.loads(json.dumps(g))
    s["train"]["twin_loss"].pop("floor")
    s["data"].pop("twin_floor_ref")
    s["name"] = "a31_gate"
    assert s == A31.gate_spec()


def rec(complete=True, val=3.0, own=-1.3, dec=0.4, parity=1e-4, n=10):
    r = {"complete": complete, "val_loss": val, "terms": {f"t{i}": (own, dec) for i in range(n)}}
    if parity is not None:
        r["parity_step1"] = parity
    if not complete:
        r = {"complete": False, "why": "x"}
    return r


def runs(**fd):
    out = {"D0r": rec()}
    for f in SEL.FLOOR_GRID:
        out[f"Fd_{SEL.lname(f)}"] = fd.get(SEL.lname(f), rec(own=-1.3, dec=1.4))
    return out


def test_rule_smallest_with_hold_and_retain():
    d = SEL.decide(runs(**{"0p3": rec(own=-1.6, dec=1.4), "1p0": rec(own=-1.32, dec=1.2), "3p0": rec(own=-1.3, dec=1.0)}))
    assert d["chosen"] == 1.0 and d["flags"] == [] and d["decisions"]["a34"]["F_1p0"] and d["decisions"]["a34"]["NF_1p0"]
    assert sum(d["decisions"]["a34"].values()) == 2


def test_rule_fallbacks():
    d = SEL.decide(runs(**{"0p3": rec(own=-1.6, dec=1.4), "1p0": rec(own=-1.3, dec=0.5), "3p0": rec(own=-1.3, dec=0.6)}))
    assert d["chosen"] == 1.0 and d["flags"] == ["retain_not_met"]
    d = SEL.decide(runs(**{"0p3": rec(own=-1.6), "1p0": rec(own=-1.5), "3p0": rec(own=-1.4)}))
    assert d["chosen"] == 3.0 and d["flags"] == ["hold_not_met"]
    d = SEL.decide(runs(**{"0p3": rec(val=3.1), "1p0": rec(complete=False), "3p0": rec(val=3.07)}))
    assert d["chosen"] is None and not any(d["decisions"]["a34"].values()) and d["flags"] == ["no_eligible_floor"]
    bad = runs()
    bad["D0r"] = rec(complete=False)
    assert SEL.decide(bad)["chosen"] is None
    # just inside the hold, the retain and the loss tolerance
    d = SEL.decide(runs(**{"0p3": rec(own=-1.349, dec=0.913, val=3.059)}))
    assert d["chosen"] == 0.3
    d = SEL.decide(runs(**{"0p3": rec(own=-1.351, dec=1.4), "1p0": rec(own=-1.3, dec=0.911), "3p0": rec(val=3.061)}))
    assert d["chosen"] == 1.0 and d["flags"] == ["retain_not_met"] and not d["table"]["3.0"]["eligible"]


def test_run_record_parity_and_health(tmp_path):
    chat = tmp_path
    (chat / "STATUS_Fd_1p0").write_text("FINISHED\n")
    for s in SEL.EARLY:
        (chat / f"early_Fd_1p0_{s}").mkdir()
        (chat / f"early_Fd_1p0_{s}" / "early.json").write_text(json.dumps({}))
    (chat / "Fd_1p0").mkdir()
    ev = [{"event": "step", "step": 1, "twin": {"floor": {"gap_absmax": 0.02}}}, {"event": "eval", "step": 1024, "val_loss": 3.0}]
    (chat / "Fd_1p0" / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    r = SEL.safe_record(str(chat), "Fd_1p0", None, None, None, True)
    assert not r["complete"] and "parity" in r["why"]
    (chat / "early_Fd_1p0_256" / "early.json").write_text(json.dumps({"health_unread": "gap"}))
    assert "health" in SEL.safe_record(str(chat), "Fd_1p0", None, None, None, True)["why"]


def own_req(G):
    return [{"id": f"g{g}", "row": r} for g in range(G) for r in ("AA", "BB")]


def own_rec(i, v, n=3, over=False):
    return {"i": i, "sum": v * n, "logprobs": [v] * n, "over_ctx": over}


def test_own_values_and_refusals():
    req = own_req(2)
    recs = [own_rec(0, -1.0), own_rec(1, -2.0, 4), own_rec(2, -0.5), own_rec(3, -1.5, 4)]
    assert REF.own_values(req, recs) == [(-1.0, -2.0, 3, 4), (-0.5, -1.5, 3, 4)]
    for bad in (recs[:3], [dict(recs[0], i=1)] + recs[1:], [own_rec(0, -1.0, over=True)] + recs[1:],
                [dict(recs[0], sum=float("nan"))] + recs[1:], [dict(recs[0], sum=-9.0)] + recs[1:]):
        with pytest.raises(ValueError):
            REF.own_values(req, bad)
    with pytest.raises(ValueError):
        REF.own_values([{"id": "g0", "row": "BB"}, {"id": "g0", "row": "AA"}] + req[2:], recs)


def test_gate_floor_checks():
    def ev(s1, s2):
        return {1: {"twin": {"floor": s1}}, 2: {"twin": {"floor": s2}}}
    ok1 = {"loss": 0.0, "active": 0.0, "gap": -0.5, "gap_absmax": 0.5 + 1e-4}
    ok2 = {"loss": 0.5, "active": 1.0, "gap": 0.5, "gap_absmax": 0.51}
    log = "floor: a\nfloor: b\n"
    assert GATE.floor_checks(ev(ok1, ok2), ev(ok1, ok2), log) == []
    assert GATE.floor_checks(ev(ok1, ok2), ev(ok1, ok2), "floor: a\n")
    assert GATE.floor_checks(ev(dict(ok1, gap_absmax=0.51), ok2), ev(ok1, ok2), log)       # parity off by 0.01
    assert GATE.floor_checks(ev(ok1, dict(ok2, active=0.75)), ev(ok1, ok2), log)
    assert GATE.floor_checks(ev(ok1, dict(ok2, loss=0.6)), ev(ok1, ok2), log)               # CUDA and CPU disagree
    assert GATE.floor_checks(ev(ok1, {"loss": None}), ev(ok1, ok2), log)
    assert GATE.floor_checks(None, ev(ok1, ok2), log)
