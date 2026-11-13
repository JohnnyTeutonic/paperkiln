"""A31's RunPod package (tools/colab_chat/a31_jobs.py, a31_select.py, a31_gate.py, a31_early.py, runner_multi.py's deadline; lane
PROTOCOL.md section 50). Claude Opus 5.5, 13 Nov 2026; revised after the launch reviews of 13 Nov 2026. No GPU, no network.

The selection and early-readout tests import the lane's readout.py (uploaded to the pod as a31_readout.py); they are skipped when
the lane's tree is not mounted."""
import json
import math
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import a31_early as E  # noqa: E402
import a31_gate as G  # noqa: E402
import a31_jobs as J  # noqa: E402
import a31_select as S  # noqa: E402

LANE = Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/cue_selective/a31")
needs_lane = pytest.mark.skipif(not (LANE / "readout.py").exists(), reason="the lane's readout.py is not mounted")


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def jobs():
    return {j["name"]: j for j in J.render()["jobs"]}


# ---------------------------------------------------------------- the jobs file

def test_jobs_file_is_its_rendering():
    assert (ROOT / "tools" / "colab_chat" / "a31_cue.json").read_text() == json.dumps(J.render(), indent=1) + "\n"


def test_specs_differ_from_s71_only_where_registered():
    base = flat(J.S71)
    assert base["train.eval_examples"] == 256 and base["train.checkpoint_every"] == 128
    allowed = {"train.seed", "data.twin_examples", "train.twin_loss.groups", "train.twin_loss.margin", "train.twin_loss.lambda",
               "train.twin_loss.mu", "export.formats"}
    names = set()
    for n, j in jobs().items():
        if "spec" not in j:
            continue
        names.add(n)
        f = flat(j["spec"])
        changed = {k for k in set(base) | set(f) if base.get(k) != f.get(k)}
        assert changed <= allowed, (n, changed - allowed)
        assert f["train.seed"] == (J.DEV_SEED if n.startswith("D") else int(n.split("_")[1]))
        tl = j["spec"]["train"].get("twin_loss")
        kind = n.split("_")[0]
        if kind in ("D0", "P"):
            assert tl is None
        else:
            assert tl["margin"] == J.MARGIN and tl["groups"] == J.GROUPS and tl["lambda"] in J.GRID
            assert tl["mu"] == (0.0 if kind in ("Dm", "M") else tl["lambda"])
        assert (j.get("promoted_by") is not None) == (kind in ("M", "N"))
        assert j["after"] == (["gate", "select"] if kind in ("M", "N") else "gate")
    for sd in J.SEEDS:
        assert f"P_{sd}" in names
        for x in J.GRID:
            assert {f"M_{sd}_{J.lname(x)}", f"N_{sd}_{J.lname(x)}"} <= names
    assert J.SEEDS == (1, 50)


def test_final_test_sets_are_scored_only_for_s71_and_the_arms_after_selection():
    js = jobs()
    for n, j in js.items():
        cmd = " ".join(j.get("cmd", []))
        reads_final = J.FINAL_REQ in cmd
        if n.startswith(("score_D", "early_")) or n in ("gate", "select"):
            assert not reads_final, n
        if n.startswith("score_D"):
            assert J.EARLY_REQ in cmd
        if reads_final:
            assert n == "score_S71" or n[len("score_"):] in js and n[len("score_"):].split("_")[0] in ("P", "M", "N"), n
            after = j["after"] if isinstance(j["after"], list) else [j["after"]]
            # after the selection: directly, or through a training job that waits for it
            assert "select" in after or js[after[0]].get("after") == ["gate", "select"], n


def test_select_waits_for_every_sweep_job_to_end_either_way():
    js = jobs()
    sel = js["select"]
    assert sel["after_terminal"] is True
    dev = [n for n in js if n.startswith("D") and "spec" in js[n]]
    assert len(dev) == 7
    want = set(dev) | {f"score_{n}" for n in dev} | {f"early_{n}_{s}" for n in dev for s in J.EARLY}
    assert set(sel["after"]) == want


def test_every_training_job_has_its_early_readouts_first():
    js = jobs()
    order = list(js)
    for n, j in js.items():
        if "spec" not in j:
            continue
        assert j["snapshot_steps"] == [128, 256, 512]
        assert all(s % j["spec"]["train"]["checkpoint_every"] == 0 for s in j["snapshot_steps"])
        for s in J.EARLY:
            e = js[f"early_{n}_{s}"]
            assert e["rung_of"] == {"job": n, "step": s} and e["light"] and "early.json" in e["outputs"]
            ref = e["cmd"][e["cmd"].index("--ref") + 1]
            assert ref == ("D0" if n.startswith("D") else "P_" + n.split("_")[1])
            assert order.index(f"early_{n}_{s}") < order.index(f"score_{n}")
    assert J.render()["max_parallel_light"] == 2


def test_gate_spec_is_two_full_rate_steps():
    g = J.gate_spec()
    assert g["train"]["steps"] == 2 and g["train"]["batch"] == 1 and g["train"]["twin_loss"]["groups"] == 2
    assert g["data"]["twin_examples"].endswith("a31_gate_twins.jsonl") and g["train"]["eval_examples"] == 1
    assert g["train"]["lr_schedule"]["warmup_steps"] == 0 and g["train"]["twin_loss"]["mu"] > 0


# ---------------------------------------------------------------- the gate

def step(loss, gn, tl=None):
    e = {"loss": loss, "grad_norm": gn}
    if tl is not None:
        e["twin"] = {"loss": tl[0], "d_a": tl[1], "d_b": tl[2], "d_n": tl[3]}
    return e


def runs(ct_gn=3.0, cc_gn=3.0, cp_gn=2.5, cpc_gn=2.5, ct_tl=(0.8, 0.3, 0.2, 0.1), cc_tl=(0.8, 0.3, 0.2, 0.1), plain_loss=2.5):
    ct = {1: step(2.5, ct_gn, ct_tl), 2: step(2.4, ct_gn, ct_tl)}
    cc = {1: step(2.5, cc_gn, cc_tl), 2: step(2.4, cc_gn, cc_tl)}
    cp = {1: step(plain_loss, cp_gn), 2: step(2.45, cp_gn)}
    cpc = {1: step(2.5, cpc_gn), 2: step(2.45, cpc_gn)}
    return ct, cc, cp, cpc


def test_gate_passes_matching_runs():
    assert G.compare(*runs()) == []


def test_gate_fails_on_a_wrong_or_missing_twin_gradient():
    assert G.compare(*runs(ct_gn=3.2, cc_gn=3.2, cp_gn=2.5, cpc_gn=2.5, ct_tl=(0.8, 0.3, 0.2, 0.1))) == []
    assert G.compare(*runs(ct_gn=2.75, cc_gn=3.0))          # CUDA's twin gradient half the CPU's (also grad_norm off by 8%)
    assert any("changes no gradient" in f for f in G.compare(*runs(ct_gn=2.5, cc_gn=2.5)))   # no twin gradient at all


def test_gate_fails_closed_on_missing_or_non_finite_values():
    ct, cc, cp, cpc = runs()
    ct[2]["twin"]["d_n"] = None                              # nlohmann writes NaN as null
    assert any("not finite" in f for f in G.compare(ct, cc, cp, cpc))
    ct, cc, cp, cpc = runs()
    cc[1]["twin"]["loss"] = float("nan")
    assert G.compare(ct, cc, cp, cpc)
    ct, cc, cp, cpc = runs()
    del ct[2]
    assert any("no step 2" in f for f in G.compare(ct, cc, cp, cpc))
    assert G.compare(None, cc, cp, cpc) == ["a run failed or timed out"]


def test_gate_fails_when_the_twin_term_moves_the_sft_batch():
    assert G.compare(*runs(plain_loss=2.5000002))


# ---------------------------------------------------------------- the early readouts' health rule

def ev_steps(n, loss=2.0, twin=True):
    tw = {"loss": 0.5, "d_a": 0.1, "d_b": 0.2, "d_n": 0.05, "active": 0.9}
    return [dict({"event": "step", "step": s, "loss": loss}, **({"twin": dict(tw)} if twin else {})) for s in range(1, n + 1)]


def evl(step, v):
    return {"event": "eval", "step": step, "val_loss": v}


def test_health_rule():
    ev = ev_steps(256) + [evl(256, 4.5)]
    assert E.health(ev, [evl(256, 4.0)], 256, False)[0] == "stop"            # 12.5% above
    assert E.health(ev, [evl(256, 4.2)], 256, False)[0] == "ok"              # 7% above
    assert E.health(ev, [evl(256, 4.0)], 128, False)[0] == "ok"              # the loss leg reads the 25% cut-off only
    assert E.health(ev, ev, 256, True)[0] == "ok"                            # a reference is never compared with itself
    assert E.health(ev_steps(256), [evl(256, 4.0)], 256, False)[0] == "unread"   # the job's eval missing
    assert E.health(ev, [], 256, False)[0] == "unread"                       # the reference's missing
    assert E.health(ev_steps(256) + [evl(256, None)], [evl(256, 4.0)], 256, False)[0] == "stop"   # null = non-finite


def test_health_reads_null_as_non_finite():
    ev = ev_steps(20)
    ev[10]["loss"] = None
    assert E.health(ev, [], 128, True) == ("stop", "non-finite loss")
    ev = ev_steps(20)
    ev[5]["twin"]["loss"] = None
    assert E.health(ev, [], 128, True)[0] == "stop"
    tm = E.train_metrics(ev_steps(256), 256)
    assert tm["step"] == 256 and abs(tm["twin_active_mean32"] - 0.9) < 1e-9 and not tm["nonfinite"]


# ---------------------------------------------------------------- the selection rule

def rec(score_sum_a=-2.0):
    return [{"i": 0, "sum": score_sum_a, "logprobs": [score_sum_a / 2] * 2}]


@pytest.fixture
def R():
    if not (LANE / "readout.py").exists():
        pytest.skip("the lane's readout.py is not mounted")
    sys.path.insert(0, str(LANE))
    import readout
    return readout


def make_chat(tmp, R, runs_spec):
    """A fake /content/chat: each run {status, val, health, dev_edge} -> files a31_select.run_record reads."""
    up = tmp / "upload"
    up.mkdir(parents=True)
    dev = [{"id": f"d{i}"} for i in range(20)]
    hd = [{"id": f"h{i}"} for i in range(5)]
    for n, t in (("dev", dev), ("heldout_dev", hd)):
        (up / f"a31_{n}_triplets.jsonl").write_text("".join(json.dumps(x) + "\n" for x in t))
    man = {"requests": 125, "sets": {"dev": {"triplets": 20, "first_request": 0}, "heldout_dev": {"triplets": 5, "first_request": 100}}}
    (up / "a31_early_req.manifest.json").write_text(json.dumps(man))
    for run, sp in runs_spec.items():
        (tmp / f"STATUS_{run}").write_text(sp.get("status", "FINISHED") + "\n")
        (tmp / run).mkdir()
        ev = ev_steps(3, twin=False) + ([evl(1024, sp["val"])] if "val" in sp else [])
        (tmp / run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
        (tmp / f"score_{run}").mkdir()
        lines = []
        edge = sp.get("edge", 0.0)
        for i in range(125):
            k = i % 5
            own = k in (0, 2)
            # AA and BB gain `edge`; NA keeps AA's score less `nui` of the edge (nui 0: no nuisance shift)
            s = -2.0 + (edge if own else 0.0) + (edge * (1.0 - sp.get("nui", 0.0)) if k == 4 else 0.0)
            lines.append(json.dumps({"i": i, "sum": s, "logprobs": [s / 2, s / 2]}))
        if sp.get("short"):
            lines = lines[:-5]
        (tmp / f"score_{run}" / "scores.jsonl").write_text("\n".join(lines) + "\n")
        for s in S.EARLY:
            if sp.get("no_early") == s:
                continue
            (tmp / f"early_{run}_{s}").mkdir()
            e = {"health_stop": "test"} if sp.get("health") and s == 256 else {}
            (tmp / f"early_{run}_{s}" / "early.json").write_text(json.dumps(e))
        if sp.get("bad_events"):
            (tmp / run / "events.jsonl").write_text("{not json\n")
    return man


def records(tmp, R):
    man = json.load(open(tmp / "upload" / "a31_early_req.manifest.json"))
    trips = {n: R.read_jsonl(tmp / "upload" / f"a31_{n}_triplets.jsonl") for n in man["sets"]}
    names = ["D0"] + [f"{p}_{S.lname(x)}" for p in ("Dm", "Dn") for x in S.GRID]
    return {r: S.safe_record(str(tmp), r, R, man, trips) for r in names}


def base_runs(**over):
    r = {"D0": {"val": 4.0, "edge": 0.0}}
    for i, x in enumerate(S.GRID):
        for p in ("Dm", "Dn"):
            r[f"{p}_{S.lname(x)}"] = {"val": 4.0 + 0.01 * i, "edge": 0.5 * (i + 1), "nui": 0.0}
    for k, v in over.items():
        r[k] = {**r[k], **v}
    return r


def test_selection_picks_by_the_rule(tmp_path, R):
    make_chat(tmp_path, R, base_runs())
    out = S.decide(records(tmp_path, R))
    # every candidate gets both right on every dev twin: a tie, to the smaller lambda; selectivity rises with lambda here
    assert out["arms"]["M"]["chosen"] == 0.1 and out["arms"]["N"]["chosen"] == 1.0
    d = out["decisions"]["a31"]
    assert sum(d.values()) == 2 and d["M_0p1"] and d["N_1p0"]


def test_a_failed_or_stopped_sweep_run_is_ineligible_not_fatal(tmp_path, R):
    make_chat(tmp_path, R, base_runs(Dm_1p0={"status": "FAILED"}, Dn_1p0={"health": True}, Dm_0p3={"short": True}))
    recs = records(tmp_path, R)
    assert not recs["Dm_1p0"]["complete"] and "status FAILED" in recs["Dm_1p0"]["why"]
    assert not recs["Dn_1p0"]["complete"] and "health" in recs["Dn_1p0"]["why"]
    assert not recs["Dm_0p3"]["complete"] and "dev scores" in recs["Dm_0p3"]["why"]
    out = S.decide(recs)
    assert out["arms"]["M"]["chosen"] == 0.1 and out["arms"]["N"]["chosen"] == 0.3


def test_a_missing_early_readout_or_unreadable_log_is_ineligible(tmp_path, R):
    make_chat(tmp_path, R, base_runs(Dm_0p1={"no_early": 256}, Dn_0p1={"bad_events": True}))
    recs = records(tmp_path, R)
    assert not recs["Dm_0p1"]["complete"] and "no early readout at step 256" in recs["Dm_0p1"]["why"]
    assert not recs["Dn_0p1"]["complete"] and "unreadable" in recs["Dn_0p1"]["why"]
    out = S.decide(recs)                   # the selection still decides: the other candidates are read
    assert out["arms"]["M"]["chosen"] == 0.3 and out["arms"]["N"]["chosen"] == 1.0


def test_no_eligible_lambda_runs_no_arm(tmp_path, R):
    make_chat(tmp_path, R, base_runs(Dm_0p1={"val": 4.2}, Dm_0p3={"val": 4.3}, Dm_1p0={"val": float("nan")}))
    out = S.decide(records(tmp_path, R))
    assert out["arms"]["M"]["chosen"] is None and not any(v for k, v in out["decisions"]["a31"].items() if k.startswith("M_"))
    assert out["arms"]["N"]["chosen"] is not None


def test_an_incomplete_d0_runs_no_arm(tmp_path, R):
    make_chat(tmp_path, R, base_runs(D0={"status": "FAILED"}))
    out = S.decide(records(tmp_path, R))
    assert not any(out["decisions"]["a31"].values())


def test_promotion_reads_the_decisions(tmp_path, monkeypatch, R):
    make_chat(tmp_path, R, base_runs())
    (tmp_path / "decisions.json").write_text(json.dumps(S.decide(records(tmp_path, R))))
    monkeypatch.setenv("JOBS_CHAT_DIR", str(tmp_path))
    import importlib
    import jobs_vm
    importlib.reload(jobs_vm)
    got = {}
    for j in J.render()["jobs"]:
        if j.get("promoted_by"):
            j = dict(j, promoted_by=dict(j["promoted_by"], files=[str(tmp_path / "decisions.json")]))
            got[j["name"]] = jobs_vm.promotion(j)
    assert got["M_1_0p1"] is True and got["M_50_0p1"] is True and got["N_1_1p0"] is True and got["N_50_1p0"] is True
    assert sum(got.values()) == 4 and len(got) == 12


# ---------------------------------------------------------------- the runner's absolute deadline

class FakeT:
    def __init__(self):
        self.stopped = 0

    def over_budget(self, st):
        return None

    def attach(self, st):
        pass

    def stop(self, st):
        self.stopped += 1

    def release(self, st):
        pass


def test_the_deadline_is_kept_across_restarts(tmp_path, monkeypatch):
    import importlib
    import runner_multi as RM
    importlib.reload(RM)
    created = []
    T = FakeT()
    monkeypatch.setattr(RM, "make_transport", lambda a: T)
    monkeypatch.setattr(RM, "launch", lambda *a, **k: created.append(1) or True)
    monkeypatch.setattr(RM, "listed", lambda: None)
    state = tmp_path / "state"
    common = ["runner_multi.py", "--commit", "x" * 40, "--jobs", "tools/colab_chat/a31_cue.json", "--name", "t", "--art", str(tmp_path / "art"),
              "--state-dir", str(state), "--backend", "runpod"]
    monkeypatch.setattr(sys, "argv", common + ["--deadline-utc", "2026-01-01T00:00:00Z"])
    RM.main()
    st = json.load(open(state / "state.json"))
    assert st["deadline_epoch"] == RM.parse_deadline_utc("2026-01-01T00:00:00Z")
    # a restart (the watchdog's) with a fresh --max-hours keeps the stored deadline and creates nothing
    monkeypatch.setattr(sys, "argv", common + ["--max-hours", "100"])
    RM.main()
    assert json.load(open(state / "state.json"))["deadline_epoch"] == st["deadline_epoch"]
    assert created == [] and T.stopped == 2


def test_health_is_unread_without_complete_evidence():
    ref = [evl(256, 4.0)]
    gap = [e for e in ev_steps(256) if e["step"] != 100] + [evl(256, 4.1)]
    assert E.health(gap, ref, 256, False)[0] == "unread"                                  # a step missing
    no_twin = ev_steps(256, twin=False) + [evl(256, 4.1)]
    assert E.health(no_twin, ref, 256, False, twin=True)[0] == "unread"                   # a twin run without twin records
    assert E.health(no_twin, ref, 256, False, twin=False)[0] == "ok"
    twice = ev_steps(256) + ev_steps(200) + [evl(256, 4.1)]                              # a resumed run's repeated steps
    assert E.health(twice, ref, 256, False, twin=True)[0] == "ok"


def test_fetch_final_asks_for_a_gguf_only_when_the_spec_exports_one():
    import runner_multi as RM
    js = {j["name"]: j for j in J.render()["jobs"]}
    assert not RM.wants_gguf(js["D0"]) and RM.wants_gguf(js["P_1"]) and RM.wants_gguf({"spec": {}})


def test_a_final_exit_releases_the_storage(tmp_path, monkeypatch):
    import importlib
    import runner_multi as RM
    importlib.reload(RM)
    released = []
    T = FakeT()
    T.release = lambda st: released.append(1)
    monkeypatch.setattr(RM, "make_transport", lambda a: T)
    monkeypatch.setattr(RM, "launch", lambda *a, **k: True)
    monkeypatch.setattr(RM, "listed", lambda: None)
    monkeypatch.setattr(sys, "argv", ["runner_multi.py", "--commit", "x" * 40, "--jobs", "tools/colab_chat/a31_cue.json", "--name", "t",
                                      "--art", str(tmp_path / "art"), "--state-dir", str(tmp_path / "s"), "--backend", "runpod",
                                      "--deadline-utc", "2026-01-01T00:00:00Z"])
    RM.main()
    assert released == [1]      # past the deadline nothing can resume here, so the volume is not left billing
