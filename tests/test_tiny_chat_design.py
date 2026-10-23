"""experiments/tiny_chat_2026-10/make_jobs.py: the registered dialogue-level rule and the matched
shapes (PREREGISTRATION.md sections 5-6)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_jobs", ROOT / "experiments" / "tiny_chat_2026-10" / "make_jobs.py")
mj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mj)


def test_levels_nominal_when_four_copies_cover_a_mix():
    d = mj.dialogue_levels(int(mj.MIX_BYTES / (4 * 0.95)) + 10)
    assert d["mode"] == "shares" and d["levels"] == [0, 25, 50, 100] and d["e1_share"] == 50


def test_levels_by_copies_when_the_corpus_is_short():
    d = mj.dialogue_levels(110_000_000)  # ~0.13 of a mix per copy
    assert d["mode"] == "copies" and d["levels"] == [0, 13, 26, 52] and d["e1_share"] == 52
    d = mj.dialogue_levels(60_000_000)
    assert d["levels"] == [0, 7, 14, 28] and d["e1_share"] == 28


def test_levels_refuse_a_corpus_too_small_to_separate():
    with pytest.raises(SystemExit):
        mj.dialogue_levels(5_000_000)


def test_shapes_matched_within_one_percent():
    for sid, s in mj.shapes().items():
        target = 20e6 if s["size"] == "20m" else 60e6
        assert abs(s["nonemb"] - target) / target < 0.01, sid
        assert s["d"] % 64 == 0 and s["heads"] * 64 == s["d"] and s["d_ff"] % 32 == 0


# ---- amendment 1 (23 Oct 2026): the data decision, the rungs and the exploratory ASHA arm ----
import json  # noqa: E402

COLAB = ROOT / "tools" / "colab_chat"


def jobs(name):
    return json.loads((COLAB / name).read_text(encoding="utf-8"))["jobs"]


def test_levels_for_the_measured_corpus():
    d = mj.dialogue_levels(149_473_532)  # tiny_chat_data.py --measure-only over the 13 pinned sources
    assert d["mode"] == "copies" and d["levels"] == [0, 17, 35, 70] and d["e1_share"] == 35


def test_design_is_final_and_matches_the_rule():
    d = json.loads((ROOT / "experiments" / "tiny_chat_2026-10" / "design.json").read_text())
    assert d["final"] and d["sources"] == mj.DIALOGUE_SOURCES and "aya_en" not in d["sources"]
    assert {k: d[k] for k in ("levels", "e1_share", "mode")} == {
        k: v for k, v in mj.dialogue_levels(d["pt_bytes"]).items() if k in ("levels", "e1_share", "mode")}
    for s in d["sources"]:  # every training source is pinned
        assert (ROOT / "corpus" / "hashes" / f"human_{s}_v1.sha256").exists(), s


def test_data_job_builds_the_pinned_sources_only():
    for f in ("tiny_chat_s0.json", "tiny_chat_e5.json", "tiny_chat_e1.json", "tiny_chat_asha.json"):
        data = next(j for j in jobs(f) if j["name"] == "data")
        cmd = data["cmd"]
        assert "--require-pinned" in cmd and cmd[cmd.index("--sources") + 1] == ",".join(mj.DIALOGUE_SOURCES)
        fetch = cmd[cmd.index("--human-fetch") + 1]
        assert "--allow-llm-risk" not in fetch and "aya_en" not in fetch and "--carry5" not in fetch


def test_every_pretraining_run_has_its_rungs():
    for f in ("tiny_chat_e5.json", "tiny_chat_e1.json", "tiny_chat_e1_60m.json"):
        js = jobs(f)
        pts = [j for j in js if "spec" in j]
        rungs = [j for j in js if j.get("rung_of")]
        assert pts and len(rungs) == 4 * len(pts)
        for p in pts:
            assert p["snapshot_steps"] == [500, 1000, 2500, 5000] and p["spec"]["train"]["eval_every"] == 500
            assert p["spec"]["train"]["steps"] == mj.STEPS  # full length: nothing stops a run early
            mine = sorted(r["rung_of"]["step"] for r in rungs if r["rung_of"]["job"] == p["name"])
            assert mine == p["snapshot_steps"]
        for r in rungs:
            assert r["light"] and r["outputs"] == ["rung.json"]
            assert f"rung_{r['rung_of']['step']}/model.safetensors" in " ".join(r["cmd"])
        assert json.loads((COLAB / f).read_text())["max_parallel_light"] == 1


def test_asha_plan_is_deterministic_and_capped():
    d = json.loads((ROOT / "experiments" / "tiny_chat_2026-10" / "design.json").read_text())
    plan = json.loads((ROOT / "experiments" / "tiny_chat_2026-10" / "asha_plan.json").read_text())
    assert plan == json.loads(json.dumps(mj.asha_plan(d["levels"])))
    assert plan["cost_job_hours"]["worst_vm_hours"] <= mj.ASHA_CAP_VM_HOURS
    assert [(r["budget"], r["keep"]) for r in plan["rounds"]] == [(1000, 27), (3000, 9), (10000, 3)]
    assert len({c["id"] for c in plan["configs"]}) == 27
    for c in plan["configs"]:
        assert c["share"] in d["levels"] and mj.ASHA_LR[0] <= c["lr"] <= mj.ASHA_LR[1]


def test_asha_jobs_wiring():
    js = {j["name"]: j for j in jobs("tiny_chat_asha.json")}
    r0 = [n for n in js if n.startswith("asha_r0_")]
    r1 = [n for n in js if n.startswith("asha_r1_")]
    r2 = [n for n in js if n.startswith("asha_r2_")]
    assert len(r0) == len(r1) == len(r2) == 27
    for k, (rid, prev) in enumerate((("r1", r0), ("r2", r1))):
        p = js[f"asha_promote_{rid}"]
        assert p["light"] and p["after_terminal"] and sorted(p["after"]) == sorted(prev)
    for n in r1 + r2:
        rid, cid = n.split("_")[1], n.split("_")[2]
        assert js[n]["after"] == f"asha_promote_{rid}"
        assert js[n]["promoted_by"]["round"] == rid and js[n]["promoted_by"]["key"] == cid
        assert js[n]["promoted_by"]["files"][-1] == "/content/chat/upload/asha_state.json"
    for n in r2:  # the finalists: full relayed runs, then the E cells' post job
        sp = js[n]["spec"]["train"]
        assert sp["steps"] == 10000 and js[n]["relay_every_steps"] == 2500
        cid = n.split("_")[2]
        assert js[f"asha_post_{cid}"]["init_job"] == n
    for n in r0 + r1:
        assert js[n]["outputs"] == ["rung.json", "run/events.jsonl"]  # every killed run's curve comes home


# ---- amendment 2 (23 Oct 2026): Stage 0's measurements, its reruns, one GPU job at a time ----
def test_throughput_comes_from_the_probes():
    sh = mj.shapes()
    assert abs(mj.tok_per_s(sh["20m_L8"]) - 27947.6) < 1 and abs(mj.tok_per_s(sh["60m_L32"]) - 10307.6) < 1
    a = mj.asha_shapes()  # an ASHA-only depth: time per token between its probed neighbours
    t3 = 1 / mj.tok_per_s(a["20m_L3"])
    assert 1 / mj.tok_per_s(sh["20m_L2"]) < t3 < 1 / mj.tok_per_s(sh["20m_L4"])
    assert mj.VM_SPEEDUP == 1.0 and mj.GPU_JOBS_PER_VM == 1


def test_gpu_jobs_one_at_a_time_and_light_jobs_capped():
    for f in ("tiny_chat_e5.json", "tiny_chat_e1.json", "tiny_chat_e1_60m.json", "tiny_chat_asha.json",
              "tiny_chat_s0_rerun.json", "tiny_chat_s0_60m_rerun.json"):
        doc = json.loads((COLAB / f).read_text())
        assert doc["max_parallel"] == 1, f
        for j in doc["jobs"]:
            if j.get("rung_of"):
                assert j["light"] and j["env"] == {"MICROTORCH_POOL_MB": mj.LIGHT_POOL_MB}
    # The Stage 0 files stay as launched for attempt 1.
    for f in ("tiny_chat_s0.json", "tiny_chat_s0_60m.json"):
        assert json.loads((COLAB / f).read_text())["max_parallel"] == 3


def test_reruns_are_exactly_the_failed_screens_unchanged():
    for size, f, n in (("20m", "tiny_chat_s0.json", 10), ("60m", "tiny_chat_s0_60m.json", 12)):
        rerun = f.replace(".json", "_rerun.json")
        first = {j["name"]: j for j in jobs(f)}
        again = [j for j in jobs(rerun) if j["name"].startswith("screen_")]
        assert sorted(j["name"] for j in again) == sorted(mj.failed_screens(size)) and len(again) == n
        for j in again:
            assert j["cmd"] == first[j["name"]]["cmd"] and j["after"] == "data"  # the same command, lr and seed
        gate = next(j for j in jobs(rerun) if j["name"] == "gate")
        assert gate["cmd"][gate["cmd"].index("--jobs") + 1] == f"tools/colab_chat/{rerun}"


def test_budget_rule_five_from_the_probes():
    rows = mj.budget_rows(mj.shapes())
    core = sum(h for _, _, h in rows[:4])
    assert [r[0].split(":")[0] for r in rows[:4]] == ["Stage 0, 20M", "Stage 0, 20M", "E5 (4 dialogue levels x 3 seeds at 20m_L8)",
                                                       "E1 20M (L2, L4, L16, L32 x 3 seeds; L8 from E5)"]
    assert abs(core - 66.2) < 0.1 and core < mj.CORE_RULE_VM_H  # the amendment's number


# ---- amendment 3 (23 Oct 2026): the floor pilot, and E5 at 60M if it fails ----
def test_pilot_is_one_e1_level_run_on_a_pilot_seed():
    doc = json.loads((COLAB / "tiny_chat_pilot.json").read_text())
    assert doc["max_parallel"] == 1 and doc["max_parallel_light"] == 1
    js = {j["name"]: j for j in doc["jobs"]}
    pts = [j for j in js.values() if "spec" in j]
    assert len(pts) == 1 and mj.PILOT_SEED not in mj.SEEDS and mj.PILOT_SEED != mj.SCREEN_SEED
    pt = pts[0]
    sp = pt["spec"]
    assert pt["name"] == f"pt_pilot_s35_r{mj.PILOT_SEED}" and sp["name"] == "tc_20m_L8"
    assert sp["train"]["seed"] == mj.PILOT_SEED and sp["train"]["steps"] == mj.STEPS
    assert sp["data"]["corpus"].endswith("/mix_s35.ids") and sp["arch"]["custom"]["layers"] == 8
    assert pt["snapshot_steps"] == [500, 1000, 2500, 5000] and pt["after"] == ["data", "gate"]
    rungs = [j for j in js.values() if j.get("rung_of")]
    assert sorted(r["rung_of"]["step"] for r in rungs) == pt["snapshot_steps"]
    for r in rungs:
        assert r["light"] and r["env"] == {"MICROTORCH_POOL_MB": mj.LIGHT_POOL_MB}
    post = js[f"post_pilot_s35_r{mj.PILOT_SEED}"]
    assert post["init_job"] == pt["name"] and "--seed" in post["cmd"] and str(mj.PILOT_SEED) in post["cmd"]
    gate = js["gate"]["cmd"]
    assert gate[gate.index("--jobs") + 1] == "tools/colab_chat/tiny_chat_pilot.json"
    assert gate[gate.index("--pilot-lr") + 1].endswith("stage0/pilot_lr.json") and "--need-pilot" not in gate
    data = js["data"]["cmd"]
    assert data[data.index("--shares") + 1] == "35"
    lr, src, prov = mj.pilot_rate(mj.lr_table()[0])
    assert sp["train"]["lr"] == lr and (prov == (src == "PROVISIONAL"))


def test_blocks_gated_on_the_pilot_verdict():
    for f, need in (("tiny_chat_e5.json", "PASS"), ("tiny_chat_e1.json", "PASS"), ("tiny_chat_e5_60m.json", "FAIL")):
        gate = next(j for j in jobs(f) if j["name"] == "gate")["cmd"]
        assert gate[gate.index("--need-pilot") + 1] == need, f
        assert gate[gate.index("--pilot-verdict") + 1] == "experiments/tiny_chat_2026-10/pilot/verdict.json"
    for f in ("tiny_chat_e1_60m.json", "tiny_chat_asha.json", "tiny_chat_pilot.json"):
        gate = next(j for j in jobs(f) if j["name"] == "gate")["cmd"]
        assert "--need-pilot" not in gate, f
    pts = [j for j in jobs("tiny_chat_e5_60m.json") if "spec" in j]
    assert sorted({j["spec"]["data"]["corpus"].rsplit("/", 1)[1] for j in pts}) == [
        "mix_s0.ids", "mix_s17.ids", "mix_s70.ids"]  # the 35% cell is E1 at 60M's L8 cell
    assert len(pts) == 9 and {j["spec"]["name"] for j in pts} == {"tc_60m_L8"}
    assert {j["spec"]["train"]["seed"] for j in pts} == set(mj.SEEDS)


def test_budget_amendment_3():
    b = mj.budget_a3_rows(mj.shapes())
    assert abs(b["pilot"] - 2.7) < 0.05 and abs(b["e5_60m"] - 40.0) < 0.05
    assert abs(b["pass"] - 154.2) < 0.05 and abs(b["fail"] - 114.1) < 0.05
