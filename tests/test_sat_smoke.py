"""SAT's CUDA smoke: its jobs file (make_sat.py --smoke) against SAT's, and sat_smoke_vm.py's pure parts.
Claude Opus 5.5, 26 Nov 2026. The end-to-end CPU rehearsal of sat_smoke_vm.py (a main run killed after a checkpoint and
resumed, then `checks` with the CPU builds of this branch and of 672501bb) is recorded in the research tree's SMOKE.md."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
import make_base2 as M  # noqa: E402
import make_sat as S  # noqa: E402
import sat_smoke_vm as V  # noqa: E402


def job(d, name):
    return next(j for j in d["jobs"] if j["name"] == name)


def test_smoke_file_is_the_builder_output():
    assert S.main(["--smoke", "--check"]) == 0
    assert json.loads((CC / S.SMOKE_FILE).read_text()) == json.loads(M.text(S.smoke()))


def test_smoke_specs_are_sats_and_satws_but_for_length_cut_cadence_and_names():
    import make_satw as SW
    sat = job(S.arm(), "pt_base2_SAT")["spec"]
    on = copy.deepcopy(job(S.smoke(), "pt_base2_SATWsmoke")["spec"])
    chk = job(S.smoke(), "sat_check")
    off = json.loads(chk["cmd"][chk["cmd"].index("--spec-json") + 1])
    t = on["train"]
    assert (t["steps"], t["symbol_targets"]["cut_step"], t["eval_every"], t["checkpoint_every"]) == (300, 200, 50, 50)
    assert on["data"]["anneal"]["from_fraction"] * t["steps"] > t["symbol_targets"]["cut_step"]   # the cut before the anneal
    # weight on: SATW's registered W; weight off: no recency_weight at all
    assert t["symbol_targets"].pop("recency_weight") == {"nearest_right": SW.RECENCY_WEIGHT} == {"nearest_right": 0.3077}
    assert "recency_weight" not in json.dumps(off)
    assert on["train"] == off["train"]
    for k in ("steps", "eval_every", "checkpoint_every"):
        t[k] = sat["train"][k]
    t["symbol_targets"]["cut_step"] = sat["train"]["symbol_targets"]["cut_step"]
    on["name"], on["out_dir"] = sat["name"], sat["out_dir"]
    assert on == sat


def test_smoke_jobs():
    d = S.smoke()
    names = [j["name"] for j in d["jobs"]]
    assert names == ["pins", "sat_pin", "sat_check", "sat_check_w", "smoke_watch", "pt_base2_SATWsmoke", "smoke_checks"]
    assert d["store_inputs"] == S.arm()["store_inputs"]
    tj = job(d, "pt_base2_SATWsmoke")
    assert tj["pod_loss_drill_step"] == 100 and tj["store_branch"] == "ckpt-SATWsmoke" and tj["snapshot_steps"] == []
    assert tj["after"] == ["pins", "sat_pin", "sat_check", "sat_check_w"] and tj["halt_on_failure"]
    assert tj["env"] == {"MICROTORCH_CHECKPOINT_TXN": "1", "MICROTORCH_STEP_TIMING": "1"}
    cw = job(d, "sat_check_w")
    assert json.loads(cw["cmd"][cw["cmd"].index("--spec-json") + 1])["train"] == tj["spec"]["train"] and cw["halt_on_failure"]
    sc = job(d, "smoke_checks")
    assert json.loads(sc["cmd"][sc["cmd"].index("--spec-json") + 1]) == tj["spec"] and sc["after"] == [tj["name"], "smoke_watch"]
    off = json.loads(sc["cmd"][sc["cmd"].index("--weight-off-spec-json") + 1])
    assert "recency_weight" not in json.dumps(off) and off["train"]["steps"] == 300
    assert job(d, "sat_pin") == job(S.arm(), "sat_pin")
    assert not any(n.startswith("rung_") for n in names)


def test_weight_sums_and_the_cut():
    spec = {"train": {"batch": 8, "accum": 2, "steps": 6, "symbol_targets": {"cut_step": 4, "recency_weight": {"nearest_right": 0.3077}}}}
    ev = [{"event": "step", "step": s, "loss": 1.0, **({"symbol_targets": {"natural_pointer_nearest_right": 10, "natural_pointer_nearest_wrong": 3,
                                                                           "natural_pointer_weight_sum": 0.3077 * 10 + 3}} if s < 4 else {})}
          for s in range(1, 7)]
    assert V.weight_report(ev, spec)["ok"]
    ev[1]["symbol_targets"]["natural_pointer_weight_sum"] = 13.0          # the weight-off sum under a weight-on spec
    assert not V.weight_report(ev, spec)["ok"]
    off = {"train": {**spec["train"], "symbol_targets": {"cut_step": 4}}}
    assert not V.weight_report(ev, off)["ok"]                             # weight off: no SATW field may appear
    bare = [{**e, "symbol_targets": {"role_loss": 1.0}} if e["step"] < 4 else e for e in ev]
    assert V.weight_report(bare, off)["ok"] and V.weight_report(bare, off)["path"] == "weight off"
    end = [{"event": "symbol_targets_end", "step": 4}]
    assert V.cut_transition(ev, end, 4)["ok"] and not V.cut_transition(ev, [], 4)["ok"]
    ev[4]["symbol_targets"] = {"w": 0}
    assert not V.cut_transition(ev, end, 4)["ok"]


def test_scaffold_rows_rule_and_step_comparison():
    spec = {"train": {"batch": 8, "accum": 2, "steps": 6, "symbol_targets": {"cut_step": 4, "twins": {"every": 10}}}}
    ev = []
    for s in range(1, 7):
        e = {"event": "step", "step": s, "loss": 1.0 / s, "grad_norm": 1.0, "lr": 0.1}
        if s < 4:
            tw = sum(1 for g in range((s - 1) * 16, s * 16) if g % 10 == 0)
            e["symbol_targets"] = {"twin_rows": tw, "natural_rows": 16 - tw, "role_loss": 2.0, "pointer_loss": 3.0}
        ev.append(e)
    r = V.scaffold_report(ev, spec)
    assert r["ok"] and r["scaffold_steps_read"] == 3
    ev[1]["symbol_targets"]["twin_rows"] += 1
    assert not V.scaffold_report(ev, spec)["ok"]
    other = json.loads(json.dumps(ev))
    assert V.compare_steps(ev, other, list(range(1, 7)))["all_equal"]
    other[4]["loss"] += 1e-7
    c = V.compare_steps(ev, other, list(range(1, 7)))
    assert not c["all_equal"] and c["first_difference"]["step"] == 5 and c["equal"] == 5


def test_copy_generation_restores_the_events_cut_at_its_checkpoint(tmp_path):
    g = tmp_path / "src" / "ckpt_gen" / "g8"
    g.mkdir(parents=True)
    (g / "state.txt").write_text("8\n{}\n")
    (g / "MANIFEST.json").write_text("{}")
    evs = tmp_path / "events.jsonl"
    evs.write_text("".join(json.dumps(e) + "\n" for e in
                           [{"event": "step", "step": 8}, {"event": "checkpoint", "step": 8}, {"event": "step", "step": 9}]))
    V.copy_generation(str(g), str(tmp_path / "dst"), str(evs))
    assert (tmp_path / "dst" / "ckpt_gen" / "live").resolve() == (tmp_path / "dst" / "ckpt_gen" / "g8").resolve()
    assert [json.loads(x)["step"] for x in (tmp_path / "dst" / "events.jsonl").read_text().splitlines()] == [8, 8]
    assert V.live_generation(str(tmp_path / "dst"))[0] == 8
