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


def test_smoke_spec_is_sats_but_for_length_cut_cadence_and_names():
    sat = job(S.arm(), "pt_base2_SAT")["spec"]
    sm = copy.deepcopy(job(S.smoke(), "pt_base2_SATsmoke")["spec"])
    t = sm["train"]
    assert (t["steps"], t["symbol_targets"]["cut_step"], t["eval_every"], t["checkpoint_every"]) == (300, 200, 50, 50)
    assert sm["data"]["anneal"]["from_fraction"] * t["steps"] > t["symbol_targets"]["cut_step"]   # the cut before the anneal
    for k in ("steps", "eval_every", "checkpoint_every"):
        t[k] = sat["train"][k]
    t["symbol_targets"]["cut_step"] = sat["train"]["symbol_targets"]["cut_step"]
    sm["name"], sm["out_dir"] = sat["name"], sat["out_dir"]
    assert sm == sat


def test_smoke_jobs():
    d = S.smoke()
    names = [j["name"] for j in d["jobs"]]
    assert names == ["pins", "sat_pin", "sat_check", "smoke_watch", "pt_base2_SATsmoke", "smoke_checks"]
    assert d["store_inputs"] == S.arm()["store_inputs"]
    tj = job(d, "pt_base2_SATsmoke")
    assert tj["pod_loss_drill_step"] == 100 and tj["store_branch"] == "ckpt-SATsmoke" and tj["snapshot_steps"] == []
    assert tj["after"] == ["pins", "sat_pin", "sat_check"] and tj["halt_on_failure"]
    assert tj["env"] == {"MICROTORCH_CHECKPOINT_TXN": "1", "MICROTORCH_STEP_TIMING": "1"}
    chk = job(d, "sat_check")
    assert json.loads(chk["cmd"][chk["cmd"].index("--spec-json") + 1])["train"] == tj["spec"]["train"]
    sc = job(d, "smoke_checks")
    assert json.loads(sc["cmd"][sc["cmd"].index("--spec-json") + 1]) == tj["spec"] and sc["after"] == [tj["name"], "smoke_watch"]
    assert job(d, "sat_pin") == job(S.arm(), "sat_pin")
    assert not any(n.startswith("rung_") for n in names)


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
