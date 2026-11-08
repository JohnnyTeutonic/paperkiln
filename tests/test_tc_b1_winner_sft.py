"""A's SFT1500 graph: the PT-only graph plus the three registered jobs, unchanged, and a pause once the bench is home. Claude Opus 5.5, 9 Nov 2026."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/colab_chat"))
import runner_multi as rm

CC = ROOT / "tools/colab_chat"


def load(name):
    return json.loads((CC / name).read_text())


def graph():
    return load("tc_b1_winner_sft_20261109.json")


def test_only_the_registered_jobs_come_back():
    old, pt_graph, g = load("tc_b1_r1.json"), load("tc_b1_winner_pt_20261108.json"), graph()
    expected = copy.deepcopy(pt_graph)
    expected["_comment"] = g["_comment"]
    added = [next(j for j in g["jobs"] if j["name"] == n) for n in ("bench_inputs", "final_diag")]
    expected["jobs"] += added + [copy.deepcopy(j) for j in old["jobs"] if j["name"] in ("sft_b1_r1", "items", "bench_b1")]
    next(j for j in expected["jobs"] if j["name"] == "pt_b1_r1")["pause_rungs"].append("bench_b1")
    assert g == expected
    for name in ("sft_b1_r1", "items", "bench_b1"):     # the registered jobs, byte for byte
        assert next(j for j in g["jobs"] if j["name"] == name) == next(j for j in old["jobs"] if j["name"] == name)


def test_the_final_diagnostics_run_first_on_the_named_final_and_never_halt_the_arm():
    jobs = graph()["jobs"]
    names = [j["name"] for j in jobs]
    d = jobs[names.index("final_diag")]
    assert names.index("final_diag") < names.index("sft_b1_r1") and d["after"] == "pt_b1_r1"
    assert not d.get("halt_on_failure") and "rung_of" not in d and "exclusive" not in d
    cmd = d["cmd"][-1]
    assert cmd.startswith("echo '") and "sha256sum -c -" in cmd and cmd.count("--device cuda") == 2
    assert "--art /content/chat/pt_b1_r1" in cmd and "model.safetensors" not in cmd and "state.txt" not in cmd
    assert d["outputs"] == ["final_diag.json", "stage/stage.json"] and d["env"]["MICROTORCH_POOL_MB"] == 1024


def test_inputs_are_checked_before_anything_is_spent():
    jobs = graph()["jobs"]
    names = [j["name"] for j in jobs]
    c = jobs[names.index("bench_inputs")]
    assert names.index("bench_inputs") < names.index("final_diag") < names.index("sft_b1_r1")
    assert c["halt_on_failure"] and c["light"] and c["after"] == "pt_b1_r1"
    cmd = c["cmd"][-1]
    bench = jobs[names.index("bench_b1")]["cmd"][-1]
    for n in (10000, 20000, 30000, 40000, 50000):
        assert f"/content/chat/pt_b1_r1/rung_{n}/model.safetensors" in cmd and f"/content/chat/pt_b1_r1/rung_{n}/model.safetensors" in bench
    assert "test ! -e /content/chat/sft_b1_r1/state.txt" in cmd      # SFT starts fresh from the final weights, never from a stale state


def test_the_recipe_is_s_fulls_sft1500_but_for_its_initial_weights():
    sft = next(j for j in graph()["jobs"] if j["name"] == "sft_b1_r1")
    s = sft["spec"]
    assert sft["init_job"] == "pt_b1_r1" and sft["after"] == "pt_b1_r1"
    assert s["train"]["objective"] == "sft" and s["train"]["steps"] == 1500 and s["train"]["lr"] == 0.0003
    assert s["train"]["batch"] == 16 and s["train"]["accum"] == 1 and s["train"]["seed"] == 1
    assert s["train"]["lr_schedule"] == {"warmup_steps": 100, "decay": "cosine", "min_lr_frac": 0.1}
    assert s["data"]["sft_examples"] == "/content/chat/data/sft_human.jsonl" and s["data"]["T"] == 1024
    pt = next(j for j in graph()["jobs"] if j["name"] == "pt_b1_r1")
    assert s["arch"] == pt["spec"]["arch"]


def test_the_arm_pauses_once_the_bench_is_home_and_never_releases_its_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "HOME", str(tmp_path))
    st = rm.State()
    g = graph()
    pt = next(j for j in g["jobs"] if j["name"] == "pt_b1_r1")
    for rung in ("rung_b1_s20000", "rung_b1_s30000", "pt_b1_r1"):
        (tmp_path / ("RELEASED_" + rung)).write_text("{}")
    for j in g["jobs"]:
        st.job(j["name"]).update(status="FINISHED", home=j["name"] != "bench_b1")
    assert rm.pause_at_rung(pt, st) is None
    st.job("bench_b1")["home"] = True
    assert rm.pause_at_rung(pt, st) == "bench_b1"
    halt = json.loads((tmp_path / "HALTED").read_text())
    assert halt["kind"] == "CUTOFF_REACHED" and halt["rung"] == "bench_b1"
    assert not st.d.get("campaign_home")


def test_the_graph_passes_the_runners_own_checks():
    rm.validate_checkpoint_families(graph()["jobs"], str(CC / "boot_jobs.sh"))
