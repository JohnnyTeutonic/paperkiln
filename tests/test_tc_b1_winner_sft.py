"""A's SFT1500 graph: the PT-only graph plus the three registered jobs, unchanged, and a pause once the bench is home. Claude Opus 5.5, 9 Nov 2026."""
import copy
import json
import os
import subprocess
import sys
import time
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
    registered = [copy.deepcopy(j) for j in old["jobs"] if j["name"] in ("sft_b1_r1", "items", "bench_b1")]
    registered[0]["after"] = ["pt_b1_r1", "bench_inputs"]
    registered[0]["spec"]["train"]["checkpoint_every"] = 25
    expected["jobs"] += added + registered
    next(j for j in expected["jobs"] if j["name"] == "pt_b1_r1")["pause_rungs"].append("bench_b1")
    assert g == expected
    for name in ("items", "bench_b1"):     # the registered jobs, byte for byte
        assert next(j for j in g["jobs"] if j["name"] == name) == next(j for j in old["jobs"] if j["name"] == name)


def test_the_final_diagnostics_run_first_on_the_named_final_and_never_halt_the_arm():
    jobs = graph()["jobs"]
    names = [j["name"] for j in jobs]
    d = jobs[names.index("final_diag")]
    assert names.index("final_diag") < names.index("sft_b1_r1") and d["after"] == ["pt_b1_r1", "bench_inputs"]
    assert not d.get("halt_on_failure") and "rung_of" not in d and "exclusive" not in d
    cmd = d["cmd"][-1]
    assert cmd.startswith("echo '") and "sha256sum -c -" in cmd and cmd.count("--device cuda") == 2
    assert "--art /content/chat/pt_b1_r1" in cmd and "model.safetensors" not in cmd and "state.txt" not in cmd
    assert d["outputs"] == ["final_diag.json", "stage/stage.json"] and d["env"]["MICROTORCH_POOL_MB"] == 1024


def test_inputs_are_checked_before_anything_is_spent():
    jobs = graph()["jobs"]
    names = [j["name"] for j in jobs]
    c = jobs[names.index("bench_inputs")]
    assert "bench_inputs" in jobs[names.index("final_diag")]["after"] and "bench_inputs" in jobs[names.index("sft_b1_r1")]["after"]
    assert "pt_b1_r1" in jobs[names.index("final_diag")]["after"] and "pt_b1_r1" in jobs[names.index("sft_b1_r1")]["after"]
    assert c["halt_on_failure"] and c["light"] and c["after"] == "pt_b1_r1"
    cmd = c["cmd"][-1]
    bench = jobs[names.index("bench_b1")]["cmd"][-1]
    for n in (10000, 20000, 30000, 40000, 50000):
        assert f"/content/chat/pt_b1_r1/rung_{n}/model.safetensors" in cmd and f"/content/chat/pt_b1_r1/rung_{n}/model.safetensors" in bench
    assert "test ! -e /content/chat/sft_b1_r1/state.txt" in cmd      # SFT starts fresh from the final weights, never from a stale state


def test_the_recipe_is_s_fulls_sft1500_but_for_its_initial_weights():
    sft = next(j for j in graph()["jobs"] if j["name"] == "sft_b1_r1")
    s = sft["spec"]
    assert sft["init_job"] == "pt_b1_r1" and sft["after"] == ["pt_b1_r1", "bench_inputs"]
    assert s["train"]["objective"] == "sft" and s["train"]["steps"] == 1500 and s["train"]["lr"] == 0.0003
    assert s["train"]["batch"] == 16 and s["train"]["accum"] == 1 and s["train"]["seed"] == 1
    assert s["train"]["lr_schedule"] == {"warmup_steps": 100, "decay": "cosine", "min_lr_frac": 0.1}
    assert s["data"]["sft_examples"] == "/content/chat/data/sft_human.jsonl" and s["data"]["T"] == 1024
    pt = next(j for j in graph()["jobs"] if j["name"] == "pt_b1_r1")
    assert s["arch"] == pt["spec"]["arch"]
    old = next(j for j in load("tc_b1_r1.json")["jobs"] if j["name"] == "sft_b1_r1")
    changed = {k for k in set(s["train"]) | set(old["spec"]["train"]) if s["train"].get(k) != old["spec"]["train"].get(k)}
    assert changed == {"checkpoint_every"} and s["train"]["checkpoint_every"] == 25      # how often state is saved, nothing of the arithmetic
    assert {k: v for k, v in s.items() if k != "train"} == {k: v for k, v in old["spec"].items() if k != "train"}
    assert sft["relay_every_steps"] == 500 and sft["env"]["MICROTORCH_CHECKPOINT_TXN"] == "1"   # home transfer every 500, publication every 25


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


def manager(tmp_path, bench_cmd):
    """The real jobs_vm.py on the graph's own dependencies of bench_inputs, final_diag and sft_b1_r1, each job a stand-in that records when it ran."""
    g = {j["name"]: j for j in graph()["jobs"]}
    log = tmp_path / "log"
    log.mkdir()

    def mark(name, rest="true"):
        return ["sh", "-c", f"echo $(date +%s.%N) > {log}/{name}.start; {rest}; r=$?; echo $(date +%s.%N) > {log}/{name}.end; exit $r"]

    jobs = {"max_parallel": 1, "max_parallel_light": 2, "jobs": [
        {"name": "pt_b1_r1", "light": True, "cmd": mark("pt_b1_r1")},
        {"name": "bench_inputs", "light": True, "after": g["bench_inputs"]["after"], "cmd": mark("bench_inputs", bench_cmd)},
        {"name": "final_diag", "after": g["final_diag"]["after"], "cmd": mark("final_diag")},
        {"name": "sft_b1_r1", "after": g["sft_b1_r1"]["after"], "cmd": mark("sft_b1_r1")}]}
    chat = tmp_path / "chat"
    chat.mkdir()
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    env = {**os.environ, "JOBS_CHAT_DIR": str(chat), "JOBS_REPO": str(tmp_path), "JOBS_TICK": "0.05"}
    subprocess.run([sys.executable, str(CC / "jobs_vm.py"), str(jf)], env=env, timeout=60, capture_output=True)
    status = {n: (chat / f"STATUS_{n}").read_text().strip() for n in ("pt_b1_r1", "bench_inputs", "final_diag", "sft_b1_r1")}
    times = {p.name: float(p.read_text()) for p in log.iterdir()}
    return status, times


def test_no_gpu_job_starts_while_the_input_check_waits_or_runs(tmp_path):
    status, t = manager(tmp_path, "sleep 1.5")
    assert status == {"pt_b1_r1": "FINISHED", "bench_inputs": "FINISHED", "final_diag": "FINISHED", "sft_b1_r1": "FINISHED"}
    for heavy in ("final_diag", "sft_b1_r1"):
        assert t[f"{heavy}.start"] >= t["bench_inputs.end"]          # not while it was WAITING on PT, nor while it RAN
    assert t["bench_inputs.start"] >= t["pt_b1_r1.end"]


def test_a_failed_input_check_starts_no_gpu_job(tmp_path):
    status, t = manager(tmp_path, "sleep 0.5; false")
    assert status["bench_inputs"] == "FAILED" and status["final_diag"] == "FAILED" and status["sft_b1_r1"] == "FAILED"
    assert "final_diag.start" not in t and "sft_b1_r1.start" not in t


def test_the_graph_passes_the_runners_own_checks():
    rm.validate_checkpoint_families(graph()["jobs"], str(CC / "boot_jobs.sh"))
