"""Carry-12's pre-freeze campaign (experiments/tiny_chat_long_2026-10/make_stage0_jobs.py, stage0_total.py):
the jobs file is the generator's output; every spec, dependency and option it names exists; the probe specs
are the core cells' shapes at T 1,024 and 16,384 tokens per step; the cost runs are the post job's Carry-12
configuration; and the measured total's arithmetic on synthetic receipts."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "tiny_chat_long_2026-10"
JOBS = ROOT / "tools" / "colab_chat" / "carry12_stage0_v1.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_jobs_file_is_the_generators_output(tmp_path):
    m = load("make_stage0_jobs")
    m.JOBS = str(tmp_path / "jobs.json")
    m.main()
    assert json.loads((tmp_path / "jobs.json").read_text()) == json.loads(JOBS.read_text())


def test_jobs_file_names_what_exists():
    doc = json.loads(JOBS.read_text())
    jobs = {j["name"]: j for j in doc["jobs"]}
    assert doc["max_parallel"] == 1
    for j in jobs.values():
        after = j.get("after") or []
        for x in [after] if isinstance(after, str) else after:
            assert x in jobs, (j["name"], x)
        cmd = j["cmd"]
        if "--spec" in cmd:
            assert (ROOT / cmd[cmd.index("--spec") + 1]).exists()
        assert (ROOT / cmd[1]).exists(), cmd[1]
    d = jobs["data"]["cmd"]
    assert d[d.index("--T") + 1] == "1024" and "--carry12" in d and d[d.index("--long-sft") + 1] == "10:4"
    assert d[d.index("--shares") + 1] == "35" and "--require-pinned" in d
    srcs = jobs["data_long3"]["cmd"][jobs["data_long3"]["cmd"].index("--sources") + 1].split(",")
    assert {"taskmaster3", "abcd", "ccpe", "p4g"} <= set(srcs) and "msc" not in srcs
    probes = [n for n in jobs if n.startswith("probe60_")]
    assert len(probes) == 5
    for n in probes:
        cmd = jobs[n]["cmd"]
        s = json.loads((ROOT / cmd[cmd.index("--spec") + 1]).read_text())
        t = s["train"]
        assert s["data"]["T"] == 1024 and t["batch"] * t["accum"] * 1024 == 16384, n
        assert not jobs[n].get("light")  # alone on the GPU
    sg = json.loads((EXP / "specs" / "probe60_Sgist_T1024.json").read_text())
    assert sg["arch"]["state_tokens"] == {"k": 4, "mask": "turn_state"} and sg["arch"]["custom"]["layers"] == 8
    x = json.loads((EXP / "specs" / "probe60_X_T1024.json").read_text())
    assert x["arch"]["custom"]["attn_res"] == {"form": "full", "scale": "sources"} and x["arch"]["custom"]["layers"] == 16
    costs = [n for n in jobs if n.startswith("cost_")]
    assert costs == ["cost_pick_e1_60m_L8", "cost_pick_e1_60m_L16", "cost_pick_e1_60m_L32"]
    for n in costs:
        cmd = jobs[n]["cmd"]
        assert "--cost-run" in cmd and "--no-near-self" in cmd and cmd[cmd.index("--ctx") + 1] == "1024"
        assert cmd[cmd.index("--max-new") + 1] == "48" and jobs[n]["after_terminal"]
        assert set(probes) <= set(jobs[n]["after"])
    for tag in ("pick_e1_60m_L8", "pick_e1_60m_L16", "pick_e1_60m_L32"):
        assert f"--upload {tag}.model.safetensors=" in doc["_comment"]


def test_total_arithmetic_on_synthetic_receipts(tmp_path):
    m = load("stage0_total")
    walls = {"jobs": {}}
    for sh, pt, post, sft in (("60m_L8", 13000.0, 11000.0, 2400.0), ("60m_L16", 14500.0, 13400.0, 2400.0)):
        for r in (1, 2, 3):
            walls["jobs"][f"pt_e1_{sh}_r{r}"] = {"wall_s": pt, "partial": False}
            walls["jobs"][f"post_e1_{sh}_r{r}"] = {"wall_s": post, "sft_s": sft, "partial": False}
    (tmp_path / "e1_60m_walls_t512.json").write_text(json.dumps(walls))
    rates = {"probe60_B_T1024": 13799 * 0.9, "probe60_B_T1024_acc2": 13000, "probe60_Sgist_T1024": 13799 * 0.85,
             "probe60_P16_T1024": 12517 * 0.9, "probe60_X_T1024": 12517 * 0.6}
    for k, v in rates.items():
        (tmp_path / f"{k}.summary.json").write_text(json.dumps({"rc": 0, "tok_per_s_steady": v, "peak_gpu_mem_mib": 1}))
    for pick in ("pick_e1_60m_L8", "pick_e1_60m_L16"):
        (tmp_path / f"cost_{pick}.l0.json").write_text(json.dumps({
            "eval": {"ember_build": {"seconds": 360, "built_here": True}},
            "cost_run": {"eval_wall_s": 7200, "timing": {}, "gen_engine": "ember", "served_ctx": 1024}}))
    assert m.main(["--stage0", str(tmp_path)]) == 0
    t = json.loads((tmp_path / "total.json").read_text())
    b = t["cells"]["B"]
    assert t["b_layout"] == "probe60_B_T1024"
    assert abs(b["pt_h"] - 13000 / 0.9 / 3600) < 0.01 and abs(b["sft_h"] - 2400 / 0.9 / 3600) < 0.01
    assert abs(b["post_rest_h"] - (11000 - 2400) / 3600) < 0.01 and abs(b["carry12_post_h"] - 7560 / 3600) < 0.01
    assert t["cells"]["D"]["pt_h"] == 0 and t["cells"]["D"]["machine_h"] == 0
    assert abs(t["cells"]["X"]["pt_h"] - 14500 / 0.6 / 3600) < 0.01
    assert abs(t["core_vm_hours"] - sum(c["vm_hours"] for c in t["cells"].values())) < 0.2
    assert t["core_usd"] == round(t["core_vm_hours"] * 0.59, 2)
    # B's 16 x 1,024 failing (out of memory): the 8 x 2 layout is used
    (tmp_path / "probe60_B_T1024.summary.json").write_text(json.dumps({"rc": 1, "tok_per_s_steady": None}))
    assert m.main(["--stage0", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "total.json").read_text())["b_layout"] == "probe60_B_T1024_acc2"
    os.remove(tmp_path / "cost_pick_e1_60m_L16.l0.json")
    assert m.main(["--stage0", str(tmp_path)]) == 1  # incomplete: nothing written over the last total
