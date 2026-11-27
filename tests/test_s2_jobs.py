"""The Stage 2 pilot's jobs file (tools/colab_chat/s2_jobs.py) against b2cue's plain arm: only the registered differences.
Claude Opus 5.5, 28 Nov 2026."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools", "colab_chat")
sys.path.insert(0, TOOLS)
import b2cue_jobs as B  # noqa: E402
import s2_jobs as P  # noqa: E402

BASE = "04191f57400caaa711eac454be79e05eeaabd2b6"


def test_file_is_current():
    assert P.main(["--check"]) == 0


def test_nine_runs_seed_by_seed():
    d = json.load(open(P.OUT))
    names = [j["name"] for j in d["jobs"]]
    assert names == [f"S2P_F{f}_N250_{s}" for s in (1, 2, 3) for f in (1, 10, 100)]
    assert d["max_parallel"] == 1


def test_only_packet_seed_and_export_differ():
    d = json.load(open(P.OUT))
    for j in d["jobs"]:
        f, seed = int(j["name"].split("_")[1][1:]), int(j["name"].split("_")[-1])
        ref = B.arm_spec("D1", P.START, seed, "P", packet=f"S2F{f}.jsonl")
        got = json.loads(json.dumps(j["spec"]))
        assert got.pop("export") == {"formats": ["safetensors"]}
        assert got["train"].pop("steps") == P.STEPS[f]
        ref.pop("export", None)
        ref["train"].pop("steps")
        assert got == ref
        s = j["spec"]
        assert s["data"]["sft_examples"] == f"/content/chat/upload/S2F{f}.jsonl"
        assert s["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
        assert (s["train"]["batch"], s["train"]["lr"], s["train"]["seed"]) == (16, 0.0001, seed)
        assert 500 <= s["train"]["steps"] <= 2000
        assert "twin_loss" not in s["train"] and "twin_examples" not in s["data"]
        assert "after" not in j and "snapshot_steps" not in j


def test_launch_commit_adds_only_these_files():
    out = subprocess.run(["git", "-C", HERE, "diff", "--name-status", BASE, "HEAD"], capture_output=True, text=True,
                         check=True).stdout.split("\n")
    rows = [r.split("\t") for r in out if r.strip()]
    own = {"tools/colab_chat/s2_jobs.py", "tools/colab_chat/s2_pilot.json", "tests/test_s2_jobs.py"}
    assert all(s == "A" and p in own for s, p in rows), rows
