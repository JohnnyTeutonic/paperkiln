#!/usr/bin/env python3
"""The transfer programme's Stage 2 variance pilot (the research tree's AI_ML/transfer_programme/stage2/REGISTRATION.md):
nine post-training runs on N250's stage-1 SFT model, three arms of format diversity at three chat seeds. Claude Opus 5.5,
28 Nov 2026. A32's renderer (b2cue_jobs.py) imported unchanged: its arm spec (S71's spec at D1) and its training job.

    python3 tools/colab_chat/s2_jobs.py            # writes s2_pilot.json
    python3 tools/colab_chat/s2_jobs.py --check    # compares it with the file on disk; non-zero on any difference

Every run is S71's spec at D1 (1,024 steps, batch 16, lr 1e-4, warm-up 64, cosine to a tenth, T 1,024) from sft_tlN250
(uploaded and bound by hash at launch), on its arm's packet (S2F1, S2F10 or S2F100: the five tasks' answer-only rows, the
same instances in every arm, built and audited by stage2/s2_build.py) at its chat seed; only the packet, the seed and the
export (safetensors only: the models are read at home with harness.py, which reads safetensors) differ from b2cue's plain
arm. No gate job (the plain SFT path of this trainer, c5295cd1, is the token ladder's and the crossed curriculum's), no
score jobs, no rung snapshots. Jobs run seed by seed, the three arms within a seed, so a cut leaves whole seeds.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2cue_jobs as B  # noqa: E402  (A32's renderer, unchanged)

U = B.U
START = f"{U}/sft_tlN250.safetensors"
ARMS = (1, 10, 100)
SEEDS = (1, 2, 3)
OUT = os.path.join(HERE, "s2_pilot.json")


def name(f, seed):
    return f"S2P_F{f}_N250_{seed}"


def spec(f, seed):
    s = B.arm_spec("D1", START, seed, "P", packet=f"S2F{f}.jsonl")
    s["export"] = {"formats": ["safetensors"]}
    return s


def render():
    jobs = []
    for seed in SEEDS:
        for f in ARMS:
            j = B.train_job(name(f, seed), spec(f, seed))
            j.pop("after")
            j.pop("snapshot_steps")
            jobs.append(j)
    return {"_comment": "Transfer programme Stage 2 variance pilot, rendered by tools/colab_chat/s2_jobs.py; edit that file.",
            "max_parallel": 1, "max_parallel_light": 1, "home_markers": True, "jobs": jobs}


def text():
    return json.dumps(render(), indent=1) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t = text()
    if "--check" in argv:
        same = os.path.exists(OUT) and open(OUT, encoding="utf-8").read() == t
        print(json.dumps({"check": "equal" if same else "different"}))
        return 0 if same else 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(t)
    print("s2_pilot.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
