#!/usr/bin/env python3
"""The QA-SFT arm's chat jobs (the research tree's AI_ML/chat_screens_claude/cue_selective/qa_sft/REGISTRATION.md, "Launch").
Claude Opus 5.5 (subagent), 25 Nov 2026.

    python3 tools/colab_chat/qa_sft_chat_jobs.py [--check]        writes qa_sft_N250.json

The coverage-SFT campaign A's renderer (coverage_sft_chat_jobs.py, its rule unchanged) with one arm in place of three:
  the start    TL_N250_1's and TL_N250_2's stage-1 weights (sft_tlN250, A's SFT1500 from N250), uploaded as
               sft_tlN250.safetensors and bound by hash at launch, as R2's and the coverage campaign's;
  the arm      S71's packet with the short-answer reading family added (S71QA.jsonl), at chat seeds 1 and 2, trained in the order
               QA_1, QA_2 on one pod;
  no early     no early health readouts and no rung snapshots, as the coverage campaign's.
Kept from R2 and the coverage campaign: the reproduction check and its gate (repro_A, gate), S71's spec at D1 with only the start and
the packet changed, and the binding score job of each arm, under the names QA_N250_<seed>, which no other job, folder or model uses.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a35_jobs as A35  # noqa: E402  (A35's renderer, unchanged)
import b2cue_jobs as B  # noqa: E402  (A32's renderer, unchanged)

BASE, SHAPE, SEEDS = "N250", "D1", A35.SEEDS
START = "sft_tlN250.safetensors"
ARMS = {"QA": "S71QA.jsonl"}
ORDER = [(arm, sd) for sd in SEEDS for arm in ARMS]          # QA_1, QA_2


def out_path():
    return os.path.join(HERE, "qa_sft_N250.json")


def arm_name(arm, seed):
    return f"{arm}_{BASE}_{seed}"


def render():
    repro = B.repro_jobs()
    first = {k: v for k, v in repro[0].items() if k != "after"}
    gate = dict(repro[1], name="gate")
    jobs = [first, gate]
    scores = []
    for arm, sd in ORDER:
        n = arm_name(arm, sd)
        spec = B.arm_spec(SHAPE, f"{B.U}/{START}", sd, "P", packet=ARMS[arm])
        job = B.train_job(n, spec, after=["gate"])
        job.pop("snapshot_steps")       # no rung snapshots on the VM or at home (as the coverage campaign's)
        jobs.append(job)
        scores.append(B.score_job(n, A35.EARLY_REQ, n))
    jobs += scores
    return {"_comment": "QA-SFT arm (lane cue_selective/qa_sft/REGISTRATION.md), rendered by "
                        "tools/colab_chat/qa_sft_chat_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def text():
    return json.dumps(render(), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    p = out_path()
    if a.check:
        if not os.path.exists(p) or open(p).read() != text():
            sys.exit(f"{os.path.basename(p)} differs from its rendering")
        print(f"{os.path.basename(p)} matches")
    else:
        with open(p, "w", newline="\n") as f:
            f.write(text())
        print(f"{len(render()['jobs'])} jobs -> {p}")


if __name__ == "__main__":
    main()
