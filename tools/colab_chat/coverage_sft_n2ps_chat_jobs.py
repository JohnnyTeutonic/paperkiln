#!/usr/bin/env python3
"""The coverage-SFT campaign D: PERS and STATE at the second pretraining seed (the research tree's
AI_ML/chat_screens_claude/cue_selective/coverage_sft/n2ps/REGISTRATION.md). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/coverage_sft_n2ps_chat_jobs.py [--check]        writes coverage_sft_ps_N250s2.json

coverage_sft_chat_jobs.py (campaign A, its rule unchanged) with two of its three arms and N2's start in place of N250's, as
campaign C (coverage_sft_n2_chat_jobs.py) did for COV:
  the start    sft_tlN250s2 (the token ladder's SFT1500 from N250s2), uploaded as sft_tlN250s2.safetensors and bound by hash at
               launch, as campaign C bound it;
  the arms     PERS (S71PERS.jsonl) and STATE (S71STATE.jsonl), campaign A's packets unchanged, at chat seeds 1 and 2, trained in
               the order PERS_1, STATE_1, PERS_2, STATE_2 on one pod; COV at N2 is campaign C's, the controls the ladder's
               TL_N250s2_1 and TL_N250s2_2;
  no early     no early readouts and no rung snapshots, as campaigns A and C.
Kept from campaign A: the reproduction check and its gate (repro_A, gate), S71's spec at D1 with only the start and the packet
changed, and each model's binding score job, under the names <ARM>_N250s2_<seed>, which no other job, folder or model uses.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a35_jobs as A35  # noqa: E402  (A35's renderer, unchanged)
import b2cue_jobs as B  # noqa: E402  (A32's renderer, unchanged)

BASE, SHAPE, SEEDS = "N250s2", "D1", A35.SEEDS
START = "sft_tlN250s2.safetensors"
ARMS = {"PERS": "S71PERS.jsonl", "STATE": "S71STATE.jsonl"}
ORDER = [(arm, sd) for sd in SEEDS for arm in ARMS]          # PERS_1, STATE_1, PERS_2, STATE_2


def out_path():
    return os.path.join(HERE, "coverage_sft_ps_N250s2.json")


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
        job.pop("snapshot_steps")       # no rung snapshots, as campaigns A and C
        jobs.append(job)
        scores.append(B.score_job(n, A35.EARLY_REQ, n))
    jobs += scores
    return {"_comment": "Coverage-SFT campaign D, PERS and STATE at N2 (lane cue_selective/coverage_sft/n2ps/REGISTRATION.md), "
                        "rendered by tools/colab_chat/coverage_sft_n2ps_chat_jobs.py; edit that file, not this one.",
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
