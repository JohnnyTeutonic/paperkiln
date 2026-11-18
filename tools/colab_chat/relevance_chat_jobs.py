#!/usr/bin/env python3
"""The relevance programme's chat screens (the research tree's AI_ML/chat_screens_claude/cue_selective/relevance/PROGRAMME.md).
Claude Opus 5.5 (a fork of the coordinating session), 19 Nov 2026.

    python3 tools/colab_chat/relevance_chat_jobs.py [--check]        writes relevance_R1_N250.json and relevance_R2_N250.json

Each screen is the token ladder's N250 chat campaign (ladder_chat_jobs.py, imported and not edited) with two changes and nothing
else, as E12's selfcorr_chat_jobs.py:
  the start    the campaign's own stage 1 is not rerun: TL_N250_1's and TL_N250_2's stage-1 weights (sft_tlN250, A's SFT1500 from
               N250) are uploaded as sft_tlN250.safetensors and bound by hash at launch;
  the packet   S71's packet with one family added (R1: S71FU.jsonl, the follow-up family; R2: S71BD.jsonl, the binding family).
Kept: the reproduction check and its gate (repro_A, gate), S71's spec at D1 at seeds 1 and 2, the early health readouts and the
binding score jobs, under the names <screen>_N250_<seed>, which no other job, folder or model uses.
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
SCREENS = {"R1": "S71FU.jsonl", "R2": "S71BD.jsonl"}


def out_path(screen):
    return os.path.join(HERE, f"relevance_{screen}_N250.json")


def arm_name(screen, seed):
    return f"{screen}_{BASE}_{seed}"


def render(screen):
    repro = B.repro_jobs()
    first = {k: v for k, v in repro[0].items() if k != "after"}
    gate = dict(repro[1], name="gate")
    jobs = [first, gate]
    scores = []
    for sd in SEEDS:
        n = arm_name(screen, sd)
        spec = B.arm_spec(SHAPE, f"{B.U}/{START}", sd, "P", packet=SCREENS[screen])
        jobs += [B.train_job(n, spec, after=["gate"])] + B.early_jobs(n, n, scored=False)
        scores.append(B.score_job(n, A35.EARLY_REQ, n))
    jobs += scores
    return {"_comment": f"Relevance programme screen {screen} (lane cue_selective/relevance/PROGRAMME.md), rendered by "
                        "tools/colab_chat/relevance_chat_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def text(screen):
    return json.dumps(render(screen), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for s in SCREENS:
        p = out_path(s)
        if a.check:
            if not os.path.exists(p) or open(p).read() != text(s):
                sys.exit(f"{os.path.basename(p)} differs from its rendering")
            print(f"{os.path.basename(p)} matches")
        else:
            with open(p, "w", newline="\n") as f:
                f.write(text(s))
            print(f"{len(render(s)['jobs'])} jobs -> {p}")


if __name__ == "__main__":
    main()
