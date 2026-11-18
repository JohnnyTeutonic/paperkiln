#!/usr/bin/env python3
"""E12, self-correction chat SFT (the research tree's AI_ML/chat_screens_claude/cue_selective/selfcorr/REGISTRATION.md).
Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.

    python3 tools/colab_chat/selfcorr_chat_jobs.py [--check]        writes selfcorr_N250.json

The token ladder's N250 chat campaign (ladder_chat_jobs.py, imported and not edited) with two changes and nothing else:
  the start    the campaign's own stage 1 is not rerun: TL_N250_1's and TL_N250_2's stage-1 weights (sft_tlN250, A's SFT1500 from
               N250, home since 17 Nov 2026) are uploaded as sft_tlN250.safetensors and bound by hash at launch, so both arms start
               from the very weights the controls started from;
  the packet   S71SC.jsonl, S71's packet with the self-correction family added (selfcorr_build.py; S71's rows and tail unchanged).
Kept: the reproduction check and its gate (repro_A, gate), S71's spec at D1 at seeds 1 and 2, the early health readouts and the
binding score jobs, under the names E12_N250_<seed>, which no other job, folder or model uses.
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
PACKET = "S71SC.jsonl"
OUT = os.path.join(HERE, "selfcorr_N250.json")


def arm_name(seed):
    return f"E12_{BASE}_{seed}"


def render():
    repro = B.repro_jobs()
    first = {k: v for k, v in repro[0].items() if k != "after"}
    gate = dict(repro[1], name="gate")
    jobs = [first, gate]
    scores = []
    for sd in SEEDS:
        n = arm_name(sd)
        spec = B.arm_spec(SHAPE, f"{B.U}/{START}", sd, "P", packet=PACKET)
        jobs += [B.train_job(n, spec, after=["gate"])] + B.early_jobs(n, n, scored=False)
        scores.append(B.score_job(n, A35.EARLY_REQ, n))
    jobs += scores
    return {"_comment": "E12 (lane cue_selective/selfcorr/REGISTRATION.md), rendered by tools/colab_chat/selfcorr_chat_jobs.py; "
                        "edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def text():
    return json.dumps(render(), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        if not os.path.exists(OUT) or open(OUT).read() != text():
            sys.exit("selfcorr_N250.json differs from its rendering")
        print("selfcorr_N250.json matches")
        return
    with open(OUT, "w", newline="\n") as f:
        f.write(text())
    print(f"{len(render()['jobs'])} jobs -> {OUT}")


if __name__ == "__main__":
    main()
