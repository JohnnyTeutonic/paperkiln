#!/usr/bin/env python3
"""E12b, silent-correction chat SFT (the research tree's AI_ML/chat_screens_claude/cue_selective/selfcorr/e12b/E12B_REGISTRATION.md).
Claude Opus 5.5 (a fork of the coordinating session), 19 Nov 2026.

    python3 tools/colab_chat/selfcorr_b_chat_jobs.py [--check]        writes selfcorr_b_N250.json

E12's campaign (selfcorr_chat_jobs.py, imported and not edited) with two changes and nothing else:
  the packet   S71SCb.jsonl: E12's S71SC.jsonl with each W-X and W-D target's correction prefix removed (e12b_build.py); every
               row, the row order, the split and the trainer's draws are E12's;
  the names    E12b_N250_<seed> for the arms, their early readouts and their score jobs.
Kept: the start (sft_tlN250.safetensors, bound by hash at launch), the 1,289 steps, the reproduction check and its gate, S71's spec
at D1 at seeds 1 and 2.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selfcorr_chat_jobs as E12  # noqa: E402  (E12's renderer, unchanged)

PACKET = "S71SCb.jsonl"
OUT = os.path.join(HERE, "selfcorr_b_N250.json")


def arm_name(seed):
    return f"E12b_{E12.BASE}_{seed}"


def render():
    old = (E12.PACKET, E12.arm_name)
    E12.PACKET, E12.arm_name = PACKET, arm_name
    try:
        out = E12.render()
    finally:
        E12.PACKET, E12.arm_name = old
    out["_comment"] = ("E12b (lane cue_selective/selfcorr/e12b/E12B_REGISTRATION.md), rendered by "
                       "tools/colab_chat/selfcorr_b_chat_jobs.py; edit that file, not this one.")
    return out


def text():
    return json.dumps(render(), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        if not os.path.exists(OUT) or open(OUT).read() != text():
            sys.exit("selfcorr_b_N250.json differs from its rendering")
        print("selfcorr_b_N250.json matches")
        return
    with open(OUT, "w", newline="\n") as f:
        f.write(text())
    print(f"{len(render()['jobs'])} jobs -> {OUT}")


if __name__ == "__main__":
    main()
