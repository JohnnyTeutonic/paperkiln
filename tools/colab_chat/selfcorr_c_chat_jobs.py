#!/usr/bin/env python3
"""E12c, correct earlier replies in place of the injected wrong ones (the research tree's
AI_ML/chat_screens_claude/cue_selective/coverage_sft/REGISTRATION.md, Part 2). Claude Opus 5.5 (a fork of the coordinating session),
22 Nov 2026.

    python3 tools/colab_chat/selfcorr_c_chat_jobs.py [--check]        writes selfcorr_c_N250.json

E12b's campaign (selfcorr_b_chat_jobs.py's rule, on E12's renderer selfcorr_chat_jobs.py, imported and not edited) with two changes
and nothing else:
  the packet   S71SCc.jsonl (e12c/e12c_build.py; checked by e12c_packet_check.py);
  the names    E12c_N250_<seed> for the arms, their early readouts and their score jobs.
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

PACKET = "S71SCc.jsonl"
OUT = os.path.join(HERE, "selfcorr_c_N250.json")


def arm_name(seed):
    return f"E12c_{E12.BASE}_{seed}"


def render():
    old = (E12.PACKET, E12.arm_name)
    E12.PACKET, E12.arm_name = PACKET, arm_name
    try:
        out = E12.render()
    finally:
        E12.PACKET, E12.arm_name = old
    out["_comment"] = ("E12c (lane cue_selective/coverage_sft/REGISTRATION.md, Part 2), rendered by "
                       "tools/colab_chat/selfcorr_c_chat_jobs.py; edit that file, not this one.")
    return out


def text():
    return json.dumps(render(), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        if not os.path.exists(OUT) or open(OUT).read() != text():
            sys.exit("selfcorr_c_N250.json differs from its rendering")
        print("selfcorr_c_N250.json matches")
        return
    with open(OUT, "w", newline="\n") as f:
        f.write(text())
    print(f"{len(render()['jobs'])} jobs -> {OUT}")


if __name__ == "__main__":
    main()
