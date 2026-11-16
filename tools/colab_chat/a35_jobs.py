#!/usr/bin/env python3
"""A35's jobs file (the root workspace's AI_ML/chat_screens_claude/a35/REGISTRATION.md): S71's chat recipe on base 2's selected base
C. Claude Fable 5.2, 16 Nov 2026. Writes tools/colab_chat/a35_c.json; `--check` compares the file on disk with a fresh rendering.

Nothing is changed but the base. Every spec is built by A32's renderer (b2cue_jobs.py, imported and not edited), at A's shape
(C continued A's weights, so it has A's shape):

  repro_A        50 steps of A's sft_b1_r1 recipe from A's final base (uploaded as base_A.safetensors), b2cue_jobs.repro_jobs's job
  gate           b2cue_early.py --repro: repro_A's step losses against sft_b1_r1's own events within 1% (reported; it always exits 0,
                 so the campaign goes on, and the scorer at home reads the record)
  sft_C          A's SFT1500 (sft_b1_r1's recipe, seed 1, with A32's held-out evaluations at 188, 375 and 750) from C at 330,000 steps
                 (uploaded as base_C.safetensors), and its watch (the non-finite stop, the readouts relayed)
  S75_1, S75_2   S71's spec (A31's rendering of it: checkpoint_every 128 and eval_examples 256, I/O and evaluation only) on S71's own
                 packet (uploaded as S71.jsonl), from sft_C, at S71's two seeds, 1 and 2. No twin term, no twin data.
  early_*        b2cue_early.py at 128, 256 and 512, --no-score: A31's health rule (each run is its own reference, so the non-finite
                 leg only) and the trainer's metrics
  score_*        `mtstudio score` on A31's early request file (its dev and held-out dev twins; reported, never judged) and
                 a31_bind.py's score.json, which binds each model's weights and GGUF for the guard evaluation at home

    python3 tools/colab_chat/a35_jobs.py [--check]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2cue_jobs as B  # noqa: E402  (A32's renderer, unchanged)

OUT = os.path.join(HERE, "a35_c.json")
SEEDS = (1, 2)                     # S71's two seeds (lane PROTOCOL.md sections 41 and 44)
PACKET = "S71.jsonl"               # S71's packet, sha256 37afd630...
EARLY_REQ = f"{B.U}/a31_early_req.jsonl"
BASE, SHAPE = "C", "A"


def arm_name(seed):
    return f"S75_{seed}"


def render():
    repro = B.repro_jobs()
    first = {k: v for k, v in repro[0].items() if k != "after"}   # the first job of the campaign: it waits on nothing
    gate = dict(repro[1], name="gate")          # the reproduction check stands where A32's gate does: every later job waits on it
    jobs = [first, gate]
    jobs += B.sft_jobs(BASE, SHAPE, f"base_{BASE}.safetensors")
    scores = []
    for sd in SEEDS:
        n = arm_name(sd)
        spec = B.arm_spec(SHAPE, f"{B.C}/sft_{BASE}/sft_{BASE}.safetensors", sd, "P", packet=PACKET)
        jobs += [B.train_job(n, spec, after=["gate", f"sft_{BASE}"], init_job=f"sft_{BASE}")] + B.early_jobs(n, n, scored=False)
        scores.append(B.score_job(n, EARLY_REQ, n))
    jobs += scores
    return {"_comment": "A35 (lane a35/REGISTRATION.md), rendered by tools/colab_chat/a35_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = json.dumps(render(), indent=1) + "\n"
    if a.check:
        if not os.path.exists(OUT) or open(OUT).read() != text:
            sys.exit("a35_c.json differs from its rendering")
        print("a35_c.json matches")
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"{len(render()['jobs'])} jobs -> {OUT}")


if __name__ == "__main__":
    main()
