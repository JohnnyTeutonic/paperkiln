#!/usr/bin/env python3
"""posttrain_calib decision rule (PREREGISTRATION.md, "Decision rule"). Frozen
at the licence commit.

    python experiments/posttrain_calib/analyze.py [receipts/measurements.json]
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = (10, 20, 50)
SEEDS = (1, 2, 3)
LO, HI = 0.25, 0.60


def decide(rows):
    """Returns (outcome, chosen R or None, per-level table). Refuses on
    missing or duplicated runs."""
    seen = {}
    for r in rows:
        key = (r["level"], r["seed"])
        if key in seen:
            raise SystemExit(f"duplicate measurement for level {key[0]} seed {key[1]}")
        seen[key] = r
    missing = [(R, s) for R in LEVELS for s in SEEDS if (R, s) not in seen]
    if missing:
        raise SystemExit(f"refusing to decide: missing runs {missing}")
    table = {}
    for R in LEVELS:
        vals = [seen[(R, s)]["truth_acc"] for s in SEEDS]
        table[R] = {"per_seed": vals, "mean": statistics.mean(vals), "sd": statistics.stdev(vals),
                    "in_band": LO <= statistics.mean(vals) <= HI,
                    "judge_reward_mean": statistics.mean(seen[(R, s)]["judge_reward"] for s in SEEDS),
                    "format_pass_truth_fail_mean": statistics.mean(seen[(R, s)]["format_pass_truth_fail"] for s in SEEDS)}
    for R in LEVELS:  # smallest qualifying level
        if table[R]["in_band"]:
            return "CHOSEN", R, table
    means = [table[R]["mean"] for R in LEVELS]
    if all(m < LO for m in means):
        return "TOO_HARD", None, table
    if all(m > HI for m in means):
        return "TOO_EASY", None, table
    return "GAP", None, table


def main(argv=None):
    path = (argv or sys.argv[1:] or [os.path.join(HERE, "receipts", "measurements.json")])[0]
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["runs"]
    outcome, R, table = decide(rows)
    lines = ["# posttrain_calib — analysis output", "",
             f"Band [{LO}, {HI}] on mean dev truth_acc over seeds {list(SEEDS)}.", "",
             "| R | per-seed truth_acc | mean | seed SD | in band | judge reward (mean) | format-pass, truth-fail (mean) |",
             "|---:|---|---:|---:|:---:|---:|---:|"]
    for L in LEVELS:
        t = table[L]
        lines.append(f"| {L} | {', '.join(f'{v:.3f}' for v in t['per_seed'])} | {t['mean']:.3f} | {t['sd']:.3f} | "
                     f"{'yes' if t['in_band'] else 'no'} | {t['judge_reward_mean']:.3f} | {t['format_pass_truth_fail_mean']:.3f} |")
    lines += ["", f"**Outcome: {outcome}**" + (f": R = {R}" if R else "")]
    text = "\n".join(lines) + "\n"
    print(text)
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "ANALYSIS.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
