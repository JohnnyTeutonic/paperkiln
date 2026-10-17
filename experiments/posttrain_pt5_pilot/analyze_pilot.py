#!/usr/bin/env python3
"""posttrain_pt5_pilot decision rule (PREREGISTRATION.md). Frozen at licence.

    python experiments/posttrain_pt5_pilot/analyze_pilot.py WORK      # WORK/beta_*/results.json
    python experiments/posttrain_pt5_pilot/analyze_pilot.py --smoke   # drive every branch
"""
from __future__ import annotations

import glob
import json
import os
import sys

GRID = (0.1, 0.5, 2.0)
ARMS = ("E", "J", "R", "Mu")
FLOOR = 0.90


def decide(by_beta):
    """by_beta: {beta: {arm: answer_rate}}. Returns (outcome, beta or None)."""
    missing = [(b, a) for b in GRID for a in ARMS if a not in by_beta.get(b, {})]
    if missing:
        raise SystemExit(f"refusing to decide: missing {missing}")
    for b in GRID:
        if all(by_beta[b][a] >= FLOOR for a in ARMS):
            return "CHOSEN", b
    return "NO_BETA", None


def smoke():
    ok = {a: 0.95 for a in ARMS}
    bad = dict(ok, Mu=0.5)
    assert decide({0.1: ok, 0.5: ok, 2.0: ok}) == ("CHOSEN", 0.1)
    assert decide({0.1: bad, 0.5: ok, 2.0: ok}) == ("CHOSEN", 0.5)
    assert decide({0.1: bad, 0.5: bad, 2.0: dict(ok, E=0.90)}) == ("CHOSEN", 2.0)  # the floor is inclusive
    assert decide({0.1: bad, 0.5: bad, 2.0: bad}) == ("NO_BETA", None)
    try:
        decide({0.1: ok, 0.5: ok})
    except SystemExit:
        pass
    else:
        raise AssertionError("accepted an incomplete grid")
    print("analyze_pilot smoke: every branch fires")


def main(argv):
    if argv and argv[0] == "--smoke":
        smoke()
        return 0
    work = argv[0]
    by_beta, rows = {}, []
    for p in sorted(glob.glob(os.path.join(work, "beta_*", "results.json"))):
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        b = res["beta"]
        m = res["results"][0]["models"]
        by_beta[b] = {a: m[a]["answer_rate"] for a in ARMS if a in m}
        rows.append((b, m))
    outcome, beta = decide(by_beta)
    L = ["# posttrain_pt5_pilot — analysis output", "",
         f"Floor: every arm's answer_rate >= {FLOOR}; choose the smallest qualifying beta.", "",
         "| beta | base | " + " | ".join(ARMS) + " |", "|---:|" + "---:|" * (len(ARMS) + 1)]
    for b, m in sorted(rows):
        L.append(f"| {b} | {m['base']['answer_rate']:.3f} | " +
                 " | ".join(f"{m[a]['answer_rate']:.3f}" for a in ARMS) + " |")
    L += ["", "Descriptive (truth_acc / judge_truth_gap_25):", ""]
    for b, m in sorted(rows):
        L.append(f"- beta {b}: " + "; ".join(f"{a} {m[a]['truth_acc']:.3f}/{m[a]['judge_truth_gap_25']:.3f}"
                                              for a in ("base",) + ARMS))
    L += ["", f"**Outcome: {outcome}**" + (f": beta = {beta}" if beta is not None else "")]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(work, "ANALYSIS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
