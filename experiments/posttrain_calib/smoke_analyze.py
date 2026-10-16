#!/usr/bin/env python3
"""Drives every branch of analyze.decide on fabricated measurements (D3):
a rule never shown capable of firing is not a rule."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402


def rows(means, spread=0.01):
    out = []
    for R, m in zip(A.LEVELS, means):
        for k, s in enumerate(A.SEEDS):
            out.append({"level": R, "seed": s, "truth_acc": m + (k - 1) * spread, "judge_reward": m + 0.1,
                        "format_pass_truth_fail": 1 - m})
    return out


def expect(means, outcome, R=None):
    got, gotR, _ = A.decide(rows(means))
    assert (got, gotR) == (outcome, R), (means, got, gotR)
    print(f"  {str(means):24s} -> {got} {gotR or ''}")


def main():
    expect([0.70, 0.40, 0.10], "CHOSEN", 20)    # smallest in band is chosen, not the largest
    expect([0.30, 0.40, 0.50], "CHOSEN", 10)
    expect([0.05, 0.10, 0.25], "CHOSEN", 50)    # the band's edge is inclusive
    expect([0.80, 0.70, 0.60], "CHOSEN", 50)
    expect([0.10, 0.20, 0.24], "TOO_HARD")
    expect([0.90, 0.80, 0.61], "TOO_EASY")
    expect([0.70, 0.65, 0.20], "GAP")
    for bad in (rows([0.3, 0.3, 0.3])[:-1], rows([0.3, 0.3, 0.3]) + rows([0.3, 0.3, 0.3])[:1]):
        try:
            A.decide(bad)
        except SystemExit as e:
            print(f"  refused: {e}")
        else:
            raise AssertionError("decide accepted an incomplete or duplicated design")
    print("smoke_analyze: every branch fires")


if __name__ == "__main__":
    main()
