#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The seed lottery: what verdict would a single-seed lab have published?

Reads the BANKED receipts of a finished, pre-registered experiment
(default: experiments/sparse_s1_boundary — 5 paired seeds x 2 attention
lanes x 3600 steps, d=256) and reports, at every budget slice, the
distribution of conclusions that a one-seed experiment would have
reached, plus the probability that two independent single-seed labs
running IDENTICAL code contradict each other on the sign of the effect.

This consumes only receipts already analysed under their own
pre-registration (sparse_s1_boundary/RESULTS.md). It introduces no new
claims; it re-renders banked data as the methodological exhibit it is.

Output: atlas/SEED_LOTTERY.md (regenerated; do not edit by hand).
"""
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RECEIPTS = os.path.join(REPO, "experiments", "sparse_s1_boundary", "receipts")
OUT = os.path.join(REPO, "atlas", "SEED_LOTTERY.md")

LANES = ("exact", "swa")
SEEDS = (1, 2, 3, 4, 5)
CUTOFF = 3600
SLICE = 400


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RECEIPTS, "run_*_events.jsonl"))):
        evals, model = [], None
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "eval":
                    evals.append((ev["step"], ev["val_loss"]))
                elif ev.get("event") == "model":
                    model = ev
        assert model is not None, path + ": no model event"
        d = model.get("d", model.get("d_model"))
        assert d == 256, f"{path}: d={d} != 256"
        lane, seed = model["attention"], model["seed"]
        assert lane in LANES and seed in SEEDS, (lane, seed)
        assert (lane, seed) not in runs, ("duplicate", lane, seed)
        runs[(lane, seed)] = sorted(evals)
    for lane in LANES:
        for s in SEEDS:
            assert (lane, s) in runs, ("missing", lane, s)
    return runs


def best_by(evals, b):
    vals = [v for step, v in evals if step <= b]
    assert vals, b
    return min(vals)


def shape_section():
    """The lottery is not only about the SIGN — it is about the SHAPE.

    Pools all 15 paired seeds (boundary + seeds receipts) and counts how
    many single seeds display an apparent BASIN: sparse ahead, then
    dense, then sparse again. atlas/THEOREM_CROSSING.md forbids that
    shape in expectation, so every occurrence is noise wearing the
    costume of a phenomenon. Same running-best convention as the
    pre-registered analyses.
    """
    dirs = [RECEIPTS,
            os.path.join(REPO, "experiments", "sparse_s1_seeds", "receipts")]
    per = {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "run_*_events.jsonl"))):
            evals, model = [], None
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("event") == "eval":
                        evals.append((ev["step"], ev["val_loss"]))
                    elif ev.get("event") == "model":
                        model = ev
            if model is None:
                continue
            per.setdefault(model["seed"], {})[model["attention"]] = \
                sorted(evals)
    budgets = list(range(SLICE, CUTOFF + 1, SLICE))
    trajs = {}
    for s, lanes in per.items():
        if not {"exact", "swa"} <= set(lanes):
            continue
        trajs[s] = [best_by(lanes["swa"], b) - best_by(lanes["exact"], b)
                    for b in budgets]
    if not trajs:
        return ""

    def changes(t, band=0.0):
        sg = [0 if abs(v) <= band else (1 if v > 0 else -1) for v in t]
        sg = [x for x in sg if x]
        return sum(1 for i in range(1, len(sg)) if sg[i] != sg[i - 1])

    n = len(trajs)
    sd = sum(statistics.pstdev([trajs[s][i] for s in trajs])
             for i in range(len(budgets))) / len(budgets)
    raw = [s for s in trajs if changes(trajs[s]) >= 2]
    banded = [s for s in trajs if changes(trajs[s], sd) >= 2]
    return (
        "## The lottery is not only about the sign — it is about the shape\n\n"
        f"Pooling all {n} paired seeds and asking a different question: how\n"
        "many single seeds show an apparent BASIN — sparse ahead, then\n"
        "dense, then sparse again?\n\n"
        "| | seeds with >= 2 sign changes |\n|---|---|\n"
        f"| raw trajectory | **{len(raw)} / {n}** |\n"
        f"| ignoring excursions <= {sd:.4f} (mean between-seed SD) | "
        f"**{len(banded)} / {n}** |\n\n"
        f"`atlas/THEOREM_CROSSING.md` proves that shape cannot exist in\n"
        f"expectation: sliding-window attention nests inside exact\n"
        "attention, so the comparison is monotone and crosses at most\n"
        f"once. Yet {len(raw)} of {n} single seeds display it — and\n"
        f"{'none' if not banded else str(len(banded))} survive once\n"
        "excursions smaller than the between-seed SD are treated as zero.\n"
        f"A one-seed experiment here has a ~{len(raw) / n:.0%} chance of\n"
        "reporting a qualitative phenomenon that does not exist. The sign\n"
        "table above says a single seed can get the direction wrong; this\n"
        "says it can invent an entire shape. Regenerate with\n"
        "`python tools/basin_check.py` for the per-seed trajectories.\n\n")


def main():
    runs = load_runs()
    budgets = list(range(SLICE, CUTOFF + 1, SLICE))
    rows = []
    for b in budgets:
        deltas = [best_by(runs[("swa", s)], b) - best_by(runs[("exact", s)], b)
                  for s in SEEDS]
        pos = sum(1 for x in deltas if x > 0)  # "exact wins" verdicts
        neg = len(SEEDS) - pos                 # "swa wins" verdicts
        p = pos / len(SEEDS)
        clash = 2.0 * p * (1.0 - p)  # P(two one-seed labs disagree on sign)
        mean = sum(deltas) / len(deltas)
        rows.append((b, deltas, pos, neg, clash, mean))

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        w = f.write
        w("# The seed lottery\n\n")
        w("*Auto-generated by `tools/seed_lottery.py` from the banked\n"
          "receipts of `experiments/sparse_s1_boundary` (pre-registered;\n"
          "analysed in its RESULTS.md). Do not edit by hand.*\n\n")
        w("**The question**: five labs run *identical* code — same corpus,\n"
          "same architecture (gpt2-nano, d=256), same hyperparameters, same\n"
          "budget — differing only in the random seed. Each publishes the\n"
          "verdict of its one run: does sliding-window attention (w=64+sink)\n"
          "beat exact attention on final validation loss?\n\n")
        w("| budget (steps) | per-seed Δ(swa−exact) | verdicts: exact / swa | "
          "P(two labs contradict) | mean Δ |\n")
        w("|---:|---|:---:|:---:|---:|\n")
        for b, deltas, pos, neg, clash, mean in rows:
            ds = ", ".join(f"{x:+.4f}" for x in deltas)
            w(f"| {b} | {ds} | {pos} / {neg} | {clash:.0%} | {mean:+.4f} |\n")
        w("\n")
        peak = max(rows, key=lambda r: r[4])
        flip_from = next((r for r in rows if r[2] > 0), None)
        w("**How to read it**: early in training every seed agrees (the\n"
          "sparse lane leads and a single-seed paper would be 'right', for a\n"
          "budget-conditional value of right). As the budget grows, the sign\n"
          f"itself becomes seed-dependent: at {peak[0]} steps, two labs\n"
          "publishing off one seed each contradict each other on the "
          f"DIRECTION of the effect {peak[4]:.0%} of the time.\n\n")
        if flip_from is not None:
            w(f"The first pro-exact seed appears at {flip_from[0]} steps; from\n"
              "there the 'conclusion' of a single-seed ablation is a lottery\n"
              "ticket. This is not a pathology of this experiment — it is what\n"
              "an honest resolution study at tiny scale looks like, and it is\n"
              "why every claim in the findings registry carries seed counts,\n"
              "paired tests, and budget scopes rather than a single-run\n"
              "verdict.\n\n")
        w(shape_section())
        w("**Where the claims live**: the registry rows S1c-budget-reversal\n"
          "(the sign flips with budget, t=+5.09, direction pre-committed) and\n"
          "the sparse_s1_boundary result (sign at 3600 UNDETERMINED at n=5;\n"
          "no B* claim licensed) — see `atlas/FINDINGS.md`. The follow-up\n"
          "pre-registration (`experiments/sparse_s1_seeds`) asks whether\n"
          "B*(256) is a point or a distribution, with the criterion fixed\n"
          "before the data existed.\n")
    print(f"wrote {OUT}")
    for b, deltas, pos, neg, clash, mean in rows:
        print(f"b={b:4}  {pos}/{neg}  clash={clash:.0%}  mean={mean:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
