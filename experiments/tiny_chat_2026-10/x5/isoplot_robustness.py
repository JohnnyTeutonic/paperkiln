#!/usr/bin/env python3
"""Descriptive: how stable amendment 5's isoplot is (the line of Y on the FineWeb panel loss through the home
plain 20M runs, and X5's residual from it).

    python experiments/tiny_chat_2026-10/x5/isoplot_robustness.py --art C:/ml_artifacts/tiny_chat --out x5/isoplot_robustness.json

For the line through every home plain 20M run (E5's mixes and E1's depths) and through the 35% mix alone: the
OLS slope and X5's mean residual, the same with each cell left out in turn (their ranges), and a bootstrap over
runs (B resamples, seed 0) of the slope and of X5's mean residual (percentile 95% intervals).
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze as an  # noqa: E402
import analyze_x5 as ax  # noqa: E402


def fit(pts):
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return my - b * mx, b


def resid(line, xs):
    a, b = line
    return sum(y - a - b * x for x, y in xs) / len(xs)


def pct(v, q):
    v = sorted(v)
    k = q * (len(v) - 1)
    f = int(k)
    return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)


def analyse(runs, xs, B):
    pts = {k: (an.get(s, "panel_fineweb"), an.get(s, "Y")) for k, s in runs.items()}
    pts = {k: v for k, v in pts.items() if None not in v}
    line = fit(list(pts.values()))
    cells = sorted({c for c, _ in pts})
    loo = {}
    for c in cells:
        l = fit([v for k, v in pts.items() if k[0] != c])
        if l:
            loo[c] = {"slope": l[1], "x5_mean_residual": resid(l, xs)}
    rng = random.Random(0)
    keys = list(pts)
    bs, br = [], []
    for _ in range(B):
        l = fit([pts[keys[rng.randrange(len(keys))]] for _ in keys])
        if l:
            bs.append(l[1])
            br.append(resid(l, xs))
    xr = [x for x, _ in pts.values()]
    return {"runs": len(pts), "cells": cells, "slope_per_nat": line[1], "x5_mean_residual": resid(line, xs),
            "panel_range": [min(xr), max(xr)], "x5_panel": [x for x, _ in xs],
            "leave_one_cell_out": {"slope_range": [min(v["slope"] for v in loo.values()), max(v["slope"] for v in loo.values())],
                                   "residual_range": [min(v["x5_mean_residual"] for v in loo.values()),
                                                      max(v["x5_mean_residual"] for v in loo.values())],
                                   "by_cell": loo},
            "bootstrap_runs": {"B": B, "slope_ci95": [pct(bs, 0.025), pct(bs, 0.975)],
                               "x5_mean_residual_ci95": [pct(br, 0.025), pct(br, 0.975)]}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=10000)
    a = ap.parse_args(argv)
    xs = []
    for s in ax.SEEDS:
        v = json.load(open(os.path.join(ax.post_dir(a.art, ax.X5, ax.CELL, s), "summary.json"), encoding="utf-8"))
        xs.append((an.get(v, "panel_fineweb"), an.get(v, "Y")))
    allr, s35 = ax.plain_20m(a.art)
    out = {"label": "DESCRIPTIVE (amendment 5 isoplot robustness)",
           "all_plain_20m": analyse(allr, xs, a.b), "s35_plain_20m": analyse(s35, xs, a.b)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    for k in ("all_plain_20m", "s35_plain_20m"):
        r = out[k]
        print(k, r["runs"], r["cells"], "slope", round(r["slope_per_nat"], 3), "resid", round(r["x5_mean_residual"], 4),
              "range", [round(x, 3) for x in r["panel_range"]],
              "LOO slope", [round(x, 3) for x in r["leave_one_cell_out"]["slope_range"]],
              "LOO resid", [round(x, 4) for x in r["leave_one_cell_out"]["residual_range"]],
              "boot slope", [round(x, 3) for x in r["bootstrap_runs"]["slope_ci95"]],
              "boot resid", [round(x, 4) for x in r["bootstrap_runs"]["x5_mean_residual_ci95"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
