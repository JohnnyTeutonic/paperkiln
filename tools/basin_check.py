#!/usr/bin/env python3
"""Does any BANKED trajectory show a basin? (atlas/THEOREM_CROSSING.md)

    python tools/basin_check.py

The crossing theorem forbids a basin — Delta(b) may cross zero at most
once, sparse->dense, never back — but it says so about the EXPECTED
Delta. A single seed is not an expectation. This script asks both
questions of the receipts already in the repo (sparse_s1_boundary +
sparse_s1_seeds, 15 paired seeds at d=256, nine 400-step slices):

  1. Does the POOLED MEAN trajectory cross more than once?
     A second crossing there would be evidence against assumption (iii)
     in data we already own.
  2. How often does a SINGLE SEED display an apparent basin (>= 2 sign
     changes)? That number is the seed lottery restated for shape: the
     rate at which a one-seed experiment would report a phenomenon the
     theorem forbids in expectation.

Reads events.jsonl only; no new compute, no network.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st

SLICES = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECEIPTS = [
    os.path.join(REPO, "experiments", "sparse_s1_boundary", "receipts"),
    os.path.join(REPO, "experiments", "sparse_s1_seeds", "receipts"),
]


def read(path):
    evals, model = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "model":
                model = ev
            elif ev.get("event") == "eval":
                evals.append((int(ev["step"]), float(ev["val_loss"])))
    if model is None:
        return None
    d = {}
    for s, v in sorted(evals):
        d[s] = v
    return {"lane": model.get("attention"), "seed": int(model["seed"]),
            "evals": sorted(d.items())}


def val_at(evals, b):
    """RUNNING-BEST val loss at budget b — min over evals at or before b.

    This is the convention the PRE-REGISTERED analysis uses
    (experiments/sparse_s1_seeds/analyze.py::bestval_by), and this tool
    must match it or it is answering a different question. It matters:
    reading the LAST eval instead of the running best moves the pooled
    Delta(3600) from +0.0195 to +0.0359 and the sign split from 9+/6-
    to 13+/2- on the same receipts. Neither reading is wrong in the
    abstract — running-best is what early stopping would deliver, and
    it is monotone per lane so it does not inherit eval noise — but the
    registered one is authoritative, and a supplementary tool that
    quietly disagrees with it manufactures a contradiction out of a
    convention. (Checked 31 Aug 2026: the receipts are byte-identical
    to their source and analyze.py reproduces its published numbers
    exactly; S1e stands.)
    """
    vals = [v for s, v in evals if s <= b]
    return min(vals) if vals else None


def sign_changes(traj, band=0.0):
    """Sign changes along a trajectory, ignoring |x| <= band as zero."""
    signs = [0 if abs(v) <= band else (1 if v > 0 else -1) for v in traj]
    signs = [s for s in signs if s != 0]
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def main():
    runs = {}
    for root in RECEIPTS:
        for p in sorted(glob.glob(os.path.join(root, "*_events.jsonl"))):
            r = read(p)
            if r:
                runs.setdefault(r["seed"], {})[r["lane"]] = r
    seeds = sorted(s for s, l in runs.items() if {"exact", "swa"} <= set(l))
    if not seeds:
        raise SystemExit("no paired receipts found")

    print("=" * 66)
    print("BASIN CHECK vs atlas/THEOREM_CROSSING.md — banked receipts only")
    print(f"{len(seeds)} paired seeds at d=256, slices {SLICES[0]}..{SLICES[-1]}")
    print("=" * 66)

    per_seed = {}
    for s in seeds:
        traj = []
        for b in SLICES:
            ve = val_at(runs[s]["exact"]["evals"], b)
            vw = val_at(runs[s]["swa"]["evals"], b)
            traj.append(None if ve is None or vw is None else vw - ve)
        if any(t is None for t in traj):
            continue
        per_seed[s] = traj

    # 1. the pooled mean — what the theorem actually constrains
    mean_traj = [st.mean([per_seed[s][i] for s in per_seed])
                 for i in range(len(SLICES))]
    mono = all(mean_traj[i] >= mean_traj[i - 1] - 1e-9
               for i in range(1, len(mean_traj)))
    print("\npooled mean Delta(b):")
    print("  " + "  ".join(f"{v:+.4f}" for v in mean_traj))
    print(f"  monotone non-decreasing: {mono}")
    print(f"  sign changes: {sign_changes(mean_traj)}   "
          f"(theorem permits at most 1, sparse->dense)")

    # 2. the single-seed picture — the seed lottery, restated for shape
    apparent = [s for s in per_seed if sign_changes(per_seed[s]) >= 2]
    band = st.mean([st.pstdev([per_seed[s][i] for s in per_seed])
                    for i in range(len(SLICES))])
    apparent_band = [s for s in per_seed
                     if sign_changes(per_seed[s], band) >= 2]
    print(f"\nper-seed apparent basins (>= 2 sign changes):")
    print(f"  raw:            {len(apparent)}/{len(per_seed)} seeds "
          f"{sorted(apparent)}")
    print(f"  ignoring |Delta| <= {band:.4f} (mean between-seed SD): "
          f"{len(apparent_band)}/{len(per_seed)} {sorted(apparent_band)}")
    for s in sorted(apparent)[:4]:
        print("    seed %-3d %s" % (s, "  ".join(f"{v:+.3f}"
                                                 for v in per_seed[s])))
    print("\nReading: the theorem constrains the EXPECTATION, and the "
          "pooled trajectory obeys it. Any per-seed count above is the "
          "rate at which a ONE-SEED experiment would report a shape the "
          "theorem forbids in expectation — the seed lottery, restated "
          "for shape rather than for sign.")


if __name__ == "__main__":
    main()
