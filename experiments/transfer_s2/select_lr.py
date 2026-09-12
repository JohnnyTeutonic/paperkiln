#!/usr/bin/env python3
"""transfer_s2 stage 1: apply the fixed learning-rate selection rule.

Rule (PREREGISTRATION.md, fixed before any stage-1 run): for each width,
among grid points passing the regime check (best val within the last
three evals in >= 2 of 3 seeds), lr* is the one with the lowest median
best val over seeds; if none passes, the one whose median best-val step is
latest; ties to the larger learning rate.

Usage:
  python3 select_lr.py --width 256 --source 0.001=/path/S_runs_root \
      --source 0.002=/path/lr_S_root ...
Each source root must contain runs/*/events.jsonl. Runs are filtered to
the exact lane and to seeds 21..23 by their model events, never by
directory names. Every number that feeds the rule is printed.
"""
import argparse
import glob
import json
import os
import statistics

SEEDS = (21, 22, 23)


def read_runs(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        ev_path = os.path.join(d, "events.jsonl")
        if not os.path.exists(ev_path):
            continue
        model, evals = None, []
        for line in open(ev_path, encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "model":
                model = e
            elif e.get("event") == "eval":
                evals[:] = [x for x in evals if x[0] < e["step"]]  # last write wins on resume
                evals.append((e["step"], e.get("val_loss", e.get("val"))))
        if model is None or not evals:
            continue
        if model.get("attention") != "exact" or model.get("seed") not in SEEDS:
            continue
        lr = round(float(model.get("lr")), 8)
        evals.sort()
        best_step, best_val = min(evals, key=lambda t: t[1])
        tail = [v for _, v in evals[-3:]]
        out[(lr, model["seed"])] = {
            "best_val": best_val, "best_step": best_step,
            "final_val": evals[-1][1], "regime_ok": best_val in tail,
            "n_evals": len(evals), "dir": d,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="LR=ROOT; the lr is checked against the model events")
    args = ap.parse_args()

    runs = {}
    for spec in args.source:
        lr_s, root = spec.split("=", 1)
        want = round(float(lr_s), 8)
        got = read_runs(root)
        for (lr, seed), r in got.items():
            if lr != want:
                continue
            runs[(lr, seed)] = r
    lrs = sorted({lr for lr, _ in runs}, reverse=True)
    print(f"width {args.width}: stage-1 table (exact lane, seeds {SEEDS})")
    print(f"{'lr':>9} {'seed':>4} {'best_val':>9} {'best_step':>9} {'final':>8} regime")
    rows = {}
    for lr in lrs:
        for seed in SEEDS:
            r = runs.get((lr, seed))
            if r is None:
                print(f"{lr:>9.5g} {seed:>4}   MISSING")
                continue
            print(f"{lr:>9.5g} {seed:>4} {r['best_val']:>9.4f} {r['best_step']:>9} "
                  f"{r['final_val']:>8.4f} {'ok' if r['regime_ok'] else 'FAIL'}")
            rows.setdefault(lr, []).append(r)
    print()
    summary = []
    for lr in lrs:
        rs = rows.get(lr, [])
        if len(rs) < 3:
            print(f"lr {lr:g}: only {len(rs)}/3 seeds present; excluded until complete")
            continue
        med_best = statistics.median(r["best_val"] for r in rs)
        med_step = statistics.median(r["best_step"] for r in rs)
        n_ok = sum(r["regime_ok"] for r in rs)
        summary.append((lr, med_best, med_step, n_ok))
        print(f"lr {lr:g}: median best val {med_best:.4f}, median best step {med_step:g}, "
              f"regime {n_ok}/3 {'PASS' if n_ok >= 2 else 'fail'}")
    if not summary:
        print("no complete grid point yet")
        return
    passing = [s for s in summary if s[3] >= 2]
    if passing:
        # lowest median best val; ties (exact equality) to the larger lr
        best = sorted(passing, key=lambda s: (s[1], -s[0]))[0]
        why = "lowest median best val among regime-passing grid points"
    else:
        best = sorted(summary, key=lambda s: (-s[2], -s[0]))[0]
        why = "no grid point passes the regime check; latest median best-val step"
    print(f"\nlr*({args.width}) = {best[0]:g}   [{why}]")


if __name__ == "__main__":
    main()
