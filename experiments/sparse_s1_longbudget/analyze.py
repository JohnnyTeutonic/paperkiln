"""Pre-registered analysis for sparse_s1_longbudget.

WRITTEN AND COMMITTED WITH PREREGISTRATION.md, BEFORE ANY RUN.

  H-OVERFIT-ORDER  argmin_b L_exact < argmin_b L_swa, paired sign test
                   over 10 seeds, one-tailed, reject at >= 8/10.
  H-SECOND-CROSS   pooled mean Delta goes +,then -, with BOTH excursions
                   clearing t*SD/sqrt(n) (t=2.262) at their slices.
                   A wobble inside the threshold is NOT a crossing.
  Z-CLOSE          first slice where the zone closes and stays closed;
                   "never closes by 12000" is a reportable result.
  W-CHECK          observed zone width vs the corollary's
                   W ~= 2 t SD / (sqrt(n) |s|).
  Threat 3         >= 8/10 seeds must show a held-out minimum strictly
                   before the final eval in at least one lane, else
                   H-SECOND-CROSS is UNTESTED, not unsupported.

Usage:  python3 analyze.py [--root /path/to/longbudget]
"""
import argparse
import glob
import json
import math
import os
import statistics as st

STEPS = 12000
SLICE = 1000                 # report grid; evals are every 200
T_CRIT = 2.262               # two-tailed, df = 9
SEEDS = list(range(41, 51))
LANES = ("exact", "swa")


def read_run(d):
    evals, model = [], None
    with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
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
        raise SystemExit(f"{d}: no model event (refuse-to-run guard)")
    dedup = {}
    for s, v in sorted(evals):
        dedup[s] = v
    return {"lane": model.get("attention"), "seed": int(model["seed"]),
            "d": int(model["d"]), "layers": int(model["layers"]),
            "evals": sorted(dedup.items())}


def load(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        if os.path.exists(os.path.join(d, "events.jsonl")):
            r = read_run(d)
            out.setdefault(r["seed"], {})[r["lane"]] = r
    return out


def val_at(evals, step):
    vals = [v for s, v in evals if s <= step]
    return vals[-1] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/c/ml_artifacts/transfer/longbudget")
    args = ap.parse_args()
    runs = load(args.root)
    seeds = [s for s in SEEDS if s in runs and
             all(l in runs[s] for l in LANES)]
    print("=" * 68)
    print("sparse_s1_longbudget PRE-REGISTERED ANALYSIS")
    print(f"seeds complete: {len(seeds)}/{len(SEEDS)}")
    print("=" * 68)
    if not seeds:
        return

    # --- Threat 3: did we reach the regime this experiment probes? ----
    reached = 0
    for s in seeds:
        hit = False
        for l in LANES:
            ev = runs[s][l]["evals"]
            if not ev:
                continue
            best_step = min(ev, key=lambda kv: kv[1])[0]
            if best_step < ev[-1][0]:
                hit = True
        reached += hit
    print(f"Threat 3 (overfit reality): {reached}/{len(seeds)} seeds show a "
          f"held-out minimum before the final eval (need >= 8)")
    regime_ok = reached >= 8

    # --- H-OVERFIT-ORDER ----------------------------------------------
    wins = 0
    print("\nH-OVERFIT-ORDER  argmin_b exact  <  argmin_b swa")
    for s in seeds:
        ae = min(runs[s]["exact"]["evals"], key=lambda kv: kv[1])[0]
        aw = min(runs[s]["swa"]["evals"], key=lambda kv: kv[1])[0]
        ok = ae < aw
        wins += ok
        print(f"  seed {s}: exact {ae:>6}  swa {aw:>6}   "
              f"{'exact first' if ok else '-'}")
    print(f"  {wins}/{len(seeds)} (reject at >= 8/10) -> "
          f"{'SUPPORTED' if wins >= 8 else 'not supported'}")

    # --- Delta trajectory, zone, crossings -----------------------------
    grid = list(range(SLICE, STEPS + 1, SLICE))
    print("\nD-SHAPE + ZONE (pooled)")
    print("     b   meanDelta        SD    thresh   zone?")
    means, sds, thr = {}, {}, {}
    for b in grid:
        ds = []
        for s in seeds:
            ve = val_at(runs[s]["exact"]["evals"], b)
            vw = val_at(runs[s]["swa"]["evals"], b)
            if ve is not None and vw is not None:
                ds.append(vw - ve)
        if len(ds) < 2:
            continue
        m, sd = st.mean(ds), st.stdev(ds)
        t = T_CRIT * sd / math.sqrt(len(ds))
        means[b], sds[b], thr[b] = m, sd, t
        print(f"  {b:>5}  {m:+.5f}  {sd:.5f}  {t:.5f}   "
              f"{'IN ZONE' if abs(m) <= t else ''}")

    # --- H-SECOND-CROSS -------------------------------------------------
    sig = [(b, means[b]) for b in grid
           if b in means and abs(means[b]) > thr[b]]
    signs = [(b, 1 if m > 0 else -1) for b, m in sig]
    changes = [(signs[i - 1][0], signs[i][0])
               for i in range(1, len(signs))
               if signs[i][1] != signs[i - 1][1]]
    print(f"\nH-SECOND-CROSS  significant sign changes: {len(changes)}")
    for a, b in changes:
        print(f"  crossing between b={a} and b={b}")
    second = any(signs[i - 1][1] > 0 and signs[i][1] < 0
                 for i in range(1, len(signs)))
    if not regime_ok:
        print("  VERDICT: UNTESTED (Threat 3 failed — the runs never "
              "reached the overfit regime)")
    else:
        print(f"  VERDICT: {'SUPPORTED — assumption (iii) FALSIFIED' if second else 'not supported'}")

    # --- Z-CLOSE + W-CHECK ---------------------------------------------
    closed = None
    for b in grid:
        if b in means and abs(means[b]) > thr[b]:
            if all(abs(means[c]) > thr[c] for c in grid
                   if c >= b and c in means):
                closed = b
                break
    print(f"\nZ-CLOSE: zone closes at b={closed}" if closed else
          "\nZ-CLOSE: the zone NEVER CLOSES by 12000 — the comparison "
          "remains statistically undetermined through the whole budget")
    if len(sig) >= 2:
        bs = [b for b, _ in sig]
        slope = ((means[bs[-1]] - means[bs[0]]) / (bs[-1] - bs[0])) or 1e-12
        sd_mid = st.mean([sds[b] for b in bs])
        w_pred = 2 * T_CRIT * sd_mid / (math.sqrt(len(seeds)) * abs(slope))
        print(f"W-CHECK: predicted zone width ~{w_pred:.0f} steps "
              f"(SD {sd_mid:.4f}, slope {slope:+.2e}/step, n={len(seeds)})")

    print("\nScope: gpt2-nano d=256 L=2, T=256, batch 4, lr 1e-3, w=64 "
          "sinks=1, TinyStories slice, CUDA venue, 12000 steps. A second "
          "crossing here falsifies THEOREM_CROSSING (iii) AT THIS "
          "PROTOCOL; its absence bounds the monotone regime to 12000 "
          "steps at this scale, not forever.")


if __name__ == "__main__":
    main()
