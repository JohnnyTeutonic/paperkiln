#!/usr/bin/env python3
"""POST-HOC, DESCRIPTIVE analyses for the transfer paper (14 Sep 2026).

Nothing here changes the frozen pre-registered analysis (../analyze.py);
every function below imports that script's own readers and sign logic so
the numbers are computed the same way. Three analyses, each labelled
post-hoc wherever it appears in the paper:

  1. per-milestone F1 (the frozen script reports the aggregate band)
  2. seeds-matter: F1 recomputed on random subsets of k seeds per arm
  3. per-edge sign table at the milestone every study reaches (800)

Usage:
  python3 posthoc.py --study s1 --S <root> --M <root>
  python3 posthoc.py --study s2 --S <root> --M <root> [--k 3 6 12] [--draws 2000]
Each root holds runs/<run>/events.jsonl.
"""
import argparse
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analyze as A  # noqa: E402  (the frozen script)


def milestone_stepper(ms):
    def step_milestone(arm, seed, label):
        target = dict(ms).get(label)
        r = arm.get(seed, {}).get("exact")
        if target is None or r is None:
            return None
        return A.step_at_milestone(r["evals"], target)
    return step_milestone


def f1_by_milestone(armS, armM, ms):
    stepper = milestone_stepper(ms)
    positions = [(sl, v) for sl, v in ms]
    mS, _ = A.sign_matrix(armS, positions, stepper)
    mM, _ = A.sign_matrix(armM, positions, stepper)
    out = []
    for sl, _ in ms:
        keys = [k for k in set(mS) & set(mM) if k[1] == sl]
        if not keys:
            out.append((sl, None, 0))
            continue
        agree = sum(1 for k in keys if mS[k] == mM[k])
        out.append((sl, agree / len(keys), len(keys)))
    total, n = A.concordance(mS, mM)
    return out, total, n


def subarm(arm, seeds):
    return {s: arm[s] for s in seeds if s in arm}


def seeds_matter(armS, armM, k, draws, rng):
    """F1 (aggregate over reached milestones) on random k-seed subsets of
    each arm, milestones recomputed from the S subset as a k-seed study
    would have done. Returns the list of F1 values."""
    seedsS, seedsM = sorted(armS), sorted(armM)
    vals = []
    for _ in range(draws):
        sS = rng.sample(seedsS, k)
        sM = rng.sample(seedsM, k)
        aS, aM = subarm(armS, sS), subarm(armM, sM)
        ms = A.milestones_from_S(aS)
        if not ms:
            continue
        _, total, n = f1_by_milestone(aS, aM, ms)
        if n:
            vals.append(total)
    return vals


def sign_table_800(armS, armM, ms):
    stepper = milestone_stepper(ms)
    positions = [(sl, v) for sl, v in ms if sl == 800]
    mS, _ = A.sign_matrix(armS, positions, stepper)
    mM, _ = A.sign_matrix(armM, positions, stepper)
    rows = []
    for (a, b) in A.edges():
        k = ((a, b), 800)
        rows.append((a, b, mS.get(k), mM.get(k)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--S", required=True)
    ap.add_argument("--M", required=True)
    ap.add_argument("--k", nargs="*", type=int, default=[3, 6, 12])
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    armS, armM = A.read_arm(args.S), A.read_arm(args.M)
    ms = A.milestones_from_S(armS)
    print(f"POST-HOC ({args.study}); seeds S={len(armS)} M={len(armM)}; "
          f"milestones from S: " + ", ".join(f"{sl}:{v:.4f}" for sl, v in ms))

    print("\n1. per-milestone F1 (POST-HOC; the pre-registered figure is the aggregate)")
    per, total, n = f1_by_milestone(armS, armM, ms)
    for sl, f, c in per:
        print(f"   milestone {sl:>4}: F1 = {f if f is None else round(f, 3)} over {c} cells")
    print(f"   aggregate: {total:.3f} over {n} cells (should equal the frozen script's primary)")

    print(f"\n2. seeds-matter (POST-HOC): F1 on random k-seed subsets, {args.draws} draws each")
    rng = random.Random(args.seed)
    for k in args.k:
        if k > min(len(armS), len(armM)):
            continue
        vals = seeds_matter(armS, armM, k, args.draws, rng)
        if not vals:
            continue
        vals.sort()
        lo, med, hi = vals[int(0.025 * len(vals))], st.median(vals), vals[int(0.975 * len(vals)) - 1]
        frac = sum(1 for v in vals if v >= 0.75) / len(vals)
        print(f"   k={k:>2}: median F1 {med:.3f}, 2.5-97.5 pct [{lo:.3f}, {hi:.3f}], "
              f"fraction of draws with F1 >= 0.75: {frac:.3f}  (n={len(vals)})")

    print("\n3. per-edge majority sign at milestone 800 (POST-HOC): +1 = first lane better")
    for a, b, sS, sM in sign_table_800(armS, armM, ms):
        flag = "" if sS == sM else "   <-- differs"
        print(f"   {a:>9} vs {b:<9}  S={sS:+d}  M={sM:+d}{flag}" if sS is not None and sM is not None
              else f"   {a:>9} vs {b:<9}  S={sS}  M={sM}")


if __name__ == "__main__":
    main()
