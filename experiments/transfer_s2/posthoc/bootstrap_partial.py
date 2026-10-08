#!/usr/bin/env python3
"""POST-HOC, DESCRIPTIVE: the Study-2 seed bootstrap with partially reached
milestone cells handled per seed (27 Sep 2026, final pre-submission check).

The frozen script (../analyze.py, licence anchor 90791ed) resamples twelve
seed indices per arm and, in each cell, skips the cell if its value count
differs from the arm's seed count -- but keeps it in the denominator. At
the last Study-2 milestone the S arm reaches the milestone in 8 of 12
seeds, so all fifteen cells there are skipped in every draw and counted
as disagreements. The printed band [0.467, 0.667] is therefore a band
around 42/75 = 0.560, not around the point estimate 47/75 = 0.627.

This script (1) reproduces the frozen band verbatim, (2) recomputes it by
drawing seed LABELS per arm and reading each cell from the drawn seeds
that have a value there, (3) prints the per-edge orientation at every
milestone for both arms (appendix table), and (4) prints the H-SCALAR
inputs. It imports the frozen script's own readers and sign logic, and it
changes nothing in that script. The verdict does not depend on either
band: the adoption rule fails on the point estimate (0.627 < 0.75).

Usage (from this directory):
  python3 bootstrap_partial.py --S ../receipts/s2_S --M ../receipts/s2_M
Each root holds run_*/events.jsonl (the receipt layout) or runs/*/events.jsonl.
"""
import argparse
import glob
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analyze as A  # noqa: E402  (the frozen script)


def read_arm_any(root):
    """Receipt layout (root/run_*) or artefact layout (root/runs/*)."""
    dirs = sorted(glob.glob(os.path.join(root, "run_*"))) or \
        sorted(glob.glob(os.path.join(root, "runs", "*")))
    arm = {}
    for d in dirs:
        if os.path.exists(os.path.join(d, "events.jsonl")):
            r = A.read_run(d)
            arm.setdefault(r["seed"], {})[r["lane"]] = r
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", required=True)
    ap.add_argument("--M", required=True)
    args = ap.parse_args()
    S, M = read_arm_any(args.S), read_arm_any(args.M)
    ms = A.milestones_from_S(S)
    print("seeds S=%d M=%d; milestones: %s" % (
        len(S), len(M), ", ".join("%d:%.4f" % (sl, v) for sl, v in ms)))

    def stepper(arm, seed, label):
        target = dict(ms).get(label)
        r = arm.get(seed, {}).get("exact")
        if target is None or r is None:
            return None
        return A.step_at_milestone(r["evals"], target)

    positions = [(sl, v) for sl, v in ms]
    matS, rawS = A.sign_matrix(S, positions, stepper)
    matM, rawM = A.sign_matrix(M, positions, stepper)
    conc, n = A.concordance(matS, matM)
    print("point estimate %.3f over %d cells" % (conc, n))
    for sl, _ in ms:
        keys = [k for k in matS if k[1] == sl and k in matM]
        ag = sum(1 for k in keys if matS[k] == matM[k])
        print("  milestone %4d: %2d/%d = %.3f  (seeds per cell S=%d M=%d)" % (
            sl, ag, len(keys), ag / len(keys), len(rawS[keys[0]]), len(rawM[keys[0]])))

    # (1) the frozen bootstrap, verbatim
    boot = A.bootstrap_concordance(rawS, rawM)
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    skipped = sum(1 for k in rawS if k in rawM and
                  (len(rawS[k]) != len(S) or len(rawM[k]) != len(M)))
    print("frozen band [%.3f, %.3f], median %.3f; cells skipped in every draw "
          "(counted as disagreements): %d" % (lo, hi, st.median(boot), skipped))

    # (2) per-seed-label bootstrap
    def by_seed(arm):
        out = {}
        for (a, b) in A.edges():
            for sl, _ in positions:
                per = {}
                for seed in sorted(arm):
                    step = stepper(arm, seed, sl)
                    if step is None:
                        continue
                    dv = A.delta(arm, seed, a, b, step)
                    if dv is not None:
                        per[seed] = dv
                out[((a, b), sl)] = per
        return out
    bS, bM = by_seed(S), by_seed(M)
    keys = sorted(set(bS) & set(bM), key=str)
    rng = random.Random(A.BOOTSTRAP_SEED)
    seedsS, seedsM = sorted(S), sorted(M)
    vals = []
    for _ in range(A.BOOTSTRAP_N):
        dS = [rng.choice(seedsS) for _ in seedsS]
        dM = [rng.choice(seedsM) for _ in seedsM]
        agree = 0
        for k in keys:
            vs = [bS[k][s] for s in dS if s in bS[k]]
            vm = [bM[k][s] for s in dM if s in bM[k]]
            if not vs or not vm:
                continue
            agree += A.majority_sign(vs) == A.majority_sign(vm)
        vals.append(agree / len(keys))
    vals.sort()
    print("per-seed band [%.3f, %.3f], median %.3f  (POST-HOC; the frozen band "
          "is the licensed one)" % (vals[int(0.025 * len(vals))],
                                     vals[int(0.975 * len(vals))], st.median(vals)))

    # (3) per-edge orientation at every milestone
    sym = {1: "+", -1: "-", 0: "0", None: "."}
    print("\nper-edge orientation at milestones %s (+ = first lane better)"
          % [sl for sl, _ in ms])
    for (a, b) in A.edges():
        rs = "".join(sym[matS.get(((a, b), sl))] for sl, _ in ms)
        rm = "".join(sym[matM.get(((a, b), sl))] for sl, _ in ms)
        fl = "".join("=" if matS.get(((a, b), sl)) == matM.get(((a, b), sl)) else "x"
                     for sl, _ in ms)
        print("  %9s vs %-9s S %s  M %s  %s" % (a, b, rs, rm, fl))

    # (4) H-SCALAR inputs at the final step
    final = A.MILESTONE_SLICES[-1]
    xs, ys = [], []
    print("\nH-SCALAR inputs (seed-mean delta at step %d)" % final)
    for (a, b) in A.edges():
        vS = [A.delta(S, s, a, b, final) for s in sorted(S)]
        vM = [A.delta(M, s, a, b, final) for s in sorted(M)]
        xs.append(st.mean([v for v in vS if v is not None]))
        ys.append(st.mean([v for v in vM if v is not None]))
        print("  %9s vs %-9s S %+.4f  M %+.4f" % (a, b, xs[-1], ys[-1]))
    print("Spearman rho = %+.3f" % A.spearman(xs, ys))


if __name__ == "__main__":
    main()
