"""Pre-registered analysis for transfer_s1: do fingerprints transfer?

WRITTEN AND COMMITTED WITH PREREGISTRATION.md, BEFORE ANY RUN. Every
rule below is fixed by that document; the licence anchor is the commit
that introduced these two files together.

  FINGERPRINT   the complete pairwise sign matrix over the six lanes
                (15 edges), as a DISTRIBUTION over seeds.
  F1 (primary)  sign-pattern concordance S vs M; "structure transfers"
                iff concordance >= 0.75 AND bootstrap p2.5 > 0.50.
  F2            shape-class concordance (flat band 0.02 = 1 SD of the
                tightest S slice, measured in sparse_s1_seeds).
  F3            b0 distributions incl. the never-crossed mass.
  H-SCALAR      Spearman rho of scalar Delta across arms over 15 edges;
                "scalars don't transfer" iff |rho| < 0.5 or CI spans 0.
                DESCRIPTIVE FOIL — n=15 edges cannot carry a headline.
  HEADLINE      "scalars don't transfer, structure does" licensed ONLY
                if F1 clears AND H-SCALAR meets its condition.
  POSITIONS     primary = matched val-loss milestones on lane L1;
                robustness = the nine 400-step slices.
  THREATS       bridge gate (halts the study), regime check, protocol
                drift, refuse-to-run (lanes from model events only).

Usage:
    python3 analyze.py --arms S=/path/S M=/path/M [L=/path/L]
    python3 analyze.py --bridge /path/bridge --banked /path/boundary
"""
import argparse
import glob
import json
import math
import os
import random
import statistics as st

SLICES = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600]
MILESTONE_SLICES = [800, 1600, 2400, 3200, 3600]
FLAT_BAND = 0.02          # 1 SD of the tightest S-arm slice (0.0179)
F1_THRESHOLD = 0.75
F1_CHANCE = 0.50
SCALAR_RHO_MAX = 0.5
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 12345    # fixed: the bootstrap must be reproducible
LANES = ["exact", "swa16s1", "swa32s1", "swa64s1", "swa128s1", "swa64s0"]


def lane_key(model):
    """Lane identity from the MODEL EVENT ONLY (refuse-to-run guard)."""
    att = model.get("attention")
    if att == "exact":
        return "exact"
    if att == "swa":
        return f"swa{int(model.get('window', -1))}s{int(model.get('sinks', -1))}"
    return f"UNKNOWN:{att}"


def read_run(d):
    evals, model = [], None
    with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "model":
                model = ev            # last segment wins (resume rule)
            elif ev.get("event") == "eval":
                evals.append((int(ev["step"]), float(ev["val_loss"])))
    if model is None:
        raise SystemExit(f"{d}: no model event (refuse-to-run guard)")
    evals.sort()
    dedup = {}
    for s, v in evals:
        dedup[s] = v                  # last occurrence wins (resume rule)
    evals = sorted(dedup.items())
    return {"lane": lane_key(model), "seed": int(model["seed"]),
            "d": int(model["d"]), "layers": int(model["layers"]),
            "evals": evals}


def read_arm(root):
    arm = {}
    for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        if not os.path.exists(os.path.join(d, "events.jsonl")):
            continue
        r = read_run(d)
        arm.setdefault(r["seed"], {})[r["lane"]] = r
    return arm


def val_at(evals, step):
    """Val loss at the last eval at or before `step`."""
    vals = [v for s, v in evals if s <= step]
    return vals[-1] if vals else None


def step_at_milestone(evals, target):
    """First eval step whose val loss has fallen to/below `target`."""
    for s, v in evals:
        if v <= target:
            return s
    return None


def edges():
    out = []
    for i in range(len(LANES)):
        for j in range(i + 1, len(LANES)):
            out.append((LANES[i], LANES[j]))
    return out


def delta(arm, seed, a, b, step):
    """loss_b - loss_a at `step`; positive => lane a is better."""
    ra, rb = arm.get(seed, {}).get(a), arm.get(seed, {}).get(b)
    if not ra or not rb:
        return None
    va, vb = val_at(ra["evals"], step), val_at(rb["evals"], step)
    if va is None or vb is None:
        return None
    return vb - va


def majority_sign(vals):
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    if pos == neg:
        return 0
    return 1 if pos > neg else -1


def milestones_from_S(armS):
    """L1 median trajectory values at the milestone slices."""
    out = []
    for sl in MILESTONE_SLICES:
        vals = []
        for seed, lanes in armS.items():
            r = lanes.get("exact")
            if r:
                v = val_at(r["evals"], sl)
                if v is not None:
                    vals.append(v)
        if vals:
            out.append((sl, st.median(vals)))
    return out


def sign_matrix(arm, positions, per_seed_step):
    """{(edge, pos): majority sign} plus the per-seed deltas behind it."""
    mat, raw = {}, {}
    for (a, b) in edges():
        for pos_label, _ in positions:
            vals = []
            for seed in sorted(arm):
                step = per_seed_step(arm, seed, pos_label)
                if step is None:
                    continue
                dv = delta(arm, seed, a, b, step)
                if dv is not None:
                    vals.append(dv)
            if vals:
                mat[((a, b), pos_label)] = majority_sign(vals)
                raw[((a, b), pos_label)] = vals
    return mat, raw


def concordance(m1, m2):
    keys = set(m1) & set(m2)
    if not keys:
        return 0.0, 0
    agree = sum(1 for k in keys if m1[k] == m2[k])
    return agree / len(keys), len(keys)


def bootstrap_concordance(rawS, rawM, n=BOOTSTRAP_N):
    """Resample SEEDS with replacement within each arm, recompute the
    majority-sign matrices and their concordance. The interval is a
    seed-noise band, NOT a p-value: cells are correlated across
    positions within an edge (stated in PREREGISTRATION.md)."""
    rng = random.Random(BOOTSTRAP_SEED)
    keys = sorted(set(rawS) & set(rawM), key=str)
    if not keys:
        return []
    nS = len(next(iter(rawS.values())))
    nM = len(next(iter(rawM.values())))
    out = []
    for _ in range(n):
        iS = [rng.randrange(nS) for _ in range(nS)]
        iM = [rng.randrange(nM) for _ in range(nM)]
        agree = 0
        for k in keys:
            vs, vm = rawS[k], rawM[k]
            if len(vs) != nS or len(vm) != nM:
                continue
            s1 = majority_sign([vs[i] for i in iS])
            s2 = majority_sign([vm[i] for i in iM])
            agree += (s1 == s2)
        out.append(agree / len(keys))
    out.sort()
    return out


def shape_class(traj):
    """traj: [Delta at each of the nine slices] -> class label."""
    if all(abs(v) < FLAT_BAND for v in traj):
        return "flat"
    ups = sum(1 for i in range(1, len(traj)) if traj[i] > traj[i - 1])
    downs = sum(1 for i in range(1, len(traj)) if traj[i] < traj[i - 1])
    if ups >= 7 and traj[-1] > 0:
        return "monotone+"
    if downs >= 7 and traj[-1] < 0:
        return "monotone-"
    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in traj]
    changes = sum(1 for i in range(1, len(signs))
                  if signs[i] != 0 and signs[i - 1] != 0
                  and signs[i] != signs[i - 1])
    if changes == 1:
        return "single-crossing"
    return "other"


def b0_of(traj):
    """Smallest slice index from which Delta stays positive; else None."""
    for i in range(len(traj)):
        if all(traj[j] > 0 for j in range(i, len(traj))):
            return SLICES[i]
    return None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def bridge_gate(bridge_root, banked_root):
    """Threat 1. Per-seed sign agreement >= 4/5 AND pooled mean within
    2 SE of banked. Failure HALTS the study."""
    print("=" * 68)
    print("THREAT 1 — NUMERICS BRIDGE (gate; the panel is blocked on it)")
    print("=" * 68)
    cuda, cpu = read_arm(bridge_root), read_arm(banked_root)
    seeds = sorted(set(cuda) & set(cpu))
    dc, dp = [], []
    for s in seeds:
        a = delta(cuda, s, "exact", "swa64s1", 3600)
        b = delta(cpu, s, "exact", "swa64s1", 3600)
        if a is None or b is None:
            continue
        dc.append(a)
        dp.append(b)
        print(f"  seed {s:>2}: cuda {a:+.5f}   cpu {b:+.5f}   "
              f"{'AGREE' if (a > 0) == (b > 0) else 'DISAGREE'}")
    if not dc:
        print("  NO COMPARABLE RUNS — gate cannot be evaluated")
        return False
    agree = sum(1 for a, b in zip(dc, dp) if (a > 0) == (b > 0))
    mc, mp = st.mean(dc), st.mean(dp)
    se = (st.stdev(dp) / math.sqrt(len(dp))) if len(dp) > 1 else float("inf")
    within = abs(mc - mp) <= 2 * se
    print(f"  sign agreement {agree}/{len(dc)} (need >= 4/5)")
    print(f"  pooled mean cuda {mc:+.5f} vs cpu {mp:+.5f} "
          f"(2 SE = {2 * se:.5f}) -> {'within' if within else 'OUTSIDE'}")
    ok = agree >= 4 and within
    print(f"  VERDICT: {'PASS — panel may run' if ok else 'FAIL — STUDY HALTS'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[],
                    help="NAME=/path pairs, e.g. S=/out/S M=/out/M")
    ap.add_argument("--bridge")
    ap.add_argument("--banked")
    args = ap.parse_args()

    if args.bridge and args.banked:
        bridge_gate(args.bridge, args.banked)
        if not args.arms:
            return

    arms = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        arms[name] = read_arm(path)
    if "S" not in arms or "M" not in arms:
        print("need at least S= and M= arms for the transfer analysis")
        return

    print("=" * 68)
    print("transfer_s1 PRE-REGISTERED ANALYSIS — does structure transfer?")
    print("=" * 68)

    # --- Threat 2 (regime) and Threat 3 (protocol drift) ---------------
    for name, arm in arms.items():
        n_ok = 0
        tot = 0
        for seed, lanes in arm.items():
            for lane, r in lanes.items():
                tot += 1
                ev = r["evals"]
                if not ev:
                    continue
                best = min(v for _, v in ev)
                tail = [v for _, v in ev[-3:]]
                if best in tail:
                    n_ok += 1
        print(f"Threat 2 (regime) {name}: best_val within last 3 evals in "
              f"{n_ok}/{tot} runs")
    drift = [delta(arms["M"], s, "exact", "swa64s1", 1200)
             for s in sorted(arms["M"])]
    drift = [d for d in drift if d is not None]
    neg = sum(1 for d in drift if d < 0)
    print(f"Threat 3 (protocol drift) M: Delta(1200) negative "
          f"{neg}/{len(drift)} (need >= 8/12; S arm precedent 15/15)")

    # --- positions -----------------------------------------------------
    ms = milestones_from_S(arms["S"])
    print("\nMatched val-loss milestones (from S arm L1 median):")
    for sl, v in ms:
        print(f"  slice {sl:>4} -> val {v:.4f}")

    def step_milestone(arm, seed, label):
        target = dict(ms).get(label)
        r = arm.get(seed, {}).get("exact")
        if target is None or r is None:
            return None
        return step_at_milestone(r["evals"], target)

    def step_slice(arm, seed, label):
        return label

    positions_primary = [(sl, v) for sl, v in ms]
    positions_robust = [(sl, None) for sl in SLICES]

    for label, positions, stepper in (
            ("PRIMARY (matched val-loss milestones)", positions_primary,
             step_milestone),
            ("ROBUSTNESS (fixed 400-step slices)", positions_robust,
             step_slice)):
        print("\n" + "-" * 68)
        print(f"F1 — SIGN-PATTERN CONCORDANCE :: {label}")
        matS, rawS = sign_matrix(arms["S"], positions, stepper)
        matM, rawM = sign_matrix(arms["M"], positions, stepper)
        conc, ncells = concordance(matS, matM)
        boot = bootstrap_concordance(rawS, rawM)
        lo = boot[int(0.025 * len(boot))] if boot else float("nan")
        hi = boot[int(0.975 * len(boot))] if boot else float("nan")
        print(f"  concordance S vs M = {conc:.3f} over {ncells} cells")
        print(f"  seed bootstrap 95% band [{lo:.3f}, {hi:.3f}] "
              f"(noise band, NOT a p-value — cells correlate within edge)")
        verdict = conc >= F1_THRESHOLD and lo > F1_CHANCE
        word = "STRUCTURE TRANSFERS" if verdict else "not adopted"
        print(f"  F1 VERDICT: {word} "
              f"(need >= {F1_THRESHOLD} and band low > {F1_CHANCE})")
        if "L" in arms:
            matL, _ = sign_matrix(arms["L"], positions, stepper)
            cSL, nSL = concordance(matS, matL)
            cML, nML = concordance(matM, matL)
            print(f"  [preliminary, 3 seeds] S vs L {cSL:.3f} ({nSL} cells), "
                  f"M vs L {cML:.3f} ({nML} cells)")

    # --- F2 shape classes ----------------------------------------------
    print("\n" + "-" * 68)
    print("F2 — SHAPE-CLASS CONCORDANCE (nine 400-step slices)")
    modal = {}
    for name in ("S", "M"):
        arm = arms[name]
        modal[name] = {}
        for (a, b) in edges():
            classes = []
            for seed in sorted(arm):
                traj = [delta(arm, seed, a, b, s) for s in SLICES]
                if any(t is None for t in traj):
                    continue
                classes.append(shape_class(traj))
            if classes:
                modal[name][(a, b)] = max(set(classes), key=classes.count)
    common = set(modal["S"]) & set(modal["M"])
    agree = sum(1 for e in common if modal["S"][e] == modal["M"][e])
    print(f"  modal-class agreement {agree}/{len(common)} edges")
    for e in sorted(common, key=str):
        flag = "" if modal["S"][e] == modal["M"][e] else "   <-- differs"
        print(f"    {e[0]:>9} vs {e[1]:<9} S={modal['S'][e]:<15} "
              f"M={modal['M'][e]:<15}{flag}")

    # --- F3 b0 distributions -------------------------------------------
    print("\n" + "-" * 68)
    print("F3 — B* AS A DISTRIBUTION (exploratory; never-crossed is a "
          "category, not a missing value)")
    for name in ("S", "M"):
        arm = arms[name]
        for (a, b) in [("exact", "swa64s1")]:
            b0s, never = [], 0
            for seed in sorted(arm):
                traj = [delta(arm, seed, a, b, s) for s in SLICES]
                if any(t is None for t in traj):
                    continue
                v = b0_of(traj)
                if v is None:
                    never += 1
                else:
                    b0s.append(v)
            med = st.median(b0s) if b0s else float("nan")
            print(f"  {name} {a} vs {b}: crossed {len(b0s)}, never {never}, "
                  f"median b0 {med}")

    # --- H-SCALAR -------------------------------------------------------
    print("\n" + "-" * 68)
    print("H-SCALAR — the committed foil (DESCRIPTIVE; n=15 edges)")
    final = MILESTONE_SLICES[-1]
    xs, ys = [], []
    for (a, b) in edges():
        vS = [delta(arms["S"], s, a, b, final) for s in sorted(arms["S"])]
        vM = [delta(arms["M"], s, a, b, final) for s in sorted(arms["M"])]
        vS = [v for v in vS if v is not None]
        vM = [v for v in vM if v is not None]
        if vS and vM:
            xs.append(st.mean(vS))
            ys.append(st.mean(vM))
    rho = spearman(xs, ys)
    print(f"  Spearman rho over {len(xs)} edges = "
          f"{'n/a' if rho is None else f'{rho:+.3f}'}")
    scal = rho is not None and abs(rho) < SCALAR_RHO_MAX
    print(f"  'scalars don't transfer' condition: "
          f"{'MET' if scal else 'not met'} (|rho| < {SCALAR_RHO_MAX})")
    print("\n  HEADLINE LICENCE requires F1 adopted AND this condition met;")
    print("  if F1 is adopted and scalars ALSO transfer, the claim is the")
    print("  weaker 'tiny-scale screening transfers, scalars included'.")

    print("\nScope: gpt2-nano family, layers=2, T=256, batch 4, lr 1e-3 "
          "FIXED across widths (declared protocol property, not a muP "
          "answer), TinyStories slice + chat7b vocab, CUDA venue, "
          "d in {256, 512, 1024}. Fingerprints are properties of THIS "
          "protocol across THIS width range.")


if __name__ == "__main__":
    main()
