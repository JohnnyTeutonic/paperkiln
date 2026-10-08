#!/usr/bin/env python3
"""POST-HOC, DESCRIPTIVE (9 Oct 2026): six-lane rankings, top-choice transfer and
selection regret across widths, and ties separated from reversals.

H-SCALAR correlates fifteen signed pairwise differences, which is not a ranking
of the six lanes and depends on how each edge is oriented. This script reports
the quantity a screen actually acts on: each arm's lane ranking by seed-mean
validation loss, the lane the small width would pick, where that lane ranks at
the larger width, and how much loss picking it costs there (regret). It also
splits each milestone's sign disagreements into reversals (opposite non-zero
majorities) and tie-involved cells (a zero majority on at least one side).

It imports the frozen ../analyze.py readers, milestone rule and sign logic and
changes no frozen script or receipt.

usage: python3 lane_ranking.py --S1 <root> --M1 <root> --S2 <root> --M2 <root>
A root holds run directories, directly or under runs/.
"""
import argparse
import glob
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analyze as A  # noqa: E402  (the frozen script)

DRAWS = 2000
SEED = 20261009


def read_arm(root):
    dirs = sorted(glob.glob(os.path.join(root, "runs", "*"))) or sorted(glob.glob(os.path.join(root, "*")))
    arm = {}
    for d in dirs:
        if os.path.exists(os.path.join(d, "events.jsonl")):
            r = A.read_run(d)
            arm.setdefault(r["seed"], {})[r["lane"]] = r
    return arm


def lane_losses(arm, position, target):
    """{lane: [per-seed val loss]} at a matched milestone (target loss) or a fixed step."""
    out = {lane: [] for lane in A.LANES}
    for seed, lanes in arm.items():
        if target is None:
            step = position
        else:
            ex = lanes.get("exact")
            step = A.step_at_milestone(ex["evals"], target) if ex else None
        if step is None:
            continue
        for lane in A.LANES:
            r = lanes.get(lane)
            v = A.val_at(r["evals"], step) if r else None
            if v is not None:
                out[lane].append(v)
    return out


def ranking(means):
    return sorted(means, key=lambda lane: means[lane])


def kendall_tau(order_a, order_b):
    pos_b = {lane: i for i, lane in enumerate(order_b)}
    conc = disc = 0
    for i in range(len(order_a)):
        for j in range(i + 1, len(order_a)):
            if pos_b[order_a[i]] < pos_b[order_a[j]]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc)


def summarise(lossS, lossM):
    meanS = {lane: st.mean(v) for lane, v in lossS.items() if v}
    meanM = {lane: st.mean(v) for lane, v in lossM.items() if v}
    rS, rM = ranking(meanS), ranking(meanM)
    pick = rS[0]
    return dict(rankS=rS, rankM=rM, tau=kendall_tau(rS, rM), pick=pick,
                pick_rank_M=rM.index(pick) + 1, regret=meanM[pick] - meanM[rM[0]],
                spread_M=meanM[rM[-1]] - meanM[rM[0]], meanS=meanS, meanM=meanM,
                nS=min(len(v) for v in lossS.values()), nM=min(len(v) for v in lossM.values()))


def bootstrap_top(lossS, lossM, rng):
    """Share of seed-bootstrap draws in which the small-width pick is also the larger-width best."""
    nS, nM = len(next(iter(lossS.values()))), len(next(iter(lossM.values())))
    same = 0
    for _ in range(DRAWS):
        iS = [rng.randrange(nS) for _ in range(nS)]
        iM = [rng.randrange(nM) for _ in range(nM)]
        mS = {lane: st.mean(v[i] for i in iS) for lane, v in lossS.items()}
        mM = {lane: st.mean(v[i] for i in iM) for lane, v in lossM.items()}
        same += ranking(mS)[0] == ranking(mM)[0]
    return same / DRAWS


def ties_and_reversals(armS, armM, ms):
    def stepper(arm, seed, label):
        target = dict(ms).get(label)
        r = arm.get(seed, {}).get("exact")
        return None if target is None or r is None else A.step_at_milestone(r["evals"], target)
    mS, _ = A.sign_matrix(armS, ms, stepper)
    mM, _ = A.sign_matrix(armM, ms, stepper)
    rows = []
    for label, _ in ms:
        keys = [k for k in set(mS) & set(mM) if k[1] == label]
        agree = sum(1 for k in keys if mS[k] == mM[k])
        tie_tie = sum(1 for k in keys if mS[k] == mM[k] == 0)
        reversal = sum(1 for k in keys if mS[k] * mM[k] == -1)
        tie_inv = sum(1 for k in keys if mS[k] != mM[k] and (mS[k] == 0 or mM[k] == 0))
        rows.append((label, len(keys), agree, tie_tie, reversal, tie_inv))
    return rows


def report(name, armS, armM, ms, rng, admitted):
    print(f"\n##### {name}: seeds S={len(armS)} M={len(armM)}; milestones from S: "
          + ", ".join(f"{sl}:{v:.4f}" for sl, v in ms))
    positions = [(f"milestone {sl} (loss {v:.4f})", sl, v) for sl, v in ms] + [("fixed step 3600", 3600, None)]
    print("position | seeds S/M | S ranking (best first) | M ranking | Kendall tau | "
          "S pick: rank at M, regret at M (loss), M best-to-worst spread | bootstrap P(S pick = M best)")
    for label, pos, target in positions:
        lossS = lane_losses(armS, pos, target)
        lossM = lane_losses(armM, pos, target)
        if any(not v for v in lossS.values()) or any(not v for v in lossM.values()):
            print(f"{label} | not reached by every lane in both arms")
            continue
        s = summarise(lossS, lossM)
        # bootstrap needs complete seed rows: keep seeds present for every lane
        p = bootstrap_top(lossS, lossM, rng) if len({len(v) for v in lossS.values()}) == 1 \
            and len({len(v) for v in lossM.values()}) == 1 else float("nan")
        note = "" if target is None or pos in admitted else "  [partially reached; outside the registered band]"
        print(f"{label} | {s['nS']}/{s['nM']} | {' > '.join(s['rankS'])} | {' > '.join(s['rankM'])} | "
              f"{s['tau']:+.3f} | {s['pick']}: rank {s['pick_rank_M']} of 6, regret {s['regret']:.4f}, "
              f"spread {s['spread_M']:.4f} | {p:.3f}{note}")
    print("\nsign disagreements by milestone: cells | agree (of which tie-tie) | reversals | tie-involved")
    for label, n, agree, tt, rev, tie in ties_and_reversals(armS, armM, ms):
        print(f"  milestone {label}: {n} | {agree} ({tt}) | {rev} | {tie}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("S1", "M1", "S2", "M2"):
        ap.add_argument("--" + k, required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    print("POST-HOC, descriptive (lane_ranking.py, 9 Oct 2026). Lane loss = seed-mean validation loss "
          "at each seed's exact-lane milestone step (or at step 3600); regret = larger-width loss of the "
          f"small-width pick minus the larger-width best; bootstrap {DRAWS} draws, seed {SEED}.")
    S1, M1 = read_arm(args.S1), read_arm(args.M1)
    ms1 = A.milestones_from_S(S1)
    report("Study 1 (shared rate 1e-3)", S1, M1, ms1, rng, admitted={800})
    S2, M2 = read_arm(args.S2), read_arm(args.M2)
    ms2 = A.milestones_from_S(S2)
    report("Study 2 (rate per width)", S2, M2, ms2, rng, admitted={800, 1600, 2400, 3200})


if __name__ == "__main__":
    main()
