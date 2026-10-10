#!/usr/bin/env python3
"""Pre-registered analysis for transfer_s3: width and learning rate, crossed.

WRITTEN WITH PREREGISTRATION.md, BEFORE ANY transfer_s3 RUN. The licence
anchor is the commit that introduces this file and PREREGISTRATION.md
together; any later change is a dated amendment there.

It reuses transfer_s2's frozen readers and sign logic (../transfer_s2/
analyze.py, licence 90791ed) by import, and adds what Study 3 needs:

  TARGETS     four exact-lane validation-loss levels, FIXED here (Study 2's
              first four milestones), identical for every cell, so two
              cells are compared at the same loss whatever their rate.
  CELLS       width x rate: S25 S5 S10 (d=256 at 2.5e-4, 5e-4, 1e-3),
              M25 M5 M10 (d=512 likewise). S5 and M25 are new; the others
              are the banked Study 1 and Study 2 arms.
  GUARD       refuse-to-run: every run's model event must record the
              cell's d, layers=2 and learning rate; lanes from model
              events only.
  REACH       a target is admitted for a pair iff each cell reaches it
              (exact-lane first crossing) in >= 9 of 12 seeds.
  F1          sign concordance over the 15 edges, per target and over the
              admitted targets; seed bootstrap (10000 draws, seed 12345)
              that redraws seed labels per cell and reads each (edge,
              target) cell from the drawn seeds that have a value there.
  H-POS       per shared-rate pair (S25-M25, S5-M5): supported iff targets
              1 and 4 are admitted, F1(t1) >= 0.75 and F1(t4) < 0.75.
  H-RATE      per shared-width pair (S25-S5, M25-M5): supported iff F1
              over the admitted targets >= 0.75 and the band's 2.5th
              percentile > 0.50.
  READING     the joint table in PREREGISTRATION.md, applied mechanically.

Usage:
  python3 analyze.py --cells S25=<root> M5=<root> S10=<root> M10=<root> \
                             S5=<root> M25=<root>
A root holds run directories directly (receipt layout) or under runs/.
"""
import argparse
import glob
import json
import os
import random
import sys

import importlib.util  # noqa: E402

_S2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transfer_s2", "analyze.py")
_spec = importlib.util.spec_from_file_location("transfer_s2_analyze", _S2)
A = importlib.util.module_from_spec(_spec)   # transfer_s2's frozen script, by path
_spec.loader.exec_module(A)

TARGETS = [(1, 3.7539), (2, 3.5152), (3, 3.3491), (4, 3.2392)]
CELL_SPEC = {"S25": (256, 2.5e-4), "S5": (256, 5e-4), "S10": (256, 1e-3),
             "M25": (512, 2.5e-4), "M5": (512, 5e-4), "M10": (512, 1e-3)}
SHARED_RATE = [("S25", "M25"), ("S5", "M5")]
SHARED_WIDTH = [("S25", "S5"), ("M25", "M5")]
DESCRIPTIVE = [("S10", "M10"), ("S25", "M5")]   # Study 1's and Study 2's pairs
REACH_MIN = 9
F1_THRESHOLD = 0.75
F1_CHANCE = 0.50
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 12345


def read_cell(name, root):
    """Read a cell, enforcing the refuse-to-run guard on d, layers and lr."""
    dirs = sorted(glob.glob(os.path.join(root, "run_*"))) or \
        sorted(glob.glob(os.path.join(root, "runs", "*")))
    d_want, lr_want = CELL_SPEC[name]
    cell, bad = {}, []
    for rd in dirs:
        ev = os.path.join(rd, "events.jsonl")
        if not os.path.exists(ev):
            continue
        r = A.read_run(rd)
        lr = None
        with open(ev, encoding="utf-8") as f:
            for line in f:
                if '"event":"model"' in line.replace(" ", ""):
                    lr = json.loads(line).get("lr")
        if r["d"] != d_want or r["layers"] != 2 or lr is None or \
                abs(lr - lr_want) > 1e-6 * lr_want + 1e-9 or r["lane"] not in A.LANES:
            bad.append(os.path.basename(rd))
            continue
        cell.setdefault(r["seed"], {})[r["lane"]] = r
    if bad:
        raise SystemExit(f"refuse-to-run: {name} has {len(bad)} runs whose model "
                         f"event does not match d={d_want}, layers=2, lr={lr_want} "
                         f"or a known lane: {bad[:5]}")
    return cell


def stepper_for(cell):
    def step(arm, seed, label):
        r = cell.get(seed, {}).get("exact")
        return None if r is None else A.step_at_milestone(r["evals"], dict(TARGETS)[label])
    return step


def reach(cell):
    return {label: sum(1 for s in cell if cell[s].get("exact") and
                       A.step_at_milestone(cell[s]["exact"]["evals"], t) is not None)
            for label, t in TARGETS}


def raw_by_seed(cell):
    """{(edge, label): {seed: delta}} at each seed's exact-lane crossing step."""
    out = {}
    step = stepper_for(cell)
    for (a, b) in A.edges():
        for label, _ in TARGETS:
            vals = {}
            for s in sorted(cell):
                st_ = step(cell, s, label)
                if st_ is None:
                    continue
                dv = A.delta(cell, s, a, b, st_)
                if dv is not None:
                    vals[s] = dv
            if vals:
                out[((a, b), label)] = vals
    return out


def signs(raw, seeds=None):
    m = {}
    for k, vals in raw.items():
        v = list(vals.values()) if seeds is None else [vals[s] for s in seeds if s in vals]
        if v:
            m[k] = A.majority_sign(v)
    return m


def f1(mx, my, labels):
    keys = [k for k in set(mx) & set(my) if k[1] in labels]
    if not keys:
        return None, 0, 0, 0
    agree = sum(1 for k in keys if mx[k] == my[k])
    rev = sum(1 for k in keys if mx[k] * my[k] == -1)
    tie = sum(1 for k in keys if mx[k] != my[k] and (mx[k] == 0 or my[k] == 0))
    return agree / len(keys), len(keys), rev, tie


def band(rx, ry, sx, sy, labels, n=BOOTSTRAP_N):
    rng = random.Random(BOOTSTRAP_SEED)
    rx = {k: v for k, v in rx.items() if k[1] in labels}
    ry = {k: v for k, v in ry.items() if k[1] in labels}
    out = []
    for _ in range(n):
        dx = [rng.choice(sx) for _ in sx]
        dy = [rng.choice(sy) for _ in sy]
        v, _, _, _ = f1(signs(rx, dx), signs(ry, dy), labels)
        if v is not None:
            out.append(v)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def compare(x, y, cells, raws, with_band=True):
    rx, ry = reach(cells[x]), reach(cells[y])
    admitted = [lab for lab, _ in TARGETS if rx[lab] >= REACH_MIN and ry[lab] >= REACH_MIN]
    mx, my = signs(raws[x]), signs(raws[y])
    per = {lab: f1(mx, my, [lab]) for lab, _ in TARGETS}
    agg = f1(mx, my, admitted) if admitted else (None, 0, 0, 0)
    bnd = band(raws[x], raws[y], sorted(cells[x]), sorted(cells[y]), admitted) \
        if (with_band and admitted) else None
    return dict(reach=(rx, ry), admitted=admitted, per=per, agg=agg, band=bnd)


def fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


def report(x, y, c):
    rx, ry = c["reach"]
    print(f"\n  {x} vs {y}")
    print("    target  loss    reach " + f"{x:>4} {y:>4}" + "   F1     cells  reversals  ties")
    for lab, t in TARGETS:
        v, n, rv, ti = c["per"][lab]
        flag = "" if lab in c["admitted"] else "   (not admitted)"
        print(f"    t{lab}      {t:.4f}  {rx[lab]:>4} {ry[lab]:>4}   {fmt(v)}   {n:>3}    {rv:>3}      {ti:>3}{flag}")
    v, n, _, _ = c["agg"]
    b = c["band"]
    print(f"    over admitted targets {c['admitted']}: F1 {fmt(v)} over {n} cells"
          + (f", band [{b[0]:.3f}, {b[1]:.3f}]" if b else ""))


def h_pos(c):
    if 1 not in c["admitted"] or 4 not in c["admitted"]:
        return "untested (target 1 or 4 not admitted)"
    f_1, f_4 = c["per"][1][0], c["per"][4][0]
    return "supported" if (f_1 >= F1_THRESHOLD and f_4 < F1_THRESHOLD) else "not supported"


def h_rate(c):
    v = c["agg"][0]
    if v is None:
        return "untested (no admitted target)"
    return "supported" if (v >= F1_THRESHOLD and c["band"][0] > F1_CHANCE) else "not supported"


def joint(pos, rate):
    def both(xs, word):
        return all(x == word for x in xs)
    if any(x.startswith("untested") for x in pos + rate):
        return "UNTESTED: a pair lacks the targets its hypothesis needs; report pair by pair"
    if both(pos, "supported") and both(rate, "supported"):
        return "POSITION: agreement decays along the loss curve at a shared rate, and rate alone does not reorder the panel"
    if both(pos, "supported") and both(rate, "not supported"):
        return "BOTH: position decays agreement at a shared rate, and rate alone also reorders the panel"
    if both(pos, "not supported") and both(rate, "not supported"):
        return "RATE: agreement holds at a shared rate, and rate alone reorders the panel; Study 2's decline is a rate effect"
    if both(pos, "not supported") and both(rate, "supported"):
        return "NEITHER: agreement holds at a shared rate and across rates at a shared width; Study 2's decline needs both factors to move"
    return "MIXED: the two pairs of a hypothesis disagree; report pair by pair"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="NAME=root for all six cells")
    ap.add_argument("--no-band", action="store_true", help="skip the bootstrap (smoke tests only)")
    args = ap.parse_args()
    roots = dict(kv.split("=", 1) for kv in args.cells)
    missing = set(CELL_SPEC) - set(roots)
    if missing:
        raise SystemExit(f"missing cells: {sorted(missing)}")
    cells = {n: read_cell(n, r) for n, r in roots.items()}
    raws = {n: raw_by_seed(c) for n, c in cells.items()}
    print("=" * 72)
    print("transfer_s3 PRE-REGISTERED ANALYSIS (width x rate, fixed loss targets)")
    print("targets: " + ", ".join(f"t{lab}={t}" for lab, t in TARGETS))
    print("seeds per cell: " + ", ".join(f"{n}={len(c)}" for n, c in sorted(cells.items())))
    print("=" * 72)
    res = {}
    for title, pairs in (("H-POS, shared rate (width varies)", SHARED_RATE),
                         ("H-RATE, shared width (rate varies)", SHARED_WIDTH),
                         ("Descriptive: Study 1's and Study 2's pairs at these targets", DESCRIPTIVE)):
        print(f"\n{title}")
        for x, y in pairs:
            res[(x, y)] = compare(x, y, cells, raws, with_band=not args.no_band)
            report(x, y, res[(x, y)])
    pos = [h_pos(res[p]) for p in SHARED_RATE]
    rate = [h_rate(res[p]) if not args.no_band else
            ("supported" if (res[p]["agg"][0] or 0) >= F1_THRESHOLD else "not supported")
            for p in SHARED_WIDTH]
    print("\nVERDICTS")
    for p, v in zip(SHARED_RATE, pos):
        print(f"  H-POS  {p[0]}-{p[1]}: {v}")
    for p, v in zip(SHARED_WIDTH, rate):
        print(f"  H-RATE {p[0]}-{p[1]}: {v}")
    print(f"\nREADING: {joint(pos, rate)}")
    return pos, rate


if __name__ == "__main__":
    main()
