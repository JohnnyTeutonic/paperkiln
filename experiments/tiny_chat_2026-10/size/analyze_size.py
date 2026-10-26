#!/usr/bin/env python3
"""Amendment 6 (PROPOSED, not yet registered): the size arm's analysis on Carry-5. Y against log2 of the
non-embedding parameter count across 20M, 60M and 150M at one depth (size/depth_choice.json), paired by seed,
with analyze.py's decide (section 7: one-sample t over seeds, n = 3, critical 4.303; the extension rule).

    python experiments/tiny_chat_2026-10/size/analyze_size.py --art C:/ml_artifacts/tiny_chat \
        --depth-choice experiments/tiny_chat_2026-10/size/depth_choice.json --out experiments/tiny_chat_2026-10/size/analysis_size.json

Blocks read: the 20M cell at the depth from ART/e1 (L2, L4, L16, L32) or ART/e5's s35 cell (L8, the same runs as
E1's L8 cell); the 60M cell from ART/e1_60m and its per-cell folders; the 150M cell from ART/e1_150m and its
per-seed folders (e1_150m_r1 ...). Refuses incomplete cells unless --partial (DESCRIPTIVE, PARTIAL, no verdict).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load("tiny_chat_analyze", os.path.join(EXP, "analyze.py"))

SESOI_SIZE = 0.01        # A6.4: 1 point of Y per doubling of non-embedding parameters (E1's per-doubling SESOI)
SESOI_LOSS = 0.01        # nats per doubling (section 3's panel-loss SESOI per doubling)
SIZES = ["20m", "60m", "150m"]
KEYS = ["Y", "Y_gold", "multi_self", "twin_self", "twin_minus_qalone_ctx", "panel_fineweb", "panel_dialogue", "sft_val"]


def sources(art, size, depth):
    """[(block folder, cell)] holding the size's cell at DEPTH. At 20M the L8 cell is E5's s35 cell (section 6)."""
    if size == "20m":
        return [("e5", "s35")] if depth == 8 else [("e1", f"L{depth}")]
    pat = "e1_60m*" if size == "60m" else "e1_150m*"
    return [(os.path.basename(d), f"L{depth}") for d in sorted(glob.glob(os.path.join(art, pat))) if os.path.isdir(d)]


def load_size(art, size, depth):
    """{seed: summary} for the size's cell at DEPTH, from every folder that holds it (split machines)."""
    out = {}
    for b, cell in sources(art, size, depth):
        for (c, seed), s in an.load(art, b).items():
            if c != cell:
                continue
            if seed in out:
                raise SystemExit(f"{size} {cell} seed {seed} found twice (second in {b}); keep one copy")
            out[seed] = s
    return out


def nonemb_of(summaries):
    ns = {(s.get("shape") or {}).get("nonemb") for s in summaries.values()}
    if len(ns) != 1 or None in ns:
        raise SystemExit(f"the cell's runs disagree on the non-embedding count, or lack it: {ns}")
    return ns.pop()


def floor_ok(ss, seeds):
    runs = {("c", r): ss[r] for r in seeds}
    return an.floor_ok(runs, "c", seeds)


def size_block(art, depth, partial=False):
    data = {z: load_size(art, z, depth) for z in SIZES}
    seeds = an.SEEDS_FIRST
    ext = [r for r in an.SEEDS_EXT if all(r in data[z] for z in SIZES)]
    if len(ext) == len(an.SEEDS_EXT):
        seeds = an.SEEDS_FIRST + an.SEEDS_EXT
    miss = [(z, r) for z in SIZES for r in seeds if r not in data[z] or data[z][r].get("primary") is None]
    res = {"depth": depth, "seeds": seeds, "missing": miss}
    have = {z: [r for r in seeds if r in data[z]] for z in SIZES}
    res["cells"] = {z: {k: _mean([an.get(data[z][r], k) for r in have[z]]) for k in KEYS} for z in SIZES}
    if miss and not partial:
        raise SystemExit(f"size arm incomplete: {miss} (use --partial for a descriptive read)")
    if miss:
        res["label"] = "DESCRIPTIVE, PARTIAL"
        return res
    n = {z: nonemb_of(data[z]) for z in SIZES}
    tot = {z: {(s.get("shape") or {}).get("total") for s in data[z].values()}.pop() for z in SIZES}
    x = [math.log2(n[z]) for z in SIZES]
    w = an.slope_weights(x)
    res["nonemb"], res["total"], res["weights_log2_nonemb"] = n, tot, w

    def per_seed(key, wts):
        return [sum(wi * an.get(data[z][r], key) for wi, z in zip(wts, SIZES)) for r in seeds]

    res["floor_ok"] = {z: floor_ok(data[z], seeds) for z in SIZES}
    bad = [z for z, v in res["floor_ok"].items() if v is False]
    floor = {"verdict": an.FLOOR_TEXT + " in " + ", ".join(bad)} if bad else None
    # HS1 (registered prediction): the slope of Y on log2(non-embedding parameters) > 0, SESOI 1 point per doubling.
    res["HS1_size_slope_Y"] = floor or an.decide(per_seed("Y", w), SESOI_SIZE, +1)
    # HS2 (registered, secondary): the step from 60M to 150M, in points per doubling, > 0 (the cost keeps falling).
    d2 = x[2] - x[1]
    res["HS2_step_60_150_per_doubling"] = floor or an.decide(
        [(an.get(data["150m"][r], "Y") - an.get(data["60m"][r], "Y")) / d2 for r in seeds], SESOI_SIZE, +1)
    # Descriptive (no verdict claimed): the 20M -> 60M step, the change of step, the gold-history slope, the twin
    # and panel slopes, and the slope over log2 total parameters.
    d1 = x[1] - x[0]
    b1 = [(an.get(data["60m"][r], "Y") - an.get(data["20m"][r], "Y")) / d1 for r in seeds]
    b2 = res["HS2_step_60_150_per_doubling"].get("per_seed") or \
        [(an.get(data["150m"][r], "Y") - an.get(data["60m"][r], "Y")) / d2 for r in seeds]
    desc = {"step_20_60_per_doubling": _desc(b1), "change_of_step_b2_minus_b1": _desc([u - v for u, v in zip(b2, b1)])}
    for k in ("Y_gold", "twin_self", "multi_self"):
        desc[f"slope_{k}"] = _desc(per_seed(k, w))
    for k in ("panel_fineweb", "panel_dialogue", "sft_val"):
        desc[f"slope_{k}"] = _desc(per_seed(k, w))
    wt = an.slope_weights([math.log2(tot[z]) for z in SIZES])
    desc["slope_Y_log2_total"] = _desc(per_seed("Y", wt))
    res["descriptive"] = desc
    res["isoplot"] = isoplot(art, data["150m"], seeds)
    return res


def _mean(xs):
    xs = [v for v in xs if v is not None]
    return {"mean": st.mean(xs) if xs else None, "n": len(xs)}


def _desc(xs):
    return {"per_seed": xs, "mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else None,
            "label": "DESCRIPTIVE"}


def isoplot(art, ss150, seeds):
    """A6.5 (descriptive): the run-level OLS line of Y on the FineWeb panel loss through every home plain 20M run
    (E5's four mixes and E1's depths, as A5.6) and the 150M runs' residuals at their own panel loss; a 150M panel
    loss below every plain 20M run's is flagged as an extrapolation."""
    pts = []
    for b in ("e1", "e5"):
        for (c, r), s in an.load(art, b).items():
            y, f = s.get("primary"), (s.get("panels") or {}).get("fineweb")
            if y is not None and f is not None:
                pts.append((f, y))
    if len(pts) < 3:
        return {"label": "DESCRIPTIVE", "note": "fewer than 3 plain 20M runs home"}
    fx = [p[0] for p in pts]
    fm, ym = st.mean(fx), st.mean(p[1] for p in pts)
    slope = sum((f - fm) * (y - ym) for f, y in pts) / sum((f - fm) ** 2 for f in fx)
    icpt = ym - slope * fm
    resid = {r: ss150[r]["primary"] - (icpt + slope * ss150[r]["panels"]["fineweb"]) for r in seeds}
    return {"label": "DESCRIPTIVE", "n_plain_20m": len(pts), "slope_Y_per_nat": slope,
            "residual_sd": st.stdev([y - (icpt + slope * f) for f, y in pts]),
            "residuals_150m": resid, "mean_residual_150m": st.mean(resid.values()),
            "extrapolation": all(ss150[r]["panels"]["fineweb"] < min(fx) for r in seeds)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--depth-choice", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    with open(a.depth_choice, encoding="utf-8") as f:
        depth = json.load(f)["depth"]
    res = size_block(a.art, depth, a.partial)
    txt = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
