#!/usr/bin/env python3
"""Arm X5's registered analysis (PREREGISTRATION.md amendment 5): Attention Residuals on the llama base
(llama_attnres, 20m_L16, cell X5) against E1's 20m_L16 cell, on Carry-5.

    python experiments/tiny_chat_2026-10/analyze_x5.py --art C:/ml_artifacts/tiny_chat [--out x5.json] [--partial]

(analyze.py runs the same block as "E1_x5_20m" when ART/e1_x5 holds post summaries.)

Inputs under ART:
  e1_x5/post_e1_x5_X5_r{s}/summary.json and carry5/m/items.jsonl   the arm's runs (seeds 1-3)
  e1/post_e1_L16_r{s}/summary.json                                   E1's home L16 runs: the registered control
  e1_x5_ctl/post_e1_x5_ctl_L16_r{s}/summary.json, carry5/m/items.jsonl
        E1's L16 post jobs re-run at the arm's commit from E1's home pretrained weights: E1's own post jobs kept
        their per-item records on their VMs, and HX5-main is an item-level test. Each re-run is compared with
        its home summary (reproduction); the test pairs X5's items with the re-run's.
  e1/, e5/                                                            every home plain 20M run (the isoplot)
  s0_x5/probe_*/summary.json                                          Stage 0's throughput probes
  e1_x5/trace_X5_r{s}/attnres_trace.json                              depth-attention traces (descriptive)

HX5-main (registered): D = Y(X5) - Y(L16) > 0 beyond the SESOI of 3 points. Unit: the item-level paired
difference d(i, s) = [self(X5) - twin(X5)] - [self(L16) - twin(L16)] for item i of Y's items (the primary
families, amendment 2) and seed s, pooled over the three seeds (3 x 888); its mean equals Y(X5) - Y(L16)
averaged over seeds. Normal-approximation 95% and 90% intervals. Verdicts, as Stage L0's: SUPPORTED (95% lower
bound > 0 and D >= 0.03), SIGNIFICANT_BELOW_SESOI (95% lower bound > 0, D < 0.03), CONTRADICTED (95% upper
bound < 0), EQUIVALENT (90% interval inside +-0.03), otherwise NOT_RESOLVED. The validity gate (A2.5) applies
to both cells on their means over seeds; a FLOOR cell ends the contrast as FLOOR. Everything else here is
descriptive.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as an  # noqa: E402  (the programme's gate, tables and loaders)

SEEDS = [1, 2, 3]
SESOI_X5 = 0.03
Z95, Z90 = 1.959964, 1.644854
X5, CTL, E1 = "e1_x5", "e1_x5_ctl", "e1"
CELL, CONTROL = "X5", "L16"
PRIMARY_FAMILIES = ("recall", "correction", "topic", "knowledge")   # amendment 2 (tiny_chat_run.py)
CONTEXT_FAMILIES = ("recall", "correction", "topic")
REPRO_KEYS = ("pt_weights_sha256", "primary", "primary_all_items", "sft_weights_sha256", "panels")
HYPOTHESIS = ("HX5-main (Jonathan's hypothesis): Y(X5) - Y(L16) > 0 beyond the SESOI of 3 points; item-level "
              "paired differences over Y's items and seeds 1-3, normal-approximation intervals, Stage L0's verdicts")
COUNTER = ("counter-hypothesis (the literature's default): any gain is a general language-model improvement: at "
           "matched FineWeb panel loss X5 sits on the plain cells' loss-to-Y line (descriptive isoplot)")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def post_dir(art, block, cell, seed):
    """ART/BLOCK/post_..., or ART/BLOCK_r{seed}/post_... where the seeds ran on machines of their own (the
    by-seed split of amendment 5)."""
    name = f"post_{block}_{cell}_r{seed}"
    for d in (os.path.join(art, block, name), os.path.join(art, f"{block}_r{seed}", name)):
        if os.path.isdir(d):
            return d
    return os.path.join(art, block, name)


def strip_ledger(d):
    """A data ledger without the fields that differ between builds of the same data (times, paths, hosts)."""
    if isinstance(d, dict):
        return {k: strip_ledger(v) for k, v in d.items()
                if not any(t in k.lower() for t in ("time", "date", "host", "wall", "elapsed", "started", "path"))}
    if isinstance(d, list):
        return [strip_ledger(x) for x in d]
    return d


def ledgers(art):
    """Every X5 machine's data.ledger.json (the arm's runs, the L16 re-runs, Stage 0) against E1's first build
    (the 25 Oct split note: a mismatch holds that machine's results for Jonathan)."""
    ref = os.path.join(art, E1, "data", "data.ledger.json")
    if not os.path.exists(ref):
        return {"reference": None}
    want = strip_ledger(jload(ref))
    out = {}
    for p in sorted(glob.glob(os.path.join(art, "e1_x5*", "data", "data.ledger.json")) +
                    glob.glob(os.path.join(art, "s0_x5", "data", "data.ledger.json"))):
        got = strip_ledger(jload(p))
        out[os.path.basename(os.path.dirname(os.path.dirname(p)))] = (
            "matches" if got == want else sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k)))
    return out


def items_of(d):
    """{id: record} from a post job's carry5/m/items.jsonl, or None when it is not home."""
    p = os.path.join(d, "carry5", "m", "items.jsonl")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    return {r["id"]: r for r in recs}


def y_items(recs):
    """{id: self - twin} over Y's items (the primary families), every record with both modes, as
    tiny_chat_run.carry5_pools pools them."""
    return {i: float(r["fc"]["self"]["acc"]) - float(r["fc"]["twin"]["acc"]) for i, r in recs.items()
            if r["family"] in PRIMARY_FAMILIES and "self" in r.get("fc", {}) and "twin" in r.get("fc", {})}


def check_items(summary, recs, where):
    """The items file must be the summary's: the mean of self - twin over Y's items equals its primary."""
    y = y_items(recs)
    m = sum(y.values()) / len(y)
    if summary.get("primary") is None or abs(m - summary["primary"]) > 1e-9:
        raise SystemExit(f"{where}: the items file gives Y {m}, its summary {summary.get('primary')}: not the same run")
    return y


def paired(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    return {"n": n, "mean": m, "sd": sd, "se": se, "ci95": [m - Z95 * se, m + Z95 * se],
            "ci90": [m - Z90 * se, m + Z90 * se]}


def verdict(s, sesoi=SESOI_X5):
    """Stage L0's rule with HX5-main's direction (> 0)."""
    lo95, hi95 = s["ci95"]
    lo90, hi90 = s["ci90"]
    if lo95 > 0:
        return "SUPPORTED" if s["mean"] >= sesoi else "SIGNIFICANT_BELOW_SESOI"
    if hi95 < 0:
        return "CONTRADICTED"
    if -sesoi < lo90 and hi90 < sesoi:
        return "EQUIVALENT"
    return "NOT_RESOLVED"


def reproduction(ctl, home):
    """The re-run's summary against E1's home summary: the pretrained weights' sha256 (the same inputs), every
    Carry-5 pool (acc and acc_norm, primary and context families, per family), Y, Y_all, the SFT weights' sha256
    and the panels."""
    diffs = {}
    for k in REPRO_KEYS:
        if ctl.get(k) != home.get(k):
            diffs[k] = {"rerun": ctl.get(k), "home": home.get(k)}
    for k in ("pool", "ctx", "acc", "acc_norm"):
        a, b = (ctl.get("carry5") or {}).get(k), (home.get("carry5") or {}).get(k)
        if a != b:
            diffs[f"carry5.{k}"] = {"rerun": a, "home": b}
    fa = ((ctl.get("carry5") or {}).get("pools") or {}).get("acc", {}).get("family")
    fb = ((home.get("carry5") or {}).get("pools") or {}).get("acc", {}).get("family")
    if fa != fb:
        diffs["carry5.pools.acc.family"] = "differ"
    return {"exact": not diffs, "differences": diffs}


def by_key(recs, ids, key):
    out = {}
    for i in ids:
        out.setdefault(str(recs[i].get(key)), []).append(i)
    return out


def isoplot(runs, xs):
    """OLS of Y on the FineWeb panel loss through RUNS {(cell, seed): summary}, and each X5 seed's residual
    at its own panel loss (flagged outside the line's range)."""
    pts = [(an.get(s, "panel_fineweb"), an.get(s, "Y")) for s in runs.values()]
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    if len(pts) < 3:
        return {"runs": len(pts), "note": "fewer than three plain runs home"}
    mx, my = st.mean(x for x, _ in pts), st.mean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    resid_sd = math.sqrt(sum((y - a - b * x) ** 2 for x, y in pts) / (len(pts) - 2)) if len(pts) > 2 else None
    lo, hi = min(x for x, _ in pts), max(x for x, _ in pts)
    per = {}
    for seed, s in xs.items():
        x, y = an.get(s, "panel_fineweb"), an.get(s, "Y")
        if x is None or y is None:
            continue
        per[seed] = {"panel_fineweb": x, "Y": y, "line_Y": a + b * x, "residual": y - a - b * x,
                     "extrapolation": not lo <= x <= hi}
    return {"runs": len(pts), "cells": sorted({c for c, _ in runs}), "intercept": a, "slope_per_nat": b,
            "residual_sd": resid_sd, "panel_range": [lo, hi], "X5": per,
            "mean_residual": st.mean(v["residual"] for v in per.values()) if per else None}


def plain_20m(art):
    """Every home plain 20M run (E5's four mixes and E1's other depths; E5's 35% cell is E1's L8), and the
    35% subset (the arm's own mix): {(cell, seed): summary}."""
    r5, r1 = an.load(art, "e5"), an.load(art, E1)
    allr = {**{(f"e5_{c}", s): v for (c, s), v in r5.items()}, **{(f"e1_{c}", s): v for (c, s), v in r1.items()}}
    s35 = {k: v for k, v in allr.items() if v.get("share") == 35}
    return allr, s35


def throughput(art):
    out = {}
    for p in sorted(glob.glob(os.path.join(art, "s0_x5", "probe_*", "summary.json"))):
        s = jload(p)
        out[os.path.basename(os.path.dirname(p))[len("probe_"):]] = {
            "tok_per_s": s.get("tok_per_s_steady"), "peak_gpu_mem_mib": s.get("peak_gpu_mem_mib"),
            "provider": (s.get("provenance") or {}).get("provider"), "gpu": s.get("gpu")}
    x, p = out.get("20m_X5", {}).get("tok_per_s"), out.get("20m_L16", {}).get("tok_per_s")
    return {"probes": out, "X5_over_L16": x / p if x and p else None}


def l4_hours(s, pt_dir):
    """S10's denominator as analyze.run_l4_hours: pretraining (its home result.json, else the post summary's
    copy) plus SFT wall-clock, in hours."""
    p = os.path.join(pt_dir, "result.json")
    pt = jload(p).get("wall_seconds") if os.path.exists(p) else (s.get("pt_result") or {}).get("wall_seconds")
    sft = (s.get("sft_result") or {}).get("wall_seconds") or s.get("sft_seconds")
    return (pt + sft) / 3600 if pt is not None and sft is not None else None


def traces(art):
    out = {}
    for s in SEEDS:
        p = os.path.join(art, X5, f"trace_{CELL}_r{s}", "attnres_trace.json")
        if os.path.exists(p):
            out[s] = jload(p).get("summary")
    return out or "not computed (tools/carry5_attnres_trace.py over the post jobs' SFT GGUFs)"


def x5(art, partial=False):
    """The X5 block; None when no X5 summary is home."""
    xs = {s: jload(os.path.join(post_dir(art, X5, CELL, s), "summary.json")) for s in SEEDS
          if os.path.exists(os.path.join(post_dir(art, X5, CELL, s), "summary.json"))}
    if not xs:
        return None
    for s, v in xs.items():
        if v.get("seed") != s or v.get("cell") != CELL:
            raise SystemExit(f"{post_dir(art, X5, CELL, s)}: summary says cell {v.get('cell')} seed {v.get('seed')}")
    home = {s: v for (c, s), v in an.load(art, E1).items() if c == CONTROL}
    ctl = {s: jload(os.path.join(post_dir(art, CTL, CONTROL, s), "summary.json")) for s in SEEDS
           if os.path.exists(os.path.join(post_dir(art, CTL, CONTROL, s), "summary.json"))}
    xi = {s: items_of(post_dir(art, X5, CELL, s)) for s in SEEDS}
    ci = {s: items_of(post_dir(art, CTL, CONTROL, s)) for s in SEEDS}
    missing = ([f"X5 seed {s} summary" for s in SEEDS if s not in xs] +
               [f"X5 seed {s} items" for s in SEEDS if s in xs and xi[s] is None] +
               [f"L16 seed {s} home summary" for s in SEEDS if s not in home] +
               [f"L16 seed {s} re-run summary" for s in SEEDS if s not in ctl] +
               [f"L16 seed {s} re-run items" for s in SEEDS if s in ctl and ci[s] is None])
    runs = {**{(CELL, s): v for s, v in xs.items()}, **{(CONTROL, s): v for s, v in home.items()}}
    have = [s for s in SEEDS if (CELL, s) in runs and (CONTROL, s) in runs]
    res = {"label": "REGISTERED (amendment 5)", "hypothesis": HYPOTHESIS, "counter_hypothesis": COUNTER,
           "seeds": SEEDS, "missing": missing,
           "cells": an.cell_table(runs, [CELL, CONTROL], have, an.CELL_KEYS),
           "families": an.family_table(runs, [CELL, CONTROL], have),
           "reproduction": {s: reproduction(ctl[s], home[s]) for s in SEEDS if s in ctl and s in home},
           "throughput": throughput(art)}
    allr, s35 = plain_20m(art)
    res["isoplot_all_plain_20m"] = isoplot(allr, xs)
    res["isoplot_s35_plain_20m"] = isoplot(s35, xs)
    res["depth_traces"] = traces(art)
    res["data_ledgers"] = led = ledgers(art)
    cost = {}
    for s, v in xs.items():
        h = l4_hours(v, os.path.join(os.path.dirname(post_dir(art, X5, CELL, s)), f"pt_{X5}_{CELL}_r{s}"))
        cost[s] = {"l4_hours": h, "Y_per_l4_hour": v["primary"] / h if h and v.get("primary") is not None else None}
    res["cost"] = cost
    res["panels_X5_minus_L16"] = {
        k: {s: an.get(xs[s], k) - an.get(home[s], k) for s in have
            if an.get(xs[s], k) is not None and an.get(home[s], k) is not None}
        for k in ("panel_fineweb", "panel_dialogue", "sft_val")}
    bad = {k: v for k, v in led.items() if v != "matches" and k != "reference"}
    if bad and not partial:
        raise SystemExit(f"data ledgers differ from E1's build: {bad}; the results wait for Jonathan (25 Oct note)")
    if missing and not partial:
        raise SystemExit(f"X5 incomplete: {missing} (use --partial for a descriptive read)")
    if missing:
        res["label"] = "DESCRIPTIVE, PARTIAL"
        return res
    # Item-level pairing: the same item and seed.
    diffs, gold, rows = [], [], {}
    for s in SEEDS:
        yx = check_items(xs[s], xi[s], post_dir(art, X5, CELL, s))
        yc = check_items(ctl[s], ci[s], post_dir(art, CTL, CONTROL, s))
        if set(yx) != set(yc):
            raise SystemExit(f"seed {s}: X5 and the L16 re-run scored different items "
                             f"({len(set(yx) ^ set(yc))} differ)")
        ids = sorted(yx)
        d = [yx[i] - yc[i] for i in ids]
        diffs += d
        rows[s] = {"items": len(ids), "Y_X5": sum(yx.values()) / len(ids), "Y_L16_rerun": sum(yc.values()) / len(ids),
                   "Y_L16_home": home[s].get("primary"), "mean_difference": sum(d) / len(d)}
        for i in ids:
            g = [r["fc"][m]["acc"] for r in (xi[s][i], ci[s][i]) for m in ("gold", "twin")]
            gold.append((g[0] - g[1]) - (g[2] - g[3]))
        for key in ("family", "distance"):
            for k, grp in by_key(xi[s], ids, key).items():
                rows.setdefault(key, {}).setdefault(k, []).extend(yx[i] - yc[i] for i in grp)
    by = {key: {k: paired(v) for k, v in sorted(rows.pop(key).items())} for key in ("family", "distance")}
    res["floor_ok"] = {c: an.floor_ok(runs, c, SEEDS) for c in (CELL, CONTROL)}
    stats = paired(diffs)
    agree = sum(1 for s in SEEDS if rows[s]["mean_difference"] > 0)
    if any(v is False for v in res["floor_ok"].values()):
        h = {"verdict": an.FLOOR_TEXT + " in " + ", ".join(c for c, v in res["floor_ok"].items() if v is False)}
    else:
        h = {"verdict": verdict(stats)}
    h.update({**stats, "sesoi": SESOI_X5, "unit": "item x seed (Y's items, primary families)",
              "per_seed": rows, "seeds_positive": f"{agree}/{len(SEEDS)}",
              "control_items": "E1's L16 post jobs re-run at the arm's commit from E1's home pretrained weights",
              "control_reproduces_home": {s: res["reproduction"][s]["exact"] for s in SEEDS}})
    res["HX5_main"] = h
    res["descriptive_seed_level_t"] = {**an.decide([rows[s]["mean_difference"] for s in SEEDS], SESOI_X5, +1),
                                       "label": "DESCRIPTIVE (the programme's seed-level t, n = 3)"}
    g = paired(gold)
    res["descriptive_Y_gold_X5_minus_L16"] = {**g, "verdict_if_registered": verdict(g), "label": "DESCRIPTIVE"}
    res["descriptive_by_family"] = {k: {**v, "label": "DESCRIPTIVE"} for k, v in by["family"].items()}
    res["descriptive_by_distance"] = {k: {**v, "label": "DESCRIPTIVE"} for k, v in by["distance"].items()}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    res = x5(a.art, a.partial)
    if res is None:
        raise SystemExit(f"no X5 post summary under {os.path.join(a.art, X5)}")
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
