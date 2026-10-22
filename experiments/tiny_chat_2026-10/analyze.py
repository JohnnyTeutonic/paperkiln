#!/usr/bin/env python3
"""The registered analysis of E5 and E1 (PREREGISTRATION.md sections 5-7), from the summaries the
runs bring home. Refuses an incomplete contrast unless --partial (then everything is labelled
DESCRIPTIVE, PARTIAL and no verdict is printed).

    python experiments/tiny_chat_2026-10/analyze.py --art C:/ml_artifacts/tiny_chat [--out analysis.json] [--partial]

ART holds the runners' artefact folders: ART/s0 (Stage 0), ART/e5, ART/e1, ART/e1_60m, each with
<job>/summary.json per job. E1's L8 cell at 20M is E5's cell at E1's dialogue level (the same runs).
Amendment 1 adds the early-ranking analysis (rung_<...>/rung.json per rung job; rung_analysis) and
a descriptive summary of the exploratory successive-halving arm (ART/asha; asha_summary).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st

SEEDS_FIRST = [1, 2, 3]
SEEDS_EXT = [4, 5]
SESOI_E5 = 0.03          # 3 points of Y
SESOI_E1 = 0.01          # 1 point of Y per doubling of depth
SESOI_LOSS_E1 = 0.01     # nats per doubling (panel losses)
FLOOR = 0.10             # validity gate: twin accuracy at least chance + 10 points
# Two-tailed t critical values: alpha 0.05 and the 90% interval (TOST at 5%), by df.
T05 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
T10 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}
DEPTHS_20 = [2, 4, 8, 16, 32]
DEPTHS_60 = [4, 8, 16, 32]


def load(art, block):
    """{(cell, seed): summary} for one block's post jobs."""
    out = {}
    for p in glob.glob(os.path.join(art, block, "post_*", "summary.json")):
        s = json.load(open(p, encoding="utf-8"))
        out[(s["cell"], s["seed"])] = s
    return out


def get(s, key):
    """One run's metric. carry5 = tiny_chat_run.carry5_read: forced-choice acc per history mode
    (gold, self, twin, noctx), chance 0.25."""
    c5 = s.get("carry5", {})
    acc = c5.get("acc", {})
    gold = acc.get("gold") - acc.get("twin") if acc.get("gold") is not None and acc.get("twin") is not None else None
    return {"Y": s.get("primary"),
            "multi_self": acc.get("self"),
            "twin_self": acc.get("twin"),
            "Y_gold": gold,
            "panel_fineweb": s.get("panels", {}).get("fineweb"),
            "panel_dialogue": s.get("panels", {}).get("dialogue"),
            "sft_val": s["sft_evals"][-1][1] if s.get("sft_evals") else None,
            "chance": c5.get("chance")}[key]


def decide(diffs, sesoi, direction=+1):
    """Paired contrast over seeds. SUPPORTED: the registered direction, |t| > crit (two-tailed 5%)
    and |mean| >= SESOI. CONTRADICTED: the opposite direction with |t| > crit. EQUIVALENT: the 90%
    interval inside +-SESOI. Otherwise NOT_RESOLVED (the extension rule applies)."""
    n = len(diffs)
    m = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = m / se if se and se > 0 else (math.inf if m else 0.0)
    crit, c90 = T05[n - 1], T10[n - 1]
    lo, hi = m - c90 * se, m + c90 * se
    if abs(t) > crit and direction * m > 0 and abs(m) >= sesoi:
        v = "SUPPORTED"
    elif abs(t) > crit and direction * m < 0:
        v = "CONTRADICTED"
    elif -sesoi < lo and hi < sesoi:
        v = "EQUIVALENT"
    elif abs(t) > crit:
        v = "SIGNIFICANT_BELOW_SESOI"
    else:
        v = "NOT_RESOLVED"
    agree = sum(1 for d in diffs if d * direction > 0)
    extend = v == "NOT_RESOLVED" and n == len(SEEDS_FIRST) and agree in (0, n)
    return {"n": n, "per_seed": diffs, "mean": m, "sd": sd, "t": t, "crit": crit, "ci90": [lo, hi],
            "sesoi": sesoi, "verdict": v, "sign_agreement": f"{agree}/{n}", "extend_to_5_seeds": extend}


def cells_complete(runs, cells, seeds):
    missing = [(c, r) for c in cells for r in seeds if (c, r) not in runs or runs[(c, r)].get("primary") is None]
    return missing


def seeds_of(runs, cells):
    have = [r for r in SEEDS_FIRST + SEEDS_EXT if all((c, r) in runs for c in cells)]
    return have if len(have) >= 3 else SEEDS_FIRST


def cell_table(runs, cells, seeds, keys):
    out = {}
    for c in cells:
        row = {}
        for k in keys:
            xs = [get(runs[(c, r)], k) for r in seeds if (c, r) in runs]
            xs = [x for x in xs if x is not None]
            row[k] = {"mean": st.mean(xs) if xs else None,
                      "se": st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None, "n": len(xs)}
        out[c] = row
    return out


def floor_ok(runs, cell, seeds):
    tw = [get(runs[(cell, r)], "twin_self") for r in seeds]
    ch = [get(runs[(cell, r)], "chance") for r in seeds]
    if any(x is None for x in tw + ch):
        return None
    return st.mean(tw) >= st.mean(ch) + FLOOR


def contrast(runs, cells, weights, seeds, key):
    return [sum(w * get(runs[(c, r)], key) for c, w in zip(cells, weights)) for r in seeds]


def e5(runs, partial, design):
    """Cells s<share> for the design's levels (design.json): d0 = 0%, dE1 = E1's level, dtop = the
    most dialogue."""
    cells = [f"s{x}" for x in design["levels"]]
    d0, de1, dtop = cells[0], f"s{design['e1_share']}", cells[-1]
    seeds = seeds_of(runs, cells)
    miss = cells_complete(runs, cells, seeds)
    res = {"design": design, "seeds": seeds, "missing": miss,
           "cells": cell_table(runs, cells, seeds, ["Y", "multi_self", "twin_self", "Y_gold", "panel_fineweb",
                                                     "panel_dialogue", "sft_val"])}
    if miss and not partial:
        raise SystemExit(f"E5 incomplete: {miss} (use --partial for a descriptive read)")
    if miss:
        res["label"] = "DESCRIPTIVE, PARTIAL"
        return res
    res["floor_ok"] = {c: floor_ok(runs, c, seeds) for c in cells}
    if all(res["floor_ok"].get(c) is not False for c in (d0, de1)):
        res["H5_primary_YE1_minus_Y0"] = decide(contrast(runs, [de1, d0], [1, -1], seeds, "Y"), SESOI_E5, +1)
    else:
        res["H5_primary_YE1_minus_Y0"] = {"verdict": "FLOOR (twin accuracy within 10 points of chance)"}
    if dtop != de1:
        res["H5_sat_Ytop_minus_YE1"] = decide(contrast(runs, [dtop, de1], [1, -1], seeds, "Y"), SESOI_E5, +1)
    res["H5_knowledge_fineweb_top_minus_0"] = decide(
        contrast(runs, [dtop, d0], [1, -1], seeds, "panel_fineweb"), 0.02, +1)
    for c in cells[1:]:
        if c != de1:
            res[f"secondary_Y_{c}_minus_{d0}"] = decide(contrast(runs, [c, d0], [1, -1], seeds, "Y"), SESOI_E5, +1)
    res["secondary_Ygold_E1_minus_0"] = decide(contrast(runs, [de1, d0], [1, -1], seeds, "Y_gold"), SESOI_E5, +1)
    # The data bar: highest mean Y among levels with twin accuracy within 3 points of the best cell's
    # and FineWeb panel loss within 0.10 nats of the 0% cell's; within SESOI, the smaller share.
    tab = res["cells"]
    best_twin = max(tab[c]["twin_self"]["mean"] for c in cells)
    ok = [c for c in cells if tab[c]["twin_self"]["mean"] >= best_twin - 0.03
          and tab[c]["panel_fineweb"]["mean"] <= tab[d0]["panel_fineweb"]["mean"] + 0.10]
    top = max(tab[c]["Y"]["mean"] for c in ok)
    res["data_bar"] = min((c for c in ok if tab[c]["Y"]["mean"] >= top - SESOI_E5), key=lambda c: int(c[1:]))
    return res


def e1(runs, depths, partial, tag):
    cells = [f"L{L}" for L in depths]
    seeds = seeds_of(runs, cells)
    miss = cells_complete(runs, cells, seeds)
    res = {"seeds": seeds, "missing": miss,
           "cells": cell_table(runs, cells, seeds, ["Y", "multi_self", "twin_self", "Y_gold", "panel_fineweb",
                                                     "panel_dialogue", "sft_val"])}
    if miss and not partial:
        raise SystemExit(f"E1 {tag} incomplete: {miss} (use --partial for a descriptive read)")
    if miss:
        res["label"] = "DESCRIPTIVE, PARTIAL"
        return res
    x = [math.log2(L) for L in depths]
    xm = st.mean(x)
    lin = [(xi - xm) / sum((xj - xm) ** 2 for xj in x) for xi in x]  # OLS slope per doubling
    q = [(xi - xm) ** 2 for xi in x]
    qm = st.mean(q)
    quad = [qi - qm for qi in q]
    res["floor_ok"] = {c: floor_ok(runs, c, seeds) for c in cells}
    if any(v is False for v in res["floor_ok"].values()):
        res["H1a_depth_slope_Y"] = {"verdict": "FLOOR in " + ", ".join(c for c, v in res["floor_ok"].items()
                                                                      if v is False)}
    else:
        res["H1a_depth_slope_Y"] = decide(contrast(runs, cells, lin, seeds, "Y"), SESOI_E1, +1)
    # H1(b): width helps knowledge/fluency -> panel loss RISES with depth (+). H-lit predicts the opposite.
    res["H1b_depth_slope_fineweb_loss"] = decide(contrast(runs, cells, lin, seeds, "panel_fineweb"), SESOI_LOSS_E1, +1)
    res["H1b_depth_slope_twin"] = decide(contrast(runs, cells, lin, seeds, "twin_self"), SESOI_E1, -1)
    res["H_interior_quadratic_Y"] = decide(contrast(runs, cells, quad, seeds, "Y"), SESOI_E1, -1)
    res["secondary_slope_Ygold"] = decide(contrast(runs, cells, lin, seeds, "Y_gold"), SESOI_E1, +1)
    res["argmax_Y"] = max(cells, key=lambda c: res["cells"][c]["Y"]["mean"])
    return res


# ---------------------------------------------------------------- early ranking (amendment 1)
RUNG_FRACTIONS = [0.05, 0.10, 0.25, 0.50]
STEPS = 10_000
RHO_MIN = 0.80        # registered threshold: Spearman rho at the rung ...
RHO_LO_MIN = 0.50     # ... with the 90% bootstrap interval's lower end at least this
BOOT = 2000
# (rung predictor, final metric): Spearman is positive when early ranks predict final ranks.
RUNG_PAIRS = [("Y_base", "Y"), ("acc_gold", "Y"), ("panel_fineweb", "panel_fineweb"),
              ("panel_dialogue", "panel_dialogue")]
HIGHER_BETTER = {"Y": True, "panel_fineweb": False, "panel_dialogue": False}


def load_rungs(art, block):
    """{(cell, seed, step): rung.json} for one block's rung jobs."""
    out = {}
    for p in glob.glob(os.path.join(art, block, "rung_*", "rung.json")):
        s = json.load(open(p, encoding="utf-8"))
        out[(s["cell"], s["seed"], s["step"])] = s
    return out


def rung_value(r, key):
    return r.get("panels", {}).get(key[len("panel_"):]) if key.startswith("panel_") else r.get(key)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def boot_ci(x, y, n=BOOT, seed=0):
    """90% percentile interval of Spearman's rho, resampling runs (pairs) with replacement."""
    import random
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(len(x)) for _ in x]
        v = spearman([x[i] for i in idx], [y[i] for i in idx])
        if v is not None:
            vals.append(v)
    vals.sort()
    if len(vals) < n // 2:
        return None, None
    return vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals)) - 1]


def rung_analysis(runs, rungs, label, steps=STEPS):
    """The registered early-ranking analysis: at each rung, per (predictor, final) pair, Spearman's
    rho across runs with its bootstrap interval, and whether keeping the top third of cells by the
    predictor's cell means keeps the final best cell. PREDICTIVE needs rho >= RHO_MIN, the lower
    end >= RHO_LO_MIN and the best cell kept; fewer than max(8, 80%) runs with the rung is
    INCOMPLETE. 100% is the post job's base model (final_rung), for the Carry-5 pairs only."""
    res = {"label": label, "runs": len(runs), "threshold": {"rho": RHO_MIN, "ci90_lower": RHO_LO_MIN,
                                                            "top_third_keeps_best_cell": True}, "rungs": {}}
    finals = {k: {"Y": s.get("primary"), "panel_fineweb": s.get("panels", {}).get("fineweb"),
                  "panel_dialogue": s.get("panels", {}).get("dialogue")} for k, s in runs.items()}
    fracs = RUNG_FRACTIONS + [1.0]
    for f in fracs:
        step = int(f * steps)
        row = {}
        for pred, fin in RUNG_PAIRS:
            if f == 1.0 and pred.startswith("panel_"):
                continue
            pts = []
            for (cell, seed), s in runs.items():
                r = (s.get("final_rung") or {}) if f == 1.0 else rungs.get((cell, seed, step))
                if r is None:
                    continue
                pv, fv = rung_value(r, pred), finals[(cell, seed)][fin]
                if pv is not None and fv is not None:
                    pts.append((pv, fv, cell))
            if len(pts) < max(8, math.ceil(0.8 * len(runs))):
                row[f"{pred}->{fin}"] = {"n": len(pts), "verdict": "INCOMPLETE"}
                continue
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            rho = spearman(x, y)
            lo, hi = boot_ci(x, y)
            cells = sorted({p[2] for p in pts})
            cm = {c: (st.mean(p[0] for p in pts if p[2] == c), st.mean(p[1] for p in pts if p[2] == c)) for c in cells}
            hb = HIGHER_BETTER[fin]
            best_final = (max if hb else min)(cells, key=lambda c: cm[c][1])
            keep = math.ceil(len(cells) / 3)
            kept = sorted(cells, key=lambda c: -cm[c][0] if hb else cm[c][0])[:keep]
            ok = rho is not None and lo is not None and rho >= RHO_MIN and lo >= RHO_LO_MIN and best_final in kept
            row[f"{pred}->{fin}"] = {"n": len(pts), "rho": rho, "ci90": [lo, hi], "cells_kept": kept,
                                     "best_final_cell": best_final, "best_kept": best_final in kept,
                                     "verdict": "PREDICTIVE" if ok else "NOT_PREDICTIVE"}
        res["rungs"][f"{round(100 * f)}%"] = row
    res["earliest_predictive"] = {
        f"{p}->{q}": next((k for k, row in res["rungs"].items() if row.get(f"{p}->{q}", {}).get("verdict") == "PREDICTIVE"),
                          None) for p, q in RUNG_PAIRS}
    return res


def asha_summary(art):
    """The exploratory arm, descriptively: each round's ranking and decisions, and the finalists'
    primary metric beside the E cells'. No verdicts."""
    plan_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asha_plan.json")
    d = os.path.join(art, "asha")
    if not os.path.isdir(d) or not os.path.exists(plan_p):
        return None
    plan = json.load(open(plan_p, encoding="utf-8"))
    out = {"label": "EXPLORATORY (not confirmatory; amendment 1)", "rounds": {}, "finalists": {}}
    for r in plan["rounds"]:
        p = os.path.join(d, f"asha_promote_{r['id']}", "decisions.json")
        if os.path.exists(p):
            out["rounds"][r["id"]] = json.load(open(p, encoding="utf-8"))
    for c in plan["configs"]:
        p = os.path.join(d, f"asha_post_{c['id']}", "summary.json")
        if os.path.exists(p):
            s = json.load(open(p, encoding="utf-8"))
            out["finalists"][c["id"]] = {**c, "Y": s.get("primary"), "panels": s.get("panels"),
                                         "final_rung": s.get("final_rung")}
    return out


def throughput(art, size):
    out = {}
    for p in sorted(glob.glob(os.path.join(art, "s0" if size == "20m" else "s0_60m", "probe_*", "summary.json"))):
        s = json.load(open(p, encoding="utf-8"))
        sid = re.sub(r"^probe_", "", os.path.basename(os.path.dirname(p)))
        out[sid] = {"tok_per_s": s.get("tok_per_s_steady"), "peak_gpu_mem_mib": s.get("peak_gpu_mem_mib")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--design", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "design.json"))
    a = ap.parse_args(argv)
    design = json.load(open(a.design, encoding="utf-8"))
    r5 = load(a.art, "e5")
    r1 = load(a.art, "e1")
    for (cell, seed), s in r5.items():  # E1's L8 cell at 20M is E5's cell at E1's dialogue level
        if cell == f"s{design['e1_share']}":
            r1[("L8", seed)] = s
    res = {"partial": a.partial}
    if r5:
        res["E5"] = e5(r5, a.partial, design)
    if any(c != "L8" for c, _ in r1):
        res["E1_20m"] = e1(r1, DEPTHS_20, a.partial, "20m")
    r60 = {(c.replace("60m_", ""), r): s for (c, r), s in load(a.art, "e1_60m").items()}
    if r60:
        res["E1_60m"] = e1(r60, DEPTHS_60, a.partial, "60m")
    # Early ranking (amendment 1): read only for complete blocks unless --partial.
    g5, g1, g60 = load_rungs(a.art, "e5"), load_rungs(a.art, "e1"), load_rungs(a.art, "e1_60m")
    for (cell, seed, step), s in list(g5.items()):  # E1's L8 cell is E5's dE1 cell: the same runs
        if cell == f"s{design['e1_share']}":
            g1[("L8", seed, step)] = s
    g60 = {(c.replace("60m_", ""), r, k): s for (c, r, k), s in g60.items()}
    early = {}
    lab = "DESCRIPTIVE, PARTIAL" if a.partial else "REGISTERED"  # without --partial, an incomplete block exited above
    for name, runs, rungs in (("E5", r5, g5), ("E1_20m", r1, g1), ("E1_60m", r60, g60)):
        if runs and (name != "E1_20m" or any(c != "L8" for c, _ in runs)):
            early[name] = rung_analysis(runs, rungs, lab)
    if r5 and any(c != "L8" for c, _ in r1):  # pooled 20M: E5's runs and E1's other depths (24 distinct runs)
        pooled = {**r5, **{k: v for k, v in r1.items() if k[0] != "L8"}}
        prungs = {**g5, **{k: v for k, v in g1.items() if k[0] != "L8"}}
        early["pooled_20m"] = rung_analysis(pooled, prungs, lab)
    res["early_ranking"] = early
    res["asha"] = asha_summary(a.art)
    res["throughput_20m"] = throughput(a.art, "20m")
    res["throughput_60m"] = throughput(a.art, "60m")
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
