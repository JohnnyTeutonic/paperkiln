#!/usr/bin/env python3
"""The registered analysis of E5 and E1 (PREREGISTRATION.md sections 5-7), from the summaries the
runs bring home. Refuses an incomplete contrast unless --partial (then everything is labelled
DESCRIPTIVE, PARTIAL and no verdict is printed).

    python experiments/tiny_chat_2026-10/analyze.py --art C:/ml_artifacts/tiny_chat [--out analysis.json] [--partial]

ART holds the runners' artefact folders: ART/s0 (Stage 0), ART/e5, ART/e1, ART/e1_60m, each with
<job>/summary.json per job. E1's L8 cell at 20M is E5's cell at E1's dialogue level (the same runs).
Amendment 1 adds the early-ranking analysis (rung_<...>/rung.json per rung job; rung_analysis) and
a descriptive summary of the exploratory successive-halving arm (ART/asha; asha_summary).
Amendment 2: Y over the primary families (the instruction family out; its generation-scored
compliance reported separately), the validity gate's second condition (twin at least 10 points
above the question alone on the context families), the per-family table with the question alone
beside the twin, and section 3's all-items Y beside the amended one.
Amendment 3 (A3.3) adds the recursion arm, E1-R (ART/e1_rec, ART/e1_rec_r8; e1_rec: HR1-HR4, the
twin pairs R2-L8, R4-L16, R8-L32, MC7-L16, FLOPs per token and Y per L4-hour) and the R8 rule
(--rec-decision OUT, from R2's and R4's post jobs).
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
FLOOR = 0.10             # validity gate (i): twin accuracy at least chance + 10 points (primary families)
FLOOR_QALONE = 0.10      # amendment 2, gate (ii): twin minus question-alone at least 10 points (context families)
MODES = ("gold", "self", "twin", "noctx")
# Two-tailed t critical values: alpha 0.05 and the 90% interval (TOST at 5%), by df.
T05 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
T10 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}
DEPTHS_20 = [2, 4, 8, 16, 32]
DEPTHS_60 = [4, 8, 16, 32]


PILOT_SEED = 99         # amendment 3: the floor pilot's seed, excluded from every analysis


def load(art, block):
    """{(cell, seed): summary} for one block's post jobs. The floor pilot (amendment 3) lives in
    ART/pilot, which no block reads; a pilot-seed summary inside a block is refused."""
    out = {}
    for p in glob.glob(os.path.join(art, block, "post_*", "summary.json")):
        s = json.load(open(p, encoding="utf-8"))
        if s.get("seed") == PILOT_SEED:
            raise SystemExit(f"{p}: seed {PILOT_SEED} is the floor pilot's, excluded from all analysis (amendment 3)")
        out[(s["cell"], s["seed"])] = s
    return out


def pilot_verdict(art):
    """Amendment 3: the floor pilot's verdict by the validity gate as amended (A2.5), on its one run:
    PASS when (i) twin accuracy over the primary families >= chance + FLOOR and (ii) twin minus
    question-alone accuracy over the context families >= FLOOR_QALONE; FAIL otherwise. Only the
    gate's quantities are written: the pilot's Y and multi-turn accuracies are not read."""
    ps = glob.glob(os.path.join(art, "pilot", "post_*", "summary.json"))
    if len(ps) != 1:
        raise SystemExit(f"{len(ps)} pilot post summaries in {os.path.join(art, 'pilot')}; the verdict needs exactly one")
    s = json.load(open(ps[0], encoding="utf-8"))
    if s.get("seed") != PILOT_SEED:
        raise SystemExit(f"{ps[0]}: seed {s.get('seed')}, not the pilot's {PILOT_SEED}")
    q = {k: get(s, k) for k in ("twin_self", "chance", "twin_ctx", "noctx_ctx", "twin_minus_qalone_ctx")}
    if any(v is None for v in q.values()):
        raise SystemExit(f"{ps[0]}: the gate's quantities are incomplete: {q}")
    i = q["twin_self"] >= q["chance"] + FLOOR
    ii = q["twin_minus_qalone_ctx"] >= FLOOR_QALONE
    return {"verdict": "PASS" if i and ii else "FAIL", "gate_i_twin_primary": i, "gate_ii_twin_minus_qalone_ctx": ii,
            "twin_primary": q["twin_self"], "chance": q["chance"], "twin_ctx": q["twin_ctx"],
            "qalone_ctx": q["noctx_ctx"], "twin_minus_qalone_ctx": q["twin_minus_qalone_ctx"],
            "thresholds": {"i": f"twin_primary >= chance + {FLOOR}", "ii": f"twin_minus_qalone_ctx >= {FLOOR_QALONE}"},
            "cell": s.get("cell"), "seed": s.get("seed"), "pt_lr": s.get("pt_lr"), "summary": ps[0],
            "excluded_from_analysis": True}


def get(s, key):
    """One run's metric. carry5 = tiny_chat_run.carry5_read: forced-choice acc per history mode
    (gold, self, twin, noctx = the question alone), chance 0.25. Amendment 2: Y, the multi-turn and
    twin accuracies and the gold-history Y are over the primary families (`pool`: recall,
    correction, topic, knowledge; the instruction family out); the floor's question-alone check is
    on the context families (`ctx`: recall, correction, topic); Y_all is section 3's frozen
    all-items definition, reported beside it."""
    c5 = s.get("carry5", {})
    if "pool" not in c5 or "ctx" not in c5:
        raise SystemExit(f"{s.get('cell')} seed {s.get('seed')}: summary without the amended Carry-5 pools "
                         "(amendment 2); rerun its post job from the amended commit")
    pool, ctx = c5["pool"], c5["ctx"]
    comp = (c5.get("pools") or {}).get("instruction_complies_gen", {})

    def diff(d, a, b):
        return d[a] - d[b] if d.get(a) is not None and d.get(b) is not None else None
    return {"Y": s.get("primary"),
            "Y_all": s.get("primary_all_items"),
            "multi_self": pool.get("self"),
            "twin_self": pool.get("twin"),
            "noctx": pool.get("noctx"),
            "twin_ctx": ctx.get("twin"),
            "noctx_ctx": ctx.get("noctx"),
            "twin_minus_qalone_ctx": diff(ctx, "twin", "noctx"),
            "Y_gold": diff(pool, "gold", "twin"),
            "instruction_complies_self": (comp.get("self") or {}).get("mean"),
            "instruction_complies_twin": (comp.get("twin") or {}).get("mean"),
            "panel_fineweb": s.get("panels", {}).get("fineweb"),
            "panel_dialogue": s.get("panels", {}).get("dialogue"),
            "sft_val": s["sft_evals"][-1][1] if s.get("sft_evals") else None,
            "chance": c5.get("chance")}[key]


CELL_KEYS = ["Y", "Y_all", "multi_self", "twin_self", "noctx", "twin_ctx", "noctx_ctx", "twin_minus_qalone_ctx",
             "Y_gold", "instruction_complies_self", "instruction_complies_twin", "panel_fineweb", "panel_dialogue",
             "sft_val"]


def family_table(runs, cells, seeds):
    """S5 as amended: per cell, per family (the instruction family included, its forced choice
    flagged as length-confounded), the mean forced-choice accuracy over seeds in every history mode,
    the question alone beside the twin."""
    out = {}
    for c in cells:
        fams = {}
        for r in seeds:
            if (c, r) not in runs:
                continue
            fam = (runs[(c, r)].get("carry5", {}).get("pools") or {}).get("acc", {}).get("family", {})
            for mode, rows in fam.items():
                for f, v in rows.items():
                    fams.setdefault(f, {}).setdefault(mode, []).append(v["mean"])
        out[c] = {f: {m: st.mean(v) for m, v in modes.items()} for f, modes in sorted(fams.items())}
        if "instruction" in out[c]:
            out[c]["instruction"]["note"] = "forced choice confounded by answer length (amendment 2); not in Y"
    return out


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
    """The validity gate as amended (amendment 2), on the cell's means over seeds: (i) twin accuracy
    over the primary families at least chance + FLOOR, and (ii) twin minus question-alone accuracy
    over the context families at least FLOOR_QALONE."""
    tw = [get(runs[(cell, r)], "twin_self") for r in seeds]
    ch = [get(runs[(cell, r)], "chance") for r in seeds]
    tq = [get(runs[(cell, r)], "twin_minus_qalone_ctx") for r in seeds]
    if any(x is None for x in tw + ch + tq):
        return None
    return st.mean(tw) >= st.mean(ch) + FLOOR and st.mean(tq) >= FLOOR_QALONE


FLOOR_TEXT = ("FLOOR (twin accuracy within 10 points of chance, or twin within 10 points of the question alone "
              "on the context families)")


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
           "cells": cell_table(runs, cells, seeds, CELL_KEYS), "families": family_table(runs, cells, seeds)}
    if miss and not partial:
        raise SystemExit(f"E5 incomplete: {miss} (use --partial for a descriptive read)")
    if miss:
        res["label"] = "DESCRIPTIVE, PARTIAL"
        return res
    res["floor_ok"] = {c: floor_ok(runs, c, seeds) for c in cells}
    if all(res["floor_ok"].get(c) is not False for c in (d0, de1)):
        res["H5_primary_YE1_minus_Y0"] = decide(contrast(runs, [de1, d0], [1, -1], seeds, "Y"), SESOI_E5, +1)
    else:
        res["H5_primary_YE1_minus_Y0"] = {"verdict": FLOOR_TEXT}
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
           "cells": cell_table(runs, cells, seeds, CELL_KEYS), "families": family_table(runs, cells, seeds)}
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
        res["H1a_depth_slope_Y"] = {"verdict": FLOOR_TEXT + " in " + ", ".join(c for c, v in res["floor_ok"].items()
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


# ---------------------------------------------------------------- the recursion arm (amendment 3, A3.3)
REC_PASSES = {"L4": 1, "R2": 2, "R4": 4, "R8": 8}          # r for HR1/HR4; E1's L4 is r = 1
REC_TWINS = {"R2": "L8", "R4": "L16", "R8": "L32", "MC7": "L16"}
SESOI_REC = 0.03                                            # HR2, HR3 and the secondary pairs: 3 points of Y
PLAIN_20M = [f"L{L}" for L in DEPTHS_20]
VOCAB, CTX = 16384, 512


def flops_per_token(arch):
    """Training FLOPs per token from a run's arch (summary["arch"], arch.custom): 6 x (applied
    non-embedding + tied head) + attention (6 L T d), with L the effective depth of a recursed model.
    make_jobs.py's formula; reproduces A3.3's table (0.19G at 20m_L4, 0.32 / 0.58 / 1.09 / 0.58G for
    R2 / R4 / R8 / MC7, 0.17 / 0.17 / 0.16G for their twins)."""
    d, dff = arch["d"], arch.get("d_ff") or 3 * arch["d"]
    rec = arch.get("recursion")
    L = rec.get("prelude", 0) + rec["block"] * rec.get("repeats", 1) + rec.get("coda", 0) if rec else arch["layers"]
    return 6 * (L * (4 * d * d + 3 * d * dff + 2 * d) + d + VOCAB * d) + 6 * L * CTX * d


def run_l4_hours(s, art, block):
    """S10's denominator: the run's wall-clock on the L4, pretraining plus SFT. Pretraining from its
    result.json at home (ART/<block>/pt_<...>/result.json; the post summary's copy when the post job ran
    on that VM); a resumed run counts its last segment only, so it is flagged."""
    pt = None
    p = os.path.join(art, block, f"pt_{block}_{s['cell']}_r{s['seed']}", "result.json")
    if os.path.exists(p):
        pt = json.load(open(p, encoding="utf-8")).get("wall_seconds")
    if pt is None:
        pt = (s.get("pt_result") or {}).get("wall_seconds")
    sft = (s.get("sft_result") or {}).get("wall_seconds")
    if sft is None:
        sft = s.get("sft_seconds")
    if pt is None or sft is None:
        return None
    return (pt + sft) / 3600


def rec_cell_table(runs, art, blocks, cells, seeds):
    """Per cell: FLOPs per token (from its arch), mean wall-clock L4-hours per run and S10 (Y per
    L4-hour, the mean over seeds of each run's Y / hours)."""
    out = {}
    for c in cells:
        rs = [(r, runs[(c, r)]) for r in seeds if (c, r) in runs]
        if not rs:
            continue
        arch = rs[0][1].get("arch")
        hrs, yph = [], []
        for r, s in rs:
            h = run_l4_hours(s, art, blocks.get(c, ""))
            if h:
                hrs.append(h)
                if s.get("primary") is not None:
                    yph.append(s["primary"] / h)
        out[c] = {"flops_per_token": flops_per_token(arch) if arch else None,
                  "l4_hours_per_run": st.mean(hrs) if hrs else None,
                  "Y_per_l4_hour": st.mean(yph) if yph else None, "runs_with_hours": len(hrs)}
    return out


def gated(runs, cells, seeds, fn):
    """A3.3: the validity gate applies to every cell; a Y contrast that uses a FLOOR cell is FLOOR."""
    bad = [c for c in cells if floor_ok(runs, c, seeds) is False]
    return {"verdict": FLOOR_TEXT + " in " + ", ".join(bad)} if bad else fn()


def slope_weights(xs):
    xm = st.mean(xs)
    return [(x - xm) / sum((y - xm) ** 2 for y in xs) for x in xs]


def readings_hr2(v):
    """HR2's two registered readings of one verdict: the literature's (not above the SESOI) and
    Jonathan's HR2-J (recursion helps carry-over). NOT_RESOLVED, SIGNIFICANT_BELOW_SESOI and FLOOR
    decide neither."""
    lit = {"SUPPORTED": "falsified", "EQUIVALENT": "confirmed", "CONTRADICTED": "confirmed"}.get(v, "undecided")
    j = {"SUPPORTED": "confirmed", "EQUIVALENT": "falsified", "CONTRADICTED": "falsified"}.get(v, "undecided")
    return {"literature_prediction": lit, "HR2_J": j}


def hr3_residuals(runs, cells, seeds):
    """Per seed, the OLS line of Y on the FineWeb panel loss through E1's five plain 20M cells; each
    recursion cell's residual at its own panel loss, flagged when that loss lies outside the plain
    cells' range in that seed."""
    per_seed, flags = [], []
    for r in seeds:
        xs = [get(runs[(c, r)], "panel_fineweb") for c in PLAIN_20M]
        ys = [get(runs[(c, r)], "Y") for c in PLAIN_20M]
        xm, ym = st.mean(xs), st.mean(ys)
        b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
        res = {}
        for c in cells:
            x = get(runs[(c, r)], "panel_fineweb")
            res[c] = get(runs[(c, r)], "Y") - (ym + b * (x - xm))
            if not min(xs) <= x <= max(xs):
                flags.append({"cell": c, "seed": r, "panel_fineweb": x, "plain_range": [min(xs), max(xs)]})
        per_seed.append(res)
    return per_seed, flags


def rec_r8_decision(runs):
    """A3.3's R8 rule, read once R2's and R4's post jobs are home: R8 runs only if R2 and R4 both pass
    the validity gate on their three seeds and the mean over seeds of Y(R4) - Y(R2) is at least 1 point
    (the SESOI per doubling). It reads only that number and the two cells' gates."""
    missing = [(c, r) for c in ("R2", "R4") for r in SEEDS_FIRST
               if (c, r) not in runs or runs[(c, r)].get("primary") is None]
    if missing:
        raise SystemExit(f"the R8 rule needs R2 and R4 on seeds {SEEDS_FIRST}: missing {missing}")
    g2, g4 = floor_ok(runs, "R2", SEEDS_FIRST), floor_ok(runs, "R4", SEEDS_FIRST)
    diffs = [get(runs[("R4", r)], "Y") - get(runs[("R2", r)], "Y") for r in SEEDS_FIRST]
    m = st.mean(diffs)
    run = bool(g2) and bool(g4) and m >= SESOI_E1
    return {"run_r8": run, "gate_R2": g2, "gate_R4": g4, "mean_Y_R4_minus_R2": m, "per_seed": diffs,
            "threshold": SESOI_E1, "seeds": SEEDS_FIRST, "rule": " ".join(rec_r8_decision.__doc__.split())}


def nearest_60m(runs60, flops):
    """The 60M plain cell nearest in FLOPs per token (log scale), with its mean Y: descriptive."""
    cells = {}
    for (c, r), s in runs60.items():
        if s.get("arch"):
            cells.setdefault(c, {"flops": flops_per_token(s["arch"]), "Y": []})["Y"].append(s.get("primary"))
    if not cells:
        return None
    c = min(cells, key=lambda k: abs(math.log(cells[k]["flops"] / flops)))
    ys = [y for y in cells[c]["Y"] if y is not None]
    return {"cell": f"60m_{c}", "flops_per_token": cells[c]["flops"], "Y_mean": st.mean(ys) if ys else None,
            "n": len(ys)}


def e1_rec(art, r1, r60, partial):
    """A3.3's E1-R block. HR1: within-seed OLS slope of Y on log2(r) over L4, R2, R4 (> 0, SESOI 1 point
    per doubling; with R8 the four-point slope is secondary). HR1-knowledge: the FineWeb panel loss's
    slope on the same cells, predicted not to fall (falsified by a SUPPORTED fall, SESOI 0.01 nats).
    HR2: per seed, the mean of Y(R2) - Y(L8) and Y(R4) - Y(L16) (SESOI 3 points; with R8 the three-pair
    mean adds Y(R8) - Y(L32), secondary), read for the literature's prediction and for HR2-J. HR3: per
    seed, the mean residual of R2 and R4 from the line of Y on FineWeb panel loss through E1's five plain
    20M cells (> 0, SESOI 3 points; R8 secondary; extrapolations flagged). HR4 (only with R8):
    the quadratic contrast of Y over log2(r) at r = 1, 2, 4, 8, descriptive. Secondary: MC7 - R4 and
    MC7 - L16. FLOPs per token and Y per L4-hour (S10) per cell; the 60M cell nearest in FLOPs beside each
    recursion cell (descriptive). The validity gate applies to every cell and a Y contrast with a FLOOR
    cell is FLOOR; panel-loss contrasts are not gated (as E1's)."""
    rr = {**load(art, "e1_rec"), **load(art, "e1_rec_r8")}
    if not rr:
        return None
    runs = {**r1, **rr}
    have_r8 = any(c == "R8" for c, _ in rr)
    core = PLAIN_20M + ["R2", "R4", "MC7"]
    cells = core + (["R8"] if have_r8 else [])
    seeds = seeds_of(runs, cells)
    miss = cells_complete(runs, cells, seeds)
    blocks = {**{c: "e1" for c in PLAIN_20M}, "L8": "e5", "R2": "e1_rec", "R4": "e1_rec", "MC7": "e1_rec",
              "R8": "e1_rec_r8"}
    res = {"label": "REGISTERED (A3.3)", "seeds": seeds, "missing": miss, "r8_ran": have_r8,
           "cells": cell_table(runs, cells, seeds, CELL_KEYS), "families": family_table(runs, cells, seeds),
           "cost": rec_cell_table(runs, art, blocks, cells, seeds),
           "matching": "non-embedding parameters only; recursion multiplies FLOPs (no compute-matched verdict); "
                       "the recursion cells carry 10.49M embedding against their twins' 7.34M (L8), 5.24M (L16) "
                       "and 3.15M (L32), so a win over a twin is ambiguous between recursion and lexical capacity"}
    if miss and not partial:
        raise SystemExit(f"E1-R incomplete: {miss} (use --partial for a descriptive read)")
    if miss:
        res["label"] = "DESCRIPTIVE, PARTIAL"
        return res
    res["floor_ok"] = {c: floor_ok(runs, c, seeds) for c in cells}
    hr1 = ["L4", "R2", "R4"]
    w1 = slope_weights([math.log2(REC_PASSES[c]) for c in hr1])
    res["HR1_passes_slope_Y"] = gated(runs, hr1, seeds,
                                      lambda: decide(contrast(runs, hr1, w1, seeds, "Y"), SESOI_E1, +1))
    k = decide(contrast(runs, hr1, w1, seeds, "panel_fineweb"), SESOI_LOSS_E1, -1)
    k["prediction"] = "falsified (a SUPPORTED fall)" if k["verdict"] == "SUPPORTED" else "not falsified"
    res["HR1_knowledge_passes_slope_fineweb_loss"] = k
    pairs = [("R2", "L8"), ("R4", "L16")]

    def pair_mean(ps):
        cs = [x for p in ps for x in p]
        w = [v for _ in ps for v in (1 / len(ps), -1 / len(ps))]
        return gated(runs, cs, seeds, lambda: decide(contrast(runs, cs, w, seeds, "Y"), SESOI_REC, +1))
    h2 = pair_mean(pairs)
    h2.update(readings_hr2(h2["verdict"]))
    res["HR2_recursion_minus_plain_depth"] = h2
    per_seed, flags = hr3_residuals(runs, ["R2", "R4"] + (["R8"] if have_r8 else []), seeds)
    res["HR3_residual_from_plain_isoloss_line"] = gated(
        runs, PLAIN_20M + ["R2", "R4"], seeds,
        lambda: {**decide([(d["R2"] + d["R4"]) / 2 for d in per_seed], SESOI_REC, +1),
                 "per_cell_mean_residual": {c: st.mean(d[c] for d in per_seed) for c in per_seed[0]},
                 "extrapolations": flags})
    res["secondary_MC7_minus_R4"] = {**gated(runs, ["MC7", "R4"], seeds, lambda: decide(
        contrast(runs, ["MC7", "R4"], [1, -1], seeds, "Y"), SESOI_REC, +1)), "label": "SECONDARY"}
    res["secondary_MC7_minus_L16"] = {**gated(runs, ["MC7", "L16"], seeds, lambda: decide(
        contrast(runs, ["MC7", "L16"], [1, -1], seeds, "Y"), SESOI_REC, +1)), "label": "SECONDARY"}
    if have_r8:
        hr8 = ["L4", "R2", "R4", "R8"]
        x8 = [math.log2(REC_PASSES[c]) for c in hr8]
        w8 = slope_weights(x8)
        res["secondary_HR1_slope_with_R8"] = {**gated(runs, hr8, seeds, lambda: decide(
            contrast(runs, hr8, w8, seeds, "Y"), SESOI_E1, +1)), "label": "SECONDARY"}
        k8 = decide(contrast(runs, hr8, w8, seeds, "panel_fineweb"), SESOI_LOSS_E1, -1)
        res["secondary_HR1_knowledge_with_R8"] = {**k8, "label": "SECONDARY"}
        h28 = pair_mean(pairs + [("R8", "L32")])
        res["secondary_HR2_three_pairs"] = {**h28, **readings_hr2(h28["verdict"]), "label": "SECONDARY"}
        res["secondary_HR3_with_R8"] = {**gated(runs, PLAIN_20M + ["R2", "R4", "R8"], seeds, lambda: decide(
            [(d["R2"] + d["R4"] + d["R8"]) / 3 for d in per_seed], SESOI_REC, +1)), "label": "SECONDARY"}
        xm = st.mean(x8)
        q = [(x - xm) ** 2 for x in x8]
        quad = [v - st.mean(q) for v in q]
        vals = contrast(runs, hr8, quad, seeds, "Y")
        res["HR4_quadratic_Y_over_log2_passes"] = {"label": "DESCRIPTIVE (no verdict)", "per_seed": vals,
                                                   "mean": st.mean(vals), "negative": st.mean(vals) < 0,
                                                   "sign_agreement": f"{sum(1 for v in vals if v < 0)}/{len(vals)} negative"}
    res["nearest_60m_in_flops"] = {c: nearest_60m(r60, res["cost"][c]["flops_per_token"])
                                   for c in ("R2", "R4", "MC7", "R8") if c in res["cost"]
                                   and res["cost"][c]["flops_per_token"]}
    return res


def throughput(art, size):
    out = {}
    for p in sorted(glob.glob(os.path.join(art, {"20m": "s0", "60m": "s0_60m", "rec": "s0_rec"}[size], "probe_*",
                                           "summary.json"))):
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
    ap.add_argument("--pilot-verdict", default=None, metavar="OUT",
                    help="amendment 3: write the floor pilot's verdict (ART/pilot) to OUT and do nothing else")
    ap.add_argument("--rec-decision", default=None, metavar="OUT",
                    help="A3.3: write the R8 rule's decision (ART/e1_rec, R2 and R4) to OUT and do nothing else")
    a = ap.parse_args(argv)
    if a.rec_decision:
        v = rec_r8_decision(load(a.art, "e1_rec"))
        os.makedirs(os.path.dirname(os.path.abspath(a.rec_decision)), exist_ok=True)
        open(a.rec_decision, "w", encoding="utf-8", newline="\n").write(json.dumps(v, indent=1) + "\n")
        print(json.dumps(v, indent=1))
        return 0
    if a.pilot_verdict:
        v = pilot_verdict(a.art)
        os.makedirs(os.path.dirname(os.path.abspath(a.pilot_verdict)), exist_ok=True)
        open(a.pilot_verdict, "w", encoding="utf-8", newline="\n").write(json.dumps(v, indent=1) + "\n")
        print(json.dumps(v, indent=1))
        return 0
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
    rec = e1_rec(a.art, r1, r60, a.partial)  # A3.3: E1-R, beside E1 at 20M (its r = 1 cell and twins)
    if rec is not None:
        res["E1_rec_20m"] = rec
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
    rrec = {**load(a.art, "e1_rec"), **load(a.art, "e1_rec_r8")}
    if rrec:  # the recursion arm's rungs: logged as E1's, read descriptively (not in amendment 1's registered set)
        early["E1_rec"] = rung_analysis(rrec, {**load_rungs(a.art, "e1_rec"), **load_rungs(a.art, "e1_rec_r8")},
                                        "DESCRIPTIVE (recursion arm, A3.3)")
    res["early_ranking"] = early
    res["asha"] = asha_summary(a.art)
    res["throughput_20m"] = throughput(a.art, "20m")
    res["throughput_60m"] = throughput(a.art, "60m")
    res["throughput_rec"] = throughput(a.art, "rec")
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
