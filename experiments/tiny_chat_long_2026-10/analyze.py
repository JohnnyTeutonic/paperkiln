#!/usr/bin/env python3
"""The registered analysis of Carry-12's cells (PREREGISTRATION_DRAFT.md sections 7-9), from the post summaries
the cells' jobs bring home (tools/colab_chat/carry12_<cell>_r<seed>.json; tiny_chat_run.py post with
--carry12-items). Refuses an incomplete contrast unless --partial (then every block is labelled DESCRIPTIVE,
PARTIAL and no verdict is printed).

    python experiments/tiny_chat_long_2026-10/analyze.py --art C:/ml_artifacts/tiny_chat [--out analysis.json]
        [--partial] [--lstar 8] [--r-sign +1|-1] [--e1-home C:/ml_artifacts/tiny_chat/e1_60m_L8]

ART holds the runners' artefact folders; every ART/*/post_c12_<cell>_r<seed>/summary.json (and ART/post_c12_*/...)
is read. Cells: W512 (E1 at 60M's L* runs re-SFT'd at T 512, Carry-12 forced choice only), B, D, S-gist (summary
cell "Sgist"), P16 (B itself when L* = 16), X, and the conditional S-full ("Sfull") and R.

One number per (cell, seed); paired contrasts over seeds 1-3 (4-5 when every cell of the contrast has them),
one-sample t (n = 3: critical 4.303), the verdict rule of E5/E1's analyze.py (decide(): SUPPORTED, CONTRADICTED,
EQUIVALENT = the 90% interval inside +-SESOI, SIGNIFICANT_BELOW_SESOI, NOT_RESOLVED) and its extension rule
(seeds 4-5 once, for a primary contrast NOT_RESOLVED with one sign on all three seeds: extend_to_5_seeds).
SESOI: 3 points for pairwise and interaction contrasts, 1 point per turn for slopes on distance, 0.02 nats for
panel losses. Y12 = forced-choice accuracy (acc) with self history over the primary families (recall,
correction, topic, load, knowledge) minus that on their twins; Y12_gold the same with gold history.

Validity gate (section 7), on each cell's means over seeds: (i) twin accuracy over the primary families at least
0.35 and (ii) twin minus question-alone over the context families (recall, correction, topic, load) at least 0.10.
A contrast that uses a cell failing either is FLOOR.

Registered contrasts (section 8):
- HT   Y12_gold(B) - Y12_gold(W512) > 0 (gold history, which both fit); HT-0, its counter, is EQUIVALENT.
- HD   Y12(D) - Y12(B) > 0; HD-0 is EQUIVALENT.
- HS1  Y12(S-gist) - Y12(B) < 0; HS1-alt holds when the verdict is EQUIVALENT or CONTRADICTED (the difference is
       at least -3 points). Descriptive beside it: the retention ratio (acc(S-gist) - acc(noctx)) /
       (acc(B) - acc(noctx)) by distance, gold history, context families.
- HS0  (conditional: only with S-full home) Y12(S-full) - Y12(B), predicted EQUIVALENT; the mask's own cost
       Y12(S-gist) - Y12(S-full) beside it.
- HR   (conditional: only with R home; --r-sign gives the 20M contrast's sign) Y12(R) - Y12(B) in that direction:
       SUPPORTED means the sign transferred.
- HX   H-gap: D_X = [Y_gold,far(X) - Y_gold,far(P16)] - [Y_near(X) - Y_near(P16)] > 0 (far: recall with gold
       history; near: the same items' near rendering; each Y is that accuracy minus recall's twin); and the slope on
       distance (1-11) of the gold-history gain X - P16 in the correction family > 0 (1 point per turn). HX-main:
       Y12(X) - Y12(P16) (counter: EQUIVALENT). The isoplot: X's residual from the line of Y12 on the FineWeb
       panel loss through the plain T 1,024 cells (B and P16) in each seed, flagged thin (two points per seed;
       none when L* = 16).
Secondary, descriptive: every cell's means (Y12, Y12_gold, the decomposition twin -> near -> gold -> self, the
validity quantities, Carry-5's Y, the panel losses, instruction compliance), accuracy by distance, and W512's
reproduction of E1's SFT (its sft_weights_sha256 against E1's home post summaries, --e1-home).
The registered S-full trigger is read from HS1 (s_full_condition); section 6's wording and HS1's direction disagree
(section 11 records it for a ruling), so both readings are reported.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tiny_chat_2026-10")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("tiny_chat_analyze", os.path.join(EXP, "analyze.py"))   # decide(), the t tables, the extension rule
decide = base.decide

SEEDS_FIRST, SEEDS_EXT = [1, 2, 3], [4, 5]
SESOI = 0.03            # pairwise and interaction contrasts: 3 points
SESOI_SLOPE = 0.01      # slopes on distance: 1 point per turn
SESOI_LOSS = 0.02       # panel losses, pairwise
TWIN_MIN = 0.35         # gate (i): chance + 10 points over the primary families
TWIN_NOCTX_MIN = 0.10   # gate (ii): twin minus question alone over the context families
FLOOR_TEXT = ("FLOOR (twin accuracy over the primary families under 0.35, or twin within 10 points of the question "
              "alone on the context families)")
PARTIAL = "DESCRIPTIVE, PARTIAL"
CORRECTION_DISTANCES = range(1, 12)


# ---------------------------------------------------------------- reading the post summaries
def load(art):
    """{(cell, seed): summary} for every post_c12_* job under ART (one or two folder levels down)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(art, "*", "post_c12_*", "summary.json"))
                    + glob.glob(os.path.join(art, "post_c12_*", "summary.json"))):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        key = (s["cell"], s["seed"])
        if key in out:
            raise SystemExit(f"{p}: a second summary for cell {key[0]} seed {key[1]}")
        s["_path"] = p
        out[key] = s
    return out


def pool(s, mode, which="primary", kind="acc"):
    c12 = s.get("carry12") or {}
    return ((c12.get("pools") or {}).get(kind, {}).get(which, {}).get(mode) or {}).get("mean")


def fam(s, mode, family, kind="acc"):
    c12 = s.get("carry12") or {}
    return ((c12.get("pools") or {}).get(kind, {}).get("family", {}).get(mode, {}).get(family) or {}).get("mean")


def sub(x, y):
    return None if x is None or y is None else x - y


def get(s, key):
    """One run's number. Y12 and Y12_gold from the post summary (carry12_eval's derived quantities); the rest
    from Carry-12's pools (forced choice, acc; over_ctx counts 0, as Carry-5)."""
    c12 = s.get("carry12") or {}
    d = c12.get("derived") or {}
    vals = {
        "Y12": s.get("Y12"), "Y12_gold": s.get("Y12_gold"),
        "twin_primary": pool(s, "twin"), "noctx_primary": pool(s, "noctx"),
        "twin_ctx": pool(s, "twin", "context"), "noctx_ctx": pool(s, "noctx", "context"),
        "twin_minus_noctx_ctx": sub(pool(s, "twin", "context"), pool(s, "noctx", "context")),
        "gold_primary": pool(s, "gold"), "self_primary": pool(s, "self"), "near_ctx": pool(s, "near", "context"),
        "split_cost_gold": d.get("split_cost_gold"), "distance_cost_gold": d.get("distance_cost_gold"),
        "own_replies_cost": d.get("own_replies_cost"),
        "Y_gold_far": sub(fam(s, "gold", "recall"), fam(s, "twin", "recall")),
        "Y_near_recall": sub(fam(s, "near", "recall"), fam(s, "twin", "recall")),
        "carry5_Y": s.get("primary"),
        "panel_fineweb": (s.get("panels") or {}).get("fineweb"),
        "panel_dialogue": (s.get("panels") or {}).get("dialogue"),
        "instruction_complies_self": ((c12.get("instruction_complies") or {}).get("self") or {}).get("mean"),
    }
    return vals[key]


CELL_KEYS = ["Y12", "Y12_gold", "twin_primary", "noctx_primary", "twin_ctx", "noctx_ctx", "twin_minus_noctx_ctx",
             "gold_primary", "self_primary", "near_ctx", "split_cost_gold", "distance_cost_gold", "own_replies_cost",
             "Y_gold_far", "Y_near_recall", "carry5_Y", "panel_fineweb", "panel_dialogue", "instruction_complies_self"]


def by_distance(s, mode, family=None):
    """{distance: acc} with gold/near/self/twin history: all families (carry12 'distance'), or one context
    family (family_distance)."""
    c12 = s.get("carry12") or {}
    rows = (c12.get("family_distance") or {}).get(mode, {}).get(family, {}) if family else \
        (c12.get("distance") or {}).get(mode, {})
    return {int(k): v["mean"] for k, v in rows.items() if v.get("mean") is not None}


# ---------------------------------------------------------------- seeds, gates, contrasts
def seeds_of(runs, cells):
    have = [r for r in SEEDS_FIRST + SEEDS_EXT if all((c, r) in runs for c in cells)]
    return have if len(have) >= 3 else SEEDS_FIRST


def missing(runs, cells, seeds, key="Y12"):
    return [(c, r) for c in cells for r in seeds if (c, r) not in runs or get(runs[(c, r)], key) is None]


def floor_ok(runs, cell, seeds):
    tw = [get(runs[(cell, r)], "twin_primary") for r in seeds if (cell, r) in runs]
    tq = [get(runs[(cell, r)], "twin_minus_noctx_ctx") for r in seeds if (cell, r) in runs]
    if not tw or any(x is None for x in tw + tq):
        return None
    return st.mean(tw) >= TWIN_MIN and st.mean(tq) >= TWIN_NOCTX_MIN


def gated(runs, cells, seeds, fn):
    bad = [c for c in cells if floor_ok(runs, c, seeds) is False]
    return {"verdict": FLOOR_TEXT + " in " + ", ".join(bad)} if bad else fn()


def paired(runs, a, b, seeds, key):
    return [get(runs[(a, r)], key) - get(runs[(b, r)], key) for r in seeds]


def cell_table(runs, cells, seeds):
    out = {}
    for c in cells:
        row = {}
        for k in CELL_KEYS:
            xs = [get(runs[(c, r)], k) for r in seeds if (c, r) in runs]
            xs = [x for x in xs if x is not None]
            row[k] = {"mean": st.mean(xs) if xs else None,
                      "se": st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None, "n": len(xs)}
        row["validity_gate"] = floor_ok(runs, c, seeds)
        out[c] = row
    return out


def ols(xs, ys):
    xm, ym = st.mean(xs), st.mean(ys)
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    return ym - b * xm, b


def correction_gain_slopes(runs, x, p, seeds):
    """Per seed, the OLS slope on distance (1-11) of the gold-history gain acc(X) - acc(P16) in the correction
    family, over the distances both cells have."""
    out = []
    for r in seeds:
        gx, gp = by_distance(runs[(x, r)], "gold", "correction"), by_distance(runs[(p, r)], "gold", "correction")
        ds = [d for d in CORRECTION_DISTANCES if d in gx and d in gp]
        fit = ols(ds, [gx[d] - gp[d] for d in ds]) if len(ds) >= 2 else None
        out.append(fit[1] if fit else None)
    return out


def isoplot_residuals(runs, plain, cell, seeds):
    """Per seed, CELL's residual from the line of Y12 on the FineWeb panel loss through the PLAIN cells (two at
    L* != 16); flagged when CELL's loss lies outside the plain cells' range that seed."""
    res, flags = [], []
    for r in seeds:
        xs = [get(runs[(c, r)], "panel_fineweb") for c in plain]
        ys = [get(runs[(c, r)], "Y12") for c in plain]
        fit = ols(xs, ys) if None not in xs + ys and len(set(xs)) >= 2 else None
        if fit is None:
            return None, [{"seed": r, "why": "no line: fewer than two distinct plain losses"}]
        x = get(runs[(cell, r)], "panel_fineweb")
        res.append(get(runs[(cell, r)], "Y12") - (fit[0] + fit[1] * x))
        if not min(xs) <= x <= max(xs):
            flags.append({"seed": r, "panel_fineweb": x, "plain_range": [min(xs), max(xs)]})
    return res, flags


CONTEXT_FAMILIES = ("recall", "correction", "topic", "load")


def ctx_by_distance(s, mode):
    """{distance: acc} over the context families (family_distance pooled by item counts)."""
    rows = ((s.get("carry12") or {}).get("family_distance") or {}).get(mode, {})
    acc = {}
    for f in CONTEXT_FAMILIES:
        for k, v in (rows.get(f) or {}).items():
            if v.get("mean") is not None and v.get("n"):
                t = acc.setdefault(int(k), [0.0, 0])
                t[0] += v["mean"] * v["n"]
                t[1] += v["n"]
    return {d: x / n for d, (x, n) in acc.items() if n}


def retention(runs, seeds):
    """(acc(S-gist) - acc(noctx)) / (acc(B) - acc(noctx)) by distance, gold history, the context families, means
    over seeds; noctx is each cell's question-alone accuracy over the context families (chance by construction)."""
    out = {}
    for d in range(1, 14):
        num, den = [], []
        for r in seeds:
            s, b = runs[("Sgist", r)], runs[("B", r)]
            gs, gb = ctx_by_distance(s, "gold").get(d), ctx_by_distance(b, "gold").get(d)
            if gs is None or gb is None:
                continue
            num.append(gs - get(s, "noctx_ctx"))
            den.append(gb - get(b, "noctx_ctx"))
        if num and st.mean(den) != 0:
            out[str(d)] = {"ratio": st.mean(num) / st.mean(den), "n_seeds": len(num)}
    return out


def w512_reproduction(runs, e1_home, lstar):
    """Descriptive: W512's re-run SFT weights against E1 at 60M's post summaries at home (same seed)."""
    out = {}
    for r in SEEDS_FIRST:
        s = runs.get(("W512", r))
        p = os.path.join(e1_home, f"post_e1_60m_L{lstar}_r{r}", "summary.json") if e1_home else None
        if s is None or not p or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            e1 = json.load(f)
        out[str(r)] = {"sft_weights_identical": s.get("sft_weights_sha256") == e1.get("sft_weights_sha256"),
                       "pt_weights_identical": s.get("pt_weights_sha256") == e1.get("pt_weights_sha256"),
                       "panel_fineweb_diff": sub((s.get("panels") or {}).get("fineweb"),
                                                 (e1.get("panels") or {}).get("fineweb"))}
    return out


# ---------------------------------------------------------------- the analysis
def analyze(runs, partial=False, lstar=8, r_sign=None, e1_home=None):
    p16 = "B" if lstar == 16 else "P16"
    core = ["W512", "B", "D", "Sgist", "X"] + ([p16] if p16 != "B" else [])
    seeds = seeds_of(runs, core)
    miss = missing(runs, [c for c in core if c != "W512"], seeds) + missing(runs, ["W512"], seeds, "Y12_gold")
    present = sorted({c for c, _ in runs})
    res = {"lstar": lstar, "p16_cell": p16, "seeds": seeds, "missing": miss,
           "cells": cell_table(runs, present, seeds),
           "sesoi": {"pairwise": SESOI, "interaction": SESOI, "slope_per_turn": SESOI_SLOPE, "loss_nats": SESOI_LOSS}}
    if miss and not partial:
        raise SystemExit(f"Carry-12 core incomplete: {miss} (use --partial for a descriptive read)")
    if miss:
        res["label"] = PARTIAL
        return res
    res["label"] = "REGISTERED (PREREGISTRATION.md section 9)"
    res["validity_gate"] = {c: floor_ok(runs, c, seeds) for c in present}

    def contrast(a, b, key, direction, sesoi=SESOI, cells=None):
        return gated(runs, cells or [a, b], seeds, lambda: decide(paired(runs, a, b, seeds, key), sesoi, direction))

    ht = contrast("B", "W512", "Y12_gold", +1)
    ht["counter_HT_0"] = "confirmed (EQUIVALENT)" if ht["verdict"] == "EQUIVALENT" else "not confirmed"
    res["HT_B_minus_W512_Y12_gold"] = ht
    hd = contrast("D", "B", "Y12", +1)
    hd["counter_HD_0"] = "confirmed (EQUIVALENT)" if hd["verdict"] == "EQUIVALENT" else "not confirmed"
    res["HD_D_minus_B"] = hd
    hs1 = contrast("Sgist", "B", "Y12", -1)
    hs1["HS1_alt"] = ("holds (the difference is at least -3 points)" if hs1["verdict"] in ("EQUIVALENT", "CONTRADICTED")
                      else "does not hold" if hs1["verdict"] in ("SUPPORTED", "SIGNIFICANT_BELOW_SESOI") else "undecided")
    res["HS1_Sgist_minus_B"] = hs1
    res["HS1_retention_ratio_by_distance"] = retention(runs, seeds)
    res["s_full_condition"] = {
        "HS1_verdict": hs1["verdict"],
        "section_6_text_HS1_CONTRADICTED": hs1["verdict"] == "CONTRADICTED",
        "s_gist_loses_HS1_SUPPORTED": hs1["verdict"] == "SUPPORTED",
        "note": "section 6 runs S-full 'only if HS1 is CONTRADICTED' and 'only if S-gist loses'; HS1's direction makes "
                "'S-gist loses' a SUPPORTED HS1. The frozen file states one trigger; both readings are reported."}
    if all(("Sfull", r) in runs for r in seeds):
        h0 = contrast("Sfull", "B", "Y12", +1)
        h0["prediction"] = "EQUIVALENT (the slots carry no information under the plain mask)"
        res["HS0_Sfull_minus_B"] = h0
        res["HS0_mask_cost_Sgist_minus_Sfull"] = {**contrast("Sgist", "Sfull", "Y12", +1), "label": "REPORTED BESIDE HS0"}
    if all(("R", r) in runs for r in seeds):
        if r_sign not in (+1, -1):
            res["HR_R_minus_B"] = {"verdict": "NOT READ: --r-sign (the 20M recursion contrast's sign) is needed"}
        else:
            hr = contrast("R", "B", "Y12", r_sign)
            hr["registered_sign"] = r_sign
            hr["sign_transferred"] = hr["verdict"] == "SUPPORTED"
            res["HR_R_minus_B"] = hr
    # HX: the gap interaction, the correction slope, the main effect and the isoplot.
    if p16 == "B":
        res["HX"] = {"note": "L* = 16: P16 is B; X is read against B"}
    dx = [(get(runs[("X", r)], "Y_gold_far") - get(runs[(p16, r)], "Y_gold_far"))
          - (get(runs[("X", r)], "Y_near_recall") - get(runs[(p16, r)], "Y_near_recall")) for r in seeds]
    res["HX_gap_D_X"] = gated(runs, ["X", p16], seeds, lambda: decide(dx, SESOI, +1))
    slopes = correction_gain_slopes(runs, "X", p16, seeds)
    res["HX_gap_correction_slope"] = gated(runs, ["X", p16], seeds, lambda: (
        decide(slopes, SESOI_SLOPE, +1) if None not in slopes
        else {"verdict": "NOT_RESOLVED", "why": "a seed without two common correction distances", "per_seed": slopes}))
    both = [res["HX_gap_D_X"].get("verdict"), res["HX_gap_correction_slope"].get("verdict")]
    floor = next((v for v in both if v and v.startswith("FLOOR")), None)
    res["HX_H_gap"] = {"verdict": floor if floor else "SUPPORTED" if both == ["SUPPORTED", "SUPPORTED"] else
                       "CONTRADICTED" if "CONTRADICTED" in both else
                       "EQUIVALENT (the counter-hypothesis's D_X)" if both[0] == "EQUIVALENT" else "NOT_RESOLVED",
                       "parts": both, "rule": "H-gap needs D_X > SESOI and a positive correction slope, both SUPPORTED"}
    xm = contrast("X", p16, "Y12", +1)
    xm["counter_no_main_effect"] = "confirmed (EQUIVALENT)" if xm["verdict"] == "EQUIVALENT" else "not confirmed"
    res["HX_main_X_minus_P16"] = xm
    plain = ["B"] + ([p16] if p16 != "B" else [])
    resid, flags = isoplot_residuals(runs, plain, "X", seeds)
    res["HX_isoplot_residual"] = ({"verdict": "NOT READ", "why": flags} if resid is None else
                                  gated(runs, plain + ["X"], seeds, lambda: {
                                      **decide(resid, SESOI, +1), "flags": flags,
                                      "thin": f"the line runs through {len(plain)} plain T 1,024 cells per seed"}))
    res["secondary_panel_fineweb_X_minus_P16"] = decide(paired(runs, "X", p16, seeds, "panel_fineweb"), SESOI_LOSS, -1)
    res["secondary_panel_fineweb_B_minus_W512"] = {
        **decide(paired(runs, "B", "W512", seeds, "panel_fineweb"), SESOI_LOSS, -1),
        "note": "B's panels hold 512 windows of 1,024 tokens, W512's of 512: not like for like"}
    res["w512_reproduces_e1"] = w512_reproduction(runs, e1_home, lstar)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--lstar", type=int, default=None,
                    help="L* (default: experiments/tiny_chat_2026-10/size/depth_choice.json)")
    ap.add_argument("--r-sign", type=int, choices=[-1, 1], default=None,
                    help="HR: the sign of the SUPPORTED 20M recursion contrast (only when R ran)")
    ap.add_argument("--e1-home", default=None, help="E1 at 60M's L* folder at home (W512's reproduction check)")
    a = ap.parse_args(argv)
    lstar = a.lstar
    if lstar is None:
        with open(os.path.join(EXP, "size", "depth_choice.json"), encoding="utf-8") as f:
            lstar = int(json.load(f)["depth"])
    res = {"partial": a.partial, "carry12": analyze(load(a.art), a.partial, lstar, a.r_sign, a.e1_home)}
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
