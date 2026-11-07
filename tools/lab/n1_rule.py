"""The frozen rules of the lab's first L4 step on pk_sft_v1 (PROTOCOL.md section 33). Claude Opus 5.5, 7 Nov 2026. Pure functions, standard library only; nothing here trains, reads a model or reaches a network.

Three readings, in order, each from the step's own models:
  gate      design A's gate on the fixed mixture: does any family stay above its specialist floor by more than 0.15 nats beyond noise, after N1 and N2?
  naming    on N1, the eligible families the selector names: positive conservative excess over the specialist floor, the largest first, at most K, each with a
            same-floor family left over for its control. The control set: as many from each floor's stratum, drawn by a registered seed from the families not named.
  targeting T doubles the named families, C the control families, both from N1's weights, at equal decision tokens. Does a named family's extra dose move it
            more than a control family's moves it, on T_E task points?

The selector tested is the controller's, excess over a specialist floor, and not the raw loss the signal read ranked. Written so that it can fail: only
TARGETING_SUPPORTED opens a later registration, and no outcome here authorises adaptive allocation.
"""
from __future__ import annotations

import math
import random

Z = 2.5
MIN_EXCESS = 0.15
K = 3
MIN_NAMED = 2
CONTROL_SEED = 20261108
CONTROL_TRIES = 20
SIGMA_FLOOR_POINTS = 4.0
TARGET_BAR_POINTS = 5.0
MIN_TASK_PAIRS = 10
UNREADABLE_SHARE = 0.5
# Two-sided 95% t quantiles: sigma comes from one pair over the task families, so its degrees of freedom are the families less one.
T_975 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
# Families with T_E task pairs, a specialist floor, and a pool that carries twice N's draws at four draws a row (build_lab_n1.py; affirm_in_role and recall_abstain do not).
ELIGIBLE = ("correction", "ended_role", "how_are_you", "like_recall", "name_use", "nameless_role", "passage", "role_mix", "roleplay", "whose_name")
# Where each family's floor comes from: a model on pk_sft_v1 that trained that family. identity_stance has none: S54 trained contrast rows only.
FLOOR_SOURCE = {"affirm_in_role": "lab_sp", "ended_role": "lab_sp", "how_are_you": "lab_sp", "like_recall": "lab_sp", "name_use": "lab_sp", "nameless_role": "lab_sp", "role_mix": "lab_sp",
                "correction": "S54", "followup": "S54", "identity_topic": "S54", "passage": "S54", "recall_abstain": "S54", "roleplay": "S54", "whose_name": "S54"}
OUTCOMES = ("VOID", "PENDING", "GATE_CLOSED", "TOO_FEW_NAMED", "TARGETING_UNREADABLE", "TARGETING_SUPPORTED", "TARGETING_REVERSED", "TARGETING_UNRESOLVED")
MEANING = {
    "VOID": "an input exists and failed its check: section 33's rerun rule",
    "PENDING": "an input is missing",
    "GATE_CLOSED": "no family stays above its floor beyond noise after the fixed mixture: the fixed curriculum stands, T and C do not run, and no comparison is registered on this base",
    "TOO_FEW_NAMED": "fewer than two eligible families could be named with a same-floor control left, or no control set fits the pools: T and C do not run, and no comparison is registered on this base",
    "TARGETING_UNREADABLE": "T_E sits at a bound in both arms on half or more of the named or of the control families, so it cannot read targeting there: an instrument limit, not a reading of the selector; no comparison is registered on this base under this plan",
    "TARGETING_SUPPORTED": "dosing the families the selector names moved them more than the same extra dose moved same-floor families it did not name, beyond noise, on one seed: a comparison may be registered, with Jonathan's dated line",
    "TARGETING_REVERSED": "the same extra dose moved the families the selector did not name more: the selector misdirects dose here, and no comparison is registered on this base",
    "TARGETING_UNRESOLVED": "no difference beyond noise either way, on one seed: no comparison is registered on this base under this plan",
}
NOT_AUTHORISED = "No outcome of this step authorises adaptive allocation. The signal read's pooled pass does not either."


def family_read(rows):
    """rows: [{family, sum, n}], one per T_M row. Nats per target by family, with the ratio estimator's standard error over rows."""
    by = {}
    for r in rows:
        by.setdefault(r["family"], []).append((float(r["sum"]), int(r["n"])))
    out = {}
    for f, xs in by.items():
        m = len(xs)
        tot_s, tot_n = sum(s for s, _ in xs), sum(n for _, n in xs)
        mean = tot_s / tot_n
        nbar = tot_n / m
        var = sum((s - mean * n) ** 2 for s, n in xs) / (m - 1) / m / nbar ** 2 if m > 1 else float("inf")
        out[f] = {"mean": mean, "se": math.sqrt(var), "rows": m}
    return out


def floors(reads):
    """reads: {source: family_read}. Each family's floor from its registered source."""
    out = {}
    for f, src in FLOOR_SOURCE.items():
        if src in reads and f in reads[src]:
            out[f] = {"floor": reads[src][f]["mean"], "floor_se": reads[src][f]["se"], "source": src}
    return out


def _cons(excess, se):
    c = excess - Z * se
    return c if c >= MIN_EXCESS else 0.0


def gate(n1, n2, fl):
    """Design A's gate on the fixed mixture, both seeds: the mean less the floor, less Z standard errors that hold both reads, the floor and the seed spread.
    The two reads are on the same rows, so treating their errors as independent overstates the error: conservative."""
    fams = {}
    for f in sorted(set(n1) & set(n2) & set(fl)):
        a, b = n1[f], n2[f]
        excess = (a["mean"] + b["mean"]) / 2 - fl[f]["floor"]
        se = math.sqrt((a["se"] ** 2 + b["se"] ** 2) / 4 + fl[f]["floor_se"] ** 2 + (a["mean"] - b["mean"]) ** 2 / 4)
        fams[f] = {"excess": round(excess, 4), "se": round(se, 4), "conservative_excess": round(_cons(excess, se), 4), "floor_source": fl[f]["source"]}
    return {"open": any(v["conservative_excess"] > 0 for v in fams.values()), "families": fams}


def naming(n1, n2, fl, feasible=None):
    """On N1, where T and C start: excess over the floor, less Z standard errors holding N1's read, the floor and one run's seed variance from the pair.
    Named, in order of conservative excess, positive only, eligible only, at most K: a family is taken only if its floor's stratum keeps an unnamed eligible family
    for every named one, so each named family has a same-floor control and contrast families are never set against mixed ones.
    Control: in each stratum, as many as were named there, drawn at CONTROL_SEED from that stratum's unnamed eligible families. If the packet plan cannot fit the sets
    in the pools (feasible returns False), every stratum is drawn again at the next seed, at most CONTROL_TRIES times. That depends on pool sizes alone, never on a read."""
    fams = {}
    for f in ELIGIBLE:
        if f not in n1 or f not in n2 or f not in fl:
            continue
        a, b = n1[f], n2[f]
        excess = a["mean"] - fl[f]["floor"]
        se = math.sqrt(a["se"] ** 2 + fl[f]["floor_se"] ** 2 + (a["mean"] - b["mean"]) ** 2 / 2)
        fams[f] = {"excess": round(excess, 4), "se": round(se, 4), "conservative_excess": round(_cons(excess, se), 4), "floor_source": fl[f]["source"]}
    ranked = sorted((f for f, v in fams.items() if v["conservative_excess"] > 0), key=lambda f: (-fams[f]["conservative_excess"], f))
    named, per = [], {}
    for f in ranked:
        s = fams[f]["floor_source"]
        room = [g for g in fams if fams[g]["floor_source"] == s and g != f and g not in named]
        if len(room) - per.get(s, 0) >= 1:
            named.append(f)
            per[s] = per.get(s, 0) + 1
        if len(named) == K:
            break
    out = {"named": named, "named_by_stratum": per, "families": fams, "skipped_for_room": [f for f in ranked if f not in named][:K]}
    if len(named) < MIN_NAMED:
        return dict(out, control=[], enough=False, why="fewer than two named with a same-floor control left")
    for k in range(CONTROL_TRIES):
        rng = random.Random(CONTROL_SEED + k)
        control = []
        for s in sorted(per):
            rest = sorted(g for g in fams if fams[g]["floor_source"] == s and g not in named)
            control += rng.sample(rest, per[s])
        control = sorted(control)
        if feasible is None or feasible(named, control):
            return dict(out, control=control, enough=True, control_seed=CONTROL_SEED + k)
    return dict(out, control=[], enough=False, why=f"no control set fits the pools in {CONTROL_TRIES} draws")


def pair_sd(points_n1, points_n2, pairs):
    """pair_sd_points: the SD of the difference between two same-allocation runs, as verdict.with_seed_noise takes it, estimated by |N1 - N2| over every T_E task pair
    (one degree of freedom), floored at the difference two independent binomial reads of that many pairs would show at the pair's mean level. It feeds no rule in this step."""
    measured = abs(points_n1 - points_n2)
    p = (points_n1 + points_n2) / 200
    floor = 100 * math.sqrt(2 * p * (1 - p) / pairs) if pairs else 0.0
    return {"measured": round(measured, 3), "binomial_floor": round(floor, 3), "pair_sd_points": round(max(measured, floor), 3), "floored": measured < floor, "pairs": pairs}


def family_sigma(fam_n1, fam_n2, families, pairs):
    """SD of one family's two-run difference in task points at the mean family size, from the pair: each difference is scaled by sqrt(pairs_f / mean pairs), so
    small families do not set the scale, and centred on the pair's mean gap. Floored. Returns (sigma used, measured, degrees of freedom, mean pairs)."""
    nbar = sum(pairs[f] for f in families) / len(families)
    e = [(fam_n1[f] - fam_n2[f]) * math.sqrt(pairs[f] / nbar) for f in families]
    mean = sum(e) / len(e)
    var = sum((x - mean) ** 2 for x in e) / (len(e) - 1)
    return max(math.sqrt(var), SIGMA_FLOOR_POINTS), math.sqrt(var), len(e) - 1, nbar


def targeting(t_pts, c_pts, named, control, task_families, sigma, pairs, df, bar=TARGET_BAR_POINTS):
    """d = T - C by family. Named gain gA = mean over named of d, less the shift s on the task families neither arm doubled; control gain gB = mean over control of -d, plus s.
    The statistic is gA - gB = mean_named(d) + mean_control(d) - 2 mean_rest(d). Var(d_f) = sigma^2 x mean pairs / pairs_f. The class needs the statistic past
    max(bar, t(df) x SE), first unless half or more of either doubled set sits at 0 or 100 points in both arms."""
    rest = [f for f in task_families if f not in named and f not in control]
    if not named or len(named) != len(control) or len(rest) < 2:
        raise ValueError("targeting: equal non-empty sets and at least two families neither arm doubled")
    nbar = sum(pairs[f] for f in task_families) / len(task_families)
    v = {f: sigma ** 2 * nbar / pairs[f] for f in task_families}
    d = {f: t_pts[f] - c_pts[f] for f in task_families}
    shift = sum(d[f] for f in rest) / len(rest)
    ga = sum(d[f] for f in named) / len(named) - shift
    gb = -sum(d[f] for f in control) / len(control) + shift
    stat = ga - gb
    se = math.sqrt(sum(v[f] for f in named) / len(named) ** 2 + sum(v[f] for f in control) / len(control) ** 2 + 4 * sum(v[f] for f in rest) / len(rest) ** 2)
    tq = T_975.get(df, 2.571 if df < 5 else 2.131)
    need = max(bar, tq * se)
    bound = sorted(f for f in named + control if t_pts[f] == c_pts[f] and t_pts[f] in (0.0, 100.0))
    unread = any(sum(1 for f in grp if f in bound) >= UNREADABLE_SHARE * len(grp) for grp in (named, control))
    out = "TARGETING_UNREADABLE" if unread else "TARGETING_SUPPORTED" if stat >= need else "TARGETING_REVERSED" if stat <= -need else "TARGETING_UNRESOLVED"
    return {"outcome": out, "statistic": round(stat, 3), "named_gain": round(ga, 3), "control_gain": round(gb, 3), "shift_on_rest": round(shift, 3), "se": round(se, 3), "t_quantile": tq, "needed": round(need, 3),
            "by_family": {f: round(d[f], 3) for f in task_families}, "rest": rest, "at_a_bound_in_both_arms": bound}


def verdict(problems, missing, gate_res, name_res, target_res):
    """First match: void, pending, gate closed, too few named, then the targeting class."""
    if problems:
        return {"outcome": "VOID", "meaning": MEANING["VOID"], "problems": problems, "authorises": NOT_AUTHORISED}
    if missing:
        return {"outcome": "PENDING", "meaning": MEANING["PENDING"], "missing": missing, "authorises": NOT_AUTHORISED}
    if not gate_res["open"]:
        return {"outcome": "GATE_CLOSED", "meaning": MEANING["GATE_CLOSED"], "authorises": NOT_AUTHORISED}
    if not name_res["enough"]:
        return {"outcome": "TOO_FEW_NAMED", "meaning": MEANING["TOO_FEW_NAMED"], "why": name_res.get("why"), "authorises": NOT_AUTHORISED}
    if target_res is None:
        return {"outcome": "PENDING", "meaning": MEANING["PENDING"], "missing": ["T and C"], "authorises": NOT_AUTHORISED}
    return {"outcome": target_res["outcome"], "meaning": MEANING[target_res["outcome"]], "authorises": NOT_AUTHORISED}
