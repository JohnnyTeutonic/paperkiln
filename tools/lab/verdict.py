"""The frozen decision rules: adaptive against fixed, and the error-directed comparison. Claude Opus 5.5, 7 Nov 2026.

Written so that fixed can win. The burden sits on the adaptive arm: every verdict but ADAPTIVE_BETTER keeps the fixed curriculum. The unit is the template group, resampled in pairs, as analytic_rubric_v2 does. Between-seed variance is added from a
separately measured noise pair, because a bootstrap over groups cannot see a model-level shift. Under three seed pairs the most an arm can reach is SCREEN_PROMISING. Task and small talk never combine, and small talk cannot offset a task loss.
Guards are non-compensatory, and a guard tripped by the adaptive arm alone counts against it. The first matching rule stands.

verdict_from_ledger is the only route to a recorded outcome: it takes a ledger and a rule, recomputes the audit and the compute from receipts, and voids anything the registration does not match.
"""
from __future__ import annotations

import math
import random

import exposure as X
import firewall as FW
from schema import Malformed, need, sha

MIN_SEED_PAIRS, MIN_GROUPS = 3, 8
ADAPTIVE = ("VOID", "GUARD_FAILED", "ADAPTIVE_NOT_EXERCISED", "ADAPTIVE_BETTER", "SCREEN_PROMISING", "FIXED_BETTER", "NO_ADAPTIVE_BENEFIT", "UNRESOLVED")
ERROR = ("VOID", "GUARD_FAILED", "SELECTION_ADDS", "SCREEN_PROMISING", "VERIFIED_ROWS_ONLY", "NEITHER_BEATS_ORDINARY", "UNRESOLVED")
MEANING = {
    "VOID": "The comparison did not hold its conditions. Nothing is read.",
    "GUARD_FAILED": "A non-compensatory guard failed. The fixed curriculum stands, and the guard is the recorded cost.",
    "ADAPTIVE_NOT_EXERCISED": "The adaptive arm never left balanced, so this compared two copies of the fixed curriculum. It tests nothing about adaptation.",
    "ADAPTIVE_BETTER": "On the frozen primary, at matched exposure and compute, this sampling rule beat this fixed mixture on this base, across the seed pairs read. It says nothing of other bases, shapes or rules.",
    "SCREEN_PROMISING": "A gain on too few seed pairs. Not a claim: it earns the replicate. The fixed curriculum stands meanwhile.",
    "FIXED_BETTER": "The fixed curriculum beat the adaptive rule by at least the practical margin. Use the fixed curriculum.",
    "NO_ADAPTIVE_BENEFIT": "The interval excludes a gain of the practical size. The fixed curriculum is simpler, so it stands.",
    "UNRESOLVED": "Neither side is shown. The fixed curriculum stands by default, and this is not evidence for either.",
    "SELECTION_ADDS": "Failure-selected rows beat randomly sampled verified rows at matched exposure, in one round, on this base.",
    "VERIFIED_ROWS_ONLY": "Verified rows beat ordinary SFT, and failure selection adds nothing to them. Keep the verified rows and drop the mining.",
    "NEITHER_BEATS_ORDINARY": "Neither verified arm beat ordinary SFT at this dose. Read the exposure audit before calling it a refutation.",
}
BANNED = ("recursive self-improvement", "recursive improvement", "intelligence explosion", "intelligence amplification", "self-improving intelligence", "autonomous intelligence")


def lint_claim(text):
    """Refuses language the design rules out, in anything a verdict or registration says of itself."""
    low = text.lower()
    hits = [b for b in BANNED if b in low]
    if hits:
        raise Malformed(f"claim language: {hits}")
    return text


def _mean(xs):
    return sum(xs) / len(xs)


def paired_boot(groups, draws, seed):
    """Percentile interval on the mean of group differences. groups: {group: difference}, already averaged over seed pairs."""
    keys = sorted(groups)
    vals = [groups[k] for k in keys]
    n = len(vals)
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return _mean(vals), means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def with_seed_noise(mean, lo, hi, pair_sd, pairs):
    """Adds between-seed variance to the group interval: pair_sd is the measured SD of the difference between two same-allocation runs, so it is known from outside and needs no t quantile on these pairs."""
    half = math.sqrt(((hi - lo) / 2) ** 2 + (1.96 * pair_sd / math.sqrt(pairs)) ** 2)
    return mean - half, mean + half


def pool_seed_pairs(pairs):
    """pairs: list of {group: adaptive - fixed}. Groups must match across pairs: a group missing from one pair would change what is being compared."""
    keys = set(pairs[0])
    if any(set(p) != keys for p in pairs):
        raise Malformed("verdict: seed pairs cover different template groups")
    return {k: _mean([p[k] for p in pairs]) for k in keys}


def _guards(guards):
    return sorted(k for k, v in guards.items() if not need(v, "holds", bool, f"guard {k}"))


def _check_rule(reg):
    if need(reg, "practical_points", float, "rule") <= 0:
        raise Malformed("rule: a practical margin is positive, frozen before training")
    if need(reg, "pair_sd_points", float, "rule") <= 0:
        raise Malformed("rule: between-seed spread is measured first, on a separate fixed pair")
    if need(reg, "min_seed_pairs", int, "rule") < MIN_SEED_PAIRS:
        raise Malformed(f"rule: no claim on under {MIN_SEED_PAIRS} seed pairs")
    if need(reg, "min_groups", int, "rule") < MIN_GROUPS:
        raise Malformed(f"rule: no claim on under {MIN_GROUPS} template groups")


def adaptive_verdict(reg, inp):
    """reg: the frozen rule: practical_points, pair_sd_points, min_seed_pairs, min_groups, boot_draws, boot_seed, compute_tolerance. inp: integrity_problems, exposure, compute, guards, task_pairs, small_talk_pairs, exercised."""
    _check_rule(reg)
    practical = reg["practical_points"]
    out = {"rule_sha": sha(reg), "meaning": None, "operating_choice": "fixed", "primary": None, "small_talk": None, "warnings": list(inp.get("exposure", {}).get("warnings", []))}
    problems = list(inp.get("integrity_problems", [])) + list(inp.get("exposure", {}).get("problems", []))
    if not inp.get("exposure", {}).get("ok", False) and not problems:
        problems.append("no exposure audit")
    comp = inp.get("compute")
    if not comp:
        problems.append("no compute receipt")
    else:
        fx, ad = comp["fixed_minutes"], comp["adaptive_minutes"]
        out["compute"] = dict(comp, overhead_minutes=round(ad - fx, 2))
        if ad > fx * (1 + reg["compute_tolerance"]):
            problems.append(f"adaptive compute {ad:.1f} passes fixed {fx:.1f} beyond tolerance: overhead is counted, not forgiven")
        if comp.get("fixed_dev_reads") != comp.get("adaptive_dev_reads"):
            problems.append("the arms had different numbers of development reads")
        if comp.get("read_share", 0) > reg.get("flag_read_share", 0.10):
            out["warnings"].append(f"reads and leg boundaries took {comp['read_share']:.1%} of the envelope: a plain fixed run would have had that compute for training")
    if not inp.get("task_pairs"):
        problems.append("no primary read")
    elif len(inp["task_pairs"][0]) < reg["min_groups"]:
        problems.append("too few template groups on the primary")
    if not inp.get("guards"):
        problems.append("no guard read")
    if problems:
        return dict(out, verdict="VOID", meaning=MEANING["VOID"], problems=problems)
    pairs = inp["task_pairs"]
    task = pool_seed_pairs(pairs)
    mean, lo, hi = paired_boot(task, reg["boot_draws"], reg["boot_seed"])
    lo, hi = with_seed_noise(mean, lo, hi, reg["pair_sd_points"], len(pairs))
    out["primary"] = {"gain_points": round(mean, 3), "interval_95_with_seed_noise": [round(lo, 3), round(hi, 3)], "groups": len(task), "seed_pairs": len(pairs), "by_seed_pair": [round(_mean(list(p.values())), 3) for p in pairs]}
    if inp.get("small_talk_pairs"):
        s_mean, s_lo, s_hi = paired_boot(pool_seed_pairs(inp["small_talk_pairs"]), reg["boot_draws"], reg["boot_seed"] + 1)
        s_lo, s_hi = with_seed_noise(s_mean, s_lo, s_hi, reg["pair_sd_points"], len(inp["small_talk_pairs"]))
        out["small_talk"] = {"gain_points": round(s_mean, 3), "interval_95_with_seed_noise": [round(s_lo, 3), round(s_hi, 3)], "reported_only": True}
    failed = _guards(inp["guards"])
    if failed:
        return dict(out, verdict="GUARD_FAILED", meaning=MEANING["GUARD_FAILED"], failed_guards=failed)
    if inp.get("exercised") is False:
        return dict(out, verdict="ADAPTIVE_NOT_EXERCISED", meaning=MEANING["ADAPTIVE_NOT_EXERCISED"])
    same_sign = all(x > 0 for x in out["primary"]["by_seed_pair"])
    if hi < 0 and mean <= -practical:
        v = "FIXED_BETTER"
    elif lo > 0 and mean >= practical and same_sign:
        v = "ADAPTIVE_BETTER" if len(pairs) >= reg["min_seed_pairs"] else "SCREEN_PROMISING"
    elif hi < practical:
        v = "NO_ADAPTIVE_BENEFIT"
    else:
        v = "UNRESOLVED"
    if v == "ADAPTIVE_BETTER":
        out["operating_choice"] = "adaptive"
    return dict(out, verdict=v, meaning=lint_claim(MEANING[v]))


def error_verdict(reg, inp):
    """Three arms from one parent: e0 ordinary continuation, e1 random verified rows, e2 failure-selected verified rows. The test is e2 against e1, not e2 against nothing. Every claim branch carries the checks of adaptive_verdict, and SELECTION_ADDS also needs the chatflow bar."""
    _check_rule(reg)
    practical = reg["practical_points"]
    out = {"rule_sha": sha(reg), "operating_choice": "ordinary"}
    problems = list(inp.get("integrity_problems", [])) + list(inp.get("exposure", {}).get("problems", []))
    if not inp.get("exposure", {}).get("ok", False) and not problems:
        problems.append("no exposure audit across the three arms")
    if inp.get("precision") is None or inp["precision"] < reg["min_failure_precision"]:
        problems.append("blind failure-precision audit missing or under its bar")
    if not inp.get("e2_minus_e1") or not inp.get("e1_minus_e0"):
        problems.append("no primary read")
    elif len(inp["e2_minus_e1"][0]) < reg["min_groups"]:
        problems.append("too few template groups on the primary")
    if not inp.get("guards"):
        problems.append("no guard read")
    if problems:
        return dict(out, verdict="VOID", meaning=MEANING["VOID"], problems=problems)
    a, b = inp["e2_minus_e1"], inp["e1_minus_e0"]
    m21, l21, h21 = paired_boot(pool_seed_pairs(a), reg["boot_draws"], reg["boot_seed"])
    m10, l10, h10 = paired_boot(pool_seed_pairs(b), reg["boot_draws"], reg["boot_seed"] + 1)
    l21, h21 = with_seed_noise(m21, l21, h21, reg["pair_sd_points"], len(a))
    l10, h10 = with_seed_noise(m10, l10, h10, reg["pair_sd_points"], len(b))
    out["primary"] = {"e2_minus_e1": [round(x, 3) for x in (m21, l21, h21)], "e1_minus_e0": [round(x, 3) for x in (m10, l10, h10)], "seed_pairs": len(a)}
    failed = _guards(inp["guards"])
    if failed:
        return dict(out, verdict="GUARD_FAILED", meaning=MEANING["GUARD_FAILED"], failed_guards=failed)
    enough = len(a) >= reg["min_seed_pairs"]
    if l21 > 0 and m21 >= practical and all(_mean(list(p.values())) > 0 for p in a):
        if not enough or not inp.get("chatflow_bar"):
            return dict(out, verdict="SCREEN_PROMISING", meaning=MEANING["SCREEN_PROMISING"])
        out["operating_choice"] = "error_selected"
        return dict(out, verdict="SELECTION_ADDS", meaning=MEANING["SELECTION_ADDS"])
    if l10 > 0 and m10 >= practical and all(_mean(list(p.values())) > 0 for p in b):
        if not enough:
            return dict(out, verdict="SCREEN_PROMISING", meaning=MEANING["SCREEN_PROMISING"])
        out["operating_choice"] = "verified"
        return dict(out, verdict="VERIFIED_ROWS_ONLY", meaning=MEANING["VERIFIED_ROWS_ONLY"])
    if h21 < practical and h10 < practical:
        return dict(out, verdict="NEITHER_BEATS_ORDINARY", meaning=MEANING["NEITHER_BEATS_ORDINARY"])
    return dict(out, verdict="UNRESOLVED", meaning=MEANING["UNRESOLVED"])


def trajectory(points):
    """Performance against compute: [(minutes, score)], trapezoid area over the span. Reported beside the verdict, and decides nothing."""
    pts = sorted(points)
    if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
        return None
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
    return round(area / (pts[-1][0] - pts[0][0]), 4)


def _groups(records, experiment_id, arm, instrument, key):
    rows = [r["body"] for r in records if r["kind"] == "evaluation" and r["body"]["experiment_id"] == experiment_id and r["body"]["arm"] == arm and r["body"]["instrument_id"] == instrument]
    if len(rows) != 1:
        return None
    return rows[0]["summary"].get(key)


def verdict_from_ledger(records, experiment_id, rule):
    """rule adds to adaptive_verdict's: pairs [[fixed_arm, adaptive_arm], ...], replay_families, replay_floor, shortcut_read, shortcut_bar. Everything else comes from the ledger."""
    from controller import registration_of
    from ledger import superseded
    reg = registration_of(records, experiment_id)
    b = reg["body"]
    problems = []
    if sha(rule) != b["frozen_rule_sha"]:
        problems.append("this rule is not the registered one")
    gone = superseded(records)
    recs = [r for r in records if r["seq"] not in gone]
    try:
        FW.scan_all(recs)
        FW.check_shortcut_read(rule.get("shortcut_read", {}), rule.get("shortcut_bar", 0.0))
    except FW.FirewallBreach as e:
        problems.append(str(e))
    instrument = b["frozen_evaluation"][0]
    legs_of = lambda arm: [r["body"] for r in recs if r["kind"] == "leg" and r["body"]["experiment_id"] == experiment_id and r["body"]["arm"] == arm]  # noqa: E731
    audits, task, talk = [], [], []
    tripped, moved = set(), False
    fx_min, ad_min, fx_reads, ad_reads, fx_read_min = 0.0, 0.0, 0, 0, 0.0
    for fx, ad in rule["pairs"]:
        if b["arms"].get(fx) != "fixed" or b["arms"].get(ad) != "adaptive":
            problems.append(f"pair {fx}, {ad} is not a registered fixed and adaptive arm")
            continue
        audits.append(X.audit_pair(legs_of(fx), legs_of(ad), b["start_checkpoint"]))
        for arm in (fx, ad):
            bad = X.replay_floor_held(legs_of(arm), set(rule["replay_families"]), rule["replay_floor"])
            if bad:
                problems.append(f"{arm}: replay under its floor at {bad}")
        mf, ma = X.minutes_from_ledger(recs, experiment_id, fx), X.minutes_from_ledger(recs, experiment_id, ad)
        fx_min += mf["total_minutes"]
        ad_min += ma["total_minutes"]
        fx_read_min += mf["read_minutes"]
        fx_reads += mf["dev_reads"]
        ad_reads += ma["dev_reads"]
        decisions = {arm: [r["body"] for r in recs if r["kind"] == "decision" and r["body"]["experiment_id"] == experiment_id and r["body"]["arm"] == arm] for arm in (fx, ad)}
        trip_a = any("guard_tripped" in d["flags"] for d in decisions[ad])
        trip_f = any("guard_tripped" in d["flags"] for d in decisions[fx])
        if trip_a and not trip_f:
            tripped.add(ad)
        moved = moved or any(abs(m - 1.0) > 1e-6 for d in decisions[ad] for m in d["multipliers"].values())
        for key, sink in (("task_groups", task), ("small_talk_groups", talk)):
            gf, ga = _groups(recs, experiment_id, fx, instrument, key), _groups(recs, experiment_id, ad, instrument, key)
            if gf is None or ga is None or set(gf) != set(ga):
                problems.append(f"{key}: missing or mismatched evaluation for {fx}, {ad}")
                continue
            sink.append({g: ga[g] - gf[g] for g in ga})
    named = {}
    for arm in {x for p in rule["pairs"] for x in p}:
        rows = [r["body"]["summary"].get("guards", {}) for r in recs if r["kind"] == "evaluation" and r["body"]["experiment_id"] == experiment_id and r["body"]["arm"] == arm]
        for r in rows:
            for k, v in r.items():
                named.setdefault(k, {"holds": True})
                named[k]["holds"] = named[k]["holds"] and bool(v["holds"])
    if set(named) != set(b["guards"]):
        problems.append(f"guards read {sorted(named)} differ from those registered {sorted(b['guards'])}")
    if tripped:
        named["family_regression"] = {"holds": False}
    bad_audits = [p for a in audits for p in a["problems"]]
    exposure = {"ok": bool(audits) and not bad_audits, "problems": bad_audits, "warnings": [w for a in audits for w in a["warnings"]]}
    total = max(1.0, fx_min)
    inp = {"integrity_problems": problems, "exposure": exposure, "task_pairs": task, "small_talk_pairs": talk, "guards": named, "exercised": moved,
           "compute": {"fixed_minutes": fx_min, "adaptive_minutes": ad_min, "fixed_dev_reads": fx_reads, "adaptive_dev_reads": ad_reads, "read_share": round(fx_read_min / total, 4)}}
    res = adaptive_verdict(rule, inp)
    res["attempts_under_this_hypothesis"] = len({r["body"]["experiment_id"] for r in records if r["kind"] == "registration" and r["body"]["hypothesis_id"] == b["hypothesis_id"]})
    return res


SIGNAL_BAR, SIGNAL_MIN_CELLS, SIGNAL_MIN_RANGE_NATS, SIGNAL_MIN_RANGE_FAMILIES = 0.5, 10, 0.30, 5


def signal_gate(dev_cells, primary_cells):
    """Does the development signal track the primary? Frozen in PLAN.md before any model is read. dev_cells and primary_cells: {(model, family): value}, dev in nats per target on T_M, lower better, primary in task points on T_E, higher better.
    Both are centred within each family, so family difficulty cannot carry the rank: the controller asks how far a family sits from its best, and this asks the same of both reads. The read needs range: if too few families differ across models on dev, the
    models cannot test the signal, and that is reported as not informative, which does not pass. Fails closed on few cells."""
    from history import spearman
    keys = sorted(set(dev_cells) & set(primary_cells))
    fams = sorted({k[1] for k in keys})
    ranged = [f for f in fams if max(dev_cells[k] for k in keys if k[1] == f) - min(dev_cells[k] for k in keys if k[1] == f) >= SIGNAL_MIN_RANGE_NATS]
    if len(keys) < SIGNAL_MIN_CELLS:
        return {"passes": False, "cells": len(keys), "spearman": None, "note": "too few cells to read"}
    if len(ranged) < SIGNAL_MIN_RANGE_FAMILIES:
        return {"passes": False, "cells": len(keys), "families_with_range": len(ranged), "spearman": None, "note": "not informative: too few families vary across these models"}

    def centred(cells):
        mean = {f: _mean([cells[k] for k in keys if k[1] == f]) for f in fams}
        return [cells[k] - mean[k[1]] for k in keys]
    rho = spearman([-x for x in centred(dev_cells)], centred(primary_cells))
    return {"passes": rho is not None and rho >= SIGNAL_BAR, "cells": len(keys), "families_with_range": len(ranged), "spearman": rho, "bar": SIGNAL_BAR}
