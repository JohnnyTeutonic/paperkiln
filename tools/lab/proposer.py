"""The experiment proposer: an opinion list with its arithmetic shown. Ranks interventions, explains the ranking, launches nothing. Claude Opus 5.5, 7 Nov 2026.

Hypothesis generation is apart from authorisation and execution. Every entry declares what the brief lists, or it is refused. Priors are named readings by their producer, set by hand, and they decide the order; nothing here learns from the ledger yet,
beyond reading a saturation flag. Entries rank only within one comparable group, the same base and endpoint scale: told recall, task points, LAMBADA and two bases are not one scale.
  information  0.5 log2(1 + prior variance / noise variance), bits, for one Gaussian read
  performance  E[max(0, effect)] under the prior, points. A measurement has none: it gets information only.
Both per L4-hour, each scaled by the group's largest ready entry, averaged. An entry with an unmet prerequisite is listed as blocked, unranked. When the SFT line is saturated, SFT entries drop below everything else ready, so better pretraining can come first.
"""
from __future__ import annotations

import math

from schema import Malformed, need, nonneg

ENGINEERING = ("t_c_all_families", "t_e_rows_built", "dev_read_on_vm", "packet_writer", "dev_signal_checked", "s54_j1_read")
REQUIRED = ("id", "class", "group", "hypothesis", "prerequisites", "estimated_l4_hours", "starting_checkpoint", "control_condition", "changed_facet", "frozen_evaluation", "primary_endpoint", "guards",
            "registration_status", "review_status", "stop_criteria", "continue_criteria", "effect_prior")


def validate(entry):
    for key in REQUIRED:
        if key not in entry:
            raise Malformed(f"intervention {entry.get('id', '?')}: missing {key}")
    p = need(entry, "effect_prior", dict, entry["id"])
    need(p, "mean_points", float, entry["id"])
    if nonneg(need(p, "sd_points", float, entry["id"]), "sd") <= 0 or nonneg(need(p, "noise_sd_points", float, entry["id"]), "noise") <= 0:
        raise Malformed(f"intervention {entry['id']}: prior and noise need positive spread")
    need(p, "producer", str, entry["id"])
    if nonneg(entry["estimated_l4_hours"], "hours") <= 0:
        raise Malformed(f"intervention {entry['id']}: no compute estimate")
    return entry


def information_bits(p):
    return 0.5 * math.log2(1 + (p["sd_points"] / p["noise_sd_points"]) ** 2)


def expected_gain(p):
    mu, sd = p["mean_points"], p["sd_points"]
    z = mu / sd
    pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return sd * pdf + mu * cdf


def saturated_from_ledger(records):
    """True if the latest decision of any arm carries the saturated flag: the one thing the proposer reads from history."""
    latest = {}
    for r in records:
        if r["kind"] == "decision":
            latest[(r["body"]["experiment_id"], r["body"]["arm"])] = r["body"]
    return any("saturated" in d["flags"] for d in latest.values())


def rank(entries, facts):
    """facts: {prerequisite: bool}. An unknown prerequisite counts as unmet."""
    ready, blocked = [], []
    saturated = bool(facts.get("sft_saturated"))
    for e in map(validate, entries):
        unmet = [p for p in e["prerequisites"] if not facts.get(p)]
        p = e["effect_prior"]
        perf = 0.0 if e["class"] == "measurement" else expected_gain(p)
        row = {"id": e["id"], "class": e["class"], "group": e["group"], "hours": e["estimated_l4_hours"], "information_bits": round(information_bits(p), 3), "expected_gain_points": round(perf, 2), "prior": f"{p['mean_points']:+.1f} +/- {p['sd_points']:.1f} points, noise {p['noise_sd_points']:.1f}",
               "prior_producer": p["producer"], "unmet": unmet, "registration_status": e["registration_status"], "review_status": e["review_status"], "needs_authorisation": True, "may_launch": False}
        (blocked if unmet else ready).append(row)
    for g in {r["group"] for r in ready}:
        mine = [r for r in ready if r["group"] == g]
        i_top = max(r["information_bits"] / r["hours"] for r in mine) or 1.0
        p_top = max(r["expected_gain_points"] / r["hours"] for r in mine) or 1.0
        for r in mine:
            r["score"] = round(0.5 * r["information_bits"] / r["hours"] / i_top + 0.5 * r["expected_gain_points"] / r["hours"] / p_top, 3)
    ready.sort(key=lambda r: (saturated and r["class"] == "sft", r["group"], -r["score"], r["id"]))
    blocked.sort(key=lambda r: (len(r["unmet"]), r["id"]))
    return {"ranked": ready, "blocked": blocked, "note": "Ranking only. Nothing here is authorised, and nothing here can launch."}
