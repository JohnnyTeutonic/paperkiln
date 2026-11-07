"""L4 costs and prerequisites for the two pilots, on two starts. Claude Opus 5.5, 7 Nov 2026. Arithmetic on recorded rates; nothing here is a measurement of a pilot.

Sources:
  s_per_step   8x768 at 1.63: correction_loop_pilot.md section 4. 12x768 at 1.83: S54 3,719.7 s for 2,048 steps and S55 1,898.8 s for 1,024, A20 receipts, 8x2 on an L4.
  overhead     19.0 minutes a session, median; 15.2 best, 31.4 worst: a18_exposure_audit.md section 7. S54 and S55 ran 18.2 and 15.7.
  transition   13 minutes across the six leg boundaries of a pair: an estimate. It is what closes the record's 2.7 L4-hours for design B, and no leg boundary has been timed.
  usd_per_hour 0.59: RunPod L4, b1_ledger.md. Colab L4 sessions spend units instead.
"""
from __future__ import annotations

STARTS = {
    "pk_sft_v1": {"label": "October's pk_sft_v1 line, 12x768, k = 0, T 512 pretraining", "s_per_step": 1.83, "start_of_adaptive": "pk_sft_v1", "parent_of_error_pilot": "S54"},
    "stronger_8x768": {"label": "the selected stronger base, if 8x768 wins, k = 4", "s_per_step": 1.63, "start_of_adaptive": "B-SFT", "parent_of_error_pilot": "B-SFT then a fixed recipe"},
    "stronger_12x768": {"label": "the selected stronger base, if 12x768 wins, k = 4", "s_per_step": 1.85, "start_of_adaptive": "B-SFT", "parent_of_error_pilot": "B-SFT then a fixed recipe"},
}
OVERHEAD = {"best": 15.2, "median": 19.0, "worst": 31.4}
TRANSITION_PER_PAIR = 13.0
SESSION_CAP = 120.0
USD_PER_HOUR = 0.59
INPUTS, SAFETENSORS = 0.7, 2.6


def train_minutes(steps, rate):
    return steps * rate / 60.0


def sessions(minutes_of_work, overhead):
    """Sessions needed under the 120-minute cap, whole arms not split."""
    n, used = 1, overhead
    for m in minutes_of_work:
        if used + m > SESSION_CAP:
            n += 1
            used = overhead
        used += m
    return n


def adaptive_pair(start, overhead="median"):
    """Two arms of four 512-step legs, with dev reads at 512, 1,024, 1,536 in both: one seed pair of the comparison, or the separate noise pair. TRANSITION_PER_PAIR is an estimate that no timed boundary backs."""
    rate = STARTS[start]["s_per_step"]
    train = train_minutes(4096, rate)
    n = sessions([train / 2, train / 2], OVERHEAD[overhead])
    total = train + n * OVERHEAD[overhead] + TRANSITION_PER_PAIR + 2 * INPUTS + 2 * SAFETENSORS
    return {"train_minutes": round(train, 1), "sessions": n, "minutes": round(total, 1), "l4_hours": round(total / 60, 2), "usd": round(total / 60 * USD_PER_HOUR, 2)}


def one_arm(start, steps, overhead="median"):
    t = train_minutes(steps, STARTS[start]["s_per_step"]) + OVERHEAD[overhead] + INPUTS + SAFETENSORS
    return {"minutes": round(t, 1), "l4_hours": round(t / 60, 2), "usd": round(t / 60 * USD_PER_HOUR, 2)}


def error_pilot(start, steps=1024, overhead="median"):
    """E0, E1 and E2 from one parent at matched exposure. 1,024 steps by default, A19's reason: at 512 the role-fact line sat under its bar on every seed read."""
    rate = STARTS[start]["s_per_step"]
    per = train_minutes(steps, rate) + SAFETENSORS
    n = sessions([per, per, per], OVERHEAD[overhead])
    total = 3 * per + n * (OVERHEAD[overhead] + INPUTS)
    return {"arms": 3, "steps_each": steps, "sessions": n, "minutes": round(total, 1), "l4_hours": round(total / 60, 2), "usd": round(total / 60 * USD_PER_HOUR, 2)}


def table():
    out = {}
    for s, spec in STARTS.items():
        pair = adaptive_pair(s)
        spec_arms = 1 if s == "pk_sft_v1" else 3
        spec_cost = round(one_arm(s, 1024)["l4_hours"] * spec_arms if s == "pk_sft_v1" else (one_arm(s, 2048)["l4_hours"] + 2 * one_arm(s, 1024)["l4_hours"]), 2)
        err = error_pilot(s)
        total = round(spec_cost + 4 * pair["l4_hours"] + err["l4_hours"], 2)
        out[s] = {"label": spec["label"], "specialist_arms": spec_arms, "specialist_l4_hours": spec_cost, "two_arm_block": pair, "two_arm_block_worst": adaptive_pair(s, "worst"), "noise_pair_l4_hours": pair["l4_hours"], "comparison_three_pairs_l4_hours": round(3 * pair["l4_hours"], 2),
                  "error_pilot": err, "error_pilot_512": error_pilot(s, 512), "all_l4_hours": total, "all_usd": round(total * USD_PER_HOUR, 2), "to_the_stage_gate_l4_hours": round(spec_cost + pair["l4_hours"], 2)}
    return out


# Status at 7 Nov 2026. yes, no, partial, or n/a. Each line is one of the brief's conditions, or the record's.
PREREQUISITES = [
    ("not the weak S-full base (brief, section 7)", "yes", "yes"),
    ("the selected stronger base exists (brief, section 7)", "n/a", "no: the race has not read"),
    ("a fixed SFT baseline on this base (brief, section 7)", "partial: S54 and S55 exist; A20's stored score reads S54 void on J1 and S55 flat", "no"),
    ("specialist floors from this base (design B)", "partial: S54 and S55; a mixed-row specialist is missing", "no: three arms"),
    ("candidate pool holds every eligible family", "no: T_C has 5 families; caps left nameless_role and affirm_in_role at 0, and contrast and stance rows are not cut", "no, the same"),
    ("development rows T_M", "yes: 744 rows, 840 decisions, 7 families", "yes"),
    ("evaluation rows T_E, by a blind author, with shortcut reads published first", "no: tables and definitions only; seven mixed families can start now", "no: the same"),
    ("per-family development read on the VM (mtstudio score is the path)", "no: wrapper unbuilt", "no: wrapper unbuilt"),
    ("packet writer from a leg plan", "no", "no"),
    ("noise floor: a separate same-allocation pair, excluded from the comparison", "no: it is the first L4 step", "no: it is the first L4 step"),
    ("development signal shown to track the primary", "no: CPU read on pk_sft_v1, S54 and S55 once T_M and T_E hold contrast rows", "no: only after its fixed arms"),
    ("A's gate: a family above 0.15 nats excess after a fixed mixture (scoping, section 6)", "unknown: the noise pair reads it", "unknown"),
    ("independent target tracker for rows, by another hand", "no", "no"),
    ("a timed leg boundary", "no", "no"),
    ("frozen registration, independent review, Jonathan's dated line", "no", "no"),
    ("does not touch the race's registration, pins, driver or budget", "yes, if run on a separate L4 line in the ledger", "yes, after the race"),
]


def prerequisites():
    return [{"condition": c, "pk_sft_v1": a, "stronger_base": b} for c, a, b in PREREQUISITES]
