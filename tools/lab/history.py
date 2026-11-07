"""Historical receipts as observational evidence, and the offline replay. Claude Opus 5.5, 7 Nov 2026.

OBSERVATIONAL, ONE SEED PER MODEL. These models were chosen by people for reasons tied to what they had just seen, on one weak base, with allocations nobody randomised. Nothing here is a trial of the controller, and it cannot show
that adaptive selection works. It serves schema design, and one question the design has to answer: on the only clean pair on record, does the loss the controller reads agree with the outcomes that matter?

snapshot() writes numbers only: family losses, leg exposures and stored outcome reads, with the source files' hashes.
"""
from __future__ import annotations

import json
from pathlib import Path

import controller as C
from schema import sha_file

BANNER = "OBSERVATIONAL: historical receipts, one seed per model, one base, allocations not randomised. Not a controlled comparison, and not evidence that adaptive selection works."
ORDER = ("P", "S23", "S30", "S34", "S51", "S52")
SOURCES = ("analysis/family_exposure.json", "analysis/family_loss.json", "analysis/a19_score_s2.json", "analysis/analytic_rubric_v2.json", "analysis/a20_bars.json")
BASE = "sfull_lineage"


def snapshot(screens_dir, fe):
    """fe: family_exposure, through adapters, for its family map and its stored read table. The result holds no row wording."""
    root = Path(screens_dir)
    load = lambda rel: json.loads((root / rel).read_text(encoding="utf-8"))
    ex, lo, sc, ar, bars = (load(s) for s in SOURCES)
    loss = {f: {m: round(v[m]["all"], 4) for m in ORDER} for f, v in lo["loss"].items()}
    se = {f: {m: round((v[m]["all"][2] - v[m]["all"][1]) / 3.92, 4) for m in v} for f, v in lo["gain_from_parent"].items()}
    leg_m, leg_c = ex["legs"]["a19/M1/train.jsonl|seed2|1024"], ex["legs"]["a16/S32/train.jsonl|seed2|1024"]

    def dose(leg, s):
        field = "sampled_supervised" if s in fe.CHAT else "sampled_decision"
        return sum(leg.get(f, {}).get(field, 0) for f in fe.SETS[s])
    stored = {name: row for name, row, _ in fe.STORED if name.startswith(("J1", "Told-value"))}
    prof = ar["consensus"]
    return {"schema": "lab-history-v1", "banner": BANNER, "producer": "Claude Opus 5.5, 7 Nov 2026", "base_id": BASE, "sources_sha256": {s: sha_file(root / s) for s in SOURCES},
            "families": sorted(loss), "per_family_rows": lo["per_family"], "loss": loss, "gain_se": se,
            "s51_minus_s52": {f: [round(x, 4) for x in lo["s51_minus_s52"][f]["all"]] for f in lo["s51_minus_s52"]},
            "leg_dose": {"M1_on_S34": {s: dose(leg_m, s) for s in loss}, "C0_on_S34": {s: dose(leg_c, s) for s in loss}},
            "leg_supervised_total": {"M1_on_S34": sum(v["sampled_supervised"] for v in leg_m.values()), "C0_on_S34": sum(v["sampled_supervised"] for v in leg_c.values())},
            "outcomes": {"j1_relevance": {m: float(stored["J1 relevance, mean of 13 turns"][m]) for m in ("S23", "S30", "S34", "S51", "S52")},
                         "told_of_108": {m: int(stored["Told-value recall, of 108"][m]) for m in ("S30", "S34", "S51", "S52")},
                         "analytic_v2_task": {m: prof["task"]["profile"][m] for m in ("S23", "S30", "S34", "S51", "S52")},
                         "analytic_v2_small_talk": {m: prof["social"]["profile"][m] for m in ("S23", "S30", "S34", "S51", "S52")},
                         "a19_verdict": sc["verdict"], "a19_reach_gain_items": sc["primary"]["gain_pool"], "a19_reach_bar": 13},
            "noise": {"told_of_108_null_sd": bars["nulls"]["different_one_step"]["pool_sd"], "told_of_108_null_p95": bars["nulls"]["different_one_step"]["pool_p95"]}}


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    top = sum((p - ma) * (q - mb) for p, q in zip(ra, rb))
    bot = (sum((p - ma) ** 2 for p in ra) * sum((q - mb) ** 2 for q in rb)) ** 0.5
    return round(top / bot, 3) if bot else None


def replay(snap):
    """What the rule would have proposed at S34, against what was run. Floors use only models that existed then; a hindsight row shows how far look-ahead would move it."""
    fams = [f for f in snap["families"] if f != "tail:human"]
    loss, se = snap["loss"], snap["gain_se"]
    out = {"banner": BANNER, "decision_point": "after S34's 2,048 steps, the state A19 started from", "evidence": "observational", "families": len(fams)}

    def run(models):
        cfg = {"schema": "lab-config-v1", "experiment_id": "history", "base_id": BASE, "arm_policies": {"replay": "adaptive"}, "allow_synthetic": True,
               "families": {f: {"balanced_share": 1.0 / len(fams), "floor": min(loss[f][m] for m in models), "floor_se": 0.03, "floor_base_id": BASE} for f in fams},
               "replay_share": 0.30, "eta": 0.5, "z": 2.0, "excess_scale": 1.0, "min_excess": 0.15, "multiplier_min": 0.5, "multiplier_max": 2.0, "regression_nats": 0.15, "min_gain_nats": 0.10, "min_dose_tokens": 4000.0, "min_rows": 20, "patience": 2,
               "legs": 4, "seed": 1, "matched_pilot": True, "leg_l4_minutes": 0.0, "budget_l4_minutes": 1000.0}
        C.load_config(cfg)
        read = {f: {"n": snap["per_family_rows"], "mean": loss[f]["S34"], "se": se[f]["S34"]} for f in fams}  # the standard error of the gain from the parent, as the nearest stored proxy for a read's
        state = {"reads": [read], "read_legs": [0], "legs": [], "failed_legs": [], "spent_l4": 0.0, "multipliers": None}
        return C.adaptive_decide(state, cfg, "replay", 1, "0" * 64)
    then = run(("P", "S23", "S30", "S34"))
    hind = run(("P", "S23", "S30", "S34", "S51", "S52"))
    m1, c0 = snap["leg_dose"]["M1_on_S34"], snap["leg_dose"]["C0_on_S34"]
    mult = [then["multipliers"][f] for f in fams]
    out["floors_then"] = {f: min(loss[f][m] for m in ("P", "S23", "S30", "S34")) for f in fams}
    out["proposal_floors_then"] = {f: then["multipliers"][f] for f in fams}
    out["proposal_floors_hindsight"] = {f: hind["multipliers"][f] for f in fams}
    out["states_floors_then"] = {f: then["families"][f]["state"] for f in fams}
    hmult = [hind["multipliers"][f] for f in fams]
    out["agreement_with_M1_dose_spearman"] = spearman(mult, [m1[f] for f in fams])
    out["agreement_with_C0_dose_spearman"] = spearman(mult, [c0[f] for f in fams])
    out["hindsight_agreement_with_M1_dose_spearman"] = spearman(hmult, [m1[f] for f in fams])
    out["hindsight_agreement_with_C0_dose_spearman"] = spearman(hmult, [c0[f] for f in fams])
    out["families_pushed"] = sorted(f for f in fams if then["families"][f]["state"] == "pushed")
    t = snap["leg_supervised_total"]
    out["equal_steps_unequal_tokens"] = {"M1_leg": t["M1_on_S34"], "C0_leg": t["C0_on_S34"], "ratio": round(t["C0_on_S34"] / t["M1_on_S34"], 2)}
    return out


def mean_excess(snap, model):
    floors = {f: min(snap["loss"][f].values()) for f in snap["families"]}
    return sum(snap["loss"][f][model] - floors[f] for f in snap["families"]) / len(snap["families"])


def signal_check(snap):
    """Does the development-style signal agree with the outcomes? Five models: description, not inference. The rubric is under development and exploratory, and reads here only as an outcome column, never as a target."""
    o = snap["outcomes"]
    models = ("S23", "S30", "S34", "S51", "S52")
    ex = [-mean_excess(snap, m) for m in models]
    rows = {"lower_mean_excess": {m: round(mean_excess(snap, m), 3) for m in models}}
    for key in ("j1_relevance", "analytic_v2_task", "analytic_v2_small_talk"):
        rows[f"spearman_with_{key}"] = spearman(ex, [o[key][m] for m in models])
    told = [m for m in models if m in o["told_of_108"]]
    rows["spearman_with_told_of_108"] = spearman([-mean_excess(snap, m) for m in told], [o["told_of_108"][m] for m in told])
    gains = snap["s51_minus_s52"]
    pair = {"family_nats_gained_summed": round(sum(max(0.0, v[0]) for v in gains.values()), 2), "family_nats_lost_summed": round(sum(max(0.0, -v[0]) for v in gains.values()), 2),
            "j1_relevance": [o["j1_relevance"]["S51"], o["j1_relevance"]["S52"]], "analytic_v2_task": [o["analytic_v2_task"]["S51"], o["analytic_v2_task"]["S52"]], "analytic_v2_small_talk": [o["analytic_v2_small_talk"]["S51"], o["analytic_v2_small_talk"]["S52"]],
            "told_of_108": [o["told_of_108"]["S51"], o["told_of_108"]["S52"]], "a19_verdict": o["a19_verdict"], "a19_reach": [o["a19_reach_gain_items"], o["a19_reach_bar"]]}
    return {"banner": BANNER, "models": rows, "s51_against_s52": pair, "reporting_rule": "On S-full's base an intervention is conditional on a weak base, not failed (Jonathan, 7 Nov 2026)."}


def observation_from_receipt(folder, screen, base_id):
    """One screen's receipts as an observation body: numbers and hashes, no logs, no wording."""
    d = Path(folder)
    res = json.loads((d / f"result_{screen}.json").read_text(encoding="utf-8"))
    home = json.loads((d / "home.json").read_text(encoding="utf-8"))
    keep = ("T", "accum", "batch", "d", "final_step", "heads", "layers", "lr", "params", "seed", "steps_requested", "tokens_per_second", "wall_seconds", "best_val")
    return {"source": f"{d.name}/result_{screen}.json", "source_sha": sha_file(d / f"result_{screen}.json"), "model": screen, "base_id": base_id, "data": dict({k: res[k] for k in keep if k in res}, gguf_sha256=home.get(screen))}
