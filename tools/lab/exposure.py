"""From shares to a leg's row plan, and the audit that matches two arms. Claude Opus 5.5, 7 Nov 2026.

Family shares are shares of sampled decision tokens, design B's unit, the one that says how much a family was taught. The trainer draws rows uniformly from the file, so at equal steps the supervised total depends on the mixture: S51's leg sampled
1,322,814 supervised targets to S52's 2,356,725 at the same 1,024 steps (family_exposure.json). A leg therefore has two targets, decision tokens and supervised tokens, registered for both arms alike.

No family share moves to meet them. A family's draws follow from its share of the decision target; replay takes the draws that are left; and the supervised gap is absorbed by the mean length of the replay rows chosen, inside a registered range, with
replay's token share held at its floor. If that cannot be done the leg fails closed. An earlier version tilted family shares by row length, which reversed the controller's choices on the demo pool (review finding 4).

The trainer's draws are deterministic given packet and seed, so a packet is recounted before launch (family_exposure.leg_counts through adapters.py) and checked against its plan by prelaunch_recount. This module does not recount.
"""
from __future__ import annotations

from schema import Malformed, need, rounded


class ExposureUnmatched(Exception):
    """The arms cannot be matched within tolerance. Fails closed: the leg does not go ahead."""


def largest_remainder(fracs, total):
    raw = {f: v * total for f, v in fracs.items()}
    out = {f: int(v) for f, v in raw.items()}
    left = total - sum(out.values())
    for f in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:left]:
        out[f] += 1
    return out


def plan_leg(decision, pool, cfg):
    """decision: the controller's. pool: {family: {rows, mean_supervised, mean_decision}}, with '_replay': {rows, mean_supervised, min_mean_supervised, max_mean_supervised}. Returns row counts and expected exposure, or raises."""
    if decision["action"] != "ALLOCATE":
        raise Malformed(f"no plan for {decision['action']}")
    steps, batch = need(cfg, "steps_per_leg", int, "cfg"), need(cfg, "batch", int, "cfg")
    d_target = need(cfg, "decision_tokens_per_leg", float, "cfg")
    s_target = need(cfg, "supervised_tokens_per_leg", float, "cfg")
    rows_wanted = need(cfg, "packet_rows", int, "cfg")
    fams = sorted(decision["shares"])
    if "_replay" not in pool or any(f not in pool for f in fams):
        raise Malformed("pool: a family has no rows")
    draws_total = steps * batch
    draws = {f: d_target * decision["shares"][f] / pool[f]["mean_decision"] for f in fams}
    used = sum(draws.values())
    if used >= draws_total:
        raise ExposureUnmatched(f"the decision target needs {used:.0f} of {draws_total} draws, leaving none for replay")
    replay_draws = draws_total - used
    sup_fam = sum(draws[f] * pool[f]["mean_supervised"] for f in fams)
    mean_replay = (s_target - sup_fam) / replay_draws
    rp = pool["_replay"]
    if not rp["min_mean_supervised"] <= mean_replay <= rp["max_mean_supervised"]:
        raise ExposureUnmatched(f"replay rows would need a mean of {mean_replay:.1f} tokens, outside {rp['min_mean_supervised']} to {rp['max_mean_supervised']}")
    replay_tokens = replay_draws * mean_replay
    share = replay_tokens / s_target
    if share < decision["replay_share"] - 1e-9:
        raise ExposureUnmatched(f"replay would hold {share:.3f} of supervised tokens, under its floor of {decision['replay_share']}")
    mix = {f: d / draws_total for f, d in draws.items()}
    mix["_replay"] = replay_draws / draws_total
    room = min(pool[f]["rows"] / mix[f] for f in mix if mix[f] > 0)
    if room < rows_wanted:
        raise ExposureUnmatched(f"the pool holds {room:.0f} rows at this mix, {rows_wanted} wanted: more repetition would be a hidden change")
    rows = largest_remainder(mix, rows_wanted)
    out = {}
    per = draws_total / rows_wanted
    for f in sorted(rows):
        n = rows[f]
        m_sup = mean_replay if f == "_replay" else pool[f]["mean_supervised"]
        m_dec = 0.0 if f == "_replay" else pool[f]["mean_decision"]
        out[f] = {"rows": n, "expected_draws": rounded(n * per, 1), "expected_supervised": rounded(n * per * m_sup, 1), "expected_decision": rounded(n * per * m_dec, 1),
                  "expected_distinct_rows": rounded(n * (1 - (1 - 1 / rows_wanted) ** draws_total), 1), "pool_rows": pool[f]["rows"]}
    return {"schema": "lab-leg-plan-v1", "experiment_id": decision["experiment_id"], "arm": decision["arm"], "leg": decision["next_leg"], "decision_sha": decision["decision_sha"], "row_seed": decision["row_seed"], "steps": steps, "batch": batch,
            "packet_training_rows": rows_wanted, "packet_tail_rows": max(1, rows_wanted // 19), "replay_mean_supervised": rounded(mean_replay, 2), "replay_token_share": rounded(share),
            "expected_supervised_total": rounded(sum(v["expected_supervised"] for v in out.values()), 1), "expected_decision_total": rounded(sum(v["expected_decision"] for v in out.values()), 1), "families": out, "executed": False}


def prelaunch_recount(plan, counts, family_tol=0.05, total_tol=0.005):
    """counts: family_exposure.leg_counts on the built packet at the leg's seed and steps. Draws are deterministic, so a miss is fixed in the packet before launch, not found after."""
    problems = []
    got = sum(v["sampled_supervised"] for v in counts.values())
    want = plan["expected_supervised_total"]
    if abs(got / want - 1) > total_tol:
        problems.append(f"supervised total {got:,.0f}, plan {want:,.0f}")
    for f, v in plan["families"].items():
        if f == "_replay" or v["expected_decision"] == 0:
            continue
        have = counts.get(f, {}).get("sampled_decision", 0)
        if abs(have / v["expected_decision"] - 1) > family_tol:
            problems.append(f"{f}: decision tokens {have:,.0f}, plan {v['expected_decision']:,.0f}")
    return problems


RECIPE_FIELDS = ("steps", "source_commit", "generator_version", "seed", "recipe")


def _sum(legs, key):
    return sum(v[key] for b in legs for v in b["exposure"].values())


def audit_pair(fixed_legs, adaptive_legs, start_checkpoint, supervised_tol=0.005, decision_tol=0.02, lr_tol=0.02, scenario_tol=0.10, scenario_void=0.25):
    """The matched-budget check on one seed pair's realised legs, leg by leg and in total. problems void the comparison; warnings print beside it."""
    problems, warnings = [], []
    fx, ad = sorted(fixed_legs, key=lambda b: b["leg"]), sorted(adaptive_legs, key=lambda b: b["leg"])
    if [b["leg"] for b in fx] != [b["leg"] for b in ad] or [b["leg"] for b in fx] != list(range(len(fx))):
        problems.append("legs differ or are not contiguous")
        return {"ok": False, "problems": problems, "warnings": warnings}
    if not fx:
        return {"ok": False, "problems": ["no legs"], "warnings": warnings}
    for name, legs in (("fixed", fx), ("adaptive", ad)):
        if legs[0]["checkpoint_in"] != start_checkpoint:
            problems.append(f"{name}: leg 0 does not start from the registered checkpoint")
    for a, b in zip(fx, ad):
        if a["status"] != "ok" or b["status"] != "ok":
            problems.append(f"leg {a['leg']}: failed leg")
            continue
        for key in RECIPE_FIELDS:
            if a[key] != b[key]:
                problems.append(f"leg {a['leg']}: {key} differs")
        for key, tol in (("sampled_supervised", supervised_tol), ("sampled_decision", decision_tol)):
            x, y = _sum([a], key), _sum([b], key)
            if abs(x - y) / max(1.0, (x + y) / 2) > tol:
                problems.append(f"leg {a['leg']}: {key} fixed {x:,.0f}, adaptive {y:,.0f}")
    if problems:
        return {"ok": False, "problems": problems, "warnings": warnings}

    def rel(key):
        x, y = _sum(fx, key), _sum(ad, key)
        return abs(x - y) / max(1.0, (x + y) / 2), x, y
    r, x, y = rel("sampled_decision_lr")
    if r > lr_tol:
        warnings.append(f"sampled_decision_lr: fixed {x:,.0f}, adaptive {y:,.0f}, {r:.2%} apart")
    r, x, y = rel("distinct_scenarios")
    if r > scenario_void:
        problems.append(f"distinct_scenarios: fixed {x:,.0f}, adaptive {y:,.0f}, {r:.2%} apart: repetition is a second changed facet")
    elif r > scenario_tol:
        warnings.append(f"distinct_scenarios: fixed {x:,.0f}, adaptive {y:,.0f}, {r:.2%} apart")
    return {"ok": not problems, "problems": problems, "warnings": warnings}


def replay_floor_held(legs, replay_families, floor, tol=0.01):
    """Replay share by sampled supervised tokens, each leg. The floor is a registered minimum, not a target the controller may trade."""
    bad = []
    for b in legs:
        total = sum(v["sampled_supervised"] for v in b["exposure"].values())
        rep = sum(v["sampled_supervised"] for k, v in b["exposure"].items() if k in replay_families)
        if rep / total < floor - tol:
            bad.append({"leg": b["leg"], "replay_share": round(rep / total, 4)})
    return bad


def minutes_from_ledger(records, experiment_id, arm):
    """Everything an arm cost, from receipts: training, its reads, CPU. decision_overhead is what a plain fixed run would not pay."""
    legs = [r["body"] for r in records if r["kind"] == "leg" and r["body"]["experiment_id"] == experiment_id and r["body"]["arm"] == arm]
    reads = [r["body"] for r in records if r["kind"] == "dev_read" and r["body"]["experiment_id"] == experiment_id and r["body"]["arm"] == arm]
    train = sum(b["compute"]["l4_minutes"] + b["compute"]["cpu_minutes"] for b in legs)
    over = sum(b["compute"]["l4_minutes"] + b["compute"]["cpu_minutes"] for b in reads)
    return {"training_minutes": rounded(train, 2), "read_minutes": rounded(over, 2), "total_minutes": rounded(train + over, 2), "dev_reads": len(reads)}
