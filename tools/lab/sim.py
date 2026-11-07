"""A synthetic world for the replay harness. Claude Opus 5.5, 7 Nov 2026.

Every number here is an assumption. This tests that the code behaves, including that it can conclude fixed is better. It says nothing about a model, and nothing from it may be cited as evidence that adaptive selection works or fails: its worlds are
built to be easy or hard for the controller on purpose, so a result in one, the brief's priority formula losing included, reflects the world as built.

It runs the real path: decide, plan_leg, realised exposure with row-length differences and small sampling noise, audit_pair, adaptive_verdict. Each arm carries a model-level seed shift on its evaluation, so a bootstrap over groups alone would be fooled.
Four legs, three reads, leg 0 balanced, as the draft pilot.

Each family has a development curve and an evaluation curve, both driven by dose. In most worlds they agree. In proxy_invalid they do not: dose lowers the development loss of families the evaluation does not reward. That is the worry from the history replay, in miniature.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import controller as C
import exposure as X
import verdict as V

DECISION_PER_LEG = 120000.0
SUPERVISED_PER_LEG = 780000.0
SEED_SD_POINTS = 3.0


@dataclass(frozen=True)
class Fam:
    start: float
    floor: float
    tau: float
    eval_start: float
    eval_floor: float
    eval_tau: float
    forget: float = 0.0     # eval loss added per 100k tokens of other families' dose past balanced
    groups: int = 10        # template groups on the evaluation
    mean_supervised: float = 90.0
    mean_decision: float = 24.0


def _uniform(n, start=3.0, floor=0.3, tau=60000.0):
    lengths = (70.0, 80.0, 90.0, 100.0, 110.0, 120.0)
    return {f"fam_{chr(97 + i)}": Fam(start, floor, tau, start, floor, tau, mean_supervised=lengths[i % 6]) for i in range(n)}


def _f(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Fam(**d)


def worlds():
    w = {}
    u = _uniform(6)
    w["uniform"] = {"fams": u, "dev_se": 0.05, "drift": 0.03, "note": "no difference among families: nothing to adapt to"}
    w["at_floor"] = {"fams": {k: _f(v, start=v.floor + 0.02, eval_start=v.floor + 0.02) for k, v in u.items()}, "dev_se": 0.05, "drift": 0.0, "note": "every family at its floor from the start: only noise could move a weight"}
    im = dict(u)
    im["fam_a"] = _f(u["fam_a"], tau=180000.0, eval_tau=180000.0, groups=20)
    im["fam_b"] = _f(u["fam_b"], tau=180000.0, eval_tau=180000.0, groups=20)
    for k in ("fam_c", "fam_d", "fam_e", "fam_f"):
        im[k] = _f(u[k], tau=25000.0, eval_tau=25000.0, groups=5)
    w["responsive_imbalance"] = {"fams": im, "dev_se": 0.05, "drift": 0.03, "note": "two slow families matter most, and development tracks evaluation"}
    hf = dict(u)
    hf["fam_a"] = _f(u["fam_a"], start=4.2, floor=4.0, eval_start=4.2, eval_floor=4.0)
    w["hard_floor"] = {"fams": hf, "dev_se": 0.05, "drift": 0.03, "note": "one family is hard and near its floor, as human rows are: raw loss would waste dose there"}
    w["noisy_read"] = {"fams": u, "dev_se": 0.35, "drift": 0.10, "note": "reads far noisier than any family difference"}
    pi = dict(u)
    for k in ("fam_a", "fam_b", "fam_c"):
        pi[k] = _f(u[k], tau=200000.0, eval_start=1.0, eval_floor=0.8, eval_tau=200000.0, groups=2)
    for k in ("fam_d", "fam_e", "fam_f"):
        pi[k] = _f(u[k], start=1.2, floor=0.9, tau=30000.0, eval_tau=90000.0, groups=20)
    w["proxy_invalid"] = {"fams": pi, "dev_se": 0.05, "drift": 0.03, "note": "the development loss rewards families the evaluation does not"}
    fg = dict(u)
    fg["fam_a"] = _f(u["fam_a"], tau=180000.0, eval_tau=180000.0)
    for k in ("fam_d", "fam_e", "fam_f"):
        fg[k] = _f(u[k], tau=40000.0, eval_tau=40000.0, forget=0.5, groups=15)
    w["interference"] = {"fams": fg, "dev_se": 0.05, "drift": 0.03, "note": "pushing one family costs others on the evaluation, and the development read does not see it"}
    return w


def base_config(fams, policy, legs=4):
    n = len(fams)
    return {"schema": "lab-config-v1", "experiment_id": "synthetic", "base_id": "synthetic_base", "arm_policies": {"a": policy}, "allow_synthetic": True,
            "families": {f: {"balanced_share": 1.0 / n, "floor": s.floor, "floor_se": 0.03, "floor_base_id": "synthetic_base"} for f, s in fams.items()},
            "replay_share": 0.30, "eta": 0.5, "z": 2.0, "excess_scale": 1.0, "min_excess": 0.15, "multiplier_min": 0.5, "multiplier_max": 2.0, "regression_nats": 0.30, "min_gain_nats": 0.16, "min_dose_tokens": 4000.0, "min_rows": 20, "patience": 2,
            "legs": legs, "seed": 7, "matched_pilot": True, "leg_l4_minutes": 20.0, "budget_l4_minutes": 400.0, "steps_per_leg": 512, "batch": 16, "packet_rows": 3000, "decision_tokens_per_leg": DECISION_PER_LEG, "supervised_tokens_per_leg": SUPERVISED_PER_LEG}


def pool_of(fams):
    p = {f: {"rows": 3000, "mean_supervised": s.mean_supervised, "mean_decision": s.mean_decision} for f, s in fams.items()}
    p["_replay"] = {"rows": 6000, "mean_supervised": 100.0, "min_mean_supervised": 40.0, "max_mean_supervised": 260.0}
    return p


def run_arm(world, policy, seed):
    """One arm through the legs. Returns its development trace, realised legs, per-family evaluation loss, the multipliers it used and its actions."""
    fams = world["fams"]
    rng = random.Random(seed)
    cfg = base_config(fams, policy)
    pool = pool_of(fams)
    dev = {f: s.start for f, s in fams.items()}
    ev = {f: s.eval_start for f, s in fams.items()}
    state = {"reads": [], "read_legs": [], "legs": [], "failed_legs": [], "spent_l4": 0.0, "multipliers": None}
    mults, actions = [], []
    head = "0" * 64
    for leg in range(cfg["legs"]):
        d = C.DECIDE[policy](state, cfg, "a", leg, head)
        actions.append(d)
        if d["action"] != "ALLOCATE":
            break
        plan = X.plan_leg(d, pool, cfg)
        mults.append(d["multipliers"])
        state["multipliers"] = d["multipliers"]
        ex = {}
        for f, s in fams.items():
            p = plan["families"][f]
            noise = 1.0 + rng.gauss(0, 0.001)
            dose = p["expected_decision"] * noise
            dev[f] = s.floor + (dev[f] - s.floor) * math.exp(-dose / s.tau) + rng.gauss(0, world["drift"])
            balanced = DECISION_PER_LEG / len(fams)
            ev[f] = s.eval_floor + (ev[f] - s.eval_floor) * math.exp(-dose / s.eval_tau) + s.forget * max(0.0, balanced - dose) / 100000.0
            ex[f] = {"draws": p["expected_draws"], "sampled_supervised": p["expected_supervised"] * noise, "sampled_decision": dose, "sampled_decision_lr": dose * 0.55, "distinct_scenarios": p["expected_distinct_rows"]}
        r = plan["families"]["_replay"]
        ex["_replay"] = {"draws": r["expected_draws"], "sampled_supervised": r["expected_supervised"], "sampled_decision": 0.0, "sampled_decision_lr": 0.0, "distinct_scenarios": r["expected_distinct_rows"]}
        state["legs"].append({"leg": leg, "status": "ok", "exposure": ex, "steps": 512, "seed": seed % 2, "checkpoint_in": "0" * 64, "source_commit": "abc1234", "generator_version": "sim", "recipe": {"lr": 1e-4}, "compute": {"l4_minutes": 20.0, "cpu_minutes": 1.0}})
        if leg < cfg["legs"] - 1:
            state["reads"].append({f: {"n": 24, "mean": dev[f] + rng.gauss(0, world["dev_se"]), "se": world["dev_se"]} for f in fams})
            state["read_legs"].append(leg)
        state["spent_l4"] += 20.0
    return {"ev": ev, "legs": state["legs"], "mults": mults, "decisions": actions, "dev": dev}


def eval_groups(world, ev, seed, items=6):
    """Points by template group on a synthetic evaluation. The group's items share its family's loss, and the whole arm carries one seed shift."""
    rng = random.Random(seed)
    shift = rng.gauss(0, SEED_SD_POINTS)
    out = {}
    for f, s in sorted(world["fams"].items()):
        p = max(0.0, min(1.0, 1.0 - ev[f] / 3.2 + shift / 100.0))
        for g in range(s.groups):
            hit = sum(rng.random() < p for _ in range(items))
            out[f"{f}:{g}"] = 100.0 * hit / items
    return out


RULE = {"practical_points": 3.0, "pair_sd_points": SEED_SD_POINTS * 2 ** 0.5, "min_seed_pairs": 3, "min_groups": 8, "boot_draws": 1000, "boot_seed": 20261107, "compute_tolerance": 0.05}


def one_comparison(world, seed, policy="adaptive", pairs=3):
    task, audits, moved, ad, tripped = [], [], False, None, False
    for p in range(pairs):
        s = seed * 1000 + p * 2
        fx = run_arm(world, "fixed", s)
        ad = run_arm(world, policy, s)
        a = eval_groups(world, ad["ev"], s + 7)
        b = eval_groups(world, fx["ev"], s + 8)
        task.append({g: a[g] - b[g] for g in a})
        audits.append(X.audit_pair(fx["legs"], ad["legs"], "0" * 64))
        moved = moved or any(abs(m - 1.0) > 1e-6 for x in ad["mults"] for m in x.values())
        tripped = tripped or (any("guard_tripped" in d["flags"] for d in ad["decisions"]) and not any("guard_tripped" in d["flags"] for d in fx["decisions"]))
    probs = [q for a in audits for q in a["problems"]]
    exposure = {"ok": not probs, "problems": probs, "warnings": [w for a in audits for w in a["warnings"]]}
    inp = {"exposure": exposure, "compute": {"fixed_minutes": 80.0 * pairs, "adaptive_minutes": 82.0 * pairs, "fixed_dev_reads": 3 * pairs, "adaptive_dev_reads": 3 * pairs}, "task_pairs": task, "guards": {"family_regression": {"holds": not tripped}}, "exercised": moved}
    return V.adaptive_verdict(RULE, inp), ad


def compare(world, n, policy="adaptive", pairs=3):
    tally = {}
    gains, top, low = [], [], []
    for i in range(n):
        res, ad = one_comparison(world, i + 1, policy, pairs)
        tally[res["verdict"]] = tally.get(res["verdict"], 0) + 1
        if res["primary"]:
            gains.append(res["primary"]["gain_points"])
        top.append(max(max(m.values()) for m in ad["mults"]))
        low.append(min(min(m.values()) for m in ad["mults"]))
    return {"verdicts": dict(sorted(tally.items())), "mean_gain_points": round(sum(gains) / max(1, len(gains)), 2), "largest_multiplier": round(max(top), 3), "smallest_multiplier": round(min(low), 3)}
