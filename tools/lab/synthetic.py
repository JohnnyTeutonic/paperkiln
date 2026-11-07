"""Synthetic ledger content for the demo and the tests. Claude Opus 5.5, 7 Nov 2026. Invented families and invented numbers. No wording, row or read from any instrument or probe."""
from __future__ import annotations

from ledger import Ledger
from schema import sha

PRODUCER = "Claude Opus 5.5, 7 Nov 2026"
DATE = "2026-11-07"
BASE = "synthetic_base"
EXP = "lab-synthetic-demo"
DEMO_NEXT_LEG = 2
FAMILIES = ("attribution_toy", "correction_toy", "recall_toy", "role_toy", "smalltalk_toy")
REPLAY = "replay_toy"
START = sha("synthetic start checkpoint")
DEV_SET = sha("toy dev set")
DEV_ID, METRIC = "toy_dev_v1", "nats_per_target"
# Family: (floor, loss after leg 0 and leg 1). correction_toy falls and stays high; recall_toy reached its floor; role_toy stalled; attribution_toy is noise around its floor.
TRACE = {"attribution_toy": (0.40, 0.46, 0.44), "correction_toy": (0.30, 2.10, 1.40), "recall_toy": (0.25, 0.31, 0.27), "role_toy": (0.35, 1.60, 1.59), "smalltalk_toy": (0.90, 0.98, 0.95), REPLAY: (0.0, 1.00, 1.00)}


def config(policy_map=None):
    fams = {f: {"balanced_share": 0.2, "floor": TRACE[f][0], "floor_se": 0.03, "floor_base_id": BASE} for f in FAMILIES}
    return {"schema": "lab-config-v1", "experiment_id": EXP, "base_id": BASE, "arm_policies": policy_map or {"fixed": "fixed", "adaptive": "adaptive"}, "allow_synthetic": True, "families": fams, "guard_families": [REPLAY], "replay_share": 0.30,
            "eta": 0.5, "z": 2.0, "excess_scale": 1.0, "min_excess": 0.15, "multiplier_min": 0.5, "multiplier_max": 2.0, "regression_nats": 0.15, "min_gain_nats": 0.16, "min_dose_tokens": 4000.0, "min_rows": 20, "patience": 1,
            "legs": 4, "seed": 11, "matched_pilot": True, "leg_l4_minutes": 40.0, "budget_l4_minutes": 400.0, "steps_per_leg": 512, "batch": 16, "packet_rows": 3000, "decision_tokens_per_leg": 90000.0, "supervised_tokens_per_leg": 650000.0}


def pool():
    return {"attribution_toy": {"rows": 900, "mean_supervised": 70.0, "mean_decision": 20.0}, "correction_toy": {"rows": 900, "mean_supervised": 120.0, "mean_decision": 30.0}, "recall_toy": {"rows": 1500, "mean_supervised": 60.0, "mean_decision": 25.0},
            "role_toy": {"rows": 900, "mean_supervised": 90.0, "mean_decision": 22.0}, "smalltalk_toy": {"rows": 1200, "mean_supervised": 45.0, "mean_decision": 15.0},
            "_replay": {"rows": 4000, "mean_supervised": 110.0, "min_mean_supervised": 50.0, "max_mean_supervised": 200.0}}


RULE = {"practical_points": 3.0, "pair_sd_points": 2.0, "min_seed_pairs": 3, "min_groups": 8, "boot_draws": 2000, "boot_seed": 1, "compute_tolerance": 0.05, "pairs": [["fixed", "adaptive"]], "replay_families": [REPLAY], "replay_floor": 0.30}


def registration(cfg=None, base=BASE, rule=None, status="registered"):
    cfg = cfg or config()
    return {"experiment_id": EXP, "base_id": base, "status": status, "hypothesis": "synthetic: exercises the schema", "hypothesis_id": "toy-h1", "changed_facet": "allocation policy", "control_arm": "fixed", "arms": cfg["arm_policies"],
            "primary": "synthetic task points", "frozen_rule_sha": sha(rule or RULE), "start_checkpoint": START, "config": cfg, "config_sha": sha(cfg), "development": {"instrument_id": DEV_ID, "set_sha": DEV_SET, "metric": METRIC},
            "budget": {"legs": 4, "steps_per_leg": 512, "batch": 16, "max_dev_reads": 8, "max_l4_minutes": 400.0}, "frozen_evaluation": ["toy_eval_v1"], "guards": ["faithfulness", "abstention", "identity", "role"], "stop_rules": ["budget", "guard"]}


def checkpoint(arm, i):
    return START if i < 0 else sha([arm, i, "out"])


def leg(arm, i, seed=1, status="ok", dose=9000.0):
    fams = {f: {"draws": dose / 10, "sampled_supervised": dose * 3, "sampled_decision": dose, "sampled_decision_lr": dose * 0.55, "distinct_scenarios": dose / 12} for f in FAMILIES}
    fams[REPLAY] = {"draws": dose, "sampled_supervised": dose * 6.5, "sampled_decision": 0.0, "sampled_decision_lr": 0.0, "distinct_scenarios": dose / 12}
    b = {"experiment_id": EXP, "base_id": BASE, "arm": arm, "leg": i, "status": status, "checkpoint_in": checkpoint(arm, i - 1), "source_commit": "abc1234", "dataset_manifest": sha([arm, i, "packet"]), "plan_sha": sha([arm, i, "plan"]), "generator_version": "synthetic-v1",
         "seed": seed, "steps": 512, "recipe": {"lr": 1e-4, "warmup": 64, "optimiser": "adamw", "batch": 16}, "compute": {"l4_minutes": 40.0, "cpu_minutes": 1.0}}
    if status == "ok":
        b.update(checkpoint_out=checkpoint(arm, i), train={"first_loss": 2.0, "last_loss": 1.5}, exposure=fams)
    else:
        b["failure_mode"] = "synthetic: pod lost"
    return b


def dev_read(arm, i, values=None, role="development", instrument=DEV_ID, se=0.04, n=24, set_sha=DEV_SET, metric=METRIC, checkpoint_sha=None):
    fams = {f: {"n": n, "mean": (values or {}).get(f, TRACE[f][min(i + 1, 2)]), "se": se} for f in FAMILIES + (REPLAY,)}
    return {"experiment_id": EXP, "base_id": BASE, "arm": arm, "leg": i, "checkpoint": checkpoint_sha or checkpoint(arm, i), "instrument_id": instrument, "instrument_role": role, "set_sha": set_sha, "metric": metric, "families": fams,
            "compute": {"l4_minutes": 1.0, "cpu_minutes": 0.0}}


def write_demo(path, legs=2, cfg=None, evidence="synthetic"):
    led = Ledger(path)
    cfg = cfg or config()
    led.append("registration", registration(cfg), PRODUCER, DATE, evidence)
    for arm in cfg["arm_policies"]:
        for i in range(legs):
            led.append("leg", leg(arm, i), PRODUCER, DATE, evidence)
            led.append("dev_read", dev_read(arm, i), PRODUCER, DATE, evidence)
    return cfg, pool()
