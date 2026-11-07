"""The curriculum controller: design B of analysis/multi_objective_scoping.md, section 4, made reproducible. Claude Opus 5.5, 7 Nov 2026.

A capped sampling heuristic between bounded SFT legs. Not Group-DRO, not a learner, not a reinforcement agent. It proposes an allocation and executes nothing.

ON RECORD (design B). Weights from a separate development instrument. Excess is family loss less a specialist floor, so a family cannot win by being hard. Capped at 0.5 to 2 times balanced. Replay stays at its share.

CHANGED OR ADDED, each for a named reason:
- Excess is scaled by a fixed `excess_scale`, not by mean excess. Dividing by the mean makes the push scale-free: 0.001 nats alone would reach the cap (review finding 9). 1 nat keeps design B's usual size, where mean excess is near 1.
- Conservative excess: excess less z standard errors, and nothing under `min_excess`, 0.15 nats, design B's bar. Noise earns no weight. z is frozen at 2 or more on a registered run.
- Marginal gain only withholds. A family with real excess whose gain's upper bound stayed under min_gain_nats for `patience` legs at adequate dose is stalled, held at the minimum share, and stays stalled.
  min_gain_nats must exceed z x the standard error of a difference, or no stall can be read, and the explanation says so per family. A measured gain never raises a weight: one leg's gain is a difference of two
  noisy reads, and a bad read at t-1 shows as a large gain at t. So the brief's suggested priority, gain x confidence / cost, chases regression to the mean; `priority_decide` keeps it for comparison in the replay harness and nothing else.
- Guards are non-compensatory, and cover replay families too. A family rising past the limit beyond noise returns the arm to balanced for the rest of the run. In a matched pilot the arm keeps going, so the comparison stands and the trip counts against it.
- A decision needs a registration on the ledger whose config hash matches, the registered development read, one contiguous checkpoint chain from the registered start, and specialist floors with provenance. Anything else fails closed.

All decisions are pure functions of ledger state, configuration and seed: lab.py audit-decisions recomputes each one.
"""
from __future__ import annotations

import math

from firewall import check_looks, controller_view
from ledger import superseded
from schema import SHA, Malformed, need, nonneg, rounded, sha

ACTIONS = ("ALLOCATE", "STOP_LEGS", "STOP_BUDGET", "STOP_SATURATED", "REVIEW_GUARD", "REVIEW_INPUT")
POLICIES = ("fixed", "adaptive", "priority")


def _floor_provenance(name, f):
    src = need(f, "floor_source", dict, f"config.{name}")
    if need(src, "kind", str, f"config.{name}.floor_source") != "specialist":
        raise Malformed(f"config.{name}: a floor must come from a model trained on that family. A floor from models that never saw it reads an undosed family as near its floor: the history replay does exactly that")
    for key in ("checkpoint", "set_sha"):
        if not SHA.match(need(src, key, str, f"config.{name}.floor_source")):
            raise Malformed(f"config.{name}.floor_source.{key}: not a sha256")
    need(src, "metric", str, f"config.{name}.floor_source")


def load_config(cfg):
    w = "config"
    if need(cfg, "schema", str, w) != "lab-config-v1":
        raise Malformed("config: wrong schema")
    need(cfg, "experiment_id", str, w)
    need(cfg, "base_id", str, w)
    fams = need(cfg, "families", dict, w)
    if len(fams) < 2:
        raise Malformed("config: at least two eligible families")
    synthetic = bool(cfg.get("allow_synthetic"))
    total = 0.0
    for name, f in fams.items():
        total += nonneg(need(f, "balanced_share", float, f"{w}.{name}"), f"{w}.{name}.balanced_share")
        nonneg(need(f, "floor", float, f"{w}.{name}"), f"{w}.{name}.floor")
        nonneg(need(f, "floor_se", float, f"{w}.{name}"), f"{w}.{name}.floor_se")
        need(f, "floor_base_id", str, f"{w}.{name}")
        if f["floor_base_id"] != cfg["base_id"]:
            raise Malformed(f"config.{name}: a floor from another base: cross-base floors confound base quality with curriculum")
        if not synthetic:
            _floor_provenance(name, f)
    if abs(total - 1.0) > 1e-9:
        raise Malformed(f"config: balanced shares sum to {total}")
    if not 0.10 <= need(cfg, "replay_share", float, w) < 1.0:
        raise Malformed("config.replay_share: a replay minimum of at least 0.10 is required")
    for key in ("eta", "z", "excess_scale", "min_excess", "multiplier_min", "multiplier_max", "regression_nats", "min_gain_nats", "min_dose_tokens"):
        if nonneg(need(cfg, key, float, w), f"{w}.{key}") == 0:
            raise Malformed(f"config.{key}: must be positive")
    if not synthetic and cfg["z"] < 2.0:
        raise Malformed("config.z: at least 2 on a registered run")
    if not 0.25 <= cfg["multiplier_min"] <= 1 <= cfg["multiplier_max"] <= 4.0:
        raise Malformed("config: multipliers must bracket 1, within 0.25 and 4")
    for key in ("min_rows", "patience", "legs", "seed"):
        need(cfg, key, int, w)
    if cfg["patience"] < 1 or cfg["min_rows"] < 1:
        raise Malformed("config: patience and min_rows are at least 1")
    need(cfg, "matched_pilot", bool, w)
    nonneg(need(cfg, "leg_l4_minutes", float, w), f"{w}.leg_l4_minutes")
    nonneg(need(cfg, "budget_l4_minutes", float, w), f"{w}.budget_l4_minutes")
    arms = need(cfg, "arm_policies", dict, w)
    if not arms or any(v not in POLICIES for v in arms.values()):
        raise Malformed("config.arm_policies: unknown policy")
    if "priority" in arms.values() and not synthetic:
        raise Malformed("config: the priority policy runs in the replay harness only")
    if len(arms) >= 2 and not cfg["matched_pilot"]:
        raise Malformed("config: two arms means a matched pilot, or one arm could stop alone")
    if not isinstance(cfg.get("family_map", {}), dict) or not isinstance(cfg.get("guard_families", []), list):
        raise Malformed("config: family_map is an object and guard_families a list")
    return cfg


def capped_simplex(balanced, raw, lo, hi, pinned=()):
    """Shares proportional to raw, each held within lo and hi times balanced, summing to one. A pinned family sits at lo exactly, and the others share the rest, if the caps allow it. Bisection on the scale, so the result is deterministic."""
    fixed = {f: lo * balanced[f] for f in pinned}
    free = [f for f in raw if f not in fixed]
    target = 1.0 - sum(fixed.values())
    if not free or sum(hi * balanced[f] for f in free) < target:
        fixed, free, target = {}, list(raw), 1.0

    def total(t):
        return sum(min(max(t * raw[f], lo * balanced[f]), hi * balanced[f]) for f in free)
    a, b = 0.0, 1.0
    while total(b) < target and b < 1e12:
        b *= 2.0
    for _ in range(200):
        m = (a + b) / 2.0
        if total(m) < target:
            a = m
        else:
            b = m
    t = (a + b) / 2.0
    s = {f: min(max(t * raw[f], lo * balanced[f]), hi * balanced[f]) for f in free}
    s.update(fixed)
    n = sum(s.values())
    return {f: v / n for f, v in s.items()}


def registration_of(records, experiment_id):
    regs = [r for r in records if r["kind"] == "registration" and r["body"]["experiment_id"] == experiment_id and r["body"]["status"] == "registered"]
    if not regs:
        raise Malformed(f"controller: {experiment_id} has no registration on the ledger")
    return regs[-1]


def _check_against_registration(reg, cfg, arm):
    b = reg["body"]
    if reg["evidence"] == "observational":
        raise Malformed("controller: observational history is for the replay harness, and drives no decision")
    if reg["evidence"] == "registered" and cfg.get("allow_synthetic"):
        raise Malformed("controller: allow_synthetic on a registered experiment")
    if b["config_sha"] != sha(cfg):
        raise Malformed("controller: this config is not the registered one")
    if cfg["arm_policies"] != b["arms"]:
        raise Malformed("controller: arm policies differ from the registration")
    if b["arms"][b["control_arm"]] != "fixed":
        raise Malformed("controller: the control arm must be the fixed policy")
    if arm not in b["arms"]:
        raise Malformed(f"controller: {arm} is not a registered arm")
    if b["base_id"] != cfg["base_id"]:
        raise Malformed("controller: the registration names another base")


def state_from_ledger(records, cfg, arm, leg):
    """What a decision may see: this arm's registered reads, its contiguous realised legs, and the decision for the previous leg."""
    reg = registration_of(records, cfg["experiment_id"])
    _check_against_registration(reg, cfg, arm)
    b = reg["body"]
    check_looks(records, cfg["experiment_id"], b["budget"]["max_dev_reads"])
    gone = superseded(records)
    reads = [r for r in controller_view(records, cfg["experiment_id"], arm, b["development"]) if r["seq"] not in gone]
    legs_all = [r for r in records if r["kind"] == "leg" and r["seq"] not in gone and r["body"]["experiment_id"] == cfg["experiment_id"]]
    legs = sorted((r["body"] for r in legs_all if r["body"]["arm"] == arm), key=lambda x: x["leg"])
    for r in reads:
        if r["evidence"] != reg["evidence"]:
            raise Malformed("controller: a read's evidence differs from the registration's")
    for x in legs_all:
        if x["body"]["base_id"] != cfg["base_id"]:
            raise Malformed("controller: a leg from another base")
    ok = [x for x in legs if x["status"] == "ok"]
    prev_out = b["start_checkpoint"]
    for i, x in enumerate(ok):
        if x["leg"] != i or x["checkpoint_in"] != prev_out:
            raise Malformed(f"controller: leg {x['leg']} does not continue the checkpoint chain from the registered start")
        prev_out = x["checkpoint_out"]
    out_of = {x["leg"]: x["checkpoint_out"] for x in ok}
    out_of[-1] = b["start_checkpoint"]
    by_leg = {}
    for r in reads:
        k = r["body"]
        if k["base_id"] != cfg["base_id"]:
            raise Malformed("controller: a read from another base")
        if k["leg"] in by_leg:
            raise Malformed(f"controller: two development reads of leg {k['leg']}")
        if k["leg"] not in out_of or k["checkpoint"] != out_of[k["leg"]]:
            raise Malformed(f"controller: the read of leg {k['leg']} is not of that leg's checkpoint")
        by_leg[k["leg"]] = k["families"]
    prior = [r["body"] for r in records if r["kind"] == "decision" and r["seq"] not in gone and r["body"]["experiment_id"] == cfg["experiment_id"] and r["body"]["arm"] == arm and r["body"]["next_leg"] == leg - 1]
    spent = sum(x["body"]["compute"]["l4_minutes"] for x in legs_all) + sum(r["body"]["compute"]["l4_minutes"] for r in records if r["kind"] == "dev_read" and r["body"]["experiment_id"] == cfg["experiment_id"])
    order = sorted(by_leg)
    return {"reads": [by_leg[k] for k in order], "read_legs": order, "legs": ok, "failed_legs": [x for x in legs if x["status"] == "failed"], "spent_l4": spent, "multipliers": prior[-1]["multipliers"] if prior else None}


def family_dose(leg_record, family, cfg):
    """Sampled dose of one controller family in one leg, through the frozen family map. A packet family the map does not cover fails closed: a silent zero would disable the stall."""
    fmap = cfg.get("family_map") or {}
    rep = set(cfg.get("guard_families", []))
    field = cfg["families"][family].get("dose_field", "sampled_decision")
    total = 0.0
    for k, v in leg_record["exposure"].items():
        name = fmap.get(k, k)
        if name not in cfg["families"] and name not in rep and name != "_replay":
            raise Malformed(f"controller: packet family {k} is not mapped to a controller family")
        if name == family:
            total += v[field]
    return total


def per_family(state, cfg):
    """Excess, conservative excess, marginal gain and stall state. Numbers shown in every explanation; only the conservative excess and the stall reach a weight."""
    out = {}
    z = cfg["z"]
    latest = state["reads"][-1]
    dose = {x["leg"]: x for x in state["legs"]}
    for f, spec in cfg["families"].items():
        v = latest[f]
        se = math.sqrt(v["se"] ** 2 + spec["floor_se"] ** 2)
        excess = v["mean"] - spec["floor"]
        cons = excess - z * se
        cons = cons if cons >= cfg["min_excess"] else 0.0
        gains = []
        run = best_run = 0
        readable = True
        pairs = list(zip(state["read_legs"], state["reads"]))
        for (_, a), (lb, b) in zip(pairs, pairs[1:]):
            g, d_se = a[f]["mean"] - b[f]["mean"], math.sqrt(a[f]["se"] ** 2 + b[f]["se"] ** 2)
            gains.append((g, d_se))
            readable = readable and z * d_se < cfg["min_gain_nats"]
            d = family_dose(dose[lb], f, cfg) if lb in dose else 0.0
            run = run + 1 if (g + z * d_se < cfg["min_gain_nats"] and d >= cfg["min_dose_tokens"]) else 0
            best_run = max(best_run, run)
        out[f] = {"loss": v["mean"], "se": v["se"], "n": v["n"], "excess": excess, "conservative_excess": cons, "last_gain": gains[-1][0] if gains else None, "last_gain_se": gains[-1][1] if gains else None,
                  "stalled": best_run >= cfg["patience"] and cons > 0, "stall_readable": readable, "insufficient": v["n"] < cfg["min_rows"]}
    return out


def guard_hits(state, cfg):
    """Families, replay included, up from their best read by more than the limit beyond the noise of both reads."""
    hits = []
    for f in sorted(set(cfg["families"]) | set(cfg.get("guard_families", []))):
        best = min(state["reads"], key=lambda r: r[f]["mean"])[f]
        cur = state["reads"][-1][f]
        if cur["mean"] - best["mean"] - cfg["z"] * math.sqrt(cur["se"] ** 2 + best["se"] ** 2) > cfg["regression_nats"]:
            hits.append(f)
    return hits


def _balanced(cfg):
    return {f: s["balanced_share"] for f, s in cfg["families"].items()}


def _finish(cfg, arm, leg, policy, action, multipliers, families, reasons, head, flags=()):
    bal = _balanced(cfg)
    raw = {f: bal[f] * multipliers[f] for f in bal}
    n = sum(raw.values())
    shares = {f: rounded(v / n) for f, v in sorted(raw.items())}
    rec = {"schema": "lab-decision-v1", "experiment_id": cfg["experiment_id"], "arm": arm, "next_leg": leg, "policy": policy, "action": action, "shares": shares, "replay_share": rounded(cfg["replay_share"]), "flags": sorted(flags),
           "multipliers": {f: rounded(multipliers[f]) for f in sorted(multipliers)}, "families": families, "reasons": reasons, "config_sha": sha(cfg), "ledger_head": head, "seed": cfg["seed"],
           "row_seed": int(sha([cfg["seed"], leg])[:8], 16), "executed": False}
    rec["decision_sha"] = sha(rec)
    return rec


NOTES = {"pushed": "real excess beyond noise: weight raised, within the cap", "held": "no excess beyond noise: balanced", "stalled": "excess remains, and adequate dose bought nothing: minimum share",
         "insufficient": "too few development rows to move it: balanced", "balanced": "fixed policy: balanced by construction", "priority": "comparison policy only"}


def _explain(fam, m, tag):
    return {"state": tag, "multiplier": rounded(m), "loss": rounded(fam["loss"]), "se": rounded(fam["se"]), "n": fam["n"], "excess": rounded(fam["excess"]), "conservative_excess": rounded(fam["conservative_excess"]),
            "last_gain_nats": None if fam["last_gain"] is None else rounded(fam["last_gain"]), "last_gain_se": None if fam["last_gain_se"] is None else rounded(fam["last_gain_se"]), "stall_readable": fam["stall_readable"], "note": NOTES[tag]}


def gate_stops(state, cfg, arm, leg, head, policy):
    """Stops that apply to every policy alike, so the fixed arm and the adaptive arm end on the same conditions."""
    one = {f: 1.0 for f in cfg["families"]}
    if leg >= cfg["legs"]:
        return _finish(cfg, arm, leg, policy, "STOP_LEGS", one, {}, [f"the {cfg['legs']} registered legs are done"], head)
    if state["spent_l4"] + cfg["leg_l4_minutes"] > cfg["budget_l4_minutes"]:
        return _finish(cfg, arm, leg, policy, "STOP_BUDGET", one, {}, [f"spent {state['spent_l4']:.1f} L4 minutes across the arms; one more leg passes {cfg['budget_l4_minutes']:.1f}"], head)
    if state["failed_legs"]:
        return _finish(cfg, arm, leg, policy, "REVIEW_INPUT", one, {}, [f"a failed leg is on record and nothing supersedes it: {state['failed_legs'][-1]['failure_mode']}"], head)
    if leg and leg - 1 not in state["read_legs"]:
        return _finish(cfg, arm, leg, policy, "REVIEW_INPUT", one, {}, [f"no development read of leg {leg - 1}: no decision without one"], head)
    want = set(cfg["families"]) | set(cfg.get("guard_families", []))
    for i, r in enumerate(state["reads"]):
        missing = sorted(want - set(r))
        if missing:
            return _finish(cfg, arm, leg, policy, "REVIEW_INPUT", one, {}, [f"families missing from read {i}: {missing}"], head)
    return None


def _guard_result(state, cfg, arm, leg, head, policy, expl):
    hits = guard_hits(state, cfg)
    if not hits:
        return None
    one = {f: 1.0 for f in cfg["families"]}
    reason = [f"families up more than {cfg['regression_nats']} nats beyond noise: {hits}. The arm stays balanced from here, and the trip counts against it"]
    return _finish(cfg, arm, leg, policy, "ALLOCATE" if cfg["matched_pilot"] else "REVIEW_GUARD", one, expl, reason, head, ["guard_tripped"])


def fixed_decide(state, cfg, arm, leg, head):
    stop = gate_stops(state, cfg, arm, leg, head, "fixed")
    if stop:
        return stop
    one = {f: 1.0 for f in cfg["families"]}
    text = ["balanced shares, the registered fixed curriculum"]
    if not state["reads"]:
        return _finish(cfg, arm, leg, "fixed", "ALLOCATE", one, {}, text, head)
    fams = per_family(state, cfg)
    expl = {f: _explain(fams[f], 1.0, "balanced") for f in sorted(fams)}
    return _guard_result(state, cfg, arm, leg, head, "fixed", expl) or _finish(cfg, arm, leg, "fixed", "ALLOCATE", one, expl, text, head)


def adaptive_decide(state, cfg, arm, leg, head):
    stop = gate_stops(state, cfg, arm, leg, head, "adaptive")
    if stop:
        return stop
    one = {f: 1.0 for f in cfg["families"]}
    if leg == 0 or not state["reads"]:
        return _finish(cfg, arm, leg, "adaptive", "ALLOCATE", one, {}, ["leg 0: balanced shares in both arms"], head)
    fams = per_family(state, cfg)
    guard = _guard_result(state, cfg, arm, leg, head, "adaptive", {f: _explain(fams[f], 1.0, "held") for f in sorted(fams)})
    if guard:
        return guard
    bal = _balanced(cfg)
    prev = state["multipliers"] or one
    live = [f for f, v in fams.items() if not v["stalled"] and not v["insufficient"] and v["conservative_excess"] > 0]
    if not live:
        expl = {f: _explain(fams[f], 1.0, "stalled" if fams[f]["stalled"] else "held") for f in sorted(fams)}  # at balanced by the flag below, so a stalled family's note reads against its multiplier of 1
        if cfg["matched_pilot"]:
            return _finish(cfg, arm, leg, "adaptive", "ALLOCATE", one, expl, ["no family has excess beyond noise, or all are stalled: balanced for the remaining legs, so both arms keep the same envelope"], head, ["saturated"])
        return _finish(cfg, arm, leg, "adaptive", "STOP_SATURATED", one, expl, ["no family has excess beyond noise, or all are stalled: more of this SFT buys nothing measurable"], head, ["saturated"])
    raw, tag = {}, {}
    for f, v in fams.items():
        if v["insufficient"]:
            raw[f], tag[f] = 1.0, "insufficient"
        elif v["stalled"]:
            raw[f], tag[f] = cfg["multiplier_min"], "stalled"
        elif v["conservative_excess"] > 0:
            raw[f], tag[f] = prev[f] * math.exp(cfg["eta"] * v["conservative_excess"] / cfg["excess_scale"]), "pushed"
        else:
            raw[f], tag[f] = 1.0, "held"
    shares = capped_simplex(bal, raw, cfg["multiplier_min"], cfg["multiplier_max"], pinned=[f for f in tag if tag[f] == "stalled"])
    mult = {f: shares[f] / bal[f] for f in bal}
    expl = {f: _explain(fams[f], mult[f], tag[f]) for f in sorted(fams)}
    return _finish(cfg, arm, leg, "adaptive", "ALLOCATE", mult, expl, [f"excess less {cfg['z']} standard errors, at least {cfg['min_excess']} nats, exp({cfg['eta']} x excess / {cfg['excess_scale']}), capped at {cfg['multiplier_min']} to {cfg['multiplier_max']} times balanced, replay at {cfg['replay_share']}"], head)


def priority_decide(state, cfg, arm, leg, head):
    """The brief's suggested estimate, expected marginal improvement x confidence / compute cost, with the same caps. For the replay harness; no configuration of a registered arm may name it. Reads a gain from the last two reads."""
    stop = gate_stops(state, cfg, arm, leg, head, "priority")
    if stop:
        return stop
    one = {f: 1.0 for f in cfg["families"]}
    if len(state["reads"]) < 2:
        return _finish(cfg, arm, leg, "priority", "ALLOCATE", one, {}, ["no gain read yet: balanced shares"], head)
    fams = per_family(state, cfg)
    bal = _balanced(cfg)
    pri = {}
    for f, v in fams.items():
        g, se = v["last_gain"], v["last_gain_se"]
        conf = abs(g) / (abs(g) + se) if abs(g) + se > 0 else 0.0
        pri[f] = max(0.0, g) * conf
    mean_p = sum(pri.values()) / len(pri)
    raw = {f: (pri[f] / mean_p if mean_p > 0 else 1.0) for f in pri}
    shares = capped_simplex(bal, raw, cfg["multiplier_min"], cfg["multiplier_max"])
    mult = {f: shares[f] / bal[f] for f in bal}
    return _finish(cfg, arm, leg, "priority", "ALLOCATE", mult, {f: _explain(fams[f], mult[f], "priority") for f in sorted(fams)}, ["gain x confidence, capped"], head)


DECIDE = {"fixed": fixed_decide, "adaptive": adaptive_decide, "priority": priority_decide}


def decide(records, cfg, arm, leg, head):
    load_config(cfg)
    if arm not in cfg["arm_policies"]:
        raise Malformed(f"config: no policy for arm {arm}")
    return DECIDE[cfg["arm_policies"][arm]](state_from_ledger(records, cfg, arm, leg), cfg, arm, leg, head)


def explain(dec):
    """Plain lines for the dry run: every family's state, and why."""
    out = [f"experiment {dec['experiment_id']}, arm {dec['arm']}, next leg {dec['next_leg']}, policy {dec['policy']}", f"action: {dec['action']} (a proposal: executed is {dec['executed']})"]
    out += [f"reason: {r}" for r in dec["reasons"]]
    if dec["flags"]:
        out.append(f"flags: {', '.join(dec['flags'])}")
    out.append(f"{'family':28s} {'share':>7s} {'x bal':>6s} {'loss':>6s} {'se':>6s} {'excess':>7s} {'cons.':>6s} {'gain':>7s}  state")
    for f, s in sorted(dec["shares"].items()):
        e = dec["families"].get(f)
        if e:
            g = "n/a" if e["last_gain_nats"] is None else f"{e['last_gain_nats']:+.3f}"
            note = e["note"] + ("" if e["stall_readable"] else " (a stall cannot be read at this noise)")
            out.append(f"{f:28s} {s:7.4f} {e['multiplier']:6.3f} {e['loss']:6.3f} {e['se']:6.3f} {e['excess']:7.3f} {e['conservative_excess']:6.3f} {g:>7s}  {e['state']}: {note}")
        else:
            out.append(f"{f:28s} {s:7.4f} {dec['multipliers'][f]:6.3f}")
    out.append(f"replay minimum {dec['replay_share']}, row seed {dec['row_seed']}, config {dec['config_sha'][:12]}, ledger head {dec['ledger_head'][:12]}, decision {dec['decision_sha'][:12]}")
    return out
