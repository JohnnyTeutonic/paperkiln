#!/usr/bin/env python3
"""The training lab's command line. Claude Opus 5.5, 7 Nov 2026. Offline: nothing here trains, generates, spends or reaches a network. Start at README.md.

  python3 tools/lab/lab.py demo --out DIR                    a synthetic ledger, then a dry run of each arm, explained
  python3 tools/lab/lab.py dry-run --ledger L --config C --arm A --leg N [--pool P] [--record --producer X --date D]
  python3 tools/lab/lab.py verify --ledger L [--head SHA]    chain and schema of every record
  python3 tools/lab/lab.py audit-decisions --ledger L --config C    recompute every recorded decision from its ledger prefix
  python3 tools/lab/lab.py replay-sim [--n 200] [--world W]  synthetic worlds, fixed against adaptive, and the brief's priority formula
  python3 tools/lab/lab.py replay-history                    the observational replay on the stored snapshot
  python3 tools/lab/lab.py propose [--facts F]               rank interventions; nothing launches
  python3 tools/lab/lab.py pilots                            L4 costs and prerequisites, both pilots, both starts
  python3 tools/lab/lab.py gate-check --action A --ledger L --today D    exit 3 and the reason unless every condition holds
  python3 tools/lab/lab.py firewall-check --splits S --sealed F --used U    pass or breach, against the pinned sealed list
  python3 tools/lab/lab.py verdict --ledger L --rule R --experiment E [--record --producer X --date D --evidence E]    the frozen rule, from the ledger alone
  python3 tools/lab/lab.py pins                              frozen files against their pins
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters as AD  # noqa: E402
import controller as C  # noqa: E402
import costs as CO  # noqa: E402
import exposure as X  # noqa: E402
import firewall as FW  # noqa: E402
import gate as G  # noqa: E402
import history as H  # noqa: E402
import proposer as P  # noqa: E402
import sim as S  # noqa: E402
import synthetic as SY  # noqa: E402
from ledger import Ledger  # noqa: E402
from schema import GENESIS, sha  # noqa: E402
import verdict as V  # noqa: E402

HERE = Path(__file__).resolve().parent
EX = HERE / "examples"


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cmd_verify(a):
    recs = Ledger(a.ledger).load(expected_head=a.head)
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(json.dumps({"records": len(recs), "head": recs[-1]["sha256"] if recs else GENESIS, "kinds": kinds}, sort_keys=True))


def do_dry_run(ledger, cfg, arm, leg, pool=None):
    recs = ledger.load()
    head = recs[-1]["sha256"] if recs else GENESIS
    dec = C.decide(recs, cfg, arm, leg, head)
    lines = C.explain(dec)
    plan = None
    if pool is not None and dec["action"] == "ALLOCATE":
        plan = X.plan_leg(dec, pool, cfg)
        lines.append(f"leg plan: {plan['packet_training_rows']} training rows, {plan['packet_tail_rows']} tail rows; expected {plan['expected_decision_total']:,.0f} decision and {plan['expected_supervised_total']:,.0f} supervised tokens; replay at {plan['replay_token_share']:.3f} of supervised, rows of mean {plan['replay_mean_supervised']} tokens")
        for f, v in plan["families"].items():
            lines.append(f"  {f:28s} rows {v['rows']:6d}  expected decision {v['expected_decision']:9,.0f}  supervised {v['expected_supervised']:9,.0f}  pool {v['pool_rows']}")
    return dec, plan, lines


def cmd_dry_run(a):
    ledger = Ledger(a.ledger)
    pool = read_json(a.pool) if a.pool else None
    dec, plan, lines = do_dry_run(ledger, read_json(a.config), a.arm, a.leg, pool)
    print("\n".join(lines))
    print("NOTHING WAS TRAINED. This is a proposal.")
    if a.record:
        if not (a.producer and a.date and a.evidence):
            raise SystemExit("--record needs --producer, --date and --evidence")
        record(ledger, dec, plan, pool, a.producer, a.date, a.evidence)
        print("recorded")


def record(ledger, dec, plan, pool, producer, date, evidence):
    """A decision, and the plan and pool hash that turn it into rows: the allocation has to be reproducible, not only the weights."""
    ledger.append("decision", dec, producer, date, evidence)
    if plan is not None:
        ledger.append("leg_plan", {"experiment_id": dec["experiment_id"], "arm": dec["arm"], "leg": dec["next_leg"], "decision_sha": dec["decision_sha"], "pool_sha": sha(pool), "plan_sha": sha(plan), "plan": plan}, producer, date, evidence)


def cmd_verdict(a):
    rule = read_json(a.rule)
    ledger = Ledger(a.ledger)
    res = V.verdict_from_ledger(ledger.load(), a.experiment, rule)
    print(json.dumps(res, indent=1, sort_keys=True))
    if a.record:
        if not (a.producer and a.date and a.evidence):
            raise SystemExit("--record needs --producer, --date and --evidence")
        ledger.append("outcome", {"experiment_id": a.experiment, "verdict": res["verdict"], "rule_sha": res["rule_sha"], "detail": res}, a.producer, a.date, a.evidence)


def cmd_audit(a):
    cfg = read_json(a.config)
    recs = Ledger(a.ledger).load()
    done = 0
    for r in recs:
        if r["kind"] != "decision":
            continue
        b = r["body"]
        if b["experiment_id"] != cfg["experiment_id"]:
            continue
        again = C.decide(recs[: r["seq"]], cfg, b["arm"], b["next_leg"], r["prev"])
        if again != b:
            raise SystemExit(f"decision at seq {r['seq']} does not reproduce")
        done += 1
    print(json.dumps({"decisions_reproduced": done}))


def cmd_replay_sim(a):
    print("SYNTHETIC: every number below comes from assumed curves. It tests the code and is no evidence about any model.")
    ws = S.worlds()
    names = [a.world] if a.world else sorted(ws)
    print(f"{'world':22s} {'policy':9s} {'gain pts':>8s} {'x range':>13s}  verdicts of {a.n}, three seed pairs each")
    for w in names:
        for pol in ("adaptive", "priority"):
            r = S.compare(ws[w], a.n, pol)
            print(f"{w:22s} {pol:9s} {r['mean_gain_points']:8.2f} {r['smallest_multiplier']:6.3f}-{r['largest_multiplier']:6.3f}  {json.dumps(r['verdicts'], sort_keys=True)}")
        print(f"  {ws[w]['note']}")


def cmd_replay_history(a):
    snap = read_json(a.snapshot or EX / "history_observational_v1.json")
    print(json.dumps({"replay": H.replay(snap), "signal_check": H.signal_check(snap)}, indent=1, sort_keys=True))


def cmd_propose(a):
    entries = read_json(EX / "interventions.json")["interventions"]
    facts = read_json(a.facts or EX / "facts_2026-11-07.json")["facts"]
    for title, f in (("ready now", facts), ("once the lab's own engineering is done (T_C families, T_E rows, the VM read, the packet writer, a dev-signal read)", dict(facts, **{k: True for k in P.ENGINEERING}))):
        res = P.rank(entries, f)
        print(f"{title} (needs Jonathan's authorisation; may_launch is False on every line):")
        for r in res["ranked"]:
            print(f"  {r['group']:18s} {r['score']:5.3f}  {r['id']:32s} {r['hours']:5.2f} h  {r['information_bits']:.3f} bits  E[gain] {r['expected_gain_points']:.2f}  prior {r['prior']}")
        if not res["ranked"]:
            print("  nothing")
        print("blocked:")
        for r in res["blocked"]:
            print(f"  {r['id']:32s} unmet: {', '.join(r['unmet'])}")


def cmd_pilots(a):
    print(json.dumps({"costs": CO.table(), "prerequisites": CO.prerequisites()}, indent=1))


def cmd_gate(a):
    try:
        print(json.dumps(G.check(read_json(a.action), Ledger(a.ledger).load(), a.today)))
    except G.Denied as e:
        print(f"DENIED: {e}")
        raise SystemExit(3) from None


def cmd_firewall(a):
    out = {"splits": FW.check_splits(read_json(a.splits))}
    out["sealed"] = FW.check_against_sealed(read_json(a.used)["groups"], FW.sealed_groups(a.sealed))
    print(json.dumps(out, sort_keys=True))


def cmd_pins(a):
    print(json.dumps(AD.verify_pins(), indent=1, sort_keys=True))


def cmd_demo(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_ledger.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; ledgers are never rewritten")
    cfg, pool = SY.write_demo(path)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    (out / "pool.json").write_text(json.dumps(pool, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    print("SYNTHETIC LEDGER. No model, no real family, no training.")
    led = Ledger(path)
    for arm in ("fixed", "adaptive"):
        dec, plan, lines = do_dry_run(led, cfg, arm, SY.DEMO_NEXT_LEG, pool)
        print("\n" + "\n".join(lines))
        record(led, dec, plan, pool, SY.PRODUCER, SY.DATE, "synthetic")
    print("\nNOTHING WAS TRAINED. Both are proposals, and both are now on the synthetic ledger for audit-decisions.")
    print(f"ledger {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("verify")
    q.add_argument("--ledger", required=True)
    q.add_argument("--head")
    q.set_defaults(fn=cmd_verify)
    q = sub.add_parser("dry-run")
    for k in ("--ledger", "--config", "--arm"):
        q.add_argument(k, required=True)
    q.add_argument("--leg", type=int, required=True)
    q.add_argument("--pool")
    q.add_argument("--record", action="store_true")
    q.add_argument("--producer")
    q.add_argument("--date")
    q.add_argument("--evidence")
    q.set_defaults(fn=cmd_dry_run)
    q = sub.add_parser("audit-decisions")
    q.add_argument("--ledger", required=True)
    q.add_argument("--config", required=True)
    q.set_defaults(fn=cmd_audit)
    q = sub.add_parser("replay-sim")
    q.add_argument("--n", type=int, default=200)
    q.add_argument("--world")
    q.set_defaults(fn=cmd_replay_sim)
    q = sub.add_parser("replay-history")
    q.add_argument("--snapshot")
    q.set_defaults(fn=cmd_replay_history)
    q = sub.add_parser("propose")
    q.add_argument("--facts")
    q.set_defaults(fn=cmd_propose)
    q = sub.add_parser("pilots")
    q.set_defaults(fn=cmd_pilots)
    q = sub.add_parser("gate-check")
    for k in ("--action", "--ledger", "--today"):
        q.add_argument(k, required=True)
    q.set_defaults(fn=cmd_gate)
    q = sub.add_parser("firewall-check")
    for k in ("--splits", "--sealed", "--used"):
        q.add_argument(k, required=True)
    q.set_defaults(fn=cmd_firewall)
    q = sub.add_parser("verdict")
    for k in ("--ledger", "--rule", "--experiment"):
        q.add_argument(k, required=True)
    q.add_argument("--record", action="store_true")
    q.add_argument("--producer")
    q.add_argument("--date")
    q.add_argument("--evidence")
    q.set_defaults(fn=cmd_verdict)
    q = sub.add_parser("pins")
    q.set_defaults(fn=cmd_pins)
    q = sub.add_parser("demo")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_demo)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
