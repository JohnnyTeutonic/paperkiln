"""The experiment ledger: append-only JSONL, hash-chained, validated on write and on read. Claude Opus 5.5, 7 Nov 2026.

There is no update or delete. A later record may name an earlier one in `supersedes`; the earlier one stays, and a registered outcome can never be superseded, so a failure cannot be rewritten to fit a later hypothesis.
A torn or altered file is refused, never repaired. One registration and one outcome per experiment, one decision per arm and leg. Every append rewrites an anchor file beside the ledger, holding the head, and load refuses a ledger
that disagrees with its anchor: that catches tail truncation. The chain is unkeyed, so a full rewrite of ledger and anchor together passes: the anchor has to be copied somewhere a run cannot reach, and `load(expected_head=...)` takes that copy.

evidence: "registered" is a prospective, frozen experiment; "observational" is imported history, never a controlled comparison; "synthetic" is the simulator or a fixture.
"""
from __future__ import annotations

import json
import os

from schema import COMMIT, DATE, GENESIS, Malformed, canon, need, nonneg, number, one_of, sha, sha_field

SCHEMA = "lab-ledger-v1"
EVIDENCE = {"registered", "observational", "synthetic"}
ROLES_DEV = {"development"}
ROLES_EVAL = {"validation", "sealed"}
ROLES = ROLES_DEV | ROLES_EVAL
OUTCOMES = {"VOID", "GUARD_FAILED", "ADAPTIVE_BETTER", "SCREEN_PROMISING", "FIXED_BETTER", "NO_ADAPTIVE_BENEFIT", "UNRESOLVED", "ADAPTIVE_NOT_EXERCISED", "SELECTION_ADDS", "VERIFIED_ROWS_ONLY", "NEITHER_BEATS_ORDINARY", "NOISE_MEASURED"}


class LedgerCorrupt(Exception):
    """The file does not match its own chain or anchor. Nothing downstream may read it."""


def _exposure(ex, where):
    if not isinstance(ex, dict) or not ex:
        raise Malformed(f"{where}: no exposure")
    for fam, v in ex.items():
        for key in ("draws", "sampled_supervised", "sampled_decision", "sampled_decision_lr", "distinct_scenarios"):
            nonneg(need(v, key, float, f"{where}.{fam}"), f"{where}.{fam}.{key}")
        if v["sampled_decision"] > v["sampled_supervised"]:
            raise Malformed(f"{where}.{fam}: more decision tokens than supervised tokens")


def _compute(c, where):
    for key in ("l4_minutes", "cpu_minutes"):
        nonneg(need(c, key, float, where), f"{where}.{key}")


def _base(b, where):
    need(b, "experiment_id", str, where)
    need(b, "base_id", str, where)


def v_registration(b):
    w = "registration"
    _base(b, w)
    one_of(need(b, "status", str, w), {"draft", "registered"}, f"{w}.status")
    for key in ("hypothesis", "hypothesis_id", "changed_facet", "control_arm", "primary"):
        need(b, key, str, w)
    if "," in b["changed_facet"] or " and " in b["changed_facet"]:
        raise Malformed(f"{w}.changed_facet: one changed experimental facet per comparison")
    sha_field(b, "start_checkpoint", w)
    sha_field(b, "frozen_rule_sha", w)
    arms = need(b, "arms", dict, w)
    if b["control_arm"] not in arms or len(arms) < 2:
        raise Malformed(f"{w}.arms: needs the control arm and at least one other")
    cfg = need(b, "config", dict, w)
    if need(b, "config_sha", str, w) != sha(cfg):
        raise Malformed(f"{w}.config_sha: does not match the config it carries")
    dev = need(b, "development", dict, w)
    need(dev, "instrument_id", str, f"{w}.development")
    sha_field(dev, "set_sha", f"{w}.development")
    need(dev, "metric", str, f"{w}.development")
    bud = need(b, "budget", dict, w)
    for key in ("legs", "steps_per_leg", "batch", "max_dev_reads"):
        if need(bud, key, int, f"{w}.budget") < 0:
            raise Malformed(f"{w}.budget.{key}: negative")
    nonneg(need(bud, "max_l4_minutes", float, f"{w}.budget"), f"{w}.budget.max_l4_minutes")
    for key in ("frozen_evaluation", "guards", "stop_rules"):
        if not need(b, key, list, w):
            raise Malformed(f"{w}.{key}: empty")


def v_review(b):
    w = "review"
    need(b, "experiment_id", str, w)
    sha_field(b, "registration_sha", w)
    need(b, "reviewer", str, w)
    one_of(need(b, "result", str, w), {"passed", "failed"}, f"{w}.result")
    sha_field(b, "findings_sha", w)


def v_leg_plan(b):
    w = "leg_plan"
    need(b, "experiment_id", str, w)
    need(b, "arm", str, w)
    need(b, "leg", int, w)
    sha_field(b, "decision_sha", w)
    sha_field(b, "pool_sha", w)
    sha_field(b, "plan_sha", w)
    need(b, "plan", dict, w)
    if b["plan_sha"] != sha(b["plan"]):
        raise Malformed(f"{w}.plan_sha: does not match the plan it carries")


def v_leg(b):
    w = "leg"
    _base(b, w)
    need(b, "arm", str, w)
    leg = need(b, "leg", int, w)
    if leg < 0:
        raise Malformed(f"{w}.leg: negative")
    status = one_of(need(b, "status", str, w), {"ok", "failed"}, f"{w}.status")
    sha_field(b, "checkpoint_in", w)
    if status == "ok":
        sha_field(b, "checkpoint_out", w)
        _exposure(need(b, "exposure", dict, w), f"{w}.exposure")
        train = need(b, "train", dict, w)
        for key in ("first_loss", "last_loss"):
            need(train, key, float, f"{w}.train")
    else:
        need(b, "failure_mode", str, w)
    if not COMMIT.match(need(b, "source_commit", str, w)):
        raise Malformed(f"{w}.source_commit: not a commit")
    sha_field(b, "dataset_manifest", w)
    sha_field(b, "plan_sha", w)
    need(b, "generator_version", str, w)
    need(b, "seed", int, w)
    if need(b, "steps", int, w) <= 0:
        raise Malformed(f"{w}.steps: not positive")
    rec = need(b, "recipe", dict, w)
    for key in ("lr", "warmup", "optimiser", "batch"):
        if key not in rec:
            raise Malformed(f"{w}.recipe: missing {key}")
    _compute(need(b, "compute", dict, w), f"{w}.compute")


def v_dev_read(b):
    w = "dev_read"
    _base(b, w)
    need(b, "arm", str, w)
    if need(b, "leg", int, w) < -1:
        raise Malformed(f"{w}.leg: -1 is the start checkpoint; nothing precedes it")
    sha_field(b, "checkpoint", w)
    need(b, "instrument_id", str, w)
    one_of(need(b, "instrument_role", str, w), ROLES, f"{w}.instrument_role")
    sha_field(b, "set_sha", w)
    need(b, "metric", str, w)
    fams = need(b, "families", dict, w)
    if not fams:
        raise Malformed(f"{w}.families: empty")
    for fam, v in fams.items():
        if need(v, "n", int, f"{w}.{fam}") <= 0:
            raise Malformed(f"{w}.{fam}.n: not positive")
        number(need(v, "mean", float, f"{w}.{fam}"), f"{w}.{fam}.mean")
        nonneg(need(v, "se", float, f"{w}.{fam}"), f"{w}.{fam}.se")
    _compute(need(b, "compute", dict, w), f"{w}.compute")


def v_evaluation(b):
    w = "evaluation"
    _base(b, w)
    need(b, "arm", str, w)
    need(b, "instrument_id", str, w)
    one_of(need(b, "instrument_role", str, w), ROLES_EVAL, f"{w}.instrument_role")
    for key in ("set_sha", "scorer_sha", "results_sha"):
        sha_field(b, key, w)
    need(b, "summary", dict, w)
    _compute(need(b, "compute", dict, w), f"{w}.compute")


def v_failure_batch(b):
    w = "failure_batch"
    need(b, "checkpoint_label", str, w)
    need(b, "split", str, w)
    one_of(need(b, "history", str, w), {"clean", "model_induced"}, f"{w}.history")
    rows = need(b, "counts", list, w)
    if not rows:
        raise Malformed(f"{w}.counts: empty")
    for r in rows:
        need(r, "cell", str, w)
        need(r, "category", str, w)
        need(r, "tier", str, w)
        need(r, "history", str, w)
        need(r, "n", int, w)


def v_decision(b):
    w = "decision"
    need(b, "experiment_id", str, w)
    need(b, "arm", str, w)
    need(b, "action", str, w)
    need(b, "next_leg", int, w)
    need(b, "reasons", list, w)
    need(b, "flags", list, w)
    need(b, "families", dict, w)
    need(b, "shares", dict, w)
    sha_field(b, "config_sha", w)
    sha_field(b, "ledger_head", w)
    sha_field(b, "decision_sha", w)
    need(b, "seed", int, w)
    if need(b, "executed", bool, w) is not False:
        raise Malformed(f"{w}.executed: the controller proposes; it executes nothing")
    if b["decision_sha"] != sha({k: v for k, v in b.items() if k != "decision_sha"}):
        raise Malformed(f"{w}.decision_sha: does not match the decision")


def v_outcome(b):
    w = "outcome"
    need(b, "experiment_id", str, w)
    one_of(need(b, "verdict", str, w), OUTCOMES, f"{w}.verdict")
    sha_field(b, "rule_sha", w)
    need(b, "detail", dict, w)


def v_observation(b):
    w = "observation"
    need(b, "source", str, w)
    sha_field(b, "source_sha", w)
    need(b, "model", str, w)
    need(b, "base_id", str, w)
    need(b, "data", dict, w)


def v_authorisation(b):
    w = "authorisation"
    sha_field(b, "action_sha", w)
    sha_field(b, "registration_sha", w)
    need(b, "granted_by", str, w)
    need(b, "quote", str, w)
    for key in ("granted", "expires"):
        if not DATE.match(need(b, key, str, w)):
            raise Malformed(f"{w}.{key}: not a date")
    nonneg(need(b, "cap_l4_minutes", float, w), f"{w}.cap_l4_minutes")
    nonneg(need(b, "cap_usd", float, w), f"{w}.cap_usd")


def v_consumption(b):
    w = "consumption"
    sha_field(b, "action_sha", w)
    need(b, "authorisation_seq", int, w)


VALIDATORS = {"registration": v_registration, "review": v_review, "leg_plan": v_leg_plan, "leg": v_leg, "dev_read": v_dev_read, "evaluation": v_evaluation, "failure_batch": v_failure_batch,
              "decision": v_decision, "outcome": v_outcome, "observation": v_observation, "authorisation": v_authorisation, "consumption": v_consumption}


def producer_named(p):
    """AGENTS.md: 'Claude' is a role shared by several models. A producer names the model and the date."""
    if not any(y in p for y in ("2026", "2027", "2028")) or p.strip().lower() in ("claude", "gpt", "codex"):
        raise Malformed(f"producer {p!r}: name the model and the date")
    return p


def validate(rec):
    w = "record"
    if need(rec, "schema", str, w) != SCHEMA:
        raise Malformed("record: wrong schema")
    if need(rec, "seq", int, w) < 0:
        raise Malformed("record: negative seq")
    prev = need(rec, "prev", str, w)
    if prev != GENESIS:
        sha_field(rec, "prev", w)
    kind = one_of(need(rec, "kind", str, w), set(VALIDATORS), "record.kind")
    producer_named(need(rec, "producer", str, w))
    if not DATE.match(need(rec, "date", str, w)):
        raise Malformed("record.date: use YYYY-MM-DD")
    one_of(need(rec, "evidence", str, w), EVIDENCE, "record.evidence")
    if rec.get("supersedes") is not None and kind == "outcome":
        raise Malformed("record: a registered outcome is never superseded")
    VALIDATORS[kind](need(rec, "body", dict, w))
    return rec


def _unique(recs, rec):
    """Rules that span records: what the chain alone cannot say."""
    k, b = rec["kind"], rec["body"]
    same = [r for r in recs if r["kind"] == k]
    if k == "outcome" and any(r["body"]["experiment_id"] == b["experiment_id"] for r in same):
        raise Malformed(f"outcome: {b['experiment_id']} already has one")
    if k == "registration" and b["status"] == "registered" and any(r["body"]["experiment_id"] == b["experiment_id"] and r["body"]["status"] == "registered" for r in same):
        raise Malformed(f"registration: {b['experiment_id']} is already registered")
    if k == "decision" and any((r["body"]["experiment_id"], r["body"]["arm"], r["body"]["next_leg"]) == (b["experiment_id"], b["arm"], b["next_leg"]) for r in same):
        raise Malformed(f"decision: {b['arm']} already has a decision for leg {b['next_leg']}")
    if rec.get("supersedes") is not None:
        s = rec["supersedes"]
        if not 0 <= s < len(recs):
            raise Malformed("record: supersedes a record that does not exist")
        if recs[s]["kind"] == "outcome":
            raise Malformed("record: a registered outcome is never superseded")
        if recs[s]["kind"] != k:
            raise Malformed("record: supersedes a record of another kind")


def superseded(recs):
    """Sequence numbers replaced by a later record. They stay in the file; readers skip them."""
    return {r["supersedes"] for r in recs if r.get("supersedes") is not None}


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        self.anchor = self.path + ".head"

    def load(self, expected_head=None):
        out, prev = [], GENESIS
        if not os.path.exists(self.path):
            if expected_head not in (None, GENESIS):
                raise LedgerCorrupt("the ledger is absent, and a head was expected")
            return out
        with open(self.path, encoding="utf-8", newline="\n") as f:
            text = f.read()
        if text and not text.endswith("\n"):
            raise LedgerCorrupt("the last line is torn")
        for i, line in enumerate(text.split("\n")[:-1]):
            try:
                rec = validate(json.loads(line))
            except (ValueError, Malformed) as e:
                raise LedgerCorrupt(f"line {i}: {e}") from e
            body = {k: v for k, v in rec.items() if k != "sha256"}
            if rec.get("sha256") != sha(body):
                raise LedgerCorrupt(f"line {i}: hash mismatch")
            if rec["seq"] != i or rec["prev"] != prev:
                raise LedgerCorrupt(f"line {i}: chain broken")
            if rec.get("supersedes") is not None and not 0 <= rec["supersedes"] < i:
                raise LedgerCorrupt(f"line {i}: supersedes a record that does not precede it")
            out.append(rec)
            prev = rec["sha256"]
        if out:
            if not os.path.exists(self.anchor):
                raise LedgerCorrupt("the ledger has no anchor")
            with open(self.anchor, encoding="utf-8") as f:
                if f.read().strip() != prev:
                    raise LedgerCorrupt("the ledger disagrees with its anchor: a record is missing, or the file was replaced")
        if expected_head is not None and prev != expected_head:
            raise LedgerCorrupt("the head differs from the recorded one")
        return out

    def head(self):
        recs = self.load()
        return recs[-1]["sha256"] if recs else GENESIS

    def append(self, kind, body, producer, date, evidence, supersedes=None):
        lock = self.path + ".lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LedgerCorrupt(f"{lock} exists: another writer, or a crashed one. A person reads this before it is removed") from None
        try:
            recs = self.load()
            rec = {"schema": SCHEMA, "seq": len(recs), "prev": recs[-1]["sha256"] if recs else GENESIS, "kind": kind, "producer": producer, "date": date,
                   "evidence": evidence, "supersedes": supersedes, "body": body}
            validate(rec)
            _unique(recs, rec)
            rec["sha256"] = sha(rec)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(canon(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            tmp = self.anchor + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(rec["sha256"] + "\n")
            os.replace(tmp, self.anchor)
            return rec
        finally:
            os.close(fd)
            os.unlink(lock)
