"""The authorisation gate for anything that spends GPU money. Claude Opus 5.5, 7 Nov 2026.

The lab holds no launcher: no code here starts a session, a pod or a job, and a test scans the folder to keep it so. This gate is what any future launcher must call, and it denies by default.
A permit needs all of these, each on the ledger: a registered experiment; a separate review record that passed, by a named reviewer who is not its producer; and a dated authorisation from Jonathan, with registered evidence, naming the exact action by hash and that
registration by hash, in date, within its caps and the registered budget, not yet consumed. `consume` is the only way to take a permit and spends it in the same locked append, so one authorisation gives one action.
There is no environment variable, flag or default grant. It is a procedural gate, not a security boundary: an agent that could write the ledger could forge a record, and only an out-of-band signature would change that. A rule binds what the code
cannot: agents never write authorisation records. Only his own dated line does, entered with its quote. `today` is passed in, so the gate reads no clock and its answer can be replayed; the caller's date is therefore trusted.
"""
from __future__ import annotations

from schema import need, nonneg, sha

GRANTOR = "Jonathan"


class Denied(Exception):
    pass


def action_sha(action):
    """The action's identity. Any change to its spec, provider, minutes or dollars makes it a different action."""
    need(action, "kind", str, "action")
    need(action, "experiment_id", str, "action")
    need(action, "spec_sha", str, "action")
    need(action, "provider", str, "action")
    if nonneg(need(action, "l4_minutes", float, "action"), "action.l4_minutes") <= 0 or nonneg(need(action, "usd", float, "action"), "action.usd") <= 0:
        raise Denied("an action with no cost is not an action this gate knows")
    return sha(action)


def check(action, records, today):
    """Returns a permit, or raises Denied naming the first unmet condition."""
    digest = action_sha(action)
    regs = [r for r in records if r["kind"] == "registration" and r["body"]["experiment_id"] == action["experiment_id"]]
    if not regs:
        raise Denied("no registration on the ledger")
    reg = regs[-1]
    if reg["evidence"] != "registered" or reg["body"]["status"] != "registered":
        raise Denied("the registration is a draft")
    reviews = [r for r in records if r["kind"] == "review" and r["body"]["experiment_id"] == action["experiment_id"] and r["body"]["registration_sha"] == reg["sha256"]]
    if not reviews or reviews[-1]["body"]["result"] != "passed":
        raise Denied("no passed review of this registration")
    if reviews[-1]["body"]["reviewer"].strip().lower() == reg["producer"].strip().lower():
        raise Denied("a registration cannot be reviewed by its producer")
    if action["l4_minutes"] > reg["body"]["budget"]["max_l4_minutes"]:
        raise Denied("the action passes the registered budget")
    grants = [r for r in records if r["kind"] == "authorisation" and r["body"]["action_sha"] == digest]
    if not grants:
        raise Denied("no authorisation names this action")
    g = grants[-1]
    b = g["body"]
    if b["granted_by"] != GRANTOR or GRANTOR not in g["producer"] or g["evidence"] != "registered":
        raise Denied(f"only {GRANTOR}'s own registered line authorises spend")
    if b["registration_sha"] != reg["sha256"]:
        raise Denied("the authorisation names another registration")
    if not b["quote"].strip():
        raise Denied("the authorisation has no quoted line")
    if not b["granted"] <= today <= b["expires"]:
        raise Denied("the authorisation is out of date")
    if action["l4_minutes"] > b["cap_l4_minutes"] or action["usd"] > b["cap_usd"]:
        raise Denied("the action passes its caps")
    if any(r["kind"] == "consumption" and r["body"]["authorisation_seq"] == g["seq"] for r in records):
        raise Denied("the authorisation is already consumed")
    return {"permit": True, "action_sha": digest, "authorisation_seq": g["seq"], "note": "a permit executes nothing"}


def consume(action, ledger, today, producer, date):
    """check and spend, as one step: a second call is denied. The lab has no caller of this; a future launcher does, and starts nothing unless it returns."""
    permit = check(action, ledger.load(), today)
    ledger.append("consumption", {"action_sha": permit["action_sha"], "authorisation_seq": permit["authorisation_seq"]}, producer, date, "registered")
    return permit
