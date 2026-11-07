"""The evaluation firewall. Claude Opus 5.5, 7 Nov 2026.

Four splits, disjoint by generator template group and by underlying scenario. The controller reads one registered development set and nothing else: an allowlist of (instrument, set hash, metric) from the registration, not a list of
bad names. Labels are self-declared, so the set hash is what counts. A sealed or validation read anywhere on the ledger stops every decision. Each read of the set spends it, and the cap comes from the registration.
The sealed list is read for its group ids and its file hash, and no other field of it is touched: analytic_rubric.py wrote it, and its text stays unread. Sealed ids enter no commit, no log and no output.
"""
from __future__ import annotations

import json

from ledger import ROLES_DEV
from schema import Malformed, need, sha_file

SPLITS = ("train", "development", "validation", "sealed")
# A second line behind the allowlist: names that may never reach the controller, matched anywhere in a normalised id.
FROZEN_INSTRUMENTS = ("chatflow", "contrast_v", "persona_v", "suite_release", "manual_v", "compose_v", "misfire", "j1", "rubric", "lambada", "bench_v", "t_e", "sealed", "t_c")
SEALED_PIN, SEALED_COUNT = "2b21ad9e034d4b3af0e88c0d2e310d318f806e59e7adb00b665bb4dd42869a45", 31   # analytic_rubric.py's sealed_templates.json: hash and count, no ids
# Known exploits. An instrument with a shortcut at or above its own bar teaches nothing, so a registered primary needs a published read of each.
REQUIRED_SHORTCUTS = ("constant_identity", "generic_acknowledgement", "blanket_abstention", "answer_dump", "last_entity_copy", "echo_question", "constant_yes")


class FirewallBreach(Exception):
    """Fails closed: the run stops, and a person reads why."""


def check_splits(manifest):
    """Pairwise disjoint on groups and on scenarios. Returns counts."""
    if need(manifest, "schema", str, "splits") != "lab-splits-v1":
        raise Malformed("splits: wrong schema")
    parts = need(manifest, "splits", dict, "splits")
    missing = [s for s in SPLITS if s not in parts]
    if missing:
        raise Malformed(f"splits: missing {missing}")
    out = {}
    for key in ("groups", "scenarios"):
        seen = {}
        for s in SPLITS:
            for x in need(parts[s], key, list, f"splits.{s}"):
                if seen.setdefault(x, s) != s:
                    raise FirewallBreach(f"a {key[:-1]} sits in both {seen[x]} and {s}")
        out[key] = {s: len(set(parts[s][key])) for s in SPLITS}
    return out


def sealed_groups(path, pin_sha=SEALED_PIN, expect=SEALED_COUNT):
    """The sealed group ids, only after the file's hash matches its pin and the count holds. Reads the one field."""
    if not pin_sha:
        raise FirewallBreach("no pin for the sealed list")
    if sha_file(path) != pin_sha:
        raise FirewallBreach("the sealed list has changed")
    with open(path, encoding="utf-8") as f:
        groups = set(json.load(f)["groups"])
    if len(groups) != expect:
        raise FirewallBreach("the sealed list does not hold its pinned count")
    return groups


def check_against_sealed(used_groups, sealed):
    """Any group the lab touches against the sealed ids. Pass or breach, with no count of hits: a count would let an arbitrary list bisect the ids."""
    if set(used_groups) & set(sealed):
        raise FirewallBreach("a sealed template group is in use")
    return {"groups_checked": len(set(used_groups)), "sealed_groups": len(set(sealed)), "overlap": 0}


def is_frozen_instrument(instrument_id):
    i = "".join(c for c in instrument_id.lower() if c.isalnum() or c == "_").replace("-", "_")
    return any(p in i for p in FROZEN_INSTRUMENTS)


def scan_all(records):
    """Every read on the ledger, whatever its experiment or arm. A non-development read in the file means a person has to look before any decision."""
    for r in records:
        if r["kind"] == "dev_read" and r["body"]["instrument_role"] not in ROLES_DEV:
            raise FirewallBreach(f"a {r['body']['instrument_role']} read is on the ledger as a dev_read, seq {r['seq']}")


def controller_view(records, experiment_id, arm, allowed):
    """The only path from the ledger to the controller. allowed is the registration's development entry. Anything else for this experiment stops the run: it does not merely drop out."""
    scan_all(records)
    reads = []
    for r in records:
        if r["kind"] != "dev_read":
            continue
        b = r["body"]
        if b["experiment_id"] != experiment_id:
            continue
        if is_frozen_instrument(b["instrument_id"]):
            raise FirewallBreach(f"{b['instrument_id']} is a frozen instrument")
        if (b["instrument_id"], b["set_sha"], b["metric"]) != (allowed["instrument_id"], allowed["set_sha"], allowed["metric"]):
            raise FirewallBreach(f"a read at leg {b['leg']} is not the registered development read")
        if b["arm"] == arm:
            reads.append(r)
    return reads


def check_looks(records, experiment_id, cap):
    """Reads of an experiment's development set, across its arms. The registration's max_dev_reads is the cap."""
    n = sum(1 for r in records if r["kind"] == "dev_read" and r["body"]["experiment_id"] == experiment_id)
    if n > cap:
        raise FirewallBreach(f"{n} development reads, cap {cap}")
    return n


def check_row_parts(rows, allowed, part_of):
    """Row provenance against the parts a use may draw on. part_of(row) gives the row's part, so synthetic rows can use the same check."""
    bad = {}
    for r in rows:
        p = part_of(r)
        if p not in allowed:
            bad[p] = bad.get(p, 0) + 1
    if bad:
        raise FirewallBreach(f"rows from forbidden parts: {bad}")
    return len(rows)


def check_shortcut_read(read, bar):
    """read: {policy: score on the instrument's primary scale}. Fails closed on a missing policy or one at or above the bar."""
    missing = [p for p in REQUIRED_SHORTCUTS if p not in read]
    if missing:
        raise FirewallBreach(f"no published read for shortcut baselines {missing}")
    high = sorted(p for p in REQUIRED_SHORTCUTS if read[p] >= bar)
    if high:
        raise FirewallBreach(f"shortcuts reach the bar: {high}")
    return {p: read[p] for p in REQUIRED_SHORTCUTS}


def prefix_stable(annotate, turns):
    """Leakage from a later question into an earlier state annotation. annotate(turns) gives one annotation per turn. The annotation of turn i must be the same
    whether the annotator sees the whole conversation or only turns up to i. Returns the first offending index, or None."""
    whole = annotate(turns)
    if len(whole) != len(turns):
        raise Malformed("annotator: one annotation per turn")
    for i in range(len(turns)):
        if annotate(turns[: i + 1])[i] != whole[i]:
            return i
    return None
