"""An independent read of verified rows. Claude Opus 5.5, 7 Nov 2026.

The generator is mixed_gen.py, wrapped through correction_split.py and never edited; its verify_row shares a module and an author with the state machine, so generator and verifier could agree in the same mistake.
This checker imports neither. It parses the segments' text and reads the row's turn metas, and it checks only what the text can show. Its targets still come from the metas, so a state machine wrong about which told value is current would pass:
that is a limit, and a tracker that derives the answer from text alone, written by another hand, is on the pilot's prerequisites.
  told_before_asked  a value a decision turn asks for appears in a user turn before this one
  gold_carries       the supervised reply on a value decision names that value
  no_lookahead       a fact's meta at turn i names a value already present in the user text up to turn i: no later turn leaks into an earlier annotation
  abstain_clean      a supervised abstention names no value the user told
check_rows reports how many times each check ran, so zero defects cannot mean nothing was checked. A row in a held part is refused by firewall.check_row_parts. Counts leave this module, never text.
"""
from __future__ import annotations

import collections
import re

TOK = re.compile(r"(<\|user\|>|<\|assistant\|>|<\|endoftext\|>)")
VALUE_OPS = {"recall", "user_name"}
MULTI_OPS = {"like_recall": "likes"}
BOTH_OP = "both_names"
ABSTAIN_OPS = {"abstain", "like_unknown", "user_name_unknown"}


def parse_turns(row):
    """[[user, reply]] from the row's segments. Written here from the segment format, so it shares no code with the generator."""
    turns, mode = [], None
    for s in row["segments"]:
        if s["loss"]:
            if not turns:
                continue
            turns[-1][1] += s["text"]
            mode = None
            continue
        for part in TOK.split(s["text"]):
            if part == "<|user|>":
                turns.append(["", ""])
                mode = "user"
            elif part == "<|assistant|>":
                mode = "reply"
            elif part == "<|endoftext|>":
                mode = None
            elif part and mode == "user":
                turns[-1][0] += part
            elif part and mode == "reply":
                turns[-1][1] += part
    return turns


def has(text, value):
    v = re.sub(r"^(a|an|the) ", "", str(value).strip(), flags=re.I)
    return bool(v) and re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", text, flags=re.I) is not None


def check_row(row, ran=None):
    """Counter of defects on one row, empty if it holds. ran, if given, counts each check applied."""
    bad = collections.Counter()
    ran = ran if ran is not None else collections.Counter()
    turns = parse_turns(row)
    metas = row["provenance"]["turns"]
    if len(turns) != len(metas):
        bad["turns_do_not_align"] += 1
        return bad
    seen = ""
    told = []
    for m, (user, reply) in zip(metas, turns):
        before = seen
        seen += " " + user
        k = m.get("k")
        if k == "fact" and m.get("value") is not None:
            ran["no_lookahead"] += 1
            if not has(seen, m["value"]):
                bad["no_lookahead"] += 1
            told.append(str(m["value"]))
        if k in VALUE_OPS:
            if m.get("value") is None:
                bad["decision_without_value"] += 1
                continue
            ran["told_before_asked"] += 1
            ran["gold_carries"] += 1
            if not has(before, m["value"]):
                bad["told_before_asked"] += 1
            if not has(reply, m["value"]):
                bad["gold_carries"] += 1
        if k in MULTI_OPS:
            for v in m.get(MULTI_OPS[k]) or []:
                ran["gold_carries"] += 1
                if not has(reply, v):
                    bad["gold_carries"] += 1
        if k == BOTH_OP:
            for v in (m.get("value"), m.get("other")):
                ran["gold_carries"] += 1
                if v is None or not has(reply, v):
                    bad["gold_carries"] += 1
        if k in ABSTAIN_OPS:
            ran["abstain_clean"] += 1
            if any(has(reply, v) for v in told):
                bad["abstain_clean"] += 1
    return bad


def check_rows(rows):
    total, ran = collections.Counter(), collections.Counter()
    defective = 0
    for r in rows:
        c = check_row(r, ran)
        total.update(c)
        defective += bool(c)
    return {"rows": len(rows), "defective_rows": defective, "defects": dict(sorted(total.items())), "checks_run": dict(sorted(ran.items()))}
