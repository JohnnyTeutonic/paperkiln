"""The T_E scorer. Claude Opus 5.5, 7 Nov 2026, before any model was read on T_E.

A rule, keyed row by row. A reply passes if it names everything in decision.key.accept and nothing in decision.key.refuse; an abstention passes on an abstention cue and no told value; a confirmation also has to hold its polarity. A reply that copies the final user turn fails, whatever it names:
the shortcut read showed a copied confirmation question naming its own key. Task points are by twin pair: a pair counts only if both twins pass, so a constant reply earns nothing where the twins' keys differ. It shares no code with build_te.py, and none with mine_score.py, because mining and evaluation stay apart. Cue lists sit outside the repository, pinned by hash here, so this file holds no wording.
"""
from __future__ import annotations

import json
import re

from schema import Malformed, sha_file

CUES_SHA = "2b9d1576ead6b5e3edad9d020ad04ee2ad6362c231774dd9d91d9f102ba69e12"
ABSTENTION = ("abstain", "like_unknown", "user_name_unknown")


def load_cues(path, pin=CUES_SHA):
    if sha_file(path) != pin:
        raise Malformed("the scorer's cue list has changed")
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    return compile_cues(c)


def compile_cues(c):
    def rx(words):
        return re.compile(r"(?<![a-z0-9'])(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")(?![a-z0-9'])", re.I)
    return {"abstain": rx(c["abstain"]), "no": rx(c["no"])}


def _core(value):
    return re.sub(r"^(a|an|the) ", "", str(value).strip(), flags=re.I)


def mention(text, value):
    v = _core(value)
    return bool(v) and re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", text, flags=re.I) is not None


def _without(text, values):
    """A refused value inside an accepted one, as a surname inside a full name, is not a mention of the refused value."""
    for v in sorted((_core(x) for x in values), key=len, reverse=True):
        if v:
            text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", " ", text, flags=re.I)
    return text


def _folded(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def last_user(row):
    """The final user turn, from the row's own segments."""
    text = "".join(s["text"] for s in row["segments"] if not s["loss"])
    return text.rsplit("<|user|>", 1)[-1].split("<|assistant|>", 1)[0]


def passes(reply, row, cues):
    d = row["decision"]
    text = (reply or "").strip()
    accept, refuse = d["key"]["accept"], d["key"]["refuse"]
    if not text or _folded(text) == _folded(last_user(row)):
        return False
    if d["k"] in ABSTENTION:
        return cues["abstain"].search(text) is not None and not any(mention(text, v) for v in refuse)
    if not all(mention(text, v) for v in accept):
        return False
    rest = _without(text, accept)
    if any(mention(rest, v) for v in refuse):
        return False
    if d["k"] == "like_affirm":
        negated = cues["no"].search(text) is not None
        return negated if row["provenance"]["turns"][d["turn"]].get("pol") == "no" else not negated
    return True


def pair_points(pairs, reply_of, cues):
    """pairs: [(row_a, row_b)]. reply_of(row) gives the reply. 100 x the share of pairs on which both twins pass."""
    if not pairs:
        raise Malformed("no pairs")
    both = sum(1 for a, b in pairs if passes(reply_of(a), a, cues) and passes(reply_of(b), b, cues))
    return round(100.0 * both / len(pairs), 2)
