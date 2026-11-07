"""The failure taxonomy and typed failure events. Claude Opus 5.5, 7 Nov 2026.

A tier says how a label was reached, and tiers are never pooled:
  verified_rule  a frozen scorer, hash-pinned, against a target the state machine computed
  heuristic      a router such as mine_score.py: value-checked regex. Its 37 of 40 failure precision on S34 is one reader's, not blind
  llm_judged     a judge's opinion: a fallible annotator, never an optimisation target where a rule exists

The mining scorer routes dose and decides no verdict (correction_loop_pilot.md, section 3), so its labels are heuristic even though
the targets behind them are verified. target_verified and the tier are separate fields for that reason.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from schema import Malformed, one_of

CATEGORIES = {
    "wrong_operation": "the reply did a different conversational operation from the one asked for",
    "missing_answer": "the requested answer is absent or partial",
    "attribution": "a value was given to the wrong speaker or subject",
    "wrong_value": "a wrong factual value, or the wrong polarity on a yes-or-no about a told fact",
    "stale_value": "a superseded value after a correction",
    "unwarranted_abstention": "abstained though the answer was available",
    "failure_to_abstain": "answered though nothing supports an answer",
    "role_identity": "a role or identity failure, including a role leaking past its end",
    "echo_stock": "an echo of the turn or a stock line in place of an answer",
    "answer_dump": "named several values without selecting the one asked for",
    "format": "a response-format failure",
    "passage": "an unsupported passage answer or a citation failure",
}
TIERS = ("verified_rule", "heuristic", "llm_judged")
HISTORIES = ("clean", "model_induced")

# mine_score.py's labels by operation. VALUE is a pass; ABSTAIN is a pass on an abstention operation. None means no failure.
ABSTENTION_OPS = {"abstain", "like_unknown", "user_name_unknown"}
IDENTITY_OPS = {"self_name", "rp_sincere"}
MINE_PASS = {"VALUE"}
MINE_UNSCORED = {"NO_VALUES", "UNSCORED"}


def from_mine_label(op, label):
    """Category for one mine_score.score() label on one operation. Raises on a label this table has not seen, so a scorer change stops here."""
    if label in MINE_UNSCORED or label in MINE_PASS:
        return None
    if op in ABSTENTION_OPS:
        if label == "ABSTAIN":
            return None
        if label in ("INVENT_OR_OTHER", "ABST_WITH_VALUE"):
            return "failure_to_abstain"
    if op in IDENTITY_OPS and label == "OTHER":
        return "role_identity"
    table = {"WRONG_VALUE": "wrong_value", "LISTS_OTHERS": "answer_dump", "ABSTAIN": "unwarranted_abstention", "OTHER": "missing_answer", "PARTIAL": "missing_answer", "DENY": "wrong_value"}
    if label not in table:
        raise Malformed(f"unmapped mine label {label!r} on {op!r}")
    return table[label]


# Frozen scorers' label codes that name one failure whatever the item. Context-dependent codes (ABSTAIN, VALUE, USER_ANSWER and the like) are not here: only
# the scorer that produced them can say whether they passed. Codes only: no instrument wording.
FROZEN_LABELS = {"WRONG_VALUE": "wrong_value", "ROLE_LEAK": "role_identity", "FOREIGN_NAME": "role_identity", "HUMAN_CLAIM": "role_identity", "DENIES_AI": "role_identity",
                 "OWNER_SWAP": "attribution", "SWAP": "attribution", "LURE": "failure_to_abstain", "INVENT": "failure_to_abstain", "PARROT": "echo_stock", "RECITAL": "echo_stock",
                 "BARE_COPY": "echo_stock", "ROLE_STOCK": "echo_stock", "EMPTY": "format"}


@dataclass(frozen=True)
class FailureEvent:
    """One failed decision. Holds ids and labels, never reply text: wording stays outside the repository."""
    row: int
    turn: int
    draw: int
    op: str
    cell: str
    category: str
    tier: str
    target_verified: bool
    history: str
    split: str

    def __post_init__(self):
        one_of(self.category, set(CATEGORIES), "event.category")
        one_of(self.tier, set(TIERS), "event.tier")
        one_of(self.history, set(HISTORIES), "event.history")

    def record(self):
        return asdict(self)


def from_frozen_label(label):
    """Category for a frozen scorer's code, or None if only that scorer can say whether it passed. Tier verified_rule, once its hash is pinned."""
    return FROZEN_LABELS.get(label)


def counts(events):
    """Rows for a failure_batch record. Tier and history stay in the key, so nothing pools across them."""
    tally = {}
    for e in events:
        key = (e.cell, e.category, e.tier, e.history)
        tally[key] = tally.get(key, 0) + 1
    return [{"cell": c, "category": g, "tier": t, "history": h, "n": n} for (c, g, t, h), n in sorted(tally.items())]
