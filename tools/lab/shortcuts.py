"""The shortcut baselines, fixed before any T_E row exists. Claude Opus 5.5, 7 Nov 2026.

Seven trivial policies, run on the frozen evaluation and published before any model is read on it. Each gives one reply per item from the item's own fields, and none sees the gold. Five are mine_score.py's POLICIES, reached through the pinned import, so their text stays
out of this repository. Two are defined here. An item is {"value", "told": [told values in order], "last_user": the final user turn, "row": the row}.

The score is task points on T_E: the share of twin pairs on which the reply passes both twins, by the frozen rule. A constant reply cannot do that where the twins' keys differ, which is why T_E is built of twins.
SHORTCUT_BAR_POINTS is frozen here: no shortcut reaches 25 points overall. On the S34 round's single items the strongest shortcut reached 21.6% of decisions, so the bar sits just above what a one-line reply earns on unpaired rows. Over the bar, the instrument is revised
before any model is read on it; a model read first would make that revision selection.
"""
from __future__ import annotations

from firewall import REQUIRED_SHORTCUTS
from schema import Malformed, sha

SHORTCUT_BAR_POINTS = 25.0
GENERIC_ACK = "Okay, got it."
FROM_MINE_SCORE = {"constant_identity": "constant_identity", "blanket_abstention": "constant_abstain", "constant_yes": "yes_always", "answer_dump": "list_all_told", "last_entity_copy": "last_told"}
LOCAL = {"generic_acknowledgement": "the constant GENERIC_ACK", "echo_question": "the final user turn, verbatim"}
DEFINITIONS = {"schema": "lab-shortcuts-v1", "bar_points": SHORTCUT_BAR_POINTS, "from_mine_score": FROM_MINE_SCORE, "local": LOCAL, "generic_ack": GENERIC_ACK, "score": "twin pairs passing both twins, 100 x share"}


def definitions_sha():
    return sha(DEFINITIONS)


def check_complete():
    have = set(FROM_MINE_SCORE) | set(LOCAL)
    if have != set(REQUIRED_SHORTCUTS):
        raise Malformed(f"shortcuts differ from the firewall's list: {sorted(have ^ set(REQUIRED_SHORTCUTS))}")
    return sorted(have)


def reply(policy, item, frozen_policies):
    """frozen_policies: mine_score.POLICIES, through adapters. The item carries no gold key, so no policy can use one."""
    if policy == "generic_acknowledgement":
        return GENERIC_ACK
    if policy == "echo_question":
        return item["last_user"]
    if policy not in FROM_MINE_SCORE:
        raise Malformed(f"unknown shortcut {policy}")
    return frozen_policies[FROM_MINE_SCORE[policy]](item.get("value"), item["told"], item.get("row"))


def read(pairs, passes, frozen_policies):
    """pairs: [(item_a, item_b)]. passes(reply_text, item) is the frozen rule. Returns {policy: task points}, ready for firewall.check_shortcut_read with SHORTCUT_BAR_POINTS."""
    if not pairs:
        raise Malformed("no pairs")
    check_complete()
    out = {}
    for p in REQUIRED_SHORTCUTS:
        both = sum(1 for a, b in pairs if passes(reply(p, a, frozen_policies), a) and passes(reply(p, b, frozen_policies), b))
        out[p] = round(100.0 * both / len(pairs), 2)
    return out
