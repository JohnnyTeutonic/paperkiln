"""The error miner's analysis half. Claude Opus 5.5, 7 Nov 2026.

mine_round.py generates, with the frozen model on CPU; mine_score.py labels. Neither is rewritten: this turns their journals into typed events and one table of cell weights.
The scorer routes dose and teaches nothing (correction_loop_pilot.md, section 3): its only channel to training is a table of weights clipped to 0.25 to 3 times the verified arm's share. No mined item, wording, value
or reply enters any arm. Clean histories and model-induced histories stay apart: own-history chains are a separate later experiment, and this table refuses them. Mining reads T_M and nothing else.
"""
from __future__ import annotations

from controller import capped_simplex
from firewall import FirewallBreach, check_row_parts
from schema import Malformed, rounded
from taxonomy import FailureEvent, from_mine_label

MINE_PART = "T_M"
CLIP = (0.25, 3.0)


def label_clean(journal, rows, score_fn, round_no=1, target_verified=True, tier="heuristic"):
    """The pilot's door. journal: mine_round's gold rows, reply text in hand. rows: the row file, indexed by journal row. score_fn: mine_score.score, imported. Each mined row's own provenance must say T_M. One round only: more needs its own registration.
    Returns (events, tallies by cell). Text goes no further than this call."""
    if round_no != 1:
        raise FirewallBreach("one corrective round only: a further round needs its own registered criteria")
    check_row_parts([rows[g["row"]] for g in journal], {MINE_PART}, lambda r: r["provenance"].get("correction_part"))
    return _tally(journal, lambda r: rows[r]["provenance"]["turns"], score_fn, MINE_PART, tier, target_verified)


def rerun_stored_round(journal, metas_of, score_fn):
    """The stored S34 round's rows predate the split, so this reruns it to check the wrap against its report, and returns tallies only. No pilot path calls it."""
    return _tally(journal, metas_of, score_fn, "a18_pool_stored_round", "heuristic", True)[1]


def _tally(journal, metas_of, score_fn, split, tier, target_verified):
    events, cells = [], {}
    for g in journal:
        metas = metas_of(g["row"])
        ok, label = score_fn(g["reply"], g["meta"], metas, g["turn"])
        if ok is None:
            continue
        cell = f"{g['k']}|{g['c']}"
        c = cells.setdefault(cell, {"n": 0, "fail": 0, "halves": [[0, 0], [0, 0]]})
        c["n"] += 1
        half = c["halves"][g["row"] % 2]
        half[1] += 1
        if not ok:
            c["fail"] += 1
            half[0] += 1
            cat = from_mine_label(g["k"], label)
            if cat is None:
                raise Malformed(f"a failure with no category: {label} on {g['k']}")
            events.append(FailureEvent(g["row"], g["turn"], g["draw"], g["k"], cell, cat, tier, target_verified, "clean", split))
    return events, cells


def halves_agree(cell, bar=0.5):
    """Both row-parity halves on the same side of the bar. A cell whose halves disagree carries no usable rate."""
    a, b = cell["halves"]
    return bool(a[1] and b[1]) and (a[0] / a[1] >= bar) == (b[0] / b[1] >= bar)


def selection_table(cells, verified_share, history="clean"):
    """The failure-selected arm's cell shares: the verified arm's, each multiplied by its failure rate relative to the mean, clipped, renormalised. A cell whose halves disagree takes the mean. Refuses model-induced histories."""
    if history != "clean":
        raise FirewallBreach("round one selects on clean histories only")
    missing = [k for k in verified_share if k not in cells or not cells[k]["n"]]
    if missing:
        raise FirewallBreach(f"cells not mined: {missing}")
    rate = {k: cells[k]["fail"] / cells[k]["n"] for k in verified_share}
    mean = sum(rate[k] * s for k, s in verified_share.items())
    if mean <= 0:
        raise Malformed("no failures to select on")
    raw = {k: s * max(1e-3, rate[k] / mean if halves_agree(cells[k]) else 1.0) for k, s in verified_share.items()}  # capped_simplex scales raw, so a zero could never reach its share
    out = capped_simplex(verified_share, raw, CLIP[0], CLIP[1])
    return {k: rounded(v) for k, v in sorted(out.items())}


def total_variation(a, b):
    return rounded(0.5 * sum(abs(a[k] - b[k]) for k in a))
