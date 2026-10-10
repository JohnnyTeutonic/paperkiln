"""Synthetic falsifiers for selection, leakage, baselines and receipt guards."""
import copy
import json
import math
from pathlib import Path

import pytest

import selection as S


def test_reversed_proxy_and_exact_decomposition():
    short, chosen = S.decision([0., 1., 2.], [3., 2., 0.], 2)
    assert short == (0, 1) and chosen == 1
    row = S.evaluate(short, chosen, [3., 2., 0.])
    assert row["hindsight_gap"] == 2.
    assert row["exclusion_gap"] == 2.
    assert row["calibration_gap"] == 0.
    # Calibration noise is a different failure from excluding the winner.
    row = S.evaluate((0, 1, 2), 0, [3., 2., 0.])
    assert row["exclusion_gap"] == 0.
    assert row["calibration_gap"] == 3.


def test_common_loss_offset_invariance():
    a = S.evaluate((0, 2), 2, [3., 2., 4.])
    b = S.evaluate((0, 2), 2, [103., 102., 104.])
    for key in a.keys() - {"loss"}:
        assert a[key] == b[key]


def small_panel():
    protocol = {"lanes": ["exact", "x", "y"], "seeds": [1, 2, 3, 4],
                "calibration_seeds": 2, "source_steps": [10], "target_steps": [20],
                "shortlist_sizes": [1, 2, 3]}
    src = {(s, lane): {10: float(i)} for s in protocol["seeds"]
           for i, lane in enumerate(protocol["lanes"])}
    tgt = {(s, lane): {20: float(2-i)} for s in protocol["seeds"]
           for i, lane in enumerate(protocol["lanes"])}
    return protocol, src, tgt


def test_known_random_expectation_and_full_shortlist_control():
    protocol, src, tgt = small_panel()
    out = S.audit(protocol, src, tgt)
    assert out["n_overlapping_splits"] == 6
    one, two, full = out["cells"]
    assert one["policy"]["hindsight_gap"]["mean"] == 2.
    assert one["random_shortlist"]["hindsight_gap"]["mean"] == 1.
    assert two["random_shortlist"]["hindsight_gap"]["mean"] == pytest.approx(1/3)
    assert full["policy"] == full["random_shortlist"] == full["direct_target"]
    assert full["policy"]["exclusion_gap"]["mean"] == 0.
    assert full["paired_contrasts"]["loss_vs_direct_target"]["mean"] == 0.


def test_heldout_changes_cannot_change_fold_selection():
    protocol, src, tgt = small_panel()
    train, test, lanes = (1, 2), (3, 4), protocol["lanes"]
    def select():
        return S.decision(S.means(src, train, lanes, 10), S.means(tgt, train, lanes, 20), 2)
    before = select()
    for seed in test:
        for name in lanes:
            tgt[seed, name][20] = -1000. if name == "exact" else 1000.
            src[seed, name][10] = -1000. if name == "y" else 1000.
    assert select() == before


def test_ties_and_negative_excess_vs_baseline():
    assert S.decision([1., 1., 1.], [2., 2., 0.], 2) == ((0, 1), 0)
    assert S.evaluate((1,), 1, [3., 2.])["excess_vs_exact"] == -1.


def receipt(tmp_path):
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    model = dict(protocol["expected_model"], event="model", d=256, lr=.00025,
                 seed=21, attention="swa", window=64, sinks=1)
    events = [{"event": "start"}, model]
    events += [{"event": "eval", "step": s, "val_loss": 3.} for s in protocol["source_steps"]]
    events += [{"event": "done", "final_step": 3600}]
    path = tmp_path / "events.jsonl"
    return protocol, events, path


def write(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def test_receipt_identity_comes_from_model(tmp_path):
    p, events, path = receipt(tmp_path)
    write(path, events)
    seed, name, values = S.read_receipt(path, p, "source")
    assert (seed, name) == (21, "swa64s1")
    assert values[3600] == 3.


@pytest.mark.parametrize("mutation", ["nan", "missing", "conflict", "scope", "done", "resume"])
def test_rejects_invalid_receipts(tmp_path, mutation):
    p, events, path = receipt(tmp_path)
    if mutation == "nan":
        events[2]["val_loss"] = math.nan
    elif mutation == "missing":
        events.pop(2)
    elif mutation == "conflict":
        events.insert(-1, {"event": "eval", "step": 400, "val_loss": 4.})
    elif mutation == "scope":
        events[1]["lr"] = .001
    elif mutation == "done":
        events.pop()
    elif mutation == "resume":
        model = copy.deepcopy(events[1])
        model["seed"] = 22
        events.insert(-1, model)
    write(path, events)
    with pytest.raises(ValueError):
        S.read_receipt(path, p, "source")


def test_rejects_unbalanced_or_duplicate_panel(tmp_path, monkeypatch):
    p, events, path = receipt(tmp_path)
    write(path, events)
    p["source"] = str(tmp_path)
    monkeypatch.setattr(S, "HERE", tmp_path)
    with pytest.raises(ValueError, match="panel differs"):
        S.load_panel(p, "source")
    another = tmp_path / "duplicate"
    another.mkdir()
    write(another / "events.jsonl", events)
    with pytest.raises(ValueError, match="duplicate"):
        S.load_panel(p, "source")
