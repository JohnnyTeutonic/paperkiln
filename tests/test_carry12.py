"""Carry-12 DRAFT (tools/carry12_build.py) and the dialogue-length tool (tools/dialogue_lengths.py):
the draft file rebuilds byte for byte, its items are well formed and balanced as the draft
pre-registration says (experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md), and the
SFT-rule arithmetic matches tools/chat_data.py. Offline: no model, no tokeniser."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import carry12_build as c12  # noqa: E402
import dialogue_lengths as dl  # noqa: E402


def items():
    with open(c12.OUT, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def test_draft_rebuilds_and_is_marked_draft():
    with open(c12.OUT, "rb") as f:
        assert c12.dump(c12.build_items()) == f.read()
    its = items()
    assert all(it["draft"] is True and it["version"] == "carry12_v0_draft" for it in its)


def test_counts_and_balance():
    its = items()
    by = {}
    for it in its:
        by.setdefault(it["family"], []).append(it)
    assert {k: len(v) for k, v in by.items()} == {"recall": 480, "correction": 360, "topic": 360, "load": 120,
                                                  "instruction": 72, "knowledge": 60}
    for fam in ("recall", "correction", "topic", "load", "knowledge"):
        tt = [it["target_turn"] for it in by[fam]]
        assert {t: tt.count(t) for t in set(tt)} == {t: len(tt) // 5 for t in (10, 11, 12, 13, 14)}
    rec = by["recall"]
    for inter in (1, 2, 3):
        win = [it["distractor_window"] for it in rec if it["interference"] == inter]
        assert win.count("early") == win.count("late") == 60
    assert all(6 <= it["distance"] <= 13 for it in rec + by["topic"] + by["load"])


def test_controls():
    for it in items():
        tt = it["target_turn"]
        assert it["near"][-1]["user"] == it["turns"][tt - 1]["user"]
        kinds = it["turn_kinds"][:tt - 1]
        events = [k for k in kinds if k not in ("opener", "small_talk", "digression", "closing")]
        assert len(it["near"]) == len(events) + 1
        if it["family"] in ("recall", "correction", "load"):
            assert it["twin"].endswith(it["noctx"])
        if it["family"] == "topic":
            assert it["turn_kinds"][it["fact_turn"]] == "digression"
            assert it["turns"][it["fact_turn"]]["user"].startswith("By the way, ")


def test_small_talk_never_names_a_value():
    values = c12.all_values()
    for it in items():
        for t, k in zip(it["turns"], it["turn_kinds"]):
            if k in ("small_talk", "digression", "opener", "closing"):
                assert not (values & set(c12.words(t["user"] + " " + t["gold"]))), (it["id"], t)


def test_sft_rule_matches_chat_data():
    # chat_data.all_turns keeps whole exchanges while the prefix is within T + 1 tokens
    assert dl.kept_turns([100, 300, 513, 600], 512) == 3
    assert dl.kept_turns([514], 512) == 0
    d = [("sft", [100, 200, 300], [40, 80, 120], [57, 117, 177])]
    s = dl.summarise(d)
    assert s["sft_rule_T512"]["turns_kept"] == 2 and s["sft_rule_T1024"]["turns_kept"] == 3
    assert s["total_tokens"] == 600


def test_alternate_is_the_data_jobs():
    """dialogue_lengths copies tiny_chat_data's alternate (it must not import the running jobs' module): the
    two function bodies, docstrings aside, are the same syntax tree."""
    import ast

    def body(path):
        tree = ast.parse(open(path, encoding="utf-8").read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "alternate")
        return [ast.dump(x) for x in fn.body[1:]]  # [0] is the docstring
    assert body(os.path.join(ROOT, "tools", "colab_chat", "tiny_chat_data.py")) ==         body(os.path.join(ROOT, "tools", "dialogue_lengths.py"))
