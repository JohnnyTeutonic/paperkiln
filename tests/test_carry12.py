"""Carry-12 v1 (tools/carry12_build.py) and the dialogue-length tool (tools/dialogue_lengths.py): the
frozen file rebuilds byte for byte and matches its pin, its items are well formed and balanced as the
pre-registration draft says (experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md), no family has
a length artefact (Carry-5's instruction family: the compliant answer always the shortest), and the
SFT-rule arithmetic matches tools/chat_data.py. Offline: no model, no tokeniser."""
import hashlib
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


def test_frozen_file_rebuilds_byte_identical_and_matches_its_pin():
    with open(c12.OUT, "rb") as f:
        disk = f.read()
    assert c12.dump(c12.build_items()) == disk
    assert hashlib.sha256(disk).hexdigest() == c12.pinned() ==         "de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797"
    its = items()
    assert all(it["version"] == "carry12_v1" and it["id"].startswith("carry12_v1:") and "draft" not in it
               for it in its)


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
    for fam, xs in by.items():  # the gold position is exactly balanced in every family
        gi = [it["gold_index"] for it in xs]
        assert [gi.count(p) for p in range(4)] == [len(xs) // 4] * 4, fam


def test_no_family_has_a_length_artefact():
    """The gold reply is uniquely the shortest (or longest) candidate on at most 35% of any family's items
    (chance-level heuristics); Carry-5's instruction family had the compliant answer shortest on all."""
    by = {}
    for it in items():
        by.setdefault(it["family"], []).append(c12.length_extremes(it))
    for fam, ext in by.items():
        assert sum(e[0] for e in ext) <= 0.35 * len(ext), fam
        assert sum(e[1] for e in ext) <= 0.35 * len(ext), fam


def test_instruction_family_is_length_neutral_and_rotates_its_forms():
    ins = [it for it in items() if it["family"] == "instruction"]
    forms = {}
    for it in ins:
        chk = it["check"]
        assert sum(c12.check_instruction(chk, c) for c in it["candidates"]) == 1
        assert c12.check_instruction(chk, it["candidates"][it["gold_index"]])
        content = [c12.words(c) for c in it["candidates"]]
        common = set(content[0]).intersection(*content[1:])
        assert all(len(set(w) - common) <= 2 for w in content)  # same content: only the form words differ
        forms[(chk["kind"], chk["form"])] = forms.get((chk["kind"], chk["form"]), 0) + 1
    assert sorted(forms.values()) == [9] * 8  # each of the 2 x 4 forms gold equally often


def test_value_gold_rate_is_near_chance():
    """Across the set a value is gold about a quarter of the times it is a candidate (noctx is chance by
    construction); load is small (20 items per slot), so only its mean is checked."""
    rate = c12.summarise(items())["value_gold_rate"]
    for fam in ("recall", "correction", "topic"):
        assert 0.23 <= rate[fam]["mean"] <= 0.27 and rate[fam]["max"] <= 0.5, (fam, rate[fam])
    assert 0.15 <= rate["load"]["mean"] <= 0.27


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


def test_small_talk_never_names_a_value_or_primes_a_candidate():
    values = c12.all_values()
    for it in items():
        if it["family"] == "instruction":
            ban = c12.banned_words(it["check"]["forms"] + it["answers"])
        else:
            ban = c12.banned_words(it["candidates"] + (it["answers"] if it["family"] == "knowledge" else []))
        for n, (t, k) in enumerate(zip(it["turns"], it["turn_kinds"]), 1):
            if k in ("small_talk", "digression", "opener", "closing"):
                assert not (values & set(c12.words(t["user"] + " " + t["gold"]))), (it["id"], t)
            if k in ("small_talk", "digression", "opener") and n < it["target_turn"]:
                assert not c12.clashes(t["user"] + " " + t["gold"], ban), (it["id"], t)


def test_no_two_items_of_a_family_share_a_twin_turns_or_near_rendering():
    seen = set()
    for it in items():
        for key in (("twin", it["twin"]), ("turns", tuple(t["user"] for t in it["turns"])),
                    ("near", tuple((t["user"], t["gold"]) for t in it["near"]))):
            k = (it["family"],) + key
            assert k not in seen, (it["id"], key[0])
            seen.add(k)


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
