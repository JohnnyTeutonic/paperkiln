"""Unit tests of tools/iftiny_checks.py (IF-tiny's code-checkable instructions) and of the built sets."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import iftiny_build  # noqa: E402
import iftiny_checks as C  # noqa: E402

CASES = [
    # (spec, reply, expected)
    ({"kind": "one_word"}, "Paris", True),
    ({"kind": "one_word"}, "  Paris.  ", True),
    ({"kind": "one_word"}, "It's", True),
    ({"kind": "one_word"}, "Paris, France", False),
    ({"kind": "one_word"}, "...", False),
    ({"kind": "one_word"}, "", False),
    ({"kind": "max_words", "n": 3}, "The big red", True),
    ({"kind": "max_words", "n": 3}, "The big red dog", False),
    ({"kind": "max_words", "n": 3}, "!", False),
    ({"kind": "lowercase"}, "the sky is blue.", True),
    ({"kind": "lowercase"}, "The sky is blue.", False),
    ({"kind": "lowercase"}, "123 ...", False),
    ({"kind": "uppercase"}, "THE SKY IS BLUE!", True),
    ({"kind": "uppercase"}, "THE SKY Is BLUE", False),
    ({"kind": "uppercase"}, "42", False),
    ({"kind": "start_with", "word": "Yes"}, "Yes, it is.", True),
    ({"kind": "start_with", "word": "Yes"}, "yes it is", True),
    ({"kind": "start_with", "word": "Yes"}, "\"Yes\" is the answer", True),
    ({"kind": "start_with", "word": "Yes"}, "Yesterday it was", False),
    ({"kind": "start_with", "word": "Yes"}, "I think yes", False),
    ({"kind": "end_with_word", "word": "done"}, "The cat sat, done.", True),
    ({"kind": "end_with_word", "word": "done"}, "Done is how it starts", False),
    ({"kind": "end_with_word", "word": "done"}, "well done!", True),
    ({"kind": "end_full_stop"}, "It is green.", True),
    ({"kind": "end_full_stop"}, "It is green.  ", True),
    ({"kind": "end_full_stop"}, "It is green...", False),
    ({"kind": "end_full_stop"}, "It is green!", False),
    ({"kind": "numbered_list", "n": 3}, "1. apple\n2. pear\n3. plum", True),
    ({"kind": "numbered_list", "n": 3}, "Here you go:\n1) apple\n2) pear\n3) plum", True),
    ({"kind": "numbered_list", "n": 3}, "1. apple\n2. pear", False),
    ({"kind": "numbered_list", "n": 3}, "1. apple\n2. pear\n3. plum\n4. fig", False),
    ({"kind": "numbered_list", "n": 3}, "1. apple\n3. pear\n2. plum", False),
    ({"kind": "numbered_list", "n": 3}, "1. apple\n2.\n3. plum", False),
    ({"kind": "numbered_list", "n": 3}, "apple, pear, plum", False),
    ({"kind": "repeat_word", "word": "apple", "n": 3}, "apple apple apple", True),
    ({"kind": "repeat_word", "word": "apple", "n": 3}, "Apple, apple, apple.", True),
    ({"kind": "repeat_word", "word": "apple", "n": 3}, "apple apple", False),
    ({"kind": "repeat_word", "word": "apple", "n": 3}, "apple apple apple pie", False),
    ({"kind": "repeat_phrase", "phrase": "The red ball."}, "the red ball", True),
    ({"kind": "repeat_phrase", "phrase": "The red ball."}, "The red ball is here.", False),
    ({"kind": "yes_no"}, "Yes.", True),
    ({"kind": "yes_no"}, "no", True),
    ({"kind": "yes_no"}, "Yes, it is.", False),
    ({"kind": "yes_no"}, "Maybe", False),
    ({"kind": "number_only"}, "4", True),
    ({"kind": "number_only"}, "Four.", True),
    ({"kind": "number_only"}, "4 legs", False),
    ({"kind": "number_only"}, "many", False),
    ({"kind": "include_word", "word": "blue"}, "The sky is Blue today.", True),
    ({"kind": "include_word", "word": "blue"}, "The sky is bluish.", False),
    ({"kind": "avoid_word", "word": "the"}, "A dog barks.", True),
    ({"kind": "avoid_word", "word": "the"}, "The dog barks.", False),
    ({"kind": "avoid_word", "word": "the"}, "", False),
    ({"kind": "avoid_word", "word": "the"}, "There it goes.", True),
    ({"kind": "quoted"}, "\"The sky is blue.\"", True),
    ({"kind": "quoted"}, "“The sky is blue”.", True),
    ({"kind": "quoted"}, "The sky is \"blue\"", False),
    ({"kind": "quoted"}, "\"\"", False),
    ({"kind": "quoted"}, "\"blue", False),
]


@pytest.mark.parametrize("spec,reply,expected", CASES)
def test_check(spec, reply, expected):
    assert C.check(spec, reply) is expected


def test_every_kind_has_cases():
    assert {s["kind"] for s, _, _ in CASES} == set(C.KINDS)


def test_unknown_kind():
    with pytest.raises(ValueError):
        C.check({"kind": "rhyme"}, "x")


def test_single_turn_set_is_consistent():
    items = iftiny_build.single_turn()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))
    assert {it["check"]["kind"] for it in items} == set(C.KINDS)
    for it in items:
        # the canned compliant reply complies and the canned violating reply does not: the checks and the
        # templates agree about what the instruction means
        assert C.check(it["check"], it["comply_example"]), it["id"]
        assert not C.check(it["check"], it["violate_example"]), it["id"]
        assert it["prompt"] != it["bare"]


def test_distance_set_history_complies():
    items = iftiny_build.distance()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))
    for it in items:
        if it["condition"] in ("instr_user", "instr_system", "nohint"):
            for _, reply in it["history"]:
                assert C.check(it["check"], reply), (it["id"], reply)
        if it["condition"] == "instr_system":
            assert it["system"] == it["instruction"]
            assert all(it["instruction"] not in u for u, _ in it["history"]) and it["instruction"] not in it["user"]
        if it["condition"] == "instr_user":
            assert it["system"] is None
            first = it["history"][0][0] if it["history"] else it["user"]
            assert first.startswith(it["instruction"] + " ")
        if it["condition"] in ("nohint", "bare"):
            assert it["system"] is None and all(it["instruction"] not in u for u, _ in it["history"])
        assert len(it["history"]) == it["distance"]


def test_frozen_files_match_builder():
    root = os.path.dirname(HERE)
    for name, fn in (("iftiny_v1.jsonl", iftiny_build.single_turn), ("ifdist_v1.jsonl", iftiny_build.distance)):
        path = os.path.join(root, "corpus", "eval", name)
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        assert rows == fn(), name
