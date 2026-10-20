"""Lever 2's dialogue text (corpus/fetchers/dialogue_pt.py and the dialogue helpers in _common):
system-turn rule, markers, alternation, SODA speakers, the oasst best branch, rendering, and a
whole build on fixture rows, without downloads."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
sys.path.insert(0, os.path.join(ROOT, "corpus"))
import _common as C  # noqa: E402
import decontam  # noqa: E402
import dialogue_pt as D  # noqa: E402


def u(t):
    return {"role": "user", "content": t}


def a(t):
    return {"role": "assistant", "content": t}


def test_boilerplate_system_is_stripped_but_an_instruction_drops():
    turns, why = C.normalise_dialogue([{"role": "system", "content": "You are a helpful assistant."}, u("Hi"), a("Hello")])
    assert why == "system_stripped" and turns == [u("Hi"), a("Hello")]
    turns, why = C.normalise_dialogue([{"role": "system", "content": "Answer only in French."}, u("Hi"), a("Bonjour")])
    assert turns is None and why == "system_instruction"
    _, why = C.normalise_dialogue([u("Hi"), {"role": "system", "content": ""}, a("Hello")])
    assert why == "system_instruction"  # a system turn after the dialogue starts is never boilerplate


def test_markers_alternation_and_trailing_user():
    assert C.normalise_dialogue([u("say <|endoftext|>"), a("ok")])[1] == "markers"
    assert C.normalise_dialogue([a("Hello"), u("Hi")])[1] == "non_alternating"
    assert C.normalise_dialogue([u("Hi"), a(" "), u("x")])[1] == "empty_turn"
    turns, why = C.normalise_dialogue([u("Hi"), a("Hello"), u("Bye")])
    assert why == "kept" and turns == [u("Hi"), a("Hello")]
    assert C.normalise_dialogue([u("Hi")])[1] == "too_short"


def test_render_and_duplicate_key():
    t = [u("Hi"), a("Hello"), u("How are you?"), a("Well.")]
    assert C.render_dialogue(t) == "<|user|>Hi<|assistant|>Hello<|endoftext|><|user|>How are you?<|assistant|>Well.<|endoftext|>\n"
    assert C.dialogue_key(t) == C.dialogue_key([u("hi!"), a("HELLO"), u("how  are you"), a("well")])


def test_soda_speakers_must_alternate():
    m, why = D.soda_dialogue({"dialogue": ["Hi.", "Hey.", "Bye."], "speakers": ["Ann", "Bo", "Ann"]})
    assert why is None and [x["role"] for x in m] == ["user", "assistant", "user"]
    assert D.soda_dialogue({"dialogue": ["a", "b", "c"], "speakers": ["Ann", "Ann", "Bo"]})[1] == "non_alternating"
    assert D.soda_dialogue({"dialogue": ["a", "b", "c"], "speakers": ["Ann", "Bo", "Cy"]})[1] == "non_alternating"


def _msg(mid, parent, role, text, rank=None, lang="en", date="2023", deleted=False):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "rank": rank, "lang": lang,
            "deleted": deleted, "review_result": True, "created_date": date}


def test_oasst_best_branch_follows_rank_then_earliest_followup():
    msgs = {m["message_id"]: m for m in [
        _msg("r", None, "prompter", "Q1"),
        _msg("a1", "r", "assistant", "worse", rank=1), _msg("a0", "r", "assistant", "best", rank=0),
        _msg("p2", "a0", "prompter", "Q2 late", date="2024"), _msg("p1", "a0", "prompter", "Q2 early", date="2023"),
        _msg("p3", "a0", "prompter", "dead end", date="2022"),
        _msg("b0", "p1", "assistant", "A2", rank=0), _msg("b1", "p2", "assistant", "A2'", rank=0),
        _msg("x", "a1", "prompter", "elsewhere")]}
    children = {}
    for m in msgs.values():
        if m["parent_id"]:
            children.setdefault(m["parent_id"], []).append(m["message_id"])
    path = D.best_branch("r", children, msgs)
    assert [p["content"] for p in path] == ["Q1", "best", "Q2 early", "A2"]


def test_oasst_branch_stops_at_non_english_or_deleted():
    msgs = {m["message_id"]: m for m in [
        _msg("r", None, "prompter", "Q1"), _msg("a0", "r", "assistant", "A1", rank=0),
        _msg("p1", "a0", "prompter", "Hola", lang="es"), _msg("b0", "p1", "assistant", "x", rank=0)]}
    children = {"r": ["a0"], "a0": ["p1"], "p1": ["b0"]}
    assert [p["content"] for p in D.best_branch("r", children, msgs)] == ["Q1", "A1"]


def test_build_counts_every_reason(tmp_path):
    qa, tc = tmp_path / "qa.jsonl", tmp_path / "tc.jsonl"
    qa.write_text(json.dumps({"prompt": "What is the capital of France?", "answers": ["paris"], "kind": "fact"}) + "\n")
    tc.write_text(json.dumps({"prompt": "Tell me something nice", "reference": "You are great."}) + "\n")
    dc = decontam.Decontaminator(str(qa), [str(tc)])
    rows = [("s#0", [u("Hi"), a("Hello")], None),
            ("s#1", [u("hi!"), a("HELLO")], None),                                  # duplicate
            ("s#2", [{"role": "system", "content": "Be terse."}, u("Hi"), a("Yo")], None),
            ("s#3", [u("What is the capital of France?"), a("Paris.")], None),      # decontaminated
            ("s#4", None, "non_alternating"),
            ("s#5", [u("Where is Paris?"), a("Paris is the capital of France.")], None)]  # flagged, kept
    rep = D.build("soda", False, dc=dc, rows=rows, out_dir=str(tmp_path / "out"))
    c = rep["counts"]
    assert (c["rows_in"], c["kept"], c["duplicate"], c["system_instruction"], c["decontaminated"],
            c["non_alternating"], c["written"]) == (6, 2, 1, 1, 1, 1, 2)
    assert len(rep["answer_flags"]) == 1 and rep["decontam_dropped"][0]["uri"] == "s#3"
    text = (tmp_path / "out" / "dialogue_soda_v1" / "text.txt").read_text(encoding="utf-8")
    assert text.count("<|endoftext|>\n") == 2 and rep["bytes"] == len(text.encode("utf-8"))
