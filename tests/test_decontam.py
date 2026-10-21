"""corpus/decontam.py: the drop rules (13-gram, exact prompt, long-prompt span), the answer flag,
and the CLI's kept rows and report."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"))
import decontam  # noqa: E402

LONG_REF = ("perhaps we can discuss why we feel bad in our lives and how to improve it together "
            "with friends and family")


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _dc(tmp_path):
    qa, tc = tmp_path / "qa.jsonl", tmp_path / "tc.jsonl"
    _write(qa, [{"prompt": "What is the capital of France?", "answers": ["paris"], "kind": "fact"}])
    _write(tc, [{"prompt": "I think we should let our feelings out and talk about something important today",
                 "reference": LONG_REF},
                {"prompt": "Hello there, how are you doing on this fine morning?", "reference": "Fine."}])
    return decontam.Decontaminator(str(qa), [str(tc)])


def test_words_normalise():
    assert decontam.words("What's THE capital--of France?") == ["what", "s", "the", "capital", "of", "france"]


def test_exact_prompt_drops(tmp_path):
    d, _ = _dc(tmp_path).check(["what is the capital of FRANCE"], "what is the capital of FRANCE")
    assert [x["rule"] for x in d] == ["prompt"]


def test_ngram_drops_reference_overlap(tmp_path):
    text = "Sure. " + " ".join(LONG_REF.split()[:13]) + " and more."
    d, _ = _dc(tmp_path).check(["hi"], text)
    assert any(x["rule"] == "ngram" and x["eval"].endswith(":reference") for x in d)


def test_twelve_words_do_not_drop(tmp_path):
    text = " ".join(LONG_REF.split()[:12])
    d, _ = _dc(tmp_path).check(["hi"], text)
    assert d == []


def test_span_drops_long_prompt_inside_a_turn(tmp_path):
    d, _ = _dc(tmp_path).check(["so, hello there how are you doing on this fine morning? ok"], "x")
    assert d == []  # the span rule reads the training text, not only user turns
    d, _ = _dc(tmp_path).check(["x"], "so, hello there how are you doing on this fine morning? ok")
    assert [x["rule"] for x in d] == ["span"]


def test_span_boundary_is_eight_words(tmp_path):
    qa, tc = tmp_path / "qa.jsonl", tmp_path / "tc.jsonl"
    _write(qa, [])
    _write(tc, [{"prompt": "one two three four five six seven"},
                {"prompt": "alpha beta gamma delta epsilon zeta eta theta"}])
    dc = decontam.Decontaminator(str(qa), [str(tc)])
    assert dc.check(["x"], "so one two three four five six seven ok")[0] == []
    d, _ = dc.check(["x"], "so alpha beta gamma delta epsilon zeta eta theta ok")
    assert [x["rule"] for x in d] == ["span"]


def test_answer_flag_needs_question_words_nearby(tmp_path):
    dc = _dc(tmp_path)
    d, f = dc.check(["tell me"], "The capital of France is Paris, a city on the Seine.")
    assert d == [] and [x["rule"] for x in f] == ["answer"]
    _, f = dc.check(["tell me"], "We went to Paris last summer and ate well.")
    assert f == []


def test_cli_keeps_clean_rows_and_reports(tmp_path):
    inp, out, rep = tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path / "rep.json"
    rows = [{"messages": [{"role": "user", "content": "What is the capital of France?"},
                          {"role": "assistant", "content": "Paris."}], "uri": "a"},
            {"messages": [{"role": "user", "content": "Tell me a joke"},
                          {"role": "assistant", "content": "Why did the chicken cross the road?"}], "uri": "b"}]
    _write(inp, rows)
    s = decontam.run(str(inp), str(out), str(rep), dc=_dc(tmp_path))
    assert s["rows_in"] == 2 and s["rows_kept"] == 1 and s["dropped_by_rule"] == {"prompt": 1}
    kept = [json.loads(line) for line in open(out)]
    assert [r["uri"] for r in kept] == ["b"]
    report = json.load(open(rep))
    assert report["dropped"][0]["uri"] == "a" and report["dropped"][0]["matches"][0]["eval"] == "qa:1"


def test_short_prompts_and_arithmetic_answers_pass(tmp_path):
    qa, tc = tmp_path / "qa.jsonl", tmp_path / "tc.jsonl"
    _write(qa, [{"prompt": "What is one plus one?", "answers": ["two"], "kind": "arith"},
                {"prompt": "Hi!", "answers": ["hello"], "kind": "turn"}])
    _write(tc, [{"prompt": "Hey there", "reference": "Hello."}])
    dc = decontam.Decontaminator(str(qa), [str(tc)])
    d, f = dc.check(["Hi!"], "Hi! One plus one is two, as everyone knows.")
    assert d == [] and f == []
    d, _ = dc.check(["What is one plus one"], "x")
    assert [x["rule"] for x in d] == ["prompt"]


def test_frozen_eval_sets_load():
    dc = decontam.Decontaminator()
    names = {e.split(":")[0] for e in dc.prompts.values()}
    assert names == {"simple_qa_v1", "tinychat_hf_prompts_v1", "chat_prompts_v1"}
    assert dc.answers and dc.patterns
