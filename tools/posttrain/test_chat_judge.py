"""tools/posttrain/chat_judge.py with a fake transport (no CLI calls): both
orders are judged, the cache is idempotent and capped, malformed verdicts go to
failures, and pairs need position agreement and matched lengths."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat_judge as J  # noqa: E402

RUBRIC = HERE / "rubrics" / "chat_v1.txt"


def fake(prefer):
    """Ranks candidates by prefer(text) descending, answering in labels."""
    calls = []

    def call(system, user):
        calls.append(user)
        cands = [ln.split(": ", 1) for ln in user.splitlines() if len(ln) > 2 and ln[1:3] == ": " and ln[0] in J.LABELS]
        ranked = sorted(cands, key=lambda c: -prefer(c[1]))
        return json.dumps({"ranking": [c[0] for c in ranked], "rationale": "fake"}), 0.01
    call.calls = calls
    return call


def reqs():
    return [
        {"prompt": "<|user|>What is your favourite food?<|assistant|>", "uri": "q1",
         "candidates": ["I like pasta with tomato sauce.", "Blue.", "I am happy today today.", "I like rice."]},
        {"prompt": "<|user|>Hi<|assistant|>Hello!<|endoftext|><|user|>How are you?<|assistant|>", "uri": "q2",
         "candidates": ["I am well, thanks.", "I am fine thank you."]},
    ]


def test_run_cache_pairs(tmp_path):
    cache = tmp_path / "c.jsonl"
    t = fake(lambda s: ("pasta" in s) * 2 + ("well" in s) - ("today today" in s) * 3)
    rep = J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 100, t)
    assert rep["calls"] == 4 and rep["failures"] == 0
    assert "Conversation so far:" in t.calls[2] and "User: Hi" in t.calls[2]
    again = J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 100, t)
    assert again["calls"] == 0  # idempotent
    out = tmp_path / "p.jsonl"
    rep = J.pairs(reqs(), str(cache), "claude-test", str(RUBRIC), str(out))
    ps = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    # q1: best "pasta" (6 words), worst "today today" (5 words): agreed, length-matched.
    assert rep["pairs"] == 2 and ps[0]["chosen"].startswith("I like pasta") and "today today" in ps[0]["rejected"]
    assert ps[0]["chosen_evidence"]["evidence_kind"] == "judge"


def test_position_bias_drops_the_prompt(tmp_path):
    cache = tmp_path / "c.jsonl"

    def first_wins(system, user):  # always ranks in label order: pure position bias
        labs = [ln[0] for ln in user.splitlines() if len(ln) > 2 and ln[1:3] == ": " and ln[0] in J.LABELS]
        return json.dumps({"ranking": labs, "rationale": "first"}), 0.0
    J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 100, first_wins)
    rep = J.pairs(reqs(), str(cache), "claude-test", str(RUBRIC), str(tmp_path / "p.jsonl"))
    assert rep["pairs"] == 0 and rep["position_disagreements"] == 2


def test_cap_and_failures(tmp_path):
    cache = tmp_path / "c.jsonl"
    rep = J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 3, lambda s, u: ("not json", 0.0))
    assert rep["calls"] == 3 and rep["failures"] == 3
    assert not cache.read_text(encoding="utf-8").strip()
    assert len((tmp_path / "c.jsonl.failures.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_parse_rejects_non_permutations():
    assert J.parse_ranking('{"ranking": ["A", "A"], "rationale": "x"}', [0, 1])[0] is None
    assert J.parse_ranking('```json\n{"ranking": ["B", "A"], "rationale": "x"}\n```', [1, 0])[0] == [0, 1]
