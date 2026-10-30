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


def test_dollar_cap_counts_earlier_runs(tmp_path):
    cache = tmp_path / "c.jsonl"
    t = fake(lambda s: len(s))  # each call reports $0.01
    r1 = J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 100, t, max_usd=0.02)
    assert r1["calls"] == 2 and abs(r1["total_spent_usd"] - 0.02) < 1e-9
    r2 = J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 100, t, max_usd=0.02)
    assert r2["calls"] == 0  # the cap covers the cache's whole history
    r3 = J.run(reqs(), str(cache), "claude-test", str(RUBRIC), 100, t, max_usd=0.05)
    assert r3["calls"] == 2


def test_second_pair_needs_its_own_agreement(tmp_path):
    cache = tmp_path / "c.jsonl"
    score = {"I like pasta with tomato sauce.": 4, "I like rice.": 3, "Blue.": 2, "I am happy today today.": 1}
    J.run(reqs()[:1], str(cache), "claude-test", str(RUBRIC), 100, fake(lambda s: score.get(s, 0)))
    out = tmp_path / "p.jsonl"
    rep = J.pairs(reqs()[:1], str(cache), "claude-test", str(RUBRIC), str(out), per_prompt=2)
    ps = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [p["rank_pair"] for p in ps] == [[0, -1]]  # rice vs Blue. fails the length match (3 vs 1 words)
    assert rep["length_mismatched"] == 1


def test_cli_runs_in_an_empty_directory(tmp_path, monkeypatch):
    seen = {}

    class Done:
        stdout = json.dumps({"result": "{}", "total_cost_usd": 0.01})

    def run(argv, **kw):
        seen.update(kw)
        return Done()

    monkeypatch.setattr(J.subprocess, "run", run)
    assert J.claude_cli_transport("claude", "m")("sys", "user") == ("{}", 0.01)
    cwd = Path(seen["cwd"])
    assert cwd.is_dir() and not any(cwd.iterdir()) and HERE.parent.parent not in cwd.parents
    (tmp_path / "x").write_text("x")
    import pytest
    with pytest.raises(SystemExit, match="must be empty"):
        J.claude_cli_transport("claude", "m", workdir=str(tmp_path))


def test_pairs_carry_how_each_reply_stopped(tmp_path):
    cache = tmp_path / "c.jsonl"
    rs = reqs()
    t = fake(lambda s: ("pasta" in s) * 2 + ("well" in s) - ("today today" in s) * 3)
    J.run(rs, str(cache), "claude-test", str(RUBRIC), 100, t)
    rs[0]["sampler"] = {"runs": [{"stop": "eos"}, {"stop": "length"}, {"stop": "length"}, {"stop": "turn"}],
                        "candidate_runs": [[0], [1], [2], [3]]}
    out = tmp_path / "p.jsonl"
    J.pairs(rs, str(cache), "claude-test", str(RUBRIC), str(out))
    ps = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert ps[0]["chosen_stop"] == "eos" and ps[0]["rejected_stop"] == "length"
    assert "chosen_stop" not in ps[1]  # a request without sampler runs


def test_cli_limit_stops_the_run(monkeypatch):
    import pytest

    class Done:
        stdout = json.dumps({"result": "You've hit your session limit", "total_cost_usd": 0.0})

    monkeypatch.setattr(J.subprocess, "run", lambda argv, **kw: Done())
    with pytest.raises(SystemExit, match="session limit"):
        J.claude_cli_transport("claude", "m")("sys", "user")


def test_workers_give_the_same_verdicts_and_respect_the_cap(tmp_path):
    t1, t3 = fake(lambda s: len(s)), fake(lambda s: len(s))
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    J.run(reqs(), str(a), "claude-test", str(RUBRIC), 100, t1)
    r = J.run(reqs(), str(b), "claude-test", str(RUBRIC), 100, t3, workers=3)
    assert r["calls"] == 4
    va = {c["key"]: c["ranking"] for c in map(json.loads, a.read_text(encoding="utf-8").splitlines())}
    vb = {c["key"]: c["ranking"] for c in map(json.loads, b.read_text(encoding="utf-8").splitlines())}
    assert va == vb
    capped = J.run(reqs(), str(tmp_path / "c.jsonl"), "claude-test", str(RUBRIC), 100, fake(len), max_usd=0.02,
                   workers=3)
    assert capped["total_spent_usd"] <= 0.02 + 1e-9


def test_cli_paid_error_preserves_reported_cost(monkeypatch):
    import pytest

    class Done:
        stdout = json.dumps({"result": "provider aborted", "is_error": True, "total_cost_usd": 0.03})

    monkeypatch.setattr(J.subprocess, "run", lambda argv, **kw: Done())
    with pytest.raises(J.JudgeStopped) as caught:
        J.claude_cli_transport("claude", "m")("sys", "user")
    assert caught.value.cost_usd == 0.03
    assert caught.value.response == "provider aborted"


def test_paid_stop_is_accounted_and_stops_new_calls(tmp_path, capsys):
    import pytest

    cache = tmp_path / "c.jsonl"
    attempts = []

    def stopped(system, user):
        attempts.append(user)
        raise J.JudgeStopped("provider aborted", 0.03)

    with pytest.raises(J.JudgeStopped):
        J.run(reqs(), str(cache), "test", str(RUBRIC), 100, stopped, max_usd=0.05)
    report = json.loads(capsys.readouterr().out)
    assert report == {"calls": 1, "cost_usd": 0.03, "failures": 1, "total_spent_usd": 0.03}
    failures = J.read_jsonl(str(cache) + ".failures.jsonl")
    assert len(attempts) == len(failures) == 1
    assert failures[0]["transport_stopped"] is True
    assert failures[0]["cost_usd"] == 0.03
    assert not cache.read_text(encoding="utf-8").strip()
    # The next run must reserve the largest failed call too: .03 + .03 exceeds .05.
    after = J.run(reqs(), str(cache), "test", str(RUBRIC), 100, fake(len), max_usd=0.05)
    assert after["calls"] == 0 and after["total_spent_usd"] == 0.03


def test_free_weekly_limit_stops_once_without_failure_ledger(tmp_path, monkeypatch, capsys):
    import pytest

    attempts = []

    class Done:
        # A limit can be returned as text without is_error; it still must stop.
        stdout = json.dumps({"result": "You've hit your weekly limit", "total_cost_usd": 0.0})

    def cli(argv, **kwargs):
        attempts.append(argv)
        return Done()

    monkeypatch.setattr(J.subprocess, "run", cli)
    cache = tmp_path / "c.jsonl"
    with pytest.raises(J.JudgeStopped, match="weekly limit"):
        J.run(reqs(), str(cache), "test", str(RUBRIC), 100,
              J.claude_cli_transport("claude", "m"), max_usd=200)
    assert len(attempts) == 1
    assert json.loads(capsys.readouterr().out)["cost_usd"] == 0
    assert J.spent(str(cache)) == 0
    assert not J.read_jsonl(str(cache) + ".failures.jsonl")
