"""judge_sidecar.py, with a fake transport: no network, no credentials."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_sidecar as js  # noqa: E402

RUBRIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubrics", "tinyarith_v1.txt")
REQS = [{"prompt": f"p{i}", "completion": f"the answer is {i}"} for i in range(5)]


def fake(replies):
    calls = []

    def t(messages):
        calls.append(messages)
        r = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r, r.encode()
    t.calls = calls
    return t


def good(v=1, why="7 is 3 + 4"):
    return json.dumps({"verdict": v, "rationale": why})


def test_cap_skip_and_append_only(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    t = fake([good()])
    out = js.run(REQS, cache, "m", RUBRIC, 3, t)
    assert out["calls"] == 3 and out["verdicts"] == 3 and out["left"] == 2 and len(t.calls) == 3
    out = js.run(REQS, cache, "m", RUBRIC, 10, t)          # resumes; earlier pairs skipped
    assert out["pending"] == 2 and out["calls"] == 2
    rows = js.read_jsonl(cache)
    assert len(rows) == 5 and all(r["rationale"] and r["reward"] == 1.0 for r in rows)
    assert all(r["rubric_version"].startswith("tinyarith_v1@") for r in rows)
    assert js.run(REQS, cache, "m", RUBRIC, 10, t)["calls"] == 0
    assert js.run(REQS, cache, "other-model", RUBRIC, 1, t)["calls"] == 1   # per judge


def test_bad_responses_never_become_verdicts(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    t = fake([good(1, "   "), "not json", json.dumps({"verdict": 2, "rationale": "x"}),
              json.dumps({"verdict": True, "rationale": "x"}), RuntimeError("HTTP 500")])
    out = js.run(REQS, cache, "m", RUBRIC, 5, t)
    assert out["verdicts"] == 0 and out["failures"] == 5
    assert js.read_jsonl(cache) == []
    fails = js.read_jsonl(cache + ".failures.jsonl")
    assert len(fails) == 5 and all("error" in f and "reward" not in f for f in fails)


def test_judge_never_sees_a_reference(tmp_path):
    t = fake([good()])
    js.run([{"prompt": "p", "completion": "c", "answer": 7}], str(tmp_path / "c.jsonl"), "m", RUBRIC, 1, t)
    user = json.loads(t.calls[0][1]["content"])
    assert set(user) == {"problem", "response"}


def test_max_calls_required_and_zero_means_zero(tmp_path):
    with pytest.raises(SystemExit):
        js.run(REQS, str(tmp_path / "c.jsonl"), "m", RUBRIC, None, fake([good()]))
    assert js.run(REQS, str(tmp_path / "c.jsonl"), "m", RUBRIC, 0, fake([good()]))["calls"] == 0


def test_env_file_reads_only_the_named_key(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OTHER=1\n# OPENAI_API_KEY=commented\nexport OPENAI_API_KEY=\"sk-test\"\n")
    assert js.read_key(str(p)) == "sk-test"
    p.write_text("OTHER=1\n")
    with pytest.raises(SystemExit):
        js.read_key(str(p))


def test_rubric_edit_changes_version(tmp_path):
    r = tmp_path / "rub.txt"
    r.write_text("grade it")
    _, v1, _ = js.rubric_identity(str(r))
    r.write_text("grade it carefully")
    _, v2, _ = js.rubric_identity(str(r))
    assert v1 != v2 and v1.startswith("rub@")
