"""Cross-language round trip: a verdict cache written by judge_sidecar.py is
read by the C++ LLMJudgeVerifier (through tools/posttrain_verify.cpp) and
every receipt comes back intact; refusals and misses stay refusals and misses.

Needs the built binary: POSTTRAIN_VERIFY_BIN=build/posttrain_verify. Skipped
when it is absent, so the Python-only suites still run anywhere.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_sidecar as js  # noqa: E402

BIN = os.environ.get("POSTTRAIN_VERIFY_BIN", "")
RUBRIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubrics", "tinyarith_v1.txt")
pytestmark = pytest.mark.skipif(not (BIN and os.path.exists(BIN)),
                                reason="set POSTTRAIN_VERIFY_BIN to the built posttrain_verify")


def verify(kind, records, *args):
    p = subprocess.run([BIN, kind, *args], input="".join(json.dumps(r) + "\n" for r in records),
                       capture_output=True, text=True, encoding="utf-8", check=True)
    return [json.loads(line) for line in p.stdout.splitlines() if line.strip()]


def test_sidecar_cache_is_read_by_cpp(tmp_path):
    cache = str(tmp_path / "cache.jsonl")
    reqs = [{"prompt": "lily has 3 apples — and 4 more", "completion": "the answer is \"7\""},
            {"prompt": "p2", "completion": "the answer is 9"},
            {"prompt": "p3", "completion": "c3"}]
    replies = iter([json.dumps({"verdict": 1, "rationale": "7 is 3 + 4 ✔"}),
                    json.dumps({"verdict": 0, "rationale": "9 is wrong"}),
                    "not json"])
    js.run(reqs, cache, "judge-m", RUBRIC, 10, lambda m: (lambda c: (c, c.encode()))(next(replies)))
    _, version, _ = js.rubric_identity(RUBRIC)
    out = verify("llm_judge", reqs + [{"prompt": "never", "completion": "asked", "seed": 17}],
                 "--cache", cache, "--judge-id", "judge-m", "--rubric-version", version)
    assert [o["reward"] for o in out] == [1.0, 0.0, -1.0, -1.0]
    assert out[0]["rationale"] == "7 is 3 + 4 ✔" and out[0]["prompt"] == reqs[0]["prompt"]
    assert all(o["evidence_kind"] == "judge" and o["evidence_uri"] == f"judge:judge-m@{version}" for o in out)
    assert out[2]["rationale"] == "no verdict on record"     # the malformed reply never became a verdict
    assert out[3]["seed"] == 17                               # extra fields are carried through
    # A different rubric version sees none of these verdicts.
    other = verify("llm_judge", reqs[:1], "--cache", cache, "--judge-id", "judge-m", "--rubric-version", "x@0")
    assert other[0]["reward"] == -1.0


def test_exact_match_and_format_records(tmp_path):
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps({"prompt": "q", "answer": 7, "uri": "tinyarith/0"}) + "\n", encoding="utf-8")
    recs = [{"prompt": "q", "completion": "the answer is 7"}, {"prompt": "q", "completion": "the answer is 8"}]
    em = verify("exact_match", recs, "--refs", str(refs))
    fm = verify("format", recs)
    assert [r["reward"] for r in em] == [1.0, 0.0] and [r["reward"] for r in fm] == [1.0, 1.0]
    assert em[0]["verifier"] == "exact_match:v1" and em[0]["evidence_uri"] == "tinyarith/0"


def test_malformed_input_line_fails_loudly():
    p = subprocess.run([BIN, "format"], input='{"prompt": "x"}\n', capture_output=True, text=True)
    assert p.returncode == 1 and "line 1" in p.stderr
