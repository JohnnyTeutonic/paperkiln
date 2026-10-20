"""sample_candidates.py without a server: seeds and temperatures per candidate, duplicate replies,
the context skip, resumption, the prompt-id check, and decontaminated prompts."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "corpus"))
import decontam  # noqa: E402
import sample_candidates as S  # noqa: E402


def encode(text):
    return [ord(c) for c in text]


def fake_send(log, reply=lambda b: f"r{b['seed']}"):
    def send(body):
        log.append(body)
        prompt = f"<|user|>{body['user_input']}<|assistant|>"
        return {"prompt_ids": encode(prompt), "reply": reply(body), "stop_reason": "eos", "tokens": 3}
    return send


def test_seeds_temperatures_and_records():
    log = []
    out = list(S.sample([("u0", "Hi"), ("u1", "Yo")], fake_send(log), 4, [0.8, 1.0], 10, 8, 1.0, encode, 256))
    assert [w for _, w in out] == ["ok", "ok"]
    req = out[1][0]
    assert req["prompt"] == "<|user|>Yo<|assistant|>" and req["candidates"] == ["r14", "r15", "r16", "r17"]
    assert [r["temperature"] for r in req["sampler"]["runs"]] == [0.8, 1.0, 0.8, 1.0]
    assert all(b["top_k"] == 0 and "repeat_penalty" not in b for b in log)


def test_identical_replies_collapse_and_skip():
    out = list(S.sample([("u0", "Hi")], fake_send([], reply=lambda b: "same"), 4, [0.9], 0, 8, 1.0, encode, 256))
    assert out == [(None, "fewer_than_two_distinct")]
    out = list(S.sample([("u0", "Hi")], fake_send([], reply=lambda b: "ab"[b["seed"] % 2]), 4, [0.9], 0, 8, 1.0,
                        encode, 256))
    assert out[0][0]["candidates"] == ["a", "b"]


def test_context_skip_resume_and_penalty():
    log = []
    items = [("u0", "x" * 300), ("u1", "Hi"), ("u2", "Yo")]
    out = list(S.sample(items, fake_send(log), 2, [0.9], 0, 8, 1.1, encode, 256, done={"u2"}))
    assert [w for _, w in out] == ["over_context", "ok"]
    assert {b["user_input"] for b in log} == {"Hi"} and all(b["repeat_penalty"] == 1.1 for b in log)


def test_prompt_id_mismatch_stops():
    def send(body):
        return {"prompt_ids": [1, 2], "reply": "r", "stop_reason": "eos", "tokens": 1}
    with pytest.raises(SystemExit):
        list(S.sample([("u0", "Hi")], send, 2, [0.9], 0, 8, 1.0, encode, 256))


def test_prompts_drop_eval_overlap(tmp_path):
    qa, tc, split = tmp_path / "qa.jsonl", tmp_path / "tc.jsonl", tmp_path / "test.jsonl"
    qa.write_text(json.dumps({"prompt": "What is the capital of France?", "answers": ["paris"], "kind": "fact"}) + "\n")
    tc.write_text("")
    split.write_text("".join(json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": "a"}],
                                         "uri": f"s#{i}"}) + "\n"
                             for i, u in enumerate(["What is the capital of France?", "Plan my week"])))
    items, dropped = S.prompts(str(split), decontam.Decontaminator(str(qa), [str(tc)]))
    assert items == [("s#1", "Plan my week")] and dropped == 1
