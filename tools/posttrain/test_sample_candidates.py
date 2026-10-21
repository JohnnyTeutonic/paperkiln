"""sample_candidates.py without a server: seeds and temperatures per candidate, duplicate replies and
their runs, the context skip, resumption (torn last line, configuration check), the prompt-id
check, and the prompt filters (eval overlap, train-split overlap, markers)."""
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


def test_identical_replies_collapse_with_their_runs():
    out = list(S.sample([("u0", "Hi")], fake_send([], reply=lambda b: "same"), 4, [0.9], 0, 8, 1.0, encode, 256))
    assert out == [({"uri": "u0", "reason": "fewer_than_two_distinct"}, "fewer_than_two_distinct")]
    out = list(S.sample([("u0", "Hi")], fake_send([], reply=lambda b: "ab"[b["seed"] % 2]), 4, [0.9], 0, 8, 1.0,
                        encode, 256))
    assert out[0][0]["candidates"] == ["a", "b"] and out[0][0]["sampler"]["candidate_runs"] == [[0, 2], [1, 3]]


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


def test_torn_last_line_is_cut_and_config_is_checked(tmp_path):
    p = tmp_path / "out.jsonl"
    good = {"uri": "a", "sampler": {"config": {"k": 4}}}
    p.write_bytes((json.dumps(good) + "\n" + '{"uri": "b", "sam').encode())
    rows = S.read_resumable(str(p))
    assert rows == [good] and p.read_bytes().endswith(b"\n")
    S.check_config(rows, {"k": 4}, str(p))
    with pytest.raises(SystemExit, match="different sampler configuration"):
        S.check_config(rows, {"k": 2}, str(p))
    S.check_config([{"uri": "c", "reason": "over_context", "config": {"k": 4}}], {"k": 4}, "skips")


def test_prompt_filters(tmp_path):
    qa, tc, split = tmp_path / "qa.jsonl", tmp_path / "tc.jsonl", tmp_path / "test.jsonl"
    qa.write_text(json.dumps({"prompt": "What is the capital of France?", "answers": ["paris"], "kind": "fact"}) + "\n")
    tc.write_text("")
    users = ["What is the capital of France?", "Plan my week", "Differentiate 3x^2 * sin(x)?", "say <|user|> now"]
    split.write_text("".join(json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": "a"}],
                                         "uri": f"s#{i}"}) + "\n" for i, u in enumerate(users)))
    train = {" ".join(decontam.words("Differentiate 3x^2 sin(x).")) }
    items, dropped = S.prompts(str(split), decontam.Decontaminator(str(qa), [str(tc)]), train_turns=train)
    assert items == [("s#1", "Plan my week")]
    assert dropped == {"decontaminated": 1, "train_overlap": 1, "markers": 1}
