"""side_by_side.py without a server or a judge: same body for both models, skips (context, identical,
invalid UTF-8), and the count (both orders agree = a win, else a tie; Wilson interval)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chat_judge as J  # noqa: E402
import side_by_side as B  # noqa: E402

RUBRIC = os.path.join(HERE, "rubrics", "chat_v1.txt")


def encode(text):
    return [ord(c) for c in text]


def server(reply, stop="eos", log=None):
    def send(body):
        if log is not None:
            log.append(body)
        return {"prompt_ids": encode(f"<|user|>{body['user_input']}<|assistant|>"), "reply": reply(body),
                "stop_reason": stop, "tokens": 5}
    return send


def test_sample_same_body_and_skips():
    la, lb = [], []
    items = [("u0", "Hi"), ("u1", "x" * 300), ("u2", "Yo"), ("u3", "Hey")]
    a = server(lambda b: "same" if b["user_input"] == "Yo" else f"a{b['seed']}", log=la)
    b = server(lambda b: "same" if b["user_input"] == "Yo" else ("bad�x" if b["user_input"] == "Hey" else "b"),
               log=lb)
    out = list(B.sample(items, a, b, 0.7, 10, 8, encode, 256))
    assert [w for _, w in out] == ["ok", "over_context", "identical", "invalid_utf8"]
    assert out[0][0]["candidates"] == ["a10", "b"] and la == lb and la[0]["seed"] == 10 and la[1]["seed"] == 12


def test_count_wins_ties_and_interval(tmp_path):
    reqs = [{"prompt": f"<|user|>q{i}<|assistant|>", "candidates": [f"good {i}", f"bad {i}"], "uri": f"u{i}",
             "sampler": {"models": ["sft", "dpo"], "runs": [{"stop": "eos"}, {"stop": "length"}],
                         "candidate_runs": [[0], [1]]}} for i in range(4)]

    def call(system, user):  # prefers "good" except on q3, where it prefers whatever is shown first
        lines = [ln for ln in user.splitlines() if ln[:3] in ("A: ", "B: ")]
        if "q3" in user:
            return json.dumps({"ranking": ["A", "B"], "rationale": "first"}), 0.0
        best = "A" if "good" in lines[0] else "B"
        return json.dumps({"ranking": [best, "B" if best == "A" else "A"], "rationale": "r"}), 0.0

    cache = tmp_path / "c.jsonl"
    J.run(reqs, str(cache), "judge", RUBRIC, 100, call)
    rep = B.count(reqs, str(cache), "judge", RUBRIC)
    assert rep["wins"] == {"sft": 3, "dpo": 0} and rep["ties"] == 1 and rep["b_win_share"] == 0.0
    assert rep["eos_share"] == {"sft": 1.0, "dpo": 0.0}
    lo, hi = B.wilson(30, 50)
    assert lo < 0.6 < hi and 0.45 < lo < 0.47


def test_dpo_pairs_drops_over_T_and_counts_stops():
    import dpo_pairs as D
    ps = [{"prompt": "ab", "chosen": "c" * 5, "rejected": "d", "chosen_stop": "length", "rejected_stop": "eos"},
          {"prompt": "ab", "chosen": "c" * 6, "rejected": "d", "chosen_stop": "eos", "rejected_stop": "eos"},
          {"prompt": "ab", "chosen": "c", "rejected": "d"}]
    kept, rep = D.build(ps, lambda s: list(s), 7)  # T + 1 = 8: 2 + 5 fits, 2 + 6 + eos does not
    assert [len(p["chosen"]) for p in kept] == [5, 1]
    assert rep["over_T"] == 1 and rep["chosen_finished"] == 1 and rep["val_pairs"] == 1
    assert rep["stops_chosen_rejected"] == {"length/eos": 1, "eos/eos": 1}
