"""Carry-5 (tools/carry5_build.py, tools/carry5_eval.py): the frozen set rebuilds byte for byte from
its generator, its items are well formed, and the scorer's checks and tie rule behave as documented.
Offline: no model, no server."""
import json
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import carry5_build as cb  # noqa: E402
import carry5_eval as ce  # noqa: E402


def items():
    with open(cb.OUT, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def test_generator_rebuilds_the_frozen_file():
    with open(cb.OUT, "rb") as f:
        assert cb.dump(cb.build_items()) == f.read()


def test_families_and_controls():
    its = items()
    fams = {}
    for it in its:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
    assert fams == {"recall": 432, "correction": 216, "instruction": 144, "topic": 180, "knowledge": 60}
    for it in its:
        assert len(it["turns"]) == 5 and it["target_turn"] in (4, 5)
        assert len(set(it["candidates"])) == 4 and 0 <= it["gold_index"] < 4
        before = " ".join(t["user"] for t in it["turns"][:it["target_turn"] - 1])
        for v in it["distractor_values"]:
            assert v in before and v in it["twin"] and v in it["candidates"]
        if it["family"] == "correction":
            assert it["old_value"] in it["candidates"] and it["old_value"] != it["candidates"][it["gold_index"]]
        if it["family"] in ("recall", "correction", "topic"):
            gold = it["candidates"][it["gold_index"]]
            assert gold in it["turns"][it["fact_turn"] - 1]["user"] and gold in it["twin"]
            assert gold not in it["noctx"]


def test_no_two_items_share_a_twin_or_user_turns():
    its = items()
    assert len({it["twin"] for it in its}) == len(its)
    assert len({tuple(t["user"] for t in it["turns"]) for it in its}) == len(its)


def test_gold_position_is_balanced():
    pos = [0] * 4
    for it in items():
        pos[it["gold_index"]] += 1
    assert min(pos) > 0.18 * sum(pos)


def test_instruction_checks():
    c = ce.check_instruction
    assert c("one_word", "White.") and not c("one_word", "Snow is white.")
    assert c("short", "A dog barks.") and not c("short", "Well, a dog barks, as far as I know.")
    assert c("yes_no", "Yes.") and c("yes_no", "No, never") and not c("yes_no", "I think so.")
    assert c("number", "4.") and c("number", "Four") and not c("number", "A cat has four legs.")


def test_reply_verdicts_contains_and_exclusive():
    it = next(x for x in items() if x["family"] == "correction")
    gold = it["candidates"][it["gold_index"]]
    assert ce.reply_verdicts(it, it["frame"].format(gold))["exclusive"]
    both = ce.reply_verdicts(it, f"{gold} or {it['old_value']}")
    assert both["contains"] and not both["exclusive"] and both["names_distractor"]


def test_ties_give_chance():
    assert ce.choose([0.0, 0.0, 0.0, 0.0], 2) == (0.25, 0)
    assert ce.choose([-1.0, -3.0, -0.5, -2.0], 2) == (1.0, 2)


def test_uniform_scorer_is_exactly_chance():
    its = items()
    scores = {f"{it['id']}|{m}|{c}": {"over_ctx": False, "sum": 0.0, "logprobs": [0.0], "argmax": []}
              for it in its for m in ce.MODES for c in range(4)}
    s = ce.aggregate(ce.item_records(its, {}, scores, None))
    assert s["fc"]["acc"]["gold"]["all"]["mean"] == 0.25
    assert s["fc"]["acc_norm"]["noctx"]["all"]["mean"] == 0.25


FAKE_SCORE = """import json, os, sys
state = sys.argv[-5] + ".n"
n = int(open(state).read()) + 1 if os.path.exists(state) else 1
open(state, "w").write(str(n))
if n <= int(os.environ["FAKE_FAILS"]):
    sys.stderr.write("CUDA error at matrix_ops.cu:44: out of memory" + chr(10))
    sys.exit(1)
out = sys.argv[sys.argv.index("--out") + 1]
with open(out, "w") as f:
    for line in open(sys.argv[sys.argv.index("--in") + 1]):
        r = json.loads(line)
        lp = [-1.0] * len(r["continuation_ids"])
        f.write(json.dumps({"key": r["key"], "over_ctx": False, "logprobs": lp, "argmax": r["continuation_ids"],
                            "sum": sum(lp)}) + chr(10))
"""


@pytest.mark.parametrize("fails", [0, 1, 2])
def test_mtstudio_score_failure_keeps_stderr_and_retries_once(tmp_path, monkeypatch, fails):
    """A failed `mtstudio score` attempt is rerun once; its stderr is kept beside the chunk and ends
    the exit message (a rung's score died on 25 Oct 2026 with exit 1 and nothing kept)."""
    fake = tmp_path / "fake_mtstudio.py"
    fake.write_text(FAKE_SCORE, encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FAKE_FAILS", str(fails))
    # The argv carry5_eval builds is [mtstudio, "score", SPEC, ...]: run the fake through a wrapper.
    wrap = tmp_path / ("mtstudio.cmd" if os.name == "nt" else "mtstudio")
    if os.name == "nt":
        wrap.write_text(f'@"{sys.executable}" "{fake}" %*\n', encoding="utf-8")
    else:
        wrap.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n', encoding="utf-8")
        wrap.chmod(0o755)

    class A:
        scorer, mtstudio = "mtstudio", str(wrap)

    reqs = [(f"k{i}", [1, 2, 3], [4, 5]) for i in range(3)]
    log = tmp_path / "score" / "chunk0000.json.out.jsonl.stderr.txt"
    if fails >= 2:
        with pytest.raises(SystemExit) as e:
            ce.score_all(A, reqs, str(spec), None, str(tmp_path))
        assert "out of memory" in str(e.value) and "failed 2 times" in str(e.value)
        assert log.read_text(encoding="utf-8").count("out of memory") == 2
        return
    res = ce.score_all(A, reqs, str(spec), None, str(tmp_path))
    assert set(res) == {"k0", "k1", "k2"} and res["k1"]["sum"] == -2.0
    assert log.exists() == (fails == 1)
    if fails == 1:
        assert "attempt 1: rc 1" in log.read_text(encoding="utf-8")


def test_template_is_the_served_chat_format():
    assert ce.template([("Hi", "Hello!")], "Bye") == "<|user|>Hi<|assistant|>Hello!<|endoftext|><|user|>Bye<|assistant|>"


@pytest.mark.skipif(not os.environ.get("MTSTUDIO_BIN"), reason="set MTSTUDIO_BIN to a built mtstudio")
def test_mtstudio_score_argmax_equals_chat_greedy(tmp_path):
    """`mtstudio score` on `mtstudio chat`'s greedy reply (plus its stop token): argmax == reply at every
    position, log-probs <= 0 summing to `sum`, and a sequence past T + 1 ids reported over_ctx."""
    import bpe
    from test_mtstudio_chat import _free_port, _get, _post
    binp = os.environ["MTSTUDIO_BIN"]
    pairs = [("hello", "Hi there!"), ("what is your name?", "I am a small model."), ("bye", "Goodbye!")]
    text = "".join(f"<|user|>{u}<|assistant|>{a}<|endoftext|>\n" for u, a in pairs) * 20
    tok = bpe.train(text, 400)
    bpe.write_gguf(str(tmp_path / "v.gguf"), tok)
    ex = [{"prompt": f"<|user|>{u}<|assistant|>", "completion": a} for u, a in pairs] * 7
    (tmp_path / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    spec = {"name": "m", "arch": {"preset": "llama-tiny"},
            "data": {"vocab": str(tmp_path / "v.gguf"), "vocab_cap": 4096, "T": 32,
                     "sft_examples": str(tmp_path / "ex.jsonl")},
            "train": {"steps": 300, "lr": 0.003, "batch": 4, "eval_every": 100, "checkpoint_every": 1000,
                      "gradmap_every": 0, "seed": 1, "objective": "sft"},
            "export": {"formats": ["safetensors"]}, "out_dir": str(tmp_path / "run")}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    assert subprocess.run([binp, "run", str(sp)], capture_output=True, timeout=1200).returncode == 0
    port = _free_port()
    proc = subprocess.Popen([binp, "chat", str(sp), "--port", str(port)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                if _get(base + "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)
        rows = []
        for hist, user in (([], "hello"), ([("hello", "Hi there!")], "what is your name?")):
            status, j = _post(base + "/chat", {"user_input": user, "top_k": 1, "max_new_tokens": 12, "repeat_last_n": 0,
                                               "history": [{"user": u, "assistant": a} for u, a in hist]})
            assert status == 200, j
            stop = [1] if j["stop_reason"] == "eos" else []
            rows.append((tok.encode(ce.template(hist, user)), tok.encode(j["reply"]) + stop))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    long_p = tok.encode("<|user|>" + "hello " * 40 + "<|assistant|>")
    with open(tmp_path / "in.jsonl", "w", encoding="utf-8") as f:
        for p, c in rows + [(long_p, [1])]:
            f.write(json.dumps({"prompt_ids": p, "continuation_ids": c}) + "\n")
    r = subprocess.run([binp, "score", str(sp), "--in", str(tmp_path / "in.jsonl"), "--out", str(tmp_path / "out.jsonl")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    out = [json.loads(x) for x in open(tmp_path / "out.jsonl", encoding="utf-8")]
    for (p, c), o in zip(rows, out):
        assert not o["over_ctx"] and o["argmax"] == c
        assert all(x <= 0 for x in o["logprobs"]) and abs(sum(o["logprobs"]) - o["sum"]) < 1e-9
    assert out[-1]["over_ctx"]
