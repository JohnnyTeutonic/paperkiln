"""Chat with a BPE chat model: the <|user|>/<|assistant|> template, end to end.

A tiny BPE model is SFT'd on ten templated exchanges, then served by
`mtstudio chat` and asked over the real /chat endpoint, with and without
history. Checks the wiring (template, stop reasons, byte decoding), plus
that the model learned the exchanges greedily. Needs MTSTUDIO_BIN.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bpe  # noqa: E402
from test_mtstudio_chat import BIN, _free_port, _get, _post  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

PAIRS = [("hello", "Hi there! How can I help?"), ("what is your name?", "I am a small paperkiln model."),
         ("do you like cats?", "Yes, cats are lovely."), ("thank you", "You are welcome!"),
         ("what colour is the sky?", "The sky is blue on a clear day."), ("good night", "Good night, sleep well."),
         ("tell me a joke", "Why did the bird sing? It was happy."), ("how are you?", "I am well, thanks."),
         ("what is 2 plus 2?", "2 plus 2 is 4."), ("bye", "Goodbye!")]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bpechat")
    text = "".join(f"<|user|>{u}<|assistant|>{a}<|endoftext|>\n" for u, a in PAIRS) * 20
    tok = bpe.train(text, 500)
    bpe.write_gguf(str(tmp / "v.gguf"), tok)
    ex = [{"prompt": f"<|user|>{u}<|assistant|>", "completion": a} for u, a in PAIRS] * 4 + \
         [{"prompt": "<|user|>hello<|assistant|>", "completion": PAIRS[0][1]}]
    (tmp / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    spec = {"name": "chat", "arch": {"preset": "llama-tiny"},
            "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 4096, "T": 32, "sft_examples": str(tmp / "ex.jsonl")},
            "train": {"steps": 400, "lr": 0.003, "batch": 4, "eval_every": 100, "checkpoint_every": 1000,
                      "gradmap_every": 0, "seed": 1, "objective": "sft"},
            "export": {"formats": ["safetensors"]}, "out_dir": str(tmp / "run")}
    (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "run", str(tmp / "spec.json")], capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stderr
    port = _free_port()
    proc = subprocess.Popen([BIN, "chat", str(tmp / "spec.json"), "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if _get(base + "/health")[0] == 200:
                break
        except OSError:
            pass
        if proc.poll() is not None:
            pytest.fail("mtstudio chat exited: " + proc.stdout.read())
        time.sleep(0.1)
    yield base
    proc.terminate()
    proc.wait(timeout=10)


@needs_bin
def test_replies_are_decoded_text_and_stop_cleanly(served):
    for req in ({"user_input": "hello", "top_k": 1, "max_new_tokens": 30},
                {"user_input": "do you like cats?", "temperature": 0.7, "seed": 2,
                 "history": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi there!"}]},
                {"user_input": "naïve café 😀", "top_k": 1, "max_new_tokens": 12}):
        status, j = _post(served + "/chat", req)
        assert status == 200, j
        assert j["stop_reason"] in ("eos", "turn", "length")
        for marker in ("<|user|>", "<|assistant|>", "<|endoftext|>"):
            assert marker not in j["reply"]


@needs_bin
def test_learned_exchanges_greedily(served):
    hits = 0
    for u, a in PAIRS:
        status, j = _post(served + "/chat", {"user_input": u, "top_k": 1, "max_new_tokens": 30})
        assert status == 200
        hits += j["reply"] == a
    assert hits >= 8, hits
