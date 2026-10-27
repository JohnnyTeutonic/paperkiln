"""`mtstudio score --batch-tokens` against the unbatched scorer on a trained toy model (LOCAL_LLM_WORK E5.2): the
same records (keys, fields, order, over_ctx, argmax, prompt_tokens), log-probs and sums within IR_SPEC 4.3's CPU
ROUNDING tolerance (1e-5 max(1, |ref|)), and the option's refusals. Bit for bit is checked where the host GEMM
gives every row its own summation order (tests/test_score_batch.cpp leg 1); here the batches are large enough
that it need not. Needs MTSTUDIO_BIN."""
import json
import os
import random
import subprocess

import pytest

BIN = os.environ.get("MTSTUDIO_BIN")
needs_bin = pytest.mark.skipif(not BIN, reason="set MTSTUDIO_BIN to a built mtstudio")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import bpe
    t = tmp_path_factory.mktemp("sb")
    pairs = [("hello", "Hi there!"), ("what is your name?", "I am a small model."), ("bye", "Goodbye!")]
    text = "".join(f"<|user|>{u}<|assistant|>{a}<|endoftext|>\n" for u, a in pairs) * 20
    tok = bpe.train(text, 400)
    bpe.write_gguf(str(t / "v.gguf"), tok)
    ex = [{"prompt": f"<|user|>{u}<|assistant|>", "completion": a} for u, a in pairs] * 7
    (t / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    spec = {"name": "m", "arch": {"preset": "llama-tiny"},
            "data": {"vocab": str(t / "v.gguf"), "vocab_cap": 4096, "T": 64, "sft_examples": str(t / "ex.jsonl")},
            "train": {"steps": 40, "lr": 0.003, "batch": 4, "eval_every": 100, "checkpoint_every": 1000,
                      "gradmap_every": 0, "seed": 1, "objective": "sft"},
            "export": {"formats": ["safetensors"]}, "out_dir": str(t / "run")}
    sp = t / "spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "run", str(sp)], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    pool = sorted(set(tok.encode(text)))  # ids the vocabulary holds
    rng = random.Random(3)
    reqs = []
    for i in range(40):
        p = [rng.choice(pool) for _ in range(rng.randrange(1, 50))]
        c = [rng.choice(pool) for _ in range(rng.randrange(1, 8))]
        reqs.append({"key": f"k{i}", "prompt_ids": p, "continuation_ids": c})
    reqs.insert(7, {"key": "long", "prompt_ids": [pool[1]] * 70, "continuation_ids": [pool[2]]})  # past T + 1: over_ctx
    reqs.append({"key": "text", "prompt": "<|user|>hello<|assistant|>", "continuation": "Hi there!"})
    inp = t / "in.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in reqs), encoding="utf-8")
    return t, sp, inp


def score(t, sp, inp, name, *extra):
    out = t / f"{name}.jsonl"
    r = subprocess.run([BIN, "score", str(sp), "--in", str(inp), "--out", str(out), *extra], capture_output=True,
                       text=True, timeout=600)
    return r, ([json.loads(x) for x in open(out, encoding="utf-8")] if r.returncode == 0 else None)


@needs_bin
@pytest.mark.parametrize("bt", ["1", "60", "200", "100000"])
def test_batched_records_equal_the_unbatched(model, bt):
    t, sp, inp = model
    _, ref = score(t, sp, inp, "ref")
    r, got = score(t, sp, inp, f"b{bt}", "--batch-tokens", bt)
    assert r.returncode == 0, r.stderr
    assert "packed batches" in r.stderr
    assert len(got) == len(ref)
    for a, b in zip(ref, got):
        assert set(a) == set(b)
        for k in a:
            if k in ("logprobs", "sum"):
                continue
            assert a[k] == b[k], (k, a, b)
        if a["over_ctx"]:
            continue
        m = max(1.0, max(abs(x) for x in a["logprobs"]))
        assert max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"])) <= 1e-5 * m
        assert abs(a["sum"] - b["sum"]) <= 1e-5 * max(1.0, abs(a["sum"]))
    assert [r["key"] for r in got if "key" in r] == [r["key"] for r in ref if "key" in r]
    assert next(r for r in got if r.get("key") == "long")["over_ctx"]


@needs_bin
def test_no_window_equals_window_on_the_host(model):
    t, sp, inp = model
    _, a = score(t, sp, inp, "w", "--batch-tokens", "200")
    _, b = score(t, sp, inp, "nw", "--batch-tokens", "200", "--no-window")
    assert a == b  # the step window is a device matter: the host computes the same bits


@needs_bin
@pytest.mark.parametrize("extra,msg", [(["--batch-tokens", "0"], "positive"),
                                       (["--no-window"], "needs --batch-tokens"),
                                       (["--batch-tokens", "64", "--kv-cache"], "not with --kv-cache")])
def test_refusals(model, extra, msg):
    t, sp, inp = model
    r, _ = score(t, sp, inp, "bad", *extra)
    assert r.returncode != 0 and msg in r.stderr
