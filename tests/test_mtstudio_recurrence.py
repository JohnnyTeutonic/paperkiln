"""mtstudio's recurrence lane (arch.custom.attention "recurrence"; spec/IR_SPEC.md 6.10), end to end on CPU.

1. With the default body (section 6.4's linear attention as a recurrence) the lane trains the kimi model:
   the same parameters, the same first loss bit for bit, then losses within rounding drift (the derived
   backward and kimi's hand-written one round differently).
2. A body no hand-written op computes (a decayed linear attention, arch.custom.recurrence) trains, exports
   a GGUF the validator accepts, with the digest key, and the numpy reference on the exported document
   gives mtstudio's own log-probabilities (mtstudio score) within 1e-5.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise. Everything is written to tmp_path.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import WORDS, corpus  # noqa: E402
import ir_conformance  # noqa: E402
import ir_conformance_mtstudio  # noqa: E402
import ir_numpy  # noqa: E402
import ir_recurrence  # noqa: E402
import ir_validate  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")


def run(tmp, name, custom, steps=20):
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    spec = {"name": name, "arch": {"preset": "gpt2-nano", "custom": dict({"d": 16, "layers": 2, "heads": 2}, **custom)},
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": {"steps": steps, "lr": 0.003, "eval_every": 1000, "checkpoint_every": 1000, "batch": 2, "seed": 3,
                      "gradmap_every": 0},
            "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(tmp / name)}
    sp = tmp / f"{name}.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "run", str(sp)], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = [json.loads(x) for x in (tmp / name / "events.jsonl").read_text().splitlines() if x.strip()]
    return spec, sp, ev


def losses(ev):
    return [e["loss"] for e in ev if e.get("event") == "step"]


def f32_weights(path):
    """{name: f32 array in numpy order} of an all-F32 GGUF (the gpt2-family export)."""
    meta, tensors, data_start, raw, _ = ir_validate.read_gguf(path)
    W = {}
    for name, dims, typ, off in tensors:
        assert typ == 0, name
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype="<f4", count=n, offset=data_start + off).reshape(list(reversed(dims))).copy()
    return W


@needs_bin
def test_default_body_trains_the_kimi_model(tmp_path):
    _, _, ek = run(tmp_path, "kimi", {"attention": "kimi"})
    _, _, er = run(tmp_path, "rec", {"attention": "recurrence"})
    lk, lr = losses(ek), losses(er)
    assert len(lk) == len(lr) == 20
    assert lk[0] == lr[0]  # the same parameters and a bit-identical forward
    drift = max(abs(a - b) / abs(a) for a, b in zip(lk, lr))
    assert drift < 1e-3, drift
    doc_k = json.loads((tmp_path / "kimi" / "kimi.ir.json").read_text(encoding="utf-8"))
    doc_r = json.loads((tmp_path / "rec" / "rec.ir.json").read_text(encoding="utf-8"))
    # The exported recurrence is spec/tools/ir_recurrence.py's rewrite of the kimi node, slot name aside.
    want = [n for n in ir_recurrence.rewrite(doc_k)["block"]["nodes"] if n["op"] == "recurrence"][0]
    got = [n for n in doc_r["block"]["nodes"] if n["op"] == "recurrence"][0]
    assert dict(got, state="lin") == want


@needs_bin
def test_a_new_body_trains_exports_and_matches_the_reference(tmp_path):
    body = ir_conformance_mtstudio.DECAYED_LINEAR_ATTENTION
    spec, sp, ev = run(tmp_path, "decay", {"attention": "recurrence", "recurrence": body}, steps=30)
    ls = losses(ev)
    assert np.mean(ls[-5:]) < np.mean(ls[:5])
    out = tmp_path / "decay"
    gguf = [p for p in out.glob("*.gguf")][0]
    assert ir_validate.file_errors(gguf) == []
    meta, *_ = ir_validate.read_gguf(gguf)
    assert meta["paperkiln.ir_sha256"][1] == ir_validate.digest(meta["paperkiln.ir"][1])
    doc = json.loads(meta["paperkiln.ir"][1])
    node = [n for n in doc["block"]["nodes"] if n["op"] == "recurrence"][0]
    assert node["update"] == body["update"] and node["readout"] == body["readout"]
    W = f32_weights(gguf)
    rng = np.random.default_rng(1)
    prompt = [int(x) for x in rng.integers(1, len(WORDS) + 1, 6)]
    cont = [int(x) for x in rng.integers(1, len(WORDS) + 1, 5)]
    req, res = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    req.write_text(json.dumps({"prompt_ids": prompt, "continuation_ids": cont}) + "\n", encoding="utf-8")
    r = subprocess.run([BIN, "score", str(sp), "--in", str(req), "--out", str(res)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(res.read_text().splitlines()[0])["logprobs"]
    lg = ir_numpy.forward(doc, W, prompt + cont[:-1])
    lp, _ = ir_conformance.logprobs_of(lg[len(prompt) - 1:], cont, [0])
    worst = float(np.max(np.abs(np.array(got) - np.array(lp))))
    assert worst < 1e-5, worst
