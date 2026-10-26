"""mtstudio's train.fused_head (the fused LM head + cross-entropy, 26 Oct 2026) end to end on CPU.

Off by default, and off is the step as before: with MTSTUDIO_BASE_BIN set to an mtstudio built at the
commit before the option (c6365b88), a pretraining run (accum 2) and an SFT run train byte-identically
under both binaries (every step's loss and grad norm, every eval, the exported safetensors and GGUF);
those two tests are skipped without it. On: the model event and result.json record it, the first step's
loss is the unfused loss bit for bit (the op's host loss is), the run tracks the unfused one to float
rounding, and a run at another chunk size is the same run bit for bit. Refusals: another family, a
chunk without the option. Needs MTSTUDIO_BIN (or ./build/mtstudio). Everything goes to tmp_path.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, DIALOGUE, EOS, tokenize_like_mtstudio, write_vocab_gguf  # noqa: E402
import test_mtstudio_sft as sft_t  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
_base = os.environ.get("MTSTUDIO_BASE_BIN")
BASE = _base if _base and Path(_base).is_file() else None
needs_base = pytest.mark.skipif(BIN is None or BASE is None,
                                reason="set MTSTUDIO_BIN and MTSTUDIO_BASE_BIN (an mtstudio built at c6365b88)")


def lm_spec(tmp, tag="lm", steps=30, accum=2, **train):
    """The spec (always named lm, so the exports are comparable) with its own out_dir per tag."""
    corpus = tmp / "corpus.txt"
    if not corpus.exists():
        text = "".join(f"user: {u} assistant: {a}{EOS}\n" for _ in range(200) for u, a in DIALOGUE)
        corpus.write_text(text)
        counts = {}
        for t in tokenize_like_mtstudio(text):
            counts[t] = counts.get(t, 0) + 1
        words = sorted((w for w in counts if w != EOS), key=lambda w: -counts[w])
        tokens = ["<unk>"] + words + [EOS]
        write_vocab_gguf(tmp / "vocab.gguf", tokens, len(tokens) - 1)
    spec = {"name": "lm", "arch": {"preset": "llama-tiny", "custom": {"d": 32, "layers": 2, "heads": 2}},
            "data": {"corpus": str(corpus), "vocab": str(tmp / "vocab.gguf"), "vocab_cap": 0, "T": 32},
            "train": dict({"steps": steps, "batch": 4, "accum": accum, "lr": 1e-2, "eval_every": 10,
                           "checkpoint_every": 1000, "gradmap_every": 0, "seed": 3}, **train),
            "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(tmp / tag)}
    p = tmp / f"{tag}.json"
    p.write_text(json.dumps(spec))
    return p


def sft_spec(tmp, tag="sft", steps=40, **train):
    """test_mtstudio_sft's spec (named sft) with its own out_dir per tag."""
    ex = [sft_t.qa(a, b) for a, b in sft_t.FACTS] * 3 + [sft_t.qa(0, 0)]
    d = tmp / f"{tag}_data"
    d.mkdir(exist_ok=True)
    base = json.loads(sft_t.setup(d, ex, steps=steps, extra_train=dict({"accum": 2, "eval_every": 20}, **train)).read_text())
    base["out_dir"] = str(tmp / tag)
    p = tmp / f"{tag}.json"
    p.write_text(json.dumps(base))
    return p


def with_out(spec, tag):
    """The same spec in another out_dir (a second binary must not resume the first's run)."""
    s = json.loads(spec.read_text())
    s["out_dir"] = s["out_dir"] + "_" + tag
    p = spec.with_name(spec.stem + "_" + tag + ".json")
    p.write_text(json.dumps(s))
    return p


def run(binary, spec):
    r = subprocess.run([binary, "run", str(spec)], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    out = Path(json.loads(spec.read_text())["out_dir"])
    ev = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines() if x.strip()]
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())
             if f.suffix in (".safetensors", ".gguf")}
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    return {"steps": [(e["step"], e["loss"], e.get("grad_norm")) for e in ev if e["event"] == "step"],
            "evals": [(e["step"], e["val_loss"]) for e in ev if e["event"] == "eval"],
            "model": next(e for e in ev if e["event"] == "model"), "files": files, "result": res}


def same_run(a, b):
    assert a["steps"] and a["steps"] == b["steps"]
    assert a["evals"] == b["evals"]
    assert a["files"] and a["files"] == b["files"]


def tracks(f, u, tol):
    assert f["model"].get("fused_head") is True and f["result"].get("fused_head") is True
    assert len(f["steps"]) == len(u["steps"])
    assert f["steps"][0][1] == u["steps"][0][1]  # the first step's loss, bit for bit
    worst = max(abs(x[1] - y[1]) / abs(y[1]) for x, y in zip(f["steps"], u["steps"]))
    assert worst <= tol, worst
    assert abs(f["evals"][-1][1] - u["evals"][-1][1]) / abs(u["evals"][-1][1]) <= tol


@needs_base
def test_off_is_the_step_before_the_option_lm(tmp_path):
    spec = lm_spec(tmp_path)
    new = run(BIN, spec)
    old = run(BASE, with_out(spec, "base"))
    same_run(new, old)
    assert "fused_head" not in new["model"] and "fused_head" not in new["result"]


@needs_base
def test_off_is_the_step_before_the_option_sft(tmp_path):
    spec = sft_spec(tmp_path)
    same_run(run(BIN, spec), run(BASE, with_out(spec, "base")))


@needs_bin
def test_off_explicit_is_the_default(tmp_path):
    a = run(BIN, lm_spec(tmp_path, "dflt"))
    b = run(BIN, lm_spec(tmp_path, "off", fused_head=False))
    same_run(a, b)
    assert "fused_head" not in a["model"] and "fused_head" not in b["model"]


@needs_bin
def test_fused_lm_tracks_unfused_and_ignores_chunk_size(tmp_path):
    u = run(BIN, lm_spec(tmp_path, "unfused"))
    f = run(BIN, lm_spec(tmp_path, "fused", fused_head=True))
    tracks(f, u, 2e-3)
    c = run(BIN, lm_spec(tmp_path, "fused_c3", fused_head=True, fused_head_chunk=3))
    assert c["model"].get("fused_head_chunk") == 3
    same_run(f, c)  # every chunk size is the same op, bit for bit


@needs_bin
def test_fused_sft_tracks_unfused(tmp_path):
    u = run(BIN, sft_spec(tmp_path, "unfused"))
    f = run(BIN, sft_spec(tmp_path, "fused", fused_head=True))
    tracks(f, u, 2e-3)
    same_run(f, run(BIN, sft_spec(tmp_path, "fused_c5", fused_head=True, fused_head_chunk=5)))


@needs_bin
def test_refusals(tmp_path):
    def err(spec):
        r = subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=600)
        assert r.returncode != 0
        return r.stdout + r.stderr
    p = lm_spec(tmp_path, "chunk_only", fused_head_chunk=8)
    assert "train.fused_head_chunk needs train.fused_head true" in err(p)
    s = json.loads(lm_spec(tmp_path, "gpt2", fused_head=True).read_text())
    s["arch"] = {"preset": "gpt2-nano", "custom": {"d": 32, "layers": 2, "heads": 2}}
    p = tmp_path / "gpt2_fused.json"
    p.write_text(json.dumps(s))
    assert "train.fused_head is wired for the llama family only" in err(p)
