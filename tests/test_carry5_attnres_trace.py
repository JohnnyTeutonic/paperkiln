"""tools/carry5_attnres_trace.py, the depth-attention trace of arm X5 (PREREGISTRATION.md amendment 5, A5.7).

Without a binary: the pooled statistics on hand-made weights (answer minus other assistant positions, the
normalised entropy, the embedding and early-source headline, the by-distance rows).
With MTSTUDIO_BIN (or ./build/mtstudio): tiny llama_attnres models (3 layers, d 32) on rendered Carry-5
conversations. At lr 0 the pseudo-queries stay zero, so every mixture must be exactly uniform (weights 1/S,
normalised entropy 1, answer minus other 0); after 80 steps at lr 3e-3 the routing must have moved. The tool's
own check (its weights reproduce the interpreter's mixture) runs on every node of both."""
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bpe  # noqa: E402
import carry5_attnres_trace as tr  # noqa: E402
from test_carry5_depth import conversations, context_items, subset  # noqa: E402
from test_mtstudio_chat import BIN  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
TOOL = [sys.executable, str(ROOT / "tools" / "carry5_attnres_trace.py")]


def test_statistics_on_hand_made_weights():
    acc = tr.Acc()
    a1 = np.array([[1.0], [1.0], [1.0]])                       # L0.attn: one source
    a2 = np.array([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])         # L0.ffn: S 2
    a4 = np.array([[0.25] * 4, [0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])  # head: S 4
    rend = {"answer_pos": [1], "other_pos": [0, 2], "prompt_ids": [0, 0], "continuation_ids": [0, 0]}
    acc.add({"id": "a", "distance": 2}, {"L0.attn": a1, "L0.ffn": a2, "head": a4}, rend)
    s = tr.summarise(acc.d)
    m = s["mixtures"]
    assert m["L0.attn"]["S"] == 1 and m["L0.attn"]["entropy_norm"]["all"] is None
    assert np.allclose(m["L0.ffn"]["answer_minus_other"], [0.9 - 0.35, 0.1 - 0.65])
    h = lambda p: -sum(x * math.log(x) for x in p) / math.log(len(p))  # noqa: E731
    assert abs(m["head"]["entropy_norm"]["answer"] - h([0.7, 0.1, 0.1, 0.1])) < 1e-12
    assert abs(m["head"]["entropy_norm"]["all"] - (1 + h([0.7, 0.1, 0.1, 0.1]) + h([0.1, 0.1, 0.1, 0.7])) / 3) < 1e-12
    hl = s["headline"]
    assert abs(hl["embedding_answer_minus_other"] - ((0.9 - 0.35) + (0.7 - 0.175)) / 2) < 1e-12
    early_head = sum(a4[1, :3]) - sum(a4[[0, 2], :3].mean(axis=0))
    assert abs(hl["early_answer_minus_other"] - early_head) < 1e-12  # only the head sees more than 3 sources
    assert hl["mixtures"] == 2 and s["by_distance"]["2"]["answer_positions"] == 1
    assert s["positions"] == {"answer": 1, "other": 2, "all": 3}


def attnres_model(tmp, tok, steps, lr):
    bpe.write_gguf(str(tmp / "v.gguf"), tok)
    (tmp / "c.txt").write_text(conversations(context_items()) * 2, encoding="utf-8")
    spec = {"name": "ar_tiny", "arch": {"preset": "llama-tiny", "rope_heads": "all",
                                        "custom": {"d": 32, "heads": 2, "layers": 3, "d_ff": 64,
                                                   "attn_res": {"form": "full", "scale": "sources"}}},
            "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 4096, "corpus": str(tmp / "c.txt"), "T": 192,
                     "max_tokens": 0},
            "train": {"steps": steps, "lr": lr, "batch": 2, "accum": 1, "eval_every": steps, "eval_windows": 2,
                      "checkpoint_every": steps, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
            "export": {"formats": ["gguf"]}, "out_dir": str(tmp / "pt")}
    p = tmp / "ar_tiny.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return tmp / "pt" / "ar_tiny.gguf"


@pytest.fixture(scope="module")
def tok():
    return bpe.train(conversations(context_items()[::4]), 1200)


@needs_bin
@pytest.mark.parametrize("steps,lr,uniform", [(1, 0.0, True), (80, 3e-3, False)])
def test_tool_on_tiny_models(tmp_path, tok, steps, lr, uniform):
    g = attnres_model(tmp_path, tok, steps, lr)
    items, its = subset(tmp_path, every=27)
    out = tmp_path / "trace"
    p = subprocess.run(TOOL + ["run", "--gguf", str(g), "--out", str(out), "--set", str(items)],
                       capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = json.loads((out / "attnres_trace.json").read_text())
    s = res["summary"]
    assert s["items"] == len(its) and not (out / "state.json").exists()
    m = s["mixtures"]
    assert set(m) == {f"L{i}.{k}" for i in range(3) for k in ("attn", "ffn")} | {"head"}
    assert [m[f"L{i}.{k}"]["S"] for i in range(3) for k in ("attn", "ffn")] == [1, 2, 3, 4, 5, 6] and m["head"]["S"] == 7
    for r in m.values():
        assert abs(sum(r["answer_mean"]) - 1) < 1e-5 and abs(sum(r["other_mean"]) - 1) < 1e-5
    ents = [r["entropy_norm"]["all"] for r in m.values() if r["S"] >= 2]
    if uniform:  # zero queries: softmax of zeros, exactly 1/S
        for r in m.values():
            assert np.allclose(r["answer_mean"], 1 / r["S"], atol=1e-7)
        assert all(abs(e - 1) < 1e-6 for e in ents) and s["headline"]["mixtures_near_uniform"] == 6
        assert abs(s["headline"]["embedding_answer_minus_other"]) < 1e-7
    else:
        assert min(ents) < 1 - 1e-4
