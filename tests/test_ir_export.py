"""The paperkiln IR in mtstudio's GGUF exports (microtorch/ir.hpp).

Tiny dense and MoE llama models train for two steps and export F32 and Q8_0
GGUFs. Checks (tools/ir_check.py): both keys present and after every other
key, every tensor the IR names exists in each file, the sidecar equals the
embedded document; the dense IR is the hand-written v0 worked example
(ir/chat_sftfinal.ir.json) with this model's dimensions; the MoE block has
one moe_ffn node in place of the dense FFN's five. Needs MTSTUDIO_BIN (or
./build/mtstudio); skipped otherwise.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from test_mtstudio_chat import BIN  # noqa: E402
from test_mtstudio_moe import MOE, events, run, setup  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
V0 = json.loads((ROOT / "ir" / "chat_sftfinal.ir.json").read_text(encoding="utf-8"))


def exports(spec):
    ex = [e for e in events(spec) if e["event"] == "export" and e["format"].startswith("gguf")]
    assert len(ex) == 2, ex
    return ex


def embedded(path):
    meta, *_ = ir_check.read_gguf(path)
    return json.loads(meta["paperkiln.ir"][1])


def v0_scaled(vocab, d, layers, heads, ctx):
    """v0 with another model's dimensions: what a dense llama-family export must say."""
    ref = copy.deepcopy(V0)
    hd = d // heads
    ref["model"].update(vocab=vocab, d=d, n_layers=layers, ctx=ctx)
    ref["block"]["repeat"] = layers
    ref["block"]["state"][0].update(heads=heads, kv_heads=heads, head_dim=hd, max_len=ctx)
    for n in ref["block"]["nodes"]:
        if n["op"] in ("rope", "attention"):
            n.update(heads=heads, head_dim=hd)
        if n["op"] == "attention":
            n.update(kv_heads=heads, scale=1.0 / hd ** 0.5)
    return ref


@needs_bin
def test_dense_ir_is_v0_at_this_size(tmp_path):
    spec = setup(tmp_path, "dense", 2, formats=("safetensors", "gguf", "gguf_q8_0"))
    run(spec)
    for e in exports(spec):
        assert ir_check.check(e["path"]) == []
        assert Path(e["ir"]).is_file()
    doc = embedded(exports(spec)[0]["path"])
    assert doc["ir_version"] == 0
    # A word-level vocabulary: only the tokenizer section is the model's own.
    assert doc["tokenizer"] == {"type": "word", "source": "gguf", "specials": {"unk": 0}, "bos": None}
    want = v0_scaled(15, 32, 2, 2, 16)
    want["tokenizer"] = doc["tokenizer"]
    assert ir_check.strip(doc) == ir_check.strip(want)


@needs_bin
def test_moe_ir_names_the_expert_tensors(tmp_path):
    spec = setup(tmp_path, "moe", 2, moe=MOE, formats=("safetensors", "gguf", "gguf_q8_0"))
    run(spec)
    for e in exports(spec):
        assert ir_check.check(e["path"]) == []
    doc = embedded(exports(spec)[0]["path"])
    ops = [n["op"] for n in doc["block"]["nodes"]]
    dense = [n["op"] for n in V0["block"]["nodes"]]
    assert ops == dense[:9] + ["moe_ffn", "add"]
    moe = doc["block"]["nodes"][9]
    assert moe["in"] == ["h2"] and moe["out"] == "f"
    assert (moe["n_experts"], moe["top_k"], moe["gate_norm"], moe["activation"]) == (4, 2, "renorm", "silu")
    assert moe["router"] == "blk.{i}.ffn_gate_inp.weight"
    assert moe["experts"] == {k: f"blk.{{i}}.ffn_{k}_exps.weight" for k in ("gate", "up", "down")}


@needs_bin
def test_checker_catches_a_wrong_ir(tmp_path):
    spec = setup(tmp_path, "d", 2, formats=("safetensors", "gguf", "gguf_q8_0"))
    run(spec)
    path = exports(spec)[0]["path"]
    bad = v0_scaled(15, 32, 2, 2, 16)
    bad["block"]["nodes"][0]["eps"] = 1e-6
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    assert any("block" in e for e in ir_check.check(path, like=tmp_path / "bad.json"))
    # A sidecar that drifted from the embedded IR is reported too.
    side = Path(exports(spec)[0]["ir"])
    doc = json.loads(side.read_text(encoding="utf-8"))
    doc["block"]["nodes"][1]["weight"] = "blk.{i}.attn_qq.weight"
    side.write_text(json.dumps(doc), encoding="utf-8")
    assert any("sidecar" in e for e in ir_check.check(path))
