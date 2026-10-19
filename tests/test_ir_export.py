"""The paperkiln IR in mtstudio's GGUF exports (microtorch/ir.hpp).

Tiny dense and MoE llama models train for two steps and export F32 and Q8_0
GGUFs. Checks (tools/ir_check.py): both keys present and after every other
key, every tensor the IR names exists in each file, the sidecar equals the
embedded document; the dense IR is the hand-written v0 worked example
(ir/chat_sftfinal.ir.json) with this model's dimensions; the MoE block has
one moe_ffn node in place of the dense FFN's five. The gpt2 family exports
in llama.cpp's gpt2 layout (shapes, metadata and bytes against the
safetensors) with a layernorm / split / gelu IR; flex exports when it has
the GPT-2 flavour, and other flavours are skipped. Needs MTSTUDIO_BIN (or
./build/mtstudio); skipped otherwise.
"""
import copy
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import MOE, WORDS, corpus, events, run, setup  # noqa: E402
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


def gpt2_setup(tmp, name, custom, formats=("safetensors", "gguf")):
    """A tiny gpt2-family spec (2 layers: the ParityLM; other depths and flavours: flex)."""
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    spec = {"name": name, "arch": {"preset": "gpt2-nano", "custom": dict({"d": 32, "layers": 2, "heads": 2}, **custom)},
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": {"steps": 2, "lr": 0.003, "eval_every": 10, "checkpoint_every": 10, "batch": 2, "seed": 3,
                      "gradmap_every": 0},
            "export": {"formats": list(formats)}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def safetensors(path):
    """{name: (shape, [float])} of an F32 safetensors file (standard library only)."""
    b = Path(path).read_bytes()
    n = struct.unpack_from("<Q", b, 0)[0]
    head = json.loads(b[8:8 + n])
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32"
        s, e = v["data_offsets"]
        out[k] = (v["shape"], list(struct.unpack_from(f"<{(e - s) // 4}f", b, 8 + n + s)))
    return out


def gguf_tensor(path, name):
    """(ne dims, [float]) of an F32 tensor of a GGUF."""
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
    _, dims, typ, off = next(t for t in tensors if t[0] == name)
    assert typ == 0, (name, typ)
    count = 1
    for x in dims:
        count *= x
    return dims, list(struct.unpack_from(f"<{count}f", raw, data_start + off))


GPT2_OPS = ["layernorm", "linear", "split", "attention", "linear", "add",
            "layernorm", "linear", "gelu", "linear", "add"]


@needs_bin
def test_gpt2_gguf_and_ir(tmp_path):
    spec = gpt2_setup(tmp_path, "g", {}, formats=("safetensors", "gguf", "gguf_q8_0"))
    run(spec)
    ev = events(spec)
    ex = [e for e in ev if e["event"] == "export" and e["format"] == "gguf"]
    assert len(ex) == 1 and ex[0]["arch"] == "gpt2", ex
    # Quantised gpt2 variants are refused by name, never silently dropped.
    assert any(e["event"] == "export_skipped" and e["format"] == "gguf_q8_0" for e in ev)
    path = ex[0]["path"]
    assert ir_check.check(path) == []
    meta, tensors, *_ = ir_check.read_gguf(path)
    assert meta["general.architecture"][1] == "gpt2"
    want = {"gpt2.context_length": 16, "gpt2.embedding_length": 32, "gpt2.block_count": 2,
            "gpt2.feed_forward_length": 128, "gpt2.attention.head_count": 2, "gpt2.vocab_size": 15}
    assert {k: meta[k][1] for k in want} == want
    assert abs(meta["gpt2.attention.layer_norm_epsilon"][1] - 1e-5) < 1e-12
    # llama.cpp's gpt2 shapes (ne order) for every tensor.
    dims = {t[0]: t[1] for t in tensors}
    d, ff, V, T = 32, 128, 15, 16
    expect = {"token_embd.weight": [d, V], "position_embd.weight": [d, T], "output_norm.weight": [d],
              "output_norm.bias": [d], "output.weight": [d, V]}
    for i in range(2):
        b = f"blk.{i}."
        expect.update({b + "attn_norm.weight": [d], b + "attn_norm.bias": [d], b + "attn_qkv.weight": [d, 3 * d],
                       b + "attn_qkv.bias": [3 * d], b + "attn_output.weight": [d, d], b + "attn_output.bias": [d],
                       b + "ffn_norm.weight": [d], b + "ffn_norm.bias": [d], b + "ffn_up.weight": [d, ff],
                       b + "ffn_up.bias": [ff], b + "ffn_down.weight": [ff, d], b + "ffn_down.bias": [d]})
    assert dims == expect
    # The bytes: projections are the trained [in, out] matrices transposed,
    # embeddings and vectors copied.
    st = safetensors(Path(path).with_suffix(".safetensors"))
    (i_, o_), w = st["attn_1.c_attn.weight"]
    _, g = gguf_tensor(path, "blk.1.attn_qkv.weight")
    assert all(g[o * i_ + i] == w[i * o_ + o] for i in range(i_) for o in range(o_))
    for gname, sname in (("token_embd.weight", "wte.weight"), ("position_embd.weight", "wpe.weight"),
                         ("blk.0.ffn_up.bias", "mlp_0.c_fc.bias"), ("output_norm.bias", "ln_f.bias")):
        assert gguf_tensor(path, gname)[1] == st[sname][1], gname
    (i_, o_), w = st["head.weight"]
    _, g = gguf_tensor(path, "output.weight")
    assert all(g[o * i_ + i] == w[i * o_ + o] for i in range(i_) for o in range(o_))
    # The IR: learned positions in embed, the gpt2 block, an untied head.
    doc = embedded(path)
    assert doc["family_hint"] == "gpt2"
    assert doc["model"] == {"vocab": V, "d": d, "n_layers": 2, "ctx": T, "dtype": "f32"}
    # ember loads non-llama files from the IR alone: the model and kv-state
    # fields must be present and agree with the GGUF's own metadata.
    g = {k.split(".", 1)[1]: v[1] for k, v in meta.items() if k.startswith("gpt2.")}
    assert (doc["model"]["d"], doc["model"]["vocab"], doc["model"]["ctx"], doc["model"]["n_layers"]) == (
        g["embedding_length"], g["vocab_size"], g["context_length"], g["block_count"])
    assert doc["block"]["repeat"] == g["block_count"]
    kv = [s for s in doc["block"]["state"] if s["kind"] == "kv"]
    assert len(kv) == 1
    heads = g["attention.head_count"]
    assert kv[0] == {"name": "kv", "kind": "kv", "heads": heads, "kv_heads": heads,
                     "head_dim": g["embedding_length"] // heads, "max_len": g["context_length"],
                     "window": 0, "sinks": 0, "dtype": "f32"}
    assert [n["op"] for n in doc["embed"]["nodes"]] == ["embedding", "pos_embedding"]
    assert doc["embed"]["nodes"][1] == {"op": "pos_embedding", "in": ["x", "pos"],
                                        "weight": "position_embd.weight", "out": "x"}
    nodes = doc["block"]["nodes"]
    assert [n["op"] for n in nodes] == GPT2_OPS
    assert nodes[0] == {"op": "layernorm", "in": ["x"], "weight": "blk.{i}.attn_norm.weight",
                        "bias": "blk.{i}.attn_norm.bias", "eps": 1e-05, "out": "h"}
    assert nodes[1]["weight"] == "blk.{i}.attn_qkv.weight" and nodes[1]["bias"] == "blk.{i}.attn_qkv.bias"
    assert nodes[2] == {"op": "split", "in": ["qkv"], "sizes": [d, d, d], "out": ["q", "k", "v"]}
    assert "rope" not in [n["op"] for n in nodes]
    assert (nodes[3]["heads"], nodes[3]["head_dim"], nodes[3]["causal"]) == (2, 16, True)
    assert nodes[8] == {"op": "gelu", "in": ["u"], "approx": "tanh", "out": "u"}
    assert doc["block"]["state"][0]["max_len"] == T
    head = doc["head"]["nodes"]
    assert [n["op"] for n in head] == ["layernorm", "linear"]
    assert head[1]["weight"] == "output.weight" and head[1]["bias"] is None and "tied_to" not in head[1]


@needs_bin
def test_gpt2_flavoured_flex_exports_and_others_skip(tmp_path):
    # Depth 3 promotes to flex; with its GPT-2 flavour it exports, d_ff included.
    spec = gpt2_setup(tmp_path / "deep", "f", {"layers": 3, "d_ff": 48})
    run(spec)
    ex = [e for e in events(spec) if e["event"] == "export" and e["format"] == "gguf"]
    assert len(ex) == 1
    assert ir_check.check(ex[0]["path"]) == []
    meta, tensors, *_ = ir_check.read_gguf(ex[0]["path"])
    assert (meta["gpt2.block_count"][1], meta["gpt2.feed_forward_length"][1]) == (3, 48)
    assert embedded(ex[0]["path"])["block"]["repeat"] == 3
    # Another flavour (rmsnorm) or attention kind (kimi) is skipped, with no IR.
    for sub, custom in (("rms", {"layers": 3, "norm": "rmsnorm"}), ("kimi", {"attention": "kimi"})):
        spec = gpt2_setup(tmp_path / sub, "s", custom)
        run(spec)
        ev = events(spec)
        assert not [e for e in ev if e["event"] == "export" and e["format"] == "gguf"]
        assert any(e["event"] == "export_skipped" and e["format"] == "gguf" for e in ev)
        assert not (tmp_path / sub / "s" / "s.ir.json").exists()


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
