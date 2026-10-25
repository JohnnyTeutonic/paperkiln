"""The paperkiln IR in mtstudio's GGUF exports (microtorch/ir.hpp).

Tiny dense and MoE llama models train for two steps and export F32 and Q8_0
GGUFs. Checks (spec/tools/ir_check.py): both keys present and after every other
key, every tensor the IR names exists in each file, the sidecar equals the
embedded document; the dense IR is the hand-written v0 worked example
(spec/examples/chat_sftfinal.ir.json) with this model's dimensions; the MoE block has
one moe_ffn node in place of the dense FFN's five. The gpt2 family exports
in llama.cpp's gpt2 layout (shapes, metadata and bytes against the
safetensors) with a layernorm / split / gelu IR, as does flex with the GPT-2
flavour. Sliding-window attention (gpt2 and flex) exports the same gpt2
tensors with family_hint "gpt2_swa" and the window and sinks on the
attention node and kv state. The other families (ir.hpp "Families": kimi,
srd, attnres, flex highway / plain / other flavours) export F32 with their
IR: every IR weight is a tensor and every tensor an IR weight, the model
fields agree with the GGUF metadata, and the state slots (kv, recurrent)
are declared in full. Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped
otherwise.
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
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import MOE, WORDS, corpus, events, run, setup  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
V0 = json.loads((ROOT / "spec" / "examples" / "chat_sftfinal.ir.json").read_text(encoding="utf-8"))


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
    assert doc["ir_version"] == 1  # spec/IR_SPEC.md section 2.3 (the worked example is a version 0 document)
    # A word-level vocabulary: only the tokenizer section is the model's own.
    assert doc["tokenizer"] == {"type": "word", "source": "gguf", "specials": {"unk": 0}, "bos": None}
    want = v0_scaled(15, 32, 2, 2, 16)
    want["tokenizer"] = doc["tokenizer"]
    want["ir_version"] = 1  # a version 0 document is a version 1 document
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


def gpt2_setup(tmp, name, custom, formats=("safetensors", "gguf"), preset="gpt2-nano"):
    """A tiny gpt2-family spec (2 layers: the ParityLM; other depths and flavours: flex)."""
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    spec = {"name": name, "arch": {"preset": preset, "custom": dict({"d": 32, "layers": 2, "heads": 2}, **custom)},
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


def gguf_weights(path):
    """{name: f32 array, numpy order (reversed ne dims)} of an all-F32 GGUF, as tools/ir_run.py reads it."""
    import numpy as np
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
    W = {}
    for name, dims, typ, off in tensors:
        assert typ == 0, name
        n = 1
        for x in dims:
            n *= x
        W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))
    return W


def test_scale_kernel_rounds_the_product_then_the_sum():
    """ir.hpp op scale: (x * w) + b, each rounded to f32, no fused multiply-add; no bias: x * w."""
    import numpy as np
    import ir_numpy
    # x * w = 1 + 2^-11 + 2^-24 exactly: f32 rounds it to 1 + 2^-11 (a tie, to even), so with
    # b = -1 the rounded order gives 2^-11 while a fused multiply-add gives 2^-11 + 2^-24.
    x = np.array([[1 + 2.0 ** -12, 2.0, -3.0]], np.float32)
    W = {"blk.0.g.weight": np.array([1 + 2.0 ** -12, 0.5, 2.0], np.float32),
         "blk.0.g.bias": np.array([-1.0, 0.25, 0.0], np.float32)}
    node = {"op": "scale", "in": ["x"], "weight": "blk.{i}.g.weight", "bias": "blk.{i}.g.bias", "out": "y"}
    env = {"x": x}
    ir_numpy.run_nodes([node], env, W, i=0)
    assert env["y"].dtype == np.float32
    assert env["y"].tolist() == [[2.0 ** -11, 1.25, -6.0]]
    fused = np.float32(np.float64(x[0, 0]) * np.float64(W["blk.0.g.weight"][0]) - 1.0)
    assert fused == 2.0 ** -11 + 2.0 ** -24
    env = {"x": x}
    ir_numpy.run_nodes([dict(node, bias=None)], env, W, i=0)
    assert env["y"].tolist() == [[1 + 2.0 ** -11, 1.0, -6.0]]
    # The last row alone is the full forward's last row (row-wise).
    xx = np.array([[0.5, -1.5, 3.0], [7.0, 0.125, -2.0]], np.float32)
    env_all, env_last = {"x": xx}, {"x": xx[-1:]}
    ir_numpy.run_nodes([node], env_all, W, i=0)
    ir_numpy.run_nodes([node], env_last, W, i=0)
    assert np.array_equal(env_all["y"][-1:], env_last["y"])


def test_scale_refuses_a_weight_of_the_wrong_width():
    import numpy as np
    import ir_numpy
    W = {"blk.1.g.weight": np.ones(4, np.float32), "blk.1.g.bias": np.zeros(3, np.float32),
         "blk.1.h.weight": np.ones(3, np.float32)}
    x = np.ones((2, 3), np.float32)
    with pytest.raises(ir_numpy.WidthError, match=r"\(rule 37\): scale tensor blk\.1\.g\.weight has length 4, not the width 3"):
        ir_numpy.run_nodes([{"op": "scale", "in": ["x"], "weight": "blk.{i}.g.weight", "bias": None, "out": "y"}],
                           {"x": x}, W, i=1)
    with pytest.raises(ir_numpy.WidthError, match=r"blk\.1\.g\.bias has length 3, not the width 2"):
        ir_numpy.run_nodes([{"op": "scale", "in": ["x"], "weight": "blk.{i}.g.bias", "bias": "blk.{i}.g.bias",
                             "out": "y"}], {"x": x[:, :2]}, W, i=1)


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
def test_gpt2_flavoured_flex_exports(tmp_path):
    # Depth 3 promotes to flex; with its GPT-2 flavour it exports, d_ff included.
    spec = gpt2_setup(tmp_path / "deep", "f", {"layers": 3, "d_ff": 48})
    run(spec)
    ex = [e for e in events(spec) if e["event"] == "export" and e["format"] == "gguf"]
    assert len(ex) == 1
    assert ir_check.check(ex[0]["path"]) == []
    meta, tensors, *_ = ir_check.read_gguf(ex[0]["path"])
    assert (meta["gpt2.block_count"][1], meta["gpt2.feed_forward_length"][1]) == (3, 48)
    assert embedded(ex[0]["path"])["block"]["repeat"] == 3


# ---- the families of microtorch/ir.hpp "Families" ----------------------------

def family_export(tmp, name, custom, arch, hint, formats=("safetensors", "gguf", "gguf_q8_0"), preset="gpt2-nano"):
    """Train a tiny model for two steps; check what every family export must
    satisfy, and return (path, meta, {tensor: ne dims}, IR)."""
    spec = gpt2_setup(tmp, name, custom, formats=formats, preset=preset)
    run(spec)
    ev = events(spec)
    ex = [e for e in ev if e["event"] == "export" and e["format"] == "gguf"]
    assert len(ex) == 1, ev[-5:]
    assert (ex[0]["arch"], ex[0]["family"]) == (arch, hint)
    # F32 only: a quantised variant is refused by name.
    assert any(e["event"] == "export_skipped" and e["format"] == "gguf_q8_0" for e in ev)
    path = ex[0]["path"]
    # Every IR weight name is a tensor; model fields and state slots complete.
    assert ir_check.check(path) == []
    meta, tensors, *_ = ir_check.read_gguf(path)
    dims = {t[0]: t[1] for t in tensors}
    doc = embedded(path)
    assert meta["general.architecture"][1] == arch and doc["family_hint"] == hint
    # And every tensor is one the IR reads: nothing in the file is unexplained.
    assert {n for _, n, _ in ir_check.weight_refs(doc)} == set(dims)
    g = {k.split(".", 1)[1]: v[1] for k, v in meta.items() if k.startswith(arch + ".")}
    m = doc["model"]
    assert (m["d"], m["vocab"], m["ctx"], m["n_layers"]) == (
        g["embedding_length"], g["vocab_size"], g["context_length"], g["block_count"])
    assert doc["block"]["repeat"] == m["n_layers"] and m["dtype"] == "f32"
    return path, meta, dims, doc


def lin_state(heads, hd, name="lin"):
    return {"name": name, "kind": "recurrent", "op": "linear_attention", "dtype": "f32",
            "tensors": [{"name": "num", "shape": [heads, hd], "init": 0},
                        {"name": "den", "shape": [heads, hd], "init": 0}]}


def kv_state(heads, hd, ctx):
    return {"name": "kv", "kind": "kv", "heads": heads, "kv_heads": heads, "head_dim": hd, "max_len": ctx,
            "window": 0, "sinks": 0, "dtype": "f32"}


GPT2_TENSORS = {"token_embd.weight", "position_embd.weight", "output_norm.weight", "output_norm.bias", "output.weight"}


def gpt2_block_tensors(layers):
    return {f"blk.{i}.{t}.{w}" for i in range(layers) for t in
            ("attn_norm", "attn_qkv", "attn_output", "ffn_norm", "ffn_up", "ffn_down") for w in ("weight", "bias")}


@needs_bin
def test_kimi_exports_gpt2_tensors_with_linear_attention(tmp_path):
    path, meta, dims, doc = family_export(tmp_path, "k", {"attention": "kimi"}, "gpt2", "kimi")
    assert set(dims) == GPT2_TENSORS | gpt2_block_tensors(2)
    assert doc["block"]["state"] == [lin_state(2, 16)]
    nodes = doc["block"]["nodes"]
    assert [n["op"] for n in nodes] == ["layernorm", "linear", "split", "linear_attention", "linear", "add",
                                        "layernorm", "linear", "gelu", "linear", "add"]
    assert nodes[3] == {"op": "linear_attention", "in": ["q", "k", "v"], "state": "lin", "heads": 2,
                        "head_dim": 16, "causal": True, "feature_map": "elu_plus_one", "den_floor": 1e-08,
                        "out": "a"}
    assert "dense" not in json.dumps(doc["modes"]) and "scans" in doc["modes"]["prefill"]


@needs_bin
def test_srd_exports_router_and_both_paths(tmp_path):
    path, meta, dims, doc = family_export(tmp_path, "r", {"attention": "srd"}, "paperkiln", "srd")
    d = 32
    pred = {f"blk.{i}.srd_pred_{a}.{w}" for i in range(2) for a in ("up", "down") for w in ("weight", "bias")}
    assert set(dims) == GPT2_TENSORS | gpt2_block_tensors(2) | pred
    assert (dims["blk.0.srd_pred_up.weight"], dims["blk.0.srd_pred_down.weight"]) == ([d, d // 4], [d // 4, d])
    assert {k: meta[k][1] for k in ("paperkiln.attention.head_count", "paperkiln.feed_forward_length")} == {
        "paperkiln.attention.head_count": 2, "paperkiln.feed_forward_length": 4 * d}
    assert doc["block"]["state"] == [kv_state(2, 16, 16), lin_state(2, 16)]
    ops = [n["op"] for n in doc["block"]["nodes"]]
    assert ops == ["layernorm", "linear", "gelu", "linear", "sub", "rms_row", "affine", "sigmoid", "affine",
                   "linear", "split", "attention", "linear_attention", "mul_col", "mul_col", "add", "linear", "add",
                   "layernorm", "linear", "gelu", "linear", "add"]
    by = {n["op"] + str(i): n for i, n in enumerate(doc["block"]["nodes"])}
    assert (by["affine6"]["scale"], by["affine6"]["shift"]) == (4.0, -1.0)
    assert (by["affine8"]["scale"], by["affine8"]["shift"], by["affine8"]["out"]) == (-1.0, 1.0, "gi")
    assert by["rms_row5"]["eps"] == 1e-08
    assert by["mul_col13"]["in"] == ["a", "g"] and by["mul_col14"]["in"] == ["l", "gi"]


@needs_bin
@pytest.mark.parametrize("layers", [2, 3])
def test_attnres_exports_depth_attention(tmp_path, layers):
    path, meta, dims, doc = family_export(tmp_path, "a", {"attention": "attnres", "layers": layers},
                                          "paperkiln", "attnres")
    q = {f"blk.{i}.attnres_{s}_q.weight" for i in range(layers) for s in ("attn", "ffn")}
    assert set(dims) == GPT2_TENSORS | gpt2_block_tensors(layers) | q | {"attnres_out_q.weight"}
    assert dims["attnres_out_q.weight"] == [32]
    # Layer 0 has one source: its query is written as zeros and never read.
    assert set(gguf_tensor(path, "blk.0.attnres_attn_q.weight")[1]) == {0.0}
    assert doc["block"]["state"] == [kv_state(2, 16, 16)]
    assert [n["op"] for n in doc["embed"]["nodes"]] == ["embedding", "pos_embedding", "list"]
    nodes = doc["block"]["nodes"]
    assert [n["op"] for n in nodes] == ["depth_attend", "layernorm", "linear", "split", "attention", "linear",
                                        "depth_attend", "layernorm", "linear", "gelu", "linear", "add", "append"]
    assert nodes[0]["in"] == ["bank"] and nodes[6]["in"] == ["bank", "o"] and nodes[0]["eps"] == 1e-05
    assert nodes[-1] == {"op": "append", "in": ["bank", "o"], "out": "bank"}
    head = doc["head"]["nodes"]
    assert [n["op"] for n in head] == ["depth_attend", "layernorm", "linear"]
    assert head[0]["weight"] == "attnres_out_q.weight"


FLEX = [  # (custom, arch, hint, extra tensors per block or None)
    ({"layers": 3, "residual": "highway"}, "paperkiln", "flex_highway", ("highway_attn", "highway_ffn")),
    ({"layers": 3, "residual": "plain"}, "gpt2", "flex_plain", ()),
    ({"layers": 3, "norm": "rmsnorm", "activation": "swiglu", "position": "sinusoidal"}, "paperkiln", "flex", None),
    ({"layers": 2, "activation": "relu", "residual": "highway", "attention": "swa", "window": 4, "sinks": 2},
     "paperkiln", "flex_highway", ("highway_attn", "highway_ffn")),
    # A runtime-defined activation (squared ReLU): relu's tensors, an expr node in place of relu.
    ({"layers": 2, "activation": "expr:max(x, 0)^2"}, "paperkiln", "flex", None),
    # RoPE with a non-SwiGLU activation is flex (ReLU^2 Wins, arXiv 2402.03804): no position table,
    # a rope node between split and attention.
    ({"layers": 2, "norm": "rmsnorm", "activation": "expr:relu(x)^2", "position": "rope"}, "paperkiln", "flex", None),
    # A runtime-defined norm (layer norm written as an expr): an expr node, then the gain and bias as a scale node.
    ({"layers": 2, "norm": "expr:(x - rowmean(x)) / sqrt(rowmean((x - rowmean(x))^2) + 1e-5)", "position": "rope"},
     "paperkiln", "flex", None),
]


@needs_bin
@pytest.mark.parametrize("custom,arch,hint,extra", FLEX)
def test_flex_flavours_export(tmp_path, custom, arch, hint, extra):
    path, meta, dims, doc = family_export(tmp_path, "x", custom, arch, hint)
    L = custom["layers"]
    if extra is not None:
        assert set(dims) == GPT2_TENSORS | gpt2_block_tensors(L) | {
            f"blk.{i}.{t}.{w}" for i in range(L) for t in extra for w in ("weight", "bias")}
    nodes = doc["block"]["nodes"]
    ops = [n["op"] for n in nodes]
    if custom.get("residual") == "highway":
        # x + sigmoid(W x + b) * (f - x), for both sublayers.
        assert ops.count("sigmoid") == 2 and ops.count("sub") == 2
        i = ops.index("sigmoid")
        assert nodes[i - 1]["weight"] == "blk.{i}.highway_attn.weight" and nodes[i - 1]["in"] == ["x"]
        assert [nodes[i + 1]["op"], nodes[i + 2]["op"], nodes[i + 3]["op"]] == ["sub", "mul", "add"]
    if custom.get("residual") == "plain":
        assert "add" not in ops
        assert [n["out"] for n in nodes if n["op"] == "linear"][1::2] == ["x", "x"]
    if custom.get("position") == "rope":
        # No table; the llama block's rope node (all heads) between split and attention.
        assert "position_embd.weight" not in dims
        assert [n["op"] for n in doc["embed"]["nodes"]] == ["embedding"]
        i = ops.index("rope")
        assert ops[i - 1] == "split" and ops[i + 1] == "attention" and ops.count("rope") == 1
        rope = nodes[i]
        assert rope["in"] == ["q", "k", "pos"] and rope["out"] == ["q", "k"]
        assert (rope["heads"], rope["head_dim"], rope["theta"], rope["rope_heads"], rope["pairing"]) == (
            2, 16, 10000.0, "all", "interleaved")
        assert any(x["name"] == "pos" for x in doc["inputs"])
    if custom.get("norm", "").startswith("expr:"):
        # expr node, then the gain g and bias b [d] as a scale node, three times.
        e = custom["norm"][5:]
        d = 32
        for where, name in ((nodes, "blk.{i}.attn_norm_gain"), (nodes, "blk.{i}.ffn_norm_gain"),
                            (doc["head"]["nodes"], "output_norm_gain")):
            k = next(j for j, n in enumerate(where) if n.get("weight") == name + ".weight")
            assert where[k - 1]["op"] == "expr" and [where[k - 1]["out"]] == where[k]["in"] == [where[k]["out"]]
            assert where[k] == {"op": "scale", "in": where[k]["in"], "weight": name + ".weight",
                                "bias": name + ".bias", "out": where[k]["out"]}
        assert [n["expr"] for n in nodes if n["op"] == "expr"] == [e, e]
        assert doc["head"]["nodes"][0] == {"op": "expr", "in": ["x"], "expr": e, "out": "h"}
        assert "layernorm" not in ops and "rmsnorm" not in ops and ops.count("scale") == 2
        # No [d, d] diagonal tensors: every gain and bias is [d].
        gains = {k: v for k, v in dims.items() if "_norm_gain." in k}
        assert len(gains) == 2 * (2 * custom["layers"] + 1) and all(v == [d] for v in gains.values())
        # The numpy IR forward with scale equals the diag(g) linear the gain used to export as,
        # bit for bit on the same weights (every off-diagonal product is +-0).
        import numpy as np
        import ir_numpy
        W = gguf_weights(path)
        diag_doc = copy.deepcopy(doc)
        for sect in ("block", "head"):
            for n in diag_doc[sect]["nodes"]:
                if n["op"] == "scale":
                    n["op"] = "linear"
        W_diag = {k: (np.diag(v).astype(np.float32) if k.endswith("_norm_gain.weight") else v) for k, v in W.items()}
        ids = [1, 4, 2, 7, 3, 9, 5, 11, 6, 2, 8, 10]
        a = ir_numpy.forward(doc, W, ids)
        b = ir_numpy.forward(diag_doc, W_diag, ids)
        assert a.dtype == b.dtype == np.float32 and np.isfinite(a).all()
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    if custom.get("norm") == "rmsnorm" and custom.get("activation") == "swiglu":
        assert ops.count("rmsnorm") == 2 and "blk.0.attn_norm.bias" not in dims
        assert ops[-6:] == ["linear", "silu", "linear", "mul", "linear", "add"]
        # Sinusoidal positions: the fixed table, sin on even columns, cos on odd.
        import math
        table = gguf_tensor(path, "position_embd.weight")[1]
        d = 32
        for p_ in (0, 5):
            for i in (0, 1, 6, 7):
                ang = p_ / 10000.0 ** (2 * (i // 2) / d)
                want = math.sin(ang) if i % 2 == 0 else math.cos(ang)
                assert abs(table[p_ * d + i] - want) < 1e-6
    if custom.get("activation") == "relu":
        assert "relu" in ops
    if custom.get("activation", "").startswith("expr:"):
        assert "relu" not in ops and ops.count("expr") == 1
        node = nodes[ops.index("expr")]
        assert node == {"op": "expr", "in": ["u"], "expr": custom["activation"][5:], "out": "u"}
        assert nodes[ops.index("expr") - 1]["weight"] == "blk.{i}.ffn_up.weight"
    if custom.get("attention") == "swa":
        att = [n for n in nodes if n["op"] == "attention"][0]
        assert (att["window"], att["sinks"]) == (4, 2)
        assert (doc["block"]["state"][0]["window"], doc["block"]["state"][0]["sinks"]) == (4, 2)


@needs_bin
@pytest.mark.parametrize("layers", [2, 3])  # 2: ParityLM(SWA); 3: flex with attention swa
def test_swa_exports_gpt2_tensors_with_a_windowed_ir(tmp_path, layers):
    window, sinks, d, T, V, heads = 4, 2, 32, 16, 15, 2
    spec = gpt2_setup(tmp_path / "swa", "w", {"layers": layers, "attention": "swa", "window": window, "sinks": sinks})
    run(spec)
    ex = [e for e in events(spec) if e["event"] == "export" and e["format"] == "gguf"]
    assert len(ex) == 1, ex
    # The event says what a GGUF-only engine would get wrong.
    assert (ex[0]["arch"], ex[0]["attention"], ex[0]["window"], ex[0]["sinks"]) == ("gpt2", "swa", window, sinks)
    assert "dense attention" in ex[0]["warning"]
    path = ex[0]["path"]
    assert ir_check.check(path) == []  # every IR weight name is a tensor of the file
    meta, tensors, *_ = ir_check.read_gguf(path)
    assert meta["general.architecture"][1] == "gpt2"
    # Exactly the dense gpt2 tensor set: no sink parameters, nothing extra.
    names = {t[0] for t in tensors}
    want = {"token_embd.weight", "position_embd.weight", "output_norm.weight", "output_norm.bias", "output.weight"}
    for i in range(layers):
        want |= {f"blk.{i}.{t}.{w}" for t in ("attn_norm", "attn_qkv", "attn_output", "ffn_norm", "ffn_up", "ffn_down")
                 for w in ("weight", "bias")}
    assert names == want
    doc = embedded(path)
    # Not "llama" (ember's dense fast path) and not plain "gpt2".
    assert doc["family_hint"] == "gpt2_swa"
    g = {k.split(".", 1)[1]: v[1] for k, v in meta.items() if k.startswith("gpt2.")}
    assert doc["model"] == {"vocab": V, "d": d, "n_layers": layers, "ctx": T, "dtype": "f32"}
    assert (doc["model"]["d"], doc["model"]["vocab"], doc["model"]["ctx"], doc["model"]["n_layers"]) == (
        g["embedding_length"], g["vocab_size"], g["context_length"], g["block_count"])
    assert doc["block"]["repeat"] == g["block_count"] == layers
    # The kv state ember allocates: windowed, with the sinks it must keep.
    assert doc["block"]["state"] == [{"name": "kv", "kind": "kv", "heads": heads, "kv_heads": heads,
                                      "head_dim": d // heads, "max_len": T, "window": window, "sinks": sinks,
                                      "dtype": "f32"}]
    nodes = doc["block"]["nodes"]
    assert [n["op"] for n in nodes] == GPT2_OPS
    att = nodes[3]
    assert att["state"] == "kv" and att["in"] == ["q", "k", "v"] and att["out"] == "a"
    assert (att["heads"], att["kv_heads"], att["head_dim"], att["causal"]) == (heads, heads, d // heads, True)
    assert (att["window"], att["sinks"]) == (window, sinks)
    assert abs(att["scale"] - (d // heads) ** -0.5) < 1e-7
    # Everything else is the dense gpt2 IR of the same shape.
    dense = gpt2_setup(tmp_path / "dense", "e", {"layers": layers})
    run(dense)
    ddoc = embedded([e for e in events(dense) if e["event"] == "export" and e["format"] == "gguf"][0]["path"])
    a, b = ir_check.strip(doc), ir_check.strip(ddoc)
    for x in (a, b):
        x.pop("family_hint"), x.pop("inputs"), x.pop("modes")
        x["block"]["state"][0].pop("window"), x["block"]["state"][0].pop("sinks")
        x["block"]["nodes"][3].pop("window"), x["block"]["nodes"][3].pop("sinks")
    assert a == b
    assert ddoc["family_hint"] == "gpt2" and ddoc["block"]["nodes"][3]["window"] == 0


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


@needs_bin
def test_mamba_exports_ssm_scan(tmp_path):
    n, d, L = 8, 32, 2
    path, meta, dims, doc = family_export(tmp_path, "m", {"d_state": n}, "paperkiln", "mamba", preset="mamba-tiny")
    want = {"token_embd.weight", "output_norm.weight", "output_norm.bias", "output.weight", "output.bias"}
    for i in range(L):
        b = f"blk.{i}."
        want |= {b + "ssm_norm.weight", b + "ssm_norm.bias", b + "ssm_a.weight", b + "ssm_b.weight",
                 b + "ssm_c.weight", b + "ssm_d.weight"}
        want |= {b + t + "." + w for t in ("ssm_in", "ssm_out", "ssm_gate", "ssm_out_gate") for w in ("weight", "bias")}
    assert set(dims) == want
    assert (dims["blk.0.ssm_a.weight"], dims["blk.0.ssm_b.weight"], dims["blk.0.ssm_c.weight"],
            dims["blk.0.ssm_d.weight"], dims["blk.0.ssm_in.weight"]) == ([n, n], [n], [n], [1], [d, n])
    # No positions, no attention metadata.
    assert [x["name"] for x in doc["inputs"]] == ["tokens"]
    assert "paperkiln.attention.head_count" not in meta
    assert doc["block"]["state"] == [{"name": "ssm", "kind": "recurrent", "op": "ssm_scan", "dtype": "f32",
                                      "tensors": [{"name": "s", "shape": [n], "init": 0}]}]
    nodes = doc["block"]["nodes"]
    assert [x["op"] for x in nodes] == ["layernorm", "linear", "ssm_scan", "linear", "linear", "gelu", "mul",
                                        "linear", "gelu", "mul", "add"]
    assert nodes[2] == {"op": "ssm_scan", "in": ["u"], "state": "ssm", "n": n, "a": "blk.{i}.ssm_a.weight",
                        "b": "blk.{i}.ssm_b.weight", "c": "blk.{i}.ssm_c.weight", "d": "blk.{i}.ssm_d.weight",
                        "out": "y"}
    assert nodes[7]["in"] == ["x"]  # the outer gate reads the block input, not the normalised one
    assert doc["head"]["nodes"][1]["bias"] == "output.bias"
