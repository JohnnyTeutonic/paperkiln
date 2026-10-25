"""The flex family's options of 26 Oct 2026 end to end (LOCAL_LLM_STACK.md section 11 item 4.2): expr
activations and norms with more inputs (a gate projection, learned scalars and per-channel vectors:
arch.custom.activation_inputs / norm_inputs), norm placements per sublayer (pre, post, sandwich,
branch_post, none: arch.custom.norm_placement) and T5's relative position bias (position t5_relative).

Each model trains 30 steps on CPU and exports an F32 GGUF. Checks: it trains (loss falls); the file
passes spec/tools/ir_check.py (keramos's validator and the file's tensors); the IR carries the expected
nodes (the learned inputs as scale nodes over a ones row, the gate's linear, the norms where the
placement puts them); spec/tools/ir_numpy.py's forward over the GGUF equals mtstudio's own forward
(`mtstudio score`) on 40 prompts within ROUNDING, greedy 40/40.

T5's bias has no keramos op yet: its export refuses (by name), and only under PAPERKILN_IR_UNRATIFIED=t5_relative
does mtstudio write the proposed attention attribute position_bias {weight, buckets}; that file is checked
against ir_numpy with the proposal's few lines patched in (PROPOSED_ATTENTION_PATCH, the text proposed to
keramos), never against keramos's validator. Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import WORDS, corpus, events, run  # noqa: E402
from test_recursion_ir import gguf_exports, parity  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

PRIMER_NORM = "expr:(x - rowmean(x)) / sqrt(abs(rowmean(x * (x - rowmean(x)))))"
CASES = {
    "gated": {"activation": "expr:relu(a) * x", "activation_inputs": ["up", "gate"]},
    "swish_beta": {"activation": "expr:x * sigmoid(a * x)", "activation_inputs": ["up", {"kind": "scalar", "init": 1.0}],
                   "position": "rope", "norm": "rmsnorm"},
    "dyt": {"norm": "expr:tanh(a * x)", "norm_inputs": ["row", {"kind": "scalar", "init": 0.5}],
            "activation": "relu", "position": "rope", "embed": "expr:x * a",
            "embed_inputs": ["row", {"kind": "scalar", "init": 8.0}]},
    "dyt_channel": {"norm": "expr:tanh(x * a)", "norm_inputs": ["row", {"kind": "channel", "init": 0.5}],
                    "activation": "gelu", "embed": "expr:x * a",
                    "embed_inputs": ["row", {"kind": "channel", "init": 8.0}]},
    # DyT on LLaMA (arXiv 2503.10622, as the chain builds it): per-place alpha inits, a learned embedding scalar.
    "dyt_llama": {"norm": "expr:tanh(a * x)", "activation": "swiglu", "position": "rope",
                  "norm_inputs": ["row", {"kind": "scalar", "init": 0.8, "init_ffn": 0.2, "init_final": 0.2}],
                  "embed": "expr:x * a", "embed_inputs": ["row", {"kind": "scalar", "init": 4.0}]},
    "post": {"norm": "layernorm", "activation": "gelu", "position": "learned", "norm_placement": "post"},
    "sandwich": {"norm": "rmsnorm", "activation": "swiglu", "position": "rope", "norm_placement": "sandwich"},
    "primer": {"norm": PRIMER_NORM, "activation": "expr:relu(x)^2", "position": "rope",
               "norm_placement": {"attn": "pre", "ffn": "branch_post"}},
    "none_highway": {"norm": "rmsnorm", "activation": "relu", "residual": "highway",
                     "norm_placement": {"attn": "none", "ffn": "branch_post"}},
}

# The proposed reference lines (keramos): T5's bias added to each head's scaled scores, by the bucket of p - j.
PROPOSED_ATTENTION_PATCH = (
    '                s = (q[:, h * hd:(h + 1) * hd] @ k[:, kh * hd:(kh + 1) * hd].T) * np.float32(nd["scale"])\n',
    '                s = (q[:, h * hd:(h + 1) * hd] @ k[:, kh * hd:(kh + 1) * hd].T) * np.float32(nd["scale"])\n'
    '                pb = nd.get("position_bias")\n'
    '                if pb:  # proposed: T5\'s relative bias W[bucket(p - j)][h], p and j absolute positions\n'
    '                    b = np.asarray(pb["buckets"])\n'
    '                    s = (s + w(pb["weight"])[b[np.clip(p - j, 0, len(b) - 1)], h]).astype(np.float32)\n')


def spec_of(tmp, name, custom, steps=30, formats=("safetensors", "gguf")):
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    spec = {"name": name, "arch": {"preset": "gpt2-nano", "custom": {"d": 32, "layers": 2, "heads": 4, **custom}},
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": {"steps": steps, "lr": 0.003, "eval_every": 15, "checkpoint_every": 15, "batch": 2, "seed": 3,
                      "gradmap_every": 0},
            "export": {"formats": list(formats)}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def doc_of(spec):
    f32 = next(e for e in gguf_exports(spec) if e["format"] == "gguf")["path"]
    return json.loads(ir_check.read_gguf(f32)[0]["paperkiln.ir"][1]), f32


def check_nodes(name, doc):
    nodes = doc["block"]["nodes"]
    ops = [n["op"] for n in nodes]
    if name == "gated":
        e = next(n for n in nodes if n["op"] == "expr")
        assert e["in"] == ["u", "g"] and e["expr"] == "relu(a) * x"
        assert any(n["op"] == "linear" and n["weight"] == "blk.{i}.ffn_gate.weight" and n["out"] == "g" for n in nodes)
    elif name == "swish_beta":
        assert {"op": "expr", "in": ["u"], "expr": "1", "out": "one"} in nodes
        assert {"op": "scale", "in": ["one"], "weight": "blk.{i}.ffn_act_p_a.weight", "bias": None, "out": "p_a"} in nodes
        assert next(n for n in nodes if n["op"] == "expr" and n["expr"] != "1")["in"] == ["u", "p_a"]
    elif name in ("dyt", "dyt_channel"):
        assert ops.count("expr") == 4  # two norms, each a ones row and its expr
        assert [n["weight"] for n in nodes if n["op"] == "scale" and n["in"] == ["one"]] == [
            "blk.{i}.attn_norm_p_a.weight", "blk.{i}.ffn_norm_p_a.weight"]
        assert [n["op"] for n in doc["head"]["nodes"]][:4] == ["expr", "scale", "expr", "scale"]
    elif name == "dyt_llama":
        assert [n["op"] for n in doc["embed"]["nodes"]] == ["embedding", "expr", "scale", "expr"]
        assert doc["embed"]["nodes"][2]["weight"] == "embd_p_a.weight"
    elif name == "post":
        assert ops == ["linear", "split", "attention", "linear", "add", "layernorm", "linear", "gelu", "linear", "add",
                       "layernorm"], ops
        assert [n["op"] for n in doc["head"]["nodes"]] == ["linear"]  # post/post: no final norm
    elif name == "sandwich":
        assert ops.count("rmsnorm") == 4
        assert [n["weight"] for n in nodes if n["op"] == "rmsnorm"] == [
            "blk.{i}.attn_norm.weight", "blk.{i}.attn_post_norm.weight", "blk.{i}.ffn_norm.weight",
            "blk.{i}.ffn_post_norm.weight"]
    elif name == "primer":
        assert ops == ["expr", "scale", "linear", "split", "rope", "attention", "linear", "add", "linear", "expr",
                       "linear", "expr", "scale", "add"], ops
    elif name == "none_highway":
        i_attn = ops.index("attention")
        assert "rmsnorm" not in ops[:i_attn] and ops.count("rmsnorm") == 1


@needs_bin
@pytest.mark.parametrize("name", list(CASES))
def test_flex_option_exports_and_numpy_parity(tmp_path, name):
    spec = spec_of(tmp_path, name, CASES[name])
    run(spec)
    ev = events(spec)
    steps = [e["loss"] for e in ev if e["event"] == "step"]
    assert steps[-1] < steps[0] - 0.3, steps
    model = next(e for e in ev if e["event"] == "model")
    assert model["family"] == "flex"
    for e in gguf_exports(spec):
        assert ir_check.check(e["path"]) == [], ir_check.check(e["path"])
        meta = ir_check.read_gguf(e["path"])[0]
        assert meta["general.architecture"][1] == "paperkiln"
    doc, _ = doc_of(spec)
    check_nodes(name, doc)
    worst, agree, n, _ = parity(spec, tmp_path)
    print(f"{name}: numpy IR vs mtstudio log-probs worst {worst:.2e}; greedy {agree}/{n}")
    assert n == 40 and agree == 40 and worst <= 1e-5


def proposed_ir_numpy():
    """spec/tools/ir_numpy.py with PROPOSED_ATTENTION_PATCH applied, as its own module."""
    src = (ROOT / "spec" / "tools" / "ir_numpy.py").read_text(encoding="utf-8")
    old, new = PROPOSED_ATTENTION_PATCH
    assert src.count(old) == 1
    spec = importlib.util.spec_from_loader("ir_numpy_proposed", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src.replace(old, new), "ir_numpy_proposed", "exec"), mod.__dict__)
    return mod


@needs_bin
def test_t5_relative_bias_refused_then_proposed(tmp_path, monkeypatch):
    custom = {"norm": "rmsnorm", "activation": "relu", "position": "t5_relative", "rel_buckets": 8,
              "rel_max_distance": 12}
    spec = spec_of(tmp_path / "refused", "t5", custom)
    monkeypatch.delenv("PAPERKILN_IR_UNRATIFIED", raising=False)
    p = subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=1800)
    assert p.returncode != 0 and "no relative position bias in attention" in (p.stdout + p.stderr)
    # It trains (safetensors written before the GGUF export refused).
    steps = [e["loss"] for e in events(spec) if e["event"] == "step"]
    assert len(steps) == 30 and steps[-1] < steps[0] - 0.3
    # Under the flag: the proposed attribute, checked against ir_numpy with the proposal's lines.
    monkeypatch.setenv("PAPERKILN_IR_UNRATIFIED", "t5_relative")
    spec = spec_of(tmp_path / "proposed", "t5", custom)
    run(spec)
    doc, f32 = doc_of(spec)
    att = next(n for n in doc["block"]["nodes"] if n["op"] == "attention")
    assert att["position_bias"]["weight"] == "rel_pos_bias.weight"
    assert att["position_bias"]["buckets"] == [0, 1, 2, 3, 4, 4, 5, 6, 6, 6, 7, 7, 7]
    names = {t[0]: list(t[1]) for t in ir_check.read_gguf(f32)[1]}
    assert names["rel_pos_bias.weight"] == [4, 8]  # ggml ne: [heads, buckets]
    import test_recursion_ir as R
    monkeypatch.setitem(sys.modules, "ir_numpy", proposed_ir_numpy())
    worst, agree, n, _ = R.parity(spec, tmp_path / "proposed")
    print(f"t5_relative (proposed IR): numpy vs mtstudio log-probs worst {worst:.2e}; greedy {agree}/{n}")
    assert n == 40 and agree == 40 and worst <= 1e-5


@needs_bin
@pytest.mark.parametrize("custom,msg", [
    ({"activation": "expr:x * a", "activation_inputs": ["up", "up"]}, "exactly once"),
    ({"activation": "relu", "activation_inputs": ["up"]}, "need an expr activation"),
    ({"norm": "expr:x * a", "norm_inputs": ["row", "gate"]}, "unknown input"),
    ({"norm_placement": "middle"}, "norm_placement must be"),
    ({"norm_placement": {"attn": "pre", "mlp": "post"}}, "unknown key mlp"),
    ({"position": "t5_relative", "attention": "swa"}, "needs exact attention"),
])
def test_bad_flex_option_specs_are_refused(tmp_path, custom, msg):
    spec = spec_of(tmp_path, "bad", custom, formats=("safetensors",))
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    if r.returncode == 0:  # plan builds no model: a refusal the model's constructor makes shows at run
        r = subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and msg in (r.stderr + r.stdout), r.stderr + r.stdout


@needs_bin
def test_llama_flavours_with_a_flex_option_are_flex(tmp_path):
    spec = spec_of(tmp_path, "s", {"norm": "rmsnorm", "activation": "swiglu", "position": "rope",
                                   "norm_placement": "sandwich"}, steps=2, formats=("safetensors",))
    run(spec)
    assert next(e for e in events(spec) if e["event"] == "model")["family"] == "flex"
    spec = spec_of(tmp_path, "l", {"qk_norm": True, "norm_placement": "post"}, steps=2, formats=("safetensors",))
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
