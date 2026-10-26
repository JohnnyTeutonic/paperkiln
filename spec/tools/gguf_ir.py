#!/usr/bin/env python3
"""A third-party GGUF's paperkiln IR, built from its llama.cpp metadata (general.architecture and the
<arch>.* keys) and tensor names: the reference for ember.cpp's C++ builder (IR_SPEC.md section 9.3). A file with paperkiln.ir needs none of this.

    python tools/gguf_ir.py MODEL.gguf [OUT.ir.json]

Families (the llama.cpp b6000 graph each mirrors, src/llama-model.cpp; the model's own HF semantics
where llama.cpp approximates):
  llama, mistral  pre-RMSNorm, interleaved RoPE (llama.cpp's convert permutes q/k for it), grouped-
                  query attention, SwiGLU; output.weight or the tied token_embd.weight. With
                  <arch>.attention.sliding_window = w the attention and its kv state get window w
                  (sinks 0: key j visible from query p iff p - w < j <= p, HF Mistral's mask).
                  llama.cpp has no architecture "mistral" (b6000 and later refuse it): its convert
                  writes Mistral checkpoints as "llama" without the window key, and its llama graph
                  has no window, so llama.cpp computes such a file without one (a writer that keeps
                  HF's sliding_window must add the key). llama with <arch>.expert_count and
                  ffn_gate_inp (Mixtral's layout): moe_ffn in place of the SwiGLU (softmax router,
                  top expert_used_count, gates renormalised over them, SiLU experts; llama.cpp's
                  llama graph).
  qwen2           llama + q/k/v biases; NEOX (half-split) RoPE. No window (the GGUF does not say
                  which layers HF's use_sliding_window would cover; Qwen2 checkpoints leave it off).
  qwen3           qwen2 without the biases, with a per-head RMSNorm on q and k before RoPE
                  (attn_q_norm, attn_k_norm [head_dim]).
  gpt2            learned positions, pre-LayerNorm, fused biased qkv split q|k|v, GELU (tanh),
                  output.weight or tied.
  gptneox         (Pythia) gpt2's tensor set without positions; partial NEOX RoPE over the first
                  rope.dimension_count dims of each head; use_parallel_residual: x + attn(ln1 x) +
                  mlp(ln2 x), else sequential; GELU exact (erf: HF's "gelu"; llama.cpp b6000 uses
                  the tanh form here, an approximation of the trained model); untied head, no bias.
  gemma3          embedding x sqrt(d) (affine), sandwich RMSNorms (attn_norm, post_attention_norm,
                  ffn_norm, post_ffw_norm), per-head q/k RMSNorm, NEOX RoPE, GeGLU (gelu tanh of
                  gate, times up), attention scale 1/sqrt(key_length) (d/heads for the 27B), tied
                  head. Layer i is local (sliding window <arch>.attention.sliding_window, RoPE base
                  10000, no RoPE scaling) when i % 6 < 5, global (window 0, rope.freq_base,
                  rope.scaling linear factor) otherwise: per-layer attributes {"per_layer": [...]}.
                  llama.cpp's convert stores the norm weights as 1 + w already, so no node carries
                  weight_offset; a GGUF holding HF's raw w would set weight_offset 1 on each.
  mamba           (Mamba-1) pre-RMSNorm; in_proj -> x | z; causal depthwise conv1d (with bias,
                  state of kernel-1 inputs) + SiLU; x_proj -> dt | B | C; dt_proj (biased);
                  selective_scan (softplus(dt), A = ssm_a as stored = -exp(A_log), D skip); times
                  SiLU(z); out_proj; residual. Head RMSNorm, output or tied.
Every other architecture stops with its name. The document has the shape IR_SPEC.md
describes; tools/ir_numpy.py runs it.
"""
from __future__ import annotations

import json
import math
import struct
import sys

IR_VERSION = 1  # IR_SPEC.md section 2.3
NEOX = {"qwen2", "qwen3"}  # llama.cpp LLAMA_ROPE_TYPE_NEOX for these
LLAMA_LIKE = {"llama", "mistral", "qwen2", "qwen3"}
WINDOWED_LLAMA = {"llama", "mistral"}  # read <arch>.attention.sliding_window


def read_meta(path):
    """(metadata {key: value}, {tensor name: numpy-order shape}) via the gguf package."""
    import gguf
    r = gguf.GGUFReader(path)
    meta = {}
    for k, f in r.fields.items():
        if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model", "tokenizer.ggml.pre",
                                                          "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id",
                                                          "tokenizer.ggml.add_bos_token"):
            if k == "tokenizer.ggml.tokens":
                meta["_n_tokens"] = len(f.data)
            continue  # big arrays: only their length matters here
        meta[k] = f.contents()
    shapes = {t.name: [int(x) for x in reversed(t.shape.tolist())] for t in r.tensors}
    return meta, shapes


def f32(v):
    """The value a float32 holds, as the shortest decimal that round-trips (ir::f32's convention). Standard
    library only (tools/ir_validate.py builds graphs for rule 38): the correctly rounded decimal of the fewest
    digits that reads back as the same float32, which is numpy's str(np.float32(v))."""
    x = struct.unpack("<f", struct.pack("<f", float(v)))[0]
    if x != x or x in (float("inf"), float("-inf")):
        return x
    for p in range(1, 10):
        d = float("%.*g" % (p, x))
        if struct.unpack("<f", struct.pack("<f", d))[0] == x:
            return d
    return x


def linear(x, weight, out, bias=None, tied_to=None):
    nd = {"op": "linear", "in": [x], "weight": weight, "bias": bias, "out": out}
    if tied_to:
        nd["tied_to"] = tied_to
    return nd


def rmsnorm(x, weight, eps, out, group=None):
    nd = {"op": "rmsnorm", "in": [x], "weight": weight, "eps": f32(eps), "out": out}
    if group:
        nd["group"] = group
    return nd


def layernorm(x, name, eps, out):
    return {"op": "layernorm", "in": [x], "weight": name + ".weight", "bias": name + ".bias", "eps": f32(eps),
            "out": out}


def op(name, ins, out, **attrs):
    return {"op": name, "in": ins, **attrs, "out": out}


def rope(H, KV, hd, n_rot, theta, pairing, freq_scale=None):
    nd = {"op": "rope", "in": ["q", "k", "pos"], "heads": H, "kv_heads": KV, "head_dim": hd, "n_rot": n_rot,
          "theta": theta, "rope_heads": "all", "pairing": pairing, "out": ["q", "k"]}
    if freq_scale is not None:
        nd["freq_scale"] = freq_scale
    return nd


def attention(H, KV, hd, scale, window=0, out="a"):
    return {"op": "attention", "in": ["q", "k", "v"], "state": "kv", "heads": H, "kv_heads": KV, "head_dim": hd,
            "causal": True, "scale": f32(scale), "window": window, "sinks": 0,
            "softmax": "max-shifted, masked entries exactly zero", "out": out}


def kv_state(H, KV, hd, ctx, window=0):
    return {"name": "kv", "kind": "kv", "heads": H, "kv_heads": KV, "head_dim": hd, "max_len": ctx,
            "window": window, "sinks": 0, "dtype": "f32"}


class Meta:
    def __init__(self, meta, shapes):
        self.meta, self.shapes = meta, shapes
        self.arch = meta["general.architecture"]

    def get(self, key, default=None):
        v = self.meta.get(self.arch + "." + key, default)
        if v is None:
            raise SystemExit(f"gguf_ir: {self.arch} file has no {self.arch}.{key}")
        return v

    def opt(self, key):
        return self.meta.get(self.arch + "." + key)

    def no_rope_scaling(self):
        if self.opt("rope.scaling.type") not in (None, "none") or "rope_freqs.weight" in self.shapes:
            raise SystemExit(f"gguf_ir: {self.arch} with RoPE scaling is not built yet")

    def attn_dims(self):
        d, H = int(self.get("embedding_length")), int(self.get("attention.head_count"))
        KV = int(self.get("attention.head_count_kv", H))
        hd = int(self.get("attention.key_length", d // H))
        return d, H, KV, hd

    def vocab(self):
        return int(self.meta.get("_n_tokens") or self.shapes["token_embd.weight"][0])


def document(m, nodes_embed, state, nodes, head, ctx):
    meta = m.meta
    eos = meta.get("tokenizer.ggml.eos_token_id")
    n_layers = int(m.get("block_count"))
    return {
        "_comment": f"built by tools/gguf_ir.py from general.architecture = {m.arch} and its metadata",
        "ir_version": IR_VERSION,
        "family_hint": m.arch,
        "model": {"vocab": m.vocab(), "d": int(m.get("embedding_length")), "n_layers": n_layers, "ctx": ctx,
                  "dtype": "f32"},
        "tokenizer": {"type": "gguf", "source": "gguf",
                      "specials": {} if eos is None else {"eot": int(eos)}, "bos": meta.get("tokenizer.ggml.bos_token_id")},
        "inputs": [{"name": "tokens", "kind": "token_ids"}, {"name": "pos", "kind": "positions"}],
        "embed": {"nodes": nodes_embed},
        "block": {"repeat": n_layers, "state": state, "nodes": nodes},
        "head": {"nodes": head},
        "outputs": [{"name": "logits", "kind": "logits"}],
        "modes": {"prefill": "all nodes over T tokens",
                  "decode": "the same graph with T = 1, attention reads and appends to the kv state, a "
                            "recurrent op reads and replaces its recurrent state"},
    }


def tied_head(m, norm_nodes):
    tied = "output.weight" not in m.shapes
    return norm_nodes + [linear("h", "output.weight", "logits", tied_to="token_embd.weight" if tied else None)]


EMBED = [{"op": "embedding", "in": ["tokens"], "weight": "token_embd.weight", "out": "x"}]
B = "blk.{i}."


def llama_like(m):
    arch = m.arch
    d, H, KV, hd = m.attn_dims()
    n_rot = int(m.get("rope.dimension_count", hd))
    ctx, eps = int(m.get("context_length")), float(m.get("attention.layer_norm_rms_epsilon"))
    theta = float(m.get("rope.freq_base", 10000.0))
    m.no_rope_scaling()
    window = int(m.opt("attention.sliding_window") or 0) if arch in WINDOWED_LLAMA else 0
    bias = "blk.0.attn_q.bias" in m.shapes
    qk_norm = "blk.0.attn_q_norm.weight" in m.shapes
    pairing = "neox" if arch in NEOX else "interleaved"

    def proj(name, out):
        return linear("h", B + name + ".weight", out, B + name + ".bias" if bias else None)

    nodes = [rmsnorm("x", B + "attn_norm.weight", eps, "h"), proj("attn_q", "q"), proj("attn_k", "k"),
             proj("attn_v", "v")]
    if qk_norm:
        nodes += [rmsnorm("q", B + "attn_q_norm.weight", eps, "q", group=hd),
                  rmsnorm("k", B + "attn_k_norm.weight", eps, "k", group=hd)]
    nodes += [
        rope(H, KV, hd, n_rot, f32(theta), pairing),
        attention(H, KV, hd, 1.0 / math.sqrt(hd), window),
        linear("a", B + "attn_output.weight", "o"),
        op("add", ["x", "o"], "x"),
        rmsnorm("x", B + "ffn_norm.weight", eps, "h2"),
    ]
    if arch == "llama" and m.opt("expert_count") and "blk.0.ffn_gate_inp.weight" in m.shapes:
        nodes.append(op("moe_ffn", ["h2"], "f", router=B + "ffn_gate_inp.weight",
                        experts={"gate": B + "ffn_gate_exps.weight", "up": B + "ffn_up_exps.weight",
                                 "down": B + "ffn_down_exps.weight"},
                        n_experts=int(m.get("expert_count")), top_k=int(m.get("expert_used_count")),
                        gate_norm="renorm", activation="silu"))
    else:
        nodes += [
            linear("h2", B + "ffn_gate.weight", "g"),
            linear("h2", B + "ffn_up.weight", "u"),
            op("silu", ["g"], "g"),
            op("mul", ["g", "u"], "gu"),
            linear("gu", B + "ffn_down.weight", "f"),
        ]
    nodes.append(op("add", ["x", "f"], "x"))
    head = tied_head(m, [rmsnorm("x", "output_norm.weight", eps, "h")])
    return document(m, EMBED, [kv_state(H, KV, hd, ctx, window)], nodes, head, ctx)


def gpt2_like(m):
    """gpt2 and gptneox: LayerNorm blocks over a fused, biased qkv (q | k | v as llama.cpp's convert
    writes it; for gptneox it has already de-interleaved HF's per-head [q k v] rows)."""
    neox = m.arch == "gptneox"
    d, H, KV, hd = m.attn_dims()
    ctx, eps = int(m.get("context_length")), float(m.get("attention.layer_norm_epsilon"))
    embed = list(EMBED)
    if not neox:
        embed.append({"op": "pos_embedding", "in": ["x", "pos"], "weight": "position_embd.weight", "out": "x"})
    nodes = [layernorm("x", B + "attn_norm", eps, "h"),
             linear("h", B + "attn_qkv.weight", "qkv", B + "attn_qkv.bias"),
             op("split", ["qkv"], ["q", "k", "v"], sizes=[H * hd, KV * hd, KV * hd])]
    if neox:
        m.no_rope_scaling()
        nodes.append(rope(H, KV, hd, int(m.get("rope.dimension_count", hd)), f32(float(m.get("rope.freq_base", 10000.0))),
                          "neox"))
    nodes += [attention(H, KV, hd, 1.0 / math.sqrt(hd)),
              linear("a", B + "attn_output.weight", "o", B + "attn_output.bias")]

    def ffn(src):
        return [layernorm(src, B + "ffn_norm", eps, "h2"),
                linear("h2", B + "ffn_up.weight", "u", B + "ffn_up.bias"),
                op("gelu", ["u"], "u", approx="none" if neox else "tanh"),
                linear("u", B + "ffn_down.weight", "f", B + "ffn_down.bias")]

    if neox and bool(m.get("use_parallel_residual", True)):
        # llama.cpp: cur = ffn(ln2 x) + x, then + attn_out; HF sums mlp + attn + x (ROUNDING apart).
        nodes += ffn("x") + [op("add", ["f", "x"], "x"), op("add", ["x", "o"], "x")]
    else:
        nodes += [op("add", ["x", "o"], "x")] + ffn("x") + [op("add", ["x", "f"], "x")]
    head = [layernorm("x", "output_norm", eps, "h")]
    if neox:
        head.append(linear("h", "output.weight", "logits"))
    else:
        head = tied_head(m, head)
    return document(m, embed, [kv_state(H, KV, hd, ctx)], nodes, head, ctx)


GEMMA3_SWA_PATTERN = 6  # llama.cpp b6000 hparams.set_swa_pattern(6): layer i local iff i % 6 < 5
GEMMA3_LOCAL_THETA = 10000.0  # llama.cpp rope_freq_base_train_swa for gemma3


def gemma3(m):
    d, H, KV, hd = m.attn_dims()
    L = int(m.get("block_count"))
    n_rot = int(m.get("rope.dimension_count", hd))
    ctx, eps = int(m.get("context_length")), float(m.get("attention.layer_norm_rms_epsilon"))
    if "rope_freqs.weight" in m.shapes:
        raise SystemExit("gguf_ir: gemma3 with rope_freqs is not built yet")
    stype = m.opt("rope.scaling.type")
    if stype not in (None, "none", "linear"):
        raise SystemExit(f"gguf_ir: gemma3 RoPE scaling {stype!r} is not built yet")
    factor = float(m.opt("rope.scaling.factor") or 1.0) if stype == "linear" else 1.0
    win = int(m.get("attention.sliding_window"))
    local = [i % GEMMA3_SWA_PATTERN < GEMMA3_SWA_PATTERN - 1 for i in range(L)]
    windows = {"per_layer": [win if loc else 0 for loc in local]}
    thetas = {"per_layer": [f32(GEMMA3_LOCAL_THETA if loc else float(m.get("rope.freq_base"))) for loc in local]}
    scales = {"per_layer": [1.0 if loc else f32(1.0 / factor) for loc in local]}
    # llama.cpp: f_attention_scale = 1/sqrt(n_embd_head_k), except 1/sqrt(n_embd / n_head) for the 27B.
    scale = 1.0 / math.sqrt(d / H) if L == 62 else 1.0 / math.sqrt(hd)
    embed = EMBED + [op("affine", ["x"], "x", scale=f32(math.sqrt(d)), shift=0.0)]
    nodes = [
        rmsnorm("x", B + "attn_norm.weight", eps, "h"),
        linear("h", B + "attn_q.weight", "q"), linear("h", B + "attn_k.weight", "k"),
        linear("h", B + "attn_v.weight", "v"),
        rmsnorm("q", B + "attn_q_norm.weight", eps, "q", group=hd),
        rmsnorm("k", B + "attn_k_norm.weight", eps, "k", group=hd),
        rope(H, KV, hd, n_rot, thetas, "neox", freq_scale=scales if factor != 1.0 else None),
        attention(H, KV, hd, scale, windows),
        linear("a", B + "attn_output.weight", "o"),
        rmsnorm("o", B + "post_attention_norm.weight", eps, "o"),
        op("add", ["x", "o"], "x"),
        rmsnorm("x", B + "ffn_norm.weight", eps, "h2"),
        linear("h2", B + "ffn_gate.weight", "g"),
        linear("h2", B + "ffn_up.weight", "u"),
        op("gelu", ["g"], "g", approx="tanh"),
        op("mul", ["g", "u"], "gu"),
        linear("gu", B + "ffn_down.weight", "f"),
        rmsnorm("f", B + "post_ffw_norm.weight", eps, "f"),
        op("add", ["x", "f"], "x"),
    ]
    head = tied_head(m, [rmsnorm("x", "output_norm.weight", eps, "h")])
    return document(m, embed, [kv_state(H, KV, hd, ctx, windows)], nodes, head, ctx)


def mamba(m):
    K, Di = int(m.get("ssm.conv_kernel")), int(m.get("ssm.inner_size"))
    N, R = int(m.get("ssm.state_size")), int(m.get("ssm.time_step_rank"))
    eps = float(m.get("attention.layer_norm_rms_epsilon"))
    ctx = int(m.get("context_length", 0)) or 1048576  # llama.cpp writes 1M: a recurrence has no length limit
    if "blk.0.ssm_dt.weight" not in m.shapes or "blk.0.ssm_d" not in m.shapes:
        raise SystemExit("gguf_ir: mamba file lacks ssm_dt.weight or ssm_d")
    if m.opt("ssm.dt_b_c_rms") or "blk.0.ssm_dt_norm.weight" in m.shapes:
        raise SystemExit("gguf_ir: mamba with dt/B/C RMSNorms (Falcon-Mamba) is not built yet")
    nodes = [
        rmsnorm("x", B + "attn_norm.weight", eps, "h"),
        linear("h", B + "ssm_in.weight", "xz"),
        op("split", ["xz"], ["xs", "z"], sizes=[Di, Di]),
        op("conv1d_causal", ["xs"], "xs", state="conv", weight=B + "ssm_conv1d.weight", bias=B + "ssm_conv1d.bias",
           channels=Di, kernel=K),
        op("silu", ["xs"], "xs"),
        linear("xs", B + "ssm_x.weight", "xdb"),
        op("split", ["xdb"], ["dt", "bm", "cm"], sizes=[R, N, N]),
        linear("dt", B + "ssm_dt.weight", "dt", B + "ssm_dt.bias"),
        op("selective_scan", ["xs", "dt", "bm", "cm"], "y", state="ssm", a=B + "ssm_a", d=B + "ssm_d", channels=Di,
           d_state=N, dt_softplus=True),
        op("silu", ["z"], "z"),
        op("mul", ["y", "z"], "y"),
        linear("y", B + "ssm_out.weight", "o"),
        op("add", ["x", "o"], "x"),
    ]
    state = [{"name": "conv", "kind": "recurrent", "op": "conv1d_causal", "dtype": "f32",
              "tensors": [{"name": "x", "shape": [K - 1, Di], "init": 0}]},
             {"name": "ssm", "kind": "recurrent", "op": "selective_scan", "dtype": "f32",
              "tensors": [{"name": "s", "shape": [Di, N], "init": 0}]}]
    head = tied_head(m, [rmsnorm("x", "output_norm.weight", eps, "h")])
    doc = document(m, EMBED, state, nodes, head, ctx)
    doc["inputs"] = [{"name": "tokens", "kind": "token_ids"}]
    return doc


BUILDERS = {**{a: llama_like for a in LLAMA_LIKE}, "gpt2": gpt2_like, "gptneox": gpt2_like, "gemma3": gemma3,
            "mamba": mamba}


def build(meta, shapes):
    arch = meta.get("general.architecture")
    if arch not in BUILDERS:
        raise SystemExit(f"gguf_ir: no IR builder for general.architecture = {arch!r}")
    return BUILDERS[arch](Meta(meta, shapes))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    meta, shapes = read_meta(argv[0])
    doc = build(meta, shapes)
    s = json.dumps(doc, indent=1)
    if len(argv) > 1:
        open(argv[1], "w", encoding="utf-8", newline="\n").write(s + "\n")
    m = doc["model"]
    att = [n for n in doc["block"]["nodes"] if n["op"] == "attention"]
    shape = f"heads {att[0]['heads']}/{att[0]['kv_heads']} x {att[0]['head_dim']}" if att else "recurrent"
    print(f"{doc['family_hint']}: d {m['d']}, {m['n_layers']} layers, {shape}, vocab {m['vocab']}, ctx {m['ctx']}, "
          f"{len(doc['block']['nodes'])} block nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
