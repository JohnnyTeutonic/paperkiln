#!/usr/bin/env python3
"""The suite's llama.cpp-layout GGUFs as Hugging Face checkpoints (model.safetensors + config.json), with
llama.cpp's convert transforms undone exactly, so that an engine that reads HF checkpoints (transformers,
ONNX Runtime GenAI's builder, vLLM) runs the same weights the suite's documents describe.

    python adapters/common/gguf_to_hf.py MODEL.gguf OUT_DIR        # writes OUT_DIR/{model.safetensors,config.json,
                                                                    #   tokenizer.json,tokenizer_config.json,keramos_export.json}
    python adapters/common/gguf_to_hf.py --check MODEL.gguf         # the round trip GGUF -> HF -> GGUF, bit for bit

The export reads general.architecture and the <arch>.* keys and never the IR document (paperkiln.ir): it is
what a converter from llama.cpp's format would do. Families (general.architecture -> transformers class):

  llama          LlamaForCausalLM; q/k rows un-permuted (llama.cpp's convert permutes HF's half-split RoPE
                 rows into interleaved pairs; the inverse is a row reindexing, exact).
  llama + experts (expert_count > 0)  MixtralForCausalLM; ffn_gate_exps | ffn_up_exps concatenated into
                 experts.gate_up_proj [E, 2 ff, d], ffn_down_exps -> experts.down_proj [E, d, ff].
  mistral        MistralForCausalLM, sliding_window = <arch>.attention.sliding_window; q/k un-permuted.
  qwen3          Qwen3ForCausalLM (NEOX RoPE: no permutation; q_norm, k_norm).
  gemma3         Gemma3ForCausalLM (Gemma3TextConfig): norm weights stored as 1 + w by llama.cpp's convert,
                 written as w = s - 1 (exact for s in [0.5, 2], Sterbenz; HF computes (1 + w) back); layer
                 types from llama.cpp's swa pattern (i % 6 < 5 sliding), local RoPE base 10000, global
                 rope.freq_base with linear scaling; query_pre_attn_scalar = key_length (scale
                 1/sqrt(head_dim), llama.cpp's except for the 27B); no softcapping (none in the GGUF).
  gptneox        GPTNeoXForCausalLM; attn_qkv re-interleaved per head ([q k v] per head, the inverse of
                 convert's de-interleave), partial_rotary_factor = rope.dimension_count / head_dim,
                 hidden_act "gelu" (exact: HF's and the IR's; llama.cpp computes tanh, its finding 2).
  gpt2           GPT2LMHeadModel; Conv1D weights are the transposes of llama.cpp's [out, in] (exact).
  mamba          MambaForCausalLM; conv1d [Di, K] -> [Di, 1, K]; A_log chosen per element so that
                 -exp(A_log) in float32 is ssm_a bit for bit where such a float32 exists (the nearest
                 otherwise; keramos_export.json counts them).

Anything else (architecture "paperkiln", other llama.cpp architectures) raises Unsupported. A tied head
(no output.weight) sets tie_word_embeddings; an output.weight bit-identical to token_embd.weight is tied too.
Only F32 tensors (the suite's); numpy and safetensors only, no torch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import ir_validate  # noqa: E402

EXPORTER_VERSION = 2  # v2: tokenizer files for "word" vocabularies; bos/eos/pad ids the GGUF lacks written as
#                       null, not left to the config class's defaults (weights unchanged from v1)


class Unsupported(Exception):
    """The GGUF's architecture has no transformers counterpart in this exporter."""


# ---------------------------------------------------------------- reading the GGUF

def read(path):
    """(metadata {key: value}, {tensor name: f32 array in numpy order [.., out, in]})."""
    meta_raw, tensors, data_start, raw, _ = ir_validate.read_gguf(path)
    meta = {k: v[1] for k, v in meta_raw.items()}
    W = {}
    for name, dims, typ, off in tensors:
        if typ != 0:
            raise Unsupported(f"{path}: {name} is not F32")
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype="<f4", count=n, offset=data_start + off).reshape(list(reversed(dims))).copy()
    return meta, W


class Arch:
    def __init__(self, meta):
        self.meta = meta
        self.name = meta.get("general.architecture")

    def __call__(self, key, default=KeyError):
        k = f"{self.name}.{key}"
        if k in self.meta:
            return self.meta[k]
        if default is KeyError:
            raise Unsupported(f"no {k}")
        return default


# ---------------------------------------------------------------- the layout transforms (and their inverses)

def unpermute_rope(w, n_head):
    """llama.cpp convert's LlamaModel.permute undone: interleaved-pair rows back to HF's half-split rows,
    per head (n_head = the tensor's head count: kv heads for k)."""
    out, rest = w.shape[0], w.shape[1:]
    return w.reshape(n_head, out // n_head // 2, 2, *rest).swapaxes(1, 2).reshape(w.shape).copy()


def permute_rope(w, n_head):
    """llama.cpp convert's LlamaModel.permute (HF half-split rows to interleaved pairs)."""
    out, rest = w.shape[0], w.shape[1:]
    return w.reshape(n_head, 2, out // n_head // 2, *rest).swapaxes(1, 2).reshape(w.shape).copy()


def interleave_qkv(w, n_head):
    """llama.cpp's q | k | v (all heads each) back to GPT-NeoX's per-head [q_h k_h v_h] rows."""
    q, k, v = np.split(w, 3, axis=0)
    hd = q.shape[0] // n_head
    parts = [x.reshape(n_head, hd, *x.shape[1:]) for x in (q, k, v)]
    return np.stack(parts, axis=1).reshape(w.shape).copy()


def deinterleave_qkv(w, n_head):
    """GPT-NeoX's per-head [q k v] rows to q | k | v (llama.cpp convert's GPTNeoXModel.modify_tensors)."""
    t = w.reshape(n_head, 3, w.shape[0] // n_head // 3, *w.shape[1:])
    return np.concatenate([t[:, i].reshape(-1, *w.shape[1:]) for i in range(3)], axis=0).copy()


def a_log_of(ssm_a):
    """A_log with -exp(A_log) == ssm_a in float32 where a float32 A_log does it: f32(log(-a)) and its float32
    neighbours (up to 64 ulps), else the neighbour nearest. Returns (A_log, number not exact)."""
    a = np.asarray(ssm_a, np.float32)
    flat = a.reshape(-1)
    out = np.empty_like(flat)
    miss = 0
    for i, x in enumerate(flat):
        target = np.float32(-x)
        if not target > 0:
            raise Unsupported(f"ssm_a has a non-negative entry {x}")
        c = np.float32(math.log(float(target)))
        best, err = c, abs(float(np.exp(c)) - float(target))
        lo = hi = c
        hit = np.exp(c) == target
        for _ in range(64):
            if hit:
                break
            lo, hi = np.nextafter(lo, np.float32(-np.inf)), np.nextafter(hi, np.float32(np.inf))
            for y in (lo, hi):
                e = abs(float(np.exp(y)) - float(target))
                if np.exp(y) == target:
                    best, hit = y, True
                    break
                if e < err:
                    best, err = y, e
        miss += not hit
        out[i] = best
    return out.reshape(a.shape), miss


def tied(W):
    return "output.weight" not in W or np.array_equal(W["output.weight"].view(np.uint32),
                                                      W["token_embd.weight"].view(np.uint32))


# ---------------------------------------------------------------- families

def _common(m, W):
    d = int(m("embedding_length"))
    return {"vocab_size": int(W["token_embd.weight"].shape[0]), "hidden_size": d,
            "num_hidden_layers": int(m("block_count")), "max_position_embeddings": int(m("context_length")),
            "tie_word_embeddings": tied(W), "torch_dtype": "float32", "use_cache": True,
            "bos_token_id": m.meta.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": m.meta.get("tokenizer.ggml.eos_token_id"),
            "pad_token_id": m.meta.get("tokenizer.ggml.padding_token_id")}


def _head(W, S):
    S["model.embed_tokens.weight"] = W["token_embd.weight"]
    S["model.norm.weight"] = W["output_norm.weight"]
    if "output.weight" in W:
        S["lm_head.weight"] = W["output.weight"]


def llama_family(m, W):
    """llama (with or without experts), mistral, qwen3."""
    a = m.name
    c = _common(m, W)
    H = int(m("attention.head_count"))
    KV = int(m("attention.head_count_kv", H))
    hd = int(m("attention.key_length", c["hidden_size"] // H))
    E = int(m("expert_count", 0))
    if m("rope.scaling.type", "none") not in ("none", None) or "rope_freqs.weight" in W:
        raise Unsupported(f"{a} with RoPE scaling")
    if a == "llama" and m("attention.sliding_window", 0):
        raise Unsupported("llama with a sliding window (no transformers Llama config carries one)")
    if a == "qwen3":
        if "blk.0.attn_q_norm.weight" not in W:
            raise Unsupported("qwen3 without q/k norms")
    elif "blk.0.attn_q_norm.weight" in W or "blk.0.attn_q.bias" in W:
        raise Unsupported(f"{a} with q/k norms or biases (not the llama graph)")
    neox = a == "qwen3"
    c.update({"intermediate_size": int(m("feed_forward_length")), "num_attention_heads": H, "num_key_value_heads": KV,
              "head_dim": hd, "hidden_act": "silu", "rms_norm_eps": float(m("attention.layer_norm_rms_epsilon")),
              "rope_parameters": {"rope_type": "default", "rope_theta": float(m("rope.freq_base", 10000.0))},
              "attention_bias": False, "attention_dropout": 0.0})
    if int(m("rope.dimension_count", hd)) != hd:
        raise Unsupported(f"{a} with partial RoPE")
    if E:
        cls, mtype = "MixtralForCausalLM", "mixtral"
        c.update({"num_local_experts": E, "num_experts_per_tok": int(m("expert_used_count")), "sliding_window": None,
                  "router_jitter_noise": 0.0, "output_router_logits": False})
    elif a == "mistral":
        cls, mtype = "MistralForCausalLM", "mistral"
        c["sliding_window"] = int(m("attention.sliding_window", 0)) or None
    elif a == "qwen3":
        cls, mtype = "Qwen3ForCausalLM", "qwen3"
        c.update({"use_sliding_window": False, "sliding_window": None})
    else:
        cls, mtype = "LlamaForCausalLM", "llama"
        c["mlp_bias"] = False
    S, transforms = {}, []
    if not neox:
        transforms.append("attn_q, attn_k rows un-permuted (llama.cpp convert's interleaved RoPE pairs to HF half-split)")
    for i in range(c["num_hidden_layers"]):
        b, p = f"blk.{i}.", f"model.layers.{i}."
        q, k = W[b + "attn_q.weight"], W[b + "attn_k.weight"]
        S[p + "self_attn.q_proj.weight"] = q if neox else unpermute_rope(q, H)
        S[p + "self_attn.k_proj.weight"] = k if neox else unpermute_rope(k, KV)
        S[p + "self_attn.v_proj.weight"] = W[b + "attn_v.weight"]
        S[p + "self_attn.o_proj.weight"] = W[b + "attn_output.weight"]
        if a == "qwen3":
            S[p + "self_attn.q_norm.weight"] = W[b + "attn_q_norm.weight"]
            S[p + "self_attn.k_norm.weight"] = W[b + "attn_k_norm.weight"]
        S[p + "input_layernorm.weight"] = W[b + "attn_norm.weight"]
        S[p + "post_attention_layernorm.weight"] = W[b + "ffn_norm.weight"]
        if E:
            S[p + "mlp.gate.weight"] = W[b + "ffn_gate_inp.weight"]
            S[p + "mlp.experts.gate_up_proj"] = np.concatenate([W[b + "ffn_gate_exps.weight"], W[b + "ffn_up_exps.weight"]],
                                                               axis=1)
            S[p + "mlp.experts.down_proj"] = W[b + "ffn_down_exps.weight"]
        else:
            S[p + "mlp.gate_proj.weight"] = W[b + "ffn_gate.weight"]
            S[p + "mlp.up_proj.weight"] = W[b + "ffn_up.weight"]
            S[p + "mlp.down_proj.weight"] = W[b + "ffn_down.weight"]
    _head(W, S)
    if E:
        transforms.append("experts: ffn_gate_exps | ffn_up_exps concatenated on the output axis into gate_up_proj")
    return cls, mtype, c, S, transforms


GEMMA3_SWA_PATTERN = 6  # llama.cpp b6000 set_swa_pattern(6): layer i local iff i % 6 < 5
GEMMA3_LOCAL_THETA = 10000.0
GEMMA3_NORMS = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
                "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm"}


def gemma3(m, W):
    c = _common(m, W)
    H = int(m("attention.head_count"))
    KV = int(m("attention.head_count_kv", H))
    hd = int(m("attention.key_length", c["hidden_size"] // H))
    L = c["num_hidden_layers"]
    if "rope_freqs.weight" in W:
        raise Unsupported("gemma3 with rope_freqs")
    stype = m("rope.scaling.type", "none")
    if stype not in ("none", "linear", None):
        raise Unsupported(f"gemma3 RoPE scaling {stype}")
    full_rope = {"rope_type": "default", "rope_theta": float(m("rope.freq_base"))}
    if stype == "linear" and float(m("rope.scaling.factor", 1.0)) != 1.0:
        full_rope = {"rope_type": "linear", "factor": float(m("rope.scaling.factor")), "rope_theta": float(m("rope.freq_base"))}
    if L == 62:
        raise Unsupported("gemma3 27B (scale 1/sqrt(d/heads)) not exported")
    c.update({"intermediate_size": int(m("feed_forward_length")), "num_attention_heads": H, "num_key_value_heads": KV,
              "head_dim": hd, "hidden_activation": "gelu_pytorch_tanh",
              "rms_norm_eps": float(m("attention.layer_norm_rms_epsilon")), "query_pre_attn_scalar": hd,
              "sliding_window": int(m("attention.sliding_window")),
              "layer_types": ["sliding_attention" if i % GEMMA3_SWA_PATTERN < GEMMA3_SWA_PATTERN - 1 else "full_attention"
                              for i in range(L)],
              "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": GEMMA3_LOCAL_THETA},
                                  "full_attention": full_rope},
              "final_logit_softcapping": None, "attn_logit_softcapping": None, "attention_bias": False,
              "attention_dropout": 0.0})
    S = {}

    def norm(x):
        return (x - np.float32(1)).astype(np.float32)

    for i in range(L):
        b, p = f"blk.{i}.", f"model.layers.{i}."
        S[p + "self_attn.q_proj.weight"] = W[b + "attn_q.weight"]
        S[p + "self_attn.k_proj.weight"] = W[b + "attn_k.weight"]
        S[p + "self_attn.v_proj.weight"] = W[b + "attn_v.weight"]
        S[p + "self_attn.o_proj.weight"] = W[b + "attn_output.weight"]
        S[p + "self_attn.q_norm.weight"] = norm(W[b + "attn_q_norm.weight"])
        S[p + "self_attn.k_norm.weight"] = norm(W[b + "attn_k_norm.weight"])
        for g, h in GEMMA3_NORMS.items():
            S[p + h + ".weight"] = norm(W[b + g + ".weight"])
        S[p + "mlp.gate_proj.weight"] = W[b + "ffn_gate.weight"]
        S[p + "mlp.up_proj.weight"] = W[b + "ffn_up.weight"]
        S[p + "mlp.down_proj.weight"] = W[b + "ffn_down.weight"]
    _head(W, S)
    S["model.norm.weight"] = norm(W["output_norm.weight"])
    c["text_config_only"] = True
    return ("Gemma3ForCausalLM", "gemma3_text", c, S,
            ["every RMSNorm weight written as s - 1 (llama.cpp convert stores 1 + w; HF computes 1 + w)"])


def gptneox(m, W):
    c = _common(m, W)
    H = int(m("attention.head_count"))
    hd = c["hidden_size"] // H
    if m("rope.scaling.type", "none") not in ("none", None):
        raise Unsupported("gptneox with RoPE scaling")
    n_rot = int(m("rope.dimension_count", hd))
    c.update({"intermediate_size": int(m("feed_forward_length")), "num_attention_heads": H, "hidden_act": "gelu",
              "layer_norm_eps": float(m("attention.layer_norm_epsilon")),
              "use_parallel_residual": bool(m("use_parallel_residual", True)),
              "rope_parameters": {"rope_type": "default", "rope_theta": float(m("rope.freq_base", 10000.0)),
                                  "partial_rotary_factor": n_rot / hd},
              "attention_bias": True, "attention_dropout": 0.0, "hidden_dropout": 0.0})
    S = {}
    for i in range(c["num_hidden_layers"]):
        b, p = f"blk.{i}.", f"gpt_neox.layers.{i}."
        S[p + "input_layernorm.weight"] = W[b + "attn_norm.weight"]
        S[p + "input_layernorm.bias"] = W[b + "attn_norm.bias"]
        S[p + "post_attention_layernorm.weight"] = W[b + "ffn_norm.weight"]
        S[p + "post_attention_layernorm.bias"] = W[b + "ffn_norm.bias"]
        S[p + "attention.query_key_value.weight"] = interleave_qkv(W[b + "attn_qkv.weight"], H)
        S[p + "attention.query_key_value.bias"] = interleave_qkv(W[b + "attn_qkv.bias"], H)
        S[p + "attention.dense.weight"] = W[b + "attn_output.weight"]
        S[p + "attention.dense.bias"] = W[b + "attn_output.bias"]
        S[p + "mlp.dense_h_to_4h.weight"] = W[b + "ffn_up.weight"]
        S[p + "mlp.dense_h_to_4h.bias"] = W[b + "ffn_up.bias"]
        S[p + "mlp.dense_4h_to_h.weight"] = W[b + "ffn_down.weight"]
        S[p + "mlp.dense_4h_to_h.bias"] = W[b + "ffn_down.bias"]
    S["gpt_neox.embed_in.weight"] = W["token_embd.weight"]
    S["gpt_neox.final_layer_norm.weight"] = W["output_norm.weight"]
    S["gpt_neox.final_layer_norm.bias"] = W["output_norm.bias"]
    if "output.weight" in W:
        S["lm_head.weight"] = W["output.weight"]
    return ("GPTNeoXForCausalLM", "gpt_neox", c, S,
            ["attn_qkv re-interleaved per head (inverse of llama.cpp convert's q | k | v)"])


def gpt2(m, W):
    c0 = _common(m, W)
    H = int(m("attention.head_count"))
    c = {"vocab_size": c0["vocab_size"], "n_embd": c0["hidden_size"], "n_layer": c0["num_hidden_layers"], "n_head": H,
         "n_positions": c0["max_position_embeddings"], "n_inner": int(m("feed_forward_length")),
         "activation_function": "gelu_new", "layer_norm_epsilon": float(m("attention.layer_norm_epsilon")),
         "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0, "scale_attn_weights": True,
         "tie_word_embeddings": c0["tie_word_embeddings"], "torch_dtype": "float32", "use_cache": True,
         "bos_token_id": c0["bos_token_id"], "eos_token_id": c0["eos_token_id"], "pad_token_id": c0["pad_token_id"]}
    S = {}
    for i in range(c["n_layer"]):
        b, p = f"blk.{i}.", f"transformer.h.{i}."
        S[p + "ln_1.weight"], S[p + "ln_1.bias"] = W[b + "attn_norm.weight"], W[b + "attn_norm.bias"]
        S[p + "attn.c_attn.weight"], S[p + "attn.c_attn.bias"] = W[b + "attn_qkv.weight"].T.copy(), W[b + "attn_qkv.bias"]
        S[p + "attn.c_proj.weight"] = W[b + "attn_output.weight"].T.copy()
        S[p + "attn.c_proj.bias"] = W[b + "attn_output.bias"]
        S[p + "ln_2.weight"], S[p + "ln_2.bias"] = W[b + "ffn_norm.weight"], W[b + "ffn_norm.bias"]
        S[p + "mlp.c_fc.weight"], S[p + "mlp.c_fc.bias"] = W[b + "ffn_up.weight"].T.copy(), W[b + "ffn_up.bias"]
        S[p + "mlp.c_proj.weight"], S[p + "mlp.c_proj.bias"] = W[b + "ffn_down.weight"].T.copy(), W[b + "ffn_down.bias"]
    S["transformer.wte.weight"] = W["token_embd.weight"]
    S["transformer.wpe.weight"] = W["position_embd.weight"]
    S["transformer.ln_f.weight"], S["transformer.ln_f.bias"] = W["output_norm.weight"], W["output_norm.bias"]
    if "output.weight" in W:
        S["lm_head.weight"] = W["output.weight"]
    return "GPT2LMHeadModel", "gpt2", c, S, ["Conv1D weights transposed ([out, in] to [in, out])"]


def mamba(m, W):
    c0 = _common(m, W)
    d = c0["hidden_size"]
    Di, N = int(m("ssm.inner_size")), int(m("ssm.state_size"))
    R, K = int(m("ssm.time_step_rank")), int(m("ssm.conv_kernel"))
    if m("ssm.dt_b_c_rms", False) or "blk.0.ssm_dt_norm.weight" in W:
        raise Unsupported("Falcon-Mamba (dt/B/C norms)")
    if Di % d:
        raise Unsupported("mamba inner size not a multiple of d")
    c = {**c0, "state_size": N, "expand": Di // d, "intermediate_size": Di, "conv_kernel": K, "time_step_rank": R,
         "use_bias": False, "use_conv_bias": "blk.0.ssm_conv1d.bias" in W,
         "layer_norm_epsilon": float(m("attention.layer_norm_rms_epsilon")), "residual_in_fp32": True,
         "hidden_act": "silu", "use_mambapy": False}
    c.pop("max_position_embeddings")
    S, misses = {}, 0
    for i in range(c["num_hidden_layers"]):
        b, p = f"blk.{i}.", f"backbone.layers.{i}."
        S[p + "norm.weight"] = W[b + "attn_norm.weight"]
        S[p + "mixer.in_proj.weight"] = W[b + "ssm_in.weight"]
        S[p + "mixer.conv1d.weight"] = W[b + "ssm_conv1d.weight"].reshape(Di, 1, K)
        if c["use_conv_bias"]:
            S[p + "mixer.conv1d.bias"] = W[b + "ssm_conv1d.bias"]
        S[p + "mixer.x_proj.weight"] = W[b + "ssm_x.weight"]
        S[p + "mixer.dt_proj.weight"] = W[b + "ssm_dt.weight"]
        S[p + "mixer.dt_proj.bias"] = W[b + "ssm_dt.bias"]
        S[p + "mixer.A_log"], miss = a_log_of(W[b + "ssm_a"])
        misses += miss
        S[p + "mixer.D"] = W[b + "ssm_d"]
        S[p + "mixer.out_proj.weight"] = W[b + "ssm_out.weight"]
    S["backbone.embeddings.weight"] = W["token_embd.weight"]
    S["backbone.norm_f.weight"] = W["output_norm.weight"]
    if "output.weight" in W:
        S["lm_head.weight"] = W["output.weight"]
    return ("MambaForCausalLM", "mamba", c, S,
            [f"A_log = the float32 whose -exp is ssm_a (numpy float32 exp; {misses} entries without an exact one)",
             "conv1d [Di, K] -> [Di, 1, K]"])


FAMILIES = {"llama": llama_family, "mistral": llama_family, "qwen3": llama_family, "gemma3": gemma3,
            "gptneox": gptneox, "gpt2": gpt2, "mamba": mamba}


def convert(meta, W):
    """(class name, model_type, config dict, {HF name: f32 array}, [transforms])."""
    arch = meta.get("general.architecture")
    if arch not in FAMILIES:
        raise Unsupported(f"general.architecture {arch!r} has no transformers export here")
    cls, mtype, c, S, tr = FAMILIES[arch](Arch(meta), W)
    c.pop("text_config_only", None)
    # None is written as null where the class default would otherwise stand in: a GGUF without special ids
    # gets none (v1 dropped them, so e.g. Gemma3TextConfig's eos 1 and MistralConfig's eos 2 appeared).
    c = {k: v for k, v in c.items() if v is not None or k in ("sliding_window", "final_logit_softcapping",
                                                               "attn_logit_softcapping", "bos_token_id",
                                                               "eos_token_id", "pad_token_id")}
    c.update({"architectures": [cls], "model_type": mtype})
    S = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in S.items()}
    return cls, mtype, c, S, tr


# ---------------------------------------------------------------- the inverse (for the round-trip test)

def back(meta, c, S):
    """The GGUF tensors {name: f32 array} the HF state S stands for (llama.cpp convert applied to S)."""
    arch = meta["general.architecture"]
    W = {}
    L = c.get("num_hidden_layers", c.get("n_layer"))
    if arch in ("llama", "mistral", "qwen3"):
        H, KV = c["num_attention_heads"], c["num_key_value_heads"]
        neox = arch == "qwen3"
        for i in range(L):
            b, p = f"blk.{i}.", f"model.layers.{i}."
            q, k = S[p + "self_attn.q_proj.weight"], S[p + "self_attn.k_proj.weight"]
            W[b + "attn_q.weight"] = q if neox else permute_rope(q, H)
            W[b + "attn_k.weight"] = k if neox else permute_rope(k, KV)
            W[b + "attn_v.weight"] = S[p + "self_attn.v_proj.weight"]
            W[b + "attn_output.weight"] = S[p + "self_attn.o_proj.weight"]
            if arch == "qwen3":
                W[b + "attn_q_norm.weight"] = S[p + "self_attn.q_norm.weight"]
                W[b + "attn_k_norm.weight"] = S[p + "self_attn.k_norm.weight"]
            W[b + "attn_norm.weight"] = S[p + "input_layernorm.weight"]
            W[b + "ffn_norm.weight"] = S[p + "post_attention_layernorm.weight"]
            if c.get("num_local_experts"):
                gu = S[p + "mlp.experts.gate_up_proj"]
                W[b + "ffn_gate_inp.weight"] = S[p + "mlp.gate.weight"]
                W[b + "ffn_gate_exps.weight"], W[b + "ffn_up_exps.weight"] = np.split(gu, 2, axis=1)
                W[b + "ffn_down_exps.weight"] = S[p + "mlp.experts.down_proj"]
            else:
                W[b + "ffn_gate.weight"] = S[p + "mlp.gate_proj.weight"]
                W[b + "ffn_up.weight"] = S[p + "mlp.up_proj.weight"]
                W[b + "ffn_down.weight"] = S[p + "mlp.down_proj.weight"]
        W["token_embd.weight"], W["output_norm.weight"] = S["model.embed_tokens.weight"], S["model.norm.weight"]
    elif arch == "gemma3":
        def norm(x):  # what HF computes: 1 + w, in float32
            return (np.float32(1) + x).astype(np.float32)
        for i in range(L):
            b, p = f"blk.{i}.", f"model.layers.{i}."
            for g, h in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                W[b + g + ".weight"] = S[p + "self_attn." + h + ".weight"]
            W[b + "attn_q_norm.weight"] = norm(S[p + "self_attn.q_norm.weight"])
            W[b + "attn_k_norm.weight"] = norm(S[p + "self_attn.k_norm.weight"])
            for g, h in GEMMA3_NORMS.items():
                W[b + g + ".weight"] = norm(S[p + h + ".weight"])
            for g in ("gate", "up", "down"):
                W[b + f"ffn_{g}.weight"] = S[p + f"mlp.{g}_proj.weight"]
        W["token_embd.weight"], W["output_norm.weight"] = S["model.embed_tokens.weight"], norm(S["model.norm.weight"])
    elif arch == "gptneox":
        H = c["num_attention_heads"]
        for i in range(L):
            b, p = f"blk.{i}.", f"gpt_neox.layers.{i}."
            W[b + "attn_norm.weight"], W[b + "attn_norm.bias"] = S[p + "input_layernorm.weight"], S[p + "input_layernorm.bias"]
            W[b + "ffn_norm.weight"] = S[p + "post_attention_layernorm.weight"]
            W[b + "ffn_norm.bias"] = S[p + "post_attention_layernorm.bias"]
            W[b + "attn_qkv.weight"] = deinterleave_qkv(S[p + "attention.query_key_value.weight"], H)
            W[b + "attn_qkv.bias"] = deinterleave_qkv(S[p + "attention.query_key_value.bias"], H)
            W[b + "attn_output.weight"], W[b + "attn_output.bias"] = S[p + "attention.dense.weight"], S[p + "attention.dense.bias"]
            W[b + "ffn_up.weight"], W[b + "ffn_up.bias"] = S[p + "mlp.dense_h_to_4h.weight"], S[p + "mlp.dense_h_to_4h.bias"]
            W[b + "ffn_down.weight"], W[b + "ffn_down.bias"] = S[p + "mlp.dense_4h_to_h.weight"], S[p + "mlp.dense_4h_to_h.bias"]
        W["token_embd.weight"] = S["gpt_neox.embed_in.weight"]
        W["output_norm.weight"], W["output_norm.bias"] = S["gpt_neox.final_layer_norm.weight"], S["gpt_neox.final_layer_norm.bias"]
    elif arch == "gpt2":
        for i in range(L):
            b, p = f"blk.{i}.", f"transformer.h.{i}."
            W[b + "attn_norm.weight"], W[b + "attn_norm.bias"] = S[p + "ln_1.weight"], S[p + "ln_1.bias"]
            W[b + "attn_qkv.weight"], W[b + "attn_qkv.bias"] = S[p + "attn.c_attn.weight"].T, S[p + "attn.c_attn.bias"]
            W[b + "attn_output.weight"], W[b + "attn_output.bias"] = S[p + "attn.c_proj.weight"].T, S[p + "attn.c_proj.bias"]
            W[b + "ffn_norm.weight"], W[b + "ffn_norm.bias"] = S[p + "ln_2.weight"], S[p + "ln_2.bias"]
            W[b + "ffn_up.weight"], W[b + "ffn_up.bias"] = S[p + "mlp.c_fc.weight"].T, S[p + "mlp.c_fc.bias"]
            W[b + "ffn_down.weight"], W[b + "ffn_down.bias"] = S[p + "mlp.c_proj.weight"].T, S[p + "mlp.c_proj.bias"]
        W["token_embd.weight"], W["position_embd.weight"] = S["transformer.wte.weight"], S["transformer.wpe.weight"]
        W["output_norm.weight"], W["output_norm.bias"] = S["transformer.ln_f.weight"], S["transformer.ln_f.bias"]
    elif arch == "mamba":
        for i in range(L):
            b, p = f"blk.{i}.", f"backbone.layers.{i}."
            W[b + "attn_norm.weight"] = S[p + "norm.weight"]
            W[b + "ssm_in.weight"] = S[p + "mixer.in_proj.weight"]
            cw = S[p + "mixer.conv1d.weight"]
            W[b + "ssm_conv1d.weight"] = cw.reshape(cw.shape[0], cw.shape[2])
            if p + "mixer.conv1d.bias" in S:
                W[b + "ssm_conv1d.bias"] = S[p + "mixer.conv1d.bias"]
            W[b + "ssm_x.weight"] = S[p + "mixer.x_proj.weight"]
            W[b + "ssm_dt.weight"], W[b + "ssm_dt.bias"] = S[p + "mixer.dt_proj.weight"], S[p + "mixer.dt_proj.bias"]
            W[b + "ssm_a"] = (-np.exp(S[p + "mixer.A_log"])).astype(np.float32)  # what HF computes: -exp(A_log)
            W[b + "ssm_d"] = S[p + "mixer.D"]
            W[b + "ssm_out.weight"] = S[p + "mixer.out_proj.weight"]
        W["token_embd.weight"], W["output_norm.weight"] = S["backbone.embeddings.weight"], S["backbone.norm_f.weight"]
    if "lm_head.weight" in S:
        W["output.weight"] = S["lm_head.weight"]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}


def round_trip(gguf):
    """{tensor: values not bit-identical} after GGUF -> HF -> GGUF ({} is exact), plus the tensors only one
    side has. A tied output.weight the exporter dropped is compared with token_embd.weight."""
    meta, W = read(gguf)
    _, _, c, S, _ = convert(meta, W)
    W2 = back(meta, c, S)
    if "output.weight" in W and "output.weight" not in W2:
        W2["output.weight"] = W2["token_embd.weight"]
    bad = {}
    for k in sorted(set(W) | set(W2)):
        if k not in W or k not in W2:
            bad[k] = "only in " + ("the GGUF" if k in W else "the export")
            continue
        a, b = W[k], W2[k]
        if a.shape != b.shape:
            bad[k] = f"shape {a.shape} != {b.shape}"
        elif not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            bad[k] = int(np.sum(a.view(np.uint32) != b.view(np.uint32)))
    return bad


# ---------------------------------------------------------------- the tokenizer

def tokenizer_files(meta, context_length=None):
    """{file name: JSON text} of an HF tokenizer for the GGUF's vocabulary, or {} where this exporter has no
    translation. tokenizer.ggml.model "word" (IR_SPEC 8.1.4's paperkiln/legacy-word: whitespace-separated
    words, an unknown word to the unknown id) is a tokenizers WordLevel model with a WhitespaceSplit
    pre-tokenizer; the special ids are the GGUF's own. Engines that need a tokenizer beside the weights
    (ONNX Runtime GenAI's builder loads one with AutoTokenizer) read these; the suite feeds ids."""
    if meta.get("tokenizer.ggml.model") != "word":
        return {}
    toks = list(meta["tokenizer.ggml.tokens"])
    unk = int(meta.get("tokenizer.ggml.unknown_token_id", 0))
    tok = {"version": "1.0", "truncation": None, "padding": None,
           "added_tokens": [{"id": unk, "content": toks[unk], "single_word": False, "lstrip": False, "rstrip": False,
                             "normalized": False, "special": True}],
           "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None, "decoder": None,
           "model": {"type": "WordLevel", "vocab": {t: i for i, t in enumerate(toks)}, "unk_token": toks[unk]}}
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": toks[unk], "clean_up_tokenization_spaces": False}
    for key, field in (("bos_token", "bos_token_id"), ("eos_token", "eos_token_id"), ("pad_token", "padding_token_id")):
        if f"tokenizer.ggml.{field}" in meta:
            cfg[key] = toks[int(meta[f"tokenizer.ggml.{field}"])]
    if context_length:
        cfg["model_max_length"] = int(context_length)
    return {"tokenizer.json": json.dumps(tok, indent=1) + "\n", "tokenizer_config.json": json.dumps(cfg, indent=1) + "\n"}


# ---------------------------------------------------------------- writing

def export(gguf, out_dir):
    """Write OUT_DIR/model.safetensors, config.json, the tokenizer (tokenizer_files) and keramos_export.json;
    return the config."""
    from safetensors.numpy import save_file
    meta, W = read(gguf)
    cls, mtype, c, S, transforms = convert(meta, W)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(S, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")
    tok = tokenizer_files(meta, c.get("max_position_embeddings", c.get("n_positions")))
    for name, text in tok.items():
        (out / name).write_text(text, encoding="utf-8")
    rt = round_trip(gguf)
    (out / "keramos_export.json").write_text(json.dumps({
        "exporter": f"keramos adapters/common/gguf_to_hf.py v{EXPORTER_VERSION}", "gguf": str(gguf),
        "general.architecture": meta.get("general.architecture"), "class": cls, "transforms": transforms,
        "tokenizer": sorted(tok) or f"none (tokenizer.ggml.model {meta.get('tokenizer.ggml.model')!r} not translated)",
        "round_trip_not_exact": rt}, indent=1) + "\n", encoding="utf-8")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--check", action="store_true", help="the round trip only")
    a = ap.parse_args(argv)
    if a.check or not a.out:
        bad = round_trip(a.gguf)
        print("round trip exact" if not bad else f"round trip NOT exact: {bad}")
        return 1 if bad else 0
    c = export(a.gguf, a.out)
    print(f"{c['architectures'][0]} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
