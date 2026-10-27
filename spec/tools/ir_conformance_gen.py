"""Builds conformance/ (IR_SPEC.md section 11): tools/ir_conformance.py --regenerate.

The pk_ models (conformance/models/pk_*/) are paperkiln's: its tools/ir_conformance_mtstudio.py
--build-models OUT trains them with mtstudio (tiny word or BPE vocabularies, d 16, 30-40 steps) into
OUT/<name>/ (<name>.gguf with the IR, <name>.safetensors and mtstudio.json, the run's spec); copy them
into conformance/models/. --regenerate needs numpy only: it writes the third-party models (random weights,
documents from tools/gguf_ir.py's builders), every case's expected values from tools/ir_numpy.py, the
refusal cases and manifest.json. Deterministic: the same models give the same files.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402
import ir_validate  # noqa: E402
from ir_conformance import SUITE, enc, evict_allowed, load_model, logprobs_of, write_gguf  # noqa: E402

# The profile of each paperkiln model of the suite (IR_SPEC 2.4).
PK_PROFILES = {
    "pk_llama": "core", "pk_moe": "core", "pk_gpt2": "core", "pk_gpt2_swa": "core", "pk_kimi": "state",
    "pk_srd": "state", "pk_attnres": "core", "pk_flex_highway": "core", "pk_flex_expr": "core",
    "pk_mamba": "state", "pk_recursive": "recursion", "pk_mor": "recursion", "pk_llama_attnres": "core",
    "pk_llama_attnres_rec": "recursion", "pk_state": "state", "pk_recurrence": "recurrence", "pk_flex_t5": "core",
    "pk_flex_shaw": "core", "pk_flex_mdha": "thirdparty"}
# Models added after the suite's first release draw their prompts from a stream of their own, so the cases of
# every earlier model stay byte-identical when the suite is regenerated.
OWN_STREAM = {"pk_recurrence", "pk_flex_t5", "tp_llama_hd16", "tp_mistral_hd16", "tp_qwen3_hd16", "pk_flex_shaw",
              "pk_flex_mdha", "tp_phi3", "tp_gemma2", "tp_qwen2moe", "tp_qwen3moe", "tp_deepseek2"}


# ---------------------------------------------------------------- third-party models (numpy)

TP = {
    "tp_qwen3": {"general.architecture": "qwen3", "qwen3.embedding_length": 16, "qwen3.block_count": 2,
                 "qwen3.attention.head_count": 4, "qwen3.attention.head_count_kv": 2, "qwen3.attention.key_length": 4,
                 "qwen3.context_length": 64, "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
                 "qwen3.rope.freq_base": 1000000.0, "qwen3.feed_forward_length": 32},
    "tp_mistral": {"general.architecture": "mistral", "mistral.embedding_length": 16, "mistral.block_count": 2,
                   "mistral.attention.head_count": 4, "mistral.attention.head_count_kv": 2,
                   "mistral.context_length": 64, "mistral.attention.layer_norm_rms_epsilon": 1e-5,
                   "mistral.rope.freq_base": 10000.0, "mistral.attention.sliding_window": 4,
                   "mistral.feed_forward_length": 32},
    "tp_gemma3": {"general.architecture": "gemma3", "gemma3.embedding_length": 16, "gemma3.block_count": 6,
                  "gemma3.attention.head_count": 2, "gemma3.attention.head_count_kv": 1,
                  "gemma3.attention.key_length": 8, "gemma3.context_length": 64,
                  "gemma3.attention.layer_norm_rms_epsilon": 1e-6, "gemma3.rope.freq_base": 1000000.0,
                  "gemma3.attention.sliding_window": 4, "gemma3.rope.scaling.type": "linear",
                  "gemma3.rope.scaling.factor": 2.0, "gemma3.feed_forward_length": 32},
    "tp_gptneox": {"general.architecture": "gptneox", "gptneox.embedding_length": 16, "gptneox.block_count": 2,
                   "gptneox.attention.head_count": 2, "gptneox.context_length": 64,
                   "gptneox.attention.layer_norm_epsilon": 1e-5, "gptneox.rope.dimension_count": 4,
                   "gptneox.rope.freq_base": 10000.0, "gptneox.use_parallel_residual": True,
                   "gptneox.feed_forward_length": 32},
    "tp_mamba1": {"general.architecture": "mamba", "mamba.embedding_length": 16, "mamba.block_count": 2,
                  "mamba.context_length": 64, "mamba.ssm.conv_kernel": 4, "mamba.ssm.inner_size": 32,
                  "mamba.ssm.state_size": 8, "mamba.ssm.time_step_rank": 2,
                  "mamba.attention.layer_norm_rms_epsilon": 1e-5},
    # v1.3.4: head_dim 16, the smallest head size onnxruntime's CPU GroupQueryAttention runs with fused rotary
    # (adapters/ortgenai/FINDINGS.txt, finding 1); every other attention model of the suite has head_dim 4 or 8.
    # Llama (MHA), Mistral (GQA 2:1, window 4) and Qwen3 (GQA 2:1, head_dim decoupled from d / heads, as in
    # the released Qwen3 checkpoints, so the file carries value_length as theirs do).
    "tp_llama_hd16": {"general.architecture": "llama", "llama.embedding_length": 32, "llama.block_count": 2,
                      "llama.attention.head_count": 2, "llama.attention.head_count_kv": 2,
                      "llama.context_length": 64, "llama.attention.layer_norm_rms_epsilon": 1e-5,
                      "llama.rope.freq_base": 10000.0, "llama.feed_forward_length": 64},
    "tp_mistral_hd16": {"general.architecture": "mistral", "mistral.embedding_length": 32, "mistral.block_count": 2,
                        "mistral.attention.head_count": 2, "mistral.attention.head_count_kv": 1,
                        "mistral.context_length": 64, "mistral.attention.layer_norm_rms_epsilon": 1e-5,
                        "mistral.rope.freq_base": 10000.0, "mistral.attention.sliding_window": 4,
                        "mistral.feed_forward_length": 64},
    "tp_qwen3_hd16": {"general.architecture": "qwen3", "qwen3.embedding_length": 32, "qwen3.block_count": 2,
                      "qwen3.attention.head_count": 4, "qwen3.attention.head_count_kv": 2,
                      "qwen3.attention.key_length": 16, "qwen3.attention.value_length": 16,
                      "qwen3.context_length": 64, "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
                      "qwen3.rope.freq_base": 1000000.0, "qwen3.feed_forward_length": 64},
    # v1.5.0 (LOCAL_LLM_WORK.md E4). Phi-3 with LongRoPE: fused qkv and gate | up, partial NEOX rotary (4 of 8
    # dims), original_ctx 6, so the model case, decode, chunks and truncate traces cross from the short factors
    # to the long ones (5.14), attn_factor 1.25, tied head.
    "tp_phi3": {"general.architecture": "phi3", "phi3.embedding_length": 16, "phi3.block_count": 2,
                "phi3.attention.head_count": 2, "phi3.attention.head_count_kv": 1, "phi3.context_length": 64,
                "phi3.attention.layer_norm_rms_epsilon": 1e-5, "phi3.rope.dimension_count": 4,
                "phi3.rope.freq_base": 10000.0, "phi3.rope.scaling.original_context_length": 6,
                "phi3.rope.scaling.attn_factor": 1.25, "phi3.attention.sliding_window": 0,
                "phi3.feed_forward_length": 32},
    # Gemma 2: local layers 0 and 2 (window 4), attention and final logit soft-capping small enough to bind.
    "tp_gemma2": {"general.architecture": "gemma2", "gemma2.embedding_length": 16, "gemma2.block_count": 4,
                  "gemma2.attention.head_count": 2, "gemma2.attention.head_count_kv": 1,
                  "gemma2.attention.key_length": 8, "gemma2.attention.value_length": 8, "gemma2.context_length": 64,
                  "gemma2.attention.layer_norm_rms_epsilon": 1e-6, "gemma2.attention.sliding_window": 4,
                  "gemma2.attn_logit_softcapping": 1.5, "gemma2.final_logit_softcapping": 2.5,
                  "gemma2.feed_forward_length": 32},
    # Qwen-MoE: raw softmax gates with a sigmoid-gated shared expert (qwen2moe); renormalised gates and q/k
    # norms (qwen3moe).
    "tp_qwen2moe": {"general.architecture": "qwen2moe", "qwen2moe.embedding_length": 16, "qwen2moe.block_count": 2,
                    "qwen2moe.attention.head_count": 4, "qwen2moe.attention.head_count_kv": 2,
                    "qwen2moe.context_length": 64, "qwen2moe.attention.layer_norm_rms_epsilon": 1e-6,
                    "qwen2moe.rope.freq_base": 1000000.0, "qwen2moe.expert_count": 4, "qwen2moe.expert_used_count": 2,
                    "qwen2moe.expert_feed_forward_length": 8, "qwen2moe.expert_shared_feed_forward_length": 24,
                    "qwen2moe.feed_forward_length": 32},
    "tp_qwen3moe": {"general.architecture": "qwen3moe", "qwen3moe.embedding_length": 16, "qwen3moe.block_count": 2,
                    "qwen3moe.attention.head_count": 4, "qwen3moe.attention.head_count_kv": 2,
                    "qwen3moe.attention.key_length": 4, "qwen3moe.context_length": 64,
                    "qwen3moe.attention.layer_norm_rms_epsilon": 1e-6, "qwen3moe.rope.freq_base": 1000000.0,
                    "qwen3moe.expert_count": 4, "qwen3moe.expert_used_count": 2,
                    "qwen3moe.expert_feed_forward_length": 8, "qwen3moe.feed_forward_length": 32},
    # DeepSeek-V2's multi-head latent attention (5.20): q through the low-rank q_a / q_b, latent rank 6, rope 4,
    # nope 4, v 3 (every width distinct from the others but nope = rope); MoE with raw gates, scale 2.5 and one
    # shared expert.
    "tp_deepseek2": {"general.architecture": "deepseek2", "deepseek2.embedding_length": 16,
                     "deepseek2.block_count": 2, "deepseek2.attention.head_count": 2,
                     "deepseek2.attention.head_count_kv": 1, "deepseek2.context_length": 64,
                     "deepseek2.attention.layer_norm_rms_epsilon": 1e-6, "deepseek2.rope.freq_base": 10000.0,
                     "deepseek2.rope.dimension_count": 4, "deepseek2.attention.q_lora_rank": 8,
                     "deepseek2.attention.kv_lora_rank": 6, "deepseek2.attention.key_length": 10,
                     "deepseek2.attention.value_length": 6, "deepseek2.attention.key_length_mla": 8,
                     "deepseek2.attention.value_length_mla": 3, "deepseek2.leading_dense_block_count": 0,
                     "deepseek2.expert_count": 4, "deepseek2.expert_used_count": 2,
                     "deepseek2.expert_feed_forward_length": 8, "deepseek2.expert_shared_count": 1,
                     "deepseek2.expert_weights_scale": 2.5, "deepseek2.expert_weights_norm": False,
                     "deepseek2.expert_gating_func": 1, "deepseek2.feed_forward_length": 32},
}
V15 = ("tp_phi3", "tp_gemma2", "tp_qwen2moe", "tp_qwen3moe", "tp_deepseek2")
TP_VOCAB = 24


def _tp_shapes(meta):
    a = meta["general.architecture"]
    g = {k[len(a) + 1:]: v for k, v in meta.items() if k.startswith(a + ".")}
    d, L, V = g["embedding_length"], g["block_count"], TP_VOCAB
    sh = {"token_embd.weight": [V, d]}
    if a == "mamba":
        Di, N, R, K = g["ssm.inner_size"], g["ssm.state_size"], g["ssm.time_step_rank"], g["ssm.conv_kernel"]
        for i in range(L):
            b = f"blk.{i}."
            sh.update({b + "attn_norm.weight": [d], b + "ssm_in.weight": [2 * Di, d], b + "ssm_conv1d.weight": [Di, K],
                       b + "ssm_conv1d.bias": [Di], b + "ssm_x.weight": [R + 2 * N, Di], b + "ssm_dt.weight": [Di, R],
                       b + "ssm_dt.bias": [Di], b + "ssm_a": [Di, N], b + "ssm_d": [Di], b + "ssm_out.weight": [d, Di]})
        sh["output_norm.weight"] = [d]
        return sh
    Hq = g["attention.head_count"]
    KV = g.get("attention.head_count_kv", Hq)
    hd = g.get("attention.key_length", d // Hq)
    ff = g["feed_forward_length"]
    if a in ("phi3", "deepseek2", "qwen2moe", "qwen3moe"):
        return _tp_shapes_v15(a, g, d, L, V, Hq, KV, hd, ff, sh)
    for i in range(L):
        b = f"blk.{i}."
        if a == "gptneox":
            sh.update({b + "attn_norm.weight": [d], b + "attn_norm.bias": [d],
                       b + "attn_qkv.weight": [(Hq + 2 * KV) * hd, d], b + "attn_qkv.bias": [(Hq + 2 * KV) * hd],
                       b + "attn_output.weight": [d, Hq * hd], b + "attn_output.bias": [d],
                       b + "ffn_norm.weight": [d], b + "ffn_norm.bias": [d], b + "ffn_up.weight": [ff, d],
                       b + "ffn_up.bias": [ff], b + "ffn_down.weight": [d, ff], b + "ffn_down.bias": [d]})
            continue
        sh.update({b + "attn_norm.weight": [d], b + "attn_q.weight": [Hq * hd, d], b + "attn_k.weight": [KV * hd, d],
                   b + "attn_v.weight": [KV * hd, d], b + "attn_output.weight": [d, Hq * hd],
                   b + "ffn_norm.weight": [d], b + "ffn_gate.weight": [ff, d], b + "ffn_up.weight": [ff, d],
                   b + "ffn_down.weight": [d, ff]})
        if a in ("qwen3", "gemma3"):
            sh.update({b + "attn_q_norm.weight": [hd], b + "attn_k_norm.weight": [hd]})
        if a in ("gemma3", "gemma2"):
            sh.update({b + "post_attention_norm.weight": [d], b + "post_ffw_norm.weight": [d]})
    if a == "gptneox":
        sh.update({"output_norm.weight": [d], "output_norm.bias": [d], "output.weight": [V, d]})
    else:
        sh["output_norm.weight"] = [d]
    return sh


def _tp_shapes_v15(a, g, d, L, V, Hq, KV, hd, ff, sh):
    """v1.5.0: phi3 (fused qkv and gate | up, LongRoPE's two factor sets, tied head), qwen2moe / qwen3moe
    (llama.cpp's expert layout, qwen2moe's shared expert and its gate), deepseek2 (MLA's tensors, MoE with a
    shared expert)."""
    for i in range(L):
        b = f"blk.{i}."
        sh.update({b + "attn_norm.weight": [d], b + "ffn_norm.weight": [d]})
        if a == "phi3":
            sh.update({b + "attn_qkv.weight": [(Hq + 2 * KV) * hd, d], b + "attn_output.weight": [d, Hq * hd],
                       b + "ffn_up.weight": [2 * ff, d], b + "ffn_down.weight": [d, ff]})
            continue
        if a in ("qwen2moe", "qwen3moe"):
            sh.update({b + "attn_q.weight": [Hq * hd, d], b + "attn_k.weight": [KV * hd, d],
                       b + "attn_v.weight": [KV * hd, d], b + "attn_output.weight": [d, Hq * hd]})
            if a == "qwen2moe":
                sh.update({b + "attn_q.bias": [Hq * hd], b + "attn_k.bias": [KV * hd], b + "attn_v.bias": [KV * hd]})
                fs = g["expert_shared_feed_forward_length"]
                sh.update({b + "ffn_gate_shexp.weight": [fs, d], b + "ffn_up_shexp.weight": [fs, d],
                           b + "ffn_down_shexp.weight": [d, fs], b + "ffn_gate_inp_shexp.weight": [1, d]})
            else:
                sh.update({b + "attn_q_norm.weight": [hd], b + "attn_k_norm.weight": [hd]})
        if a == "deepseek2":
            dr, r = g["rope.dimension_count"], g["attention.kv_lora_rank"]
            dn, dv, qr = g["attention.key_length_mla"] - dr, g["attention.value_length_mla"], g["attention.q_lora_rank"]
            fs = g["expert_feed_forward_length"] * g["expert_shared_count"]
            sh.update({b + "attn_q_a.weight": [qr, d], b + "attn_q_a_norm.weight": [qr],
                       b + "attn_q_b.weight": [Hq * (dn + dr), qr], b + "attn_kv_a_mqa.weight": [r + dr, d],
                       b + "attn_kv_a_norm.weight": [r], b + "attn_k_b.weight": [Hq, r, dn],
                       b + "attn_v_b.weight": [Hq, dv, r], b + "attn_output.weight": [d, Hq * dv],
                       b + "ffn_gate_shexp.weight": [fs, d], b + "ffn_up_shexp.weight": [fs, d],
                       b + "ffn_down_shexp.weight": [d, fs]})
        E, fe = g["expert_count"], g["expert_feed_forward_length"]
        sh.update({b + "ffn_gate_inp.weight": [E, d], b + "ffn_gate_exps.weight": [E, fe, d],
                   b + "ffn_up_exps.weight": [E, fe, d], b + "ffn_down_exps.weight": [E, d, fe]})
    sh["output_norm.weight"] = [d]
    if a == "phi3":
        n2 = g["rope.dimension_count"] // 2
        sh.update({"rope_factors_short.weight": [n2], "rope_factors_long.weight": [n2]})
    else:
        sh["output.weight"] = [V, d]
    return sh


def _gguf_meta(meta, V, d, ctx, L):
    """read_gguf-style metadata for a written model: the architecture's keys, a word tokenizer."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, bool):
            out[k] = (7, v)
        elif isinstance(v, int):
            out[k] = (4, v)
        elif isinstance(v, float):
            out[k] = (6, v)
        else:
            out[k] = (8, v)
    out["tokenizer.ggml.model"] = (8, "word")
    out["tokenizer.ggml.tokens"] = (9, ["<unk>"] + [f"w{i}" for i in range(1, V)])
    return out


def build_tp_models():
    for name, meta in TP.items():
        rng = np.random.default_rng(sum(map(ord, name)))
        shapes = _tp_shapes(meta)
        doc = gguf_ir.build(dict(meta), shapes)
        W = {}
        for t, sh in shapes.items():
            if t.startswith("rope_factors_"):  # v1.5.0: LongRoPE's divisors, short near 1, long larger
                a = (1.0 if "short" in t else 2.0) + (0.5 if "short" in t else 4.0) * np.abs(rng.standard_normal(sh))
            elif t.endswith("norm.weight"):
                a = 1.0 + 0.1 * rng.standard_normal(sh)
            elif t.endswith("ssm_a"):
                a = -np.exp(0.5 * rng.standard_normal(sh))
            elif t.endswith(".bias") or t.endswith("ssm_d"):
                a = 0.1 * rng.standard_normal(sh)
            else:
                a = rng.standard_normal(sh) / np.sqrt(sh[-1])
            W[t] = a.astype(np.float32)
        dst = SUITE / "models" / name
        dst.mkdir(parents=True, exist_ok=True)
        m = doc["model"]
        write_gguf(dst / f"{name}.gguf", _gguf_meta(meta, TP_VOCAB, m["d"], m["ctx"], m["n_layers"]), W, doc)


# ---------------------------------------------------------------- expected values

def model_dirs():
    return sorted(p for p in (SUITE / "models").iterdir() if p.is_dir() and (p / (p.name + ".gguf")).is_file())


def _profile(name):
    if name.startswith("tp_"):
        return "thirdparty"
    return PK_PROFILES.get(name, "core")


def _state_info(doc, W):
    k = 0
    for n in doc["embed"]["nodes"]:
        if n["op"] == "embedding" and n.get("state_weight"):
            k = W[n["state_weight"]].shape[0]
    return k


def _conversation(rng, V, k, turns):
    """Template ids (IR_SPEC.md 8.2; bpe specials 1 eot, 2 user, 3 assistant): exchanges with state blocks."""
    def words(n):
        return [int(x) for x in rng.integers(4, V, n)]
    ids = []
    for u, a in turns:
        ids += [2] + words(u) + [3] + words(a) + [1] + list(range(V, V + k))
    return ids


def model_case(name, doc, W, rng):
    V = doc["model"]["vocab"]
    k = _state_info(doc, W)
    routed = doc["block"]["nodes"][0]["op"] == "depth_route"
    prompts = []
    for j in range(4):
        if k:
            ids = _conversation(rng, V, k, [(2, 2)] * (j % 2 + 1)) + [2] + [int(x) for x in rng.integers(4, V, 2)] + [3]
        else:
            ids = [int(x) for x in rng.integers(1, V, 3 + j)]
        cur, cont = list(ids), []
        for _ in range(6):
            lg = ir_numpy.forward(doc, W, cur)[-1].astype(np.float64)
            lg[0] = -np.inf
            cont.append(int(np.argmax(lg)))
            cur.append(cont[-1])
        trace = []
        lg = ir_numpy.forward(doc, W, ids + cont[:-1], trace=trace)
        lp, am = logprobs_of(lg[len(ids) - 1:], cont, [0])
        p = {"ids": ids, "continuation": cont, "logprobs": lp, "argmax": am}
        if routed:
            p["depth"] = trace
        prompts.append(p)
    return {"model": f"models/{name}", "class": "ROUNDING", "tol": 1e-5, "banned": [0], "prompts": prompts}


def _kv_expect(doc, ids, trace, evict, k):
    from ir_conformance import _reference_kv
    kv = _reference_kv(doc, ids, trace, evict, k)
    return ({sl: [[len(p) for p in al] for al in apps] for sl, apps in kv.items()},
            {sl: [al for al in apps] for sl, apps in kv.items()})


def _expect_after(doc, W, ids, n_new, evict, k, logits=True):
    trace, states = [], {}
    e = {"len": len(ids)}
    if ids:
        lg = ir_numpy.forward(doc, W, ids, trace=trace, states=states)
        if logits and n_new:
            e["logits"] = enc(lg[-n_new:])
        if trace and n_new:
            e["depth"] = trace[-n_new:]
        rows, pos = _kv_expect(doc, ids, trace, evict, k)
        if rows:
            e["kv_rows"], e["kv_positions"] = rows, pos
        if states:
            L = doc["block"]["repeat"]
            e["recurrent"] = {sl: [{t: enc(v) for t, v in per[i].items()} for i in range(L)] for sl, per in states.items()}
    return e


def trace_case(name, doc, W, steps, evict=None, score_form=False):
    """Expected reports after each step, from the reference over the session's ids."""
    k = _state_info(doc, W)
    ids, snaps, expect = [], {}, []
    ev = evict is not False
    for st in steps:
        m = st["method"]
        if m == "forward":
            ids = ids + st["ids"]
            expect.append(_expect_after(doc, W, ids, len(st["ids"]), ev, k))
        elif m == "truncate":
            ids = ids[:st["n"]]
            expect.append(_expect_after(doc, W, ids, 0, ev, k, logits=False))
        elif m == "snapshot":
            snaps[st["tag"]] = list(ids)
            expect.append({"len": len(ids)})
        elif m == "restore":
            ids = list(snaps[st["tag"]])
            expect.append(_expect_after(doc, W, ids, 0, ev, k, logits=False))
    case = {"model": f"models/{name}", "evict": evict, "class": "ROUNDING", "tol": 1e-5, "steps": steps,
            "expect": expect}
    if score_form:
        allf = [x for st in steps if st["method"] == "forward" for x in st["ids"]]
        V = doc["model"]["vocab"]
        cut = max([i for i, t in enumerate(allf) if t >= V], default=0) + 1
        trace = []
        lg = ir_numpy.forward(doc, W, allf[:-1], trace=trace)
        cont = allf[cut:]
        lp, am = logprobs_of(lg[cut - 1:], cont, [0])
        sf = {"prompt": allf[:cut], "continuation": cont, "evict": bool(k) and ev, "logprobs": lp, "argmax": am}
        if trace:
            sf["depth"] = trace
        if k and ev:
            sf["kv_rows"] = [len(p) for p in evict_allowed(allf[:-1], V, k)]
        case["score_form"] = sf
    return case


SCORE_FORM = {"pk_llama", "pk_moe", "pk_recursive", "pk_mor", "pk_state"}


def traces(name, doc, W, rng):
    V = doc["model"]["vocab"]
    k = _state_info(doc, W)
    out = {}
    if k:
        conv = _conversation(rng, V, k, [(2, 2), (2, 3)]) + [2, int(rng.integers(4, V)), int(rng.integers(4, V)), 3,
                                                             int(rng.integers(4, V))]
        first = conv.index(V + k - 1) + 1  # the call ends exactly where block 0 completes (lazy rows legal)
        steps = [{"method": "forward", "ids": conv[:first]}, {"method": "forward", "ids": conv[first:first + 1]}]
        steps += [{"method": "forward", "ids": conv[i:i + 2]} for i in range(first + 1, len(conv), 2)]
        out["evict_on"] = trace_case(name, doc, W, steps, None, score_form=True)
        out["evict_off"] = trace_case(name, doc, W, steps, False, score_form=True)
        new = [int(rng.integers(4, V)) for _ in range(3)]
        # Into segment 1, whose text rows were dropped when block 1 completed: re-prefill from its start.
        out["rewind_evicted"] = trace_case(name, doc, W, [
            {"method": "forward", "ids": conv}, {"method": "truncate", "n": first + 3},
            {"method": "forward", "ids": new}])
        # Into the current segment (its rows are held): keep the rows before the cut.
        out["rewind_current"] = trace_case(name, doc, W, [
            {"method": "forward", "ids": conv}, {"method": "truncate", "n": len(conv) - 2},
            {"method": "forward", "ids": new[:2]}])
        # A cut exactly at a segment's start (after block 0): nothing to re-prefill.
        out["rewind_at_block"] = trace_case(name, doc, W, [
            {"method": "forward", "ids": conv}, {"method": "truncate", "n": first},
            {"method": "forward", "ids": [2] + new}])
        return out
    ids = [int(x) for x in rng.integers(1, V, 9)]
    steps = [{"method": "forward", "ids": ids[:4]}] + [{"method": "forward", "ids": [t]} for t in ids[4:]]
    out["decode"] = trace_case(name, doc, W, steps, None, score_form=name in SCORE_FORM)
    if any(s["kind"] == "recurrent" for s in doc["block"].get("state", [])):
        other = [int(x) for x in rng.integers(1, V, 3)]
        out["snapshot"] = trace_case(name, doc, W, [
            {"method": "forward", "ids": ids[:4]}, {"method": "snapshot", "tag": "p"},
            {"method": "forward", "ids": ids[4:7]}, {"method": "restore", "tag": "p"},
            {"method": "forward", "ids": other}])
    elif name in ("pk_llama", "pk_mor", "tp_qwen3", "tp_llama_hd16", "tp_qwen3_hd16", "tp_phi3", "tp_deepseek2"):
        out["truncate"] = trace_case(name, doc, W, [
            {"method": "forward", "ids": ids}, {"method": "truncate", "n": 5},
            {"method": "forward", "ids": [int(x) for x in rng.integers(1, V, 2)]}])
    return out


# ---------------------------------------------------------------- op fixtures

def op_cases():
    rng = np.random.default_rng(20261025)

    def r(*sh, s=1.0):
        return (s * rng.standard_normal(sh)).astype(np.float32)

    cases = {}

    def add(op, name, node, inputs, weights, cls, state_slot=None):
        env = {k: (np.array(v) if isinstance(v, list) and k in ("tokens", "pos") else v) for k, v in inputs.items()}
        states = {}
        ir_numpy.run_nodes([node], env, weights, i=0, states=states)
        outs = node["out"] if isinstance(node["out"], list) else [node["out"]]
        expect = {}
        for o in outs:
            expect[o] = [enc(x) for x in env[o]] if isinstance(env[o], list) else enc(env[o])
        c = {"name": name, "node": node,
             "inputs": {k: (v if k in ("tokens", "pos") else ([enc(x) for x in v] if isinstance(v, list) else enc(v)))
                        for k, v in inputs.items()},
             "weights": {k: enc(v) for k, v in weights.items()}, "expect": expect, "class": cls,
             "tol": 1e-5}
        if states:
            c["expect_state"] = {sl: {t: enc(v) for t, v in per[0].items()} for sl, per in states.items()}
        cases.setdefault(op, []).append(c)

    tab, st = r(8, 6), r(2, 6)
    add("embedding", "rows", {"op": "embedding", "in": ["tokens"], "weight": "e", "out": "x"},
        {"tokens": [3, 0, 5, 1]}, {"e": tab}, "EXACT")
    add("embedding", "state_weight", {"op": "embedding", "in": ["tokens"], "weight": "e", "state_weight": "s",
                                      "out": "x"}, {"tokens": [2, 8, 9, 1]}, {"e": tab, "s": st}, "EXACT")
    add("pos_embedding", "rows", {"op": "pos_embedding", "in": ["x", "pos"], "weight": "p", "out": "y"},
        {"x": r(4, 6), "pos": [0, 1, 2, 3]}, {"p": r(8, 6)}, "EXACT")
    x = r(3, 6)
    add("linear", "bias", {"op": "linear", "in": ["x"], "weight": "w", "bias": "b", "out": "y"}, {"x": x},
        {"w": r(5, 6), "b": r(5)}, "ROUNDING")
    add("linear", "no_bias", {"op": "linear", "in": ["x"], "weight": "w", "bias": None, "out": "y"}, {"x": x},
        {"w": r(5, 6)}, "ROUNDING")
    add("linear", "tied_to", {"op": "linear", "in": ["x"], "weight": "absent", "bias": None, "tied_to": "e",
                              "out": "y"}, {"x": x}, {"e": r(7, 6)}, "ROUNDING")
    add("scale", "bias", {"op": "scale", "in": ["x"], "weight": "g", "bias": "b", "out": "y"}, {"x": x},
        {"g": r(6), "b": r(6)}, "EXACT")
    add("scale", "no_bias", {"op": "scale", "in": ["x"], "weight": "g", "bias": None, "out": "y"}, {"x": x},
        {"g": r(6)}, "EXACT")
    add("split", "three", {"op": "split", "in": ["x"], "sizes": [2, 3, 1], "out": ["a", "b", "c"]}, {"x": x}, {},
        "EXACT")
    add("rmsnorm", "row", {"op": "rmsnorm", "in": ["x"], "weight": "w", "eps": 1e-05, "out": "y"}, {"x": x},
        {"w": r(6)}, "ROUNDING")
    add("rmsnorm", "group", {"op": "rmsnorm", "in": ["x"], "weight": "w", "eps": 1e-06, "group": 3, "out": "y"},
        {"x": x}, {"w": r(3)}, "ROUNDING")
    add("rmsnorm", "weight_offset", {"op": "rmsnorm", "in": ["x"], "weight": "w", "eps": 1e-06, "weight_offset": 1.0,
                                     "out": "y"}, {"x": x}, {"w": r(6, s=0.1)}, "ROUNDING")
    add("layernorm", "row", {"op": "layernorm", "in": ["x"], "weight": "w", "bias": "b", "eps": 1e-05, "out": "y"},
        {"x": x}, {"w": r(6), "b": r(6)}, "ROUNDING")
    wide = np.concatenate([r(2, 6, s=3.0), np.array([[0, -0.0, 1e-8, -1e-8, 20, -20]], np.float32)])
    for ap in ("tanh", "none"):
        add("gelu", ap, {"op": "gelu", "in": ["x"], "approx": ap, "out": "y"}, {"x": wide}, {}, "ROUNDING")
    add("silu", "x", {"op": "silu", "in": ["x"], "out": "y"}, {"x": wide}, {}, "ROUNDING")
    add("sigmoid", "x", {"op": "sigmoid", "in": ["x"], "out": "y"}, {"x": wide}, {}, "ROUNDING")
    add("relu", "x", {"op": "relu", "in": ["x"], "out": "y"}, {"x": wide}, {}, "EXACT")
    a, b = r(3, 6), r(3, 6)
    for op in ("add", "mul", "sub"):
        add(op, "ab", {"op": op, "in": ["a", "b"], "out": "y"}, {"a": a, "b": b}, {}, "EXACT")
    add("affine", "scale_shift", {"op": "affine", "in": ["x"], "scale": 0.7071067690849304, "shift": -1.25,
                                  "out": "y"}, {"x": x}, {}, "EXACT")
    add("mul_col", "x_c", {"op": "mul_col", "in": ["x", "c"], "out": "y"}, {"x": x, "c": r(3, 1)}, {}, "EXACT")
    add("rms_row", "eps", {"op": "rms_row", "in": ["x"], "eps": 1e-08, "out": "y"}, {"x": x}, {}, "ROUNDING")
    q8, k8, pos = r(4, 8), r(4, 8), [0, 1, 5, 9]
    rope = {"op": "rope", "in": ["q", "k", "pos"], "heads": 2, "head_dim": 4, "theta": 10000.0, "rope_heads": "all",
            "pairing": "interleaved", "out": ["q", "k"]}
    add("rope", "interleaved", rope, {"q": q8, "k": k8, "pos": pos}, {}, "ROUNDING")
    add("rope", "neox", dict(rope, pairing="neox"), {"q": q8, "k": k8, "pos": pos}, {}, "ROUNDING")
    add("rope", "neox_partial", dict(rope, pairing="neox", n_rot=2), {"q": q8, "k": k8, "pos": pos}, {}, "ROUNDING")
    add("rope", "freq_scale", dict(rope, pairing="neox", freq_scale=0.5), {"q": q8, "k": k8, "pos": pos}, {},
        "ROUNDING")
    add("rope", "grouped_query", dict(rope, kv_heads=1), {"q": q8, "k": r(4, 4), "pos": pos}, {}, "ROUNDING")
    att = {"op": "attention", "in": ["q", "k", "v"], "state": "kv", "heads": 2, "kv_heads": 2, "head_dim": 4,
           "causal": True, "scale": 0.5, "window": 0, "sinks": 0, "out": "a"}
    qa, ka, va = r(6, 8), r(6, 8), r(6, 8)
    add("attention", "causal", att, {"q": qa, "k": ka, "v": va}, {}, "ROUNDING")
    add("attention", "window_sinks", dict(att, window=2, sinks=1), {"q": qa, "k": ka, "v": va}, {}, "ROUNDING")
    add("attention", "grouped_query", dict(att, kv_heads=1), {"q": qa, "k": r(6, 4), "v": r(6, 4)}, {}, "ROUNDING")
    moe = {"op": "moe_ffn", "in": ["x"], "router": "r", "experts": {"gate": "g", "up": "u", "down": "d"},
           "n_experts": 4, "top_k": 2, "gate_norm": "renorm", "activation": "silu", "out": "y"}
    mw = {"r": r(4, 6), "g": r(4, 8, 6), "u": r(4, 8, 6), "d": r(4, 6, 8)}
    add("moe_ffn", "renorm", moe, {"x": x}, mw, "ROUNDING")
    add("moe_ffn", "raw", dict(moe, gate_norm="raw"), {"x": x}, mw, "ROUNDING")
    add("list", "two", {"op": "list", "in": ["a", "b"], "out": "L"}, {"a": a, "b": b}, {}, "EXACT")
    add("append", "one", {"op": "append", "in": ["L", "b"], "out": "L2"}, {"L": [a], "b": b}, {}, "EXACT")
    s3 = [r(3, 6), r(3, 6)]
    add("depth_attend", "one_source", {"op": "depth_attend", "in": ["L"], "weight": "q", "eps": 1e-05, "out": "h"},
        {"L": [a]}, {"q": r(6)}, "EXACT")
    add("depth_attend", "three_sources", {"op": "depth_attend", "in": ["L", "x"], "weight": "q", "eps": 1e-05,
                                          "out": "h"}, {"L": s3, "x": x}, {"q": r(6)}, "ROUNDING")
    for nm, e, ins in (("gelu_sigmoid", "x*sigmoid(1.702*x)", ["x"]), ("relu_squared", "max(x, 0)^2", ["x"]),
                       ("where_rowmean", "where(x - a, x, a) + rowmean(x)", ["x", "a"]),
                       ("layer_norm", "(x - rowmean(x)) / sqrt(rowmean((x - rowmean(x))^2) + 1e-5)", ["x"])):
        add("expr", nm, {"op": "expr", "in": ins, "expr": e, "out": "y"}, {"x": x, "a": b}, {}, "ROUNDING")
    add("linear_attention", "two_heads", {"op": "linear_attention", "in": ["q", "k", "v"], "state": "lin", "heads": 2,
                                          "head_dim": 4, "causal": True, "feature_map": "elu_plus_one",
                                          "den_floor": 1e-08, "out": "y"}, {"q": qa, "k": ka, "v": va}, {}, "ROUNDING")
    add("ssm_scan", "n4", {"op": "ssm_scan", "in": ["u"], "state": "ssm", "n": 4, "a": "A", "b": "B", "c": "C",
                           "d": "Dk", "out": "y"}, {"u": r(5, 4)},
        {"A": r(4, 4, s=0.3), "B": r(4), "C": r(4), "Dk": r(1)}, "ROUNDING")
    add("conv1d_causal", "k4", {"op": "conv1d_causal", "in": ["x"], "state": "conv", "weight": "w", "bias": "b",
                                "channels": 3, "kernel": 4, "out": "y"}, {"x": r(5, 3)}, {"w": r(3, 4), "b": r(3)},
        "ROUNDING")
    add("selective_scan", "d3n2", {"op": "selective_scan", "in": ["x", "dt", "B", "C"], "state": "ssm", "a": "A",
                                   "d": "Dk", "channels": 3, "d_state": 2, "dt_softplus": True, "out": "y"},
        {"x": r(5, 3), "dt": np.concatenate([r(4, 3), np.array([[25.0, -3.0, 0.0]], np.float32)]), "B": r(5, 2),
         "C": r(5, 2)}, {"A": -np.exp(r(3, 2, s=0.5)), "Dk": r(3)}, "ROUNDING")
    # recurrence (IR_SPEC 6.10): the four hand-written ops' rewrites on those ops' own fixtures (6.10.6), then
    # bodies of their own: a learned initial state with an index offset, and an EXACT exponential average.
    import ir_recurrence
    from ir_conformance import dec
    for op in ("linear_attention", "ssm_scan", "conv1d_causal", "selective_scan"):
        hw = cases[op][0]
        node = ir_recurrence.REWRITES[op](hw["node"])
        ins = {k: dec(v) for k, v in hw["inputs"].items()}
        add("recurrence", f"{op}_body", node, ins, {k: dec(v) for k, v in hw["weights"].items()},
            "EXACT" if ir_recurrence.compile_node(node).exact else "ROUNDING")
        cases["recurrence"][-1]["reproduces"] = f"op/{op}/{hw['name']}"
    for name, node, ins, ws in recurrence_bodies(r):
        add("recurrence", name, node, ins, ws, "EXACT" if ir_recurrence.compile_node(node).exact else "ROUNDING")
    # v1.2 (cases added after the first release draw after every earlier one, so those stay byte-identical):
    # rope_heads "first" (5.14: head 0 of q and of k); depth_attend without eps (5.18: 1e-5); a row
    # reduction of a constant (5.19: w additions); an expression exactly 200 deep (5.19); a recurrence read
    # whose index is larger than the declared dimension (6.10.3: +0 outside).
    add("rope", "first_head", dict(rope, rope_heads="first"), {"q": q8, "k": k8, "pos": pos}, {}, "ROUNDING")
    add("depth_attend", "eps_default", {"op": "depth_attend", "in": ["L", "x"], "weight": "q", "out": "h"},
        {"L": s3, "x": x}, {"q": r(6)}, "ROUNDING")
    add("expr", "rowsum_constant", {"op": "expr", "in": ["x"], "expr": "rowsum(0.1)", "out": "y"},
        {"x": np.zeros((1, 2048), np.float32)}, {}, "ROUNDING")
    add("expr", "depth_200", {"op": "expr", "in": ["x"], "expr": "(" * 39 + "-x" + ")" * 39, "out": "y"},
        {"x": x}, {}, "ROUNDING")
    pad = {"op": "recurrence", "in": ["x"], "state": "pad", "index": {"c": 3, "j": 4}, "inputs": ["x[j]"],
           "weights": {"W[c,c]": "W"}, "update": ["h[c] = h[c] + sum(j, W[c,j] * x[j])"],
           "readout": "y[c] = h[c]", "out": "y"}
    add("recurrence", "read_beyond_declaration", pad, {"x": r(5, 4)}, {"W": r(3, 3, s=0.5)},
        "EXACT" if ir_recurrence.compile_node(pad).exact else "ROUNDING")
    # v1.3 (5.15): T5's relative position bias, W[buckets[min(p - j, D - 1)], h] added to each head's scaled
    # score: a table shorter than the rows (the last bucket covers every larger distance), with a window and
    # sinks, and under grouped-query attention (the bias is per query head).
    q9, k9, v9 = r(9, 8), r(9, 8), r(9, 8)
    pbt = r(5, 2)
    pb = {"weight": "rel", "buckets": [0, 1, 2, 3, 3, 4]}
    add("attention", "position_bias", dict(att, position_bias=pb), {"q": q9, "k": k9, "v": v9}, {"rel": pbt},
        "ROUNDING")
    add("attention", "position_bias_window_sinks", dict(att, window=3, sinks=1, position_bias=pb),
        {"q": q9, "k": k9, "v": v9}, {"rel": pbt}, "ROUNDING")
    add("attention", "position_bias_grouped_query", dict(att, kv_heads=1, position_bias=pb),
        {"q": q9, "k": r(9, 4), "v": r(9, 4)}, {"rel": pbt}, "ROUNDING")
    # v1.4 (5.15): Shaw's relative key and value vectors by the clipped distance n = min(p - j, K): K = 3 below
    # the 9 rows (the last row covers every larger distance); vectors shared by the heads (rows of head_dim)
    # and one set per head (rows of heads * head_dim); keys only; with a window and sinks; under grouped-query
    # attention (the query head's vectors); K = 0 (one vector for every distance).
    qs, ks, vs = r(9, 8), r(9, 8), r(9, 8)
    rk1, rv1, rk2, rv2 = r(4, 4, s=0.5), r(4, 4, s=0.5), r(4, 8, s=0.5), r(4, 8, s=0.5)
    rel = {"key": "rk", "value": "rv", "max_distance": 3}
    qkv = {"q": qs, "k": ks, "v": vs}
    add("attention", "relative_kv", dict(att, relative_kv=rel), qkv, {"rk": rk1, "rv": rv1}, "ROUNDING")
    add("attention", "relative_kv_per_head", dict(att, relative_kv=rel), qkv, {"rk": rk2, "rv": rv2}, "ROUNDING")
    add("attention", "relative_kv_keys_only", dict(att, relative_kv=dict(rel, value=None)), qkv, {"rk": rk1},
        "ROUNDING")
    add("attention", "relative_kv_window_sinks", dict(att, window=3, sinks=1, relative_kv=rel), qkv,
        {"rk": rk1, "rv": rv1}, "ROUNDING")
    add("attention", "relative_kv_grouped_query", dict(att, kv_heads=1, relative_kv=rel),
        {"q": qs, "k": r(9, 4), "v": r(9, 4)}, {"rk": rk2, "rv": rv2}, "ROUNDING")
    add("attention", "relative_kv_distance_zero", dict(att, relative_kv=dict(rel, max_distance=0)), qkv,
        {"rk": r(1, 4, s=0.5), "rv": r(1, 4, s=0.5)}, "ROUNDING")
    # v1.5 (5.15, 5.14, 5.5, 5.20): Gemma 2's attention soft cap, alone, with a window and sinks, and before a
    # position bias; LongRoPE's per-pair divisors (one tensor; the short and the long set, chosen by the
    # largest position + 1 = 10 against original_ctx 10 and 9) and attn_factor; a per-group split; multi-head
    # latent attention (two heads, a window and sinks, one head).
    add("attention", "softcap", dict(att, softcap=0.5), {"q": q9, "k": k9, "v": v9}, {}, "ROUNDING")
    add("attention", "softcap_window_sinks", dict(att, window=3, sinks=1, softcap=0.5), {"q": q9, "k": k9, "v": v9},
        {}, "ROUNDING")
    add("attention", "softcap_position_bias", dict(att, softcap=0.5, position_bias=pb), {"q": q9, "k": k9, "v": v9},
        {"rel": pbt}, "ROUNDING")
    f_short, f_long = (1.0 + 0.5 * np.abs(r(2))).astype(np.float32), (2.0 + 4.0 * np.abs(r(2))).astype(np.float32)
    rq = {"q": q8, "k": k8, "pos": pos}
    add("rope", "freq_factors", dict(rope, pairing="neox", freq_factors="f"), rq, {"f": f_long}, "ROUNDING")
    for nm, oc in (("freq_factors_short", 10), ("freq_factors_long", 9)):
        add("rope", nm, dict(rope, pairing="neox", freq_factors={"short": "fs", "long": "fl", "original_ctx": oc}),
            rq, {"fs": f_short, "fl": f_long}, "ROUNDING")
    add("rope", "attn_factor", dict(rope, attn_factor=1.25), rq, {}, "ROUNDING")
    add("rope", "freq_factors_partial_attn_factor", dict(rope, pairing="neox", n_rot=2, attn_factor=1.25,
                                                         freq_factors="f1"), rq, {"f1": f_long[:1]}, "ROUNDING")
    add("split", "groups", {"op": "split", "in": ["x"], "sizes": [2, 1], "groups": 2, "out": ["a", "b"]}, {"x": x}, {},
        "EXACT")
    mla = {"op": "mla_attention", "in": ["qn", "qr", "c", "kr"], "state": "kv", "heads": 2, "kv_rank": 3,
           "qk_nope": 2, "qk_rope": 2, "v_head_dim": 2, "k_b": "kb", "v_b": "vb", "scale": 0.5, "causal": True,
           "window": 0, "sinks": 0, "out": "a"}
    mi = {"qn": r(7, 4), "qr": r(7, 4), "c": r(7, 3), "kr": r(7, 2)}
    mw = {"kb": r(2, 3, 2), "vb": r(2, 2, 3)}
    add("mla_attention", "two_heads", mla, mi, mw, "ROUNDING")
    add("mla_attention", "window_sinks", dict(mla, window=3, sinks=1), mi, mw, "ROUNDING")
    add("mla_attention", "one_head", dict(mla, heads=1), {"qn": mi["qn"][:, :2], "qr": mi["qr"][:, :2], "c": mi["c"],
                                                          "kr": mi["kr"]}, {"kb": mw["kb"][:1], "vb": mw["vb"][:1]},
        "ROUNDING")
    return cases


def recurrence_bodies(r, square=True):
    """(name, node, inputs, weights): recurrences beyond the four rewrites, on draws of r."""
    rnn = {"op": "recurrence", "in": ["x"], "state": "rnn", "index": {"i": 3, "j": 3}, "inputs": ["x[i]"],
           "weights": {"W[i,j]": "W", "U[i,j]": "U", "h0[i]": "h0"}, "init": ["h[i] = h0[i]"],
           "update": ["h[i] = tanh(sum(j, W[i,j] * h[j]) + sum(j, U[i,j] * x[j]))"],
           "readout": "y[i] = h[i] + 0.5 * h[i - 1]", "out": "y"}
    ema = {"op": "recurrence", "in": ["x", "g"], "state": "avg", "index": {"c": 4}, "inputs": ["x[c]", "g[c]"],
           "weights": {"a[c]": "decay"},
           "update": ["m[c] = a[c] * m[c] + (1 - a[c]) * (x[c] * g[c])", "n[c] = n[c] + 1"],
           "readout": "y[c] = m[c] / (1 - a[c] ^ n[c]) + max(x[c], 0)", "out": "y"}
    ema_exact = dict(ema, readout="y[c] = m[c] / n[c] + max(x[c], 0)")
    # 6.10.1 (v1.1.1): a declaration may repeat an index; W[c,c] is a square weight, read as W[c,j].
    sq = {"op": "recurrence", "in": ["x"], "state": "sq", "index": {"c": 3, "j": 3}, "inputs": ["x[c]"],
              "weights": {"W[c,c]": "W"}, "update": ["h[c] = sum(j, W[c,j] * h[j]) + x[c]"],
              "readout": "y[c] = h[c]", "out": "y"}
    return [("rnn_init_offset", rnn, {"x": r(6, 3)}, {"W": r(3, 3, s=0.5), "U": r(3, 3, s=0.5), "h0": r(3)}),
            ("ema_bias_corrected", ema, {"x": r(5, 4), "g": r(5, 4)}, {"decay": np.full(4, 0.875, np.float32)}),
            ("ema_exact", ema_exact, {"x": r(5, 4), "g": r(5, 4)}, {"decay": np.full(4, 0.875, np.float32)}),
            ] + ([("square_weight_declaration", sq, {"x": r(5, 3)}, {"W": r(3, 3, s=0.5)})] if square else [])


def grad_cases():
    """Gradient cases for recurrence (IR_SPEC 6.10.5): d sum(out * out_grad) by central differences of the
    reference in float64, for every input and weight, on small fixtures of their own."""
    import ir_recurrence
    rng = np.random.default_rng(6105)

    def r(*sh, s=1.0):
        return (s * rng.standard_normal(sh)).astype(np.float32)

    hw = {"linear_attention": ({"op": "linear_attention", "in": ["q", "k", "v"], "state": "lin", "heads": 2,
                                "head_dim": 2, "causal": True, "feature_map": "elu_plus_one", "den_floor": 1e-08,
                                "out": "y"}, {"q": r(4, 4), "k": r(4, 4), "v": r(4, 4)}, {}),
          "ssm_scan": ({"op": "ssm_scan", "in": ["u"], "state": "ssm", "n": 3, "a": "A", "b": "B", "c": "C",
                        "d": "Dk", "out": "y"}, {"u": r(4, 3)}, {"A": r(3, 3, s=0.3), "B": r(3), "C": r(3), "Dk": r(1)}),
          "conv1d_causal": ({"op": "conv1d_causal", "in": ["x"], "state": "conv", "weight": "w", "bias": "b",
                             "channels": 2, "kernel": 3, "out": "y"}, {"x": r(5, 2)}, {"w": r(2, 3), "b": r(2)}),
          "selective_scan": ({"op": "selective_scan", "in": ["x", "dt", "B", "C"], "state": "ssm", "a": "A", "d": "Dk",
                              "channels": 2, "d_state": 2, "dt_softplus": True, "out": "y"},
                             {"x": r(4, 2), "dt": r(4, 2), "B": r(4, 2), "C": r(4, 2)},
                             {"A": -np.exp(r(2, 2, s=0.5)), "Dk": r(2)})}
    items = [(f"{op}_body", ir_recurrence.REWRITES[op](nd), ins, ws) for op, (nd, ins, ws) in hw.items()]
    items += recurrence_bodies(r, square=False)
    out = []

    def case(name, node, ins, ws):
        prog = ir_recurrence.compile_node(node)
        xs = [ins[n] for n in node["in"]]
        wl = {local: ws[t] for local, _, t in prog.weights}
        y, _ = ir_recurrence.run(prog, xs, wl)
        g = r(*y.shape)
        fd = ir_recurrence.fd_grads(prog, xs, wl, g)
        expect = {}
        for (local, _), value in zip(prog.inputs, node["in"]):
            expect[value] = enc(fd[local].astype(np.float32))
        for local, _, t in prog.weights:
            expect[t] = enc(fd[local].reshape(np.shape(ws[t])).astype(np.float32))
        out.append({"name": name, "node": node, "inputs": {k: enc(v) for k, v in ins.items()},
                    "weights": {k: enc(v) for k, v in ws.items()}, "out_grad": enc(g), "expect": expect,
                    "tol": 1e-4})

    for it in items:
        case(*it)
    # Cases added after the first release draw after every earlier one, so those stay byte-identical.
    case(*[b for b in recurrence_bodies(r) if b[0] == "square_weight_declaration"][0])
    return {"_comment": "spec/IR_SPEC.md section 6.10.5: expect holds d sum(out * out_grad) / d(each input value "
                        "and weight tensor), central differences of tools/ir_recurrence.py in float64 (h = 1e-5 "
                        "relative). A case passes within tol * max(1, max|expect|) per tensor.", "cases": out}


def attention64(node, ins, ws):
    """IR_SPEC 5.15's attention with relative_kv in float64 (no window, sinks or mask: the gradient cases'
    nodes), independent of tools/ir_numpy.py: e = (q . (k + rk[n])) * scale, n = min(p - j, K), softmax over
    j <= p, out = sum_j a (v_j + rv[n])."""
    q, k, v = (np.asarray(ins[n], np.float64) for n in node["in"])
    H, KV, hd, sc = node["heads"], node["kv_heads"], node["head_dim"], float(node["scale"])
    rel = node["relative_kv"]
    K = rel["max_distance"]
    tk = np.asarray(ws[rel["key"]], np.float64)
    tv = np.asarray(ws[rel["value"]], np.float64) if rel.get("value") is not None else None
    T = q.shape[0]
    n_of = np.clip(np.arange(T)[:, None] - np.arange(T)[None, :], 0, K)
    out = []
    for h in range(H):
        kh = h // (H // KV)
        qh, kk, vv = q[:, h * hd:(h + 1) * hd], k[:, kh * hd:(kh + 1) * hd], v[:, kh * hd:(kh + 1) * hd]
        ck = slice(0, hd) if tk.shape[1] == hd else slice(h * hd, (h + 1) * hd)
        e = np.einsum("pc,pjc->pj", qh, kk[None, :, :] + tk[:, ck][n_of]) * sc
        e[np.arange(T)[None, :] > np.arange(T)[:, None]] = -np.inf
        a = np.exp(e - e.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)
        vals = vv[None, :, :] + (tv[:, slice(0, hd) if tv.shape[1] == hd else slice(h * hd, (h + 1) * hd)][n_of]
                                 if tv is not None else 0.0)
        out.append(np.einsum("pj,pjc->pc", a, vals))
    return np.concatenate(out, axis=1)


def grad_attention_cases():
    """v1.4 (IR_SPEC 5.15, section 11.2's gradient kind): attention with relative_kv, d sum(out * out_grad) by
    central differences of attention64 in float64 (h = 1e-6 relative), for q, k, v and the tables: vectors
    shared by the heads, one set per head under grouped-query attention, keys only."""
    rng = np.random.default_rng(1803)

    def r(*sh, s=1.0):
        return (s * rng.standard_normal(sh)).astype(np.float32)

    att = {"op": "attention", "in": ["q", "k", "v"], "state": "kv", "heads": 2, "kv_heads": 2, "head_dim": 3,
           "causal": True, "scale": 0.5, "window": 0, "sinks": 0, "out": "a"}
    rel = {"key": "rk", "value": "rv", "max_distance": 2}
    items = [("relative_kv", dict(att, relative_kv=rel), {"q": r(5, 6), "k": r(5, 6), "v": r(5, 6)},
              {"rk": r(3, 3, s=0.5), "rv": r(3, 3, s=0.5)}),
             ("relative_kv_per_head_grouped_query", dict(att, kv_heads=1, relative_kv=rel),
              {"q": r(5, 6), "k": r(5, 3), "v": r(5, 3)}, {"rk": r(3, 6, s=0.5), "rv": r(3, 6, s=0.5)}),
             ("relative_kv_keys_only", dict(att, relative_kv=dict(rel, value=None)),
              {"q": r(5, 6), "k": r(5, 6), "v": r(5, 6)}, {"rk": r(3, 3, s=0.5)})]
    out = []
    for name, node, ins, ws in items:
        y = attention64(node, ins, ws)
        g = r(*y.shape).astype(np.float64)
        expect = {}
        for src in (ins, ws):
            for t, val in src.items():
                base = np.asarray(val, np.float64)
                d = np.zeros_like(base)
                for idx in np.ndindex(base.shape):
                    h = 1e-6 * max(1.0, abs(base[idx]))
                    for sgn in (1, -1):
                        pert = base.copy()
                        pert[idx] += sgn * h
                        a2 = dict(ins) if src is ins else ins
                        w2 = dict(ws) if src is ws else ws
                        (a2 if src is ins else w2)[t] = pert
                        d[idx] += sgn * float((attention64(node, a2, w2) * g).sum())
                    d[idx] /= 2 * h
                expect[t] = enc(d.astype(np.float32))
        out.append({"name": name, "node": node, "inputs": {k: enc(v) for k, v in ins.items()},
                    "weights": {k: enc(v) for k, v in ws.items()}, "out_grad": enc(g.astype(np.float32)),
                    "expect": expect, "tol": 1e-4})
    return {"_comment": "spec/IR_SPEC.md section 5.15 (revision 1.4): expect holds d sum(out * out_grad) / d(each "
                        "input value and weight tensor) of attention with relative_kv, central differences of "
                        "tools/ir_conformance_gen.py's attention64 in float64 (h = 1e-6 relative). A case passes "
                        "within tol * max(1, max|expect|) per tensor.", "cases": out}


REC_MODELS = ("pk_kimi", "pk_srd", "pk_mamba", "tp_mamba1")


def recurrence_traces():
    """(name, rewritten document, trace case): each model of REC_MODELS with its hand-written recurrent ops
    replaced by recurrences (IR_SPEC 6.10.6), run through prefill, token-by-token decode, a snapshot and
    a restore, expected values from the ORIGINAL document (today's outputs)."""
    import ir_recurrence
    out = []
    for name in REC_MODELS:
        doc, W, meta, _ = load_model(SUITE / "models" / name)
        rng = np.random.default_rng(sum(map(ord, name)) + 610)
        V = doc["model"]["vocab"]
        ids = [int(x) for x in rng.integers(1, V, 9)]
        other = [int(x) for x in rng.integers(1, V, 3)]
        steps = ([{"method": "forward", "ids": ids[:4]}, {"method": "snapshot", "tag": "p"}]
                 + [{"method": "forward", "ids": [t]} for t in ids[4:]]
                 + [{"method": "restore", "tag": "p"}, {"method": "forward", "ids": other}])
        case = trace_case(name, doc, W, steps)
        case["ir"] = f"recurrence/{name}.ir.json"
        out.append((name, ir_recurrence.rewrite(doc), case))
    return out


# ---------------------------------------------------------------- refusals

def refusals():
    """(name, case) pairs: one fault each, on the suite's models (rule numbers: IR_SPEC.md 10.1)."""
    out = []

    def doc_case(name, base, rule, section, f):
        doc, *_ = load_model(SUITE / "models" / base)
        doc = copy.deepcopy(doc)
        f(doc)
        out.append((name, {"doc": doc, "gguf": f"models/{base}/{base}.gguf", "rule": rule, "section": section}))

    def node(doc, op, k=0):
        return [n for n in doc["block"]["nodes"] if n["op"] == op][k]

    def kv(doc):
        return [s for s in doc["block"]["state"] if s["kind"] == "kv"][0]

    L = "pk_llama"
    doc_case("unknown_op", L, 4, "3.5", lambda d: d["block"]["nodes"].append({"op": "warp", "in": ["x"], "out": "x"}))
    doc_case("unknown_attribute", L, 4, "3.5", lambda d: node(d, "attention").update(temperature=2.0))
    doc_case("unknown_top_key", L, 1, "3.2", lambda d: d.update(profiles=["core"]))
    doc_case("version_2", L, 2, "2.3", lambda d: d.update(ir_version=2))
    doc_case("repeat_ne_n_layers", L, 3, "3.3", lambda d: d["block"].update(repeat=3))
    doc_case("model_ctx_missing", L, 3, "3.3", lambda d: d["model"].pop("ctx"))
    doc_case("per_layer_wrong_length", L, 6, "3.8", lambda d: node(d, "rope").update(theta={"per_layer": [10000.0]}))
    doc_case("per_layer_in_head", L, 6, "3.8", lambda d: d["head"]["nodes"][0].update(eps={"per_layer": [1e-5, 1e-5]}))
    doc_case("add_one_input", L, 7, "3.5", lambda d: node(d, "add").update(**{"in": ["x"]}))
    doc_case("rope_pairing_unknown", L, 5, "5.14", lambda d: node(d, "rope").update(pairing="halves"))
    doc_case("rope_n_rot_odd", L, 10, "5.14", lambda d: node(d, "rope").update(n_rot=3))
    doc_case("attention_not_causal", L, 14, "5.15", lambda d: node(d, "attention").update(causal=False))
    doc_case("attention_heads_not_multiple", L, 10, "5.15", lambda d: node(d, "attention").update(heads=3))
    doc_case("attention_window_differs", L, 12, "5.15", lambda d: node(d, "attention").update(window=4))
    doc_case("slot_declared_twice", L, 11, "6.1", lambda d: d["block"]["state"].append(copy.deepcopy(kv(d))))
    doc_case("slot_unread", L, 11, "6.1", lambda d: d["block"]["state"].append(dict(copy.deepcopy(kv(d)), name="kv2")))
    doc_case("slot_unknown_kind", L, 11, "6.1", lambda d: kv(d).update(kind="paged"))
    doc_case("attention_no_slot", L, 11, "6.1", lambda d: node(d, "attention").update(state="nowhere"))
    doc_case("kv_slot_field_missing", L, 12, "6.2", lambda d: kv(d).pop("max_len"))
    doc_case("tensor_index_without_weight_index", L, 19, "7.2",
             lambda d: node(d, "linear").update(tensor_index="application"))
    doc_case("name_with_i_outside_block", L, "17a", "3.7",
             lambda d: d["head"]["nodes"][0].update(weight="blk.{i}.output_norm.weight"))
    doc_case("evict_on_causal", L, 16, "6.8.3", lambda d: kv(d).update(evict="turn_state"))
    doc_case("missing_tensor", L, 25, "3.7", lambda d: node(d, "linear").update(weight="blk.{i}.nope.weight"))
    doc_case("gelu_approx_unknown", "pk_gpt2", 5, "5.8", lambda d: node(d, "gelu").update(approx="sigmoid"))
    doc_case("split_count", "pk_gpt2", 10, "5.5", lambda d: node(d, "split").update(sizes=[16, 32]))
    doc_case("expr_does_not_parse", "pk_flex_expr", 9, "5.19", lambda d: node(d, "expr").update(expr="max(x,"))
    doc_case("moe_top_k_too_large", "pk_moe", 10, "5.16", lambda d: node(d, "moe_ffn").update(top_k=5))
    doc_case("moe_gate_norm_unknown", "pk_moe", 5, "5.16", lambda d: node(d, "moe_ffn").update(gate_norm="topk"))
    doc_case("recurrent_dtype_f16", "pk_kimi", 13, "6.3",
             lambda d: [s for s in d["block"]["state"] if s["kind"] == "recurrent"][0].update(dtype="f16"))
    doc_case("recurrent_shape_mismatch", "pk_kimi", 13, "6.3",
             lambda d: [s for s in d["block"]["state"] if s["kind"] == "recurrent"][0]["tensors"][0].update(shape=[3, 8]))
    doc_case("recurrent_op_mismatch", "pk_kimi", 13, "6.3",
             lambda d: [s for s in d["block"]["state"] if s["kind"] == "recurrent"][0].update(op="ssm_scan"))
    doc_case("linear_attention_not_causal", "pk_kimi", 14, "6.4",
             lambda d: node(d, "linear_attention").update(causal=False))
    doc_case("conv_kernel_one", "tp_mamba1", 10, "6.6", lambda d: node(d, "conv1d_causal").update(kernel=1))
    doc_case("weight_index_wrong_length", "pk_recursive", 18, "7.1",
             lambda d: d["block"]["weight_index"]["per_layer"].pop())
    doc_case("weight_index_unused_layer", "pk_recursive", 18, "7.1",
             lambda d: d["block"]["weight_index"].update(per_layer=[0, 2, 2, 3, 3][:d["block"]["repeat"]]))
    doc_case("weight_index_negative", "pk_recursive", 18, "7.1",
             lambda d: d["block"]["weight_index"]["per_layer"].__setitem__(0, -1))
    # The nine malformed MoR documents ember refuses (ember experiments/recursion_2026-10-23/mor_refusals.py).
    M = "pk_mor"
    doc_case("mor_route_not_first", M, 20, "7.4", lambda d: d["block"]["nodes"].insert(0, d["block"]["nodes"].pop(1)))
    doc_case("mor_expert_choice", M, 5, "7.7", lambda d: node(d, "depth_route").update(route="expert_choice"))

    def steps(d):
        return node(d, "depth_route")["step"]["per_layer"]

    doc_case("mor_unequal_steps", M, 20, "7.4", lambda d: node(d, "depth_route").update(
        step={"per_layer": [-1, 0, 0, 1, -1]}))
    doc_case("mor_steps_descending", M, 20, "7.4", lambda d: node(d, "depth_route").update(
        step={"per_layer": [-1, 2, 1, 0, -1]}))
    doc_case("mor_step_length_wrong", M, 6, "3.8", lambda d: steps(d).pop())
    doc_case("mor_no_kv_rows_active", M, 21, "7.5", lambda d: kv(d).pop("kv_rows"))
    doc_case("mor_kv_from", M, 22, "7.6", lambda d: kv(d).update(kv_from={"per_layer": list(range(d["block"]["repeat"]))}))
    doc_case("mor_gate_unknown", M, 5, "7.4", lambda d: node(d, "depth_route").update(gate="sigmoid"))
    doc_case("mor_router_with_i", M, 20, "7.7", lambda d: node(d, "depth_route").update(weight="rec.{i}.router.weight"))
    doc_case("mor_kv_rows_shared", M, 5, "7.5", lambda d: kv(d).update(kv_rows="shared"))
    doc_case("mor_route_in_head", M, 20, "7.4", lambda d: d["head"]["nodes"].insert(0, copy.deepcopy(node(d, "depth_route"))))
    S = "pk_state"
    doc_case("turn_state_with_window", S, 15, "6.8.2", lambda d: (node(d, "attention").update(window=4),
                                                                    kv(d).update(window=4)))
    doc_case("turn_state_without_state_weight", S, 15, "6.8.2", lambda d: d["embed"]["nodes"][0].pop("state_weight"))
    doc_case("evict_without_mask", S, 16, "6.8.3", lambda d: node(d, "attention").pop("mask"))
    doc_case("evict_unknown", S, 5, "6.8.3", lambda d: kv(d).update(evict="oldest"))
    doc_case("mask_unknown", S, 5, "6.8.2", lambda d: node(d, "attention").update(mask="sliding"))
    doc_case("state_weight_with_i", S, 17, "6.8.1", lambda d: d["embed"]["nodes"][0].update(state_weight="state_{i}.weight"))
    doc_case("two_state_weights", S, 17, "6.8.1", lambda d: d["embed"]["nodes"].append(dict(copy.deepcopy(
        d["embed"]["nodes"][0]), out="x2")))
    doc_case("state_weight_wrong_name", S, 25, "3.7", lambda d: d["embed"]["nodes"][0].update(state_weight="nope.weight"))
    # recurrence (6.10): faults in pk_kimi and pk_mamba with their recurrent ops rewritten (6.10.6).
    import ir_recurrence

    def rec_case(name, base, rule, f):
        def g(d):
            d.clear()
            d.update(ir_recurrence.rewrite(load_model(SUITE / "models" / base)[0]))
            f(d)
        doc_case(name, base, rule, "6.10", g)

    def rec(d):
        return node(d, "recurrence")

    def rslot(d):
        return [s for s in d["block"]["state"] if s.get("op") == "recurrence"][0]

    rec_case("recurrence_body_does_not_parse", "pk_kimi", 31, lambda d: rec(d)["update"].__setitem__(1, "den[h,e] = den[h,e] +"))
    rec_case("recurrence_name_twice", "pk_kimi", 31, lambda d: rec(d)["let"].append("num[h,e] = 1"))
    rec_case("recurrence_rowsum", "pk_kimi", 31, lambda d: rec(d).update(readout="y[h,e] = rowsum(num[h,e])"))
    rec_case("recurrence_index_size_zero", "pk_kimi", 31, lambda d: rec(d)["index"].update(e=0))
    rec_case("recurrence_index_out_of_scope", "pk_kimi", 32, lambda d: rec(d).update(readout="y[h,e] = sum(h, num[h,e])"))
    rec_case("recurrence_init_reads_input", "pk_kimi", 32, lambda d: rec(d).update(init=["num[h,e] = q[h,e]"]))
    rec_case("recurrence_rank_mismatch", "pk_kimi", 32, lambda d: rec(d)["update"].__setitem__(1, "den[h,e] = den[h] + 1"))
    rec_case("recurrence_index_undeclared", "pk_kimi", 32, lambda d: rec(d).update(readout="y[h,e] = num[h,f]"))
    rec_case("recurrence_slot_shape", "pk_kimi", 33, lambda d: rslot(d)["tensors"][0].update(shape=[16]))
    rec_case("recurrence_slot_missing_tensor", "pk_kimi", 33, lambda d: rslot(d)["tensors"].pop())
    rec_case("recurrence_init_flag", "pk_kimi", 33, lambda d: rslot(d)["tensors"][0].update(init="body"))
    rec_case("recurrence_weight_size", "pk_mamba", 34, lambda d: rec(d)["weights"].update({"A[r,j]": rec(d)["weights"]["B[r]"]}))
    # v1.1.1 (6.10.1, 6.10.2): an assignment that repeats an index; a bare name that is not an index in scope.
    rec_case("recurrence_assignment_repeats_index", "pk_kimi", 32, lambda d: rec(d).update(readout="y[h,h] = num[h,h]"))
    rec_case("recurrence_bare_undeclared_name", "pk_kimi", 32, lambda d: rec(d).update(readout="y[h,e] = num[h,e] + foo"))
    # v1.2: rule 36 (reads of values no node writes, 3.5 and 7.4) and rule 20's contiguous steps (7.4).
    doc_case("reads_unwritten_value", L, 36, "3.5", lambda d: node(d, "add")["in"].__setitem__(1, "nowhere"))
    doc_case("embed_writes_no_x", L, 36, "3.5", lambda d: d["embed"]["nodes"][0].update(out="h"))
    doc_case("head_writes_no_logits", L, 36, "3.4", lambda d: d["head"]["nodes"][-1].update(out="y"))

    def mor_outside(d):
        d["embed"]["nodes"].append({"op": "affine", "in": ["x"], "scale": 1.0, "shift": 0.0, "out": "e"})
        d["block"]["nodes"].insert(1, {"op": "add", "in": ["x", "e"], "out": "x"})
    doc_case("mor_reads_outside_value", M, 36, "7.4", mor_outside)
    doc_case("mor_step_gap", M, 20, "7.4", lambda d: node(d, "depth_route").update(
        step={"per_layer": [-1, 0, -1, 1, 2]}))
    # v1.3.1 (10.1): under depth_route a step node reading tokens, or depth_route's out, is a rule 17a fault
    # (rule 36 follows as a consequence: neither is in the step's scope, 7.4(b)).
    doc_case("mor_reads_tokens", M, "17a", "3.5", lambda d: d["block"]["nodes"].insert(
        1, {"op": "add", "in": ["x", "tokens"], "out": "x"}))
    doc_case("mor_reads_route_out", M, "17a", "7.4", lambda d: d["block"]["nodes"].insert(
        1, {"op": "add", "in": ["x", node(d, "depth_route")["out"]], "out": "x"}))
    # v1.3: rule 39 (5.15), position_bias's buckets and its table.
    doc_case("position_bias_bucket_negative", L, 39, "5.15", lambda d: node(d, "attention").update(
        position_bias={"weight": "token_embd.weight", "buckets": [0, -1]}))
    doc_case("position_bias_buckets_empty", L, 39, "5.15", lambda d: node(d, "attention").update(
        position_bias={"weight": "token_embd.weight", "buckets": []}))
    doc_case("position_bias_table_shape", L, 39, "5.15", lambda d: node(d, "attention").update(
        position_bias={"weight": "token_embd.weight", "buckets": [0, 1, 2]}))
    doc_case("position_bias_per_layer", L, 6, "3.8", lambda d: node(d, "attention").update(
        position_bias={"per_layer": [{"weight": "token_embd.weight", "buckets": [0]}] * d["block"]["repeat"]}))
    # v1.3.5 (7.2): tensor_index on llama_attnres under recursion (weight_index [0, 1, 1, 2], stored layers 0-2).
    # In the head (rule 19); a value other than "application" (rule 5); on an op that names no tensor (rule 4);
    # on a node whose application-indexed name the file lacks: a tagged rmsnorm reads blk.3.attn_norm.weight at
    # application 3, past the stored layers (rule 25), which an engine applying the tag to depth_attend alone
    # accepts.
    AR = "pk_llama_attnres_rec"
    doc_case("tensor_index_in_head", AR, 19, "7.2", lambda d: d["head"]["nodes"][0].update(tensor_index="application"))
    doc_case("tensor_index_value_unknown", AR, 5, "7.2", lambda d: node(d, "depth_attend").update(tensor_index="stored"))
    doc_case("tensor_index_on_affine", AR, 4, "3.5", lambda d: node(d, "affine").update(tensor_index="application"))
    doc_case("tensor_index_absent_application_tensor", AR, 25, "3.7",
             lambda d: node(d, "rmsnorm").update(tensor_index="application"))
    # v1.4: rule 40 (5.15), relative_kv's fields, its tables' shapes and its combination with position_bias.
    doc_case("relative_kv_distance_negative", L, 40, "5.15", lambda d: node(d, "attention").update(
        relative_kv={"key": "token_embd.weight", "max_distance": -1}))
    doc_case("relative_kv_without_key", L, 40, "5.15", lambda d: node(d, "attention").update(
        relative_kv={"value": "token_embd.weight", "max_distance": 2}))
    doc_case("relative_kv_unknown_field", L, 40, "5.15", lambda d: node(d, "attention").update(
        relative_kv={"key": "token_embd.weight", "max_distance": 2, "buckets": [0]}))
    doc_case("relative_kv_key_shape", L, 40, "5.15", lambda d: node(d, "attention").update(
        relative_kv={"key": "token_embd.weight", "value": None, "max_distance": 3}))
    doc_case("relative_kv_value_shape", L, 40, "5.15", lambda d: node(d, "attention").update(
        relative_kv={"key": "token_embd.weight", "value": "output_norm.weight",
                     "max_distance": d["model"]["vocab"] - 1}))
    doc_case("relative_kv_with_position_bias", L, 40, "5.15", lambda d: node(d, "attention").update(
        relative_kv={"key": "token_embd.weight", "max_distance": d["model"]["vocab"] - 1},
        position_bias={"weight": "token_embd.weight", "buckets": [0]}))
    doc_case("relative_kv_per_layer", L, 6, "3.8", lambda d: node(d, "attention").update(
        relative_kv={"per_layer": [{"key": "token_embd.weight", "max_distance": 0}] * d["block"]["repeat"]}))
    # v1.5: rules 10 (softcap, attn_factor, split groups), 41 (freq_factors), 42 (mla_attention's tensors), 12 (its
    # kv slot), 7, 8, 14 and 6.
    G2, P3, DS = "tp_gemma2", "tp_phi3", "tp_deepseek2"
    doc_case("softcap_not_positive", G2, 10, "5.15", lambda d: node(d, "attention").update(softcap=0.0))
    doc_case("attn_factor_not_positive", P3, 10, "5.14", lambda d: node(d, "rope").update(attn_factor=-1.25))
    doc_case("freq_factors_without_original_ctx", P3, 41, "5.14",
             lambda d: node(d, "rope")["freq_factors"].pop("original_ctx"))
    doc_case("freq_factors_unknown_field", P3, 41, "5.14",
             lambda d: node(d, "rope")["freq_factors"].update(medium="rope_factors_short.weight"))
    doc_case("freq_factors_tensor_shape", P3, 41, "5.14",
             lambda d: node(d, "rope").update(freq_factors="output_norm.weight"))
    doc_case("freq_factors_per_layer", P3, 6, "3.8", lambda d: node(d, "rope").update(
        freq_factors={"per_layer": ["rope_factors_short.weight"] * d["block"]["repeat"]}))
    doc_case("split_groups_zero", DS, 10, "5.5", lambda d: node(d, "split").update(groups=0))
    doc_case("mla_slot_head_dim", DS, 12, "5.20", lambda d: kv(d).update(head_dim=8))
    doc_case("mla_slot_kv_heads", DS, 12, "5.20", lambda d: kv(d).update(kv_heads=2))
    doc_case("mla_k_b_shape", DS, 42, "5.20", lambda d: node(d, "mla_attention").update(k_b="blk.{i}.attn_v_b.weight"))
    doc_case("mla_reads_three", DS, 7, "3.5", lambda d: node(d, "mla_attention")["in"].pop())
    doc_case("mla_without_k_b", DS, 8, "3.5", lambda d: node(d, "mla_attention").pop("k_b"))
    doc_case("mla_not_causal", DS, 14, "5.20", lambda d: node(d, "mla_attention").update(causal=False))
    # v1.2: rule 35 (3.2), a duplicate member name at any depth; the documents as JSON texts.
    base = json.dumps(load_model(SUITE / "models" / L)[0])

    def text_case(name, text):
        out.append((name, {"text": text, "gguf": f"models/{L}/{L}.gguf", "rule": 35, "section": "3.2"}))
    text_case("duplicate_member_top", base[:-1] + ', "family_hint": "llama"}')
    assert base.count('{"op": "attention", ') == 1
    text_case("duplicate_member_node", base.replace('{"op": "attention", ', '{"op": "attention", "causal": true, ', 1))
    text_case("duplicate_member_prose", base[:-1] + ', "_note": {"a": 1, "a": 2}}')
    # Container rules: GGUFs whose metadata breaks one rule.
    gg = []

    def gguf_case(name, base, rule, section, f):
        doc, W, meta, _ = load_model(SUITE / "models" / base)
        meta, doc = dict(meta), copy.deepcopy(doc)
        f(meta, doc)
        path = SUITE / "refuse" / f"{name}.gguf"
        write_gguf(path, meta, W, doc)
        gg.append((name, {"gguf": f"refuse/{name}.gguf", "rule": rule, "section": section}))

    gguf_case("version_key_mismatch", L, 24, "3.1", lambda m, d: None)
    gguf_case("state_tokens_k_mismatch", S, 28, "6.8.1",
              lambda m, d: m.__setitem__("paperkiln.state_tokens.k", (4, 3)))
    gguf_case("state_template_not_v1", S, 28, "8.2", lambda m, d: m.__setitem__(
        "paperkiln.chat_template", (8, json.dumps({"name": "paperkiln", "version": 2}))))
    gguf_case("weight_index_block_count", "pk_recursive", 27, "7.1",
              lambda m, d: m.__setitem__("paperkiln.block_count", (4, 9)))
    gguf_case("state_tokens_arch_llama", S, 28, "8.2", lambda m, d: m.__setitem__("general.architecture", (8, "llama")))
    # v1.2: an embedded document holding a duplicate member name (rule 35).
    doc, W, meta, _ = load_model(SUITE / "models" / L)
    text = json.dumps(doc, separators=(",", ":"))
    assert text.count('"model":{') == 1
    write_gguf(SUITE / "refuse" / "duplicate_member_embedded.gguf", dict(meta), W,
               ir_text=text.replace('"model":{', '"model":{"ctx":64,', 1))
    gg.append(("duplicate_member_embedded", {"gguf": "refuse/duplicate_member_embedded.gguf", "rule": 35,
                                             "section": "3.2"}))
    # version_key_mismatch: rewrite the version key by hand (write_gguf derives it from the document).
    _patch_version_key(SUITE / "refuse" / "version_key_mismatch.gguf", 7)

    # v1.3: rule 38 (3.1), a llama.cpp architecture on a file whose document is not that architecture's graph.
    def relabelled(m, arch):
        old = m["general.architecture"][1]
        out_m = {}
        for k, v in m.items():
            out_m[arch + k[len(old):] if k.startswith(old + ".") and not k.startswith("paperkiln.ir") else k] = v
        out_m["general.architecture"] = (8, arch)
        return out_m

    def arch_case(name, base, f):
        doc, W, meta, _ = load_model(SUITE / "models" / base)
        meta = f(dict(meta))
        write_gguf(SUITE / "refuse" / f"{name}.gguf", meta, W, doc)
        gg.append((name, {"gguf": f"refuse/{name}.gguf", "rule": 38, "section": "3.1"}))
    # kimi's linear attention under gpt2 (the pre-1.3 export): llama.cpp computes softmax gpt2 on it.
    arch_case("architecture_kimi_as_gpt2", "pk_kimi", lambda m: relabelled(m, "gpt2"))
    arch_case("architecture_swa_as_gpt2", "pk_gpt2_swa", lambda m: relabelled(m, "gpt2"))
    # A llama graph whose metadata states another RoPE base than its document.
    arch_case("architecture_metadata_disagrees", L, lambda m: dict(m, **{"llama.rope.freq_base": (6, 500000.0)}))
    # An architecture section 9.3 does not define (nothing states what a program building from it computes;
    # "phi3" until v1.5.0, which defines it).
    arch_case("architecture_without_graph", L, lambda m: relabelled(m, "falcon"))
    # v1.5.0: a gemma2 file whose embedded document drops the attention soft cap its metadata states.
    doc, W, meta, _ = load_model(SUITE / "models" / "tp_gemma2")
    doc = copy.deepcopy(doc)
    for n in doc["block"]["nodes"]:
        n.pop("softcap", None)
    write_gguf(SUITE / "refuse" / "architecture_gemma2_softcap_dropped.gguf", dict(meta), W, doc)
    gg.append(("architecture_gemma2_softcap_dropped", {"gguf": "refuse/architecture_gemma2_softcap_dropped.gguf",
                                                       "rule": 38, "section": "3.1"}))
    # v1.3.1: rule 38 judges the file's embedded document even when a valid document is read in place of it.
    gg.append(("architecture_substituted_document", {
        "doc": load_model(SUITE / "models" / "pk_kimi")[0], "gguf": "refuse/architecture_kimi_as_gpt2.gguf",
        "rule": 38, "section": "3.1"}))
    return out + gg


def _patch_version_key(path, value):
    meta, tensors, data_start, raw, (i0, i1) = ir_validate.read_gguf(path)
    key = b"paperkiln.ir_version"
    at = raw.index(key) + len(key)
    raw = bytearray(raw)
    raw[at + 4:at + 8] = int(value).to_bytes(4, "little")  # after the uint32 type tag
    Path(path).write_bytes(bytes(raw))


# ---------------------------------------------------------------- digest vectors (IR_SPEC 3.10)

# RFC 8785 Appendix B's numbers (as IEEE doubles), with the forms ECMAScript writes; checked against
# V8 (Node 24) when the vectors were made.
RFC8785_NUMBERS = [
    "0x0000000000000000", "0x0000000000000001", "0x8000000000000001", "0x7fefffffffffffff", "0xffefffffffffffff",
    "0x4340000000000000", "0xc340000000000000", "0x4430000000000000", "0x44b52d02c7e14af5", "0x44b52d02c7e14af6",
    "0x44b52d02c7e14af7", "0x444b1ae4d6e2ef4e", "0x444b1ae4d6e2ef4f", "0x444b1ae4d6e2ef50", "0x3eb0c6f7a0b5ed8c",
    "0x3eb0c6f7a0b5ed8d", "0x41b3de4355555553", "0x41b3de4355555554", "0x41b3de4355555555", "0x41b3de4355555556",
    "0x41b3de4355555557", "0xbecbf647612f3696", "0x43143ff3c1cb0959"]


def digest_vectors():
    """{"vectors": [{"name", "text", "canonical", "sha256"} | {"name", "text", "refused": reason}]}: the
    serialiser on numbers, strings and member names (with prose=True: '_' keys kept, so the serialiser is
    tested alone), the digest of every suite model's document, and texts that have no digest."""
    import struct
    vec = []

    def ok(name, text, prose=False):
        vec.append({"name": name, "text": text, "prose": prose, "canonical": ir_validate.canonical(text, prose),
                    "sha256": ir_validate.digest(text, prose)})

    def refused(name, text, why):
        try:
            ir_validate.digest(text)
        except ir_validate.DigestError:
            vec.append({"name": name, "text": text, "refused": why})
            return
        raise SystemExit(f"digest vector {name}: expected no digest")

    for h in RFC8785_NUMBERS:
        x = struct.unpack(">d", bytes.fromhex(h[2:]))[0]
        ok(f"rfc8785_number_{h[2:]}", json.dumps(x), prose=True)
    for name, x in (("f32_eps", 1e-05), ("f32_rope_theta", 10000.0), ("f32_scale", 0.3535533845424652),
                    ("f32_den_floor", 1e-08), ("integer_2p53", 2 ** 53), ("negative_integer", -7),
                    ("minus_zero_integer_form", "-0"), ("exponent_form", "1E+2")):
        ok(name, x if isinstance(x, str) else json.dumps(x), prose=True)
    strs = {"controls": "a\u0000b\u001f\u007f", "escapes": 'q"b\\ tab\tnl\ncr\rbs\bff\f', "line_separators":
            "  ", "astral": "\U0001F600", "accents": "é é", "names": "blk.{i}.attn_q.weight"}
    for name, s in strs.items():
        ok(f"string_{name}", json.dumps(s, ensure_ascii=False), prose=True)
        ok(f"string_{name}_escaped", json.dumps(s, ensure_ascii=True), prose=True)
    ok("key_order_utf16", json.dumps({"": 1, "\U0001F600": 2, "b": 3, "a": 4, "A": 5, "é": 6, "": 7},
                                     ensure_ascii=False), prose=True)
    ok("nesting", '{"b": [1, {"d": true, "c": null}], "a": {"z": [], "y": {}}}', prose=True)
    ok("prose_removed", '{"op": "relu", "_note": "dropped", "in": ["x"], "out": "y", "x": {"_a": 1, "b": 2}}')
    ok("prose_kept_in_serialiser_test", '{"op": "relu", "_note": "kept", "in": ["x"]}', prose=True)
    for md in model_dirs():
        doc_text = ir_validate.document_text(md / f"{md.name}.gguf")
        ok(f"document_{md.name}", doc_text)
    refused("negative_zero", '{"op": "affine", "in": ["x"], "scale": 1, "shift": -0.0, "out": "y"}',
            "a negative zero (RFC 8785 writes it as 0)")
    refused("integer_beyond_2p53", "9007199254740993", "an integer beyond 2^53 (not I-JSON)")
    refused("overflow", "1e400", "a number beyond the double range")
    refused("duplicate_key", '{"op": "relu", "op": "silu"}', "a duplicate member name")
    refused("lone_surrogate", '"\\ud800"', "a lone surrogate (not well-formed Unicode)")
    refused("nan_literal", "NaN", "not JSON")
    # Step 1 runs on the whole document: what a '_' member holds can take the digest away (v1.1.1).
    refused("prose_overflow", '{"op": "relu", "_a": 1e400}', "a number beyond the double range under a '_' key")
    refused("prose_negative_zero", '{"op": "relu", "_a": -0.0}', "a negative zero under a '_' key")
    refused("prose_integer_beyond_2p53", '{"op": "relu", "_a": 9007199254740993}', "an integer beyond 2^53 under a '_' key")
    refused("prose_lone_surrogate", '{"op": "relu", "_a": "\\ud800"}', "a lone surrogate under a '_' key")
    refused("prose_key_lone_surrogate", '{"op": "relu", "_\\ud800": 1}', "a lone surrogate in a '_' key's name")
    refused("prose_duplicate_key", '{"op": "relu", "_a": 1, "_a": 2}', "a duplicate '_' member name")
    # v1.2 (3.10 step 1): the integer range is judged by the number's lexical form. A number with a fraction or
    # an exponent is its nearest binary64 value; only an integer token must lie within -2^53 .. 2^53.
    ok("fraction_form_beyond_2p53", "9007199254740993.0", prose=True)
    ok("exponent_form_1e300", "1e300", prose=True)
    ok("negative_integer_2p53", "-9007199254740992", prose=True)
    refused("negative_integer_beyond_2p53", "-9007199254740993", "an integer beyond -2^53 (not I-JSON)")
    return {"_comment": "spec/IR_SPEC.md section 3.10: the IR digest. canonical is the RFC 8785 serialisation "
                        "(with prose false, every '_' key removed first: the digest); sha256 is the digest of "
                        "text. A refused text has no digest.", "vectors": vec}


# ---------------------------------------------------------------- dequantisation fixtures (IR_SPEC 4.7)

def quant_cases():
    """Random blocks of every storage type tools/ir_quant.py defines (fp16 fields finite: normal, subnormal,
    signed, zero, the largest), two rows of two blocks each, with their f32 values (EXACT)."""
    import ir_quant
    rng = np.random.default_rng(20261025)
    halves = {"Q4_0": [0], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}
    special = np.array([0.0, -0.0, 6e-08, -3e-05, 65504.0, 1e-04, -0.5], np.float64)
    cases = []
    for name, (gid, qk, size) in ir_quant.TYPES.items():
        rows, per_row = 2, (2 * qk if qk > 1 else 8)
        n = rows * per_row
        if qk == 1:  # F32, F16, BF16: values, the special ones included
            vals = np.concatenate([rng.standard_normal(n - len(special)) * 0.5, special]).astype(np.float32)
            raw = (vals.astype("<f4") if name == "F32" else vals.astype("<f2") if name == "F16"
                   else (vals.view(np.uint32) >> 16).astype("<u2")).tobytes()
        else:
            nb = n // qk
            b = rng.integers(0, 256, (nb, size), dtype=np.uint8)
            pool = np.concatenate([rng.standard_normal(16) * 0.05, special])
            for at in halves[name]:
                h = rng.choice(pool, nb).astype("<f2")
                b[:, at:at + 2] = h.view(np.uint8).reshape(nb, 2)
            raw = b.tobytes()
        vals = ir_quant.dequantize(name, raw, n).reshape(rows, per_row)
        cases.append({"name": name.lower(), "type": name, "ggml_type": gid, "rows": rows, "n": per_row,
                      "bytes": raw.hex(), "expect": enc(vals), "class": "EXACT"})
    return {"_comment": "spec/IR_SPEC.md section 4.7: each case's bytes are rows x n elements of the storage type, "
                        "row-major; expect is their f32 values, EXACT on every backend.", "cases": cases}


# ---------------------------------------------------------------- v1.2: accept, chat and truncate cases

def accept_cases():
    """IR_SPEC 11.2: the documents every consumer must accept: each suite model with its own document, each
    rewritten document in place of its model's, and each worked document alone (examples/, no GGUF)."""
    cases = [{"name": f"model_{md.name}", "gguf": f"models/{md.name}/{md.name}.gguf"} for md in model_dirs()]
    cases += [{"name": f"rewritten_{n}", "gguf": f"models/{n}/{n}.gguf", "ir": f"recurrence/{n}.ir.json"}
              for n in REC_MODELS]
    cases += [{"name": f"example_{p.name[:-len('.ir.json')]}", "ir": f"../examples/{p.name}"}
              for p in sorted((ROOT / "examples").glob("*.ir.json"))]
    # v1.3.1 (3.2, rule 35): a valid document read in place of a file whose embedded document holds a duplicate
    # member name is accepted; rule 35 judges the document read.
    (SUITE / "accept" / "pk_llama.ir.json").write_text(
        json.dumps(load_model(SUITE / "models" / "pk_llama")[0], indent=1) + "\n", encoding="utf-8")
    cases.append({"name": "substituted_over_duplicate_embedded", "gguf": "refuse/duplicate_member_embedded.gguf",
                  "ir": "accept/pk_llama.ir.json"})
    # v1.3.5 (7.2): pk_llama_attnres_rec's document without its tensor_index tags (also the document of the
    # trace pk_llama_attnres_rec_untagged).
    (SUITE / "accept" / f"{UNTAGGED}_untagged.ir.json").write_text(
        json.dumps(untagged_document(), indent=1) + "\n", encoding="utf-8")
    cases.append({"name": "untagged_application_queries", "gguf": f"models/{UNTAGGED}/{UNTAGGED}.gguf",
                  "ir": f"accept/{UNTAGGED}_untagged.ir.json"})
    # v1.4.1 (3.7, 7.1): pk_flex_shaw under weight_index [1, 0]; relative_kv's names read stored layer m_i.
    (SUITE / "accept" / f"{PERMUTED}_weight_index.ir.json").write_text(
        json.dumps(permuted_document(), indent=1) + "\n", encoding="utf-8")
    cases.append({"name": "relative_kv_weight_index", "gguf": f"models/{PERMUTED}/{PERMUTED}.gguf",
                  "ir": f"accept/{PERMUTED}_weight_index.ir.json"})
    # v1.4.2 (5.15, 7.5; Q16): pk_mor under window 3, sinks 1: a window under depth_route counts positions.
    (SUITE / "accept" / f"{MOR_WINDOW}_window.ir.json").write_text(
        json.dumps(mor_window_document(), indent=1) + "\n", encoding="utf-8")
    cases.append({"name": "mor_window", "gguf": f"models/{MOR_WINDOW}/{MOR_WINDOW}.gguf",
                  "ir": f"accept/{MOR_WINDOW}_window.ir.json"})
    for c in cases:
        errs = ir_validate.check_errors(SUITE / c["gguf"] if "gguf" in c else None,
                                        SUITE / c["ir"] if "ir" in c else None)
        if errs:
            raise SystemExit(f"accept case {c['name']}: the reference refuses it: {errs[:2]}")
    return {"_comment": "spec/IR_SPEC.md section 11.2: documents every consumer must accept (check returns "
                        "accepted). gguf is a suite model; ir, when present, is read in place of its own document, "
                        "or alone when there is no gguf (the document rules only).", "cases": cases}


def chat_cases():
    """IR_SPEC 8.1.3 and 8.1.6: chat/render.json's conversations and records (paperkiln's
    tests/chat_template_cases.json, the hand-written cases of gate G5: 32 v1 cases, and from v1.4.2 the 42 of
    paperkiln/2, paperkiln 22f8fe6d), each case's expected text or refusal from tools/ir_chat.py. A case whose
    file already states a verdict (paperkiln's, when it was imported) must get the same one here."""
    import ir_chat
    src = json.loads((SUITE / "chat" / "render.json").read_text(encoding="utf-8"))
    cases = []
    for c in src["cases"]:
        was = c.get("expect", True if c.get("refused") else None)
        c = {k: v for k, v in c.items() if k not in ("expect", "refused")}
        rec = src["records"][c["record"]] if isinstance(c["record"], str) else c["record"]
        try:
            c["expect"] = ir_chat.render(rec, c["messages"], c.get("generation_prompt", False), c.get("thinking"),
                                         c.get("tools"))
        except ir_chat.ChatError:
            c["refused"] = True
        if was is not None and was != c.get("expect", True):
            raise SystemExit(f"chat case {c['name']}: tools/ir_chat.py gives {c.get('expect', 'a refusal')!r}, the "
                             f"file states {was!r}")
        cases.append(c)
    return {"_comment": "spec/IR_SPEC.md sections 8.1.3 and 8.1.6: render_chat(record, messages, generation_prompt, "
                        "thinking, tools) gives expect, or refuses where refused is true. A case's record is a key of "
                        "records or a record itself; tools, when present, is the conversation's tool definitions. "
                        "Inputs: paperkiln's tests/chat_template_cases.json (gate G5).",
            "records": src["records"], "cases": cases}


def parse_reply_cases():
    """IR_SPEC 8.1.6 (v1.4.2): chat/parse_reply.json, paperkiln's 12 parse_reply cases (tests/chat_template_cases.json
    at 22f8fe6d): a reply's ids over a toy vocabulary (tokens: id -> text), split on call_id. expect from
    tools/ir_chat.py, null where a part does not parse; a stated verdict must be reproduced."""
    import ir_chat
    src = json.loads((SUITE / "chat" / "parse_reply.json").read_text(encoding="utf-8"))
    cases = []
    for c in src["cases"]:
        def dec(ids, tok=c["tokens"]):
            return "".join(tok[str(i)] for i in ids)
        got = ir_chat.parse_reply(c["ids"], c["call_id"], dec)
        if "expect" in c and c["expect"] != got:
            raise SystemExit(f"parse_reply case {c['name']}: tools/ir_chat.py gives {got!r}, the file states "
                             f"{c['expect']!r}")
        cases.append(dict(c, expect=got))
    return {"_comment": "spec/IR_SPEC.md section 8.1.6: parse_reply(tokens, ids, call_id) splits the reply's ids at "
                        "every call_id and decodes each part by concatenating its ids' texts from tokens; expect is "
                        "{content, tool_calls}, or null where a later part is not name + newline + arguments with a "
                        "non-empty name (the engine then returns the whole reply as content). Inputs: paperkiln's "
                        "tests/chat_template_cases.json (parse_reply).", "cases": cases}


def extra_traces():
    """v1.2 (IR_SPEC 6.9): truncate on models with recurrent state (a fresh run's state over the first n tokens:
    a snapshot or a run from the start). Each draws from a stream of its own."""
    out = []
    for name in ("pk_srd", "tp_mamba1", "pk_flex_mdha"):  # pk_flex_mdha: v1.4, a conv slot beside the kv slot
        doc, W, meta, _ = load_model(SUITE / "models" / name)
        rng = np.random.default_rng(sum(map(ord, name)) + 1200)
        V = doc["model"]["vocab"]
        ids = [int(x) for x in rng.integers(1, V, 9)]
        out.append((f"{name}_truncate", trace_case(name, doc, W, [
            {"method": "forward", "ids": ids}, {"method": "truncate", "n": 5},
            {"method": "forward", "ids": [int(x) for x in rng.integers(1, V, 2)]}])))
    return out


def window_traces():
    """v1.3 (IR_SPEC 6.9): a rewind into a sliding window, behind rows a windowed cache would have dropped: the
    state a fresh run over the first n tokens leaves (the engine keeps or re-prefills them). Each draws from a
    stream of its own."""
    out = []
    for name, total, cut, new in (("pk_gpt2_swa", 14, 7, 3), ("tp_gemma3", 48, 10, 4), ("tp_mistral_hd16", 14, 7, 3),
                                  ("tp_gemma2", 14, 7, 3)):
        doc, W, meta, _ = load_model(SUITE / "models" / name)
        rng = np.random.default_rng(sum(map(ord, name)) + 1300)
        V = doc["model"]["vocab"]
        ids = [int(x) for x in rng.integers(1, V, total)]
        out.append((f"{name}_rewind_window", trace_case(name, doc, W, [
            {"method": "forward", "ids": ids}, {"method": "truncate", "n": cut},
            {"method": "forward", "ids": [int(x) for x in rng.integers(1, V, new)]}])))
    return out


UNTAGGED = "pk_llama_attnres_rec"


def untagged_document():
    """v1.3.5 (7.2): pk_llama_attnres_rec's document without its tensor_index tags. Valid (every
    rec_app.{m_i} is a tensor of the file; a pseudo-query may serve several source counts), and another
    function: application i reads the queries of application m_i."""
    doc = copy.deepcopy(load_model(SUITE / "models" / UNTAGGED)[0])
    for n in doc["block"]["nodes"]:
        n.pop("tensor_index", None)
    return doc


PERMUTED = "pk_flex_shaw"


def permuted_document():
    """v1.4.1 (3.7, 7.1): pk_flex_shaw's document with weight_index [1, 0]: application 0 reads stored layer 1's
    tensors and application 1 layer 0's, relative_kv's key and value among them (the file has architecture
    paperkiln and block_count 2, so the document is valid). An engine that binds relative_kv's names to the
    application index instead of m_i computes another function."""
    doc = copy.deepcopy(load_model(SUITE / "models" / PERMUTED)[0])
    doc["block"]["weight_index"] = {"per_layer": [1, 0]}
    return doc


def permuted_traces():
    """v1.4.1: the trace of the permuted document over pk_flex_shaw's file (a prefill of 4, then token-by-token
    decode). Draws from a stream of its own."""
    doc, W, meta, _ = load_model(SUITE / "models" / PERMUTED)
    alt = permuted_document()
    rng = np.random.default_rng(sum(map(ord, PERMUTED)) + 1600)
    V = doc["model"]["vocab"]
    ids = [int(x) for x in rng.integers(1, V, 8)]
    case = trace_case(PERMUTED, alt, W, [{"method": "forward", "ids": ids[:4]}]
                      + [{"method": "forward", "ids": [t]} for t in ids[4:]])
    case["ir"] = f"accept/{PERMUTED}_weight_index.ir.json"
    return [(f"{PERMUTED}_weight_index", case)]


MOR_WINDOW = "pk_mor"


def mor_window_document():
    """v1.4.2 (5.15, 7.5; Q16): pk_mor's document with window 3 and sinks 1 on its kv slot and attention, every
    application. Under depth_route a step's queries and keys are its active rows at their absolute positions, and
    the window counts positions, never rows: a key at position j is visible from a query at p iff j <= p and
    (j < 1 or j >= p - 2), among the rows the application holds. An engine that counts the window in the active
    rows computes another function."""
    doc = copy.deepcopy(load_model(SUITE / "models" / MOR_WINDOW)[0])
    for s in doc["block"]["state"]:
        if s["kind"] == "kv":
            s["window"], s["sinks"] = 3, 1
    for n in doc["block"]["nodes"]:
        if n["op"] == "attention":
            n["window"], n["sinks"] = 3, 1
    return doc


def mor_window_traces():
    """v1.4.2 (Q16): the windowed document over pk_mor's file: a prefill of 4, a call of 3, then token-by-token
    decode to 14 ids (prefill = decode under the window); and a rewind into the window (14 ids, a cut to 8, 3 new
    ids: the state of a fresh run over the first 8, 6.9). Each draws from a stream of its own."""
    doc, W, meta, _ = load_model(SUITE / "models" / MOR_WINDOW)
    alt = mor_window_document()
    V = doc["model"]["vocab"]
    rng = np.random.default_rng(sum(map(ord, MOR_WINDOW)) + 1700)
    ids = [int(x) for x in rng.integers(1, V, 14)]
    dec = trace_case(MOR_WINDOW, alt, W, [{"method": "forward", "ids": ids[:4]}, {"method": "forward", "ids": ids[4:7]}]
                     + [{"method": "forward", "ids": [t]} for t in ids[7:]])
    rng = np.random.default_rng(sum(map(ord, MOR_WINDOW)) + 1800)
    ids = [int(x) for x in rng.integers(1, V, 14)]
    rew = trace_case(MOR_WINDOW, alt, W, [{"method": "forward", "ids": ids}, {"method": "truncate", "n": 8},
                                         {"method": "forward", "ids": [int(x) for x in rng.integers(1, V, 3)]}])
    for c in (dec, rew):
        c["ir"] = f"accept/{MOR_WINDOW}_window.ir.json"
    return [(f"{MOR_WINDOW}_window", dec), (f"{MOR_WINDOW}_window_rewind", rew)]


def untagged_traces():
    """v1.3.5 (7.2): the trace of the untagged document over pk_llama_attnres_rec's file (a prefill of 4, then
    token-by-token decode), its expected values from that document, so an engine that reads the application
    index wherever a name is rec_app.{i}, or in every node once one is tagged, fails it while the tagged
    model's cases pass. Draws from a stream of its own."""
    doc, W, meta, _ = load_model(SUITE / "models" / UNTAGGED)
    alt = untagged_document()
    rng = np.random.default_rng(sum(map(ord, UNTAGGED)) + 1500)
    V = doc["model"]["vocab"]
    ids = [int(x) for x in rng.integers(1, V, 8)]
    case = trace_case(UNTAGGED, alt, W, [{"method": "forward", "ids": ids[:4]}]
                      + [{"method": "forward", "ids": [t]} for t in ids[4:]])
    case["ir"] = f"accept/{UNTAGGED}_untagged.ir.json"
    return [(f"{UNTAGGED}_untagged", case)]


def chunk_traces():
    """v1.3 (H1 of adapters/hf/FINDINGS.txt): every model's decode state after forwards of several ids over an
    existing state (a prefill of 4, then 3 ids, 1 id and 2 ids), so an engine that handles only n = 1
    correctly after a prefill fails a case of its own. Each draws from a stream of its own."""
    out = []
    for md in model_dirs():
        name = md.name
        doc, W, meta, _ = load_model(md)
        rng = np.random.default_rng(sum(map(ord, name)) + 1400)
        V = doc["model"]["vocab"]
        ids = [int(x) for x in rng.integers(1, V, 10)]
        out.append((f"{name}_chunks", trace_case(name, doc, W, [
            {"method": "forward", "ids": ids[:4]}, {"method": "forward", "ids": ids[4:7]},
            {"method": "forward", "ids": ids[7:8]}, {"method": "forward", "ids": ids[8:]}])))
    return out


# ---------------------------------------------------------------- regenerate

def regenerate():
    (SUITE / "models").mkdir(parents=True, exist_ok=True)
    for sub in ("ops", "state", "refuse"):
        d = SUITE / sub  # emptied file by file (OneDrive can hold a directory handle open)
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    build_tp_models()
    manifest = []
    rng = np.random.default_rng(11)
    for md in model_dirs():
        name = md.name
        doc, W, meta, path = load_model(md)
        errs = ir_validate.file_errors(path)
        if errs:
            raise SystemExit(f"{name}: the validator refuses the model: {errs[:3]}")
        mrng = np.random.default_rng(sum(map(ord, name))) if name in OWN_STREAM else rng
        case = model_case(name, doc, W, mrng)
        (md / "case.json").write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
        manifest.append({"id": f"model/{name}", "kind": "model", "file": f"models/{name}/case.json",
                         "model": f"models/{name}", "profile": _profile(name), "class": "ROUNDING"})
        for tname, t in traces(name, doc, W, mrng).items():
            f = f"state/{name}_{tname}.json"
            (SUITE / f).write_text(json.dumps(t, indent=1) + "\n", encoding="utf-8")
            manifest.append({"id": f"trace/{name}_{tname}", "kind": "trace", "file": f, "model": f"models/{name}",
                             "profile": "state" if tname.startswith(("evict", "rewind", "snapshot")) else _profile(name)})
    for op, cs in op_cases().items():
        f = f"ops/{op}.json"
        (SUITE / f).write_text(json.dumps({"op": op, "cases": cs}, indent=1) + "\n", encoding="utf-8")
        manifest.append({"id": f"op/{op}", "kind": "op", "file": f,
                         "profile": {"recurrence": "recurrence", "mla_attention": "thirdparty"}.get(op, "core"),
                         "n": len(cs)})
    for name, case in refusals():
        f = f"refuse/{name}.json"
        (SUITE / f).write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
        manifest.append({"id": f"refuse/{name}", "kind": "refuse", "file": f, "rule": case["rule"],
                         "section": case["section"]})
    (SUITE / "digest").mkdir(exist_ok=True)
    dv = digest_vectors()
    (SUITE / "digest" / "vectors.json").write_text(json.dumps(dv, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    manifest.append({"id": "digest/vectors", "kind": "digest", "file": "digest/vectors.json", "profile": "core",
                     "n": len(dv["vectors"])})
    (SUITE / "quant").mkdir(exist_ok=True)
    qc = quant_cases()
    (SUITE / "quant" / "blocks.json").write_text(json.dumps(qc, indent=1) + "\n", encoding="utf-8")
    manifest.append({"id": "dequant/blocks", "kind": "dequant", "file": "quant/blocks.json", "profile": "core",
                     "n": len(qc["cases"])})
    (SUITE / "grad").mkdir(exist_ok=True)
    gc = grad_cases()
    (SUITE / "grad" / "recurrence.json").write_text(json.dumps(gc, indent=1) + "\n", encoding="utf-8")
    manifest.append({"id": "grad/recurrence", "kind": "grad", "file": "grad/recurrence.json", "profile": "recurrence",
                     "n": len(gc["cases"])})
    ga = grad_attention_cases()
    (SUITE / "grad" / "attention.json").write_text(json.dumps(ga, indent=1) + "\n", encoding="utf-8")
    manifest.append({"id": "grad/attention", "kind": "grad", "file": "grad/attention.json", "profile": "core",
                     "n": len(ga["cases"])})
    (SUITE / "recurrence").mkdir(exist_ok=True)
    for name, doc, case in recurrence_traces():
        (SUITE / "recurrence" / f"{name}.ir.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        f = f"state/{name}_recurrence.json"
        (SUITE / f).write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
        manifest.append({"id": f"trace/{name}_recurrence", "kind": "trace", "file": f, "model": f"models/{name}",
                         "profile": "recurrence"})
    for name, case in (extra_traces() + window_traces() + chunk_traces() + untagged_traces() + permuted_traces()
                       + mor_window_traces()):
        f = f"state/{name}.json"
        (SUITE / f).write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
        manifest.append({"id": f"trace/{name}", "kind": "trace", "file": f, "model": case["model"],
                         "profile": "state"})
    (SUITE / "accept").mkdir(exist_ok=True)
    ac = accept_cases()
    (SUITE / "accept" / "documents.json").write_text(json.dumps(ac, indent=1) + "\n", encoding="utf-8")
    manifest.append({"id": "accept/documents", "kind": "accept", "file": "accept/documents.json", "profile": "core",
                     "n": len(ac["cases"])})
    cc = chat_cases()
    (SUITE / "chat" / "render.json").write_text(json.dumps(cc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest.append({"id": "chat/render", "kind": "chat", "file": "chat/render.json", "profile": "chat",
                     "n": len(cc["cases"])})
    pc = parse_reply_cases()
    (SUITE / "chat" / "parse_reply.json").write_text(json.dumps(pc, indent=1, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
    manifest.append({"id": "chat/parse_reply", "kind": "chat", "file": "chat/parse_reply.json", "profile": "chat",
                     "n": len(pc["cases"])})
    (SUITE / "manifest.json").write_text(json.dumps({
        "_comment": "spec/IR_SPEC.md section 11: the conformance suite's cases. Generated by tools/ir_conformance.py "
                    "--regenerate from the models in models/ (expected values: tools/ir_numpy.py).",
        "ir_version": 1, "cases": manifest}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} cases")
