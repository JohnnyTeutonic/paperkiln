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
    "pk_llama_attnres_rec": "recursion", "pk_state": "state", "pk_recurrence": "recurrence"}
# Models added after the suite's first release draw their prompts from a stream of their own, so the cases of
# every earlier model stay byte-identical when the suite is regenerated.
OWN_STREAM = {"pk_recurrence"}


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
}
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
        if a == "gemma3":
            sh.update({b + "post_attention_norm.weight": [d], b + "post_ffw_norm.weight": [d]})
    if a == "gptneox":
        sh.update({"output_norm.weight": [d], "output_norm.bias": [d], "output.weight": [V, d]})
    else:
        sh["output_norm.weight"] = [d]
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
            if t.endswith("norm.weight"):
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
    elif name in ("pk_llama", "pk_mor", "tp_qwen3"):
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
    doc_case("unknown_op", L, 4, "10.1 rule 1", lambda d: d["block"]["nodes"].append({"op": "warp", "in": ["x"], "out": "x"}))
    doc_case("unknown_attribute", L, 4, "10.1 rule 1", lambda d: node(d, "attention").update(temperature=2.0))
    doc_case("unknown_top_key", L, 1, "10.1 rule 1", lambda d: d.update(profiles=["core"]))
    doc_case("version_2", L, 2, "2.3", lambda d: d.update(ir_version=2))
    doc_case("repeat_ne_n_layers", L, 3, "3.3", lambda d: d["block"].update(repeat=3))
    doc_case("model_ctx_missing", L, 3, "3.3", lambda d: d["model"].pop("ctx"))
    doc_case("per_layer_wrong_length", L, 6, "3.8", lambda d: node(d, "rope").update(theta={"per_layer": [10000.0]}))
    doc_case("per_layer_in_head", L, 6, "3.8", lambda d: d["head"]["nodes"][0].update(eps={"per_layer": [1e-5, 1e-5]}))
    doc_case("add_one_input", L, 7, "10.1 rule 7", lambda d: node(d, "add").update(**{"in": ["x"]}))
    doc_case("rope_pairing_unknown", L, 5, "10.1 rule 1", lambda d: node(d, "rope").update(pairing="halves"))
    doc_case("rope_n_rot_odd", L, 10, "5.14", lambda d: node(d, "rope").update(n_rot=3))
    doc_case("attention_not_causal", L, 14, "5.15", lambda d: node(d, "attention").update(causal=False))
    doc_case("attention_heads_not_multiple", L, 10, "5.15", lambda d: node(d, "attention").update(heads=3))
    doc_case("attention_window_differs", L, 12, "5.15", lambda d: node(d, "attention").update(window=4))
    doc_case("slot_declared_twice", L, 11, "6.1", lambda d: d["block"]["state"].append(copy.deepcopy(kv(d))))
    doc_case("slot_unread", L, 11, "6.1", lambda d: d["block"]["state"].append(dict(copy.deepcopy(kv(d)), name="kv2")))
    doc_case("slot_unknown_kind", L, 11, "10.1 rule 1", lambda d: kv(d).update(kind="paged"))
    doc_case("attention_no_slot", L, 11, "6.1", lambda d: node(d, "attention").update(state="nowhere"))
    doc_case("kv_slot_field_missing", L, 12, "6.2", lambda d: kv(d).pop("max_len"))
    doc_case("tensor_index_without_weight_index", L, 19, "7.2",
             lambda d: node(d, "linear").update(tensor_index="application"))
    doc_case("name_with_i_outside_block", L, 9, "3.7",
             lambda d: d["head"]["nodes"][0].update(weight="blk.{i}.output_norm.weight"))
    doc_case("evict_on_causal", L, 16, "6.8.3", lambda d: kv(d).update(evict="turn_state"))
    doc_case("missing_tensor", L, 25, "3.7", lambda d: node(d, "linear").update(weight="blk.{i}.nope.weight"))
    doc_case("gelu_approx_unknown", "pk_gpt2", 5, "10.1 rule 1", lambda d: node(d, "gelu").update(approx="sigmoid"))
    doc_case("split_count", "pk_gpt2", 10, "5.5", lambda d: node(d, "split").update(sizes=[16, 32]))
    doc_case("expr_does_not_parse", "pk_flex_expr", 9, "5.19", lambda d: node(d, "expr").update(expr="max(x,"))
    doc_case("moe_top_k_too_large", "pk_moe", 10, "5.16", lambda d: node(d, "moe_ffn").update(top_k=5))
    doc_case("moe_gate_norm_unknown", "pk_moe", 5, "10.1 rule 1", lambda d: node(d, "moe_ffn").update(gate_norm="topk"))
    doc_case("recurrent_dtype_f16", "pk_kimi", 13, "10.1 rule 1",
             lambda d: [s for s in d["block"]["state"] if s["kind"] == "recurrent"][0].update(dtype="f16"))
    doc_case("recurrent_shape_mismatch", "pk_kimi", 13, "6.3",
             lambda d: [s for s in d["block"]["state"] if s["kind"] == "recurrent"][0]["tensors"][0].update(shape=[3, 8]))
    doc_case("recurrent_op_mismatch", "pk_kimi", 13, "6.1",
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
    doc_case("mor_expert_choice", M, 20, "7.7", lambda d: node(d, "depth_route").update(route="expert_choice"))

    def steps(d):
        return node(d, "depth_route")["step"]["per_layer"]

    doc_case("mor_unequal_steps", M, 20, "7.4", lambda d: node(d, "depth_route").update(
        step={"per_layer": [-1, 0, 0, 1, -1]}))
    doc_case("mor_steps_descending", M, 20, "7.4", lambda d: node(d, "depth_route").update(
        step={"per_layer": [-1, 2, 1, 0, -1]}))
    doc_case("mor_step_length_wrong", M, 20, "7.4", lambda d: steps(d).pop())
    doc_case("mor_no_kv_rows_active", M, 21, "7.5", lambda d: kv(d).pop("kv_rows"))
    doc_case("mor_kv_from", M, 22, "7.6", lambda d: kv(d).update(kv_from={"per_layer": list(range(d["block"]["repeat"]))}))
    doc_case("mor_gate_unknown", M, 20, "7.4", lambda d: node(d, "depth_route").update(gate="sigmoid"))
    doc_case("mor_router_with_i", M, 20, "7.7", lambda d: node(d, "depth_route").update(weight="rec.{i}.router.weight"))
    doc_case("mor_kv_rows_shared", M, 21, "7.5", lambda d: kv(d).update(kv_rows="shared"))
    doc_case("mor_route_in_head", M, 20, "7.4", lambda d: d["head"]["nodes"].insert(0, copy.deepcopy(node(d, "depth_route"))))
    S = "pk_state"
    doc_case("turn_state_with_window", S, 15, "6.8.2", lambda d: (node(d, "attention").update(window=4),
                                                                    kv(d).update(window=4)))
    doc_case("turn_state_without_state_weight", S, 15, "6.8.2", lambda d: d["embed"]["nodes"][0].pop("state_weight"))
    doc_case("evict_without_mask", S, 16, "6.8.3", lambda d: node(d, "attention").pop("mask"))
    doc_case("evict_unknown", S, 5, "10.1 rule 1", lambda d: kv(d).update(evict="oldest"))
    doc_case("mask_unknown", S, 5, "10.1 rule 1", lambda d: node(d, "attention").update(mask="sliding"))
    doc_case("state_weight_with_i", S, 17, "6.8.1", lambda d: d["embed"]["nodes"][0].update(state_weight="state_{i}.weight"))
    doc_case("two_state_weights", S, 17, "6.8.1", lambda d: d["embed"]["nodes"].append(dict(copy.deepcopy(
        d["embed"]["nodes"][0]), out="x2")))
    doc_case("state_weight_wrong_name", S, 28, "6.8.1", lambda d: d["embed"]["nodes"][0].update(state_weight="nope.weight"))
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
    # version_key_mismatch: rewrite the version key by hand (write_gguf derives it from the document).
    _patch_version_key(SUITE / "refuse" / "version_key_mismatch.gguf", 7)
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
        manifest.append({"id": f"op/{op}", "kind": "op", "file": f, "profile": "recurrence" if op == "recurrence" else "core",
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
    (SUITE / "recurrence").mkdir(exist_ok=True)
    for name, doc, case in recurrence_traces():
        (SUITE / "recurrence" / f"{name}.ir.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        f = f"state/{name}_recurrence.json"
        (SUITE / f).write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
        manifest.append({"id": f"trace/{name}_recurrence", "kind": "trace", "file": f, "model": f"models/{name}",
                         "profile": "recurrence"})
    (SUITE / "manifest.json").write_text(json.dumps({
        "_comment": "spec/IR_SPEC.md section 11: the conformance suite's cases. Generated by tools/ir_conformance.py "
                    "--regenerate from the models in models/ (expected values: tools/ir_numpy.py).",
        "ir_version": 1, "cases": manifest}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} cases")
