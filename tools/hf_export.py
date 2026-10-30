#!/usr/bin/env python3
"""paperkiln -> Hugging Face export (docs/ECOSYSTEM.md feature 4).

    python tools/hf_export.py OUT_DIR [--hf-dir OUT_DIR/hf]

Takes a finished plain llama, exact two-block GPT2, or plain GPT2/Llama-flavour flex run (events.jsonl +
<name>.safetensors + <name>.gguf) and writes an HF-layout folder that
`transformers.AutoModelForCausalLM.from_pretrained` opens directly:

    config.json            LlamaConfig/GPT2Config from the recorded model event
    model.safetensors      weights renamed to model.* and converted from
                           microtorch's [in, out] Linear layout to HF's
                           [out, in]; norm gammas squeezed [1,d] -> [d].
                           GPT2 Conv1D weights retain [in,out]; its untied
                           output head alone is transposed. Llama-flavour flex
                           splits fused QKV, permutes q/k weights AND biases,
                           preserves biased attention and an untied head.
    tokenizer.json (+cfg)  exact llama-BPE reconstruction from GGUF, or
                           native paperkiln byte-BPE, preserving token IDs and
                           byte decoding; legacy word-level vocabularies use
                           the C-locale byte scanner and original spacing.

The receipt lives in tools/hf_export_verify.py: transformers loads the
folder and checks greedy token IDs plus raw logits against teacher-forced
`mtstudio score` on the same encoded prompt and continuation — the same parity standard
ember.cpp serving is held to.

RoPE: HF Llama rotates every head. A run recorded with rope_heads "first"
(or with no rope_heads, i.e. trained before the field existed) rotates
head 0 only, so with more than one head HF cannot reproduce it; such runs
are refused unless --allow-legacy-rope is given.

Exports stage all five managed files before publication. Existing managed files require
--overwrite; unrelated files are preserved, incompatible loader/shard leftovers are refused.
Caught publication errors restore previous managed bytes. Existing-directory replacement is
rollback-protected, not crash-atomic; retained recovery files are reported if rollback fails.

Requires: numpy, safetensors, tokenizers;
transformers+torch only for the verify step.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
from safetensors.numpy import load_file, save_file


def model_event(out_dir, events=None):
    with open(events or os.path.join(out_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.lstrip().startswith("{"):
                continue
            e = json.loads(line)
            if e.get("event") == "model":
                return e
    raise SystemExit("no model event in events.jsonl")


def start_name(out_dir, events=None):
    with open(events or os.path.join(out_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.lstrip().startswith("{"):
                continue
            e = json.loads(line)
            if e.get("event") == "start":
                return e["name"]
    raise SystemExit("no start event")


def gguf_tokens(path):
    """Minimal GGUF v3 metadata reader for tokenizer.ggml.tokens — no
    dependency, and we are parsing our OWN exporter's output
    (gguf.hpp), so the key set and types are known."""
    import struct

    def u32(f): return struct.unpack("<I", f.read(4))[0]
    def u64(f): return struct.unpack("<Q", f.read(8))[0]
    def s(f): return f.read(u64(f)).decode("utf-8", errors="replace")
    SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8,
             12: 8}

    def skip_value(f, t):
        if t in SIZES:
            f.read(SIZES[t])
        elif t == 8:
            s(f)
        elif t == 9:
            et, n = u32(f), u64(f)
            for _ in range(n):
                skip_value(f, et)
        else:
            raise SystemExit(f"gguf: unknown value type {t}")

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        u32(f)                       # version
        u64(f)                       # tensor count
        n_kv = u64(f)
        for _ in range(n_kv):
            key = s(f)
            t = u32(f)
            if key == "tokenizer.ggml.tokens" and t == 9:
                et, n = u32(f), u64(f)
                if et != 8:
                    raise SystemExit("tokens array is not strings")
                return [s(f) for _ in range(n)]
            skip_value(f, t)
    raise SystemExit(f"no tokenizer.ggml.tokens in {path}")


def rope_perm(dk):
    """paperkiln's apply_rope rotates ADJACENT pairs (x[2j], x[2j+1]) —
    the llama.cpp/GGUF interleaved convention. HF Llama rotates HALVES
    (x[j], x[j+dk/2]). Same mathematics under a permutation of each
    head's dimensions, so q/k output rows are permuted at export: HF
    position p takes our dim 2p (first half) or 2(p-dk/2)+1 (second)."""
    half = dk // 2
    return [2 * p if p < half else 2 * (p - half) + 1 for p in range(dk)]


def convert_weights(sd, n_heads):
    out = {}
    for name, w in sd.items():
        hf = name if name == "lm_head.weight" else "model." + name
        if "proj.weight" in name:
            w = np.ascontiguousarray(w.T)      # [in,out] -> [out,in]
        if "q_proj.weight" in name or "k_proj.weight" in name:
            dk = w.shape[0] // n_heads
            perm = rope_perm(dk)
            w = w.reshape(n_heads, dk, -1)[:, perm, :].reshape(w.shape)
            w = np.ascontiguousarray(w)
        if "layernorm.weight" in name or name == "norm.weight":
            w = w.reshape(-1)                  # [1,d] -> [d]
        out[hf] = w.astype(np.float32)
    return out


def validate_gpt2(m):
    """ParityLM or the exact GPT2-flavour FlexLM subset, both untied."""
    flex = m.get("family") == "flex"
    placement = m.get("norm_placement", "pre")
    if flex:
        required = {"attention", "norm", "activation", "position", "residual", "final_norm", "norm_placement", "d_ff"}
        if not required <= m.keys():
            raise ValueError("plain flex HF export requires the complete resolved flavour record")
        # Flex writes each placement separately; refuse missing/extra slots.
        placement_ok = placement == {"attn": "pre", "ffn": "pre"}
    else:
        placement_ok = placement == "pre"
    if (m.get("attention", "exact") != "exact" or m.get("norm", "layernorm") != "layernorm"
            or m.get("activation", "gelu") != "gelu" or m.get("position", "learned") != "learned"
            or m.get("residual", "residual") != "residual" or not m.get("final_norm", True)
            or not placement_ok):
        raise ValueError("HF GPT2 export requires exact attention, pre LayerNorm, tanh GELU, learned positions and residuals")
    if any(m.get(k) for k in ("thirdparty", "qk_norm", "recursion", "attn_res", "moe", "state_tokens", "tie_head", "tie_embeddings",
                              "qkv_conv", "embed", "embed_expr", "embed_inputs", "embed_before_positions", "embed_before_pos",
                              "logits", "logits_expr", "activation_inputs", "norm_inputs", "t5_relative", "shaw_relative", "ssm")):
        raise ValueError("modified or tied-head GPT2 exports are unsupported")
    if m.get("gelu_approx", "tanh") != "tanh" or m.get("layer_norm_epsilon", 1e-5) != 1e-5:
        raise ValueError("GPT2 source GELU/LayerNorm constants differ from the supported mapping")
    if (any(type(m.get(k)) is not int or m[k] < 1 for k in ("d", "heads", "layers", "vocab", "T"))
            or m["d"] % m["heads"] or (not flex and m["layers"] != 2)
            or type(m.get("d_ff", 4*m["d"])) is not int or m.get("d_ff", 4*m["d"]) < 1
            or (not flex and m.get("d_ff", 4*m["d"]) != 4*m["d"])):
        raise ValueError("unsupported GPT2 geometry (ParityLM has two blocks and 4d MLPs; plain flex permits arbitrary positive depth/MLP width)")


def convert_gpt2_weights(sd, m):
    """Conv1D weights already use [in,out]; transpose only the untied head."""
    validate_gpt2(m)
    d, vocab, ctx = m["d"], m["vocab"], m["T"]
    ff = m.get("d_ff", 4*d)
    flex = m.get("family") == "flex"
    layout = {"wte.weight": ("transformer.wte.weight", (vocab, d)),
              "wpe.weight": ("transformer.wpe.weight", (ctx, d)),
              "ln_f.weight": ("transformer.ln_f.weight", (1, d)),
              "ln_f.bias": ("transformer.ln_f.bias", (1, d)),
              "head.weight": ("lm_head.weight", (d, vocab))}
    for b in range(m["layers"]):
        for native, hf, width_in, width_out in (("attn", "attn.c_attn", d, 3*d),
                                                ("attn", "attn.c_proj", d, d),
                                                ("mlp", "mlp.c_fc", d, ff),
                                                ("mlp", "mlp.c_proj", ff, d)):
            suffix = hf.split(".", 1)[1]
            prefix = f"layers.{b}.{native}" if flex else f"{native}_{b}"
            layout[f"{prefix}.{suffix}.weight"] = (f"transformer.h.{b}.{hf}.weight", (width_in, width_out))
            layout[f"{prefix}.{suffix}.bias"] = (f"transformer.h.{b}.{hf}.bias", (1, width_out))
        for native, hf in (("ln1", "ln_1"), ("ln2", "ln_2")):
            for part in ("weight", "bias"):
                key = f"layers.{b}.{hf}.{part}" if flex else f"{native}_{b}.{part}"
                layout[key] = (f"transformer.h.{b}.{hf}.{part}", (1, d))
    if set(sd) != set(layout):
        raise ValueError("GPT2 source tensor inventory differs from the complete supported mapping")
    out = {}
    for key, (name, shape) in layout.items():
        w = sd[key]
        if w.shape != shape or not np.isfinite(w).all():
            raise ValueError("invalid GPT2 source tensor: " + key)
        if key == "head.weight":
            w = w.T
        elif name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")) or key.endswith(".bias"):
            w = w.reshape(-1)
        out[name] = np.ascontiguousarray(w, dtype=np.float32)
    return out


def hf_model_type(m):
    if m.get("family") in ("llama", "gpt2"):
        return m["family"]
    if m.get("family") == "flex":
        flavour = (m.get("norm"), m.get("activation"), m.get("position"))
        if flavour == ("layernorm", "gelu", "learned"):
            return "gpt2"
        if flavour == ("rmsnorm", "swiglu", "rope"):
            return "llama"
    raise ValueError("HF export supports only plain llama/GPT2 and their exact flex flavours")


def validate_flex_llama(m):
    """Committed FlexLM has biased MHA, all-head RoPE and an untied head."""
    required = {"attention", "norm", "activation", "position", "residual", "final_norm", "norm_placement", "d_ff"}
    if m.get("family") != "flex" or not required <= m.keys():
        raise ValueError("Llama-flavour flex requires the complete resolved flavour record")
    if ((m["norm"], m["activation"], m["position"]) != ("rmsnorm", "swiglu", "rope")
            or m["attention"] != "exact" or m["residual"] != "residual" or m["final_norm"] is not True
            or m["norm_placement"] != {"attn": "pre", "ffn": "pre"}):
        raise ValueError("HF Llama-flavour flex requires exact attention, pre RMSNorm, SwiGLU, RoPE and residuals")
    if any(m.get(k) for k in ("thirdparty", "qk_norm", "recursion", "attn_res", "moe", "state_tokens", "tie_embeddings", "tie_head",
                              "qkv_conv", "embed", "embed_expr", "embed_inputs", "embed_before_positions", "embed_before_pos",
                              "logits", "logits_expr", "activation_inputs", "norm_inputs", "t5_relative", "shaw_relative", "ssm")):
        raise ValueError("modified Llama-flavour flex exports are unsupported")
    if (any(type(m.get(k)) is not int or m[k] < 1 for k in ("d", "heads", "layers", "vocab", "T", "d_ff"))
            or m["d"] % m["heads"] or (m["d"] // m["heads"]) % 2):
        raise ValueError("unsupported Llama-flavour flex geometry")
    if (m.get("kv_heads", m["heads"]) != m["heads"] or m.get("rope_pairing", "interleaved") != "interleaved"
            or m.get("rope_dims", m["d"] // m["heads"]) != m["d"] // m["heads"]
            or m.get("rope_heads", "all") != "all" or m.get("rms_norm_eps", 1e-5) != 1e-5
            or type(m.get("tie_head", False)) is not bool):
        raise ValueError("alternate RoPE, RMSNorm constants or head semantics are unsupported")
    theta = m.get("rope_theta", 10000.0)
    if type(theta) not in (int, float) or not np.isfinite(theta) or theta <= 0:
        raise ValueError("invalid recorded rope_theta")
    return theta


def convert_flex_llama_weights(sd, m):
    validate_flex_llama(m)
    d, ff, vocab, heads = m["d"], m["d_ff"], m["vocab"], m["heads"]
    layout = {"wte.weight": (vocab, d), "norm.weight": (1, d), "head.weight": (d, vocab)}
    for b in range(m["layers"]):
        base = f"layers.{b}."
        layout.update({base+"norm1.weight": (1, d), base+"norm2.weight": (1, d),
                       base+"attn.c_attn.weight": (d, 3*d), base+"attn.c_attn.bias": (1, 3*d),
                       base+"attn.c_proj.weight": (d, d), base+"attn.c_proj.bias": (1, d),
                       base+"mlp.gate_proj.weight": (d, ff), base+"mlp.up_proj.weight": (d, ff),
                       base+"mlp.down_proj.weight": (ff, d)})
    if set(sd) != set(layout):
        raise ValueError("Llama-flavour flex tensor inventory differs from complete mapping")
    for key, shape in layout.items():
        if sd[key].shape != shape or not np.isfinite(sd[key]).all():
            raise ValueError("invalid Llama-flavour flex source tensor: " + key)
    out = {"model.embed_tokens.weight": sd["wte.weight"], "model.norm.weight": sd["norm.weight"].reshape(-1),
           "lm_head.weight": sd["head.weight"].T}
    dk = d // heads
    perm = rope_perm(dk)
    for b in range(m["layers"]):
        base, hf = f"layers.{b}.", f"model.layers.{b}."
        for norm, name in (("norm1", "input_layernorm"), ("norm2", "post_attention_layernorm")):
            out[hf+name+".weight"] = sd[base+norm+".weight"].reshape(-1)
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            weight = sd[base+"attn.c_attn.weight"][:, i*d:(i+1)*d].T
            bias = sd[base+"attn.c_attn.bias"].reshape(3, d)[i]
            if i < 2:
                weight = weight.reshape(heads, dk, d)[:, perm, :].reshape(d, d)
                bias = bias.reshape(heads, dk)[:, perm].reshape(d)
            out[hf+"self_attn."+name+".weight"] = weight
            out[hf+"self_attn."+name+".bias"] = bias
        out[hf+"self_attn.o_proj.weight"] = sd[base+"attn.c_proj.weight"].T
        out[hf+"self_attn.o_proj.bias"] = sd[base+"attn.c_proj.bias"].reshape(-1)
        for name in ("gate_proj", "up_proj", "down_proj"):
            out[hf+"mlp."+name+".weight"] = sd[base+"mlp."+name+".weight"].T
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


def map_weights(sd, m):
    kind = hf_model_type(m)
    if m["family"] == "flex" and kind == "llama":
        return convert_flex_llama_weights(sd, m)
    return convert_gpt2_weights(sd, m) if kind == "gpt2" else convert_weights(sd, m["heads"])


def native_bpe_tokenizer(gguf_path):
    """Serializable HF equivalent of bpe.py's byte classes and ranked merges.

    Every non-ASCII Unicode character consists entirely of native letter bytes.
    The whitespace branch stops before exactly a space followed by a letter.
    Added tokens are only the three native control strings; literal <unk> is text.
    """
    import bpe
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    native = bpe.read_gguf(gguf_path)
    base = len(bpe.SPECIALS) + 256
    if (native.tokens[:4] != [t.encode() for t in bpe.SPECIALS]
            or native.tokens[4:base] != [bytes([b]) for b in range(256)]
            or len(native.tokens) != base + len(native.merges)):
        raise ValueError("invalid native byte-BPE vocabulary layout")
    for rank, (a, b) in enumerate(native.merges):
        if min(a, b) < 0 or max(a, b) >= base + rank or native.tokens[base + rank] != native.tokens[a] + native.tokens[b]:
            raise ValueError("invalid native byte-BPE ranked merge")
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    next_code = 256
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(next_code)
            next_code += 1
    byte_map = dict(zip(bs, map(chr, cs)))
    tokens = ["".join(byte_map[b] for b in token) for token in native.tokens]
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate native byte-BPE token strings are unsupported")
    vocab = {token: i for i, token in enumerate(tokens)}
    merges = [(tokens[a], tokens[b]) for a, b in native.merges]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=False))
    letter = r"[A-Za-z'\x80-\x{10FFFF}]"
    space = r"[ \t\n\r\x0b\x0c]"
    pattern = (" ?" + letter + "+|[0-9]|(?:(?! " + letter + ")" + space + ")+|"
               r"[^A-Za-z'0-9 \t\n\r\x0b\x0c\x80-\x{10FFFF}]+")
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(sp, special=True, normalized=False) for sp in bpe.SPECIALS[1:]])
    return tok


def legacy_word_tokenizer(tokens):
    """Serializable C-locale wordtok semantics for lossless UTF-8 vocabularies.

    UTF-8 input is still consumed byte by byte: non-ASCII bytes are separate
    punctuation/unknown tokens, never Unicode letters or Unicode whitespace.
    Refuse vocabulary shapes the serialized spacing decoder cannot preserve.
    This matches the source process's initial C locale; custom C locales and
    invalid UTF-8 byte-string APIs are outside the HF text-tokenizer contract.
    """
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    if (not tokens or len(set(tokens)) != len(tokens) or
            any(not t or any(c in " \t\n\r\v\f" for c in t)
                or (len(t) > 1 and t[0] in ".,!?;:)") for t in tokens)):
        raise ValueError("legacy word-level HF export requires unique nonempty UTF-8 tokens without ASCII whitespace "
                         "or multi-character tokens beginning with closing punctuation")
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    next_code = 256
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(next_code)
            next_code += 1
    byte_map = dict(zip(bs, map(chr, cs)))
    mapped = ["".join(byte_map[b] for b in t.encode("utf-8")) for t in tokens]
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(mapped)}, unk_token=mapped[0]))
    tok.normalizer = normalizers.Sequence([normalizers.Replace(chr(c), chr(c + 32)) for c in range(65, 91)])
    whitespace = "".join(byte_map[b] for b in b" \t\n\r\v\f")
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        pre_tokenizers.Split(Regex("[" + whitespace + "]+"), behavior="removed"),
        pre_tokenizers.Split(Regex("[a-z0-9']+|[^a-z0-9']"), behavior="isolated")])
    # Prefix each token before ByteLevel fuses the sequence, then reproduce
    # wordtok::detokenize's exact seven attaching punctuation characters.
    tok.decoder = decoders.Sequence([
        decoders.Replace(Regex("^"), byte_map[32]), decoders.ByteLevel(),
        *[decoders.Replace(" " + c, c) for c in ".,!?;:)"],
        decoders.Replace("( ", "("), decoders.Strip(" ", 1, 0)])
    if "<|endoftext|>" in tokens:
        tok.add_special_tokens([AddedToken("<|endoftext|>", special=True, normalized=False)])
    return tok


def legacy_word_vocab(path):
    """Read losslessly: GGUF invalid UTF-8 byte strings cannot be HF text tokens."""
    from llama3_bpe import read_gguf_kv
    tokens = read_gguf_kv(str(path), {"tokenizer.ggml.tokens"}).get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
        raise ValueError("legacy word-level HF export requires valid UTF-8 GGUF token strings")
    return tokens


def legacy_word_ids(tokens, text):
    """Independent byte-scanning mirror of C-locale wordtok::tokenize."""
    vocab = {t.encode("utf-8"): i for i, t in enumerate(tokens)}
    eos = b"<|endoftext|>"
    raw, ids, word = text.encode("utf-8"), [], bytearray()
    def flush():
        if word:
            ids.append(vocab.get(bytes(word), 0))
            word.clear()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == 60 and eos in vocab and raw.startswith(eos, i):
            flush()
            ids.append(vocab[eos])
            i += len(eos)
            continue
        if 65 <= c <= 90 or 97 <= c <= 122 or 48 <= c <= 57 or c == 39:
            word.append(c + 32 if 65 <= c <= 90 else c)
        else:
            flush()
            if c not in b" \t\n\r\v\f":
                ids.append(vocab.get(bytes([c]), 0))
        i += 1
    flush()
    return ids


def write_tokenizer(hf_dir, tokens, gguf_path=None):
    if gguf_path:
        from llama3_bpe import from_gguf, read_gguf_kv
        kind = read_gguf_kv(gguf_path, {"tokenizer.ggml.model"}).get("tokenizer.ggml.model")
        if kind == "gpt2":
            tok = from_gguf(gguf_path)
            tok.save(os.path.join(hf_dir, "tokenizer.json"))
            vocab = tok.get_vocab()
            with open(os.path.join(hf_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump({"tokenizer_class": "PreTrainedTokenizerFast",
                           "model_max_length": 1_000_000}, f, indent=1)
            return vocab, None
        if kind == "bpe":
            tok = native_bpe_tokenizer(gguf_path)
            tok.save(os.path.join(hf_dir, "tokenizer.json"))
            with open(os.path.join(hf_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump({"tokenizer_class": "PreTrainedTokenizerFast",
                           "model_max_length": 1_000_000}, f, indent=1)
            return tok.get_vocab(), None
        if kind not in (None, "word", "wordlevel"):
            raise ValueError(f"unsupported tokenizer model: {kind!r}")
    if gguf_path:
        tokens = legacy_word_vocab(gguf_path)
    tok = legacy_word_tokenizer(tokens)
    tok.save(os.path.join(hf_dir, "tokenizer.json"))
    with open(os.path.join(hf_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1_000_000,
                   "clean_up_tokenization_spaces": False}, f, indent=1)
    return tok.get_vocab(), tokens[0]



def validate_model(m):
    """Only the plain tied-head Llama mapping is supported by this exporter."""
    if (m.get("attention", "exact") != "exact" or m.get("norm", "rmsnorm") != "rmsnorm"
            or m.get("activation", "swiglu") != "swiglu" or m.get("position", "rope") != "rope"
            or m.get("residual", "residual") != "residual" or not m.get("final_norm", True)
            or m.get("norm_placement", "pre") != "pre"):
        raise ValueError("HF Llama export requires exact attention, pre RMSNorm, SwiGLU, RoPE and residuals")
    if (m.get("thirdparty") or m.get("kv_heads", m["heads"]) != m["heads"]
            or m.get("qk_norm") or m.get("rope_pairing", "interleaved") != "interleaved"
            or m.get("rope_dims", m["d"] // m["heads"]) != m["d"] // m["heads"]):
        raise ValueError("thirdparty/GQA/partial or alternate RoPE/qk-normalized Llama export is unsupported")
    if any(m.get(k) for k in ("recursion", "attn_res", "moe", "state_tokens")):
        raise ValueError("recursion, attention residuals, MoE and state-token exports are unsupported")
    if not m.get("tie_embeddings", m.get("tie_head", True)):
        raise ValueError("untied output embeddings are unsupported")
    theta = m.get("rope_theta", 10000.0)
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("invalid recorded rope_theta")
    return theta


EXPORT_FILES = frozenset({"config.json", "model.safetensors", "tokenizer.json",
                          "tokenizer_config.json", "export_receipt.json"})


def export_destination(target, overwrite):
    """Reject redirecting links and loader-affecting leftovers; preserve unrelated files."""
    target = Path(target).absolute()
    for component in (target, *target.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError("HF export target must not contain symlinks or junctions")
    if target.exists():
        if not target.is_dir():
            raise ValueError("HF export target is not a directory")
        for child in target.iterdir():
            if child.name in EXPORT_FILES:
                if not overwrite:
                    raise ValueError("HF export already exists; use --overwrite to replace its managed files")
                if child.is_symlink() or not child.is_file():
                    raise ValueError("HF managed output must be a regular file")
            elif (child.suffix in (".safetensors", ".bin") or child.name.endswith(".index.json")
                  or child.name in {"generation_config.json", "special_tokens_map.json", "added_tokens.json",
                                    "vocab.json", "merges.txt"}):
                raise ValueError(f"incompatible stale HF loader artifact: {child.name}")
    return target


@contextmanager
def staged_export(target, overwrite=False):
    """Publish a complete export; caught failures roll back existing managed bytes.

    Existing unrelated files stay in place. Directory publication for a new target is atomic;
    replacing an existing directory's files is rollback-protected, not crash-atomic. A failed
    rollback retains the staging/backup directory and reports its path for manual recovery.
    """
    target = export_destination(target, overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ("." + target.name + ".export.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        stage = Path(tempfile.mkdtemp(prefix="." + target.name + ".export-", dir=target.parent))
        build = stage / "build"
        backup = stage / "backup"
        build.mkdir()
        backup.mkdir()
    except BaseException:
        lock.unlink()
        raise
    preserve = False
    try:
        yield str(build)
        if {p.name for p in build.iterdir()} != EXPORT_FILES or any(
                not p.is_file() or p.is_symlink() for p in build.iterdir()):
            raise ValueError("staged HF export is incomplete or has unexpected files")
        export_destination(target, overwrite)  # recheck before any destination mutation
        if not target.exists():
            os.replace(build, target)
        else:
            moved, published = [], []
            try:
                for name in sorted(EXPORT_FILES):
                    if (target / name).exists():
                        os.replace(target / name, backup / name)
                        moved.append(name)
                for name in sorted(EXPORT_FILES):
                    os.replace(build / name, target / name)
                    published.append(name)
            except BaseException:
                try:
                    for name in reversed(published):
                        os.replace(target / name, build / name)
                    for name in reversed(moved):
                        os.replace(backup / name, target / name)
                except BaseException as recovery:
                    preserve = True
                    raise RuntimeError(f"HF publication rollback failed; preserve recovery files at {stage}") from recovery
                raise
    finally:
        if not preserve:
            shutil.rmtree(stage)
            lock.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--hf-dir")
    ap.add_argument("--events", help="recorded JSONL events or mixed training log")
    ap.add_argument("--overwrite", action="store_true", help="replace existing managed HF files after successful staging")
    ap.add_argument("--allow-legacy-rope", action="store_true",
                    help="export a rope_heads=first run anyway (HF rotates every head, so outputs differ)")
    args = ap.parse_args()
    target = args.hf_dir or os.path.join(args.out_dir, "hf")
    with staged_export(target, args.overwrite) as stage:
        args.hf_dir = stage
        count, vocab, unk = export_model(args)
    print(f"HF export -> {target} ({count} tensors, vocab {vocab}, unk={unk!r})")
    print("verify: python tools/hf_export_verify.py", target, args.out_dir)


def export_model(args):
    m = model_event(args.out_dir, args.events)
    kind = hf_model_type(m)
    rope = m.get("rope_heads", "first")
    if m["family"] == "llama" and rope != "all" and m.get("heads", 1) > 1 and not args.allow_legacy_rope:
        raise SystemExit(
            f"this run has rope_heads={rope} (RoPE on head 0 of {m['heads']}); "
            "HF Llama rotates every head and cannot reproduce it. Retrain with "
            "arch.rope_heads=all, or pass --allow-legacy-rope to export anyway")
    theta = (validate_flex_llama(m) if m["family"] == "flex" else validate_model(m)) if kind == "llama" else None
    if kind == "gpt2":
        validate_gpt2(m)
    name = start_name(args.out_dir, args.events)
    hf_dir = args.hf_dir or os.path.join(args.out_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)

    gguf_path = os.path.join(args.out_dir, f"{name}.gguf")
    tokens = gguf_tokens(gguf_path)
    vocab, unk = write_tokenizer(hf_dir, tokens, gguf_path)
    if len(vocab) != m["vocab"]:
        raise ValueError("tokenizer vocabulary does not match model vocabulary")
    sd = load_file(os.path.join(args.out_dir, f"{name}.safetensors"))
    if m["family"] == "llama" and "lm_head.weight" in sd:
        raise ValueError("untied output-head tensor is unsupported")
    save_file(map_weights(sd, m),
              os.path.join(hf_dir, "model.safetensors"))


    from llama3_bpe import read_gguf_kv
    special_ids = read_gguf_kv(gguf_path, {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"})
    if kind == "llama":
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": m["d"],
            "intermediate_size": 3 * m["d"] if "d_ff" not in m else m["d_ff"],
            "num_hidden_layers": m["layers"],
            "num_attention_heads": m["heads"],
            "num_key_value_heads": m["heads"],
            "vocab_size": m["vocab"],
            "max_position_embeddings": m["T"],
            "rms_norm_eps": 1e-5,  # ops::kRmsNormEps, including the training tape
            "rope_theta": theta,
            "hidden_act": "silu",
            "tie_word_embeddings": m["family"] != "flex",
            "attention_bias": m["family"] == "flex",
            "mlp_bias": False,
            "torch_dtype": "float32",
            "bos_token_id": special_ids.get("tokenizer.ggml.bos_token_id", vocab.get("<s>")),
            "eos_token_id": special_ids.get("tokenizer.ggml.eos_token_id", vocab.get("</s>")),
        }
    else:
        cfg = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
               "n_embd": m["d"], "n_inner": m.get("d_ff", 4*m["d"]), "n_layer": m["layers"],
               "n_head": m["heads"], "vocab_size": m["vocab"], "n_positions": m["T"], "n_ctx": m["T"],
               "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
               "tie_word_embeddings": False, "resid_pdrop": 0., "embd_pdrop": 0., "attn_pdrop": 0.,
               "scale_attn_weights": True, "scale_attn_by_inverse_layer_idx": False,
               "reorder_and_upcast_attn": False, "add_cross_attention": False, "torch_dtype": "float32",
               "bos_token_id": special_ids.get("tokenizer.ggml.bos_token_id", vocab.get("<s>")),
               "eos_token_id": special_ids.get("tokenizer.ggml.eos_token_id", vocab.get("<|endoftext|>"))}
    with open(os.path.join(hf_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(hf_dir, "export_receipt.json"), "w", encoding="utf-8") as f:
        json.dump({"events": os.path.abspath(args.events) if args.events else None}, f)
    return len(sd), len(tokens), unk



if __name__ == "__main__":
    main()
