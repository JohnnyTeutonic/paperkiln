#!/usr/bin/env python3
"""paperkiln -> Hugging Face export (docs/ECOSYSTEM.md feature 4).

    python tools/hf_export.py OUT_DIR [--hf-dir OUT_DIR/hf]

Takes a finished llama-family mtstudio run directory (events.jsonl +
<name>.safetensors + <name>.gguf) and writes an HF-layout folder that
`transformers.AutoModelForCausalLM.from_pretrained` opens directly:

    config.json            LlamaConfig from the run's own model event
    model.safetensors      weights renamed to model.* and converted from
                           microtorch's [in, out] Linear layout to HF's
                           [out, in]; norm gammas squeezed [1,d] -> [d]
    tokenizer.json (+cfg)  exact llama-BPE reconstruction from GGUF, or
                           native paperkiln byte-BPE, preserving token IDs and
                           byte decoding. Legacy word-level exports are refused.

The receipt lives in tools/hf_export_verify.py: transformers loads the
folder and greedy-generates ARGMAX-IDENTICAL continuations to
`mtstudio sample --topk 1` on the same prompt â€” the same parity standard
ember.cpp serving is held to.

RoPE: HF Llama rotates every head. A run recorded with rope_heads "first"
(or with no rope_heads, i.e. trained before the field existed) rotates
head 0 only, so with more than one head HF cannot reproduce it; such runs
are refused unless --allow-legacy-rope is given.

Requires: safetensors, tokenizers; gguf (pip) for the vocab;
transformers+torch only for the verify step.
"""
from __future__ import annotations

import argparse
import json
import os

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
    """Minimal GGUF v3 metadata reader for tokenizer.ggml.tokens â€” no
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
    """paperkiln's apply_rope rotates ADJACENT pairs (x[2j], x[2j+1]) â€”
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
    raise ValueError("legacy word-level HF export is unsupported: paperkiln lowercases ASCII words, "
                     "keeps apostrophes and splits punctuation by byte; HF Whitespace is not equivalent")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--hf-dir")
    ap.add_argument("--events", help="recorded JSONL events or mixed training log")
    ap.add_argument("--allow-legacy-rope", action="store_true",
                    help="export a rope_heads=first run anyway (HF will rotate "
                         "every head, so its outputs will differ)")
    args = ap.parse_args()
    m = model_event(args.out_dir, args.events)
    if m.get("family") != "llama":
        raise SystemExit("HF export is llama-family only (LlamaForCausalLM "
                         f"mapping); this run is family={m.get('family')}")
    rope = m.get("rope_heads", "first")
    if rope != "all" and m.get("heads", 1) > 1 and not args.allow_legacy_rope:
        raise SystemExit(
            f"this run has rope_heads={rope} (RoPE on head 0 of {m['heads']}); "
            "HF Llama rotates every head and cannot reproduce it. Retrain with "
            "arch.rope_heads=all, or pass --allow-legacy-rope to export anyway")
    theta = validate_model(m)
    name = start_name(args.out_dir, args.events)
    hf_dir = args.hf_dir or os.path.join(args.out_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)

    gguf_path = os.path.join(args.out_dir, f"{name}.gguf")
    tokens = gguf_tokens(gguf_path)
    vocab, unk = write_tokenizer(hf_dir, tokens, gguf_path)
    if len(vocab) != m["vocab"]:
        raise ValueError("tokenizer vocabulary does not match model vocabulary")
    sd = load_file(os.path.join(args.out_dir, f"{name}.safetensors"))
    if "lm_head.weight" in sd:
        raise ValueError("untied output-head tensor is unsupported")
    save_file(convert_weights(sd, m["heads"]),
              os.path.join(hf_dir, "model.safetensors"))


    from llama3_bpe import read_gguf_kv
    special_ids = read_gguf_kv(gguf_path, {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"})
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
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
        "bos_token_id": special_ids.get("tokenizer.ggml.bos_token_id", vocab.get("<s>")),
        "eos_token_id": special_ids.get("tokenizer.ggml.eos_token_id", vocab.get("</s>")),
    }
    with open(os.path.join(hf_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(hf_dir, "export_receipt.json"), "w", encoding="utf-8") as f:
        json.dump({"events": os.path.abspath(args.events) if args.events else None}, f)
    print(f"HF export -> {hf_dir}  ({len(sd)} tensors, vocab {len(tokens)}, "
          f"unk={unk!r})")
    print("verify: python tools/hf_export_verify.py", hf_dir, args.out_dir)


if __name__ == "__main__":
    main()
