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
    tokenizer.json (+cfg)  the run's word-level vocabulary (read from the
                           exported GGUF, which embeds it) as a
                           tokenizers WordLevel model

The receipt lives in tools/hf_export_verify.py: transformers loads the
folder and greedy-generates ARGMAX-IDENTICAL continuations to
`mtstudio sample --topk 1` on the same prompt — the same parity standard
ember.cpp serving is held to.

Requires: safetensors, tokenizers; gguf (pip) for the vocab;
transformers+torch only for the verify step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file, save_file


def model_event(out_dir):
    with open(os.path.join(out_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("event") == "model":
                return e
    raise SystemExit("no model event in events.jsonl")


def start_name(out_dir):
    with open(os.path.join(out_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
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


def write_tokenizer(hf_dir, tokens):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    vocab = {t: i for i, t in enumerate(tokens)}
    unk = "<unk>" if "<unk>" in vocab else tokens[0]
    tok = Tokenizer(WordLevel(vocab, unk_token=unk))
    tok.pre_tokenizer = Whitespace()
    tok.save(os.path.join(hf_dir, "tokenizer.json"))
    with open(os.path.join(hf_dir, "tokenizer_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast",
                   "unk_token": unk, "model_max_length": 1_000_000}, f,
                  indent=1)
    return vocab, unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--hf-dir")
    args = ap.parse_args()
    m = model_event(args.out_dir)
    if m.get("family") != "llama":
        raise SystemExit("HF export is llama-family only (LlamaForCausalLM "
                         f"mapping); this run is family={m.get('family')}")
    name = start_name(args.out_dir)
    hf_dir = args.hf_dir or os.path.join(args.out_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)

    sd = load_file(os.path.join(args.out_dir, f"{name}.safetensors"))
    save_file(convert_weights(sd, m["heads"]),
              os.path.join(hf_dir, "model.safetensors"))

    tokens = gguf_tokens(os.path.join(args.out_dir, f"{name}.gguf"))
    vocab, unk = write_tokenizer(hf_dir, tokens)

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
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
        "bos_token_id": vocab.get("<s>", 0),
        "eos_token_id": vocab.get("</s>", 0),
    }
    with open(os.path.join(hf_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print(f"HF export -> {hf_dir}  ({len(sd)} tensors, vocab {len(tokens)}, "
          f"unk={unk!r})")
    print("verify: python tools/hf_export_verify.py", hf_dir, args.out_dir)


if __name__ == "__main__":
    main()
