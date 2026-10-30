#!/usr/bin/env python3
"""Sizes for R3's decision (experiments/lowbit_2026-10-28).

    python3 size.py gguf FILE.gguf ...                 bytes and bits per weight by tensor class, per file
    python3 size.py ternary MODEL.safetensors          the BitLinear model's real size, packed

gguf: every tensor's stored bytes from the GGUF's own type table (gguf-py), grouped as the block
projections (attn_q/k/v/output, ffn_gate/up/down), token_embd, output and the norms, plus the file size.

ternary: the latent weights of a BitLinear model (arch.custom.bitlinear) give each block projection's codes
q = clamp(round(W s), -1, 1) and scale s = 1 / max(mean|W|, 1e-5) (ops::ternary_scale's formula, mean in
double), so the model is exactly those codes and one float per tensor. Packings: 5 trits per byte (243 <=
256; 1.6 bits per weight, as llama.cpp's TQ1_0 packs them before its per-256-block f16 scale) and 2 bits per
weight (TQ2_0's layout), each with one f32 scale per tensor; the token embedding (the tied head) and the norms
are not ternary in BitNet and are priced at F16, Q8_0 (34 bytes per 32) and Q6_K (210 bytes per 256: the
type llama-quantize gives the head in Q4_K_M; at ne0 not a multiple of 256 it falls back to Q8_0). llama.cpp's
TQ1_0/TQ2_0 need ne0 to be a multiple of 256 too, which 20m_L16 (320, 864) is not: the packed sizes are what
an engine with its own row layout would store.
"""
import json
import math
import os
import struct
import sys

import numpy as np

PROJ = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def gguf_sizes(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    groups = {}
    for t in r.tensors:
        n = int(np.prod([int(x) for x in t.shape]))
        b = int(t.n_bytes)
        kind = t.name.split(".")[2] if t.name.startswith("blk.") else t.name.split(".")[0]
        cls = "proj" if kind in PROJ else ("norm" if "norm" in kind else kind)
        g = groups.setdefault(cls, {"params": 0, "bytes": 0, "types": {}})
        g["params"] += n
        g["bytes"] += b
        g["types"][t.tensor_type.name] = g["types"].get(t.tensor_type.name, 0) + 1
    for g in groups.values():
        g["bpw"] = round(8 * g["bytes"] / g["params"], 3)
    return {"file": os.path.basename(path), "file_bytes": os.path.getsize(path), "groups": groups}


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        buf = f.read()
    return {k: np.frombuffer(buf[v["data_offsets"][0]:v["data_offsets"][1]], dtype=np.float32).reshape(v["shape"])
            for k, v in h.items() if k != "__metadata__"}


def ternary(path):
    sd = read_st(path)
    proj = {k: v for k, v in sd.items() if k.startswith("layers.") and k.endswith("_proj.weight")}
    other = {k: v for k, v in sd.items() if k not in proj}
    n_proj = sum(v.size for v in proj.values())
    zeros = 0
    for v in proj.values():
        mean = float(np.abs(v.astype(np.float64)).sum() / v.size)
        s = np.float32(1.0) / np.float32(max(np.float32(mean), np.float32(1e-5)))
        q = np.clip(np.rint(v * s), -1, 1)
        zeros += int((q == 0).sum())
    emb = sd["embed_tokens.weight"]
    n_emb = emb.size
    n_norm = sum(v.size for k, v in other.items() if k != "embed_tokens.weight")
    proj_trit = sum(math.ceil(v.size / 5) + 4 for v in proj.values())
    proj_2bit = sum(math.ceil(v.size / 4) + 4 for v in proj.values())
    emb_price = {"f16": 2 * n_emb, "q8_0": n_emb // 32 * 34, "q6_k": n_emb // 256 * 210}
    out = {"params_proj": n_proj, "params_embedding": n_emb, "params_norm": n_norm,
           "zero_share": round(zeros / n_proj, 4), "proj_bytes_5_per_byte": proj_trit, "proj_bytes_2bit": proj_2bit,
           "embedding_bytes": emb_price, "norm_bytes_f32": 4 * n_norm,
           "total_bytes": {f"{p}+emb_{e}": pb + eb + 4 * n_norm
                           for p, pb in (("trit5", proj_trit), ("2bit", proj_2bit)) for e, eb in emb_price.items()}}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "gguf":
        print(json.dumps([gguf_sizes(p) for p in sys.argv[2:]], indent=1))
    else:
        print(json.dumps(ternary(sys.argv[2]), indent=1))
