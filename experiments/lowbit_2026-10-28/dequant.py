#!/usr/bin/env python3
"""Weights-only, untied safetensors for a paperkiln llama GGUF of any tensor type (F32, Q8_0, Q4_0, Q4_K,
Q5_K, Q6_K, Q5_0, Q5_1), for `mtstudio` (LOCAL_LLM_WORK R3, experiments/lowbit_2026-10-28).

    python dequant.py MODEL.gguf OUT.safetensors

Every tensor is dequantised with gguf-py (llama.cpp b6000's reference dequantisers; ember's are bit-identical
to them, ember.cpp/experiments/kquant_2026-10-25), then mapped onto paperkiln's state_dict as
experiments/benchmarks_2026-10-26/rebuild_weights.py does (projections transposed back to [in, out], norms
[1, d]). A llama.cpp quantised file stores token_embd and output separately (Q4_K_M: Q4_K and Q6_K), so the
head is written as its own lm_head.weight (output transposed to [d, vocab]) and the model must be loaded with
arch.custom.tie_embeddings false. ember computes with exactly these dequantised values in F32, so mtstudio on
OUT is ember's quantised model up to F32 kernel rounding; llama.cpp additionally quantises activations to
Q8_K in its quantised matmuls, which this does not model.
"""
import json
import os
import struct
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

PROJ = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj"}
NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


def tensors(path):
    """{name: float32 array in numpy order [ne1, ne0]} and {name: type name}."""
    r = GGUFReader(path)
    out, types = {}, {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape)]
        a = dequantize(t.data, t.tensor_type).astype(np.float32, copy=False).reshape(shape)
        out[t.name], types[t.name] = a, t.tensor_type.name
    return out, types


def state_dict(path):
    W, types = tensors(path)
    sd = {"embed_tokens.weight": W.pop("token_embd.weight")}
    head = W.pop("output.weight", None)
    sd["lm_head.weight"] = np.ascontiguousarray((head if head is not None else sd["embed_tokens.weight"]).T)
    sd["norm.weight"] = W.pop("output_norm.weight").reshape(1, -1)
    for name in sorted(W):
        parts = name.split(".")
        if parts[0] != "blk" or len(parts) != 4:
            sys.exit(f"unexpected tensor {name}")
        l, kind = parts[1], parts[2]
        if kind in PROJ:
            sd[f"layers.{l}.{PROJ[kind]}.weight"] = np.ascontiguousarray(W[name].T)
        elif kind in NORM:
            sd[f"layers.{l}.{NORM[kind]}.weight"] = W[name].reshape(1, -1)
        else:
            sys.exit(f"unexpected tensor {name}")
    return sd, types


def write_st(sd, path):
    header, off = {}, 0
    keys = sorted(sd)
    for k in keys:
        nbytes = sd[k].size * 4
        header[k] = {"dtype": "F32", "shape": list(sd[k].shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for k in keys:
            f.write(np.ascontiguousarray(sd[k], dtype="<f4").tobytes())
    os.replace(path + ".tmp", path)


def main():
    gguf, out = sys.argv[1], sys.argv[2]
    sd, types = state_dict(gguf)
    write_st(sd, out)
    count = {}
    for t in types.values():
        count[t] = count.get(t, 0) + 1
    print(json.dumps({"gguf": gguf, "out": out, "tensors": len(sd), "types": count}))


if __name__ == "__main__":
    main()
