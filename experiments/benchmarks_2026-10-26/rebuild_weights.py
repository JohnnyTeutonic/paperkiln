#!/usr/bin/env python3
"""Weights-only safetensors for a paperkiln llama GGUF (F32), for `mtstudio score`.

    python rebuild_weights.py MODEL.gguf OUT.safetensors [--check-against REF.safetensors REF.gguf]

The tiny-chat picks' model.safetensors hold the SFT run's AdamW state, not the weights (tiny_chat_run.py pick
copies the out_dir's optim.safetensors at dc29b5e6; best.json's weights_sha256 is that file's hash). Their
weights are model.gguf. This writes paperkiln's state_dict from it: the inverse of src/gguf.cpp's
export_gguf_llama (token_embd -> embed_tokens as stored; every projection transposed back from GGUF's
[out][in] to paperkiln's [in, out]; norms as [1, d]; output.weight dropped, since paperkiln ties the head to
embed_tokens, and checked equal to token_embd). --check-against runs the same inverse on a model that has
both files (pk_sft_v1) and requires every tensor bit-identical to its real safetensors. The round trip
(mtstudio export of OUT back to GGUF, tensors bit-identical to MODEL.gguf) is checked separately.
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "spec", "tools"))
import ir_check  # noqa: E402

PROJ = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj"}
NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


def state_dict(gguf):
    meta, tensors, ds, raw, _ = ir_check.read_gguf(gguf)
    if any(t[2] != 0 for t in tensors):
        sys.exit(f"{gguf}: not all F32")
    W = {name: np.frombuffer(raw, dtype=np.float32, count=int(np.prod(dims)), offset=ds + off)
         .reshape(list(reversed(dims))) for name, dims, _, off in tensors}
    sd = {"embed_tokens.weight": W.pop("token_embd.weight")}
    out = W.pop("output.weight", None)
    if out is not None and not np.array_equal(out, sd["embed_tokens.weight"]):
        sys.exit("output.weight differs from token_embd.weight: an untied head, not handled here")
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
    return sd


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        buf = f.read()
    out = {}
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(buf[a:b], dtype=np.float32).reshape(v["shape"])
    return out, base


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
    if "--check-against" in sys.argv:
        i = sys.argv.index("--check-against")
        ref_st, ref_gguf = sys.argv[i + 1], sys.argv[i + 2]
        mine, real = state_dict(ref_gguf), read_st(ref_st)[0]
        if set(mine) != set(real):
            sys.exit(f"keys differ: {sorted(set(mine) ^ set(real))[:8]}")
        bad = [k for k in real if mine[k].shape != real[k].shape or not np.array_equal(mine[k], real[k])]
        if bad:
            sys.exit(f"{len(bad)} tensors differ from {ref_st}, e.g. {bad[:4]}")
        print(f"inverse checked on {ref_gguf}: {len(real)} tensors bit-identical to {ref_st}")
    sd = state_dict(gguf)
    write_st(sd, out)
    print(json.dumps({"gguf": gguf, "out": out, "tensors": len(sd)}))


if __name__ == "__main__":
    main()
