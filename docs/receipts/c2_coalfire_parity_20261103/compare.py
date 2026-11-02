"""C2 comparison: coalfire's and paperkiln's logits for the same ids, against an independent float64 forward of the
HF export (HF LlamaForCausalLM semantics, written here from the config: RMSNorm x / sqrt(mean(x^2) + eps) * w,
rotate-half RoPE with base rope_theta, causal softmax attention, SiLU-gated MLP, tied or untied head).

  python3 compare.py HF_DIR IDS COALFIRE.f32 PAPERKILN.f32 OUT.json

Pass: every row of each engine within 1e-4 of the referee (max |a - ref| / max(1, max |ref row|)), identical argmax
per row, finite. Engine-against-engine figures are reported alongside. Exit 1 on failure.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

BOUND = 1e-4


def load_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(f"{name}: dtype {meta['dtype']}")
        a, b = meta["data_offsets"]
        out[name] = np.frombuffer(raw[base + a:base + b], dtype="<f4").reshape(meta["shape"]).astype(np.float64)
    return out


def referee(hf_dir, ids):
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    W = load_safetensors(Path(hf_dir) / "model.safetensors")
    d, H, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    KV = cfg.get("num_key_value_heads", H)
    hd = d // H
    eps, theta = cfg["rms_norm_eps"], cfg.get("rope_theta", 10000.0)
    T = len(ids)

    def rms(x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

    inv = theta ** (-np.arange(0, hd, 2) / hd)              # [hd/2]
    ang = np.arange(T)[:, None] * inv[None, :]               # [T, hd/2]
    cos = np.concatenate([np.cos(ang)] * 2, axis=-1)         # [T, hd]
    sin = np.concatenate([np.sin(ang)] * 2, axis=-1)

    def rope(x):                                             # x [heads, T, hd], rotate-half
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return x * cos + np.concatenate([-x2, x1], axis=-1) * sin

    h = W["model.embed_tokens.weight"][ids]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for l in range(L):
        p = f"model.layers.{l}."
        x = rms(h, W[p + "input_layernorm.weight"])
        q = (x @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, hd).transpose(1, 0, 2)
        k = (x @ W[p + "self_attn.k_proj.weight"].T).reshape(T, KV, hd).transpose(1, 0, 2)
        v = (x @ W[p + "self_attn.v_proj.weight"].T).reshape(T, KV, hd).transpose(1, 0, 2)
        q, k = rope(q), rope(k)
        k, v = np.repeat(k, H // KV, axis=0), np.repeat(v, H // KV, axis=0)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        a = (s / s.sum(-1, keepdims=True)) @ v               # [H, T, hd]
        h = h + a.transpose(1, 0, 2).reshape(T, d) @ W[p + "self_attn.o_proj.weight"].T
        x = rms(h, W[p + "post_attention_layernorm.weight"])
        g = x @ W[p + "mlp.gate_proj.weight"].T
        u = x @ W[p + "mlp.up_proj.weight"].T
        h = h + (g / (1 + np.exp(-g)) * u) @ W[p + "mlp.down_proj.weight"].T
    h = rms(h, W["model.norm.weight"])
    head = W["model.embed_tokens.weight"] if cfg.get("tie_word_embeddings") else W["lm_head.weight"]
    return h @ head.T, cfg


def rows_err(a, ref):
    return np.max(np.abs(a - ref), axis=1) / np.maximum(1.0, np.max(np.abs(ref), axis=1))


def main():
    hf_dir, ids_path, cf_path, pk_path, out_path = sys.argv[1:6]
    ids = [int(t) for t in Path(ids_path).read_text().split()]
    ref, cfg = referee(hf_dir, ids)
    V = ref.shape[1]
    report = {"ids": len(ids), "vocab": V, "bound": BOUND, "config": cfg, "engines": {}}
    ok = True
    logits = {}
    for name, p in (("coalfire", cf_path), ("paperkiln", pk_path)):
        a = np.fromfile(p, dtype="<f4").astype(np.float64)
        if a.size != len(ids) * V:
            report["engines"][name] = {"pass": False, "reason": f"{a.size} values, expected {len(ids) * V}"}
            ok = False
            continue
        a = a.reshape(len(ids), V)
        logits[name] = a
        e = rows_err(a, ref)
        argmax = bool(np.all(a.argmax(1) == ref.argmax(1)))
        finite = bool(np.isfinite(a).all())
        passed = finite and argmax and float(e.max()) <= BOUND
        report["engines"][name] = {"pass": passed, "max_rel_vs_float64": float(e.max()),
                                   "median_rel_vs_float64": float(np.median(e)), "argmax_identical": argmax,
                                   "finite": finite}
        ok = ok and passed
    if len(logits) == 2:
        e = rows_err(logits["coalfire"], logits["paperkiln"])
        report["coalfire_vs_paperkiln"] = {"max_rel": float(e.max()),
                                           "argmax_identical": bool(np.all(logits["coalfire"].argmax(1) ==
                                                                           logits["paperkiln"].argmax(1)))}
    report["pass"] = ok
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
