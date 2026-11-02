"""C2 diagnosis: which variant of the float64 referee reproduces coalfire's own logits?

  python3 diagnose.py HF_DIR IDS COALFIRE.f32

Variants change one assumption each against compare.py's HF semantics and report max relative error against
coalfire's logits. A variant that lands near float32 rounding (about 1e-6) identifies what coalfire computes.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_safetensors, rows_err  # noqa: E402


def forward(W, cfg, ids, rope_mode="half", eps=None, theta=None, pos0=0, scale=None, final_norm=True,
            norm="rms", unpermute=False):
    d, H, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    hd = d // H
    eps = cfg["rms_norm_eps"] if eps is None else eps
    theta = cfg.get("rope_theta", 10000.0) if theta is None else theta
    T = len(ids)

    def nrm(x, w):
        if norm == "layer":
            mu = x.mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * w
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

    pos = np.arange(T) + pos0
    inv = theta ** (-np.arange(0, hd, 2) / hd)
    ang = pos[:, None] * inv[None, :]

    def rope(x):  # x [heads, T, hd]
        if rope_mode == "none":
            return x
        if rope_mode == "half":
            c, s = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([np.sin(ang)] * 2, -1)
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return x * c + np.concatenate([-x2, x1], -1) * s
        # adjacent pairs (0,1), (2,3), ...
        c, s = np.cos(ang), np.sin(ang)
        xe, xo = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = xe * c - xo * s
        out[..., 1::2] = xe * s + xo * c
        return out

    def unperm(w):  # [out, in] HF rotate-half rows back to interleaved order, per head
        w = w.reshape(H, 2, hd // 2, -1)
        return w.transpose(0, 2, 1, 3).reshape(H * hd, -1)

    h = W["model.embed_tokens.weight"][ids]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for l in range(L):
        p = f"model.layers.{l}."
        wq, wk = W[p + "self_attn.q_proj.weight"], W[p + "self_attn.k_proj.weight"]
        if unpermute:
            wq, wk = unperm(wq), unperm(wk)
        x = nrm(h, W[p + "input_layernorm.weight"])
        q = (x @ wq.T).reshape(T, H, hd).transpose(1, 0, 2)
        k = (x @ wk.T).reshape(T, H, hd).transpose(1, 0, 2)
        v = (x @ W[p + "self_attn.v_proj.weight"].T).reshape(T, H, hd).transpose(1, 0, 2)
        q, k = rope(q), rope(k)
        sc = (1 / np.sqrt(hd)) if scale is None else scale
        s = q @ k.transpose(0, 2, 1) * sc + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        a = (s / s.sum(-1, keepdims=True)) @ v
        h = h + a.transpose(1, 0, 2).reshape(T, d) @ W[p + "self_attn.o_proj.weight"].T
        x = nrm(h, W[p + "post_attention_layernorm.weight"])
        g, u = x @ W[p + "mlp.gate_proj.weight"].T, x @ W[p + "mlp.up_proj.weight"].T
        h = h + (g / (1 + np.exp(-g)) * u) @ W[p + "mlp.down_proj.weight"].T
    if final_norm:
        h = nrm(h, W["model.norm.weight"])
    head = W["model.embed_tokens.weight"] if cfg.get("tie_word_embeddings") else W["lm_head.weight"]
    return h @ head.T


def main():
    hf_dir, ids_path, cf_path = sys.argv[1:4]
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    W = load_safetensors(Path(hf_dir) / "model.safetensors")
    ids = [int(t) for t in Path(ids_path).read_text().split()]
    cf = np.fromfile(cf_path, dtype="<f4").astype(np.float64).reshape(len(ids), -1)
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    variants = {
        "HF semantics (compare.py)": {},
        "no RoPE": {"rope_mode": "none"},
        "adjacent-pair RoPE on the exported (permuted) weights": {"rope_mode": "pairs"},
        "adjacent-pair RoPE on un-permuted weights": {"rope_mode": "pairs", "unpermute": True},
        "rotate-half RoPE on un-permuted weights": {"rope_mode": "half", "unpermute": True},
        "positions start at 1": {"pos0": 1},
        "eps 1e-6": {"eps": 1e-6},
        "LayerNorm instead of RMSNorm": {"norm": "layer"},
        "no final norm": {"final_norm": False},
        "attention scale 1/hd": {"scale": 1.0 / hd},
        "attention scale 1": {"scale": 1.0},
    }
    out = {}
    for name, kw in variants.items():
        ref = forward(W, cfg, ids, **kw)
        e = rows_err(cf, ref)
        out[name] = {"max_rel": float(e.max()), "median_rel": float(np.median(e)),
                     "argmax_identical": bool(np.all(cf.argmax(1) == ref.argmax(1)))}
        print(f"{e.max():10.3e} {np.median(e):10.3e} {str(out[name]['argmax_identical']):>6}  {name}")
    first = rows_err(cf[:1], forward(W, cfg, ids[:1]))  # one position: attention reduces to v, RoPE has no effect
    print(f"position 0 only, HF semantics: {first.max():.3e}")
    out["position 0 only (HF semantics)"] = float(first.max())
    Path(hf_dir).parent.joinpath("diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
