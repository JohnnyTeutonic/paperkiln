"""C2 diagnosis per sub-op: the L2 norms coalfire prints under TCPP_LAYER_TRACE ([SUBOP] lines) against the same
quantities from the float64 referee of the HF export, layer by layer. The first diverging sub-op names the op.

  python3 diagnose_subops.py HF_DIR IDS SUBOP_LOG
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_safetensors  # noqa: E402

KEYS = ["in", "ln1", "attn", "res1", "ln2", "ffn", "out"]


def referee_subops(W, cfg, ids):
    d, H, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    hd, eps, theta, T = d // H, cfg["rms_norm_eps"], cfg.get("rope_theta", 10000.0), len(ids)
    rms = lambda x, w: x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w
    ang = np.arange(T)[:, None] * (theta ** (-np.arange(0, hd, 2) / hd))[None, :]
    cos, sin = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([np.sin(ang)] * 2, -1)

    def rope(x):
        return x * cos + np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sin

    mask = np.triu(np.full((T, T), -np.inf), 1)
    h = W["model.embed_tokens.weight"][ids]
    rows = []
    for l in range(L):
        p = f"model.layers.{l}."
        r = {"in": h}
        x = rms(h, W[p + "input_layernorm.weight"]); r["ln1"] = x
        q = rope((x @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, hd).transpose(1, 0, 2))
        k = rope((x @ W[p + "self_attn.k_proj.weight"].T).reshape(T, H, hd).transpose(1, 0, 2))
        v = (x @ W[p + "self_attn.v_proj.weight"].T).reshape(T, H, hd).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        a = ((s / s.sum(-1, keepdims=True)) @ v).transpose(1, 0, 2).reshape(T, d) @ W[p + "self_attn.o_proj.weight"].T
        r["attn"] = a
        h = h + a; r["res1"] = h
        x = rms(h, W[p + "post_attention_layernorm.weight"]); r["ln2"] = x
        g, u = x @ W[p + "mlp.gate_proj.weight"].T, x @ W[p + "mlp.up_proj.weight"].T
        f = (g / (1 + np.exp(-g)) * u) @ W[p + "mlp.down_proj.weight"].T; r["ffn"] = f
        h = h + f; r["out"] = h
        rows.append({k: float(np.linalg.norm(v)) for k, v in r.items()})
    return rows


def main():
    hf_dir, ids_path, log_path = sys.argv[1:4]
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    W = load_safetensors(Path(hf_dir) / "model.safetensors")
    ids = [int(t) for t in Path(ids_path).read_text().split()]
    ref = referee_subops(W, cfg, ids)
    cf = []
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("[SUBOP]"):
            cf.append({k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", line)})
    cf = cf[-len(ref):]  # the last forward's layers
    out = []
    for l, (a, b) in enumerate(zip(cf, ref)):
        row = {k: {"coalfire": a.get(k), "referee": b[k], "rel": abs(a.get(k, np.nan) - b[k]) / b[k]} for k in KEYS}
        out.append(row)
        print(f"L{l} " + "  ".join(f"{k} {a.get(k, float('nan')):.5g}/{b[k]:.5g}" for k in KEYS))
    Path(hf_dir).parent.joinpath("diagnosis_subops.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
