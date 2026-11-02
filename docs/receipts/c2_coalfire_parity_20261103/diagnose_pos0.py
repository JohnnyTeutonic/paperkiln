"""C2 diagnosis at position 0, where attention returns v and RoPE is the identity, so only the embedding, norms,
MLP and head matter. Each variant changes one assumption; one near float32 rounding names what coalfire computes.

  python3 diagnose_pos0.py HF_DIR IDS COALFIRE.f32
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_safetensors, rows_err  # noqa: E402


def pos0(W, cfg, tok, emb_scale=1.0, norm_plus_one=False, norm_weight=True, act="silu", swap_gate_up=False,
         residual=True, gelu_tanh=False, dropout_scale=1.0, ffn_out_scale=1.0, attn_out_scale=1.0):
    d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    eps = cfg["rms_norm_eps"]

    def nrm(x, w):
        y = x / np.sqrt(np.mean(x * x) + eps)
        if not norm_weight:
            return y
        return y * (1 + w) if norm_plus_one else y * w

    def a(g):
        if act == "silu":
            return g / (1 + np.exp(-g))
        if act == "gelu":
            from math import erf
            return np.array([0.5 * v * (1 + erf(v / np.sqrt(2))) for v in g])
        if act == "relu":
            return np.maximum(g, 0)
        return g

    h = W["model.embed_tokens.weight"][tok] * emb_scale
    for l in range(L):
        p = f"model.layers.{l}."
        x = nrm(h, W[p + "input_layernorm.weight"])
        v = x @ W[p + "self_attn.v_proj.weight"].T            # one position: softmax weight 1 on itself
        att = v @ W[p + "self_attn.o_proj.weight"].T * attn_out_scale
        h = h + att if residual else att
        x = nrm(h, W[p + "post_attention_layernorm.weight"])
        gw, uw = W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"]
        if swap_gate_up:
            gw, uw = uw, gw
        f = (a(x @ gw.T) * (x @ uw.T)) @ W[p + "mlp.down_proj.weight"].T * ffn_out_scale * dropout_scale
        h = h + f if residual else f
    h = nrm(h, W["model.norm.weight"])
    head = W["model.embed_tokens.weight"] if cfg.get("tie_word_embeddings") else W["lm_head.weight"]
    return h @ head.T


def main():
    hf_dir, ids_path, cf_path = sys.argv[1:4]
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    W = load_safetensors(Path(hf_dir) / "model.safetensors")
    ids = [int(t) for t in Path(ids_path).read_text().split()]
    cf = np.fromfile(cf_path, dtype="<f4").astype(np.float64).reshape(len(ids), -1)[0]
    d = cfg["hidden_size"]
    variants = {
        "HF semantics": {}, "embedding x sqrt(d)": {"emb_scale": np.sqrt(d)}, "norm (1 + w)": {"norm_plus_one": True},
        "norms without weight": {"norm_weight": False}, "GELU (erf)": {"act": "gelu"}, "ReLU": {"act": "relu"},
        "gate/up swapped": {"swap_gate_up": True}, "FFN output x 0.9 (dropout scaling)": {"dropout_scale": 0.9},
        "FFN output x 1/0.9 (inverted dropout)": {"dropout_scale": 1 / 0.9},
        "attention output x 0.9": {"attn_out_scale": 0.9}, "attention output x 0": {"attn_out_scale": 0.0},
        "FFN output x 0": {"ffn_out_scale": 0.0},
    }
    out = {}
    for name, kw in variants.items():
        ref = pos0(W, cfg, ids[0], **kw)
        e = float(rows_err(cf[None, :], ref[None, :]).max())
        out[name] = {"max_rel": e, "argmax_identical": bool(cf.argmax() == ref.argmax())}
        print(f"{e:10.3e} {str(out[name]['argmax_identical']):>6}  {name}")
    Path(hf_dir).parent.joinpath("diagnosis_pos0.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
