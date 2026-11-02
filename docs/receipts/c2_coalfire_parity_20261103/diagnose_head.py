"""C2 diagnosis from the logits backwards. If coalfire computed logits = h @ head.T with the exported head, they lie in
the head's column space (rank d = hidden size, vocab > d). Solve for h by least squares, per row, against each
candidate head; the residual says which head coalfire used, and the recovered h can be compared with the referee's
final-normed hidden state.

  python3 diagnose_head.py HF_DIR IDS COALFIRE.f32
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_safetensors  # noqa: E402
from diagnose import forward  # noqa: E402


def main():
    hf_dir, ids_path, cf_path = sys.argv[1:4]
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    W = load_safetensors(Path(hf_dir) / "model.safetensors")
    ids = [int(t) for t in Path(ids_path).read_text().split()]
    cf = np.fromfile(cf_path, dtype="<f4").astype(np.float64).reshape(len(ids), -1)
    heads = {"lm_head.weight": W.get("lm_head.weight"), "embed_tokens (tied)": W["model.embed_tokens.weight"]}
    out = {}
    for name, M in heads.items():
        if M is None:
            continue
        h, *_ = np.linalg.lstsq(M, cf.T, rcond=None)          # M [V, d], h [d, T]
        resid = np.linalg.norm(M @ h - cf.T, axis=0) / np.linalg.norm(cf.T, axis=0)
        out[name] = {"relative_residual_max": float(resid.max()), "relative_residual_median": float(np.median(resid))}
        print(f"{name}: residual max {resid.max():.3e}, median {np.median(resid):.3e}")
        if name == "lm_head.weight":
            hcf = h.T                                           # coalfire's final hidden rows, as the head saw them
    # the referee's final-normed hidden state
    ref_logits = forward(W, cfg, ids)
    href, *_ = np.linalg.lstsq(heads["lm_head.weight"], ref_logits.T, rcond=None)
    href = href.T
    rms = lambda x: np.sqrt(np.mean(x * x, axis=1))
    cos = np.sum(hcf * href, 1) / (np.linalg.norm(hcf, axis=1) * np.linalg.norm(href, axis=1))
    print("recovered hidden rows: rms coalfire", np.round(rms(hcf)[:4], 4), "referee", np.round(rms(href)[:4], 4))
    print("cosine(coalfire h, referee h) per row: min", round(float(cos.min()), 5), "median", round(float(np.median(cos)), 5))
    gam = W["model.norm.weight"]
    print("final-norm gamma: mean", round(float(gam.mean()), 4), "min", round(float(gam.min()), 4), "max", round(float(gam.max()), 4))
    out["hidden"] = {"rms_coalfire_first_rows": rms(hcf)[:8].tolist(), "rms_referee_first_rows": rms(href)[:8].tolist(),
                     "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos))}
    Path(hf_dir).parent.joinpath("diagnosis_head.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
