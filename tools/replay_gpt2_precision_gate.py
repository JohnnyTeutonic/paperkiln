"""Independent full-model NumPy forward and raw CPU/CUDA gate replay; no model calls."""

import argparse
import json
from pathlib import Path
import numpy as np

FORWARD_BOUND = 0.005
GRADIENT_BOUND = 0.03
GRADIENT_NORM_FLOOR = 1e-8
LOSS_BOUND = 0.05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matrix(record):
    a = np.asarray(record["values"], dtype=np.float32)
    require(
        a.size == record["rows"] * record["cols"] and np.isfinite(a).all(),
        "invalid raw tensor",
    )
    return a.reshape(record["rows"], record["cols"])


def rounded(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    hi = u >> 16
    lo = u & 65535
    return (
        ((hi + ((lo > 32768) | ((lo == 32768) & ((hi & 1) != 0)))) << 16)
        .astype(np.uint32)
        .view(np.float32)
    )


def product(a, b, bf16=False):
    a, b = (rounded(a), rounded(b)) if bf16 else (a, b)
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)


def forward(record):
    """No C++/IR interpreter imports. Fixed two-block model; FP32 projections."""
    p = {name: matrix(value) for name, value in record["parameters"].items()}
    ids = np.asarray(record["ids"], dtype=int)
    sl = record["seq_len"]
    d = record["d"]
    heads = record["heads"]
    dk = d // heads
    require(
        (record["vocab"], d, heads, record["layers"], sl) == (16, 8, 2, 2, 4),
        "fixed model differs",
    )
    require(ids.tolist() == [1, 2, 3, 4, 1, 2, 3, 4], "fixed ids differ")
    x = (p["wte.weight"][ids] + p["wpe.weight"][np.arange(len(ids)) % sl]).astype(
        np.float32
    )

    def norm(a, name):
        a = a.astype(np.float64)
        mean = a.mean(axis=1, keepdims=True)
        var = ((a - mean) ** 2).mean(axis=1, keepdims=True)
        z = ((a - mean) / np.sqrt(var + 1e-5)).astype(np.float32)
        return (z * p[name + ".weight"] + p[name + ".bias"]).astype(np.float32)

    def linear(a, name):
        z = product(a, p[name + ".weight"])
        return (z + p[name + ".bias"]).astype(np.float32) if name + ".bias" in p else z

    visible = (
        np.arange(len(ids))[:, None] // sl == np.arange(len(ids))[None, :] // sl
    ) & (np.arange(len(ids))[None, :] <= np.arange(len(ids))[:, None])
    for block in range(2):
        qkv = linear(norm(x, f"ln1_{block}"), f"attn_{block}.c_attn")
        out = []
        for h in range(heads):
            q = qkv[:, h * dk : (h + 1) * dk]
            k = qkv[:, d + h * dk : d + (h + 1) * dk]
            v = qkv[:, 2 * d + h * dk : 2 * d + (h + 1) * dk]
            scores = product(q, k.T, True).astype(np.float64) / np.sqrt(dk)
            scores = np.where(visible, scores, -np.inf)
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            a = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
            out.append(product(a, v, True))
        x = (x + linear(np.concatenate(out, axis=1), f"attn_{block}.c_proj")).astype(
            np.float32
        )
        z = linear(norm(x, f"ln2_{block}"), f"mlp_{block}.c_fc").astype(np.float64)
        z = (
            0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        ).astype(np.float32)
        x = (x + linear(z, f"mlp_{block}.c_proj")).astype(np.float32)
    return linear(norm(x, "ln_f"), "head")


def error(a, b):
    require(a.shape == b.shape, "tensor shapes differ")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) / max(
        1.0, float(np.max(np.abs(b)))
    )


def expected_shapes():
    result = {
        "wte.weight": (16, 8),
        "wpe.weight": (4, 8),
        "ln_f.weight": (1, 8),
        "ln_f.bias": (1, 8),
        "head.weight": (8, 16),
    }
    for b in range(2):
        for norm in ("ln1", "ln2"):
            for kind in ("weight", "bias"):
                result[f"{norm}_{b}.{kind}"] = (1, 8)
        for base, dims in (
            (f"attn_{b}.c_attn", (8, 24)),
            (f"attn_{b}.c_proj", (8, 8)),
            (f"mlp_{b}.c_fc", (8, 32)),
            (f"mlp_{b}.c_proj", (32, 8)),
        ):
            result[base + ".weight"] = dims
            result[base + ".bias"] = (1, dims[1])
    return result


def check(record, require_cuda=False):
    require(
        record["pass"] is True and record["actual_cuda"] is require_cuda,
        "backend proof differs",
    )
    require(
        record["default_preserved_logits_gradients_bit_equal"] is True,
        "missing default bit proof",
    )
    shapes = expected_shapes()
    require(
        set(record["parameters"]) == set(record["gradients"]) == set(shapes),
        "complete fixed parameter inventory differs",
    )
    for name, shape in shapes.items():
        require(
            matrix(record["parameters"][name]).shape
            == matrix(record["gradients"][name]).shape
            == shape,
            "fixed parameter/gradient shape differs: " + name,
        )
    actual = matrix(record["logits"])
    expected = forward(record)
    err = error(actual, expected)
    require(err <= FORWARD_BOUND, "full-model independent forward bound")
    require(
        set(record["gradients"]) == set(record["parameters"]),
        "gradient inventory differs",
    )
    grads = [matrix(g) for g in record["gradients"].values()]
    require(any(np.any(g != 0) for g in grads), "all model gradients zero")
    losses = np.asarray(record["losses"], dtype=float)
    require(
        record["steps"] == 16 and losses.shape == (16,) and np.isfinite(losses).all(),
        "short loss trace differs",
    )
    if require_cuda:
        require(
            record["bf16_profile_counts"].get("gemm_bf16", 0) == 24,
            "missing six attention GEMMs",
        )
        require(
            record["training_profile_counts"].get("gemm_bf16", 0) == 16 * 24,
            "missing training BF16 GEMMs",
        )
        require(
            record["default_profile_counts"].get("gemm_bf16", 0) == 0,
            "default contaminated",
        )
    return err


def replay(cpu, cuda=None, inference=None):
    result = {
        "pass": True,
        "cpu_forward_error": check(cpu),
        "network_calls": 0,
        "quality_or_speed_claim": False,
        "bounds": {
            "forward_normalized_max": FORWARD_BOUND,
            "cpu_cuda_gradient_normalized_max": GRADIENT_BOUND,
            "per_tensor_gradient_relative_norm_max": GRADIENT_BOUND,
            "gradient_norm_floor": GRADIENT_NORM_FLOOR,
            "loss_trace_absolute_max": LOSS_BOUND,
        },
    }
    if cuda is not None:
        result["cuda_forward_error"] = check(cuda, True)
        require(cpu["parameters"] == cuda["parameters"], "initial weights differ")
        ge = max(
            error(matrix(cuda["gradients"][k]), matrix(cpu["gradients"][k]))
            for k in cpu["gradients"]
        )
        tensor_norm_errors = {}
        for name in cpu["gradients"]:
            expected = matrix(cpu["gradients"][name]).astype(np.float64)
            actual = matrix(cuda["gradients"][name]).astype(np.float64)
            reference_norm = float(np.linalg.norm(expected))
            actual_norm = float(np.linalg.norm(actual))
            require(
                reference_norm == 0 or actual_norm > 0,
                "nonzero expected gradient tensor was zeroed",
            )
            relative = float(np.linalg.norm(actual - expected)) / max(
                reference_norm, GRADIENT_NORM_FLOOR
            )
            require(
                relative <= GRADIENT_BOUND,
                "per-tensor relative gradient norm bound: " + name,
            )
            tensor_norm_errors[name] = relative
        le = float(
            np.max(np.abs(np.asarray(cuda["losses"]) - np.asarray(cpu["losses"])))
        )
        require(
            ge <= GRADIENT_BOUND and le <= LOSS_BOUND,
            "CPU/CUDA surrogate gradient or loss bound",
        )
        result.update(
            actual_cuda=True,
            cpu_cuda_gradient_error=ge,
            cpu_cuda_loss_error=le,
            per_tensor_relative_gradient_norm_errors=tensor_norm_errors,
        )
    if inference is not None:
        require(
            inference["pass"] is True and inference["actual_cuda"] is True,
            "loaded inference backend",
        )
        require(
            inference["loaded_attention_precision"] == "fp32",
            "loaded inference precision",
        )
        c = inference["profile_counts"]
        require(
            c.get("gemm_bf16", 0) + c.get("gemm_head_bf16", 0) == 0,
            "BF16 loaded inference",
        )
        require(
            c.get("gemm", 0) + c.get("gemm_head", 0) > 0, "missing FP32 inference GEMMs"
        )
        matrix(inference["logits"])
        result["loaded_inference_profiler_fp32_only"] = True
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--cuda", type=Path)
    p.add_argument("--inference", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    def load(path):
        return json.loads(path.read_text()) if path else None

    r = replay(load(a.cpu), load(a.cuda), load(a.inference))
    text = json.dumps(r, indent=1) + "\n"
    if a.out:
        with a.out.open("x") as f:
            f.write(text)
    print(text, end="")
