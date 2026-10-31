"""Independent binary64 Mamba1 equations and every-parameter central differences.
No production/IR evaluator is imported. Raw frozen weights are data, not derivatives.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np

BOUNDS = {
    "forward_normalized": 3e-5,
    "gradient_normalized": 3e-5,
    "tensor_gradient_relative_l2": 0.02,
    "fd_stability_relative_l2": 0.02,
    "relative_l2_floor": 1e-12,
}
IDS = [0, 1, 2, 3, 4, 5, 6, 0]
TARGETS = [1, 2, 3, 4, 5, 6, 0, 1]


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def shapes(depth):
    result = {"embed.weight": (7, 8), "norm_f.weight": (1, 8), "lm_head.weight": (8, 7)}
    for i in range(depth):
        p = f"layers.{i}."
        result.update(
            {
                p + n: s
                for n, s in {
                    "norm.weight": (1, 8),
                    "conv1d.weight": (8, 3),
                    "conv1d.bias": (1, 8),
                    "A_log": (8, 2),
                    "D": (1, 8),
                    "in_proj.weight": (8, 16),
                    "x_proj.weight": (8, 6),
                    "dt_proj.weight": (2, 8),
                    "dt_proj.bias": (1, 8),
                    "out_proj.weight": (8, 8),
                }.items()
            }
        )
    return result


def rms(x, w):
    return (
        x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + float(np.float32(1e-5))) * w
    )


def silu(x):
    return x / (1 + np.exp(-x))


def forward(weights, depth):
    h = weights["embed.weight"][IDS].copy()
    for layer in range(depth):
        p = f"layers.{layer}."
        residual = h
        xz = rms(h, weights[p + "norm.weight"]) @ weights[p + "in_proj.weight"]
        xs = xz[:, :8].reshape(2, 4, 8)
        z = xz[:, 8:]
        conv = np.zeros((2, 4, 8), dtype=np.float64)
        for time in range(4):
            for tap in range(3):
                source = time + tap - 2
                if source >= 0:
                    conv[:, time, :] += (
                        xs[:, source, :] * weights[p + "conv1d.weight"][:, tap]
                    )
        xs = silu(conv.reshape(8, 8) + weights[p + "conv1d.bias"])
        xdb = xs @ weights[p + "x_proj.weight"]
        raw_dt = (
            xdb[:, :2] @ weights[p + "dt_proj.weight"] + weights[p + "dt_proj.bias"]
        )
        delta = np.where(raw_dt > 20, raw_dt, np.log1p(np.exp(raw_dt)))
        bm = xdb[:, 2:4]
        cm = xdb[:, 4:6]
        a = -np.exp(weights[p + "A_log"])
        scan = np.zeros((8, 8), dtype=np.float64)
        for sequence in range(2):
            state = np.zeros((8, 2), dtype=np.float64)
            for time in range(4):
                row = sequence * 4 + time
                dt = delta[row, :, None]
                state = (
                    np.exp(dt * a) * state + dt * bm[row, None, :] * xs[row, :, None]
                )
                scan[row] = (
                    np.sum(state * cm[row, None, :], axis=1)
                    + weights[p + "D"][0] * xs[row]
                )
        h = residual + (scan * silu(z)) @ weights[p + "out_proj.weight"]
    logits = rms(h, weights["norm_f.weight"]) @ weights["lm_head.weight"]
    maximum = logits.max(axis=1)
    logsum = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    loss = float(np.mean(logsum - logits[np.arange(8), TARGETS]))
    return logits, loss


def ce_difference(positive, negative):
    # CE(positive)-CE(negative), without subtracting nearly equal O(1) losses.
    shifted = negative - negative.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    change = positive - negative
    return float(
        np.mean(
            np.log1p(np.sum(probabilities * np.expm1(change), axis=1))
            - change[np.arange(8), TARGETS]
        )
    )


def finite_differences(weights, depth):
    gradients = {}
    stability = {}
    for name, array in weights.items():
        grad = np.zeros_like(array)
        other = np.zeros_like(array)
        step = (
            0.1
            if name.endswith("dt_proj.weight")
            else 0.01
            if name.endswith("A_log") or name.endswith("dt_proj.bias")
            else 0.0001
        )
        for index in np.ndindex(array.shape):
            original = array[index]
            for h, dest in ((step, other), (step / 2, grad)):
                array[index] = original + h
                plus = forward(weights, depth)[0]
                array[index] = original - h
                minus = forward(weights, depth)[0]
                dest[index] = ce_difference(plus, minus) / (2 * h)
            array[index] = original
        denom = max(float(np.linalg.norm(grad)), BOUNDS["relative_l2_floor"])
        stability[name] = float(np.linalg.norm(grad - other)) / denom
        need(
            stability[name] <= BOUNDS["fd_stability_relative_l2"],
            "FD step stability " + name,
        )
        gradients[name] = grad
    return gradients, stability


def array(value, size, label):
    need(isinstance(value, list) and len(value) == size, label + " inventory")
    need(
        all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        ),
        label + " finite",
    )
    return np.array(value, dtype=np.float64)


def load(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.startswith("{")
    ]


def replay(rows, fixture, gpu=False):
    need(len(rows) == 3 and isinstance(rows[-1], dict), "summary inventory")
    summary = rows[-1]
    need(
        set(summary) == {"summary", "cases", "actual_cuda", "pass"}
        and summary["summary"] is True
        and type(summary["cases"]) is int
        and summary["cases"] == 2
        and summary["actual_cuda"] is gpu
        and summary["pass"] is True,
        "summary",
    )
    need(isinstance(fixture, list) and len(fixture) == 2, "fixture cases")
    results = []
    for depth, row, frozen in zip((1, 2), rows[:2], fixture):
        need(
            type(row.get("depth")) is int
            and row.get("depth") == depth
            and row.get("actual_cuda") is gpu
            and row.get("devcheck") is True,
            "backend/depth",
        )
        for name, value in {
            "seed": 37,
            "d": 8,
            "vocab": 7,
            "state": 2,
            "expand": 1,
            "dt_rank": 2,
            "conv": 3,
            "batch": 2,
            "seq_len": 4,
            "ids": IDS,
            "targets": TARGETS,
            "logits_shape": [8, 7],
            "rms_eps": float(np.float32(1e-5)),
        }.items():
            need(row.get(name) == value, "registered geometry/fixture " + name)
            if type(value) is int:
                need(type(row[name]) is int, "integer geometry " + name)
            if isinstance(value, list):
                need(
                    isinstance(row[name], list)
                    and all(type(x) is int for x in row[name]),
                    "integer fixture " + name,
                )
        expected = shapes(depth)
        params = row.get("parameters")
        need(
            isinstance(params, dict) and set(params) == set(expected), "parameter names"
        )
        need(
            set(frozen) == {"depth", "parameters"}
            and type(frozen.get("depth")) is int
            and frozen.get("depth") == depth
            and set(frozen.get("parameters", {})) == set(expected),
            "frozen inventory",
        )
        weights = {}
        actual_grad = {}
        for name, shape in expected.items():
            need(
                params[name].get("shape") == list(shape)
                and frozen["parameters"][name].get("shape") == list(shape),
                "parameter shape " + name,
            )
            need(
                set(params[name]) == {"shape", "values", "gradient"}
                and set(frozen["parameters"][name]) == {"shape", "values"},
                "tensor field inventory " + name,
            )
            need(
                all(type(x) is int for x in params[name]["shape"])
                and all(type(x) is int for x in frozen["parameters"][name]["shape"]),
                "integer tensor shape " + name,
            )
            weights[name] = array(
                params[name].get("values"), math.prod(shape), name + " weights"
            ).reshape(shape)
            frozen_values = array(
                frozen["parameters"][name].get("values"),
                math.prod(shape),
                name + " fixture",
            ).reshape(shape)
            need(
                np.array_equal(weights[name], frozen_values),
                "frozen weight values " + name,
            )
            need(
                np.array_equal(
                    weights[name], weights[name].astype(np.float32).astype(np.float64)
                ),
                "FP32 raw weights " + name,
            )
            actual_grad[name] = array(
                params[name].get("gradient"), math.prod(shape), name + " gradient"
            ).reshape(shape)
        if gpu:
            counts = row.get("gpu_n", {})
            ms = row.get("gpu_ms", {})
            for op, count in {
                "selective_scan_fwd": depth,
                "selective_scan_bwd": depth,
                "conv1d_causal_fwd": depth,
                "conv1d_causal_bwd": 3 * depth,
                "rmsnorm_fwd": depth + 1,
                "rmsnorm_bwd": 2 * (depth + 1),
                "gemm": 12 * depth + 3,
            }.items():
                need(
                    type(counts.get(op)) is int
                    and counts.get(op) == count
                    and op in ms
                    and isinstance(ms[op], (int, float))
                    and not isinstance(ms[op], bool)
                    and math.isfinite(ms[op])
                    and ms[op] >= 0,
                    "actual device profile " + op,
                )
        else:
            need(row.get("gpu_n") == {} and row.get("gpu_ms") == {}, "CPU profile")
        logits, loss = forward(weights, depth)
        observed = array(row.get("logits"), 56, "logits").reshape(8, 7)
        forward_error = float(np.max(abs(logits - observed) / (1 + abs(logits))))
        need(forward_error <= BOUNDS["forward_normalized"], "forward bound")
        need(
            isinstance(row.get("loss"), (int, float))
            and math.isfinite(row["loss"])
            and abs(loss - row["loss"]) / (1 + abs(loss))
            <= BOUNDS["forward_normalized"],
            "loss bound",
        )
        gradients, stability = finite_differences(weights, depth)
        metrics = {}
        for name, ref in gradients.items():
            actual = actual_grad[name]
            absolute = float(np.max(abs(actual - ref) / (1 + abs(ref))))
            relative = float(np.linalg.norm(actual - ref)) / max(
                float(np.linalg.norm(ref)), BOUNDS["relative_l2_floor"]
            )
            need(
                absolute <= BOUNDS["gradient_normalized"]
                and relative <= BOUNDS["tensor_gradient_relative_l2"],
                "gradient bound " + name,
            )
            metrics[name] = {
                "elements": ref.size,
                "max_normalized_error": absolute,
                "relative_l2_error": relative,
                "fd_stability_relative_l2": stability[name],
                "reference_l2_norm": float(np.linalg.norm(ref)),
                "reference_gradient": ref.reshape(-1).tolist(),
            }
        results.append(
            {
                "depth": depth,
                "parameter_tensors": len(expected),
                "parameter_elements": sum(math.prod(x) for x in expected.values()),
                "forward_normalized_error": forward_error,
                "loss_absolute_error": abs(loss - row["loss"]),
                "gradients": metrics,
            }
        )
    return {
        "pass": True,
        "actual_cuda": gpu,
        "scope": "two tiny composed Mamba1 forward/all-parameter derivatives against independent binary64 equations and central differences; no quality, speed or full residency claim",
        "bounds": BOUNDS,
        "cases": results,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--fixture", required=True)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            replay(
                load(a.log), json.loads(Path(a.fixture).read_text()), a.require_cuda
            ),
            indent=2,
        )
    )
