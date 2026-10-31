"""Independent fixed-fixture full-array NumPy replay; no production implementation import."""

import argparse
import json
import math
from pathlib import Path
import numpy as np

OPS = [
    "exp",
    "div",
    "where",
    "rowsum",
    "rowmax",
    "broadcast_cols",
    "relu_",
    "sigmoid_",
    "scale_",
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fixture(v, shape):
    return np.array(v, dtype=np.float32).astype(np.float64).reshape(shape)


def matrix(raw, shape):
    need(
        set(raw) == {"shape", "values"} and raw["shape"] == list(shape),
        "exact matrix schema/shape",
    )
    a = np.asarray(raw["values"], dtype=np.float64)
    need(
        a.shape == (int(np.prod(shape)),) and np.isfinite(a).all(),
        "full finite matrix inventory",
    )
    return a.reshape(shape)


def replay(doc, require_cuda=False):
    gpu = doc.get("actual_cuda")
    need(type(gpu) is bool and (not require_cuda or gpu), "required backend")
    need(
        doc.get("pass") is True and len(doc.get("cases", [])) == len(OPS), "completion"
    )
    worst = 0.0
    records = []
    for op, row in zip(OPS, doc["cases"]):
        need(
            row.get("op") == op and row.get("actual_cuda") is gpu,
            "ordered unique cases/backend",
        )
        a = (
            fixture([0.2, 0.8], (2, 1))
            if op == "broadcast_cols"
            else fixture([0.2, -0.6, 1.1, 0.8, -1.3, 0.4], (2, 3))
        )
        inputs = {"a": a}
        grads = {}
        shape = (2, 1) if op in ["rowsum", "rowmax"] else (2, 3)
        u = (
            fixture([-0.4, 0.7], shape)
            if shape == (2, 1)
            else fixture([-0.4, 0.7, 1.2, -0.9, 0.3, 0.8], shape)
        )
        inputs["upstream_weights"] = u
        dy = u / u.size
        if op in ["div", "where"]:
            b = fixture([0.7, 1.2, 0.9, 1.4, 0.8, 1.1], (2, 3))
            inputs["b"] = b
        if op == "exp":
            y = np.exp(a)
            grads = {"a": dy * y}
        elif op == "div":
            y = a / b
            grads = {"a": dy / b, "b": -dy * a / (b * b)}
        elif op == "where":
            c = fixture([-1, 2, -3, 4, -5, 6], (2, 3))
            inputs["c"] = c
            y = np.where(c > 0, a, b)
            grads = {"a": dy * (c > 0), "b": dy * (c <= 0)}
        elif op == "rowsum":
            y = a.sum(axis=1, keepdims=True)
            grads = {"a": np.broadcast_to(dy, a.shape)}
        elif op == "rowmax":
            y = a.max(axis=1, keepdims=True)
            da = np.zeros_like(a)
            da[np.arange(2), a.argmax(axis=1)] = dy[:, 0]
            grads = {"a": da}
        elif op == "broadcast_cols":
            y = np.broadcast_to(a, (2, 3))
            grads = {"a": dy.sum(axis=1, keepdims=True)}
        else:
            inputs["w"] = np.eye(3)
            if op == "relu_":
                y = np.maximum(a, 0)
                dx = dy * (a > 0)
            elif op == "sigmoid_":
                y = 1 / (1 + np.exp(-a))
                dx = dy * y * (1 - y)
            else:
                y = a * float(np.float32(1.7))
                dx = dy * float(np.float32(1.7))
            grads = {"a": dx, "w": a.T @ dx}
        need(set(row["inputs"]) == set(inputs), "complete input inventory")
        for n, v in inputs.items():
            need(
                np.array_equal(matrix(row["inputs"][n], v.shape), v),
                "fixed fixture " + n,
            )
        need(set(row["gradients"]) == set(grads), "exact gradient inventory")
        arrays = {"output": (matrix(row["output"], y.shape), y)}
        for n, v in grads.items():
            arrays["gradient_" + n] = (matrix(row["gradients"][n], v.shape), v)
        for n, (actual, ref) in arrays.items():
            if n.startswith("gradient_") and np.linalg.norm(ref) > 0:
                need(np.linalg.norm(actual) > 0, "nonzero expected gradient tensor")
            error = float(np.max(abs(actual - ref) / (1 + abs(ref))))
            need(
                math.isfinite(error) and error <= 3e-5,
                "fixed normalized bound " + op + " " + n,
            )
            worst = max(worst, error)
        expected_stale = (
            2
            if op in ["div", "relu_", "sigmoid_", "scale_"]
            else 3
            if op == "where"
            else 1
        )
        need(
            type(row["stale_inputs_poisoned"]) is int
            and row["stale_inputs_poisoned"] == (expected_stale if gpu else 0),
            "stale input proof",
        )
        need(row["stale_gradient_poisoned"] is gpu, "stale gradient proof")
        records.append(
            {"op": op, "gradient_names": sorted(grads), "output_shape": list(y.shape)}
        )
    return {
        "pass": True,
        "actual_cuda": gpu,
        "cases": records,
        "maximum_normalized_error": worst,
        "bound": 3e-5,
        "scope": "nine named helper cases only; no expected-old-failure, speed or broad audit closure",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    print(json.dumps(replay(json.loads(a.raw.read_text()), a.require_cuda)))
