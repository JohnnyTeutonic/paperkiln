"""Replay raw explicit attention training-op arrays against an independent NumPy oracle."""

import argparse
import json
import math
from pathlib import Path
import numpy as np


def require(condition, message="attention replay contract failed"):
    if not condition:
        raise ValueError(message)


def quantized(a, bf):
    a = np.asarray(a, dtype=np.float32)
    if not bf:
        return a.astype(np.float64)
    u = a.view(np.uint32)
    hi = u >> 16
    lo = u & 65535
    rounded = (
        (hi + ((lo > 32768) | (lo == 32768) & (hi & 1 != 0)) << 16)
        .astype(np.uint32)
        .view(np.float32)
    )
    return rounded.astype(np.float64)


def product(a, b, ta, tb, bf):
    a = quantized(a, bf)
    b = quantized(b, bf)
    return ((a.T if ta else a) @ (b.T if tb else b)).astype(np.float32)


def oracle(row):
    q, k, v, g = [
        np.asarray(row[x], dtype=np.float32)
        for x in ("q", "k", "v", "upstream_before_mean")
    ]
    require(
        q.ndim == k.ndim == v.ndim == g.ndim == 2, "attention replay contract failed"
    )
    require(
        q.shape == k.shape and q.shape[0] == v.shape[0] and (v.shape == g.shape),
        "attention replay contract failed",
    )
    require(
        all((np.isfinite(x).all() for x in (q, k, v, g))),
        "attention replay contract failed",
    )
    n = q.shape[0]
    sl = row["seq"]
    require(
        type(sl) is int and sl > 0 and (n == sl * row["batch"]),
        "attention replay contract failed",
    )
    bf = row["bf16"]
    require(
        type(bf) is bool and type(row["causal"]) is bool,
        "attention replay contract failed",
    )
    scores = (product(q, k, False, True, bf) * np.float32(0.5)).astype(np.float32)
    p = np.zeros_like(scores)
    for i in range(n):
        lo = i // sl * sl
        hi = i + 1 if row["causal"] else lo + sl
        exp = np.exp(
            scores[i, lo:hi].astype(np.float64) - float(scores[i, lo:hi].max())
        )
        p[i, lo:hi] = (exp / exp.sum()).astype(np.float32)
    dy = (g / np.float32(g.size)).astype(np.float32)
    ds = product(dy, v, False, True, bf)
    ds = (
        0.5
        * p.astype(np.float64)
        * (
            ds.astype(np.float64)
            - (p.astype(np.float64) * ds).sum(axis=1, keepdims=True)
        )
    ).astype(np.float32)
    return {
        "y": product(p, v, False, False, bf),
        "dq": product(ds, k, False, False, bf),
        "dk": product(ds, q, True, False, bf),
        "dv": product(p, dy, True, False, bf),
    }


def fixture(rows, cols, seed):
    return np.array(
        [
            [
                0.37 * math.sin((i + 1) * (j + 3) + seed)
                + 0.21 * math.cos(i * 3 + j + seed)
                for j in range(cols)
            ]
            for i in range(rows)
        ],
        dtype=np.float32,
    )


def registered_shape(row, metric):
    shapes = {
        (1, 1): (3, 2),
        (7, 1): (5, 3),
        (4, 2): (8, 5),
        (17, 2): (9, 7),
        (3, 2): (5, 4),
    }
    require(type(row["seq"]) is int and type(row["batch"]) is int, "shape types differ")
    key = (row["seq"], row["batch"])
    require(key in shapes, "unregistered sequence/batch shape")
    d, dv = shapes[key]
    require(metric.get("d") == d and metric.get("dv") == dv, "registered widths differ")
    n = row["seq"] * row["batch"]
    for field, cols, seed in [
        ("q", d, 1),
        ("k", d, 5),
        ("v", dv, 9),
        ("upstream_before_mean", dv, 13),
    ]:
        actual = np.asarray(row[field], dtype=np.float64)
        require(
            actual.shape == (n, cols) and np.isfinite(actual).all(),
            "registered raw shape differs",
        )
        require(
            np.max(np.abs(actual - fixture(n, cols, seed).astype(np.float64))) < 1e-7,
            "raw input differs from registered fixture",
        )


def replay(log, require_cuda=False):
    rows = [
        json.loads(line)
        for line in Path(log).read_text().splitlines()
        if line.startswith("{")
    ]
    final = rows[-1]
    require(
        final
        == {
            "pass": True,
            "cases": 44,
            "actual_cuda": require_cuda,
            "devcheck": True,
            "default_fp32_bit_equal": True,
        },
        "attention replay contract failed",
    )
    raw = [r for r in rows if r.get("raw_case") is True]
    metrics = [r for r in rows if r.get("case") is True]
    require(len(raw) == len(metrics) == 44, "attention replay contract failed")
    keys = set()
    worst = 0
    for r, m in zip(raw, metrics):
        registered_shape(r, m)
        key = tuple((r[x] for x in ("seq", "batch", "causal", "bf16", "grad_mask")))
        require(key not in keys, "attention replay contract failed")
        keys.add(key)
        require(
            all(
                (m[x] == r[x] for x in ("seq", "batch", "causal", "bf16", "grad_mask"))
            ),
            "attention replay contract failed",
        )
        mask = r["grad_mask"]
        require(
            type(mask) is int and 0 <= mask <= 7, "attention replay contract failed"
        )
        expected = (
            2
            + bool(mask & 4)
            + (1 + bool(mask & 1) + bool(mask & 2) if mask & 3 else 0)
        )
        require(r["expected_gemms"] == expected, "attention replay contract failed")
        require(
            r["measured_device_gemms"] == (expected if require_cuda else 0),
            "attention replay contract failed",
        )
        bound = 0.005 if r["bf16"] else 3e-05
        require(
            type(m["worst"]) in (int, float)
            and math.isfinite(m["worst"])
            and (0 <= m["worst"] < bound),
            "attention replay contract failed",
        )
        refs = oracle(r)
        for name, bit in [("y", 0), ("dq", 1), ("dk", 2), ("dv", 4)]:
            if bit and (not mask & bit):
                require(name not in r, "attention replay contract failed")
                continue
            a = np.asarray(r[name], dtype=np.float64)
            b = refs[name].astype(np.float64)
            require(
                a.shape == b.shape and np.isfinite(a).all(),
                "attention replay contract failed",
            )
            error = float(np.max(np.abs(a - b) / (1 + np.abs(b))))
            require(error < bound, (key, name, error, bound))
            worst = max(worst, error)
    expected_keys = set()
    for bf in (False, True):
        for causal in (False, True):
            for sl, batch in ((1, 1), (7, 1), (4, 2), (17, 2)):
                expected_keys.add((sl, batch, causal, bf, 7))
            for mask in range(7):
                expected_keys.add((3, 2, causal, bf, mask))
    require(keys == expected_keys, "attention replay contract failed")
    return {
        "pass": True,
        "cases": 44,
        "actual_cuda": require_cuda,
        "independent_numpy_replay": True,
        "worst_normalized_error": worst,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log")
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    print(json.dumps(replay(a.log, a.require_cuda), indent=2))
