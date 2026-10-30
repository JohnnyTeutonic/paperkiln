"""Independent NumPy oracle for experimental BF16 attention products.

This is a surrogate backward: each GEMM rounds both operands to BF16 RN-even,
accumulates in FP32, and returns FP32; softmax and its derivative stay FP32.
It is NOT the mathematical derivative of the discontinuous rounding operator.
Dense equal-length MHA only, contiguous [batch, head, sequence, head_dim],
causal or unmasked, no dropout/bias/GQA/state. Production ops are untouched.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = 0x54414642  # little-endian BFAT
HEADER = struct.Struct("<8I")
ARRAY_NAMES = ("scores", "prob", "dp", "ds", "y", "dq", "dk", "dv")


def bf16(x):
    """Round float32 to BF16 with ties to even; retain NaNs as quiet NaNs."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    with np.errstate(over="ignore"):
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(
            0xFFFF0000
        )
    nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    rounded = np.where(
        nan, (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded
    )
    return np.asarray(rounded, dtype=np.uint32).view(np.float32)


def gemm(a, b, low_precision=True):
    # Independent float64 reduction of explicit BF16/FP32 operands, then FP32.
    a, b = (bf16(a), bf16(b)) if low_precision else (a, b)
    return (np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)).astype(
        np.float32
    )


def probabilities(scores, causal):
    scores = np.asarray(scores, dtype=np.float64)
    if causal:
        scores = np.where(
            np.triu(np.ones(scores.shape[-2:], dtype=bool), 1), -np.inf, scores
        )
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def softmax_backward(prob, dp):
    p = np.asarray(prob, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)
    return (p * (dp - (p * dp).sum(axis=-1, keepdims=True))).astype(np.float32)


def reference(q, k, v, dy, causal=True, low_precision=True):
    q, k, v, dy = [np.asarray(x, dtype=np.float32) for x in (q, k, v, dy)]
    if len({x.shape for x in (q, k, v, dy)}) != 1 or q.ndim != 4:
        raise ValueError("q/k/v/dy must have the same [B,H,S,D] shape")
    if any(not np.isfinite(x).all() for x in (q, k, v, dy)):
        raise ValueError("finite inputs required")
    scale = np.float32(1 / np.sqrt(q.shape[-1]))
    scores = gemm(q, k.swapaxes(-1, -2), low_precision) * scale
    prob = probabilities(scores, causal)
    y = gemm(prob, v, low_precision)
    dp = gemm(dy, v.swapaxes(-1, -2), low_precision)
    ds = softmax_backward(prob, dp)
    dq = gemm(ds, k, low_precision) * scale
    dk = gemm(ds.swapaxes(-1, -2), q, low_precision) * scale
    dv = gemm(prob.swapaxes(-1, -2), dy, low_precision)
    return dict(zip(ARRAY_NAMES, (scores, prob, dp, ds, y, dq, dk, dv)))


def staged_oracle(q, k, v, dy, actual, causal, low_precision):
    """Check every product using its ACTUAL input, isolating rounding midpoints."""
    scale = np.float32(1 / np.sqrt(q.shape[-1]))
    return {
        "scores": gemm(q, k.swapaxes(-1, -2), low_precision) * scale,
        "prob": probabilities(actual["scores"], causal),
        "dp": gemm(dy, v.swapaxes(-1, -2), low_precision),
        "ds": softmax_backward(actual["prob"], actual["dp"]),
        "y": gemm(actual["prob"], v, low_precision),
        "dq": gemm(actual["ds"], k, low_precision) * scale,
        "dk": gemm(actual["ds"].swapaxes(-1, -2), q, low_precision) * scale,
        "dv": gemm(actual["prob"].swapaxes(-1, -2), dy, low_precision),
    }


def validate_shape(shape):
    if len(shape) != 4 or any(
        not isinstance(x, (int, np.integer)) or x < 1 for x in shape
    ):
        raise ValueError("four positive dimensions required")
    b, h, s, d = map(int, shape)
    if b * h > 256 or s > 2048 or d > 256 or d % 8 or b * h * s * s > 16_777_216:
        raise ValueError("unsupported shape/workspace size")


def encode_input(q, k, v, dy, causal, low_precision):
    validate_shape(q.shape)
    if type(causal) is not bool or type(low_precision) is not bool:
        raise ValueError("mask/precision must be boolean")
    if len({x.shape for x in (q, k, v, dy)}) != 1:
        raise ValueError("input shapes differ")
    if any(not np.isfinite(x).all() for x in (q, k, v, dy)):
        raise ValueError("finite inputs required")
    return HEADER.pack(MAGIC, 1, *q.shape, int(causal), int(low_precision)) + b"".join(
        np.asarray(x, dtype="<f4").tobytes() for x in (q, k, v, dy)
    )


def write_input(path, q, k, v, dy, causal, low_precision):
    data = encode_input(q, k, v, dy, causal, low_precision)
    with Path(path).open("xb") as f:
        f.write(data)


def read_output(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("short header")
    magic, version, b, h, s, d, causal, low = HEADER.unpack_from(data)
    if (magic, version) != (MAGIC, 1) or causal not in (0, 1) or low not in (0, 1):
        raise ValueError("invalid binary header")
    validate_shape((b, h, s, d))
    n, p = b * h * s * d, b * h * s * s
    if len(data) != HEADER.size + 4 * (4 * n + 4 * p):
        raise ValueError("wrong binary output length")
    arrays, offset = {}, HEADER.size
    for name in ARRAY_NAMES:
        shape = (b, h, s, s) if name in ARRAY_NAMES[:4] else (b, h, s, d)
        size = int(np.prod(shape))
        arrays[name] = (
            np.frombuffer(data, dtype="<f4", count=size, offset=offset)
            .reshape(shape)
            .copy()
        )
        offset += size * 4
    return arrays
