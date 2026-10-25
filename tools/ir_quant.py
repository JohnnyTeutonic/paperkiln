"""The IR's dequantisation reference (spec/IR_SPEC.md section 4.7): the f32 value of every element of a
stored GGUF tensor, per storage type, as ggml's scalar dequantize_row_* compute it (llama.cpp b6000,
commit 4762ad7316dcdec20016ab5985fb46a27902204d, ggml/src/ggml-quants.c; block layouts in
ggml/src/ggml-common.h).

    dequantize(type, raw: bytes, n: int) -> float32 array [n]     type: a name below or its ggml id
    TYPES                                  {name: (ggml id, elements per block, bytes per block)}

Every value is EXACT: each block's products are exact in f32 (an fp16 scale has 11 significant bits,
the integer factors at most 7 + 5 or 6 + 5 more), so at most one rounding happens (the - m of Q5_1,
Q4_K and Q5_K) and a fused multiply-add gives the same bits. numpy only.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32
# name: (ggml type id, elements per block, bytes per block)
TYPES = {"F32": (0, 1, 4), "F16": (1, 1, 2), "Q4_0": (2, 32, 18), "Q5_0": (6, 32, 22), "Q5_1": (7, 32, 24),
         "Q8_0": (8, 32, 34), "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210),
         "BF16": (30, 1, 2)}
BY_ID = {v[0]: k for k, v in TYPES.items()}


def _half(b):
    """fp16 fields (little-endian) as f32: exact."""
    return np.frombuffer(np.ascontiguousarray(b).tobytes(), dtype="<f2").astype(F32)


def _blocks(raw, n, name):
    _, qk, size = TYPES[name]
    if n % qk:
        raise ValueError(f"{name}: {n} elements is not a multiple of the block size {qk}")
    nb = n // qk
    if len(raw) != nb * size:
        raise ValueError(f"{name}: {len(raw)} bytes for {nb} blocks of {size}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(nb, size)


def _scale_min_k4(scales):
    """get_scale_min_k4 for j = 0..7 over [nb, 12] bytes: (sc, m) [nb, 8], 6-bit integers."""
    q = scales.astype(np.int32)
    sc = np.zeros((q.shape[0], 8), np.int32)
    m = np.zeros_like(sc)
    for j in range(8):
        if j < 4:
            sc[:, j], m[:, j] = q[:, j] & 63, q[:, j + 4] & 63
        else:
            sc[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
            m[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return sc, m


def dequantize(kind, raw, n):
    name = BY_ID[kind] if isinstance(kind, int) else kind.upper()
    if name not in TYPES:
        raise ValueError(f"no dequantisation reference for storage type {kind}")
    raw = bytes(raw)
    if name == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n).astype(F32)
    if name == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n).astype(F32)
    if name == "BF16":
        u = np.frombuffer(raw, dtype="<u2", count=n).astype(np.uint32) << 16
        return u.view(F32).copy()
    b = _blocks(raw, n, name)
    nb = b.shape[0]
    if name == "Q8_0":  # y = qs[j] * d
        d = _half(b[:, 0:2])[:, None]
        q = b[:, 2:34].view(np.int8).astype(F32)
        return (q * d).reshape(-1)
    if name == "Q4_0":  # y[j] = ((qs[j] & 15) - 8) d, y[j + 16] = ((qs[j] >> 4) - 8) d
        d = _half(b[:, 0:2])[:, None]
        qs = b[:, 2:18].astype(np.int32)
        q = np.concatenate([(qs & 0xF) - 8, (qs >> 4) - 8], axis=1).astype(F32)
        return (q * d).reshape(-1)
    if name in ("Q5_0", "Q5_1"):  # the fifth bit of value j is bit j of qh (bits 16.. for the high nibbles)
        d = _half(b[:, 0:2])[:, None]
        at = 2 if name == "Q5_0" else 4
        qh = b[:, at:at + 4].copy().view("<u4").astype(np.int64)  # [nb, 1]
        qs = b[:, at + 4:at + 20].astype(np.int64)
        j = np.arange(16)
        h0 = ((qh >> j) << 4) & 0x10
        h1 = (qh >> (j + 12)) & 0x10
        q = np.concatenate([(qs & 0xF) | h0, (qs >> 4) | h1], axis=1)
        if name == "Q5_0":
            return ((q - 16).astype(F32) * d).reshape(-1)
        m = _half(b[:, 2:4])[:, None]
        return ((q.astype(F32) * d) + m).reshape(-1)  # the product exact, then one rounding
    if name in ("Q4_K", "Q5_K"):
        d = _half(b[:, 0:2])[:, None]
        dmin = _half(b[:, 2:4])[:, None]
        sc, m = _scale_min_k4(b[:, 4:16])
        d1 = d * sc.astype(F32)  # [nb, 8], one per 32 values: exact
        m1 = dmin * m.astype(F32)
        if name == "Q4_K":
            qs = b[:, 16:144].reshape(nb, 4, 32).astype(np.int32)
            q = np.stack([qs & 0xF, qs >> 4], axis=2)  # [nb, 4 chunks of 64, low then high nibbles, 32]
        else:
            qh = b[:, 16:48].astype(np.int32)  # [nb, 32]
            qs = b[:, 48:176].reshape(nb, 4, 32).astype(np.int32)
            lo, hi = [], []
            for c in range(4):  # u1 = 1 << 2c, u2 = 2 << 2c
                lo.append((qs[:, c] & 0xF) + np.where(qh & (1 << (2 * c)), 16, 0))
                hi.append((qs[:, c] >> 4) + np.where(qh & (2 << (2 * c)), 16, 0))
            q = np.stack([np.stack(lo, axis=1), np.stack(hi, axis=1)], axis=2)
        q = q.reshape(nb, 8, 32).astype(F32)  # sub-block s = 2 * chunk + (0 low, 1 high)
        return ((d1[:, :, None] * q) - m1[:, :, None]).reshape(-1)
    if name == "Q6_K":  # y = d * sc * (q - 32), (d * sc) first
        ql = b[:, 0:128].astype(np.int32)
        qh = b[:, 128:192].astype(np.int32)
        sc = b[:, 192:208].view(np.int8).astype(F32)
        d = _half(b[:, 208:210])[:, None]
        out = np.zeros((nb, 256), F32)
        for half in range(2):  # n = 0, 128
            L, H, S = ql[:, 64 * half:64 * half + 64], qh[:, 32 * half:32 * half + 32], sc[:, 8 * half:8 * half + 8]
            for li in range(32):
                is_ = li // 16
                q1 = ((L[:, li] & 0xF) | (((H[:, li] >> 0) & 3) << 4)) - 32
                q2 = ((L[:, li + 32] & 0xF) | (((H[:, li] >> 2) & 3) << 4)) - 32
                q3 = ((L[:, li] >> 4) | (((H[:, li] >> 4) & 3) << 4)) - 32
                q4 = ((L[:, li + 32] >> 4) | (((H[:, li] >> 6) & 3) << 4)) - 32
                base = 128 * half + li
                for off, qv, s in ((0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)):
                    out[:, base + off] = (d[:, 0] * S[:, is_ + s]) * qv.astype(F32)
        return out.reshape(-1)
    raise AssertionError(name)
