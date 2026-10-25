"""tools/ir_quant.py, the dequantisation reference (IR_SPEC.md section 4.7): bit for bit with gguf-py's
dequantize (when installed) on random blocks of every type, and EXACT: every value equals the f32 rounding
of the block's exact real value (computed in float64, where every product and sum here is exact)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ir_quant as q  # noqa: E402

QUANT = ["Q4_0", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]


def random_blocks(name, nb, rng):
    """Random block bytes whose fp16 fields are finite (normal, subnormal, signed, zero)."""
    _, qk, size = q.TYPES[name]
    b = rng.integers(0, 256, (nb, size), dtype=np.uint8)
    halves = {"Q4_0": [0], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}
    vals = np.concatenate([rng.standard_normal(nb * 2) * 0.05, [0.0, -0.0, 6e-8, -3e-5, 65504.0, 1e-4]])
    for at in halves[name]:
        h = rng.choice(vals, nb).astype("<f2")
        b[:, at:at + 2] = h.view(np.uint8).reshape(nb, 2)
    return b


@pytest.mark.parametrize("name", QUANT)
def test_matches_gguf_py(name):
    gq = pytest.importorskip("gguf")
    rng = np.random.default_rng(sum(map(ord, name)))
    b = random_blocks(name, 64, rng)
    n = 64 * q.TYPES[name][1]
    got = q.dequantize(name, b.tobytes(), n)
    want = gq.quants.dequantize(b.reshape(1, -1), getattr(gq.GGMLQuantizationType, name)).reshape(-1)
    assert got.dtype == np.float32 and got.shape == (n,)
    assert np.array_equal(got.view(np.uint32), want.astype(np.float32).view(np.uint32))


def _f64_reference(name, b):
    """The same formulas evaluated in float64 (each product and the final sum exact there)."""
    nb = b.shape[0]
    h = lambda x: np.frombuffer(np.ascontiguousarray(x).tobytes(), "<f2").astype(np.float64)  # noqa: E731
    if name == "Q8_0":
        return (b[:, 2:].view(np.int8).astype(np.float64) * h(b[:, :2])[:, None]).reshape(-1)
    if name == "Q4_0":
        qs = b[:, 2:].astype(np.int64)
        return (np.concatenate([(qs & 15) - 8, (qs >> 4) - 8], 1) * h(b[:, :2])[:, None]).reshape(-1)
    if name in ("Q5_0", "Q5_1"):
        at = 2 if name == "Q5_0" else 4
        qh = b[:, at:at + 4].copy().view("<u4").astype(np.int64)
        qs = b[:, at + 4:].astype(np.int64)
        j = np.arange(16)
        v = np.concatenate([(qs & 15) | (((qh >> j) << 4) & 16), (qs >> 4) | ((qh >> (j + 12)) & 16)], 1)
        d = h(b[:, :2])[:, None]
        return ((v - 16) * d if name == "Q5_0" else v * d + h(b[:, 2:4])[:, None]).reshape(-1)
    if name in ("Q4_K", "Q5_K"):
        d, dmin = h(b[:, :2])[:, None], h(b[:, 2:4])[:, None]
        sc, m = q._scale_min_k4(b[:, 4:16])
        if name == "Q4_K":
            qs = b[:, 16:].reshape(nb, 4, 32).astype(np.int64)
            qq = np.stack([qs & 15, qs >> 4], 2).reshape(nb, 8, 32)
        else:
            qh = b[:, 16:48].astype(np.int64)
            qs = b[:, 48:].reshape(nb, 4, 32).astype(np.int64)
            qq = np.stack([np.stack([(qs[:, c] & 15) + np.where(qh & (1 << 2 * c), 16, 0) for c in range(4)], 1),
                           np.stack([(qs[:, c] >> 4) + np.where(qh & (2 << 2 * c), 16, 0) for c in range(4)], 1)],
                          2).reshape(nb, 8, 32)
        return ((d * sc)[:, :, None] * qq - (dmin * m)[:, :, None]).reshape(-1)
    if name == "Q6_K":
        ql, qh = b[:, :128].astype(np.int64), b[:, 128:192].astype(np.int64)
        sc, d = b[:, 192:208].view(np.int8).astype(np.float64), h(b[:, 208:210])
        out = np.zeros((nb, 256))
        for half in range(2):
            L, H, S = ql[:, 64 * half:64 * half + 64], qh[:, 32 * half:32 * half + 32], sc[:, 8 * half:8 * half + 8]
            for li in range(32):
                qv = [((L[:, li] & 15) | ((H[:, li] & 3) << 4)) - 32, ((L[:, li + 32] & 15) | (((H[:, li] >> 2) & 3) << 4)) - 32,
                      ((L[:, li] >> 4) | (((H[:, li] >> 4) & 3) << 4)) - 32, ((L[:, li + 32] >> 4) | (((H[:, li] >> 6) & 3) << 4)) - 32]
                for k, off in enumerate((0, 32, 64, 96)):
                    out[:, 128 * half + li + off] = d * S[:, li // 16 + 2 * k] * qv[k]
        return out.reshape(-1)
    raise AssertionError(name)


@pytest.mark.parametrize("name", QUANT)
def test_every_value_is_the_rounded_exact_value(name):
    rng = np.random.default_rng(7 + sum(map(ord, name)))
    b = random_blocks(name, 64, rng)
    got = q.dequantize(name, b.tobytes(), 64 * q.TYPES[name][1])
    want = _f64_reference(name, b).astype(np.float32)  # one rounding of the exact value
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_plain_types_and_shape_errors():
    x = np.array([1.5, -0.0, 3.0e-41, 65504.0], np.float32)
    assert np.array_equal(q.dequantize("F32", x.tobytes(), 4).view(np.uint32), x.view(np.uint32))
    assert np.array_equal(q.dequantize("F16", x.astype("<f2").tobytes(), 4), x.astype("<f2").astype(np.float32))
    bf = (x.view(np.uint32) >> 16).astype("<u2")
    assert np.array_equal(q.dequantize(30, bf.tobytes(), 4).view(np.uint32), (bf.astype(np.uint32) << 16))
    with pytest.raises(ValueError):
        q.dequantize("Q8_0", bytes(34), 33)
    with pytest.raises(ValueError):
        q.dequantize("Q4_K", bytes(143), 256)
    with pytest.raises(ValueError):
        q.dequantize("IQ2_XS", bytes(4), 1)
