"""Independent NumPy full-network forward/backward for a registered plain Flex subset.

No production model, IR interpreter or framework autograd is imported. The BF16
backward is the declared six-GEMM surrogate, not a derivative of bit rounding.
"""

import argparse
import json
from pathlib import Path
import numpy as np

SHAPES = [(1, 12, "fp32"), (1, 12, "bf16"), (3, 20, "fp32"), (3, 20, "bf16")]
IDS = [1, 2, 3, 4, 6, 5, 2, 1]
TARGETS = [2, 3, 4, 1, 5, 2, 1, 6]
ELEMENT_BOUND = 3e-4
TENSOR_RELATIVE_BOUND = 0.03
NORM_FLOOR = 1e-10


def need(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    return np.asarray(value, dtype=np.float32)


def bf16(value):
    bits = f32(value).view(np.uint32)
    high, low = bits >> 16, bits & 65535
    increment = (low > 32768) | ((low == 32768) & ((high & 1) != 0))
    return ((high + increment.astype(np.uint32)) << 16).view(np.float32)


def product(a, b, mixed=False):
    if mixed:
        a, b = bf16(a), bf16(b)
    return f32(a.astype(np.float64) @ b.astype(np.float64))


class Value:
    def __init__(self, data, parents=(), pullback=None):
        self.data = f32(data)
        self.parents = parents
        self.pullback = pullback
        self.grad = np.zeros_like(self.data)

    def add_grad(self, value):
        self.grad = f32(self.grad + value)


def add(a, b):
    def back(g):
        a.add_grad(g)
        b.add_grad(g)

    return Value(f32(a.data + b.data), (a, b), back)


def linear(a, w, b=None):
    y = product(a.data, w.data)
    if b is not None:
        y = f32(y + b.data)

    def back(g):
        a.add_grad(product(g, w.data.T))
        w.add_grad(product(a.data.T, g))
        if b is not None:
            b.add_grad(f32(g.astype(np.float64).sum(0, keepdims=True)))

    return Value(y, (a, w) if b is None else (a, w, b), back)


def norm(a, gain, bias):
    x = a.data.astype(np.float64)
    centered = x - x.mean(1, keepdims=True)
    inv = 1 / np.sqrt((centered**2).mean(1, keepdims=True) + 1e-5)
    z = f32(centered * inv)
    y = f32(f32(z * gain.data) + bias.data)

    def back(g):
        gg = g.astype(np.float64)
        zz = z.astype(np.float64)
        scaled = gg * gain.data.astype(np.float64)
        dx = inv * (
            scaled
            - scaled.mean(1, keepdims=True)
            - zz * (scaled * zz).mean(1, keepdims=True)
        )
        a.add_grad(f32(dx))
        gain.add_grad(f32((gg * zz).sum(0, keepdims=True)))
        bias.add_grad(f32(gg.sum(0, keepdims=True)))

    return Value(y, (a, gain, bias), back)


def gelu(a):
    x = a.data.astype(np.float64)
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x**3))
    y = f32(0.5 * x * (1 + t))
    slope = 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * c * (1 + 3 * 0.044715 * x**2)
    return Value(y, (a,), lambda g: a.add_grad(f32(g * slope)))


def attention(qkv, mixed):
    rows, d = qkv.data.shape[0], 8
    sl, heads, width = 4, 2, 4
    visible = (np.arange(rows)[:, None] // sl == np.arange(rows)[None, :] // sl) & (
        np.arange(rows)[None, :] <= np.arange(rows)[:, None]
    )
    cache, outputs = [], []
    scale = np.float32(1 / np.sqrt(width))
    for h in range(heads):
        q = qkv.data[:, h * width : (h + 1) * width]
        k = qkv.data[:, d + h * width : d + (h + 1) * width]
        v = qkv.data[:, 2 * d + h * width : 2 * d + (h + 1) * width]
        scores = f32(product(q, k.T, mixed) * scale).astype(np.float64)
        scores = np.where(visible, scores, -np.inf)
        e = np.exp(scores - scores.max(1, keepdims=True))
        p = f32(e / e.sum(1, keepdims=True))
        outputs.append(product(p, v, mixed))
        cache.append((q, k, v, p))

    def back(g):
        dqkv = np.zeros_like(qkv.data)
        for h, (q, k, v, p) in enumerate(cache):
            upstream = g[:, h * width : (h + 1) * width]
            dv = product(p.T, upstream, mixed)
            dp = product(upstream, v.T, mixed).astype(np.float64)
            pp = p.astype(np.float64)
            ds = f32(scale * pp * (dp - (dp * pp).sum(1, keepdims=True)))
            ds = np.where(visible, ds, np.float32(0))
            dqkv[:, h * width : (h + 1) * width] = product(ds, k, mixed)
            dqkv[:, d + h * width : d + (h + 1) * width] = product(ds.T, q, mixed)
            dqkv[:, 2 * d + h * width : 2 * d + (h + 1) * width] = dv
        qkv.add_grad(dqkv)

    return Value(np.concatenate(outputs, axis=1), (qkv,), back)


def gather(table, indices):
    def back(g):
        full = np.zeros_like(table.data)
        np.add.at(full, indices, g)
        table.add_grad(full)

    return Value(table.data[indices], (table,), back)


def cross_entropy(logits):
    z = logits.data.astype(np.float64)
    shifted = z - z.max(1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(1, keepdims=True))
    p = np.exp(shifted - logsum)
    loss = -(shifted - logsum)[np.arange(len(TARGETS)), TARGETS].mean()

    def back(g):
        dx = p.copy()
        dx[np.arange(len(TARGETS)), TARGETS] -= 1
        logits.add_grad(f32(dx / len(TARGETS) * float(g[0, 0])))

    return Value([[loss]], (logits,), back)


def reverse(loss):
    seen, order = set(), []

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        for parent in value.parents:
            visit(parent)
        order.append(value)

    visit(loss)
    loss.grad = np.ones_like(loss.data)
    for value in reversed(order):
        if value.pullback:
            value.pullback(value.grad)


def expected_shapes(layers, hidden):
    result = {
        "wte.weight": (7, 8),
        "wpe.weight": (4, 8),
        "ln_f.weight": (1, 8),
        "ln_f.bias": (1, 8),
        "head.weight": (8, 7),
    }
    for block in range(layers):
        prefix = f"layers.{block}."
        for name in ("ln_1", "ln_2"):
            for kind in ("weight", "bias"):
                result[prefix + name + "." + kind] = (1, 8)
        for name, shape in [
            ("attn.c_attn", (8, 24)),
            ("attn.c_proj", (8, 8)),
            ("mlp.c_fc", (8, hidden)),
            ("mlp.c_proj", (hidden, 8)),
        ]:
            result[prefix + name + ".weight"] = shape
            result[prefix + name + ".bias"] = (1, shape[1])
    return result


def fixed_parameter(name, shape):
    h = 2166136261
    for c in name.encode("ascii"):
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    buckets = (
        ((h + np.arange(np.prod(shape), dtype=np.uint64) * 17) & 0xFFFFFFFF) % 37
    ).astype(np.int64) - 18
    gain = "ln_" in name and name.endswith(".weight")
    return f32((1 + buckets / 512) if gain else buckets / 64).reshape(shape)


def full_model(parameters, layers, mixed):
    p = {name: Value(value) for name, value in parameters.items()}
    x = add(
        gather(p["wte.weight"], IDS), gather(p["wpe.weight"], np.arange(len(IDS)) % 4)
    )

    def ln(a, name):
        return norm(a, p[name + ".weight"], p[name + ".bias"])

    def fc(a, name):
        return linear(a, p[name + ".weight"], p.get(name + ".bias"))

    for block in range(layers):
        prefix = f"layers.{block}."
        x = add(
            x,
            fc(
                attention(fc(ln(x, prefix + "ln_1"), prefix + "attn.c_attn"), mixed),
                prefix + "attn.c_proj",
            ),
        )
        x = add(
            x,
            fc(
                gelu(fc(ln(x, prefix + "ln_2"), prefix + "mlp.c_fc")),
                prefix + "mlp.c_proj",
            ),
        )
    logits = fc(ln(x, "ln_f"), "head")
    loss = cross_entropy(logits)
    reverse(loss)
    return logits.data, loss.data, {name: value.grad for name, value in p.items()}


def matrix(record, shape):
    need(
        set(record) == {"shape", "values"}
        and isinstance(record["shape"], list)
        and all(type(x) is int for x in record["shape"])
        and record["shape"] == list(shape),
        "exact tensor schema/shape",
    )
    value = np.asarray(record["values"], dtype=np.float64)
    need(
        value.shape == (int(np.prod(shape)),) and np.isfinite(value).all(),
        "full finite tensor inventory",
    )
    return f32(value.reshape(shape))


def element_error(actual, reference):
    return float(
        np.max(np.abs(actual.astype(np.float64) - reference) / (1 + np.abs(reference)))
    )


def replay(doc, require_cuda=False):
    need(
        doc.get("pass") is True and doc.get("actual_cuda") is require_cuda,
        "backend identity",
    )
    need(len(doc.get("cases", [])) == len(SHAPES), "complete fixed case inventory")
    results = []
    for row, (layers, hidden, precision) in zip(doc["cases"], SHAPES):
        need(
            all(
                type(row.get(k)) is int
                for k in ["layers", "d_ff", "d", "heads", "vocab", "seq_len"]
            ),
            "integer geometry required",
        )
        need(
            (row.get("layers"), row.get("d_ff"), row.get("precision"))
            == (layers, hidden, precision),
            "ordered registered case",
        )
        need(
            (row.get("d"), row.get("heads"), row.get("vocab"), row.get("seq_len"))
            == (8, 2, 7, 4),
            "fixed model geometry",
        )
        need(
            row.get("ids") == IDS
            and row.get("targets") == TARGETS
            and all(type(x) is int for x in row["ids"] + row["targets"]),
            "fixed token fixture",
        )
        proof = row["default_proof"]
        need(
            set(proof)
            == {
                "legacy_fp32_logits_and_all_gradients_bit_equal",
                "IR_precision_independent",
                "reload_fp32_bit_equal",
            }
            and all(v is True for v in proof.values()),
            "default and inference controls",
        )
        shapes = expected_shapes(layers, hidden)
        need(
            set(row["parameters"]) == set(row["gradients"]) == set(shapes),
            "full parameter/gradient inventory",
        )
        parameters = {}
        for name, shape in shapes.items():
            value = matrix(row["parameters"][name], shape)
            need(
                np.array_equal(value, fixed_parameter(name, shape)),
                "source-generated parameter fixture: " + name,
            )
            parameters[name] = value
        logits, loss, gradients = full_model(parameters, layers, precision == "bf16")
        forward_error = element_error(matrix(row["logits"], (8, 7)), logits)
        loss_error = element_error(matrix(row["loss"], (1, 1)), loss)
        need(
            max(forward_error, loss_error) <= ELEMENT_BOUND,
            "independent forward/loss bound",
        )
        errors, relative = {}, {}
        for name, shape in shapes.items():
            actual = matrix(row["gradients"][name], shape)
            reference = gradients[name]
            error = element_error(actual, reference)
            refnorm = float(np.linalg.norm(reference.astype(np.float64)))
            normerror = float(
                np.linalg.norm(actual.astype(np.float64) - reference)
            ) / max(refnorm, NORM_FLOOR)
            need(
                refnorm == 0 or np.linalg.norm(actual) > 0,
                "zeroed gradient tensor: " + name,
            )
            need(
                error <= ELEMENT_BOUND and normerror <= TENSOR_RELATIVE_BOUND,
                "independent full gradient bound: " + name,
            )
            errors[name], relative[name] = error, normerror
        if require_cuda:
            wanted = layers * 2 * 6 if precision == "bf16" else 0
            need(
                type(row["profile"].get("gemm_bf16", 0)) is int
                and row["profile"].get("gemm_bf16", 0) == wanted,
                "six GEMMs per layer/head precision proof",
            )
        results.append(
            {
                "layers": layers,
                "d_ff": hidden,
                "precision": precision,
                "full_gradient_tensors": len(shapes),
                "forward_error": forward_error,
                "loss_error": loss_error,
                "maximum_gradient_element_error": max(errors.values()),
                "maximum_gradient_relative_norm_error": max(relative.values()),
                "gradient_relative_norm_errors": relative,
            }
        )
    return {
        "pass": True,
        "actual_cuda": require_cuda,
        "bounds": {
            "normalized_element": ELEMENT_BOUND,
            "tensor_relative_norm": TENSOR_RELATIVE_BOUND,
            "norm_floor": NORM_FLOOR,
        },
        "cases": results,
        "scope": "canonical tiny plain Flex model only; BF16 gradients use explicit six-GEMM surrogate; no quality/speed/adoption claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(replay(json.loads(args.raw.read_text()), args.require_cuda)))
