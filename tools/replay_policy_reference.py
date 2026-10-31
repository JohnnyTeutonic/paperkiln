"""Independent binary64 public policy-KL loss/gradient reference, no framework imports."""

import argparse
import json
import math
import struct
from pathlib import Path

BOUND = 1e-6


def need(ok, message):
    if not ok:
        raise ValueError(message)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def values(doc, shape):
    need(isinstance(doc, dict) and set(doc) == {"shape", "values"}, "matrix inventory")
    need(
        doc["shape"] == shape and all(type(n) is int for n in doc["shape"]),
        "matrix shape",
    )
    v = doc["values"]
    need(isinstance(v, list) and len(v) == math.prod(shape), "matrix length")
    need(all(type(x) in (int, float) and math.isfinite(x) for x in v), "matrix finite")
    return v


def error(a, b):
    return max(abs(x - y) / (1 + abs(y)) for x, y in zip(a, b))


def reference(logits, ref, w, k):
    loss = 0.0
    gradient = []
    for row, target in enumerate([2, 0]):
        x = logits[3 * row : 3 * row + 3]
        mx = max(x)
        lz = mx + math.log(sum(math.exp(v - mx) for v in x))
        lp = [v - lz for v in x]
        kl = sum(math.exp(v) * (v - ref[3 * row + j]) for j, v in enumerate(lp))
        loss += -w[row] * f32(lp[target]) + k[row] * kl
        gradient += [
            f32(
                1.5
                * (
                    w[row] * (math.exp(f32(v)) - (j == target))
                    + k[row] * math.exp(f32(v)) * (f32(v) - ref[3 * row + j] - kl)
                )
            )
            for j, v in enumerate(lp)
        ]
    return [f32(loss)], gradient


def replay(d, variant, case, cuda=False):
    need(
        variant in ["old", "fixed"] and case in ["mixed", "policy_only", "kl_only"],
        "registered case",
    )
    need(
        d.get("variant") == variant
        and d.get("case") == case
        and d.get("actual_cuda") is cuda
        and d.get("pass") is True
        and "error" not in d,
        "identity/pass",
    )
    need(
        d.get("reference_stale_before") is cuda and d.get("poisoned") is cuda,
        "stale reference provenance",
    )
    need(
        d.get("reference_stale_after_op") is (cuda and variant == "old"),
        "consumer materialization provenance",
    )
    w = [0.75, -0.25] if case != "kl_only" else [0.0, 0.0]
    k = [0.5, 1.0] if case != "policy_only" else [0.0, 0.0]
    need(
        d.get("w") == w
        and d.get("k") == k
        and d.get("targets") == [2, 0]
        and d.get("upstream") == 1.5,
        "registered recipe",
    )
    logits = [-0.75, 0.125, 0.5, 0.25, -0.375, 1.0]
    ref = [f32(math.log(f32(x))) for x in [0.2, 0.3, 0.5, 0.6, 0.3, 0.1]]
    need(values(d["logits"], [2, 3]) == logits, "logit fixture")
    need(error(values(d["reference"], [2, 3]), ref) < 2e-7, "reference fixture")
    loss, gradient = reference(logits, ref, w, k)
    actual_loss = values(d["loss"], [1, 1])
    actual_gradient = values(d["gradient"], [2, 3])
    le, ge = error(actual_loss, loss), error(actual_gradient, gradient)
    profile = d.get("profile")
    need(isinstance(profile, dict), "profile")
    if cuda:
        need(
            sum(
                v
                for key, v in profile.items()
                if key.startswith("gemm") and type(v) is int
            )
            == 1
            and all(type(v) is int for v in profile.values()),
            "one actual FP32 reference GEMM",
        )
    expected_old = cuda and variant == "old" and case != "policy_only"
    if expected_old:
        wrong_loss, wrong_gradient = reference(logits, [9.0] * 6, w, k)
        need(
            le > 1e-3 and ge > 1e-3, "old must fail both correct loss and full gradient"
        )
        need(
            error(actual_loss, wrong_loss) <= BOUND
            and error(actual_gradient, wrong_gradient) <= BOUND,
            "old causal poisoned-reference prediction",
        )
    else:
        need(
            le <= BOUND and ge <= BOUND, "independent correct loss/full-gradient bound"
        )
    return {
        "pass": True,
        "actual_cuda": cuda,
        "variant": variant,
        "case": case,
        "expected_old_numeric_failure": expected_old,
        "correct_loss_error": le,
        "correct_gradient_error": ge,
        "bound": BOUND,
        "scope": "public constant-reference Matrix consumer only; no historical CLI invalidity claim",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--variant", choices=["old", "fixed"], required=True)
    p.add_argument("--case", choices=["mixed", "policy_only", "kl_only"], required=True)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            replay(json.loads(a.raw.read_text()), a.variant, a.case, a.require_cuda)
        )
    )
