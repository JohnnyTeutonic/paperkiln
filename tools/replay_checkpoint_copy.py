"""Independent binary64 checkpoint copy reference; source-bound tiny cases only."""

import argparse
import json
import math
from pathlib import Path

BOUND = 1e-6


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fixture(r, c, seed):
    return [
        [((i * c + j) * 7 + seed) % 19 / 8 - 9 / 8 for j in range(c)] for i in range(r)
    ]


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def transpose(a):
    return list(map(list, zip(*a)))


def flat(a):
    return [v for row in a for v in row]


def matrix(doc, reference, name):
    need(
        isinstance(doc, dict) and set(doc) == {"shape", "values"}, name + " matrix keys"
    )
    need(
        doc["shape"] == [len(reference), len(reference[0])]
        and all(type(n) is int for n in doc["shape"]),
        name + " shape",
    )
    expected = flat(reference)
    need(
        isinstance(doc["values"], list) and len(doc["values"]) == len(expected),
        name + " values",
    )
    need(
        all(type(v) in (float, int) and math.isfinite(v) for v in doc["values"]),
        name + " finite",
    )
    error = max(abs(a - b) / (1 + abs(b)) for a, b in zip(doc["values"], expected))
    need(error <= BOUND, name + " numeric")
    return error


def replay(doc, variant, case, cuda=False):
    need(variant in ("old", "fixed") and case in ("output", "input"), "registered case")
    need(
        doc.get("variant") == variant
        and doc.get("case") == case
        and doc.get("actual_cuda") is cuda,
        "identity",
    )
    observations = doc.get("observations")
    need(isinstance(observations, list) and len(observations) >= 1, "observations")
    first = observations[0]
    need(first.get("grad_enabled") is False, "NoGrad forward")
    for row in observations:
        need(
            set(row)
            == {
                "grad_enabled",
                "input_stale",
                "output_stale_before_local_materialize",
                "output_stale_on_return",
            }
            and all(type(v) is bool for v in row.values()),
            "observation inventory",
        )
    need(
        doc.get("input_stale_before_checkpoint") is (cuda and case == "input"),
        "initial input provenance",
    )
    need(
        first["output_stale_before_local_materialize"] is cuda,
        "deferred output provenance",
    )
    need(
        first["output_stale_on_return"] is (cuda and case == "output"),
        "return provenance",
    )
    if cuda and variant == "old":
        need(doc.get("pass") is False, "old must reject")
        need(
            doc.get("error")
            == "DEVCHECK: host read of device-stale tensor at checkpoint "
            + case
            + " copy",
            "exact old guard",
        )
        need(len(observations) == 1, "old no recompute")
        if case == "input":
            need(
                doc.get("input_stale_before_backward") is True
                and first["input_stale"] is True,
                "input backward reached stale",
            )
        else:
            need(
                "input_stale_before_backward" not in doc,
                "output rejection precedes backward",
            )
        need("gradients" not in doc, "failed old has no gradients")
        return {"pass": True, "expected_old_guard": case, "actual_cuda": True}
    need(doc.get("pass") is True and "error" not in doc, "successful control")
    need(
        len(observations) == 2 and observations[1]["grad_enabled"] is True,
        "recomputation",
    )
    need(observations[1]["input_stale"] is False, "detached input host-authoritative")
    need(
        doc.get("input_stale_before_backward") is (cuda and case == "input"),
        "backward input provenance",
    )
    w = fixture(4, 2, 3)
    if case == "input":
        a, b = fixture(2, 3, 1), fixture(3, 4, 2)
        x = mm(a, b)
        parameters = {"a": a, "b": b, "w": w}
    else:
        x = fixture(2, 4, 4)
        parameters = {"x": x, "w": w}
    y = mm(x, w)
    dy = [[0.125, -0.0625], [0.1875, 0.3125]]
    dx = mm(dy, transpose(w))
    grads = {"w": mm(transpose(x), dy)}
    if case == "input":
        grads.update(a=mm(dx, transpose(b)), b=mm(transpose(a), dx))
    else:
        grads["x"] = dx
    need(
        set(doc.get("parameters", {})) == set(parameters)
        and set(doc.get("gradients", {})) == set(grads),
        "full parameter/gradient inventory",
    )
    errors = [
        matrix(doc["parameters"][n], v, "fixture " + n) for n, v in parameters.items()
    ]
    # All fixtures are exact binary fractions; require identity, not merely the output bound.
    need(max(errors) == 0, "fixture identity")
    errors = [
        matrix(doc["logits"], y, "forward"),
        matrix(
            doc["loss"],
            [[sum(a * b for a, b in zip(flat(y), [0.5, -0.25, 0.75, 1.25])) / 4]],
            "loss",
        ),
    ]
    errors += [
        matrix(doc["gradients"][n], g, "gradient " + n) for n, g in grads.items()
    ]
    profile = doc.get("profile")
    need(isinstance(profile, dict), "profile inventory")
    if cuda:
        need(
            any(
                type(v) is int and v > 0
                for k, v in profile.items()
                if k.startswith("gemm")
            ),
            "actual GEMM profile",
        )
    return {
        "pass": True,
        "actual_cuda": cuda,
        "case": case,
        "variant": variant,
        "gradient_tensors": len(grads),
        "bound": BOUND,
        "maximum_normalized_error": max(errors),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--variant", choices=["old", "fixed"], required=True)
    p.add_argument("--case", choices=["output", "input"], required=True)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            replay(json.loads(a.raw.read_text()), a.variant, a.case, a.require_cuda)
        )
    )
