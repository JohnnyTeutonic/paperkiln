"""Independent fixed-fixture embedding gather/scatter replay; assertions are not used."""

import argparse
import json
import math
from pathlib import Path

CASES = (
    "deferred_fallback",
    "deferred_device",
    "write_through_fallback",
    "leaf_fallback",
    "empty_ids",
    "negative_id",
    "large_id",
)
INPUT = [(i - 6) / 8 for i in range(12)]
UPSTREAM = [(i + 1) / 16 for i in range(9)]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def values(actual, expected, name, bound=2e-5):
    need(isinstance(actual, list) and len(actual) == len(expected), name + " inventory")
    need(
        all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in actual
        ),
        name + " finite",
    )
    return max((abs(float(a) - e) for a, e in zip(actual, expected)), default=0)


def replay(rows, gpu=False, selected="all", expect_failure=False):
    if expect_failure:
        need(
            gpu and selected == "deferred_fallback",
            "old failure mode requires exact deferred fallback",
        )
    wanted = list(CASES) if selected == "all" else [selected]
    need(all(n in CASES for n in wanted), "unknown selected case")
    need(
        len(rows) == len(wanted) + 1 and [r.get("case") for r in rows[:-1]] == wanted,
        "complete ordered case inventory",
    )
    summary = rows[-1]
    need(
        summary.get("summary") is True and summary.get("cases") == len(wanted),
        "summary inventory",
    )
    need(summary.get("actual_cuda") is gpu, "summary backend")
    if gpu:
        need(summary.get("devcheck_enabled") is True, "missing DEVCHECK")
    worst = 0
    for r, name in zip(rows[:-1], wanted):
        computed = name != "leaf_fallback"
        deferred = name != "write_through_fallback"
        stale = gpu and computed and deferred
        need(
            r.get("actual_cuda") is gpu
            and r.get("computed_table") is computed
            and r.get("deferred") is deferred,
            "case mode",
        )
        need(
            r.get("stale_operand_proved") is stale and r.get("host_poison") is stale,
            "stale proof",
        )
        need(
            r.get("ops_enabled") is (gpu and name == "deferred_device"), "fallback mode"
        )
        need(
            r.get("input_shape") == [4, 3]
            and values(r.get("input"), INPUT, "input") == 0,
            "fixed input fixture",
        )
        if name == "empty_ids":
            need(
                r.get("ids") == []
                and r.get("empty_rejected") is True
                and r.get("pass") is True,
                "empty refusal control",
            )
            continue
        if name in ("negative_id", "large_id"):
            need(
                r.get("ids") == [-1 if name == "negative_id" else 4]
                and r.get("bounds_rejected") is True,
                "bounds control",
            )
            need(r.get("pass") is True, "bounds pass")
            continue
        ids = [] if name == "empty_ids" else [2, 0, 2]
        need(
            r.get("ids") == ids and r.get("output_shape") == [len(ids), 3],
            "gather geometry",
        )
        need(values(r.get("upstream"), UPSTREAM, "upstream") == 0, "fixed upstream")
        expected = [INPUT[i * 3 + c] for i in ids for c in range(3)]
        grad = [0.0] * 12
        for row, i in enumerate(ids):
            for c in range(3):
                grad[i * 3 + c] += UPSTREAM[row * 3 + c] / 9
        out_error = values(r.get("output"), expected, "output")
        grad_error = values(r.get("input_gradient"), grad, "gradient")
        error = max(out_error, grad_error)
        worst = max(worst, error)
        need(
            isinstance(r.get("max_abs_error"), (int, float))
            and math.isfinite(r["max_abs_error"])
            and abs(error - r["max_abs_error"]) <= 1e-9,
            "recorded error",
        )
        if expect_failure:
            need(
                gpu
                and wanted == ["deferred_fallback"]
                and out_error > 0.01
                and grad_error <= 2e-5,
                "expected old numeric failure",
            )
            need(
                r.get("pass") is False and summary.get("pass") is False,
                "old failure status",
            )
        else:
            need(error <= 2e-5 and r.get("pass") is True, "numeric bound")
    if not expect_failure:
        need(summary.get("pass") is True, "summary pass")
    return {
        "pass": True,
        "actual_cuda": gpu,
        "cases": len(wanted),
        "independent_replay": True,
        "old_numeric_failure_verified": expect_failure,
        "worst_abs_error": worst,
        "bound": 2e-5,
    }


def load(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.startswith("{")
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--case", default="all", choices=("all",) + CASES)
    p.add_argument("--expect-numerical-failure", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            replay(load(a.log), a.require_cuda, a.case, a.expect_numerical_failure),
            indent=2,
        )
    )
