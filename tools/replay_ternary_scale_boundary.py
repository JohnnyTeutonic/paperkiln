"""Strict prospective old/fixed real-device boundary replay; no model claims."""

import json
import math
import struct
from pathlib import Path

PREFIX = "TERNARY_BOUNDARY_JSON "
VALUES = [-0.25, 0.5, 1.0, -1.5, 0.75, -0.125]


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except (OverflowError, struct.error):
        raise ValueError("invalid float32 scale") from None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate raw field")
        result[key] = value
    return result


def replay(log, mode, cuda, returncode):
    if mode not in ("old", "fixed") or type(cuda) is not bool:
        raise ValueError("invalid gate mode")
    rows = [
        json.loads(s[len(PREFIX) :], object_pairs_hook=unique_object)
        for s in Path(log).read_text().splitlines()
        if s.startswith(PREFIX)
    ]
    if len(rows) != (2 if cuda else 1):
        raise ValueError("exact raw row inventory required")
    if rows[0] != {"kind": "fresh", "scale": 1.4545455, "zero_scale": 100000}:
        raise ValueError("fresh controls differ")
    if not cuda:
        if returncode != 0:
            raise ValueError("fresh controls failed")
        return {"verified": True, "actual_cuda": False, "mode": mode}
    row = rows[1]
    if set(row) != {
        "kind",
        "actual_cuda",
        "scale",
        "host_stale_after",
        "host_stale_before",
        "values",
    }:
        raise ValueError("deferred fields differ")
    if (
        row["kind"] != "deferred"
        or row["actual_cuda"] is not True
        or row["host_stale_before"] is not True
    ):
        raise ValueError("real CUDA required")
    if type(row["scale"]) not in (int, float) or not math.isfinite(row["scale"]):
        raise ValueError("nonfinite scale")
    v = row["values"]
    if (
        type(v) is not list
        or len(v) != 6
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
    ):
        raise ValueError("full finite restored tensor required")
    if mode == "old":
        if (
            returncode != 1
            or row["host_stale_after"] is not True
            or row["scale"] != 100000
            or v != [0] * 6
        ):
            raise ValueError("expected old boundary failure absent")
    else:
        if (
            returncode != 0
            or row["host_stale_after"] is not False
            or f32(row["scale"]) != f32(16 / 11)
            or v != VALUES
        ):
            raise ValueError("fixed device restoration differs")
    return {
        "verified": True,
        "actual_cuda": True,
        "mode": mode,
        "full_tensor_values_checked": 6,
        "expected_old_failure": mode == "old",
    }
