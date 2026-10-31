"""Strict independent raw-array replay for the frozen transpose-load matrix."""

import argparse
import json
import math
from pathlib import Path
import numpy as np

SHAPES = (
    (1, 1, 0),
    (1, 1, 1),
    (2, 3, 7),
    (31, 33, 35),
    (32, 32, 32),
    (33, 31, 65),
    (64, 96, 64),
    (65, 67, 63),
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def replay(text):
    lines = iter(text.splitlines())
    header = next(lines, "")
    require(header.startswith("DEVICE ") and "L4" in header, "actual L4 required")
    seen = set()
    worst = 0.0
    for f in range(4):
        for m, n, k in SHAPES:
            for mode in range(3):
                fields = next(lines, "").split()
                require(len(fields) == 7 and fields[0] == "CASE", "case header")
                ident = tuple(map(int, fields[1:6]))
                require(ident == (f, m, n, k, mode), "registered case identity/order")
                reported = float(fields[6])
                require(
                    math.isfinite(reported) and 0 <= reported <= 3e-5, "reported error"
                )
                arrays = {}
                for name, size in (
                    ("A", m * k),
                    ("B", k * n),
                    ("I", m * n),
                    ("C", m * n),
                    ("OLD", m * n),
                ):
                    row = next(lines, "").split()
                    require(
                        row and row[0] == name and len(row) == size + 1,
                        "array name/size",
                    )
                    v = np.array([float.fromhex(x) for x in row[1:]], dtype=np.float32)
                    require(np.isfinite(v).all(), "nonfinite array")
                    arrays[name] = v
                expected_a = np.array(
                    [(int((i * 17 + 3) % 41) - 20) / 31 for i in range(m * k)],
                    dtype=np.float32,
                )
                expected_b = np.array(
                    [(int((i * 13 + 7) % 43) - 21) / 29 for i in range(k * n)],
                    dtype=np.float32,
                )
                expected_i = np.array(
                    [(int(i % 11) - 5) / 17 for i in range(m * n)], dtype=np.float32
                )
                for name, value in (
                    ("A", expected_a),
                    ("B", expected_b),
                    ("I", expected_i),
                ):
                    require(
                        np.array_equal(
                            arrays[name].view(np.uint32), value.view(np.uint32)
                        ),
                        "fixture mismatch " + name,
                    )
                require(
                    np.array_equal(
                        arrays["C"].view(np.uint32), arrays["OLD"].view(np.uint32)
                    ),
                    "old/candidate bits",
                )
                a = arrays["A"].reshape((k, m) if f & 2 else (m, k))
                b = arrays["B"].reshape((n, k) if f & 1 else (k, n))
                if f & 2:
                    a = a.T
                if f & 1:
                    b = b.T
                ref = a.astype(np.float64) @ b.astype(np.float64)
                if mode:
                    ref += arrays["I"].reshape(m, n)
                error = float(
                    np.max(np.abs(arrays["C"].reshape(m, n) - ref) / (1 + np.abs(ref)))
                )
                require(error <= 3e-5, "independent double error")
                require(abs(error - reported) <= 1e-12, "reported error disagrees")
                worst = max(worst, error)
                seen.add(ident)
    require(next(lines, "") == "PASS 96", "completion marker")
    require(not any(x.strip() for x in lines), "trailing output")
    return {
        "pass": True,
        "cases": len(seen),
        "maximum_normalized_error": worst,
        "bound": 3e-5,
    }


def timings(text):
    rows = text.splitlines()
    require(rows and rows[0].startswith("DEVICE ") and "L4" in rows[0], "timing device")
    data = {}
    index = 1
    for f in range(4):
        for pair in range(7):
            for leg in range(2):
                fields = rows[index].split()
                index += 1
                old = (not leg) if pair % 2 else bool(leg)
                require(
                    len(fields) == 5
                    and fields[0] == "TIMING"
                    and tuple(map(int, fields[1:4])) == (f, pair, int(old)),
                    "timing identity",
                )
                ms = float(fields[4])
                require(math.isfinite(ms) and ms > 0, "timing finite positive")
                data.setdefault(str(f), {"old": [], "candidate": []})[
                    "old" if old else "candidate"
                ].append(ms)
    require(rows[index:] == ["TIMING_PASS 56"], "timing complete")
    return {
        "scope": "descriptive CUDA-event kernel timing only; 256 cubed; alternating matched pairs; no whole-model speed claim",
        "flags": data,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--timing", type=Path)
    a = p.parse_args()
    result = replay(a.log.read_text())
    if a.timing:
        result["timing"] = timings(a.timing.read_text())
    print(json.dumps(result))
