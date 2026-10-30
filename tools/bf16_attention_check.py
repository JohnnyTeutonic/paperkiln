"""Frozen, resumable numerical gate for the standalone BF16 attention probe.

No training inference, HTTP/model calls, or production-dispatch changes.
Every GEMM is checked against an independent float64 reduction of its actual
rounded operands (3e-5 normalized error). End-to-end bounds are 3e-5 FP32 and
5e-3 BF16; these are test-matrix gates, not universal precision/quality claims.
The BF16 backward uses the explicitly documented straight-through surrogate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from bf16_attention_reference import (
    ARRAY_NAMES,
    encode_input,
    read_output,
    reference,
    staged_oracle,
    write_input,
)

SHAPES = [
    (1, 1, 1, 8),
    (2, 3, 7, 8),
    (1, 2, 16, 16),
    (1, 3, 33, 24),
    (2, 2, 64, 32),
    (1, 4, 128, 64),
    (1, 8, 256, 64),
    (1, 4, 512, 64),
]
STAGE_BOUND = 3e-5
BF16_END_BOUND = 5e-3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)  # atomic publication that cannot overwrite an existing result
    tmp.unlink()


def error(got, expected):
    if got.shape != expected.shape or not np.isfinite(got).all():
        raise ValueError("shape/nonfinite output")
    return float(
        np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64)))
        / max(1.0, float(np.max(np.abs(expected))))
    )


def audit_output(inputs, actual, causal, low):
    expected = reference(*inputs, causal=causal, low_precision=low)
    stages = staged_oracle(*inputs, actual, causal, low)
    stage_errors = {name: error(actual[name], stages[name]) for name in ARRAY_NAMES}
    end_errors = {name: error(actual[name], expected[name]) for name in ARRAY_NAMES}
    if max(stage_errors.values()) > STAGE_BOUND:
        raise ValueError("per-stage numerical bound exceeded: " + str(stage_errors))
    bound = BF16_END_BOUND if low else STAGE_BOUND
    if max(end_errors.values()) > bound:
        raise ValueError("end-to-end numerical bound exceeded: " + str(end_errors))
    np.testing.assert_allclose(actual["prob"].sum(-1), 1, atol=2e-6, rtol=0)
    if causal:
        assert not np.any(np.triu(actual["prob"], 1))
        assert not np.any(np.triu(actual["ds"], 1))
    return {
        "stage_errors": stage_errors,
        "end_errors": end_errors,
        "stage_bound": STAGE_BOUND,
        "end_bound": bound,
    }


def main():
    if not __debug__:
        raise RuntimeError("optimized Python disables verification; refusing")
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    binary, root = Path(args.binary).resolve(), args.output.resolve()
    if args.replay and not root.is_dir():
        raise RuntimeError("replay requires existing results")
    if not args.replay:
        root.mkdir(parents=True, exist_ok=True)
    if (root / "HALTED").exists():
        raise RuntimeError("HALTED evidence retained; this result cannot be retried")
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("bf16_attention_reference.py").resolve(),
        Path(__file__).with_name("bf16_attention_probe.cu").resolve(),
    ]

    def snapshot():
        return {"binary": sha(binary), "sources": {p.name: sha(p) for p in sources}}

    frozen = snapshot()
    pins = {
        **frozen,
        "shapes": SHAPES,
        "stage_bound": STAGE_BOUND,
        "bf16_end_bound": BF16_END_BOUND,
        "seed": 20261031,
        "scope": "standalone dense MHA primitive; no production or loss/performance adoption claim",
        "resume": "completed receipts only; interrupted case is preserved and halts for review",
    }

    def check_pins():
        assert snapshot() == frozen, "source/binary changed during matrix"

    if (root / "manifest.json").exists():
        assert json.loads((root / "manifest.json").read_text()) == json.loads(
            json.dumps(pins)
        )
    else:
        if args.replay:
            raise RuntimeError("replay requires manifest")
        immutable_json(root / "manifest.json", pins)
    if (root / "report.json").exists() and not args.replay:
        raise RuntimeError("complete result already exists; use offline replay")
    cases = []
    for index, shape in enumerate(SHAPES):
        rng = np.random.default_rng(20261031 + index)
        inputs = [
            rng.normal(0, 0.2 if index < 3 else 1.0, shape).astype(np.float32)
            for _ in range(4)
        ]
        if index == 1:
            inputs[0].flat[:4] = np.array(
                [1.00390625, 1.01171875, -1.00390625, -1.01171875], dtype=np.float32
            )
        for causal in (False, True):
            for low in (False, True):
                name = f"case{index:02d}_mask{int(causal)}_bf16{int(low)}"
                inf, outf, log, err, receipt = [
                    root / (name + suffix)
                    for suffix in (".in.bin", ".out.bin", ".stdout", ".stderr", ".json")
                ]
                completed = receipt.exists()
                check_pins()
                try:
                    if not completed:
                        if args.replay:
                            raise RuntimeError(
                                "replay requires every completed case receipt"
                            )
                        # Exclusive writes preserve an interrupted case: no automatic scientific retry.
                        write_input(inf, *inputs, causal, low)
                        with log.open("xb") as stdout, err.open("xb") as stderr:
                            result = subprocess.run(
                                [str(binary), str(inf), str(outf)],
                                stdout=stdout,
                                stderr=stderr,
                                timeout=180,
                            )
                        if result.returncode:
                            raise RuntimeError(
                                f"probe failed with status {result.returncode}"
                            )
                    check_pins()
                    assert inf.read_bytes() == encode_input(*inputs, causal, low)
                    assert outf.read_bytes()[:32] == inf.read_bytes()[:32], (
                        "output header differs"
                    )
                    timing = json.loads(log.read_text())
                    assert set(timing) == {
                        "device",
                        "device_uuid",
                        "sm",
                        "iterations",
                        "forward_backward_ms",
                    }
                    assert timing["device"] == "NVIDIA L4" and timing["sm"] == 89
                    assert (
                        timing["iterations"] == 20 and type(timing["iterations"]) is int
                    )
                    assert isinstance(
                        timing["forward_backward_ms"], (int, float)
                    ) and not isinstance(timing["forward_backward_ms"], bool)
                    assert (
                        np.isfinite(timing["forward_backward_ms"])
                        and timing["forward_backward_ms"] > 0
                    )
                    uuid = timing["device_uuid"]
                    assert (
                        isinstance(uuid, str)
                        and len(uuid) == 32
                        and all(c in "0123456789abcdef" for c in uuid)
                    )
                    if cases:
                        assert uuid == cases[0]["timing"]["device_uuid"], (
                            "device changed mid-matrix"
                        )
                    record = {
                        "case": name,
                        "shape": list(shape),
                        "causal": causal,
                        "bf16": low,
                        "input_sha256": sha(inf),
                        "output_sha256": sha(outf),
                        "stdout_sha256": sha(log),
                        "stderr_sha256": sha(err),
                        "timing": timing,
                        "audit": audit_output(inputs, read_output(outf), causal, low),
                    }
                    check_pins()
                    if completed:
                        assert json.loads(receipt.read_text()) == record, (
                            "case receipt differs from raw evidence"
                        )
                    else:
                        immutable_json(receipt, record)
                        print(name, "PASS", flush=True)
                    cases.append(record)
                except Exception as exc:
                    if not args.replay and not (root / "HALTED").exists():
                        immutable_json(
                            root / "HALTED", {"case": name, "error": str(exc)}
                        )
                    raise
    check_pins()
    report = {"status": "COMPLETE", "cases": cases, "pins": pins}
    if args.replay:
        assert json.loads((root / "report.json").read_text()) == json.loads(
            json.dumps(report)
        )
    else:
        immutable_json(root / "report.json", report)
    check_pins()
    print("all", len(cases), "cases complete", flush=True)


if __name__ == "__main__":
    main()
