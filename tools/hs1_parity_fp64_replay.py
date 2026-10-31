"""Verify retained FP64 diagnostic rows against the pinned native CPU rows."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def replay(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "fp64_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("artifact hash differs:" + name)
    report = json.loads((folder / "fp64_report.json").read_text())
    if report["predeclared_checks"] != {
        "max_absolute_full_logits_error": 0.001,
        "ordinary_native_top2_gap": 0.001,
        "ordinary_argmax_must_match": True,
    }:
        raise ValueError("prospective diagnostic check changed")
    offsets = {}
    maximum = 0.0
    ordinary = 0
    positions = 0
    for row in report["rows"]:
        run = row["run"]
        native = np.frombuffer(
            gzip.decompress((folder / (run + "_full.logits.f32.gz")).read_bytes()),
            dtype="<f4",
        ).reshape(-1, 16384)
        data = (folder / row["raw_file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["raw_sha256"]:
            raise ValueError("FP64 row hash differs")
        reference = np.frombuffer(gzip.decompress(data), dtype="<f8").reshape(-1, 16384)
        if len(reference) != len(row["positions"]) or not np.isfinite(reference).all():
            raise ValueError("FP64 row count/finite check failed")
        offset = offsets.get(run, 0)
        compatible = True
        for i, saved in enumerate(row["positions"]):
            old, new = native[offset + i].astype(np.float64), reference[i].copy()
            error = float(np.max(np.abs(old - new)))
            old[0], new[0] = -np.inf, -np.inf
            oldtop = np.argsort(-old, kind="stable")[:2]
            top = np.argsort(-new, kind="stable")[:2]
            gap = float(old[oldtop[0]] - old[oldtop[1]])
            tests = {
                "native_argmax": int(oldtop[0]),
                "fp64_argmax": int(top[0]),
                "fp64_runner_up": int(top[1]),
                "fp64_top2_gap": float(new[top[0]] - new[top[1]]),
                "fp64_argmax_minus_saved": float(
                    new[top[0]] - new[saved["saved_chosen"]]
                ),
                "native_top2_gap": gap,
                "ordinary": gap > 0.001,
                "max_abs_error": error,
            }
            if any(saved[key] != value for key, value in tests.items()):
                raise ValueError("per-position diagnostic differs")
            compatible &= error <= 0.001 and (gap <= 0.001 or oldtop[0] == top[0])
            ordinary += gap > 0.001
            positions += 1
            maximum = max(maximum, error)
        if bool(compatible) != row["compatibility_pass"]:
            raise ValueError("compatibility predicate differs")
        offsets[run] = offset + len(reference)
    if positions != 58 or len(report["rows"]) != 4:
        raise ValueError("expected four B replies/58 positions")
    return {
        "status": "VERIFIED_FP64_DIAGNOSTIC",
        "positions": positions,
        "ordinary_positions": ordinary,
        "max_abs_error": maximum,
        "scope": "Numerical sensitivity diagnostic; not historical oracle or parity waiver",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    print(json.dumps(replay(parser.parse_args().folder), indent=2))
