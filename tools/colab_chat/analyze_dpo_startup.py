#!/usr/bin/env python3
"""Read-only analysis of one process's opt-in DPO startup stderr markers.

This does not locate the exact instant of a halt, infer a CUDA cause, advance a
checkpoint clock, or establish any training/adoption verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PREFIX = "DPO_STARTUP_PHASE "
MAX_BYTES = 64 * 1024 * 1024
FIELDS = {"phase", "elapsed_seconds", "pairs_completed", "pairs_total", "step", "diagnostic_only"}
ORDER = ["reference_model_load_begin", "reference_model_load_end", "reference_cache_begin",
         "reference_cache_end", "first_training_step_begin", "first_training_step_end"]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def analyze(raw: bytes, expected_pairs: int, start_step: int = 0):
    if type(expected_pairs) is not int or expected_pairs <= 0:
        raise ValueError("expected_pairs must be a positive integer")
    if type(start_step) is not int or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    if len(raw) > MAX_BYTES:
        raise ValueError("stderr exceeds bounded input size")
    text = raw.decode("utf-8", errors="strict")
    rows, next_index, progress, previous_time = [], 0, 0, -1.0
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            continue
        row = json.loads(line[len(PREFIX):], object_pairs_hook=unique_object)
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("unexpected diagnostic fields")
        if row["diagnostic_only"] is not True:
            raise ValueError("marker is not diagnostic-only")
        for key in ("pairs_completed", "pairs_total", "step"):
            if type(row[key]) is not int:
                raise ValueError("counts and step must be integers")
        elapsed = row["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < previous_time or elapsed < 0:
            raise ValueError("invalid or decreasing diagnostic clock")
        if row["pairs_total"] != expected_pairs:
            raise ValueError("reference-pair universe changed")
        phase, completed = row["phase"], row["pairs_completed"]
        if phase == "reference_cache_progress":
            if next_index != 3 or not progress < completed <= expected_pairs or row["step"] != 0:
                raise ValueError("invalid reference progress or phase order")
            progress = completed
        else:
            if next_index >= len(ORDER) or phase != ORDER[next_index]:
                raise ValueError("unknown, repeated or out-of-order phase")
            if completed != (0 if next_index < 3 else expected_pairs):
                raise ValueError("phase completion count differs")
            if next_index == 3 and progress != expected_pairs:
                raise ValueError("cache end lacks complete progress marker")
            if row["step"] != (start_step + 1 if next_index >= 4 else 0):
                raise ValueError("phase training step differs")
            next_index += 1
        previous_time = elapsed
        rows.append(row)
    by_phase = {r["phase"]: r for r in rows if r["phase"] != "reference_cache_progress"}
    durations = {}
    for label, begin, end in [
        ("reference_model_load", ORDER[0], ORDER[1]),
        ("reference_cache", ORDER[2], ORDER[3]),
        ("first_training_iteration", ORDER[4], ORDER[5]),
    ]:
        if begin in by_phase and end in by_phase:
            durations[label] = by_phase[end]["elapsed_seconds"] - by_phase[begin]["elapsed_seconds"]
    return {
        "kind": "DPO_STARTUP_OBSERVATION_V1", "diagnostic_only": True,
        "stderr_sha256": hashlib.sha256(raw).hexdigest(), "stderr_bytes": len(raw),
        "expected_pairs": expected_pairs, "start_step": start_step,
        "markers": len(rows), "last_observed_phase": rows[-1]["phase"] if rows else None,
        "last_observed_elapsed_seconds": rows[-1]["elapsed_seconds"] if rows else None,
        "reference_pairs_observed_complete": progress,
        "reference_cache_end_observed": ORDER[3] in by_phase,
        "first_iteration_end_observed": ORDER[5] in by_phase,
        "completed_phase_seconds": durations,
        "checkpoint_proof": False, "adoption_verdict": None,
        "limitation": "Last observation is not the exact halt phase; clocks begin at reference setup, not process launch. No extrapolated completion time or causal GPU diagnosis.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stderr", type=Path)
    parser.add_argument("--expected-pairs", required=True, type=int)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.stderr.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    report = analyze(raw, args.expected_pairs, args.start_step)
    # Exclusive creation protects original logs and previous observations.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
