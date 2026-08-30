#!/usr/bin/env python3
"""Validate an events.jsonl against docs/EVENTS_SPEC.md (contract v1).

    python tools/validate_events.py RUN_events.jsonl [more.jsonl ...]
    python tools/validate_events.py --in-flight tail.jsonl   # no `done` needed

Exit 0 iff every file satisfies the spec, including the resume-segment
rules (a killed-and-resumed run appends a fresh start+model segment;
step numbers restart across the boundary; model scope keys must agree
across segments). This is the machine half of the Atlas submission
standard: a findings.jsonl row's receipts must pass here before the
row is even read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODEL_REQUIRED = ("family", "d", "layers", "seed")


def validate(path: str, in_flight: bool) -> list:
    errs = []
    n_start = n_done = n_step = 0
    seg_last_step = 0          # step counter, reset at each segment start
    seg_has_model = False
    first_model = None         # scope dict from the first segment's model
    first_event = None
    done_seen_before_last = False
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        return [f"unreadable: {e}"]
    with fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"line {i}: not JSON ({e})")
                continue
            if not isinstance(ev, dict) or "event" not in ev:
                errs.append(f"line {i}: no \"event\" key")
                continue
            kind = ev["event"]
            if first_event is None:
                first_event = kind
                if kind != "start":
                    errs.append("start is not the first event")
            if n_done and kind != "done":
                done_seen_before_last = True
            if kind == "start":
                n_start += 1
                seg_last_step = 0
                seg_has_model = False
            elif kind == "model":
                seg_has_model = True
                missing = [k for k in MODEL_REQUIRED if k not in ev]
                if missing:
                    errs.append(f"line {i}: model event missing required "
                                f"key(s) {missing}")
                scope = {k: ev.get(k) for k in MODEL_REQUIRED}
                if first_model is None:
                    first_model = scope
                elif scope != first_model:
                    errs.append(f"line {i}: model scope differs across "
                                f"segments ({scope} vs {first_model}) — "
                                f"file mixes two runs")
            elif kind == "step":
                n_step += 1
                if not seg_has_model:
                    errs.append(f"line {i}: step precedes this segment's "
                                f"model event")
                    seg_has_model = True      # report once per segment
                s = ev.get("step")
                if not isinstance(s, (int, float)):
                    errs.append(f"line {i}: step event missing numeric "
                                f"\"step\"")
                elif s <= seg_last_step:
                    errs.append(f"line {i}: step {s} not increasing within "
                                f"segment (prev {seg_last_step})")
                else:
                    seg_last_step = int(s)
            elif kind == "eval":
                if not isinstance(ev.get("val_loss"), (int, float)):
                    errs.append(f"line {i}: eval event missing numeric "
                                f"\"val_loss\"")
            elif kind == "done":
                n_done += 1
    if n_start == 0:
        errs.append("no start event")
    if first_model is None:
        errs.append("no model event")
    if n_step == 0:
        errs.append("no step events")
    if not in_flight:
        if n_done != 1:
            errs.append(f"expected exactly one done event, saw {n_done} "
                        f"(in-flight file? pass --in-flight)")
        elif done_seen_before_last:
            errs.append("done is not the last event")
    return errs


def check_provenance(path: str) -> list:
    """Receipts must name the code that made them (docs/EVENTS_SPEC.md).
    Looks for provenance.json beside the events file — either in the run
    directory or named <run>_provenance.json next to a flattened copy."""
    d = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    stem = base[:-len("_events.jsonl")] if base.endswith("_events.jsonl") \
        else ""
    cands = [os.path.join(d, "provenance.json")]
    if stem:
        cands.append(os.path.join(d, stem + "_provenance.json"))
    for c in cands:
        if os.path.exists(c):
            try:
                with open(c, encoding="utf-8") as f:
                    p = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                return [f"provenance unreadable: {e}"]
            errs = []
            if not p.get("repo_commit"):
                errs.append("provenance has no repo_commit")
            if p.get("repo_dirty"):
                errs.append("provenance says repo_dirty=true — this run is "
                            "not reproducible from any commit")
            return errs
    return ["no provenance.json beside the receipt (run mtsweep, or see "
            "docs/EVENTS_SPEC.md)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--in-flight", action="store_true",
                    help="accept files without a done event")
    ap.add_argument("--require-provenance", action="store_true",
                    help="also require a provenance.json naming a clean "
                         "commit — the Atlas submission setting")
    args = ap.parse_args()
    bad = 0
    for path in args.files:
        errs = validate(path, args.in_flight)
        if args.require_provenance:
            errs = errs + check_provenance(path)
        if errs:
            bad += 1
            print(f"FAIL {path}")
            for e in errs[:20]:
                print(f"  {e}")
            if len(errs) > 20:
                print(f"  ... {len(errs) - 20} more")
        else:
            print(f"ok   {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
