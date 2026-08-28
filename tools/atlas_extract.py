#!/usr/bin/env python3
"""Atlas stage 0b: behavioural feature extraction (atlas/ARCHITECTURE_ATLAS.md).

Turns a finished mtstudio out_dir into one Atlas row: the structural echo
from result.json joined with behavioural features computed from
events.jsonl — the y_i vector, extracted from the stream the trainer
already emits.

    python tools/atlas_extract.py OUT_DIR [OUT_DIR ...] [--jsonl rows.jsonl]
    python tools/atlas_extract.py --selftest

Behavioural features (each named, each cheap, each defined here rather
than in prose):
  best_val, final_train_loss   endpoint quality
  steps_to_half_gap            convergence speed: first step at which the
                               train loss has closed half the gap between
                               its initial value and its final value
  loss_auc_norm                mean train loss over the run divided by the
                               initial loss — area under the (normalised)
                               curve; lower = faster descent, sample-
                               efficiency proxy
  grad_norm_mean/max           gradient scale
  grad_spike_count             instability: post-warmup steps (first 5%
                               excluded) with grad_norm > 3x the
                               post-warmup median — the warmup transient
                               was ~85% of the old count and is now its
                               own metric
  grad_init_transient          max grad_norm in the first 5% of steps /
                               post-warmup median: how violent was init
  loss_tail_std                late-run noise: std of the last 20% of
                               train-loss samples
  gate_mean_final              SRD only: mean gate at the end (density)
No seed-variance here by design: variance is a property of a CELL (same
config, several seeds), so it is computed by mtsweep's aggregator, not
per run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys


def read_events(path):
    events = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial tail line from a killed run
    return events


def behavioural_features(events):
    steps = [e for e in events if e.get("event") == "step"]
    evals = [e for e in events if e.get("event") == "eval"]
    feats = {}
    if not steps:
        return feats

    losses = [e["loss"] for e in steps if "loss" in e]
    if losses:
        first, final = losses[0], statistics.fmean(losses[-max(1, len(losses) // 10):])
        feats["final_train_loss"] = final
        feats["loss_auc_norm"] = statistics.fmean(losses) / first if first else None
        half = first - 0.5 * (first - final)
        feats["steps_to_half_gap"] = next(
            (e["step"] for e, l in zip(steps, losses) if l <= half), None)
        tail = losses[-max(2, len(losses) // 5):]
        feats["loss_tail_std"] = statistics.pstdev(tail)

    gnorms = [e["grad_norm"] for e in steps if e.get("grad_norm") is not None]
    if gnorms:
        feats["grad_norm_mean"] = statistics.fmean(gnorms)
        feats["grad_norm_max"] = max(gnorms)
        # Instability, NOT the init transient: measured 2026-08-01 on the
        # Stage-2 corpus, ~85% of >3x-median exceedances sat in the first
        # 5% of steps (median spike position: the 1% mark) — that is the
        # warmup story, and it was contaminating the instability story
        # (atlas/ATLAS_STAGE2_RESULTS.md finding #6). Split them: spikes are
        # counted over the post-warmup 95% against the post-warmup
        # median; the transient gets its own metric.
        warm = max(1, len(gnorms) // 20)
        body = gnorms[warm:] if len(gnorms) > warm + 3 else gnorms
        med = statistics.median(body)
        spikes = sum(1 for g in body if med > 0 and g > 3 * med)
        feats["grad_spike_count"] = spikes
        # RATE per 1,000 post-warmup steps: token-matched designs vary the
        # step count (Stage 3: 600 vs 1200 steps at equal tokens), which
        # makes a raw COUNT mechanically higher in longer runs — the
        # Stage-3 ctx→spikes "signal" was partly that. Rate is the
        # cross-budget-comparable column; the count stays for continuity.
        feats["grad_spike_rate"] = 1000.0 * spikes / len(body) if body else None
        feats["grad_init_transient"] = (max(gnorms[:warm]) / med) if med > 0 else None

    if evals:
        feats["best_val"] = min(e["val_loss"] for e in evals)
        feats["n_evals"] = len(evals)

    gates = [e["gate"] for e in steps if "gate" in e]
    if gates:
        feats["gate_mean_final"] = statistics.fmean(gates[-max(1, len(gates) // 10):])
    return feats


def extract_row(out_dir):
    """result.json (structural + outcome) + events.jsonl (behavioural)."""
    row = {"out_dir": out_dir}
    rpath = os.path.join(out_dir, "result.json")
    epath = os.path.join(out_dir, "events.jsonl")
    if os.path.exists(rpath):
        with open(rpath, "r", encoding="utf-8") as f:
            row.update(json.load(f))
    if os.path.exists(epath):
        feats = behavioural_features(read_events(epath))
        # result.json's best_val is authoritative when both exist
        for k, v in feats.items():
            row.setdefault(k, v)
    row["complete"] = os.path.exists(rpath)
    return row


def selftest():
    fixture = os.path.join(os.path.dirname(__file__), "..", "studio", "sample_events.jsonl")
    feats = behavioural_features(read_events(fixture))
    required = ["final_train_loss", "loss_auc_norm", "steps_to_half_gap",
                "grad_norm_mean", "grad_spike_count"]
    missing = [k for k in required if k not in feats]
    print(json.dumps(feats, indent=2, sort_keys=True))
    if missing:
        print(f"SELFTEST FAIL: missing {missing}")
        return 1
    if not (0 < feats["loss_auc_norm"] <= 1.5):
        print("SELFTEST FAIL: loss_auc_norm out of range")
        return 1
    print("SELFTEST OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="*")
    ap.add_argument("--jsonl", help="append rows to this file instead of stdout")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.out_dirs:
        ap.error("give at least one out_dir (or --selftest)")
    rows = [extract_row(d) for d in args.out_dirs]
    if args.jsonl:
        with open(args.jsonl, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"wrote {len(rows)} row(s) -> {args.jsonl}")
    else:
        for r in rows:
            print(json.dumps(r, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
