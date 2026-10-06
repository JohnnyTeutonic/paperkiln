#!/usr/bin/env python3
"""Replication as a CLI verb.

    python tools/reproduce.py S3-lrxopt                 # plan + cost (no compute)
    python tools/reproduce.py S3-lrxopt --run           # actually re-run it
    python tools/reproduce.py S3-lrxopt --check-only \
           --rows experiments/atlas_stage3/atlas_rows.jsonl   # verify vs receipts

Every experimental finding in atlas/findings.jsonl carries its mtsweep
manifest and a machine-checkable `check` (metric, factor or pair,
expected direction or null, t threshold). This tool:

  default     prints the plan AND THE COST (runs, estimated wall-clock
              from the original receipts' own wall_seconds) — reproduce
              tells you the price before spending your machine
  --run       executes the manifest into a FRESH out_root, analyzes,
              and issues a verdict: REPLICATED (same sign, |t| over
              threshold), DID-NOT-REPLICATE (sign flip or signal
              where null expected), or UNDERPOWERED/AMBIGUOUS
  --check-only  runs the same verdict logic against an EXISTING rows
              file — verifying the registry against its own committed
              receipts, or against rows a collaborator sends you

Replication failures are results: append them to the registry, don't
delete the row.

RoPE head coverage. Llama-family findings registered before
arch.rope_heads existed (ROPE_HEADS_SINCE) were produced by runs that
rotate the FIRST head only. `--run` reproduces them as registered: when
the manifest does not state arch.rope_heads, the replay's base spec gets
arch.rope_heads = "first" (the manifest file itself is never edited).
`--rope-heads all` re-runs the same design with RoPE on every head
instead: a new experiment, not a replication of the registered row.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_analyze import interactions, main_effects  # noqa: E402
from atlas_findings import load as load_findings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Date arch.rope_heads entered the spec format (tools/mtstudio.cpp). Every
# finding registered before it ran llama models with rope_heads=first.
ROPE_HEADS_SINCE = "2026-10-06"


def replay_rope_heads(finding, sweep, override):
    """The arch.rope_heads a replay pins, or None to leave the sweep as is."""
    from mtsweep import any_llama, rope_heads_stated
    if not any_llama(sweep):
        return None
    if override in ("first", "all"):
        return override
    if rope_heads_stated(sweep):
        return None
    return "first" if finding.get("date", "") < ROPE_HEADS_SINCE else None


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("complete")]


def verdict(finding, rows, factors):
    chk = finding["check"]
    metric = chk["metric"]
    if "pair" in chk:
        ints = interactions(rows, factors, metric)
        pa, pb = (p.split(".")[-1] for p in chk["pair"])
        key = next((k for k in ints if pa in k and pb in k), None)
        if key is None:
            return "AMBIGUOUS", "interaction pair not found in rows"
        eff, t = ints[key]["effect"], ints[key]["t"]
    else:
        effs = main_effects(rows, factors, metric)
        if chk["factor"] not in effs:
            return "AMBIGUOUS", f"factor {chk['factor']} not found in rows"
        eff, t = effs[chk["factor"]]["effect"], effs[chk["factor"]]["t"]
    if chk.get("expect") == "null":
        if abs(t) < chk.get("max_abs_t", 2.0):
            return "REPLICATED", f"null holds: t={t:+.2f}"
        return "DID-NOT-REPLICATE", f"signal where null registered: t={t:+.2f}"
    want = chk["direction"]
    if eff * want > 0 and abs(t) >= chk.get("min_abs_t", 2.0):
        return "REPLICATED", f"effect={eff:+.4g} t={t:+.2f} (registered t={finding.get('t')})"
    if eff * want < 0 and abs(t) >= 2.0:
        return "DID-NOT-REPLICATE", f"SIGN FLIP: effect={eff:+.4g} t={t:+.2f}"
    return "UNDERPOWERED/AMBIGUOUS", f"effect={eff:+.4g} t={t:+.2f}"


def estimate_hours(finding):
    """Cost from the original receipts' own wall_seconds, if present —
    receipts first, then the manifest's sibling rows file."""
    cands = [r for r in finding.get("receipts", [])
             if r.endswith("atlas_rows.jsonl")]
    m = finding.get("manifest", "")
    if m:
        cands.append(os.path.join(os.path.dirname(m), "atlas_rows.jsonl"))
    for rec in cands:
        try:
            rows = load_rows(os.path.join(ROOT, rec))
            secs = sum(r.get("wall_seconds", 0) for r in rows)
            if secs:
                return secs / 3600.0
        except OSError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("finding_id")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--rows", help="existing atlas_rows.jsonl for --check-only")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="fresh out_root (default /tmp/reproduce_<id>)")
    ap.add_argument("--rope-heads", choices=("registered", "first", "all"),
                    default="registered",
                    help="RoPE head coverage for --run: 'registered' (default) "
                         "replays what the finding was produced with (first "
                         "for rows registered before arch.rope_heads); 'all' "
                         "re-runs the design with every head rotated")
    args = ap.parse_args()

    finding = next((f for f in load_findings() if f["id"] == args.finding_id),
                   None)
    if finding is None:
        raise SystemExit(f"unknown finding {args.finding_id} "
                         "(see tools/atlas_findings.py render)")
    print(f"[{finding['id']}] {finding['claim']}")
    print(f"  status: {finding['status']}  scope: {finding['scope']}")
    if "check" not in finding:
        raise SystemExit("this finding has no machine check "
                         "(non-sweep evidence; see its receipts)")
    manifest = finding.get("manifest", "")
    if not manifest.endswith(".json"):
        raise SystemExit(f"manifest {manifest} is not an mtsweep manifest; "
                         "follow the receipts by hand")

    if args.check_only:
        rows_path = args.rows
        if not rows_path:
            raise SystemExit("--check-only needs --rows")
        with open(os.path.join(ROOT, manifest), encoding="utf-8") as f:
            factors = json.load(f)["factors"]
        rows = load_rows(rows_path)
        v, why = verdict(finding, rows, factors)
        print(f"  {v}: {why}  ({len(rows)} rows from {rows_path})")
        sys.exit(0 if v == "REPLICATED" else 1)

    with open(os.path.join(ROOT, manifest), encoding="utf-8") as f:
        sweep = json.load(f)
    n_runs = None
    est = estimate_hours(finding)
    print(f"  manifest: {manifest}  design={sweep.get('design')}  "
          f"seeds={sweep.get('seeds')}")
    rope = replay_rope_heads(finding, sweep, args.rope_heads)
    if rope:
        from mtsweep import pin_rope_heads
        sweep = pin_rope_heads(sweep, rope)
        print(f"  rope_heads: {rope}"
              + (" (as registered: the finding predates arch.rope_heads)"
                 if args.rope_heads == "registered" else " (override)"))
    if est:
        print(f"  COST: ~{est:.1f} h serial CPU (from the original run's own "
              f"wall_seconds); --jobs {args.jobs} => ~{est / args.jobs:.1f} h")
    if not args.run:
        print("  (dry plan — pass --run to spend the compute, or --check-only "
              "--rows FILE to verify existing rows)")
        return

    out_root = args.out or f"/tmp/reproduce_{finding['id']}"
    sweep = dict(sweep, out_root=out_root)
    sweep_path = f"/tmp/reproduce_{finding['id']}_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep, f)
    print(f"  running into {out_root} ...")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "tools", "mtsweep.py"), sweep_path,
                    "--jobs", str(args.jobs)], check=True)
    rows = load_rows(os.path.join(out_root, "atlas_rows.jsonl"))
    v, why = verdict(finding, rows, sweep["factors"])
    print(f"  {v}: {why}")
    print("  (a DID-NOT-REPLICATE is a result — append it to the registry, "
          "do not delete the row)")
    sys.exit(0 if v == "REPLICATED" else 1)


if __name__ == "__main__":
    main()
