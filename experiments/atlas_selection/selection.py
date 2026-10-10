"""Seed-held-out decision audit. Standard library only; no training/API calls.

Run: python selection.py run --output results_v1.json
Replay: python selection.py verify results_v1.json
All paths are relative to this file, never the current working directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(scores, candidates):
    """Only calibration scores enter selection; lane index resolves ties."""
    return min(candidates, key=lambda i: (scores[i], i))


def decision(source, calibration, k):
    shortlist = tuple(sorted(range(len(source)), key=lambda i: (source[i], i))[:k])
    return shortlist, choose(calibration, shortlist)


def evaluate(shortlist, chosen, heldout):
    """Finite-panel hindsight benchmark, NOT population simple regret."""
    best_all = min(heldout)
    best_short = min(heldout[i] for i in shortlist)
    return {
        "loss": heldout[chosen],
        "excess_vs_exact": heldout[chosen] - heldout[0],
        "hindsight_gap": heldout[chosen] - best_all,
        "exclusion_gap": best_short - best_all,
        "calibration_gap": heldout[chosen] - best_short,
    }


def lane(model):
    if model["attention"] == "exact":
        return "exact"
    if model["attention"] == "swa":
        return f"swa{model['window']}s{model['sinks']}"
    raise ValueError(f"unrecognised attention: {model['attention']}")


def read_receipt(path, protocol, role):
    """Fail closed on incomplete, non-finite, ambiguous or out-of-scope data.

    Resume segments may repeat evaluations, but only identical values may
    repeat. No malformed-line skipping or silent missing-run intersection.
    """
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not events or events[0].get("event") != "start":
        raise ValueError(f"{path}: missing initial start")
    if events[-1].get("event") != "done" or sum(e.get("event") == "done" for e in events) != 1:
        raise ValueError(f"{path}: must contain one terminal done")
    if events[-1].get("final_step") != protocol["completed_step"]:
        raise ValueError(f"{path}: incomplete run")
    models = [e for e in events if e.get("event") == "model"]
    if not models or any(m != models[0] for m in models):
        raise ValueError(f"{path}: missing or inconsistent model events")
    model = models[0]
    expected = dict(protocol["expected_model"], d=protocol[f"{role}_width"])
    if any(model.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{path}: model outside declared scope")
    if not math.isclose(model["lr"], protocol["expected_lr"][role], rel_tol=1e-6):
        raise ValueError(f"{path}: unexpected learning rate")
    evals = {}
    for e in events:
        if e.get("event") == "eval":
            step, value = e["step"], e["val_loss"]
            if type(step) is not int or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path}: invalid evaluation")
            if step in evals and evals[step] != value:
                raise ValueError(f"{path}: conflicting evaluation at {step}")
            evals[step] = value
    if not set(protocol[f"{role}_steps"]).issubset(evals):
        raise ValueError(f"{path}: required evaluation missing")
    return model["seed"], lane(model), evals


def load_panel(protocol, role):
    root = (HERE / protocol[role]).resolve()
    paths = sorted(root.rglob("events.jsonl"))
    panel, receipts = {}, []
    for path in paths:
        seed, name, evals = read_receipt(path, protocol, role)
        key = (seed, name)
        if key in panel:
            raise ValueError(f"duplicate seed/lane {key}")
        panel[key] = evals
        receipts.append({"role": role, "path": path.relative_to(root).as_posix(), "sha256": sha(path)})
    expected = set(itertools.product(protocol["seeds"], protocol["lanes"]))
    if set(panel) != expected:
        raise ValueError(f"{role}: panel differs from manifest; missing={expected-set(panel)}, extra={set(panel)-expected}")
    return panel, receipts


def means(panel, seeds, lanes, step):
    return [st.fmean(panel[s, name][step] for s in seeds) for name in lanes]


def quantile(xs, p):
    ys = sorted(xs)
    x = (len(ys) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return ys[lo] + (ys[hi] - ys[lo]) * (x - lo)


def summarise(rows):
    return {key: {"mean": st.fmean(r[key] for r in rows),
                  "split_p05": quantile([r[key] for r in rows], .05),
                  "split_p95": quantile([r[key] for r in rows], .95)}
            for key in rows[0]}


def audit(protocol, source, target):
    lanes, seeds = protocol["lanes"], protocol["seeds"]
    n, m = len(lanes), protocol["calibration_seeds"]
    if lanes[0] != "exact" or len(seeds) != len(set(seeds)) or not 0 < m < len(seeds):
        raise ValueError("invalid panel/split specification")
    if any(k < 1 or k > n for k in protocol["shortlist_sizes"]):
        raise ValueError("invalid shortlist size")
    folds = list(itertools.combinations(seeds, m))
    # Same seed IDs are withheld at BOTH widths. This prevents shared seed
    # effects from leaking from the source panel into target evaluation.
    cache = []
    for train in folds:
        test = [s for s in seeds if s not in train]
        cache.append((
            {t: means(source, train, lanes, t) for t in protocol["source_steps"]},
            {t: means(target, train, lanes, t) for t in protocol["target_steps"]},
            {t: means(target, test, lanes, t) for t in protocol["target_steps"]},
        ))
    cells = []
    for target_step in protocol["target_steps"]:
        for k in protocol["shortlist_sizes"]:
            # Exact expectation over uniformly random size-k subsets, with
            # the SAME target calibration rule as the informed shortlist.
            random_rows, direct_rows = [], []
            for _, cal, held in cache:
                c, h = cal[target_step], held[target_step]
                rr = [evaluate(subset, choose(c, subset), h)
                      for subset in itertools.combinations(range(n), k)]
                random_rows.append({key: st.fmean(r[key] for r in rr) for key in rr[0]})
                direct_rows.append(evaluate(tuple(range(n)), choose(c, range(n)), h))
            for source_step in protocol["source_steps"]:
                rows, contrasts = [], []
                choices, included = Counter(), Counter()
                for index, (src, cal, held) in enumerate(cache):
                    shortlist, chosen = decision(src[source_step], cal[target_step], k)
                    row = evaluate(shortlist, chosen, held[target_step])
                    rows.append(row)
                    choices[lanes[chosen]] += 1
                    included.update(lanes[i] for i in shortlist)
                    contrasts.append({
                        "loss_vs_random_shortlist": row["loss"] - random_rows[index]["loss"],
                        "loss_vs_direct_target": row["loss"] - direct_rows[index]["loss"],
                    })
                cells.append({
                    "source_step": source_step, "target_step": target_step, "k": k,
                    "policy": summarise(rows), "random_shortlist": summarise(random_rows),
                    "direct_target": summarise(direct_rows), "paired_contrasts": summarise(contrasts),
                    "selected_counts": dict(sorted(choices.items())),
                    "shortlisted_counts": dict(sorted(included.items())),
                    "acquisition_budget": {
                        "source_run_steps": n * m * source_step,
                        "target_run_steps": (k * m * target_step if k > 1 else 0),
                        "random_target_run_steps": (k * m * target_step if k > 1 else 0),
                        "direct_target_run_steps": n * m * target_step,
                        "heldout_scoring_run_steps_per_policy": (len(seeds) - m) * target_step,
                    },
                })
    return {"n_seeds": len(seeds), "n_overlapping_splits": len(folds),
            "split_ids_sha256": hashlib.sha256(canonical(folds).encode()).hexdigest(),
            "cells": cells}


def build_report():
    print("Loading and validating source panel...", flush=True)
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    source, source_receipts = load_panel(protocol, "source")
    print("Loading and validating target panel...", flush=True)
    target, target_receipts = load_panel(protocol, "target")
    print("Computing exhaustive seed-held-out decisions...", flush=True)
    return {
        "status": protocol["status"], "protocol": protocol,
        "code_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "protocol.json"),
        "receipts": source_receipts + target_receipts,
        "interpretation": {
            "split_percentiles": "descriptive sensitivity over overlapping splits, NOT confidence intervals",
            "hindsight_gap": "finite held-out-panel best is an optimistic oracle, NOT population simple regret",
            "generalisation": "seed-held-out only; same corpus, validation tokens, family and width pair",
            "budget": "acquisition run-steps by width, NOT measured FLOPs, latency or a cost-matched comparison",
            "positions": "fixed optimizer steps; no interpolation, best-checkpoint selection or matched-loss claim",
            "tuning": "learning rates already selected in transfer_s2; no retuning here",
        },
        "analysis": audit(protocol, source, target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.output.exists():
        raise SystemExit("output exists; choose a new immutable report path")
    fresh = build_report()
    if args.command == "verify":
        saved = json.loads(args.report.read_text(encoding="utf-8"))
        if canonical(saved) != canonical(fresh):
            raise SystemExit("FAIL: receipt, protocol, source or recomputed result mismatch")
        print("VERIFIED: all input hashes, selections, baselines, decompositions and summaries")
    else:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(fresh, f, indent=2, allow_nan=False)
            f.write("\n")
        print(f"Wrote {args.output}: {len(fresh['receipts'])} runs, "
              f"{fresh['analysis']['n_overlapping_splits']} overlapping splits, "
              f"{len(fresh['analysis']['cells'])} cells; POST-HOC")


if __name__ == "__main__":
    main()
