#!/usr/bin/env python3
"""PT4 decision rules (PREREGISTRATION.md, "Decision rules"). Frozen at the
licence commit.

    python experiments/posttrain_pt4/analyze.py WORK/results.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

N_SEEDS = 5
T_CRIT = 2.776  # two-tailed, df = 4


def paired(results, a, b):
    d = [r["points"][a]["M"] - r["points"][b]["M"] for r in results]
    m = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    t = m / se if se > 0 else (math.inf if m > 0 else (-math.inf if m < 0 else 0.0))
    return {"contrast": f"M({a}) - M({b})", "mean": m, "se": se, "t": t, "per_seed": d}


def decide(results):
    if len(results) != N_SEEDS or len({r["seed"] for r in results}) != N_SEEDS:
        raise SystemExit(f"refusing to decide: need {N_SEEDS} distinct seeds")
    for r in results:
        for p in ("base", "open", "closed"):
            if p not in r["points"]:
                raise SystemExit(f"refusing to decide: seed {r['seed']} lacks point {p}")
    base = [r["points"]["base"]["M"] for r in results]
    threshold = statistics.mean(base) + 3 * statistics.stdev(base)
    over = sum(r["points"]["open"]["M"] > threshold for r in results)
    a = paired(results, "open", "base")
    b = paired(results, "open", "closed")
    a_ok = a["mean"] > 0 and a["t"] > T_CRIT and over >= 4
    b_ok = b["mean"] > 0 and b["t"] > T_CRIT
    return {"threshold": threshold, "open_over_threshold": over, "a": dict(a, holds=a_ok), "b": dict(b, holds=b_ok),
            "verdict": "MONITOR_VALIDATED" if a_ok and b_ok else
                       ("MONITOR_BLIND" if not a_ok else "MONITOR_CANNOT_SEE_CLOSURE"),
            "truth": {p: [r["points"][p]["truth_acc"] for r in results] for p in ("base", "open", "closed")},
            "M": {p: [r["points"][p]["M"] for r in results] for p in ("base", "open", "closed")}}


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    if res.get("repo_dirty"):
        print("WARNING: results were produced from a dirty tree")
    d = decide(res["results"])
    L = ["# PT4 — analysis output", "", f"commit {res.get('repo_commit')}", "",
         f"threshold (mean + 3 SD of M_base) = {d['threshold']:.4f}; open runs above it: {d['open_over_threshold']}/5",
         ""]
    for k in ("a", "b"):
        c = d[k]
        L.append(f"- F-PT4({k}) {c['contrast']}: mean {c['mean']:+.4f}, SE {c['se']:.4f}, t {c['t']:+.2f} "
                 f"-> {'holds' if c['holds'] else 'FAILS'}; per seed {', '.join(f'{x:+.3f}' for x in c['per_seed'])}")
    L += ["", f"**Verdict: {d['verdict']}**", ""]
    for p in ("base", "open", "closed"):
        L.append(f"- {p}: M {', '.join(f'{x:.3f}' for x in d['M'][p])}; "
                 f"truth {', '.join(f'{x:.3f}' for x in d['truth'][p])}")
    text = "\n".join(L) + "\n"
    print(text)
    out = os.path.dirname(os.path.abspath(path))
    with open(os.path.join(out, "ANALYSIS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(out, "analysis.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
