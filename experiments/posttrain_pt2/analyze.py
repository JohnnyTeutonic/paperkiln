#!/usr/bin/env python3
"""PT2 decision rules (PREREGISTRATION.md, "Hypotheses and decision rules").
Frozen at the licence commit.

    python experiments/posttrain_pt2/analyze.py WORK/results.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

QS = ("10", "25")
T_CRIT = 2.262  # two-tailed, df = 9
N_SEEDS = 10


def paired(results, metric, a, b):
    d = [r["models"][a][metric] - r["models"][b][metric] for r in results]
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(len(d))
    t = m / se if se > 0 else (math.inf if m > 0 else (-math.inf if m < 0 else 0.0))
    return {"contrast": f"{metric}: {a} - {b}", "mean": m, "se": se, "t": t, "n": len(d),
            "per_seed": d, "positive": sum(x > 0 for x in d), "negative": sum(x < 0 for x in d)}


def verdict(c, crit=T_CRIT):
    if c["mean"] > 0 and c["t"] > crit:
        return "SUPPORTED"
    if c["mean"] < 0 and c["t"] < -crit:
        return "CONTRADICTED"
    return "NOT_RESOLVED"


def decide(results):
    if len(results) != N_SEEDS or len({r["seed"] for r in results}) != N_SEEDS:
        raise SystemExit(f"refusing to decide: need {N_SEEDS} distinct seeds, got {len(results)}")
    for r in results:
        for m in ("base", "E", "J10", "R10", "J25", "R25", "X"):
            if m not in r["models"]:
                raise SystemExit(f"refusing to decide: seed {r['seed']} lacks model {m}")
    out = {"H1": {}, "H2": {}, "H3": {}, "descriptive": {}}
    for q in QS:
        c = paired(results, f"judge_truth_gap_{q}", f"J{q}", f"R{q}")
        out["H1"][q] = dict(c, verdict=verdict(c))
        c = paired(results, "truth_acc", "E", f"J{q}")
        out["H2"][q] = dict(c, verdict=verdict(c))
        c = paired(results, "truth_acc", f"R{q}", f"J{q}")
        out["H3"][q] = dict(c, verdict=verdict(c) if abs(c["t"]) > T_CRIT else "NOT_RESOLVED")
    out["descriptive"]["E_minus_X_truth"] = paired(results, "truth_acc", "E", "X")
    for m in ("E", "J10", "R10", "J25", "R25", "X"):
        out["descriptive"][f"hack_frac_{m}_minus_base"] = paired(results, "hack_frac", m, "base")
        out["descriptive"][f"distractor_frac_{m}_minus_base"] = paired(results, "distractor_frac", m, "base")
    out["descriptive"]["K"] = [r["K"] for r in results]
    out["descriptive"]["confusion"] = {r["seed"]: r["confusion"] for r in results}
    return out


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    if res.get("repo_dirty"):
        print("WARNING: results were produced from a dirty tree")
    d = decide(res["results"])
    L = ["# PT2 — analysis output", "", f"level R = {res['level']}; commit {res.get('repo_commit')}", ""]
    for h in ("H1", "H2", "H3"):
        for q, c in d[h].items():
            L.append(f"- **{h} (q=0.{q})** {c['contrast']}: mean {c['mean']:+.4f}, SE {c['se']:.4f}, "
                     f"t {c['t']:+.2f}, signs +{c['positive']}/-{c['negative']} -> **{c['verdict']}**")
            L.append(f"  per seed: {', '.join(f'{x:+.3f}' for x in c['per_seed'])}")
    L += ["", "Descriptive:"]
    for k, c in d["descriptive"].items():
        if isinstance(c, dict) and "mean" in c:
            L.append(f"- {c['contrast']}: mean {c['mean']:+.4f}, SE {c['se']:.4f}, t {c['t']:+.2f}")
    L.append(f"- K per seed: {d['descriptive']['K']}")
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "ANALYSIS.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(text)
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "analysis.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(d, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
