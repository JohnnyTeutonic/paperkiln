#!/usr/bin/env python3
"""PT5 decision rules (PREREGISTRATION.md, "Decision rules"). Frozen at the
licence commit.

    python experiments/posttrain_pt5/analyze.py WORK/results.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

N_SEEDS = 10
T_CRIT = 2.262  # two-tailed, df = 9
GAP = "judge_truth_gap_25"
MODELS = ("base", "E", "J", "R", "Mu", "Mw", "Md")


def paired(results, metric, a, b):
    d = [r["models"][a][metric] - r["models"][b][metric] for r in results]
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(len(d))
    t = m / se if se > 0 else (math.inf if m > 0 else (-math.inf if m < 0 else 0.0))
    return {"contrast": f"{metric}: {a} - {b}", "mean": m, "sd": sd, "se": se, "t": t, "per_seed": d,
            "positive": sum(x > 0 for x in d), "negative": sum(x < 0 for x in d)}


def decide(results):
    if len(results) != N_SEEDS or len({r["seed"] for r in results}) != N_SEEDS:
        raise SystemExit(f"refusing to decide: need {N_SEEDS} distinct seeds, got {len(results)}")
    for r in results:
        for m in MODELS:
            if m not in r["models"]:
                raise SystemExit(f"refusing to decide: seed {r['seed']} lacks model {m}")
    a = paired(results, GAP, "J", "R")
    if a["mean"] > 0 and a["t"] > T_CRIT:
        va = "SUPPORTED"
    elif a["mean"] < 0 and a["t"] < -T_CRIT:
        va = "CONTRADICTED"
    else:
        va = "NOT_RESOLVED"
    b1 = paired(results, GAP, "Mu", "Mw")
    b2 = paired(results, "truth_acc", "Md", "Mw")
    i_ok = b1["mean"] > 0 and b1["t"] > T_CRIT
    ii_ok = b2["mean"] < b2["sd"]
    vb = ("PROVENANCE_WEIGHTING_EARNED" if i_ok and ii_ok else
          "WEIGHTING_INERT" if not i_ok else "DROP_BEATS_WEIGHT")
    desc = {"E_minus_J_truth": paired(results, "truth_acc", "E", "J"),
            "R_minus_J_truth": paired(results, "truth_acc", "R", "J"),
            "J_minus_R_distractor": paired(results, "distractor_frac", "J", "R"),
            "N": [r["N"] for r in results], "composition": {r["seed"]: r["composition"] for r in results}}
    return {"F_PT5a": dict(a, verdict=va), "F_PT5b": {"i": dict(b1, holds=i_ok), "ii": dict(b2, holds=ii_ok),
                                                      "verdict": vb}, "descriptive": desc}


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    if res.get("repo_dirty"):
        print("WARNING: results were produced from a dirty tree")
    d = decide(res["results"])
    L = ["# PT5 — analysis output", "", f"commit {res.get('repo_commit')}", ""]
    c = d["F_PT5a"]
    L.append(f"- **F-PT5a** {c['contrast']}: mean {c['mean']:+.4f}, SE {c['se']:.4f}, t {c['t']:+.2f}, "
             f"signs +{c['positive']}/-{c['negative']} -> **{c['verdict']}**")
    for k in ("i", "ii"):
        c = d["F_PT5b"][k]
        L.append(f"- F-PT5b({k}) {c['contrast']}: mean {c['mean']:+.4f}, SD {c['sd']:.4f}, t {c['t']:+.2f} "
                 f"-> {'holds' if c['holds'] else 'fails'}")
    L += [f"- **F-PT5b: {d['F_PT5b']['verdict']}**", "", "Descriptive:"]
    for k, c in d["descriptive"].items():
        if isinstance(c, dict) and "mean" in c:
            L.append(f"- {c['contrast']}: mean {c['mean']:+.4f}, SE {c['se']:.4f}, t {c['t']:+.2f}")
    L.append(f"- N per seed: {d['descriptive']['N']}")
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
