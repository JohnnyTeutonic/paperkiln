#!/usr/bin/env python3
"""Paired contrasts for the turn-distance set (tools/iftiny_eval.py's items.jsonl.gz per model).

    python analyze_if.py DIR [DIR ...] [--json OUT]

Per model and distance d, over the 84 (family, sequence) pairs: instruction (user placement) minus nohint
(the same compliant history without the instruction: what the format alone carries over), instruction
minus bare, and system minus user placement. Each is a mean paired difference with a normal-approximation
95% interval (sd / sqrt(84)); descriptive, no correction for the many cells.
"""
import gzip
import json
import math
import os
import sys


def load(d):
    out = {}
    for tag in sorted(os.listdir(d)):
        p = os.path.join(d, tag, "items.jsonl.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                out[tag] = [json.loads(x) for x in f if x.strip()]
    return out


def paired(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return {"n": n, "diff": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)]}


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        args = args[:i] + args[i + 2:]
    res = {}
    for d in args:
        for tag, recs in load(d).items():
            cells = {}
            for r in recs:
                if r["set"] != "ifdist":
                    continue
                _, fam, seq, dd, cond = r["id"].split("/")
                cells.setdefault((dd, cond), {})[(fam, seq)] = int(r["comply"])
            res[tag] = {}
            for dd in ("d0", "d1", "d2", "d4"):
                keys = sorted(cells[(dd, "instr_user")])
                g = {c: [cells[(dd, c)][k] for k in keys] for c in ("instr_user", "instr_system", "nohint", "bare")}
                res[tag][dd] = {"user_minus_nohint": paired(g["instr_user"], g["nohint"]),
                                "user_minus_bare": paired(g["instr_user"], g["bare"]),
                                "system_minus_user": paired(g["instr_system"], g["instr_user"])}
    lines = ["| model | d | user - nohint | user - bare | system - user |", "|---|---|---|---|---|"]
    for tag, v in res.items():
        for dd, x in v.items():
            f = lambda c: f"{c['diff']:+.3f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"  # noqa: E731
            lines.append(f"| {tag} | {dd} | {f(x['user_minus_nohint'])} | {f(x['user_minus_bare'])} | "
                         f"{f(x['system_minus_user'])} |")
    print("\n".join(lines))
    if out_json:
        with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
