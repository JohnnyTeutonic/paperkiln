#!/usr/bin/env python3
"""Check 7: the overlap filter on T_E rows, as a black box. Claude Opus 5.5, 7 Nov 2026. Windows Python, through the CPU budget. Counts only.

build_a19.py's filter runs unchanged, with its last rule off, and correction_split.py's rewrite of that last rule follows, waiving only strings common in A19's own rows: the path T_M and T_C took, with a self-check against A19's kept counts.
Rows are dropped by whole pair. Kept rows go to the output file; which were dropped is not reported.

  python tools/lab/te_filter.py --rows ROWS.jsonl --out KEPT.jsonl --scratch DIR --report REPORT.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LAB_ROOT", "C:/Users/jonat/OneDrive/Documents/research_portfolio_complete"))
SCREENS = ROOT / "AI_ML" / "chat_screens_claude"
A = "C:/ml_artifacts/chat_screens_claude/"


def lines(p):
    with open(p, encoding="utf-8") as f:
        return [x for x in f.read().split("\n") if x.strip()]


def jl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run(rows_in, scratch):
    """mixed_slot rows through the filter. Returns (kept rows, report)."""
    sys.path.insert(0, str(SCREENS))
    sys.path.insert(0, str(ROOT / "microtorch" / "tools"))
    import build_a19 as B
    import build_screens as b
    import correction_split as CS
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    eval_repo = str(ROOT / "tinychat-eval")
    B.COMMON_LEAF = 0
    pre = scratch / "a19_pre"
    if not (pre / "filter_report.json").exists():
        B.cmd_filter(argparse.Namespace(mixed=A + "packets/a19/pool", a17=A + "packets/a17", out=str(pre), eval_repo=eval_repo))
    ind, empty = scratch / "input", scratch / "empty"
    ind.mkdir(exist_ok=True)
    empty.mkdir(exist_ok=True)
    for f in ("pool.jsonl", "heldout.jsonl"):
        (empty / f).write_text("", encoding="utf-8")
    jl(ind / "pool.jsonl", rows_in)
    (ind / "heldout.jsonl").write_text("", encoding="utf-8")
    B.cmd_filter(argparse.Namespace(mixed=str(ind), a17=str(empty), out=str(scratch / "te_pre"), eval_repo=eval_repo))
    leaves, sizes = CS.short_strings(b)
    pre_rows = {k: [json.loads(x) for x in lines(pre / f"{f}.jsonl")] for k, f in (("mixed/pool", "mixed_pool"), ("stance/pool", "stance_pool"), ("mixed/heldout", "mixed_heldout"), ("stance/heldout", "stance_heldout"))}
    df = collections.Counter()
    total = sum(len(v) for v in pre_rows.values())
    held = {}
    for k, rows in pre_rows.items():
        held[k] = [CS.found_in(r, b, leaves, sizes) for r in rows]
        for f in held[k]:
            df.update(f)
    rare = {x for x, c in df.items() if c < 0.01 * total}
    common = set(df) - rare
    a19rep = json.loads(Path(A + "packets/a19/filtered/filter_report.json").read_text(encoding="utf-8"))
    for k, rows in pre_rows.items():
        bad = {r["provenance"]["pair_id"] for r, f in zip(rows, held[k]) if f & rare}
        kept = sum(1 for r in rows if r["provenance"]["pair_id"] not in bad)
        if kept != a19rep["summary"][k]["rows_kept"]:
            raise SystemExit(f"the rewritten rule does not reproduce A19 on {k}")
    del pre_rows, held
    rows = [json.loads(x) for x in lines(scratch / "te_pre" / "mixed_pool.jsonl")]
    bad_pairs = {r["provenance"]["pair_id"] for r in rows if CS.found_in(r, b, leaves, sizes) - common}
    keep = [r for r in rows if r["provenance"]["pair_id"] not in bad_pairs]
    fr = json.loads((scratch / "te_pre" / "filter_report.json").read_text(encoding="utf-8"))
    steps = {k: {"ids_dropped": st["mixed/pool"].get("ids_dropped"), "rows_kept": st["mixed/pool"].get("rows_kept")} for k, st in fr["steps"].items() if isinstance(st, dict) and "mixed/pool" in st}
    return keep, {"steps": steps, "rows_before_last_rule": len(rows), "pairs_dropped_by_last_rule": len(bad_pairs), "rows_kept": len(keep)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("--rows", "--out", "--scratch", "--report"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in lines(a.rows)]
    keep, rep = run(rows, a.scratch)
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    jl(a.out, keep)
    start = collections.Counter(r["provenance"]["family"] for r in rows)
    kept = collections.Counter(r["provenance"]["family"] for r in keep)
    rep["rows_in"] = len(rows)
    rep["kept_by_family"] = {f: [kept[f], start[f]] for f in sorted(start)}
    Path(a.report).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
