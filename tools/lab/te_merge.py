#!/usr/bin/env python3
"""T_E stage A as registered: filter survivors from the first set and the top-up, up to 50 pairs a family. Claude Opus 5.5, 7 Nov 2026, written before the top-up exists.

The rule is fixed here and selects on nothing but the filter. Files are read in the order given, rows in file order. Each pair goes to its cell, by family, operation and context, until the cell holds its number from the brief's table. A cell short of pairs stays short, is reported, and borrows from no other cell, so
the operation mix holds. Pair ids are prefixed by file, because two authors may reuse one.

  python3 tools/lab/te_merge.py --in FILTERED_A.jsonl FILTERED_TOPUP.jsonl --out MERGED.jsonl --receipt RECEIPT.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import Malformed, sha_file  # noqa: E402
from te_verify import STAGE_A, pairs_of  # noqa: E402


def merge(sets, spec):
    """sets: [(tag, rows)]. Returns (kept rows, shortfall by cell, kept by source)."""
    room = {(fam, op, c): n for fam, cells in spec.items() for (op, c), n in cells.items()}
    kept, by_source = [], collections.Counter()
    for tag, rows in sets:
        by = pairs_of(rows)
        for (fam, pid), v in by.items():   # dicts keep file order
            if set(v) != {"a", "b"}:
                raise Malformed(f"a filtered file holds half a pair: {pid}")
            b = v["b"]
            cell = (fam, b["provenance"]["values"].get("pair_op") or b["decision"]["k"], b["decision"]["c"])
            if room.get(cell, 0) <= 0:
                continue
            room[cell] -= 1
            by_source[tag] += 1
            for x in (v["a"], v["b"]):
                r = json.loads(json.dumps(x))
                r["provenance"]["pair_id"] = f"{tag}_{pid}"
                r["conversation_id"] = f"{tag}-{r['conversation_id']}"
                kept.append(r)
    return kept, {f"{k[0]}/{k[1]}/{k[2]}": n for k, n in sorted(room.items()) if n > 0}, dict(by_source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    sets = []
    for i, f in enumerate(a.files):
        with open(f, encoding="utf-8") as h:
            sets.append((f"s{i}", [json.loads(x) for x in h]))
    kept, short, by_source = merge(sets, STAGE_A)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fam = collections.Counter(r["provenance"]["family"] for r in kept)
    rec = {"producer": "Claude Opus 5.5, 7 Nov 2026", "inputs_sha256": {f: sha_file(f) for f in a.files}, "output_sha256": sha_file(a.out), "rows": len(kept), "rows_by_family": dict(sorted(fam.items())), "pairs_by_source": by_source, "shortfall_by_cell": short}
    Path(a.receipt).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
