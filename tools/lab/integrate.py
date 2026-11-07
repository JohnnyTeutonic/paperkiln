#!/usr/bin/env python3
"""The wrapped frozen code on real files, counts only. CPU, read-only, no model. Claude Opus 5.5, 7 Nov 2026.

Run through the CPU budget from WSL:
  python3 scripts/cpu_budget.py run --session claude --job lab-integrate --cores 2 --resume "rerun" -- python3 tools/lab/integrate.py --out REPORT.json

Checks: the pins; the independent row checker on T_C and T_M; the parts rows claim; family_exposure's recount against its stored total; mine_score through the wrap against the stored S34 round; the sealed pin and the rubric pool's
groups against it. Prints counts and ids of checks. No row text, no reply and no sealed id leaves this script.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters as AD  # noqa: E402
import firewall as FW  # noqa: E402
import miner as M  # noqa: E402
import rowcheck as RC  # noqa: E402

ART = Path(os.environ.get("LAB_ARTEFACTS", "/mnt/c/ml_artifacts/chat_screens_claude"))


def rows(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def check_rows():
    out = {}
    for name, rel, allowed in (("T_C", "packets/correction_v1/candidate_pool.jsonl", {"T_C"}), ("T_M", "packets/correction_v1/mine_rows.jsonl", {"T_M"})):
        rs = rows(ART / rel)
        FW.check_row_parts(rs, allowed, lambda r: r["provenance"].get("correction_part"))
        out[name] = RC.check_rows(rs)
        out[name]["families"] = dict(sorted(collections.Counter(r["provenance"]["family"] for r in rs).items()))
    return out


def check_exposure():
    root = AD.screens_dir()
    stored = json.loads((root / "analysis/family_exposure.json").read_text(encoding="utf-8"))["checks"]["sampled_supervised_by_leg"]["a19/M1/train.jsonl|seed2|1024"]
    counts = AD.leg_counts(ART / "packets/a19/M1/train.jsonl", ART / "runs/a16-s1-S34-20261106/S34/model.gguf", 2, 1024)
    got = sum(v["sampled_supervised"] for v in counts.values())
    AD_exposure = AD.to_exposure(counts)
    return {"stored_sampled_supervised": stored, "recounted": got, "equal": got == stored, "families": len(AD_exposure)}


def check_miner():
    ms = AD.mine_score()
    pool = rows(ART / "packets/a18/filtered/mixed_pool.jsonl")
    gold = rows(ART / "correction_round_s34/gold.jsonl")
    cells = M.rerun_stored_round(gold, lambda r: pool[r]["provenance"]["turns"], ms.score)
    stored = json.loads((AD.screens_dir() / "analysis/correction_base_receipts/mine_report.json").read_text(encoding="utf-8"))["cells"]
    diff = [k for k, v in cells.items() if round(v["fail"] / v["n"], 3) != stored[k]["fail_rate"]]
    return {"cells": len(cells), "stored_cells": len(stored), "cells_differing": len(diff)}


def check_sealed():
    sealed = FW.sealed_groups(ART / "rel_instrument/analytic/sealed_templates.json")
    pool = json.loads((AD.screens_dir() / "analysis/analytic_rubric_v2_pool.json").read_text(encoding="utf-8"))["turns"]
    used = {t["group"] for t in pool}
    return FW.check_against_sealed(used, sealed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    a = p.parse_args()
    rep = {"pins": AD.verify_pins(), "rows": check_rows(), "exposure": check_exposure(), "miner": check_miner(), "sealed": check_sealed()}
    text = json.dumps(rep, indent=1, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
