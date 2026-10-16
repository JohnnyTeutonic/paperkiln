#!/usr/bin/env python3
"""Stratified audit of scored records (charter 0001, Phase 2).

    python tools/posttrain/audit_sample.py draw  SCORED.jsonl --n 40 --seed 1 --out-dir AUDIT/
    #  ... a person edits AUDIT/audit.jsonl: for each item they review, set
    #      "audited": true, "audit_reward": 0 or 1, "audit_rationale", "annotator"
    python tools/posttrain/audit_sample.py merge SCORED.jsonl AUDIT/audit.jsonl --out-dir AUDIT/

`draw` stratifies by (evidence kind, reward bucket, accepted). The buckets
are abstain, 0, between, and 1. Every non-empty stratum gets at least one
item and the rest is allocated in proportion. It writes audit.jsonl (the
editable file) and audit_sheet.md (for reading). Each item carries the
sha256 of the record it was drawn from, so an edited item cannot be merged
onto a different record.

`merge` validates every item marked audited. It needs a 0/1 verdict, a
non-empty rationale and an annotator, and fails loudly otherwise. It
writes:
  human_audit.jsonl      the HumanAuditVerifier format (prompt, completion,
                         reward, rationale, annotator): audited verdicts as
                         evaluation evidence
  scored.audited.jsonl   the scored records with "audited" set where a
                         person looked, plus the audit verdict beside the
                         machine one. It is never written over the input.
  audit_summary.json     agreement between the machine verdict and the
                         person's, per evidence kind
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record_sha(r):
    core = {k: r[k] for k in ("prompt", "completion", "reward", "rationale", "evidence_kind", "evidence_uri")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bucket(reward):
    if reward < 0:
        return "abstain"
    if reward == 0:
        return "0"
    if reward == 1:
        return "1"
    return "between"


def stratum(r):
    return (r["evidence_kind"], bucket(r["reward"]), bool(r.get("accepted", False)))


def allocate(sizes, n):
    """At least one per non-empty stratum, the rest by largest remainder."""
    keys = sorted(k for k, s in sizes.items() if s > 0)
    if n < len(keys):
        raise ValueError(f"--n {n} is smaller than the {len(keys)} non-empty strata")
    total = sum(sizes[k] for k in keys)
    n = min(n, total)
    alloc = {k: 1 for k in keys}
    rest = n - len(keys)
    quotas = {k: rest * sizes[k] / total for k in keys}
    for k in keys:
        alloc[k] += min(int(quotas[k]), sizes[k] - 1)
    left = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quotas[k] - int(quotas[k])), k)):
        if left <= 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            left -= 1
    return alloc


def draw(scored, n, seed):
    groups = {}
    for i, r in enumerate(scored):
        groups.setdefault(stratum(r), []).append(i)
    alloc = allocate({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    items = []
    for k in sorted(alloc):
        for i in sorted(rng.sample(groups[k], alloc[k])):
            r = scored[i]
            items.append({"audit_id": f"a{len(items):04d}", "record_index": i, "record_sha256": record_sha(r),
                          "stratum": list(k), "prompt": r["prompt"], "completion": r["completion"],
                          "machine_reward": r["reward"], "machine_rationale": r["rationale"],
                          "evidence_kind": r["evidence_kind"], "evidence_uri": r["evidence_uri"],
                          "audited": False, "audit_reward": None, "audit_rationale": "", "annotator": ""})
    return items, alloc


def sheet(items, alloc):
    L = ["# Audit sheet", "",
         "Edit `audit.jsonl`, not this file. For each item reviewed set `audited: true`, `audit_reward` (0 or 1),",
         "`audit_rationale` (why) and `annotator`. Then run `audit_sample.py merge`.", "",
         "Strata (evidence kind, reward bucket, accepted) -> items: " +
         "; ".join(f"{k} -> {v}" for k, v in sorted(alloc.items())), ""]
    for it in items:
        L += [f"## {it['audit_id']}  `{it['evidence_kind']}`  machine reward {it['machine_reward']}", "",
              f"- prompt: {it['prompt']}", f"- completion: {it['completion']}",
              f"- machine rationale: {it['machine_rationale']}", f"- evidence: `{it['evidence_uri']}`", ""]
    return "\n".join(L)


def merge(scored, items):
    out = [dict(r) for r in scored]
    human, agree = [], {}
    for it in items:
        if not it.get("audited"):
            continue
        i = it["record_index"]
        if not 0 <= i < len(scored) or record_sha(scored[i]) != it["record_sha256"]:
            raise SystemExit(f"{it['audit_id']}: does not match the scored record it was drawn from")
        if it.get("audit_reward") not in (0, 1) or isinstance(it.get("audit_reward"), bool):
            raise SystemExit(f"{it['audit_id']}: audit_reward must be 0 or 1")
        if not str(it.get("audit_rationale", "")).strip():
            raise SystemExit(f"{it['audit_id']}: an audit needs a rationale")
        if not str(it.get("annotator", "")).strip():
            raise SystemExit(f"{it['audit_id']}: an audit needs an annotator")
        out[i]["audited"] = True
        out[i]["audit_reward"] = it["audit_reward"]
        out[i]["audit_rationale"] = it["audit_rationale"]
        out[i]["annotator"] = it["annotator"]
        human.append({"prompt": it["prompt"], "completion": it["completion"], "reward": it["audit_reward"],
                      "rationale": it["audit_rationale"], "annotator": it["annotator"]})
        a = agree.setdefault(it["evidence_kind"], {"audited": 0, "agree": 0, "machine_abstained": 0})
        a["audited"] += 1
        if scored[i]["reward"] < 0:
            a["machine_abstained"] += 1
        elif float(scored[i]["reward"] >= 0.5) == float(it["audit_reward"]):
            a["agree"] += 1
    return out, human, agree


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("scored")
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--out-dir", required=True)
    m = sub.add_parser("merge")
    m.add_argument("scored")
    m.add_argument("audit")
    m.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    scored = read_jsonl(a.scored)
    if a.cmd == "draw":
        items, alloc = draw(scored, a.n, a.seed)
        write_jsonl(os.path.join(a.out_dir, "audit.jsonl"), items)
        with open(os.path.join(a.out_dir, "audit_sheet.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(sheet(items, alloc))
        print(f"{len(items)} items drawn from {len(alloc)} strata -> {a.out_dir}")
        return 0
    out, human, agree = merge(scored, read_jsonl(a.audit))
    write_jsonl(os.path.join(a.out_dir, "scored.audited.jsonl"), out)
    write_jsonl(os.path.join(a.out_dir, "human_audit.jsonl"), human)
    with open(os.path.join(a.out_dir, "audit_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"audited": len(human), "by_evidence_kind": agree}, f, indent=1)
    print(f"{len(human)} audited items merged -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
