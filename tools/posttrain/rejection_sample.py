#!/usr/bin/env python3
"""Rejection sampling with receipts (charter 0001, Phase 2).

    python tools/posttrain/rejection_sample.py \\
        --mtstudio build/mtstudio --verify build/posttrain_verify \\
        --spec run/spec.json --prompts corpus/eval/tinyarith_r20_dev_v1.jsonl \\
        --verifier exact_match --n 8 --temp 1.0 --seed 17 --keep top1 \\
        --out-dir runs/rs_exact

1. `mtstudio sample-set` draws N completions per prompt by pure temperature
   sampling (no top-k, no banned tokens), so each record carries its policy
   log-probability.
2. `posttrain_verify <kind>` attaches a VerifierResult to every record:
   reward, rationale, evidence kind and URI, audited flag.
3. Selection keeps records, never rewrites them. `--keep topK` keeps the K
   best per prompt (ties to the lower sample index); `--keep ge:X` keeps
   rewards >= X. Abstentions are never kept.

Outputs in --out-dir: samples.jsonl (raw), scored.jsonl (every record,
with "accepted"), accepted.jsonl, summary.json. Each record also gets
`checkpoint_sha256` and `spec_sha256`, so a record names the exact weights
that produced it. Verifiers that need reference answers (exact_match,
synthetic_judge) get them from the prompt file's "answer" and "uri" fields
unless --refs is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

REF_KINDS = {"exact_match", "synthetic_judge"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_keep(s):
    if s.startswith("top"):
        k = int(s[3:])
        if k < 1:
            raise ValueError("--keep topK needs K >= 1")
        return ("top", k)
    if s.startswith("ge:"):
        return ("ge", float(s[3:]))
    raise ValueError(f"--keep must be topK or ge:X, got {s!r}")


def select(records, keep):
    """Mark each record accepted or not. Pure function of the scored records."""
    mode, arg = keep
    by_prompt = {}
    for i, r in enumerate(records):
        by_prompt.setdefault(r["prompt_index"], []).append(i)
    accepted = set()
    for idxs in by_prompt.values():
        live = [i for i in idxs if records[i]["reward"] >= 0]
        if mode == "top":
            live.sort(key=lambda i: (-records[i]["reward"], records[i]["sample_index"]))
            accepted.update(live[:arg])
        else:
            accepted.update(i for i in live if records[i]["reward"] >= arg)
    return [dict(r, accepted=i in accepted) for i, r in enumerate(records)]


def summarise(records, keep):
    n = len(records)
    abst = sum(r["reward"] < 0 for r in records)
    scored = [r["reward"] for r in records if r["reward"] >= 0]
    acc = [r for r in records if r["accepted"]]
    prompts = {r["prompt_index"] for r in records}
    return {"records": n, "prompts": len(prompts), "abstained": abst,
            "mean_reward": sum(scored) / len(scored) if scored else None,
            "accepted": len(acc), "prompts_with_an_accepted_record": len({r["prompt_index"] for r in acc}),
            "accepted_mean_reward": sum(r["reward"] for r in acc) / len(acc) if acc else None,
            "stop_eos": sum(r.get("stop") == "eos" for r in records),
            "keep": f"{keep[0]}:{keep[1]}",
            "evidence_kinds": sorted({r["evidence_kind"] for r in records}),
            "verifiers": sorted({r["verifier"] for r in records})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--verify", required=True, help="the posttrain_verify binary")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--verifier-arg", action="append", default=[],
                    help="extra posttrain_verify arguments, e.g. --verifier-arg=--seed --verifier-arg=3")
    ap.add_argument("--refs")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--keep", default="top1")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    keep = parse_keep(a.keep)
    os.makedirs(a.out_dir, exist_ok=True)
    samples = os.path.join(a.out_dir, "samples.jsonl")
    subprocess.run([a.mtstudio, "sample-set", a.spec, "--prompts", a.prompts, "--out", samples,
                    "--n", str(a.n), "--temp", repr(a.temp), "--seed", str(a.seed),
                    "--max-new", str(a.max_new)], check=True)
    vargs = list(a.verifier_arg)
    if a.verifier in REF_KINDS:
        refs = a.refs
        if not refs:
            refs = os.path.join(a.out_dir, "refs.jsonl")
            write_jsonl(refs, [{"prompt": p["prompt"], "answer": p["answer"], "uri": p["uri"]}
                               for p in read_jsonl(a.prompts)])
        vargs = ["--refs", refs] + vargs
    with open(samples, "rb") as fin:
        out = subprocess.run([a.verify, a.verifier, *vargs], stdin=fin, capture_output=True, check=True).stdout
    records = [json.loads(x) for x in out.decode("utf-8").splitlines() if x.strip()]
    ck = records[0]["checkpoint"] if records else None
    ck_sha = sha256_file(ck) if ck and os.path.exists(ck) else None
    sp_sha = sha256_file(a.spec)
    for r in records:
        r["checkpoint_sha256"] = ck_sha
        r["spec_sha256"] = sp_sha
    records = select(records, keep)
    write_jsonl(os.path.join(a.out_dir, "scored.jsonl"), records)
    write_jsonl(os.path.join(a.out_dir, "accepted.jsonl"), [r for r in records if r["accepted"]])
    summary = summarise(records, keep)
    summary.update({"spec": a.spec, "spec_sha256": sp_sha, "checkpoint_sha256": ck_sha, "prompts_file": a.prompts,
                    "prompts_sha256": sha256_file(a.prompts), "n": a.n, "temperature": a.temp, "seed": a.seed})
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
