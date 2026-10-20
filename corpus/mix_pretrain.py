#!/usr/bin/env python3
"""Pretraining text from several sources at set ratios, in two parts: the main mix and an
annealing tail with its own ratios (lever 2, LOCAL_LLM_STACK.md section 7).

    python corpus/mix_pretrain.py OUT_PREFIX \
        --source fineweb=corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt \
        --source soda=corpus/data/dialogue_soda_v1/text.txt  ... \
        --main fineweb=0.7,soda=0.1,oasst=0.05,ultrachat=0.1,everyday=0.05 \
        --tail fineweb=0.5,soda=0.2,oasst=0.1,ultrachat=0.15,everyday=0.05 \
        --tail-fraction 0.15 --total-bytes 2000000000 --seed 1

Source files hold documents that each end with "<|endoftext|>" and a newline (fineweb_edu.py's
and dialogue_pt.py's format). Ratios and budgets are in bytes of document text; a ratio of 0
or an absent source takes nothing from it. --total-bytes max takes the largest total every
budget fits.

Each source's documents are taken in a seeded random order with one cursor across both parts,
so no document appears twice; within each part the chosen documents are interleaved by a
seeded shuffle at document granularity, never in blocks. Outputs:
  OUT_PREFIX.main.txt, OUT_PREFIX.tail.txt   the two parts
  OUT_PREFIX.ledger.json                     parameters, per source and part documents and
                                             bytes, input and output sha256
The parts are separate files because mtstudio draws training windows at random positions of
its corpus: an annealing tail only works if the trainer switches to the tail file for the
last tail-fraction of its steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from array import array

SEP = b"<|endoftext|>\n"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def index_documents(path, chunk=1 << 24):
    """(starts, ends) byte offsets of every document (separator included)."""
    starts, ends = array("Q"), array("Q")
    pos, start, tail = 0, 0, b""
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            buf = tail + block
            base = pos - len(tail)
            i = 0
            while True:
                j = buf.find(SEP, i)
                if j < 0:
                    break
                end = base + j + len(SEP)
                starts.append(start)
                ends.append(end)
                start = end
                i = j + len(SEP)
            tail = buf[max(i, len(buf) - len(SEP) + 1):]
            pos += len(block)
    if start != pos:
        raise ValueError(f"{path}: {pos - start} bytes after the last document separator")
    return starts, ends


def parse_ratios(text, names):
    out = {}
    for item in text.split(","):
        k, _, v = item.partition("=")
        if k not in names:
            raise SystemExit(f"ratio for unknown source {k!r}")
        out[k] = float(v)
    total = sum(out.values())
    if total <= 0 or any(v < 0 for v in out.values()):
        raise SystemExit(f"bad ratios {text!r}")
    return {k: out.get(k, 0.0) / total for k in names}


def mix(out_prefix, sources, main, tail, tail_fraction, total_bytes, seed):
    names = list(sources)
    index = {s: index_documents(p) for s, p in sources.items()}
    avail = {s: sum(e - b for b, e in zip(*index[s])) for s in names}
    parts = {"main": (1.0 - tail_fraction, main), "tail": (tail_fraction, tail)}
    if total_bytes == "max":
        need = {s: sum(frac * r[s] for frac, r in parts.values()) for s in names}
        total_bytes = int(min(avail[s] / need[s] for s in names if need[s] > 0))
    order = {}
    for s in names:
        perm = list(range(len(index[s][0])))
        random.Random(f"{seed}:order:{s}").shuffle(perm)
        order[s] = perm
    cursor = {s: 0 for s in names}
    files = {s: open(p, "rb") for s, p in sources.items()}
    ledger = {"seed": seed, "tail_fraction": tail_fraction, "total_bytes_target": total_bytes,
              "unit": "bytes of document text", "sources": {}, "parts": {}}
    try:
        for part, (frac, ratios) in parts.items():
            picks, per = [], {}
            for s in names:
                budget = int(total_bytes * frac * ratios[s])
                got = n = 0
                starts, ends = index[s]
                while got < budget:
                    if cursor[s] >= len(order[s]):
                        raise SystemExit(f"{s}: out of documents in part {part} ({got} of {budget} bytes)")
                    d = order[s][cursor[s]]
                    cursor[s] += 1
                    picks.append((s, d))
                    got += ends[d] - starts[d]
                    n += 1
                per[s] = {"ratio": ratios[s], "budget_bytes": budget, "documents": n, "bytes": got}
            random.Random(f"{seed}:interleave:{part}").shuffle(picks)
            path = f"{out_prefix}.{part}.txt"
            with open(path, "wb") as out:
                for s, d in picks:
                    f = files[s]
                    f.seek(index[s][0][d])
                    out.write(f.read(index[s][1][d] - index[s][0][d]))
            ledger["parts"][part] = {"fraction": frac, "file": os.path.basename(path), "sources": per,
                                     "documents": len(picks), "bytes": os.path.getsize(path),
                                     "sha256": sha256_file(path)}
    finally:
        for f in files.values():
            f.close()
    for s, p in sources.items():
        ledger["sources"][s] = {"path": p, "sha256": sha256_file(p), "documents": len(index[s][0]),
                                "bytes": avail[s], "documents_used": cursor[s]}
    with open(f"{out_prefix}.ledger.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(ledger, f, indent=1)
    return ledger


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mix pretraining sources into a main part and an annealing tail.")
    ap.add_argument("out_prefix")
    ap.add_argument("--source", action="append", required=True, help="NAME=PATH (repeat)")
    ap.add_argument("--main", required=True, help="NAME=RATIO,... for the main part")
    ap.add_argument("--tail", required=True, help="NAME=RATIO,... for the annealing tail")
    ap.add_argument("--tail-fraction", type=float, default=0.15)
    ap.add_argument("--total-bytes", default="max", help="an integer, or max")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    sources = dict(s.split("=", 1) for s in a.source)
    if not 0.0 <= a.tail_fraction < 1.0:
        raise SystemExit("--tail-fraction must be in [0, 1)")
    total = a.total_bytes if a.total_bytes == "max" else int(a.total_bytes)
    ledger = mix(a.out_prefix, sources, parse_ratios(a.main, sources), parse_ratios(a.tail, sources),
                 a.tail_fraction, total, a.seed)
    print(json.dumps({p: {"bytes": v["bytes"], "documents": v["documents"]} for p, v in ledger["parts"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
