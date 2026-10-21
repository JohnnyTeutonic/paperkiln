#!/usr/bin/env python3
"""FineWeb-Edu (HuggingFaceFW/fineweb-edu, config sample-10BT): the general
text for the chat model's base (experiments/chat_general/PLAN.md; registry
fineweb_edu_v1, licence odc-by).

    python corpus/fetchers/fineweb_edu.py shard0_v1 [--pin]   # one 2.15 GB shard, ~750M GPT-2 tokens
    python corpus/fetchers/fineweb_edu.py shard1_v1 [--pin]   # the next shards (lever 3), shard1-3
    python corpus/fetchers/fineweb_edu.py sample_v1 [--pin]   # the first 50 MB of shard0's train text

One pinned parquet shard (sample/10BT/00N_00000.parquet at revision REV)
covers far more tokens than a night of training uses; shardN_v1 is shard N
(0-3; 1-3 are lever 3's more pretraining tokens, pinned on their first
fetch, on a VM, with the hashes file brought home). Each document goes
to train or val by a hash of its id (99.5 / 0.5), independent of order.
Outputs in corpus/data/<id>/:
  pretrain_train.txt   train documents, <|endoftext|> between them
  val.txt              val documents, the same format (held-out bits/byte)
sample_v1 keeps only the first SAMPLE_BYTES of shard0's train text, for
tokenizer training and tests on a laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

REPO = "HuggingFaceFW/fineweb-edu"
REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARDS = {f"shard{n}_v1": f"sample/10BT/{n:03d}_00000.parquet" for n in range(4)}  # 14 shards at REV
SAMPLE_BYTES = 50 * 1024 * 1024
EOT = "<|endoftext|>"


def split_of(doc_id):
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return "val" if h < 5 else "train"


def documents(shard):
    """(id, text) in shard order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REV)
    f = pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=2048, columns=["id", "text"]):
        d = batch.to_pydict()
        for i, t in zip(d["id"], d["text"]):
            # A literal end-of-text marker inside a document would read as a
            # document boundary; none is expected, and any is dropped.
            yield i, t.replace(EOT, "")


def build(cid, shard, sample, pin):
    names = ["pretrain_train.txt"] if sample else ["pretrain_train.txt", "val.txt"]
    if C.already_done(cid, names):
        print(f"{cid}: present and verified")
        return
    tr = open(C.out_path(cid, "pretrain_train.txt"), "w", encoding="utf-8", newline="\n")
    va = None if sample else open(C.out_path(cid, "val.txt"), "w", encoding="utf-8", newline="\n")
    counts = {"train": 0, "val": 0}
    written = 0
    for doc_id, text in documents(shard):
        text = text.strip()
        if not text:
            continue
        sp = split_of(doc_id)
        if sample and sp != "train":
            continue
        block = text + "\n" + EOT + "\n"
        if sp == "train":
            if sample and written + len(block.encode("utf-8")) > SAMPLE_BYTES:
                break
            tr.write(block)
            written += len(block.encode("utf-8"))
        else:
            va.write(block)
        counts[sp] += 1
    tr.close()
    if va:
        va.close()
    print(f"{cid}: {counts}")
    C.finish(cid, [C.out_path(cid, n) for n in names],
             f"{REPO}@{REV} {shard}{' first 50 MB of train' if sample else ''}, split 99.5/0.5 by sha256(id)",
             pin=pin)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch one pinned FineWeb-Edu shard.")
    ap.add_argument("target", choices=[*SHARDS, "sample_v1"])
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    try:
        build(f"fineweb_edu_{a.target}", SHARDS.get(a.target, SHARDS["shard0_v1"]), a.target == "sample_v1", a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
