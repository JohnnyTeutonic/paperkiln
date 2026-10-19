#!/usr/bin/env python3
"""smol-smoltalk (HuggingFaceTB/smol-smoltalk, Apache-2.0): Hugging Face's SFT
mix for its smallest SmolLM2 models, used here to teach the general-text chat
model simple question answering and greetings, which TinyChat (emotional
small talk only) cannot (experiments/chat_general/PLAN.md).

    python corpus/fetchers/smol_smoltalk.py qa_v1 [--pin]

Kept sources: smol-magpie-ultra-short (everyday questions, short answers),
openhermes-50k (general-knowledge Q&A) and everyday-conversations (greetings,
small talk). Dropped: code (self-oss-instruct: a 256-token context mangles
it), email summarising and rewriting (long inputs), and smol-contraints
(format-constraint prompts the model would only parrot). System messages are
dropped; only alternating user/assistant turns from the first user turn are
kept. The SFT builder (tools/chat_data.py) drops examples over the context.

Outputs in corpus/data/smol_smoltalk_qa_v1/: dialogues_{train,val,test}.jsonl
({"messages": [...], "uri", "source"}), split 98/1/1 by a hash of the first
user turn, independent of order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

REPO = "HuggingFaceTB/smol-smoltalk"
REV = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
SHARDS = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
KEEP = {"smol-magpie-ultra-short", "openhermes-50k", "everyday-conversations"}


def split_of(text):
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h == 0 else ("val" if h == 1 else "train")


def turns(messages):
    """Alternating user/assistant turns from the first user turn, or None."""
    msgs = [m for m in messages if m["role"] in ("user", "assistant")]
    out = []
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want:
            break
        text = m["content"].strip()
        if not text:
            break
        out.append({"role": m["role"], "content": text})
    if len(out) % 2:
        out = out[:-1]
    return out or None


def build(pin):
    cid = "smol_smoltalk_qa_v1"
    names = ["dialogues_train.jsonl", "dialogues_val.jsonl", "dialogues_test.jsonl"]
    if C.already_done(cid, names):
        print(f"{cid}: present and verified")
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    outs = {k: open(C.out_path(cid, f"dialogues_{k}.jsonl"), "w", encoding="utf-8", newline="\n")
            for k in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0, "dropped_source": 0, "dropped_shape": 0}
    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REV)
        f = pq.ParquetFile(path)
        row = 0
        for batch in f.iter_batches(batch_size=4096, columns=["messages", "source"]):
            d = batch.to_pydict()
            for msgs, src in zip(d["messages"], d["source"]):
                row += 1
                if src not in KEEP:
                    counts["dropped_source"] += 1
                    continue
                t = turns(msgs)
                if t is None:
                    counts["dropped_shape"] += 1
                    continue
                sp = split_of(t[0]["content"])
                outs[sp].write(json.dumps({"messages": t, "uri": f"{REPO}@{REV[:8]}/{shard}#{row}", "source": src},
                                          ensure_ascii=False) + "\n")
                counts[sp] += 1
    for f in outs.values():
        f.close()
    print(f"{cid}: {counts}")
    C.finish(cid, [C.out_path(cid, n) for n in names],
             f"{REPO}@{REV} train shards 0-3, sources {sorted(KEEP)}, split 98/1/1 by sha256 of the first user turn",
             pin=pin)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch the QA/conversation part of smol-smoltalk at a pinned revision.")
    ap.add_argument("target", choices=["qa_v1"])
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    try:
        build(a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
