#!/usr/bin/env python3
"""Vocabulary-coverage probe of candidate corpora (corpus ledger 0002).

    python experiments/corpus_probe/probe.py fetch    # samples via the HF rows API
    python experiments/corpus_probe/probe.py measure  # coverage table -> RESULTS.json

A probe, not a corpus: it answers one question before any fetcher is written.
How much of each candidate can a 4,096-entry word-level vocabulary represent?
The answer decides whether a candidate is worth a C1 arm. Each sample is the
first N rows the Hugging Face datasets-server returns for the named
config/split. That API serves the dataset's head, so the head revision is
recorded beside the sample at fetch time, and the sample's sha256 is the
receipt.

Three vocabularies per candidate:
  research   vocab_chat7b_cap4096_v1 (every finding so far)
  quickstart vocab_tinystories_valid_v1
  own        a 4,096 frequency vocabulary built on the first half of the
             sample's documents, measured on the second half (no leakage)
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import corpus_quality as cq  # noqa: E402

N_ROWS = 1000
CANDIDATES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    "simplewiki": ("wikimedia/wikipedia", "20231101.simple", "train", "text"),
}
VOCABS = {
    "research": os.path.join(ROOT, "corpus/data/vocab_chat7b_cap4096_v1/chat7b_cap4096.gguf"),
    "quickstart": os.path.join(ROOT, "corpus/data/vocab_tinystories_valid_v1/tinystories_vocab.gguf"),
}
REFERENCE = {"tinystories_slice": os.path.join(ROOT, "corpus/data/tinystories_slice_v1/train-0-small.txt")}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def fetch():
    for name, (ds, config, split, field) in CANDIDATES.items():
        out = os.path.join(HERE, "samples", f"{name}.txt")
        if os.path.exists(out):
            print(f"{name}: sample present")
            continue
        head = get_json(f"https://huggingface.co/api/datasets/{ds}")["sha"]
        docs = []
        for off in range(0, N_ROWS, 100):
            url = (f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={config}"
                   f"&split={split}&offset={off}&length=100")
            for attempt in range(4):
                try:
                    rows = get_json(url)["rows"]
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
            docs += [" ".join(str(r["row"][field]).split()) for r in rows]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(d + "\n" for d in docs if d))
        meta = {"dataset": ds, "config": config, "split": split, "head_revision_at_fetch": head,
                "rows": len(docs), "fetched": time.strftime("%Y-%m-%d"), "sha256": cq.sha256_file(out)}
        with open(out + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, indent=1)
        print(name, meta)


def own_vocab(train_bytes):
    counts = collections.Counter(w for w in cq.tokenize_bytes(train_bytes, {}, 0)[1])
    eos = cq.EOS
    words = [w for w, _ in counts.most_common() if all(b < 128 for b in w) and w != eos]
    return [b"<unk>"] + words[:4095]


def measure():
    results = {}
    targets = {n: os.path.join(HERE, "samples", f"{n}.txt") for n in CANDIDATES}
    targets.update(REFERENCE)
    for name, path in targets.items():
        with open(path, "rb") as f:
            data = f.read()
        docs = cq.split_documents(data)
        half = len(docs) // 2
        train, test = b"\n".join(docs[:half]) + b"\n", b"\n".join(docs[half:]) + b"\n"
        row = {"sha256": cq.sha256_file(path), "documents": len(docs),
               "doc_tokens": cq.percentiles([len(cq.tokenize_bytes(d, {}, 0)[0]) for d in docs])}
        for vname, vpath in VOCABS.items():
            c = cq.coverage(test, cq.read_gguf_tokens(vpath), 4096, 0)
            row[vname] = {"unk_rate": round(c["unk_rate"], 4), "tokens_per_byte": round(c["tokens_per_byte"], 4),
                          "top_unk": c["top_unk"][:8]}
        c = cq.coverage(test, own_vocab(train), 4096, 0)
        row["own"] = {"unk_rate": round(c["unk_rate"], 4), "tokens_per_byte": round(c["tokens_per_byte"], 4),
                      "top_unk": c["top_unk"][:8]}
        row["measured_on"] = "second half of the documents"
        results[name] = row
        print(f"{name:18s} research {row['research']['unk_rate']:.3f}  quickstart "
              f"{row['quickstart']['unk_rate']:.3f}  own {row['own']['unk_rate']:.3f}  docs {len(docs)}")
    with open(os.path.join(HERE, "RESULTS.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    {"fetch": fetch, "measure": measure}[sys.argv[1]]()
