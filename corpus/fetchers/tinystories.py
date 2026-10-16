#!/usr/bin/env python3
"""TinyStories corpora, from a pinned revision of roneneldan/TinyStories.

    python corpus/fetchers/tinystories.py slice_v1     # the research slice
    python corpus/fetchers/tinystories.py valid_v1     # the quickstart corpus

Source: Eldan & Li (2023), Hugging Face dataset roneneldan/TinyStories,
revision REVISION below (the dataset's head since 12 Aug 2024); licence
cdla-sharing-1.0 as declared in that revision's dataset card.

tinystories_slice_v1 is the corpus behind every Atlas, sparse-attention,
transfer and highway finding: the first 5000 non-empty stories of the
train split, whitespace-collapsed, one per line, with NO end-of-story
token: it was written on 12 Jul 2026 by an early version of
transformer_cpp/scripts/prepare_tinystories.py, before that script began
appending " <|endoftext|>", so stories are separated only by a line break
(whitespace to the tokenizer). It was written on Windows, so its line
endings are CRLF. This fetcher writes both properties explicitly, so the
bytes, and the pinned sha256, are the historical ones.

tinystories_valid_v1 is what tools/get_tinystories_data.py writes for the
quickstart specs: the first 20 MiB of TinyStories-valid.txt, raw bytes.

Requires huggingface_hub (and pandas + pyarrow for slice_v1).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
EOT = "<|endoftext|>"
TRAIN_SHARD0 = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"


def _download(filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION)


def fetch_slice_v1(pin=False, n_stories=5000):
    cid, name = "tinystories_slice_v1", "train-0-small.txt"
    if C.already_done(cid, [name]):
        print(f"{cid}: present and verified")
        return
    import pandas as pd
    texts = pd.read_parquet(_download(TRAIN_SHARD0), columns=["text"])["text"]
    out = C.out_path(cid, name)
    written = 0
    with open(out, "w", encoding="utf-8", newline="\r\n") as f:
        for story in texts:
            line = " ".join(str(story).split())
            if not line:
                continue
            f.write(line + "\n")
            written += 1
            if written >= n_stories:
                break
    if written < n_stories:
        C.die(f"{cid}: shard 0 had only {written} non-empty stories")
    C.finish(cid, [out], f"{REPO}@{REVISION} {TRAIN_SHARD0}, first {n_stories} stories, CRLF",
             pin=pin)


def fetch_valid_v1(pin=False, max_mb=20.0):
    cid, name = "tinystories_valid_v1", "tinystories_valid.txt"
    if C.already_done(cid, [name]):
        print(f"{cid}: present and verified")
        return
    src = _download("TinyStories-valid.txt")
    out = C.out_path(cid, name)
    with open(src, "rb") as f, open(out, "wb") as g:
        g.write(f.read(int(max_mb * 1024 * 1024)))
    C.finish(cid, [out], f"{REPO}@{REVISION} TinyStories-valid.txt, first {max_mb} MiB raw",
             pin=pin)


TARGETS = {"slice_v1": fetch_slice_v1, "valid_v1": fetch_valid_v1}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch pinned TinyStories corpora.")
    ap.add_argument("target", choices=sorted(TARGETS) + ["all"])
    ap.add_argument("--pin", action="store_true",
                    help="first fetch of a new corpus id only: record its hashes")
    a = ap.parse_args(argv)
    try:
        for t in (sorted(TARGETS) if a.target == "all" else [a.target]):
            TARGETS[t](pin=a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
