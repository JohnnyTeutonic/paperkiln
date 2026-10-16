#!/usr/bin/env python3
"""Frozen held-out text for bits-per-byte evaluation (charter 0001 §8.2, the
fixed evaluation panel).

    python corpus/fetchers/eval_heldout.py           # build and verify against pins
    python corpus/fetchers/eval_heldout.py --pin     # first build only

  corpus/eval/tinystories_heldout_v1.txt  TinyStories train-split stories
      100000..100499 (shard 0, roneneldan/TinyStories@f54c09fd), one per line.
      No corpus in the registry draws on them: the research slice is stories
      0..4999 of the same shard, and the quickstart corpus is the validation
      file.
  corpus/eval/tinychat_heldout_v1.txt     500 TinyChat v2 dialogues at seed 999,
      a seed no registered corpus uses.

The sets are frozen like every file in corpus/eval/. Overlap with every
registered training corpus is measured into corpus/eval/heldout_overlap_v1.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common as C  # noqa: E402
import tinystories as TS  # noqa: E402

ROOT = os.path.dirname(C.CORPUS_DIR)
EVAL = os.path.join(C.CORPUS_DIR, "eval")


def stories(lo=100000, hi=100500):
    import pandas as pd
    texts = pd.read_parquet(TS._download(TS.TRAIN_SHARD0), columns=["text"])["text"]
    out = [" ".join(str(t).split()) for t in texts.iloc[lo:hi]]
    if len(out) != hi - lo or not all(out):
        C.die("shard 0 does not hold the expected stories")
    return out


def chats(n=500, seed=999):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "get_tinychat_data.py"), "--version", "2",
                        "--dialogues", str(n), "--seed", str(seed), "--out-dir", tmp],
                       check=True, stdout=subprocess.DEVNULL)
        with open(os.path.join(tmp, "data", "tinychat.txt"), encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]


def freeze(eid, lines, pin):
    path = os.path.join(EVAL, f"{eid}.txt")
    body = "".join(x + "\n" for x in lines).encode("utf-8")
    want = C.expected_hashes(eid).get(os.path.basename(path))
    if want is None:
        if not pin:
            C.die(f"{eid}: not pinned; run with --pin once")
        with open(path, "wb") as f:
            f.write(body)
        C.pin_hashes(eid, {os.path.basename(path): C.sha256_file(path)}, f"FROZEN evaluation set ({eid})")
        print(f"{eid}: PINNED")
        return path
    if C.hashlib.sha256(body).hexdigest() != want:
        C.die(f"{eid}: the source no longer reproduces the frozen set")
    if not os.path.exists(path) or C.sha256_file(path) != want:
        with open(path, "wb") as f:
            f.write(body)
    print(f"{eid}: OK (frozen, reproduces identically)")
    return path


def overlap(eval_lines, corpus_path, n=8):
    with open(corpus_path, encoding="utf-8", errors="replace") as f:
        train = [" ".join(x.split()) for x in f if x.strip()]
    tl = set(train)
    grams = set()
    for x in train:
        w = x.lower().split()
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    eg = set()
    for x in eval_lines:
        w = x.lower().split()
        eg.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return {"exact_line_overlap": sum(" ".join(x.split()) in tl for x in eval_lines) / len(eval_lines),
            f"{n}gram_overlap": len(eg & grams) / len(eg) if eg else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Frozen held-out evaluation text.")
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(EVAL, exist_ok=True)
    sets = {"tinystories_heldout_v1": stories(), "tinychat_heldout_v1": chats()}
    for eid, lines in sets.items():
        freeze(eid, lines, a.pin)
    corpora = {cid: os.path.join(C.DATA_DIR, cid, name) for cid, name in [
        ("tinystories_slice_v1", "train-0-small.txt"), ("tinystories_valid_v1", "tinystories_valid.txt"),
        ("tinychat_v2_1500", "tinychat.txt"), ("tinychat_v2_4000", "tinychat.txt")]}
    report = {eid: {cid: overlap(lines, p) for cid, p in corpora.items() if os.path.exists(p)}
              for eid, lines in sets.items()}
    with open(os.path.join(EVAL, "heldout_overlap_v1.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
