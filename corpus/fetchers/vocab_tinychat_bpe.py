#!/usr/bin/env python3
"""The chat model's tokenizer: byte-level BPE, 8192 entries, trained by
tools/bpe.py on tinychat_hf_sample_v1's pretraining text plus the chat
specials. Training is deterministic, so the vocabulary is pinned by the
hash of the GGUF it writes.

    python corpus/fetchers/vocab_tinychat_bpe.py [--pin]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common as C  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(C.CORPUS_DIR), "tools"))
import bpe  # noqa: E402

CID = "vocab_tinychat_bpe8k_v1"
SOURCE = ("tinychat_hf_sample_v1", "pretrain_train.txt")
SIZE = 8192


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    if C.already_done(CID, ["tinychat_bpe8k.gguf"]):
        print(f"{CID}: present and verified")
        return 0
    if not C.already_done(SOURCE[0], [SOURCE[1]]):
        C.die(f"{SOURCE[0]} missing or unverified: run corpus/fetchers/tinychat_hf.py sample_v1")
    with open(os.path.join(C.DATA_DIR, *SOURCE), encoding="utf-8") as f:
        text = f.read()
    tok = bpe.train(text, SIZE)
    out = C.out_path(CID, "tinychat_bpe8k.gguf")
    bpe.write_gguf(out, tok)
    held = text[-200000:]
    print(f"{CID}: {len(tok.tokens)} tokens; {len(held.encode()) / len(tok.encode(held)):.2f} bytes/token on the "
          f"last 200k characters")
    try:
        C.finish(CID, [out], f"tools/bpe.py train on {SOURCE[0]}/{SOURCE[1]}, vocab {SIZE}", pin=a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
