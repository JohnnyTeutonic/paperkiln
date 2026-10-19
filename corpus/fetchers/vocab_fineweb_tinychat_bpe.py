#!/usr/bin/env python3
"""The general-text chat model's tokenizer (experiments/chat_general/PLAN.md):
byte-level BPE, 8192 entries, trained by tools/bpe.py on a mix of 80%
FineWeb-Edu (the base's text) and 20% TinyChat (the SFT's text), plus the
chat specials, so both tokenise well. Training is deterministic, so the
vocabulary is pinned by the hash of the GGUF it writes.

    python corpus/fetchers/vocab_fineweb_tinychat_bpe.py [--pin]

Mix: the first 40 MB of fineweb_edu_sample_v1 and the first 10 MB of
tinychat_hf_sample_v1, each cut at a document boundary. Bytes per token are
reported on 200k characters of each source that the vocabulary was not
trained on (the text just after each cut).
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

CID = "vocab_fineweb_tinychat_bpe8k_v1"
SOURCES = [("fineweb_edu_sample_v1", "pretrain_train.txt", 40 * 1024 * 1024),
           ("tinychat_hf_sample_v1", "pretrain_train.txt", 10 * 1024 * 1024)]
SIZE = 8192
EOT = "<|endoftext|>\n"


def cut(text, nbytes):
    """The longest prefix within nbytes that ends at a document boundary, and the text after it."""
    head = text.encode("utf-8")[:nbytes].decode("utf-8", errors="ignore")
    i = head.rfind(EOT)
    i = len(head) if i < 0 else i + len(EOT)
    return text[:i], text[i:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    if C.already_done(CID, ["fineweb_tinychat_bpe8k.gguf"]):
        print(f"{CID}: present and verified")
        return 0
    parts, held = [], {}
    for cid, name, nbytes in SOURCES:
        if not C.already_done(cid, [name]):
            C.die(f"{cid} missing or unverified: fetch it first")
        with open(os.path.join(C.DATA_DIR, cid, name), encoding="utf-8") as f:
            train, rest = cut(f.read(), nbytes)
        parts.append(train)
        held[cid] = rest[:200000]
    tok = bpe.train("".join(parts), SIZE)
    out = C.out_path(CID, "fineweb_tinychat_bpe8k.gguf")
    bpe.write_gguf(out, tok)
    rates = {cid: len(h.encode()) / len(tok.encode(h)) for cid, h in held.items()}
    print(f"{CID}: {len(tok.tokens)} tokens; bytes/token on held-out text: "
          + ", ".join(f"{k} {v:.2f}" for k, v in rates.items()))
    try:
        C.finish(CID, [out], "tools/bpe.py train on 40 MB fineweb_edu_sample_v1 + 10 MB tinychat_hf_sample_v1, "
                 f"vocab {SIZE}", pin=a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
