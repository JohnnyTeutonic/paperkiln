#!/usr/bin/env python3
"""The quickstart vocabulary: tools/get_tinystories_data.py's frequency
vocabulary over tinystories_valid_v1, rebuilt with the standard library.

    python corpus/fetchers/vocab_tinystories_valid.py
    python corpus/fetchers/vocab_tinystories_valid.py --check-against story3k.gguf

get_tinystories_data.py writes its GGUF with the `gguf` package, whose byte
layout depends on that package's version. mtstudio reads only
`tokenizer.ggml.tokens`, so the identity of the vocabulary is its token list.
This fetcher rebuilds the list by the same procedure (importing the same
tokenizer function), pins its writer-independent token-list hash, and writes
the GGUF with the stdlib writer. `--check-against` compares the list with the
vocabulary embedded in a released model (the chat-v1 story3k.gguf).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common as C  # noqa: E402
from vocab_chat7b import token_list_sha, write_vocab_gguf  # noqa: E402

ROOT = os.path.dirname(C.CORPUS_DIR)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from get_tinystories_data import EOS, tokenize_like_mtstudio  # noqa: E402

CID = "vocab_tinystories_valid_v1"
SOURCE = ("tinystories_valid_v1", "tinystories_valid.txt")
LIST = os.path.join(C.CORPUS_DIR, "vocab", "tinystories_valid_4096.json")
VOCAB = 4096


def build_tokens():
    """get_tinystories_data.py main(), the vocabulary half, verbatim in effect."""
    src = os.path.join(C.DATA_DIR, *SOURCE)
    if not C.already_done(SOURCE[0], [SOURCE[1]]):
        C.die(f"{SOURCE[0]} missing or unverified: run corpus/fetchers/tinystories.py valid_v1")
    counts = collections.Counter()
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            counts.update(tokenize_like_mtstudio(line))
    words = [w for w, _ in counts.most_common() if w.isascii() and w != EOS]
    special = ["<unk>"] + ([EOS] if counts[EOS] else [])
    return special + words[: VOCAB - len(special)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the quickstart vocabulary.")
    ap.add_argument("--check-against", metavar="GGUF")
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    toks = build_tokens()
    sha = token_list_sha(toks)
    if not os.path.exists(LIST):
        if not a.pin:
            C.die(f"{LIST} absent: run once with --pin")
        with open(LIST, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"source": "corpus tinystories_valid_v1 via tools/get_tinystories_data.py procedure",
                       "token_list_sha256": sha, "tokens": toks}, f, ensure_ascii=True, indent=0)
            f.write("\n")
    with open(LIST, encoding="utf-8") as f:
        rec = json.load(f)
    if rec["token_list_sha256"] != sha or rec["tokens"] != toks:
        C.die(f"rebuilt token list differs from {LIST} (sha {sha} vs {rec['token_list_sha256']})")
    if a.check_against:
        from corpus_quality import read_gguf_tokens
        theirs = [t.decode("utf-8") for t in read_gguf_tokens(a.check_against)]
        if theirs[: len(toks)] != toks:
            first = next(i for i, (x, y) in enumerate(zip(theirs, toks)) if x != y) if len(theirs) >= len(toks) else len(theirs)
            C.die(f"{a.check_against}: token lists differ (first at id {first})")
        print(f"token list matches {a.check_against} ({len(toks)} tokens)")
    out = C.out_path(CID, "tinystories_vocab.gguf")
    if not C.already_done(CID, ["tinystories_vocab.gguf"]):
        write_vocab_gguf(out, toks, toks.index(EOS) if EOS in toks else 0)
        try:
            C.finish(CID, [out], f"token_list_sha256 {sha}", pin=a.pin)
        except C.HashMismatch as e:
            C.die(str(e))
    else:
        print(f"{CID}: present and verified")
    print(f"token_list_sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
