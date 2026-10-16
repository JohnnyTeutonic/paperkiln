#!/usr/bin/env python3
"""The research vocabulary: chat7b's first 4096 tokens, as a vocabulary-only GGUF.

    python corpus/fetchers/vocab_chat7b.py              # build + verify
    python corpus/fetchers/vocab_chat7b.py --extract PATH/TO/chat7b.gguf   # (re)derive the token list

Every Atlas, sparse-attention, transfer and highway run read
`tokenizer.ggml.tokens` from transformer_cpp/releases/chat7b.gguf (58 MB,
sha256 9c4b2719..., not published) and kept its first 4096 entries
(`data.vocab_cap`). mtstudio reads nothing else from that file, so a GGUF
holding only those 4096 tokens, in order, gives identical token ids. The
token list is committed at corpus/vocab/chat7b_cap4096.json; this script
writes the GGUF from it with the standard-library writer that
tools/get_tinychat_data.py uses, and verifies both.

The vocabulary was built by transformer_cpp's whitespace tokenizer from
TinyStories-Instruct, so 1503 of its 4095 non-<unk> entries carry attached
punctuation ("day,", "said.") that mtstudio's tokenizer can never produce,
and the bare full stop, comma, "!" and "?" are absent: see
corpus/ledger/0000-legacy-unknown.md and registry row
C0-legacy-vocab-coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

CID = "vocab_chat7b_cap4096_v1"
LIST = os.path.join(C.CORPUS_DIR, "vocab", "chat7b_cap4096.json")
SOURCE_SHA = "9c4b2719a7b9443a27f198ebe5845a9c00f4dd7b7ab69343d30f72421ee8453c"
CAP = 4096


def token_list_sha(tokens):
    """Writer-independent identity of a vocabulary: sha256 of its tokens,
    each length-prefixed (u64 LE) so no token boundary is ambiguous."""
    h = hashlib.sha256()
    for t in tokens:
        b = t.encode("utf-8", "surrogateescape")
        h.update(struct.pack("<Q", len(b)) + b)
    return h.hexdigest()


def extract(gguf_path):
    sys.path.insert(0, os.path.join(os.path.dirname(C.CORPUS_DIR), "tools"))
    from corpus_quality import read_gguf_tokens
    if C.sha256_file(gguf_path) != SOURCE_SHA:
        C.die(f"{gguf_path} is not the historical chat7b.gguf (sha256 {SOURCE_SHA})")
    toks = [t.decode("utf-8", "surrogateescape") for t in read_gguf_tokens(gguf_path)[:CAP]]
    os.makedirs(os.path.dirname(LIST), exist_ok=True)
    with open(LIST, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source": "transformer_cpp/releases/chat7b.gguf", "source_sha256": SOURCE_SHA,
                   "cap": CAP, "token_list_sha256": token_list_sha(toks), "tokens": toks},
                  f, ensure_ascii=True, indent=0)
        f.write("\n")
    print(f"wrote {LIST}: {len(toks)} tokens, token_list_sha256 {token_list_sha(toks)}")


def write_vocab_gguf(path, tokens, eos_id):
    def s(x):
        b = x.encode("utf-8", "surrogateescape")
        return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s("llama"),
          s("tokenizer.ggml.model") + struct.pack("<I", 8) + s("word"),
          s("tokenizer.ggml.tokens") + struct.pack("<II", 9, 8) + struct.pack("<Q", len(tokens))
          + b"".join(s(t) for t in tokens),
          s("tokenizer.ggml.eos_token_id") + struct.pack("<I", 4) + struct.pack("<I", eos_id)]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kv)))
        for item in kv:
            f.write(item)
        f.write(b"\0" * ((-f.tell()) % 32))


def build(pin=False):
    name = "chat7b_cap4096.gguf"
    if C.already_done(CID, [name]):
        print(f"{CID}: present and verified")
        return
    with open(LIST, encoding="utf-8") as f:
        rec = json.load(f)
    toks = rec["tokens"]
    if len(toks) != CAP or token_list_sha(toks) != rec["token_list_sha256"]:
        C.die(f"{LIST}: token list does not match its recorded hash")
    out = C.out_path(CID, name)
    write_vocab_gguf(out, toks, toks.index("<|endoftext|>"))
    C.finish(CID, [out], f"first {CAP} tokens of chat7b.gguf (sha256 {SOURCE_SHA}), "
             f"token_list_sha256 {rec['token_list_sha256']}", pin=pin)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the chat7b cap-4096 vocabulary GGUF.")
    ap.add_argument("--extract", metavar="CHAT7B_GGUF")
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    if a.extract:
        extract(a.extract)
        return 0
    try:
        build(pin=a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
