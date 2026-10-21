#!/usr/bin/env python3
"""Pre-tokenise a text corpus into the .ids file mtstudio reads in place of
the text (data.corpus / data.anneal.corpus ending in .ids), so a run or a
resume does not re-tokenise gigabytes on the VM.

    python3 tools/corpus_ids.py TEXT VOCAB.gguf OUT.ids [--threads N]

The ids equal mtstudio's own tokenisation of the whole file: the text is cut
only right after a special token (<|endoftext|> and the turn markers), and
both encoders split at special tokens before anything else. The vocabulary
decides the encoder: tokenizer.ggml.model "bpe" (tools/bpe.py) or "gpt2" /
pre "llama-bpe" (tools/llama3_bpe.py). Layout, little-endian: "PKIDS001",
uint32 id width (2 when the vocabulary fits, else 4), uint32 vocabulary
size, uint64 count, ids. Prints a JSON summary with the output's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llama3_bpe  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<unk>"]
BATCH = 4096


def chunks(path, size=1 << 24):
    """Pieces of the file, each ending right after a special token (or at the
    end of the file). Read as text with strict UTF-8, so bad bytes surface."""
    pat = re.compile("|".join(map(re.escape, SPECIALS)))
    buf = ""
    with open(path, encoding="utf-8", errors="strict", newline="") as f:
        while True:
            block = f.read(size)
            buf += block
            start = 0
            for m in pat.finditer(buf):
                yield buf[start:m.end()]
                start = m.end()
            buf = buf[start:]
            if not block:
                if buf:
                    yield buf
                return


def encoder(vocab):
    kv = llama3_bpe.read_gguf_kv(vocab, {"tokenizer.ggml.model", "tokenizer.ggml.tokens"})
    n = len(kv["tokenizer.ggml.tokens"])
    if kv["tokenizer.ggml.model"] == "bpe":
        import bpe
        tok = bpe.read_gguf(vocab)
        return n, lambda texts: [tok.encode(t) for t in texts]
    tok = llama3_bpe.from_gguf(vocab)
    return n, lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("text")
    ap.add_argument("vocab")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=0, help="HF tokenizers threads (0 = its default)")
    a = ap.parse_args(argv)
    if a.threads:
        os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    n_vocab, enc = encoder(a.vocab)
    width = 2 if n_vocab <= 1 << 16 else 4
    fmt = "H" if width == 2 else "I"
    count = 0
    tmp = a.out + ".partial"
    with open(tmp, "wb") as f:
        f.write(b"PKIDS001" + struct.pack("<IIQ", width, n_vocab, 0))
        batch = []

        def flush():
            nonlocal count
            for ids in enc(batch):
                f.write(struct.pack(f"<{len(ids)}{fmt}", *ids))
                count += len(ids)
            batch.clear()
        for piece in chunks(a.text):
            batch.append(piece)
            if len(batch) >= BATCH:
                flush()
        flush()
        f.seek(16)
        f.write(struct.pack("<Q", count))
    os.replace(tmp, a.out)
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    print(json.dumps({"out": a.out, "tokens": count, "width": width, "vocab": n_vocab, "sha256": h.hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
