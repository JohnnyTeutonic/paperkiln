#!/usr/bin/env python3
"""Byte-level BPE: train, encode, decode (the reference implementation).

    python tools/bpe.py train CORPUS --vocab 8192 --out VOCAB.gguf [--max-mb 50]
    python tools/bpe.py encode VOCAB.gguf "some text"

Why: the word-level tokenizer (include/microtorch/word_tokenizer.hpp) cannot
represent open text. A 4096-word vocabulary loses 17-20% of FineWeb-Edu or
Simple English Wikipedia tokens to <unk> even when fitted to them
(experiments/corpus_probe/). Byte-level BPE is lossless by construction:
every UTF-8 string round-trips, and there is no <unk>.

The algorithm, kept readable:
- Pre-tokenize the UTF-8 BYTES into chunks: a run of letters (with one
  optional leading space), a single digit (so numbers stay compositional),
  a run of other non-space bytes, or a run of whitespace. Merges never
  cross a chunk boundary. Classes are byte classes so that C++ needs no
  Unicode tables: a letter is an ASCII letter, an apostrophe, or ANY byte
  >= 0x80 (so a multibyte character stays inside its word); a digit is
  0-9; whitespace is the C isspace set.
- Vocabulary: specials, then the 256 single bytes, then one token per merge
  in the order it was learned.
- Training: count chunk frequencies once, then repeatedly merge the most
  frequent adjacent pair, ties broken by the smaller pair of token ids, so
  the result is deterministic. Pair counts are updated incrementally.
- Encoding: within a chunk, repeatedly apply the lowest-ranked merge
  present. Byte-identical to include/microtorch/bpe_tokenizer.hpp, which
  tests/test_bpe.py checks.

The vocabulary GGUF stores `tokenizer.ggml.model = "bpe"`, the tokens (as
raw bytes), and `tokenizer.ggml.merges` as "a b" lines of token ids.
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import os
import struct
import sys

SPECIALS = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"]


_SPACE = frozenset(b" \t\n\r\x0b\x0c")
_SPECIAL_BYTES = [sp.encode() for sp in SPECIALS[1:]]


def _letter(c):
    return (0x41 <= c <= 0x5A) or (0x61 <= c <= 0x7A) or c == 0x27 or c >= 0x80


def _digit(c):
    return 0x30 <= c <= 0x39


def pretokenize(text):
    """Chunks (bytes) of the UTF-8 text, in order; they concatenate back to it."""
    b = text.encode("utf-8") if isinstance(text, str) else text
    out = []
    i, n = 0, len(b)
    while i < n:
        sp = next((x for x in _SPECIAL_BYTES if b.startswith(x, i)), None)
        if sp is not None:
            out.append(sp)
            i += len(sp)
            continue
        c = b[i]
        j = i + 1 if (c == 0x20 and i + 1 < n and _letter(b[i + 1])) else i
        if _letter(b[j]):
            k = j
            while k < n and _letter(b[k]):
                k += 1
        elif _digit(c):
            k = i + 1
        elif c in _SPACE:
            k = i
            while k < n and b[k] in _SPACE and not (b[k] == 0x20 and k + 1 < n and _letter(b[k + 1])):
                k += 1
            k = max(k, i + 1)
        else:
            k = i
            while k < n and not (b[k] in _SPACE or _letter(b[k]) or _digit(b[k]) or
                                 any(b.startswith(x, k) for x in _SPECIAL_BYTES)):
                k += 1
            k = max(k, i + 1)
        out.append(b[i:k])
        i = k
    return out


class BPE:
    def __init__(self, tokens, merges):
        self.tokens = tokens                    # list[bytes]
        self.merges = merges                    # list[(a, b)] in rank order
        self.rank = {m: r for r, m in enumerate(merges)}
        self.merged = {m: len(SPECIALS) + 256 + r for r, m in enumerate(merges)}
        self.special = {s.encode(): i for i, s in enumerate(SPECIALS)}

    def encode_chunk(self, chunk):
        if chunk in self.special:
            return [self.special[chunk]]
        ids = [len(SPECIALS) + b for b in chunk]
        while len(ids) > 1:
            best, at = None, -1
            for k in range(len(ids) - 1):
                r = self.rank.get((ids[k], ids[k + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, k
            if best is None:
                break
            ids[at:at + 2] = [self.merged[(ids[at], ids[at + 1])]]
        return ids

    def encode(self, text):
        out = []
        for ch in pretokenize(text):
            out += self.encode_chunk(ch)
        return out

    def decode(self, ids):
        b = b"".join(self.tokens[i] if i >= len(SPECIALS) else SPECIALS[i].encode() for i in ids)
        return b.decode("utf-8", errors="replace")


def train(text, vocab_size):
    n_merges = vocab_size - len(SPECIALS) - 256
    if n_merges < 0:
        raise ValueError("vocab_size too small")
    specials = {sp.encode() for sp in SPECIALS}
    freq = collections.Counter(ch for ch in pretokenize(text) if ch not in specials)
    words = [[len(SPECIALS) + b for b in w] for w in freq]
    counts = [freq[w] for w in freq]
    pair = collections.Counter()
    where = collections.defaultdict(set)
    for wi, w in enumerate(words):
        for k in range(len(w) - 1):
            pair[(w[k], w[k + 1])] += counts[wi]
            where[(w[k], w[k + 1])].add(wi)
    heap = [(-c, p) for p, c in pair.items()]
    heapq.heapify(heap)
    merges = []
    tokens = [s.encode() for s in SPECIALS] + [bytes([b]) for b in range(256)]
    while len(merges) < n_merges and heap:
        c, p = heapq.heappop(heap)
        if -c != pair.get(p, 0) or -c <= 0:
            continue  # stale entry
        new = len(tokens)
        merges.append(p)
        tokens.append(tokens[p[0]] + tokens[p[1]])
        for wi in list(where[p]):
            w = words[wi]
            k = 0
            while k < len(w) - 1:
                if w[k] == p[0] and w[k + 1] == p[1]:
                    # retract the pairs touching this occurrence, merge, re-add
                    if k > 0:
                        q = (w[k - 1], w[k])
                        pair[q] -= counts[wi]
                        heapq.heappush(heap, (-pair[q], q))
                    if k + 2 < len(w):
                        q = (w[k + 1], w[k + 2])
                        pair[q] -= counts[wi]
                        heapq.heappush(heap, (-pair[q], q))
                    w[k:k + 2] = [new]
                    if k > 0:
                        q = (w[k - 1], w[k])
                        pair[q] += counts[wi]
                        where[q].add(wi)
                        heapq.heappush(heap, (-pair[q], q))
                    if k + 1 < len(w):
                        q = (w[k], w[k + 1])
                        pair[q] += counts[wi]
                        where[q].add(wi)
                        heapq.heappush(heap, (-pair[q], q))
                k += 1
        pair[p] = 0
        del where[p]
    return BPE(tokens, merges)


# ---- GGUF (the minimal tokenizer-only subset mtstudio reads) ----------------

def _s(b):
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, bpe, eos="<|endoftext|>"):
    kv = [_s(b"general.architecture") + struct.pack("<I", 8) + _s(b"llama"),
          _s(b"tokenizer.ggml.model") + struct.pack("<I", 8) + _s(b"bpe"),
          _s(b"tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(bpe.tokens)) + b"".join(_s(t) for t in bpe.tokens),
          _s(b"tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, len(bpe.merges))
          + b"".join(_s(f"{a} {b}".encode()) for a, b in bpe.merges),
          _s(b"tokenizer.ggml.eos_token_id") + struct.pack("<II", 4, SPECIALS.index(eos))]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
        f.write(b"\0" * ((-f.tell()) % 32))


def read_gguf(path):
    buf = open(path, "rb").read()
    p = 0

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from("<" + fmt, buf, p)[0]
        p += struct.calcsize("<" + fmt)
        return v

    def rs():
        nonlocal p
        n = rd("Q")
        s = buf[p:p + n]
        p += n
        return s
    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(t):
        if t in scal:
            return rd(scal[t])
        if t == 8:
            return rs()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [val(et) for _ in range(n)]
        raise ValueError(t)
    assert rd("I") == 0x46554747
    rd("I"), rd("Q")
    kv = {}
    for _ in range(rd("Q")):
        k = rs().decode()
        kv[k] = val(rd("I"))
    merges = [tuple(int(x) for x in m.split()) for m in kv["tokenizer.ggml.merges"]]
    return BPE(kv["tokenizer.ggml.tokens"], merges)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("corpus")
    t.add_argument("--vocab", type=int, default=8192)
    t.add_argument("--max-mb", type=float, default=50.0)
    t.add_argument("--out", required=True)
    e = sub.add_parser("encode")
    e.add_argument("vocab")
    e.add_argument("text")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        with open(a.corpus, encoding="utf-8", errors="replace") as f:
            text = f.read(int(a.max_mb * 1024 * 1024))
        bpe = train(text, a.vocab)
        write_gguf(a.out, bpe)
        print(json.dumps({"vocab": len(bpe.tokens), "merges": len(bpe.merges), "out": a.out}))
        return 0
    bpe = read_gguf(a.vocab)
    ids = bpe.encode(a.text)
    print(json.dumps({"ids": ids, "pieces": [bpe.tokens[i].decode("utf-8", "replace") for i in ids],
                      "roundtrip": bpe.decode(ids) == a.text}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
