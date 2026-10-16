#!/usr/bin/env python3
"""Corpus measurements for the corpus registry and corpus cards
(docs/decisions/0001-posttrain-charter.md, sections 8-9).

    python tools/corpus_quality.py coverage CORPUS VOCAB.gguf [--cap 4096] [--max-tokens 400000]
    python tools/corpus_quality.py sha256 FILE

Every number here is computed the way the trainer sees the corpus.
`tokenize_bytes` is a byte-exact mirror of wordtok::tokenize
(include/microtorch/word_tokenizer.hpp), NOT of the Unicode-aware mirror in
tools/get_tinystories_data.py: C++ classifies one byte at a time with the C
locale, so each byte of a multibyte UTF-8 character is its own
single-character token, which no word-level vocabulary contains, so it
becomes <unk> (id 0). The two mirrors agree on ASCII text only.

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys

EOS = b"<|endoftext|>"
_SPACE = frozenset(b" \t\n\r\x0b\x0c")  # C isspace, "C" locale


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_gguf_tokens(path):
    """`tokenizer.ggml.tokens` from a GGUF header, in id order (the subset of
    GGUF that mtstudio's read_gguf_vocab reads)."""
    with open(path, "rb") as f:
        buf = f.read()
    p = 0

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from("<" + fmt, buf, p)
        p += struct.calcsize("<" + fmt)
        return v[0]

    def rd_str():
        nonlocal p
        n = rd("Q")
        s = buf[p:p + n]
        p += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}

    def skip(vt):
        if vt in scalar:
            rd(scalar[vt])
        elif vt == 8:
            rd_str()
        elif vt == 9:
            et, n = rd("I"), rd("Q")
            for _ in range(n):
                skip(et)
        else:
            raise ValueError(f"unknown GGUF value type {vt}")

    if rd("I") != 0x46554747:
        raise ValueError(f"{path}: not GGUF")
    rd("I")
    rd("Q")
    n_kv = rd("Q")
    for _ in range(n_kv):
        key = rd_str()
        vt = rd("I")
        if key == b"tokenizer.ggml.tokens" and vt == 9:
            et, n = rd("I"), rd("Q")
            if et != 8:
                raise ValueError("tokenizer.ggml.tokens is not a string array")
            return [rd_str() for _ in range(n)]
        skip(vt)
    raise ValueError(f"{path}: no tokenizer.ggml.tokens")


def build_vocab(tokens, cap=0):
    """token bytes -> id, as mtstudio builds it: truncate to `cap`, first
    occurrence wins (std::map::emplace)."""
    if cap and cap < len(tokens):
        tokens = tokens[:cap]
    vocab = {}
    for i, t in enumerate(tokens):
        vocab.setdefault(t, i)
    return vocab


def tokenize_bytes(data, vocab, max_tokens=0):
    """Byte-exact wordtok::tokenize. Returns (ids, words, bytes_consumed),
    where words[i] is the surface bytes behind ids[i]."""
    ids, words, cur = [], [], bytearray()
    eos = vocab.get(EOS, -1)
    limit = max_tokens if max_tokens else len(data) + 1
    i, n = 0, len(data)

    def flush():
        if cur:
            w = bytes(cur)
            ids.append(vocab.get(w, 0))
            words.append(w)
            cur.clear()

    while i < n:
        if len(ids) >= limit:
            break
        c = data[i]
        if c == 0x3C and eos >= 0 and data.startswith(EOS, i):
            flush()
            ids.append(eos)
            words.append(EOS)
            i += len(EOS)
            continue
        if (0x41 <= c <= 0x5A) or (0x61 <= c <= 0x7A) or (0x30 <= c <= 0x39) or c == 0x27:
            cur.append(c + 32 if 0x41 <= c <= 0x5A else c)
        else:
            flush()
            if c not in _SPACE:
                w = bytes([c])
                ids.append(vocab.get(w, 0))
                words.append(w)
        i += 1
    flush()  # unconditional, as in C++: a word cut by the cap still counts
    return ids, words, i


def coverage(data, tokens, cap=0, max_tokens=0, val_frac=0.05):
    """Coverage of a corpus under a capped vocabulary, as the trainer sees it.

    unk is counted by id 0 *and* by surface form: a token whose own surface
    is the vocabulary's id-0 string is not unknown."""
    vocab = build_vocab(tokens, cap)
    ids, words, used = tokenize_bytes(data, vocab, max_tokens)
    id0 = tokens[0] if tokens else None
    unk = [w for t, w in zip(ids, words) if t == 0 and w != id0]
    nonascii = sum(1 for w in unk if any(b > 127 for b in w))
    distinct = {}
    for w in unk:
        distinct[w] = distinct.get(w, 0) + 1
    top = sorted(distinct.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    val_start = len(ids) - len(ids) // 20 if val_frac == 0.05 else int(len(ids) * (1 - val_frac))
    val_unk = sum(1 for t, w in zip(ids[val_start:], words[val_start:]) if t == 0 and w != id0)
    return {
        "tokens": len(ids),
        "bytes_consumed": used,
        "tokens_per_byte": len(ids) / used if used else 0.0,
        "vocab_size": len(vocab),
        "unk_tokens": len(unk),
        "unk_rate": len(unk) / len(ids) if ids else 0.0,
        "unk_nonascii_byte_tokens": nonascii,
        "unk_distinct_types": len(distinct),
        "val_tokens": len(ids) - val_start,
        "val_unk_rate": val_unk / (len(ids) - val_start) if len(ids) > val_start else 0.0,
        "top_unk": [[w.decode("utf-8", "backslashreplace"), c] for w, c in top],
    }


def split_documents(data):
    """Documents as the corpora delimit them: on <|endoftext|> when the corpus
    carries it, else one document per non-empty line."""
    if EOS in data:
        parts = data.split(EOS)
    else:
        parts = data.splitlines()
    return [p.strip() for p in parts if p.strip()]


def percentiles(values, qs=(50, 90, 99)):
    """Nearest-rank percentiles (no interpolation, so every value reported
    is one that occurs)."""
    if not values:
        return {f"p{q}": None for q in qs}
    v = sorted(values)
    return {f"p{q}": v[max(0, -(-q * len(v) // 100) - 1)] for q in qs}


def exact_duplicate_rate(docs):
    """Fraction of documents whose exact bytes occurred earlier in the corpus."""
    seen, dup = set(), 0
    for d in docs:
        if d in seen:
            dup += 1
        else:
            seen.add(d)
    return dup / len(docs) if docs else 0.0


def composition(data):
    """Document count, length distribution in bytes and in tokenizer tokens
    (vocabulary-independent: the token boundaries do not depend on the vocab),
    and the exact-duplicate rate."""
    docs = split_documents(data)
    tok_lens = [len(tokenize_bytes(d, {}, 0)[0]) for d in docs]
    return {
        "bytes": len(data),
        "documents": len(docs),
        "doc_bytes": percentiles([len(d) for d in docs]),
        "doc_tokens": percentiles(tok_lens),
        "tokens_total": sum(tok_lens),
        "exact_duplicate_rate": exact_duplicate_rate(docs),
        "nonascii_bytes": sum(1 for b in data if b > 127),
        "crlf_lines": data.count(b"\r\n"),
        "has_eos_marker": EOS in data,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("corpus")
    c.add_argument("vocab")
    c.add_argument("--cap", type=int, default=4096)
    c.add_argument("--max-tokens", type=int, default=400000,
                   help="mtstudio's data.max_tokens (default 400000; 0 = whole file)")
    s = sub.add_parser("sha256")
    s.add_argument("file")
    m = sub.add_parser("composition")
    m.add_argument("corpus")
    a = ap.parse_args(argv)
    if a.cmd == "sha256":
        print(sha256_file(a.file))
        return 0
    if a.cmd == "composition":
        with open(a.corpus, "rb") as f:
            out = composition(f.read())
        out["sha256"] = sha256_file(a.corpus)
        json.dump(out, sys.stdout, indent=1)
        print()
        return 0
    with open(a.corpus, "rb") as f:
        data = f.read()
    out = coverage(data, read_gguf_tokens(a.vocab), a.cap, a.max_tokens)
    out.update({"corpus_sha256": sha256_file(a.corpus), "vocab_sha256": sha256_file(a.vocab),
                "cap": a.cap, "max_tokens": a.max_tokens})
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
