#!/usr/bin/env python3
"""The next base model's vocabulary (LOCAL_LLM_STACK.md sections 7-8): byte-level BPE in the Llama 3
scheme, so llama.cpp, Ollama and LM Studio tokenise our models natively.

    python corpus/fetchers/vocab_llama3_bpe.py train --source PATH=BYTES ... --out DIR [--size 16384]
    python corpus/fetchers/vocab_llama3_bpe.py check --dir DIR --heldout PATH=BYTES ... [--llama-tokenize BIN]

Scheme (decided with 2e, 21 Oct 2026):
- pre-tokeniser: the llama3 regex (LLAMA3_SPLIT, as tokenizers' Split, behaviour "isolated"),
  then the GPT-2 byte-to-unicode mapping (ByteLevel, no prefix space);
- BPE merges as strings, trained by HF tokenizers' BpeTrainer;
- ids 0-3 are the chat specials <unk>, <|endoftext|>, <|user|>, <|assistant|>, control tokens
  that are never split;
- GGUF: tokenizer.ggml.model "gpt2", tokenizer.ggml.pre "llama-bpe", tokens in byte-level form,
  merges "a b", token types (3 = control for the specials), add_bos / add_eos false.

train takes the first BYTES of each source text file, cut at a document boundary ("<|endoftext|>"
and a newline), and writes DIR/tokenizer.json, DIR/vocab_llama3_bpe.gguf (a vocabulary-only GGUF
with the minimal llama hyperparameters llama.cpp's loader requires) and DIR/train_ledger.json
(sources with sha256 and bytes used, and the GGUF's sha256).

check tokenises held-out text (the bytes after each training cut, or any file) plus a fixed set of
hard cases (digit runs, contractions, newlines, non-ASCII letters, the specials) with
tokenizer.json and with llama.cpp (llama-tokenize, or llama-cpp-python's vocab-only load), and
exits 1 on any difference. Bytes per token per held-out source go into DIR/check.json.

Pre-tokenised corpora (.ids) for mtstudio come from tools/corpus_ids.py with the trained GGUF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys

SPECIALS = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"]
LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
EOT = "<|endoftext|>\n"
GGUF_NAME = "vocab_llama3_bpe.gguf"
HARD_CASES = [
    "Hello world", " leading space", "trailing space ", "two  spaces", "tabs\tand\nnewlines\n\n\nthree",
    "It's they're we've I'm you'll he'd SHE'S", "12345678901 3.14159 1,000,000 2026-10-21",
    "café naïve Zürich São Paulo Øresund", "日本語のテキスト", "Ελληνικά κείμενα", "emoji 🙂👍🏽 done",
    "<|user|>What is the capital of France?<|assistant|>Paris.<|endoftext|>",
    "x<|endoftext|>y", "<|user|> spaced <|assistant|>", "code: def f(x): return x**2  # comment",
    "URL https://example.com/a?b=c&d=e", "punctuation!!! ... ??? ---", "\r\nwindows\r\nlines",
]


def _utf8_prefix(raw):
    """raw cut back to a whole UTF-8 character, decoded strictly: invalid bytes in the source
    raise (llama.cpp turns them into U+FFFD before splitting, so they could never round-trip)."""
    k = len(raw)
    while k > 0 and k > len(raw) - 4 and (raw[k - 1] & 0xC0) == 0x80:
        k -= 1
    if k > 0 and raw[k - 1] >= 0xC0:
        k -= 1
    return raw[:k].decode("utf-8", errors="strict")


def cut(path, nbytes):
    """(training text, the text just after it): the longest prefix within nbytes that ends at a
    document boundary."""
    with open(path, "rb") as f:
        raw = f.read(nbytes + 1_000_000)
    head = _utf8_prefix(raw[:nbytes])
    i = head.rfind(EOT)
    i = len(head) if i < 0 else i + len(EOT)
    text = _utf8_prefix(raw)
    return text[:i], text[i:i + 200_000]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    # ignore_merges: a pre-token that is itself in the vocabulary is emitted whole, as llama.cpp's
    # llama-bpe does (2e's include/microtorch/llama3_tokenizer.hpp follows the same rule).
    tok = Tokenizer(models.BPE(unk_token=None, fuse_unk=False, byte_fallback=False, ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3_SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    return tok


def train(sources, size, out):
    from tokenizers import pre_tokenizers, trainers
    os.makedirs(out, exist_ok=True)
    texts, ledger = [], {"scheme": "llama3 split + GPT-2 byte level BPE", "size": size, "specials": SPECIALS,
                         "sources": []}
    for path, nbytes in sources:
        t, _ = cut(path, nbytes)
        texts.append(t)
        ledger["sources"].append({"path": path, "sha256": sha256(path), "bytes_requested": nbytes,
                                  "bytes_used": len(t.encode("utf-8"))})
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=size, min_frequency=2, special_tokens=SPECIALS, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def docs():
        for t in texts:
            for d in t.split(EOT):
                if d:
                    yield d + EOT
    tok.train_from_iterator(docs(), trainer=trainer)
    for i, s in enumerate(SPECIALS):
        if tok.token_to_id(s) != i:
            raise SystemExit(f"special {s} has id {tok.token_to_id(s)}, not {i}")
    tj = os.path.join(out, "tokenizer.json")
    tok.save(tj)
    gguf = os.path.join(out, GGUF_NAME)
    write_gguf(gguf, tj)
    ledger["tokenizer_json_sha256"] = sha256(tj)
    ledger["gguf_sha256"] = sha256(gguf)
    ledger["vocab_size"] = tok.get_vocab_size()
    with open(os.path.join(out, "train_ledger.json"), "w") as f:
        json.dump(ledger, f, indent=1)
    print(json.dumps({k: ledger[k] for k in ("vocab_size", "gguf_sha256")}))


# ---- GGUF (vocabulary only, in the form llama.cpp's loader reads) ---------
def _s(b):
    return struct.pack("<Q", len(b)) + b


def _kv_str(k, v):
    return _s(k.encode()) + struct.pack("<I", 8) + _s(v.encode())


def _kv_u32(k, v):
    return _s(k.encode()) + struct.pack("<II", 4, v)


def _kv_f32(k, v):
    return _s(k.encode()) + struct.pack("<If", 6, v)


def _kv_bool(k, v):
    return _s(k.encode()) + struct.pack("<IB", 7, 1 if v else 0)


def _kv_strs(k, vs):
    return _s(k.encode()) + struct.pack("<IIQ", 9, 8, len(vs)) + b"".join(_s(v.encode()) for v in vs)


def _kv_i32s(k, vs):
    return _s(k.encode()) + struct.pack("<IIQ", 9, 5, len(vs)) + b"".join(struct.pack("<i", v) for v in vs)


def write_gguf(path, tokenizer_json):
    tj = json.load(open(tokenizer_json, encoding="utf-8"))
    vocab = tj["model"]["vocab"]                          # byte-level token string -> id
    added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
    n = max(list(vocab.values()) + list(added.values())) + 1
    tokens = [""] * n
    for t, i in vocab.items():
        tokens[i] = t
    for t, i in added.items():
        tokens[i] = t
    if "" in tokens:
        raise SystemExit("vocabulary has gaps")
    merges = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in tj["model"]["merges"]]
    # llama.cpp token types: 2 = unknown (<unk>), 3 = control (the other specials), 1 = normal.
    types = [2 if t == "<unk>" else 3 if t in SPECIALS else 1 for t in tokens]
    kv = [_kv_str("general.architecture", "llama"), _kv_str("general.name", "paperkiln llama3-bpe vocabulary"),
          # The minimal llama hyperparameters llama.cpp's loader requires even for a vocabulary-only load.
          _kv_u32("llama.context_length", 2048), _kv_u32("llama.embedding_length", 256),
          _kv_u32("llama.block_count", 1), _kv_u32("llama.feed_forward_length", 512),
          _kv_u32("llama.attention.head_count", 1), _kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
          _kv_u32("llama.vocab_size", n),
          _kv_str("tokenizer.ggml.model", "gpt2"), _kv_str("tokenizer.ggml.pre", "llama-bpe"),
          _kv_strs("tokenizer.ggml.tokens", tokens), _kv_i32s("tokenizer.ggml.token_type", types),
          _kv_strs("tokenizer.ggml.merges", merges),
          _kv_u32("tokenizer.ggml.bos_token_id", 1), _kv_u32("tokenizer.ggml.eos_token_id", 1),
          _kv_u32("tokenizer.ggml.unknown_token_id", 0), _kv_u32("tokenizer.ggml.padding_token_id", 0),
          _kv_bool("tokenizer.ggml.add_bos_token", False), _kv_bool("tokenizer.ggml.add_eos_token", False)]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
        f.write(b"\0" * ((-f.tell()) % 32))


# ---- check -----------------------------------------------------------------
def llama_cpp_ids(gguf, texts, binary=None):
    """llama.cpp's ids for each text: llama-tokenize if given, else llama-cpp-python."""
    if binary:
        out = []
        for t in texts:   # through stdin, so text starting with "-" is never read as a flag
            r = subprocess.run([binary, "-m", gguf, "--stdin", "--ids", "--log-disable", "--no-bos"],
                               input=t, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                raise SystemExit(f"llama-tokenize failed: {r.stderr[-400:]}")
            out.append(json.loads(r.stdout.strip().splitlines()[-1]))
        return out
    from llama_cpp import Llama
    m = Llama(model_path=gguf, vocab_only=True, verbose=False)
    return [m.tokenize(t.encode("utf-8"), add_bos=False, special=True) for t in texts]


def check(d, heldout, binary=None):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    gguf = os.path.join(d, GGUF_NAME)
    samples, report = list(HARD_CASES), {"bytes_per_token": {}}
    for path, nbytes in heldout:
        _, rest = cut(path, nbytes)
        if not rest.strip():    # the whole file went into training: nothing held out to measure
            report["bytes_per_token"][path] = None
            continue
        lines = [x for x in rest.split("\n") if x.strip()][:300]
        samples += lines
        enc = tok.encode(rest).ids
        report["bytes_per_token"][path] = round(len(rest.encode("utf-8")) / max(1, len(enc)), 3)
    hf = [tok.encode(t).ids for t in samples]
    lc = llama_cpp_ids(gguf, samples, binary)
    diffs = [{"text": t[:200], "hf": a[:40], "llama_cpp": b[:40]} for t, a, b in zip(samples, hf, lc) if a != b]
    roundtrip = sum(tok.decode(ids, skip_special_tokens=False) != t for t, ids in zip(samples, hf))
    report.update({"samples": len(samples), "differences": len(diffs), "first_differences": diffs[:20],
                   "roundtrip_failures": roundtrip, "engine": binary or "llama-cpp-python"})
    with open(os.path.join(d, "check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("samples", "differences", "roundtrip_failures", "bytes_per_token")}))
    return 0 if not diffs and not roundtrip else 1


def _pairs(items):
    out = []
    for s in items:
        p, _, b = s.rpartition("=")
        out.append((p, int(b)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--source", action="append", required=True, help="PATH=BYTES (repeat)")
    t.add_argument("--size", type=int, default=16384)
    t.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", required=True)
    c.add_argument("--heldout", action="append", default=[], help="PATH=BYTES: text after the training cut (bytes_per_token null when nothing is left)")
    c.add_argument("--llama-tokenize", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(_pairs(a.source), a.size, a.out)
        return 0
    return check(a.dir, _pairs(a.heldout), a.llama_tokenize)


if __name__ == "__main__":
    sys.exit(main())
