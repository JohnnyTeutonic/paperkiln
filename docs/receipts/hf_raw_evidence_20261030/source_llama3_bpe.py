#!/usr/bin/env python3
"""The Python side of include/microtorch/llama3_tokenizer.hpp: an HF
tokenizers Tokenizer rebuilt from a llama.cpp-form vocabulary GGUF
(tokenizer.ggml.model "gpt2", pre "llama-bpe"), configured as llama.cpp
encodes it. tests/test_llama3_tokenizer.py holds the three equal: this,
the C++ encoder and the HF tokenizer the vocabulary was trained as.

    from llama3_bpe import from_gguf
    tok = from_gguf("vocab.gguf"); tok.encode(text, add_special_tokens=False).ids

Requires tokenizers.
"""
from __future__ import annotations

import struct

LLAMA3 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
          r"\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIAL_TYPES = {2, 3, 4}  # unknown, control, user-defined: llama.cpp parses these out first


def read_gguf_kv(path, keys=None):
    """The metadata of a GGUF as a dict (only `keys`, when given). Strings
    come back as str (bytes when not UTF-8), arrays as lists."""
    buf = open(path, "rb").read()
    p = 0
    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

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
        try:
            return s.decode("utf-8")
        except UnicodeDecodeError:
            return s  # a paperkiln "bpe" vocabulary's byte tokens

    def val(t):
        if t in scal:
            return rd(scal[t])
        if t == 8:
            return rs()
        if t == 9:
            et, n = rd("I"), rd("Q")
            if et in scal:  # bulk-read numeric arrays
                nonlocal p
                f = "<" + str(n) + scal[et]
                v = list(struct.unpack_from(f, buf, p))
                p += struct.calcsize(f)
                return v
            return [val(et) for _ in range(n)]
        raise ValueError(f"GGUF type {t}")
    if rd("I") != 0x46554747:
        raise ValueError(f"{path}: not GGUF")
    rd("I"), rd("Q")
    kv = {}
    for _ in range(rd("Q")):
        k = rs()
        v = val(rd("I"))
        if keys is None or k in keys:
            kv[k] = v
    return kv


def from_gguf(path):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    kv = read_gguf_kv(path, {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
                             "tokenizer.ggml.merges", "tokenizer.ggml.token_type"})
    if kv.get("tokenizer.ggml.model") != "gpt2" or kv.get("tokenizer.ggml.pre") != "llama-bpe":
        raise ValueError(f"{path}: tokenizer is {kv.get('tokenizer.ggml.model')!r}/{kv.get('tokenizer.ggml.pre')!r}, "
                         "not 'gpt2'/'llama-bpe'")
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type", [1] * len(tokens))
    vocab = {}
    for i, t in enumerate(tokens):
        vocab[t] = i  # a repeated string keeps its last id, as llama.cpp
    merges = [tuple(m.split(" ")) for m in kv["tokenizer.ggml.merges"]]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(tokens[i], special=True, normalized=False)
                            for i, t in enumerate(types) if t in SPECIAL_TYPES])
    return tok
