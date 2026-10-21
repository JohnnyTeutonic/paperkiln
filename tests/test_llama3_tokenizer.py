"""Llama 3 byte-level BPE (include/microtorch/llama3_tokenizer.hpp).

The C++ encoder (LLAMA3_ENCODE_BIN, or build/llama3_encode) must agree with
two independent references on adversarial text (digits, contractions in
both cases, newline and whitespace runs, non-ASCII letters, marks, emoji,
special tokens glued to words):
- its pre-tokeniser with the llama3 regex as published (Python `regex`);
- its ids with HF tokenizers' ByteLevel BPE (llama3 Split, ignore_merges)
  trained here and written to a vocab GGUF of the llama.cpp form (model
  "gpt2", pre "llama-bpe", string merges, token types).
llama.cpp's own llama-tokenize is the third leg, in the vocab trainer's
check (corpus/fetchers). Skipped without the binary or the Python packages.
"""
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
LLAMA3 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
          r"\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIALS = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"]
PIECES = ["a", "Z", "ab", " ", "  ", "   ", "9", "12345", "3.14", "'", "'s", "'S", "'LL", "'re", "'Ve", "'d",
          "!!", "...", "$", "é", "e\u0301", "ß", "中文", "텍스트", "Ελλάδα", "١٢٣", "½", "😀", "✓", "\t", "\r",
          "\n", "\r\n", "\n\n", " \n ", "\u00a0", "\u3000", "\u2028", "<|", "|>", "<|user|>", "<|endoftext|>",
          "x<|assistant|>y", "don't", "WE'VE", "I'm", "naïve", "x\u0300\u0301", "\u200b", "\x0b"]
TEXT = ("Once upon a time, there was a little girl named Lily. She loved to play outside!\n"
        "\"Can I have 12 apples?\" asked Tom. 'Yes,' said Mom. It's 2026; we'll see 1,000,000 of them.\n"
        "<|user|> what is 3 plus 4 ?<|assistant|> the answer is 7 .<|endoftext|>\n"
        "naïve café — 中文 텍스트 ✓ 😀\n\ttabs  and   spaces\r\n\n\n  indented line\n") * 30


def _binary():
    c = os.environ.get("LLAMA3_ENCODE_BIN") or str(ROOT / "build" / "llama3_encode")
    return c if Path(c).is_file() else None


def _records(seed, n):
    rng = random.Random(seed)
    recs = TEXT.split("\n")[:12]
    for _ in range(n):
        recs.append("".join(rng.choice(PIECES) for _ in range(rng.randrange(1, 30))))
    return recs


def _run(arg, recs):
    r = subprocess.run([_binary(), arg], input="\0".join(recs).encode(), capture_output=True, timeout=300)
    assert r.returncode == 0, r.stderr.decode()
    out = r.stdout.decode().split("\n")[:-1]
    assert len(out) == len(recs)
    return out


needs_bin = pytest.mark.skipif(_binary() is None, reason="set LLAMA3_ENCODE_BIN to the built llama3_encode")


@needs_bin
def test_pre_tokeniser_matches_the_llama3_regex():
    regex = pytest.importorskip("regex")
    recs = _records(1, 2000)
    for rec, line in zip(recs, _run("--split", recs)):
        cpp = [bytes.fromhex(w).decode("utf-8") for w in line.split()]
        assert cpp == regex.findall(LLAMA3, rec), repr(rec)


def _write_gguf(path, tokens, merges, types):
    def s(x):
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kv = [("tokenizer.ggml.model", 8, s("gpt2")), ("tokenizer.ggml.pre", 8, s("llama-bpe")),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, len(tokens)) + b"".join(s(t) for t in tokens)),
          ("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, len(types)) + struct.pack(f"<{len(types)}i", *types)),
          ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, len(merges)) + b"".join(s(m) for m in merges))]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
        for k, t, v in kv:
            f.write(s(k) + struct.pack("<I", t) + v)


@pytest.fixture(scope="module")
def hf(tmp_path_factory):
    """An HF ByteLevel BPE trained as the vocabulary trainer does, and its GGUF."""
    tk = pytest.importorskip("tokenizers")
    from tokenizers import Regex, decoders, models, pre_tokenizers, trainers
    tok = tk.Tokenizer(models.BPE(ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator([TEXT] + _records(3, 300), trainer)
    vocab = tok.get_vocab()
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    assert [vocab[t] for t in tokens] == list(range(len(tokens))) and tokens[:4] == SPECIALS
    merges = [" ".join(m) if isinstance(m, list) else m for m in json.loads(tok.to_str())["model"]["merges"]]
    p = tmp_path_factory.mktemp("v") / "v.gguf"
    _write_gguf(p, tokens, merges, [2] + [3] * 3 + [1] * (len(tokens) - 4))
    return tok, p


@needs_bin
def test_ids_match_hf_tokenizers(hf):
    tok, p = hf
    recs = _records(2, 2000)
    for rec, line in zip(recs, _run(str(p), recs)):
        assert list(map(int, line.split())) == tok.encode(rec, add_special_tokens=False).ids, repr(rec)


def test_from_gguf_rebuilds_the_trained_tokenizer(hf):
    import llama3_bpe
    tok, p = hf
    again = llama3_bpe.from_gguf(str(p))
    for rec in _records(4, 1000):
        assert again.encode(rec, add_special_tokens=False).ids == tok.encode(rec, add_special_tokens=False).ids


def _read_ids(path):
    b = open(path, "rb").read()
    assert b[:8] == b"PKIDS001"
    width, nv, n = struct.unpack_from("<IIQ", b, 8)
    return nv, list(struct.unpack_from(f"<{n}{'H' if width == 2 else 'I'}", b, 24))


def _corpus():
    rng = random.Random(5)
    docs = ["".join(rng.choice(PIECES) for _ in range(rng.randrange(1, 60))) for _ in range(300)]
    return "<|endoftext|>\n".join(docs) + "<|user|>hi<|assistant|>hello<|endoftext|>"


def test_corpus_ids_equal_whole_text_encoding(hf, tmp_path, monkeypatch):
    import corpus_ids
    import llama3_bpe
    tok, p = hf
    text = _corpus()
    src = tmp_path / "c.txt"
    src.write_bytes(text.encode("utf-8"))
    orig = corpus_ids.chunks
    monkeypatch.setattr(corpus_ids, "chunks", lambda path: orig(path, size=97))  # specials straddle blocks
    out = tmp_path / "c.ids"
    assert corpus_ids.main([str(src), str(p), str(out)]) == 0
    nv, ids = _read_ids(out)
    assert nv == tok.get_vocab_size() and ids == llama3_bpe.from_gguf(str(p)).encode(text, add_special_tokens=False).ids


def test_corpus_ids_with_a_paperkiln_bpe_vocab(tmp_path, monkeypatch):
    import bpe
    import corpus_ids
    text = _corpus()
    tok = bpe.train(text, 400)
    v = tmp_path / "v.gguf"
    bpe.write_gguf(str(v), tok)
    src = tmp_path / "c.txt"
    src.write_bytes(text.encode("utf-8"))
    orig = corpus_ids.chunks
    monkeypatch.setattr(corpus_ids, "chunks", lambda path: orig(path, size=97))
    out = tmp_path / "c.ids"
    assert corpus_ids.main([str(src), str(v), str(out)]) == 0
    assert _read_ids(out) == (len(tok.tokens), tok.encode(text))


def test_chat_data_sizes_with_a_llama3_vocab(hf):
    import chat_data
    import llama3_bpe
    _, p = hf
    full = llama3_bpe.from_gguf(str(p))
    rng = random.Random(6)
    chat_data._init_tok(str(p))
    for _ in range(200):
        msgs = []
        for k in range(2 * rng.randrange(1, 4)):
            msgs.append({"role": "user" if k % 2 == 0 else "assistant",
                         "content": "".join(rng.choice(PIECES[:40]) for _ in range(rng.randrange(1, 12)))})
        d, out = chat_data._sized(json.dumps({"messages": msgs}))
        if out is None:
            continue
        for ex, n in out:
            text = ex["prompt"] + ex["completion"] + "<|endoftext|>"
            assert n == len(full.encode(text, add_special_tokens=False).ids), repr(text)


def test_chat_data_sizes_with_a_paperkiln_bpe_vocab(tmp_path):
    import bpe
    import chat_data
    tok = bpe.train(TEXT, 400)
    v = tmp_path / "v.gguf"
    bpe.write_gguf(str(v), tok)
    chat_data._init_tok(str(v))
    rng = random.Random(7)
    for _ in range(200):
        msgs = [{"role": "user" if k % 2 == 0 else "assistant",
                 "content": "".join(rng.choice(PIECES[:40]) for _ in range(rng.randrange(1, 12)))}
                for k in range(2 * rng.randrange(1, 4))]
        d, out = chat_data._sized(json.dumps({"messages": msgs}))
        for ex, n in out or []:
            text = ex["prompt"] + ex["completion"] + "<|endoftext|>"
            assert n == len(tok.encode(text)), repr(text)
