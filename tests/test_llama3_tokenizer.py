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
import os
from pathlib import Path
import random
import struct
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LLAMA3 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
          r"\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIALS = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"]
PIECES = ["a", "Z", "ab", " ", "  ", "   ", "9", "12345", "3.14", "'", "'s", "'S", "'LL", "'re", "'Ve", "'d",
          "!!", "...", "$", "é", "é", "ß", "中文", "텍스트", "Ελλάδα", "١٢٣", "½", "😀", "✓", "\t", "\r",
          "\n", "\r\n", "\n\n", " \n ", " ", "　", " ", "<|", "|>", "<|user|>", "<|endoftext|>",
          "x<|assistant|>y", "don't", "WE'VE", "I'm", "naïve", "x̀́", "​", "\x0b"]
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


pytestmark = pytest.mark.skipif(_binary() is None, reason="set LLAMA3_ENCODE_BIN to the built llama3_encode")


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


def test_ids_match_hf_tokenizers(tmp_path):
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
    import json
    merges = [" ".join(m) if isinstance(m, list) else m for m in json.loads(tok.to_str())["model"]["merges"]]
    types = [2] + [3] * 3 + [1] * (len(tokens) - 4)
    p = tmp_path / "v.gguf"
    _write_gguf(p, tokens, merges, types)
    recs = _records(2, 2000)
    for rec, line in zip(recs, _run(str(p), recs)):
        cpp = list(map(int, line.split()))
        assert cpp == tok.encode(rec, add_special_tokens=False).ids, repr(rec)
