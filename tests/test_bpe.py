"""Byte-level BPE (tools/bpe.py, include/microtorch/bpe_tokenizer.hpp).

Python properties run anywhere. The parity test runs the C++ encoder
(BPE_ENCODE_BIN, or build/bpe_encode) on the same lines and requires
identical ids; it is skipped without the binary. Files go to tmp_path.
"""
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bpe  # noqa: E402

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside!\n"
    "\"Can I have 12 apples?\" asked Tom. 'Yes,' said Mom.\n"
    "<|user|> what is 3 plus 4 ?<|assistant|> the answer is 7 .<|endoftext|>\n"
    "naïve café — 中文 텍스트 ✓ 😀\n\ttabs  and   spaces\r\n"
) * 40


@pytest.fixture(scope="module")
def tok():
    return bpe.train(TEXT, 400)


def test_vocabulary_layout(tok):
    # A small corpus can run out of pairs before the target size: never more.
    assert len(tok.tokens) == len(bpe.SPECIALS) + 256 + len(tok.merges) <= 400 and len(tok.merges) > 50
    assert tok.tokens[:len(bpe.SPECIALS)] == [s.encode() for s in bpe.SPECIALS]
    assert tok.tokens[len(bpe.SPECIALS):len(bpe.SPECIALS) + 256] == [bytes([b]) for b in range(256)]
    for r, (a, b) in enumerate(tok.merges):
        assert tok.tokens[len(bpe.SPECIALS) + 256 + r] == tok.tokens[a] + tok.tokens[b]


def test_training_is_deterministic(tok):
    again = bpe.train(TEXT, 400)
    assert again.merges == tok.merges and again.tokens == tok.tokens


def test_pretokenize_partitions_the_bytes():
    rng = random.Random(0)
    alphabet = "ab Z9'.,!\n\t\r é中😀<|>" + "".join(bpe.SPECIALS)
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        assert b"".join(bpe.pretokenize(s)) == s.encode()


def test_lossless_and_no_unk(tok):
    rng = random.Random(1)
    for _ in range(200):
        s = "".join(chr(rng.choice([rng.randrange(32, 127), rng.randrange(160, 0x3000), 0x1F600, 10, 9])) for _ in
                    range(rng.randrange(0, 60)))
        ids = tok.encode(s)
        assert tok.decode(ids) == s
        assert 0 not in ids


def test_specials_are_atomic_and_digits_split(tok):
    ids = tok.encode("<|user|> 42<|assistant|><|endoftext|>")
    assert ids[0] == bpe.SPECIALS.index("<|user|>") and ids[-1] == bpe.SPECIALS.index("<|endoftext|>")
    assert bpe.SPECIALS.index("<|assistant|>") in ids
    assert [tok.tokens[i] for i in tok.encode("1234")] == [b"1", b"2", b"3", b"4"]


def test_compression(tok):
    sample = TEXT[:2000]
    assert len(sample.encode()) / len(tok.encode(sample)) > 2.0


def test_gguf_roundtrip(tok, tmp_path):
    p = tmp_path / "v.gguf"
    bpe.write_gguf(str(p), tok)
    back = bpe.read_gguf(str(p))
    assert back.tokens == tok.tokens and back.merges == tok.merges
    assert back.encode(TEXT[:500]) == tok.encode(TEXT[:500])


def _binary():
    c = os.environ.get("BPE_ENCODE_BIN") or str(ROOT / "build" / "bpe_encode")
    return c if Path(c).is_file() else None


@pytest.mark.skipif(_binary() is None, reason="set BPE_ENCODE_BIN to the built bpe_encode")
def test_cpp_encoder_matches_python(tok, tmp_path):
    p = tmp_path / "v.gguf"
    bpe.write_gguf(str(p), tok)
    lines = [l + "\n" for l in TEXT.split("\n")[:200]]
    rng = random.Random(2)
    for _ in range(200):  # adversarial lines: random bytes of every class, specials glued in
        lines.append("".join(rng.choice(["a", "Z", " ", "  ", "9", "'", "!!", "é", "中", "😀", "\t", "\r",
                                          "<|", "|>", "<|user|>", "<|endoftext|>", "x<|assistant|>y"])
                             for _ in range(rng.randrange(1, 25))) + "\n")
    text = "".join(lines)
    r = subprocess.run([_binary(), str(p)], input=text.encode(), capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode()
    cpp = [list(map(int, l.split())) if l.strip() else [] for l in r.stdout.decode().split("\n")[:-1]]
    py = [tok.encode(l) for l in lines]
    assert len(cpp) == len(py)
    for i, (a, b) in enumerate(zip(cpp, py)):
        assert a == b, (i, lines[i], a, b)
