"""Tests for tools/corpus_quality.py and corpus/fetchers/_common.py.

The tokenizer cases pin the byte-level behaviour of wordtok::tokenize
(include/microtorch/word_tokenizer.hpp); the C0 measurement depends on it.
Files are written to pytest's tmp_path, never into the repository.
"""
import importlib
import os
import struct
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))

import corpus_quality as cq  # noqa: E402

EOS = b"<|endoftext|>"


def vocab_of(*toks):
    return cq.build_vocab([b"<unk>"] + [t.encode() if isinstance(t, str) else t for t in toks])


def words(data, vocab, max_tokens=0):
    return cq.tokenize_bytes(data, vocab, max_tokens)[1]


def test_words_digits_apostrophes_lowercased():
    v = vocab_of("lily's", "42", "the")
    ids, w, _ = cq.tokenize_bytes(b"Lily's 42 THE", v)
    assert w == [b"lily's", b"42", b"the"]
    assert ids == [1, 2, 3]


def test_punctuation_is_its_own_token_and_unknown_maps_to_zero():
    v = vocab_of("hi", ",")
    ids, w, _ = cq.tokenize_bytes(b"hi, hi.", v)
    assert w == [b"hi", b",", b"hi", b"."]
    assert ids == [1, 2, 1, 0]


def test_non_ascii_is_one_unknown_token_per_byte():
    # U+2019 RIGHT SINGLE QUOTATION MARK is three UTF-8 bytes; C's isalpha
    # in the "C" locale rejects each, so "lily's" splits around them.
    v = vocab_of("lily", "s")
    w = words("lily’s".encode("utf-8"), v)
    assert w == [b"lily", b"\xe2", b"\x80", b"\x99", b"s"]


def test_crlf_and_all_c_whitespace_separate_without_tokens():
    v = vocab_of("a", "b")
    assert words(b"a\r\nb\x0b\x0ca\tb", v) == [b"a", b"b", b"a", b"b"]


def test_eos_is_one_token_only_when_in_vocab():
    with_eos = cq.build_vocab([b"<unk>", EOS, b"a"])
    assert cq.tokenize_bytes(b"a <|endoftext|> a", with_eos)[0] == [2, 1, 2]
    without = vocab_of("a")
    assert words(b"a <|endoftext|>", without) == [b"a", b"<", b"|", b"endoftext", b"|", b">"]


def test_max_tokens_cap_matches_cpp_including_the_one_extra_token():
    # C++ checks the cap once per byte, but one punctuation byte can add two
    # ids (flush the word in progress, then the punctuation itself), so the
    # stream can end one past the cap: the 400001 in every research receipt.
    v = vocab_of("a", "bb", "c")
    ids, _, used = cq.tokenize_bytes(b"a , bb c", v, max_tokens=2)
    assert len(ids) == 2 and used == 3
    ids, w, _ = cq.tokenize_bytes(b"bb. c", v, max_tokens=1)
    assert w == [b"bb", b"."]


def test_vocab_cap_and_first_occurrence_wins():
    v = cq.build_vocab([b"<unk>", b"x", b"x", b"y"], cap=3)
    assert v == {b"<unk>": 0, b"x": 1}


def write_gguf(path, tokens, extra_kv=True):
    def s(b):
        return struct.pack("<Q", len(b)) + b
    kv = []
    if extra_kv:  # values of every skippable kind, before the tokens
        kv.append(s(b"general.architecture") + struct.pack("<I", 8) + s(b"llama"))
        kv.append(s(b"x.u32") + struct.pack("<II", 4, 7))
        kv.append(s(b"x.arr") + struct.pack("<IIQ", 9, 6, 2) + struct.pack("<ff", 1.0, 2.0))
    kv.append(s(b"tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens))
              + b"".join(s(t) for t in tokens))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))


def test_gguf_token_roundtrip(tmp_path):
    toks = [b"<unk>", EOS, b"day,", b"\xe2\x80\x99", b"the"]
    p = tmp_path / "v.gguf"
    write_gguf(p, toks)
    assert cq.read_gguf_tokens(str(p)) == toks


def test_gguf_rejects_non_gguf(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ValueError):
        cq.read_gguf_tokens(str(p))


def test_coverage_counts_unknowns_and_held_out_tail():
    toks = [b"<unk>", b"a"]
    data = b"a . a . a . a . a . a . a . a . a . a ." * 2  # 40 tokens, half unknown
    c = cq.coverage(data, toks, cap=0, max_tokens=0)
    assert c["tokens"] == 40 and c["unk_tokens"] == 20 and c["unk_rate"] == 0.5
    assert c["val_tokens"] == 2 and c["val_unk_rate"] == 0.5
    assert c["top_unk"] == [[".", 20]]


def test_percentiles_nearest_rank_and_duplicates():
    assert cq.percentiles(list(range(1, 101))) == {"p50": 50, "p90": 90, "p99": 99}
    assert cq.percentiles([5]) == {"p50": 5, "p90": 5, "p99": 5}
    assert cq.exact_duplicate_rate([b"a", b"b", b"a", b"a"]) == 0.5


def test_documents_split_on_eos_else_lines():
    assert cq.split_documents(b"x y <|endoftext|>\nz <|endoftext|>\n") == [b"x y", b"z"]
    assert cq.split_documents(b"x y\r\n\r\nz\r\n") == [b"x y", b"z"]


@pytest.fixture
def common(tmp_path, monkeypatch):
    import _common
    importlib.reload(_common)
    monkeypatch.setattr(_common, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(_common, "HASH_DIR", str(tmp_path / "hashes"))
    return _common


def test_fetch_requires_pin_then_verifies_then_refuses_repin(common):
    p = common.out_path("c", "f.txt")
    with open(p, "wb") as f:
        f.write(b"hello")
    with pytest.raises(common.HashMismatch):
        common.finish("c", [p], "note")              # unpinned, no --pin
    common.finish("c", [p], "note", pin=True)        # first fetch pins
    assert common.already_done("c", ["f.txt"])
    common.finish("c", [p], "note")                  # verifies
    with pytest.raises(RuntimeError):
        common.pin_hashes("c", {"f.txt": "0" * 64}, "again")
    with open(p, "wb") as f:
        f.write(b"hellp")
    assert not common.already_done("c", ["f.txt"])
    with pytest.raises(common.HashMismatch):
        common.finish("c", [p], "note")              # changed bytes fail loudly
