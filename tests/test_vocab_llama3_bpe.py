"""corpus/fetchers/vocab_llama3_bpe.py on a tiny synthetic corpus: the specials hold ids 0-3 and are
never split, the llama3 split groups digits by three, every hard case round-trips, the GGUF holds
the llama.cpp keys, and encode writes the same ids as tokenizer.json. llama.cpp's own tokenisation
is the `check` subcommand's job (it needs llama.cpp), not this test's."""
import json
import os
import struct
import sys

import pytest

tokenizers = pytest.importorskip("tokenizers")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "fetchers"))
import vocab_llama3_bpe as V  # noqa: E402

EOT = "<|endoftext|>\n"


def read_gguf_kv(path):
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
        return s.decode("utf-8")
    scal = {4: "I", 5: "i", 6: "f", 7: "?"}

    def val(t):
        if t in scal:
            return rd(scal[t])
        if t == 8:
            return rs()
        et, n = rd("I"), rd("Q")
        return [val(et) for _ in range(n)]
    assert buf[:4] == b"GGUF"
    p = 4
    rd("I"), rd("Q")
    return {rs(): val(rd("I")) for _ in range(rd("Q"))}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    d = tmp_path_factory.mktemp("vocab")
    src = d / "src.txt"
    docs = []
    for i in range(400):
        docs.append(f"The quick brown fox number {i} jumps over the lazy dog. It's {i * 37} metres away.")
        docs.append(f"<|user|>How are you today, friend {i}?<|assistant|>I'm well, thank you!<|endoftext|>")
        docs.append("Café naïve Zürich: ünïcödé letters, and 日本語 text.\n\nNew paragraph here.")
    src.write_bytes("".join(x + EOT for x in docs).encode("utf-8"))
    out = d / "out"
    V.train([(str(src), src.stat().st_size)], 600, str(out))
    return out


def test_specials_and_split(trained):
    tok = tokenizers.Tokenizer.from_file(str(trained / "tokenizer.json"))
    assert [tok.token_to_id(s) for s in V.SPECIALS] == [0, 1, 2, 3]
    ids = tok.encode("<|user|>Hi<|assistant|>").ids
    assert ids[0] == 2 and ids[-1] == 3
    pieces = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str("12345 it's")]
    assert pieces[:2] == ["123", "45"]


def test_hard_cases_round_trip(trained):
    tok = tokenizers.Tokenizer.from_file(str(trained / "tokenizer.json"))
    for t in V.HARD_CASES:
        assert tok.decode(tok.encode(t).ids, skip_special_tokens=False) == t


def test_gguf_has_the_llama_cpp_keys(trained):
    kv = read_gguf_kv(str(trained / V.GGUF_NAME))
    tok = tokenizers.Tokenizer.from_file(str(trained / "tokenizer.json"))
    assert kv["tokenizer.ggml.model"] == "gpt2" and kv["tokenizer.ggml.pre"] == "llama-bpe"
    assert len(kv["tokenizer.ggml.tokens"]) == tok.get_vocab_size() == kv["llama.vocab_size"]
    assert kv["tokenizer.ggml.tokens"][:4] == V.SPECIALS and kv["tokenizer.ggml.token_type"][:5] == [2, 3, 3, 3, 1]
    assert all(" " in m for m in kv["tokenizer.ggml.merges"]) and kv["tokenizer.ggml.add_bos_token"] is False
    ledger = json.load(open(trained / "train_ledger.json"))
    assert ledger["gguf_sha256"] == V.sha256(str(trained / V.GGUF_NAME))


def test_gguf_rebuilds_the_same_tokenizer(trained):
    # 2e's tools/llama3_bpe.py rebuilds the HF tokenizer from a llama.cpp-form vocab GGUF, the
    # path every other consumer takes; it must reproduce tokenizer.json's ids exactly.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import llama3_bpe
    rebuilt = llama3_bpe.from_gguf(str(trained / V.GGUF_NAME))
    tok = tokenizers.Tokenizer.from_file(str(trained / "tokenizer.json"))
    for t in V.HARD_CASES + ["The quick brown fox number 12 jumps.", "Café 日本語 it's"]:
        assert rebuilt.encode(t).ids == tok.encode(t).ids, t


def test_cut_keeps_whole_characters_and_rejects_bad_bytes(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes(("é" * 10 + EOT + "zz").encode())
    train_text, rest = V.cut(str(p), 5)       # 5 bytes would split the third "é"
    assert train_text == "éé" and rest.startswith("é" * 8)
    p.write_bytes(b"ok" + EOT.encode() + b"\xff\xfe bad")
    with pytest.raises(UnicodeDecodeError):
        V.cut(str(p), 100)
