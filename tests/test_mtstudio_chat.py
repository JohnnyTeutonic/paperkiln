"""mtstudio chat, end-of-text and the corpus cap, end to end on CPU.

The tokenizer mirror runs anywhere. The rest trains a tiny llama-family
model for 300 steps on a synthetic dialogue corpus (about five seconds on
four CPU cores, enough to learn its five exchanges), then
starts `mtstudio chat` on a free localhost port and talks to it. It needs
a built binary: set MTSTUDIO_BIN, or build to ./build/mtstudio; it is
skipped otherwise. Everything is written to pytest's tmp_path (never the
OneDrive tree).
"""
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from get_tinystories_data import EOS, tokenize_like_mtstudio  # noqa: E402


def test_python_tokenizer_keeps_end_of_text_whole():
    toks = list(tokenize_like_mtstudio("The end.<|endoftext|>Once upon"))
    assert toks == ["the", "end", ".", EOS, "once", "upon"]
    assert list(tokenize_like_mtstudio("end<|endoftext|><|endoftext|>")) == ["end", EOS, EOS]
    # A partial marker is ordinary punctuation, as before.
    assert list(tokenize_like_mtstudio("a <|endof")) == ["a", "<", "|", "endof"]


# ---- end to end ----

def _binary():
    cand = os.environ.get("MTSTUDIO_BIN") or str(ROOT / "build" / "mtstudio")
    return cand if Path(cand).is_file() else None


BIN = _binary()
needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")


def _gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_vocab_gguf(path, tokens, eos_id=None):
    """Minimal tokenizer-only GGUF (v3) that mtstudio's reader accepts."""
    meta = [(_gguf_str("tokenizer.ggml.model") + struct.pack("<I", 8) + _gguf_str("word"))]
    meta.append(_gguf_str("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens))
                + b"".join(_gguf_str(t) for t in tokens))
    if eos_id is not None:
        meta.append(_gguf_str("tokenizer.ggml.eos_token_id") + struct.pack("<II", 4, eos_id))
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, 0, len(meta)) + b"".join(meta))


def read_gguf_meta(path):
    """uint32 metadata + the token list of a GGUF (enough for these checks)."""
    b = Path(path).read_bytes()
    p = 0

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from(fmt, b, p)
        p += struct.calcsize(fmt)
        return v[0]

    def rs():
        nonlocal p
        n = rd("<Q")
        s = b[p:p + n].decode()
        p += n
        return s

    assert rd("<I") == 0x46554747
    rd("<I")
    rd("<Q")
    out, sizes = {}, {4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    for _ in range(rd("<Q")):
        key, vt = rs(), rd("<I")
        if vt in sizes:
            out[key] = rd(sizes[vt])
        elif vt == 8:
            out[key] = rs()
        elif vt == 9:
            et, n = rd("<I"), rd("<Q")
            out[key] = [rs() if et == 8 else rd(sizes[et]) for _ in range(n)]
        else:
            raise ValueError(f"meta type {vt}")
    return out


DIALOGUE = [
    ("hello there", "hi! how are you?"),
    ("how are you?", "i am good, thank you."),
    ("what is your name?", "my name is kiln."),
    ("do you like cats?", "yes, i like cats and dogs."),
    ("tell me a story", "once upon a time there was a cat."),
]


def build_run(tmp, max_tokens=None, eos=True):
    corpus = tmp / "corpus.txt"
    text = "".join(f"user: {u} assistant: {a}{EOS if eos else ''}\n"
                   for _ in range(200) for u, a in DIALOGUE)
    corpus.write_text(text)
    counts = {}
    for t in tokenize_like_mtstudio(text):
        counts[t] = counts.get(t, 0) + 1
    words = sorted((w for w in counts if w != EOS), key=lambda w: -counts[w])
    # EOS last, not at the exporter's default id 1, so the export check
    # below can tell "found it" from "kept the default".
    tokens = ["<unk>"] + words + ([EOS] if eos else [])
    vocab = tmp / "vocab.gguf"
    write_vocab_gguf(vocab, tokens, len(tokens) - 1 if eos else None)
    out = tmp / "run"
    spec = {
        "name": "chatty",
        "arch": {"preset": "llama-tiny", "custom": {"d": 32, "layers": 1, "heads": 2}},
        "data": {"corpus": str(corpus), "vocab": str(vocab), "vocab_cap": 0, "T": 32},
        "train": {"steps": 300, "batch": 4, "lr": 1e-2, "eval_every": 100,
                  "checkpoint_every": 300},
        "export": {"formats": ["safetensors", "gguf"]},
        "out_dir": str(out),
    }
    if max_tokens is not None:
        spec["data"]["max_tokens"] = max_tokens
    sp = tmp / "spec.json"
    sp.write_text(json.dumps(spec))
    r = subprocess.run([BIN, "run", str(sp)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    return out, tokens, events


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    if BIN is None:
        pytest.skip("set MTSTUDIO_BIN to a built mtstudio")
    return build_run(tmp_path_factory.mktemp("chat"))


@needs_bin
def test_run_records_spec_and_exports_eos(trained):
    out, tokens, events = trained
    assert json.loads((out / "spec.json").read_text())["name"] == "chatty"
    meta = read_gguf_meta(out / "chatty.gguf")
    assert meta["tokenizer.ggml.tokens"][-1] == EOS
    assert meta["tokenizer.ggml.eos_token_id"] == len(tokens) - 1
    data = next(e for e in events if e["event"] == "data")
    assert data["tokens"] > 5000  # default cap (400000) is far above this corpus


@needs_bin
def test_corpus_cap_from_spec(tmp_path):
    _, _, events = build_run(tmp_path, max_tokens=2000)
    data = next(e for e in events if e["event"] == "data")
    assert data["tokens"] in (2000, 2001)  # the cap can admit one trailing token, as before


@needs_bin
def test_chat_server(trained):
    out, tokens, _ = trained
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([BIN, "chat", str(out), "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(200):
            try:
                if _get(base + "/health")[0] == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None:
                pytest.fail("mtstudio chat exited: " + proc.stdout.read())
            time.sleep(0.1)
        status, body = _get(base + "/health")
        health = json.loads(body)
        assert status == 200 and health["status"] == "ok" and health["eos"] is True
        assert health["vocab"] == len(tokens)

        status, page = _get(base + "/")
        assert status == 200 and b"<html" in page and b"chat" in page

        assert _get(base + "/card")[0] == 404
        (out / "card.json").write_text(json.dumps({"name": "chatty card"}))
        status, card = _get(base + "/card")
        assert status == 200 and json.loads(card)["name"] == "chatty card"

        # Greedy and seeded: well-formed, bounded, stops for a stated reason.
        for req in ({"user_input": "hello there", "top_k": 1, "max_new_tokens": 40},
                    {"user_input": "do you like cats?", "temperature": 0.8, "seed": 3,
                     "history": [{"role": "user", "content": "hello there"},
                                 {"role": "assistant", "content": "hi!"}]},
                    {"user_input": "zebra quantum", "top_p": 0.5, "max_new_tokens": 5}):
            status, j = _post(base + "/chat", req)
            assert status == 200, j
            assert isinstance(j["reply"], str) and j["reply"] == j["reply"].strip()
            assert j["stop_reason"] in ("eos", "turn", "length")
            assert 0 <= j["tokens"] <= req.get("max_new_tokens", 60)
            assert EOS not in j["reply"]
            assert "user:" not in j["reply"] and "assistant:" not in j["reply"]
            if j["stop_reason"] == "length":
                assert j["tokens"] == req.get("max_new_tokens", 60)

        # A learnt exchange ends its turn at end-of-text, not at the cap.
        status, j = _post(base + "/chat", {"user_input": "what is your name?", "top_k": 1})
        assert status == 200 and j["stop_reason"] == "eos" and j["reply"], j

        # Same seed, same reply.
        req = {"user_input": "tell me a story", "temperature": 1.0, "seed": 11}
        assert _post(base + "/chat", req)[1] == _post(base + "/chat", req)[1]

        assert _post(base + "/chat", b"not json")[0] == 400
        assert _post(base + "/chat", {"max_new_tokens": 3})[0] == 400
        assert _post(base + "/chat", {"user_input": "hi", "top_k": "many"})[0] == 400
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@needs_bin
def test_legacy_vocab_without_eos_still_chats(tmp_path):
    out, tokens, _ = build_run(tmp_path, eos=False)
    assert EOS not in tokens
    meta = read_gguf_meta(out / "chatty.gguf")
    assert meta["tokenizer.ggml.eos_token_id"] == 1  # the exporter's default, unchanged
    port = _free_port()
    proc = subprocess.Popen([BIN, "chat", str(out / "spec.json"), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(200):
            try:
                if _get(base + "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)
        status, j = _post(base + "/chat", {"user_input": "hello there", "top_k": 1,
                                           "max_new_tokens": 12})
        # No end-of-text in this vocabulary: the next "user :" ends the turn.
        assert status == 200 and j["stop_reason"] == "turn" and j["reply"], j
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
