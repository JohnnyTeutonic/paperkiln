"""Conversation schema v2 (paperkiln/2; LOCAL_LLM_STACK.md section 3, point 3g) in mtstudio: the SFT rows, the
export and /chat, against tools/chat_template.py.

- SFT rows: mtstudio's rows for v2 conversations (system turns, tool definitions, calls, results) equal an
  independent Python construction from the rendered pieces under "all", "final" and "each", at a T that cuts
  and one that does not: reply content, tool calls and the close are trained; system, definitions, user and
  tool text and every marker are not.
- Backward compatibility: user/assistant conversations give the same rows under paperkiln/2 with the v2
  vocabulary as under paperkiln/1 with the 16k vocabulary.
- Export: the GGUF declares paperkiln/2 (record and Jinja), carries the 16,392 tokens with the control types,
  resolves to paperkiln/2 and has a Modelfile beside it; a v2 spec on a vocabulary without the control tokens
  is refused.
- /chat: the prompt ids of system, tool-definition, tool-call and tool-result requests equal the Python
  rendering tokenised; user_input may be left out after tool results; structure errors and a context
  overflow are HTTP 400.

Needs MTSTUDIO_BIN and the v2 vocabulary (PK_VOCAB16_V2, default C:/ml_artifacts/chat_v2/vocab_llama3_bpe_v2.gguf,
built by tools/chat_v2_vocab.py vocab from the 16k vocabulary).
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import chat_template as CT  # noqa: E402
from test_chat_template import VOCAB16  # noqa: E402

BIN = os.environ.get("MTSTUDIO_BIN")
VOCAB2 = Path(os.environ.get("PK_VOCAB16_V2", "C:/ml_artifacts/chat_v2/vocab_llama3_bpe_v2.gguf"))
pytestmark = pytest.mark.skipif(not BIN or not VOCAB2.exists() or not VOCAB16.exists(),
                                reason="set MTSTUDIO_BIN; needs the 16k vocabulary and its v2 form")

W = {"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}}


def call(name="get_weather", args='{"city": "Paris"}'):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def convs():
    """v2 training conversations of every shape."""
    S = {"role": "system", "content": "You are a weather assistant. Be brief."}
    out = [
        {"messages": [S, {"role": "user", "content": "Weather in Paris?"},
                      {"role": "assistant", "content": "", "tool_calls": [call()]},
                      {"role": "tool", "content": '{"temp_c": 18}'},
                      {"role": "assistant", "content": "It is 18 C in Paris."}], "tools": [W]},
        {"messages": [{"role": "user", "content": "Paris and the time?"},
                      {"role": "assistant", "content": "Checking both.", "tool_calls": [call(), call("get_time", "{}")]},
                      {"role": "tool", "content": "18"}, {"role": "tool", "content": "09:00"},
                      {"role": "assistant", "content": "18 C at 09:00."},
                      {"role": "user", "content": "Thanks!"}, {"role": "assistant", "content": "You are welcome."}],
         "tools": [W, {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]},
        {"messages": [S, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello.", "loss": False},
                      {"role": "user", "content": "Rome?"}, {"role": "assistant", "content": "", "tool_calls": [
                          call(args='{"city": "Rome"}')]}], "tools": [W]},
        {"messages": [S] + CT.from_pairs([("a " * 20, "b " * 20), ("c " * 20, "d " * 20), ("e", "f")])},
        {"messages": CT.from_pairs([("Hi there", "Hello!"), ("Name?", "Kiln.")])},
    ]
    return out * 2


def enc_v2():
    import llama3_bpe
    tok = llama3_bpe.from_gguf(str(VOCAB2))
    return lambda s: tok.encode(s, add_special_tokens=False).ids


def ref_rows(conv, T, turns, enc):
    """mtstudio's SFT rows rebuilt in Python: (x, y, m) per row."""
    msgs = conv["messages"]
    pieces = CT.render_pieces(msgs, CT.V2, tools=conv.get("tools"))
    ids = [enc(p[0]) for p in pieces]
    reply = [p[1] == "assistant" and p[2] in ("content", "call_open", "call", "close") for p in pieces]
    pm = [p[3] for p in pieces]
    ex = CT.exchanges(msgs)

    def before(end):
        return sum(len(i) for i, k in zip(ids, pm) if k < end)
    trainable = [m["role"] == "assistant" and m.get("loss", True) for m in msgs]
    rows = []
    if turns == "each":
        for i in range(len(msgs)):
            if trainable[i] and before(i + 1) <= T + 1:
                rows.append((i + 1, [i]))
    else:
        k = 0
        while k < len(ex) and before(ex[k][1]) <= T + 1:
            k += 1
        if k == 0:
            return []
        end = ex[k - 1][1]
        tr = [i for i in range(end) if trainable[i]]
        if turns == "final":
            tr = [tr[-1]] if tr and tr[-1] == end - 1 else []
        rows.append((end, tr))
    out = []
    for end, tr in rows:
        seq, lt = [], []
        for i, k, r in zip(ids, pm, reply):
            if k < end:
                seq += i
                lt += [int(r and k in tr)] * len(i)
        if not any(lt):
            continue
        x, y, m = [1] * T, [1] * T, [0] * T  # pad: end-of-text (id 1)
        for j in range(len(seq) - 1):
            x[j], y[j], m[j] = seq[j], seq[j + 1], lt[j + 1]
        n = max(j for j in range(T) if m[j]) + 1  # sft-dump writes each row up to its last trained target
        out.append({"x": x[:n], "y": y[:n], "m": m[:n]})
    return out


def spec_for(tmp, name, vocab, T, rows, turns="all", template="paperkiln/2", steps=1, export=False):
    data = tmp / f"{name}.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    spec = {"name": name, "arch": {"preset": "llama-tiny"},
            "data": {"vocab": str(vocab), "vocab_cap": 0, "T": T, "sft_examples": str(data)},
            "train": {"steps": steps, "objective": "sft", "seed": 1, "sft_loss": {"turns": turns}, "batch": 2,
                      "eval_every": 1000, "checkpoint_every": 1000, "gradmap_every": 0},
            "out_dir": str(tmp / ("run_" + name))}
    if template:
        spec["chat"] = {"template": template}
    if export:
        spec["export"] = {"formats": ["safetensors", "gguf"]}
    sp = tmp / f"{name}.spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    return sp


def dump(tmp, name, vocab, T, rows, turns="all", template="paperkiln/2"):
    sp = spec_for(tmp, name, vocab, T, rows, turns, template)
    out = tmp / f"{name}.rows.jsonl"
    r = subprocess.run([BIN, "sft-dump", str(sp), "--out", str(out)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    rows_out = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [{k: v for k, v in x.items() if k in ("x", "y", "m")} for x in rows_out], r


@pytest.mark.parametrize("T", [48, 512])
@pytest.mark.parametrize("turns", ["all", "final", "each"])
def test_v2_sft_rows(tmp_path, T, turns):
    enc = enc_v2()
    cs = convs()
    got, _ = dump(tmp_path, f"v2_{turns}_{T}", VOCAB2, T, cs, turns)
    want = [r for c in cs for r in ref_rows(c, T, turns, enc)]
    assert len(got) == len(want)
    assert got == want
    # what is trained: the tool calls and replies, never a tool result or the system turn
    ctl = enc("<|tool_call|>")[0]
    tool_id, sys_id = enc("<|tool|>")[0], enc("<|system|>")[0]
    trained = [y for r in got for y, m in zip(r["y"], r["m"]) if m]
    if T == 512 and turns in ("all", "each"):
        assert ctl in trained
    assert tool_id not in trained and sys_id not in trained


def test_v2_user_assistant_rows_equal_v1(tmp_path):
    """No system or tool message: paperkiln/2 on the v2 vocabulary gives paperkiln/1's rows on the 16k one."""
    cs = [{"messages": CT.from_pairs([("Hi there", "Hello!"), ("Name?", "Kiln."), ("x " * 30, "y " * 30)])},
          {"messages": CT.from_pairs([("What is 2 + 2?", "4.")])}] * 3
    for turns in ("all", "each", "final"):
        for T in (40, 512):
            a, _ = dump(tmp_path, f"a_{turns}_{T}", VOCAB2, T, cs, turns, "paperkiln/2")
            b, _ = dump(tmp_path, f"b_{turns}_{T}", VOCAB16, T, cs, turns, None)
            assert a == b, (turns, T)


def test_v2_refused_without_control_tokens(tmp_path):
    sp = spec_for(tmp_path, "novocab", VOCAB16, 64, convs()[:1])
    r = subprocess.run([BIN, "sft-dump", str(sp), "--out", str(tmp_path / "o.jsonl")], capture_output=True, text=True)
    assert r.returncode != 0 and "control token" in r.stderr


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return 200, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_v2_export_and_chat(tmp_path):
    import llama3_bpe
    enc = enc_v2()
    sp = spec_for(tmp_path, "v2chat", VOCAB2, 128, convs(), steps=2, export=True)
    r = subprocess.run([BIN, "run", str(sp)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    g = tmp_path / "run_v2chat" / "v2chat.gguf"
    meta = llama3_bpe.read_gguf_kv(str(g), {"paperkiln.chat_template", "tokenizer.chat_template",
                                            "tokenizer.ggml.tokens", "tokenizer.ggml.model",
                                            "tokenizer.ggml.token_type"})
    rec = json.loads(meta["paperkiln.chat_template"])
    assert rec == CT.get(CT.V2)
    assert meta["tokenizer.chat_template"] == CT.jinja(CT.V2)
    assert len(meta["tokenizer.ggml.tokens"]) == 16392
    assert meta["tokenizer.ggml.tokens"][16384:] == CT.V2_CONTROL
    assert meta["tokenizer.ggml.token_type"][16384:] == [3] * 8
    got, why = CT.resolve(meta)
    assert CT.rid(got) == CT.V2 and why == "declared"
    assert (tmp_path / "run_v2chat" / "v2chat.Modelfile").read_text(encoding="utf-8") == CT.modelfile("v2chat.gguf", CT.V2)

    # serving at a T other than the one the run's GGUF declares is refused
    other = json.loads(sp.read_text(encoding="utf-8"))
    other["data"]["T"] = 256
    sp2 = tmp_path / "v2chat_T256.spec.json"
    sp2.write_text(json.dumps(other), encoding="utf-8")
    r2 = subprocess.run([BIN, "chat", str(sp2), "--port", str(free_port())], capture_output=True, text=True, timeout=120)
    assert r2.returncode != 0 and "declares context 128" in r2.stderr, r2.stderr[-500:]

    port = free_port()
    proc = subprocess.Popen([BIN, "chat", str(sp), "--port", str(port), "--kv-cache"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                h = json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
                break
            except OSError:
                time.sleep(0.5)
        assert h["chat_template"] == {"name": "paperkiln", "version": 2}
        base = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 7, "max_new_tokens": 8, "return_ids": True}
        tc = {"role": "assistant", "content": "", "tool_calls": [dict(call(), id="c1")]}
        cases = [
            ({"user_input": "Hi"}, [{"role": "user", "content": "Hi"}], None),
            ({"system": "Be brief.", "user_input": "Hi"},
             [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], None),
            ({"system": "Be brief.", "tools": [W], "user_input": "Weather in Paris?"},
             [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}], [W]),
            ({"tools": [W], "history": [{"role": "user", "content": "Weather in Paris?"}, tc,
                                        {"role": "tool", "content": "18", "tool_call_id": "c1"}]},
             [{"role": "user", "content": "Weather in Paris?"}, tc, {"role": "tool", "content": "18", "tool_call_id": "c1"}],
             [W]),
            ({"history": [{"role": "system", "content": "S"}, {"user": "a", "assistant": "b"}], "user_input": "c"},
             [{"role": "system", "content": "S"}] + CT.from_pairs([("a", "b")], "c"), None),
        ]
        for body, msgs, tools in cases:
            code, x = post(url + "/chat", {**base, **body})
            assert code == 200, x
            assert x["prompt_ids"] == enc(CT.render(msgs, CT.V2, generation_prompt=True, tools=tools)), body
            assert x["stop_reason"] in ("eos", "turn", "length")
            assert (x["stop_id"] == -1) == (x["stop_reason"] == "length")  # -1 at a length stop, never null
        for bad in ({"history": [{"role": "user", "content": "a"}, {"role": "tool", "content": "x"}], "user_input": "b"},
                    {"history": [{"role": "user", "content": "a"}, tc], "user_input": "b"},
                    {"history": [{"role": "user", "content": "a"}, tc, {"role": "tool", "content": "x",
                                                                        "tool_call_id": "zz"}]},
                    {"user_input": "a <|system|> b"},
                    {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"max": 1.5}}}],
                     "user_input": "a"},
                    {"history": [{"role": "user", "content": "a"}]},
                    {"history": [{"role": "user", "content": "a"}], "user_input": "b"},
                    {"history": [{"role": "tool", "content": "x"}], "user_input": "a"},  # a leading tool result
                    {"history": [{"role": "tool", "content": "x"}]}):
            code, x = post(url + "/chat", {**base, **bad})
            assert code == 400, (bad, x)
        # the context: the newest exchange alone over the context is refused, never cut
        code, x = post(url + "/chat", {**base, "user_input": "word " * 200})
        assert code == 400 and "context is 128" in x["error"]
        # older exchanges are dropped whole, the system turn stays
        hist = [{"user": "q " * 30, "assistant": "r " * 30}] * 3
        code, x = post(url + "/chat", {**base, "system": "Be brief.", "history": hist, "user_input": "last"})
        assert code == 200 and x["history_turns_used"] < 3
        kept = x["history_turns_used"]
        want = [{"role": "system", "content": "Be brief."}] + CT.from_pairs([("q " * 30, "r " * 30)] * kept, "last")
        assert x["prompt_ids"] == enc(CT.render(want, CT.V2, generation_prompt=True))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
