"""Per-turn state tokens through mtstudio, the GGUF and the IR (arch.state_tokens; IR_SPEC_DRAFT.md
sections 1 and 5 in experiments/tiny_chat_long_2026-10/: embedding state_weight, attention mask
"turn_state", kv slot evict "turn_state", paperkiln.state_tokens.k).

A tiny BPE chat model with k = 4 state slots and the turn_state mask trains on CPU on synthetic multi-
turn dialogues (a fact stated in turn 1 and asked for in a later turn, so it has to cross the state
rows), and exports F32 and Q8_0 GGUFs. Checks:
  - the serialiser: the state_tokens event's block counts equal tools/state_tokens.py's over the same
    stream (the early-stop rule included: one reply in the corpus has no <|endoftext|>), and an SFT run
    on segments rows and on a prompt/completion row whose history holds an unfinished reply;
  - the files pass spec/tools/ir_check.py; architecture paperkiln, paperkiln.state_tokens.k = 4,
    state_embd.weight [d, 4] stored F32 in both variants; the IR's embedding carries state_weight,
    its attention mask turn_state, its kv slot evict turn_state, family llama_state;
  - spec/tools/ir_numpy.py's forward over the F32 GGUF against `mtstudio score` on 40 multi-turn prompts
    whose ids hold the state blocks (prompt 7 holds a reply cut short, closed by the template):
    log-probabilities of numpy's greedy continuation within ROUNDING (1e-5 of max(1, max |ref|)) and
    greedy 40/40, for the full forward, the KV cache, the KV cache with eviction (prefill chunks that
    end at each state block, and token by token); eviction on against off identical to the bit; the
    cache holds n 4 state rows plus the current turn under eviction;
  - text prompts scored by mtstudio (its own insertion) equal the ids prompts (Python's insertion);
  - `mtstudio chat --kv-cache --evict` replies equal `chat` and `chat --kv-cache` over a multi-turn
    conversation, with fewer kv rows;
  - malformed or unwired specs are refused.

STATE_GATE_DIR=DIR also writes the gate files for ember (F32 GGUF, IR, Q8_0 GGUF, vocabulary, spec,
score_in.jsonl / score_out*.jsonl, the text prompts, a manifest with sha256) into DIR/state4/;
STATE_GATE_STEPS sets the gate model's training steps (default 600).
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN  # noqa: E402
import bpe  # noqa: E402
import ir_check  # noqa: E402
import state_tokens  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

K = 4
EOT, USER, ASSIST = 1, 2, 3  # tools/bpe.py's SPECIALS order: <unk>, <|endoftext|>, <|user|>, <|assistant|>
NAMES = ["ann", "bob", "cara", "dan", "eve", "finn", "gus", "hana"]
COLOURS = ["red", "blue", "green", "gold", "pink", "grey"]
PETS = ["cat", "dog", "fish", "bird", "frog"]
SMALL = [("how are you?", "i am well, thank you."), ("is it cold today?", "a little, take a coat."),
         ("tell me a joke.", "why did the bird sing? it was happy."), ("do you like tea?", "yes, tea is lovely."),
         ("what day is it?", "i am not sure, sorry.")]


def dialogue(rng, cut=False):
    """Turns (user, assistant): a name and a colour or a pet stated early, small talk, then a question."""
    name, colour, pet = rng.choice(NAMES), rng.choice(COLOURS), rng.choice(PETS)
    turns = [(f"my name is {name}.", f"hello {name}, nice to meet you.")]
    if rng.random() < 0.5:
        turns.append((f"i like {colour}.", f"{colour} is a fine colour."))
        q = ("what colour do i like?", f"you like {colour}.")
    else:
        turns.append((f"i have a {pet}.", f"a {pet} is a good friend."))
        q = ("what pet do i have?", f"you have a {pet}.")
    for _ in range(rng.randint(0, 2)):
        turns.append(rng.choice(SMALL))
    turns.append(rng.choice([q, ("what is my name?", f"your name is {name}.")]))
    return turns


def render(turns, cut_at=None):
    """The chat template's text; cut_at = i: turn i's reply stops after its first word, no <|endoftext|>."""
    out = ""
    for i, (u, a) in enumerate(turns):
        if i == cut_at:
            out += f"<|user|>{u}<|assistant|>{a.split()[0]}"
        else:
            out += f"<|user|>{u}<|assistant|>{a}<|endoftext|>"
    return out


def corpus(n=1500, seed=0):
    rng = random.Random(seed)
    docs = [render(dialogue(rng)) for _ in range(n)]
    docs[5] = render(dialogue(random.Random(99)), cut_at=1)  # a reply that stopped early
    return "\n".join(docs) + "\n"


def tok(tmp):
    return bpe.read_gguf(str(tmp / "v.gguf"))


def spec_of(tmp, name, steps=200, mask="turn_state", d=32, objective="lm", extra_train=None, extra_data=None,
            formats=("safetensors", "gguf", "gguf_q8_0"), arch_extra=None, state=True):
    tmp.mkdir(parents=True, exist_ok=True)
    if not (tmp / "v.gguf").exists():
        text = corpus()
        bpe.write_gguf(str(tmp / "v.gguf"), bpe.train(text, 400))
        (tmp / "c.txt").write_text(text, encoding="utf-8")
    arch = {"preset": "llama-tiny", "custom": {"d": d, "heads": 2, "layers": 2}}
    if state:
        arch["state_tokens"] = {"k": K, "mask": mask}
    arch.update(arch_extra or {})
    data = {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 128, "max_tokens": 0}
    data.update(extra_data or {})
    train = {"steps": steps, "lr": 0.004, "eval_every": 100, "checkpoint_every": 1000, "batch": 4, "seed": 3,
             "gradmap_every": 0, "objective": objective,
             "lr_schedule": {"warmup_steps": 20, "decay": "cosine", "min_lr_frac": 0.1}}
    train.update(extra_train or {})
    spec = {"name": name, "arch": arch, "data": data, "train": train, "export": {"formats": list(formats)},
            "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def run(spec):
    r = subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def events(spec):
    out = Path(json.loads(Path(spec).read_text())["out_dir"])
    return [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines() if x.strip()]


def exports(spec):
    return {e["format"]: e["path"] for e in events(spec) if e["event"] == "export" and e["format"].startswith("gguf")}


def weights(path):
    import numpy as np
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
    W = {}
    for name, dims, typ, off in tensors:
        assert typ == 0, name
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))
    return meta, W


def insert(ids, vocab):
    return state_tokens.insert(ids, vocab, K, EOT, USER, ASSIST)


CUT_PROMPT = 7


def prompts(tk, vocab, n=40, seed=11):
    """(text, ids): multi-turn prompts ending at the last question's <|assistant|>; the ids carry the state
    blocks (tools/state_tokens.py). Prompt CUT_PROMPT has its second reply cut short."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        turns = dialogue(rng)
        text = render(turns[:-1], cut_at=1 if i == CUT_PROMPT else None) + f"<|user|>{turns[-1][0]}<|assistant|>"
        ids, blocks, closed = insert(tk.encode(text), vocab)
        assert closed == (1 if i == CUT_PROMPT else 0)
        out.append((text, ids))
    return out


def greedy(doc, W, ids, vocab, n_new=10):
    """numpy greedy (<unk> banned, as mtstudio score's argmax); stops after <|endoftext|>."""
    import numpy as np
    import ir_numpy
    ids, out, lps = list(ids), [], []
    for _ in range(n_new):
        lg = ir_numpy.forward(doc, W, ids)[-1].astype(np.float64)
        ban = lg.copy()
        ban[0] = -np.inf
        t = int(np.argmax(ban))
        lse = lg.max() + np.log(np.exp(lg - lg.max()).sum())
        lps.append(lg[t] - lse)
        out.append(t)
        ids.append(t)
        if t == EOT:
            break
    return out, lps


def score(spec, rows, path, *flags):
    req = path.with_name(path.stem + "_in.jsonl")
    req.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    r = subprocess.run([BIN, "score", str(spec), "--in", str(req), "--out", str(path), *flags], capture_output=True,
                       text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("state")
    steps = int(os.environ.get("STATE_GATE_STEPS", "600")) if os.environ.get("STATE_GATE_DIR") else 200
    d = 64 if os.environ.get("STATE_GATE_DIR") else 32
    spec = spec_of(tmp, "state4", steps=steps, d=d)
    run(spec)
    return tmp, spec


@needs_bin
def test_serialiser_counts_and_model_event(model):
    tmp, spec = model
    ev = events(spec)
    st = [e for e in ev if e["event"] == "state_tokens"]
    assert len(st) == 1 and st[0]["corpus"] == "data.corpus"
    tk = tok(tmp)
    vocab = len(tk.tokens)
    m = next(e for e in ev if e["event"] == "model")
    assert m["vocab"] == vocab
    ids = tk.encode((tmp / "c.txt").read_text(encoding="utf-8"))
    want, blocks, closed = insert(ids, vocab)
    assert st[0]["tokens_before"] == len(ids) and st[0]["tokens_after"] == len(want)
    assert st[0]["blocks"] == blocks and st[0]["closed_early"] == closed == 1
    assert m["state_tokens"] == {"k": K, "mask": "turn_state", "params": K * m["d"]}
    steps = [e for e in ev if e["event"] == "step"]
    assert steps[-1]["loss"] < steps[0]["loss"] - 1.0, [e["loss"] for e in steps[::20]]


@needs_bin
def test_export_and_ir(model):
    tmp, spec = model
    ex = exports(spec)
    assert set(ex) == {"gguf", "gguf_q8_0"}
    d = next(e for e in events(spec) if e["event"] == "model")["d"]
    for path in ex.values():
        assert ir_check.check(path) == [], ir_check.check(path)
        meta, tensors, *_ = ir_check.read_gguf(path)
        assert meta["general.architecture"][1] == "paperkiln"
        assert meta["paperkiln.state_tokens.k"][1] == K
        st = [t for t in tensors if t[0] == "state_embd.weight"]
        assert len(st) == 1 and st[0][1] == [d, K] and st[0][2] == 0  # ne {d, k}, F32
    doc = json.loads(ir_check.read_gguf(ex["gguf"])[0]["paperkiln.ir"][1])
    assert doc["family_hint"] == "llama_state"
    assert doc["embed"]["nodes"] == [{"op": "embedding", "in": ["tokens"], "weight": "token_embd.weight",
                                      "state_weight": "state_embd.weight", "out": "x"}]
    att = [n for n in doc["block"]["nodes"] if n["op"] == "attention"]
    assert len(att) == 1 and att[0]["mask"] == "turn_state" and att[0]["window"] == 0 and att[0]["sinks"] == 0
    assert doc["block"]["state"][0]["evict"] == "turn_state"
    sp = doc["tokenizer"]["specials"]
    assert (sp["eot"], sp["user"], sp["assistant"]) == (EOT, USER, ASSIST)
    # Without the state tensor ir_check fails: the key is load-bearing.
    bad = json.loads(json.dumps(doc))
    bad["embed"]["nodes"][0]["state_weight"] = "nope.weight"
    meta, tensors, *_ = ir_check.read_gguf(ex["gguf"])
    assert ir_check.state_token_errors(bad, meta, tensors)
    bad = json.loads(json.dumps(doc))
    del bad["block"]["nodes"][[n["op"] for n in bad["block"]["nodes"]].index("attention")]["mask"]
    assert ir_check.state_token_errors(bad, meta, tensors)  # evict without the mask


def parity(tmp, spec, out_dir=None):
    import numpy as np
    f32 = exports(spec)["gguf"]
    meta, W = weights(f32)
    doc = json.loads(meta["paperkiln.ir"][1])
    tk = tok(tmp)
    vocab = doc["model"]["vocab"]
    ps = prompts(tk, vocab)
    rows, cont = [], []
    for text, ids in ps:
        assert state_tokens.error(ids, vocab, K, EOT, USER, ASSIST) == ""
        out, lps = greedy(doc, W, ids, vocab)
        rows.append({"prompt_ids": ids, "continuation_ids": out})
        cont.append((out, lps))
    od = Path(out_dir or tmp)
    res = {"full": score(spec, rows, od / "score_out.jsonl"),
           "kv": score(spec, rows, od / "score_out_kv.jsonl", "--kv-cache"),
           "evict": score(spec, rows, od / "score_out_evict.jsonl", "--kv-cache", "--evict"),
           "evict_step1": score(spec, rows, od / "score_out_evict_step1.jsonl", "--kv-cache", "--evict", "--kv-step",
                                "1")}
    summary = {"n": len(rows)}
    for key, got in res.items():
        worst, agree = 0.0, 0
        for (out, lps), g in zip(cont, got):
            assert not g["over_ctx"]
            ref = np.array(g["logprobs"])
            worst = max(worst, float(np.max(np.abs(np.array(lps) - ref)) / max(1.0, float(np.max(np.abs(ref))))))
            agree += g["argmax"] == out
        summary[key] = {"worst": worst, "agree": agree}
    # Eviction changes no logit: identical records but for kv_rows.
    same = sum(a["logprobs"] == b["logprobs"] and a["argmax"] == b["argmax"]
               for a, b in zip(res["kv"], res["evict"]))
    summary["evict_vs_keep_identical"] = same
    blocks = [g["state_blocks"] for g in res["kv"]]
    summary["kv_rows_keep"] = [g["kv_rows"] for g in res["kv"]]
    summary["kv_rows_evict"] = [g["kv_rows"] for g in res["evict"]]
    summary["state_blocks"] = blocks
    # Text prompts: mtstudio's own insertion gives the same ids, so the same scores to the bit.
    trows = [{"prompt": text, "continuation_ids": c[0]} for (text, _), c in zip(ps, cont)]
    tres = score(spec, trows, od / "score_out_text.jsonl")
    summary["text_vs_ids_identical"] = sum(a["logprobs"] == b["logprobs"] and a["prompt_tokens"] == len(r["prompt_ids"])
                                           for a, b, r in zip(tres, res["full"], rows))
    return summary, rows, ps, doc, f32


@needs_bin
def test_numpy_ir_vs_mtstudio_and_eviction(model):
    tmp, spec = model
    (tmp / "gate_work").mkdir(exist_ok=True)
    s, rows, ps, doc, f32 = parity(tmp, spec, tmp / "gate_work")
    print("state tokens k=4: " + json.dumps({k: v for k, v in s.items() if k not in ("kv_rows_keep", "kv_rows_evict",
                                                                                         "state_blocks")}))
    print(f"kv rows (keep / evict) for the first 5 prompts: {s['kv_rows_keep'][:5]} / {s['kv_rows_evict'][:5]}; "
          f"state blocks {s['state_blocks'][:5]}")
    assert s["n"] == 40
    for key in ("full", "kv", "evict", "evict_step1"):
        assert s[key]["agree"] == 40, (key, s[key])
        assert s[key]["worst"] <= 1e-5, (key, s[key])
    assert s["evict_vs_keep_identical"] == 40
    assert s["text_vs_ids_identical"] == 40
    for keep, ev, b in zip(s["kv_rows_keep"], s["kv_rows_evict"], s["state_blocks"]):
        assert b >= 1 and keep > ev
    # Under eviction: b blocks of K rows plus the current turn (the last <|user|> on).
    for ev, b, r in zip(s["kv_rows_evict"], s["state_blocks"], rows):
        ctx = r["prompt_ids"] + r["continuation_ids"][:-1]
        last_user = len(ctx) - 1 - ctx[::-1].index(USER)
        assert ev == K * b + (len(ctx) - last_user)
    if os.environ.get("STATE_GATE_DIR"):
        write_gate(tmp, spec, s, rows, ps, doc, f32)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gate(tmp, spec, s, rows, ps, doc, f32):
    out = Path(os.environ["STATE_GATE_DIR"]) / "state4"
    out.mkdir(parents=True, exist_ok=True)
    work = tmp / "gate_work"
    f32 = Path(f32)
    q8 = Path(exports(spec)["gguf_q8_0"])
    files = [f32, f32.with_suffix(".ir.json"), q8, tmp / "v.gguf"]
    for f in files:
        shutil.copyfile(f, out / f.name)
    (out / "score_in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for name in ("score_out.jsonl", "score_out_kv.jsonl", "score_out_evict.jsonl", "score_out_evict_step1.jsonl",
                 "score_out_text.jsonl"):
        shutil.copyfile(work / name, out / name)
    (out / "prompts_text.jsonl").write_text(
        "".join(json.dumps({"i": i, "prompt": t, "prompt_ids": ids, "truncated_reply": i == CUT_PROMPT}) + "\n"
                for i, (t, ids) in enumerate(ps)), encoding="utf-8")
    shutil.copyfile(spec, out / "spec.json")
    names = [f.name for f in files] + ["score_in.jsonl", "score_out.jsonl", "score_out_kv.jsonl",
                                       "score_out_evict.jsonl", "score_out_evict_step1.jsonl", "score_out_text.jsonl",
                                       "prompts_text.jsonl", "spec.json"]
    ev = events(spec)
    manifest = {
        "model": "state4", "family_hint": doc["family_hint"], "state_tokens": {"k": K, "mask": "turn_state"},
        "vocab": doc["model"]["vocab"], "state_ids": list(range(doc["model"]["vocab"], doc["model"]["vocab"] + K)),
        "specials": doc["tokenizer"]["specials"],
        "train": {"steps": next(e for e in ev if e["event"] == "start")["steps"],
                  "final_loss": [e for e in ev if e["event"] == "step"][-1]["loss"],
                  "state_tokens_event": next(e for e in ev if e["event"] == "state_tokens")},
        "truncated_reply_prompt": CUT_PROMPT,
        "numpy_vs_mtstudio": {k: v for k, v in s.items()},
        "files": {n: sha256(out / n) for n in names},
        "_note": "score_in.jsonl: prompt_ids (the template's state blocks included, built by tools/state_tokens.py; "
                 "prompt 7's second reply stops after one word and the template closes it with <|endoftext|> and a "
                 "block) and numpy's greedy continuation (stopping after <|endoftext|>). score_out.jsonl: mtstudio "
                 "score, full forward; score_out_kv.jsonl: --kv-cache (prefill chunks ending at each state block); "
                 "score_out_evict.jsonl: --kv-cache --evict (identical logprobs; kv_rows = rows per layer held after "
                 "the context: 4 per finished turn plus the current turn); score_out_evict_step1.jsonl: the same "
                 "token by token; score_out_text.jsonl: the text prompts in prompts_text.jsonl, mtstudio inserting "
                 "the blocks itself (identical to score_out.jsonl). Gate (IR_SPEC_DRAFT section 5, gate 2/5): greedy "
                 "40/40 and logprobs within ROUNDING, eviction on vs off bit-equal on CPU, rounding on CUDA."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


@needs_bin
def test_sft_rows_get_state_blocks(model):
    tmp, spec = model
    tk = tok(tmp)
    vocab = len(tk.tokens)
    rng = random.Random(5)
    rows, want_blocks, want_closed = [], 0, 0
    for i in range(30):
        turns = dialogue(rng)
        segs = []
        for u, a in turns:
            segs += [{"text": f"<|user|>{u}<|assistant|>", "loss": False}, {"text": a, "loss": True}]
        rows.append({"segments": segs})
        ids = []
        for sg in segs:
            ids += tk.encode(sg["text"]) + ([EOT] if sg["loss"] else [])
        out, b, c = insert(ids, vocab)
        want_blocks += b - 1  # the block that would end the row is left out
    # A prompt/completion row whose history holds a reply cut short.
    hist = render([("my name is ann.", "hello ann, nice to meet you."), ("i like red.", "red is a fine colour.")],
                  cut_at=1)
    rows.append({"prompt": hist + "<|user|>what is my name?<|assistant|>", "completion": "your name is ann."})
    out, b, c = insert(tk.encode(rows[-1]["prompt"]) + tk.encode(rows[-1]["completion"]) + [EOT], vocab)
    want_blocks += b - 1
    want_closed += c
    rows += rows[:4]  # a few more, so the held-out tail is not the special row
    for r in rows[-4:]:
        ids = []
        for sg in r["segments"]:
            ids += tk.encode(sg["text"]) + ([EOT] if sg["loss"] else [])
        want_blocks += insert(ids, vocab)[1] - 1
    (tmp / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    d = next(e for e in events(spec) if e["event"] == "model")["d"]
    sspec = spec_of(tmp, "sft4", steps=20, objective="sft", formats=("safetensors",), d=d,
                    extra_data={"sft_examples": str(tmp / "sft.jsonl")},
                    extra_train={"init_from": str(Path(json.loads(spec.read_text())["out_dir"]) / "state4.safetensors"),
                                 "eval_every": 10})
    run(sspec)
    st = [e for e in events(sspec) if e["event"] == "state_tokens"]
    assert len(st) == 1 and st[0]["corpus"] == "data.sft_examples"
    assert st[0]["blocks"] == want_blocks and st[0]["closed_early"] == want_closed == 1
    assert st[0]["rows_dropped_over_T"] == 0 and st[0]["turns_cut_over_T"] == 0
    assert all(e["loss"] == e["loss"] for e in events(sspec) if e["event"] == "step")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def chat_replies(spec, flags, convo):
    port = _free_port()
    proc = subprocess.Popen([BIN, "chat", str(spec), "--port", str(port), *flags], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if proc.poll() is not None:
                pytest.fail("mtstudio chat exited: " + proc.stdout.read())
            time.sleep(0.1)
        history, out = [], []
        for u in convo:
            j = _post(base + "/chat", {"user_input": u, "history": history, "top_k": 1, "max_new_tokens": 12})
            out.append((j["reply"], j.get("kv_rows")))
            history += [{"role": "user", "content": u}, {"role": "assistant", "content": j["reply"]}]
        return out
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@needs_bin
def test_chat_eviction_same_replies(model):
    tmp, spec = model
    convo = ["my name is eve.", "i have a frog.", "how are you?", "what is my name?", "what pet do i have?"]
    plain = chat_replies(spec, [], convo)
    kv = chat_replies(spec, ["--kv-cache"], convo)
    ev = chat_replies(spec, ["--kv-cache", "--evict"], convo)
    print("chat replies:", [r for r, _ in plain], "kv rows keep", [n for _, n in kv], "evict", [n for _, n in ev])
    assert [r for r, _ in plain] == [r for r, _ in kv] == [r for r, _ in ev]
    assert all(e < k for (_, k), (_, e) in zip(kv[1:], ev[1:]))


@needs_bin
@pytest.mark.parametrize("arch_extra,extra_train,formats,msg", [
    ({"state_tokens": {"k": 0}}, None, ("safetensors",), "k must be >= 1"),
    ({"state_tokens": {"mask": "window"}}, None, ("safetensors",), "mask must be turn_state or causal"),
    ({"state_tokens": {"slots": 4}}, None, ("safetensors",), "unknown key slots"),
    ({"moe": {"experts": 4, "top_k": 2}}, None, ("safetensors",), "with arch.moe is not wired"),
    ({"custom": {"d": 32, "heads": 2, "recursion": {"block": 2, "repeats": 2, "schedule": "mor"}}}, None,
     ("safetensors",), "schedule mor is not wired"),
    (None, None, ("safetensors", "gguf_q4_0"), "no state embedding"),
    ({"custom": {"d": 32, "heads": 2, "attn_res": {"form": "full"}}}, None, ("safetensors",),
     "with arch.custom.attn_res is not wired"),
])
def test_bad_specs_are_refused(tmp_path, arch_extra, extra_train, formats, msg):
    spec = spec_of(tmp_path, "bad", formats=formats, arch_extra=None)
    j = json.loads(spec.read_text())
    for k, v in (arch_extra or {}).items():
        if k == "state_tokens":
            j["arch"]["state_tokens"].update(v)
        elif k == "custom":
            j["arch"]["custom"].update(v)
        else:
            j["arch"][k] = v
    spec.write_text(json.dumps(j), encoding="utf-8")
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and msg in r.stderr, r.stderr


@needs_bin
def test_plan_prints_the_state_tokens(tmp_path):
    spec = spec_of(tmp_path, "plan")
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "state_tokens: k 4 after every finished reply, mask turn_state" in r.stdout, r.stdout


def test_python_template_matches_the_spec():
    V = 10
    ids = [2, 5, 3, 6, 1, 7, 2, 8, 3, 9, 2, 5, 3]  # reply 9 has no <|endoftext|>
    out, b, c = state_tokens.insert(ids, V, 2, EOT, USER, ASSIST)
    assert out == [2, 5, 3, 6, 1, 10, 11, 7, 2, 8, 3, 9, 1, 10, 11, 2, 5, 3] and (b, c) == (2, 1)
    assert state_tokens.error(out, V, 2, EOT, USER, ASSIST) == ""
    assert state_tokens.error([2, 5, 3, 6, 1, 11, 10], V, 2, EOT, USER, ASSIST)
    assert state_tokens.codes(out, V, 2) == [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4]
