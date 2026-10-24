"""tools/chat_template.py: the conversation schema and the template as data
(LOCAL_LLM_STACK.md section 3, points 3-5).

G1, v1 bit identity: the renderer gives exactly the strings today's builders gave (frozen
copies below, LEGACY_*), on every frozen prompt set, every Carry-5 and Carry-12 item in every
history mode, and the pinned human corpora (render_document reproduces each pinned text.txt
sha256); per-piece tokenisation equals whole-string tokenisation under both vocabularies.
G5 (paperkiln side): the generated Jinja renders as the record does (jinja2) on 1,000 random
conversations; the third-party records render as the families' own HF templates; the hand-written
edge cases in tests/chat_template_cases.json, which tests/test_chat_template.cpp also runs.
"""
import hashlib
import json
import os
import random
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "corpus" / "fetchers"))
import chat_template as CT  # noqa: E402

CASES = ROOT / "tests" / "chat_template_cases.json"
VOCAB16 = Path(os.environ.get("PK_VOCAB16", "C:/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"))
VOCAB8 = ROOT / "corpus" / "data" / "vocab_fineweb_tinychat_bpe8k_v1" / "fineweb_tinychat_bpe8k.gguf"
THIRD = Path(os.environ.get("PK_THIRDPARTY", "C:/ml_artifacts/thirdparty"))


# ---------------------------------------------------------------- the builders as they were (frozen copies)
def LEGACY_turn(u, a):  # chat_data.turn
    return f"<|user|>{u}<|assistant|>{a}<|endoftext|>"


def LEGACY_examples(messages):  # chat_data.examples
    hist = ""
    for k in range(0, len(messages) - 1, 2):
        u, a = messages[k]["content"], messages[k + 1]["content"]
        yield {"prompt": hist + f"<|user|>{u}<|assistant|>", "completion": a}
        hist += LEGACY_turn(u, a)


def LEGACY_segments(messages, k):  # chat_data.all_turns' row
    segs = []
    for j in range(k):
        segs.append({"text": f"<|user|>{messages[2 * j]['content']}<|assistant|>", "loss": False})
        segs.append({"text": messages[2 * j + 1]["content"], "loss": True})
    return {"segments": segs}


def LEGACY_render_dialogue(turns):  # _common.render_dialogue
    return "".join(f"<|user|>{turns[k]['content']}<|assistant|>{turns[k + 1]['content']}<|endoftext|>"
                   for k in range(0, len(turns) - 1, 2)) + "\n"


def LEGACY_template(history, user):  # carry5_eval.template
    return "".join(f"<|user|>{u}<|assistant|>{a}<|endoftext|>" for u, a in history) + f"<|user|>{user}<|assistant|>"


def LEGACY_prompt(p):  # chat_eval's single-turn prompts
    return f"<|user|>{p}<|assistant|>"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


PROMPT_SETS = ["chat_prompts_v1", "simple_qa_v1", "tinychat_hf_prompts_v1"]


def carry_prompts(path):
    """(legacy, new) prompt strings for every item and mode of a Carry set, the self mode with
    synthetic replies (an empty one, a newline-only one and longer ones among them)."""
    out = []
    for n, it in enumerate(read_jsonl(path)):
        tt = it["target_turn"]
        turns = it["turns"]
        gold_hist = [(t["user"], t["gold"]) for t in turns[:tt - 1]]
        fake = [("" if (n + k) % 17 == 0 else "\n" if (n + k) % 19 == 0 else t["gold"][::-1] + " ok")
                for k, t in enumerate(turns[:tt - 1])]
        self_hist = [(t["user"], r) for t, r in zip(turns, fake)]
        legacy = {"gold": LEGACY_template(gold_hist, turns[tt - 1]["user"]),
                  "twin": LEGACY_template([], it["twin"]), "noctx": LEGACY_template([], it["noctx"]),
                  "self": LEGACY_template(self_hist, turns[tt - 1]["user"])}
        for mode, s in legacy.items():
            new = CT.render(CT.from_carry_item(it, mode, fake), generation_prompt=True)
            out.append((s, new))
        # every turn of the self play: carry5_eval.template, as delegated
        import carry5_eval
        for k in range(len(turns)):
            h = self_hist[:k]
            out.append((LEGACY_template(h, turns[k]["user"]), carry5_eval.template(h, turns[k]["user"])))
    return out


# ---------------------------------------------------------------- G1
def test_g1_prompt_sets():
    n = 0
    for name in PROMPT_SETS:
        for p in read_jsonl(ROOT / "corpus" / "eval" / f"{name}.jsonl"):
            assert CT.render(CT.from_prompt(p["prompt"]), generation_prompt=True) == LEGACY_prompt(p["prompt"])
            assert CT.prompt([], p["prompt"]) == LEGACY_prompt(p["prompt"])
            n += 1
    assert n > 300


@pytest.mark.parametrize("name", ["carry5_v1", "carry12_v1"])
def test_g1_carry(name):
    pairs = carry_prompts(ROOT / "corpus" / "eval" / f"{name}.jsonl")
    bad = [(a, b) for a, b in pairs if a != b]
    assert not bad, bad[:2]
    assert len(pairs) > (5000 if name == "carry5_v1" else 1000)


def human_corpora():
    d = ROOT / "corpus" / "data"
    return sorted(p.name for p in d.glob("human_*_v1") if (p / "dialogues.jsonl").exists() and (p / "text.txt").exists())


@pytest.mark.parametrize("cid", human_corpora())
def test_g1_pinned_corpus_text(cid):
    """render_document over the pinned dialogues reproduces the pinned text.txt byte for byte,
    and chat_data's builders (now delegated) give the strings they gave before."""
    import _common as C
    import chat_data
    pinned = C.expected_hashes(cid)
    assert "text.txt" in pinned
    h = hashlib.sha256()
    n = 0
    for d in read_jsonl(ROOT / "corpus" / "data" / cid / "dialogues.jsonl"):
        m = d["messages"]
        s = CT.render_document(m)
        assert s == LEGACY_render_dialogue(m) == C.render_dialogue(m)
        h.update(s.encode("utf-8"))
        if n % 7 == 0:  # the SFT builders on a seventh of each corpus
            assert list(chat_data.examples(m)) == list(LEGACY_examples(m))
            k = len(m) // 2
            assert chat_data.all_turns(m, [(None, 0)] * k, 10 ** 9) == (LEGACY_segments(m, k), k)
            assert CT.to_segments(CT.from_pairs(chat_data._pairs(m)[:1])) == LEGACY_segments(m, 1)
        n += 1
    assert h.hexdigest() == pinned["text.txt"], cid


def _tokenizers():
    out = []
    if VOCAB8.exists():
        import bpe
        t = bpe.read_gguf(str(VOCAB8))
        out.append(("bpe8k", t.encode))
    if VOCAB16.exists():
        import llama3_bpe
        t16 = llama3_bpe.from_gguf(str(VOCAB16))
        out.append(("llama3_16k", lambda s: t16.encode(s, add_special_tokens=False).ids))
    return out


def test_g1_ids_pieces_equal_whole():
    """Both vocabularies: a rendering tokenised whole equals its pieces tokenised one by one (the
    property mtstudio's SFT loader and the C++ renderer rely on), and the specials are one id each."""
    toks = _tokenizers()
    if not toks:
        pytest.skip("no vocabulary on disk")
    convs = []
    for it in read_jsonl(ROOT / "corpus" / "eval" / "carry5_v1.jsonl")[::9]:
        convs.append((CT.from_carry_item(it, "gold"), True))
    for p in read_jsonl(ROOT / "corpus" / "eval" / "chat_prompts_v1.jsonl")[::5]:
        convs.append((CT.from_prompt(p["prompt"]) , True))
        convs.append(([{"role": "system", "content": "Answer in one word."}] + CT.from_prompt(p["prompt"]), True))
    for d in read_jsonl(ROOT / "corpus" / "data" / "human_dailydialog_v1" / "dialogues.jsonl")[:150]:
        convs.append((d["messages"], False))
    for name, enc in toks:
        for i, s in enumerate(CT.V1_SPECIALS[1:], 1):
            assert enc(s) == [i], (name, s)
        for m, gp in convs:
            pieces = CT.render_pieces(m, generation_prompt=gp)
            whole = enc("".join(p[0] for p in pieces))
            assert whole == [i for p in pieces for i in enc(p[0])], name


def test_vocab16_specials_at_0_3():
    """Resolution step 2 detects v1 by the token texts at ids 0-3, which both paperkiln BPE
    vocabularies carry (8k "bpe" and 16k "llama3_bpe")."""
    import llama3_bpe
    seen = 0
    for v in (VOCAB16, VOCAB8):
        if not v.exists():
            continue
        meta = llama3_bpe.read_gguf_kv(str(v))
        toks = [x.decode("latin-1") if isinstance(x, bytes) else x for x in meta["tokenizer.ggml.tokens"][:4]]
        assert toks == CT.V1_SPECIALS, v
        rec, why = CT.resolve(meta)
        assert CT.rid(rec) == CT.V1 and "ids 0-3" in why
        seen += 1
    if not seen:
        pytest.skip("no vocabulary on disk")


def test_resolution_order():
    v1 = {"tokenizer.ggml.tokens": CT.V1_SPECIALS + ["a"], "tokenizer.ggml.model": "gpt2"}
    assert CT.rid(CT.resolve(v1)[0]) == CT.V1
    declared = dict(v1, **{"paperkiln.chat_template": json.dumps(dict(CT.get(CT.V1), system={"mode": "none"}))})
    rec, why = CT.resolve(declared)
    assert why == "declared" and rec["system"]["mode"] == "none"
    word = {"tokenizer.ggml.tokens": ["<unk>", "the"], "tokenizer.ggml.model": "word"}
    assert CT.rid(CT.resolve(word)[0]) == CT.LEGACY_WORD
    none, why = CT.resolve({"tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.model": "gpt2"})
    assert none is None and "raw completion" in why
    bad = dict(v1, **{"paperkiln.chat_template": json.dumps(dict(CT.get(CT.V1), content="lower"))})
    with pytest.raises(CT.TemplateError):
        CT.resolve(bad)


def test_resolution_third_party_files():
    import llama3_bpe
    files = {"chatml-qwen2.5/1": THIRD / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct-fp16.gguf",
             "chatml-qwen3/1": THIRD / "qwen3-0.6b" / "Qwen3-0.6B-Q8_0.gguf",
             "gemma3/1": THIRD / "gemma-3-270m-it" / "gemma-3-270m-it-f32.gguf"}
    seen = 0
    for want, f in files.items():
        if not f.exists():
            continue
        rec, why = CT.resolve(llama3_bpe.read_gguf_kv(str(f)))
        assert rec is not None and CT.rid(rec) == want, (f, why)
        # Gemma 3 from llama.cpp b6000 carries no tokenizer.chat_template: the marker check finds it.
        assert ("markers" in why) == (want == "gemma3/1"), why
        seen += 1
    if not seen:
        pytest.skip("no third-party GGUFs on disk")


# ---------------------------------------------------------------- loss policy
def test_trained_policy():
    m = CT.from_pairs([("a", "b"), ("c", "d"), ("e", "f")])
    assert CT.trained(m, "all") == [(m, [1, 3, 5])]
    assert CT.trained(m, "final") == [(m, [5])]
    assert CT.trained(m, "each") == [(m[:2], [1]), (m[:4], [3]), (m, [5])]
    m[3]["loss"] = False
    assert CT.trained(m, "all") == [(m, [1, 5])]
    assert [k for _, k in CT.trained(m, "each")] == [[1], [5]]
    with pytest.raises(CT.TemplateError):
        CT.trained(m, "last")


# ---------------------------------------------------------------- parse
def test_parse_round_trips(tmp_path):
    m = CT.from_pairs([("Hi", "Hello"), ("How are you?", "Well.")])
    rows = [CT.to_segments(m), {"prompt": CT.prompt([("Hi", "Hello")], "How are you?"), "completion": "Well."},
            {"text": CT.render_document(m)}, {"prompt": "<|user|>x<|user|>y<|assistant|>", "completion": "z"}]
    m2 = [dict(x) for x in m]
    m2[1]["loss"] = False
    rows.append(CT.to_segments(m2))
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "conv.jsonl"
    r = CT.main(["parse", str(src), "--out", str(out)]) or json.loads((tmp_path / "conv.jsonl.ledger.json").read_text())
    assert r["rows"] == 4 and r["refused"] == 1 and r["by_form"] == {"segments": 2, "prompt_completion": 1, "document": 1}
    convs = read_jsonl(out)
    for c in convs[:3]:
        assert c["messages"] == m
    assert convs[0]["_sft_loss_turns"] == "all" and convs[1]["_sft_loss_turns"] == "final"
    assert convs[3]["messages"][1] == {"role": "assistant", "content": "Hello", "loss": False}
    assert r["source_sha256"] == CT.sha256_file(src)


# ---------------------------------------------------------------- G5: Jinja, third-party, edge cases
def _jinja_env():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment()

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env.globals["raise_exception"] = raise_exception
    return env, jinja2


WORDS = ["hello", "Paris", "the cat", "", "\n", "  spaced  ", "naïve café", "x" * 40, "a\nb", "{{ not jinja }}", "%}"]


def random_conversation(rng):
    msgs = []
    if rng.random() < 0.4:
        msgs.append({"role": "system", "content": rng.choice(WORDS[:3] + ["Be brief.", "Answer in one word."])})
    n = rng.randint(0, 6)
    for _ in range(n):
        msgs += [{"role": "user", "content": rng.choice(WORDS)}, {"role": "assistant", "content": rng.choice(WORDS)}]
    gp = rng.random() < 0.5
    if gp or not n:
        msgs.append({"role": "user", "content": rng.choice(WORDS)})
        gp = True
    return msgs, gp


def test_g5_jinja_matches_record():
    env, jinja2 = _jinja_env()
    rng = random.Random(24)
    for tid in (CT.V1, CT.LEGACY_WORD):
        for mode in ("fold", "none"):
            rec = CT.get(tid)
            rec["system"]["mode"] = mode
            jt = env.from_string(CT.jinja(rec))
            ok = refused = 0
            for _ in range(1000):
                m, gp = random_conversation(rng)
                try:
                    want = CT.render(m, rec, generation_prompt=gp)
                except CT.TemplateError:
                    with pytest.raises(jinja2.exceptions.TemplateError):
                        jt.render(messages=m, add_generation_prompt=gp)
                    refused += 1
                    continue
                assert jt.render(messages=m, add_generation_prompt=gp) == want
                ok += 1
            assert ok > 500 and (refused > 100) == (mode == "none")


def test_g5_third_party_records_match_hf():
    env, _ = _jinja_env()
    files = {"chatml-qwen2.5/1": THIRD / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct-fp16.gguf",
             "chatml-qwen3/1": THIRD / "qwen3-0.6b" / "Qwen3-0.6B-Q8_0.gguf",
             "gemma3/1": THIRD / "gemma-3-270m-it" / "chat_template.jinja"}
    import llama3_bpe
    rng = random.Random(7)
    seen = 0
    for tid, f in files.items():
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8") if f.suffix == ".jinja" else \
            llama3_bpe.read_gguf_kv(str(f), {"tokenizer.chat_template"})["tokenizer.chat_template"]
        assert hashlib.sha256(src.encode("utf-8")).hexdigest() in CT.get(tid)["match"]
        jt = env.from_string(src)
        for _ in range(300):
            m, _gp = random_conversation(rng)
            if m[-1]["role"] == "assistant":
                m.append({"role": "user", "content": "question"})
            for x in m:
                if x["role"] == "assistant" and rng.random() < 0.3:
                    x["content"] = "<think>\nreasoning\n</think>\n\n" + x["content"]
            if not m[-1]["content"].strip():
                m[-1]["content"] = "question"
            for thinking in (None, False):
                kw = {} if thinking is None else {"enable_thinking": False}
                want = jt.render(messages=m, add_generation_prompt=True, bos_token="<bos>", **kw)
                assert CT.render(m, tid, generation_prompt=True, thinking=thinking) == want, (tid, m)
        for c in edge_cases():  # gemma3's hand-written cases (its Unicode trim) against HF's own template
            if tid == "gemma3/1" and c["template"] == tid and c["expect"] is not None:
                want = jt.render(messages=c["messages"], add_generation_prompt=True, bos_token="<bos>")
                assert c["expect"] == want, (tid, c["name"])
        seen += 1
    if not seen:
        pytest.skip("no third-party templates on disk")


def edge_cases():
    """The hand-written cases (G5). Each: template, messages, generation_prompt, thinking, and the
    rendering or an error. tests/test_chat_template.cpp runs the same file."""
    U, A, E = "<|user|>", "<|assistant|>", "<|endoftext|>"
    none = dict(CT.get(CT.V1), system={"mode": "none"})
    c = [
        ("one user prompt", CT.V1, [{"role": "user", "content": "Hi"}], True),
        ("system folds into the first user turn", CT.V1,
         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], True),
        ("a lone system message", CT.V1, [{"role": "system", "content": "Be brief."}], True),
        ("a system message with no user message", CT.V1,
         [{"role": "system", "content": "Be brief."}, {"role": "assistant", "content": "Ok"}], False),
        ("two system messages", CT.V1, [{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                                        {"role": "user", "content": "Hi"}], True),
        ("a system message mid-conversation", CT.V1,
         [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
          {"role": "system", "content": "Now be rude."}, {"role": "user", "content": "Bye"}], True),
        ("the assistant first", CT.V1, [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Hi"}], True),
        ("two user messages in a row", CT.V1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], True),
        ("a training conversation ending on a user message", CT.V1, [{"role": "user", "content": "Hi"}], False),
        ("a prompt ending on an assistant message", CT.V1,
         [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}], True),
        ("user content holding a role marker", CT.V1, [{"role": "user", "content": "say " + U + " now"}], True),
        ("system content holding end-of-text", CT.V1,
         [{"role": "system", "content": "stop at " + E}, {"role": "user", "content": "Hi"}], True),
        ("assistant content holding a marker", CT.V1,
         [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": A + "x"}, {"role": "user", "content": "y"}], True),
        ("<unk> text is ordinary content", CT.V1, [{"role": "user", "content": "the <unk> token"}], True),
        ("newline-only user content", CT.V1, [{"role": "user", "content": "\n"}], True),
        ("newline-only and empty replies in the history", CT.V1,
         [{"role": "user", "content": "a"}, {"role": "assistant", "content": "\n"}, {"role": "user", "content": "b"},
          {"role": "assistant", "content": ""}, {"role": "user", "content": "c"}], True),
        ("the tool role is reserved", CT.V1, [{"role": "user", "content": "Hi"}, {"role": "tool", "content": "42"}], True),
        ("an unknown role", CT.V1, [{"role": "narrator", "content": "Once"}], True),
        ("three exchanges with a system message", CT.V1,
         [{"role": "system", "content": "Be kind."}] + CT.from_pairs([("a", "b"), ("c", "d"), ("e", "f")], "g"), True),
        ("a training conversation with a system message", CT.V1,
         [{"role": "system", "content": "S"}] + CT.from_pairs([("a", "b"), ("c", "d")]), False),
        ("loss on a user message", CT.V1, [{"role": "user", "content": "Hi", "loss": True}], True),
        ("unicode content", CT.V1, [{"role": "user", "content": "Grüße, 東京, 🙂"}], True),
        ("system mode none refuses a system message", none,
         [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}], True),
        ("system mode none without one", none, [{"role": "user", "content": "Hi"}], True),
        ("legacy word-level prompt", CT.LEGACY_WORD, CT.from_pairs([("hi", "hello")], "how are you"), True),
        ("gemma3 trims and folds", "gemma3/1",
         [{"role": "system", "content": " Be brief. "}, {"role": "user", "content": "  Hi \n"}], True),
        # HF's trim is str.strip(): Unicode whitespace at the ends goes, a zero-width space stays.
        ("gemma3 trims unicode edge whitespace", "gemma3/1",
         [{"role": "user", "content": " 　Hi there \n"}], True),
        ("gemma3 trims unicode whitespace in system and history", "gemma3/1",
         [{"role": "system", "content": " Be brief. "}, {"role": "user", "content": " Q1\u0085"},
          {"role": "assistant", "content": " A1 "}, {"role": "user", "content": "\x1cQ2\x1f"}], True),
        ("gemma3 keeps inner unicode whitespace and zero-width space", "gemma3/1",
         [{"role": "user", "content": "​A B​"}], True),
        ("qwen3 strips past reasoning", "chatml-qwen3/1",
         CT.from_pairs([("a", "<think>\nr\n</think>\n\nb")], "c"), True),
        ("qwen2.5 writes its default system turn", "chatml-qwen2.5/1", [{"role": "user", "content": "Hi"}], True),
    ]
    out = []
    for name, t, m, gp in c:
        row = {"name": name, "template": t, "messages": m, "generation_prompt": gp}
        try:
            row["expect"] = CT.render(m, t, generation_prompt=gp)
        except CT.TemplateError:
            row["expect"] = None
        out.append(row)
    out.append({"name": "qwen3 thinking off", "template": "chatml-qwen3/1", "messages": [{"role": "user", "content": "Hi"}],
                "generation_prompt": True, "thinking": False,
                "expect": CT.render([{"role": "user", "content": "Hi"}], "chatml-qwen3/1", True, thinking=False)})
    return out


def fixture():
    """Everything tests/test_chat_template.cpp checks the C++ twin against."""
    none = dict(CT.get(CT.V1), system={"mode": "none"})
    reg = CT._registry()
    jin = {"paperkiln/1": CT.jinja(CT.V1), "paperkiln/1 none": CT.jinja(none),
           "paperkiln/legacy-word": CT.jinja(CT.LEGACY_WORD)}
    res = [
        {"declared": "", "tokens": CT.V1_SPECIALS + ["a"], "model": "gpt2", "chat_template": "", "arch": "llama",
         "expect": CT.V1},
        {"declared": json.dumps(none), "tokens": CT.V1_SPECIALS, "model": "bpe", "chat_template": "", "arch": "llama",
         "expect": CT.V1},
        {"declared": "", "tokens": ["<unk>", "the"], "model": "word", "chat_template": "", "arch": "gpt2",
         "expect": CT.LEGACY_WORD},
        {"declared": "", "tokens": ["<bos>", "<start_of_turn>", "<end_of_turn>"], "model": "llama", "chat_template": "",
         "arch": "gemma3", "expect": "gemma3/1"},
        {"declared": "", "tokens": ["<bos>", "<start_of_turn>"], "model": "llama", "chat_template": "",
         "arch": "gemma3", "expect": None},
        {"declared": "", "tokens": ["a", "b"], "model": "gpt2", "chat_template": "{{ x }}", "arch": "llama",
         "expect": None},
    ]
    q25 = THIRD / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct-fp16.gguf"
    if q25.exists():
        import llama3_bpe
        jt = llama3_bpe.read_gguf_kv(str(q25), {"tokenizer.chat_template"})["tokenizer.chat_template"]
        res.append({"declared": "", "tokens": ["a"], "model": "gpt2", "chat_template": jt, "arch": "qwen2",
                    "expect": "chatml-qwen2.5/1"})
    for r in res:
        meta = {"tokenizer.ggml.tokens": r["tokens"], "tokenizer.ggml.model": r["model"],
                "general.architecture": r["arch"]}
        if r["declared"]:
            meta["paperkiln.chat_template"] = r["declared"]
        if r["chat_template"]:
            meta["tokenizer.chat_template"] = r["chat_template"]
        got = CT.resolve(meta)[0]
        assert (CT.rid(got) if got else None) == r["expect"], r
    return {"cases": edge_cases(), "records": {k: v for k, v in reg.items()}, "jinja": jin,
            "modelfile": {"paperkiln/1": CT.modelfile("model.gguf")}, "resolve": res,
            "sha256": {"abc": hashlib.sha256(b"abc").hexdigest(), "": hashlib.sha256(b"").hexdigest()}}


def test_g5_edge_cases_frozen():
    """The cases file is what the renderer gives now (the C++ test reads the same file); refused
    cases are refused, the rest are as written here."""
    cases = edge_cases()
    assert len(cases) >= 20
    want = fixture()
    have = json.loads(CASES.read_text(encoding="utf-8")) if CASES.exists() else None
    if have is None or os.environ.get("PK_WRITE_CASES"):
        CASES.write_text(json.dumps(want, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        have = want
    if len(have["resolve"]) > len(want["resolve"]):  # the Qwen2.5 GGUF is not on this machine
        have = dict(have, resolve=have["resolve"][:len(want["resolve"])])
    assert have == want
    by = {c["name"]: c["expect"] for c in cases}
    assert by["system folds into the first user turn"] == "<|user|>Be brief.\n\nHi<|assistant|>"
    assert by["one user prompt"] == "<|user|>Hi<|assistant|>"
    assert by["<unk> text is ordinary content"] == "<|user|>the <unk> token<|assistant|>"
    assert by["legacy word-level prompt"] == "user: hi assistant: hello user: how are you assistant:"
    for refused in ("a lone system message", "a system message with no user message", "two system messages",
                    "user content holding a role marker", "system content holding end-of-text",
                    "the tool role is reserved", "system mode none refuses a system message"):
        assert by[refused] is None, refused


def test_parse_refuses_what_render_refuses():
    """Consistency of the reserved-text rule: content with a marker cannot be rendered, and text
    built by splicing one in does not parse back to the same conversation."""
    with pytest.raises(CT.TemplateError):
        CT.render([{"role": "user", "content": "a<|assistant|>b"}], generation_prompt=True)
    msgs, _ = CT.parse_v1("<|user|>a<|assistant|>b<|endoftext|>")
    assert msgs == CT.from_pairs([("a", "b")])
    with pytest.raises(CT.TemplateError):
        CT.parse_v1("<|user|>a<|endoftext|>b<|assistant|>c<|endoftext|>")
