"""Base 2's corpus build: the ratio arithmetic, the document filter, the pins, and base2_data.sh end to end on a few
megabytes of toy sources against a store on local disk (hf_store.py's dir: backend), with the real tokeniser and the
real scenario engine.

No network, no GPU. The end-to-end tests need the 16k vocabulary (PK_VOCAB16) and the scenario engine's folder
(BASE2_TEST_ENGINE, in the author's research tree); they are skipped without them. POSIX only."""
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "corpus" / "fetchers"))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX paths and bash")

import base2_corpus as B  # noqa: E402
import base2_pins as P  # noqa: E402

VOCAB = Path(os.environ.get("PK_VOCAB16", "/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"))
ENGINE = Path(os.environ.get("BASE2_TEST_ENGINE", "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/"
                                                  "AI_ML/chat_screens_claude/scenario_engine"))
SEP = "<|endoftext|>\n"
# The plan's section 2.2 sizes: one copy of each source after filtering (the human text is 198 MB, section 2.1).
PLAN_AVAIL = {"fineweb": 13.8e9, "simplewiki": 252e6, "human": 198123090, "soda": 832205592, "ultrachat": 1173884012,
              "everyday": 1853283}


# ---------------------------------------------------------------- arithmetic
def test_recipe_n_is_the_plans_table_at_the_plans_sizes():
    r = B.recipe_n(PLAN_AVAIL)
    main_mb, tail_mb = r["total_bytes"] * (1 - r["tail_fraction"]) / 1e6, r["total_bytes"] * r["tail_fraction"] / 1e6
    # 14,000 and 2,600 MB in the plan with every dialogue byte used; here 99% of each source is used.
    assert abs(main_mb / 0.99 - 14000) < 30 and abs(tail_mb / 0.99 - 2600) < 6
    plan_main = {"fineweb": 0.74, "simplewiki": 0.018, "human": 0.0312, "soda": 0.0874, "ultrachat": 0.1232, "everyday": 0.0002}
    plan_tail = {"fineweb": 0.53, "simplewiki": 0.0, "human": 0.0606, "soda": 0.1697, "ultrachat": 0.2394, "everyday": 0.0004}
    for s in B.NATURAL:
        assert abs(r["main"][s] - plan_main[s]) < 6e-4, s
        assert abs(r["tail"][s] - plan_tail[s]) < 6e-4, s
    assert abs(sum(r["main"].values()) - 1) < 1e-12 and abs(sum(r["tail"].values()) - 1) < 1e-12
    assert abs(r["dialogue_share_of_bytes"] - 0.28) < 0.005   # A's overall dialogue share, 28% here
    # No source is asked for more than 99% of its copies, so a part's last whole document never runs a source dry.
    for s in B.NATURAL:
        assert r["demand_bytes"][s] <= B.COPIES[s] * PLAN_AVAIL[s] * 0.99 * (1 + 1e-9), s
    for s in B.DIALOGUE:
        assert abs(r["demand_bytes"][s] - B.COPIES[s] * PLAN_AVAIL[s] * 0.99) < 4, s
    assert B.COPIES == {"fineweb": 1, "simplewiki": 1, "human": 3, "soda": 2, "ultrachat": 2, "everyday": 2}


def test_recipe_n_follows_the_measured_sizes_and_refuses_a_short_fineweb():
    a = dict(PLAN_AVAIL, simplewiki=300e6, soda=700e6)
    r = B.recipe_n(a)
    main_bytes = r["total_bytes"] * (1 - r["tail_fraction"])
    assert abs(r["main"]["simplewiki"] * main_bytes - 0.99 * 300e6) < 2
    assert abs(r["main"]["fineweb"] + r["main"]["simplewiki"] - 0.758) < 1e-12
    assert r["tail"]["fineweb"] == 0.53 and r["tail"]["simplewiki"] == 0
    assert r["total_bytes"] < B.recipe_n(PLAN_AVAIL)["total_bytes"]   # less dialogue, a smaller corpus at the same shares
    with pytest.raises(SystemExit):
        B.recipe_n(dict(PLAN_AVAIL, fineweb=10e9))
    with pytest.raises(SystemExit):
        B.recipe_n({k: v for k, v in PLAN_AVAIL.items() if k != "everyday"})
    capped = B.recipe_n(dict(PLAN_AVAIL, simplewiki=900e6))
    assert capped["simplewiki_capped"] and capped["main"]["simplewiki"] == B.SIMPLEWIKI_MAX


def n_records(rn, bpt_main=4.375, bpt_tail=4.409):
    main_b, tail_b = int(rn["total_bytes"] * (1 - rn["tail_fraction"])), int(rn["total_bytes"] * rn["tail_fraction"])
    led = {"seed": 1, "sources": {s: {} for s in B.NATURAL}, "parts": {"main": {"bytes": main_b}, "tail": {"bytes": tail_b}}}
    return led, {"main": int(main_b / bpt_main), "tail": int(tail_b / bpt_tail)}


@pytest.mark.parametrize("scn_bpt", [2.9, 3.6, 4.38, 5.2])
def test_recipe_s_gives_fifteen_percent_of_tokens_at_any_bytes_a_token(scn_bpt):
    rn = B.recipe_n(PLAN_AVAIL)
    led, toks = n_records(rn)
    scn_tokens = 39_000_000
    r = B.recipe_s(rn, led, toks, int(scn_tokens * scn_bpt), scn_tokens, 200000, 200000, 0.02)
    for p in ("main", "tail"):
        part = r["parts"][p]
        scn_t = part["scenario_bytes"] / scn_bpt
        nat_t = part["natural_bytes"] / part["n_bytes_per_token"]
        assert abs(scn_t / (scn_t + nat_t) - 0.15) < 1e-6
        assert abs(scn_t + nat_t - toks[p]) / toks[p] < 1e-6        # S's token count is N's
        assert abs(sum(r[p].values()) - 1) < 1e-9
        for s in B.NATURAL:                                         # every natural budget is 85% of N's
            assert abs(r[p][s] * part["bytes"] - 0.85 * rn[p][s] * led["parts"][p]["bytes"]) < 2, (p, s)
        # The byte ratio is 15% only when the two kinds of text tokenise alike.
        assert (r[p]["scenario"] < 0.15) == (scn_bpt < part["n_bytes_per_token"])
    assert abs(r["total_bytes"] - (r["parts"]["main"]["bytes"] + r["parts"]["tail"]["bytes"])) <= 1
    need = r["parts"]["main"]["scenario_bytes"] + r["parts"]["tail"]["scenario_bytes"]
    assert r["scenario_documents"] * r["scenario_bytes_per_document"] >= need * 1.03 / 0.98
    assert r["scenario_documents"] == r["scenario_shards"] * 200000


def test_subset_reads_the_ledgers():
    n = {"seed": 1, "sources": {"a": {"sha256": "x", "documents_used": 10}, "b": {"sha256": "y", "documents_used": 5}}}
    s = {"seed": 1, "sources": {"a": {"sha256": "x", "documents_used": 9}, "b": {"sha256": "y", "documents_used": 5},
                                "scenario": {"sha256": "z", "documents_used": 3}}}
    assert B.subset(n, s) == []
    s["sources"]["a"]["documents_used"] = 11
    assert B.subset(n, s)
    s["sources"]["a"] = {"sha256": "other", "documents_used": 1}
    assert B.subset(n, s)
    assert B.subset(n, {"seed": 2, "sources": n["sources"]})


# ---------------------------------------------------------------- documents
def test_docfilter_drops_held_out_text_and_excluded_documents(tmp_path):
    import _common as C
    a = " ".join(f"word{i}" for i in range(20))
    held = [{"role": "user", "content": a}, {"role": "assistant", "content": "fine then"}]
    val = tmp_path / "val.txt"
    val.write_text(C.render_dialogue(held), encoding="utf-8")
    sft = tmp_path / "sft.jsonl"
    sft_words = "only sft text " + " ".join(f"x{i}" for i in range(14))
    sft.write_text(json.dumps({"messages": [{"role": "user", "content": sft_words}]}) + "\n")
    panel = tmp_path / "panel.txt"
    panel.write_text("a held-out scenario document\n" + SEP)
    docs = ["plain article " + a + "\n",                                       # a val 13-gram in a plain document
            "<|user|>" + sft_words + "<|assistant|>ok",                        # an SFT 13-gram in a chat document
            "a held-out scenario document\n",                                  # equal to a panel document
            "an ordinary article about nothing in particular\n",
            # a rendered dialogue: its last exchange's end-of-text marker and newline are the document separator
            "<|user|>hello there<|assistant|>hello to you<|endoftext|><|user|>and again<|assistant|>yes"]
    src = tmp_path / "in.txt"
    src.write_text("".join(d + SEP for d in docs), encoding="utf-8")
    led = B.docfilter(str(ROOT), [str(val), str(sft)], str(panel), str(src), str(tmp_path / "out.txt"), str(tmp_path / "led.json"))
    assert (led["documents_in"], led["documents_kept"]) == (5, 2)
    assert (led["dropped_heldout_13gram"], led["dropped_equal_excluded"], led["dropped_eval_rules"]) == (2, 1, 0)
    kept = (tmp_path / "out.txt").read_text()
    assert kept == docs[3] + SEP + docs[4] + SEP
    assert led["sha256"] == hashlib.sha256(kept.encode()).hexdigest() and led["carry5_v1_sha256"] and led["carry12_v1_sha256"]
    assert B.doc_views(docs[4]) == (["hello there", "and again"], "\nhello there\nhello to you\n\nand again\nyes")


def test_docfilter_applies_the_eval_rules_to_user_turns_and_text(tmp_path):
    """A document that repeats a frozen eval prompt goes, whether it is a chat turn or running text."""
    rows = [json.loads(ln) for ln in open(ROOT / "corpus" / "eval" / "chat_prompts_v1.jsonl", encoding="utf-8") if ln.strip()]
    prompt = next(r["prompt"] for r in rows if len(r["prompt"].split()) >= 10 and "<|" not in r["prompt"]
                  and "\n" not in r["prompt"])
    (tmp_path / "val.txt").write_text("<|user|>nothing<|assistant|>at all<|endoftext|>\n")
    (tmp_path / "sft.jsonl").write_text("")
    src = tmp_path / "in.txt"
    src.write_text(f"<|user|>{prompt}<|assistant|>an answer{SEP}" + f"Some text. {prompt} More text.\n{SEP}"
                   + f"harmless words only\n{SEP}", encoding="utf-8")
    led = B.docfilter(str(ROOT), [str(tmp_path / "val.txt"), str(tmp_path / "sft.jsonl")], None, str(src),
                      str(tmp_path / "out.txt"), str(tmp_path / "led.json"))
    assert (led["documents_in"], led["documents_kept"], led["dropped_eval_rules"]) == (3, 1, 2)


def test_head_docs_cuts_at_a_document(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("".join(f"document {i} é\n{SEP}" for i in range(50)), encoding="utf-8")
    n = B.head_docs(str(src), str(tmp_path / "b.txt"), 200)
    out = (tmp_path / "b.txt").read_bytes()
    assert n == len(out) <= 200 and out.endswith(SEP.encode()) and src.read_bytes().startswith(out)


def test_fineweb_fetcher_names_every_shard_of_the_sample():
    import fineweb_edu as F
    assert len(F.SHARDS) == 14
    assert F.SHARDS["shard0_v1"] == "sample/10BT/000_00000.parquet" and F.SHARDS["shard13_v1"] == "sample/10BT/013_00000.parquet"
    assert F.REV == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    for n in (0, 1):   # the two pinned shards keep their pins
        assert (ROOT / "corpus" / "hashes" / f"fineweb_edu_shard{n}_v1.sha256").exists()


def test_data_script_is_the_plans_build():
    text = (CC / "base2_data.sh").read_text()
    assert "\r" not in text
    subprocess.run(["bash", "-n", str(CC / "base2_data.sh")], check=True)
    assert os.access(CC / "base2_data.sh", os.X_OK)
    assert 'FW_SHARDS="2 3 4 5"' in text and "shard0_v1/val.txt" in text
    assert "tc_b1_filter.py" in text and "--carry12" in text and "simplewiki --allow-sa" in text
    for src in ("soda", "ultrachat", "everyday", "taskmaster3", "abcd", "ccpe", "p4g", "aya_en", "msc"):
        assert f"--source \"{src}=" in text, src
    assert 'copies 3 "$WORK/human3.txt"' in text and 'copies 2 "$WORK/soda2.txt"' in text
    assert "--seed 1" in text and 'PINS "$ENGINE" engine' in text
    # S-full's data job, word for word as base A's data job ran it: the panels and the split depend on it.
    b1 = (CC / "base_b1_data.sh").read_text()
    line = next(ln for ln in b1.splitlines() if "--shares 35" in ln)
    assert line.strip() in text
    assert next(ln for ln in b1.splitlines() if ln.startswith("S13=")) in text
    assert next(ln for ln in b1.splitlines() if ln.startswith("LONG=")) in text


# ---------------------------------------------------------------- pins
def sha(b):
    return hashlib.sha256(b).hexdigest()


def toy_pins():
    return {"files": {"f.txt": sha(b"f")}, "sources": {"src/a.txt": None}, "engine": {"commit": None, "files": {}},
            "N": {"N.main.ids": None, "N.tail.ids": None}, "S": {"S.main.ids": None, "S.tail.ids": None},
            "panels": {"panel_fineweb.ids": sha(b"pf"), "panel_dialogue.ids": sha(b"pd"), "panel_scenario.ids": None},
            "vocab": {"vocab_llama3_bpe.gguf": sha(b"v")}, "a_weights": {"tc_b1_r1.safetensors": sha(b"w")},
            "rehearsal": {"rehearsal.ids": None}}


def test_pins_record_on_a_build_and_refuse_on_an_arm(tmp_path):
    d, up = tmp_path / "data", tmp_path / "upload"
    d.mkdir()
    up.mkdir()
    for name, body in (("N.main.ids", b"nm"), ("N.tail.ids", b"nt"), ("S.main.ids", b"sm"), ("S.tail.ids", b"st"),
                       ("panel_fineweb.ids", b"pf"), ("panel_dialogue.ids", b"pd"), ("panel_scenario.ids", b"ps")):
        (d / name).write_bytes(body)
    (up / "vocab_llama3_bpe.gguf").write_bytes(b"v")
    (up / "tc_b1_r1.safetensors").write_bytes(b"w")
    pins = toy_pins()
    assert P.check(pins, str(d), "build-N")[0] == [] and P.check(pins, str(d), "build-S")[0] == []
    for mode in ("arm-N", "arm-S", "arm-C"):
        assert any("no pin" in x for x in P.check(pins, str(d), mode, str(up))[0]), mode
    done = P.adopt(pins, {m: {"observed": P.check(pins, str(d), m)[1]} for m in ("build-N", "build-S")})
    assert sorted(done) == ["N/N.main.ids", "N/N.tail.ids", "S/S.main.ids", "S/S.tail.ids", "panels/panel_scenario.ids"]
    for mode in ("arm-N", "arm-S", "arm-C"):
        assert P.check(pins, str(d), mode, str(up))[0] == [], mode
    (d / "S.tail.ids").write_bytes(b"other")
    assert P.check(pins, str(d), "arm-N", str(up))[0] == []
    assert P.check(pins, str(d), "arm-S", str(up))[0] and P.check(pins, str(d), "arm-C", str(up))[0]
    assert P.check(pins, str(d), "build-S")[0]                       # a set pin is compared on a build too
    with pytest.raises(SystemExit):
        P.adopt(pins, {"build-S": {"observed": P.check(pins, str(d), "build-S")[1]}})
    (d / "S.tail.ids").write_bytes(b"st")
    (up / "tc_b1_r1.safetensors").write_bytes(b"not A")
    assert P.check(pins, str(d), "arm-S", str(up))[0] == [] and P.check(pins, str(d), "arm-C", str(up))[0]
    (up / "vocab_llama3_bpe.gguf").write_bytes(b"x")
    assert P.check(pins, str(d), "arm-N", str(up))[0] and P.check(pins, str(d), "vocab", str(up))[0]
    (d / "panel_fineweb.ids").write_bytes(b"x")
    assert any("panel_fineweb" in x for x in P.check(pins, str(d), "build-S")[0])
    (d / "N.main.ids").unlink()
    assert any("missing" in x for x in P.check(pins, str(d), "build-N")[0])


def test_engine_pins_refuse_until_pinned_and_refuse_a_stray_module(tmp_path):
    e = tmp_path / "engine"
    e.mkdir()
    for f in P.ENGINE_FILES:
        (e / f).write_text(f"# {f}\n")
    (e / "test_scenario_engine.py").write_text("# tests are not sent\n")
    pins = toy_pins()
    pins["engine"]["files"] = {f: None for f in P.ENGINE_FILES}
    assert len(P.check(pins, str(e), "engine")[0]) == len(P.ENGINE_FILES)        # null pins refuse
    assert P.adopt(pins, {"engine": {"observed": P.check(pins, str(e), "engine")[1]}}) == []   # and are never adopted
    P.pin_engine(pins, str(e), commit="abc")
    assert pins["engine"]["commit"] == "abc" and P.check(pins, str(e), "engine")[0] == []
    (e / "extra.py").write_text("x = 1\n")
    assert any("not pinned" in x for x in P.check(pins, str(e), "engine")[0])
    (e / "extra.py").unlink()
    (e / "banks.py").write_text("# revised\n")
    assert P.check(pins, str(e), "engine")[0]
    with pytest.raises(SystemExit):
        P.pin_engine(pins, str(e))
    P.pin_engine(pins, str(e), repin=True)
    assert P.check(pins, str(e), "engine")[0] == []
    (e / "rules.py").unlink()
    with pytest.raises(SystemExit):
        P.pin_engine(pins, str(e), repin=True)


def test_committed_pins():
    pins = json.load(open(CC / "base2_pins.json"))
    b1 = json.load(open(CC / "tc_b1_pins.json"))
    assert pins["files"] == b1["files"]                         # stage 1 reruns base A's first job on the same sources
    assert pins["panels"]["panel_fineweb.ids"].startswith("c1aa6d42") and pins["panels"]["panel_dialogue.ids"].startswith("7881aa1c")
    assert set(pins["engine"]["files"]) >= set(P.ENGINE_FILES)
    assert all(len(v) == 64 for v in (*pins["vocab"].values(), *pins["a_weights"].values()))
    for g in ("sources", "N", "S", "rehearsal"):
        assert pins[g], g
    for mode, (groups, _) in P.MODES.items():
        for name, _base in groups:
            assert name in pins, (mode, name)
    if VOCAB.exists():
        assert P.sha256(str(VOCAB)) == pins["vocab"]["vocab_llama3_bpe.gguf"]


def test_pins_cli_writes_each_modes_record(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "N.main.ids").write_bytes(b"1")
    (d / "N.tail.ids").write_bytes(b"2")
    pf = tmp_path / "pins.json"
    pf.write_text(json.dumps(toy_pins()))
    assert P.main([str(d), "build-N", "--pins", str(pf), "--out", str(tmp_path / "o" / "pins.json")]) == 0
    obs = json.load(open(d / "base2_pins_observed.json"))
    assert obs["build-N"]["observed"]["N/N.main.ids"] == sha(b"1") and json.load(open(tmp_path / "o" / "pins.json")) == obs
    with pytest.raises(SystemExit):
        P.main([str(d), "arm-N", "--pins", str(pf), "--upload", str(d)])
    assert set(json.load(open(d / "base2_pins_observed.json"))) == {"build-N", "arm-N"}
    assert P.main(["adopt", str(d / "base2_pins_observed.json"), "--pins", str(pf)]) == 0
    assert json.load(open(pf))["N"]["N.main.ids"] == sha(b"1")


def test_store_has_reads_a_local_store(tmp_path):
    if not (CC / "hf_store.py").exists():
        pytest.skip("hf_store.py is the store agent's file")
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'store'}"}))
    d = tmp_path / "set"
    d.mkdir()
    (d / "a.txt").write_text("a")
    os.environ["BASE2_STORE_RETRY_S"] = "0"
    assert B.store_has(str(cfg), "base2-build", "build/stage1") is False
    subprocess.run([sys.executable, str(CC / "hf_store.py"), "push", "--config", str(cfg), "--branch", "base2-build",
                    "--prefix", "build/stage1", "--dir", str(d)], check=True, capture_output=True)
    assert B.store_has(str(cfg), "base2-build", "build/stage1") is True
    assert B.store_has(str(cfg), "main", "build/stage1") is False
    assert B.main(["store-has", str(tmp_path / "absent.json"), "main", "x"]) == 2   # an unreadable store is not "absent"


# ---------------------------------------------------------------- end to end on toy sources
SYL = ["ba", "ko", "lu", "ti", "re", "mo", "sa", "ne", "vi", "du", "pa", "gi", "zo", "fe", "ka", "ru", "mi", "to", "le", "wa"]


def word(rng):
    return "".join(rng.choice(SYL) for _ in range(rng.randint(1, 4)))


def sentence(rng, n):
    return " ".join(word(rng) for _ in range(n))


def article(rng):
    return sentence(rng, rng.randint(40, 160)) + "\n" + SEP


def dialogue(rng):
    import _common as C
    turns = []
    for _ in range(rng.randint(1, 4)):
        turns += [{"role": "user", "content": sentence(rng, rng.randint(4, 14))},
                  {"role": "assistant", "content": sentence(rng, rng.randint(4, 16))}]
    return turns, C.render_dialogue(turns)


def write_until(path, rng, make, size, extra=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for d in extra:
            f.write(d)
        while n < size:
            d = make(rng)
            f.write(d)
            n += len(d)


def build_fixture(fx):
    """The files stage 1's fetches leave, in miniature: four FineWeb-Edu shards (one holding a document of the panel and
    a repeat), Simple English Wikipedia, the human split, the filtered dialogue sources and the two panels."""
    rng = random.Random(20261111)
    val = [article(rng) for _ in range(30)]
    (fx / "fw" / "fineweb_edu_shard0_v1").mkdir(parents=True)
    (fx / "fw" / "fineweb_edu_shard0_v1" / "val.txt").write_text("".join(val), encoding="utf-8")
    twice = article(rng)
    for n in (2, 3, 4, 5):
        write_until(fx / "fw" / f"fineweb_edu_shard{n}_v1" / "pretrain_train.txt", rng, article, 1_150_000,
                    extra=[val[n], twice] if n in (3, 4) else [twice] if n == 5 else [])
    write_until(fx / "fw" / "simplewiki.txt", rng, article, 95_000)
    write_until(fx / "fw" / "sample.txt", rng, article, 300_000)
    held = [dialogue(rng) for _ in range(40)]
    (fx / "human_val.txt").write_text("".join(t for _, t in held[:20]), encoding="utf-8")
    (fx / "human_sft.jsonl").write_text("".join(json.dumps({"messages": m}) + "\n" for m, _ in held[20:]), encoding="utf-8")
    write_until(fx / "human_pt.txt", rng, lambda r: dialogue(r)[1], 60_000)
    for name, size in (("soda", 260_000), ("ultrachat", 360_000), ("everyday", 120_000), ("taskmaster3", 12_000),
                       ("abcd", 8_000), ("ccpe", 4_000), ("p4g", 4_000), ("aya_en", 4_000), ("msc", 10_000)):
        write_until(fx / "filtered" / f"{name}.txt", rng, lambda r: dialogue(r)[1], size)
    for name in ("panel_fineweb.ids", "panel_dialogue.ids"):
        (fx / name).write_bytes(b"PKIDS001" + struct.pack("<IIQ", 2, 16384, 4) + struct.pack("<4H", 1, 2, 3, 4))
    return {"val": val, "twice": twice}


def docs_of(path):
    return [d for d in Path(path).read_text(encoding="utf-8").split(SEP) if d]


def ids_tokens(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"PKIDS001"
    width, _vocab, count = struct.unpack("<IIQ", head[8:])
    assert os.path.getsize(path) == 24 + width * count
    return count


class Mini:
    def __init__(self, tmp):
        self.tmp = tmp
        self.fx = tmp / "fixture"
        self.known = build_fixture(self.fx)
        self.store = tmp / "store"
        self.cfg = tmp / "store.json"
        self.cfg.write_text(json.dumps({"repo": f"dir:{self.store}"}))
        up = tmp / "upload"
        up.mkdir()
        self.vocab = up / "vocab_llama3_bpe.gguf"
        os.symlink(VOCAB, self.vocab)
        # The engine reaches a VM through the store (base2_seed.py's set seed/scenario_engine) and is pulled beside the
        # vocabulary; its files are pinned as they stand, as `pin-engine` pins them at home.
        seed = tmp / "seed_engine"
        seed.mkdir()
        for f in P.engine_files(str(ENGINE)):
            shutil.copyfile(ENGINE / f, seed / f)
        self.hf("push", "--branch", "main", "--prefix", "seed/scenario_engine", "--dir", str(seed))
        self.engine = up / "scenario_engine"
        self.hf("pull", "--branch", "main", "--prefix", "seed/scenario_engine", "--to", str(self.engine))
        pins = json.load(open(CC / "base2_pins.json"))
        pins["files"] = {}
        for name in ("panel_fineweb.ids", "panel_dialogue.ids"):   # the toy panels stand in for base A's
            pins["panels"][name] = P.sha256(str(self.fx / name))
        P.pin_engine(pins, str(self.engine), commit="as it stood for this test")
        self.pins = tmp / "pins.json"
        self.pins.write_text(json.dumps(pins))
        self.logs = []

    def hf(self, *args):
        subprocess.run([sys.executable, str(CC / "hf_store.py"), args[0], "--config", str(self.cfg), *args[1:]],
                       check=True, capture_output=True)

    def commits(self, store=None):
        p = (store or self.store) / "commits.jsonl"
        return [json.loads(ln) for ln in open(p)] if p.exists() else []

    def run(self, out, mode, cfg=None, expect=0, **env):
        e = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BASE2_STORE_CONFIG": str(cfg or self.cfg), "BASE2_PINS": str(self.pins),
             "BASE2_ENGINE_DIR": str(self.engine), "BASE2_FIXTURE": str(self.fx), "BASE2_THREADS": "2", "BASE2_JOBS": "2",
             "BASE2_SCN_SHARD_DOCS": "400", "BASE2_SCN_HELDOUT_DOCS": "200", "BASE2_MIN_FREE_GB": "0",
             "BASE2_TIMING_BYTES": "150000", "BASE2_KEEP_MIX_TEXT": "1", "BASE2_STORE_RETRY_S": "0",
             # The engine's validator enforces rates (a three-word ending under 2% of turns, cue and genre shares)
             # that a file of a few hundred documents cannot meet; it runs at the build's real sizes only.
             "BASE2_SCN_VALIDATE": "0", "PYTHONDONTWRITEBYTECODE": "1", **env}
        r = subprocess.run(["bash", "tools/colab_chat/base2_data.sh", str(out), str(self.vocab), mode], cwd=ROOT, env=e,
                           capture_output=True, text=True)
        self.logs.append(r.stdout + r.stderr)
        assert r.returncode == expect, (r.stdout[-3000:], r.stderr[-3000:])
        return r.stdout + r.stderr

    def fork_store(self, name):
        """A copy of the store as it stands, with its own config: a second VM that never saw the later stages."""
        dst = self.tmp / name
        shutil.copytree(self.store, dst)
        cfg = self.tmp / f"{name}.json"
        cfg.write_text(json.dumps({"repo": f"dir:{dst}"}))
        return cfg


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    if not VOCAB.exists() or not (ENGINE / "generate.py").exists() or not (CC / "hf_store.py").exists():
        pytest.skip("needs the 16k vocabulary, the scenario engine's folder and hf_store.py")
    m = Mini(tmp_path_factory.mktemp("base2_mini"))
    m.a = m.tmp / "vm_a"    # stage 1, then lost
    m.b = m.tmp / "vm_b"    # stages 2 and 3
    m.out1 = m.run(m.a, "stage1")
    m.n1 = len(m.commits())
    m.out1_again = m.run(m.a, "stage1")
    m.after1 = m.fork_store("store_after1")
    m.out2 = m.run(m.b, "stage2")
    m.n2 = len(m.commits())
    m.out2_again = m.run(m.b, "stage2")
    m.after2, m.after2_e, m.after2_f = (m.fork_store(f"store_after2{x}") for x in ("", "_e", "_f"))
    m.out3 = m.run(m.b, "stage3")
    m.n3 = len(m.commits())
    m.out3_again = m.run(m.b, "all")
    return m


def test_stage1_joins_filters_and_pushes(mini):
    led = json.load(open(mini.a / "base2_stage1.json"))
    join = led["fineweb_join"]
    # tc_b1_fineweb.py counts the empty piece after the val file's last separator as a 31st key.
    assert join["dropped_equal_val"] == 2 and join["dropped_repeat"] == 2 and join["val_documents"] == 31
    fw = docs_of(mini.a / "src" / "fineweb2345.txt")
    assert not {d + SEP for d in fw} & set(mini.known["val"])          # no document of the FineWeb panel
    assert fw.count(mini.known["twice"][:-len(SEP)]) == 1
    assert led["simplewiki_filter"]["documents_kept"] == led["simplewiki_filter"]["documents_in"] > 0
    assert (mini.a / "src" / "human1.txt").read_bytes() == b"".join(
        (mini.fx / p).read_bytes() for p in ("human_pt.txt", *[f"filtered/{s}.txt" for s in
                                                             ("taskmaster3", "abcd", "ccpe", "p4g", "aya_en", "msc")]))
    assert led["tokeniser_timing"]["ids"]["tokens"] > 0
    assert led["pins"]["stage1"]["differences"] == [] and led["pins"]["vocab"]["differences"] == []
    sets = [(c["branch"], sorted({a.rsplit("/", 1)[0] for a in c["adds"]})) for c in mini.commits()[1:mini.n1]]
    assert sets == [("base2-build", ["build/src"]), ("main", ["corpus/panels"]), ("base2-build", ["build/stage1"])]
    assert (mini.store / "main" / "corpus" / "panels" / "panel_fineweb.ids").exists()


def test_every_stage_is_idempotent(mini):
    assert "stage 1 done" in mini.out1 and "stage 1: finished already" in mini.out1_again
    assert "stage 2 done" in mini.out2 and "stage 2: finished already" in mini.out2_again
    assert "stage 3 done" in mini.out3
    for n in (1, 2, 3):
        assert f"stage {n}: finished already" in mini.out3_again
    # A finished stage pushes nothing again: the store's commits are those of the three first runs.
    assert len(mini.commits()) == mini.n3
    assert mini.n1 == 4 and mini.n2 - mini.n1 == 2 and mini.n3 - mini.n2 == 3
    # Stage 2 ran on a VM that never saw stage 1: it pulled the sources and stage 1's record from the store.
    assert "pulled base2-build:build/src" in mini.out2 and (mini.b / "base2_stage1.json").exists()
    assert not (mini.b / "fw").exists()


def test_a_lost_stage_rebuilds_the_same_bytes(mini):
    """A second VM that repeats stage 2, and one that repeats stage 3, from what the store held at that point."""
    want = json.load(open(mini.b / "base2_pins_observed.json"))
    c = mini.tmp / "vm_c"
    out = mini.run(c, "stage2", cfg=mini.after1)
    assert "stage 2 done" in out
    got = json.load(open(c / "base2_pins_observed.json"))
    assert got["build-N"]["observed"] == want["build-N"]["observed"]
    d = mini.tmp / "vm_d"
    out = mini.run(d, "stage3", cfg=mini.after2)
    assert "stage 3 done" in out and "pulled base2-build:build/stage2" in out
    got = json.load(open(d / "base2_pins_observed.json"))
    assert got["build-S"]["observed"] == want["build-S"]["observed"]
    assert json.load(open(d / "scenario.ledger.json"))["sha256"] == json.load(open(mini.b / "scenario.ledger.json"))["sha256"]
    # A stage stopped half-way goes on from what it has: with its record and its ids gone, the mix's text is still
    # there, so the rerun tokenises again and reaches the same ids.
    for f in ("base2_stage2.json", "N.main.ids", "N.tail.ids", "ids_N.json"):
        (c / f).unlink()
    shutil.rmtree(mini.tmp / "store_after1" / "base2-build" / "build" / "stage2")
    before = (c / "N.ledger.json").stat().st_mtime_ns
    out = mini.run(c, "stage2", cfg=mini.after1)
    assert "stage 2 done" in out and (c / "N.ledger.json").stat().st_mtime_ns == before
    assert json.load(open(c / "base2_pins_observed.json"))["build-N"]["observed"] == want["build-N"]["observed"]


def regen_scenario(mini):
    """The scenario documents of S's two parts, as one file (the build's own joined stream is scratch and gone)."""
    p = mini.tmp / "s_scenario.txt"
    if not p.exists():
        natural = set()
        for f in ("fineweb2345.txt", "simplewiki.txt", "human1.txt", "soda.txt", "ultrachat.txt", "everyday.txt"):
            natural |= set(docs_of(mini.b / "src" / f))
        docs = [d for part in ("main", "tail") for d in docs_of(mini.b / f"S.{part}.txt") if d not in natural]
        p.write_text("".join(d + SEP for d in docs), encoding="utf-8")
    return p


def test_scenario_share_is_fifteen_percent_of_tokens_by_the_real_tokeniser(mini):
    import corpus_ids
    _, enc = corpus_ids.encoder(str(mini.vocab))
    natural = set()
    for p in ("fineweb2345.txt", "simplewiki.txt", "human1.txt", "soda.txt", "ultrachat.txt", "everyday.txt"):
        natural |= set(docs_of(mini.b / "src" / p))
    rec = json.load(open(mini.b / "base2_stage3.json"))
    for part in ("main", "tail"):
        docs = docs_of(mini.b / f"S.{part}.txt")
        scn = [d for d in docs if d not in natural]
        nat = [d for d in docs if d in natural]
        assert scn and nat
        t_s = sum(len(x) for x in enc([d + SEP for d in scn]))
        t_n = sum(len(x) for x in enc([d + SEP for d in nat]))
        share = t_s / (t_s + t_n)
        assert 0.14 <= share <= 0.16, (part, share)
        assert abs(share - 0.15) < 0.005, (part, share)       # the measured ratio lands well inside the bounds
        # The build's own figure: the same scenario tokens over the ids file's count. Counted a document at a time, the
        # total is within a token a document of the file's.
        ids = ids_tokens(mini.b / f"S.{part}.ids")
        assert abs(ids - (t_s + t_n)) <= len(docs)
        mine = rec[f"share_{part}"]
        assert mine["ok"] and mine["scenario_documents"] == len(scn) and mine["scenario_tokens"] == t_s
        assert mine["tokens"] == ids and mine["tokens_from"] == "ids"
        assert abs(mine["scenario_share_of_tokens"] - share) < 0.001
        whole = B.share(str(mini.b / f"S.{part}.txt"), str(regen_scenario(mini)), str(mini.vocab))
        assert whole["tokens"] == t_s + t_n and whole["tokens_from"] == "documents" and whole["scenario_tokens"] == t_s
        # S keeps N's token count in each part, to the size of a few documents.
        assert abs(ids - ids_tokens(mini.b / f"N.{part}.ids")) / ids < 0.02, part
    r = rec["recipe"]
    # The toy natural text tokenises more finely than scenario text, so the byte ratio is well above 15%.
    assert r["scenario_bytes_per_token"] > r["parts"]["main"]["n_bytes_per_token"] and r["main"]["scenario"] > 0.16
    assert rec["scenario_sample"]["tokens"] > 0 and rec["scenario"]["ok"] and rec["scenario"]["dropped_share"] <= 0.02


def test_s_takes_natural_documents_that_n_took_and_no_panel_document(mini):
    n = collections.Counter(docs_of(mini.b / "N.main.txt") + docs_of(mini.b / "N.tail.txt"))
    s_docs = docs_of(mini.b / "S.main.txt") + docs_of(mini.b / "S.tail.txt")
    natural = set(n)
    s_nat = collections.Counter(d for d in s_docs if d in natural)
    s_scn = [d for d in s_docs if d not in natural]
    assert s_nat and not s_nat - n                       # with multiplicity: a copy in S is a copy in N
    assert 0.80 < sum(len(d) * c for d, c in s_nat.items()) / sum(len(d) * c for d, c in n.items()) < 0.88
    assert json.load(open(mini.b / "subset_S_in_N.json")) == {"subset": True, "differences": []}
    val = {d[:-len(SEP)] for d in mini.known["val"]}
    assert not val & set(n) and not val & set(s_docs)    # the FineWeb panel's documents are in no mix
    # The scenario documents are the pinned engine's training stream; none is a document of the held-out panel.
    assert len(set(s_scn)) == len(s_scn)
    regen = mini.tmp / "regen"
    regen.mkdir()
    subprocess.run([sys.executable, str(mini.engine / "generate.py"), "--seed", "20261111", "--docs", "200", "--heldout",
                    "--out", str(regen / "held.txt"), "--meta", os.devnull], check=True, capture_output=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    held = set(docs_of(regen / "held.txt"))
    assert len(held) == 200 and not held & set(s_scn)
    assert ids_tokens(mini.b / "panel_scenario.ids") == json.load(open(mini.b / "panel_scenario.json"))["tokens"] > 0
    # Each document is a whole one of the stream: shard 0 regenerated holds the ones S took from it.
    subprocess.run([sys.executable, str(mini.engine / "generate.py"), "--seed", "20261111", "--docs", "400", "--start", "0",
                    "--out", str(regen / "s0.txt"), "--meta", os.devnull], check=True, capture_output=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert len(set(docs_of(regen / "s0.txt")) & set(s_scn)) > 50


def test_dialogue_copies_and_the_stores_final_sets(mini):
    led = json.load(open(mini.b / "N.ledger.json"))
    one = {s: len(docs_of(mini.b / "src" / f"{f}.txt")) for s, f in (("human", "human1"), ("soda", "soda"),
                                                                        ("ultrachat", "ultrachat"), ("everyday", "everyday"))}
    assert led["sources"]["human"]["documents"] == 3 * one["human"]
    for s in ("soda", "ultrachat", "everyday"):
        assert led["sources"][s]["documents"] == 2 * one[s]
        assert led["sources"][s]["documents_used"] > 0.95 * led["sources"][s]["documents"]   # 99% of the bytes
    assert led["seed"] == 1 and led["parts"]["tail"]["sources"]["simplewiki"]["documents"] == 0
    main = mini.store / "main" / "corpus"
    for arm in ("N", "S"):
        for part in ("main", "tail"):
            assert (main / arm / f"{arm}.{part}.ids").read_bytes() == (mini.b / f"{arm}.{part}.ids").read_bytes()
    assert (main / "panel_scenario" / "panel_scenario.ids").exists() and (main / "panels" / "panel_dialogue.ids").exists()
    assert not (mini.b / "work").exists() and not (mini.b / "scn").exists()   # stage 3 leaves no scratch text


def test_a_stream_the_filter_cuts_too_far_is_refused(mini):
    e = mini.tmp / "vm_e"
    out = mini.run(e, "stage3", cfg=mini.after2_e, expect=1, BASE2_SCN_MAX_DROP="-0.5")
    assert (e / "scenario.ledger.json.refused").exists() and not (e / "S.main.ids").exists()
    assert "stage 3 done" not in out


def test_an_unpinned_engine_generates_nothing(mini):
    f = mini.tmp / "vm_f"
    pins = json.load(open(mini.pins))
    pins["engine"]["files"]["generate.py"] = "0" * 64
    bad = mini.tmp / "pins_bad.json"
    bad.write_text(json.dumps(pins))
    out = mini.run(f, "stage3", cfg=mini.after2_f, expect=1, BASE2_PINS=str(bad))
    assert "not the pinned bytes" in out and not (f / "scn" / "panel_scenario.txt").exists()


def test_rehearsal_ids_are_built_once_and_then_pulled(mini):
    g, h = mini.tmp / "vm_g", mini.tmp / "vm_h"
    cfg = mini.fork_store("store_reh")
    out = mini.run(g, "rehearsal", cfg=cfg)
    assert "pushed main:corpus/rehearsal" in out and ids_tokens(g / "rehearsal.ids") > 10000
    out = mini.run(h, "rehearsal", cfg=cfg)
    assert "pulled main:corpus/rehearsal" in out and "pushed" not in out
    assert (h / "rehearsal.ids").read_bytes() == (g / "rehearsal.ids").read_bytes()
