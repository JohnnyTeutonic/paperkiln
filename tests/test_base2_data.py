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


def test_data_script_says_what_decontamination_covers_and_orders_publication():
    text = (CC / "base2_data.sh").read_text()
    head = text[:text.index("set -euo pipefail")]
    assert "What the decontamination covers, source by source, and what it does not:" in head
    assert "FineWeb-Edu: exact copies of the FineWeb panel's documents are dropped, and repeats inside the shards. Nothing else:" in head
    assert "FineWeb-Edu is NOT put through the eval rules, Carry-5, Carry-12 or the held-out 13-gram check" in head
    assert "nothing looser than" in head and "equality is checked against that panel" in head
    # FineWeb-Edu's only filter in the script is the join that base A's data job used.
    assert text.count("tc_b1_fineweb.py") == 1 and "docfilter" not in text[text.index("tc_b1_fineweb.py") - 400:text.index("tc_b1_fineweb.py") + 200]
    # Built is not published: in every stage the receipt is the last push, after the output sets, and a mix's text
    # and the scratch go only after it.
    for n, sets, last in ((1, ["build/src", "corpus/panels"], None), (2, ["corpus/N"], "drop_mix_text N"),
                          (3, ["corpus/panel_scenario", "corpus/S"], "drop_mix_text S")):
        body = text[text.index(f"\npublish{n}() {{"):]
        body = body[:body.index("\n}\n")]
        receipt = body.index(f'push_set "$BB" build/stage{n} ')
        for s in sets:
            assert 0 < body.index(f" {s} ") < receipt, (n, s)
        assert body.count("push_set ") == len(sets) + 1
        if last:
            assert body.index(last) > receipt
        build = text[text.index(f"\nbuild{n}() {{"):text.index(f"\npublish{n}() {{")]
        assert "push_set" not in build and "drop_mix_text" not in build and 'rm -rf "$OUT/scn"' not in build
        assert f'"$OUT/base2_stage{n}.json"' not in build        # the receipt's file is written by the publication alone
    run = text[text.index("\nrun_stage() {"):]
    assert run.index('in_store "$BB" "build/stage$n"') < run.index('"build$n"') < run.index('"publish$n"')
    assert '[ -s "$OUT/base2_stage$1.json" ] && return 0' not in text   # a local file never marks a stage finished
    # A push is tried again, and counts only when the store's manifest is read back against the local bytes.
    push = text[text.index("\npush_set() {"):text.index("\npull_set() {")]
    assert "for try in 1 2 3 4; do" in push and push.count("B2 store-verify") == 2 and "exit 1" in push


def test_one_version_of_the_stores_client_library():
    text = (CC / "base2_data.sh").read_text()
    assert text.count("HF_HUB_VERSION=${BASE2_HF_HUB_VERSION:-0.35.3}") == 1
    assert 'pip -q install "huggingface_hub==$HF_HUB_VERSION"' in text
    assert text.count("pip ") == 1 and "export HF_HUB_DISABLE_IMPLICIT_TOKEN=1" in text
    # The same version as the boot's, whenever the boot names one.
    import re
    for other in ("boot_base2.sh", "hf_store.py", "ckpt_courier.py"):
        p = CC / other
        if p.exists():
            for v in re.findall(r"huggingface_hub==([0-9][0-9.]*)", p.read_text()):
                assert v == "0.35.3", (other, v)
            for v in re.findall(r"^HUB(?:_VERSION)? ?= ?\"?([0-9][0-9.]*)", p.read_text(), flags=re.M):
                assert v == "0.35.3", (other, v)    # the store agent's own name for it, in the boot and in hf_store.py


# ---------------------------------------------------------------- cached files and their inputs
def test_a_cached_file_is_reused_only_with_the_inputs_it_was_built_from(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE2_HASH_CACHE", str(tmp_path / "cache.json"))
    a, b, out = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "recipe.json"
    a.write_text("one")
    b.write_text("two")
    out.write_text("{}")
    assert not B.bound(str(out), [str(a), str(b)], ["k=1"]) and not out.exists()      # no record: never trusted
    out.write_text("{}")
    B.bind(str(out), [str(a), str(b)], ["k=1"])
    assert B.bound(str(out), [str(a), str(b)], ["k=1"]) and out.exists()
    assert B.main(["bound", str(out), "--input", str(a), "--input", str(b), "--key", "k=1"]) == 0
    assert not B.bound(str(out), [str(a), str(b)], ["k=2"]) and not out.exists()      # another setting
    out.write_text("{}")
    B.bind(str(out), [str(a), str(b)], ["k=1"])
    b.write_text("TWO")                                                               # an input's bytes change
    assert not B.bound(str(out), [str(a), str(b)], ["k=1"])
    assert not out.exists() and not Path(str(out) + ".inputs.json").exists()          # the stale file is gone
    out.write_text("{}")
    B.bind(str(out), [str(a), str(b)], ["k=1"])
    assert not B.bound(str(out), [str(a)], ["k=1"])                                   # another set of inputs
    out.write_text("{}")
    B.bind(str(out), [str(a), str(b)], ["k=1"])
    b.unlink()
    assert not B.bound(str(out), [str(a), str(b)], ["k=1"])                           # an input is gone
    out.write_text("{}")
    with pytest.raises(SystemExit):
        B.bind(str(out), [str(a), str(b)], [])
    # The remembered hash follows the file's size and time: a rewrite is read again.
    a.write_text("one")
    h = B.file_sha(str(a))
    a.write_text("eno")
    os.utime(a, ns=(1, 1))
    assert B.file_sha(str(a)) != h == hashlib.sha256(b"one").hexdigest()


def test_every_cached_file_of_the_script_is_bound():
    """No `[ -s FILE ] ||` in front of a recipe, a ledger, an ids record or a share: each goes through cached/stamp."""
    text = (CC / "base2_data.sh").read_text()
    for f in ('"$OUT/recipe_N.json"', '"$OUT/recipe_S.json"', '"$OUT/$arm.ledger.json"', '"$OUT/ids_$arm.json"',
              '"$OUT/share_S.$part.json"', '"$OUT/scenario_sample.json"', '"$OUT/panel_scenario.json"', '"$led"',
              '"$WORK/scenario.txt"', '"$SRC/fineweb2345.txt"', '"$SRC/simplewiki.txt"', '"$SRC/human1.txt"', '"$OUT/N.keys"',
              '"$OUT/subset_docs_S_in_N.json"'):
        assert f"cached {f} " in text and f"stamp {f} " in text, f
        assert f"[ -s {f} ] ||" not in text, f


def test_subset_by_document(tmp_path):
    n1, n2, scn, s = tmp_path / "n1.txt", tmp_path / "n2.txt", tmp_path / "scn.txt", tmp_path / "s.txt"
    n1.write_text(f"a\n{SEP}b\n{SEP}a\n{SEP}")
    n2.write_text(f"c\n{SEP}")
    scn.write_text(f"x\n{SEP}y\n{SEP}")
    assert B.doc_keys([str(n1), str(n2)], str(tmp_path / "n.keys")) == {"documents": 4, "distinct": 3}
    s.write_text(f"a\n{SEP}x\n{SEP}c\n{SEP}a\n{SEP}")
    r = B.subset_docs(str(tmp_path / "n.keys"), str(scn), [str(s)])
    assert r == {"natural_documents_in_S": 3, "scenario_documents_in_S": 1, "documents_in_N": 4,
                 "documents_of_S_not_in_N": 0, "subset": True}
    s.write_text(f"a\n{SEP}a\n{SEP}a\n{SEP}")                                # a third copy N does not hold
    assert not B.subset_docs(str(tmp_path / "n.keys"), str(scn), [str(s)])["subset"]
    s.write_text(f"b\n{SEP}z\n{SEP}")                                        # a document of neither
    r = B.subset_docs(str(tmp_path / "n.keys"), str(scn), [str(s)])
    assert not r["subset"] and r["documents_of_S_not_in_N"] == 1
    assert B.main(["subset-docs", "--n-keys", str(tmp_path / "n.keys"), "--scenario", str(scn), "--out",
                   str(tmp_path / "r.json"), str(s)]) == 1


def test_token_report_and_split(tmp_path):
    n, s = tmp_path / "ids_N.json", tmp_path / "ids_S.json"
    n.write_text('{"out": "/d/N.main.ids", "tokens": 1000}\n{"out": "/d/N.tail.ids", "tokens": 200}\n')
    s.write_text('{"out": "/d/S.main.ids", "tokens": 1004}\n{"out": "/d/S.tail.ids", "tokens": 199}\n')
    r = B.token_report(str(n), str(s), 0.01)
    assert r["N"] == {"main": 1000, "tail": 200} and r["S"] == {"main": 1004, "tail": 199} and r["ok"]
    assert r["N_total"] == 1200 and r["S_total"] == 1203 and abs(r["relative_difference"]["main"] - 0.004) < 1e-12
    assert not B.token_report(str(n), str(s), 0.003)["ok"]
    src = tmp_path / "a.txt"
    src.write_text("".join(f"document {i}\n{SEP}" for i in range(40)), encoding="utf-8")
    na, nb = B.split_docs(str(src), str(tmp_path / "m.txt"), str(tmp_path / "t.txt"), 700)
    m, t = (tmp_path / "m.txt").read_text(), (tmp_path / "t.txt").read_text()
    assert m + t == src.read_text() and na == len(m) <= 700 < na + 30 and nb == len(t) and m.endswith(SEP) and na != nb
    with pytest.raises(SystemExit):
        B.split_docs(str(src), str(tmp_path / "m2.txt"), str(tmp_path / "t2.txt"), 10 ** 9)


# ---------------------------------------------------------------- pins
def sha(b):
    return hashlib.sha256(b).hexdigest()


def toy_pins():
    return {"files": {"f.txt": sha(b"f")}, "sources": {"src/a.txt": None}, "engine": {"commit": None, "files": {}},
            "N": {"N.main.ids": None, "N.tail.ids": None}, "S": {"S.main.ids": None, "S.tail.ids": None},
            "panels": {"panel_fineweb.ids": sha(b"pf"), "panel_dialogue.ids": sha(b"pd"), "panel_scenario.ids": None},
            "vocab": {"vocab_llama3_bpe.gguf": sha(b"v")}, "a_weights": {"tc_b1_r1.safetensors": sha(b"w")},
            "rehearsal": {"rehearsal.main.ids": None, "rehearsal.tail.ids": None}}


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
    both = json.load(open(d / "base2_pins_observed.json"))
    assert set(both) == {"build-N", "arm-N"} and both["arm-N"]["differences"]
    # A record that carries differences comes from a refused run: nothing of it is adopted, not even its clean modes.
    before = pf.read_text()
    with pytest.raises(SystemExit) as e:
        P.main(["adopt", str(d / "base2_pins_observed.json"), "--pins", str(pf)])
    assert "carries differences" in str(e.value) and pf.read_text() == before
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"build-N": both["build-N"]}))
    assert P.main(["adopt", str(clean), "--pins", str(pf)]) == 0
    assert json.load(open(pf))["N"]["N.main.ids"] == sha(b"1")
    (d / "N.tail.ids").unlink()                                   # a build with a missing file is refused too
    with pytest.raises(SystemExit):
        P.main([str(d), "build-N", "--pins", str(pf)])
    with pytest.raises(SystemExit):
        P.main(["adopt", str(d / "base2_pins_observed.json"), "--pins", str(pf)])


def test_adopt_receipts_checks_the_load_probe_and_writes_the_shard_hashes_where_they_are_committed(tmp_path, capsys):
    """The launch reviews of 12 Nov 2026: the adoption was a by-hand extraction of each receipt's built.pins, the
    load probe's hashes were never compared with what stage 3 published (the S pins are null while it runs), and
    the FineWeb-Edu shard 2 to 5 hash files the plan commits (work item 6: corpus/hashes/, beside shards 0 and 1)
    went nowhere. adopt-receipts does all of it in one step, refuses before writing anything, and prints the
    adoption commit's files."""
    vm = tmp_path / "vm"
    (vm / "src").mkdir(parents=True)
    for name, body in (("N.main.ids", b"nm"), ("N.tail.ids", b"nt"), ("S.main.ids", b"sm"), ("S.tail.ids", b"st"),
                       ("panel_fineweb.ids", b"pf"), ("panel_dialogue.ids", b"pd"), ("panel_scenario.ids", b"ps"),
                       ("src/a.txt", b"a")):
        (vm / name).write_bytes(body)
    repo = tmp_path / "repo"
    (repo / "corpus" / "hashes").mkdir(parents=True)
    (repo / "tools" / "colab_chat").mkdir(parents=True)
    pf = repo / "tools" / "colab_chat" / "base2_pins.json"
    pf.write_text(json.dumps(toy_pins()))
    (repo / "corpus" / "hashes" / "fineweb_edu_shard0_v1.sha256").write_text("committed\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "start"], check=True)
    obs = {m: {"observed": P.check(toy_pins(), str(vm), m)[1], "differences": []} for m in ("stage1", "build-N", "build-S")}
    build = tmp_path / "build"
    (build / "data").mkdir(parents=True)
    (build / "load_probe").mkdir()

    def published(*names):                              # store-verify's record of a published set
        return {"verified": True, "differences": [], "files": {f: {"bytes": 2, "sha256": sha((vm / f).read_bytes())} for f in names}}

    pubs = {1: {"published_panels": published("panel_fineweb.ids", "panel_dialogue.ids")},
            2: {"published_N": published("N.main.ids", "N.tail.ids")},
            3: {"published_S": published("S.main.ids", "S.tail.ids"), "published_panel_scenario": published("panel_scenario.ids")}}

    def receipt(n, modes, **more):
        (build / "data" / f"base2_stage{n}.json").write_text(json.dumps(
            {"stage": str(n), "built": {"stage": str(n), "pins": {m: obs[m] for m in modes}}, **pubs[n], **more}))

    # The observed record grows on one VM: stage 3's receipt carries every mode run before it.
    for n, modes in ((1, ["stage1"]), (2, ["stage1", "build-N"]), (3, ["stage1", "build-N", "build-S"])):
        receipt(n, modes)
    for n in (2, 3, 4, 5):
        (build / "data" / f"fineweb_edu_shard{n}_v1.sha256").write_text(f"{sha(bytes([n]))}  pretrain_train.txt\n")
    probe = build / "load_probe" / "loadprobe_pins.json"
    (build / "load_probe" / "loadprobe.json").write_text(json.dumps({"returncode": 0, "seconds_to_first_step": 41.5}))
    state_dir = tmp_path / "build_runner"
    state_dir.mkdir()

    def adopt():
        return P.main(["adopt-receipts", str(build), "--pins", str(pf), "--repo", str(repo), "--state-dir", str(state_dir)])

    before = pf.read_text()
    # (1) The probe pulled S's tail with other bytes than stage 3 hashed before it published: refused, nothing written.
    other = json.loads(json.dumps(obs["build-S"]))
    other["observed"]["S/S.tail.ids"] = sha(b"other")
    probe.write_text(json.dumps({"build-S": other}))
    with pytest.raises(SystemExit) as e:
        adopt()
    assert "S/S.tail.ids: stage 3 hashed" in str(e.value) and "the load probe pulled" in str(e.value)
    assert pf.read_text() == before and [p.name for p in (repo / "corpus" / "hashes").iterdir()] == ["fineweb_edu_shard0_v1.sha256"]
    # (2) A probe with a difference of its own, and a stage receipt with no pins (N would stay null): refused.
    probe.write_text(json.dumps({"build-S": {**obs["build-S"], "differences": ["panels/panel_fineweb.ids: x against y"]}}))
    stage2 = (build / "data" / "base2_stage2.json").read_text()
    (build / "data" / "base2_stage2.json").write_text(json.dumps({"stage": "2", "built": {"stage": "2"}}))
    (build / "data" / "base2_stage3.json").write_text(json.dumps(
        {"stage": "3", "built": {"pins": {m: obs[m] for m in ("stage1", "build-S")}}}))
    with pytest.raises(SystemExit) as e:
        adopt()
    said = str(e.value)
    assert "load probe: panels/panel_fineweb.ids" in said and "stage 2: its receipt carries no built.pins" in said
    assert "N/N.main.ids: still no pin after the adoption, and an arm reads it" in said and pf.read_text() == before
    (build / "data" / "base2_stage2.json").write_text(stage2)
    receipt(3, ["stage1", "build-N", "build-S"])
    # The second launch review: (a) the load probe's 20 steps did not complete; (b) the build's runner halted; (c) a
    # published set the store's manifest lists at other bytes than the build hashed (N, from stage 2's receipt).
    probe.write_text(json.dumps({"build-S": obs["build-S"]}))
    lp = build / "load_probe" / "loadprobe.json"
    lp.write_text(json.dumps({"returncode": 1, "seconds_to_first_step": None}))
    (state_dir / "HALTED").write_text("{}")
    pubs[2]["published_N"]["files"]["N.tail.ids"]["sha256"] = sha(b"published other bytes")
    receipt(2, ["stage1", "build-N"])
    with pytest.raises(SystemExit) as e:
        adopt()
    said = str(e.value)
    assert "load probe: its 20-step run did not complete (return code 1, first step None)" in said
    assert "the build's state folder holds HALTED" in said
    assert f"stage 2: published_N lists N.tail.ids at {sha(b'published other bytes')}, and the build hashed {sha(b'nt')}" in said
    assert pf.read_text() == before
    lp.write_text(json.dumps({"returncode": 0, "seconds_to_first_step": 41.5}))
    (state_dir / "HALTED").unlink()
    pubs[2]["published_N"]["files"]["N.tail.ids"]["sha256"] = sha(b"nt")
    receipt(2, ["stage1", "build-N"])
    # (3) A shard hash file already committed with other contents: refused.
    probe.write_text(json.dumps({"build-S": obs["build-S"]}))
    (repo / "corpus" / "hashes" / "fineweb_edu_shard3_v1.sha256").write_text("something else\n")
    with pytest.raises(SystemExit) as e:
        adopt()
    assert "corpus/hashes/fineweb_edu_shard3_v1.sha256: committed, and the build's first fetch wrote another" in str(e.value)
    # (4) The build as it should be: every pin an arm reads is set, the shard hashes are where the plan commits them
    # (one already there with the same bytes is left as it is), and the adoption commit's files are printed as git
    # sees them: the new, untracked hash files too (`git diff` would not show them), shard 3's among them.
    (repo / "corpus" / "hashes" / "fineweb_edu_shard3_v1.sha256").write_text(
        (build / "data" / "fineweb_edu_shard3_v1.sha256").read_text())
    capsys.readouterr()
    assert adopt() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["commit_files"] == ["corpus/hashes/fineweb_edu_shard2_v1.sha256", "corpus/hashes/fineweb_edu_shard3_v1.sha256",
                                   "corpus/hashes/fineweb_edu_shard4_v1.sha256", "corpus/hashes/fineweb_edu_shard5_v1.sha256",
                                   "tools/colab_chat/base2_pins.json"]
    assert out["written"] == ["corpus/hashes/fineweb_edu_shard2_v1.sha256", "corpus/hashes/fineweb_edu_shard4_v1.sha256",
                              "corpus/hashes/fineweb_edu_shard5_v1.sha256"]
    assert sorted(out["adopted"]) == ["N/N.main.ids", "N/N.tail.ids", "S/S.main.ids", "S/S.tail.ids",
                                      "panels/panel_scenario.ids", "sources/src/a.txt"]
    pins = json.loads(pf.read_text())
    up = tmp_path / "up"
    up.mkdir()
    (up / "vocab_llama3_bpe.gguf").write_bytes(b"v")
    (up / "tc_b1_r1.safetensors").write_bytes(b"w")
    for mode in ("arm-N", "arm-S", "arm-C"):
        assert P.check(pins, str(vm), mode, str(up))[0] == [], mode
    for n in (2, 4, 5):
        assert (repo / "corpus" / "hashes" / f"fineweb_edu_shard{n}_v1.sha256").read_text() == f"{sha(bytes([n]))}  pretrain_train.txt\n"
    # The plan's place for them: beside the committed shards 0 and 1, which fineweb_edu.py --pin writes.
    assert (ROOT / "corpus" / "hashes" / "fineweb_edu_shard0_v1.sha256").exists() and P.FW_HASHES[0] == "fineweb_edu_shard2_v1.sha256"
    # The load probe's job sends its hashes home, so this check has them.
    assert "loadprobe_pins.json" in next(j for j in json.load(open(CC / "base2_data.json"))["jobs"] if j["name"] == "load_probe")["outputs"]
    assert 'PINS "$OUT" build-S --out "${1:?probe out dir}/loadprobe_pins.json"' in (CC / "base2_data.sh").read_text()


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
    # Read back after a push: the store's manifest against the local bytes.
    assert B.store_verify(str(cfg), "base2-build", "build/stage1", [str(d / "a.txt")])[0] == []
    (d / "a.txt").write_text("b")
    assert B.store_verify(str(cfg), "base2-build", "build/stage1", [str(d / "a.txt")])[0]
    (d / "c.txt").write_text("c")
    assert "not in the store's manifest" in B.store_verify(str(cfg), "base2-build", "build/stage1", [str(d / "c.txt")])[0][0]
    assert B.main(["store-verify", str(cfg), "base2-build", "build/absent", str(d / "c.txt")]) == 2


def test_only_a_missing_file_or_branch_reads_as_absent():
    """A wrong or expired token gets "repository not found"; an outage gets a cache miss. Neither is an unfinished
    stage. The classes stand in for huggingface_hub's by name, as hf_store.py matches them."""
    class StoreError(Exception):
        pass

    class Missing(StoreError):          # hf_store.Missing: what both backends raise for a path the store lacks
        pass

    def make(name, *bases):
        return type(name, bases or (Exception,), {})
    http = make("HfHubHTTPError")
    entry, rev = make("EntryNotFoundError", http), make("RevisionNotFoundError", http)
    remote = make("RemoteEntryNotFoundError", entry)
    local = make("LocalEntryNotFoundError", entry, FileNotFoundError)
    repo, gated = make("RepositoryNotFoundError", http), None
    gated = make("GatedRepoError", repo)
    for e in (entry("x"), rev("x"), remote("x"), Missing("main:a/manifest.json is not in the store")):
        assert B.reads_as_absent(e, Missing)
    for e in (repo("401"), gated("x"), local("offline"), http("500"), ConnectionError("x"), TimeoutError("x"),
              StoreError("token file not found: /x"), StoreError("main:a/manifest.json is not in the store")):
        assert not B.reads_as_absent(e, Missing), e      # the last: the class says "absent", never an error's wording
    # Wrapped by the store's own error, the cause decides.
    try:
        try:
            raise repo("401")
        except Exception as inner:
            raise StoreError("fetch failed") from inner
    except StoreError as wrapped:
        assert not B.reads_as_absent(wrapped, Missing)
    try:
        try:
            raise entry("404")
        except Exception as inner:
            raise StoreError("fetch failed") from inner
    except StoreError as wrapped:
        assert B.reads_as_absent(wrapped, Missing)
    # The real library's classes, when it is installed: the two that mean absent and the ones that must not.
    errors = pytest.importorskip("huggingface_hub.errors")
    import hf_store
    assert B.reads_as_absent(errors.EntryNotFoundError("404"), hf_store.Missing)
    assert B.reads_as_absent(errors.RevisionNotFoundError("404"), hf_store.Missing)
    for cls in (errors.RepositoryNotFoundError, errors.GatedRepoError, errors.LocalEntryNotFoundError, errors.HfHubHTTPError):
        assert not B.reads_as_absent(cls("x"), hf_store.Missing), cls


# ---------------------------------------------------------------- the real HfBackend over a stand-in for the hub
HUB_SITE = ROOT / "tests" / "hub_standin_site"
HUB_TOKEN = "hf_FAKEtokenFAKEtokenFAKEtoken0123"


def hub_config(tmp, name="hub"):
    """A store config that names a repo on the hub (not dir:), a token file, and the folder the stand-in hub keeps."""
    root = tmp / name
    root.mkdir()
    tok = tmp / f"{name}.token"
    tok.write_text(HUB_TOKEN + "\n")
    tok.chmod(0o600)
    cfg = tmp / f"{name}.json"
    cfg.write_text(json.dumps({"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tok)}))
    return cfg, root


def hub_env(root):
    """The environment under which every python child reaches the stand-in through the real HfBackend."""
    return {"HUB_STANDIN_ROOT": str(root), "PYTHONPATH": f"{HUB_SITE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)}


def test_an_absent_path_or_branch_on_the_real_backend_reads_as_absent_and_nothing_else_does(tmp_path, monkeypatch):
    """store-has through hf_store.HfBackend, the class the VM uses, over an API that raises the library's own
    errors. The hub's "no such file" and "no such branch" are answers: False, exit 1, the stage is built. A repo
    that is not found (a wrong or expired token), a refused login and a dead connection are not answers: they are
    tried again and then raised, exit 2, and the build stops. (The review of 12 Nov 2026: HfBackend.fetch turned
    the hub's two errors into hf_store.Missing and reads_as_absent knew that class on a local store only, so the
    first `in_store main corpus/rehearsal` on the real store stopped the build.)"""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    sys.path.insert(0, str(ROOT / "tests"))
    import hf_store
    import hub_standin
    cfg, root = hub_config(tmp_path)
    monkeypatch.setenv("HUB_STANDIN_ROOT", str(root))
    monkeypatch.setenv("BASE2_STORE_RETRY_S", "0")
    monkeypatch.setattr(huggingface_hub, "HfApi", hub_standin.DiskApi)
    assert isinstance(hf_store.open_store(str(cfg)).b, hf_store.HfBackend)
    assert B.store_has(str(cfg), "main", "corpus/rehearsal") is False          # EntryNotFoundError at the hub
    assert B.store_has(str(cfg), "base2-build", "build/stage1") is False       # RevisionNotFoundError: no such branch
    log = (root / "calls.log").read_text()
    assert "raise EntryNotFoundError corpus/rehearsal/manifest.json" in log and "raise RevisionNotFoundError base2-build" in log
    assert B.main(["store-has", str(cfg), "main", "corpus/rehearsal"]) == 1
    d = tmp_path / "set"
    d.mkdir()
    (d / "a.txt").write_text("a")
    assert hf_store.main(["push", "--config", str(cfg), "--branch", "main", "--prefix", "corpus/rehearsal", "--dir", str(d), "--limit", "60"]) == 0
    assert B.store_has(str(cfg), "main", "corpus/rehearsal") is True
    assert B.store_verify(str(cfg), "main", "corpus/rehearsal", [str(d / "a.txt")])[0] == []
    with monkeypatch.context() as quick:
        quick.setattr(hf_store.time, "sleep", lambda s: None)        # the waits between tries
        for mode in ("repo_not_found", "auth", "transport"):
            (root / "fail.json").write_text(json.dumps({"mode": mode}))
            (root / "calls.log").write_text("")
            with pytest.raises(Exception) as err:
                B.store_has(str(cfg), "main", "corpus/absent")
            assert not isinstance(err.value, hf_store.Missing), mode
            assert (root / "calls.log").read_text().count("hf_hub_download") >= 3, mode   # tried again, then it stands
            assert B.main(["store-has", str(cfg), "main", "corpus/absent"]) == 2, mode
            assert B.main(["store-verify", str(cfg), "main", "corpus/rehearsal", str(d / "a.txt")]) == 2, mode
    # A request that never returns is ended at the limit: a failed try, exit 2, not a job that waits without end.
    (root / "fail.json").write_text(json.dumps({"mode": "hang"}))
    monkeypatch.setenv("BASE2_STORE_LIMIT_S", "1")
    import time
    t0 = time.time()
    assert B.main(["store-has", str(cfg), "main", "corpus/rehearsal"]) == 2
    assert B.main(["store-verify", str(cfg), "main", "corpus/rehearsal", str(d / "a.txt")]) == 2
    assert time.time() - t0 < 60


def test_an_unreadable_store_stops_the_build_and_is_not_an_unfinished_stage(tmp_path, monkeypatch):
    if not (CC / "hf_store.py").exists():
        pytest.skip("hf_store.py is the store agent's file")
    import hf_store
    monkeypatch.setenv("BASE2_STORE_RETRY_S", "0")
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'store'}"}))
    RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    calls = tmp_path / "calls"

    def refuse(self, branch, path, dest, rev=None):
        with open(calls, "a") as f:                                   # the call runs in a bounded child: count on disk
            f.write(path + "\n")
        raise RepositoryNotFoundError("401 Client Error")
    monkeypatch.setattr(hf_store.DirBackend, "fetch", refuse)
    monkeypatch.setattr(hf_store.time, "sleep", lambda s: None)
    with pytest.raises(hf_store.StoreError, match="RepositoryNotFoundError"):
        B.store_has(str(cfg), "base2-build", "build/stage1")
    assert len(calls.read_text().splitlines()) >= 3                   # tried again, then it stands
    assert B.main(["store-has", str(cfg), "base2-build", "build/stage1"]) == 2
    text = (CC / "base2_data.sh").read_text()
    fn = text[text.index("\nin_store() {"):text.index("\nsets_of() {")]
    assert '[ "$rc" = 0 ] && return 0' in fn and '[ "$rc" = 1 ] && return 1' in fn and "exit 1" in fn


def test_every_store_call_of_the_data_script_has_a_limit():
    """The shell passes --limit to every hf_store.py call, and store-has and store-verify bound their own reads."""
    text = (CC / "base2_data.sh").read_text()
    calls = [ln for ln in text.splitlines() if "python3 $CC/hf_store.py" in ln]
    assert len(calls) == text.count("$CC/hf_store.py") == 2 and all('--limit "$XFER_LIMIT_S"' in ln for ln in calls), calls
    assert "XFER_LIMIT_S=${BASE2_STORE_XFER_LIMIT_S:-5400}" in text and "export BASE2_STORE_LIMIT_S=${BASE2_STORE_LIMIT_S:-300}" in text
    src = (CC / "base2_corpus.py").read_text()
    for fn in ("def store_has(", "def store_verify("):
        body = src[src.index(fn):src.index("\ndef ", src.index(fn) + 1)]
        assert "hf_store.bounded(" in body, fn
    # The watcher makes no store call at all (it used the repository's hourly commit limit by itself, 12 Nov 2026).
    body = src[src.index("def drill_watch("):src.index("\ndef ", src.index("def drill_watch(") + 1)]
    assert "hf_store" not in body and "open_store" not in body
    # The build's commits against the hub's limit of 128 an hour: one a set, eight for the whole build and one for
    # the rehearsal's corpus, each tried at most four times; and every push passes the VM's commit budget.
    sets = [ln for ln in text.splitlines() if ln.strip().startswith("push_set ")]
    assert len(sets) == 9 and sum("corpus/rehearsal" in ln for ln in sets) == 1
    assert "for try in 1 2 3 4; do" in text and 4 * (len(sets) - 1) == 32 < 128 // 2
    push = next(ln for ln in text.splitlines() if "python3 $CC/hf_store.py push" in ln) + text.split("python3 $CC/hf_store.py push", 1)[1].split("\n", 2)[1]
    assert '--commit-budget "$COMMIT_BUDGET"' in push and '--commit-wait "$COMMIT_WAIT_S"' in push
    assert "COMMIT_BUDGET=${BASE2_STORE_COMMIT_BUDGET:-30}" in text and "per writer, not per repository" in text


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

    def vm_config(self, out, cfg=None):
        """The store config as ONE VM holds it: the same store, with that VM's own commit ledger. On a real VM the
        ledger lies beside the VM's config (CHAT/store_commits.json) and counts that VM's commits alone. Here every
        "VM" is a folder of one test session, and with one config for all of them they shared one ledger: the 31st
        push of the whole file met the data job's budget of 30 an hour and waited its 15 minutes, which no VM of
        the real build does (eight commits and a branch, 32 with every retry)."""
        src = Path(cfg or self.cfg)
        d = json.loads(src.read_text())
        d.setdefault("commit_ledger", str(self.tmp / f"{Path(out).name}.store_commits.json"))
        dst = self.tmp / f"{Path(out).name}.{src.name}"
        dst.write_text(json.dumps(d))
        return dst

    def run(self, out, mode, cfg=None, expect=0, **env):
        e = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BASE2_STORE_CONFIG": str(self.vm_config(out, cfg)), "BASE2_PINS": str(self.pins),
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

    def fork_store(self, name, src=None):
        """A copy of the store as it stands (or of a kept copy), with its own config: a second VM that never saw the
        later stages."""
        dst = self.tmp / name
        shutil.copytree(src or self.store, dst)
        cfg = self.tmp / f"{name}.json"
        cfg.write_text(json.dumps({"repo": f"dir:{dst}"}))
        return cfg

    def after(self, stage, name):
        """A fresh copy of the store as it stood when that stage had just been published."""
        return self.fork_store(name, self.tmp / f"snap{stage}")


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
    m.fork_store("snap1")
    m.out2 = m.run(m.b, "stage2")
    m.n2 = len(m.commits())
    m.out2_again = m.run(m.b, "stage2")
    m.fork_store("snap2")
    m.out3 = m.run(m.b, "stage3")
    m.n3 = len(m.commits())
    m.out3_again = m.run(m.b, "all")
    return m


def test_stage1_joins_filters_and_pushes(mini):
    receipt = json.load(open(mini.a / "base2_stage1.json"))
    led = receipt["built"]
    # The receipt names what was read back from the store, set by set.
    assert receipt["published_src"]["verified"] and receipt["published_panels"]["verified"]
    assert set(receipt["published_src"]["files"]) == {"fineweb2345.txt", "simplewiki.txt", "human1.txt", "soda.txt",
                                                      "ultrachat.txt", "everyday.txt", "human_val.txt", "human_sft.jsonl"}
    assert receipt["published_src"]["files"]["fineweb2345.txt"]["sha256"] == P.sha256(str(mini.a / "src" / "fineweb2345.txt"))
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
    cfg_c = mini.after(1, "store_c")
    out = mini.run(c, "stage2", cfg=cfg_c)
    assert "stage 2 done" in out
    got = json.load(open(c / "base2_pins_observed.json"))
    assert got["build-N"]["observed"] == want["build-N"]["observed"]
    d = mini.tmp / "vm_d"
    out = mini.run(d, "stage3", cfg=mini.after(2, "store_d"))
    assert "stage 3 done" in out and "pulled base2-build:build/stage2" in out
    got = json.load(open(d / "base2_pins_observed.json"))
    assert got["build-S"]["observed"] == want["build-S"]["observed"]
    assert json.load(open(d / "scenario.ledger.json"))["sha256"] == json.load(open(mini.b / "scenario.ledger.json"))["sha256"]
    # A stage stopped half-way goes on from what it has: with its receipt and its ids gone, the mix's text is still
    # there, so the rerun tokenises again and reaches the same ids. The set it had published is found in the store
    # with the same bytes and is not sent again.
    for f in ("base2_stage2.json", "N.main.ids", "N.tail.ids", "ids_N.json"):
        (c / f).unlink()
    shutil.rmtree(mini.tmp / "store_c" / "base2-build" / "build" / "stage2")
    before = (c / "N.ledger.json").stat().st_mtime_ns
    n_commits = len(mini.commits(mini.tmp / "store_c"))
    out = mini.run(c, "stage2", cfg=cfg_c)
    assert "stage 2 done" in out and (c / "N.ledger.json").stat().st_mtime_ns == before
    assert "published main:corpus/N (the store held these bytes already)" in out
    assert len(mini.commits(mini.tmp / "store_c")) == n_commits + 1          # the receipt alone
    assert json.load(open(c / "base2_pins_observed.json"))["build-N"]["observed"] == want["build-N"]["observed"]


def test_a_changed_input_rebuilds_everything_built_from_it(mini):
    """A recipe, a mix and ids left by an earlier run are never reused once a source's bytes differ."""
    k = mini.tmp / "vm_k"
    cfg = mini.after(1, "store_k")
    mini.run(k, "stage2", cfg=cfg)
    first = json.load(open(k / "base2_pins_observed.json"))["build-N"]["observed"]
    recipe = json.load(open(k / "recipe_N.json"))
    (k / "base2_stage2.json").unlink()
    shutil.rmtree(mini.tmp / "store_k" / "base2-build" / "build" / "stage2")
    more = "<|user|>a new line of talk here<|assistant|>and its answer" + SEP
    with open(k / "src" / "everyday.txt", "a", encoding="utf-8") as f:       # one more dialogue in one source
        f.write(more)
    out = mini.run(k, "stage2", cfg=cfg)
    assert "stage 2 done" in out
    again = json.load(open(k / "recipe_N.json"))
    assert again["available_bytes"]["everyday"] == recipe["available_bytes"]["everyday"] + len(more)
    assert again["total_bytes"] != recipe["total_bytes"]
    second = json.load(open(k / "base2_pins_observed.json"))["build-N"]["observed"]
    assert second["N/N.main.ids"] != first["N/N.main.ids"]
    # The store now holds the rebuilt ids, read back against the local bytes.
    assert P.sha256(str(mini.tmp / "store_k" / "main" / "corpus" / "N" / "N.main.ids")) == second["N/N.main.ids"]


def test_a_failed_push_is_not_a_finished_stage_and_the_rerun_publishes_without_rebuilding(mini):
    p = mini.tmp / "vm_p"
    cfg = mini.after(1, "store_p")
    locked = mini.tmp / "store_p" / "main" / "corpus"
    os.chmod(locked, 0o555)                                          # the store refuses the new set
    try:
        out = mini.run(p, "stage2", cfg=cfg, expect=1, BASE2_KEEP_MIX_TEXT="0")
    finally:
        os.chmod(locked, 0o755)
    assert "stage 2 built" in out and "stage 2 done" not in out
    assert out.count("push of main:corpus/N failed or did not verify") == 4 and "main:corpus/N is not published; stopping" in out
    assert not (p / "base2_stage2.json").exists() and (p / "base2_stage2.built.json").exists()
    assert not (mini.tmp / "store_p" / "base2-build" / "build" / "stage2").exists()     # no receipt in the store
    assert (p / "N.main.txt").exists() and (p / "N.tail.txt").exists()                  # the text waits for publication
    kept = {f: (p / f).stat().st_mtime_ns for f in ("N.ledger.json", "N.main.ids", "N.tail.ids", "ids_N.json", "recipe_N.json",
                                                    "N.keys", "N.main.txt")}
    out = mini.run(p, "stage2", cfg=cfg, BASE2_KEEP_MIX_TEXT="0")
    assert "stage 2 done" in out and "published main:corpus/N (try 1, verified)" in out
    for f in ("N.ledger.json", "N.main.ids", "N.tail.ids", "ids_N.json", "recipe_N.json", "N.keys"):
        assert (p / f).stat().st_mtime_ns == kept[f], f                                 # nothing was built again
    assert not (p / "N.main.txt").exists() and not (p / "N.tail.txt").exists()          # gone only now
    receipt = json.load(open(p / "base2_stage2.json"))
    assert receipt["published_N"]["verified"] and receipt["built"]["documents"]["documents"] > 0
    stored = mini.tmp / "store_p" / "main" / "corpus" / "N" / "N.main.ids"
    assert stored.read_bytes() == (p / "N.main.ids").read_bytes()
    assert (mini.tmp / "store_p" / "base2-build" / "build" / "stage2" / "base2_stage2.json").exists()
    assert "stage 2: finished already" in mini.run(p, "stage2", cfg=cfg)


def test_a_receipt_without_its_outputs_stops_the_build(mini):
    q = mini.tmp / "vm_q"
    cfg = mini.after(2, "store_q")
    shutil.rmtree(mini.tmp / "store_q" / "main" / "corpus" / "N")
    out = mini.run(q, "stage2", cfg=cfg, expect=1)
    assert "the store holds stage 2's receipt and not main:corpus/N; stopping" in out and "finished already" not in out
    # A local receipt the store does not hold is not believed either: the stage builds and publishes.
    r = mini.tmp / "vm_r"
    r.mkdir()
    (r / "base2_stage2.json").write_text(json.dumps({"stage": "2"}))
    out = mini.run(r, "stage2", cfg=mini.after(1, "store_r"))
    assert "stage 2 built" in out and "stage 2 done" in out
    assert json.load(open(r / "base2_stage2.json"))["published_N"]["verified"]


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
    rec = json.load(open(mini.b / "base2_stage3.json"))["built"]
    for part in ("main", "tail"):
        docs = docs_of(mini.b / f"S.{part}.txt")
        scn = [d for d in docs if d not in natural]
        nat = [d for d in docs if d in natural]
        assert scn and nat
        # An independent count, each document on its own with the newline after it (the other convention).
        t_s = sum(len(x) for x in enc([d + SEP for d in scn]))
        t_n = sum(len(x) for x in enc([d + SEP for d in nat]))
        share = t_s / (t_s + t_n)
        assert 0.14 <= share <= 0.16, (part, share)
        assert abs(share - 0.15) < 0.005, (part, share)       # the measured ratio lands well inside the bounds
        ids = ids_tokens(mini.b / f"S.{part}.ids")
        mine = rec[f"share_{part}"]
        assert mine["ok"] and mine["scenario_documents"] == len(scn)
        assert mine["tokens"] == ids and mine["tokens_from"] == "ids" and mine["boundary_error_tokens"] == 0
        # The two conventions differ by where a newline's tokens fall: at most one token a scenario document.
        assert abs(mine["scenario_tokens"] - t_s) <= len(scn) and abs(mine["scenario_share_of_tokens"] - share) <= mine["convention_bound"]
        assert mine["convention_bound"] == len(scn) / ids < 0.002
        # No boundary error: every document encoded as the file encodes it sums to the ids file's own count, exactly.
        whole = B.share(str(mini.b / f"S.{part}.txt"), str(regen_scenario(mini)), str(mini.vocab))
        assert whole["tokens"] == ids and whole["tokens_from"] == "documents"
        assert whole["scenario_tokens"] == mine["scenario_tokens"]
        # N's and S's actual token counts, as the receipt reports them: S keeps N's count in each part.
        counts = rec["token_counts"]
        assert counts["S"][part] == ids and counts["N"][part] == ids_tokens(mini.b / f"N.{part}.ids")
        assert abs(counts["relative_difference"][part]) < 0.01 and counts["ok"]
    assert rec["token_counts"]["N_total"] == sum(rec["token_counts"]["N"].values())
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
    # The build's own check is by document too: S's against the digests of N's, which outlive N's text.
    by_doc = json.load(open(mini.b / "base2_stage3.json"))["built"]["subset_documents"]
    assert by_doc["subset"] and by_doc["documents_of_S_not_in_N"] == 0
    assert by_doc["natural_documents_in_S"] == sum(s_nat.values()) and by_doc["scenario_documents_in_S"] == len(s_scn)
    assert by_doc["documents_in_N"] == sum(n.values())
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
    out = mini.run(e, "stage3", cfg=mini.after(2, "store_e"), expect=1, BASE2_SCN_MAX_DROP="-0.5")
    assert (e / "scenario.ledger.json.refused").exists() and not (e / "S.main.ids").exists()
    assert "stage 3 done" not in out


def test_an_unpinned_engine_generates_nothing(mini):
    f = mini.tmp / "vm_f"
    pins = json.load(open(mini.pins))
    pins["engine"]["files"]["generate.py"] = "0" * 64
    bad = mini.tmp / "pins_bad.json"
    bad.write_text(json.dumps(pins))
    out = mini.run(f, "stage3", cfg=mini.after(2, "store_f"), expect=1, BASE2_PINS=str(bad))
    assert "not the pinned bytes" in out and not (f / "scn" / "panel_scenario.txt").exists()


STUB_VALIDATOR = """import json
import os
import sys

args = sys.argv[1:]
name = os.path.basename(args[0])
bad = bool(os.environ.get("STUB_FAIL")) and os.environ["STUB_FAIL"] in name
with open(args[args.index("--report") + 1], "w") as f:
    json.dump({"verdict": "FAIL" if bad else "PASS", "file": name, "heldout": "--heldout" in args}, f)
sys.exit(1 if bad else 0)
"""


def test_the_validator_runs_on_every_scenario_shard(mini):
    """The engine's own validator enforces rates a few hundred documents cannot meet, so a stand-in with its command
    line takes its place here: the build must call it on the held-out panel and on every shard, and stop on one FAIL."""
    eng = mini.tmp / "engine_stub"
    shutil.copytree(mini.engine, eng, ignore=shutil.ignore_patterns("__pycache__", "base2_pins_observed.json"))
    (eng / "validate.py").write_text(STUB_VALIDATOR)
    pins = json.load(open(mini.pins))
    P.pin_engine(pins, str(eng), commit="stand-in validator", repin=True)
    pf = mini.tmp / "pins_stub.json"
    pf.write_text(json.dumps(pins))
    env = {"BASE2_SCN_VALIDATE": "1", "BASE2_ENGINE_DIR": str(eng), "BASE2_PINS": str(pf)}
    v = mini.tmp / "vm_v"
    out = mini.run(v, "stage3", cfg=mini.after(2, "store_v"), **env)
    assert "stage 3 done" in out and "validator on every shard" in out
    led = json.load(open(v / "scenario.ledger.json"))
    shards = json.load(open(v / "recipe_S.json"))["scenario_shards"]
    assert shards >= 3 and led["validator"] == {"shards": shards, "verdicts": ["PASS"] * shards, "passed": True}
    assert json.load(open(v / "scenario_validate_heldout.json")) == {"verdict": "PASS", "file": "panel_scenario.txt.tmp",
                                                                    "heldout": True}
    # With the validator on, the same stream as with it off.
    assert led["sha256"] == json.load(open(mini.b / "scenario.ledger.json"))["sha256"]
    assert json.load(open(mini.b / "scenario.ledger.json"))["validator"]["passed"] is None
    w = mini.tmp / "vm_w"
    out = mini.run(w, "stage3", cfg=mini.after(2, "store_w"), expect=1, STUB_FAIL="raw_2.", **env)
    assert "scenario shard 2 fails the engine's validator" in out and "stage 3 built" not in out
    assert not (w / "S.main.ids").exists() and not (w / "scenario.ledger.json").exists()
    assert not (mini.tmp / "store_w" / "main" / "corpus" / "S").exists()
    # A shard that never met the validator is not accepted where every shard must hold its PASS.
    led_dir = mini.tmp / "ledger_check"
    shutil.copytree(w / "scn", led_dir)
    (led_dir / "validate_2.json").unlink(missing_ok=True)
    for k in range(shards):
        if not (led_dir / f"shard_{k}.ledger.json").exists():
            shutil.copyfile(led_dir / "shard_0.ledger.json", led_dir / f"shard_{k}.ledger.json")
    r = B.scn_ledger(str(led_dir), shards, str(v / "recipe_S.json"), json.load(open(v / "recipe_S.json")), 20261111, 0.02,
                     validate=True)
    assert r["validator"]["passed"] is False and r["validator"]["verdicts"][2] is None and not r["ok"]


FAKE_TRAINER = """#!/usr/bin/env python3
import json
import os
import sys

spec = json.load(open(sys.argv[2]))
os.makedirs(spec["out_dir"], exist_ok=True)
seen = {f: os.path.exists(os.path.join(os.environ["PROBE_DATA"], f)) for f in ("S.main.ids", "S.tail.ids")}
json.dump(seen, open(os.environ["PROBE_SAW"], "w"))
with open(os.path.join(spec["out_dir"], "events.jsonl"), "w") as f:
    f.write(json.dumps({"event": "step", "step": 1}) + chr(10))
"""


def test_the_load_probe_pulls_and_checks_s_on_a_vm_that_has_no_ids(mini):
    """Every stage skipped: the VM holds receipts and no ids. The probe's mode brings S's ids and the panels from the
    store, checked against the manifests, before the trainer starts."""
    fresh = mini.tmp / "vm_l"
    out = mini.run(fresh, "all")
    assert all(f"stage {n}: finished already" in out for n in (1, 2, 3))
    assert not (fresh / "S.main.ids").exists()
    fake = mini.tmp / "fake_mtstudio.py"
    fake.write_text(FAKE_TRAINER)
    os.chmod(fake, 0o755)
    saw = mini.tmp / "probe_saw.json"
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BASE2_STORE_CONFIG": str(mini.cfg), "BASE2_PINS": str(mini.pins),
         "BASE2_STORE_RETRY_S": "0", "JOBS_MTSTUDIO": str(fake), "PROBE_DATA": str(fresh), "PROBE_SAW": str(saw)}
    r = subprocess.run(["bash", "tools/colab_chat/base2_data.sh", str(fresh), str(mini.vocab), "loadprobe", str(mini.tmp / "probe")],
                       cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "pulled main:corpus/S" in r.stdout and json.load(open(saw)) == {"S.main.ids": True, "S.tail.ids": True}
    for f in ("S.main.ids", "S.tail.ids"):
        assert (fresh / f).read_bytes() == (mini.store / "main" / "corpus" / "S" / f).read_bytes()
    probe = json.load(open(mini.tmp / "probe" / "loadprobe.json"))
    assert probe["returncode"] == 0 and probe["seconds_to_first_step"] is not None
    assert json.load(open(fresh / "base2_pins_observed.json"))["build-S"]["differences"] == []
    # A copy on the VM that is not the store's bytes is replaced before the probe, never loaded.
    (fresh / "S.tail.ids").write_bytes(b"not the ids")
    r = subprocess.run(["bash", "tools/colab_chat/base2_data.sh", str(fresh), str(mini.vocab), "loadprobe", str(mini.tmp / "probe2")],
                       cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert (fresh / "S.tail.ids").read_bytes() == (mini.store / "main" / "corpus" / "S" / "S.tail.ids").read_bytes()
    # With no S in the store there is nothing to load, and the probe says so by failing.
    cfg = mini.after(2, "store_l")
    r = subprocess.run(["bash", "tools/colab_chat/base2_data.sh", str(mini.tmp / "vm_l2"), str(mini.vocab), "loadprobe",
                        str(mini.tmp / "probe3")], cwd=ROOT, env={**e, "BASE2_STORE_CONFIG": str(cfg)}, capture_output=True, text=True)
    assert r.returncode != 0 and not (mini.tmp / "probe3" / "loadprobe.json").exists()


def test_rehearsal_corpora_are_two_sizes_built_once_and_then_pulled(mini):
    g, h = mini.tmp / "vm_g", mini.tmp / "vm_h"
    cfg = mini.fork_store("store_reh")
    out = mini.run(g, "rehearsal", cfg=cfg, BASE2_REH_MAIN_BYTES="200000")
    assert "published main:corpus/rehearsal" in out
    main, tail = ids_tokens(g / "rehearsal.main.ids"), ids_tokens(g / "rehearsal.tail.ids")
    assert main > 1.5 * tail > 10000                                  # a main and a tail of different sizes
    rec = [json.loads(ln) for ln in open(g / "rehearsal.json")]
    assert [os.path.basename(x["out"]) for x in rec] == ["rehearsal.main.ids", "rehearsal.tail.ids"]
    assert [x["tokens"] for x in rec] == [main, tail]
    obs = json.load(open(g / "base2_pins_observed.json"))["rehearsal"]
    assert obs["differences"] == [] and set(obs["observed"]) == {"vocab/vocab_llama3_bpe.gguf", "rehearsal/rehearsal.main.ids",
                                                                 "rehearsal/rehearsal.tail.ids"}
    out = mini.run(h, "rehearsal", cfg=cfg, BASE2_REH_MAIN_BYTES="200000")
    assert "pulled main:corpus/rehearsal" in out and "published" not in out
    for f in ("rehearsal.main.ids", "rehearsal.tail.ids"):
        assert (h / f).read_bytes() == (g / f).read_bytes()


def test_the_rehearsal_builds_on_a_hub_that_lacks_its_corpus_and_stops_on_one_it_cannot_read(mini):
    """base2_data.sh rehearsal, end to end, with every store call of the script going through hf_store.HfBackend
    (tests/hub_standin.py stands in for the hub and raises the library's own errors). The first VM finds no
    main:corpus/rehearsal: that is an answer, so it builds, publishes and verifies. The second VM, part b's, finds the
    first's corpus and pulls it. A hub that says "repository not found", refuses the login, cannot be reached or does
    not answer stops the build with nothing built."""
    pytest.importorskip("huggingface_hub")
    cfg, root = hub_config(mini.tmp, "hub_reh")
    env = hub_env(root)
    a, b = mini.tmp / "vm_reh_a", mini.tmp / "vm_reh_b"
    out = mini.run(a, "rehearsal", cfg=cfg, BASE2_REH_MAIN_BYTES="200000", **env)
    assert "published main:corpus/rehearsal (try 1, verified)" in out
    log = (root / "calls.log").read_text()
    assert "raise EntryNotFoundError corpus/rehearsal/manifest.json" in log        # the absent answer, from the hub
    assert sum(ln.startswith("create_commit main files: corpus/rehearsal") for ln in log.splitlines()) == 1
    stored = root / "branches" / "main" / "corpus" / "rehearsal"
    assert sorted(os.listdir(stored)) == ["manifest.json", "rehearsal.json", "rehearsal.main.ids", "rehearsal.tail.ids"]
    assert (stored / "rehearsal.main.ids").read_bytes() == (a / "rehearsal.main.ids").read_bytes()
    out = mini.run(b, "rehearsal", cfg=cfg, BASE2_REH_MAIN_BYTES="200000", **env)
    assert "pulled main:corpus/rehearsal" in out and "published" not in out          # part b reuses part a's corpus
    for f in ("rehearsal.main.ids", "rehearsal.tail.ids"):
        assert (b / f).read_bytes() == (a / f).read_bytes()
    assert HUB_TOKEN not in "".join(mini.logs[-2:]) and HUB_TOKEN not in (root / "calls.log").read_text()
    for mode, extra in (("repo_not_found", {}), ("auth", {}), ("transport", {}), ("hang", {"BASE2_STORE_LIMIT_S": "1"})):
        (root / "fail.json").write_text(json.dumps({"mode": mode}))
        c = mini.tmp / f"vm_reh_{mode}"
        out = mini.run(c, "rehearsal", cfg=cfg, expect=1, BASE2_REH_MAIN_BYTES="200000", **env, **extra)
        assert "the store could not be read (main:corpus/rehearsal); stopping" in out, mode
        assert not (c / "rehearsal.main.ids").exists() and not (c / "work").exists(), mode
    (root / "fail.json").unlink()
