"""tools/draft_table.py, ember's corpus draft table (EMBER_DRAFT=table) written beside a chat model's GGUF, and
mtstudio's opt-in export.draft_table.

- The vendored builder equals ember's experiments/draft_2026-10-27/dr_table.py byte for byte on the same ids, and
  its tokenisation equals dr_table.py tokenize's on the same text (EMBER_DR_TABLE, default the sibling ember.cpp
  checkout; skipped without it). On the full corpora ember's h_c8_p10.bin and s_c8_p10.bin are reproduced byte for
  byte (experiments/draft_table_2026-10-27, receipts; not rerun here, ten minutes of CPU).
- JSONL rows ({"messages"}, {"segments"}, {"prompt", "completion"}) give the tokens of the same dialogue as chat
  text, and the file passes the loader's checks (draft_table.h DraftTable::load); a toy corpus drafts its cycle.
- mtstudio (MTSTUDIO_BIN): with export.draft_table the run writes the table beside the GGUF, equal to the tool's
  own build; without it nothing changes (no table, no log, the GGUF byte-identical to the run with the table);
  `mtstudio export` writes it too; a spec that cannot have a table refuses before training.

The tokeniser tests need the 16k paperkiln vocabulary (PK_VOCAB16, as tests/test_chat_template.py) and tokenizers.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import chat_template as CT  # noqa: E402
import draft_table as D  # noqa: E402
from test_chat_template import VOCAB16  # noqa: E402

EMBER_DR = Path(os.environ.get("EMBER_DR_TABLE", str(ROOT.parent / "ember.cpp" / "experiments" / "draft_2026-10-27"
                                                       / "dr_table.py")))
BIN = os.environ.get("MTSTUDIO_BIN")
needs_ember = pytest.mark.skipif(not EMBER_DR.is_file(), reason="set EMBER_DR_TABLE to ember's dr_table.py")
needs_vocab = pytest.mark.skipif(not VOCAB16.is_file(), reason="set PK_VOCAB16 to the 16k paperkiln vocabulary")
needs_bin = pytest.mark.skipif(not BIN or not VOCAB16.is_file(), reason="set MTSTUDIO_BIN and PK_VOCAB16")

DIALOGUES = [
    [("Hi, how are you today?", "I'm well, thanks. How about you?"), ("Fine. Any plans?", "Just reading a book.")],
    [("What is the capital of France?", "The capital of France is Paris.")],
    [("Can you help me?\n\nI need a recipe.", "Of course. What would you like to cook?"),
     ("Pasta, please.", "Boil the pasta, then add the sauce.\r\nEnjoy!")],
]


def ember_module():
    spec = importlib.util.spec_from_file_location("ember_dr_table", EMBER_DR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def structured_ids(n, vocab, seed):
    """A corpus with repeated phrases (so every order has contexts above the count floor) and noise."""
    rng = np.random.default_rng(seed)
    phrases = [rng.integers(0, vocab, size=rng.integers(3, 9)) for _ in range(40)]
    out = []
    while sum(len(p) for p in out) < n:
        out.append(phrases[rng.integers(0, len(phrases))] if rng.random() < 0.8 else rng.integers(0, vocab, size=3))
    return np.concatenate(out)[:n].astype(np.uint16)


def chat_text(dialogues):
    return "".join(CT.render_document(CT.from_pairs(d)) for d in dialogues)


@needs_ember
@pytest.mark.parametrize("vocab,min_count,min_conf,orders", [(300, 8, 0.10, 4), (16384, 2, 0.05, 4), (50, 4, 0.25, 3)])
def test_builder_equals_embers_byte_for_byte(tmp_path, vocab, min_count, min_conf, orders):
    E = ember_module()
    ids = structured_ids(60_000, vocab, seed=vocab)
    corpus = tmp_path / "c.u16"
    ids.tofile(corpus)
    out = tmp_path / "ember.bin"
    E.build(argparse.Namespace(corpus=str(corpus), out=str(out), vocab_size=vocab, orders=orders, min_count=min_count,
                               min_conf=min_conf, prune_redundant=1))
    ours = D.table_bytes(vocab, D.build_tables(ids, vocab, orders, min_count, min_conf, 1, log=lambda s: None))
    assert ours == out.read_bytes()
    assert D.load_problem(ours, vocab) == ""
    v, tables = D.read(str(out))
    assert v == vocab and sum(t["keys"].size for t in tables) > 100
    for ctx in ([int(ids[5])], [int(x) for x in ids[10:13]], [int(x) for x in ids[20:24]]):
        assert D.key_of(ctx) == E.key_of(ctx)


@needs_ember
@needs_vocab
def test_tokenisation_equals_embers(tmp_path):
    pytest.importorskip("tokenizers")
    E = ember_module()
    text = tmp_path / "text.txt"
    text.write_text(chat_text(DIALOGUES * 3), encoding="utf-8", newline="")
    out = tmp_path / "e.u16"
    E.tokenize(argparse.Namespace(vocab=str(VOCAB16), out=str(out), text=[str(text)], max_bytes=0,
                                  pk_tools=str(ROOT / "tools")))
    vocab, tok, tmpl, _ = D.model_vocab(str(VOCAB16))
    ids, recs, sha = D.tokenise([str(text)], tok, tmpl, log=lambda s: None)
    assert ids.tobytes() == out.read_bytes() and sha == hashlib.sha256(out.read_bytes()).hexdigest()
    assert recs[0]["tokens"] == ids.size and vocab == 16384 and CT.rid(tmpl) == "paperkiln/1"
    # --max-bytes: whole lines, the one that crosses the limit included, as dr_table.py.
    E.tokenize(argparse.Namespace(vocab=str(VOCAB16), out=str(out), text=[str(text)], max_bytes=150,
                                  pk_tools=str(ROOT / "tools")))
    cut, _, _ = D.tokenise([str(text)], tok, tmpl, max_bytes=150, log=lambda s: None)
    assert cut.tobytes() == out.read_bytes() and 0 < cut.size < ids.size


@needs_vocab
def test_jsonl_rows_give_the_dialogues_tokens(tmp_path):
    pytest.importorskip("tokenizers")
    vocab, tok, tmpl, _ = D.model_vocab(str(VOCAB16))
    text = tmp_path / "text.txt"
    text.write_text(chat_text(DIALOGUES), encoding="utf-8", newline="")
    want, _, _ = D.tokenise([str(text)], tok, tmpl, log=lambda s: None)
    msgs = [CT.from_pairs(d) for d in DIALOGUES]
    forms = {
        "messages": [{"messages": m} for m in msgs],
        "segments": [CT.to_segments(m) for m in msgs],
        # a one-exchange dialogue as a {prompt, completion} row (mtstudio's per-turn form)
        "prompt_completion": [{"prompt": CT.render(m[:-1], CT.V1, generation_prompt=True),
                               "completion": m[-1]["content"]} for m in msgs if len(m) == 2],
    }
    for form, rows in forms.items():
        p = tmp_path / f"{form}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        got, recs, _ = D.tokenise([str(p)], tok, tmpl, log=lambda s: None)
        if form == "prompt_completion":
            one = tmp_path / "one.txt"
            one.write_text(chat_text([DIALOGUES[1]]), encoding="utf-8", newline="")
            want_one, _, _ = D.tokenise([str(one)], tok, tmpl, log=lambda s: None)
            assert got.tolist() == want_one.tolist()
        else:
            assert got.tolist() == want.tolist(), form
        assert recs[0]["form"] == "jsonl"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"input": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(D.Refused, match="none of messages"):
        D.tokenise([str(bad)], tok, tmpl, log=lambda s: None)


def test_loader_checks_and_the_drafter():
    ids = np.array([5, 6, 7, 8, 9] * 200, dtype=np.uint16)
    tables = D.build_tables(ids, 16, 4, 8, 0.10, 1, log=lambda s: None)
    data = D.table_bytes(16, tables)
    assert D.load_problem(data, 16) == ""
    # back-off pruning: orders 2 and 3 repeat order 1's answers; order 4 (hashed) is kept
    assert [int(t["keys"].size) for t in tables] == [5, 0, 0, 5]
    assert D.draft(tables, [3, 5], 4) == [6, 7, 8, 9] and D.draft(tables, [9, 5, 6, 7, 8], 2) == [9, 5]
    assert D.draft(tables, [11], 4) == []
    assert D.load_problem(data, 32) == "built for vocabulary size 16, the model has 32"
    assert D.load_problem(b"EMBERDT0" + data[8:], 16) == "not a draft table"
    assert D.load_problem(data + b"\0" * 8, 16) == "trailing bytes"
    assert D.load_problem(data[:-8], 16) == "truncated tables"
    swapped = bytearray(data)
    k0 = 16 + 8 * 4  # order 1's first two keys
    swapped[k0:k0 + 16] = data[k0 + 8:k0 + 16] + data[k0:k0 + 8]
    assert D.load_problem(bytes(swapped), 16) == "entries out of range or not ascending"
    with pytest.raises(D.Refused, match="outside the vocabulary"):
        D.build_tables(ids, 8, 4, 8, 0.10, 1, log=lambda s: None)


# --------------------------------------------------------------------------
# mtstudio export.draft_table

def sft_spec(tmp, run, draft=None, formats=("safetensors", "gguf")):
    """A two-step SFT run named "m" into tmp/run (one name, so two runs' files compare byte for byte)."""
    data = tmp / "sft.jsonl"
    if not data.exists():
        data.write_text("".join(json.dumps({"messages": CT.from_pairs(d)}) + "\n" for d in DIALOGUES * 20),
                        encoding="utf-8")
    spec = {"name": "m", "arch": {"preset": "llama-tiny", "custom": {"d": 32, "layers": 1, "heads": 2}},
            "data": {"vocab": str(VOCAB16), "vocab_cap": 0, "T": 64, "sft_examples": str(data)},
            "train": {"steps": 2, "objective": "sft", "seed": 1, "batch": 2, "eval_every": 100,
                      "checkpoint_every": 100, "gradmap_every": 0},
            "export": {"formats": list(formats)}, "out_dir": str(tmp / run)}
    if draft is not None:
        spec["export"]["draft_table"] = draft
    p = tmp / f"{run}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def mts(*args, env=None):
    return subprocess.run([BIN, *map(str, args)], capture_output=True, text=True, timeout=1800, cwd=ROOT, env=env)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@needs_bin
def test_mtstudio_export_draft_table_opt_in(tmp_path):
    pytest.importorskip("tokenizers")
    off = mts("run", sft_spec(tmp_path, "off"))
    assert off.returncode == 0, off.stdout[-2000:] + off.stderr[-2000:]
    on = mts("run", sft_spec(tmp_path, "on", True))
    assert on.returncode == 0, on.stdout[-2000:] + on.stderr[-2000:]
    # Off: nothing new, nothing said.
    assert not list((tmp_path / "off").glob("*draft_table*")) and "draft_table" not in off.stdout
    # On: the table and its receipt beside the GGUF, equal to the tool's own build from data.sft_examples.
    table = tmp_path / "on" / "m.draft_table.bin"
    rec = json.loads((tmp_path / "on" / "m.draft_table.bin.json").read_text())
    assert rec["sources"][0]["path"] == str(tmp_path / "sft.jsonl") and rec["template"] == "paperkiln/1"
    ref = tmp_path / "ref.bin"
    D.build(str(tmp_path / "on" / "m.gguf"), str(ref), [str(tmp_path / "sft.jsonl")], log=lambda s: None)
    assert table.read_bytes() == ref.read_bytes() and rec["sha256"] == sha(ref)
    assert D.load_problem(table.read_bytes(), 16384) == ""
    assert any(json.loads(x).get("format") == "draft_table" for x in on.stdout.splitlines() if x.startswith("{"))
    # The table changes no other export: the same files, the table's three aside, and the same bytes in them.
    names_off = {p.name for p in (tmp_path / "off").iterdir()}
    names_on = {p.name for p in (tmp_path / "on").iterdir()}
    assert names_on - names_off == {"m.draft_table.bin", "m.draft_table.bin.json", "m.draft_table.log"}
    assert names_off <= names_on
    for f in ("m.gguf", "m.safetensors", "m.ir.json"):
        assert sha(tmp_path / "on" / f) == sha(tmp_path / "off" / f), f
    # `mtstudio export` writes it as well, here from explicit sources.
    spec = sft_spec(tmp_path, "on", {"sources": [str(tmp_path / "sft.jsonl")], "max_bytes": 0})
    ex = mts("export", spec, "--from", tmp_path / "on" / "m.safetensors", "--formats", "gguf",
             "--out-dir", tmp_path / "re")
    assert ex.returncode == 0, ex.stdout[-2000:] + ex.stderr[-2000:]
    assert (tmp_path / "re" / "m.draft_table.bin").read_bytes() == ref.read_bytes()
    assert sha(tmp_path / "re" / "m.gguf") == sha(tmp_path / "off" / "m.gguf")


@needs_bin
def test_mtstudio_refuses_a_table_it_cannot_write(tmp_path):
    for draft, why in (({"sorces": ["x"]}, "not a key it takes"), ("yes", "must be true, false or")):
        r = mts("run", sft_spec(tmp_path, "bad", draft))
        assert r.returncode != 0 and why in r.stderr, (draft, r.stderr[-500:])
    r = mts("run", sft_spec(tmp_path, "nogguf", True, formats=("safetensors",)))
    assert r.returncode != 0 and "needs a GGUF export" in r.stderr
    lm = json.loads(sft_spec(tmp_path, "lm", True).read_text())
    lm["train"]["objective"] = "lm"
    del lm["data"]["sft_examples"]
    lm["data"]["corpus"] = str(tmp_path / "sft.jsonl")
    (tmp_path / "lm.json").write_text(json.dumps(lm), encoding="utf-8")
    r = mts("run", tmp_path / "lm.json")
    assert r.returncode != 0 and "data.sft_examples, or export.draft_table.sources" in r.stderr
    # A missing tool refuses before training, not after it.
    r = mts("run", sft_spec(tmp_path, "early", True),
            env={**os.environ, "MTSTUDIO_DRAFT_TABLE": str(tmp_path / "nope.py")})
    assert r.returncode != 0 and "not found" in r.stderr
    assert not (tmp_path / "early" / "m.safetensors").exists()
