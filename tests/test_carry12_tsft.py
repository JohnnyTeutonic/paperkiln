"""tools/carry12_tsft_build.py, cell TS's template-SFT set (experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md
sections 0, 4 and 6), and the data job's --template-sft path: the set is seeded and byte-reproducible (its pin), its
design holds (families, N 1-12 balanced, one target per dialogue, at most 16 turns), the disjointness check against
Carry-12 v1 and Carry-5 v1 passes and catches planted overlaps, the committed receipts are the tool's output, and
tiny_chat_data.py --template-sft writes TS's SFT set (the human rows once each plus the 4,224 template rows) without
changing the job's other outputs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import carry12_tsft_build as TS  # noqa: E402

LONG = ROOT / "experiments" / "tiny_chat_long_2026-10"


@pytest.fixture(scope="module")
def rows():
    return TS.build_rows()


def test_seeded_and_pinned(rows):
    data = TS.dump(rows)
    assert data == TS.dump(TS.build_rows())  # deterministic
    assert TS.sha256_bytes(data) == TS.pinned() == "3dbe77b5008fa0f0056416e393f9611b7907ad4d6cf29956ca1898c165c28e0c"
    got, sha = TS.build_checked()
    assert got == data and sha == TS.pinned()


def test_design(rows):
    from collections import Counter
    fam = Counter(r["meta"]["family"] for r in rows)
    assert fam == {"recall": 2160, "correction": 1440, "instruction": 624} and len(rows) == 4224
    for f, n in fam.items():
        by_n = Counter(r["meta"]["N"] for r in rows if r["meta"]["family"] == f)
        assert sorted(by_n) == list(range(1, 13)) and set(by_n.values()) == {n // 12}, f
    assert Counter(r["meta"]["slot"] for r in rows if r["meta"]["family"] == "recall") == {s: 360 for s in TS.SLOTS}
    assert all(r["meta"]["n_turns"] <= 16 and r["meta"]["target_turn"] == r["meta"]["n_turns"] - 1 for r in rows)
    assert len({tuple(m["content"] for m in r["messages"][0::2]) for r in rows}) == len(rows)  # no two alike
    # one carry target per dialogue: the question's slot is asked once, at the target turn
    for r in rows[:500] + rows[2160:2700]:
        q = [k for k, x in enumerate(r["meta"]["turn_kinds"]) if x == "question"]
        assert q == [r["meta"]["target_turn"] - 1], r["uri"]
    # correction: the old value is held before the fix, the question needs the new one
    for r in rows[2160:2160 + 200]:
        m = r["meta"]
        assert m["old_value"] in r["messages"][2 * (m["plant_turn"] - 1)]["content"]
        assert m["value"] in r["messages"][2 * m["target_turn"] - 1]["content"]
    # instruction: the four forms per kind rotate (each gold 78 times)
    forms = Counter(r["meta"]["form"] for r in rows if r["meta"]["family"] == "instruction")
    assert set(forms.values()) == {78} and len(forms) == 8
    # no turn comes from the human dialogue pool: every user turn is one of this tool's texts
    own = {q for q, _ in TS.SMALL_TALK + TS.OPENERS + TS.CLOSINGS} | {q for a in TS.ARCS for q, _ in a}
    for r in rows[::37]:
        for m, k in zip(r["messages"][0::2], r["meta"]["turn_kinds"]):
            if k in ("small_talk", "opener", "closing"):
                assert m["content"] in own


def test_disjoint_from_carry12_and_carry5(rows):
    rep = TS.disjoint_report(rows)
    assert rep["disjoint"] is True and rep["decontam_py"]["rows_dropped"] == 0
    for name in ("carry12", "carry5"):
        r = rep[name]
        assert r["carry5_build_scan"] == {} and r["entity_words_in_text"] == {} and r["equal_turns_n"] == 0
        assert r["value_overlap"]["whole_strings"] == [] and r["value_overlap"]["content_words"] == []
    receipt = json.loads((LONG / "carry12_tsft_v1_disjoint.json").read_text(encoding="utf-8"))
    assert receipt["sha256"] == TS.pinned() and receipt["disjoint"] is True
    assert {k: v for k, v in receipt.items() if k not in ("version", "sha256")} == json.loads(json.dumps(rep))


def test_the_check_catches_planted_overlap(rows, monkeypatch):
    """A Carry-12 value among the boats, a Carry-12 user turn as small talk and a Carry-12 reply sentence: every
    one is caught."""
    c12 = [json.loads(x) for x in open(TS.CARRY12, encoding="utf-8") if x.strip()]
    turn = next(t["user"] for t in c12[0]["turns"] if len(t["user"].split()) >= 6)
    bad = json.loads(json.dumps(rows[:40]))
    bad[0]["messages"][0]["content"] = turn
    bad[1]["messages"][1]["content"] = "Rain kept me in. That is annoying."
    bad[2]["messages"][0]["content"] = "My boat is named Biscuit."
    monkeypatch.setattr(TS, "BOATS", TS.BOATS + ["Leeds"])
    monkeypatch.setitem(TS.SLOTS["boat"], "values", TS.BOATS + ["Leeds"])
    rep = TS.disjoint_report(bad)
    assert rep["disjoint"] is False
    r = rep["carry12"]
    assert r["equal_turns_n"] >= 1 and "sentence" in r["carry5_build_scan"]
    assert "leeds" in r["value_overlap"]["content_words"] and "biscuit" in r["entity_words_in_text"]
    assert rep["decontam_py"]["rows_dropped"] >= 1  # the user turn is a Carry-12 prompt of 4+ words


def test_committed_ledger_is_the_tools_output(rows):
    led = json.loads((LONG / "carry12_tsft_v1_build.json").read_text(encoding="utf-8"))
    fresh = TS.ledger(rows, TS.dump(rows))
    assert {k: v for k, v in led.items() if k != "tokens"} == json.loads(json.dumps(fresh))
    assert led["tokens"]["over_1024"] == 0 and led["tokens"]["per_row"]["max"] <= 1025
    assert led["tokens"]["vocab_sha256"] == "0b880aa073a7e27c0770d5ecc5215b55d6d415080d20bd3dc7b169ecabadf198"


def test_data_job_template_sft(tmp_path):
    """--template-sft (with --carry12): sft_human_tsft.jsonl holds sft_human.jsonl's rows once each plus the 4,224
    template rows, shuffled; the ledger counts them; the job's other outputs equal the job without the option; and
    --template-sft without --carry12 is refused."""
    from test_tiny_chat_pipeline import DATA, dialogues, fineweb, vocab
    fineweb(tmp_path / "fw", 400)
    dialogues(tmp_path / "human" / "toy", 300)
    vocab(tmp_path / "v.gguf")

    def run(out, *extra, ok=True):
        p = subprocess.run(DATA + ["--out", str(out), "--vocab", str(tmp_path / "v.gguf"), "--fineweb",
                                   str(tmp_path / "fw"), "--shares", "35", "--mix-bytes", "60000", "--T", "4096",
                                   "--human-dir", str(tmp_path / "human"), "--human-fetch", "", *extra],
                           capture_output=True, text=True, cwd=ROOT)
        assert (p.returncode == 0) == ok, p.stdout[-2000:] + p.stderr[-2000:]
        return json.loads((out / "data.ledger.json").read_text()) if ok else p.stderr
    err = run(tmp_path / "bad", "--template-sft", ok=False)
    assert "--template-sft needs --carry12" in err
    base = run(tmp_path / "plain", "--carry12")
    led = run(tmp_path / "tsft", "--carry12", "--template-sft")
    ts = led["template_sft"]
    sft = (tmp_path / "tsft" / "sft_human.jsonl").read_text().splitlines()
    mixed = (tmp_path / "tsft" / "sft_human_tsft.jsonl").read_text().splitlines()
    assert ts["sha256"] == TS.pinned() and ts["dialogues"] == 4224 and ts["decontam"]["rows_dropped"] == 0
    assert ts["rows_human"] == len(sft) and ts["rows_template"] == 4224 and len(mixed) == ts["rows_out"]
    assert abs(ts["template_share"] - 4224 / (len(sft) + 4224)) < 1e-4
    src = [json.loads(x)["mix_source"] for x in mixed]
    assert src.count("tsft_raw.jsonl") == 4224 and src[:50].count("tsft_raw.jsonl") not in (0, 50)  # shuffled
    human = sorted(json.dumps({k: v for k, v in json.loads(x).items() if k != "mix_source"}, sort_keys=True)
                   for x in mixed if json.loads(x)["mix_source"] == "sft_raw.jsonl")
    assert human == sorted(json.dumps({k: v for k, v in json.loads(x).items() if k != "mix_source"}, sort_keys=True)
                           for x in sft)
    assert all(sum(s["loss"] for s in json.loads(x)["segments"]) >= 2 for x in mixed
               if json.loads(x)["mix_source"] == "tsft_raw.jsonl")
    for f in ("sft_human.jsonl", "mix_s35.ids", "panel_fineweb.ids", "panel_dialogue.ids"):
        assert led["outputs"][f]["sha256"] == base["outputs"][f]["sha256"], f
    assert "sft_human_tsft.jsonl" in led["outputs"] and "template_sft" not in base
    assert (tmp_path / "tsft" / "tsft_decontam.json").exists()
    assert not (tmp_path / "tsft" / "tsft_dialogues.jsonl").exists() and not (tmp_path / "tsft" / "tsft_raw.jsonl").exists()
