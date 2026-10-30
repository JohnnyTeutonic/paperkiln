"""Prospective cache provenance: no network, tokenizers, models or historical ledgers."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/colab_chat/tiny_chat_data.py"
spec = importlib.util.spec_from_file_location("tiny_chat_cache_test", SOURCE)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


@pytest.fixture
def cached(tmp_path):
    for name in data.HUMAN_CACHE_FILES:
        (tmp_path / name).write_text(name + "\n")
    contract = {"args": {"carry12": True}, "dependencies": {"source": "sha"}}
    ledger = {"human": {"files": {"toy": "sha"}, "split": {"pt": 1, "val": 1, "sft": 1}},
              "steps": {"human_fetch": {"sources": ["toy"]}, "decontam": {"rows_dropped": 2}}}
    data.save_human_cache(str(tmp_path), contract, ledger)
    return tmp_path, contract, ledger


def test_cached_retry_restores_exact_complete_provenance_without_fetch(cached, monkeypatch):
    out, contract, original = cached
    def no_fetch(*args, **kwargs):
        pytest.fail("cache resume must never fetch/recompute")
    monkeypatch.setattr(data, "sh", no_fetch)
    restored = {"steps": {"later": "present"}}
    data.restore_human_cache(str(out), contract, restored)
    assert restored["human"] == original["human"]
    assert restored["steps"]["human_fetch"] == original["steps"]["human_fetch"]
    assert restored["steps"]["decontam"] == original["steps"]["decontam"]
    assert restored["steps"]["later"] == "present"


@pytest.mark.parametrize("name", data.HUMAN_CACHE_FILES)
def test_tampered_or_missing_split_and_report_fail_closed(cached, name):
    out, contract, original = cached
    (out / name).write_text("changed")
    with pytest.raises(SystemExit, match="unverified human cache"):
        data.restore_human_cache(str(out), contract, {})
    (out / name).unlink()
    with pytest.raises(SystemExit, match="unverified human cache"):
        data.restore_human_cache(str(out), contract, {})


@pytest.mark.parametrize("key", ["args", "dependencies"])
def test_changed_semantics_or_source_hash_rejected(cached, key):
    out, contract, original = cached
    changed = copy.deepcopy(contract)
    changed[key]["changed"] = True
    with pytest.raises(SystemExit, match="contract differs"):
        data.restore_human_cache(str(out), changed, {})


@pytest.mark.parametrize("failure", ["missing", "truncated", "provenance"])
def test_legacy_or_damaged_receipt_preserves_existing_ledger(cached, failure):
    out, contract, original = cached
    ledger = out / "data.ledger.json"
    ledger.write_text(json.dumps(original))
    before = ledger.read_bytes()
    receipt = out / "human.cache.json"
    if failure == "missing":
        receipt.unlink()
    elif failure == "truncated":
        receipt.write_text("{")
    else:
        value = json.loads(receipt.read_text())
        value["steps"].pop("decontam")
        receipt.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="unverified human cache"):
        data.restore_human_cache(str(out), contract, {})
    assert ledger.read_bytes() == before


def test_actual_measure_only_retry_after_split_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "human" / "toy"
    source.mkdir(parents=True)
    rows = [{"messages": [{"role": "user", "content": f"cache test unique question {i}"},
                          {"role": "assistant", "content": f"cache test unique answer {i}"}],
             "uri": f"toy/{i}"} for i in range(60)]
    (source / "dialogues.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (source / "text.txt").write_text("toy source text")
    out = tmp_path / "out"
    args = ["--out", str(out), "--human-dir", str(tmp_path / "human"),
            "--sources", "toy", "--human-fetch", "", "--measure-only"]
    assert data.main(args) == 0
    ledger_path = out / "data.ledger.json"
    original = json.loads(ledger_path.read_text())
    ledger_path.unlink()  # models a failure after split checkpoint, before final ledger exists
    def no_recompute(*args, **kwargs):
        pytest.fail("cached retry attempted fresh decontamination")
    monkeypatch.setattr(data.decontam, "run", no_recompute)
    assert data.main(args) == 0
    assert json.loads(ledger_path.read_text()) == original
    before = ledger_path.read_bytes()
    (out / "human.cache.json").unlink()
    with pytest.raises(SystemExit, match="unverified human cache"):
        data.main(args)
    assert ledger_path.read_bytes() == before
