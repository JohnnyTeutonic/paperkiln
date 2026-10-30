"""Filesystem regressions for extension provenance and the registered merge."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("extension_check", Path(__file__).with_name("check_extension_results.py"))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def ready(tmp_path):
    for _, ref, campaign, post, cell in check.CELLS:
        write(tmp_path / ref / "data/data.ledger.json", {"hash": "same", "elapsed": 1})
        for seed in (4, 5):
            folder = tmp_path / campaign.format(seed=seed)
            write(folder / "data/data.ledger.json", {"hash": "same", "elapsed": 2})
            write(folder / post.format(seed=seed) / "summary.json", {"cell": cell, "seed": seed, "primary": -.1})
    return tmp_path


def test_ready_and_idempotent_registered_merge(ready):
    receipt = check.audit(ready)
    assert receipt["ready"] and len(receipt["ledgers"]) == 6
    targets = check.merge_20m(ready, receipt)
    assert len(targets) == 2
    assert check.merge_20m(ready, receipt) == targets
    for seed in (4, 5):
        source = ready / f"e5_s35_r{seed}/post_e5_s35_r{seed}/summary.json"
        assert source.read_bytes() == (ready / f"e5/post_e5_s35_r{seed}/summary.json").read_bytes()


def test_pending_blocks_all_merge(ready):
    (ready / "e1_60m_L8_r5/post_e1_60m_L8_r5/summary.json").unlink()
    receipt = check.audit(ready)
    assert not receipt["ready"] and receipt["pending"]
    with pytest.raises(ValueError, match="pending"):
        check.merge_20m(ready, receipt)
    assert not (ready / "e5/post_e5_s35_r4/summary.json").exists()


def test_ledger_hash_mismatch_blocks_merge(ready):
    write(ready / "e5_s35_r4/data/data.ledger.json", {"hash": "changed", "elapsed": 2})
    receipt = check.audit(ready)
    assert not receipt["ready"] and receipt["errors"]
    with pytest.raises(ValueError):
        check.merge_20m(ready, receipt)


@pytest.mark.parametrize("field,value", [("seed", 3), ("seed", True), ("cell", "L8"), ("primary", None)])
def test_wrong_or_incomplete_report_rejected(ready, field, value):
    p = ready / "e5_s35_r4/post_e5_s35_r4/summary.json"
    s = json.loads(p.read_text())
    s[field] = value
    write(p, s)
    assert not check.audit(ready)["ready"]


def test_existing_different_report_is_preserved(ready):
    target = ready / "e5/post_e5_s35_r5/summary.json"
    write(target, {"original": "preserve"})
    with pytest.raises(ValueError, match="different"):
        check.merge_20m(ready, check.audit(ready))
    assert json.loads(target.read_text()) == {"original": "preserve"}
    assert not (ready / "e5/post_e5_s35_r4/summary.json").exists()


def test_changed_source_after_audit_rejected(ready):
    receipt = check.audit(ready)
    write(ready / "e5_s35_r4/post_e5_s35_r4/summary.json", {"modified": True})
    with pytest.raises(ValueError, match="source changed"):
        check.merge_20m(ready, receipt)


def test_receipt_output_is_immutable(ready):
    out = ready / "receipt.json"
    assert check.main(["--art", str(ready), "--out", str(out)]) == 0
    with pytest.raises(SystemExit):
        check.main(["--art", str(ready), "--out", str(out)])
