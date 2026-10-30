"""Durable parity evidence and independent offline replay, without model downloads."""

import builtins
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hf_export_verify as verify


def fixture():
    hf = np.array([[9, 3, 2, 1], [9, 0, 4, 2]], dtype=np.float32)
    native = hf.copy()
    native[0, 2] += np.float32(0.00001)
    ids = [1, 2]
    error = verify.assess(ids, {"argmax": ids}, hf, native, 2e-4, 2e-4)
    receipt = dict(
        status="HF-PARITY-OK",
        continuation_ids=ids,
        mtstudio_argmax=ids,
        max_abs_logit_error=error,
        atol=2e-4,
        rtol=2e-4,
    )
    return receipt, hf, native


def publish(tmp_path):
    receipt, hf, native = fixture()
    target = tmp_path / "parity.json"
    verify.publish_evidence(target, receipt, hf, native, [0])
    return target


def test_offline_replay_recomputes_raw_numeric_and_masked_ids_without_models(
    tmp_path, monkeypatch
):
    target = publish(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "tokenizers", "safetensors"}:
            raise AssertionError("offline replay imported model dependencies")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    got = verify.replay_evidence(target)
    assert got["status"] == "HF-PARITY-REPLAY-OK" and got["rows"] == 2
    assert got["max_abs_logit_error"] == fixture()[0]["max_abs_logit_error"]


def test_receipt_and_evidence_cannot_be_overwritten(tmp_path):
    target = publish(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        verify.publish_evidence(target, *fixture(), [0])
    assert all(p.read_bytes() == data for p, data in before.items())
    target.unlink()
    with pytest.raises(FileExistsError):
        verify.publish_evidence(target, *fixture(), [0])
    assert (
        before[next(p for p in before if p.suffix == ".npz")]
        == next(tmp_path.rglob("*.npz")).read_bytes()
    )


def test_failed_publication_cleans_only_owned_new_files(tmp_path):
    target = tmp_path / "parity.json"
    sentinel = tmp_path / "unrelated"
    sentinel.write_bytes(b"keep")
    receipt, hf, native = fixture()
    receipt["invalid"] = object()
    with pytest.raises(TypeError):
        verify.publish_evidence(target, receipt, hf, native, [0])
    assert (
        not target.exists() and not target.with_name(target.name + ".evidence").exists()
    )
    assert sentinel.read_bytes() == b"keep"


def rewrite(target, change):
    receipt = json.loads(target.read_text())
    change(receipt)
    target.write_text(json.dumps(receipt))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAIL"),
        ("max_abs_logit_error", 0),
        ("continuation_ids", [0, 2]),
        ("mtstudio_argmax", [2, 2]),
        ("atol", float("inf")),
    ],
)
def test_inconsistent_receipt_refuses(tmp_path, field, value):
    target = publish(tmp_path)
    rewrite(target, lambda r: r.update({field: value}))
    with pytest.raises(ValueError):
        verify.replay_evidence(target)


def test_raw_corruption_refuses_before_loading(tmp_path):
    target = publish(tmp_path)
    raw = next(tmp_path.rglob("*.npz"))
    with raw.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash/size"):
        verify.replay_evidence(target)


def test_evidence_traversal_refuses(tmp_path):
    target = publish(tmp_path)
    rewrite(target, lambda r: r["evidence"].update(file="../outside.npz"))
    with pytest.raises(ValueError, match="invalid retained"):
        verify.replay_evidence(target)


@pytest.mark.parametrize(
    "change",
    ["native_ids", "banned_range", "banned_all", "float64", "nonfinite", "extra"],
)
def test_raw_evidence_semantics_rechecked_even_with_updated_manifest(tmp_path, change):
    target = publish(tmp_path)
    raw = next(tmp_path.rglob("*.npz"))
    with np.load(raw, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    if change == "native_ids":
        arrays["mt_logits"][0, 3] = 10
    if change == "banned_range":
        arrays["banned"] = np.array([99])
    if change == "banned_all":
        arrays["banned"] = np.arange(4)
    if change == "float64":
        arrays["hf_logits"] = arrays["hf_logits"].astype(np.float64)
    if change == "nonfinite":
        arrays["hf_logits"][0, 0] = np.inf
    if change == "extra":
        arrays["opaque"] = np.array([0])
    np.savez(raw, **arrays)
    rewrite(
        target,
        lambda r: r["evidence"].update(
            bytes=raw.stat().st_size,
            sha256=verify.hashlib.sha256(raw.read_bytes()).hexdigest(),
        ),
    )
    with pytest.raises(ValueError):
        verify.replay_evidence(target)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_assess_refuses_invalid_tolerance(value):
    _, hf, native = fixture()
    with pytest.raises(ValueError, match="tolerances"):
        verify.assess([1, 2], {"argmax": [1, 2]}, hf, native, value, 0)


def test_assess_refuses_hf_nonfinite_and_empty_rows():
    _, hf, native = fixture()
    hf[0, 0] = np.inf
    with pytest.raises(ValueError, match="invalid output"):
        verify.assess([1, 2], {"argmax": [1, 2]}, hf, native, 1, 1)
    empty = np.zeros((0, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="invalid output"):
        verify.assess([], {"argmax": []}, empty, empty, 1, 1)


def test_publication_exposes_only_complete_json(tmp_path, monkeypatch):
    target = tmp_path / "parity.json"
    real_link = verify.os.link
    observed = []

    def inspect_link(source, destination):
        assert not Path(destination).exists()
        value = json.loads(Path(source).read_text())
        assert value["status"] == "HF-PARITY-OK"
        assert next(tmp_path.rglob("*.npz")).is_file()
        real_link(source, destination)
        observed.append(verify.replay_evidence(destination)["rows"])

    monkeypatch.setattr(verify.os, "link", inspect_link)
    verify.publish_evidence(target, *fixture(), [0])
    assert observed == [2]
    assert not list(tmp_path.glob("*.incoming"))


def test_publication_race_preserves_foreign_receipt(tmp_path, monkeypatch):
    target = tmp_path / "parity.json"

    def race(source, destination):
        Path(destination).write_bytes(b"other publisher")
        raise FileExistsError(destination)

    monkeypatch.setattr(verify.os, "link", race)
    with pytest.raises(FileExistsError):
        verify.publish_evidence(target, *fixture(), [0])
    assert target.read_bytes() == b"other publisher"
    assert not target.with_name(target.name + ".evidence").exists()
    assert not list(tmp_path.glob("*.incoming"))


def test_provenance_parses_exact_captured_bytes_and_detects_later_change(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    events = tmp_path / "events.jsonl"
    events.write_text(
        "build note\n"
        + json.dumps({"event": "start", "name": "original"})
        + "\n"
        + json.dumps({"event": "model", "family": "flex"})
        + "\n"
    )
    (hf / "export_receipt.json").write_text(json.dumps({"events": str(events)}))
    model, source, event_path, before = verify.snapshot_provenance(hf, tmp_path)
    assert (
        model["family"] == "flex"
        and source.name == "original.safetensors"
        and event_path == events
    )
    assert verify.hashes([hf / "export_receipt.json", events]) == before
    events.write_text(events.read_text().replace("original", "replacement"))
    assert verify.hashes([hf / "export_receipt.json", events]) != before


def test_raw_greedy_is_checked_separately_from_numeric_agreement(tmp_path):
    target = publish(tmp_path)
    raw = next(tmp_path.rglob("*.npz"))
    with np.load(raw, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    for name in ("hf_logits", "mt_logits"):
        arrays[name][1, 3] = 20
    np.savez(raw, **arrays)
    rewrite(
        target,
        lambda r: r["evidence"].update(
            bytes=raw.stat().st_size,
            sha256=verify.hashlib.sha256(raw.read_bytes()).hexdigest(),
        ),
    )
    with pytest.raises(ValueError, match="raw masked greedy"):
        verify.replay_evidence(target)


def test_native_bool_ids_are_not_integers(tmp_path):
    target = publish(tmp_path)
    rewrite(target, lambda r: r.update(mtstudio_argmax=[True, 2]))
    with pytest.raises(ValueError, match="invalid retained token"):
        verify.replay_evidence(target)


def test_interrupt_after_atomic_link_keeps_complete_receipt_and_rows(
    tmp_path, monkeypatch
):
    real_link = verify.os.link

    def interrupt(source, destination):
        real_link(source, destination)
        raise KeyboardInterrupt()

    monkeypatch.setattr(verify.os, "link", interrupt)
    target = tmp_path / "parity.json"
    with pytest.raises(KeyboardInterrupt):
        verify.publish_evidence(target, *fixture(), [0])
    assert verify.replay_evidence(target)["rows"] == 2
    assert not list(tmp_path.glob("*.incoming"))
