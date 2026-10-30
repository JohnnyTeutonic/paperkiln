"""Transactional HF publication: failures cannot silently mix old and new artifacts."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hf_export as export  # noqa: E402


def write_set(directory, version):
    for name in export.EXPORT_FILES:
        (Path(directory) / name).write_bytes((version + ":" + name).encode())


def snapshot(directory):
    return {p.name: p.read_bytes() for p in Path(directory).iterdir()}


def test_staging_failure_preserves_previous_export_and_unrelated_file(tmp_path):
    target = tmp_path / "hf"
    target.mkdir()
    write_set(target, "old")
    (target / "README.md").write_bytes(b"user model card")
    before = snapshot(target)
    with pytest.raises(RuntimeError, match="injected"):
        with export.staged_export(target, overwrite=True) as stage:
            write_set(stage, "new")
            raise RuntimeError("injected tokenizer/weight/config failure")
    assert snapshot(target) == before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure_index", [1, 4, 6, 9, 10])
def test_publish_move_failure_rolls_back_all_managed_bytes(tmp_path, monkeypatch, failure_index):
    target = tmp_path / "hf"
    target.mkdir()
    write_set(target, "old")
    (target / "README.md").write_bytes(b"keep")
    before = snapshot(target)
    original = export.os.replace
    calls = []
    def replace(src, dst):
        calls.append((src, dst))
        if len(calls) == failure_index:
            raise OSError("injected publish failure")
        return original(src, dst)
    monkeypatch.setattr(export.os, "replace", replace)
    with pytest.raises(OSError, match="injected publish"):
        with export.staged_export(target, overwrite=True) as stage:
            write_set(stage, "new")
    assert snapshot(target) == before
    assert list(tmp_path.iterdir()) == [target]


def test_successful_explicit_overwrite_publishes_complete_set_preserves_unrelated(tmp_path):
    target = tmp_path / "hf"
    target.mkdir()
    write_set(target, "old")
    (target / "README.md").write_bytes(b"keep")
    with export.staged_export(target, overwrite=True) as stage:
        write_set(stage, "new")
    assert snapshot(target) == {**{n: ("new:" + n).encode() for n in export.EXPORT_FILES}, "README.md": b"keep"}
    assert list(tmp_path.iterdir()) == [target]


def test_existing_export_requires_explicit_overwrite(tmp_path):
    target = tmp_path / "hf"
    target.mkdir()
    write_set(target, "old")
    before = snapshot(target)
    with pytest.raises(ValueError, match="overwrite"):
        with export.staged_export(target):
            pytest.fail("must refuse before building")
    assert snapshot(target) == before


@pytest.mark.parametrize("stale", ["model-00001-of-00002.safetensors", "model.safetensors.index.json",
                                  "pytorch_model.bin", "generation_config.json", "added_tokens.json"])
def test_stale_loader_artifacts_are_not_silently_mixed_or_deleted(tmp_path, stale):
    target = tmp_path / "hf"
    target.mkdir()
    (target / stale).write_bytes(b"preserve incompatible input")
    with pytest.raises(ValueError, match="incompatible"):
        with export.staged_export(target, overwrite=True):
            pytest.fail("must refuse")
    assert snapshot(target) == {stale: b"preserve incompatible input"}


def test_new_export_only_appears_when_complete_and_missing_file_refuses(tmp_path):
    target = tmp_path / "hf"
    with pytest.raises(ValueError, match="incomplete"):
        with export.staged_export(target) as stage:
            (Path(stage) / "config.json").write_text("{}")
            assert not target.exists()
    assert not target.exists()
    with export.staged_export(target) as stage:
        write_set(stage, "new")
        assert not target.exists()
    assert set(snapshot(target)) == export.EXPORT_FILES


def test_link_target_or_managed_file_refused_without_touching_link_destination(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "secret").write_bytes(b"unchanged")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlinks"):
        with export.staged_export(link, overwrite=True):
            pytest.fail("link target accepted")
    target = tmp_path / "hf"
    target.mkdir()
    (target / "config.json").symlink_to(real / "secret")
    with pytest.raises(ValueError, match="regular file"):
        with export.staged_export(target, overwrite=True):
            pytest.fail("managed link accepted")
    assert (real / "secret").read_bytes() == b"unchanged"


def test_simultaneous_export_refuses_without_disturbing_first_stage(tmp_path):
    target = tmp_path / "hf"
    with export.staged_export(target) as stage:
        write_set(stage, "first")
        with pytest.raises(FileExistsError):
            with export.staged_export(target):
                pytest.fail("parallel writer accepted")
    assert (target / "config.json").read_bytes() == b"first:config.json"


def test_failed_rollback_keeps_recovery_bytes_and_writer_lock(tmp_path, monkeypatch):
    target = tmp_path / "hf"
    target.mkdir()
    write_set(target, "old")
    original = export.os.replace
    calls = []
    def replace(src, dst):
        calls.append((src, dst))
        if len(calls) >= 7:  # one new file published, then publication and rollback fail
            raise OSError("injected persistent disk error")
        return original(src, dst)
    monkeypatch.setattr(export.os, "replace", replace)
    with pytest.raises(RuntimeError, match="preserve recovery files"):
        with export.staged_export(target, overwrite=True) as stage:
            write_set(stage, "new")
    retained = list(tmp_path.glob(".hf.export-*"))
    assert len(retained) == 1
    assert (retained[0] / "backup/model.safetensors").read_bytes() == b"old:model.safetensors"
    assert (tmp_path / ".hf.export.lock").exists()
    with pytest.raises(FileExistsError):
        with export.staged_export(target, overwrite=True):
            pytest.fail("must block until operator recovers previous export")


def test_cli_failed_reexport_preserves_previous_export_byte_for_byte(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    target = run / "hf"
    target.mkdir()
    write_set(target, "old")
    (target / "README.md").write_bytes(b"user card")
    before = snapshot(target)
    def broken_export(args):
        (Path(args.hf_dir) / "tokenizer.json").write_bytes(b"partial new tokenizer")
        raise ValueError("injected mapping validation failure")
    monkeypatch.setattr(export, "export_model", broken_export)
    monkeypatch.setattr(sys, "argv", ["hf_export.py", str(run), "--overwrite"])
    with pytest.raises(ValueError, match="mapping validation"):
        export.main()
    assert snapshot(target) == before
    assert list(run.iterdir()) == [target]
