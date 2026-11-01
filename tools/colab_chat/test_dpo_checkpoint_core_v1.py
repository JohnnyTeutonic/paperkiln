import json
import shutil
import struct
from unittest.mock import patch
import pytest
import dpo_checkpoint_core_v1 as cp

SOURCE = "a" * 40


def tensor(path, items):
    header = {}
    payload = b""
    for name, value in items.items():
        start = len(payload)
        payload += struct.pack("<f", value)
        header[name] = {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def flat(path, step=25, value=0.25):
    path.mkdir(exist_ok=True, parents=True)
    tensor(path / "model.safetensors", {"w": value})
    tensor(
        path / "optim.safetensors", {"adamw.0.m.0": value / 2, "adamw.0.v.0": value / 4}
    )
    (path / "state.txt").write_text(
        str(step)
        + "\n"
        + json.dumps({"adamw_t": [step], "best_val": 1.0, "evals_flat": 0})
        + "\n"
    )
    (path / "events.jsonl").write_text(
        json.dumps({"event": "step", "step": step}) + "\n"
    )
    return path


def publish(path, step=25, value=0.25):
    flat(path, step, value)
    return cp.publish(path, step, 1, 2, 1, SOURCE, "b" * 64)


def test_complete_roundtrip_and_newer_priority(tmp_path):
    remote = tmp_path / "remote"
    publish(remote)
    home = tmp_path / "home"
    cp.copy_current(remote / ".dpo_checkpoint_v1", home)
    resumed = tmp_path / "resumed"
    m = cp.restore_flat(home, resumed, SOURCE)
    assert m["step"] == 25 and m["rng"]["draws"] == 50
    assert all(
        (remote / n).read_bytes() == (resumed / n).read_bytes() for n in cp.FILES
    )
    publish(remote, 50, 0.5)
    assert cp.copy_current(home, remote / ".dpo_checkpoint_v1") == {
        "kept_newer": True,
        "step": 50,
    }
    assert cp.current_unlocked(remote / ".dpo_checkpoint_v1")[1]["step"] == 50


def test_inflight_retention_does_not_block_new_checkpoints(tmp_path):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    pinned, manifest = cp.pin_inflight(root)
    for step in [50, 75, 100]:
        publish(tmp_path, step, step / 100)
        assert pinned.is_dir()
        assert len(list(root.glob("generation-*"))) == 2
        assert cp.current_unlocked(root)[1]["step"] == step
    cp.release_inflight(root, cp.sha(pinned / "manifest.json"))
    assert len(list(root.glob("generation-*"))) == 1
    assert not (root / "INFLIGHT").exists()


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
def test_interrupted_publication_preserves_committed(tmp_path, failure_at):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    before = cp.current_unlocked(root)[1]
    flat(tmp_path, 50, 0.5)
    original = shutil.copyfileobj
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("simulated interrupted write")
        return original(*args, **kwargs)

    with patch.object(cp.shutil, "copyfileobj", broken), pytest.raises(OSError):
        cp.publish(tmp_path, 50, 1, 2, 1, SOURCE, "b" * 64)
    assert cp.current_unlocked(root)[1] == before


@pytest.mark.parametrize(
    "target",
    ["model.safetensors", "optim.safetensors", "state.txt", "rng", "source", "step"],
)
def test_tampered_or_mixed_generation_refuses(tmp_path, target):
    publish(tmp_path)
    folder, manifest = cp.current_unlocked(tmp_path / ".dpo_checkpoint_v1")
    if target in cp.FILES:
        (folder / target).write_bytes(b"changed")
    else:
        if target == "rng":
            manifest["rng"]["draws"] += 1
        elif target == "source":
            manifest["source_commit"] = "wrong"
        else:
            manifest["step"] += 25
        (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        cp.current_unlocked(tmp_path / ".dpo_checkpoint_v1")


def test_same_step_different_bytes_refused(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    publish(a)
    publish(b, 25, 0.9)
    with pytest.raises(ValueError):
        cp.copy_current(a / ".dpo_checkpoint_v1", b / ".dpo_checkpoint_v1")


def test_missing_optimizer_is_not_cold_resume(tmp_path):
    flat(tmp_path)
    (tmp_path / "optim.safetensors").unlink()
    with pytest.raises(ValueError):
        cp.publish(tmp_path, 25, 1, 2, 1, SOURCE, "b" * 64)


def test_disk_preflight_preserves_previous(tmp_path):
    publish(tmp_path)
    before = cp.current_unlocked(tmp_path / ".dpo_checkpoint_v1")[1]
    flat(tmp_path, 50)
    with (
        patch.object(
            cp.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(1, 1, 0)
        ),
        pytest.raises(ValueError),
    ):
        cp.publish(tmp_path, 50, 1, 2, 1, SOURCE, "b" * 64)
    assert cp.current_unlocked(tmp_path / ".dpo_checkpoint_v1")[1] == before


def test_invalid_ack_retains_inflight(tmp_path):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    cp.pin_inflight(root)
    with pytest.raises(ValueError):
        cp.release_inflight(root, "wrong")
    assert (root / "INFLIGHT").is_symlink()


@pytest.mark.parametrize("operation", ["pin", "release"])
def test_slow_full_validation_does_not_hold_publisher_lock(tmp_path, operation):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    old, _ = cp.current_unlocked(root)
    digest = cp.sha(old / "manifest.json")
    if operation == "release":
        cp.pin_inflight(root)
    entered, allow, advanced = ctx.Event(), ctx.Event(), ctx.Event()

    def consumer():
        original = cp.validate_generation
        def slow(folder):
            entered.set()
            if not allow.wait(20):
                raise RuntimeError("test validation barrier timed out")
            return original(folder)
        cp.validate_generation = slow
        if operation == "pin":
            cp.pin_inflight(root)
        else:
            cp.release_inflight(root, digest)

    def producer():
        publish(tmp_path, 50, .5)
        publish(tmp_path, 75, .75)
        advanced.set()

    reader = ctx.Process(target=consumer)
    writer = ctx.Process(target=producer)
    reader.start()
    try:
        assert entered.wait(5)
        writer.start()
        # Controlled barrier, not a laptop storage-throughput claim: the
        # verifier cannot finish until we explicitly allow it below.
        assert advanced.wait(5)
        assert old.is_dir()
        assert cp.current_unlocked(root)[1]["step"] == 75
    finally:
        allow.set()
        reader.join(10)
        if writer.pid is not None:
            writer.join(10)
        for process in [reader, writer]:
            if process.pid is not None and process.is_alive():
                process.terminate(); process.join()
    assert reader.exitcode == 0 and writer.exitcode == 0
    if operation == "pin":
        assert old.is_dir()
        cp.release_inflight(root, digest)
    assert not old.exists()


@pytest.mark.parametrize("operation", ["pin", "release", "restore"])
def test_lease_and_resume_boundaries_still_detect_payload_corruption(tmp_path, operation):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    folder, _ = cp.current_unlocked(root)
    digest = cp.sha(folder / "manifest.json")
    if operation == "release":
        cp.pin_inflight(root)
    payload = folder / "model.safetensors"
    data = bytearray(payload.read_bytes()); data[-1] ^= 1; payload.write_bytes(data)
    with pytest.raises(ValueError, match="hash differs"):
        if operation == "pin":
            cp.pin_inflight(root)
        elif operation == "release":
            cp.release_inflight(root, digest)
        else:
            cp.restore_flat(root, tmp_path / "restore", SOURCE)
    if operation != "restore":
        assert (root / "INFLIGHT").is_symlink()


def test_phase_evidence_tracks_actual_durable_commit_not_heartbeat(tmp_path):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    rows = [json.loads(s) for s in (root / "phases.jsonl").read_text().splitlines()]
    names = [r["event"] for r in rows]
    assert names[0:2] == ["publish.lock_wait", "publish.lock_acquired"]
    assert names[-1] == "publish.current_durable"
    assert names.count("publish.copy_fsync_complete") == len(cp.FILES)
    assert names.count("publish.hash_complete") == len(cp.FILES)
    assert rows[-1]["generation"] == (root / "CURRENT").readlink().name
    assert all(type(r["monotonic"]) is float and type(r["unix"]) is float for r in rows)



def test_stale_ack_cannot_release_newer_lease(tmp_path):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    first, _ = cp.pin_inflight(root)
    stale = cp.sha(first / "manifest.json")
    cp.release_inflight(root, stale)
    publish(tmp_path, 50)
    second, _ = cp.pin_inflight(root)
    with pytest.raises(ValueError, match="acknowledgement differs"):
        cp.release_inflight(root, stale)
    assert (root / "INFLIGHT").readlink().name == second.name
    cp.release_inflight(root, cp.sha(second / "manifest.json"))


def test_phase_journal_refuses_symlink_destination(tmp_path):
    target = tmp_path / "untouched"; target.write_text("preserve")
    (tmp_path / "phases.jsonl").symlink_to(target)
    with pytest.raises(OSError):
        cp.phase(tmp_path, "test")
    assert target.read_text() == "preserve"



def test_publish_symlink_root_writes_nothing_outside(tmp_path):
    run = tmp_path / "run"; external = tmp_path / "external"
    flat(run); external.mkdir()
    (external / "sentinel").write_text("preserve")
    (run / ".dpo_checkpoint_v1").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cp.publish(run, 25, 1, 2, 1, SOURCE, "b" * 64)
    assert sorted(p.name for p in external.iterdir()) == ["sentinel"]
    assert (external / "sentinel").read_text() == "preserve"



def test_publisher_phase_stdout_fallback_is_telemetry(tmp_path, capsys):
    publish(tmp_path)
    lines = [s.removeprefix("DPO_CHECKPOINT_PHASE ") for s in capsys.readouterr().out.splitlines() if s.startswith("DPO_CHECKPOINT_PHASE ")]
    rows = [json.loads(s) for s in lines]
    assert rows[-1]["event"] == "publish.current_durable"
    assert cp.current_unlocked(tmp_path / ".dpo_checkpoint_v1")[1]["step"] == 25


def test_phase_syncs_only_at_durable_commit_and_halt(tmp_path, monkeypatch):
    import os
    original = cp.os.fsync
    synced = []
    def observed(fd):
        if os.readlink(f"/proc/self/fd/{fd}").endswith("phases.jsonl"):
            synced.append(fd)
        return original(fd)
    monkeypatch.setattr(cp.os, "fsync", observed)
    publish(tmp_path)
    assert len(synced) == 1
    root = tmp_path / ".dpo_checkpoint_v1"
    cp.phase(root, "manager.active_start")
    cp.phase(root, "pin.validation_start")
    assert len(synced) == 1
    cp.phase(root, "manager.gap_halt")
    assert len(synced) == 2



@pytest.mark.parametrize("fault", ["open", "write", "fsync"])
def test_diagnostic_io_failure_never_invalidates_publication(tmp_path, monkeypatch, capsys, fault):
    import os, errno
    original = getattr(cp.os, fault)
    def broken(first, *args, **kwargs):
        target = str(first) if fault == "open" else os.readlink(f"/proc/self/fd/{first}")
        if target.endswith("phases.jsonl"):
            raise OSError(errno.EIO, "injected diagnostic-only failure")
        return original(first, *args, **kwargs)
    monkeypatch.setattr(cp.os, fault, broken)
    publish(tmp_path)
    assert cp.current_unlocked(tmp_path / ".dpo_checkpoint_v1")[1]["step"] == 25
    assert '"status": "WRITE_FAILED"' in capsys.readouterr().out



@pytest.mark.parametrize("operation", ["pin", "release"])
def test_corrupt_retained_generation_raises_distinct_fatal_error(tmp_path, operation):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    folder, _ = cp.current_unlocked(root)
    digest = cp.sha(folder / "manifest.json")
    if operation == "release":
        cp.pin_inflight(root)
    target = folder / "model.safetensors"
    raw = bytearray(target.read_bytes()); raw[-1] ^= 1; target.write_bytes(raw)
    with pytest.raises(cp.PinnedGenerationValidationError, match="PINNED_GENERATION_VALIDATION_FAILED"):
        if operation == "pin":
            cp.pin_inflight(root)
        else:
            cp.release_inflight(root, digest)
    assert (root / "INFLIGHT").readlink().name == folder.name



@pytest.mark.parametrize("operation", ["pin", "release"])
def test_corrupt_retained_metadata_has_distinct_fatal_error(tmp_path, operation):
    publish(tmp_path)
    root = tmp_path / ".dpo_checkpoint_v1"
    folder, _ = cp.pin_inflight(root)
    digest = cp.sha(folder / "manifest.json")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["rng"]["draws"] += 1
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(cp.PinnedGenerationValidationError, match="retained metadata"):
        if operation == "pin":
            cp.pin_inflight(root)
        else:
            cp.release_inflight(root, digest)
    assert (root / "INFLIGHT").is_symlink()
