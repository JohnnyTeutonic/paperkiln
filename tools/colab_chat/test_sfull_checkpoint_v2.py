import json
import shutil
import struct
from unittest.mock import patch
import pytest
import sfull_checkpoint_v2 as cp

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
    return cp.publish(path, step, 1, 2, 1, SOURCE)


def test_complete_roundtrip_and_newer_priority(tmp_path):
    remote = tmp_path / "remote"
    publish(remote)
    home = tmp_path / "home"
    cp.copy_current(remote / ".checkpoint_v2", home)
    resumed = tmp_path / "resumed"
    m = cp.restore_flat(home, resumed, SOURCE)
    assert m["step"] == 25 and m["rng"]["draws"] == 50
    assert all(
        (remote / n).read_bytes() == (resumed / n).read_bytes() for n in cp.FILES
    )
    publish(remote, 50, 0.5)
    assert cp.copy_current(home, remote / ".checkpoint_v2") == {
        "kept_newer": True,
        "step": 50,
    }
    assert cp.current_unlocked(remote / ".checkpoint_v2")[1]["step"] == 50


def test_inflight_retention_does_not_block_new_checkpoints(tmp_path):
    publish(tmp_path)
    root = tmp_path / ".checkpoint_v2"
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
    root = tmp_path / ".checkpoint_v2"
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
        cp.publish(tmp_path, 50, 1, 2, 1, SOURCE)
    assert cp.current_unlocked(root)[1] == before


@pytest.mark.parametrize(
    "target",
    ["model.safetensors", "optim.safetensors", "state.txt", "rng", "source", "step"],
)
def test_tampered_or_mixed_generation_refuses(tmp_path, target):
    publish(tmp_path)
    folder, manifest = cp.current_unlocked(tmp_path / ".checkpoint_v2")
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
        cp.current_unlocked(tmp_path / ".checkpoint_v2")


def test_same_step_different_bytes_refused(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    publish(a)
    publish(b, 25, 0.9)
    with pytest.raises(ValueError):
        cp.copy_current(a / ".checkpoint_v2", b / ".checkpoint_v2")


def test_missing_optimizer_is_not_cold_resume(tmp_path):
    flat(tmp_path)
    (tmp_path / "optim.safetensors").unlink()
    with pytest.raises(ValueError):
        cp.publish(tmp_path, 25, 1, 2, 1, SOURCE)


def test_disk_preflight_preserves_previous(tmp_path):
    publish(tmp_path)
    before = cp.current_unlocked(tmp_path / ".checkpoint_v2")[1]
    flat(tmp_path, 50)
    with (
        patch.object(
            cp.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(1, 1, 0)
        ),
        pytest.raises(ValueError),
    ):
        cp.publish(tmp_path, 50, 1, 2, 1, SOURCE)
    assert cp.current_unlocked(tmp_path / ".checkpoint_v2")[1] == before


def test_invalid_ack_retains_inflight(tmp_path):
    publish(tmp_path)
    root = tmp_path / ".checkpoint_v2"
    cp.pin_inflight(root)
    with pytest.raises(ValueError):
        cp.release_inflight(root, "wrong")
    assert (root / "INFLIGHT").is_symlink()
