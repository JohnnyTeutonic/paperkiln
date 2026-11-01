import shutil
import pytest
import dpo_checkpoint_core_v1 as cp
import dpo_checkpoint_transport_v1 as tr
from test_dpo_checkpoint_core_v1 import flat

SOURCE = "a" * 40


def job():
    return {
        "name": "pk_dpo_v1",
        "dpo_full_checkpoint_v1": {
            "every_steps": 10,
            "max_gap_seconds": 180,
            "steps": 100,
            "source_commit": SOURCE,
            "seed": 1,
            "batch": 8,
            "accum": 1,
            "contract_sha256": "b" * 64,
        },
    }


def ready(base, step):
    out = base / job()["name"]
    flat(out, step, step / 100)
    cp.publish(out, step, 1, 8, 1, SOURCE, "b" * 64)
    return out


def uploaded(home, upload):
    files, leases = tr.home_uploads([job()], home)
    upload.mkdir(parents=True)
    for name, source in files:
        shutil.copyfile(source, upload / name)
    return leases


def test_complete_transport_preserves_optimizer_rng_and_events(tmp_path):
    home, remote, upload = (tmp_path / x for x in ["home", "remote", "upload"])
    original = ready(home, 25)
    leases = uploaded(home, upload)
    assert tr.recover_remote([job()], remote, upload)[0]["step"] == 25
    actual = remote / job()["name"]
    assert all(
        (actual / n).read_bytes() == (original / n).read_bytes() for n in cp.FILES
    )
    assert cp.current_unlocked(tr.root_for(remote, job()))[1]["rng"]["draws"] == 200
    tr.release_uploads(leases)
    assert not (tr.root_for(home, job()) / "INFLIGHT").exists()


def test_older_home_never_overwrites_newer_remote(tmp_path):
    home, remote, upload = (tmp_path / x for x in ["home", "remote", "upload"])
    ready(home, 25)
    newer = ready(remote, 50)
    expected = (newer / "model.safetensors").read_bytes()
    uploaded(home, upload)
    assert tr.recover_remote([job()], remote, upload)[0]["kept_newer"]
    assert (newer / "model.safetensors").read_bytes() == expected
    assert cp.current_unlocked(tr.root_for(remote, job()))[1]["step"] == 50


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_optimizer",
        "changed_optimizer",
        "manifest_symlink",
        "source",
        "flat_cold",
    ],
)
def test_incomplete_or_mixed_transport_refused(tmp_path, mutation):
    home, remote, upload = (tmp_path / x for x in ["home", "remote", "upload"])
    ready(home, 25)
    uploaded(home, upload)
    if mutation == "flat_cold":
        shutil.rmtree(upload)
        upload.mkdir()
        flat(remote / job()["name"], 25)
    elif mutation == "manifest_symlink":
        p = next(upload.glob("*.manifest.json"))
        target = tmp_path / "saved.json"
        p.rename(target)
        p.symlink_to(target)
    elif mutation == "source":
        j = job()
        j["dpo_full_checkpoint_v1"]["source_commit"] = "b" * 40
        with pytest.raises(ValueError):
            tr.recover_remote([j], remote, upload)
        return
    else:
        p = next(upload.glob("*.optim.safetensors"))
        if mutation == "missing_optimizer":
            p.unlink()
        else:
            p.write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError)):
        tr.recover_remote([job()], remote, upload)
    assert not (tr.root_for(remote, job()) / "CURRENT").exists()


def test_gap_monitor_active_only_and_reset(tmp_path):
    import dpo_checkpoint_manager_v1 as op

    j = job()
    folder = str((tmp_path / j["name"]).resolve())
    starts = {}
    op.gap_check([j], tmp_path, set(), starts, 0, 1000)
    assert not starts
    op.gap_check([j], tmp_path, {folder}, starts, 0, 1000)
    with pytest.raises(ValueError, match="gap"):
        op.gap_check([j], tmp_path, {folder}, starts, 0, 1181)
    ready(tmp_path, 25)
    op.gap_check([j], tmp_path, {folder}, starts, 0, 1182)
    op.gap_check([j], tmp_path, {folder}, starts, 0, 1362)
    with pytest.raises(ValueError, match="gap"):
        op.gap_check([j], tmp_path, {folder}, starts, 0, 1363)


def test_final_home_requires_complete_final_optimizer_bundle(tmp_path):
    assert tr.final_home_ready(job(), tmp_path) is False
    ready(tmp_path, 25)
    assert tr.final_home_ready(job(), tmp_path) is False
    ready(tmp_path, 100)
    assert tr.final_home_ready(job(), tmp_path) is True
    current, _ = cp.current_unlocked(tr.root_for(tmp_path, job()))
    (current / "optim.safetensors").unlink()
    with pytest.raises(ValueError):
        tr.final_home_ready(job(), tmp_path)



@pytest.mark.parametrize("telemetry", ["ok", "download_failed", "partial"])
def test_real_relay_snapshots_telemetry_after_verified_import_and_release(tmp_path, monkeypatch, telemetry):
    import contextlib, io, json, types
    home, remote = tmp_path / "home", tmp_path / "remote"
    ready(remote, 25)
    original_root_for = tr.root_for
    remote_root = original_root_for(remote, job())
    home_root = original_root_for(home, job())
    home_root.mkdir(parents=True)
    previous = b'{"event":"previous"}\n'
    (home_root / "phases.snapshot.jsonl").write_bytes(previous)
    monkeypatch.setattr(tr, "root_for", lambda base, j: original_root_for(remote if str(base) == "/content/chat" else base, j))

    def rpc(exec_py, expression):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(expression, {"cp": cp, "json": json})
        return json.loads([s[len("S_FULL_RPC "):] for s in output.getvalue().splitlines() if s.startswith("S_FULL_RPC ")][-1])

    monkeypatch.setattr(tr, "rpc", rpc)
    captures = []
    def download(source, target, timeout):
        if source.endswith("phases.jsonl"):
            # Diagnostic capture must happen after the integrity boundary and
            # remote acknowledgement, never on behalf of checkpoint success.
            assert cp.current_unlocked(home_root)[1]["step"] == 25
            assert not (remote_root / "INFLIGHT").exists()
            captures.append(source)
            if telemetry == "download_failed":
                return 1
            if telemetry == "partial":
                from pathlib import Path
                Path(target).write_bytes(b'{"event":')
                return 0
        shutil.copyfile(source, target)
        return 0

    saved = []
    st = types.SimpleNamespace(d={}, save=lambda: saved.append(True))
    assert tr.relay(None, download, st, job(), home)
    record = st.d["dpo_full_checkpoint_v1"][job()["name"]]
    assert record["full_optimizer"] is True and record["step"] == 25 and saved
    assert captures
    assert record["phase_snapshot"]["diagnostic_only"] is True
    assert record["phase_snapshot"]["status"] == ("CAPTURED" if telemetry == "ok" else "CAPTURE_FAILED")
    if telemetry == "ok":
        rows = [json.loads(s) for s in (home_root / "phases.snapshot.jsonl").read_text().splitlines()]
        assert any(r["event"] == "publish.current_durable" for r in rows)
        assert rows[-1]["event"] == "release.complete"
        assert not (home_root / "phases.snapshot.tmp").exists()
    else:
        assert (home_root / "phases.snapshot.jsonl").read_bytes() == previous
    restored = tmp_path / "restored"
    assert cp.restore_flat(home_root, restored, SOURCE)["rng"]["draws"] == 200
    assert all((restored / n).read_bytes() == (remote / job()["name"] / n).read_bytes() for n in cp.FILES)
