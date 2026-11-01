import os
import signal
import subprocess
import pytest
import sfull_preboot_recovery_v1 as recovery


def run(*args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    run("git", "-C", str(origin), "init", "-q")
    run("git", "-C", str(origin), "config", "user.email", "fixture@example.invalid")
    run("git", "-C", str(origin), "config", "user.name", "Fixture")
    (origin / "source.txt").write_text("original\n")
    run("git", "-C", str(origin), "add", "source.txt")
    run("git", "-C", str(origin), "commit", "-qm", "original")
    first = run("git", "-C", str(origin), "rev-parse", "HEAD")
    (origin / "source.txt").write_text("next\n")
    run("git", "-C", str(origin), "commit", "-qam", "next")
    second = run("git", "-C", str(origin), "rev-parse", "HEAD")
    root = tmp_path / "content"
    root.mkdir()
    digest, _ = recovery.archive_identity(origin, first)
    return root, origin, first, second, digest


def clone(repo, head=None):
    root, origin, first, second, digest = repo
    target = root / "sfull_control"
    run("git", "clone", "-q", str(origin), str(target))
    run("git", "-C", str(target), "checkout", "-q", "--detach", head or first)
    return target


def recover(repo):
    root, origin, first, second, digest = repo
    return recovery.recover(root, str(origin), first, first, digest)


def test_stale_index_lock_preserved_not_unlinked(repo):
    target = clone(repo)
    (target / ".git/index.lock").write_text("interrupted")
    assert recover(repo)["status"] == "SOURCE_RECOVERED_ORIGINAL_PATH"
    assert (
        repo[0] / ".sfull_preboot_v1/quarantined-control/.git/index.lock"
    ).read_text() == "interrupted"
    assert not (target / ".git/index.lock").exists()
    assert (target / "source.txt").read_text() == "original\n"
    assert not (repo[0] / "pk").exists()


def test_unborn_interrupted_checkout_is_quarantined(repo):
    root, origin, first, *_ = repo
    target = root / "sfull_control"
    target.mkdir()
    run("git", "-C", str(target), "init", "-q")
    run("git", "-C", str(target), "remote", "add", "origin", str(origin))
    run("git", "-C", str(target), "fetch", "-q", "origin", first)
    # A real interrupted Git subprocess leaves its lock and incomplete source behind.
    script = target / "interrupt.sh"
    script.write_text("printf held > .git/index.lock; kill -STOP $$; sleep 30\n")
    process = subprocess.Popen(
        [
            "git",
            "-C",
            str(target),
            "-c",
            "alias.interrupted=!sh ./interrupt.sh",
            "interrupted",
        ],
        start_new_session=True,
    )
    try:
        import time

        end = time.monotonic() + 3
        while not (target / ".git/index.lock").exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert (target / ".git/index.lock").exists()
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        assert recover(repo)["previous_control_state"] == "unborn"
        assert (root / ".sfull_preboot_v1/quarantined-control/interrupt.sh").exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_real_active_git_owner_refused(repo):
    target = clone(repo)
    process = subprocess.Popen(
        ["git", "-C", str(target), "hash-object", "--stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        with pytest.raises(recovery.ActiveOwner):
            recover(repo)
        assert not (repo[0] / ".sfull_preboot_v1").exists()
    finally:
        process.terminate()
        process.communicate()


@pytest.mark.parametrize(
    "mutation", ["dirty", "wrong_revision", "wrong_archive", "numeric_dirty"]
)
def test_source_mismatch_never_replaced(repo, mutation):
    root, origin, first, second, digest = repo
    target = clone(repo, second if mutation == "wrong_revision" else first)
    if mutation == "dirty":
        (target / "source.txt").write_text("author work")
    if mutation == "numeric_dirty":
        run("git", "clone", "-q", str(origin), str(root / "pk"))
        run("git", "-C", str(root / "pk"), "checkout", "-q", "--detach", first)
        (root / "pk/source.txt").write_text("author work")
    args = (
        root,
        str(origin),
        first,
        first,
        "0" * 64 if mutation == "wrong_archive" else digest,
    )
    with pytest.raises(ValueError):
        recovery.recover(*args)
    assert (
        target.exists()
        and not (root / ".sfull_preboot_v1/quarantined-control").exists()
    )


def test_existing_lock_owner_refused(repo):
    import fcntl

    chat = repo[0] / "chat"
    chat.mkdir()
    with (chat / "sfull_preboot_v1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(recovery.ActiveOwner):
            recover(repo)


def test_clean_retry_does_not_replace_or_duplicate_sources(repo):
    target = clone(repo)
    inode = target.stat().st_ino
    assert recover(repo)["status"] == "ALREADY_CLEAN"
    assert target.stat().st_ino == inode


def test_git_dir_owner_from_outside_checkout_refused(repo):
    target = clone(repo)
    process = subprocess.Popen(
        ["git", "--git-dir=" + str(target / ".git"), "hash-object", "--stdin"],
        cwd=repo[0],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        with pytest.raises(recovery.ActiveOwner):
            recover(repo)
    finally:
        process.terminate()
        process.communicate()


def test_git_metadata_symlink_refused(repo):
    target = clone(repo)
    metadata = repo[0] / "metadata"
    (target / ".git").rename(metadata)
    (target / ".git").symlink_to(metadata, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary Git checkout"):
        recover(repo)


def test_publication_interruption_keeps_original_and_staged_source(repo, monkeypatch):
    target = clone(repo)
    (target / ".git/index.lock").write_text("stale")
    original = recovery.os.rename

    def interrupt(source, destination):
        if str(source).endswith("stage-control"):
            raise OSError("injected publication interruption")
        return original(source, destination)

    monkeypatch.setattr(recovery.os, "rename", interrupt)
    with pytest.raises(OSError, match="publication interruption"):
        recover(repo)
    assert (
        repo[0] / ".sfull_preboot_v1/quarantined-control/.git/index.lock"
    ).read_text() == "stale"
    assert (
        repo[0] / ".sfull_preboot_v1/stage-control/source.txt"
    ).read_text() == "original\n"
    with pytest.raises(ValueError, match="prior quarantine"):
        recover(repo)
