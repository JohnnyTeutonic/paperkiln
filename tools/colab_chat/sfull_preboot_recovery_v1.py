"""One-shot source-only recovery; never unlink Git locks or touch training data."""

from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


class ActiveOwner(RuntimeError):
    pass


def git(folder, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(folder), *args],
        check=check,
        capture_output=True,
        text=True,
        env=os.environ | {"GIT_OPTIONAL_LOCKS": "0"},
    )


def archive_identity(folder, commit):
    process = subprocess.Popen(
        ["git", "-C", str(folder), "archive", "--format=tar", commit],
        stdout=subprocess.PIPE,
    )
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: process.stdout.read(1 << 20), b""):
        digest.update(block)
        size += len(block)
    process.stdout.close()
    if process.wait() != 0:
        raise ValueError("source archive failed")
    return digest.hexdigest(), size


def ancestors():
    result = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            pid = int(
                Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1]
            )
        except (OSError, ValueError, IndexError):
            break
    return result


def active_owners(roots):
    ignored = ancestors()
    result = []
    roots = [p.resolve() for p in roots]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ignored:
            continue
        try:
            argv = [
                os.fsdecode(x)
                for x in (proc / "cmdline").read_bytes().split(b"\0")
                if x
            ]
            if not argv:
                continue
            names = {Path(x).name for x in argv}
            manager = bool(
                names & {"jobs_vm.py", "boot_jobs.sh", "boot_sfull_v2_20261101.sh"}
            ) or ("sfull_operational_v2.py" in names and "manage" in argv)
            if manager:
                result.append(int(proc.name))
                continue
            if Path(argv[0]).name not in {"git", "mtstudio"}:
                continue
            result.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def inspect_existing(folder, expected, origin, allow_unborn):
    if folder.is_symlink():
        raise ValueError("source checkout symlink refused")
    if not folder.exists():
        return "absent"
    if (folder / ".git").is_symlink() or not (folder / ".git").is_dir():
        raise ValueError("source path is not an owned ordinary Git checkout")
    if git(folder, "remote", "get-url", "origin").stdout.strip() != origin:
        raise ValueError("source origin differs")
    head = git(folder, "rev-parse", "--verify", "HEAD", check=False)
    if head.returncode:
        if allow_unborn:
            return "unborn"
        raise ValueError("numeric source is incomplete")
    if head.stdout.strip() != expected:
        raise ValueError("existing source revision differs")
    if git(folder, "diff", "--quiet", "HEAD", "--", check=False).returncode:
        raise ValueError("tracked source is dirty or unreadable")
    return "stale_lock" if (folder / ".git/index.lock").exists() else "clean"


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    sync_dir(path.parent)


def sync_dir(folder):
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def recover(
    root,
    origin,
    control_commit,
    numerical_commit,
    expected_archive,
    expected_archive_bytes=0,
):
    root = Path(root).resolve()
    chat = root / "chat"
    chat.mkdir(parents=True, exist_ok=True)
    control = root / "sfull_control"
    numeric = root / "pk"
    workspace = root / ".sfull_preboot_v1"
    with (chat / "sfull_preboot_v1.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ActiveOwner("another source recovery owns the boot lock") from error
        if active_owners([control, numeric]):
            raise ActiveOwner("live Git/boot/manager owner; source recovery refused")
        control_state = inspect_existing(control, control_commit, origin, True)
        numeric_state = inspect_existing(numeric, numerical_commit, origin, False)
        if numeric_state not in {"absent", "clean"}:
            raise ValueError(
                "numeric checkout requires independent review; never altered"
            )
        if control_state == "clean":
            actual, _ = archive_identity(control, control_commit)
            if actual != expected_archive:
                raise ValueError("existing source archive differs")
            return {
                "status": "ALREADY_CLEAN",
                "numeric_unchanged": True,
                "archive_sha256": actual,
            }
        if workspace.is_symlink():
            raise ValueError("recovery workspace symlink refused")
        workspace.mkdir(exist_ok=True)
        stage = workspace / "stage-control"
        quarantine = workspace / "quarantined-control"
        if quarantine.exists() or quarantine.is_symlink():
            raise ValueError(
                "prior quarantine preserved; refuse another source replacement"
            )
        if stage.exists() or stage.is_symlink():
            raise ValueError(
                "interrupted staging preserved; review before another attempt"
            )
        if shutil.disk_usage(root).free < 2 * expected_archive_bytes + (64 << 20):
            raise ValueError("insufficient source staging space")
        stage.mkdir()
        git(stage, "init", "-q")
        git(stage, "remote", "add", "origin", origin)
        git(
            stage,
            "fetch",
            "-q",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            control_commit,
        )
        git(stage, "checkout", "-q", "--detach", control_commit)
        if inspect_existing(stage, control_commit, origin, False) != "clean":
            raise ValueError("staged source is not clean")
        actual, size = archive_identity(stage, control_commit)
        if actual != expected_archive:
            raise ValueError("staged source archive differs")
        if active_owners([control, numeric]):
            raise ActiveOwner("owner appeared while staging; no publication")
        if inspect_existing(control, control_commit, origin, True) != control_state:
            raise ValueError("source changed during staging")
        report = {
            "status": "STAGED_VERIFIED",
            "control_commit": control_commit,
            "archive_sha256": actual,
            "archive_bytes": size,
            "previous_control_state": control_state,
            "numeric_state": numeric_state,
            "numeric_unchanged": True,
            "stage": str(stage),
            "quarantine": str(quarantine),
        }
        atomic_json(chat / "sfull_preboot_recovery_v1.json", report)
        if control.exists():
            os.rename(control, quarantine)
        os.rename(stage, control)
        sync_dir(root)
        sync_dir(workspace)
        report["status"] = "SOURCE_RECOVERED_ORIGINAL_PATH"
        atomic_json(chat / "sfull_preboot_recovery_v1.json", report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    a = p.parse_args()
    policy = json.loads(Path(a.policy).read_text())
    try:
        print(
            json.dumps(
                recover(
                    "/content",
                    policy["origin"],
                    policy["control_commit"],
                    policy["numerical_commit"],
                    policy["control_archive_sha256"],
                    policy["control_archive_bytes"],
                )
            )
        )
    except ActiveOwner:
        raise
    except Exception:
        Path("/content/chat/STATUS").write_text("BOOT_FAILED\n")
        raise


if __name__ == "__main__":
    main()
