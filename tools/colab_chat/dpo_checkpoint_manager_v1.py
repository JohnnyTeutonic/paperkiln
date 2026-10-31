"""DPO-only normal jobs_vm owner with durable-volume cadence fail-stop."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import dpo_checkpoint_v1 as dpo
from dpo_checkpoint_transport_v1 import config


def validate_jobs(historical, jobs_path, upload):
    upload = Path(upload)
    binding = json.loads((upload / "dpo_binding.json").read_text())
    contract, digest = dpo.load_contract(upload / "dpo_checkpoint_contract_v1.json")
    if (
        binding["runtime_jobs_sha256"]
        != hashlib.sha256(Path(jobs_path).read_bytes()).hexdigest()
        or binding["checkpoint_contract_sha256"] != digest
    ):
        raise ValueError("DPO operational inputs differ from binding")
    head = subprocess.check_output(
        ["git", "-C", str(historical), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != contract["source_commit"] or binding["source_commit"] != head:
        raise ValueError("DPO admitted source differs")
    jobs = json.loads(Path(jobs_path).read_text())["jobs"]
    training = [j for j in jobs if "spec" in j]
    if len(training) != 1 or training[0]["name"] != "pk_dpo_v1":
        raise ValueError("unexpected DPO training job inventory")
    job = training[0]
    if (
        job.get("after") != "dpo_preflight"
        or job.get("init_file") != "pksft.safetensors"
    ):
        raise ValueError(
            "DPO training must follow source preflight and fixed SFT initialization"
        )
    c = config(job)
    if c["contract_sha256"] != digest:
        raise ValueError("DPO checkpoint job contract differs")
    actual = json.loads(json.dumps(job["spec"]))
    actual.update(name="pk_dpo_v1", out_dir="/content/chat/pk_dpo_v1")
    actual["train"]["init_from"] = "/content/chat/upload/pksft.safetensors"
    if actual != contract["runtime_spec"]:
        raise ValueError("DPO runtime recipe differs")
    if (
        any(
            c[k] != contract["runtime_spec"]["train"][k]
            for k in ["steps", "seed", "batch", "accum"]
        )
        or c["source_commit"] != head
    ):
        raise ValueError("DPO checkpoint monitor recipe differs")
    if not jobs or jobs[0]["name"] != "dpo_preflight":
        raise ValueError("DPO source preflight must precede training")
    return contract, digest


def finish(chat, jobs, manager_rc):
    names = [j["name"] for j in jobs["jobs"]]
    okay = manager_rc == 0 and all(
        (Path(chat) / ("STATUS_" + n)).is_file()
        and (Path(chat) / ("STATUS_" + n)).read_text().strip() == "FINISHED"
        for n in names
    )
    status = "FINISHED" if okay else "JOBS_FAILED"
    temp = Path(chat) / "STATUS.tmp"
    temp.write_text(status + "\n")
    os.replace(temp, Path(chat) / "STATUS")
    return 0 if okay else 1


def active_training_folders(historical, manager_group):
    result = set()
    binary = (Path(historical) / "build_cuda/mtstudio").resolve()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if (
                len(argv) < 3
                or not argv[0]
                or Path(os.fsdecode(argv[0])).resolve() != binary
                or argv[1] != b"run"
            ):
                continue
            if os.getpgid(int(entry.name)) != manager_group:
                continue
            spec = json.loads(Path(os.fsdecode(argv[2])).read_text())
            result.add(str(Path(spec["out_dir"]).resolve()))
        except (OSError, ValueError, KeyError):
            continue
    return result


def gap_check(jobs, chat, active, starts, now, clock):
    from dpo_checkpoint_transport_v1 import config, root_for

    for job in jobs:
        c = config(job)
        if not c:
            continue
        folder = str((Path(chat) / job["name"]).resolve())
        if folder not in active:
            starts.pop(folder, None)
            continue
        observed = starts.setdefault(folder, {"clock": clock, "step": 0})
        root = root_for(chat, job)
        if (root / "CURRENT").is_symlink():
            # The synchronous publisher committed and fsynced the immutable generation.
            m = json.loads((root / "CURRENT/manifest.json").read_text())
            if (
                m["source_commit"] != c["source_commit"]
                or m["step"] > c["steps"]
                or m["step"] < observed["step"]
            ):
                raise ValueError("checkpoint monitor source/step differs or regressed")
            if m["step"] == c["steps"]:
                continue
            if m["step"] > observed["step"]:
                observed.update(clock=clock, step=m["step"])
        gap = clock - observed["clock"]
        if gap > c["max_gap_seconds"]:
            raise ValueError(
                "committed persistent-volume checkpoint gap exceeded180seconds: "
                + job["name"]
            )


def manage(historical, chat, jobs_path):
    import signal
    import time
    from dpo_checkpoint_transport_v1 import recover_remote

    chat = Path(chat)
    marker = chat / "DPO_CHECKPOINT_V1_HALTED.json"
    if marker.exists():
        raise ValueError("preserved checkpoint halt requires review")
    jobs = json.loads(Path(jobs_path).read_text())
    contract, digest = validate_jobs(historical, jobs_path, chat / "upload")
    recover_remote(jobs["jobs"], chat, chat / "upload")
    env = os.environ | {"JOBS_REPO": str(historical), "JOBS_CHAT_DIR": str(chat)}
    env.update(
        {
            "MICROTORCH_DPO_CHECKPOINT_PUBLISH_V1": str(
                Path(historical) / "tools/colab_chat/dpo_checkpoint_v1.py"
            ),
            "DPO_NUMERICAL_SOURCE_COMMIT": contract["source_commit"],
            "DPO_CHECKPOINT_CONTRACT": str(
                chat / "upload/dpo_checkpoint_contract_v1.json"
            ),
        }
    )
    p = subprocess.Popen(
        [
            "python3",
            "-u",
            str(Path(historical) / "tools/colab_chat/jobs_vm.py"),
            str(jobs_path),
        ],
        cwd=historical,
        env=env,
        start_new_session=True,
    )
    starts = {}
    try:
        while p.poll() is None:
            active = active_training_folders(historical, p.pid)
            gap_check(jobs["jobs"], chat, active, starts, time.time(), time.monotonic())
            time.sleep(2)
        return finish(chat, jobs, p.returncode)
    except Exception as error:
        # Only the process group created above is owned and terminated.
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        with marker.open("x") as f:
            json.dump({"kind": "DPO_CHECKPOINT_V1_HALTED", "reason": str(error)}, f)
            f.flush()
            os.fsync(f.fileno())
        with (chat / "boot.log").open("a") as f:
            f.write("DPO_CHECKPOINT_V1_HALTED: " + str(error) + "\n")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical", default="/content/pk")
    p.add_argument("--chat", default="/content/chat")
    p.add_argument("--jobs", required=True)
    args = p.parse_args()
    try:
        return manage(args.historical, args.chat, args.jobs)
    except Exception:
        chat = Path(args.chat)
        chat.mkdir(parents=True, exist_ok=True)
        temp = chat / "STATUS.tmp"
        temp.write_text("BOOT_FAILED\n")
        os.replace(temp, chat / "STATUS")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
