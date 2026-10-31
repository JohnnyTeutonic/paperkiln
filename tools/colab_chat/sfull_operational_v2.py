"""Explicit operational overlay; historical training source and recipe stay frozen."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

NUMERICAL_COMMIT = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
V2_SHA = "ef002f238c1d3d54aa27e6e4981ef9b5590d7930c1feeabc245ac4a607c2edf0"
REFERENCE_SHA = "acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_jobs(original, seed, control="/content/sfull_control"):
    jobs = json.loads(json.dumps(original))
    jobs["_comment"] = (
        "V2 operational input-verification overlay only. Seeds, model, data, training, scoring and historical numerical source unchanged; historic parity qualification retained."
    )
    for job in jobs["jobs"]:
        if job["name"] == "sfull_inputs":
            job["cmd"] = [
                "python3",
                f"{control}/tools/colab_chat/sfull_verify_inputs_v2.py",
                "--data",
                "/content/chat/data",
                "--repo",
                "/content/pk",
                "--reference",
                "experiments/tiny_chat_long_2026-10/receipts/sfull_prospective_20261031/data_reference.json",
                "--out",
                "{out}/verified_inputs.json",
            ]
        if job["name"] == "gate":
            job["cmd"][job["cmd"].index("--jobs") + 1] = (
                f"{control}/tools/colab_chat/carry12_sfull_v2_r{seed}.json"
            )
    for job in jobs["jobs"]:
        if job["name"].startswith("pt_"):
            job["spec"]["train"]["checkpoint_every"] = 25
            job["relay_every_steps"] = 25
            job["full_checkpoint_v2"] = {
                "relative": ".",
                "every_steps": 25,
                "max_gap_seconds": 180,
                "steps": 10000,
                "source_commit": NUMERICAL_COMMIT,
                "seed": seed,
                "batch": 16,
                "accum": 1,
            }
        elif job["name"].startswith("post_"):
            job["cmd"][job["cmd"].index("--sft-template") + 1] = (
                f"{control}/tools/colab_chat/sfull_sft_template_v2.json"
            )
            job["full_checkpoint_v2"] = {
                "relative": "sft",
                "every_steps": 25,
                "max_gap_seconds": 180,
                "steps": 1500,
                "source_commit": NUMERICAL_COMMIT,
                "seed": seed,
                "batch": 16,
                "accum": 1,
            }
    return jobs


def validate(historical, control, jobs_path):
    historical, control, jobs_path = map(Path, (historical, control, jobs_path))
    head = subprocess.check_output(
        ["git", "-C", str(historical), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != NUMERICAL_COMMIT:
        raise ValueError("historical numerical commit differs")
    subprocess.run(
        ["git", "-C", str(historical), "diff", "--exit-code", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
    )
    if sha(control / "tools/colab_chat/sfull_verify_inputs_v2.py") != V2_SHA:
        raise ValueError("reviewed V2 verifier source differs")
    reference = (
        historical
        / "experiments/tiny_chat_long_2026-10/receipts/sfull_prospective_20261031/data_reference.json"
    )
    if sha(reference) != REFERENCE_SHA:
        raise ValueError("original reference differs")
    actual = json.loads(jobs_path.read_text())
    matched = []
    for seed in (1, 2, 3):
        original = json.loads(
            (historical / f"tools/colab_chat/carry12_sfull_r{seed}.json").read_text()
        )
        if actual == expected_jobs(original, seed):
            matched.append(seed)
    if len(matched) != 1:
        raise ValueError("V2 jobs differ beyond exact reviewed operational overlay")
    original_sft = json.loads(
        (
            historical / "experiments/tiny_chat_long_2026-10/specs/sft_T1024.json"
        ).read_text()
    )
    original_sft["train"]["checkpoint_every"] = 25
    if (
        json.loads(
            (control / "tools/colab_chat/sfull_sft_template_v2.json").read_text()
        )
        != original_sft
    ):
        raise ValueError("SFT differs beyond checkpoint interval")
    return {
        "numerical_commit": head,
        "seed": matched[0],
        "jobs_sha256": sha(jobs_path),
        "V2_sha256": V2_SHA,
        "reference_sha256": REFERENCE_SHA,
        "arithmetic_data_optimization_recipe_changed": False,
        "checkpoint_io_interval_amendment": True,
        "historical_parity_clean_pass_claimed": False,
    }


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
    from sfull_checkpoint_transport_v2 import config, root_for

    for job in jobs:
        c = config(job)
        if not c:
            continue
        folder = str((Path(chat) / job["name"] / c["relative"]).resolve())
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


def manage(historical, control, chat, jobs_path):
    import signal
    import time
    from sfull_checkpoint_transport_v2 import recover_remote

    chat = Path(chat)
    marker = chat / "CHECKPOINT_V2_HALTED.json"
    if marker.exists():
        raise ValueError("preserved checkpoint halt requires review")
    jobs = json.loads(Path(jobs_path).read_text())
    recover_remote(jobs["jobs"], chat, chat / "upload")
    env = os.environ | {"JOBS_REPO": str(historical), "JOBS_CHAT_DIR": str(chat)}
    env["S_FULL_CHECKPOINT_TARGETS"] = json.dumps(
        [
            str((chat / j["name"] / j["full_checkpoint_v2"]["relative"]).resolve())
            for j in jobs["jobs"]
            if "full_checkpoint_v2" in j
        ]
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
            json.dump({"kind": "S_FULL_CHECKPOINT_V2_HALTED", "reason": str(error)}, f)
            f.flush()
            os.fsync(f.fileno())
        with (chat / "boot.log").open("a") as f:
            f.write("CHECKPOINT_V2_HALTED: " + str(error) + "\n")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["validate", "manage"])
    p.add_argument("--historical", default="/content/pk")
    p.add_argument("--control", default="/content/sfull_control")
    p.add_argument("--chat", default="/content/chat")
    p.add_argument("--jobs", required=True)
    a = p.parse_args()
    try:
        proof = validate(a.historical, a.control, a.jobs)
        print(json.dumps(proof), flush=True)
        if a.action == "validate":
            return 0
        return manage(a.historical, a.control, a.chat, a.jobs)
    except Exception:
        if a.action == "manage":
            temp = Path(a.chat) / "STATUS.tmp"
            temp.write_text("BOOT_FAILED\n")
            os.replace(temp, Path(a.chat) / "STATUS")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
