"""Offline S-full archival/source readiness freeze; never enables jobs or rents compute."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import finalize_sfull as final
import prepare_sfull as prep

ROOT = prep.ROOT
PROOF = final.PROOF
REF = "refs/heads/codex/archived-carry12-sfull-20261031"
CPU = Path(
    "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/scripts/cpu_budget.py"
)
VOCAB = Path("/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf")


def git(*args, data=None, env=None):
    return (
        subprocess.check_output(["git", "-C", str(ROOT), *args], input=data, env=env)
        .decode()
        .strip()
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    manifest_path = PROOF / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = sorted(manifest["artifacts_sha256"])
    for path, expected in manifest["artifacts_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("packet bytes changed: " + path)
        exists = (
            subprocess.run(
                ["git", "-C", str(ROOT), "cat-file", "-e", prep.PIN + ":" + path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if exists:
            raise ValueError("archive delta must add only new paths: " + path)
    try:
        commit = git("rev-parse", "--verify", REF)
    except subprocess.CalledProcessError:
        with tempfile.TemporaryDirectory(prefix="sfull-index-") as temporary:
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
            git("read-tree", prep.PIN, env=env)
            for path in paths:
                blob = git(
                    "hash-object", "-w", "--stdin", data=(ROOT / path).read_bytes()
                )
                git(
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    "100644",
                    blob,
                    path,
                    env=env,
                )
            tree = git("write-tree", env=env)
            commit = git(
                "commit-tree",
                tree,
                "-p",
                prep.PIN,
                data=b"Archive conditional S-full L8 causal-mask jobs only; review, no launch\n",
            )
            git("update-ref", REF, commit, "")
    if git("rev-parse", commit + "^") != prep.PIN:
        raise ValueError("archive parent differs")
    if git("diff", "--name-only", prep.PIN, commit).splitlines() != paths:
        raise ValueError("archive delta differs")
    for path, expected in manifest["artifacts_sha256"].items():
        data = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", commit + ":" + path]
        )
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("archive blob differs: " + path)
    reference = json.loads(final.REFERENCE.read_text())
    if sha(VOCAB) != reference["vocab_sha256"]:
        raise ValueError("local upload vocabulary differs")
    manifest.update(
        archive_commit=commit,
        status="READY_FOR_FUNDING_REVIEW_WITH_RUNTIME_DATA_GATE",
        ready_for_launch=False,
        engineering_remaining=[
            "independent funding/parity/readiness review before any enable"
        ],
        runtime_data_preverified=False,
    )
    put(manifest_path, manifest)
    put(
        final.LAUNCH,
        {
            "parent": prep.PIN,
            "commit": commit,
            "manifest_sha256": sha(manifest_path),
            "branch": REF,
            "changed_paths": paths,
            "enabled": False,
            "runtime_data_gate_required": True,
        },
    )
    final.common.CPU_BUDGET = CPU
    source_hashes = {str(path.resolve()): sha(path) for path in final.sources()}
    source_hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    # The preparation tool is archived as provenance, not a runtime finalizer dependency.
    put(
        PROOF / "preparation_source.json",
        {"sha256": sha(Path(__file__)), "source": str(Path(__file__).resolve())},
    )
    source_hashes.pop(str(Path(__file__).resolve()))
    put(
        PROOF / "source_freeze.json",
        {
            "source_hashes": source_hashes,
            "scope": final.SCOPE,
            "prospective": True,
            "data_outcomes_absent": True,
        },
    )
    commands = []
    for seed in prep.SEEDS:
        commands.append(
            {
                "seed": seed,
                "enabled": False,
                "serial_predecessor": seed - 1 if seed > 1 else None,
                "requires_previous_campaign_home_and_budget_review": True,
                "argv": [
                    "python3",
                    "/home/jonat/pk_e10_component_20261031/tools/colab_chat/runner_multi.py",
                    "--backend",
                    "runpod",
                    "--gpu",
                    "L4",
                    "--cloud",
                    "SECURE",
                    "--volume-gb",
                    "80",
                    "--image",
                    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                    "--commit",
                    commit,
                    "--jobs",
                    f"tools/colab_chat/carry12_sfull_r{seed}.json",
                    "--boot-script",
                    "tools/colab_chat/boot_sfull.sh",
                    "--name",
                    f"pk-c12-sfull-r{seed}",
                    "--art",
                    f"/mnt/c/ml_artifacts/tiny_chat/c12_Sfull_r{seed}",
                    "--state-dir",
                    f"/home/jonat/pk_c12_sfull_r{seed}_runner",
                    "--max-usd",
                    "7",
                    "--max-hours",
                    "23.5",
                    "--upload",
                    "vocab_llama3_bpe.gguf=" + str(VOCAB),
                ],
            }
        )
    final_command = [
        "python3",
        str(final.HERE / "finalize_sfull.py"),
        "--cpu-budget",
        str(CPU),
        "--art",
        "/mnt/c/ml_artifacts/tiny_chat",
        "--state-dir",
        "/home/jonat/pk_c12_sfull_finalizer",
        "--out",
        "/mnt/c/ml_artifacts/tiny_chat/carry12_sfull_final",
        "--max-hours",
        "96",
        "--source-freeze",
        str(PROOF / "source_freeze.json"),
    ]
    controller = Path("/home/jonat/pk_e10_component_20261031")
    controller_proof = {
        "commit": subprocess.check_output(
            ["git", "-C", str(controller), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(controller / "tools/colab_chat" / name): sha(
                controller / "tools/colab_chat" / name
            )
            for name in [
                "runner_multi.py",
                "runpod_transport.py",
                "runpod_api.py",
                "runpod_sweep.py",
            ]
        },
        "review_again_at_funding": True,
    }
    readiness = {
        "status": manifest["status"],
        "no_paid_launch": True,
        "no_watchdog_entries_enabled": True,
        "archive_commit": commit,
        "archive_parent": prep.PIN,
        "archive_changed_paths": paths,
        "source_freeze_sha256": sha(PROOF / "source_freeze.json"),
        "vocab_upload_sha256": sha(VOCAB),
        "expected_raw_human_sha256": reference["human"]["files"],
        "expected_products": reference["outputs"],
        "runtime_data_preverified": False,
        "runtime_gate": "all13 raw sources +5 products +vocab +evaluations +decontamination exact before PT; mismatch blocks; no fallback/no automatic rerun",
        "historical_parity_blocker": "six original HS1 history mismatches retained; not waived",
        "funding_required_not_inferred": True,
        "notification_threshold_usd": 20,
        "offered_topup_usd_uncredited": 40,
        "existing_provider_min_balance_accessor_constraint_usd": 15,
        "separate_long_run_funding_review_required": True,
        "tokenizer_runtime_requirement": "tokenizers==0.23.3",
        "modern_local_controller": controller_proof,
        "per_seed_cap_usd": 7,
        "total_cap_usd": 21,
        "expected_gpu_hours": [18, 27],
        "expected_gpu_usd": [11, 16],
        "serial_queue_only": True,
        "expected_serial_gpu_hours_combined": [18, 27],
        "operational_max_hours_per_seed": 23.5,
        "operational_max_hours_all_three_if_fully_used": 70.5,
        "clock_policy": "separate per-seed clocks; never one23.5h clock for all3; any later continuation needs explicit funding/budget review; no auto cap raise",
        "campaigns": commands,
        "finalizer_argv": final_command,
    }
    put(PROOF / "readiness.json", readiness)
    print(
        json.dumps(
            {
                "archive_commit": commit,
                "readiness": str(PROOF / "readiness.json"),
                "status": readiness["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
