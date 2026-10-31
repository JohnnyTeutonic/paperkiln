import copy
import hashlib
import json
from unittest.mock import patch
import pytest
import dpo_checkpoint_manager_v1 as manager

SOURCE = "a" * 40


@pytest.fixture
def bound(tmp_path):
    upload = tmp_path / "upload"
    upload.mkdir()
    spec = {
        "name": "pk_dpo_v1",
        "out_dir": "/content/chat/pk_dpo_v1",
        "data": {
            "vocab": "/content/chat/upload/vocab.gguf",
            "pref_pairs": "/content/chat/upload/dpo_pairs.jsonl",
        },
        "train": {
            "objective": "provenance_po",
            "optimizer": "adamw",
            "batch": 8,
            "accum": 1,
            "seed": 1,
            "steps": 20,
            "checkpoint_every": 10,
            "init_from": "/content/chat/upload/pksft.safetensors",
        },
    }
    contract = {
        "kind": "DPO_FULL_STATE_CHECKPOINT_V1",
        "source_commit": SOURCE,
        "runtime_spec": spec,
        "inputs_sha256": {
            p: "b" * 64 for p in [*spec["data"].values(), spec["train"]["init_from"]]
        },
        "dataset_manifest_sha256": "c" * 64,
    }
    c = upload / "dpo_checkpoint_contract_v1.json"
    c.write_text(json.dumps(contract))
    digest = hashlib.sha256(c.read_bytes()).hexdigest()
    jobs = {
        "jobs": [
            {
                "name": "dpo_preflight",
                "cmd": ["python3", "tools/posttrain/dpo_vm_preflight_v7.py"],
            },
            {
                "name": "pk_dpo_v1",
                "after": "dpo_preflight",
                "spec": spec,
                "init_file": "pksft.safetensors",
                "dpo_full_checkpoint_v1": {
                    "every_steps": 10,
                    "max_gap_seconds": 180,
                    "steps": 20,
                    "source_commit": SOURCE,
                    "seed": 1,
                    "batch": 8,
                    "accum": 1,
                    "contract_sha256": digest,
                },
            },
        ]
    }
    path = upload / "DPO_RUNTIME_JOBS_V1.json"
    path.write_text(json.dumps(jobs))
    binding = {
        "source_commit": SOURCE,
        "runtime_jobs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_contract_sha256": digest,
    }
    b = upload / "dpo_binding.json"
    b.write_text(json.dumps(binding))
    with patch.object(manager.subprocess, "check_output", return_value=SOURCE + "\n"):
        yield tmp_path, path, upload, jobs, b, binding


def test_exact_bound_instance(bound):
    repo, path, upload, *_ = bound
    assert manager.validate_jobs(repo, path, upload)[0]["source_commit"] == SOURCE


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_job",
        "spec_lr",
        "monitor_steps",
        "monitor_source",
        "second_train",
        "contract",
        "source",
    ],
)
def test_mutated_jobs_or_contract_refused(bound, mutation):
    repo, path, upload, jobs, b, binding = bound
    if mutation == "contract":
        binding["checkpoint_contract_sha256"] = "0" * 64
    elif mutation == "source":
        binding["source_commit"] = "f" * 40
    elif mutation == "extra_job":
        jobs["jobs"].append({"name": "undeclared", "cmd": ["true"]})
    elif mutation == "second_train":
        jobs["jobs"].append(copy.deepcopy(jobs["jobs"][1]))
    elif mutation == "spec_lr":
        jobs["jobs"][1]["spec"]["train"]["lr"] = 99
    elif mutation == "monitor_steps":
        jobs["jobs"][1]["dpo_full_checkpoint_v1"]["steps"] = 99
    else:
        jobs["jobs"][1]["dpo_full_checkpoint_v1"]["source_commit"] = "f" * 40
    path.write_text(json.dumps(jobs))
    if mutation != "extra_job":
        binding["runtime_jobs_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    b.write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        manager.validate_jobs(repo, path, upload)


def test_owned_manager_gap_failure_preserves_halt_and_unrelated_process(
    tmp_path, monkeypatch
):
    import subprocess
    import sys

    repo = tmp_path / "repo"
    manager_path = repo / "tools/colab_chat/jobs_vm.py"
    manager_path.parent.mkdir(parents=True)
    manager_path.write_text("import time\ntime.sleep(30)\n")
    chat = tmp_path / "chat"
    chat.mkdir()
    jobs = tmp_path / "jobs.json"
    jobs.write_text('{"jobs": []}')
    monkeypatch.setattr(
        manager, "validate_jobs", lambda *args: ({"source_commit": SOURCE}, "b" * 64)
    )
    import dpo_checkpoint_transport_v1 as transport

    monkeypatch.setattr(transport, "recover_remote", lambda *args: [])
    monkeypatch.setattr(manager, "active_training_folders", lambda *args: set())

    def fail(*args):
        raise ValueError("injected durable gap")

    monkeypatch.setattr(manager, "gap_check", fail)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    try:
        with pytest.raises(ValueError, match="durable gap"):
            manager.manage(repo, chat, jobs)
        assert unrelated.poll() is None
        marker = chat / "DPO_CHECKPOINT_V1_HALTED.json"
        before = marker.read_bytes()
        with pytest.raises(ValueError, match="preserved"):
            manager.manage(repo, chat, jobs)
        assert marker.read_bytes() == before
    finally:
        unrelated.terminate()
        unrelated.wait()
