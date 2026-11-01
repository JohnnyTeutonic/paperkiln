import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

CANDIDATE = Path("/home/jonat/sfull_boot_recovery_candidate_v2_20261101")
FROZEN = Path("/home/jonat/pk_sfull_operation_v2_20261101")
if not CANDIDATE.is_dir():
    pytest.skip("private recovery candidate unavailable", allow_module_level=True)


def load():
    path = CANDIDATE / "tools/colab_chat/runner_multi.py"
    spec = importlib.util.spec_from_file_location("reviewed_recovery_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code():
    return load().boot_launch_code(
        "1af22cfd66f95cd40dcd98cf4ffae57cb3de4c61",
        "tools/colab_chat/carry12_sfull_v2_r1.json",
        "tools/colab_chat/boot_sfull_v2_20261101.sh",
    )


def test_generated_remote_code_compiles_and_only_original_args():
    remote = code()
    compile(remote, "remote", "exec")
    for value in [
        "1af22cfd66f95cd40dcd98cf4ffae57cb3de4c61",
        "tools/colab_chat/carry12_sfull_v2_r1.json",
    ]:
        assert value in remote


def test_exact_reverse_function_proof():
    original = (FROZEN / "tools/colab_chat/runner_multi.py").read_text()
    amended = (CANDIDATE / "tools/colab_chat/runner_multi.py").read_text()

    def split(source):
        node = next(
            x
            for x in ast.parse(source).body
            if isinstance(x, ast.FunctionDef) and x.name == "boot_launch_code"
        )
        lines = source.splitlines(keepends=True)
        return (
            "".join(lines[: node.lineno - 1]),
            "".join(lines[node.lineno - 1 : node.end_lineno]),
            "".join(lines[node.end_lineno :]),
        )

    a, b, c = split(amended)
    _, old, _ = split(original)
    assert a + old + c == original


def test_actual_help_and_repo_relative_jobs_load_equivalence():
    script = CANDIDATE / "tools/colab_chat/runner_multi.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--max-usd" in result.stdout
    repo = Path(load().HERE).parents[1]
    for seed in (1, 2, 3):
        rel = f"tools/colab_chat/carry12_sfull_v2_r{seed}.json"
        assert (repo / rel).read_bytes() == (FROZEN / rel).read_bytes()
        assert json.loads((repo / rel).read_text())["jobs"]
    for name in [
        "runpod_api.py",
        "runpod_transport.py",
        "jobs_vm.py",
        "sfull_checkpoint_v2.py",
        "sfull_checkpoint_transport_v2.py",
    ]:
        assert (script.parent / name).read_bytes() == (
            FROZEN / "tools/colab_chat" / name
        ).read_bytes()


def test_remote_preparation_preserves_previous_status_no_provider(
    tmp_path, monkeypatch
):
    remote = code().replace("/content", str(tmp_path))
    chat = tmp_path / "chat"
    chat.mkdir()
    (chat / "STATUS").write_text("BOOT_FAILED\n")
    calls = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    exec(compile(remote, "offline_remote", "exec"), {})
    assert (chat / "sfull_preboot_status_before_v1.txt").read_text() == "BOOT_FAILED\n"
    assert (chat / "STATUS").read_text() == "BOOTING\n"
    assert len(calls) == 1 and calls[0][0][1] == str(
        tmp_path / "sfull_boot_recovery_capsule_v1.py"
    )
    assert calls[0][0][-2:] == [
        "1af22cfd66f95cd40dcd98cf4ffae57cb3de4c61",
        "tools/colab_chat/carry12_sfull_v2_r1.json",
    ]
    policy = json.loads((tmp_path / "sfull_preboot_policy_v1.json").read_text())
    assert policy["control_archive_bytes"] == 388659200


def test_old_frozen_source_anchors_still_match():
    freeze = FROZEN / "docs/receipts/sfull_launch_v2_20261101/source_freeze_v2.json"
    pins = json.loads(freeze.read_text())["source_hashes"]
    assert len(pins) == 37
    for path, digest in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
