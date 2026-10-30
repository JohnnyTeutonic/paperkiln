"""Offline operational tests for baseline-first component evidence relay."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e10_vm", HERE / "e10_component_vm.py")
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)


@pytest.mark.parametrize(
    "mode", ["complete", "candidate_halted", "baseline_halted", "tampered"]
)
def test_baseline_first_and_preserved(tmp_path, monkeypatch, mode):
    source = tmp_path / "source"
    runner = source / "tools/row_pick_profile/run.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# frozen test source")
    upload = tmp_path / "upload"
    upload.mkdir()
    archive = upload / "e10_component_source.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source / "tools", arcname="tools")
    manifest = {
        "ember_commit": "a" * 40,
        "vm_sha256": hashlib.sha256(Path(vm.__file__).read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {
            "tools/row_pick_profile/run.py": hashlib.sha256(
                runner.read_bytes()
            ).hexdigest()
        },
    }
    if mode == "tampered":
        manifest["source_sha256"]["tools/row_pick_profile/run.py"] = "b" * 64
    (upload / "e10_component_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["vm", str(output), str(upload)])
    calls = []

    def fake(command, **kwargs):
        leg = "baseline" if "--baseline-only" in command else "candidate"
        calls.append(leg)
        out = Path(command[command.index("--output") + 1])
        out.mkdir()
        status = "HALTED" if mode == leg + "_halted" else "COMPLETE"
        (out / "receipt.json").write_text(json.dumps({"status": status}))

    monkeypatch.setattr(vm.subprocess, "run", fake)
    vm.main()
    result = json.loads((output / "summary.json").read_text())
    assert calls == (
        []
        if mode == "tampered"
        else ["baseline"]
        if mode == "baseline_halted"
        else ["baseline", "candidate"]
    )
    assert result["status"] == ("COMPLETE" if mode == "complete" else "HALTED")
    assert (output / "out.zip").exists()
    if mode == "candidate_halted":
        assert result["baseline"] == "COMPLETE"
        assert (
            json.loads((output / "baseline/receipt.json").read_text())["status"]
            == "COMPLETE"
        )
