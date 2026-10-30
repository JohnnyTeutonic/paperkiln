"""Packaging/lifecycle fixtures only: compiler, hardware and checker are mocked."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bf16_vm", HERE / "bf16_attention_vm.py")
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)


@pytest.fixture
def upload(tmp_path):
    path = tmp_path / "upload"
    path.mkdir()
    for name in vm.FILES:
        source = HERE / name if name == "bf16_attention_vm.py" else HERE.parent / name
        (path / name).write_bytes(source.read_bytes())
    manifest = {"schema":"paperkiln.bf16_attention.upload.v1",
                "files":{name:vm.sha(path / name) for name in vm.FILES}}
    (path / vm.MANIFEST).write_text(json.dumps(manifest))
    return path


def fake_execute(mode="pass", calls=None):
    calls = calls if calls is not None else []
    def execute(argv, log, timeout, cwd):
        calls.append(argv)
        text, rc = "mock command\n", 0
        if argv[0] == "nvidia-smi":
            text = "NVIDIA L4, GPU-fixture, fixture-driver\n"
        elif argv[:2] == ["nvcc", "--version"]:
            text = "Cuda compilation tools, release 12.4, V12.4.0\n"
        elif argv[0] == "nvcc":
            if mode == "compile-fail":
                text, rc = "injected compiler error\n", 7
            else:
                Path(argv[argv.index("-o") + 1]).write_bytes(b"CPU FIXTURE; not a CUDA executable")
        elif len(argv) > 1 and argv[1].endswith("bf16_attention_check.py"):
            results = Path(argv[argv.index("--output") + 1])
            results.mkdir(exist_ok=True)
            if mode == "negative":
                (results / "HALTED").write_text(json.dumps({"case":"fixture", "error":"per-stage numerical bound exceeded"}))
                text, rc = "injected numerical negative\n", 1
            elif "--replay" not in argv:
                (results / "report.json").write_text(json.dumps({"status":"COMPLETE", "cases":[{}]*32}))
                (results / "fixture.out.bin").write_bytes(b"raw CPU packaging fixture")
        log.write_text(text)
        return rc
    return execute


def summary(out):
    return json.loads((out / "attempt_summary.json").read_text())


def test_success_archive_exact_sources_and_idempotent_relay(upload, tmp_path):
    out = tmp_path / "job"
    calls = []
    assert vm.run(upload, out, execute=fake_execute(calls=calls)) == 0
    data = summary(out)
    assert data["status"] == "PASS" and data["numerical_pass"] is True
    with zipfile.ZipFile(out / "out.zip") as bundle:
        assert bundle.testzip() is None
        assert bundle.read("attempt_summary.json") == (out / "attempt_summary.json").read_bytes()
        for name in vm.FILES:
            assert bundle.read("evidence/sources/" + name) == (upload / name).read_bytes()
        assert "evidence/results/fixture.out.bin" in bundle.namelist()
    receipt = json.loads((out / "archive_receipt.json").read_text())
    assert receipt["archive_sha256"] == vm.sha(out / "out.zip")
    before = len(calls)
    assert vm.run(upload, out, execute=fake_execute(calls=calls)) == 0
    assert len(calls) == before  # relay/recovery never repeats scientific execution


def test_numerical_negative_returns_packaged_attempt_not_pass(upload, tmp_path):
    out = tmp_path / "negative"
    assert vm.run(upload, out, execute=fake_execute("negative")) == 0
    data = summary(out)
    assert data["status"] == "HALTED" and data["numerical_pass"] is False
    assert data["checker_returncode"] == 1
    with zipfile.ZipFile(out / "out.zip") as bundle:
        assert "evidence/results/HALTED" in bundle.namelist()
    assert not any(command["name"] == "offline_replay" for command in data["commands"])


def test_operational_compile_error_is_packaged_with_returncode(upload, tmp_path):
    out = tmp_path / "compile-error"
    assert vm.run(upload, out, execute=fake_execute("compile-fail")) == 0
    data = summary(out)
    assert data["status"] == "OPERATIONAL_FAILURE" and not data["numerical_pass"]
    assert data["commands"][-1]["returncode"] == 7
    assert not any(command["name"] == "checker" for command in data["commands"])


def test_packaging_error_is_fatal_without_completed_archive_receipt(upload, tmp_path):
    out = tmp_path / "archive-error"
    def broken(*args):
        raise OSError("injected packaging failure")
    with pytest.raises(OSError, match="packaging"):
        vm.run(upload, out, execute=fake_execute(), pack=broken)
    assert not (out / "out.zip").exists()
    assert not (out / "archive_receipt.json").exists()
    assert summary(out)["archive_required"] is True


def test_tampered_upload_refused_before_any_command(upload, tmp_path):
    (upload / "bf16_attention_probe.cu").write_text("changed")
    calls = []
    out = tmp_path / "tamper"
    assert vm.run(upload, out, execute=fake_execute(calls=calls)) == 0
    assert calls == []
    assert summary(out)["status"] == "OPERATIONAL_FAILURE"
    assert "SHA mismatch" in summary(out)["errors"][0]


def test_missing_manifest_still_packages_operational_failure(upload, tmp_path):
    (upload / vm.MANIFEST).unlink()
    out = tmp_path / "missing-manifest"
    assert vm.run(upload, out, execute=fake_execute()) == 0
    assert not summary(out)["numerical_pass"]
    assert "manifest is missing" in summary(out)["errors"][0]


def test_interrupted_attempt_preserved_without_execution(upload, tmp_path):
    out = tmp_path / "interrupted"
    work = out / "bf16_attempt"
    work.mkdir(parents=True)
    (work / "partial.bin").write_bytes(b"retain exact partial bytes")
    calls = []
    assert vm.run(upload, out, execute=fake_execute(calls=calls)) == 0
    assert calls == []
    assert summary(out)["status"] == "OPERATIONAL_FAILURE"
    with zipfile.ZipFile(out / "out.zip") as bundle:
        assert bundle.read("evidence/partial.bin") == b"retain exact partial bytes"


def test_hardlink_preflight_before_gpu_or_compiler(upload, tmp_path, monkeypatch):
    def no_links(*args):
        raise OSError("hardlinks unavailable")
    monkeypatch.setattr(vm, "hardlink_preflight", no_links)
    calls = []
    out = tmp_path / "no-links"
    assert vm.run(upload, out, execute=fake_execute(calls=calls)) == 0
    assert calls == []
    assert "hardlinks unavailable" in summary(out)["errors"][0]


def test_bounded_command_normal_and_timeout_reaps_owned_process(tmp_path):
    log = tmp_path / "normal.log"
    assert vm.command([sys.executable,"-c","print('normal output')"], log, 3, tmp_path) == 0
    assert "normal output" in log.read_text()
    pidfile = tmp_path / "pid"
    code = "import os,time; from pathlib import Path; Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(30)"
    assert vm.command([sys.executable,"-c",code], tmp_path / "timeout.log", .2, tmp_path) == 124
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_archive_preserves_pre1980_files_without_timestamp_failure(tmp_path):
    work = tmp_path / "evidence"
    work.mkdir()
    raw = work / "old.bin"
    raw.write_bytes(b"exact historical raw bytes")
    summary_path = tmp_path / "attempt_summary.json"
    summary_path.write_text('{"fixture":true}')
    os.utime(raw, (1, 1))
    os.utime(summary_path, (1, 1))
    archive = tmp_path / "out.zip"
    vm.package(work, summary_path, archive)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("evidence/old.bin") == b"exact historical raw bytes"
        assert bundle.getinfo("evidence/old.bin").date_time[0] == 1980
        assert bundle.testzip() is None
