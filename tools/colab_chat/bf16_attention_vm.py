"""One standalone BF16 diagnostic attempt; zero exit means evidence packaged, not PASS."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import zipfile

FILES = ("bf16_attention_probe.cu", "bf16_attention_reference.py", "bf16_attention_check.py",
         "test_bf16_attention_reference.py", "bf16_attention_vm.py")
MANIFEST = "bf16_attention_manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)


def hardlink_preflight(directory):
    fd, name = tempfile.mkstemp(prefix="hardlink-probe-", dir=directory)
    os.close(fd)
    target = name + ".link"
    try:
        os.link(name, target)
        if os.stat(name).st_ino != os.stat(target).st_ino:
            raise RuntimeError("hardlink identity check failed")
    finally:
        if os.path.exists(target):
            os.unlink(target)
        os.unlink(name)


def command(argv, log, timeout, cwd):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONOPTIMIZE": "0"}
    with log.open("xb") as output:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def package(work, summary_path, archive):
    fd, temp = tempfile.mkstemp(prefix="out.zip.", dir=archive.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, strict_timestamps=False) as bundle:
            bundle.write(summary_path, "attempt_summary.json")
            for path in sorted(work.rglob("*")):
                if path.is_symlink():
                    raise RuntimeError("refusing symlink in diagnostic evidence")
                if path.is_file():
                    bundle.write(path, "evidence/" + path.relative_to(work).as_posix())
        with zipfile.ZipFile(temp) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError("diagnostic archive CRC check failed")
        os.link(temp, archive)  # never replace an existing/foreign archive
    finally:
        os.unlink(temp)


def run(upload, out, *, execute=command, pack=package):
    upload, out = Path(upload), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path, archive, receipt_path = (out / name for name in
        ("attempt_summary.json", "out.zip", "archive_receipt.json"))
    manifest_path = upload / MANIFEST
    manifest_data = manifest_path.read_bytes() if manifest_path.is_file() else None
    manifest_sha = hashlib.sha256(manifest_data).hexdigest() if manifest_data is not None else None
    # Manager recovery may relay this exact completed attempt, never execute it again.
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old["manifest_sha256"] != manifest_sha or old["archive_sha256"] != sha(archive) or old["summary_sha256"] != sha(summary_path):
            raise RuntimeError("existing diagnostic artifact differs; refusing retry")
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None or bundle.read("attempt_summary.json") != summary_path.read_bytes():
                raise RuntimeError("existing archive is invalid")
        return 0
    if summary_path.exists() or archive.exists():
        raise RuntimeError("partial publication retained; operator review required, no retry")
    work = out / "bf16_attempt"
    interrupted = work.exists()
    work.mkdir(exist_ok=True)
    summary = {"schema": "paperkiln.bf16_attention.attempt.v1", "status": "OPERATIONAL_FAILURE",
               "numerical_pass": False, "manifest_sha256": manifest_sha,
               "commands": [], "errors": [], "checker_returncode": None, "archive_required": True,
               "exit_meaning": "zero exit means diagnostic artifacts published, not numerical PASS",
               "scope": "standalone fixed32 dense MHA cases; explicit BF16 STE surrogate; no production adoption"}
    try:
        if interrupted:
            raise RuntimeError("interrupted attempt preserved; numerical execution is never automatically retried")
        hardlink_preflight(work)
        if manifest_data is None:
            raise RuntimeError("upload manifest is missing")
        manifest = json.loads(manifest_data)
        if manifest.get("schema") != "paperkiln.bf16_attention.upload.v1" or set(manifest["files"]) != set(FILES):
            raise RuntimeError("unexpected upload manifest/inventory")
        source_data = {name: (upload / name).read_bytes() for name in FILES}
        for name in FILES:
            if hashlib.sha256(source_data[name]).hexdigest() != manifest["files"][name]:
                raise RuntimeError("upload SHA mismatch: " + name)
        if sha(__file__) != manifest["files"]["bf16_attention_vm.py"]:
            raise RuntimeError("running worker differs from frozen upload")
        sources = work / "sources"
        sources.mkdir()
        for name in (*FILES, MANIFEST):
            with (sources / name).open("xb") as output:
                output.write(manifest_data if name == MANIFEST else source_data[name])
        summary["source_sha256"] = manifest["files"]

        def call(name, argv, timeout):
            log = work / (name + ".log")
            try:
                rc = execute(argv, log, timeout, work)
            except Exception as exc:
                summary["commands"].append({"name": name, "argv": list(map(str, argv)),
                                            "returncode": None, "error": str(exc),
                                            "log_sha256": sha(log) if log.is_file() else None})
                raise
            summary["commands"].append({"name": name, "argv": list(map(str, argv)), "returncode": rc,
                                        "log_sha256": sha(log)})
            return rc, log.read_text(errors="replace")

        rc, hardware = call("hardware", ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], 30)
        if rc or len(hardware.strip().splitlines()) != 1 or not hardware.startswith("NVIDIA L4,"):
            raise RuntimeError("requires exactly one NVIDIA L4")
        rc, nvcc = call("nvcc", ["nvcc", "--version"], 30)
        if rc or "release 12.4," not in nvcc:
            raise RuntimeError("requires the configured CUDA12.4 compiler")
        rc, _ = call("numpy_install", [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                         "--no-input", "--no-deps", "--index-url", "https://pypi.org/simple", "numpy==1.26.4"], 300)
        if rc:
            raise RuntimeError("NumPy1.26.4 installation failed")
        rc, _ = call("python_numpy", [sys.executable, "-c", "import sys,numpy; print(sys.version); print(numpy.__version__); assert numpy.__version__ == '1.26.4'"], 30)
        if rc:
            raise RuntimeError("NumPy version check failed")
        binary = work / "bf16_attention_probe"
        rc, _ = call("compile", ["nvcc", "-O3", "-std=c++17", "-arch=sm_89",
                               str(sources / "bf16_attention_probe.cu"), "-o", str(binary), "-lcublas"], 600)
        if rc or not binary.is_file():
            raise RuntimeError("standalone CUDA compilation failed")
        summary["binary_sha256"] = sha(binary)
        argv = [sys.executable, str(sources / "bf16_attention_check.py"), "--binary", str(binary),
                "--output", str(work / "results")]
        rc, _ = call("checker", argv, 1800)
        summary["checker_returncode"] = rc
        halted = work / "results/HALTED"
        if rc:
            if halted.is_file():
                summary["status"] = "HALTED"
                summary["halted"] = json.loads(halted.read_text())
            raise RuntimeError("numerical checker did not pass; raw attempt retained")
        report = json.loads((work / "results/report.json").read_text())
        if report.get("status") != "COMPLETE" or len(report.get("cases", [])) != 32:
            raise RuntimeError("fixed32 case report is incomplete")
        rc, _ = call("offline_replay", argv + ["--replay"], 600)
        if rc:
            raise RuntimeError("raw numerical evidence replay failed")
        if sha(binary) != summary["binary_sha256"] or any(sha(sources / name) != manifest["files"][name] for name in FILES):
            raise RuntimeError("source/binary drift before packaging")
        summary.update(status="PASS", numerical_pass=True, report_sha256=sha(work / "results/report.json"))
    except Exception as exc:
        summary["errors"].append(str(exc))
    immutable_json(summary_path, summary)
    pack(work, summary_path, archive)  # packaging errors propagate as job failure
    immutable_json(receipt_path, {"schema": "paperkiln.bf16_attention.archive.v1",
        "manifest_sha256": manifest_sha, "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
        "summary_sha256": sha(summary_path), "attempt_status": summary["status"],
        "numerical_pass": summary["numerical_pass"], "diagnostic_worker_returncode": 0})
    print(json.dumps({"diagnostic_packaged": True, "attempt_status": summary["status"],
                      "numerical_pass": summary["numerical_pass"]}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.upload, args.out))
