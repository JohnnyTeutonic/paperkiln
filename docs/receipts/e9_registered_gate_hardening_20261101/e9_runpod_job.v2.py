"""Run the staged E9 gate on a remote VM; always return explicit diagnostic receipts."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import shutil
import subprocess
import zipfile

FILES = ("ember.tgz", "ember_base.tgz", "inputs.tgz", "e9_gate_vm.sh", "HEAD")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def claim_attempt(outdir, content):
    """No attempt may overwrite or retry existing scientific evidence."""
    outdir, content = Path(outdir), Path(content)
    output = content / "out"
    for directory in (outdir, output):
        if directory.is_symlink() or (directory.exists() and
                (not directory.is_dir() or any(directory.iterdir()))):
            raise ValueError("preserve existing E9 evidence: " + str(directory))
    markers = [content / name for name in
               ("GATE_DONE", "GATE_FAILED", "PHASES", "gate_vm.log", "gate_out.zip")]
    markers.extend(content.glob("gate_out_*.zip"))
    if any(path.exists() or path.is_symlink() for path in markers):
        raise ValueError("preserve existing E9 terminal/partial attempt evidence")
    # jobs_vm creates only the empty job directory; its pid/rc/status/log live
    # outside that directory. O_EXCL serializes concurrent wrapper starts.
    claim = content / ".e9_gate_attempt.json"
    fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"kind": "E9_ONE_DURABLE_ATTEMPT", "started_unix": time.time(),
                   "wrapper_sha256": sha(Path(__file__)), "outdir": str(outdir)}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(content, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    # The durable claim is intentionally never removed, including setup failure.


def run(outdir, content=Path("/content")):
    outdir, content = Path(outdir), Path(content)
    claim_attempt(outdir, content)  # MUST precede every output write and catch block.
    outdir.mkdir(parents=True, exist_ok=True)
    output = content / "out"
    output.mkdir(exist_ok=True)
    execution = {"gate_pass": False, "process_exit_code": None, "complete_marker": False}
    try:
        upload = content / "chat/upload"
        manifest = json.loads((upload / "STAGING.json").read_text())
        if (manifest.get("formats") != ["f16"] or manifest.get("deadline_minutes") != 95
                or manifest.get("phases") != ["g1", "g2", "g3", "g4", "tfd_qwen3", "tfd_qwen25"]):
            raise ValueError("campaign scope/deadline differs from staged E9 plan")
        for name in FILES:
            source = upload / name
            expected = manifest["files"][name]
            if source.stat().st_size != expected["bytes"] or sha(source) != expected["sha256"]:
                raise ValueError("staged hash mismatch: " + name)
            shutil.copyfile(source, content / name)
        shutil.copyfile(upload / "STAGING.json", output / "STAGING.json")
        baseline = (content / "HEAD").read_text().strip()
        if baseline != manifest["baseline_ember_commit"]:
            raise ValueError("baseline commit differs from manifest")
        phases = ",".join(manifest["phases"])
        env = dict(os.environ, KVQ_FORMATS="f16", KVQ_BUILD_JOBS="4")
        result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=15s", "95m", "bash",
                                 str(content / "e9_gate_vm.sh"), baseline, phases], env=env, check=False)
        execution["process_exit_code"] = result.returncode
        execution["complete_marker"] = (content / "GATE_DONE").exists()
        report = output / "gate_summary.json"
        summary = json.loads(report.read_text()) if report.exists() else {"pass": False, "errors": ["missing gate summary"]}
        passed = result.returncode == 0 and execution["complete_marker"] and summary.get("pass") is True
        if not passed:
            summary["pass"] = False
            summary.setdefault("errors", []).append("gate process failed, timed out, or lacks completion marker")
            failure = content / "GATE_FAILED"
            if failure.exists():
                reason = failure.read_text(errors="replace").strip()
                summary["errors"].append(reason)
                summary["classification"] = "BASELINE_UNSTABLE" if "BASELINE_UNSTABLE" in reason else "FAILED"
        execution["gate_pass"] = passed
    except (OSError, ValueError, KeyError) as exc:
        summary = {"pass": False, "errors": [str(exc)]}
    (output / "gate_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (output / "execution.json").write_text(json.dumps(execution, indent=1) + "\n")
    for name in ("gate_summary.json", "execution.json"):
        shutil.copyfile(output / name, outdir / name)
    log = content / "gate_vm.log"
    (outdir / "vm.log").write_text(log.read_text(errors="replace") if log.exists() else "VM gate did not start\n")
    with zipfile.ZipFile(outdir / "out.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in output.rglob("*"):
            if source.is_file() and source.suffix != ".bin":
                archive.write(source, str(source.relative_to(output)))
    print(json.dumps(execution))
    # Command completion means diagnostics are home. Only gate_summary.pass is a gate verdict.
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    raise SystemExit(run(parser.parse_args().outdir))
