"""Supervised E10 component job: original-only baseline, then optional candidate."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    upload = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/content/chat/upload")
    manifest = json.loads((upload / "e10_component_manifest.json").read_text())
    archive = upload / "e10_component_source.tar"
    result = {
        "status": "HALTED",
        "scope": "component only; no improvement/adoption inference",
        "source_commit": manifest["ember_commit"],
    }
    try:
        if sha(Path(__file__)) != manifest["vm_sha256"]:
            raise ValueError("VM wrapper hash mismatch")
        if sha(archive) != manifest["archive_sha256"]:
            raise ValueError("archive hash mismatch")
        source = output / "source_tree"
        source.mkdir(exist_ok=False)
        with tarfile.open(archive) as tar:
            if any(
                m.name.startswith("/")
                or ".." in Path(m.name).parts
                or not (m.isfile() or m.isdir())
                for m in tar.getmembers()
            ):
                raise ValueError("archive contains unsafe/non-file members")
            tar.extractall(source)
        for name, expected in manifest["source_sha256"].items():
            if sha(source / name) != expected:
                raise ValueError("source hash mismatch: " + name)
        runner = source / "tools/row_pick_profile/run.py"
        for leg in ["baseline", "candidate"]:
            command = [sys.executable, str(runner), "--output", str(output / leg)]
            if leg == "baseline":
                command.append("--baseline-only")
            with (output / (leg + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    timeout=780,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            receipt = json.loads((output / leg / "receipt.json").read_text())
            result[leg] = receipt["status"]
            if receipt["status"] != "COMPLETE":
                break
        result["status"] = (
            "COMPLETE" if result.get("candidate") == "COMPLETE" else "HALTED"
        )
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "vm.py").write_bytes(Path(__file__).read_bytes())
        with zipfile.ZipFile(
            output / "out.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive_out:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path.name != "out.zip":
                    archive_out.write(path, str(path.relative_to(output)))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
