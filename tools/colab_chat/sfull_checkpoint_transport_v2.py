"""Opt-in full checkpoint relay through the existing normal runner transport."""

from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import sfull_checkpoint_v2 as cp

CONTROL = "/content/sfull_control/tools/colab_chat"


def config(job):
    value = job.get("full_checkpoint_v2")
    if not value:
        return None
    if (
        set(value)
        != {
            "relative",
            "every_steps",
            "max_gap_seconds",
            "steps",
            "source_commit",
            "seed",
            "batch",
            "accum",
        }
        or value["relative"] not in (".", "sft")
        or value["every_steps"] != 25
        or value["max_gap_seconds"] != 180
        or value["seed"] not in (1, 2, 3)
        or value["batch"] != 16
        or value["accum"] != 1
        or value["steps"] not in (10000, 1500)
        or not re.fullmatch("[0-9a-f]{40}", value["source_commit"])
    ):
        raise ValueError("unsupported full checkpoint contract")
    if any(
        type(value[k]) is not int
        for k in ["every_steps", "max_gap_seconds", "steps", "seed", "batch", "accum"]
    ):
        raise ValueError("integer checkpoint contract required")
    if (
        job["name"].endswith(str(value["seed"])) is False
        or (
            job["name"].startswith("pt_")
            and (value["relative"] != "." or value["steps"] != 10000)
        )
        or (
            job["name"].startswith("post_")
            and (value["relative"] != "sft" or value["steps"] != 1500)
        )
    ):
        raise ValueError("checkpoint job/phase binding differs")
    if not re.fullmatch("(pt|post)_c12_Sfull_r[123]", job["name"]):
        raise ValueError("unsupported checkpoint job")
    return value


def root_for(base, job):
    c = config(job)
    return Path(base) / job["name"] / c["relative"] / ".checkpoint_v2"


def prefix_for(job, manifest_sha):
    c = config(job)
    rel = "root" if c["relative"] == "." else "sft"
    return "S_FULL_V2__" + job["name"] + "__" + rel + "__" + manifest_sha


def home_uploads(jobs, art):
    result = []
    leases = []
    for job in jobs:
        if not config(job):
            continue
        root = root_for(art, job)
        if not (root / "CURRENT").is_symlink():
            continue
        folder, manifest = cp.pin_inflight(root)
        check_manifest(job, manifest)
        digest = cp.sha(folder / "manifest.json")
        prefix = prefix_for(job, digest)
        result.extend(
            (prefix + "." + n, str(folder / n)) for n in [*cp.FILES, "manifest.json"]
        )
        leases.append((root, digest))
    return result, leases


def release_uploads(leases):
    for root, digest in leases:
        cp.release_inflight(root, digest)


def check_manifest(job, manifest):
    c = config(job)
    if (
        manifest["source_commit"] != c["source_commit"]
        or manifest["rng"]["recipe_seed"] != c["seed"]
        or manifest["rng"]["batch"] != c["batch"]
        or manifest["rng"]["accum"] != c["accum"]
        or manifest["step"] > c["steps"]
    ):
        raise ValueError("checkpoint source/recipe/budget differs")


def rpc(exec_py, expression):
    code = (
        "import sys,json\nsys.path.insert(0,"
        + repr(CONTROL)
        + ")\nimport sfull_checkpoint_v2 as cp\n"
        + "try:\n exec(" + repr(expression) + ")\n"
        + "except cp.PinnedGenerationValidationError as error:\n"
        + " print('S_FULL_RPC_ERROR '+json.dumps({'kind':'PINNED_GENERATION_VALIDATION_FAILED','error':str(error)}))\n"
        + " raise SystemExit(3)\n"
    )
    rc, out = exec_py(code)
    failures = [line[len("S_FULL_RPC_ERROR "):] for line in out.splitlines()
                if line.startswith("S_FULL_RPC_ERROR ")]
    if rc != 0 and len(failures) == 1:
        failure = json.loads(failures[0])
        if failure.get("kind") == "PINNED_GENERATION_VALIDATION_FAILED":
            raise cp.PinnedGenerationValidationError(failure["error"])
    rows = [
        line[len("S_FULL_RPC ") :]
        for line in out.splitlines()
        if line.startswith("S_FULL_RPC ")
    ]
    if rc != 0 or len(rows) != 1:
        raise RuntimeError("checkpoint RPC failed: " + out[-300:])
    return json.loads(rows[0])


def capture_phase_snapshot(download, remote, home):
    """Best-effort diagnostics only, after verified import and lease release.

    Neither a snapshot nor capture progress proves checkpoint integrity or
    resets the remote durable-generation cadence monitor.
    """
    home = Path(home)
    temp = home / "phases.snapshot.tmp"
    try:
        if temp.is_symlink():
            raise ValueError("telemetry scratch cannot be a symlink")
        if download(str(Path(remote) / "phases.jsonl"), str(temp), timeout=60) != 0:
            raise OSError("phase telemetry download failed")
        if not temp.is_file() or temp.is_symlink():
            raise ValueError("phase telemetry snapshot missing")
        raw = temp.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("incomplete phase telemetry snapshot")
        for line in raw.splitlines():
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                raise ValueError("invalid phase telemetry record")
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, home / "phases.snapshot.jsonl")
        cp.sync_dir(home)
        return {"status": "CAPTURED", "diagnostic_only": True,
                "bytes": len(raw), "sha256": cp.sha(home / "phases.snapshot.jsonl")}
    except Exception as error:
        # Preserve any previous complete snapshot and the valid checkpoint.
        # The state receipt and ordinary runner log retain a capture failure.
        record = {"status": "CAPTURE_FAILED", "diagnostic_only": True,
                  "error_type": type(error).__name__}
        try:
            print("S_FULL_CHECKPOINT_TELEMETRY " + json.dumps(record), flush=True)
        except (BrokenPipeError, OSError):
            pass
        return record



def relay(exec_py, download, st, job, art):
    config(job)
    remote = str(root_for("/content/chat", job))
    home = root_for(art, job)
    local = cp.current_unlocked(home) if (home / "CURRENT").is_symlink() else None
    if local:
        check_manifest(job, local[1])
    last = local[1]["step"] if local else 0
    expression = f"""root=cp.Path({remote!r})
if not (root/'CURRENT').is_symlink():
 print('S_FULL_RPC '+json.dumps({{'pending':True}}))
elif not (root/'INFLIGHT').is_symlink() and json.loads((root/'CURRENT'/'manifest.json').read_text())['step'] <= {last}:
 print('S_FULL_RPC '+json.dumps({{'pending':True}}))
else:
 folder,manifest=cp.pin_inflight(root)
 print('S_FULL_RPC '+json.dumps({{'folder':str(folder),'manifest':manifest,'manifest_sha256':cp.sha(folder/'manifest.json')}}))"""
    r = rpc(exec_py, expression)
    if r.get("pending"):
        return False
    manifest = r["manifest"]
    check_manifest(job, manifest)
    folder = Path(r["folder"])
    if folder.parent != Path(remote) or not folder.name.startswith("generation-"):
        raise ValueError("remote generation path differs")
    home.mkdir(parents=True, exist_ok=True)
    stage = home / (".stage-transfer-" + r["manifest_sha256"])
    stage.mkdir(exist_ok=True)
    cp.require_space(home, [])  # at least metadata and safety margin
    required = sum(v["bytes"] for v in manifest["files"].values()) + (64 << 20)
    if shutil.disk_usage(home).free < required:
        raise ValueError("insufficient disk for full home checkpoint")
    for name in [*cp.FILES, "manifest.json"]:
        if download(str(folder / name), str(stage / name), timeout=2400) != 0:
            return True
    if cp.sha(stage / "manifest.json") != r["manifest_sha256"]:
        raise ValueError("relay manifest differs")
    imported = cp.commit_import(home, stage)
    rpc(
        exec_py,
        f"cp.release_inflight({remote!r},{r['manifest_sha256']!r})\nprint('S_FULL_RPC '+json.dumps({{'released':True}}))",
    )
    st.d.setdefault("full_checkpoint_v2", {})[job["name"]] = {
        "step": imported["step"],
        "manifest_sha256": r["manifest_sha256"],
        "full_optimizer": True,
        "rng_draws": manifest["rng"]["draws"],
        "phase_snapshot": capture_phase_snapshot(download, remote, home),
    }
    st.save()
    return True


def recover_remote(jobs, chat, upload):
    """After input reassembly and before jobs_vm, import only complete home bundles."""
    chat, upload = Path(chat), Path(upload)
    result = []
    for job in jobs:
        c = config(job)
        if not c:
            continue
        root = root_for(chat, job)
        head = prefix_for(job, "")
        for path in sorted(upload.glob(head + "*.manifest.json")):
            digest = path.name[len(head) : -len(".manifest.json")]
            if path.is_symlink() or not path.is_file():
                raise ValueError("home manifest must be regular")
            if not re.fullmatch("[0-9a-f]{64}", digest) or cp.sha(path) != digest:
                raise ValueError("home manifest filename/binding differs")
            manifest = json.loads(path.read_text())
            check_manifest(job, manifest)
            root.mkdir(parents=True, exist_ok=True)
            stage = root / ".stage-import"
            stage.mkdir(exist_ok=True)
            prefix = prefix_for(job, digest)
            cp.require_space(
                root,
                [upload / (prefix + "." + n) for n in [*cp.FILES, "manifest.json"]],
            )
            for name in [*cp.FILES, "manifest.json"]:
                source = upload / (prefix + "." + name)
                if source.is_symlink() or not source.is_file():
                    raise ValueError("incomplete home upload")
                shutil.copyfile(source, stage / name)
                with (stage / name).open("rb") as f:
                    cp.os.fsync(f.fileno())
            result.append(cp.commit_import(root, stage))
            for name in [*cp.FILES, "manifest.json"]:
                (upload / (prefix + "." + name)).unlink()
        folder = chat / job["name"] / c["relative"]
        if (root / "CURRENT").is_symlink():
            m = cp.restore_flat(root, folder, c["source_commit"])
            check_manifest(job, m)
        elif any((folder / n).exists() for n in cp.FILES):
            raise ValueError("uncommitted flat checkpoint; cold/mixed resume refused")
    return result


def final_home_ready(job, art):
    """A finished process is not home until its final full state is verified."""
    c = config(job)
    if not c:
        return True
    root = root_for(art, job)
    if not (root / "CURRENT").is_symlink():
        return False
    with cp.locked(root, False):
        _, manifest = cp.current_unlocked(root)
        check_manifest(job, manifest)
        return manifest["step"] == c["steps"]
