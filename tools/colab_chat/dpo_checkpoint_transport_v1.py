"""Opt-in full checkpoint relay through the existing normal runner transport."""

from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import dpo_checkpoint_core_v1 as cp

CONTROL = "/content/pk/tools/colab_chat"


def config(job):
    value = job.get("dpo_full_checkpoint_v1")
    if value is None:
        return None
    required = {
        "every_steps",
        "max_gap_seconds",
        "steps",
        "source_commit",
        "seed",
        "batch",
        "accum",
        "contract_sha256",
    }
    if set(value) != required or job["name"] != "pk_dpo_v1":
        raise ValueError("unsupported DPO checkpoint contract")
    if any(
        type(value[k]) is not int
        for k in ["every_steps", "max_gap_seconds", "steps", "seed", "batch", "accum"]
    ):
        raise ValueError("integer DPO checkpoint contract required")
    if (
        value["every_steps"] != 10
        or value["max_gap_seconds"] != 180
        or value["steps"] < 1
        or value["seed"] != 1
        or value["batch"] != 8
        or value["accum"] != 1
        or not re.fullmatch("[0-9a-f]{40}", value["source_commit"])
        or not re.fullmatch("[0-9a-f]{64}", value["contract_sha256"])
    ):
        raise ValueError("unsupported DPO checkpoint bounds")
    return value


def root_for(base, job):
    config(job)
    return Path(base) / job["name"] / ".dpo_checkpoint_v1"


def prefix_for(job, manifest_sha):
    config(job)
    return "DPO_FULL_V1__" + job["name"] + "__" + manifest_sha


def home_uploads(jobs, art):
    result = []
    leases = []
    for job in jobs:
        if not config(job):
            continue
        root = root_for(art, job)
        if not (root / "CURRENT").is_symlink():
            folder = Path(art) / job["name"]
            if any(
                (folder / n).exists()
                for n in ["model.safetensors", "optim.safetensors", "state.txt"]
            ):
                raise ValueError("DPO home partial flat checkpoint refused")
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
        manifest["operational_contract_sha256"] != c["contract_sha256"]
        or manifest["source_commit"] != c["source_commit"]
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
        + ")\nimport dpo_checkpoint_core_v1 as cp\n"
        + expression
        + "\n"
    )
    rc, out = exec_py(code)
    rows = [
        line[len("S_FULL_RPC ") :]
        for line in out.splitlines()
        if line.startswith("S_FULL_RPC ")
    ]
    if rc != 0 or len(rows) != 1:
        raise RuntimeError("checkpoint RPC failed: " + out[-300:])
    return json.loads(rows[0])


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
    st.d.setdefault("dpo_full_checkpoint_v1", {})[job["name"]] = {
        "step": imported["step"],
        "manifest_sha256": r["manifest_sha256"],
        "full_optimizer": True,
        "rng_draws": manifest["rng"]["draws"],
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
        folder = chat / job["name"]
        if (root / "CURRENT").is_symlink():
            m = cp.restore_flat(root, folder, c["source_commit"])
            check_manifest(job, m)
        elif any((folder / n).exists() for n in cp.FILES):
            raise ValueError("uncommitted flat checkpoint; cold/mixed resume refused")
    return result


def final_home_ready(job, art):
    c = config(job)
    root = root_for(art, job)
    if not (root / "CURRENT").is_symlink():
        return False
    with cp.locked(root, False) as home:
        item = cp.current_unlocked(home)
        check_manifest(job, item[1])
        return item[1]["step"] == c["steps"]
