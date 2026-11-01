"""Atomic complete generations for the opt-in S-full checkpoint I/O amendment.

publish is called synchronously by the training save function after all four
legacy files are closed. Pointer/retention changes use LOCK; pinned payload
validation uses a separate lease lock so it cannot block publication.
No optimizer arithmetic or random-number generator is implemented here.
"""

from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import uuid

FILES = ("model.safetensors", "optim.safetensors", "state.txt", "events.jsonl")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(root, exclusive=True):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("checkpoint root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "LOCK").open("a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield root


def phase(root, event, **details):
    """Durable operational evidence; never a cadence heartbeat or integrity verdict."""
    record = dict(event=event, unix=time.time(), monotonic=time.monotonic(), pid=os.getpid(), **details)
    root = Path(root)
    if root.is_symlink():
        raise ValueError("checkpoint root cannot be a symlink")
    path = root / "phases.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        data = (json.dumps(record, sort_keys=True) + "\n").encode()
        if os.write(fd, data) != len(data):
            raise OSError("incomplete checkpoint phase record")
        os.fsync(fd)
    finally:
        os.close(fd)
    if event.startswith("publish."):
        # The normal training log is a fallback copy of operational telemetry.
        # A closed stdout pipe must not invalidate a durable checkpoint.
        try:
            print("S_FULL_CHECKPOINT_PHASE " + data.decode().rstrip(), flush=True)
        except (BrokenPipeError, OSError):
            pass


@contextmanager
def lease_locked(root):
    # Serializes lease users only. Publishers never acquire this lock, so a
    # slow full-payload verifier cannot block durable publication.
    root = Path(root)
    if root.is_symlink():
        raise ValueError("checkpoint root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "LEASE_LOCK").open("a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield root


def tensor_inventory(path):
    size = Path(path).stat().st_size
    with Path(path).open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors length")
        n = struct.unpack("<Q", raw)[0]
        if n < 2 or n > min(16 << 20, size - 8):
            raise ValueError("invalid safetensors header length")
        header = json.loads(f.read(n))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("empty checkpoint tensor set")
    spans = []
    for name, item in tensors.items():
        if item.get("dtype") != "F32" or len(item.get("shape", [])) != 2:
            raise ValueError("expected Matrix F32 checkpoint")
        shape = item["shape"]
        if any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("invalid tensor shape")
        offsets = item.get("data_offsets")
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
        ):
            raise ValueError("invalid tensor offsets")
        a, b = offsets
        if a < 0 or b - a != 4 * shape[0] * shape[1] or b > size - 8 - n:
            raise ValueError("truncated or malformed tensor payload")
        spans.append((a, b))
    spans.sort()
    if (
        spans[0][0] != 0
        or any(x[1] != y[0] for x, y in zip(spans, spans[1:]))
        or spans[-1][1] != size - 8 - n
    ):
        raise ValueError("overlapping, missing or trailing tensor payload")
    return {k: v["shape"] for k, v in tensors.items()}


def inspect_flat(folder, *, expected_step=None):
    folder = Path(folder)
    for name in FILES:
        if not (folder / name).is_file() or (folder / name).is_symlink():
            raise ValueError("complete regular weights/optimizer/state required")
    lines = (folder / "state.txt").read_text().splitlines()
    if len(lines) != 2:
        raise ValueError("exact complete two-line state required")
    step = int(lines[0])
    state = json.loads(lines[1])
    if step < 1 or (expected_step is not None and step != expected_step):
        raise ValueError("checkpoint step differs")
    if (
        type(state.get("adamw_t")) is not list
        or len(state["adamw_t"]) != 1
        or type(state["adamw_t"][0]) is not int
        or state["adamw_t"] != [step]
    ):
        raise ValueError("full single-AdamW timestep required; cold resume refused")
    model = tensor_inventory(folder / "model.safetensors")
    optim = tensor_inventory(folder / "optim.safetensors")
    m = {
        k.replace("adamw.0.m.", "", 1): v
        for k, v in optim.items()
        if k.startswith("adamw.0.m.")
    }
    v = {
        k.replace("adamw.0.v.", "", 1): v
        for k, v in optim.items()
        if k.startswith("adamw.0.v.")
    }
    if (
        not m
        or set(m) != {str(i) for i in range(len(m))}
        or m != v
        or len(optim) != 2 * len(m)
        or sorted(map(tuple, m.values())) != sorted(map(tuple, model.values()))
    ):
        raise ValueError("optimizer moment inventory differs from complete model")
    return step


def current_unlocked(root, pointer="CURRENT", *, metadata_only=False):
    if pointer not in ("CURRENT", "INFLIGHT"):
        raise ValueError("unknown pointer")
    link = Path(root) / pointer
    if not link.exists() and not link.is_symlink():
        return None
    if not link.is_symlink():
        raise ValueError("CURRENT must be an atomic generation pointer")
    target = os.readlink(link)
    if Path(target).name != target or not target.startswith("generation-"):
        raise ValueError("unsafe generation pointer")
    folder = Path(root) / target
    return generation_metadata(folder) if metadata_only else validate_generation(folder)


def generation_metadata(folder):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("generation must be a regular directory")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema") != 2 or set(manifest.get("files", {})) != set(FILES):
        raise ValueError("complete generation manifest required")
    rng = manifest.get("rng", {})
    if (
        set(rng) != {"engine", "initial_seed", "draws", "recipe_seed", "batch", "accum"}
        or rng.get("engine") != "std::mt19937"
        or any(
            type(rng.get(k)) is not int
            for k in ["initial_seed", "draws", "recipe_seed", "batch", "accum"]
        )
        or rng["batch"] < 1
        or rng["accum"] < 1
        or rng["recipe_seed"] < 0
        or rng["initial_seed"] != 123 + 1000003 * rng["recipe_seed"]
        or rng["draws"] != manifest["step"] * rng["batch"] * rng["accum"]
        or type(manifest.get("source_commit")) is not str
        or len(manifest["source_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in manifest["source_commit"])
    ):
        raise ValueError("RNG reconstruction or source binding differs")
    inspect_flat(folder, expected_step=manifest["step"])
    return folder, manifest


def validate_generation(folder):
    folder, manifest = generation_metadata(folder)
    for name, want in manifest["files"].items():
        p = folder / name
        if p.stat().st_size != want["bytes"] or sha(p) != want["sha256"]:
            raise ValueError("checkpoint payload or generation hash differs")
    inspect_flat(folder, expected_step=manifest["step"])
    return folder, manifest


def cleanup_unlocked(root):
    keep = {
        os.readlink(Path(root) / n)
        for n in ["CURRENT", "INFLIGHT"]
        if (Path(root) / n).is_symlink()
    }
    for p in Path(root).iterdir():
        if p.name.startswith(("generation-", ".stage-")) and p.name not in keep:
            if p.is_symlink() or not p.is_dir():
                raise ValueError("unexpected checkpoint scratch path")
            shutil.rmtree(p)


def require_space(root, paths):
    required = sum(Path(p).stat().st_size for p in paths) + (64 << 20)
    if shutil.disk_usage(root).free < required:
        raise ValueError("insufficient free space for complete temporary generation")


def pin_inflight(root):
    with lease_locked(root) as folder:
        phase(folder, "pin.lock_wait")
        with locked(folder):
            phase(folder, "pin.lock_acquired")
            item = current_unlocked(folder, "INFLIGHT", metadata_only=True)
            if item is None:
                item = current_unlocked(folder, metadata_only=True)
                if item is None:
                    raise ValueError("no committed checkpoint to pin")
                temp = folder / (".pin-" + uuid.uuid4().hex)
                os.symlink(item[0].name, temp)
                os.replace(temp, folder / "INFLIGHT")
                sync_dir(folder)
            expected = sha(item[0] / "manifest.json")
        # INFLIGHT protects the generation throughout validation, including
        # failure. Another lease user cannot release it while this runs.
        phase(folder, "pin.validation_start", generation=item[0].name)
        checked = validate_generation(item[0])
        if checked[1] != item[1] or sha(item[0] / "manifest.json") != expected:
            raise ValueError("pinned manifest changed during validation")
        phase(folder, "pin.validation_complete", generation=item[0].name)
        return checked


def release_inflight(root, expected_manifest_sha):
    with lease_locked(root) as folder:
        phase(folder, "release.lock_wait")
        with locked(folder):
            phase(folder, "release.lock_acquired")
            item = current_unlocked(folder, "INFLIGHT", metadata_only=True)
            if item is None or sha(item[0] / "manifest.json") != expected_manifest_sha:
                raise ValueError("inflight acknowledgement differs")
        phase(folder, "release.validation_start", generation=item[0].name)
        checked = validate_generation(item[0])
        if checked[1] != item[1]:
            raise ValueError("pinned manifest changed during validation")
        phase(folder, "release.validation_complete", generation=item[0].name)
        with locked(folder):
            now = current_unlocked(folder, "INFLIGHT", metadata_only=True)
            if now is None or now[0] != item[0] or sha(now[0] / "manifest.json") != expected_manifest_sha:
                raise ValueError("inflight acknowledgement differs")
            (folder / "INFLIGHT").unlink()
            sync_dir(folder)
            cleanup_unlocked(folder)
        phase(folder, "release.complete")


def publish(folder, step, seed, batch, accum, source):
    folder = Path(folder)
    if (
        any(type(v) is not int or v < 1 for v in [step, batch, accum])
        or type(seed) is not int
        or seed < 0
        or len(source) != 40
        or any(c not in "0123456789abcdef" for c in source)
    ):
        raise ValueError("invalid checkpoint recipe binding")
    inspect_flat(folder, expected_step=step)
    checkpoint_root = folder / ".checkpoint_v2"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    phase(checkpoint_root, "publish.lock_wait", step=step)
    with locked(checkpoint_root) as root:
        phase(root, "publish.lock_acquired", step=step)
        # Prior immutable publisher output is inspected for identity/step only.
        # All new payloads and consumption/import/restore boundaries retain
        # complete validation; this does not assert prior-payload integrity.
        old = current_unlocked(root, metadata_only=True)
        if old and (old[1]["source_commit"] != source or any(
            old[1]["rng"][key] != value for key, value in
            [("recipe_seed", seed), ("batch", batch), ("accum", accum)]
        )):
            raise ValueError("checkpoint source/recipe drift")
        if old and step <= old[1]["step"]:
            raise ValueError("checkpoint step must advance")
        cleanup_unlocked(root)
        require_space(root, [folder / n for n in FILES])
        stage = root / (".stage-" + uuid.uuid4().hex)
        stage.mkdir()
        manifest = {
            "schema": 2,
            "step": step,
            "source_commit": source,
            "created_unix": time.time(),
            "rng": {
                "engine": "std::mt19937",
                "initial_seed": 123 + 1000003 * seed,
                "draws": step * batch * accum,
                "recipe_seed": seed,
                "batch": batch,
                "accum": accum,
            },
            "files": {},
        }
        for name in FILES:
            phase(root, "publish.copy_fsync_start", step=step, file=name)
            target = stage / name
            with (folder / name).open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 4 << 20)
                dst.flush()
                os.fsync(dst.fileno())
            phase(root, "publish.copy_fsync_complete", step=step, file=name)
            phase(root, "publish.hash_start", step=step, file=name)
            manifest["files"][name] = {
                "bytes": target.stat().st_size,
                "sha256": sha(target),
            }
            phase(root, "publish.hash_complete", step=step, file=name)
        inspect_flat(stage, expected_step=step)
        manifest["committed_unix"] = time.time()
        with (stage / "manifest.json").open("x") as f:
            json.dump(manifest, f, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        sync_dir(stage)
        generation = root / ("generation-" + uuid.uuid4().hex)
        os.rename(stage, generation)
        temp = root / (".pointer-" + uuid.uuid4().hex)
        os.symlink(generation.name, temp)
        os.replace(temp, root / "CURRENT")
        sync_dir(root)
        phase(root, "publish.current_durable", step=step, generation=generation.name)
        cleanup_unlocked(root)
        return manifest


def copy_current(source_root, destination_root):
    """Atomic remote->home or home->remote bundle publication; no cold fallback."""
    source_root, destination_root = map(Path, (source_root, destination_root))
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("source and destination must differ")
    with locked(source_root, False) as src:
        item = current_unlocked(src)
        if item is None:
            raise ValueError("no committed source checkpoint")
        source, manifest = item
        with locked(destination_root) as dst:
            old = current_unlocked(dst)
            if old:
                if (
                    old[1]["source_commit"] != manifest["source_commit"]
                    or old[1]["rng"]["recipe_seed"] != manifest["rng"]["recipe_seed"]
                ):
                    raise ValueError("checkpoint recipe/source differs")
                if old[1]["step"] > manifest["step"]:
                    return {"kept_newer": True, "step": old[1]["step"]}
                if old[1]["step"] == manifest["step"]:
                    if old[1] != manifest:
                        raise ValueError("same-step different generation refused")
                    return {"already_present": True, "step": manifest["step"]}
            cleanup_unlocked(dst)
            require_space(dst, [source / n for n in [*FILES, "manifest.json"]])
            stage = dst / (".stage-" + uuid.uuid4().hex)
            stage.mkdir()
            for name in [*FILES, "manifest.json"]:
                with (source / name).open("rb") as a, (stage / name).open("xb") as b:
                    shutil.copyfileobj(a, b, 4 << 20)
                    b.flush()
                    os.fsync(b.fileno())
            for name, want in manifest["files"].items():
                if sha(stage / name) != want["sha256"]:
                    raise ValueError("interrupted or mixed transfer")
            inspect_flat(stage, expected_step=manifest["step"])
            sync_dir(stage)
            generation = dst / ("generation-" + uuid.uuid4().hex)
            os.rename(stage, generation)
            temp = dst / (".pointer-" + uuid.uuid4().hex)
            os.symlink(generation.name, temp)
            os.replace(temp, dst / "CURRENT")
            sync_dir(dst)
            cleanup_unlocked(dst)
            return {"copied": True, "step": manifest["step"]}


def commit_import(root, stage):
    root, stage = map(Path, (root, stage))
    if (
        stage.parent.resolve() != root.resolve()
        or not stage.name.startswith(".stage-")
        or stage.is_symlink()
    ):
        raise ValueError("import stage outside checkpoint root")
    _, manifest = validate_generation(stage)
    with locked(root) as dst:
        old = current_unlocked(dst)
        if old:
            if old[1]["source_commit"] != manifest["source_commit"] or old[1][
                "rng"
            ] != {**manifest["rng"], "draws": old[1]["rng"]["draws"]}:
                raise ValueError("import recipe/source differs")
            if old[1]["step"] > manifest["step"]:
                shutil.rmtree(stage)
                return {"kept_newer": True, "step": old[1]["step"]}
            if old[1]["step"] == manifest["step"]:
                if old[1] != manifest:
                    raise ValueError("same-step different generation refused")
                shutil.rmtree(stage)
                return {"already_present": True, "step": manifest["step"]}
        generation = dst / ("generation-" + uuid.uuid4().hex)
        os.rename(stage, generation)
        temp = dst / (".pointer-" + uuid.uuid4().hex)
        os.symlink(generation.name, temp)
        os.replace(temp, dst / "CURRENT")
        sync_dir(dst)
        cleanup_unlocked(dst)
        return {"copied": True, "step": manifest["step"]}


def restore_flat(root, folder, expected_source):
    """Call before starting the training process. Published generations remain authority."""
    with locked(root, False) as src:
        item = current_unlocked(src)
        if item is None:
            raise ValueError("no complete resume generation; refuse cold start")
        generation, manifest = item
        if manifest["source_commit"] != expected_source:
            raise ValueError("resume source differs")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        require_space(folder, [generation / n for n in FILES])
        for name in FILES:
            temp = folder / (name + ".restore")
            with (generation / name).open("rb") as a, temp.open("wb") as b:
                shutil.copyfileobj(a, b, 4 << 20)
                b.flush()
                os.fsync(b.fileno())
            os.replace(temp, folder / name)
        sync_dir(folder)
        inspect_flat(folder, expected_step=manifest["step"])
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["publish"])
    p.add_argument("--out", required=True)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--batch", required=True, type=int)
    p.add_argument("--accum", required=True, type=int)
    p.add_argument("--source", required=True)
    a = p.parse_args()
    targets = os.environ.get("S_FULL_CHECKPOINT_TARGETS")
    if targets is not None:
        allowed = json.loads(targets)
        if not isinstance(allowed, list) or not all(
            isinstance(x, str) for x in allowed
        ):
            raise ValueError("invalid checkpoint target scope")
        if str(Path(a.out).resolve()) not in allowed:
            return
    print(json.dumps(publish(a.out, a.step, a.seed, a.batch, a.accum, a.source)))


if __name__ == "__main__":
    main()
