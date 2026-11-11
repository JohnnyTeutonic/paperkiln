#!/usr/bin/env python3
"""Home side, once per campaign: put the files a VM cannot make into the checkpoint store's `main` branch.

    python3 tools/colab_chat/base2_seed.py --store-repo OWNER/NAME --store-token-file PATH \\
        --weights /mnt/c/ml_artifacts/tiny_chat/b1_r1/pt_b1_r1/tc_b1_r1.safetensors \\
        --vocab /mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf \\
        --engine-dir .../AI_ML/chat_screens_claude/scenario_engine --record base2_seed_record.json

Three sets, each one commit with a manifest (hf_store.py): seed/weights (arm C's start, about 291 MB through the
home uplink, 9 to 13 minutes), seed/vocab, and seed/scenario_engine (the engine's Python files, tests left out). A
set already in the store with the same sha256 of every file is not sent again; one that differs is refused, never
overwritten (--replace sends it anyway). The record lists every file's size and sha256, for base2_pins.json. There
is no default repo and no default token path. --dry-run hashes and prints the plan without opening the store.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_store  # noqa: E402


def engine_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".py") and not f.startswith("test_"))


def plan(a):
    sets = []
    if a.weights:
        sets.append(("seed/weights", [a.weights]))
    if a.vocab:
        sets.append(("seed/vocab", [a.vocab]))
    if a.engine_dir:
        sets.append(("seed/scenario_engine", [os.path.join(a.engine_dir, f) for f in engine_files(a.engine_dir)]))
    out = []
    for prefix, paths in sets:
        files = {}
        for p in paths:
            sha, n = hf_store.sha256_file(p)
            files[os.path.basename(p)] = {"bytes": n, "sha256": sha, "source": os.path.abspath(p)}
        if not files:
            raise hf_store.StoreError(f"{prefix}: no files")
        out.append((prefix, files))
    return out


def seed(store, sets, replace=False, log=print):
    """Push each set unless the store holds the same bytes. Returns {prefix: "pushed" | "present"}."""
    tree = store.b.tree("main") or {}
    done = {}
    for prefix, files in sets:
        if f"{prefix}/{hf_store.MANIFEST}" in tree and not replace:
            have = store.read_manifest("main", prefix)["files"]
            same = {f: (v["bytes"], v["sha256"]) for f, v in have.items()} == \
                   {f: (v["bytes"], v["sha256"]) for f, v in files.items()}
            if not same:
                raise hf_store.StoreError(f"{prefix} is in the store with other bytes; nothing overwritten (--replace)")
            done[prefix] = "present"
            log(f"{prefix}: already in the store, same sha256; not sent")
            continue
        with tempfile.TemporaryDirectory(prefix="base2_seed.") as tmp:
            for f, v in files.items():  # links, so a 291 MB file is not copied to be sent
                try:
                    os.symlink(v["source"], os.path.join(tmp, f))
                except OSError:
                    shutil.copyfile(v["source"], os.path.join(tmp, f))
            t0 = time.time()
            man = store.push_set("main", prefix, tmp, kind="seed", names=sorted(files))
        for f, v in files.items():
            if (man["files"][f]["bytes"], man["files"][f]["sha256"]) != (v["bytes"], v["sha256"]):
                raise hf_store.StoreError(f"{prefix}/{f} changed while it was sent")
        mb = sum(v["bytes"] for v in files.values()) / 1e6
        done[prefix] = "pushed"
        log(f"{prefix}: {len(files)} files, {mb:.1f} MB pushed in {time.time() - t0:.0f} s")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store-repo", required=True, help="OWNER/NAME of the store repo (no default)")
    ap.add_argument("--store-token-file", help="file holding the write token, mode 600 (no default)")
    ap.add_argument("--store-repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--weights")
    ap.add_argument("--vocab")
    ap.add_argument("--engine-dir")
    ap.add_argument("--record", required=True, help="where the JSON record of sizes and hashes is written")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    token = None
    try:
        sets = plan(a)
        rec = {"repo": a.store_repo, "repo_type": a.store_repo_type, "branch": "main", "at": time.time(),
               "sets": {p: {f: {"bytes": v["bytes"], "sha256": v["sha256"]} for f, v in files.items()} for p, files in sets}}
        if a.dry_run:
            print(json.dumps(rec, indent=1))
            return 0
        store = hf_store.open_store({"repo": a.store_repo, "repo_type": a.store_repo_type,
                                     "token_file": a.store_token_file})
        token = store.b.token
        rec["result"] = seed(store, sets, a.replace)
    except Exception as e:  # noqa: BLE001 one redacted line
        print(hf_store.redact(f"base2_seed failed: {type(e).__name__}: {e}", token), file=sys.stderr)
        return 1
    with open(a.record + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(a.record + ".tmp", a.record)
    print(f"record: {a.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
