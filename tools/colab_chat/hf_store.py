#!/usr/bin/env python3
"""A checkpoint and input store on a private Hugging Face repo, for campaigns that outlive their VMs (base 2).

    python3 hf_store.py push   --config STORE.json --branch main --prefix corpus/N --dir DIR
    python3 hf_store.py pull   --config STORE.json --branch main --prefix corpus/N --to DIR
    python3 hf_store.py inputs --config STORE.json --jobs JOBS.json --root /content
    python3 hf_store.py steps  --config STORE.json --branch ckpt-N
    python3 hf_store.py prune  --config STORE.json --branch ckpt-N [--keep 2]
    python3 hf_store.py adopt-token  --config STORE.json --from FILE --dir DIR
    python3 hf_store.py forget-token --config STORE.json

Layout. A "set" is a folder of the repo with a manifest.json beside its files: the format, the kind, and the size
and sha256 of every file. A push is one commit that holds the files and the manifest, so a set is in the store
whole or not at all; a pull fetches the manifest, then each file, and refuses a file whose size or sha256 differs.

    main      inputs (any prefix), marks/<job>/rung_<step>/ (weights at a mark step). Only grows: prune() refuses it.
    ckpt-*    gens/g<step>/ : one transactional checkpoint generation of one arm (model.safetensors,
              optim.safetensors, state.txt, the trainer's MANIFEST.json, events.jsonl) and its manifest.json.

A resume takes the newest generation that verifies, and falls back to the one before it (pull_generation). prune()
removes generations older than the newest two that are whole in the store's listing, and nothing else.

STORE.json: {"repo": "owner/name", "repo_type": "dataset", "token_file": PATH, "reclaim": true,
"min_push_interval_s": 0, "branches": {job: branch}}. There is no default repo and no default token path. "repo": "dir:/some/folder" is a
store on local disk with the same rules (tests and local rehearsals; no token). A jobs file names the inputs a VM
needs as "store_inputs": [{"branch": "main", "prefix": P, "to": "chat/upload"}], "to" relative to --root.

The token is read from token_file (mode 600, refused otherwise) and handed to the library call by call. It is never
printed, logged, put on a command line or in the environment, and every error text passes through redact().

Assumed of Hugging Face and not tested here (the tests run against DirBackend and a recording stand-in for HfApi):
create_commit is atomic and takes adds and deletes together; list_repo_tree gives each file's size and, for LFS
files, its sha256; super_squash_history plus permanently_delete_lfs_files frees the storage of removed generations;
the account's private storage quota and commit rate limits. The rehearsal measures them.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

FORMAT = "paperkiln.store/1"
MANIFEST = "manifest.json"
GEN_FILES = ("model.safetensors", "optim.safetensors", "state.txt", "MANIFEST.json")  # one trainer generation
GEN_FORMAT = "paperkiln.ckpt_gen/1"  # include/microtorch/checkpoint_txn.hpp
MARK_FILES = ("model.safetensors", "state.txt")  # what jobs_vm.take_snapshot copies
# Repos this tool never writes to or prunes: the screens' and cloud_eval's inputs repo holds pinned revisions.
REFUSED_REPOS = ("JohnnyTeutonic/chat-screens-inputs",)


class StoreError(Exception):
    pass


class VerifyError(StoreError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def read_token(path):
    """The token in a file only its owner can read. Never returned to anything that prints."""
    try:
        st = os.stat(path)
    except OSError:
        raise StoreError(f"token file not found: {path}") from None
    if st.st_mode & 0o077:
        raise StoreError(f"token file {path} is readable by others; chmod 600")
    tok = open(path).read().strip()
    if not tok:
        raise StoreError(f"token file {path} is empty")
    return tok


def redact(text, token):
    text = str(text)
    return text.replace(token, "[token]") if token else text


def gen_step(name):
    return int(name[1:]) if name[:1] == "g" and name[1:].isdigit() else None


# ---------------------------------------------------------------- backends
class DirBackend:
    """The store on local disk: ROOT/<branch>/<path>, commits listed in ROOT/commits.jsonl. A commit copies its
    files in, the manifest last, under a lock. Tests and local rehearsals only."""
    token = None

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _p(self, branch, path=""):
        return os.path.join(self.root, branch, *path.split("/")) if path else os.path.join(self.root, branch)

    def ensure_branch(self, branch):
        os.makedirs(self._p(branch), exist_ok=True)

    def tree(self, branch):
        d = self._p(branch)
        if not os.path.isdir(d):
            return None
        out = {}
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                sha, n = sha256_file(p)
                out[os.path.relpath(p, d).replace(os.sep, "/")] = {"bytes": n, "sha256": sha}
        return out

    def commit(self, branch, adds, deletes, message):
        self.ensure_branch(branch)
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for path, src in sorted(adds.items(), key=lambda kv: kv[0].endswith("/" + MANIFEST) or kv[0] == MANIFEST):
                dst = self._p(branch, path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst + ".incoming")
                os.replace(dst + ".incoming", dst)
            for path in deletes:
                p = self._p(branch, path)
                if os.path.exists(p):
                    os.remove(p)
                    d = os.path.dirname(p)
                    while d != self._p(branch) and not os.listdir(d):
                        os.rmdir(d)
                        d = os.path.dirname(d)
            with open(os.path.join(self.root, "commits.jsonl"), "a") as f:
                f.write(json.dumps({"branch": branch, "adds": sorted(adds), "deletes": sorted(deletes),
                                    "message": message, "at": time.time()}) + "\n")

    def fetch(self, branch, path, dest):
        src = self._p(branch, path)
        if not os.path.isfile(src):
            raise StoreError(f"{branch}:{path} is not in the store")
        shutil.copyfile(src, dest)

    def reclaim(self, branch, dead, protected):
        with open(os.path.join(self.root, "commits.jsonl"), "a") as f:
            f.write(json.dumps({"branch": branch, "reclaim": sorted(set(dead) - set(protected)), "at": time.time()}) + "\n")
        return len(set(dead) - set(protected))


class HfBackend:
    """The same five operations on a Hugging Face repo. `api` is huggingface_hub.HfApi() unless a test passes its own."""

    def __init__(self, repo, repo_type, token, api=None):
        if repo in REFUSED_REPOS:
            raise StoreError(f"{repo} is the pinned inputs repo; the store must be its own repo")
        self.repo, self.repo_type, self.token = repo, repo_type, token
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api = api

    def _kw(self):
        return {"repo_type": self.repo_type, "token": self.token}

    def ensure_branch(self, branch):
        if branch != "main":
            self.api.create_branch(self.repo, branch=branch, exist_ok=True, **self._kw())

    def tree(self, branch):
        try:
            entries = list(self.api.list_repo_tree(self.repo, recursive=True, revision=branch, **self._kw()))
        except Exception as e:  # noqa: BLE001 the library's RevisionNotFoundError, by name so a stand-in can raise it
            if type(e).__name__ == "RevisionNotFoundError":
                return None
            raise
        out = {}
        for e in entries:
            size = getattr(e, "size", None)
            if size is None:  # a folder
                continue
            lfs = getattr(e, "lfs", None)
            sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            out[e.path] = {"bytes": size, "sha256": sha}
        return out

    def commit(self, branch, adds, deletes, message):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        self.ensure_branch(branch)
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=src) for p, src in sorted(adds.items())]
        ops += [CommitOperationDelete(path_in_repo=p) for p in sorted(deletes)]
        self.api.create_commit(self.repo, ops, commit_message=message, revision=branch, **self._kw())

    def fetch(self, branch, path, dest):
        stage = tempfile.mkdtemp(prefix=".hfstage.", dir=os.path.dirname(os.path.abspath(dest)))
        try:
            got = self.api.hf_hub_download(self.repo, path, revision=branch, local_dir=stage, **self._kw())
            os.replace(got, dest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def reclaim(self, branch, dead, protected):
        """Free the storage of files a prune removed from `branch`: squash that branch's history, then delete the
        LFS objects no listed tree still names. `protected` are the sha256 of every file still in a branch."""
        self.api.super_squash_history(self.repo, branch=branch, commit_message="squash: superseded generations removed",
                                      **self._kw())
        gone = set(dead) - set(protected)
        files = [f for f in self.api.list_lfs_files(self.repo, **self._kw()) if getattr(f, "oid", None) in gone]
        if files:
            self.api.permanently_delete_lfs_files(self.repo, files, rewrite_history=False, **self._kw())
        return len(files)


# ---------------------------------------------------------------- the store
class Store:
    def __init__(self, backend, reclaim=False, log=None, branches=()):
        self.b, self.reclaim_on, self._log, self.branches = backend, reclaim, log, set(branches)
        self._manifests = {}

    def log(self, msg):
        msg = redact(msg, self.b.token)
        if self._log:
            self._log(msg)
        else:
            print(msg, flush=True)

    # -- sets
    def push_set(self, branch, prefix, folder, kind="files", names=None, meta=None):
        """Hash `names` (default: every file in folder, not recursive) and commit them with a manifest. One commit."""
        names = sorted(names if names is not None else
                       [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f != MANIFEST])
        if not names:
            raise StoreError(f"nothing to push in {folder}")
        files = {}
        for f in names:
            sha, n = sha256_file(os.path.join(folder, f))
            files[f] = {"bytes": n, "sha256": sha}
        man = {"format": FORMAT, "kind": kind, "files": files, "pushed_at": time.time(), **(meta or {})}
        with tempfile.TemporaryDirectory(prefix="store_manifest.") as tmp:
            mp = os.path.join(tmp, MANIFEST)
            with open(mp, "w") as fh:
                json.dump(man, fh, indent=1)
            # The files are hashed again from the bytes on disk by whoever pulls; a file that changes between the
            # hash above and the upload shows there as a refused pull, never as a silent difference.
            adds = {f"{prefix}/{f}": os.path.join(folder, f) for f in names}
            adds[f"{prefix}/{MANIFEST}"] = mp
            self.b.commit(branch, adds, [], f"{kind}: {prefix}")
        self._manifests[(branch, prefix)] = man
        return man

    def read_manifest(self, branch, prefix):
        key = (branch, prefix)
        if key not in self._manifests:
            with tempfile.TemporaryDirectory(prefix="store_manifest.") as tmp:
                p = os.path.join(tmp, MANIFEST)
                self.b.fetch(branch, f"{prefix}/{MANIFEST}", p)
                try:
                    man = json.load(open(p))
                except ValueError:
                    raise VerifyError(f"{branch}:{prefix}: manifest is not JSON") from None
            if not isinstance(man, dict) or man.get("format") != FORMAT or not isinstance(man.get("files"), dict):
                raise VerifyError(f"{branch}:{prefix}: bad manifest")
            self._manifests[key] = man
        return self._manifests[key]

    def pull_set(self, branch, prefix, dest, names=None):
        """Fetch a set into dest, each file checked against the manifest. A file already there with the right size
        and sha256 is kept. Raises VerifyError (and leaves no wrong file under its name) on any difference."""
        man = self.read_manifest(branch, prefix)
        os.makedirs(dest, exist_ok=True)
        for f in sorted(names if names is not None else man["files"]):
            if f not in man["files"]:
                raise VerifyError(f"{branch}:{prefix}: the manifest does not list {f}")
            want, dst = man["files"][f], os.path.join(dest, f)
            if os.path.isfile(dst) and os.path.getsize(dst) == want["bytes"] and sha256_file(dst)[0] == want["sha256"]:
                continue
            tmp = dst + ".pull.tmp"
            try:
                self.b.fetch(branch, f"{prefix}/{f}", tmp)
            except StoreError as e:
                raise VerifyError(f"{branch}:{prefix}/{f}: missing from the store ({e})") from None
            sha, n = sha256_file(tmp)
            if n != want["bytes"] or sha != want["sha256"]:
                os.remove(tmp)
                raise VerifyError(f"{branch}:{prefix}/{f}: sha256 or size differs from the manifest")
            os.replace(tmp, dst)
        return man

    # -- generations
    def steps(self, branch, tree=None):
        """Steps of the generations whose manifest is in the branch's listing, ascending. [] for no branch."""
        tree = self.b.tree(branch) if tree is None else tree
        out = set()
        for p in tree or {}:
            parts = p.split("/")
            if len(parts) == 3 and parts[0] == "gens" and parts[2] == MANIFEST and gen_step(parts[1]) is not None:
                out.add(gen_step(parts[1]))
        return sorted(out)

    def head_step(self, branch):
        """The newest pushed step, read from that generation's manifest; None for an empty branch."""
        steps = self.steps(branch)
        if not steps:
            return None
        return int(self.read_manifest(branch, f"gens/g{steps[-1]}")["step"])

    def whole_in_listing(self, branch, step, tree):
        """True when the listing holds every file of the generation's manifest at its size (and sha256, where the
        listing gives one). No download beyond the manifest."""
        try:
            man = self.read_manifest(branch, f"gens/g{step}")
        except (StoreError, OSError):
            return False
        if man.get("step") != step or not set(GEN_FILES) <= set(man["files"]):
            return False
        for f, want in man["files"].items():
            e = tree.get(f"gens/g{step}/{f}")
            if not e or e["bytes"] != want["bytes"] or (e.get("sha256") and e["sha256"] != want["sha256"]):
                return False
        return True

    def whole_steps(self, branch):
        """Steps of the generations that are whole in the listing, ascending."""
        tree = self.b.tree(branch) or {}
        return [s for s in self.steps(branch, tree) if self.whole_in_listing(branch, s, tree)]

    def push_generation(self, branch, job, step, folder):
        """One generation, staged in `folder` (GEN_FILES and events.jsonl), as one commit."""
        if not branch.startswith("ckpt-"):
            raise StoreError(f"generations go to a ckpt-* branch, not {branch}")
        check_generation(folder, step)
        names = [*GEN_FILES] + (["events.jsonl"] if os.path.exists(os.path.join(folder, "events.jsonl")) else [])
        return self.push_set(branch, f"gens/g{step}", folder, kind="generation", names=names,
                             meta={"job": job, "step": step})

    def pull_generation(self, branch, dest):
        """The newest generation that verifies, into dest; the one before it when the newest does not, and so on.
        Returns (step, [(rejected step, why)]), or (None, []) when the branch holds no generation. Raises
        StoreError when generations exist and none verifies: a resume never starts from step 0 in that case."""
        steps = self.steps(branch)
        rejected = []
        for step in reversed(steps):
            d = os.path.join(dest, f"g{step}")
            for attempt in (1, 2):  # a download damaged on the way is fetched once more before the step is given up
                shutil.rmtree(d, ignore_errors=True)
                try:
                    man = self.pull_set(branch, f"gens/g{step}", d)
                    if man.get("kind") != "generation" or man.get("step") != step:
                        raise VerifyError("the manifest names another step")
                    check_generation(d, step)
                    return step, rejected
                except (VerifyError, OSError) as e:
                    shutil.rmtree(d, ignore_errors=True)
                    self._manifests.pop((branch, f"gens/g{step}"), None)
                    if attempt == 2:
                        rejected.append((step, str(e)))
                        self.log(f"{branch}: generation g{step} refused twice ({e}); trying the one before")
        if steps:
            raise StoreError(f"{branch}: no generation verifies ({rejected}); refusing to resume")
        return None, rejected

    def prune(self, branch, keep=2):
        """Remove generations older than the newest `keep` whole ones. Never main, never fewer than two kept,
        never a path outside gens/. Returns the steps removed."""
        if not branch.startswith("ckpt-"):
            raise StoreError(f"prune refuses {branch}: only ckpt-* branches hold generations")
        if keep < 2:
            raise StoreError("prune keeps at least two generations")
        tree = self.b.tree(branch) or {}
        for key in [k for k in self._manifests if k[0] == branch]:  # a generation pushed again at a step has a new manifest
            del self._manifests[key]
        whole = [s for s in self.steps(branch, tree) if self.whole_in_listing(branch, s, tree)]
        if len(whole) <= keep:
            return []
        floor = whole[-keep]  # everything at or above it stays, whole or not
        dead = sorted(p for p in tree if p.startswith("gens/") and (gen_step(p.split("/")[1]) or 0) < floor
                      and gen_step(p.split("/")[1]) is not None)
        if not dead:
            return []
        gone = sorted({gen_step(p.split("/")[1]) for p in dead})
        self.b.commit(branch, {}, dead, f"prune: generations {gone} superseded by g{whole[-1]}")
        for s in gone:
            self._manifests.pop((branch, f"gens/g{s}"), None)
        if self.reclaim_on:
            try:
                shas = {tree[p]["sha256"] for p in dead if tree[p].get("sha256")}
                kept = {e["sha256"] for p, e in tree.items() if p not in dead and e.get("sha256")}
                for other in {"main", *self.branches} - {branch}:  # a mark holds the same bytes as its generation's weights
                    kept |= {e["sha256"] for e in (self.b.tree(other) or {}).values() if e.get("sha256")}
                n = self.b.reclaim(branch, shas, kept)
                self.log(f"{branch}: storage of {n} removed objects released")
            except Exception as e:  # noqa: BLE001 the generations are gone from the branch either way
                self.log(f"{branch}: storage not released ({type(e).__name__}: {e}); will try at the next prune")
        return gone

    # -- marks
    def mark_steps(self, job, tree=None):
        tree = self.b.tree("main") if tree is None else tree
        out = set()
        for p in tree or {}:
            parts = p.split("/")
            if len(parts) == 4 and parts[0] == "marks" and parts[1] == job and parts[3] == MANIFEST \
                    and parts[2].startswith("rung_") and parts[2][5:].isdigit():
                out.add(int(parts[2][5:]))
        return sorted(out)

    def push_mark(self, job, step, folder):
        """Weights and state.txt at a mark step, to main, once: a mark already in the store is left as it is."""
        if step in self.mark_steps(job):
            return False
        if file_step(os.path.join(folder, "state.txt")) != step:
            raise StoreError(f"{job}: mark folder is not at step {step}")
        self.push_set("main", f"marks/{job}/rung_{step}", folder, kind="mark", names=list(MARK_FILES),
                      meta={"job": job, "step": step})
        return True

    def pull_mark(self, job, step, dest):
        man = self.pull_set("main", f"marks/{job}/rung_{step}", dest, names=list(MARK_FILES))
        if man.get("step") != step or file_step(os.path.join(dest, "state.txt")) != step:
            raise VerifyError(f"{job}: mark rung_{step} names another step")
        return man


def file_step(p):
    try:
        return int(open(p).readline().strip() or 0)
    except (OSError, ValueError):
        return None


def check_generation(folder, step):
    """A staged or pulled generation is whole by the trainer's own MANIFEST.json (checkpoint_txn.hpp invalid()):
    format, step, and the size and sha256 of the three checkpoint files; state.txt names the step."""
    try:
        m = json.load(open(os.path.join(folder, "MANIFEST.json")))
    except (OSError, ValueError):
        raise VerifyError("no trainer manifest") from None
    if not isinstance(m, dict) or m.get("format") != GEN_FORMAT or m.get("step") != step:
        raise VerifyError("the trainer manifest names another step or format")
    for f in GEN_FILES[:3]:
        want = (m.get("files") or {}).get(f)
        p = os.path.join(folder, f)
        if not want or not os.path.isfile(p):
            raise VerifyError(f"missing {f}")
        sha, n = sha256_file(p)
        if n != want.get("bytes") or sha != want.get("sha256"):
            raise VerifyError(f"{f} differs from the trainer manifest")
    if file_step(os.path.join(folder, "state.txt")) != step:
        raise VerifyError("state.txt names another step")


# ---------------------------------------------------------------- configuration
def load_config(path):
    cfg = json.load(open(path))
    if not cfg.get("repo"):
        raise StoreError(f"{path}: no \"repo\" (there is no default store)")
    return cfg


def open_store(cfg, log=None, api=None):
    """A Store from a config dict or its path. "dir:PATH" repos need no token."""
    if isinstance(cfg, str):
        cfg = load_config(cfg)
    repo = cfg["repo"]
    kw = {"reclaim": bool(cfg.get("reclaim")), "log": log, "branches": (cfg.get("branches") or {}).values()}
    if repo.startswith("dir:"):
        return Store(DirBackend(repo[4:]), **kw)
    if not cfg.get("token_file"):
        raise StoreError("the store config names no token_file (there is no default token path)")
    return Store(HfBackend(repo, cfg.get("repo_type", "dataset"), read_token(cfg["token_file"]), api=api), **kw)


def retry(fn, tries=4, delay=20, log=print, what="store call", token=None):
    """Network calls of a boot: a few tries, then the error. A VerifyError is not retried by the caller's wish only."""
    for i in range(tries):
        try:
            return fn()
        except VerifyError:
            raise
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            log(redact(f"{what} failed ({type(e).__name__}: {e}); try {i + 2} of {tries} in {delay} s", token))
            time.sleep(delay)


def adopt_token(config, src, directory):
    """Boot: move the uploaded token to `directory` (a tmpfs on the VM, so it never sits on a volume), mode 600,
    and record its path in the config. With no new upload, an adopted token stays as it is."""
    cfg = json.load(open(config))
    if cfg["repo"].startswith("dir:") and not os.path.exists(src):
        return
    if os.path.exists(src):
        os.makedirs(directory, mode=0o700, exist_ok=True)
        dst = os.path.join(directory, "token")
        fd = os.open(dst + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(open(src).read())
        os.replace(dst + ".tmp", dst)
        os.remove(src)
        cfg["token_file"] = dst
        with open(config + ".tmp", "w") as f:
            json.dump(cfg, f, indent=1)
        os.replace(config + ".tmp", config)
    elif not (cfg.get("token_file") and os.path.exists(cfg["token_file"])):
        raise StoreError("no token was uploaded and none is adopted")


def forget_token(config, also=()):
    """Stop: remove the token from the VM. True when no copy is left."""
    paths = list(also)
    try:
        tf = json.load(open(config)).get("token_file")
        if tf:
            paths.append(tf)
    except (OSError, ValueError):
        pass
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    return not any(os.path.exists(p) for p in paths)


def pull_inputs(store, jobs_path, root, log=print):
    """A jobs file's "store_inputs", each set verified into ROOT/<to>."""
    for item in json.load(open(jobs_path)).get("store_inputs", []):
        dest = os.path.join(root, item["to"])
        t0 = time.time()
        man = retry(lambda item=item, dest=dest: store.pull_set(item.get("branch", "main"), item["prefix"], dest,
                                                                item.get("files")),
                    log=log, what=f"pull of {item['prefix']}", token=store.b.token)
        names = item.get("files") or sorted(man["files"])
        mb = sum(man["files"][f]["bytes"] for f in names) / 1e6
        log(f"input {item['prefix']}: {len(names)} files, {mb:.1f} MB verified in {time.time() - t0:.0f} s -> {dest}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, *args):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        for a, kw in args:
            p.add_argument(a, **kw)

    req = {"required": True}
    cmd("push", ("--branch", req), ("--prefix", req), ("--dir", req), ("--kind", {"default": "files"}))
    cmd("pull", ("--branch", req), ("--prefix", req), ("--to", req))
    cmd("inputs", ("--jobs", req), ("--root", req))
    cmd("steps", ("--branch", req))
    cmd("prune", ("--branch", req), ("--keep", {"type": int, "default": 2}))
    cmd("adopt-token", ("--from", {"required": True, "dest": "src"}), ("--dir", req))
    cmd("forget-token", ("--also", {"action": "append", "default": []}))
    a = ap.parse_args(argv)
    if a.cmd == "adopt-token":
        adopt_token(a.config, a.src, a.dir)
        print("token adopted")
        return 0
    if a.cmd == "forget-token":
        gone = forget_token(a.config, a.also)
        print("TOKEN_GONE" if gone else "TOKEN_LEFT")
        return 0 if gone else 1
    token = None
    try:
        store = open_store(a.config)
        token = store.b.token
        if a.cmd == "push":
            man = store.push_set(a.branch, a.prefix, a.dir, kind=a.kind)
            print(json.dumps({"pushed": a.prefix, "files": man["files"]}, indent=1))
        elif a.cmd == "pull":
            man = retry(lambda: store.pull_set(a.branch, a.prefix, a.to), what=f"pull of {a.prefix}", token=token)
            print(json.dumps({"pulled": a.prefix, "files": sorted(man["files"])}))
        elif a.cmd == "inputs":
            pull_inputs(store, a.jobs, a.root)
        elif a.cmd == "steps":
            steps = store.steps(a.branch)
            print(json.dumps({"branch": a.branch, "steps": steps, "head": steps[-1] if steps else None}))
        elif a.cmd == "prune":
            print(json.dumps({"branch": a.branch, "removed": store.prune(a.branch, a.keep)}))
    except Exception as e:  # noqa: BLE001 one redacted line, never a traceback that could carry a header
        print(redact(f"hf_store {a.cmd} failed: {type(e).__name__}: {e}", token), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
