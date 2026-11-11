#!/usr/bin/env python3
"""A checkpoint and input store on a private Hugging Face repo, for campaigns that outlive their VMs (base 2).

    python3 hf_store.py push   --config STORE.json --branch main --prefix corpus/N --dir DIR
    python3 hf_store.py pull   --config STORE.json --branch main --prefix corpus/N --to DIR
    python3 hf_store.py inputs --config STORE.json --jobs JOBS.json --root /content
    python3 hf_store.py steps  --config STORE.json --branch ckpt-N
    python3 hf_store.py prune  --config STORE.json --branch ckpt-N [--keep 2]
    python3 hf_store.py release --config STORE.json
    python3 hf_store.py adopt-token  --config STORE.json --from FILE --dir DIR
    python3 hf_store.py forget-token --config STORE.json [--chat CHAT]
Every command but the two token ones takes --limit SECONDS: the call runs in a child that is ended at the limit.

Layout. A "set" is a folder of the repo with a manifest.json beside its files: the format, the kind, and the size
and sha256 of every file. A push is one commit that holds the files and the manifest; a pull fetches the manifest,
then each file, and refuses a file whose size or sha256 differs.

    main      inputs (any prefix), marks/<job>/rung_<step>/ (weights at a mark step), locks/. prune() refuses it.
    ckpt-*    gens/g<step>/ : one transactional checkpoint generation of one arm (model.safetensors,
              optim.safetensors, state.txt, the trainer's MANIFEST.json, events.jsonl) and its manifest.json.

Generations. A generation's manifest names its campaign, job, the sha256 of the job's spec and of the inputs the VM
pulled, and the writer (the runner's launch count as "epoch", and a boot id). A generation is never written twice:
a step whose manifest is listed is left alone, and a push names the branch head it was computed against, so a
second writer's commit in between is refused by the store (DirBackend) or by the hub (parent_commit). A resume
(pull_generation) takes the newest generation that verifies by download, gives up one only on a verification
error (a transport error is retried, then raised), refuses a generation of another campaign, job, spec or inputs,
and refuses when the branch lists nothing at or above the step home has seen pushed. prune() removes generations
only behind two that the caller has verified, by its own push or by download; a listing alone never counts.

Storage. Removing a generation from a branch does not free its bytes. release() does, and is opt-in ("reclaim" in
the config): under a lock file on main it deletes the LFS objects that no branch's tree names and that are older
than a grace period, so an object another writer has uploaded and not yet committed is not touched.

STORE.json: {"repo": "owner/name", "repo_type": "dataset", "token_file": PATH, "reclaim": false,
"min_push_interval_s": 0, "branches": {job: branch}, "campaign": NAME, "epoch": N, "min_step": {job: step},
"first_start": [jobs]}. There is no default repo and no default token path. "repo": "dir:/some/folder" is a
store on local disk with the same rules (tests and local rehearsals; no token). A jobs file names the inputs a VM
needs as "store_inputs": [{"branch": "main", "prefix": P, "to": "chat/upload"}], "to" relative to --root.

The token. It is read from token_file (mode 600, refused otherwise) and handed to the library call by call; it is
not printed, logged, put on a command line or exported, and every error text passes through redact(). On a VM it
arrives as an uploaded file on the VM's disk and is moved to a tmpfs by the runner's next call (adopt_token), so
it is on disk for the seconds between the two. HF_HUB_DISABLE_IMPLICIT_TOKEN is set here, so no other download in a
process that imported this file sends a token it was not given.

Assumed of Hugging Face and not tested here (the tests run against DirBackend and a recording stand-in for HfApi,
with huggingface_hub pinned to HUB_VERSION): create_commit is atomic, takes adds and deletes together and refuses
a stale parent_commit; list_repo_tree gives each file's size and, for LFS files, its sha256; list_lfs_files names
each object by that sha256 (release() checks this before it deletes anything, and refuses otherwise);
permanently_delete_lfs_files frees the bytes; the account's storage quota and commit rate limits.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import select
import shutil
import signal
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

FORMAT = "paperkiln.store/1"
MANIFEST = "manifest.json"
HUB_VERSION = "0.35.3"  # the version the calls below were written and tested against; the boot installs it
GEN_FILES = ("model.safetensors", "optim.safetensors", "state.txt", "MANIFEST.json")  # one trainer generation
GEN_FORMAT = "paperkiln.ckpt_gen/1"  # include/microtorch/checkpoint_txn.hpp
MARK_FILES = ("model.safetensors", "state.txt")  # what jobs_vm.take_snapshot copies
IDENTITY = ("campaign", "job", "spec_sha256", "inputs_sha256")
LOCK = "locks/release.json"
LOCK_STALE_S = 1800
RELEASE_GRACE_S = 7200
# Repos this tool never writes to or prunes: the screens' and cloud_eval's inputs repo holds pinned revisions.
REFUSED_REPOS = ("JohnnyTeutonic/chat-screens-inputs",)


class StoreError(Exception):
    pass


class VerifyError(StoreError):
    """Bytes that are not what a manifest says: the only reason a generation is given up for an older one."""


class Missing(StoreError):
    """A path the store does not hold."""


class IdentityError(StoreError):
    """A generation of another campaign, job, spec or inputs, or history missing below a step home has seen."""


class ConflictError(StoreError):
    """The branch moved between the read and the commit."""


ERRORS = {c.__name__: c for c in (StoreError, VerifyError, Missing, IdentityError, ConflictError)}


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


def spec_hash(job):
    """The sha256 of a job's spec as the jobs file gives it."""
    return hashlib.sha256(json.dumps(job.get("spec"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hub_version():
    try:
        import huggingface_hub
        return huggingface_hub.__version__
    except Exception:  # noqa: BLE001 a dir: store needs no library
        return None


# ---------------------------------------------------------------- a call with a wall-time limit
_CHILDREN = set()


def end_children():
    """End every bounded() child of this process (a daemon's SIGTERM handler calls it)."""
    for pid in list(_CHILDREN):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def bounded(fn, timeout, token=None, while_waiting=None):
    """fn() in a forked child that is ended after `timeout` seconds, so a hung transfer stops nothing else. The
    result (JSON-able, of any size: the pipe is read while the child runs) or the child's error, of the same store
    error class, with the token redacted. while_waiting() runs about once a second in the parent meanwhile."""
    if not timeout:
        return fn()
    sys.stdout.flush()  # so the child does not inherit, and write again, what is still buffered here
    sys.stderr.flush()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = {"ok": fn()}
        except BaseException as e:  # noqa: BLE001
            out = {"err": redact(f"{e}", token)[:2000], "type": type(e).__name__}
        try:
            sys.stdout.flush()  # what fn printed: _exit below drops unwritten buffers
            sys.stderr.flush()
            data = json.dumps(out).encode()
            while data:
                data = data[os.write(w, data):]
        finally:
            os._exit(0)
    os.close(w)
    _CHILDREN.add(pid)
    t0, last, buf = time.time(), 0.0, b""
    try:
        while True:
            ready, _, _ = select.select([r], [], [], 0.2)
            if ready:
                b = os.read(r, 1 << 16)
                if not b:
                    break
                buf += b
            now = time.time()
            if while_waiting and now - last >= 1:
                last = now
                while_waiting()
            if now - t0 > timeout:
                os.kill(pid, signal.SIGKILL)
                raise StoreError(f"store call over {timeout:.0f} s: ended")
    finally:
        os.close(r)
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass
        _CHILDREN.discard(pid)
    try:
        out = json.loads(buf)
    except ValueError:
        raise StoreError("store call ended without a result") from None
    if "err" in out:
        cls = ERRORS.get(out.get("type"))
        raise cls(out["err"]) if cls else StoreError(f"{out.get('type')}: {out['err']}")
    return out["ok"]


# ---------------------------------------------------------------- backends
class DirBackend:
    """The store on local disk: ROOT/<branch>/<path>, commits listed in ROOT/commits.jsonl. A commit copies its
    files in, the manifest last, under a lock, and is refused when `parent` is not the branch's head. Tests and
    local rehearsals only."""
    token = None

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _p(self, branch, path=""):
        return os.path.join(self.root, branch, *path.split("/")) if path else os.path.join(self.root, branch)

    def _log(self):
        p = os.path.join(self.root, "commits.jsonl")
        return [json.loads(ln) for ln in open(p).read().splitlines()] if os.path.exists(p) else []

    def ensure_branch(self, branch):
        os.makedirs(self._p(branch), exist_ok=True)

    def branches(self):
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))

    def head(self, branch):
        if not os.path.isdir(self._p(branch)):
            return None
        return str(sum(1 for c in self._log() if c.get("branch") == branch and "message" in c))

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

    def commit(self, branch, adds, deletes, message, parent=None):
        self.ensure_branch(branch)
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if parent is not None and parent != self.head(branch):
                raise ConflictError(f"{branch} moved (head {self.head(branch)}, expected {parent})")
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
            raise Missing(f"{branch}:{path} is not in the store")
        shutil.copyfile(src, dest)

    def lfs_objects(self):
        return []  # a folder has no object store: nothing to release

    def delete_lfs(self, objects):
        pass


class HfBackend:
    """The same operations on a Hugging Face repo. `api` is huggingface_hub.HfApi() unless a test passes its own.
    Every library call is made with keyword arguments."""

    def __init__(self, repo, repo_type, token, api=None):
        if repo in REFUSED_REPOS:
            raise StoreError(f"{repo} is the pinned inputs repo; the store must be its own repo")
        self.repo, self.repo_type, self.token = repo, repo_type, token
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api = api

    def _kw(self):
        return {"repo_id": self.repo, "repo_type": self.repo_type, "token": self.token}

    def ensure_branch(self, branch):
        if branch != "main":
            self.api.create_branch(branch=branch, exist_ok=True, **self._kw())

    def branches(self):
        return sorted(b.name for b in self.api.list_repo_refs(**self._kw()).branches)

    def head(self, branch):
        try:
            return self.api.repo_info(revision=branch, **self._kw()).sha
        except Exception as e:  # noqa: BLE001 the library's not-found types, by name so a stand-in can raise them
            if type(e).__name__ == "RevisionNotFoundError":
                return None
            raise

    def tree(self, branch):
        try:
            entries = list(self.api.list_repo_tree(recursive=True, revision=branch, **self._kw()))
        except Exception as e:  # noqa: BLE001
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

    def commit(self, branch, adds, deletes, message, parent=None):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        self.ensure_branch(branch)
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=src) for p, src in sorted(adds.items())]
        ops += [CommitOperationDelete(path_in_repo=p) for p in sorted(deletes)]
        self.api.create_commit(operations=ops, commit_message=message, revision=branch, parent_commit=parent, **self._kw())

    def fetch(self, branch, path, dest):
        stage = tempfile.mkdtemp(prefix=".hfstage.", dir=os.path.dirname(os.path.abspath(dest)))
        try:
            got = self.api.hf_hub_download(filename=path, revision=branch, local_dir=stage, **self._kw())
            os.replace(got, dest)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("EntryNotFoundError", "RevisionNotFoundError"):
                raise Missing(f"{branch}:{path} is not in the store") from None
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def lfs_objects(self):
        """[{"ids": the object's ids as the library gives them, "pushed_at": epoch seconds or None, "ref": handle}]"""
        out = []
        for f in self.api.list_lfs_files(**self._kw()):
            at = getattr(f, "pushed_at", None)
            out.append({"ids": [x for x in (getattr(f, "oid", None), getattr(f, "file_oid", None)) if x],
                        "pushed_at": at.timestamp() if hasattr(at, "timestamp") else None, "ref": f})
        return out

    def delete_lfs(self, objects):
        self.api.permanently_delete_lfs_files(lfs_files=[o["ref"] for o in objects], rewrite_history=False, **self._kw())


# ---------------------------------------------------------------- the store
class Store:
    def __init__(self, backend, reclaim=False, log=None, tries=4, delay=20):
        self.b, self.reclaim_on, self._log, self.tries, self.delay = backend, reclaim, log, tries, delay
        self._manifests = {}

    def log(self, msg):
        msg = redact(msg, self.b.token)
        if self._log:
            self._log(msg)
        else:
            print(msg, flush=True)

    def fetch(self, branch, path, dest):
        """One file, a transport failure tried again (Missing is an answer, not a failure)."""
        for i in range(self.tries):
            try:
                return self.b.fetch(branch, path, dest)
            except Missing:
                raise
            except Exception as e:  # noqa: BLE001
                if i == self.tries - 1:
                    raise
                self.log(f"fetch of {branch}:{path} failed ({type(e).__name__}: {e}); try {i + 2} of {self.tries}")
                time.sleep(self.delay)

    # -- sets
    def push_set(self, branch, prefix, folder, kind="files", names=None, meta=None, parent=None):
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
            self.b.commit(branch, adds, [], f"{kind}: {prefix}", parent=parent)
        self._manifests[(branch, prefix)] = man
        return man

    def read_manifest(self, branch, prefix, fresh=False):
        key = (branch, prefix)
        if fresh or key not in self._manifests:
            with tempfile.TemporaryDirectory(prefix="store_manifest.") as tmp:
                p = os.path.join(tmp, MANIFEST)
                self.fetch(branch, f"{prefix}/{MANIFEST}", p)
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
        and sha256 is kept. A file that differs is fetched once more, then VerifyError; no wrong file is left
        under its name. A transport failure is retried by fetch() and then raised as itself."""
        man = self.read_manifest(branch, prefix)
        os.makedirs(dest, exist_ok=True)
        for f in sorted(names if names is not None else man["files"]):
            if f not in man["files"]:
                raise VerifyError(f"{branch}:{prefix}: the manifest does not list {f}")
            want, dst = man["files"][f], os.path.join(dest, f)
            if os.path.isfile(dst) and os.path.getsize(dst) == want["bytes"] and sha256_file(dst)[0] == want["sha256"]:
                continue
            tmp = dst + ".pull.tmp"
            for attempt in (1, 2):
                try:
                    self.fetch(branch, f"{prefix}/{f}", tmp)
                except Missing as e:
                    raise VerifyError(f"{branch}:{prefix}/{f}: missing from the store ({e})") from None
                sha, n = sha256_file(tmp)
                if n == want["bytes"] and sha == want["sha256"]:
                    break
                os.remove(tmp)
                if attempt == 2:
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
        """The newest listed step, read from that generation's manifest; None for an empty branch."""
        steps = self.steps(branch)
        if not steps:
            return None
        return int(self.read_manifest(branch, f"gens/g{steps[-1]}")["step"])

    def head(self, branch):
        """What a writer reads before each push: the branch's commit, its listed steps, and who wrote the newest
        generation ({"commit", "steps", "step", "epoch", "boot_id"}; step None for an empty branch)."""
        commit = self.b.head(branch)
        steps = self.steps(branch)
        out = {"commit": commit, "steps": steps, "step": None, "epoch": None, "boot_id": None}
        if steps:
            man = self.read_manifest(branch, f"gens/g{steps[-1]}", fresh=True)
            out.update(step=steps[-1], epoch=man.get("epoch"), boot_id=man.get("boot_id"))
        return out

    def whole_in_listing(self, branch, step, tree):
        """True when the listing holds every file of the generation's manifest at its size (and sha256, where the
        listing gives one). A necessary check before a prune or a completion, never a verification by itself."""
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
        tree = self.b.tree(branch) or {}
        return [s for s in self.steps(branch, tree) if self.whole_in_listing(branch, s, tree)]

    def push_generation(self, branch, job, step, folder, identity=None, parent=None):
        """One generation, staged in `folder` (GEN_FILES and events.jsonl), as one commit computed against the
        branch head `parent`. A step already listed is never written again."""
        if not branch.startswith("ckpt-"):
            raise StoreError(f"generations go to a ckpt-* branch, not {branch}")
        check_generation(folder, step)
        if step in self.steps(branch):
            raise ConflictError(f"{branch}: g{step} is already in the store; a generation is written once")
        names = [*GEN_FILES] + (["events.jsonl"] if os.path.exists(os.path.join(folder, "events.jsonl")) else [])
        return self.push_set(branch, f"gens/g{step}", folder, kind="generation", names=names,
                             meta={**(identity or {}), "job": job, "step": step}, parent=parent)

    def pull_generation(self, branch, dest, expect=None, min_step=0):
        """The newest generation that verifies, into dest/g<step>; the one before it only when the newer one fails
        verification. Returns (step, [(rejected step, why)]), or (None, []) for a branch with no generation.
        Raises IdentityError when the branch lists nothing at or above min_step (the step home has seen pushed) or
        the generation is another campaign's, job's, spec's or inputs'; StoreError when generations exist and none
        verifies; a transport error as itself. So a resume never starts lower because something could not be read."""
        steps = self.steps(branch)
        if min_step and (not steps or steps[-1] < min_step):
            raise IdentityError(f"{branch}: home has seen step {min_step} pushed and the store lists "
                                f"{'up to g%d' % steps[-1] if steps else 'no generation'}; refusing to resume")
        rejected = []
        for step in reversed(steps):
            d = os.path.join(dest, f"g{step}")
            try:
                try:
                    man = self.pull_set(branch, f"gens/g{step}", d)
                except Missing as e:
                    raise VerifyError(str(e)) from None
                if man.get("kind") != "generation" or man.get("step") != step:
                    raise VerifyError("the manifest names another step")
                check_identity(man, expect, f"{branch}: g{step}")
                check_generation(d, step)
                return step, rejected
            except IdentityError:
                raise
            except VerifyError as e:
                shutil.rmtree(d, ignore_errors=True)
                self._manifests.pop((branch, f"gens/g{step}"), None)
                rejected.append((step, str(e)))
                self.log(f"{branch}: generation g{step} refused ({e}); trying the one before")
        if steps:
            raise StoreError(f"{branch}: no generation verifies ({rejected}); refusing to resume")
        return None, rejected

    def verify_by_download(self, branch, step, expect=None):
        """True when the generation downloads whole (every sha256, the trainer's manifest, the identity)."""
        with tempfile.TemporaryDirectory(prefix="store_verify.") as tmp:
            try:
                man = self.pull_set(branch, f"gens/g{step}", tmp)
                check_identity(man, expect, f"{branch}: g{step}")
                check_generation(tmp, step)
                return man.get("step") == step
            except (VerifyError, Missing, IdentityError):
                return False

    def prune(self, branch, keep=2, verified=(), protect=()):
        """Remove the generations older than the `keep` newest of `verified`: the steps the caller has verified by
        its own push or by download, and that are still whole in the listing. With fewer than `keep` such
        generations nothing is removed. Nothing at or above the lowest step in `protect` (a mark staged and not
        yet pushed) is removed. Never main, never a path outside gens/. Returns the steps removed."""
        if not branch.startswith("ckpt-"):
            raise StoreError(f"prune refuses {branch}: only ckpt-* branches hold generations")
        if keep < 2:
            raise StoreError("prune keeps at least two generations")
        tree = self.b.tree(branch) or {}
        for key in [k for k in self._manifests if k[0] == branch]:
            del self._manifests[key]
        good = [s for s in self.steps(branch, tree) if s in set(verified) and self.whole_in_listing(branch, s, tree)]
        if len(good) < keep:
            return []
        floor = min([good[-keep], *protect])
        dead = sorted(p for p in tree if p.startswith("gens/") and gen_step(p.split("/")[1]) is not None
                      and gen_step(p.split("/")[1]) < floor)
        if not dead:
            return []
        gone = sorted({gen_step(p.split("/")[1]) for p in dead})
        self.b.commit(branch, {}, dead, f"prune: generations {gone} behind verified g{good[-keep]} and g{good[-1]}")
        for s in gone:
            self._manifests.pop((branch, f"gens/g{s}"), None)
        return gone

    def release(self, owner, grace=RELEASE_GRACE_S, now=None):
        """Free storage by difference: every LFS object that no branch's tree names (main, every ckpt-*, any
        other) and that was pushed more than `grace` seconds ago is deleted for good. One releaser at a time, by a
        lock file on main taken with a conditional commit. Returns the number deleted, or None when another holds
        the lock. Refuses when the listing's hashes cannot be matched to the object ids."""
        now = time.time() if now is None else now
        if not self._lock(owner, now):
            return None
        try:
            named = set()
            for br in self.b.branches():
                named |= {e["sha256"] for e in (self.b.tree(br) or {}).values() if e.get("sha256")}
            objs = self.b.lfs_objects()
            ids = {i for o in objs for i in o["ids"]}
            if not named <= ids:
                raise StoreError("release refused: the trees name objects the LFS listing does not show by sha256")
            dead = [o for o in objs if not named.intersection(o["ids"]) and o["pushed_at"] is not None
                    and now - o["pushed_at"] > grace]
            if dead:
                self.b.delete_lfs(dead)
            return len(dead)
        finally:
            self._unlock()

    def _lock(self, owner, now):
        tree = self.b.tree("main") or {}
        if LOCK in tree:
            with tempfile.TemporaryDirectory(prefix="store_lock.") as tmp:
                try:
                    self.b.fetch("main", LOCK, os.path.join(tmp, "l"))
                    held = json.load(open(os.path.join(tmp, "l")))
                except (StoreError, ValueError, OSError):
                    held = {}
            if now - held.get("at", 0) < LOCK_STALE_S:
                return False
        with tempfile.TemporaryDirectory(prefix="store_lock.") as tmp:
            p = os.path.join(tmp, "l")
            json.dump({"owner": owner, "at": now}, open(p, "w"))
            try:
                self.b.commit("main", {LOCK: p}, [], f"lock: release by {owner}", parent=self.b.head("main"))
            except Exception:  # noqa: BLE001 main moved under us, or the hub refused: someone else may hold it
                return False
        return True

    def _unlock(self):
        try:
            self.b.commit("main", {}, [LOCK], "unlock: release")
        except Exception as e:  # noqa: BLE001 a lock left behind goes stale by itself
            self.log(f"release lock not removed ({type(e).__name__}: {e}); it expires in {LOCK_STALE_S} s")

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

    def push_mark(self, job, step, folder, identity=None):
        """Weights and state.txt at a mark step, to main, once: a mark already in the store is left as it is."""
        if step in self.mark_steps(job):
            return False
        if file_step(os.path.join(folder, "state.txt")) != step:
            raise StoreError(f"{job}: mark folder is not at step {step}")
        self.push_set("main", f"marks/{job}/rung_{step}", folder, kind="mark", names=list(MARK_FILES),
                      meta={**(identity or {}), "job": job, "step": step})
        return True

    def pull_mark(self, job, step, dest, expect=None):
        man = self.pull_set("main", f"marks/{job}/rung_{step}", dest, names=list(MARK_FILES))
        check_identity(man, expect, f"{job}: mark rung_{step}")
        if man.get("step") != step or file_step(os.path.join(dest, "state.txt")) != step:
            raise VerifyError(f"{job}: mark rung_{step} names another step")
        return man


def file_step(p):
    try:
        return int(open(p).readline().strip() or 0)
    except (OSError, ValueError):
        return None


def check_identity(man, expect, what):
    for k in IDENTITY:
        if expect and expect.get(k) is not None and man.get(k) != expect[k]:
            raise IdentityError(f"{what} is of another {k} ({str(man.get(k))[:16]}, expected {str(expect[k])[:16]}); refusing")


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


def open_store(cfg, log=None, api=None, **kw):
    """A Store from a config dict or its path. "dir:PATH" repos need no token."""
    if isinstance(cfg, str):
        cfg = load_config(cfg)
    repo = cfg["repo"]
    kw = {"reclaim": bool(cfg.get("reclaim")), "log": log, **kw}
    if repo.startswith("dir:"):
        return Store(DirBackend(repo[4:]), **kw)
    if not cfg.get("token_file"):
        raise StoreError("the store config names no token_file (there is no default token path)")
    return Store(HfBackend(repo, cfg.get("repo_type", "dataset"), read_token(cfg["token_file"]), api=api), **kw)


def retry(fn, tries=4, delay=20, log=print, what="store call", token=None):
    """A few tries, then the error. A VerifyError is an answer and is not tried again."""
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


def token_copies(config, also=()):
    """Every path a token can be at on this machine: the adopted file, its temporaries, and the given paths."""
    paths = list(also)
    try:
        tf = json.load(open(config)).get("token_file")
    except (OSError, ValueError):
        tf = None
    if tf:
        paths += [tf, *glob.glob(tf + ".tmp*")]
    return paths


def adopt_token(config, src, directory):
    """Move the uploaded token to `directory` (a tmpfs on the VM), mode 600, and record its path in the config.
    With no new upload, an adopted token stays as it is. No temporary is left behind on a failure."""
    cfg = json.load(open(config))
    if cfg["repo"].startswith("dir:") and not os.path.exists(src):
        return
    if os.path.exists(src):
        os.makedirs(directory, mode=0o700, exist_ok=True)
        dst = os.path.join(directory, "token")
        tmp = f"{dst}.tmp{os.getpid()}"
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(open(src).read())
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        os.remove(src)
        cfg["token_file"] = dst
        with open(config + ".tmp", "w") as f:
            json.dump(cfg, f, indent=1)
        os.replace(config + ".tmp", config)
    elif not (cfg.get("token_file") and os.path.exists(cfg["token_file"])):
        raise StoreError("no token was uploaded and none is adopted")


def stop_courier(chat, wait=15):
    """End the courier of this VM and its children (it runs in a process group of its own) and wait for them, so
    nothing that holds the token in memory outlives the file."""
    try:
        pid = int(open(os.path.join(chat, "courier.pid")).read())
    except (OSError, ValueError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except OSError:
            return
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                os.waitpid(pid, os.WNOHANG)  # reaps it when it is our child
            except OSError:
                pass
            state = ""
            try:
                state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
            except (OSError, IndexError):
                return
            if state == "Z":
                return
            time.sleep(0.1)


def forget_token(config, also=(), chat=None):
    """Remove the token from the VM: the courier and its children first (when `chat` is given), then every copy.
    True when no copy is left."""
    if chat:
        stop_courier(chat)
        also = [*also, os.path.join(chat, "upload", "store_token")]
    paths = token_copies(config, also)
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    return not any(os.path.exists(p) for p in paths)


def pull_inputs(store, jobs_path, root, log=print, record=None):
    """A jobs file's "store_inputs", each set verified into ROOT/<to>. `record` gets every file's sha256 and one
    digest of them all ("inputs_sha256"), which each generation's manifest then carries."""
    sets = {}
    for item in json.load(open(jobs_path)).get("store_inputs", []):
        dest = os.path.join(root, item["to"])
        t0 = time.time()
        man = store.pull_set(item.get("branch", "main"), item["prefix"], dest, item.get("files"))
        names = item.get("files") or sorted(man["files"])
        mb = sum(man["files"][f]["bytes"] for f in names) / 1e6
        sets[f"{item.get('branch', 'main')}:{item['prefix']}"] = {f: man["files"][f]["sha256"] for f in names}
        log(f"input {item['prefix']}: {len(names)} files, {mb:.1f} MB verified in {time.time() - t0:.0f} s -> {dest}")
    if record:
        rec = {"sets": sets, "inputs_sha256": inputs_hash(sets)}
        with open(record + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(record + ".tmp", record)


def inputs_hash(sets):
    return hashlib.sha256(json.dumps(sets, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preflight(store, branches, inputs):
    """From home, before a session is created: every job's branch can be listed and every input set's manifest
    read, with each of its files in the listing at its size. Returns {branch: newest step}; raises otherwise."""
    out = {}
    for branch in sorted(set(branches)):
        steps = store.steps(branch)
        out[branch] = steps[-1] if steps else None
    trees = {}
    for item in inputs:
        br = item.get("branch", "main")
        man = store.read_manifest(br, item["prefix"], fresh=True)
        trees[br] = trees.get(br) or store.b.tree(br) or {}
        for f in item.get("files") or sorted(man["files"]):
            e = trees[br].get(f"{item['prefix']}/{f}")
            if f not in man["files"] or not e or e["bytes"] != man["files"][f]["bytes"]:
                raise VerifyError(f"{br}:{item['prefix']}/{f} is not in the store as its manifest says")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, *args):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--limit", type=float, default=0, help="seconds before the call is ended (0: none)")
        for a, kw in args:
            p.add_argument(a, **kw)

    req = {"required": True}
    cmd("push", ("--branch", req), ("--prefix", req), ("--dir", req), ("--kind", {"default": "files"}))
    cmd("pull", ("--branch", req), ("--prefix", req), ("--to", req))
    cmd("inputs", ("--jobs", req), ("--root", req), ("--record", {"default": None}))
    cmd("steps", ("--branch", req))
    cmd("prune", ("--branch", req), ("--keep", {"type": int, "default": 2}))
    cmd("release", ("--owner", {"default": "cli"}))
    cmd("adopt-token", ("--from", {"required": True, "dest": "src"}), ("--dir", req))
    cmd("forget-token", ("--also", {"action": "append", "default": []}), ("--chat", {"default": None}))
    a = ap.parse_args(argv)
    if a.cmd == "adopt-token":
        try:
            adopt_token(a.config, a.src, a.dir)
        except (StoreError, OSError, ValueError) as e:
            print(f"hf_store adopt-token failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print("token adopted")
        return 0
    if a.cmd == "forget-token":
        gone = forget_token(a.config, a.also, a.chat)
        print("TOKEN_GONE" if gone else "TOKEN_LEFT")
        return 0 if gone else 1
    token = None

    def work():
        if a.cmd == "push":
            man = store.push_set(a.branch, a.prefix, a.dir, kind=a.kind)
            return json.dumps({"pushed": a.prefix, "files": man["files"]}, indent=1)
        if a.cmd == "pull":
            man = store.pull_set(a.branch, a.prefix, a.to)
            return json.dumps({"pulled": a.prefix, "files": sorted(man["files"])})
        if a.cmd == "inputs":
            pull_inputs(store, a.jobs, a.root, record=a.record)
            return ""
        if a.cmd == "steps":
            steps = store.steps(a.branch)
            return json.dumps({"branch": a.branch, "steps": steps, "head": steps[-1] if steps else None})
        if a.cmd == "prune":  # by hand: the generations are verified by download here, newest first
            good = []
            for s in reversed(store.steps(a.branch)):
                if len(good) < a.keep and store.verify_by_download(a.branch, s):
                    good.append(s)
            return json.dumps({"branch": a.branch, "verified": sorted(good),
                               "removed": store.prune(a.branch, a.keep, verified=good)})
        return json.dumps({"released": store.release(a.owner)})

    try:
        store = open_store(a.config)
        token = store.b.token
        out = bounded(work, a.limit, token)
        if out:
            print(out)
    except Exception as e:  # noqa: BLE001 one redacted line, never a traceback that could carry a header
        print(redact(f"hf_store {a.cmd} failed: {type(e).__name__}: {e}", token), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
