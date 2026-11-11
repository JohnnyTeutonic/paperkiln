"""A stand-in for huggingface_hub.HfApi on local disk, so that tests reach hf_store.HfBackend, the real class, from
any process: this one (monkeypatch huggingface_hub.HfApi) or the children a shell script starts (tests/hub_standin_site
is put on PYTHONPATH; its sitecustomize.py installs this class as huggingface_hub.HfApi when HUB_STANDIN_ROOT is set).

It raises the library's own exception classes, as the hub does: huggingface_hub.errors.EntryNotFoundError for a path
that is not there, RevisionNotFoundError for a branch that is not there, RepositoryNotFoundError for a repo the token
cannot see, HfHubHTTPError for a refused login, requests.ConnectionError for a transport failure. Nothing here opens a
socket.

    HUB_STANDIN_ROOT/branches/<branch>/<path>   the files
    HUB_STANDIN_ROOT/heads.json                 {branch: commit count}; a commit id is "<branch>@<count>"
    HUB_STANDIN_ROOT/fail.json                  {"mode": "repo_not_found" | "auth" | "transport" | "hang"}: every call fails so
    HUB_STANDIN_ROOT/calls.log                  one line a call: the method and what it was asked for or raised
"""
import fcntl
import json
import os
import shutil
import time

from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
import requests


class DiskApi:
    def __init__(self, root=None):
        self.root = root or os.environ["HUB_STANDIN_ROOT"]
        os.makedirs(os.path.join(self.root, "branches", "main"), exist_ok=True)

    # -- plumbing
    def _note(self, text):
        with open(os.path.join(self.root, "calls.log"), "a") as f:
            f.write(text + "\n")

    def _gate(self, name, what=""):
        try:
            mode = json.load(open(os.path.join(self.root, "fail.json"))).get("mode")
        except (OSError, ValueError):
            mode = None
        self._note(f"{name} {what}" + (f" -> {mode}" if mode else ""))
        if mode == "repo_not_found":
            raise RepositoryNotFoundError("404 Client Error. Repository Not Found for url: https://huggingface.co/api/datasets/x. "
                                          "Please make sure you specified the correct `repo_id` and `repo_type`.")
        if mode == "auth":
            raise HfHubHTTPError("401 Client Error: Unauthorized for url: https://huggingface.co/api/datasets/x (Invalid credentials)")
        if mode == "transport":
            raise requests.ConnectionError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded")
        if mode == "hang":
            time.sleep(3600)

    def _heads(self):
        try:
            return json.load(open(os.path.join(self.root, "heads.json")))
        except (OSError, ValueError):
            return {"main": 0}

    def _dir(self, revision):
        branch = (revision or "main").split("@")[0]
        d = os.path.join(self.root, "branches", branch)
        if not os.path.isdir(d):
            self._note(f"  raise RevisionNotFoundError {revision}")
            raise RevisionNotFoundError(f"404 Client Error. Revision Not Found for url: .../{revision}. Invalid rev id: {revision}")
        return branch, d

    # -- the calls hf_store.HfBackend makes, every argument by keyword
    def create_branch(self, *, repo_id, branch, exist_ok=False, repo_type=None, token=None):
        self._gate("create_branch", branch)
        os.makedirs(os.path.join(self.root, "branches", branch), exist_ok=True)

    def list_repo_refs(self, *, repo_id, repo_type=None, token=None):
        self._gate("list_repo_refs")
        names = sorted(os.listdir(os.path.join(self.root, "branches")))
        return type("Refs", (), {"branches": [type("Ref", (), {"name": n})() for n in names]})()

    def repo_info(self, *, repo_id, revision=None, repo_type=None, token=None):
        self._gate("repo_info", str(revision))
        branch, _ = self._dir(revision)
        return type("Info", (), {"sha": f"{branch}@{self._heads().get(branch, 0)}"})()

    def list_repo_tree(self, *, repo_id, recursive=False, revision=None, repo_type=None, token=None):
        self._gate("list_repo_tree", str(revision))
        _, d = self._dir(revision)
        out = []
        for root, _, files in os.walk(d):
            for f in sorted(files):
                p = os.path.join(root, f)
                out.append(type("RepoFile", (), {"path": os.path.relpath(p, d).replace(os.sep, "/"),
                                                 "size": os.path.getsize(p), "lfs": None})())
        return out

    def create_commit(self, *, repo_id, operations, commit_message, revision=None, parent_commit=None, repo_type=None, token=None):
        self._gate("create_commit", f"{revision} {commit_message}")
        branch, d = self._dir(revision)
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            heads = self._heads()
            if parent_commit is not None and parent_commit != f"{branch}@{heads.get(branch, 0)}":
                raise HfHubHTTPError("412 Client Error: Precondition Failed (a commit has happened since)")
            for o in operations:
                dst = os.path.join(d, *o.path_in_repo.split("/"))
                if type(o).__name__ == "CommitOperationAdd":
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(o.path_or_fileobj, dst + ".incoming")
                    os.replace(dst + ".incoming", dst)
                elif os.path.exists(dst):
                    os.remove(dst)
            heads[branch] = heads.get(branch, 0) + 1
            with open(os.path.join(self.root, "heads.json"), "w") as f:
                json.dump(heads, f)

    def hf_hub_download(self, *, repo_id, filename, revision=None, local_dir=None, repo_type=None, token=None):
        self._gate("hf_hub_download", f"{revision}:{filename}")
        _, d = self._dir(revision)
        src = os.path.join(d, *filename.split("/"))
        if not os.path.isfile(src):
            self._note(f"  raise EntryNotFoundError {filename}")
            raise EntryNotFoundError(f"404 Client Error. Entry Not Found for url: .../resolve/{revision}/{filename}.")
        dst = os.path.join(local_dir, *filename.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def list_lfs_files(self, *, repo_id, repo_type=None, token=None):
        self._gate("list_lfs_files")
        return []

    def permanently_delete_lfs_files(self, *, repo_id, lfs_files, rewrite_history=True, repo_type=None, token=None):
        self._gate("permanently_delete_lfs_files")
