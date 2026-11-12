"""A stand-in for huggingface_hub.HfApi on local disk, so that tests reach hf_store.HfBackend, the real class, from
any process: this one (monkeypatch huggingface_hub.HfApi) or the children a shell script starts (tests/hub_standin_site
is put on PYTHONPATH; its sitecustomize.py installs this class as huggingface_hub.HfApi when HUB_STANDIN_ROOT is set).

It raises the library's own exception classes, as the hub does: huggingface_hub.errors.EntryNotFoundError for a path
that is not there, RevisionNotFoundError for a branch that is not there, RepositoryNotFoundError for a repo the token
cannot see, HfHubHTTPError for a refused login and for HTTP 429, requests.ConnectionError for a transport failure.
Nothing here opens a socket.

    HUB_STANDIN_ROOT/branches/<branch>/<path>   the files
    HUB_STANDIN_ROOT/heads.json                 {branch: commit count}; a commit id is "<branch>@<count>"
    HUB_STANDIN_ROOT/fail.json                  {"mode": "repo_not_found" | "auth" | "transport" | "hang"}: every call fails so
    HUB_STANDIN_ROOT/calls.log                  one line a call: the method and what it was asked for or raised
    HUB_STANDIN_ROOT/limit.json                 optional: {"per_window": N, "window_s": 3600, "retry_after": S | "auto" | null}.
                                                The hub's limit on commit creation for the whole repository: with N
                                                commits taken in the window, the next create_commit is refused with the
                                                library's HfHubHTTPError, status 429, and a Retry-After header ("auto": the
                                                seconds until there is room; null: no header).
    HUB_STANDIN_ROOT/now.txt                    optional: the hub's clock, a number (a test's own clock); else time.time()
    HUB_STANDIN_ROOT/commit_log.jsonl           one line a create_commit that reached the limit check:
                                                {"at", "branch", "message", "ok", "adds", "deletes"}
    HUB_STANDIN_ROOT/lfs.json                   the object store: {sha256: {"size", "pushed_at"}} for every *.safetensors
                                                file ever UPLOADED and not yet deleted for good. create_commit uploads
                                                first (the object is there before the commit is asked for, and stays
                                                when the commit is refused), then commits; a commit that names an object
                                                the store does not hold is refused (400). A listing gives such a file
                                                its sha256; list_lfs_files and permanently_delete_lfs_files work on this
                                                index; a download of a file whose object is gone fails.
    HUB_STANDIN_ROOT/damage.json                optional: {"path": P, "bytes": N}: the next commit that adds P leaves
                                                it N bytes long on the hub (a file that arrived short). Used once.
    HUB_STANDIN_ROOT/lose.json                  optional: {"path": P}: the next commit that adds P lands, and P's object
                                                is then gone from the object store: a pointer whose object is missing.
                                                The listing still gives P its size and sha256; a read of its bytes
                                                (DiskSession, hf_hub_download) fails. Used once.
    HUB_STANDIN_ROOT/probe.json                 optional: {"status": 403}: while it is there, the ranged read of every
                                                large file is redirected and the object store answers with that status
                                                (what one does for a key it will not show: no verdict for the reader).
The ranged read's revision may be a commit id ("<branch>@<n>", quoted by the library); a 206 names the whole size.
A new branch starts from main's tree (create_branch), as on the hub, and its creation counts against limit.json.
preupload_lfs_files is the upload phase by itself; DiskApi.http_session answers the one-byte ranged read.
DiskApi.after_upload, when a test sets it, is called between a commit's upload and its commit.
"""
import datetime
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import urllib.parse

from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
import requests

LFS_SUFFIX = ".safetensors"


_SHA = {}


def file_sha(p):
    """The file's sha256; kept by path, size and modification time, so a listing does not hash a large file again."""
    st = os.stat(p)
    key = (p, st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


class DiskSession:
    """The one HTTP call hf_store makes itself: a ranged GET of a file's first byte at the hub's download address
    (https://huggingface.co/datasets/<repo>/resolve/<revision>/<path>). Answers as the hub and its object store
    do: 206 and one byte; 404 with the hub's own error code for a path that is not there; a redirect and then the
    object store's bare 404 for an LFS pointer whose object is gone; the fail.json modes as the API's calls."""

    def __init__(self, api):
        self.api = api

    def get(self, url, headers=None, timeout=None, stream=False, allow_redirects=True):
        headers = headers or {}
        assert headers.get("Range") == "bytes=0-0", headers.get("Range")
        assert str(headers.get("authorization") or headers.get("Authorization") or "").startswith("Bearer "), "no token header"
        rest = url.split("/resolve/", 1)[1]
        revision, path = rest.split("/", 1)
        revision = urllib.parse.unquote(revision)  # the library quotes it: a commit id "<branch>@<n>"
        self.api._gate("probe", f"{revision}:{path}")
        resp = requests.Response()
        resp.url = url
        resp.raw = __import__("io").BytesIO(b"")
        d = os.path.join(self.api.root, "branches", revision.split("@")[0])
        src = os.path.join(d, *path.split("/"))
        if not os.path.isfile(src):
            resp.status_code = 404
            resp.headers["X-Error-Code"] = "EntryNotFound"
            return resp
        refuse = self.api._json("probe.json", {}).get("status")
        if refuse and src.endswith(LFS_SUFFIX):  # probe.json: the object store's answer to every read of a large file
            self.api._note(f"  probe {refuse} from the object store: {path}")
            first = requests.Response()
            first.status_code, first.url = 302, url
            resp.status_code, resp.url = int(refuse), "https://cdn-lfs.example/objects/" + file_sha(src) + "?X-Amz-Signature=SIGNED"
            resp.history = [first]
            return resp
        if src.endswith(LFS_SUFFIX) and file_sha(src) not in self.api._json("lfs.json", {}):
            self.api._note(f"  probe 404 from the object store: {path}")
            first = requests.Response()
            first.status_code, first.url = 302, url
            resp.status_code, resp.url = 404, "https://cdn-lfs.example/objects/" + file_sha(src)
            resp.history = [first]
            return resp
        resp.status_code = 206
        resp.headers["Content-Range"] = f"bytes 0-0/{os.path.getsize(src)}"
        with open(src, "rb") as f:
            resp.raw = __import__("io").BytesIO(f.read(1))
        return resp


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

    def _now(self):
        try:
            return float(open(os.path.join(self.root, "now.txt")).read())
        except (OSError, ValueError):
            return time.time()

    def _json(self, name, default):
        try:
            return json.load(open(os.path.join(self.root, name)))
        except (OSError, ValueError):
            return default

    def _put(self, name, obj):
        p = os.path.join(self.root, name)
        with open(p + ".tmp", "w") as f:
            json.dump(obj, f)
        os.replace(p + ".tmp", p)

    def _heads(self):
        return self._json("heads.json", {"main": 0})

    def _dir(self, revision):
        branch = (revision or "main").split("@")[0]
        d = os.path.join(self.root, "branches", branch)
        if not os.path.isdir(d):
            self._note(f"  raise RevisionNotFoundError {revision}")
            raise RevisionNotFoundError(f"404 Client Error. Revision Not Found for url: .../{revision}. Invalid rev id: {revision}")
        return branch, d

    # -- the calls hf_store.HfBackend makes, every argument by keyword
    def create_branch(self, *, repo_id, branch, exist_ok=False, repo_type=None, token=None):
        """A new branch starts from main's head, as on the hub: its tree is main's tree. Its creation is counted
        against the repository's commit limit and refused with the same 429 (the rehearsal's logs, 12 Nov 2026)."""
        self._gate("create_branch", branch)
        d = os.path.join(self.root, "branches", branch)
        if os.path.isdir(d):
            return
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._limited(branch, "create_branch", [])
            shutil.copytree(os.path.join(self.root, "branches", "main"), d)

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
                size = os.path.getsize(p)
                lfs = {"sha256": file_sha(p), "size": size} if f.endswith(LFS_SUFFIX) else None
                out.append(type("RepoFile", (), {"path": os.path.relpath(p, d).replace(os.sep, "/"), "size": size, "lfs": lfs})())
        return out

    def _limited(self, branch, message, ops):
        """The repository's commit limit (limit.json), under the commit lock. Returns the time of this commit, or
        raises the library's own error for HTTP 429 as the hub sends it."""
        lim, now = self._json("limit.json", None), self._now()
        rec = {"at": now, "branch": branch, "message": message, "ok": True,
               "adds": sorted(o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationAdd"),
               "deletes": sorted(o.path_in_repo for o in ops if type(o).__name__ != "CommitOperationAdd")}
        log = os.path.join(self.root, "commit_log.jsonl")
        if lim:
            window = float(lim.get("window_s", 3600))
            past = [json.loads(ln) for ln in open(log).read().splitlines()] if os.path.exists(log) else []
            taken = sorted(c["at"] for c in past if c["ok"] and now - c["at"] < window)
            if len(taken) >= int(lim["per_window"]):
                with open(log, "a") as f:
                    f.write(json.dumps({**rec, "ok": False}) + "\n")
                self._note(f"  raise 429 create_commit {branch}")
                resp = requests.Response()
                resp.status_code = 429
                resp.url = f"https://huggingface.co/api/datasets/x/commit/{branch}"
                wait = lim.get("retry_after")
                if wait == "auto":
                    wait = math.ceil(taken[len(taken) - int(lim["per_window"])] + window - now)
                if wait is not None:
                    resp.headers["Retry-After"] = str(wait)
                raise HfHubHTTPError(f"429 Client Error: Too Many Requests for url: {resp.url}\n\nYou have exceeded the rate "
                                     f"limit for repository commit creation ({lim['per_window']} per hour).", response=resp)
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return now

    def _upload(self, operations):
        """Phase one, the upload, as the library does it: every LFS file's object is in the object store BEFORE the
        commit is asked for, and stays there whatever the commit's answer is. An object the store holds already
        is not sent again and keeps its time. Returns the LFS adds."""
        big = [o for o in operations if type(o).__name__ == "CommitOperationAdd" and o.path_in_repo.endswith(LFS_SUFFIX)]
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            objects = self._json("lfs.json", {})
            for o in big:
                sha = file_sha(o.path_or_fileobj)
                if sha not in objects:
                    objects[sha] = {"size": os.path.getsize(o.path_or_fileobj), "pushed_at": self._now()}
                    self._note(f"  upload {o.path_in_repo} {sha[:12]}")
            self._put("lfs.json", objects)
        return big

    def preupload_lfs_files(self, *, repo_id, additions, token=None, repo_type=None, revision=None, create_pr=None,
                            num_threads=5, free_memory=True, gitignore_content=None):
        """The library's own first phase, called by itself: the large files of `additions` go to the object store
        now; a create_commit with the same operations then uploads nothing."""
        self._gate("preupload_lfs_files", str(revision))
        self._dir(revision)
        self._upload(list(additions))

    @property
    def http_session(self):
        """What hf_store.HfBackend.probe reads a byte through, in place of the library's requests session."""
        return DiskSession(self)

    def create_commit(self, *, repo_id, operations, commit_message, revision=None, parent_commit=None, repo_type=None, token=None):
        self._gate("create_commit", f"{revision} {commit_message}")
        branch, d = self._dir(revision)
        operations = list(operations)
        big = self._upload(operations)
        if callable(getattr(self, "after_upload", None)):
            self.after_upload(branch, operations)  # a test's hook: what another party does between the two phases
        # Phase two, the commit: refused for a stale parent, past the limit, or when it names an object that is
        # not in the object store.
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            heads = self._heads()
            if parent_commit is not None and parent_commit != f"{branch}@{heads.get(branch, 0)}":
                raise HfHubHTTPError("412 Client Error: Precondition Failed (a commit has happened since)")
            self._limited(branch, commit_message, operations)
            objects = self._json("lfs.json", {})
            for o in big:
                if file_sha(o.path_or_fileobj) not in objects:
                    self._note(f"  raise 400 create_commit {branch}: {o.path_in_repo} names a missing object")
                    raise HfHubHTTPError(f"400 Client Error: Bad Request for url: .../commit/{branch} (LFS object of "
                                         f"{o.path_in_repo} does not exist)")
            damage = self._json("damage.json", {})
            for o in operations:
                dst = os.path.join(d, *o.path_in_repo.rstrip("/").split("/"))
                if type(o).__name__ == "CommitOperationAdd":
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(o.path_or_fileobj, dst + ".incoming")
                    os.replace(dst + ".incoming", dst)
                    if damage.get("path") == o.path_in_repo:  # the file arrives short: damage.json, used once
                        with open(dst, "r+b") as f:
                            f.truncate(int(damage.get("bytes", 0)))
                        os.remove(os.path.join(self.root, "damage.json"))
                        self._note(f"  damaged {o.path_in_repo}")
                elif o.is_folder:  # CommitOperationDelete reads a path that ends in "/" as a folder
                    shutil.rmtree(dst, ignore_errors=True)
                elif os.path.exists(dst):
                    os.remove(dst)
            lose = self._json("lose.json", {})
            for o in big:  # lose.json, used once: the commit lands, and the object of this file is gone after it
                if lose.get("path") == o.path_in_repo:
                    objects.pop(file_sha(o.path_or_fileobj), None)
                    self._put("lfs.json", objects)
                    os.remove(os.path.join(self.root, "lose.json"))
                    self._note(f"  lost the object of {o.path_in_repo}")
            heads[branch] = heads.get(branch, 0) + 1
            self._put("heads.json", heads)

    def hf_hub_download(self, *, repo_id, filename, revision=None, local_dir=None, repo_type=None, token=None):
        self._gate("hf_hub_download", f"{revision}:{filename}")
        _, d = self._dir(revision)
        src = os.path.join(d, *filename.split("/"))
        if not os.path.isfile(src):
            self._note(f"  raise EntryNotFoundError {filename}")
            raise EntryNotFoundError(f"404 Client Error. Entry Not Found for url: .../resolve/{revision}/{filename}.")
        if src.endswith(LFS_SUFFIX) and file_sha(src) not in self._json("lfs.json", {}):
            self._note(f"  raise EntryNotFoundError {filename} (its object was deleted for good)")
            raise EntryNotFoundError(f"404 Client Error. Entry Not Found for url: .../resolve/{revision}/{filename} (LFS object gone).")
        dst = os.path.join(local_dir, *filename.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def list_lfs_files(self, *, repo_id, repo_type=None, token=None):
        self._gate("list_lfs_files")
        utc = datetime.timezone.utc
        return [type("LFSFileInfo", (), {"oid": sha, "file_oid": "blob" + sha[:8], "size": o["size"],
                                         "pushed_at": datetime.datetime.fromtimestamp(o["pushed_at"], utc)})()
                for sha, o in sorted(self._json("lfs.json", {}).items())]

    def permanently_delete_lfs_files(self, *, repo_id, lfs_files, rewrite_history=True, repo_type=None, token=None):
        self._gate("permanently_delete_lfs_files", f"{len(list(lfs_files))} objects")
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            objects = self._json("lfs.json", {})
            for f in lfs_files:
                objects.pop(f.oid, None)
            self._put("lfs.json", objects)
