#!/usr/bin/env python3
"""A checkpoint and input store on a private Hugging Face repo, for campaigns that outlive their VMs (base 2).

    python3 hf_store.py push   --config STORE.json --branch main --prefix corpus/N --dir DIR
    python3 hf_store.py pull   --config STORE.json --branch main --prefix corpus/N --to DIR
    python3 hf_store.py inputs --config STORE.json --jobs JOBS.json --root /content
    python3 hf_store.py steps  --config STORE.json --branch ckpt-N
    python3 hf_store.py prune  --config STORE.json --branch ckpt-N [--keep 2]
    python3 hf_store.py release --config STORE.json [--grace SECONDS]
    python3 hf_store.py adopt-token  --config STORE.json --from FILE --dir DIR
    python3 hf_store.py forget-token --config STORE.json [--chat CHAT]
Every command but the two token ones takes --limit SECONDS: the call runs in a child that is ended at the limit;
and --commit-budget N and --commit-wait SECONDS (see "Commits" below).

Layout. A "set" is a folder of the repo with a manifest.json beside its files: the format, the kind, and the size
and sha256 of every file. A push is one commit that holds the files and the manifest; a pull fetches the manifest,
then each file, and refuses a file whose size or sha256 differs.

    main      inputs (any prefix), marks/<job>/rung_<step>/ (weights at a mark step), locks/ (the release lock
              and locks/released.json, the ledgers a release has finished with). prune() refuses it.
    ckpt-*    gens/g<step>/ : one transactional checkpoint generation of one arm (model.safetensors,
              optim.safetensors, state.txt, the trainer's MANIFEST.json, events.jsonl) and its manifest.json.
              pruned/g<step>.json : the ledger of a generation a commit removed (its LFS objects and sizes, the
              time), written by that same commit.

Generations. A generation's manifest names its campaign, job, the sha256 of the job's spec and of the inputs the VM
pulled, and the writer (the runner's launch count as "epoch", and a boot id). A generation is never written twice:
a step whose manifest is listed is left alone, and a push names the branch head it was computed against, so a
second writer's commit in between is refused by the store (DirBackend) or by the hub (parent_commit). A resume
(pull_generation) takes the newest generation that verifies by download, gives up one only on a verification
error (a transport error is retried, then raised), refuses a generation of another campaign, job, spec or inputs,
and refuses when the branch lists nothing at or above the step home has seen pushed. Generations are removed only
behind two that the caller has verified: by a download, or by a listing taken AFTER the commit that added the
generation, which shows every file of its manifest at its size and LFS sha256 (listed_whole). The commit that adds a
generation never counts that generation, so a branch holds three for the time between two saves, and FOUR for
one save after a session is replaced: the new VM has verified only the generation it restored, removes nothing
until it has verified one of its own, and its next save then removes two in one commit, each with its ledger.

Storage. Removing a generation from a branch does not free its bytes. release() does, and deletes only what a
commit once named and a later commit removed: the commit that removes a generation writes its ledger
(pruned/g<step>.json) in the same commit, and a release, under a lock file on main, deletes an LFS object only when
(a) a ledger names it, (b) EVERY ledger that names it is older than the grace period, and (c) no branch's tree,
listed under the lock, names it. A ledger that cannot be read stops all deletion in that round. An object that was
uploaded and never committed is in no ledger and is never deleted by a release, however long its commit waits
(what is not closed: an upload that reuses the bytes of an object a ledger names; see release_report). Such objects
are reported ("orphan_bytes") and removed only by hand, `hf_store.py release --orphans`, with no VM writing. The
grace period then covers only a reader of a generation that was just removed (a restore's download), and only if
that download fits inside it; it is never under MIN_RELEASE_GRACE_S. Release runs from home, by the runner, on a timer (runner_multi.py --store-release-storage); a
VM never releases. The lock keeps two releases apart and nothing else.

Commits. The hub limits commit creation per repository (128 an hour on 12 Nov 2026, answered with HTTP 429; a
branch's creation is refused by the same limit), and every writer of a campaign shares the one repo. So: a save is
ONE commit (push_generation adds the generation, removes the pruned ones and writes their ledgers together); a
branch is created once and remembered beside the ledger; and every commit and branch creation a Store makes passes a
client-side budget (CommitBudget: "store_commit_budget_per_hour" in the config, counted over a rolling hour in a
small file, "commit_ledger"). THE BUDGET IS PER WRITER, NOT PER REPOSITORY: the processes that share one ledger file
(a VM's courier and its data job; one runner at home) share one count, and nothing ties the ledgers of different
machines together. The defaults are set so that three arms' ceilings sum under the hub's limit (3 x 24 for the
couriers and 3 x 12 for the runners: 108 of 128); only the hub's own 429 bounds the repository. A generation, a mark,
an input set, a branch and the lock of the release at a campaign's end are essential: with the budget spent they
wait their turn, at most ESSENTIAL_WAIT_S (15 minutes), and are then sent whatever the count is, for the hub to
answer. A timed release's lock, a prune by itself and anything else are refused once the budget is within its
reserve (a quarter) of spent. A call the hub answers with 429 is
raised as RateLimited with the hub's Retry-After (seconds or an HTTP date), and nothing passes the budget for at
least that long, at most MAX_429_WAIT_S (30 minutes).

STORE.json: {"repo": "owner/name", "repo_type": "dataset", "token_file": PATH,
"min_push_interval_s": 0, "branches": {job: branch}, "campaign": NAME, "epoch": N, "min_step": {job: step},
"first_start": [jobs], "mark_ns": "", "push_cap_bytes": 0, "store_commit_budget_per_hour": 0, "commit_ledger": PATH,
"commit_wait_s": 0, "commit_window_s": 3600, "commit_essential_wait_s": 900}. There is no default repo and no
default token path.
"repo": "dir:/some/folder" is a store on local disk with the same rules (tests and local rehearsals; no token). A
jobs file names the inputs a VM needs as "store_inputs": [{"branch": "main", "prefix": P, "to": "chat/upload"}],
"to" relative to --root. "mark_ns" is put before the job's name in its marks' paths (marks/<mark_ns><job>/), so a
second attempt under another campaign name never reads the first one's marks.

The token. It is read from token_file (mode 600, refused otherwise) and handed to the library call by call; it is
not printed, logged, put on a command line or exported, and every error text passes through redact(). On a VM it
arrives as an uploaded file on the VM's disk and is moved to a tmpfs by the runner's next call (adopt_token), so
it is on disk for the seconds between the two. forget_token() removes the file the config names and, always,
every token* file in the tmpfs folder (PK_STORE_TOKEN_DIR, else /dev/shm/pk_store, and the config's "token_dir"),
so an adoption that died half way leaves nothing it cannot find. HF_HUB_DISABLE_IMPLICIT_TOKEN is set here, so no
other download in a process that imported this file sends a token it was not given.

Assumed of Hugging Face and not tested here (the tests run against DirBackend and a recording stand-in for HfApi,
with huggingface_hub pinned to HUB_VERSION): create_commit is atomic, takes adds and deletes (of a folder too,
CommitOperationDelete with a path that ends in "/") together and refuses a stale parent_commit; list_repo_tree gives
each file's size and, for LFS files, its sha256; list_lfs_files names each object by that sha256 (release() checks
this before it deletes anything, and refuses otherwise); the account's storage quota; that a refused commit does not
count against the hub's commit limit (the budget counts every commit it lets through, refused or not). Seen on the
real hub in the rehearsal of 12 Nov 2026: permanently_delete_lfs_files frees the bytes and each object carries its
size; the commit limit and its 429.
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import email.utils
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import select
import shutil
import signal
import sys
import tempfile
import time
import uuid

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

FORMAT = "paperkiln.store/1"
MANIFEST = "manifest.json"
HUB_VERSION = "0.35.3"  # the version the calls below were written and tested against; the boot installs it
GEN_FILES = ("model.safetensors", "optim.safetensors", "state.txt", "MANIFEST.json")  # one trainer generation
GEN_FORMAT = "paperkiln.ckpt_gen/1"  # include/microtorch/checkpoint_txn.hpp
MARK_FILES = ("model.safetensors", "state.txt")  # what jobs_vm.take_snapshot copies
IDENTITY = ("campaign", "job", "spec_sha256", "inputs_sha256")
LOCK = "locks/release.json"
RELEASED = "locks/released.json"  # main: {branch: [steps]}, the ledgers a release has finished with
PRUNED = "pruned"  # a ckpt-* branch: pruned/g<step>.json, the ledger of a generation a commit removed
PRUNED_FORMAT = "paperkiln.pruned/1"
LOCK_STALE_S = 1800
RELEASE_GRACE_S = 7200
# Repos this tool never writes to or prunes: the screens' and cloud_eval's inputs repo holds pinned revisions.
REFUSED_REPOS = ("JohnnyTeutonic/chat-screens-inputs",)
TOKEN_DIR = "/dev/shm/pk_store"  # the VM's tmpfs folder for the adopted token (PK_STORE_TOKEN_DIR moves it)


def refused_repo(repo):
    """True for a repo this tool never writes to. The hub's names do not tell case apart, so neither does this."""
    return str(repo).lower() in {r.lower() for r in REFUSED_REPOS}


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


class RateLimited(StoreError):
    """The hub answered a commit with HTTP 429. The text carries "retry_after_s=N", so the wait survives the pipe
    of a bounded() call."""

    @property
    def retry_after(self):
        m = re.search(r"retry_after_s=(\d+)", str(self))
        return int(m.group(1)) if m else 0


class BudgetSpent(StoreError):
    """The client-side commit budget has no room for this commit now; nothing was sent. The text carries
    "room in N s"."""

    @property
    def wait_s(self):
        m = re.search(r"room in (\d+) s", str(self))
        return int(m.group(1)) if m else 60


ERRORS = {c.__name__: c for c in (StoreError, VerifyError, Missing, IdentityError, ConflictError, RateLimited, BudgetSpent)}
MIN_429_WAIT_S = 60       # nothing passes the budget for at least this long after a 429 that names no wait
MAX_429_WAIT_S = 1800     # nor for longer than this, whatever wait the hub names
ESSENTIAL_WAIT_S = 900    # an essential commit waits this long for room in the budget, then is sent anyway
MIN_RELEASE_GRACE_S = 600  # a release never deletes what a ledger younger than this names
RELEASE_SKEW_S = 120       # added to the grace: a ledger's time is a VM's clock, the release's is home's


def rate_limited(e, now=None):
    """Seconds the hub asks for when `e` is its HTTP 429 (0 when it names none), else None. The wait is read from
    Retry-After, as seconds or as an HTTP date, else from the `t=` of a RateLimit header. Not capped here."""
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    if code != 429 and not (code is None and re.search(r"\b429\b", str(e)) and type(e).__name__ == "HfHubHTTPError"):
        return None
    headers = getattr(resp, "headers", None) or {}
    after = headers.get("Retry-After") or headers.get("retry-after")
    if after is not None:
        try:
            wait = float(str(after).strip())
            if math.isfinite(wait):
                return max(0.0, wait)
        except ValueError:
            pass
        try:  # an HTTP date: the wait is until then
            when = email.utils.parsedate_to_datetime(str(after).strip())
            if when.tzinfo is None:
                when = when.replace(tzinfo=datetime.timezone.utc)
            return max(0.0, when.timestamp() - (time.time() if now is None else now))
        except (TypeError, ValueError, OverflowError):
            pass
    m = re.search(r"\bt=(\d+)", str(headers.get("RateLimit") or headers.get("ratelimit") or ""))
    return float(m.group(1)) if m else 0.0


def capped_429_wait(seconds):
    """The wait that is honoured after a 429: at least a minute, at most half an hour."""
    try:
        seconds = float(seconds or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    if not math.isfinite(seconds):
        seconds = MAX_429_WAIT_S
    return min(max(seconds, MIN_429_WAIT_S), MAX_429_WAIT_S)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def http_status(e):
    """The HTTP status an error carries: its response's, else the one its text names ("HTTP 403", "403 Client
    Error"); None for an error that names none (a transport failure, a time-out)."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    m = re.search(r"\bHTTP (\d{3})\b|\b([1-5]\d\d) (?:Client|Server) Error", str(e))
    return int(m.group(1) or m.group(2)) if m else None


def range_total(resp):
    """The whole size a ranged reply names (Content-Range: "bytes 0-0/N", or "bytes */N" with a 416); None when it
    names none."""
    text = str((getattr(resp, "headers", None) or {}).get("Content-Range") or "")
    m = re.fullmatch(r"\s*bytes\s+(?:\d+-\d+|\*)/(\d+)\s*", text)
    return int(m.group(1)) if m else None


class CommitBudget:
    """At most `per_hour` commits in any rolling hour FROM ONE WRITER, counted in a small JSON file. The processes
    given the same path share the count (a courier and its bounded() children, the data job's separate hf_store.py
    calls on the same VM); nothing ties the files of different machines together, so this is a ceiling for a
    writer, not for the repository. Every commit let through is counted when it is sent, whether or not the hub
    then takes it.

    An essential commit (a generation, a mark, an input set, a branch) may use the whole budget, and when there is
    no room it waits at most `essential_wait` seconds from the moment it first asked: after that it passes whatever
    the count is, and the hub answers. Any other commit is refused once the count is within `reserve` of the
    budget, so a lock or a clean-up never takes an essential commit's place. After a 429 nothing passes, forced or
    not, until the hub's wait is over (block(); between MIN_429_WAIT_S and MAX_429_WAIT_S).

    The file is checked on every read: a wrong type, a number that is not finite or a file that is not JSON makes
    it a malformed ledger, which is moved aside (PATH.bad<time>) and replaced by one that counts half the budget as
    just used; a time in the future (a clock set back) is set to now, and a block never reaches further than
    MAX_429_WAIT_S ahead. A clock set forward ages the count early. When the file cannot be read or written at
    all the count is kept in memory. Nothing here raises but BudgetSpent. Without a path the count is this
    process's and its children's own."""

    def __init__(self, per_hour, path=None, clock=time.time, sleep=time.sleep, reserve=None, window=3600.0,
                 essential_wait=ESSENTIAL_WAIT_S):
        self.per_hour, self.clock, self.sleep = int(per_hour), clock, sleep
        self.window, self.essential_wait = float(window), float(essential_wait)
        self.reserve = max(0, min(self.per_hour - 1, max(2, self.per_hour // 4))) if reserve is None else int(reserve)
        if path is None:
            fd, path = tempfile.mkstemp(prefix="pk_store_commits.", suffix=".json")
            os.close(fd)
            atexit.register(_remove_own, path, os.getpid())
        self.path = path
        self._mem = None  # the count, once the file could not be used

    WINDOW = property(lambda self: self.window)

    def _blank(self):
        return {"commits": [], "blocked_until": 0, "refused": 0, "http_429": 0, "reported_at": None, "essential_since": None}

    def _checked(self, d, now):
        """The ledger with every time brought into [.., now] and every block within reach, or None when it is
        malformed."""
        if not isinstance(d, dict):
            return None
        out = self._blank()
        commits = d.get("commits", [])
        if not isinstance(commits, list):
            return None
        for c in commits:
            if not (isinstance(c, list) and c and _num(c[0])):
                return None
            out["commits"].append([min(float(c[0]), now), str(c[1]) if len(c) > 1 else ""])
        for k in ("refused", "http_429"):
            v = d.get(k, 0) or 0
            if not (_num(v) and v >= 0):
                return None
            out[k] = int(v)
        b = d.get("blocked_until", 0) or 0
        if not _num(b):
            return None
        out["blocked_until"] = min(float(b), now + MAX_429_WAIT_S)
        for k in ("reported_at", "essential_since"):
            v = d.get(k)
            if v is not None and not _num(v):
                return None
            out[k] = None if v is None else min(float(v), now)
        return out

    def _read(self, now):
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return self._blank()
        if not raw.strip():
            return self._blank()  # made a moment ago and not yet written
        try:
            d = self._checked(json.loads(raw.decode("utf-8")), now)
        except ValueError:  # not JSON, or not text
            d = None
        if d is None:  # malformed: aside, and a new one that leaves room for half a budget
            try:
                os.replace(self.path, f"{self.path}.bad{int(now)}")
            except OSError:
                pass
            d = self._blank()
            d["commits"] = [[now, "recovered"] for _ in range(self.per_hour // 2)]
        return d

    def _edit(self, fn):
        now = self.clock()
        if self._mem is None:
            d, ran, out = None, False, None
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path + ".lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    d = self._read(now)
                    d["commits"] = [c for c in d["commits"] if now - c[0] < self.window]
                    out, ran = fn(d, now), True
                    tmp = f"{self.path}.tmp{os.getpid()}"
                    with open(tmp, "w") as f:
                        json.dump(d, f)
                    os.replace(tmp, self.path)
                    return out
            except OSError:
                # The file cannot be used: the count goes on in memory, from the history that could still be
                # read. An edit that had run before the write failed is kept and is not run a second time.
                if d is None:
                    try:
                        d = self._read(now)
                    except OSError:
                        d = self._blank()
                self._mem = d
                if ran:
                    return out
        d = self._mem
        d["commits"] = [c for c in d["commits"] if now - c[0] < self.window]  # it ages here as it does in the file
        d["blocked_until"] = min(d["blocked_until"], now + MAX_429_WAIT_S)
        return fn(d, now)

    def _wait(self, d, now, essential, need, force=False):
        blocked = max(0.0, d["blocked_until"] - now)
        if force:
            return blocked
        limit = self.per_hour if essential else self.per_hour - self.reserve
        times = sorted(c[0] for c in d["commits"])
        over = len(times) + need - limit
        if over > len(times):
            room = self.window  # more than the budget holds at once: never
        else:
            room = max(0.0, times[over - 1] + self.window - now) if over > 0 else 0.0
        if essential:
            if room > 0:
                if d["essential_since"] is None:
                    d["essential_since"] = now  # the wait's clock starts when an essential commit first asks
                room = min(room, max(0.0, d["essential_since"] + self.essential_wait - now))
            else:
                d["essential_since"] = None
        return max(room, blocked)

    def count(self):
        return self._edit(lambda d, now: len(d["commits"]))

    def wait_s(self, essential=True, need=1):
        """Seconds until a commit of this kind would pass; 0 when it would now. For an essential commit this starts
        the clock of its wait."""
        return self._edit(lambda d, now: self._wait(d, now, essential, need))

    def admit(self, what, essential=True, need=1, force=False, patience=0, reserved=False):
        """Count the commit and return, or raise BudgetSpent with nothing counted. need: the commits this one
        stands for, all counted now (a lock is admitted with its unlock: two). reserved: a commit that was counted
        then (that unlock): not counted again. force: counted whatever the count is (a second try of an unlock).
        Reserved and forced commits still wait out a 429. patience: seconds an essential commit may wait here."""
        left = patience
        while True:
            def go(d, now):
                w = self._wait(d, now, essential, need, force=force or reserved)
                if w <= 0:
                    if not reserved:
                        d["commits"] += [[now, what]] * need
                    if essential and not force and not reserved:
                        d["essential_since"] = None
                return w
            w = self._edit(go)
            if w <= 0:
                return
            if essential and w <= left:
                nap = min(w, 5.0)
                self.sleep(nap)
                left -= nap
                continue

            def refused(d, now):
                d["refused"] += 1
            self._edit(refused)
            raise BudgetSpent(f"the commit budget of {self.per_hour} an hour has no room ({what} not sent); "
                              f"room in {math.ceil(w)} s")

    def block(self, seconds):
        """The hub answered 429: nothing passes for capped_429_wait(seconds)."""
        def go(d, now):
            d["blocked_until"] = min(max(d["blocked_until"], now + capped_429_wait(seconds)), now + MAX_429_WAIT_S)
            d["http_429"] += 1
        self._edit(go)

    def report(self, every=None, peek=False):
        """{"commits": sent in the last hour, "budget", "refused": since the last report, "http_429": since the
        last report}, at most once every `every` seconds (default: an hour; 0: now); None in between. The first
        call starts the clock. peek: the numbers as they are, and the hour's report is left to whoever makes it."""
        every = self.window if every is None else every

        def go(d, now):
            if not peek and every and d.get("reported_at") is None:
                d["reported_at"] = now
                return None
            if not peek and every and now - d["reported_at"] < every:
                return None
            out = {"commits": len(d["commits"]), "budget": self.per_hour, "refused": d["refused"], "http_429": d["http_429"]}
            if not peek:
                d["reported_at"], d["refused"], d["http_429"] = now, 0, 0
            return out
        return self._edit(go)

    # -- the branches this writer has seen exist, beside the ledger: one create_branch a branch, across processes
    def _branches(self, repo):
        try:
            with open(self.path + ".branches") as f:
                d = json.load(f)
            if isinstance(d, dict) and d.get("repo") == repo and isinstance(d.get("branches"), list):
                return [b for b in d["branches"] if isinstance(b, str)]
        except (OSError, ValueError):
            pass
        return []

    def knows_branch(self, repo, branch):
        return branch in self._branches(repo)

    def note_branch(self, repo, branch, there=True):
        try:
            with open(self.path + ".lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                have = [b for b in self._branches(repo) if b != branch] + ([branch] if there else [])
                tmp = f"{self.path}.branches.tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump({"repo": repo, "branches": sorted(have)}, f)
                os.replace(tmp, self.path + ".branches")
        except OSError:
            pass


def _remove_own(path, pid):
    if os.getpid() == pid:
        for p in (path, path + ".lock", path + ".branches"):
            try:
                os.remove(p)
            except OSError:
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


def ledger_steps(tree):
    """The steps whose ledger (pruned/g<step>.json) a branch's listing holds, ascending."""
    out = set()
    for p in tree or {}:
        m = re.fullmatch(rf"{PRUNED}/g(\d+)\.json", p)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def campaign_bytes(report, branches, mark_folders):
    """From a release's measurement (release_report), the bytes one campaign holds in the store, each LFS object
    once: what its own branches name under gens/, what its own branches' ledgers name that is still stored, and its
    own marks. Nothing a branch inherited from main, nothing of another campaign, no object that is nobody's."""
    own = {}
    for b in branches:
        own.update(report["gens"].get(b) or {})
        own.update(report["pending"].get(b) or {})
    for m in mark_folders:
        own.update(report["marks"].get(m) or {})
    return sum(own.values())


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
            code = http_status(e)
            if code is not None and http_status(Exception(out["err"])) != code:
                out["err"] += f" [HTTP {code}]"  # the status crosses the pipe when the text does not name it
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

    def tree(self, branch, rev=None):  # rev: a folder has no history; commit() checks the head under its lock
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

    def probe(self, branch, path, rev=None, size=None):
        p = self._p(branch, path)
        return os.path.isfile(p) and (size is None or os.path.getsize(p) == size)

    def commit(self, branch, adds, deletes, message, parent=None, before_send=None):
        self.ensure_branch(branch)
        with open(os.path.join(self.root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if before_send is not None:
                adds = {**adds, **before_send()}
            if parent is not None and parent != self.head(branch):
                raise ConflictError(f"{branch} moved (head {self.head(branch)}, expected {parent})")
            for path, src in sorted(adds.items(), key=lambda kv: kv[0].endswith("/" + MANIFEST) or kv[0] == MANIFEST):
                dst = self._p(branch, path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst + ".incoming")
                os.replace(dst + ".incoming", dst)
            for path in deletes:  # a path that ends in "/" is a folder, removed with all it holds
                p = self._p(branch, path.rstrip("/"))
                if os.path.exists(p):
                    shutil.rmtree(p) if path.endswith("/") else os.remove(p)
                    d = os.path.dirname(p)
                    while d != self._p(branch) and not os.listdir(d):
                        os.rmdir(d)
                        d = os.path.dirname(d)
            with open(os.path.join(self.root, "commits.jsonl"), "a") as f:
                f.write(json.dumps({"branch": branch, "adds": sorted(adds), "deletes": sorted(deletes),
                                    "message": message, "at": time.time()}) + "\n")

    def fetch(self, branch, path, dest, rev=None):
        src = self._p(branch, path)
        if not os.path.isfile(src):
            raise Missing(f"{branch}:{path} is not in the store")
        shutil.copyfile(src, dest)

    def lfs_objects(self):
        """A folder has no object store apart from its files: each file of each branch is its own object, so a
        release here finds nothing unnamed, deletes nothing, and measures what the branches hold."""
        out = {}
        for br in self.branches():
            for path, e in (self.tree(br) or {}).items():
                out.setdefault(e["sha256"], {"ids": [e["sha256"]], "pushed_at": os.path.getmtime(self._p(br, path)),
                                             "bytes": e["bytes"], "ref": None})
        return list(out.values())

    def delete_lfs(self, objects):
        pass


class HfBackend:
    """The same operations on a Hugging Face repo. `api` is huggingface_hub.HfApi() unless a test passes its own.
    Every library call is made with keyword arguments."""

    def __init__(self, repo, repo_type, token, api=None):
        if refused_repo(repo):
            raise StoreError(f"{repo} is the pinned inputs repo; the store must be its own repo")
        self.repo, self.repo_type, self.token = repo, repo_type, token
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api = api
        self._known = {"main"}  # branches known to exist: create_branch is called once for each, not before every commit
        self.upload_path = None  # of the last commit that wrote a ledger: "upload_first", or "fallback (the error)"

    def _kw(self):
        return {"repo_id": self.repo, "repo_type": self.repo_type, "token": self.token}

    def ensure_branch(self, branch):
        if branch not in self._known:
            self.api.create_branch(branch=branch, exist_ok=True, **self._kw())
            self._known.add(branch)

    def known_branch(self, branch, there=True):
        """The caller has seen the branch exist (in a bounded() child, whose own note is lost with it), or not."""
        self._known.add(branch) if there else self._known.discard(branch)

    def knows_branch(self, branch):
        return branch in self._known

    def branches(self):
        return sorted(b.name for b in self.api.list_repo_refs(**self._kw()).branches)

    def head(self, branch):
        try:
            return self.api.repo_info(revision=branch, **self._kw()).sha
        except Exception as e:  # noqa: BLE001 the library's not-found types, by name so a stand-in can raise them
            if type(e).__name__ == "RevisionNotFoundError":
                return None
            raise

    def tree(self, branch, rev=None):
        """The branch's listing; with rev (a commit id from head()), the listing of that commit."""
        try:
            entries = list(self.api.list_repo_tree(recursive=True, revision=rev or branch, **self._kw()))
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

    def commit(self, branch, adds, deletes, message, parent=None, before_send=None):
        """One create_commit: the adds and the deletes together (a delete whose path ends in "/" is a folder's:
        CommitOperationDelete reads the trailing slash). The branch is the caller's to have created
        (Store.ensure_branch, which passes the budget): a commit that names its parent is on a branch the caller
        has read.

        before_send (a ledger's writer): the adds' large files are uploaded first (preupload_lfs_files, which
        create_commit then does not repeat), before_send() is called, and the {path: file} it returns are added
        to the commit that is sent next. What it writes therefore carries the time the commit is sent, not the
        time the upload began. Should the pinned library not take the call as written here (a TypeError or an
        AttributeError), the upload is left to create_commit and before_send() is called just before it: the old
        order, in which the ledger's time is the upload's START. Which of the two ran is kept in `upload_path`
        ("upload_first", or "fallback (the error)"); push_generation hands it back and the courier logs it."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=src) for p, src in sorted(adds.items())]
        self.upload_path = None
        if before_send is not None:
            if ops:
                try:
                    self.api.preupload_lfs_files(additions=ops, revision=branch, **self._kw())
                    self.upload_path = "upload_first"
                except (TypeError, AttributeError) as e:
                    self.upload_path = redact(f"fallback ({type(e).__name__}: {e})", self.token)[:200]
            ops += [CommitOperationAdd(path_in_repo=p, path_or_fileobj=src) for p, src in sorted(before_send().items())]
        ops += [CommitOperationDelete(path_in_repo=p) for p in sorted(deletes)]
        self.api.create_commit(operations=ops, commit_message=message, revision=branch, parent_commit=parent, **self._kw())
        self._known.add(branch)

    def probe(self, branch, path, rev=None, size=None):
        """One byte of a file, read through the hub's download address: True when the bytes are there, False when
        the hub or its object store answers that they are not (a pointer whose object is gone), an error when the
        answer could not be had. The listing cannot tell these apart for an LFS file: its size and hash come from
        the pointer. A ranged GET of the first byte, at the commit `rev` when one is given (the commit the caller's
        listing was taken at), else at the branch's name; the token goes in the request's header, as the library
        builds it, and nowhere else. A redirect to the object store is followed (requests drops the token on the
        way).

        size: the file's size as the listing gives it. With it the reply is held against the listing: a reply that
        names the object's whole size (Content-Range) and names another one is False, and a 416 (no byte in that
        range) is True only for a file listed as empty whose reply names no other size. A 416 that cannot be held
        against a size is an error, not an answer.

        A 403 or a 401, from the hub or from the object store after the redirect, is NO VERDICT: an error. An
        object store may answer 403 for a key that is not there, and it answers 403 for a signed address that has
        expired or that it refuses; one reply cannot tell these apart, and "gone" is never taken back (the
        generation is not whole for good, and a run's last one halts its campaign). What keeps such a check from
        reading "unread" for ever without a word is the caller's count: the courier reports a check that stays
        unread ckpt_courier.UNREAD_REPORT_AFTER times running, with this status (log, status and
        CHAT/COURIER_CHECK_UNREAD), and the runner repeats it at home. The error's text names the HTTP status and
        where it came from, and not the address (after a redirect that is a signed one)."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
        url = hf_hub_url(repo_id=self.repo, filename=path, repo_type=self.repo_type, revision=rev or branch)
        session = getattr(self.api, "http_session", None) or get_session()
        r = session.get(url, headers={**build_hf_headers(token=self.token), "Range": "bytes=0-0"}, timeout=30,
                        stream=True, allow_redirects=True)
        try:
            where = "from the object store, after the hub's redirect" if r.history else "from the hub"
            total = range_total(r)
            if r.status_code == 416:
                if size == 0 and total in (None, 0):
                    return True  # there, and empty, as listed
                if size and total is not None and total != size:
                    return False  # the object there is of another size than the listing gives the file
                raise StoreError(f"HTTP 416 on the byte read of {branch}:{path} ({where}): no byte in the range, and "
                                 f"the reply cannot be held against the listed size ({size})")
            try:
                hf_raise_for_status(r)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "EntryNotFoundError":
                    return False  # the hub: no such file at this revision
                if r.status_code in (404, 410) and r.history and type(e).__name__ not in ("RepositoryNotFoundError", "RevisionNotFoundError"):
                    return False  # the object store, after the hub's redirect: no such object
                raise StoreError(f"HTTP {r.status_code} on the byte read of {branch}:{path} ({where}; "
                                 f"{type(e).__name__})") from None
            if size is not None and total is not None and total != size:
                return False  # a reply for an object of another size than the listing gives the file
            return len(next(r.iter_content(1), b"")) == 1 or size == 0
        finally:
            r.close()

    def fetch(self, branch, path, dest, rev=None):
        """One file. Only the hub saying "no such file" or "no such branch" is Missing; a repository that is not
        found (what a wrong or expired token gets), a refused login and a transport failure are raised as they are."""
        stage = tempfile.mkdtemp(prefix=".hfstage.", dir=os.path.dirname(os.path.abspath(dest)))
        try:
            got = self.api.hf_hub_download(filename=path, revision=rev or branch, local_dir=stage, **self._kw())
            os.replace(got, dest)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("EntryNotFoundError", "RevisionNotFoundError"):
                raise Missing(f"{branch}:{path} is not in the store") from None
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def lfs_objects(self):
        """[{"ids": the object's ids as the library gives them, "pushed_at": epoch seconds or None, "bytes": its
        size or None, "ref": handle}]"""
        out = []
        for f in self.api.list_lfs_files(**self._kw()):
            at, size = getattr(f, "pushed_at", None), getattr(f, "size", None)
            out.append({"ids": [x for x in (getattr(f, "oid", None), getattr(f, "file_oid", None)) if x],
                        "pushed_at": at.timestamp() if hasattr(at, "timestamp") else None,
                        "bytes": size if isinstance(size, int) else None, "ref": f})
        return out

    def delete_lfs(self, objects):
        self.api.permanently_delete_lfs_files(lfs_files=[o["ref"] for o in objects], rewrite_history=False, **self._kw())


# ---------------------------------------------------------------- the store
class Store:
    def __init__(self, backend, log=None, tries=4, delay=20, mark_ns="", budget=None, budget_wait=0, clock=time.time):
        """budget: a CommitBudget every commit of this Store passes, or None for no limit of its own. budget_wait:
        seconds an essential commit may wait inside the call for its turn (0: it is refused and the caller waits).
        clock: the time a pruned generation's ledger carries."""
        self.b, self._log, self.tries, self.delay = backend, log, tries, delay
        self.mark_ns = mark_ns or ""
        self.budget, self.budget_wait, self.clock = budget, budget_wait, clock
        self._manifests = {}

    def _limited(self, e, what):
        """Raise RateLimited when `e` is the hub's 429 (its wait kept, and the budget closed for that long, between
        a minute and half an hour); return otherwise, for the caller to raise `e` as itself."""
        wait = rate_limited(e)
        if wait is None:
            return
        if self.budget:
            self.budget.block(wait)
        text = redact(str(e), self.b.token).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
        raise RateLimited(f"HTTP 429 from the store on {what} ({text}); "
                          f"retry_after_s={math.ceil(capped_429_wait(wait))}") from None

    def ensure_branch(self, branch):
        """The branch exists: created at most once by this writer. The hub counts a branch's creation against the
        repository's commit limit and refuses it with the same 429 (the rehearsal's logs), so it passes the budget
        as an essential commit, and its 429 is read as a commit's. That it exists is kept in this process and, with
        a budget, in a file beside the ledger, so the data job's separate processes ask once between them."""
        b = self.b
        if not hasattr(b, "knows_branch"):
            return b.ensure_branch(branch)  # a folder on disk: nothing is asked of a hub
        if b.knows_branch(branch):
            return
        if self.budget and self.budget.knows_branch(b.repo, branch):
            b.known_branch(branch)
            return
        if self.budget:
            self.budget.admit("branch", essential=True, patience=self.budget_wait)
        try:
            b.ensure_branch(branch)
        except Exception as e:  # noqa: BLE001
            self._limited(e, f"the creation of {branch}")
            raise
        if self.budget:
            self.budget.note_branch(b.repo, branch)

    def commit(self, branch, adds, deletes, message, parent=None, essential=True, need=1, force=False, reserved=False,
               late=None):
        """The one way a commit leaves this Store: through the budget, and with the hub's 429 read. Raises
        BudgetSpent (nothing was sent) or RateLimited (the hub refused; its wait is kept and the budget lets
        nothing through until it is over); any other failure as itself. A commit that names no parent is on a
        branch nobody has read: the branch is created first, once (ensure_branch). late: called when the adds'
        large files are uploaded and the commit is about to be sent; the {path: file} it returns go into the same
        commit (a ledger, whose time must be that moment's)."""
        if parent is None:
            self.ensure_branch(branch)
        if self.budget:
            self.budget.admit(message.split(":")[0], essential=essential, need=need, force=force, reserved=reserved,
                              patience=self.budget_wait if essential else 0)
        try:
            if late is None:
                self.b.commit(branch, adds, deletes, message, parent=parent)
            else:
                self.b.commit(branch, adds, deletes, message, parent=parent, before_send=late)
        except Exception as e:  # noqa: BLE001
            self._limited(e, f"a commit to {branch}")
            if type(e).__name__ == "RevisionNotFoundError" and hasattr(self.b, "known_branch"):
                self.b.known_branch(branch, there=False)  # it was there once and is gone: created again next time
                if self.budget:
                    self.budget.note_branch(self.b.repo, branch, there=False)
            raise

    def log(self, msg):
        msg = redact(msg, self.b.token)
        if self._log:
            self._log(msg)
        else:
            print(msg, flush=True)

    def fetch(self, branch, path, dest, rev=None):
        """One file, a transport failure tried again (Missing is an answer, not a failure)."""
        for i in range(self.tries):
            try:
                return self.b.fetch(branch, path, dest, rev=rev) if rev else self.b.fetch(branch, path, dest)
            except Missing:
                raise
            except Exception as e:  # noqa: BLE001
                if i == self.tries - 1:
                    raise
                self.log(f"fetch of {branch}:{path} failed ({type(e).__name__}: {e}); try {i + 2} of {self.tries}")
                time.sleep(self.delay)

    # -- sets
    def push_set(self, branch, prefix, folder, kind="files", names=None, meta=None, parent=None, deletes=(), late=None):
        """Hash `names` (default: every file in folder, not recursive) and commit them with a manifest. One commit,
        which also removes `deletes` and adds what `late()` returns at the moment the commit is sent
        (push_generation's pruned generations and their ledgers)."""
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
            self.commit(branch, adds, list(deletes), f"{kind}: {prefix}", parent=parent, late=late)
        self._manifests[(branch, prefix)] = man
        return man

    def read_manifest(self, branch, prefix, fresh=False, rev=None):
        key = (branch, prefix)
        if fresh or key not in self._manifests:
            with tempfile.TemporaryDirectory(prefix="store_manifest.") as tmp:
                p = os.path.join(tmp, MANIFEST)
                self.fetch(branch, f"{prefix}/{MANIFEST}", p, rev=rev)
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

    def listed_whole(self, branch, step, files=None, probe=True):
        """True when a listing taken NOW shows the generation whole and its large files are really there; False
        when a COMPLETED comparison says otherwise; an error when anything could not be read. The three are kept
        apart: an unread manifest or listing is "not verified yet", never "damaged" (the second review, 12 Nov
        2026: a manifest fetch that failed after a good listing read as not whole for good).

        Whole: the manifest is listed and, read again from the store, names this step (and, with `files`, exactly
        the sizes and hashes the pusher computed from its own bytes); every file of it is listed at that size and,
        for an LFS file, that sha256; and (probe) one byte of each LFS file can be read. The listing's size and
        hash of an LFS file come from its pointer, so only the read shows a pointer whose object is gone.
        All three reads are of ONE commit, the branch's head when this check looked: the listing is taken at it,
        and the manifest and the bytes are read at it, never at the moving branch name, so a save that lands in
        between changes nothing here. Each byte read is held against the file's listed size (HfBackend.probe).
        Reads, not commits. This, after the commit that added the generation, is what makes it count as verified."""
        prefix = f"gens/g{step}"
        self._manifests.pop((branch, prefix), None)
        head = self.b.head(branch)
        tree = self.b.tree(branch, rev=head) if head else None
        if tree is None:
            raise StoreError(f"{branch} is not listed")
        if f"{prefix}/{MANIFEST}" not in tree:
            return False
        try:
            man = self.read_manifest(branch, prefix, fresh=True, rev=head)
        except VerifyError:
            return False  # read, and not a manifest
        if man.get("step") != step or not set(GEN_FILES) <= set(man["files"]) or (files is not None and man["files"] != files):
            return False
        large = []
        for f, want in man["files"].items():
            e = tree.get(f"{prefix}/{f}")
            if not e or e["bytes"] != want["bytes"] or (e.get("sha256") and e["sha256"] != want["sha256"]):
                return False
            if e.get("sha256"):
                large.append((f"{prefix}/{f}", e["bytes"]))
        return all(self.b.probe(branch, p, rev=head, size=n) for p, n in large) if probe else True

    def push_generation(self, branch, job, step, folder, identity=None, parent=None, prune=None):
        """One generation, staged in `folder` (GEN_FILES and events.jsonl), as one commit computed against the
        branch head `parent`. A step already listed is never written again.

        prune: {"keep": 2, "verified": steps, "protect": steps}, or None for no removal. With it the SAME commit
        removes the generations behind the `keep` newest of the caller's verified steps that are whole in the
        listing the commit is computed against. The generation being added NEVER counts among them: whether it
        arrived whole is known only from a listing after this commit (listed_whole), so it can protect nothing
        yet, and the branch holds one generation more until the next save (two more, four in all, for the one save
        after a session is replaced: the new VM's first save has one verified generation, the restored one, and
        removes nothing; its second removes two, each with its own ledger). The rules are prune()'s: nothing is
        removed with fewer than `keep` such generations, nothing at or above the lowest step in `protect`, nothing
        outside gens/. The same commit writes each removed generation's ledger, pruned/g<step>.json (its LFS
        objects and sizes, the time), which is what lets a release delete those objects and no others, and removes
        the ledgers a release has finished with (main:locks/released.json). The ledger's time is taken after the
        new generation's files are hashed and uploaded, immediately before the commit is sent, so a slow upload
        does not use up the grace a reader of the removed generation is owed. The removed steps come back as
        "pruned" beside the manifest's own keys, and with them "ledger_upload": which order ran in a commit that
        wrote a ledger on the hub ("upload_first", or "fallback (the error)": HfBackend.commit), else None."""
        if not branch.startswith("ckpt-"):
            raise StoreError(f"generations go to a ckpt-* branch, not {branch}")
        check_generation(folder, step)
        tree = (self.b.tree(branch, rev=parent) if parent else self.b.tree(branch)) or {}
        if step in self.steps(branch, tree):
            raise ConflictError(f"{branch}: g{step} is already in the store; a generation is written once")
        gone = []
        if prune:
            gone, _ = self._prune_plan(branch, int(prune.get("keep", 2)), prune.get("verified") or (),
                                       prune.get("protect") or (), tree)
        names = [*GEN_FILES] + (["events.jsonl"] if os.path.exists(os.path.join(folder, "events.jsonl")) else [])
        with tempfile.TemporaryDirectory(prefix="store_ledger.") as tmp:
            finished = self._finished_ledgers(branch, gone, tree) if prune else []
            man = self.push_set(branch, f"gens/g{step}", folder, kind="generation", names=names,
                                meta={**(identity or {}), "job": job, "step": step}, parent=parent,
                                deletes=[f"gens/g{s}/" for s in gone] + finished,
                                late=(lambda: self._write_ledgers(branch, gone, tree, tmp)) if gone else None)
        for s in gone:
            self._manifests.pop((branch, f"gens/g{s}"), None)
        return {**man, "pruned": gone, "ledger_upload": getattr(self.b, "upload_path", None) if gone else None}

    def _prune_plan(self, branch, keep, verified, protect, tree):
        """(steps to remove, the two it is behind): the generations of `tree` below the `keep` newest of the
        verified steps that are whole in that listing. ([], "") with fewer than `keep`."""
        if keep < 2:
            raise StoreError("prune keeps at least two generations")
        for key in [k for k in self._manifests if k[0] == branch]:
            del self._manifests[key]
        good = [s for s in self.steps(branch, tree) if s in set(verified) and self.whole_in_listing(branch, s, tree)]
        if len(good) < keep:
            return [], ""
        floor = min([good[-keep], *protect])
        gone = sorted({gen_step(p.split("/")[1]) for p in tree if p.startswith("gens/")
                       and gen_step(p.split("/")[1]) is not None and gen_step(p.split("/")[1]) < floor})
        return gone, f"behind verified g{good[-keep]} and g{good[-1]}"

    def _write_ledgers(self, branch, gone, tree, tmp):
        """{path: file}: the ledgers of the generations `gone` of `tree`, written into tmp NOW, with this moment as
        their time. Called immediately before the commit that removes them is sent."""
        adds, now = {}, self.clock()
        for s in gone:
            objects = [{"path": p, "sha256": e["sha256"], "bytes": e["bytes"]} for p, e in sorted(tree.items())
                       if p.startswith(f"gens/g{s}/") and e.get("sha256")]
            p = os.path.join(tmp, f"g{s}.json")
            with open(p, "w") as f:
                json.dump({"format": PRUNED_FORMAT, "branch": branch, "step": s, "at": now, "objects": objects}, f)
            adds[f"{PRUNED}/g{s}.json"] = p
        return adds

    def _finished_ledgers(self, branch, gone, tree):
        """[paths] of the branch's older ledgers that a release has finished with (main:locks/released.json), to
        remove in this commit. Whatever is wrong with that record, nothing is removed and the save goes on."""
        have = [s for s in ledger_steps(tree) if s not in gone]
        if not have:
            return []
        try:
            done = self._released().get(branch) or []
        except Exception as e:  # noqa: BLE001 main not read: the ledgers stay until a later save
            self.log(f"{branch}: the record of finished ledgers was not read ({type(e).__name__}); none is removed now")
            done = []
        return [f"{PRUNED}/g{s}.json" for s in have if s in done]

    def _released(self):
        """main:locks/released.json, {branch: [steps]}: the ledgers whose objects a release found all gone or all
        named again. {} when there is none. Checked: a record that is not {text: [whole numbers]} throughout is
        said in the log and taken for no record at all, so a save never fails on it and never removes a ledger on
        its word."""
        with tempfile.TemporaryDirectory(prefix="store_lock.") as tmp:
            p = os.path.join(tmp, "r")
            try:
                self.b.fetch("main", RELEASED, p)
            except Missing:
                return {}
            try:
                with open(p, "rb") as f:
                    d = json.loads(f.read().decode("utf-8"))
            except ValueError:
                d = None
        ok = isinstance(d, dict) and all(isinstance(k, str) and isinstance(v, list)
                                         and all(isinstance(s, int) and not isinstance(s, bool) for s in v) for k, v in d.items())
        if not ok:
            self.log(f"main:{RELEASED} is not well formed; ignored (no ledger is removed on its word)")
            return {}
        return d

    def _read_ledger(self, branch, step, rev=None):
        """A ledger as its branch holds it (at the commit `rev`, when given), checked; None for one that cannot be
        read or is not well formed. A release that meets one deletes nothing at all in that round."""
        with tempfile.TemporaryDirectory(prefix="store_ledger.") as tmp:
            p = os.path.join(tmp, "l")
            try:
                self.fetch(branch, f"{PRUNED}/g{step}.json", p, rev=rev)
                with open(p, "rb") as f:
                    d = json.loads(f.read().decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                self.log(f"{branch}: ledger of g{step} not read ({type(e).__name__})")
                return None
        ok = isinstance(d, dict) and d.get("format") == PRUNED_FORMAT and _num(d.get("at")) and isinstance(d.get("objects"), list) \
            and all(isinstance(o, dict) and isinstance(o.get("sha256"), str) and _num(o.get("bytes")) for o in d["objects"])
        if not ok:
            self.log(f"{branch}: ledger of g{step} is not well formed")
            return None
        return d

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
        """Remove the generations older than the `keep` newest of `verified`: the steps the caller has verified, by
        download or by a listing taken after the commit that added them, and that are still whole in the listing.
        With fewer than `keep` such generations nothing is removed. Nothing at or above the lowest step in
        `protect` (a mark staged and not yet pushed) is removed. Never main, never a path outside gens/. Returns
        the steps removed. A commit of its own, conditional on the head it read, with the removed generations'
        ledgers in it, for a hand's clean-up (the CLI); the courier removes in the commit of its next generation
        (push_generation's `prune`). Not essential to the budget."""
        if not branch.startswith("ckpt-"):
            raise StoreError(f"prune refuses {branch}: only ckpt-* branches hold generations")
        head = self.b.head(branch)
        tree = (self.b.tree(branch, rev=head) if head else self.b.tree(branch)) or {}
        gone, why = self._prune_plan(branch, keep, verified, protect, tree)
        if not gone:
            return []
        with tempfile.TemporaryDirectory(prefix="store_ledger.") as tmp:
            finished = self._finished_ledgers(branch, gone, tree)
            self.commit(branch, {}, [f"gens/g{s}/" for s in gone] + finished, f"prune: generations {gone} {why}",
                        parent=head, essential=False, late=lambda: self._write_ledgers(branch, gone, tree, tmp))
        for s in gone:
            self._manifests.pop((branch, f"gens/g{s}"), None)
        return gone

    def release(self, owner, grace=RELEASE_GRACE_S, now=None):
        """release_report()'s count of deleted objects, or None when another releaser holds the lock."""
        r = self.release_report(owner, grace=grace, now=now)
        return None if r["held"] else r["objects"]

    def release_report(self, owner, grace=RELEASE_GRACE_S, now=None, orphans=False, essential=False):
        """Free the storage of removed generations, and nothing else. An LFS object is deleted for good only when
        (a) at least one ledger (pruned/g<step>.json on some branch, written by the commit that removed the
        generation) names it; (b) EVERY ledger that names it is older than `grace` + RELEASE_SKEW_S seconds; and
        (c) no branch's tree names it. Heads, trees and ledgers are all read here, under the lock, each tree and
        each ledger at the commit that was its branch's head when the release looked (never by the moving branch
        name). So what is deleted was committed once and has since been removed from every place that held it: an
        object on main, a mark's weights, an object a kept generation shares with a removed one, and an object
        that was uploaded and is not committed yet are never deleted, whatever their age. Two removed generations
        that held the same bytes (a run and its control) are released together, once the later ledger is old
        enough. A ledger that cannot be read or is not well formed stops ALL deletion in that round: it may name an
        object another ledger also names, and a round later it may be readable.

        The grace. A ledger's time is the removing VM's clock, taken just before its commit is sent; RELEASE_SKEW_S
        is added here for a difference between that clock and this one. `grace` protects a reader of a generation
        that was just removed (a restore that listed it before the removal) ONLY IF ITS DOWNLOAD FITS INSIDE IT;
        there is no lease, and a download that takes longer can lose the object under it. What keeps that from
        mattering: a restore takes the newest generation, and a generation is removed only when two newer ones
        have been verified by its branch's one writer, which the runner starts only after the session before it is
        stopped. Refused under MIN_RELEASE_GRACE_S.

        Not closed: an upload that reuses the bytes of an object a ledger names (the same weights pushed again
        under another path). The hub skips such an upload; if every ledger naming the object is old and no tree
        names it yet, a release in the seconds between that upload's check and its commit deletes it, and the
        commit names a missing object. The post-commit check (listed_whole) reads a byte of each large file and
        says so; nothing is removed on that generation's account.

        One releaser at a time, by a lock file on main: two commits, the lock and its removal, both counted when
        the lock is taken. The removal's commit also records on main (locks/released.json) the ledgers that are
        finished with (every object gone, or named by a tree again); their branch's next save removes them.

        orphans=True (by hand only, `hf_store.py release --orphans`, WITH NO VM WRITING): also delete the objects
        that no tree and no ledger names and that were pushed more than `grace` ago: uploads whose commit never
        came, and what was removed before there were ledgers. Beside a writer this can delete an upload from under
        its commit; the timed release never does it.

        A tree that cannot be read REFUSES the round (StoreError; nothing is deleted and the lock is removed): a
        branch whose head is not found, or whose listing at that head is not given, is never taken for a branch
        that names nothing, because what it names would then be deleted from under it.

        essential=True (the release at a campaign's end): its lock may use the whole commit budget, without the
        reserve, and from the moment it first asks it is owed the budget's essential wait, after which it passes
        whatever the count is (CommitBudget). The waiting is the caller's: with no room now this returns held,
        and "why" carries the seconds until there is room.

        Returns {"held": the lock was not taken (another releaser holds it, main could not be read, or the budget
        had no room; "why" says which, the budget as "budget: room in N s"), "objects": deleted, "bytes": freed}
        and, for a release that ran, a measurement in which every
        object counts once: "gens" {branch: {sha256: bytes}} (what each branch names under gens/), "marks" {folder
        under marks/: {sha256: bytes}}, "pending" {branch: {sha256: bytes}} (in that branch's ledgers, still
        stored, not named), "orphan_objects" and "orphan_bytes" (in no tree and no ledger, still stored),
        "stored_bytes" (every object left), "ledgers", "ledgers_done" and "ledgers_unread" (with any unread the
        measurement is incomplete: what those ledgers hold is in nobody's "pending"). A size the store does not
        give is not taken for zero: "stored_bytes" and "orphan_bytes" are then None and "unknown_sizes" says how
        many. Refuses when the listing's hashes cannot be matched to the object ids."""
        if grace < MIN_RELEASE_GRACE_S:
            raise StoreError(f"release refused: a grace of {grace:.0f} s is under the minimum of {MIN_RELEASE_GRACE_S} s "
                             "(a restore may be reading a generation that was just removed)")
        now = time.time() if now is None else now
        mine = self._lock(owner, now, essential=essential)
        if not mine:
            return {"held": True, "objects": 0, "bytes": 0, "why": self._lock_why}
        done = None
        try:
            heads = {br: self.b.head(br) for br in self.b.branches()}
            trees = {br: self.b.tree(br, rev=h) if h else None for br, h in heads.items()}
            unlisted = sorted(br for br, t in trees.items() if t is None)
            if unlisted:
                raise StoreError(f"release refused: the tree of {', '.join(unlisted)} could not be read (no head, or no "
                                 "listing at it); an unread tree is never taken for an empty one, and nothing is "
                                 "deleted in this round")
            named = {e["sha256"] for t in trees.values() for e in t.values() if e.get("sha256")}
            gens = {br: {e["sha256"]: e["bytes"] for p, e in t.items() if p.startswith("gens/") and e.get("sha256")}
                    for br, t in trees.items() if br != "main"}
            marks = {}
            for p, e in trees.get("main", {}).items():
                parts = p.split("/")
                if len(parts) >= 3 and parts[0] == "marks" and e.get("sha256"):
                    marks.setdefault(parts[1], {})[e["sha256"]] = e["bytes"]
            ledgers = {(br, s): self._read_ledger(br, s, rev=heads[br]) for br, t in sorted(trees.items()) for s in ledger_steps(t)}
            unread = sorted(k for k, led in ledgers.items() if led is None)
            young = set()  # objects that some ledger younger than the grace names
            ledgered = set()
            for led in ledgers.values():
                for o in (led or {}).get("objects", []):
                    ledgered.add(o["sha256"])
                    if not now - led["at"] > grace + RELEASE_SKEW_S:
                        young.add(o["sha256"])
            objs = self.b.lfs_objects()
            by_id = {i: o for o in objs for i in o["ids"]}
            if not named <= set(by_id):
                raise StoreError("release refused: the trees name objects the LFS listing does not show by sha256")
            dead, pending, finished = {}, {}, {}
            for (br, s), led in sorted(ledgers.items()):
                if led is None:
                    continue
                waiting = False
                for o in led["objects"]:
                    obj = by_id.get(o["sha256"])
                    if obj is None or o["sha256"] in named:
                        continue  # gone already, or named by a tree again: counted where it is named
                    if o["sha256"] not in young and not unread:
                        dead[id(obj)] = (obj, o["bytes"])  # every ledger that names it is old; it goes, once
                    else:
                        pending.setdefault(br, {})[o["sha256"]] = o["bytes"]
                        waiting = True
                if not waiting:
                    finished.setdefault(br, []).append(s)
            if unread:
                self.log(f"release: {len(unread)} ledger(s) not read ({', '.join(f'{b}:g{s}' for b, s in unread)}); nothing "
                         "is deleted in this round")
            orphan = [o for o in objs if not named.intersection(o["ids"]) and not ledgered.intersection(o["ids"])]
            if orphans and not unread:
                for o in orphan:
                    if o["pushed_at"] is not None and now - o["pushed_at"] > grace:
                        dead[id(o)] = (o, o.get("bytes"))
            if dead:
                self.b.delete_lfs([o for o, _ in dead.values()])
            done = finished
            left = [o for o in objs if id(o) not in dead]
            left_orphans = [o for o in orphan if id(o) not in dead]
            unknown = sum(1 for o in left if o.get("bytes") is None)
            freed = sum((o.get("bytes") if o.get("bytes") is not None else n) or 0 for o, n in dead.values())
            return {"held": False, "objects": len(dead), "bytes": freed,
                    "stored_bytes": None if unknown else sum(o["bytes"] for o in left),
                    "orphan_objects": len(left_orphans),
                    "orphan_bytes": None if any(o.get("bytes") is None for o in left_orphans) else sum(o["bytes"] for o in left_orphans),
                    "unknown_sizes": unknown, "gens": gens, "marks": marks, "pending": pending,
                    "pending_bytes": sum({k: v for d in pending.values() for k, v in d.items()}.values()),
                    "ledgers": len(ledgers), "ledgers_done": sum(len(v) for v in finished.values()),
                    "ledgers_unread": len(unread)}
        finally:
            self._unlock(mine, done)
    def _read_lock(self, head):
        """The lock file as main's commit `head` holds it; {} for one that is not JSON. A lock that cannot be
        fetched raises: an unread lock is never taken for a stale one."""
        with tempfile.TemporaryDirectory(prefix="store_lock.") as tmp:
            p = os.path.join(tmp, "l")
            self.b.fetch("main", LOCK, p, rev=head)
            try:
                with open(p) as f:
                    held = json.load(f)
            except ValueError:
                return {}
        return held if isinstance(held, dict) else {}

    def _lock(self, owner, now, essential=False):
        """Take the release lock; the lock's record, or None when it is not ours. In this order: main's head is
        read; the lock is looked for in the tree of THAT commit; the lock is committed with that commit as its
        parent. So a lock (or any other commit) that reaches main after the head was read refuses this commit, and
        a lock that was there before is seen. Two releasers can never both hold it. With a commit budget the lock
        is not essential, is taken only with room for itself and its removal, and both are counted here. An
        essential lock (the release at a campaign's end) may use the whole budget and is owed the budget's
        essential wait. Why a lock was not taken is left in self._lock_why."""
        self._lock_why = "another releaser holds the lock"
        if self.budget:
            w = self.budget.wait_s(essential=essential, need=2)
            if w > 0:
                self._lock_why = f"budget: room in {math.ceil(w)} s"
                self.log(f"release lock not taken: the commit budget has no room for it now (room in {math.ceil(w)} s)")
                return None
        try:
            self.b.ensure_branch("main")
            head = self.b.head("main")
            tree = self.b.tree("main", rev=head) if head else None
            if tree is None:  # an unread main is never taken for one that holds no lock
                self._lock_why = "main could not be listed"
                return None
            if LOCK in tree:
                if now - self._read_lock(head).get("at", 0) < LOCK_STALE_S:
                    return None
        except Exception as e:  # noqa: BLE001 main or the lock could not be read: someone else may hold it
            self._lock_why = f"main or the lock could not be read ({type(e).__name__})"
            return None
        mine = {"owner": owner, "at": now, "nonce": uuid.uuid4().hex}
        with tempfile.TemporaryDirectory(prefix="store_lock.") as tmp:
            p = os.path.join(tmp, "l")
            with open(p, "w") as f:
                json.dump(mine, f)
            try:
                self.commit("main", {LOCK: p}, [], f"lock: release by {owner}", parent=head, essential=essential, need=2)
            except Exception as e:  # noqa: BLE001 main moved since the head was read, or the hub refused
                self._lock_why = f"the lock's commit was refused ({type(e).__name__})"
                return None
        return mine

    def _unlock(self, mine, done=None):
        """Remove the lock only while it is the one this release took (same nonce), with a conditional commit on
        the head it was read at. Main moves under it (marks, inputs), so it is read again a few times. The first
        try was counted with the lock; a later one is counted as it is sent. Neither is sent into a 429's wait: a
        lock left behind goes stale by itself. done ({branch: [steps]}, from a release that ran to its end): the
        same commit writes it to locks/released.json, or removes that file when there is nothing to say."""
        err = None
        for i in range(4):
            try:
                head = self.b.head("main")
                tree = self.b.tree("main", rev=head) if head else None
                if tree is None:
                    raise StoreError("main could not be listed")  # not "no lock there": it is read again
                if LOCK not in tree:
                    return
                if self._read_lock(head).get("nonce") != mine["nonce"]:
                    self.log("release lock is another releaser's now; left as it is")
                    return
                with tempfile.TemporaryDirectory(prefix="store_lock.") as tmp:
                    adds, deletes = {}, [LOCK]
                    if done:
                        p = os.path.join(tmp, "r")
                        with open(p, "w") as f:
                            json.dump({br: sorted(steps) for br, steps in sorted(done.items())}, f)
                        adds[RELEASED] = p
                    elif done is not None and RELEASED in tree:
                        deletes.append(RELEASED)
                    self.commit("main", adds, deletes, "unlock: release", parent=head, essential=False,
                                reserved=i == 0, force=i > 0)
                return
            except (RateLimited, BudgetSpent) as e:  # the hub's wait is on: the lock goes stale by itself
                err = e
                break
            except Exception as e:  # noqa: BLE001 a lock left behind goes stale by itself
                err = e
        self.log(f"release lock not removed ({type(err).__name__}: {err}); it expires in {LOCK_STALE_S} s")

    # -- marks
    def mark_prefix(self, job, step):
        return f"marks/{self.mark_ns}{job}/rung_{step}"

    def mark_steps(self, job, tree=None):
        tree = self.b.tree("main") if tree is None else tree
        out = set()
        for p in tree or {}:
            parts = p.split("/")
            if len(parts) == 4 and parts[0] == "marks" and parts[1] == self.mark_ns + job and parts[3] == MANIFEST \
                    and parts[2].startswith("rung_") and parts[2][5:].isdigit():
                out.add(int(parts[2][5:]))
        return sorted(out)

    def push_mark(self, job, step, folder, identity=None):
        """Weights and state.txt at a mark step, to main, once: a mark already in the store is left as it is."""
        if step in self.mark_steps(job):
            return False
        if file_step(os.path.join(folder, "state.txt")) != step:
            raise StoreError(f"{job}: mark folder is not at step {step}")
        self.push_set("main", self.mark_prefix(job, step), folder, kind="mark", names=list(MARK_FILES),
                      meta={**(identity or {}), "job": job, "step": step})
        return True

    def pull_mark(self, job, step, dest, expect=None):
        man = self.pull_set("main", self.mark_prefix(job, step), dest, names=list(MARK_FILES))
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
    """A Store from a config dict or its path. "dir:PATH" repos need no token. With "store_commit_budget_per_hour"
    above 0 every commit passes a CommitBudget of that size, counted in "commit_ledger" (default, for a config
    given by its path: store_commits.json beside it, so every process on a VM shares one count)."""
    ledger = None
    if isinstance(cfg, str):
        ledger = os.path.join(os.path.dirname(os.path.abspath(cfg)), "store_commits.json")
        cfg = load_config(cfg)
    repo = cfg["repo"]
    kw = {"log": log, "mark_ns": cfg.get("mark_ns") or "", **kw}
    per_hour = int(cfg.get("store_commit_budget_per_hour") or 0)
    if per_hour > 0 and "budget" not in kw:
        kw["budget"] = CommitBudget(per_hour, cfg.get("commit_ledger") or ledger,
                                    window=float(cfg.get("commit_window_s") or 3600),
                                    essential_wait=float(cfg.get("commit_essential_wait_s", ESSENTIAL_WAIT_S)))
        kw.setdefault("budget_wait", float(cfg.get("commit_wait_s") or 0))
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
    """Every path a token can be at on this machine: the given paths, the file the config names and its
    temporaries, and, whatever the config says or fails to say, every token* file in the tmpfs folder (the
    environment's, the default, and the config's "token_dir"). The glob is what finds a token whose adoption died
    after the move and before the config named it."""
    paths = list(also)
    try:
        cfg = json.load(open(config))
    except (OSError, ValueError):
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}
    tf = cfg.get("token_file")
    if tf:
        paths += [tf, *glob.glob(tf + ".tmp*")]
    for d in (os.environ.get("PK_STORE_TOKEN_DIR") or TOKEN_DIR, cfg.get("token_dir")):
        if d:
            paths += glob.glob(os.path.join(d, "token*"))
    return list(dict.fromkeys(paths))


def _write_config(config, cfg):
    with open(config + ".tmp", "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(config + ".tmp", config)


def adopt_token(config, src, directory):
    """Move the uploaded token to `directory` (a tmpfs on the VM), mode 600, and record its path in the config.
    With no new upload, an adopted token stays as it is. The folder is recorded in the config ("token_dir") before
    the token moves, and forget_token() removes every token* file in it, so a failure at any later step (the move,
    the removal of the upload, the config's second write) leaves no copy that the clean-up cannot find."""
    cfg = json.load(open(config))
    if cfg["repo"].startswith("dir:") and not os.path.exists(src):
        return
    if os.path.exists(src):
        os.makedirs(directory, mode=0o700, exist_ok=True)
        if cfg.get("token_dir") != directory:
            cfg["token_dir"] = directory
            _write_config(config, cfg)
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
        _write_config(config, cfg)
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
        p.add_argument("--commit-budget", type=int, default=None,
                       help="commits an hour this machine may send to the store, counted with every other process "
                            "that reads this config (default: the config's \"store_commit_budget_per_hour\"; 0: no limit)")
        p.add_argument("--commit-wait", type=float, default=None,
                       help="seconds a push may wait, inside --limit, for room in that budget (default: the config's "
                            "\"commit_wait_s\", else 0: it fails at once and the caller tries again)")
        for a, kw in args:
            p.add_argument(a, **kw)

    req = {"required": True}
    cmd("push", ("--branch", req), ("--prefix", req), ("--dir", req), ("--kind", {"default": "files"}))
    cmd("pull", ("--branch", req), ("--prefix", req), ("--to", req))
    cmd("inputs", ("--jobs", req), ("--root", req), ("--record", {"default": None}))
    cmd("steps", ("--branch", req))
    cmd("prune", ("--branch", req), ("--keep", {"type": int, "default": 2}))
    cmd("release", ("--owner", {"default": "cli"}),
        ("--grace", {"type": float, "default": RELEASE_GRACE_S,
                     "help": f"seconds: what a younger ledger names is spared (never under {MIN_RELEASE_GRACE_S})"}),
        ("--orphans", {"action": "store_true",
                       "help": "also delete objects that no tree and no ledger names (uploads never committed). RUN "
                               "WITH NO VM WRITING: beside a writer this can take an upload from under its commit"}))
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
        r = store.release_report(a.owner, grace=a.grace, orphans=a.orphans)
        return json.dumps({"released": None if r["held"] else r["objects"], "bytes": r["bytes"],
                           "stored_bytes": r.get("stored_bytes"), "orphan_bytes": r.get("orphan_bytes"),
                           "orphan_objects": r.get("orphan_objects"), "pending_bytes": r.get("pending_bytes")})

    try:
        cfg = load_config(a.config)
        if a.commit_budget is not None:  # the flag sets the size; the count is the one every process beside this config shares
            cfg["store_commit_budget_per_hour"] = a.commit_budget
        if a.commit_wait is not None:
            cfg["commit_wait_s"] = a.commit_wait
        cfg.setdefault("commit_ledger", os.path.join(os.path.dirname(os.path.abspath(a.config)), "store_commits.json"))
        store = open_store(cfg)
        token = store.b.token
        out = bounded(work, a.limit, token)
        if out:
            print(out)
        if store.budget and a.cmd in ("push", "prune", "release"):
            rep = store.budget.report(0, peek=True)
            print("stat " + json.dumps({"what": "commits", **rep}, sort_keys=True), file=sys.stderr)
    except Exception as e:  # noqa: BLE001 one redacted line, never a traceback that could carry a header
        print(redact(f"hf_store {a.cmd} failed: {type(e).__name__}: {e}", token), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
