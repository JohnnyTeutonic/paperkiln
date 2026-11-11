#!/usr/bin/env python3
"""A small Jarvis Labs client for runner_multi.py --backend jarvis and jarvis_sweep.py (7 Nov 2026).

    python3 tools/colab_chat/jarvis_api.py status      # balance, VMs, filesystems, L4 VM stock

Every call goes through the `jl` CLI's JSON output (~/.local/bin/jl, 0.2.19), not its Python SDK: the CLI
has its own virtual environment, so nothing here depends on the runner's interpreter. jl keeps its own
token; this module never reads it. Only the fields used below leave this module: `jl status --json` also
carries the account's name, address and phone number, so balance() returns one number and no error
message ever quotes jl's output, only the text of its {"error": ...} reply.

Guardrails, enforced by create_vm() itself so no caller can skip them: GPU_TYPE and REGION only, never
spot (no code path passes --spot, and a VM reporting is_spot is destroyed by the transport); at most
MAX_VMS instances on the account, counted through the API before every create, under a lock so two
runners cannot both pass the count; no create below MIN_BALANCE dollars (zero balance permanently
deletes Jarvis data); a price above MAX_USD_PER_HR refused; a spend ceiling (max_usd, required, at most
SPEND_CEILING) that the next hour must fit under.

`jl create` POSTs and then polls for up to 600 s, so a failure, a timeout or a kill can leave a VM whose
id the caller never saw. Callers give every create a unique name, record it before calling, and look
the name up afterwards (find_by_name). destroy_vm() believes the listing, not jl's reply.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

import runpod_api

JL = os.path.expanduser(os.environ.get("JARVIS_JL", "~/.local/bin/jl"))
HOME = os.path.expanduser(os.environ.get("JARVIS_HOME", "~/.jarvis"))  # registry, keep list, logs; not jl's config
GPU_TYPE = "L4"            # the study GPU; no other type enters an arm (microtorch/AGENTS.md section 8)
REGION = "IN2"             # the only region with L4 VMs (7 Nov 2026)
REGION_IDS = ("IN2", "india-noida-01")  # jl's display code and the backend's name for it
VM_DISK_GB = 100           # Jarvis's minimum for a VM
SSH_USER = "ubuntu"
MAX_VMS = 1                # account-wide, every create: one validated job, then raise it deliberately
MAX_FILESYSTEMS = 2        # one per campaign
FLOOR = 8.0                # Jonathan's working floor (7 Nov 2026): the balance never goes under it
MIN_BALANCE = 10.0         # dollars; no create below it
MIN_BALANCE_RUNNING = 9.0  # dollars; a running VM is destroyed below it: $1 above FLOOR covers teardown (an L4 at $0.50/h for two hours)
MAX_USD_PER_HR = 0.50      # the listed L4 price is 0.44; a rise beyond this needs a person
SPEND_CEILING = 15.0       # the most any one runner's --max-usd may be: Jonathan's working limit of $15 spent (7 Nov 2026)
OPENING_CREDIT = 50.0      # the account's credit when Jonathan opened it (7 Nov 2026); spend since then is OPENING_CREDIT - balance
ACCOUNT_CAP = 15.0         # account-wide spend since opening, every runner and charge included (mount tests, filesystems, VMs)
FS_MIN_GB, FS_MAX_GB = 50, 200
FS_USD_PER_GB_MONTH = 0.10
JL_TIMEOUT = 90            # one read or destroy; jl's own HTTP limits are 10 s connect + 120 s read
CREATE_TIMEOUT = 720       # jl polls for 600 s after the POST
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,40}\Z")
DEAD_STATUSES = ("Destroying", "Failed", "Paused")  # never comes back on its own; the runner replaces it


class JarvisError(runpod_api.RunPodError):
    pass


class Refused(JarvisError, runpod_api.Refused):
    """A guardrail refused the action before anything was sent to Jarvis: never retried blindly."""


def _kill(process):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _jl(args, timeout=JL_TIMEOUT):
    """The parsed JSON reply of `jl ARGS --json`. JarvisError on a timeout, a non-zero exit, an
    {"error": ...} reply or output that is not JSON; the message never quotes the output itself."""
    what = " ".join(a for a in args[:2] if not a.startswith("-") and not a.isdigit())
    env = dict(os.environ, NO_COLOR="1", TERM="dumb", COLUMNS="100000")
    try:
        process = subprocess.Popen([JL, *args, "--json"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, env=env, start_new_session=True)
    except OSError as e:
        raise JarvisError(f"jl {what}: cannot run jl: {type(e).__name__}") from None
    try:
        out, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(process)
        raise JarvisError(f"jl {what}: no answer in {timeout} s") from None
    except BaseException:  # the runner's signal handler raises SystemExit: never leave jl polling behind
        _kill(process)
        raise
    try:
        data = json.loads(out)
    except ValueError:
        data = None
    if isinstance(data, dict) and data.get("error"):
        raise JarvisError(f"jl {what}: {str(data['error'])[:300]}")
    if process.returncode != 0:
        raise JarvisError(f"jl {what}: exit {process.returncode}")
    if data is None:
        raise JarvisError(f"jl {what}: output is not JSON ({len(out)} bytes)")
    return data


@contextlib.contextmanager
def _lock(name):
    """A non-blocking account-wide lock, released by the kernel even after kill -9."""
    os.makedirs(HOME, exist_ok=True)
    with open(os.path.join(HOME, name + ".lock"), "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused(f"another {name} is in progress") from None
        yield


# ---------------------------------------------------------------- account
def balance():
    """Dollars in the account. Grants are not counted: the floor protects paid-for data."""
    d = _jl(["status"])
    try:
        return float(d["balance"]["balance"])
    except (KeyError, TypeError, ValueError):
        raise JarvisError("jl status: no balance in the reply") from None


def account_spent(bal):
    """Dollars spent on the account since it opened, from the balance: every charge counts, whoever made it (Codex's review, 7 Nov 2026).
    A top-up would make this read low, so the balance floors still apply on their own."""
    return OPENING_CREDIT - bal


def checked_account_cap(cap):
    """An account-wide cap for one runner: at most ACCOUNT_CAP; a validation run takes a smaller one (cuda_validate_v1: $3)."""
    if cap is None or not 0 < cap <= ACCOUNT_CAP:
        raise Refused(f"an account-wide cap up to ${ACCOUNT_CAP:.0f} is required, got {cap}")
    return cap


def stock():
    """{"free": n, "usd_per_hr": rate} for on-demand L4 VMs in REGION, or None when none is listed."""
    rows = _jl(["gpus"])
    if not isinstance(rows, list):
        raise JarvisError("jl gpus: unexpected reply")
    for r in rows:  # L4 has one row for VMs and one for containers, with different stock
        if (r.get("gpu_type"), r.get("region"), r.get("workload_type")) == (GPU_TYPE, REGION, "vm"):
            free = r.get("effective_num_free_devices")
            return {"free": int(r.get("num_free_devices") or 0) if free is None else int(free),
                    "usd_per_hr": float(r["price_per_hour"])}
    return None


# ---------------------------------------------------------------- VMs
def list_instances():
    d = _jl(["list"])
    if not isinstance(d, list):
        raise JarvisError("jl list: unexpected reply")
    return d


def get_vm(machine_id):
    """The instance, or None when the listing no longer holds it. Raises when the listing fails:
    "not found" is only ever concluded from a listing that worked."""
    return next((v for v in list_instances() if str(v.get("machine_id")) == str(machine_id)), None)


def find_by_name(name, instances=None):
    return [v for v in (list_instances() if instances is None else instances) if v.get("name") == name]


def ssh_endpoint(vm):
    """(ip, port) of a Running VM from its ssh_command, or None. The command is parsed, never run: it
    turns host-key checking off."""
    if vm.get("status") != "Running" or not vm.get("ssh_command"):
        return None
    try:
        parts = shlex.split(vm["ssh_command"])
    except ValueError:
        return None
    host, user, port, i = None, None, 22, 1
    while i < len(parts):
        a = parts[i]
        if a in ("-p", "-l", "-o", "-i"):
            value = parts[i + 1] if i + 1 < len(parts) else None
            if a == "-p" and value and value.isdigit():
                port = int(value)
            elif a == "-l":
                user = value
            i += 2
            continue
        if not a.startswith("-"):
            host = a
        i += 1
    if host and "@" in host:
        user, host = host.rsplit("@", 1)
    if user != SSH_USER or not host or not re.fullmatch(r"[A-Za-z0-9.:-]+", host):
        return None
    return host, port


def vm_faults(vm):
    """Why this VM must not run a job: anything other than one on-demand L4 VM in REGION."""
    faults = []
    if vm.get("gpu_type") != GPU_TYPE:
        faults.append(f"GPU {vm.get('gpu_type')}")
    if vm.get("num_gpus") not in (1, None):
        faults.append(f"{vm.get('num_gpus')} GPUs")
    if vm.get("template") != "vm":
        faults.append(f"template {vm.get('template')}")
    if vm.get("is_spot"):
        faults.append("spot")
    if vm.get("region") not in REGION_IDS:
        faults.append(f"region {vm.get('region')}")
    return faults


def create_vm(name, fs_id, spent, max_usd, gpu=GPU_TYPE, timeout=CREATE_TIMEOUT, account_cap=ACCOUNT_CAP):
    """`jl create` for one on-demand L4 VM behind the guardrails; returns the instance. Refused means
    nothing was sent to Jarvis; JarvisError means a VM may exist, so look its name up."""
    account_cap = checked_account_cap(account_cap)
    if gpu != GPU_TYPE:
        raise Refused(f"only {GPU_TYPE} runs on Jarvis, not {gpu}")
    if not NAME_RE.match(name or ""):
        raise Refused(f"VM name {name!r} needs 1 to 40 letters, digits, hyphens or underscores")
    if fs_id is None:
        raise Refused("no VM without a filesystem: its checkpoints would die with it")
    if max_usd is None or not 0 < max_usd <= SPEND_CEILING:
        raise Refused(f"a spend ceiling up to ${SPEND_CEILING:.0f} is required, got {max_usd}")
    with _lock("create"):
        n = len(list_instances())
        if n >= MAX_VMS:
            raise Refused(f"{n} Jarvis instance(s) exist (cap {MAX_VMS}); refusing to create another")
        bal = balance()
        if bal < MIN_BALANCE:
            raise Refused(f"Jarvis balance ${bal:.2f} is below ${MIN_BALANCE:.0f}; refusing to create a VM")
        s = stock()
        if not s or s["free"] < 1:
            raise Refused(f"no {GPU_TYPE} VM free in {REGION}")
        if s["usd_per_hr"] > MAX_USD_PER_HR:
            raise Refused(f"{GPU_TYPE} lists at ${s['usd_per_hr']:.2f}/h, above ${MAX_USD_PER_HR:.2f}/h")
        if spent + s["usd_per_hr"] > max_usd:
            raise Refused(f"${spent:.2f} spent plus an hour at ${s['usd_per_hr']:.2f} passes the ${max_usd:.2f} cap")
        if account_spent(bal) + s["usd_per_hr"] > account_cap:
            raise Refused(f"${account_spent(bal):.2f} spent on the account plus an hour at ${s['usd_per_hr']:.2f} passes the account cap ${account_cap:.2f}")
        if bal - s["usd_per_hr"] < MIN_BALANCE_RUNNING:
            raise Refused(f"an hour at ${s['usd_per_hr']:.2f} would take the balance ${bal:.2f} under ${MIN_BALANCE_RUNNING:.0f}")
        vm = _jl(["create", "--gpu", GPU_TYPE, "--vm", "--region", REGION, "--storage", str(VM_DISK_GB),
                  "--fs-id", str(fs_id), "--name", name, "--yes"], timeout)
    if not isinstance(vm, dict) or not vm.get("machine_id"):
        raise JarvisError("jl create: no machine id in the reply")
    return dict(vm, usd_per_hr=s["usd_per_hr"])


def destroy_vm(machine_id):
    """True once the listing shows the VM gone or being destroyed; False when it still stands. jl's
    reply alone proves nothing. Raises when the listing fails."""
    try:
        _jl(["destroy", str(machine_id), "--yes"])
    except JarvisError:
        pass  # already gone, or a real failure: the listing below decides which
    vm = get_vm(machine_id)
    return vm is None or vm.get("status") == "Destroying"


# ---------------------------------------------------------------- filesystems
def list_filesystems():
    d = _jl(["filesystem", "list"])
    if not isinstance(d, list):
        raise JarvisError("jl filesystem list: unexpected reply")
    return d


def filesystem_create(name, gb):
    """A filesystem in REGION; returns its id. Same floor as a VM: a filesystem bills until removed."""
    if not 1 <= len(name) <= 30:
        raise Refused(f"filesystem name {name!r} needs 1 to 30 characters")
    if not FS_MIN_GB <= gb <= FS_MAX_GB:
        raise Refused(f"filesystem size {gb} GB is outside {FS_MIN_GB} to {FS_MAX_GB} GB")
    with _lock("filesystem"):
        n = len(list_filesystems())
        if n >= MAX_FILESYSTEMS:
            raise Refused(f"{n} Jarvis filesystems exist (cap {MAX_FILESYSTEMS}); refusing to create another")
        bal = balance()
        if bal < MIN_BALANCE:
            raise Refused(f"Jarvis balance ${bal:.2f} is below ${MIN_BALANCE:.0f}; refusing a filesystem")
        d = _jl(["filesystem", "create", "--name", name, "--storage", str(gb), "--region", REGION, "--yes"])
    if not isinstance(d, dict) or d.get("fs_id") is None:
        raise JarvisError("jl filesystem create: no id in the reply")
    return int(d["fs_id"])


def filesystem_remove(fs_id):
    """True once the listing no longer holds the filesystem."""
    try:
        _jl(["filesystem", "remove", str(fs_id), "--yes"])
    except JarvisError:
        pass
    return not any(str(f.get("fs_id")) == str(fs_id) for f in list_filesystems())


def ledger_usd(led, now=None):
    """Estimated dollars of a runner's ledger: VMs at their recorded $/h from start to stop (or now),
    filesystems at FS_USD_PER_GB_MONTH from creation to deletion (or now). Same shape as RunPod's."""
    now = now or time.time()
    vms = sum(p["usd_per_hr"] * max(0.0, (p.get("stop") or now) - p["start"]) / 3600 for p in led.get("pods", []))
    fss = sum(v["gb"] * FS_USD_PER_GB_MONTH * max(0.0, (v.get("deleted") or now) - v["created"]) / (30 * 86400)
              for v in led.get("volumes", []))
    return vms + fss


def status():
    keep = ("machine_id", "name", "status", "gpu_type", "region", "is_spot", "fs_id", "public_ip")
    return {"balance": balance(), "instances": [{k: v.get(k) for k in keep} for v in list_instances()],
            "filesystems": [{k: f.get(k) for k in ("fs_id", "fs_name", "storage", "region")}
                            for f in list_filesystems()],
            "L4_vm_stock": stock()}


if __name__ == "__main__":
    if sys.argv[1:] == ["status"]:
        print(json.dumps(status(), indent=1))
    else:
        sys.exit(__doc__)
