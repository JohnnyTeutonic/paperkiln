#!/usr/bin/env python3
"""No idle Jarvis spend (7 Nov 2026): destroy every instance no live runner owns. Run by the Windows
watchdog every tick, from WSL:

    python3 tools/colab_chat/jarvis_sweep.py [--dry-run] [--log ~/.jarvis/sweep.log]

Runners (runner_multi.py --backend jarvis) register their state dir in ~/.jarvis/runners.json before any
create. An instance is kept when a live runner's state records it, or when a live runner is creating that
exact name right now ("creating" within the last 30 minutes), or a live runner of that name's state is
mid-write. Anything else is destroyed, paused instances included: a paused VM's disk still bills. The stop
goes into the dead runner's ledger, or a new ledger entry when the runner died between creating the VM and
recording it, or ~/.jarvis/orphans.jsonl when no runner knows the VM. Ids or names in ~/.jarvis/keep.json
(a JSON list) are never touched: VMs made by hand.

The listing is read before the runner states. A runner writes "creating" before it calls jl, so any VM in
the listing is already in the state read after it; the other order would destroy a VM created in between.

An unreadable registry stops the sweep: it could otherwise condemn every live runner's VM. Filesystems
are only reported: they hold checkpoints, and release() removes them with its own proof.
Exit 0 unless the API failed, a destroy failed or the registry is unreadable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jarvis_api as api  # noqa: E402
import runpod_sweep  # noqa: E402  (its proof that a runner is alive; its own globals stay unused)

REGISTRY = os.path.join(api.HOME, "runners.json")
KEEP = os.path.join(api.HOME, "keep.json")
ORPHANS = os.path.join(api.HOME, "orphans.jsonl")
CREATING_S = 1800
LOG = None


def read_keep():
    """The ids and names never to touch: an empty set when keep.json is absent; None when it exists but is not a JSON list of
    strings and integers."""
    if not os.path.exists(KEEP):
        return set()
    try:
        with open(KEEP) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, list) or not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in d):
        return None
    return {str(x) for x in d}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def read_registry():
    """The registry (empty when it does not exist yet); None when it exists and cannot be read."""
    if not os.path.exists(REGISTRY):
        return {}
    reg = runpod_sweep.jload(REGISTRY)
    return reg if isinstance(reg, dict) else None


def read_runners(reg, extra):
    runners = []
    for d in sorted(set(list(reg) + [os.path.expanduser(x) for x in extra])):
        st = None
        for _ in range(3):  # a live runner rewrites state.json in place: retry a torn read
            st = runpod_sweep.jload(os.path.join(d, "state.json"))
            if st is not None or not os.path.exists(os.path.join(d, "state.json")):
                break
            time.sleep(1)
        runners.append({"dir": d, "st": st, "alive": runpod_sweep.runner_alive(d), "R": (st or {}).get("jarvis") or {},
                        "name": (reg.get(d) or {}).get("name")})
    return runners


def spared(vm, runners, now):
    """A live runner has a claim on this VM: it records it, is creating it, or its state cannot be read."""
    vm_id, name = str(vm.get("machine_id")), vm.get("name") or ""
    for r in runners:
        if not r["alive"]:
            continue
        if str((r["R"].get("pod") or {}).get("id")) == vm_id:
            return True
        c = r["R"].get("creating") or {}
        if c.get("name") == name and now - c.get("started", 0) < CREATING_S:
            return True
        if r["st"] is None and name.rsplit("-", 1)[0] == r["name"]:
            return True
    return False


def record(vm, runners, now):
    """Write the destroyed VM down where it belongs. A dead runner's state is safe to write."""
    vm_id, name = str(vm.get("machine_id")), vm.get("name")
    for r in runners:
        c = r["R"].get("creating") or {}
        pod = r["R"].get("pod") or {}
        if str(pod.get("id")) != vm_id and c.get("name") != name:
            continue
        st = runpod_sweep.jload(os.path.join(r["dir"], "state.json"))  # re-read
        if st is None:
            continue
        R = st.get("jarvis") or {}
        closed = False
        for e in R.get("ledger", {}).get("pods", []):
            if e["id"] == vm_id and not e.get("stop"):
                e["stop"], e["stopped_by"], closed = now, "sweep", True
        if not closed and c.get("name") == name:
            R.setdefault("ledger", {}).setdefault("pods", []).append(
                {"id": vm_id, "name": name, "gpu": vm.get("gpu_type"), "usd_per_hr": api.MAX_USD_PER_HR,
                 "dc": api.REGION, "volume": vm.get("fs_id"), "start": c.get("started", now), "stop": now,
                 "stopped_by": "sweep", "image": "jarvis-vm-template", "image_digest": None})
        if str((R.get("pod") or {}).get("id")) == vm_id:
            R["pod"] = None
            st["endpoint"] = None
        if (R.get("creating") or {}).get("name") == name:
            R["creating"] = None
        runpod_sweep.save_json(os.path.join(r["dir"], "state.json"), st)
        return r["dir"]
    with open(ORPHANS, "a") as f:
        f.write(json.dumps({"at": now, "id": vm_id, "name": name, "status": vm.get("status"),
                            "gpu": vm.get("gpu_type"), "fs_id": vm.get("fs_id")}) + "\n")
    return ORPHANS


def report_filesystems(runners):
    for f in api.list_filesystems():
        fs_id = str(f.get("fs_id"))
        owners = [r for r in runners if str((r["R"].get("volume") or {}).get("id")) == fs_id]
        if not owners:
            log(f"filesystem {fs_id} ({f.get('fs_name')}, {f.get('storage')} GB): no runner state records it; left alone")
        elif all(r["st"] and r["st"].get("campaign_home") and not r["alive"] for r in owners):
            log(f"filesystem {fs_id} ({f.get('fs_name')}): campaign is home and it is still billed; left alone")


def main(argv=None):
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--state-dir", action="append", default=[], help="also consider this runner state dir")
    a = ap.parse_args(argv)
    LOG = os.path.expanduser(a.log) if a.log else None
    if LOG:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if not os.path.exists(api.JL):
        return 0  # no Jarvis account on this machine
    os.makedirs(api.HOME, exist_ok=True)
    try:
        vms = api.list_instances()  # first: see the module docstring
    except api.JarvisError as e:
        log(f"sweep: API failed: {e}")
        return 1
    reg = read_registry()
    if reg is None:
        log(f"sweep: {REGISTRY} is unreadable; destroying nothing")
        return 1
    keep = read_keep()
    if keep is None:   # as the registry: a keep list that cannot be read protects nothing, so nothing is destroyed (launch reviews, 11 Nov 2026)
        log(f"sweep: {KEEP} is unreadable or not a JSON list of ids and names; destroying nothing")
        return 1
    runners = read_runners(reg, a.state_dir)
    failed = False
    for vm in vms:
        vm_id, name = str(vm.get("machine_id")), vm.get("name")
        if vm_id in keep or name in keep:
            continue
        now = time.time()
        if spared(vm, runners, now):
            continue
        owner = next((r for r in runners if str((r["R"].get("pod") or {}).get("id")) == vm_id), None)
        why = f"its runner ({owner['dir']}) is not alive" if owner else "no live runner records it"
        log(f"VM {vm_id} ({name}, {vm.get('status')}, {vm.get('gpu_type')}): {why}; "
            f"{'would destroy (dry run)' if a.dry_run else 'destroying'}")
        if a.dry_run:
            continue
        try:
            ok = api.destroy_vm(vm_id)
        except api.JarvisError as e:
            log(f"VM {vm_id}: destroy failed: {e}")
            failed = True
            continue
        if ok:
            log(f"VM {vm_id}: destroyed; recorded in {record(vm, runners, time.time())}")
        else:
            log(f"VM {vm_id}: NOT destroyed")
            failed = True
    try:
        report_filesystems(runners)
    except api.JarvisError as e:
        log(f"sweep: filesystem listing failed: {e}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
