#!/usr/bin/env python3
"""No idle RunPod spend (24 Oct 2026): terminate every pod no live runner owns, delete the volumes of
campaigns that are home. Run by the Windows watchdog (scripts/colab_watchdog.ps1) every tick, from WSL:

    python3 tools/colab_chat/runpod_sweep.py [--dry-run] [--log ~/.runpod/sweep.log]

Runners (runner_multi.py --backend runpod) register their state dir in ~/.runpod/runners.json. A
runner is alive when its runner.pid names a live process whose command line is runner_multi.py with
that --state-dir. The one explicitly registered source-pinned data controller is also
recognised after its full manifest and anchored root admission validate. For each pod on the account:
  - kept when a live runner's state records it as its pod, or a live runner is creating a pod of
    that name right now (state "creating" within the last 30 minutes);
  - otherwise terminated (its runner died, e.g. kill -9, or no runner state knows it), and the stop
    written into the dead runner's ledger.
Pod ids or names in ~/.runpod/keep.json (a JSON list) are never touched: pods made by hand.
For each network volume: deleted when its runner is not alive and its campaign is home (state
"campaign_home" plus every recorded job explicitly home, with no active pod); otherwise left (a volume
without a runner state is reported, never deleted). Exit 0 unless the API failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_api as api  # noqa: E402

REGISTRY = os.path.expanduser("~/.runpod/runners.json")
KEEP = os.path.expanduser("~/.runpod/keep.json")
CREATING_S = 1800
LOG = None


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def jload(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    tmp = f"{path}.sweep{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# Only this source-pinned, root-admitted in-process runner is additionally known.
# Ownership remains valid during deadline-triggered stop/relay, not just while training.
ADMITTED_CONTROLLERS = {
    "/home/jonat/pk_sfull_data_controller_20261031/tools/colab_chat/sfull_data_only_controller_20261031.py": {
        "sha256": "8c582ebba3f752567fff98dd9587f75fd37a055d7bd9cab75ca725c7dea3dbe7",
        "manifest": "/home/jonat/pk_sfull_data_controller_20261031/docs/receipts/sfull_data_only_admission_20261031/manifest.json",
        "manifest_sha256": "3640b1595afb5847e66081b7a6fc10766caed7c4fa5b9cb61b2e3957fb1b4ea9",
    }
}


def admitted_controller_alive(args, state_dir):
    """Recognise exact reviewed code, command, state and anchored root admission."""
    import hashlib
    import importlib.util
    import subprocess
    from pathlib import Path

    try:
        if len(args) != 8 or os.path.basename(args[0]) not in {"python3", "python3.11"}:
            return False
        known = ADMITTED_CONTROLLERS.get(args[1])
        if known is None:
            return False
        script = Path(args[1])
        manifest = Path(known["manifest"])
        admission = Path(state_dir) / "admission.json"
        expected = [args[0], str(script), "--manifest", str(manifest),
                    "--manifest-sha256", known["manifest_sha256"], "--admission", str(admission)]
        if args != expected or hashlib.sha256(script.read_bytes()).hexdigest() != known["sha256"]:
            return False
        anchor = json.loads((Path(state_dir) / "admission_anchor.json").read_text())
        if anchor != {"manifest_sha256": known["manifest_sha256"],
                      "admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest()}:
            return False
        spec = importlib.util.spec_from_file_location("sweep_admitted_controller", script)
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        root = script.parents[2]
        m = controller.verify_manifest(manifest, known["manifest_sha256"], root)
        if os.path.realpath(m["state_dir"]) != os.path.realpath(state_dir):
            return False
        # Revalidate the anchored launch, not wall-clock freshness. An expired launch
        # still owns its pod while the live controller stops and relays it.
        a = json.loads(admission.read_text())
        controller.validate_admission(m, known["manifest_sha256"], admission,
                                      controller.timestamp(a["admitted_at_utc"]), True)
        return True
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError, subprocess.SubprocessError):
        return False


def runner_alive(state_dir):
    """The runner of state_dir is running: runner.pid names a live runner_multi.py with this --state-dir."""
    try:
        pid = int(open(os.path.join(state_dir, "runner.pid")).read().strip())
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except (OSError, ValueError):
        return False
    args = [a.decode(errors="replace") for a in (argv[:-1] if argv and argv[-1] == b"" else argv)]
    if not any(a.endswith("runner_multi.py") for a in args):
        return admitted_controller_alive(args, state_dir)
    for i, a in enumerate(args):
        v = args[i + 1] if a == "--state-dir" and i + 1 < len(args) else (a.split("=", 1)[1]
                                                                          if a.startswith("--state-dir=") else None)
        if v and os.path.realpath(os.path.expanduser(v)) == os.path.realpath(state_dir):
            return True
    return False


def campaign_home(state_dir, st):
    """Deletion proof comes from complete current state, never a log line or retirement."""
    jobs = st.get("jobs")
    R = st.get("runpod") or {}
    return (st.get("campaign_home") is True and isinstance(jobs, dict) and bool(jobs)
            and all(isinstance(job, dict) and job.get("home") is True for job in jobs.values())
            and not R.get("pod") and not st.get("endpoint")
            and not R.get("creating"))


def main(argv=None):
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--state-dir", action="append", default=[], help="also consider this runner state dir")
    a = ap.parse_args(argv)
    LOG = os.path.expanduser(a.log) if a.log else None
    if not os.path.exists(api.KEY_PATH):
        return 0  # no RunPod account on this machine
    reg = jload(REGISTRY, {}) or {}
    dirs = sorted(set(list(reg) + [os.path.expanduser(d) for d in a.state_dir]))
    runners = []
    for d in dirs:
        st = None
        for _ in range(3):  # a live runner rewrites state.json in place: retry a torn read
            st = jload(os.path.join(d, "state.json"))
            if st is not None or not os.path.exists(os.path.join(d, "state.json")):
                break
            time.sleep(1)
        runners.append({"dir": d, "st": st, "alive": runner_alive(d), "R": (st or {}).get("runpod") or {},
                        "name": (reg.get(d) or {}).get("name")})
    keep = set(jload(KEEP, []) or [])
    try:
        pods, vols = api.list_pods(), api.list_volumes()
    except api.RunPodError as e:
        log(f"sweep: API failed: {e}")
        return 1
    now = time.time()
    acted = False
    for p in pods:
        pid, name = p.get("id"), p.get("name")
        if pid in keep or name in keep:
            continue
        owner = next((r for r in runners if (r["R"].get("pod") or {}).get("id") == pid), None)
        if owner and owner["alive"]:
            continue
        # A live runner of this name whose state could not be read (mid-write), or that is creating
        # this pod and has not recorded its id yet: never touched.
        if any(r["alive"] and r["name"] == name and (r["st"] is None or (r["R"].get("creating") and
                                                                         now - r["R"]["creating"] < CREATING_S))
               for r in runners):
            continue
        why = (f"its runner ({owner['dir']}) is not alive" if owner else "no runner state records it")
        log(f"pod {pid} ({name}, {p.get('desiredStatus')}, ${p.get('costPerHr')}/h): {why}; "
            f"{'would terminate (dry run)' if a.dry_run else 'terminating'}")
        acted = True
        if a.dry_run:
            continue
        try:
            ok = api.terminate_pod(pid)
        except api.RunPodError as e:
            log(f"pod {pid}: terminate failed: {e}")
            continue
        st = jload(os.path.join(owner["dir"], "state.json")) if ok and owner else None  # re-read
        if st is not None:
            R = st.get("runpod") or {}
            for e in R.get("ledger", {}).get("pods", []):
                if e["id"] == pid and not e.get("stop"):
                    e["stop"], e["stopped_by"] = time.time(), "sweep"
            if (R.get("pod") or {}).get("id") == pid:
                R["pod"] = None
                st["endpoint"] = None
            save_json(os.path.join(owner["dir"], "state.json"), st)
        log(f"pod {pid}: {'terminated' if ok else 'NOT terminated'}")
    for v in vols:
        vid = v.get("id")
        if vid in keep or v.get("name") in keep:
            continue
        owner = next((r for r in runners if (r["R"].get("volume") or {}).get("id") == vid
                      or any(x.get("id") == vid for x in r["R"].get("retired_volumes", []))), None)
        if not owner:
            log(f"volume {vid} ({v.get('name')}, {v.get('size')} GB, {v.get('dataCenterId')}): no runner state "
                "records it; left alone")
            continue
        owners = [r for r in runners if (r["R"].get("volume") or {}).get("id") == vid
                  or any(x.get("id") == vid for x in r["R"].get("retired_volumes", []))]
        # Re-read immediately before deletion: the initial snapshot may predate a rescue/restart.
        # Shared/legacy/missing state and retirement alone never authorize deleting checkpoints.
        if any(runner_alive(r["dir"]) or not campaign_home(
                r["dir"], jload(os.path.join(r["dir"], "state.json"), {}) or {}) for r in owners):
            continue
        if any(p.get("networkVolumeId") == vid or
               (p.get("networkVolume") or {}).get("id") == vid for p in pods):
            continue
        log(f"volume {vid} ({v.get('name')}): campaign {owner['dir']} is verified home; "
            f"{'would delete (dry run)' if a.dry_run else 'deleting'}")
        acted = True
        if a.dry_run:
            continue
        try:
            ok = api.delete_volume(vid)
        except api.RunPodError as e:
            log(f"volume {vid}: delete failed: {e}")
            continue
        st = jload(os.path.join(owner["dir"], "state.json")) if ok else None
        if st is not None:
            R = st.get("runpod") or {}
            for e in R.get("ledger", {}).get("volumes", []):
                if e["id"] == vid and not e.get("deleted"):
                    e["deleted"] = time.time()
            if (R.get("volume") or {}).get("id") == vid:
                R["volume"] = None
            R["retired_volumes"] = [x for x in R.get("retired_volumes", []) if x.get("id") != vid]
            save_json(os.path.join(owner["dir"], "state.json"), st)
        log(f"volume {vid}: {'deleted' if ok else 'NOT deleted'}")
    if not acted and LOG is None:
        print(f"sweep: {len(pods)} pod(s), {len(vols)} volume(s), {len(runners)} runner(s); nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
