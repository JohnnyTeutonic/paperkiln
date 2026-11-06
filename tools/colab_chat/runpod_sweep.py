#!/usr/bin/env python3
"""No idle RunPod spend (24 Oct 2026): terminate every pod no live runner owns, delete the volumes of
campaigns that are home. Run by the Windows watchdog (scripts/colab_watchdog.ps1) every tick, from WSL:

    python3 tools/colab_chat/runpod_sweep.py [--dry-run] [--log ~/.runpod/sweep.log]

Runners (runner_multi.py --backend runpod) register their state dir in ~/.runpod/runners.json. A
runner is alive when its runner.pid names a live process whose command line is runner_multi.py with
that --state-dir. The one explicitly registered source-pinned data controller is also
recognised after its full manifest and anchored root admission validate. Exact root-admitted
Sfull runpy guards are additionally recognized with their source/admission hashes and live PID start identity. For each pod on the account:
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



# These launchers execute runner_multi in-process via runpy; /proc keeps the guard argv.
# Adding a recovery owner requires a separate reviewed source AND admission hash.
ADMITTED_ROOT_GUARDS = {
    "/home/jonat/sfull_v2_seed1_root_admission_20261101/guarded_seed1.py": {
        "kind": "seed1",
        "sha256": "d4fe41271c7aaf4325dd019fdc2886098d54490f0f61e28f78bdc8a13132e69a",
        "admission": "/home/jonat/sfull_v2_seed1_root_admission_20261101/admission.json",
        "admission_sha256": "579cc75c2be8e1821e043b4c7dda3286a67020d533a51192a59f7e2af9a4851b",
        "state_dir": "/home/jonat/sfull_v2_states_20261101/pk_c12_sfull_r1_runner",
    },
    "/home/jonat/sfull_boot_recovery_root_20261101/guarded_recovery.py": {
        "kind": "recovery",
        "sha256": "2f75c9abf767dd3dc1274dc05e2956b06e20147d23e96429a354ad6a30f83cf8",
        "admission": "/home/jonat/sfull_boot_recovery_root_20261101/recovery_admission.json",
        "admission_sha256": "2a5c1638a718dc352e0f2361c6bab95925c7cf0c2da36958eca386aeaacb7ec4",
        "state_dir": "/home/jonat/sfull_v2_states_20261101/pk_c12_sfull_r1_runner",
        "original_guard": "/home/jonat/sfull_v2_seed1_root_admission_20261101/guarded_seed1.py",
    },
}


def process_snapshot(pid):
    """Only process identity metadata; never read process environment or credentials."""
    from pathlib import Path
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid owner PID")
    root = Path('/proc') / str(pid)
    raw = (root / 'cmdline').read_bytes()
    fields_raw = raw.split(b'\0')
    if fields_raw and fields_raw[-1] == b'':
        fields_raw.pop()
    args = [item.decode('utf-8') for item in fields_raw]
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ValueError('zombie owner')
    ticks = int(fields[19])
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    return {'args': args, 'start_ticks': ticks,
            'started_epoch': boot + ticks / os.sysconf('SC_CLK_TCK'),
            'uid': root.stat().st_uid, 'exe': os.path.realpath(root / 'exe')}


def load_guard(path, started_epoch, source):
    """Validate admission at this same process's start, preserving cleanup after expiry."""
    import datetime
    import types
    module = types.ModuleType('sweep_pinned_root_guard')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    class AtProcessStart(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime.fromtimestamp(started_epoch, tz)
    module.datetime = types.SimpleNamespace(datetime=AtProcessStart, timezone=datetime.timezone)
    return module


def validate_root_guard(args, state_dir, started_epoch):
    """No cached approval: rehash every pinned source/admission and rerun full validators."""
    import hashlib
    import types
    from pathlib import Path
    if len(args) != 3 or os.path.basename(args[0]) not in {'python3', 'python3.11'}:
        return False
    known = ADMITTED_ROOT_GUARDS.get(args[1])
    if known is None or args[2] != known['admission']:
        return False
    if os.path.realpath(state_dir) != os.path.realpath(known['state_dir']):
        return False
    script, admission = Path(args[1]), Path(args[2])
    if script.is_symlink() or admission.is_symlink():
        return False
    source = script.read_bytes()
    admission_raw = admission.read_bytes()
    if (hashlib.sha256(source).hexdigest() != known['sha256']
            or hashlib.sha256(admission_raw).hexdigest() != known['admission_sha256']):
        return False
    record = json.loads(admission_raw)
    if known['kind'] == 'seed1':
        module = load_guard(script, started_epoch, source)
    elif known['kind'] == 'recovery':
        old = ADMITTED_ROOT_GUARDS.get(known['original_guard'])
        if (old is None or old['kind'] != 'seed1'
                or record.get('original_guard') != known['original_guard']
                or record.get('original_admission') != old['admission']
                or record.get('source_pins', {}).get(known['original_guard']) != old['sha256']
                or record['source_pins'].get(old['admission']) != old['admission_sha256']):
            return False
        # Use the reviewed recovery validator unchanged, with a private original module
        # whose deadline clock is the owning PID's start. Never patch global importlib.
        old_source = Path(known['original_guard']).read_bytes()
        if (hashlib.sha256(old_source).hexdigest() != old['sha256']
                or hashlib.sha256(Path(old['admission']).read_bytes()).hexdigest() != old['admission_sha256']):
            return False
        original = load_guard(known['original_guard'], started_epoch, old_source)
        module = types.ModuleType('sweep_pinned_recovery_guard')
        module.__file__ = str(script)
        exec(compile(source, str(script), 'exec'), module.__dict__)
        token = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        def pinned_spec(name, path):
            if name != 'original_sfull_root_guard' or str(path) != known['original_guard']:
                raise ValueError('unexpected recovery import')
            return token
        def pinned_module(value):
            if value is not token:
                raise ValueError('unexpected recovery module')
            return original
        module.importlib = types.SimpleNamespace(util=types.SimpleNamespace(
            spec_from_file_location=pinned_spec, module_from_spec=pinned_module))
    else:
        return False
    argv = module.validate(record)
    if argv.count('--state-dir') != 1 or argv.count('--deadline-utc') != 1:
        return False
    return (os.path.realpath(argv[argv.index('--state-dir') + 1]) == os.path.realpath(state_dir)
            and argv[argv.index('--deadline-utc') + 1] == record['deadline_utc'])


def admitted_root_guard_pid_alive(pid, state_dir):
    """Live exact PID + start identity; source approval cannot survive PID reuse or drift."""
    try:
        before = process_snapshot(pid)
        trusted_python = {os.path.realpath(sys.executable)}
        for path in ['/usr/bin/python3', '/usr/bin/python3.11']:
            if os.path.isfile(path):
                trusted_python.add(os.path.realpath(path))
        if (before['uid'] != os.getuid() or before['exe'] not in trusted_python
                or not 0 < before['started_epoch'] <= time.time() + 1):
            return False
        if not validate_root_guard(before['args'], state_dir, before['started_epoch']):
            return False
        return before == process_snapshot(pid)
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError,
            SyntaxError, IndexError, StopIteration):
        return False


def runner_alive(state_dir):
    """The runner of state_dir is running: runner.pid names a live runner_multi.py with this --state-dir."""
    try:
        pid = int(open(os.path.join(state_dir, "runner.pid")).read().strip())
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except (OSError, ValueError):
        return False
    args = [a.decode(errors="replace") for a in (argv[:-1] if argv and argv[-1] == b"" else argv)]
    if len(args) > 1 and args[1] in ADMITTED_ROOT_GUARDS:
        if not admitted_root_guard_pid_alive(pid, state_dir):
            return False
        try:
            return int(open(os.path.join(state_dir, "runner.pid")).read().strip()) == pid
        except (OSError, ValueError):
            return False
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
