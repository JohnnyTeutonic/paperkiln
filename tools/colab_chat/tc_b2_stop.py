#!/usr/bin/env python3
"""Ends one arm of the race (stronger_base_spec.md, section 11; PROTOCOL section 30). Claude Opus 5.5, 7 Nov 2026. Run by hand, in WSL.

    python3 tools/colab_chat/tc_b2_stop.py --state-dir DIR --reason "..." [--release-volume]

Writes DIR/HALTED (the watchdog's stop_file, which then disables the entry), SIGTERMs the runner by its pid file, and waits for the
pod to be gone. The runner's own handler ends the pod and keeps the volume. --release-volume then deletes that arm's volume, by the
id its state.json holds and no other, and closes the ledger entry: the losing arm's last step. The winner's state directory is
never touched. Refused unless the state directory names a run of the race and its volume exists under its own name.
"""
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_api as api  # noqa: E402

RACE = ("pk-tc-b1-r1", "pk-tc-b2-l12")


def owned_volume(state):
    r = state.get("runpod", {})
    v = r.get("volume")
    return v if v and any(x.get("id") == v["id"] for x in r.get("ledger", {}).get("volumes", [])) else None


def stop(home, reason, release, wait=180, sleep=time.sleep):
    state = json.load(open(os.path.join(home, "state.json")))
    vol = owned_volume(state)
    pods = state.get("runpod", {}).get("ledger", {}).get("pods", [])
    names = {p.get("name") for p in pods}
    if not names or not names <= set(RACE):
        raise SystemExit(f"{home} is not a run of the race: {sorted(names)}")
    marker = os.path.join(home, "HALTED")
    if not os.path.exists(marker):
        with open(marker, "x") as f:
            json.dump({"kind": "RACE_ARM_STOPPED", "reason": reason, "at": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
    pidf = os.path.join(home, "runner.pid")
    if os.path.exists(pidf):
        try:
            os.kill(int(open(pidf).read()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    t0 = time.time()
    while time.time() - t0 < wait:
        state = json.load(open(os.path.join(home, "state.json")))
        if not state.get("runpod", {}).get("pod"):
            break
        sleep(5)
    else:
        raise SystemExit("the pod is still recorded; the sweep ends it once the runner is dead. Volume kept. Run this again")
    if release:
        if os.path.exists(pidf):
            raise SystemExit("the runner has not exited; volume kept")
        if not vol:
            print("no volume recorded")
            return state
        if not any(v.get("id") == vol["id"] for v in api.list_volumes()):
            print(f"volume {vol['id']} is already gone")
        elif not api.delete_volume(vol["id"]):
            raise SystemExit(f"delete of {vol['id']} failed")
        now = time.time()
        for e in state["runpod"]["ledger"]["volumes"]:
            if e["id"] == vol["id"] and not e.get("deleted"):
                e["deleted"] = now
        state["runpod"]["volume"] = None
        tmp = os.path.join(home, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, os.path.join(home, "state.json"))
        print(f"volume {vol['id']} deleted; ledger closed at ${api.ledger_usd(state['runpod']['ledger']):.2f}")
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--release-volume", action="store_true")
    a = ap.parse_args(argv)
    stop(os.path.expanduser(a.state_dir), a.reason, a.release_volume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
