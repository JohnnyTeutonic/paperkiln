#!/usr/bin/env python3
"""By-hand steps of the race (stronger_base_spec.md, section 11; PROTOCOL.md section 30). Claude Opus 5.5, 7 Nov 2026. Run in WSL.

    python3 tools/colab_chat/tc_b2_stop.py stop --arm NAME --state-dir DIR --reason "..."
    python3 tools/colab_chat/tc_b2_stop.py release-volume --arm NAME --state-dir DIR --reason "the lead's word, dated"
    python3 tools/colab_chat/tc_b2_stop.py release-rung --arm NAME --state-dir DIR --rung RUNG_JOB [--ahead] --reason "the verdict or Jonathan's line"
    python3 tools/colab_chat/tc_b2_stop.py reset-failed --arm NAME --state-dir DIR --reason "Codex's go, dated"

stop writes DIR/HALTED (the watchdog's stop_file, which then disables the entry), SIGTERMs a live runner and waits for its pod to go.
Volume and checkpoints stay. release-volume deletes that arm's volume, by the id its state holds and only if the provider's name for
it is NAME-vol: the loser's last step, and only on the lead's word. release-rung lets a paused arm go past a cut-off: it keeps the
pause marker as HALTED.<rung> and writes DIR/RELEASED_<rung>; --ahead releases a later rung before the arm reaches it, for a winner with no
read left. The lead then sets a fresh deadline and enables the entry.
"""
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_api as api  # noqa: E402
import runpod_sweep as sweep  # noqa: E402

RACE = ("pk-tc-b1-r1", "pk-tc-b2-l12")


def read(home):
    return json.load(open(os.path.join(home, "state.json")))


def check_arm(state, arm):
    if arm not in RACE:
        raise SystemExit(f"{arm} is not an arm of the race")
    led = state.get("runpod", {}).get("ledger", {})
    names = {p.get("name") for p in led.get("pods", [])} | {
        (v.get("name") or "")[:-len("-vol")] for v in led.get("volumes", []) if (v.get("name") or "").endswith("-vol")}
    if names != {arm}:
        raise SystemExit(f"this state directory is {sorted(names)}, not {arm}")


def stop(home, arm, reason, wait=180, sleep=time.sleep):
    check_arm(read(home), arm)
    marker = os.path.join(home, "HALTED")
    if not os.path.exists(marker):
        with open(marker, "x") as f:
            json.dump({"kind": "RACE_ARM_STOPPED", "reason": reason, "at": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
    t0 = time.time()
    while time.time() - t0 < wait:
        if sweep.runner_alive(home):
            try:
                os.kill(int(open(os.path.join(home, "runner.pid")).read()), signal.SIGTERM)
            except (OSError, ValueError):
                pass
        elif not read(home).get("runpod", {}).get("pod"):
            return read(home)
        sleep(5)
    raise SystemExit("the pod is still recorded: the sweep ends it once the runner is dead. Run this again")


def release_volume(home, arm, reason):
    state = read(home)
    check_arm(state, arm)
    if sweep.runner_alive(home) or state.get("runpod", {}).get("pod"):
        raise SystemExit("the arm is not stopped: volume kept")
    vol = state["runpod"].get("volume")
    if not vol:
        raise SystemExit("no volume recorded")
    live = [v for v in api.list_volumes() if v.get("id") == vol["id"]]
    if live and live[0].get("name") != f"{arm}-vol":
        raise SystemExit(f"volume {vol['id']} is named {live[0].get('name')}, not {arm}-vol")
    if live and not api.delete_volume(vol["id"]):
        raise SystemExit(f"delete of {vol['id']} failed")
    now = time.time()
    for e in state["runpod"]["ledger"]["volumes"]:
        if e["id"] == vol["id"] and not e.get("deleted"):
            e["deleted"] = now
    state["runpod"]["volume"] = None
    state["release_reason"] = reason
    tmp = os.path.join(home, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(home, "state.json"))
    print(f"volume {vol['id']} deleted; ledger closed at ${api.ledger_usd(state['runpod']['ledger']):.2f}")
    return state


def release_rung(home, arm, rung, reason, ahead=False):
    check_arm(read(home), arm)
    marker = os.path.join(home, "HALTED")
    paused = os.path.exists(marker) and json.load(open(marker)).get("rung") == rung
    if not paused and not ahead:
        raise SystemExit(f"this arm is not paused at {rung}")
    with open(os.path.join(home, f"RELEASED_{rung}"), "x") as f:
        json.dump({"reason": reason, "at": time.time()}, f)
    if paused:
        os.replace(marker, os.path.join(home, f"HALTED.{rung}"))


def reset_failed(home, arm, reason):
    """After a halted job has been read and repaired: every unfinished job goes back to not started, in state.json here and, through prepare(), on the
    volume, where FAILED otherwise persists. Home jobs and checkpoints are untouched. The marker is kept as HALTED.failed. Only a stopped arm."""
    state = read(home)
    check_arm(state, arm)
    if sweep.runner_alive(home) or state.get("runpod", {}).get("pod"):
        raise SystemExit("the arm is not stopped")
    marker = os.path.join(home, "HALTED")
    if not os.path.exists(marker) or json.load(open(marker)).get("kind") != "JOB_FAILED_STATE_KEPT":
        raise SystemExit("this arm is not halted on a failed job")
    names = [n for n, j in state["jobs"].items() if not j.get("home")]
    if any(j.get("relayed_step") for j in state["jobs"].values()):
        raise SystemExit("this arm has trained: a reset here is not for that")
    for n in names:
        state["jobs"][n] = {"relayed_step": 0, "status": None, "home": False}
    state["reset_jobs"] = names
    state["reset_reason"] = {"reason": reason, "at": time.time()}
    state["booted"] = False
    tmp = os.path.join(home, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(home, "state.json"))
    os.replace(marker, os.path.join(home, "HALTED.failed"))
    print(f"reset: {names}")
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["stop", "release-volume", "release-rung", "reset-failed"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--rung")
    ap.add_argument("--ahead", action="store_true")
    a = ap.parse_args(argv)
    home = os.path.expanduser(a.state_dir)
    if a.cmd == "stop":
        stop(home, a.arm, a.reason)
    elif a.cmd == "release-volume":
        release_volume(home, a.arm, a.reason)
    elif a.cmd == "reset-failed":
        reset_failed(home, a.arm, a.reason)
    else:
        if not a.rung:
            ap.error("--rung is required")
        release_rung(home, a.arm, a.rung, a.reason, a.ahead)
    return 0


if __name__ == "__main__":
    sys.exit(main())
