#!/usr/bin/env python3
"""E9's start gate (research tree: cue_selective/diversity/REGISTRATION.md section 7 and amendment A4).

    python3 -u tools/colab_chat/e9_launch.py --name pk-e9-<ARM> --commit <40-character SHA>
    python3 -u tools/colab_chat/e9_launch.py --guard --campaign e9-diversity --commit <SHA>
    python3 tools/colab_chat/e9_launch.py --status

The token ladder's gate (ladder_launch.py), imported and NOT edited, run with E9's tables: its arms, launch folder, guard,
cap and jobs files are swapped in this process only. Everything the ladder's gate does holds unchanged: seed 2 off, read as
the ladder reads it; admission through a GO record in E9's launch folder; per-arm deadlines written ONCE at an arm's first
start and passed at every restart, so a restart never extends an arm; hour limits (26 L4-hours an arm, the 1B rung's);
room, disk and balance checks; the units guard. E9 adds, before an arm's first start:

  LADDER ENDED  the token ladder has ended: every arm of the ladder's own table is home, or its stop is confirmed (HALTED
                with stop_confirmed, or DEADLINE stopped), or never launched; and no process carries a ladder arm's name;
  THE CAP       a unit ledger: the four corpus builds' worst case (6 L4-hours each, 46.6 units, reserved from the start)
                plus the worst case of every E9 arm that has started (26 L4-hours at 1.94 units each, counted in
                full even when finished) plus this arm's must not pass CAP_UNITS (350). Six arms fit (349.2); a seventh
                (the last in order) is refused (halted, E9_CAP) rather than run;
  THE ORDER     slot 1: D2, then N1Bs50; slot 2: D3, then S1Bs50; slot 3: D4, then the seed-2 D arm, then D4ops. The
                seed-2 D arm is the one the lead names in E9_LAUNCH_DIR/REPLICATE.json after seed 1's reading (the onset
                level's, or D4s50); the other two seed-2 D arms halt. D4ops waits for the replicate arm to be home or
                stopped. So the cap drops D4ops first, then the replicate.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_launch as LL  # noqa: E402  (the ladder's gate, unchanged)

S2 = LL.S2
LADDER_ARMS = dict(LL.ARMS)          # the ladder's own arms, captured before E9's are swapped in
LADDER_NAMES = tuple(LADDER_ARMS)
LADDER_OWN = LL.OWN_COPY              # the ladder arms' own deadline copy, captured before E9's name is swapped in
E9_LAUNCH_DIR = "/home/jonat/e9_launch"
E9_GUARD = {"campaign": "e9-diversity", "dir": "/home/jonat/e9_units_guard"}
CAP_UNITS = 350.0
GUARD_STOP_AT = 330.0
ARM_HOURS = LL.MAX_HOURS["1B"]       # 26
ARM_WORST_UNITS = ARM_HOURS * LL.L4_UNITS_PER_HOUR
BUILD_RESERVE_UNITS = 4 * 6 * LL.L4_UNITS_PER_HOUR   # the four corpus builds (e9_data_<ARM>.json), 6 L4-hours each, worst case
REPLICATES = ("D2s50", "D3s50", "D4s50")
SLOTS = {1: ["D2", "N1Bs50"], 2: ["D3", "S1Bs50"], 3: ["D4", "REPLICATE", "D4ops"]}
OWN_COPY = "e9_launch.json"


def _arms():
    out = {}
    for slot, names in SLOTS.items():
        for i, arm in enumerate(names):
            members = REPLICATES if arm == "REPLICATE" else (arm,)
            for m in members:
                before = names[i - 1] if i else None
                before = None if before == "REPLICATE" else before          # D4ops waits for the replicate (below)
                out[f"pk-e9-{m}"] = {"arm": m, "rung": "1B", "slot": slot, "state_dir": f"/home/jonat/e9_{m}_runner",
                                     "art": f"/mnt/c/ml_artifacts/e9/arm_{m}", "campaign": f"e9-{m}",
                                     "before": f"pk-e9-{before}" if before else None,
                                     "jobs": f"tools/colab_chat/{'ladder' if m[0] == 'N' or m[0] == 'S' else 'e9'}_{m}.json"}
    return out


ARMS = _arms()
_ORIG_DECIDE = LL.decide_arm
_ORIG_RUNNER = LL.runner_argv


def runner_argv(name, commit, deadline_utc):
    """The ladder's runner command with E9's jobs file and names (hours and every other flag the ladder's)."""
    argv = _ORIG_RUNNER(name, commit, deadline_utc)
    a = ARMS[name]
    return [a["jobs"] if x == f"tools/colab_chat/ladder_{a['arm']}.json" else x for x in argv]


def _stopped(state_dir):
    return LL.seed2_arm_stopped(state_dir) or not (S2.launched(state_dir) or os.path.lexists(os.path.join(state_dir, LADDER_OWN)))


def ladder_ended(cmdlines=None, arms=None):
    """None when the ladder has ended (as above), else why not."""
    arms = LADDER_ARMS if arms is None else arms
    for n, x in arms.items():
        if not _stopped(x["state_dir"]):
            return f"the ladder's arm {n} has not stopped"
    lines = LL.scan() if cmdlines is None else cmdlines
    if lines is None:
        return "the process table cannot be read"
    for line in lines:
        if any(n in line for n in arms):
            return "a process of the ladder is running"
    return None


def committed_units(arms=None):
    arms = ARMS if arms is None else arms
    return BUILD_RESERVE_UNITS + sum(ARM_WORST_UNITS for x in arms.values()
               if S2.launched(x["state_dir"]) or os.path.lexists(os.path.join(x["state_dir"], OWN_COPY)))


def replicate_choice(launch_dir=None):
    rec = LL.read_json(os.path.join(launch_dir or E9_LAUNCH_DIR, "REPLICATE.json"))
    return (rec or {}).get("arm") if (rec or {}).get("arm") in REPLICATES and (rec or {}).get("who") else None


def decide_arm(name, commit, now, slots, rows_fn, bal_fn, launch_dir=None, arms=None, cmdlines=None, **kw):
    arms = ARMS if arms is None else arms
    launch_dir = E9_LAUNCH_DIR if launch_dir is None else launch_dir
    a = arms[name]
    started = S2.launched(a["state_dir"]) or os.path.lexists(os.path.join(a["state_dir"], OWN_COPY))
    if not started:
        why = ladder_ended(cmdlines)
        if why:
            return "wait", f"the token ladder has not ended ({why}): E9 starts after it"
        if a["arm"] in REPLICATES:
            choice = replicate_choice(launch_dir)
            if choice is None:
                return "wait", "the seed-2 D arm waits for REPLICATE.json (seed 1's reading names it)"
            if choice != a["arm"]:
                return "halt", ("E9_NOT_REPLICATE", f"REPLICATE.json names {choice}, not {a['arm']}")
        if a["arm"] == "D4ops":
            choice = replicate_choice(launch_dir)
            rep = arms.get(f"pk-e9-{choice}") if choice else None
            if rep is None or not LL.seed2_arm_stopped(rep["state_dir"]):
                return "wait", "D4ops runs last: after the seed-2 D arm is home or stopped"
        if committed_units(arms) + ARM_WORST_UNITS > CAP_UNITS + 1e-9:
            return "halt", ("E9_CAP", f"{committed_units(arms):.1f} units committed; this arm's worst case "
                            f"{ARM_WORST_UNITS:.1f} would pass the cap of {CAP_UNITS:g}")
    return _ORIG_DECIDE(name, commit, now, slots, rows_fn, bal_fn, launch_dir=launch_dir, arms=arms, **kw)


def install():
    """Swap E9's tables into the ladder's gate (this process only)."""
    LL.ARMS = ARMS
    LL.LAUNCH_DIR = E9_LAUNCH_DIR
    LL.GUARD = E9_GUARD
    LL.CAP_UNITS = CAP_UNITS
    LL.GUARD_STOP_AT = GUARD_STOP_AT
    LL.OWN_COPY = OWN_COPY
    LL.WORST_HOURS = int(CAP_UNITS // ARM_WORST_UNITS) * ARM_HOURS
    LL.WORST_UNITS = round(LL.WORST_HOURS * LL.L4_UNITS_PER_HOUR, 1)
    LL.runner_argv = runner_argv
    LL.decide_arm = decide_arm


def main(argv=None):
    install()
    return LL.main(argv)


if __name__ == "__main__":
    sys.exit(main())
