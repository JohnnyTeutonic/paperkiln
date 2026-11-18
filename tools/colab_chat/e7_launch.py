#!/usr/bin/env python3
"""E7's start gate (research tree: AI_ML/chat_screens_claude/base2/e7_shape/REGISTRATION.md, section 6).

    python3 -u tools/colab_chat/e7_launch.py --name pk-e7-E7A --commit <40-character SHA>          the arm
    python3 -u tools/colab_chat/e7_launch.py --guard --campaign e7-shape --commit <SHA>            its units guard
    python3 tools/colab_chat/e7_launch.py --status

The token ladder's gate (ladder_launch.py), imported and NOT edited, run with E7's table: one arm (E7A, the ladder's N1B
with base A's shape; tools/colab_chat/e7_E7A.json), its own launch folder (/home/jonat/e7_launch: GO.json, launch.json),
its own units guard (campaign e7-shape), its own cap. Everything the ladder's gate does for an arm holds unchanged: seed 2
off, read as the ladder reads it (its lead note from the LADDER's launch folder); admission through a GO record in E7's
launch folder; a deadline written ONCE at the arm's first start and passed at every restart; the hour limit of a 1 B rung
(26 L4-hours); room, disk and balance checks; the units guard. E7 adds, before the arm's first start:

  LADDER ENDED   the token ladder was admitted and has ended (E9's reviewed reading, e9_launch.ladder_ended, imported):
                 E7 shares no Colab week with the ladder. Colab allows three L4 sessions and the ladder uses three; and the
                 ladder's units guard counts the account's whole balance, so E7's units spent beside it would count
                 against the ladder's stop.
  E9 NOT LIVE    E9 has not been admitted (no /home/jonat/e9_launch/launch.json) and none of its arms or corpus builds has
                 started: E9 needs all three sessions. If E9 is live, E7 halts (E7_AFTER_E9) for the lead to read; it does
                 not wait a week holding a gate open.
  THE CAP        the arm's worst case (26 L4-hours, 50.44 units) fits under CAP_UNITS (72).

Claude Opus 5.5 (a fork of the coordinating session), 19 Nov 2026.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_launch as LL  # noqa: E402  (the ladder's gate, unchanged)
import e9_launch as E9  # noqa: E402  (E9's reviewed ladder-end reading, unchanged; its install() is never called here)

S2 = LL.S2
E7_LAUNCH_DIR = "/home/jonat/e7_launch"
E7_GUARD = {"campaign": "e7-shape", "dir": "/home/jonat/e7_units_guard"}
E9_LAUNCH_DIR = E9.E9_LAUNCH_DIR
ARM_HOURS = LL.MAX_HOURS["1B"]                                   # 26
ARM_WORST_UNITS = ARM_HOURS * LL.L4_UNITS_PER_HOUR               # 50.44
CAP_UNITS = 72.0
GUARD_STOP_AT = round(CAP_UNITS - LL.UNITS_MARGIN, 1)            # 52.0: above the worst case, below the cap
OWN_COPY = "e7_launch.json"
ARMS = {"pk-e7-E7A": {"arm": "E7A", "rung": "1B", "slot": 1, "state_dir": "/home/jonat/e7_E7A_runner",
                      "art": "/mnt/c/ml_artifacts/e7/arm_E7A", "campaign": "e7-E7A", "before": None,
                      "jobs": "tools/colab_chat/e7_E7A.json"}}
_ORIG_DECIDE = LL.decide_arm
_ORIG_RUNNER = LL.runner_argv
_ORIG_UNITS = LL.units_needed


def runner_argv(name, commit, deadline_utc):
    """The ladder's runner command with E7's jobs file and names (hours and every other flag the ladder's)."""
    argv = _ORIG_RUNNER(name, commit, deadline_utc)
    a = ARMS[name]
    out = [a["jobs"] if x == f"tools/colab_chat/ladder_{a['arm']}.json" else x for x in argv]
    assert a["jobs"] in out, out
    return out


def _started(x, own=OWN_COPY):
    return S2.launched(x["state_dir"]) or os.path.lexists(os.path.join(x["state_dir"], own))


def committed_units(arms=None):
    arms = ARMS if arms is None else arms
    return sum(ARM_WORST_UNITS for x in arms.values() if _started(x))


def e9_live(e9_launch_dir=None, e9_arms=None, e9_builds=None):
    """None when E9 is not live (never admitted, nothing of it started), else why it is."""
    d = E9_LAUNCH_DIR if e9_launch_dir is None else e9_launch_dir
    if os.path.lexists(os.path.join(d, "launch.json")):
        return "E9 has been admitted (its launch.json exists)"
    for n, x in (E9.ARMS if e9_arms is None else e9_arms).items():
        if E9._started(x):
            return f"E9's arm {n} has started"
    for n, x in (E9.BUILDS if e9_builds is None else e9_builds).items():
        if E9._started(x, E9.BUILD_COPY):
            return f"E9's corpus build {n} has started"
    return None


def decide_arm(name, commit, now, slots, rows_fn, bal_fn, launch_dir=None, arms=None, cmdlines=None,
               ladder_ended=None, e9=None, **kw):
    arms = ARMS if arms is None else arms
    launch_dir = E7_LAUNCH_DIR if launch_dir is None else launch_dir
    a = arms[name]
    if not _started(a):
        why = (ladder_ended or E9.ladder_ended)(cmdlines, rows=rows_fn() if rows_fn else None)
        if why:
            return "wait", f"the token ladder has not ended ({why}): E7 starts after it"
        why = (e9 or e9_live)()
        if why:
            return "halt", ("E7_AFTER_E9", f"{why}: E7 runs before E9 or not at all on this route; the lead reads it")
        if committed_units(arms) + ARM_WORST_UNITS > CAP_UNITS + 1e-9:
            return "halt", ("E7_CAP", f"{committed_units(arms):.1f} units committed; the arm's worst case "
                            f"{ARM_WORST_UNITS:.1f} would pass the cap of {CAP_UNITS:g}")
    return _ORIG_DECIDE(name, commit, now, slots, rows_fn, bal_fn, launch_dir=launch_dir, arms=arms, **kw)


def units_needed(name, arms, now, rec):
    return _ORIG_UNITS(name, ARMS if arms is None else arms, now, rec)


def install():
    """Swap E7's table into the ladder's gate (this process only)."""
    LL.ARMS = ARMS
    LL.LAUNCH_DIR = E7_LAUNCH_DIR
    LL.GUARD = E7_GUARD
    LL.CAP_UNITS = CAP_UNITS
    LL.GUARD_STOP_AT = GUARD_STOP_AT
    LL.OWN_COPY = OWN_COPY
    LL.WORST_HOURS = ARM_HOURS
    LL.WORST_UNITS = round(LL.WORST_HOURS * LL.L4_UNITS_PER_HOUR, 1)
    LL.runner_argv = runner_argv
    LL.decide_arm = decide_arm
    LL.units_needed = units_needed
    LL.seed2_section_ended = E9.seed2_section_ended      # seed 2 read from the LADDER's launch folder, as E9 reads it
    LL.seed2_off = E9.seed2_off


def main(argv=None):
    install()
    return LL.main(argv)


if __name__ == "__main__":
    sys.exit(main())
