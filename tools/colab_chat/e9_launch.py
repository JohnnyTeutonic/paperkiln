#!/usr/bin/env python3
"""E9's start gate (research tree: cue_selective/diversity/REGISTRATION.md section 7, amendments A4 to A6).

    python3 -u tools/colab_chat/e9_launch.py --name pk-e9-<ARM> --commit <40-character SHA>          an arm
    python3 -u tools/colab_chat/e9_launch.py --build pk-e9-data-<CORPUS> --commit <SHA>             a corpus build
    python3 -u tools/colab_chat/e9_launch.py --guard --campaign e9-diversity --commit <SHA>
    python3 tools/colab_chat/e9_launch.py --status

The token ladder's gate (ladder_launch.py), imported and NOT edited, run with E9's tables: its arms, launch folder, guard,
cap and jobs files are swapped in this process only. Everything the ladder's gate does for an arm holds unchanged: seed 2
off, read as the ladder reads it (its lead note from the LADDER's launch folder); admission through a GO record in E9's
launch folder; per-arm deadlines written ONCE at an arm's first start and passed at every restart; hour limits (26 L4-hours
an arm); room, disk and balance checks; the units guard. E9 adds, before an arm's first start:

  LADDER ENDED  the ladder was admitted (its launch.json exists) and has ended: every arm of the ladder's own table is home,
                or its stop is confirmed (HALTED with stop_confirmed, or DEADLINE stopped), or it never launched and holds
                HALTED or the ladder's week deadline has passed; and no process carries a ladder arm's name (as --name X or
                --name=X);
  BASELINES     the lead's BASELINES.json records the ladder's N1B and S1B as valid and reused;
  THE CAP       a unit ledger: each started corpus build folder at 11.64 units (6 L4-hours), each required build not yet
                started reserved at the same, and each started arm at 50.44 (26 L4-hours), all in full even when finished;
                this arm's worst case must fit under CAP_UNITS (350). Three builds and six arms are 337.6;
  THE ORDER     slot 1: D2, then N1Bs50; slot 2: D3, then S1Bs50; slot 3: D4, then the seed-2 D arm (named once in
                REPLICATE.json, frozen when one starts), then D4ops after every planned arm is home or stopped.

A CORPUS BUILD (--build, A6) is gated the same way: the ladder has ended and seed 2 is off (the ladder's own reading); its
start fits the ledger; D4ops's build waits until a planned arm is known not to start (it holds HALTED without having
launched), so that its build and its arm fit the cap. Its deadline is written ONCE at its first start (first start + 6
hours, in its own state folder `/home/jonat/e9_data_<CORPUS>_runner`, never an arm's) and passed as --deadline-utc at every
start, with --max-hours 6, so no restart extends it. After the deadline it is halted for a person to read.

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
LADDER_OWN = LL.OWN_COPY              # the ladder arms' own deadline copy
LADDER_LAUNCH_DIR = LL.LAUNCH_DIR     # the ladder's launch folder (its launch.json, its SEED2_ENDED.json)
E9_LAUNCH_DIR = "/home/jonat/e9_launch"
E9_GUARD = {"campaign": "e9-diversity", "dir": "/home/jonat/e9_units_guard"}
CAP_UNITS = 350.0
ARM_HOURS = LL.MAX_HOURS["1B"]       # 26
ARM_WORST_UNITS = ARM_HOURS * LL.L4_UNITS_PER_HOUR             # 50.44
BUILD_HOURS = 6
BUILD_UNITS = BUILD_HOURS * LL.L4_UNITS_PER_HOUR               # 11.64
REQUIRED_BUILDS = ("D2", "D3", "D4")
# The guard counts from admission. All four builds may come before or after it; it stops early enough that the builds,
# the guard's own stop and the 20 units the ladder keeps for stopping sessions stay under the cap: 350 - 4 x 11.64 - 20.
GUARD_STOP_AT = round(CAP_UNITS - 4 * BUILD_UNITS - LL.UNITS_MARGIN, 1)    # 283.4
REPLICATES = ("D2s50", "D3s50", "D4s50")
SLOTS = {1: ["D2", "N1Bs50"], 2: ["D3", "S1Bs50"], 3: ["D4", "REPLICATE", "D4ops"]}
OWN_COPY = "e9_launch.json"
BUILD_COPY = "e9_build.json"
PLAN = ("D2", "D3", "D4", "N1Bs50", "S1Bs50")          # the required arms besides the replicate (D4ops is optional)


def _arms():
    out = {}
    for slot, names in SLOTS.items():
        for i, arm in enumerate(names):
            members = REPLICATES if arm == "REPLICATE" else (arm,)
            for m in members:
                before = names[i - 1] if i else None
                before = None if before == "REPLICATE" else before          # D4ops waits for every planned arm
                out[f"pk-e9-{m}"] = {"arm": m, "rung": "1B", "slot": slot, "state_dir": f"/home/jonat/e9_{m}_runner",
                                     "art": f"/mnt/c/ml_artifacts/e9/arm_{m}", "campaign": f"e9-{m}",
                                     "before": f"pk-e9-{before}" if before else None,
                                     "jobs": f"tools/colab_chat/{'ladder' if m[0] in 'NS' else 'e9'}_{m}.json"}
    return out


def _builds():
    return {f"pk-e9-data-{c}": {"corpus": c, "state_dir": f"/home/jonat/e9_data_{c}_runner",
                                "art": f"/mnt/c/ml_artifacts/e9/data_{c}", "jobs": f"tools/colab_chat/e9_data_{c}.json"}
            for c in ("D2", "D3", "D4", "D4ops")}


ARMS = _arms()
BUILDS = _builds()
_ORIG_DECIDE = LL.decide_arm
_ORIG_RUNNER = LL.runner_argv
_ORIG_UNITS = LL.units_needed
_ORIG_SEED2_ENDED = LL.seed2_section_ended
_ORIG_SEED2_OFF = LL.seed2_off


def runner_argv(name, commit, deadline_utc):
    """The ladder's runner command with E9's jobs file and names (hours and every other flag the ladder's)."""
    argv = _ORIG_RUNNER(name, commit, deadline_utc)
    a = ARMS[name]
    return [a["jobs"] if x == f"tools/colab_chat/ladder_{a['arm']}.json" else x for x in argv]


def build_argv(name, commit, deadline_utc, builds=None):
    """A corpus build's runner command: the ladder's arm command shape (runner_multi.py with the store's repo and token,
    base 2's boot script) with the build's jobs file, its own folder, --max-hours 6 and its recorded deadline, and without
    the three checkpoint-store flags of a training arm (--store-campaign, --store-first-start, --store-release-storage)."""
    b = (BUILDS if builds is None else builds)[name]
    return [sys.executable, "-u", "tools/colab_chat/runner_multi.py", "--commit", commit, "--jobs", b["jobs"],
            "--name", name, "--art", b["art"], "--state-dir", b["state_dir"], "--gpu", "L4",
            "--max-hours", str(BUILD_HOURS), "--deadline-utc", deadline_utc, "--ckpt-store", "hf",
            "--boot-script", "tools/colab_chat/boot_base2.sh", "--store-repo", S2.STORE_REPO,
            "--store-token-file", S2.TOKEN_FILE]


# ---------------------------------------------------------------- the ladder's end, and seed 2 read from the ladder's folder
def seed2_section_ended(now, launch_dir=None, **kw):
    return _ORIG_SEED2_ENDED(now, launch_dir=LADDER_LAUNCH_DIR if launch_dir is None else launch_dir, **kw)


def seed2_off(now=None, cmdlines=None, rows=None, seed2_arms=None, launch_dir=None, seed2_launch=None):
    return _ORIG_SEED2_OFF(now=now, cmdlines=cmdlines, rows=rows, seed2_arms=seed2_arms,
                           launch_dir=LADDER_LAUNCH_DIR if launch_dir is None else launch_dir, seed2_launch=seed2_launch)


def _names_in(line, names):
    """Whether a process line (an argv list or a string) carries one of the names, as --name X or --name=X."""
    if isinstance(line, (list, tuple)):
        for i, x in enumerate(line):
            if x in names or (x.startswith("--name=") and x[len("--name="):] in names):
                return True
            if x == "--name" and i + 1 < len(line) and line[i + 1] in names:
                return True
        return False
    return any(n in line for n in names)


def ladder_ended(cmdlines=None, arms=None, launch_dir=None, now=None):
    """None when the ladder has ended (as the docstring says), else why not."""
    arms = LADDER_ARMS if arms is None else arms
    rec = LL.read_json(os.path.join(LADDER_LAUNCH_DIR if launch_dir is None else launch_dir, "launch.json"))
    if not rec:
        return "the ladder has no launch.json: it was never admitted, so it has not ended"
    week = LL.parse_iso(rec.get("deadline_utc"))
    now = time.time() if now is None else now
    for n, x in arms.items():
        d = x["state_dir"]
        if LL.seed2_arm_stopped(d):
            continue
        launched = S2.launched(d) or os.path.lexists(os.path.join(d, LADDER_OWN))
        if launched:
            return f"the ladder's arm {n} has not stopped"
        if not (os.path.lexists(os.path.join(d, "HALTED")) or (week is not None and now > week)):
            return f"the ladder's arm {n} never launched and could still start (no HALTED, the ladder's week not over)"
    lines = LL.scan() if cmdlines is None else cmdlines
    if lines is None:
        return "the process table cannot be read"
    for line in lines:
        if _names_in(line, tuple(arms)):
            return "a process of the ladder is running"
    return None


# ---------------------------------------------------------------- the ledger
def _started(x, own=OWN_COPY):
    return S2.launched(x["state_dir"]) or os.path.lexists(os.path.join(x["state_dir"], own))


def committed_units(arms=None, builds=None):
    arms = ARMS if arms is None else arms
    builds = BUILDS if builds is None else builds
    units = 0.0
    for b in builds.values():
        if _started(b, BUILD_COPY) or b["corpus"] in REQUIRED_BUILDS:      # required builds are reserved from the start
            units += BUILD_UNITS
    return units + sum(ARM_WORST_UNITS for x in arms.values() if _started(x))


def replicate_choice(launch_dir=None, arms=None):
    """The seed-2 D arm: frozen once one has started (a later REPLICATE.json cannot admit a second); else the lead's file."""
    arms = ARMS if arms is None else arms
    begun = [x["arm"] for x in arms.values() if x["arm"] in REPLICATES and _started(x)]
    if begun:
        return begun[0]
    rec = LL.read_json(os.path.join(launch_dir or E9_LAUNCH_DIR, "REPLICATE.json"))
    return (rec or {}).get("arm") if (rec or {}).get("arm") in REPLICATES and (rec or {}).get("who") else None


def plan_arms(arms, launch_dir=None):
    rep = replicate_choice(launch_dir, arms) or "D4s50"   # until it is named, one seed-2 D arm is planned for
    return {n: x for n, x in arms.items() if x["arm"] in PLAN or x["arm"] == rep}


def baselines_ok(launch_dir=None):
    """Section 2's fallback (E9's own N1B or S1B) is not automated: the lead records that the ladder's are valid."""
    rec = LL.read_json(os.path.join(launch_dir or E9_LAUNCH_DIR, "BASELINES.json")) or {}
    return rec.get("N1B") == "valid" and rec.get("S1B") == "valid" and bool(rec.get("who")) and bool(rec.get("when"))


def units_needed(name, arms, now, rec):
    """The ladder's worst case of what is left, over the registered plan only (and `name`, if D4ops), not every entry."""
    keep = plan_arms(arms)
    if name in arms and name not in keep:
        keep = dict(keep, **{name: arms[name]})
    return _ORIG_UNITS(name, keep, now, rec)


# ---------------------------------------------------------------- an arm
def decide_arm(name, commit, now, slots, rows_fn, bal_fn, launch_dir=None, arms=None, cmdlines=None, builds=None, **kw):
    arms = ARMS if arms is None else arms
    launch_dir = E9_LAUNCH_DIR if launch_dir is None else launch_dir
    a = arms[name]
    if not _started(a):
        why = ladder_ended(cmdlines)
        if why:
            return "wait", f"the token ladder has not ended ({why}): E9 starts after it"
        if not baselines_ok(launch_dir):
            return "wait", "BASELINES.json does not record the ladder's N1B and S1B as valid and reused"
        if a["arm"] in REPLICATES:
            choice = replicate_choice(launch_dir, arms)
            if choice is None:
                return "wait", "the seed-2 D arm waits for REPLICATE.json (seed 1's reading names it)"
            if choice != a["arm"]:
                return "halt", ("E9_NOT_REPLICATE", f"REPLICATE.json names {choice}, not {a['arm']}")
        if a["arm"] == "D4ops":
            if replicate_choice(launch_dir, arms) is None:
                return "wait", "D4ops runs last: the seed-2 D arm is not yet named"
            waiting = [n for n, x in plan_arms(arms, launch_dir).items() if not LL.seed2_arm_stopped(x["state_dir"])]
            if waiting:
                return "wait", f"D4ops runs last: after every planned arm is home or stopped ({', '.join(sorted(waiting))})"
        if committed_units(arms, builds) + ARM_WORST_UNITS > CAP_UNITS + 1e-9:
            return "halt", ("E9_CAP", f"{committed_units(arms, builds):.1f} units committed; this arm's worst case "
                            f"{ARM_WORST_UNITS:.1f} would pass the cap of {CAP_UNITS:g}")
    return _ORIG_DECIDE(name, commit, now, slots, rows_fn, bal_fn, launch_dir=launch_dir, arms=arms, **kw)


# ---------------------------------------------------------------- a corpus build (A6)
def decide_build(name, commit, now, arms=None, builds=None, cmdlines=None, seed2=None, launch_dir=None):
    """(kind, what) as decide_arm: ("exec", argv) | ("wait", why) | ("halt", (kind, why))."""
    arms = ARMS if arms is None else arms
    builds = BUILDS if builds is None else builds
    launch_dir = E9_LAUNCH_DIR if launch_dir is None else launch_dir
    b = builds[name]
    own = os.path.join(b["state_dir"], BUILD_COPY)
    if os.path.lexists(os.path.join(b["state_dir"], "HALTED")):
        return "halt", ("E9_BUILD_HALTED", "this build holds HALTED: the gate runs nothing; a person reads it")
    rec = LL.read_json(own)
    if rec is None:
        why = ladder_ended(cmdlines)
        if why:
            return "wait", f"the token ladder has not ended ({why}): no E9 build starts before it"
        off = (seed2 or seed2_off)(now=now)
        if off:
            return "wait", f"seed 2 is not off ({off})"
        if b["corpus"] == "D4ops":
            plan = plan_arms(arms, launch_dir)
            dropped = [n for n, x in plan.items() if not _started(x) and os.path.lexists(os.path.join(x["state_dir"], "HALTED"))]
            if not dropped:
                return "wait", "D4ops's build waits until a planned arm is known not to start"
        extra = 0.0 if b["corpus"] in REQUIRED_BUILDS else BUILD_UNITS
        if committed_units(arms, builds) + extra > CAP_UNITS + 1e-9:
            return "halt", ("E9_CAP", "this build would pass the cap")
        os.makedirs(b["state_dir"], exist_ok=True)
        LL.write_once(own, {"first_start_utc": LL.iso(now), "deadline_utc": LL.iso(now + BUILD_HOURS * 3600),
                            "max_hours": BUILD_HOURS, "commit": commit})
        rec = LL.read_json(own)
    dl = LL.parse_iso((rec or {}).get("deadline_utc"))
    if dl is None:
        return "halt", ("E9_BUILD_RECORD", "the build's own deadline record cannot be read")
    if now >= dl:
        return "halt", ("E9_BUILD_DEADLINE", f"past the build's deadline {rec['deadline_utc']}")
    return "exec", build_argv(name, commit, rec["deadline_utc"], builds)


def install():
    """Swap E9's tables into the ladder's gate (this process only)."""
    LL.ARMS = ARMS
    LL.LAUNCH_DIR = E9_LAUNCH_DIR
    LL.GUARD = E9_GUARD
    LL.CAP_UNITS = CAP_UNITS
    LL.GUARD_STOP_AT = GUARD_STOP_AT
    LL.OWN_COPY = OWN_COPY
    LL.WORST_HOURS = 6 * ARM_HOURS
    LL.WORST_UNITS = round(LL.WORST_HOURS * LL.L4_UNITS_PER_HOUR, 1)
    LL.runner_argv = runner_argv
    LL.decide_arm = decide_arm
    LL.units_needed = units_needed
    LL.seed2_section_ended = seed2_section_ended
    LL.seed2_off = seed2_off


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    install()
    if "--build" in argv:
        i = argv.index("--build")
        name = argv[i + 1]
        commit = argv[argv.index("--commit") + 1] if "--commit" in argv else ""
        if name not in BUILDS or len(commit) != 40:
            raise SystemExit(f"--build is one of {sorted(BUILDS)}, with --commit the full launch SHA")
        if S2.head_commit() != commit:
            raise SystemExit(f"this worktree is not at {commit}; the gate runs only from the launch commit")
        os.chdir(S2.REPO)
        said = None
        while True:
            now = time.time()
            shared = LL.take_lock(os.path.join(E9_LAUNCH_DIR, "gate.lock"), wait=True)
            try:
                kind, what = decide_build(name, commit, now)
            finally:
                shared.close()
            if kind == "exec":
                print(f"{LL.iso(now)} e9_launch: starting {' '.join(what[1:])}", flush=True)
                os.execv(what[0], what)
            if kind == "halt":
                p = os.path.join(BUILDS[name]["state_dir"], "HALTED")
                if not os.path.exists(p):
                    os.makedirs(BUILDS[name]["state_dir"], exist_ok=True)
                    S2.write_file(p, dict(S2.halt_record(what[0], what[1], now), stop_confirmed=None,
                                          note="written by E9's gate, which stops no session itself"))
                print(f"{LL.iso(now)} e9_launch: {what[1]}: HALTED ({what[0]})", flush=True)
                return 0
            if what != said:
                print(f"{LL.iso(now)} e9_launch: waiting: {what}", flush=True)
                said = what
            time.sleep(LL.POLL_S)
    return LL.main(argv)


if __name__ == "__main__":
    sys.exit(main())
