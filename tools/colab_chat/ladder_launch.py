#!/usr/bin/env python3
"""The start gate of the token ladder's pretraining arms and of its units guard (the research tree's
AI_ML/chat_screens_claude/base2/token_ladder/REGISTRATION.md). Claude Fable 5.2, 17 Nov 2026.

The watchdog runs one of these for each arm and one for the guard, from the ladder's worktree at the launch commit:

    python3 -u tools/colab_chat/ladder_launch.py --name pk-tl-<ARM> --commit <40-character SHA>
    python3 -u tools/colab_chat/ladder_launch.py --guard --campaign token-ladder --commit <SHA>
    python3 tools/colab_chat/ladder_launch.py --status          one reading of every condition, printed; starts nothing

It is base 2's second-seed gate (base2_seed2_launch.py, imported for its helpers and not edited) with the ladder's
conditions. A gate checks every POLL_S seconds and replaces itself (exec) with the registered command when every condition
holds; anything it cannot read counts as not holding, so an unreadable listing, balance or file makes it wait.

Seed 2 comes first (PROTOCOL 48.3). Its units guard counts every unit spent on the account, so a ladder session opened while
seed 2 runs would be charged to seed 2's cap and could stop it. No arm and no guard of the ladder starts until ALL hold:
  S1. each seed-2 arm's entry in scripts/colab_driver.json (found once by its state folder) reads "enabled": false;
  S2. no process carries a seed-2 arm's --name or --state-dir;
  S3. seed 2's units-guard entry (found once by its state folder) reads "enabled": false, and no process carries
      --campaign base2-seed2 (the waiting gate and the guard itself both do);
  S4. Colab's listing answers and lists no seed-2 session.
The ladder never writes to seed 2's folders, entries or processes: these are read only. Closing seed 2 is its lead's step.

Admission, once, by the first arm whose gate finds all of these (it writes LAUNCH_DIR/launch.json through a hard link):
  A1. LAUNCH_DIR/GO.json exists and validates (the lane's launch step writes it after its own refusals: Jonathan's quoted
      line, the reviews file and its ruling, this commit);
  A2. S1 to S4;
  A3. no folder of the ladder has launched;
  A4. the listing has room for three L4 sessions;
  A5. the unit balance is at least MIN_BALANCE. Every unit is bought before admission: the guard reads the cap as the fall of
      the balance since admission, so a purchase during the ladder would hide spending from it.
The record holds the admission time, the ladder's one deadline (admission plus WEEK_HOURS), the balance, the commit and the
bounds. Every arm's state folder is created then, so the guard's stop reaches arms that have not started.

An arm that has not launched then starts when: S1 to S4; the guard is running (not asked of the admitting arm, which starts
at admission; the guard starts within one poll of it); the arm before it in its slot (make_ladder.SLOTS) is home with its
last release made; the listing has room for one L4; C: has at least MIN_FREE_GB free. It halts instead (HALTED, no
session made) when: the guard has asked for a stop; the arm before it holds HALTED; or the arm's expected hours no longer
fit before the deadline (LADDER_NO_TIME). A launched arm (state.json "launches" at least 1) runs its runner at once on every
start with the recorded deadline; the runner's own restart rules apply.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base2_seed2_launch as S2  # noqa: E402  (helpers and seed 2's names; unchanged)
import make_ladder as L  # noqa: E402

LAUNCH_DIR = "/home/jonat/token_ladder_launch"
GUARD = {"campaign": "token-ladder", "dir": "/home/jonat/token_ladder_units_guard"}
WEEK_HOURS = 168                 # the whole ladder ends within seven days of admission
CAP_UNITS = 760.0                # the pretraining campaign's cap, in Colab units (about USD 76)
MIN_BALANCE = CAP_UNITS          # the balance admission needs
GUARD_STOP_AT = 740.0            # the cap less 20 for stopping the sessions
MIN_FREE_GB = 8                  # an arm brings about 2 GB home
POLL_S = 300
EXPECT_HOURS = {"1B": 18.5, "2B": 38.0, "T": 114.0}     # at about 3,300 steps an hour on an L4 (base 2's measured rate)
EXPECT_UNITS = {"1B": 36, "2B": 74, "T": 221}           # at 1.94 units an hour
MAX_HOURS = {"1B": 26, "2B": 50, "T": 140}              # each runner's own limit: the expectation and a fifth, rounded up
OWN_COPY = "ladder_launch.json"
GO_LINE = ("Jonathan, direct chat, 17 Nov 2026: \"If you think it's worth it, you make all the decisions. I am happy perhaps "
           "to wait a week maximum if you really want. just make your own decisions.\"")


def _arms():
    out = {}
    for arm in L.FIRST_SEED_ARMS:
        r = L.RUNGS[arm]
        slot = L.SLOTS[r["slot"]]
        i = slot.index(arm)
        out[f"pk-tl-{arm}"] = {"arm": arm, "rung": r["rung"], "state_dir": f"/home/jonat/ladder_{arm}_runner",
                               "art": f"/mnt/c/ml_artifacts/base2/arm_{arm}", "campaign": f"ladder-{arm}",
                               "before": f"pk-tl-{slot[i - 1]}" if i else None}
    return out


ARMS = _arms()
iso, parse_iso, finite, read_json = S2.iso, S2.parse_iso, S2.finite, S2.read_json


def runner_argv(name, commit, deadline_utc):
    """Base 2's arm command (base2_seed2_launch.runner_argv) with the ladder's names, jobs file and hours."""
    a = ARMS[name]
    return [sys.executable, "-u", "tools/colab_chat/runner_multi.py", "--commit", commit,
            "--jobs", f"tools/colab_chat/ladder_{a['arm']}.json", "--name", name, "--art", a["art"],
            "--state-dir", a["state_dir"], "--gpu", "L4", "--max-hours", str(MAX_HOURS[a["rung"]]),
            "--deadline-utc", deadline_utc, "--ckpt-store", "hf", "--boot-script", "tools/colab_chat/boot_base2.sh",
            "--store-repo", S2.STORE_REPO, "--store-token-file", S2.TOKEN_FILE, "--store-campaign", a["campaign"],
            "--store-first-start", "--store-release-storage"]


def guard_argv(start_balance):
    return [sys.executable, "-u", "tools/colab_chat/base2_units_guard.py", "--campaign", GUARD["campaign"],
            "--stop-at", f"{GUARD_STOP_AT:g}", "--start-balance", repr(float(start_balance)), "--every-min", "30",
            "--log", os.path.join(GUARD["dir"], "guard.jsonl"),
            *[x for a in ARMS.values() for x in ("--state-dir", a["state_dir"])]]


# ---------------------------------------------------------------- records

def go_problem(rec, commit):
    if not isinstance(rec, dict):
        return "GO.json is absent or unreadable"
    if rec.get("line") != GO_LINE:
        return "GO.json does not hold Jonathan's line as registered"
    if rec.get("commit") != commit:
        return f"GO.json is for commit {rec.get('commit')}, not {commit}"
    if not (isinstance(rec.get("reviews"), str) and rec["reviews"]) or parse_iso(rec.get("at_utc")) is None:
        return "GO.json names no reviews file or no time"
    return None


def record_problem(rec, commit=None):
    if not isinstance(rec, dict):
        return "not a JSON object"
    at, dl = parse_iso(rec.get("at_utc")), parse_iso(rec.get("deadline_utc"))
    if at is None or dl is None:
        return "admission time or deadline missing or not in the form YYYY-MM-DDTHH:MM:SSZ"
    if dl != at + WEEK_HOURS * 3600:
        return "the deadline is not the admission time plus WEEK_HOURS"
    b = rec.get("start_balance")
    if not finite(b) or b < MIN_BALANCE:
        return "no finite start balance at least MIN_BALANCE"
    if rec.get("min_balance") != MIN_BALANCE or rec.get("guard_stop_at") != GUARD_STOP_AT or rec.get("week_hours") != WEEK_HOURS:
        return "the bounds recorded are not the registered ones"
    c = rec.get("commit")
    if not (isinstance(c, str) and len(c) == 40):
        return "no full commit"
    if commit is not None and c != commit:
        return f"recorded for commit {c}, not {commit}"
    return None


def load_record(path, commit=None):
    if not os.path.lexists(path):
        return "absent", None
    rec = read_json(path)
    why = record_problem(rec, commit) if rec is not None else "unreadable or not a JSON object"
    return ("invalid", why) if why else ("valid", rec)


def write_launch(rec, launch_dir):
    """launch.json, once: a hard link of a finished temporary file, so a second writer meets the first's file."""
    os.makedirs(launch_dir, exist_ok=True)
    path = os.path.join(launch_dir, "launch.json")
    tmp = os.path.join(launch_dir, f".launch.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        pass
    finally:
        os.remove(tmp)
    return load_record(path, rec.get("commit"))


# ---------------------------------------------------------------- the checks (each: None when it holds, else why)

def _has_pair(argv, flag, value):
    return any(x == flag and argv[i + 1] == value for i, x in enumerate(argv[:-1])) or f"{flag}={value}" in argv


def seed2_off(path=None, cmdlines=None, rows=None, seed2_arms=None, seed2_guard=None):
    """S1 to S4. `rows` is one answered Colab listing ([(name, hardware)]) or None."""
    path = S2.DRIVER_JSON if path is None else path
    arms = S2.ARMS if seed2_arms is None else seed2_arms
    guard = S2.GUARD if seed2_guard is None else seed2_guard
    d = read_json(path)
    drivers = (d or {}).get("drivers")
    if not isinstance(drivers, list):
        return "scripts/colab_driver.json is unreadable"
    for what, folder in [(n, a["state_dir"]) for n, a in arms.items()] + [("seed 2's units guard", guard["dir"])]:
        es = [e for e in drivers if isinstance(e, dict) and e.get("state_dir") == folder]
        if len(es) != 1:
            return f"{what}: its entry was not found once by its state folder in scripts/colab_driver.json"
        if es[0].get("enabled") is not False:
            return f"{what}: its watchdog entry is still enabled (seed 2 has not ended, or its guard is not closed)"
    argvs = (cmdlines or S2.process_cmdlines)()
    for argv in argvs:
        for n, a in arms.items():
            if _has_pair(argv, "--name", n) or _has_pair(argv, "--state-dir", a["state_dir"]):
                return f"a process of seed 2's arm {n} is still running"
        if _has_pair(argv, "--campaign", guard["campaign"]):
            return "seed 2's units guard (or its gate) is still running"
    if rows is None:
        return "Colab's listing did not answer"
    held = sorted(n for n, _ in rows if n in arms)
    if held:
        return f"Colab lists seed 2's sessions {held}"
    return None


def guard_running(cmdlines=None):
    return any(any(x.endswith("base2_units_guard.py") for x in argv) and _has_pair(argv, "--campaign", GUARD["campaign"])
               for argv in (cmdlines or S2.process_cmdlines)())


def guard_stop_asked(log=None):
    """True when the ladder's guard has logged a stop request, or its log cannot be parsed (an unknown counts as asked)."""
    log = os.path.join(GUARD["dir"], "guard.jsonl") if log is None else log
    if not os.path.exists(log):
        return False
    try:
        with open(log, encoding="utf-8") as f:
            return any(json.loads(ln).get("stop_requested") for ln in f if ln.strip())
    except (OSError, ValueError, AttributeError):
        return True


def ended_home(state_dir):
    st = read_json(os.path.join(state_dir, "state.json"))
    return bool(st) and st.get("campaign_home") is True and st.get("store_release_final_done") is True


def free_gb(path="/mnt/c"):
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return None


def stop_marked(state_dir):
    return any(os.path.exists(os.path.join(state_dir, f)) for f in ("STOP_REQUEST", "STOP_REQUEST.taken", "HALTED"))


# ---------------------------------------------------------------- the decisions

def decide_arm(name, commit, now, slots, rows_fn, bal_fn, launch_dir=None, seed2=None, guard_up=None, stop_asked=None,
               free=None, arms=None, clock=time.time):
    """(kind, what): ("exec", argv) | ("wait", why) | ("halt", (kind, why)) | ("refuse", why)."""
    launch_dir = LAUNCH_DIR if launch_dir is None else launch_dir
    arms = ARMS if arms is None else arms
    a = arms[name]
    own = os.path.join(a["state_dir"], OWN_COPY)
    path = os.path.join(launch_dir, "launch.json")
    if S2.launched(a["state_dir"]):
        state, rec = load_record(own, commit)
        if state != "valid":
            state, rec = load_record(path, commit)
            if state != "valid":
                return "refuse", f"this folder has launched and has no valid launch record ({rec or 'absent'}): a person reads it"
        return "exec", runner_argv(name, commit, rec["deadline_utc"])
    why = go_problem(read_json(os.path.join(launch_dir, "GO.json")), commit)
    if why:
        return "refuse", why
    if stop_marked(a["state_dir"]) or (stop_asked or guard_stop_asked)():
        return "halt", ("LADDER_UNITS_STOP", "a stop was asked of the ladder before this arm launched")
    state, rec = load_record(path, commit)
    if state == "invalid":
        return "refuse", f"launch.json does not validate ({rec}); it is never replaced: a person reads it"
    rows = rows_fn()
    why = (seed2 or seed2_off)(rows=rows)
    if why:
        return "wait", why
    f = (free or free_gb)()
    if f is None or f < MIN_FREE_GB:
        return "wait", f"C: has {f if f is None else round(f, 1)} GB free, under {MIN_FREE_GB}"
    before = a["before"]
    if before is not None:
        if os.path.exists(os.path.join(arms[before]["state_dir"], "HALTED")):
            return "halt", ("LADDER_BEFORE_HALTED", f"{before}, the arm before this one in its slot, holds HALTED: a person reads it")
        if not ended_home(arms[before]["state_dir"]):
            return "wait", f"{before}, the arm before this one in its slot, is not home with its last release made"
    if state == "absent":
        if before is not None:
            return "wait", "the ladder has not been admitted (a slot's first arm admits it)"
        if any(S2.launched(x["state_dir"]) for x in arms.values()):
            return "refuse", "a folder of the ladder has launched with no launch.json: a person reads it"
        why = S2.room(slots, rows, len(L.SLOTS))
        if why:
            return "wait", why
        b = bal_fn()
        if b is None:
            return "wait", "the unit balance could not be read"
        if b < MIN_BALANCE:
            return "wait", f"the unit balance {b:.2f} is under {MIN_BALANCE:g}: every unit is bought before admission"
        at = int(clock())
        state, rec = write_launch({"at_utc": iso(at), "deadline_utc": iso(at + WEEK_HOURS * 3600), "start_balance": b,
                                   "first": name, "commit": commit, "min_balance": MIN_BALANCE,
                                   "guard_stop_at": GUARD_STOP_AT, "week_hours": WEEK_HOURS,
                                   "checks": "GO.json; seed 2's entries disabled, its processes gone, no seed-2 session "
                                             "listed; L4 room for three; balance at least the cap"}, launch_dir)
        if state != "valid":
            return "refuse", f"launch.json does not validate after the admission ({rec})"
        for x in arms.values():
            os.makedirs(x["state_dir"], exist_ok=True)
        S2.write_file(own, rec)
        return "exec", runner_argv(name, commit, rec["deadline_utc"])
    deadline = parse_iso(rec["deadline_utc"])
    if now + EXPECT_HOURS[a["rung"]] * 3600 > deadline:
        return "halt", ("LADDER_NO_TIME", f"{EXPECT_HOURS[a['rung']]} expected hours no longer fit before {rec['deadline_utc']}")
    if not (guard_up or guard_running)():
        return "wait", "the ladder's units guard is not running"
    why = S2.room(slots, rows, 1)
    if why:
        return "wait", why
    os.makedirs(a["state_dir"], exist_ok=True)
    S2.write_file(own, rec)
    return "exec", runner_argv(name, commit, rec["deadline_utc"])


def decide_guard(now, commit, launch_dir=None, seed2=None, rows_fn=None):
    launch_dir = LAUNCH_DIR if launch_dir is None else launch_dir
    why = go_problem(read_json(os.path.join(launch_dir, "GO.json")), commit)
    if why:
        return "refuse", why
    state, rec = load_record(os.path.join(launch_dir, "launch.json"), commit)
    if state == "invalid":
        return "refuse", f"launch.json does not validate ({rec}); it is never replaced: a person reads it"
    if state == "absent":
        return "wait", "the ladder has not been admitted"
    why = (seed2 or seed2_off)(rows=rows_fn() if rows_fn else None)
    if why:
        return "wait", why
    return "exec", guard_argv(rec["start_balance"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", choices=sorted(ARMS))
    ap.add_argument("--commit")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--campaign")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    if a.guard and a.campaign != GUARD["campaign"]:
        ap.error(f"--guard runs with --campaign {GUARD['campaign']}")
    if not (a.guard or a.status) and not a.name:
        ap.error("--name, --guard or --status")
    if (a.name or a.guard) and (not a.commit or len(a.commit) != 40):
        ap.error("--commit is the full 40-character launch SHA")
    if (a.name or a.guard) and S2.head_commit() != a.commit:
        sys.exit(f"this worktree is not at {a.commit}; the gate runs only from the launch commit")
    os.chdir(S2.REPO)
    try:
        slots = S2.load_slots()
    except Exception as e:  # noqa: BLE001  a gate that cannot count slots waits; it never takes them as free
        print(f"{iso(time.time())} ladder_launch: scripts/colab_slots.py could not be loaded ({type(e).__name__})", flush=True)
        slots = None
    rows_fn = lambda: S2.listing_rows(slots)  # noqa: E731
    said = None
    while True:
        now = time.time()
        if a.status:
            rows = rows_fn()
            print(json.dumps({"now_utc": iso(now), "seed2": seed2_off(rows=rows) or "off: entries disabled, no process, no session",
                              "room_for_three": S2.room(slots, rows, 3) or "yes", "balance": S2.balance(),
                              "min_balance": MIN_BALANCE, "free_gb": free_gb(), "guard_running": guard_running(),
                              "guard_stop_asked": guard_stop_asked(),
                              "launch": load_record(os.path.join(LAUNCH_DIR, "launch.json")),
                              "launched": sorted(n for n, x in ARMS.items() if S2.launched(x["state_dir"])),
                              "home": sorted(n for n, x in ARMS.items() if ended_home(x["state_dir"]))}, indent=1))
            return 0
        if a.guard:
            kind, what = decide_guard(now, a.commit, rows_fn=rows_fn)
            folder = GUARD["dir"]
        else:
            kind, what = decide_arm(a.name, a.commit, now, slots, rows_fn, S2.balance)
            folder = ARMS[a.name]["state_dir"]
        if kind == "exec":
            print(f"{iso(now)} ladder_launch: starting {' '.join(what[1:])}", flush=True)
            os.execv(what[0], what)
        if kind == "halt":
            hkind, why = what
            p = os.path.join(folder, "HALTED")
            if not os.path.exists(p):
                S2.write_file(p, S2.halt_record(hkind, why, now))
            print(f"{iso(now)} ladder_launch: {why}: HALTED ({hkind}) written, nothing runs", flush=True)
            return 0
        if kind == "refuse":
            print(f"{iso(now)} ladder_launch: refused: {what}", flush=True)
            return 2
        if what != said:
            print(f"{iso(now)} ladder_launch: waiting: {what}", flush=True)
            said = what
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
