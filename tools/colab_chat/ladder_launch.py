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

SEED 2 (PROTOCOL 48.3) IS NEVER TOUCHED. Its folders, watchdog entries, processes and launch record are read only.
"Seed 2 is off" means ALL of:
  S1. each seed-2 arm's entry in scripts/colab_driver.json (found once by its state folder) reads "enabled": false;
  S2. no process carries a seed-2 arm's --name or --state-dir;
  S3. seed 2's units-guard entry (found once by its state folder) reads "enabled": false, and no process carries
      --campaign base2-seed2 (the waiting gate and the guard itself both do);
  S4. Colab's listing answers and lists no seed-2 session.
Anything else is "seed 2 is live". Closing seed 2 and its guard is its lead's step, never the ladder's.

WHILE SEED 2 IS LIVE the two campaigns draw on one unit balance, and seed 2's guard counts every unit the account spends.
Its stop comes when the balance falls to its starting balance less its bound (both read from seed 2's launch record:
463.19 - 420 = 43.19 units). The ladder then runs under these limits:
  O1. slot 1 only (make_ladder.SLOTS: N1B, then S1B, then NT): one L4 at a time, beside seed 2's two, so the account's
      three-L4 limit holds; the listing must show room for one;
  O2. at each start, the balance less seed 2's stop balance must cover both campaigns to seed 2's own deadline with
      OVERLAP_MARGIN to spare: seed 2's two sessions and the ladder's one, at L4_UNITS_PER_HOUR each, for every hour
      left to that deadline. Seed 2 cannot spend past its deadline, so this bounds what the pair can take while it lives;
  O3. the ladder's units guard does NOT run (it would count seed 2's spend as the ladder's and stop the ladder on it).
      Until seed 2 is off the ladder's units are bounded by wall time: one session, each runner's --max-hours and the
      ladder's deadline.
Slots 2 and 3 (N2B, S2B) and the guard start only once seed 2 is off.

Admission, once, by a slot's first arm (it writes LAUNCH_DIR/launch.json through a hard link):
  A1. LAUNCH_DIR/GO.json exists and validates (the lane's launch step writes it after its own refusals: Jonathan's quoted
      line, the reviews file and its ruling, this commit);
  A2. no folder of the ladder has launched;
  A3. the arm's own start conditions below;
  A4. the unit balance covers the ladder's expected units (EXPECT_TOTAL) and 30, on top of what is left of seed 2's own
      cap while seed 2 is live (seed2_cap_left: its registered cap less what the account has spent since its admission,
      the purchase of 17 Nov 2026 counted; the ladder has spent nothing at admission, so that spend is seed 2's).
The record holds the admission time, the ladder's one deadline (admission plus WEEK_HOURS), the balance, the commit and the
bounds. Every arm's state folder is created then, so the guard's stop reaches arms that have not started.

An arm that has not launched starts when: the arm before it in its slot is home with its last release made; the listing has
room for one L4; C: has at least MIN_FREE_GB free; its expected hours fit before the deadline; and, for slot 1 while seed 2
is live, O2; otherwise (seed 2 off) the ladder's guard is running. It halts instead (HALTED, no session made) when a stop
has been asked of the ladder, the arm before it holds HALTED, or its hours no longer fit (LADDER_NO_TIME). A launched arm
(state.json "launches" at least 1) that holds HALTED is never run by the gate (a person reads it). Otherwise a launched arm
runs its runner at once with the recorded deadline: that resumes or adopts the session it already has and asks for no new
slot, and a pending STOP_REQUEST is taken by that runner, whose confirmed stop is the only thing that ends a live session.

Every decision of an arm that has not launched is made under one lock (LAUNCH_DIR/gate.lock), so two gates never admit or
take a slot on the same reading. An arm about to start writes START_PENDING in its folder; the other gates count it as an L4
already taken until its session is in Colab's listing or the arm has ended (home, or HALTED). It never lapses by age: a start
that hangs is restarted by the watchdog and still owns its slot, and one that is truly stuck is a person's to reconcile. Each gate holds its arm's own lock (state folder, gate.lock) for its whole life and
passes it to the runner, so an arm never has two gates or two runners.

The cap also holds with the guard blind. Slot 1 can run at most WEEK_HOURS (the deadline), and N2B and S2B at most their
runners' 50 hours each: 268 L4-hours, 520 units, under CAP_UNITS. The guard is the earlier stop, not the only one.

The guard is base2_units_guard.py, unchanged, as its own instance: campaign "token-ladder", its own folder and log, the
ladder's five state folders. It starts once seed 2 is off, with the balance read at that moment as its start, and a bound
of GUARD_STOP_AT less what one L4 can have spent since admission (L4_UNITS_PER_HOUR an hour: the ladder's overlap spend,
counted high). Both are written once to GUARD/guard_start.json and reused on every restart. Units bought while it runs
would hide spending from it (its "spent" is the fall of the balance): none are bought during the ladder.
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
SEED2_LAUNCH = os.path.join(S2.LAUNCH_DIR, "launch.json")
WEEK_HOURS = 168                 # the whole ladder ends within seven days of admission
L4_UNITS_PER_HOUR = 1.94         # one L4 session (base 2's measured rate; seed 2's guard logs 3.88 for two)
EXPECT_HOURS = {"1B": 18.5, "2B": 38.0, "T": 114.0}     # at about 3,300 steps an hour on an L4 (base 2's measured rate)
EXPECT_UNITS = {"1B": 41, "2B": 79, "T": 226}           # training at 1.94 units an hour, and about 5 for the three rung jobs
MAX_HOURS = {"1B": 26, "2B": 50, "T": 140}              # each runner's own limit: the expectation and a fifth, rounded up
EXPECT_TOTAL = sum(EXPECT_UNITS[L.RUNGS[a]["rung"]] for a in L.LAUNCHED_ARMS)     # 466
CAP_UNITS = 540.0                # the pretraining campaign's cap, in Colab units (about USD 54)
GUARD_STOP_AT = 520.0            # the cap less 20 for stopping the sessions
SEED2_PURCHASED = 500.0          # units bought while seed 2 ran (PROTOCOL 48.3, 17 Nov 2026): its guard does not see them
OVERLAP_MARGIN = 100.0
MIN_FREE_GB = 8                  # an arm brings about 2 GB home
POLL_S = 300
OWN_COPY = "ladder_launch.json"
GO_LINE = ("Jonathan, direct chat, 17 Nov 2026: \"If you think it's worth it, you make all the decisions. I am happy perhaps "
           "to wait a week maximum if you really want. just make your own decisions.\"")


def _arms():
    out = {}
    for arm in L.LAUNCHED_ARMS:
        r = L.RUNGS[arm]
        slot = L.SLOTS[r["slot"]]
        i = slot.index(arm)
        out[f"pk-tl-{arm}"] = {"arm": arm, "rung": r["rung"], "slot": r["slot"], "state_dir": f"/home/jonat/ladder_{arm}_runner",
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


def guard_argv(start_balance, stop_at):
    return [sys.executable, "-u", "tools/colab_chat/base2_units_guard.py", "--campaign", GUARD["campaign"],
            "--stop-at", f"{stop_at:g}", "--start-balance", repr(float(start_balance)), "--every-min", "30",
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
    if not finite(rec.get("start_balance")):
        return "no finite start balance"
    if rec.get("cap_units") != CAP_UNITS or rec.get("guard_stop_at") != GUARD_STOP_AT or rec.get("week_hours") != WEEK_HOURS:
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


def write_once(path, rec):
    """`rec` at `path` unless a file is already there: a hard link of a finished temporary file, so a second writer meets
    the first's file. Returns what is then in place (parsed, or None)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{time.monotonic_ns()}.tmp"
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
    return read_json(path)


# ---------------------------------------------------------------- the checks (each: None when it holds, else why)

def scan(proc="/proc"):
    """[argv] of every process, or None when the process table cannot be listed (an unknown is never an empty table)."""
    try:
        if not any(p.isdigit() for p in os.listdir(proc)):
            return None
    except OSError:
        return None
    return S2.process_cmdlines(proc)


def pending_others(name, arms, now, rows):
    """How many other arms hold START_PENDING, are not in the listing and have not ended: L4s being taken. A reservation is
    kept until the session is seen or the arm has ended (home or HALTED); neither its age nor a recorded launch frees it."""
    listed = {n for n, _ in (rows or [])}
    n = 0
    for other, x in arms.items():
        if other == name or other in listed:
            continue
        d = x["state_dir"]
        if not os.path.lexists(os.path.join(d, "START_PENDING")):
            continue
        if ended_home(d) or os.path.exists(os.path.join(d, "HALTED")):
            continue
        n += 1
    return n


def _has_pair(argv, flag, value):
    return any(x == flag and argv[i + 1] == value for i, x in enumerate(argv[:-1])) or f"{flag}={value}" in argv


def seed2_off(path=None, cmdlines=None, rows=None, seed2_arms=None, seed2_guard=None):
    """S1 to S4: None when seed 2 is off, else why it counts as live. `rows` is one answered Colab listing or None."""
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
    argvs = (cmdlines or scan)()
    if argvs is None:
        return "the process table could not be read"
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


def overlap_problem(balance, now, seed2_launch=None):
    """O2, asked while seed 2 is live: None when the balance, less the balance at which seed 2's guard stops seed 2, covers
    seed 2's two sessions and the ladder's one for every hour left to seed 2's deadline, with OVERLAP_MARGIN to spare."""
    rec = read_json(SEED2_LAUNCH if seed2_launch is None else seed2_launch)
    start, stop_at, dl = (rec or {}).get("start_balance"), (rec or {}).get("guard_stop_at"), parse_iso((rec or {}).get("deadline_utc"))
    if not finite(start) or not finite(stop_at) or dl is None:
        return "seed 2's launch record could not be read: its guard's stop balance is unknown"
    if not finite(balance):
        return "the unit balance could not be read"
    hours = max(0.0, (dl - now) / 3600.0)
    room = balance - (start - stop_at)
    need = 3 * L4_UNITS_PER_HOUR * hours + OVERLAP_MARGIN
    if room < need:
        return (f"the balance {balance:.1f} is {room:.1f} above seed 2's stop balance {start - stop_at:.1f}; both campaigns to "
                f"seed 2's deadline ({hours:.1f} h, three L4s) and the margin need {need:.1f}")
    return None


def seed2_cap_left(balance, seed2_launch=None):
    """What is left of seed 2's registered cap, in units, or None when it cannot be read. Read before the ladder has spent
    anything, so the account's spend since seed 2's admission is seed 2's own."""
    rec = read_json(SEED2_LAUNCH if seed2_launch is None else seed2_launch)
    start, cap = (rec or {}).get("start_balance"), (rec or {}).get("min_balance")
    if not finite(start) or not finite(cap) or not finite(balance):
        return None
    return max(0.0, cap - (start + SEED2_PURCHASED - balance))


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
               free=None, arms=None, overlap=None, cap_left=None, clock=time.time):
    """(kind, what): ("exec", argv) | ("wait", why) | ("halt", (kind, why)) | ("refuse", why)."""
    launch_dir = LAUNCH_DIR if launch_dir is None else launch_dir
    arms = ARMS if arms is None else arms
    a = arms[name]
    own = os.path.join(a["state_dir"], OWN_COPY)
    path = os.path.join(launch_dir, "launch.json")
    if S2.launched(a["state_dir"]):
        if os.path.exists(os.path.join(a["state_dir"], "HALTED")):
            return "halt", ("LADDER_HALTED", "this arm holds HALTED: the gate runs nothing; a person reads it")
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
    before = a["before"]
    if before is not None:
        if os.path.exists(os.path.join(arms[before]["state_dir"], "HALTED")):
            return "halt", ("LADDER_BEFORE_HALTED", f"{before}, the arm before this one in its slot, holds HALTED: a person reads it")
        if not ended_home(arms[before]["state_dir"]):
            return "wait", f"{before}, the arm before this one in its slot, is not home with its last release made"
    if state == "absent" and before is not None:
        return "wait", "the ladder has not been admitted (a slot's first arm admits it)"
    if state == "valid" and now + EXPECT_HOURS[a["rung"]] * 3600 > parse_iso(rec["deadline_utc"]):
        return "halt", ("LADDER_NO_TIME", f"{EXPECT_HOURS[a['rung']]} expected hours no longer fit before {rec['deadline_utc']}")
    f = (free or free_gb)()
    if f is None or f < MIN_FREE_GB:
        return "wait", f"C: has {f if f is None else round(f, 1)} GB free, under {MIN_FREE_GB}"
    rows = rows_fn()
    live = (seed2 or seed2_off)(rows=rows)
    b = None
    if live:
        if a["slot"] != 1:
            return "wait", f"seed 2 is live ({live}): only slot 1 runs beside it"
        b = bal_fn()
        why = (overlap or overlap_problem)(b, now)
        if why:
            return "wait", why
    elif state == "valid" and not (guard_up or guard_running)():
        return "wait", "seed 2 is off and the ladder's units guard is not running yet"
    why = S2.room(slots, rows, 1 + pending_others(name, arms, now, rows))
    if why:
        return "wait", why
    if state == "absent":
        if any(S2.launched(x["state_dir"]) for x in arms.values()):
            return "refuse", "a folder of the ladder has launched with no launch.json: a person reads it"
        b = bal_fn() if b is None else b
        if not finite(b):
            return "wait", "the unit balance could not be read"
        left = (cap_left or seed2_cap_left)(b) if live else 0.0
        if left is None:
            return "wait", "what is left of seed 2's cap could not be read"
        need = EXPECT_TOTAL + 30 + left
        if b < need:
            return "wait", f"the unit balance {b:.2f} is under {need:.1f} (the ladder's expected units and 30" + (f", and {left:.1f} left of seed 2's cap)" if live else ")")
        at = int(clock())
        got = write_once(path, {"at_utc": iso(at), "deadline_utc": iso(at + WEEK_HOURS * 3600), "start_balance": b,
                                "first": name, "commit": commit, "cap_units": CAP_UNITS, "guard_stop_at": GUARD_STOP_AT,
                                "week_hours": WEEK_HOURS, "seed2_live_at_admission": bool(live)})
        state, rec = load_record(path, commit)
        if state != "valid" or got is None:
            return "refuse", f"launch.json does not validate after the admission ({rec})"
        if rec.get("first") != name or rec.get("at_utc") != iso(at):
            return "wait", "another arm admitted the ladder first: this arm starts as an admitted arm at its next check"
        for x in arms.values():
            os.makedirs(x["state_dir"], exist_ok=True)
    os.makedirs(a["state_dir"], exist_ok=True)
    S2.write_file(own, rec)
    S2.write_file(os.path.join(a["state_dir"], "START_PENDING"), {"at": now, "at_utc": iso(now), "name": name})
    return "exec", runner_argv(name, commit, rec["deadline_utc"])


def decide_guard(now, commit, bal_fn, launch_dir=None, seed2=None, rows_fn=None, guard_dir=None, clock=time.time):
    launch_dir = LAUNCH_DIR if launch_dir is None else launch_dir
    guard_dir = GUARD["dir"] if guard_dir is None else guard_dir
    why = go_problem(read_json(os.path.join(launch_dir, "GO.json")), commit)
    if why:
        return "refuse", why
    state, rec = load_record(os.path.join(launch_dir, "launch.json"), commit)
    if state == "invalid":
        return "refuse", f"launch.json does not validate ({rec}); it is never replaced: a person reads it"
    if state == "absent":
        return "wait", "the ladder has not been admitted"
    gs_path = os.path.join(guard_dir, "guard_start.json")
    gs = read_json(gs_path)
    if gs is None and os.path.lexists(gs_path):
        return "refuse", "guard_start.json is unreadable; it is never replaced: a person reads it"
    why = (seed2 or seed2_off)(rows=rows_fn() if rows_fn else None)      # before EVERY start of the guard, a restart too
    if why:
        return "wait", f"seed 2 is live ({why}): the ladder's guard would count its units"
    if gs is None:
        b = bal_fn()
        if not finite(b):
            return "wait", "the unit balance could not be read"
        at = int(clock())
        hours = max(0.0, (at - parse_iso(rec["at_utc"])) / 3600.0)
        gs = write_once(gs_path, {"at_utc": iso(at), "admitted_utc": rec["at_utc"], "commit": commit, "start_balance": b,
                                  "hours_since_admission": round(hours, 3),
                                  "overlap_units_counted": round(L4_UNITS_PER_HOUR * hours, 2),
                                  "stop_at": round(GUARD_STOP_AT - L4_UNITS_PER_HOUR * hours, 2)})
    why = guard_start_problem(gs, rec, commit, now)
    if why:
        return "refuse", f"guard_start.json is not this admission's record ({why}): a person reads it"
    if gs["stop_at"] <= 0:
        return "halt", ("LADDER_UNITS_STOP", "the ladder's cap was already counted as spent when its guard could start")
    return "exec", guard_argv(gs["start_balance"], gs["stop_at"])


def guard_start_problem(gs, rec, commit, now=None):
    """None when gs is the guard record of this admission with the registered bound, recomputed; else why."""
    if not isinstance(gs, dict) or not finite(gs.get("start_balance")) or not finite(gs.get("stop_at")):
        return "no start balance or bound"
    at, adm = parse_iso(gs.get("at_utc")), parse_iso(rec["at_utc"])
    if at is None or gs.get("admitted_utc") != rec["at_utc"] or gs.get("commit") != commit:
        return "not written for this admission and commit"
    if at > (time.time() if now is None else now) + 60:
        return "written in the future"
    if not (adm <= at <= parse_iso(rec["deadline_utc"])):
        return "written outside the ladder's week"
    if abs(gs["stop_at"] - round(GUARD_STOP_AT - L4_UNITS_PER_HOUR * (at - adm) / 3600.0, 2)) > 0.011:
        return "the bound is not the registered one less the overlap"
    return None


def take_lock(path, wait):
    """An exclusive lock on `path`, kept by the returned file object; None if it is held and `wait` is false."""
    import fcntl
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except OSError:
        f.close()
        return None
    return f


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
    if a.name or a.guard:
        mine = take_lock(os.path.join(ARMS[a.name]["state_dir"] if a.name else GUARD["dir"], "gate.lock"), wait=False)
        if mine is None:
            print(f"{iso(time.time())} ladder_launch: another gate or runner holds this folder's lock: exiting", flush=True)
            return 0
        os.set_inheritable(mine.fileno(), True)      # the runner (exec) keeps it: one gate or runner a folder
    while True:
        now = time.time()
        if a.status:
            rows = rows_fn()
            b = S2.balance()
            live = seed2_off(rows=rows)
            print(json.dumps({"now_utc": iso(now), "seed2": live or "off: entries disabled, no process, no session",
                              "overlap_rule": (overlap_problem(b, now) or "holds") if live else "not asked: seed 2 is off",
                              "room_for_one_l4": S2.room(slots, rows, 1) or "yes", "balance": b,
                              "admission_needs": (None if seed2_cap_left(b) is None else EXPECT_TOTAL + 30 + seed2_cap_left(b)) if live else EXPECT_TOTAL + 30,
                              "free_gb": free_gb(), "guard_running": guard_running(), "guard_stop_asked": guard_stop_asked(),
                              "launch": load_record(os.path.join(LAUNCH_DIR, "launch.json")),
                              "launched": sorted(n for n, x in ARMS.items() if S2.launched(x["state_dir"])),
                              "home": sorted(n for n, x in ARMS.items() if ended_home(x["state_dir"]))}, indent=1))
            return 0
        if a.guard:
            kind, what = decide_guard(now, a.commit, S2.balance, rows_fn=rows_fn)
            folder = GUARD["dir"]
        else:
            shared = take_lock(os.path.join(LAUNCH_DIR, "gate.lock"), wait=True)     # one decision at a time, all arms
            try:
                kind, what = decide_arm(a.name, a.commit, time.time(), slots, rows_fn, S2.balance)
            finally:
                shared.close()      # START_PENDING, written under the lock, holds the slot from here
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
