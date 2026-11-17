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

SEED 2 FIRST, AND NO OVERLAP (PROTOCOL 48.3: no other Colab use of the lane during its section). Nothing of the ladder
starts, and the ladder is not admitted, until seed 2 is off. Seed 2's folders, entries, processes and records are read only.
"Seed 2 is off" is a positive reading, ALL of:
  S1. its section has ended, by one of: both arms' state.json read campaign_home and store_release_final_done; or the time
      is an hour past the deadline in seed 2's own launch record (nothing of it can run after that); or the lead has written
      LAUNCH_DIR/SEED2_ENDED.json (who, when, why) to say the section is closed. A HALTED arm is not an end: it may be resumed;
  S2. the process table can be read and no process carries a seed-2 arm's --name or --state-dir;
  S3. Colab's listing answers and lists no seed-2 session.
It does not depend on seed 2's watchdog entries, which may be enabled, disabled or removed.

ADMISSION, once, by a slot's first arm (it writes LAUNCH_DIR/launch.json through a hard link): GO.json validates (only the
lane's launch_pt.sh --go writes it, after its refusals); seed 2 is off; no folder of the ladder has launched; the arm's own
start conditions hold. The record holds the admission time, the ladder's one deadline (admission plus WEEK_HOURS: the week
runs from the ladder's own first start), the balance, the commit and the bounds. Every arm's state folder is created then,
so the guard's stop reaches arms that have not started.

AN ARM'S START (never launched before). All of: seed 2 off; the arm before it in its slot home with its last release made;
the listing shows room for one L4 beyond the starts already reserved; C: has MIN_FREE_GB free; its whole hour limit and
CHAT_HOURS for its chat campaign fit before the ladder's deadline (else HALTED, LADDER_NO_TIME); the balance covers the worst
case of everything not yet finished, this arm included, and UNITS_MARGIN (units_needed; else it waits: no units are bought for
the ladder); and, for every arm but the admitting one, the ladder's guard is running. At its first start the arm writes ONCE,
in its own folder, arm_deadline_utc = the earlier of the ladder's deadline and that moment plus its hour limit.

EVERY start of an arm's runner, the first and each restart, passes that same recorded arm deadline as --deadline-utc. The
runner takes the earlier of it and now plus --max-hours, so a restart can never extend an arm. After its arm deadline an arm is
never run again (HALTED, LADDER_ARM_DEADLINE). The cap therefore holds by wall time alone, whether or not the guard can read
the balance: the arms' hour limits sum to WORST_HOURS L4-hours, WORST_UNITS units.

A launched arm (state.json "launches" at least 1, or a session of its own name in the listing) that holds HALTED is not run
by the gate; its watchdog entry names HALTED as its stop file, so the watchdog disables it for a person to read. Otherwise a
launched arm inside its deadline runs its runner at once: that resumes or adopts the session it has and asks for no new slot,
and a pending STOP_REQUEST is taken by that runner, whose confirmed stop ends the session.

Every decision is made under one lock (LAUNCH_DIR/gate.lock). An arm about to start writes START_PENDING; other gates count
it as an L4 taken until its session is listed or the arm has ended. Each gate holds its folder's own lock for its life and
the runner inherits it: one gate or runner a folder.

THE GUARD is base2_units_guard.py, unchanged, as its own instance: campaign "token-ladder", its own folder and log, the
ladder's five state folders, started once the ladder is admitted (seed 2 is then off, and is asked again at every start of
the guard). Its start balance is the admission's and its bound GUARD_STOP_AT; both are in the admission record. It is the
earlier stop; the wall-time bound above is the one that does not depend on reading the balance.
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
SEED2_GRACE_S = 3600             # seed 2's section counts as ended an hour past its own deadline
WEEK_HOURS = 168                 # the whole ladder ends within seven days of its own first start
L4_UNITS_PER_HOUR = 1.94         # one L4 session (base 2's measured rate)
EXPECT_HOURS = {"1B": 18.5, "2B": 38.0, "T": 98.5}      # at about 3,300 steps an hour on an L4 (base 2's measured rate)
# Each arm's hour limit: its expectation and 40% (base 2 saw 15 to 40% over), rounded up. An arm never runs longer.
MAX_HOURS = {"1B": 26, "2B": 54, "T": 138}
CHAT_HOURS = 12                  # what an arm's chat campaign needs after it, inside the week
WORST_HOURS = sum(MAX_HOURS[L.RUNGS[a]["rung"]] for a in L.LAUNCHED_ARMS)             # 298
WORST_UNITS = round(WORST_HOURS * L4_UNITS_PER_HOUR, 1)                                 # 578.1
CAP_UNITS = 580.0                # the pretraining campaign's cap: the wall-time worst case, rounded up (about USD 58)
GUARD_STOP_AT = 560.0            # the guard's earlier stop: the cap less 20 for stopping the sessions
UNITS_MARGIN = 20.0
MIN_FREE_GB = 8                  # an arm brings about 2 GB home
POLL_S = 300
OWN_COPY = "ladder_launch.json"


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
    """Base 2's arm command (base2_seed2_launch.runner_argv) with the ladder's names, jobs file and hours. deadline_utc is
    the ARM's recorded deadline, the same at every start."""
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
    """GO.json is written only by the lane's launch_pt.sh --go, which holds the line to the registration's form."""
    if not isinstance(rec, dict):
        return "GO.json is absent or unreadable"
    line = rec.get("line")
    if not (isinstance(line, str) and line.startswith("Jonathan, ") and len(line) >= 30):
        return "GO.json does not hold Jonathan's dated line"
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


def own_problem(own, rec, name):
    """None when `own` is this arm's first-start record under admission `rec`: the admission's fields and an arm deadline no
    later than the ladder's and no later than the first start plus the arm's hour limit."""
    if not isinstance(own, dict):
        return "absent or unreadable"
    first, arm_dl = parse_iso(own.get("first_start_utc")), parse_iso(own.get("arm_deadline_utc"))
    if first is None or arm_dl is None or own.get("name") != name:
        return "no first start, no arm deadline, or another arm's record"
    if own.get("at_utc") != rec["at_utc"] or own.get("deadline_utc") != rec["deadline_utc"] or own.get("commit") != rec["commit"]:
        return "not written under this admission"
    limit = MAX_HOURS[ARMS[name]["rung"]] * 3600
    if arm_dl != min(parse_iso(rec["deadline_utc"]), first + limit) or first < parse_iso(rec["at_utc"]):
        return "the arm deadline is not the earlier of the ladder's deadline and the first start plus the hour limit"
    return None


# ---------------------------------------------------------------- the checks (each: None when it holds, else why)

def _has_pair(argv, flag, value):
    return any(x == flag and argv[i + 1] == value for i, x in enumerate(argv[:-1])) or f"{flag}={value}" in argv


def scan(proc="/proc"):
    """[argv] of every process, or None when the process table cannot be listed (an unknown is never an empty table)."""
    try:
        if not any(p.isdigit() for p in os.listdir(proc)):
            return None
    except OSError:
        return None
    return S2.process_cmdlines(proc)


def ended_home(state_dir):
    st = read_json(os.path.join(state_dir, "state.json"))
    return bool(st) and st.get("campaign_home") is True and st.get("store_release_final_done") is True


def seed2_section_ended(now, launch_dir=None, seed2_arms=None, seed2_launch=None):
    """S1: how seed 2's section is known to have ended ("home" | "deadline" | "lead"), or None."""
    arms = S2.ARMS if seed2_arms is None else seed2_arms
    if all(ended_home(a["state_dir"]) for a in arms.values()):
        return "home"
    dl = parse_iso((read_json(SEED2_LAUNCH if seed2_launch is None else seed2_launch) or {}).get("deadline_utc"))
    if dl is not None and now >= dl + SEED2_GRACE_S:
        return "deadline"
    lead = read_json(os.path.join(LAUNCH_DIR if launch_dir is None else launch_dir, "SEED2_ENDED.json"))
    if lead and all(isinstance(lead.get(k), str) and lead[k] for k in ("by", "why")) and parse_iso(lead.get("at_utc")) is not None:
        return "lead"
    return None


def seed2_off(now=None, cmdlines=None, rows=None, seed2_arms=None, launch_dir=None, seed2_launch=None):
    """S1 to S3: None when seed 2 is off, else why it counts as live. `rows` is one answered Colab listing or None."""
    now = time.time() if now is None else now
    arms = S2.ARMS if seed2_arms is None else seed2_arms
    if seed2_section_ended(now, launch_dir, arms, seed2_launch) is None:
        return ("seed 2's section has not ended (its arms are not both home with the last release made, its deadline and an "
                "hour have not passed, and the lead has written no SEED2_ENDED.json)")
    argvs = (cmdlines or scan)()
    if argvs is None:
        return "the process table could not be read"
    for argv in argvs:
        for n, a in arms.items():
            if _has_pair(argv, "--name", n) or _has_pair(argv, "--state-dir", a["state_dir"]):
                return f"a process of seed 2's arm {n} is still running"
    if rows is None:
        return "Colab's listing did not answer"
    held = sorted(n for n, _ in rows if n in arms)
    if held:
        return f"Colab lists seed 2's sessions {held}"
    return None


def pending_others(name, arms, rows):
    """How many other arms hold START_PENDING, are not in the listing and have not ended: L4s being taken. A reservation is
    kept until the session is seen or the arm has ended (home or HALTED); neither its age nor a recorded launch frees it."""
    listed = {n for n, _ in (rows or [])}
    n = 0
    for other, x in arms.items():
        d = x["state_dir"]
        if other == name or other in listed or not os.path.lexists(os.path.join(d, "START_PENDING")):
            continue
        if ended_home(d) or os.path.exists(os.path.join(d, "HALTED")):
            continue
        n += 1
    return n


def units_needed(name, arms, now, rec):
    """The worst case, in units, of everything of the ladder not yet finished, `name` included: an arm that has started can
    still run to its recorded arm deadline; one that has not, its whole hour limit (or to the ladder's deadline if sooner).
    The four low arms come first: the top rung, while it has not started, is counted only in its own start (and in --status,
    name None), so a balance that cannot carry it leaves it waiting and never holds the low arms back."""
    week = parse_iso(rec["deadline_utc"]) if rec else now + WEEK_HOURS * 3600
    hours = 0.0
    for other, x in arms.items():
        d = x["state_dir"]
        if ended_home(d) or os.path.exists(os.path.join(d, "HALTED")):
            continue
        own = read_json(os.path.join(d, OWN_COPY))
        arm_dl = parse_iso((own or {}).get("arm_deadline_utc")) if other != name else None
        if arm_dl is None and x["rung"] == "T" and name not in (other, None):
            continue
        if arm_dl is not None:
            hours += max(0.0, (arm_dl - now) / 3600.0)
        else:
            hours += max(0.0, min(MAX_HOURS[x["rung"]], (week - now) / 3600.0))
    return hours * L4_UNITS_PER_HOUR + UNITS_MARGIN


def guard_running(cmdlines=None):
    argvs = (cmdlines or scan)()
    return bool(argvs) and any(any(x.endswith("base2_units_guard.py") for x in argv) and _has_pair(argv, "--campaign", GUARD["campaign"])
                               for argv in argvs)


def guard_stop_asked(log=None):
    """True when the ladder's guard has logged a stop request. A line that cannot be parsed is skipped, as the guard itself
    skips it (a torn last line after a crash is not a stop); a log that cannot be opened counts as asked."""
    log = os.path.join(GUARD["dir"], "guard.jsonl") if log is None else log
    if not os.path.exists(log):
        return False
    try:
        with open(log, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return True
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("stop_requested"):
            return True
    return False


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
    own_path = os.path.join(a["state_dir"], OWN_COPY)
    path = os.path.join(launch_dir, "launch.json")
    rows = rows_fn()
    listed = rows is not None and any(n == name for n, _ in rows)
    if S2.launched(a["state_dir"]) or listed:
        # a session of this arm exists or has existed: no new slot is asked; the recorded arm deadline is passed again
        if os.path.exists(os.path.join(a["state_dir"], "HALTED")):
            return "halt", ("LADDER_HALTED", "this arm holds HALTED: the gate runs nothing; a person reads it")
        state, rec = load_record(path, commit)
        own = read_json(own_path)
        why = rec if state != "valid" else own_problem(own, rec, name)
        if state != "valid" or why:
            return "refuse", f"this arm has a session or has launched, and has no valid admission and first-start record ({why or 'absent'}): a person reads it"
        if now >= parse_iso(own["arm_deadline_utc"]):
            return "halt", ("LADDER_ARM_DEADLINE", f"this arm's deadline {own['arm_deadline_utc']} has passed: it is not run again"
                            + ("; A SESSION OF ITS NAME IS STILL LISTED: a person stops it" if listed else ""))
        return "exec", runner_argv(name, commit, own["arm_deadline_utc"])
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
    limit = MAX_HOURS[a["rung"]]
    if state == "valid" and now + (limit + CHAT_HOURS) * 3600 > parse_iso(rec["deadline_utc"]):
        return "halt", ("LADDER_NO_TIME", f"this arm's {limit} hours and {CHAT_HOURS} for its chat campaign no longer fit before {rec['deadline_utc']}")
    why = (seed2 or seed2_off)(now=now, rows=rows)
    if why:
        return "wait", f"seed 2 is live ({why}): nothing of the ladder starts beside it"
    f = (free or free_gb)()
    if f is None or f < MIN_FREE_GB:
        return "wait", f"C: has {f if f is None else round(f, 1)} GB free, under {MIN_FREE_GB}"
    if state == "valid" and not (guard_up or guard_running)():
        return "wait", "the ladder's units guard is not running yet"
    why = S2.room(slots, rows, 1 + pending_others(name, arms, rows))
    if why:
        return "wait", why
    b = bal_fn()
    if not finite(b):
        return "wait", "the unit balance could not be read"
    need = units_needed(name, arms, now, rec if state == "valid" else None)
    if b < need:
        return "wait", (f"the unit balance {b:.1f} is under {need:.1f}: the worst case of everything not yet finished, this arm "
                        "included, and the margin (no units are bought for the ladder)")
    if state == "absent":
        if any(S2.launched(x["state_dir"]) for x in arms.values()):
            return "refuse", "a folder of the ladder has launched with no launch.json: a person reads it"
        at = int(clock())
        got = write_once(path, {"at_utc": iso(at), "deadline_utc": iso(at + WEEK_HOURS * 3600), "start_balance": b,
                                "first": name, "commit": commit, "cap_units": CAP_UNITS, "guard_stop_at": GUARD_STOP_AT,
                                "week_hours": WEEK_HOURS, "worst_hours": WORST_HOURS, "worst_units": WORST_UNITS})
        state, rec = load_record(path, commit)
        if state != "valid" or got is None:
            return "refuse", f"launch.json does not validate after the admission ({rec})"
        if rec.get("first") != name or rec.get("at_utc") != iso(at):
            return "wait", "another arm admitted the ladder first: this arm starts as an admitted arm at its next check"
        for x in arms.values():
            os.makedirs(x["state_dir"], exist_ok=True)
    first = int(clock())
    arm_dl = min(parse_iso(rec["deadline_utc"]), first + limit * 3600)
    own = write_once(own_path, {**rec, "name": name, "first_start_utc": iso(first), "arm_deadline_utc": iso(arm_dl),
                                "max_hours": limit})
    why = own_problem(own, rec, name)
    if why:
        return "refuse", f"{OWN_COPY} in this arm's folder does not validate ({why}); it is never replaced: a person reads it"
    S2.write_file(os.path.join(a["state_dir"], "START_PENDING"), {"at": now, "at_utc": iso(now), "name": name})
    return "exec", runner_argv(name, commit, own["arm_deadline_utc"])


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
    why = (seed2 or seed2_off)(now=now, rows=rows_fn() if rows_fn else None)      # before EVERY start of the guard
    if why:
        return "wait", f"seed 2 is live ({why}): the ladder's guard would count its units"
    return "exec", guard_argv(rec["start_balance"])


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
            state, rec = load_record(os.path.join(LAUNCH_DIR, "launch.json"))
            print(json.dumps({"now_utc": iso(now), "seed2": seed2_off(now=now, rows=rows) or f"off ({seed2_section_ended(now)})",
                              "room_for_three_l4": S2.room(slots, rows, 3) or "yes", "balance": S2.balance(),
                              "worst_case_units_of_what_is_left_and_margin": round(units_needed(None, ARMS, now, rec if state == "valid" else None), 1),
                              "worst_hours": WORST_HOURS, "worst_units": WORST_UNITS, "cap_units": CAP_UNITS,
                              "free_gb": free_gb(), "guard_running": guard_running(), "guard_stop_asked": guard_stop_asked(),
                              "launch": [state, rec],
                              "arm_deadlines": {n: (read_json(os.path.join(x["state_dir"], OWN_COPY)) or {}).get("arm_deadline_utc") for n, x in ARMS.items()},
                              "launched": sorted(n for n, x in ARMS.items() if S2.launched(x["state_dir"])),
                              "home": sorted(n for n, x in ARMS.items() if ended_home(x["state_dir"]))}, indent=1))
            return 0
        shared = take_lock(os.path.join(LAUNCH_DIR, "gate.lock"), wait=True)     # one decision at a time, arms and guard
        try:
            if a.guard:
                kind, what = decide_guard(time.time(), a.commit, rows_fn=rows_fn)
                folder = GUARD["dir"]
            else:
                kind, what = decide_arm(a.name, a.commit, time.time(), slots, rows_fn, S2.balance)
                folder = ARMS[a.name]["state_dir"]
        finally:
            shared.close()      # START_PENDING, written under the lock, holds the slot from here
        if kind == "exec":
            print(f"{iso(now)} ladder_launch: starting {' '.join(what[1:])}", flush=True)
            os.execv(what[0], what)
        if kind == "halt":
            hkind, why = what
            p = os.path.join(folder, "HALTED")
            if not os.path.exists(p):
                S2.write_file(p, dict(S2.halt_record(hkind, why, now), stop_confirmed=None,
                                      note="written by the ladder's gate, which stops no session itself"))
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
