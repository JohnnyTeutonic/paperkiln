#!/usr/bin/env python3
"""The start of base 2's second seed (PROTOCOL.md section 48.3): arms N2 and S2, and their units guard, each started by
the watchdog through this gate, which waits until section 48's arms have ended and then runs the registered command.

    python3 -u tools/colab_chat/base2_seed2_launch.py --name pk-b2s2-N2 --commit SHA    # arm N2's watchdog entry
    python3 -u tools/colab_chat/base2_seed2_launch.py --name pk-b2s2-S2 --commit SHA    # arm S2's
    python3 -u tools/colab_chat/base2_seed2_launch.py --guard --campaign base2-seed2 --commit SHA   # their guard
    python3 tools/colab_chat/base2_seed2_launch.py --status                            # one check, printed; nothing runs

Why a gate. The three live arms (N, S and C, section 48) hold Colab's three L4s and 108 of the store repository's 128
commits an hour until they end, and section 48's units guard counts every unit the account spends. So N2 and S2 start only
when N, S and C have ended, and then take N's and S's places: the same repository, token, switches, release turns and
store code, at the trainer's seed 50 (make_base2.SECOND_SEED).

ADMISSION, the pair's first start, needs all of these at one check (a failed or unreadable check is a wait, said once):
  1. the time, read again just before the record is written, is before WINDOW_END (2026-11-21T00:00:00Z). Past it with no
     admission, each folder gets HALTED (SEED2_WINDOW_CLOSED; no session was made), which the watchdog's stop_file reads;
  2. each of section 48's arm folders has ended: state.json reads campaign_home and store_release_final_done, and holds no
     STOP_UNRESOLVED or OWNERSHIP_MISMATCH; or the lead has written LAUNCH_DIR/CLEARED_<arm> for an arm that ended any
     other way, once its stop is confirmed, its storage released by hand, and it will not be restarted;
  3. Colab's listing answers and holds no session of section 48 (pk-b2-N, pk-b2-S, pk-b2-C, pk-b2-data, pk-b2-data-r2);
  4. section 48's units guard is off: its entry in scripts/colab_driver.json (by its exact name, SECTION48_GUARD_NAME) is
     present with "enabled": false, AND no process runs `base2_units_guard.py --campaign base2` (disabling an entry does
     not end its process; left running, it would count this section's units as section 48's);
  5. the listing has room for an L4 for each arm of this section (scripts/colab_slots.py's rule);
  6. Colab's unit balance (ccu_read.py, as the units guard reads it) is at least MIN_BALANCE;
  7. no folder of this section has launched (a launched folder with no admission record is a person's to read).
It then writes LAUNCH_DIR/launch.json once (a hard link of a finished file, so two gates cannot both write it): the
admission time, the deadline (admission plus MAX_HOURS, one literal for both arms), the balance read, the commit and the
bounds. Before it runs its runner, each arm copies the record into its own folder (seed2_launch.json).

After admission, an arm that has not launched needs checks 3 and 5 for itself, and halts instead (HALTED, no session made):
at or after the recorded deadline (SEED2_DEADLINE_BEFORE_LAUNCH), or PARTNER_LATE_S after admission
(SEED2_PARTNER_LATE), in which case it also writes STOP_REQUEST into its partner's folder if the partner has launched, so
one arm never runs alone for nothing. An arm whose folder has launched (state.json "launches" at least 1) runs its runner
at once on every start, with the deadline from its own copy; the runner's own restart rules apply.

A launch record that exists but does not validate (its fields, its commit, the registered bounds) is never replaced:
a gate that meets one refuses (exit 2) with the reason, except that a launched arm runs on its own valid copy.

The guard waits for admission and check 4, then runs base2_units_guard.py for N2's and S2's folders with the admission's
balance as its start and GUARD_STOP_AT as its bound. While it waits its command line names --campaign base2-seed2 and not
the guard's file name; after its exec it does carry `base2_units_guard.py`, section 48's entry's match, which is why check 4
requires that entry disabled (it must stay disabled while this section's guard runs).

Each gate is one process that checks every POLL_S seconds and then replaces itself (exec) with the registered command, so
the watchdog's match ("--name pk-b2s2-N2", "--name pk-b2s2-S2", "--campaign base2-seed2") holds while it waits and after.
The scripts it reads outside the launch commit (scripts/colab_slots.py, scripts/colab_driver.json, ccu_read.py) are read
live. Claude Opus 5.5, 13 Nov 2026.
"""
from __future__ import annotations

import argparse
import calendar
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import base2_units_guard as guard_mod  # noqa: E402  the balance as the units guard reads it

RESEARCH = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
DRIVER_JSON = f"{RESEARCH}/scripts/colab_driver.json"
SLOTS_PY = f"{RESEARCH}/scripts/colab_slots.py"
LAUNCH_DIR = "/home/jonat/base2_seed2_launch"
WINDOW_END = calendar.timegm((2026, 11, 21, 0, 0, 0))      # no admission at or after 2026-11-21T00:00:00Z
MAX_HOURS = 120
PARTNER_LATE_S = 24 * 3600   # an arm not launched a day after admission halts, and stops its partner
MIN_BALANCE = 440.0     # the cap of section 48.3, in units: the balance admission needs
GUARD_STOP_AT = 420.0   # the cap less 20 for stopping the sessions
POLL_S = 300
STORE_REPO = "johnnyteutonic/base2-store"
TOKEN_FILE = "/home/jonat/.hf_base2_token"
SECTION48_ARMS = {"N": "/home/jonat/base2_arm_N_runner", "S": "/home/jonat/base2_arm_S_runner",
                  "C": "/home/jonat/base2_arm_C_runner"}
SECTION48_SESSIONS = ("pk-b2-N", "pk-b2-S", "pk-b2-C", "pk-b2-data", "pk-b2-data-r2")
SECTION48_GUARD_NAME = "base2 units guard (PROTOCOL 48; stop at 620 units spent)"   # its entry's exact name
SECTION48_GUARD_CAMPAIGN = "base2"
ARMS = {"pk-b2s2-N2": {"arm": "N2", "state_dir": "/home/jonat/base2_seed2_N2_runner",
                       "art": "/mnt/c/ml_artifacts/base2/arm_N2", "campaign": "base2-N2"},
        "pk-b2s2-S2": {"arm": "S2", "state_dir": "/home/jonat/base2_seed2_S2_runner",
                       "art": "/mnt/c/ml_artifacts/base2/arm_S2", "campaign": "base2-S2"}}
GUARD = {"campaign": "base2-seed2", "dir": "/home/jonat/base2_seed2_units_guard"}
OWN_COPY = "seed2_launch.json"


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_iso(s):
    try:
        return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def runner_argv(name, commit, deadline_utc):
    """Section 48's arm command with N2's or S2's names (48.3): nothing else differs."""
    a = ARMS[name]
    return [sys.executable, "-u", "tools/colab_chat/runner_multi.py", "--commit", commit,
            "--jobs", f"tools/colab_chat/base2_{a['arm']}.json", "--name", name, "--art", a["art"],
            "--state-dir", a["state_dir"], "--gpu", "L4", "--max-hours", str(MAX_HOURS), "--deadline-utc", deadline_utc,
            "--ckpt-store", "hf", "--boot-script", "tools/colab_chat/boot_base2.sh", "--store-repo", STORE_REPO,
            "--store-token-file", TOKEN_FILE, "--store-campaign", a["campaign"], "--store-first-start",
            "--store-release-storage"]


def guard_argv(start_balance):
    return [sys.executable, "-u", "tools/colab_chat/base2_units_guard.py", "--campaign", GUARD["campaign"],
            "--stop-at", f"{GUARD_STOP_AT:g}", "--start-balance", repr(float(start_balance)), "--every-min", "30",
            "--log", os.path.join(GUARD["dir"], "guard.jsonl"),
            *[x for a in ARMS.values() for x in ("--state-dir", a["state_dir"])]]


# ---------------------------------------------------------------- the launch record

def record_problem(rec, commit=None):
    """None when rec is a valid admission record (for this commit, when given), else why not."""
    if not isinstance(rec, dict):
        return "not a JSON object"
    at, dl = parse_iso(rec.get("at_utc")), parse_iso(rec.get("deadline_utc"))
    if at is None or dl is None:
        return "admission time or deadline missing or not in the form YYYY-MM-DDTHH:MM:SSZ"
    if at >= WINDOW_END:
        return "admitted at or after the window's end"
    if dl != at + MAX_HOURS * 3600:
        return "the deadline is not the admission time plus MAX_HOURS"
    b = rec.get("start_balance")
    if not finite(b) or b < MIN_BALANCE:
        return "no finite start balance at least MIN_BALANCE"
    if rec.get("min_balance") != MIN_BALANCE or rec.get("guard_stop_at") != GUARD_STOP_AT or rec.get("max_hours") != MAX_HOURS:
        return "the bounds recorded are not the registered ones"
    c = rec.get("commit")
    if not (isinstance(c, str) and len(c) == 40):
        return "no full commit"
    if commit is not None and c != commit:
        return f"recorded for commit {c}, not {commit}"
    return None


def load_record(path, commit=None):
    """("absent", None) | ("valid", rec) | ("invalid", why)."""
    if not os.path.lexists(path):
        return "absent", None
    rec = read_json(path)
    why = record_problem(rec, commit) if rec is not None else "unreadable or not a JSON object"
    return ("invalid", why) if why else ("valid", rec)


def write_launch(rec, launch_dir=LAUNCH_DIR):
    """launch.json, once: a hard link of a finished temporary file, so a second writer meets the first's file. Returns
    load_record's reading of what is then in place."""
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


def write_file(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- the checks (each: None when it holds, else why)

def section48_ended(launch_dir=LAUNCH_DIR, arms=SECTION48_ARMS):
    for arm, d in arms.items():
        if os.path.exists(os.path.join(launch_dir, f"CLEARED_{arm}")):
            continue
        for bad in ("STOP_UNRESOLVED", "OWNERSHIP_MISMATCH"):
            if os.path.exists(os.path.join(d, bad)):
                return f"section 48's arm {arm} holds {bad}"
        st = read_json(os.path.join(d, "state.json"))
        if st is None:
            return f"section 48's arm {arm}: state.json unreadable"
        if st.get("campaign_home") is not True or st.get("store_release_final_done") is not True:
            return f"section 48's arm {arm} has not ended with its last release made"
    return None


def process_cmdlines(proc="/proc"):
    """[argv] of every readable process."""
    out = []
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return out
    for p in pids:
        try:
            with open(os.path.join(proc, p, "cmdline"), "rb") as f:
                raw = f.read()
        except OSError:
            continue
        if raw:
            out.append([x.decode("utf-8", "replace") for x in raw.rstrip(b"\0").split(b"\0")])
    return out


def runs_guard_of(argvs, campaign):
    """Whether any argv is base2_units_guard.py with --campaign exactly `campaign`."""
    for argv in argvs:
        if any(x.endswith("base2_units_guard.py") for x in argv):
            for i, x in enumerate(argv[:-1]):
                if x == "--campaign" and argv[i + 1] == campaign:
                    return True
            if f"--campaign={campaign}" in argv:
                return True
    return False


def guard48_off(path=DRIVER_JSON, cmdlines=process_cmdlines):
    d = read_json(path)
    drivers = (d or {}).get("drivers")
    entries = [e for e in drivers if isinstance(e, dict) and e.get("name") == SECTION48_GUARD_NAME] \
        if isinstance(drivers, list) else []
    if len(entries) != 1:
        return "section 48's units guard entry not found once by its name in scripts/colab_driver.json"
    if entries[0].get("enabled") is not False:
        return "section 48's units guard entry is still enabled"
    if runs_guard_of(cmdlines(), SECTION48_GUARD_CAMPAIGN):
        return "section 48's units guard process is still running (its entry is disabled; end the process)"
    return None


def load_slots(path=SLOTS_PY):
    spec = importlib.util.spec_from_file_location("colab_slots", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def listing_rows(slots):
    """[(name, hardware)] from one answered listing, or None."""
    if slots is None:
        return None
    try:
        return slots.parse(slots.listing())
    except slots.Unreadable:
        return None


def no_section48_session(rows):
    if rows is None:
        return "Colab's listing did not answer (or scripts/colab_slots.py could not be loaded)"
    held = sorted(n for n, _ in rows if n in SECTION48_SESSIONS)
    return f"Colab lists section 48's sessions {held}" if held else None


def launched(state_dir):
    st = read_json(os.path.join(state_dir, "state.json"))
    n = (st or {}).get("launches")
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


def room(slots, rows, n):
    if rows is None or slots is None:
        return "Colab's listing did not answer"
    if n <= 0:
        return None
    return None if slots.room(rows, "l4", n) else f"no room for {n} more L4 ({slots.summary(rows)})"


def balance(read=None):
    r = (read or guard_mod.read_ccu)()
    b = r.get("currentBalance")
    return b if finite(b) else None


def halt_record(kind, why, now):
    return {"kind": kind, "at": now, "at_utc": iso(now), "stop_confirmed": True, "why": why}


# ---------------------------------------------------------------- the decisions

def decide_arm(name, commit, now, slots, rows_fn, bal_fn, launch_dir=LAUNCH_DIR, s48=None, g48=None, clock=time.time):
    """(kind, what): ("exec", argv) | ("wait", why) | ("halt", (kind, why)) | ("stop_partner_and_halt", (partner state
    dir, kind, why)) | ("refuse", why). rows_fn and bal_fn are called only when needed; clock is read again just before
    the admission record is written."""
    a = ARMS[name]
    own = os.path.join(a["state_dir"], OWN_COPY)
    path = os.path.join(launch_dir, "launch.json")
    if launched(a["state_dir"]):
        state, rec = load_record(own, commit)
        if state != "valid":
            state, rec = load_record(path, commit)
            if state != "valid":
                return "refuse", f"this folder has launched and has no valid launch record ({rec or 'absent'}): a person reads it"
        return "exec", runner_argv(name, commit, rec["deadline_utc"])
    state, rec = load_record(path, commit)
    if state == "invalid":
        return "refuse", f"launch.json does not validate ({rec}); it is never replaced: a person reads it"
    if state == "absent":
        if now >= WINDOW_END:
            return "halt", ("SEED2_WINDOW_CLOSED", "the start window closed with no admission; no session was made")
        if any(launched(x["state_dir"]) for x in ARMS.values()):
            return "refuse", "a folder of this section has launched with no launch.json: a person reads it"
        why = (s48 or section48_ended)(launch_dir) or (g48 or guard48_off)()
        if why:
            return "wait", why
        rows = rows_fn()
        why = no_section48_session(rows) or room(slots, rows, len(ARMS))
        if why:
            return "wait", why
        b = bal_fn()
        if b is None:
            return "wait", "the unit balance could not be read"
        if b < MIN_BALANCE:
            return "wait", f"the unit balance {b:.2f} is under {MIN_BALANCE:g}"
        at = int(clock())
        if at >= WINDOW_END:
            return "halt", ("SEED2_WINDOW_CLOSED", "the start window closed during the admission check")
        state, rec = write_launch({"at_utc": iso(at), "deadline_utc": iso(at + MAX_HOURS * 3600), "start_balance": b,
                                   "first": name, "commit": commit, "min_balance": MIN_BALANCE,
                                   "guard_stop_at": GUARD_STOP_AT, "max_hours": MAX_HOURS,
                                   "checks": "section 48 ended; its guard entry disabled and its process gone; no section 48 "
                                             "session listed; L4 room for both arms; balance at least the cap"}, launch_dir)
        if state != "valid":
            return "refuse", f"launch.json does not validate after the admission ({rec})"
        write_file(own, rec)
        return "exec", runner_argv(name, commit, rec["deadline_utc"])
    deadline, admitted = parse_iso(rec["deadline_utc"]), parse_iso(rec["at_utc"])
    if now >= deadline:
        return "halt", ("SEED2_DEADLINE_BEFORE_LAUNCH", "the section's deadline passed before this arm launched")
    if now >= admitted + PARTNER_LATE_S:
        partner = [x["state_dir"] for n, x in ARMS.items() if n != name]
        return "stop_partner_and_halt", (partner[0], "SEED2_PARTNER_LATE",
                                         "this arm had not launched a day after admission; its partner is stopped")
    rows = rows_fn()
    why = no_section48_session(rows) or room(slots, rows, 1)
    if why:
        return "wait", why
    write_file(own, rec)
    return "exec", runner_argv(name, commit, rec["deadline_utc"])


def decide_guard(now, launch_dir=LAUNCH_DIR, g48=None, commit=None):
    state, rec = load_record(os.path.join(launch_dir, "launch.json"), commit)
    if state == "invalid":
        return "refuse", f"launch.json does not validate ({rec}); it is never replaced: a person reads it"
    if state == "absent":
        if now >= WINDOW_END:
            return "halt", ("SEED2_WINDOW_CLOSED", "the start window closed with no admission")
        return "wait", "the pair has not been admitted"
    why = (g48 or guard48_off)()
    if why:
        return "wait", why
    return "exec", guard_argv(rec["start_balance"])


def head_commit():
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=60)
    return p.stdout.strip() if p.returncode == 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", choices=sorted(ARMS))
    ap.add_argument("--commit")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--campaign")
    ap.add_argument("--status", action="store_true", help="one check of admission's conditions, printed")
    a = ap.parse_args(argv)
    if a.guard and a.campaign != GUARD["campaign"]:
        ap.error(f"--guard runs with --campaign {GUARD['campaign']}")
    if not (a.guard or a.status) and not a.name:
        ap.error("--name, --guard or --status")
    if (a.name or a.guard) and (not a.commit or len(a.commit) != 40):
        ap.error("--commit is the full 40-character launch SHA")
    if (a.name or a.guard) and head_commit() != a.commit:
        sys.exit(f"this worktree is not at {a.commit}; the gate runs only from the launch commit")
    os.chdir(REPO)
    try:
        slots = load_slots()
    except Exception as e:  # noqa: BLE001  a gate that cannot count slots waits; it never takes them as free
        print(f"{iso(time.time())} base2_seed2_launch: scripts/colab_slots.py could not be loaded ({type(e).__name__})",
              flush=True)
        slots = None
    said = None
    while True:
        now = time.time()
        if a.status:
            rows = listing_rows(slots)
            out = {"now_utc": iso(now), "window_end_utc": iso(WINDOW_END),
                   "section48_ended": section48_ended() or "yes", "section48_guard": guard48_off() or "off",
                   "section48_sessions": no_section48_session(rows) or "none listed",
                   "room_for_two": room(slots, rows, 2) or "yes", "balance": balance(),
                   "launch": load_record(os.path.join(LAUNCH_DIR, "launch.json"))}
            print(json.dumps(out, indent=1))
            return 0
        if a.guard:
            kind, what = decide_guard(now, commit=a.commit)
            folder = GUARD["dir"]
        else:
            kind, what = decide_arm(a.name, a.commit, now, slots, lambda: listing_rows(slots), balance)
            folder = ARMS[a.name]["state_dir"]
        if kind == "exec":
            print(f"{iso(now)} base2_seed2_launch: starting {' '.join(what[1:])}", flush=True)
            os.execv(what[0], what)
        if kind == "stop_partner_and_halt":
            partner, hkind, why = what
            if launched(partner) and not os.path.exists(os.path.join(partner, "STOP_REQUEST.taken")):
                with open(os.path.join(partner, "STOP_REQUEST"), "w", encoding="utf-8") as f:
                    f.write(f"base2_seed2_launch: {a.name} had not launched a day after admission (section 48.3)\n")
            kind, what = "halt", (hkind, why)
        if kind == "halt":
            hkind, why = what
            p = os.path.join(folder, "HALTED")
            if not os.path.exists(p):
                write_file(p, halt_record(hkind, why, now))
            print(f"{iso(now)} base2_seed2_launch: {why}: HALTED ({hkind}) written, nothing runs", flush=True)
            return 0
        if kind == "refuse":
            print(f"{iso(now)} base2_seed2_launch: refused: {what}", flush=True)
            return 2
        if what != said:
            print(f"{iso(now)} base2_seed2_launch: waiting: {what}", flush=True)
            said = what
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
