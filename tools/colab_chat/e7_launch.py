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
    if LL.ended_home(a["state_dir"]):
        return "halt", ("E7_HOME", "the arm is home with its last release made: nothing runs again")
    if _started(a):
        # A1 (GPT review, point 1): E9 live is re-read at every start. A session of the arm that may be up is handed to the
        # runner (it resumes it, or finishes a stop); a REPLACEMENT session, a new allocation, is never made while E9 is live.
        why = (e9 or e9_live)()
        if why:
            rows = rows_fn() if rows_fn else None
            listed = rows is None or any(n == name for n, _ in rows)
            if not listed:
                return "halt", ("E7_AFTER_E9", f"{why}: no replacement session for E7 while E9 is live; the lead reads it")
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


# ---------------------------------------------------------------- A1: the enforced preflight and the GO record (GPT, point 2)
RESEARCH = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/base2/e7_shape"
REVIEWS = "base2/e7_shape/reviews/reviews_answered.md"
HF_ACCOUNT_GB, PUSH_CAP_GB, HF_MAX_AGE_S = 1000.0, 200.0, 3 * 86400
# The trainer, runner, boot and every file the arm runs that is not E7's own: identical to the ladder's launch commit
PARITY_EXCLUDE = (":(exclude)tools/colab_chat", ":(exclude)tests")
PARITY_FILES = ("tools/colab_chat/runner_multi.py", "tools/colab_chat/boot_base2.sh", "tools/colab_chat/tiny_chat_run.py",
                "tools/colab_chat/base2_pins.py", "tools/colab_chat/base2_pins.json", "tools/colab_chat/base2_units_guard.py",
                "tools/colab_chat/base2_rung_D1.json", "tools/colab_chat/ladder_launch.py", "tools/colab_chat/make_ladder.py",
                "tools/colab_chat/make_base2.py", "tools/colab_chat/ladder_N1B.json", "tools/colab_chat/base2_seed2_launch.py")


def _git(*args):
    import subprocess
    p = subprocess.run(["git", *args], cwd=S2.REPO, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def preflight(commit, line, research=RESEARCH, now=None, git=None, ladder_launch_dir=None):
    """None when every registered launch precondition that is not the gate's own holds, else the list of failures."""
    import time as _t
    git = git or _git
    now = _t.time() if now is None else now
    bad = []
    if not (isinstance(line, str) and line.startswith("Jonathan, ") and len(line) >= 30):
        bad.append("the GO line is not Jonathan's dated line ('Jonathan, ...', at least 30 characters)")
    rc, head = git("rev-parse", "HEAD")
    if rc or head != commit:
        bad.append(f"the worktree is not at {commit}")
    rc, remote = git("branch", "-r", "--contains", commit)
    if rc or "origin/claude/e7-shape" not in remote:
        bad.append("the launch commit is not on origin/claude/e7-shape (the VM fetches it)")
    try:
        text = open(os.path.join(research, "reviews", "reviews_answered.md"), encoding="utf-8").read()
    except OSError:
        text = ""
    if not any(x.startswith("LAUNCH RULING: go (") for x in text.splitlines()):
        bad.append("reviews/reviews_answered.md holds no line beginning 'LAUNCH RULING: go ('")
    hf = LL.read_json(os.path.join(research, "checks", "hf_storage.json")) or {}
    at = LL.parse_iso(hf.get("at_utc"))
    if not (LL.finite(hf.get("used_gb")) and at is not None and hf.get("who") and now - at <= HF_MAX_AGE_S):
        bad.append("checks/hf_storage.json does not record the account's used storage within three days")
    elif HF_ACCOUNT_GB - float(hf["used_gb"]) < PUSH_CAP_GB:
        bad.append(f"the HF account has under {PUSH_CAP_GB:g} GB free for the arm's push cap")
    rec = LL.read_json(os.path.join(ladder_launch_dir or E9.LADDER_LAUNCH_DIR, "launch.json")) or {}
    lc = rec.get("commit")
    if not (isinstance(lc, str) and len(lc) == 40):
        bad.append("the ladder's launch.json names no launch commit to hold the trainer to")
    else:
        rc, _ = git("diff", "--quiet", lc, commit, "--", ".", *PARITY_EXCLUDE)
        rc2, _ = git("diff", "--quiet", lc, commit, "--", *PARITY_FILES)
        if rc or rc2:
            bad.append(f"the trainer, runner or a reused file differs from the ladder's launch commit {lc[:8]}")
    return bad or None


def write_go(commit, line, research=RESEARCH, launch_dir=None, **kw):
    bad = preflight(commit, line, research, **kw)
    if bad:
        return bad
    d = E7_LAUNCH_DIR if launch_dir is None else launch_dir
    p = os.path.join(d, "GO.json")
    if os.path.lexists(p):
        return ["GO.json already exists: it is never replaced"]
    import time as _t
    rec = {"line": line, "commit": commit, "reviews": REVIEWS, "at_utc": LL.iso(int(_t.time())), "preflight": "passed"}
    got = LL.write_once(p, rec)
    return None if got == rec else ["GO.json was written by another process first: a person reads it"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    install()
    if "--preflight" in argv or "--go" in argv:
        commit = argv[argv.index("--commit") + 1] if "--commit" in argv else ""
        line = argv[argv.index("--go") + 1] if "--go" in argv else "Jonathan, (preflight only, no line given here)"
        bad = write_go(commit, line) if "--go" in argv else preflight(commit, line)
        print("PREFLIGHT_OK" if not bad else "REFUSED:\n- " + "\n- ".join(bad), flush=True)
        return 0 if not bad else 2
    return LL.main(argv)


if __name__ == "__main__":
    sys.exit(main())
