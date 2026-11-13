"""base2_seed2_launch.py, the gate that starts base 2's second seed (PROTOCOL.md section 48.3), and the seed-2 jobs files.
CPU only; no Colab call, no balance read, no store. Claude Opus 5.5, 13 Nov 2026."""
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "colab_chat"))
import base2_seed2_launch as L  # noqa: E402
import make_base2 as M  # noqa: E402

SHA = "a" * 40
# Section 48's arm N command as scripts/colab_driver.json holds it (PROTOCOL.md 48.2), with its deadline literal.
S48_N = ("python3 -u tools/colab_chat/runner_multi.py --commit 711160a2ba7edb6962fed00da7c91a3c5545ded1 "
         "--jobs tools/colab_chat/base2_N.json --name pk-b2-N --art /mnt/c/ml_artifacts/base2/arm_N "
         "--state-dir /home/jonat/base2_arm_N_runner --gpu L4 --max-hours 120 --deadline-utc 2026-11-17T19:48:06Z "
         "--ckpt-store hf --boot-script tools/colab_chat/boot_base2.sh --store-repo johnnyteutonic/base2-store "
         "--store-token-file /home/jonat/.hf_base2_token --store-campaign base2-N --store-first-start "
         "--store-release-storage")
S48_GUARD = ("python3 -u tools/colab_chat/base2_units_guard.py --campaign base2 --stop-at 620 --start-balance 949.73 "
             "--every-min 30 --log /home/jonat/base2_units_guard/guard.jsonl --state-dir /home/jonat/base2_data_runner "
             "--state-dir /home/jonat/base2_arm_N_runner --state-dir /home/jonat/base2_arm_S_runner "
             "--state-dir /home/jonat/base2_arm_C_runner --state-dir /home/jonat/base2_data_runner_r2")
S48_MATCHES = ["--name pk-b2-N", "--name pk-b2-S", "--name pk-b2-C", "--name pk-b2-data", "--name pk-b2-data-r2",
               "base2_units_guard.py"]
SEED2_MATCHES = ["--name pk-b2s2-N2", "--name pk-b2s2-S2", "--campaign base2-seed2"]


def cmdline(argv):
    return " ".join(["python3", *argv[1:]])


# ---------------------------------------------------------------- the commands

def test_the_arm_command_is_section_48s_with_the_second_seeds_names_only():
    for name, base in (("pk-b2s2-N2", "N"), ("pk-b2s2-S2", "S")):
        got = cmdline(L.runner_argv(name, "711160a2ba7edb6962fed00da7c91a3c5545ded1", "2026-11-17T19:48:06Z"))
        a = L.ARMS[name]
        want = (S48_N.replace("base2_N.json", f"base2_{a['arm']}.json").replace("--name pk-b2-N", f"--name {name}")
                .replace("/mnt/c/ml_artifacts/base2/arm_N", a["art"]).replace("/home/jonat/base2_arm_N_runner", a["state_dir"])
                .replace("--store-campaign base2-N", f"--store-campaign {a['campaign']}"))
        assert got == want, (name, base)


def test_the_guard_command_is_section_48s_for_the_second_seeds_folders():
    got = cmdline(L.guard_argv(512.25))
    assert got == ("python3 -u tools/colab_chat/base2_units_guard.py --campaign base2-seed2 --stop-at 420 "
                   "--start-balance 512.25 --every-min 30 --log /home/jonat/base2_seed2_units_guard/guard.jsonl "
                   "--state-dir /home/jonat/base2_seed2_N2_runner --state-dir /home/jonat/base2_seed2_S2_runner")
    assert L.MIN_BALANCE - L.GUARD_STOP_AT == 20


def test_no_watchdog_match_of_section_48_sees_a_second_seed_process_and_none_of_ours_sees_section_48s():
    gates = [f"python3 -u tools/colab_chat/base2_seed2_launch.py --name {n} --commit {SHA}" for n in L.ARMS]
    gates.append(f"python3 -u tools/colab_chat/base2_seed2_launch.py --guard --campaign base2-seed2 --commit {SHA}")
    runners = [cmdline(L.runner_argv(n, SHA, "2026-11-22T00:00:00Z")) for n in L.ARMS]
    for m in S48_MATCHES:
        for line in gates + runners:  # the waiting gates and the arms' runners: never taken for a section 48 process
            assert m not in line, (m, line)
    # The seed-2 guard after its exec does carry the guard's file name: it runs only once section 48's guard entry is
    # disabled (check 4), so that match is then no longer read.
    assert "base2_units_guard.py" in cmdline(L.guard_argv(500.0))
    s48 = [S48_N.replace("pk-b2-N", f"pk-b2-{x}").replace("arm_N", f"arm_{x}").replace("base2-N", f"base2-{x}")
           .replace("base2_N.json", f"base2_{x}.json") for x in "NSC"] + [S48_GUARD]
    for m in SEED2_MATCHES:
        for line in s48:
            assert m not in line, (m, line)
    for n in L.ARMS:  # each seed-2 entry's own match holds for its gate and for the runner it becomes
        assert f"--name {n}" in gates[list(L.ARMS).index(n)] and f"--name {n}" in runners[list(L.ARMS).index(n)]


def test_names_folders_and_campaigns_are_new():
    for name, a in L.ARMS.items():
        assert not name.startswith("pk-b2-") and a["campaign"] in ("base2-N2", "base2-S2")
        assert "base2_arm_" not in a["state_dir"] and a["state_dir"].startswith("/home/jonat/")
        for d in L.SECTION48_ARMS.values():
            assert a["state_dir"] != d


# ---------------------------------------------------------------- the checks

from pathlib import Path  # noqa: E402


class Slots:
    """colab_slots.py's rule, loaded from the real file when it is there, else its two functions as written."""
    def __init__(self):
        try:
            self.m = L.load_slots()
        except (OSError, FileNotFoundError):
            self.m = None

    def room(self, rows, kind, n=1):
        if self.m:
            return self.m.room(rows, kind, n)
        used, l4 = len(rows), sum(1 for _, h in rows if h != "CPU")
        return used + n <= 6 and (kind == "cpu" or l4 + n <= 3)

    def summary(self, rows):
        return f"{len(rows)} rows"


S48_GUARD_ARGV = S48_GUARD.split()
SEED2_GUARD_ARGV = ["python3", "-u", "tools/colab_chat/base2_units_guard.py", "--campaign", "base2-seed2", "--stop-at", "420"]


def ended(tmp, arms=("N", "S", "C"), home=True, released=True):
    dirs = {}
    for x in arms:
        d = tmp / f"s48_{x}"
        d.mkdir()
        (d / "state.json").write_text(json.dumps({"campaign_home": home, "store_release_final_done": released}))
        dirs[x] = str(d)
    return dirs


def driver(path, s48_enabled=False, extra=()):
    entries = [{"name": L.SECTION48_GUARD_NAME, "enabled": s48_enabled, "match": "base2_units_guard.py"}, *extra]
    path.write_text(json.dumps({"drivers": entries}))


@pytest.fixture
def world(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    arms = {}
    for name, a in L.ARMS.items():
        arms[name] = {**a, "state_dir": str(tmp_path / f"{a['arm']}_runner")}
    monkeypatch.setattr(L, "ARMS", arms)
    s48 = ended(tmp_path)
    drv = tmp_path / "colab_driver.json"
    driver(drv)
    w = {"launch": str(launch), "s48": s48, "driver": str(drv), "slots": Slots(), "rows": [], "bal": 500.0,
         "now": L.WINDOW_END - 4 * 86400, "procs": [["bash"], SEED2_GUARD_ARGV]}

    def decide(name, commit=SHA):
        return L.decide_arm(name, commit, w["now"], w["slots"], lambda: w["rows"], lambda: w["bal"],
                            launch_dir=w["launch"], s48=lambda ld: L.section48_ended(ld, w["s48"]),
                            g48=lambda: L.guard48_off(w["driver"], cmdlines=lambda: w["procs"]), clock=lambda: w["now"])
    w["decide"] = decide
    w["g48"] = lambda: L.guard48_off(w["driver"], cmdlines=lambda: w["procs"])
    return w


def launch_rec(w):
    return json.load(open(os.path.join(w["launch"], "launch.json")))


def mark_launched(name):
    d = L.ARMS[name]["state_dir"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"launches": 1}, f)


def admit(world, name="pk-b2s2-N2"):
    kind, argv = world["decide"](name)
    assert kind == "exec"
    mark_launched(name)
    return launch_rec(world)


def test_admission_writes_one_deadline_and_both_arms_run_with_it(world):
    kind, argv = world["decide"]("pk-b2s2-N2")
    assert kind == "exec"
    rec = launch_rec(world)
    assert rec["at_utc"] == L.iso(world["now"]) and rec["deadline_utc"] == L.iso(world["now"] + 120 * 3600)
    assert rec["start_balance"] == 500.0 and rec["commit"] == SHA and L.record_problem(rec, SHA) is None
    assert argv[argv.index("--deadline-utc") + 1] == rec["deadline_utc"]
    own = json.load(open(os.path.join(L.ARMS["pk-b2s2-N2"]["state_dir"], L.OWN_COPY)))
    assert own == rec
    mark_launched("pk-b2s2-N2")
    world["rows"] = [("pk-b2s2-N2", "L4")]
    world["now"] += 900   # S2 starts a quarter of an hour later, with N2's deadline
    kind, argv = world["decide"]("pk-b2s2-S2")
    assert kind == "exec" and argv[argv.index("--deadline-utc") + 1] == rec["deadline_utc"]
    assert launch_rec(world) == rec
    assert json.load(open(os.path.join(L.ARMS["pk-b2s2-S2"]["state_dir"], L.OWN_COPY))) == rec


def test_a_restart_runs_at_once_on_the_arms_own_copy_whatever_the_checks_read(world):
    rec = admit(world)
    world["rows"], world["bal"], world["now"] = None, None, L.WINDOW_END + 86400
    open(os.path.join(world["s48"]["N"], "STOP_UNRESOLVED"), "w").close()
    os.remove(os.path.join(world["launch"], "launch.json"))   # even with the shared record gone
    kind, argv = world["decide"]("pk-b2s2-N2")
    assert kind == "exec" and argv[argv.index("--deadline-utc") + 1] == rec["deadline_utc"]
    open(os.path.join(world["launch"], "launch.json"), "w").write("{broken")   # or broken
    assert world["decide"]("pk-b2s2-N2")[0] == "exec"


def test_a_launched_folder_with_no_valid_record_is_refused(world):
    mark_launched("pk-b2s2-N2")
    assert world["decide"]("pk-b2s2-N2")[0] == "refuse"
    # and the other arm does not take the admission path while one has launched
    assert world["decide"]("pk-b2s2-S2")[0] == "refuse"
    assert not os.path.exists(os.path.join(world["launch"], "launch.json"))


@pytest.mark.parametrize("damage", ["broken", "not_object", "commit", "deadline", "balance", "bounds", "after_window"])
def test_an_invalid_launch_record_is_never_replaced(world, damage):
    rec = admit(world)
    p = os.path.join(world["launch"], "launch.json")
    bad = dict(rec)
    if damage == "broken":
        text = "{"
    elif damage == "not_object":
        text = "[1]"
    else:
        if damage == "commit":
            bad["commit"] = "b" * 40
        elif damage == "deadline":
            bad["deadline_utc"] = L.iso(L.parse_iso(rec["deadline_utc"]) + 3600)
        elif damage == "balance":
            bad["start_balance"] = float("nan")
        elif damage == "bounds":
            bad["guard_stop_at"] = 500.0
        elif damage == "after_window":
            bad["at_utc"] = L.iso(L.WINDOW_END)
            bad["deadline_utc"] = L.iso(L.WINDOW_END + 120 * 3600)
        text = json.dumps(bad)
    open(p, "w").write(text)
    kind, why = world["decide"]("pk-b2s2-S2")
    assert kind == "refuse" and "never replaced" in why
    assert open(p).read() == text
    assert L.decide_guard(world["now"], world["launch"], world["g48"], SHA)[0] == "refuse"


@pytest.mark.parametrize("change, why", [
    ("home", "has not ended"), ("released", "has not ended"), ("unresolved", "STOP_UNRESOLVED"),
    ("ownership", "OWNERSHIP_MISMATCH"), ("no_state", "unreadable"), ("guard_on", "still enabled"),
    ("guard_missing", "not found"), ("guard_twice", "not found"), ("guard_process", "process is still running"),
    ("listing", "did not answer"), ("no_slots_module", "did not answer"), ("s48_listed", "section 48's sessions"),
    ("one_slot", "no room for 2"), ("balance_low", "under 440"), ("balance_unread", "could not be read")])
def test_each_condition_of_admission_holds_it_back(world, change, why):
    if change in ("home", "released"):
        st = {"campaign_home": change != "home", "store_release_final_done": change != "released"}
        open(os.path.join(world["s48"]["C"], "state.json"), "w").write(json.dumps(st))
    elif change == "unresolved":
        open(os.path.join(world["s48"]["S"], "STOP_UNRESOLVED"), "w").close()
    elif change == "ownership":
        open(os.path.join(world["s48"]["N"], "OWNERSHIP_MISMATCH"), "w").close()
    elif change == "no_state":
        os.remove(os.path.join(world["s48"]["N"], "state.json"))
    elif change == "guard_on":
        driver(Path(world["driver"]), s48_enabled=True)
    elif change == "guard_missing":
        open(world["driver"], "w").write(json.dumps({"drivers": []}))
    elif change == "guard_twice":
        driver(Path(world["driver"]), extra=[{"name": L.SECTION48_GUARD_NAME, "enabled": False}])
    elif change == "guard_process":
        world["procs"].append(S48_GUARD_ARGV)
    elif change == "listing":
        world["rows"] = None
    elif change == "no_slots_module":
        world["slots"] = None
    elif change == "s48_listed":
        world["rows"] = [("pk-b2-C", "L4")]
    elif change == "one_slot":
        world["rows"] = [("someone", "L4"), ("other", "L4")]
    elif change == "balance_low":
        world["bal"] = 439.99
    elif change == "balance_unread":
        world["bal"] = None
    kind, what = world["decide"]("pk-b2s2-N2")
    assert kind == "wait" and why in what, what
    assert not os.path.exists(os.path.join(world["launch"], "launch.json"))


def test_the_new_guard_entry_named_like_section_48s_does_not_hold_admission_back(world):
    from pathlib import Path
    driver(Path(world["driver"]), extra=[{"name": "base2 units guard (PROTOCOL 48.3; base2-seed2)", "enabled": True,
                                          "match": "--campaign base2-seed2"}])
    assert world["decide"]("pk-b2s2-N2")[0] == "exec"


def test_only_section_48s_guard_campaign_counts_as_its_process():
    assert L.runs_guard_of([S48_GUARD_ARGV], "base2")
    assert not L.runs_guard_of([SEED2_GUARD_ARGV], "base2")
    assert L.runs_guard_of([SEED2_GUARD_ARGV], "base2-seed2")
    assert not L.runs_guard_of([["python3", "x.py", "--campaign", "base2"]], "base2")
    assert L.runs_guard_of([["/usr/bin/python3", "-u", "/abs/tools/colab_chat/base2_units_guard.py", "--campaign=base2"]],
                           "base2")


def test_the_process_scan_reads_proc(tmp_path):
    for pid, argv in (("12", S48_GUARD_ARGV), ("13", ["sleep", "5"]), ("x", ["no"])):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    got = L.process_cmdlines(str(tmp_path))
    assert S48_GUARD_ARGV in got and ["sleep", "5"] in got and ["no"] not in got


def test_the_balance_bar_is_inclusive_and_cpu_runtimes_leave_room(world):
    world["bal"] = 440.0
    world["rows"] = [("cpu-a", "CPU"), ("cpu-b", "CPU"), ("one", "L4")]
    assert world["decide"]("pk-b2s2-N2")[0] == "exec"


def test_the_window_is_read_again_just_before_the_record(world):
    world["now"] = L.WINDOW_END - 1
    clock = iter([L.WINDOW_END])
    kind, what = L.decide_arm("pk-b2s2-N2", SHA, world["now"], world["slots"], lambda: [], lambda: 500.0,
                              launch_dir=world["launch"], s48=lambda ld: L.section48_ended(ld, world["s48"]),
                              g48=world["g48"], clock=lambda: next(clock))
    assert kind == "halt" and what[0] == "SEED2_WINDOW_CLOSED"
    assert not os.path.exists(os.path.join(world["launch"], "launch.json"))


def test_the_lead_clears_an_arm_that_ended_another_way(world):
    open(os.path.join(world["s48"]["C"], "state.json"), "w").write(json.dumps({"campaign_home": False}))
    assert world["decide"]("pk-b2s2-N2")[0] == "wait"
    open(os.path.join(world["launch"], "CLEARED_C"), "w").close()
    assert world["decide"]("pk-b2s2-N2")[0] == "exec"


def test_the_second_arm_needs_its_own_slot_and_no_section_48_session(world):
    admit(world)
    world["rows"] = [("pk-b2s2-N2", "L4"), ("x", "L4"), ("y", "L4")]
    assert world["decide"]("pk-b2s2-S2")[0] == "wait"
    world["rows"] = [("pk-b2s2-N2", "L4"), ("pk-b2-S", "L4")]
    assert world["decide"]("pk-b2s2-S2")[0] == "wait"
    world["rows"] = [("pk-b2s2-N2", "L4"), ("x", "L4")]
    world["bal"] = 1.0   # the balance is read once, at admission
    assert world["decide"]("pk-b2s2-S2")[0] == "exec"


def test_a_partner_a_day_late_halts_and_stops_the_first(world):
    rec = admit(world)
    world["rows"] = [("pk-b2s2-N2", "L4"), ("x", "L4"), ("y", "L4")]
    world["now"] = L.parse_iso(rec["at_utc"]) + 24 * 3600 - 1
    assert world["decide"]("pk-b2s2-S2")[0] == "wait"
    world["now"] += 1
    kind, what = world["decide"]("pk-b2s2-S2")
    assert kind == "stop_partner_and_halt" and what[0] == L.ARMS["pk-b2s2-N2"]["state_dir"] and what[1] == "SEED2_PARTNER_LATE"


def test_an_unlaunched_arm_at_the_deadline_halts_without_a_slot(world, monkeypatch):
    monkeypatch.setattr(L, "PARTNER_LATE_S", 10 ** 9)
    rec = admit(world)
    world["rows"] = None
    world["now"] = L.parse_iso(rec["deadline_utc"])
    kind, what = world["decide"]("pk-b2s2-S2")
    assert kind == "halt" and what[0] == "SEED2_DEADLINE_BEFORE_LAUNCH"


def test_past_the_window_with_no_admission_each_folder_halts(world):
    world["now"] = L.WINDOW_END
    kind, what = world["decide"]("pk-b2s2-N2")
    assert kind == "halt" and what[0] == "SEED2_WINDOW_CLOSED"
    assert L.decide_guard(L.WINDOW_END, world["launch"], world["g48"])[0] == "halt"
    rec = L.halt_record(*what, world["now"])
    assert rec["kind"] == "SEED2_WINDOW_CLOSED" and rec["stop_confirmed"] is True
    assert not os.path.exists(os.path.join(world["launch"], "launch.json"))


def test_the_window_and_bounds_are_the_registered_ones():
    assert L.iso(L.WINDOW_END) == "2026-11-21T00:00:00Z" and L.MAX_HOURS == 120 and L.MIN_BALANCE == 440
    assert L.GUARD_STOP_AT == 420 and L.PARTNER_LATE_S == 86400


def test_the_launch_record_is_written_once_also_when_two_gates_race(tmp_path):
    import threading
    base = {"at_utc": "2026-11-17T00:00:00Z", "deadline_utc": "2026-11-22T00:00:00Z", "start_balance": 450.0,
            "commit": SHA, "min_balance": 440.0, "guard_stop_at": 420.0, "max_hours": 120}
    out = []
    start = threading.Barrier(8)

    def go(i):
        start.wait()
        out.append(L.write_launch({**base, "first": f"gate{i}"}, str(tmp_path)))
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert os.listdir(tmp_path) == ["launch.json"]
    assert len({json.dumps(r) for r in out}) == 1 and out[0][0] == "valid"


def test_the_guard_waits_for_admission_and_for_section_48s_guard_to_be_off(world):
    assert L.decide_guard(world["now"], world["launch"], world["g48"])[0] == "wait"
    admit(world)
    world["procs"].append(S48_GUARD_ARGV)
    assert L.decide_guard(world["now"], world["launch"], world["g48"])[0] == "wait"
    world["procs"].pop()
    kind, argv = L.decide_guard(L.WINDOW_END + 1, world["launch"], world["g48"])   # after admission the window no longer matters
    assert kind == "exec" and argv[argv.index("--start-balance") + 1] == "500.0"


# ---------------------------------------------------------------- the jobs files

def normalise(doc, arm):
    """A seed-2 jobs file with its names and seed put back to its recipe's arm."""
    base = M.SEED2[arm]
    text = json.dumps(doc)
    text = re.sub(rf"(pt_base2_|base2_|ckpt-|rung_|BASE2_){arm}(?![A-Za-z0-9])", rf"\g<1>{base}", text)
    d = json.loads(text)
    d["_comment"] = M.arm(base)["_comment"]
    for k in ("recipe_of", "seed"):
        d["_base2"].pop(k)
    d["_base2"]["arm"] = base
    d["jobs"][1]["spec"]["train"]["seed"] = 1
    for j in d["jobs"][2:]:
        i = j["cmd"].index("--seed")
        assert j["cmd"][i + 1] == str(M.SECOND_SEED)
        j["cmd"][i + 1] = "1"
    return d


@pytest.mark.parametrize("arm", ["N2", "S2"])
def test_n2_and_s2_are_n_and_s_at_the_second_seed_and_nothing_else(arm):
    doc = M.arm(arm)
    assert doc["jobs"][1]["spec"]["train"]["seed"] == M.SECOND_SEED == 50
    assert doc["_base2"]["seed"] == 50 and doc["_base2"]["recipe_of"] == M.SEED2[arm]
    assert normalise(doc, arm) == M.arm(M.SEED2[arm])
    assert doc["jobs"][0]["cmd"][3] == f"arm-{M.SEED2[arm]}"   # the pins job checks the same inputs as N's or S's
    with open(os.path.join(os.path.dirname(HERE), "tools", "colab_chat", f"base2_{arm}.json"), encoding="utf-8") as f:
        assert json.load(f) == doc


def test_the_second_seed_meets_no_generator_stream_of_the_first():
    """nn::Llama gives block i's seven matrices the generator seeds seed + 100 (i + 1) + k, k = 1 to 7 (src/llama.cpp);
    at seed 2 most would be seed 1's streams in other roles."""
    src = open(os.path.join(os.path.dirname(HERE), "src", "llama.cpp"), encoding="utf-8").read()
    assert "seed + 100 * (unsigned)(i + 1)" in src and all(f"seed + {k})" in src for k in range(1, 8))
    assert len(M.block_streams(2) & M.block_streams(1)) == 6 * 17 + 6   # the case the review found
    assert not M.block_streams(M.SECOND_SEED) & M.block_streams(1)
    assert {M.SECOND_SEED + 11, M.SECOND_SEED + 14}.isdisjoint({12, 15})   # token and state embeddings


def test_the_rungs_read_the_same_shape_and_panels_at_the_second_seed():
    for arm in ("N2", "S2"):
        for j, k in zip(M.arm(arm)["jobs"][2:], M.arm(M.SEED2[arm])["jobs"][2:]):
            spec = lambda c: c[c.index("--spec") + 1]  # noqa: E731
            panels = lambda c: [c[i + 1] for i, x in enumerate(c) if x == "--panel"]  # noqa: E731
            assert spec(j["cmd"]) == spec(k["cmd"]) == M.SHAPE_D1 and panels(j["cmd"]) == panels(k["cmd"])
    assert M.rung_d1()["train"]["seed"] == 1   # the rung's shape spec is unchanged
