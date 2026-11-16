"""The token ladder's start gate (tools/colab_chat/ladder_launch.py). While seed 2 is live only slot 1 runs, under the
balance rule, and the ladder's guard does not; slots 2 and 3 and the guard wait for seed 2 to be off; the admission is
written once with the registered bounds; a slot's arms run one after another; a stop reaches arms that have not started.
Nothing here opens a session, reads a balance or touches a real folder. Claude Fable 5.2, 17 Nov 2026."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "colab_chat"))
import base2_seed2_launch as S2  # noqa: E402
import ladder_launch as G  # noqa: E402
import make_ladder as L  # noqa: E402

COMMIT = "a" * 40
NOW = 1_800_000_000
LIVE = "pk-b2s2-N2: its watchdog entry is still enabled"


class Slots:
    def __init__(self, free):
        self.free = free

    def room(self, rows, kind, n=1):
        return kind == "l4" and n <= self.free

    def summary(self, rows):
        return f"{self.free} free"


@pytest.fixture
def world(tmp_path):
    arms = {n: dict(a, state_dir=str(tmp_path / a["arm"])) for n, a in G.ARMS.items()}
    ld = tmp_path / "launch"
    ld.mkdir()
    (ld / "GO.json").write_text(json.dumps({"line": G.GO_LINE, "commit": COMMIT, "reviews": "reviews/x.md",
                                            "at_utc": S2.iso(NOW - 60)}))
    return {"arms": arms, "ld": str(ld), "tmp": tmp_path, "gd": str(tmp_path / "guard")}


def decide(world, name, seed2=None, free_l4=3, bal=900.0, guard_up=True, stop=False, free=20.0, now=NOW, overlap=None, left=380.0):
    return G.decide_arm(name, COMMIT, now, Slots(free_l4), lambda: [], lambda: bal, launch_dir=world["ld"],
                        seed2=lambda rows=None: seed2, guard_up=lambda: guard_up, stop_asked=lambda: stop,
                        free=lambda: free, arms=world["arms"], overlap=lambda b, t: overlap, cap_left=lambda b: left, clock=lambda: now)


def guard(world, seed2=None, bal=500.0, now=NOW):
    return G.decide_guard(now, COMMIT, lambda: bal, launch_dir=world["ld"], seed2=lambda rows=None: seed2,
                          guard_dir=world["gd"], clock=lambda: now)


def home(world, name, launches=1):
    d = world["arms"][name]["state_dir"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"launches": launches, "campaign_home": True, "store_release_final_done": True}, f)


# ---------------------------------------------------------------- seed 2

def drivers(tmp, n2=False, s2=False, guard=False, drop=None):
    es = [{"name": "n2", "enabled": n2, "state_dir": S2.ARMS["pk-b2s2-N2"]["state_dir"]},
          {"name": "s2", "enabled": s2, "state_dir": S2.ARMS["pk-b2s2-S2"]["state_dir"]},
          {"name": "g", "enabled": guard, "state_dir": S2.GUARD["dir"]}]
    es = [e for e in es if e["name"] != drop]
    p = tmp / "driver.json"
    p.write_text(json.dumps({"drivers": es}))
    return str(p)


def test_seed2_off_needs_all_three_entries_disabled_no_process_and_no_session(tmp_path):
    ok = dict(cmdlines=lambda: [["python3", "x.py"]], rows=[("pk-tl-N1B", "L4")])
    assert G.seed2_off(path=drivers(tmp_path), **ok) is None
    for kw in ({"n2": True}, {"s2": True}, {"guard": True}):
        assert "still enabled" in G.seed2_off(path=drivers(tmp_path, **kw), **ok)
    for drop in ("n2", "s2", "g"):
        assert "not found once" in G.seed2_off(path=drivers(tmp_path, drop=drop), **ok)
    assert "unreadable" in G.seed2_off(path=str(tmp_path / "none.json"), **ok)
    p = drivers(tmp_path)
    live = [["python3", "-u", "runner_multi.py", "--name", "pk-b2s2-S2"]]
    assert "pk-b2s2-S2" in G.seed2_off(path=p, cmdlines=lambda: live, rows=[])
    live = [["python3", "runner_multi.py", "--state-dir", S2.ARMS["pk-b2s2-N2"]["state_dir"]]]
    assert "pk-b2s2-N2" in G.seed2_off(path=p, cmdlines=lambda: live, rows=[])
    for g in (["python3", "base2_seed2_launch.py", "--guard", "--campaign", "base2-seed2"],
              ["python3", "base2_units_guard.py", "--campaign", "base2-seed2", "--stop-at", "420"]):
        assert "guard" in G.seed2_off(path=p, cmdlines=lambda g=g: [g], rows=[])
    assert "did not answer" in G.seed2_off(path=p, cmdlines=lambda: [], rows=None)
    assert "pk-b2s2-N2" in G.seed2_off(path=p, cmdlines=lambda: [], rows=[("pk-b2s2-N2", "L4")])


def test_the_gate_never_writes_to_seed2s_names():
    src = open(G.__file__, encoding="utf-8").read()
    assert "os.kill" not in src and "STOP_REQUEST\"), \"w\"" not in src and "[\"enabled\"] =" not in src
    for a in G.ARMS.values():
        assert "seed2" not in a["state_dir"] and a["state_dir"] not in [x["state_dir"] for x in S2.ARMS.values()]
        assert a["campaign"] not in [x["campaign"] for x in S2.ARMS.values()] and not a["campaign"].startswith("base2")
    assert G.GUARD["dir"] != S2.GUARD["dir"] and G.GUARD["campaign"] != S2.GUARD["campaign"]
    assert not set(G.ARMS) & set(S2.ARMS)
    assert not G.LAUNCH_DIR.startswith(S2.LAUNCH_DIR) and G.SEED2_LAUNCH == S2.LAUNCH_DIR + "/launch.json"
    assert set(G.ARMS) == {f"pk-tl-{a}" for a in ("N1B", "S1B", "NT", "N2B", "S2B")}


def seed2_record(tmp, start=463.19, stop_at=420.0, hours=109.0):
    p = tmp / "seed2_launch.json"
    p.write_text(json.dumps({"start_balance": start, "guard_stop_at": stop_at, "deadline_utc": S2.iso(NOW + int(hours * 3600))}))
    return str(p)


def test_the_overlap_rule_covers_both_campaigns_to_seed2s_deadline_with_the_margin(tmp_path):
    p = seed2_record(tmp_path)
    # the day's figures: 921.16 in balance, seed 2's guard stops at 463.19 - 420 = 43.19, 109 hours to its deadline
    assert G.overlap_problem(921.16, NOW, p) is None
    need = 3 * 1.94 * 109 + 100          # 734.4
    assert G.overlap_problem(43.19 + need + 0.01, NOW, p) is None
    assert "need" in G.overlap_problem(43.19 + need - 0.5, NOW, p)
    assert G.overlap_problem(422.77, NOW, p) is not None          # the balance before the purchase: no overlap
    assert G.overlap_problem(43.19 + 100.01, NOW + 109 * 3600, p) is None       # at seed 2's deadline only the margin is asked
    assert "could not be read" in G.overlap_problem(None, NOW, p)
    assert "unknown" in G.overlap_problem(921.16, NOW, str(tmp_path / "none.json"))
    (tmp_path / "bad.json").write_text(json.dumps({"start_balance": 463.19}))
    assert "unknown" in G.overlap_problem(921.16, NOW, str(tmp_path / "bad.json"))
    # what the pair can take while seed 2 lives (seed 2's two L4s and the ladder's one, to its deadline) leaves the margin
    assert 921.16 - 43.19 - 3 * 1.94 * 109 > G.OVERLAP_MARGIN
    # what is left of seed 2's cap: 440 less what the account spent since its admission, the purchase counted
    p2 = tmp_path / "s2.json"
    p2.write_text(json.dumps({"start_balance": 463.19, "min_balance": 440.0}))
    assert abs(G.seed2_cap_left(921.16, str(p2)) - (440 - 42.03)) < 1e-6
    assert G.seed2_cap_left(400.0, str(p2)) == 0.0 and G.seed2_cap_left(None, str(p2)) is None
    assert G.seed2_cap_left(921.16, str(tmp_path / "none.json")) is None
    assert 921.16 - G.seed2_cap_left(921.16, str(p2)) >= G.EXPECT_TOTAL + 30          # the day's balance admits the ladder


def test_while_seed2_is_live_only_slot_1_runs_and_only_under_the_overlap_rule(world):
    for name in ("pk-tl-N2B", "pk-tl-S2B"):
        kind, why = decide(world, name, seed2=LIVE)
        assert kind == "wait" and "only slot 1" in why
    assert decide(world, "pk-tl-N1B", seed2=LIVE, overlap="the balance is short")[0] == "wait"
    assert decide(world, "pk-tl-N1B", seed2=LIVE, free_l4=0)[0] == "wait"          # seed 2 holds two of the three L4s
    assert decide(world, "pk-tl-N1B", seed2=LIVE, bal=G.EXPECT_TOTAL + 30 + 380.0 - 1)[0] == "wait"
    assert decide(world, "pk-tl-N1B", seed2=LIVE, left=None)[0] == "wait"
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))
    kind, argv = decide(world, "pk-tl-N1B", seed2=LIVE, guard_up=False)            # no ladder guard while seed 2 is live
    assert kind == "exec"
    rec = json.load(open(os.path.join(world["ld"], "launch.json")))
    assert rec["seed2_live_at_admission"] is True and rec["deadline_utc"] == S2.iso(NOW + 168 * 3600)
    assert guard(world, seed2=LIVE)[0] == "wait" and not os.path.exists(world["gd"])
    # slot 1's next arm starts beside seed 2 too, under the same rule, with no guard
    home(world, "pk-tl-N1B")
    assert decide(world, "pk-tl-S1B", seed2=LIVE, guard_up=False, overlap="short")[0] == "wait"
    assert decide(world, "pk-tl-S1B", seed2=LIVE, guard_up=False)[0] == "exec"
    assert decide(world, "pk-tl-N2B", seed2=LIVE)[0] == "wait"


def test_once_seed2_is_off_the_other_slots_start_and_every_start_needs_the_guard(world):
    assert decide(world, "pk-tl-N1B", seed2=LIVE)[0] == "exec"
    assert decide(world, "pk-tl-N2B", guard_up=False) == ("wait", "seed 2 is off and the ladder's units guard is not running yet")
    assert decide(world, "pk-tl-N2B", free_l4=0)[0] == "wait"
    kind, argv = decide(world, "pk-tl-N2B", bal=0.0)
    assert kind == "exec" and argv[argv.index("--jobs") + 1] == "tools/colab_chat/ladder_N2B.json"
    home(world, "pk-tl-N1B")
    assert decide(world, "pk-tl-S1B", guard_up=False)[0] == "wait"
    assert decide(world, "pk-tl-S1B")[0] == "exec"


# ---------------------------------------------------------------- GO, admission

def test_no_go_file_or_another_commits_or_another_line_refuses(world):
    go = os.path.join(world["ld"], "GO.json")
    rec = json.load(open(go))
    for bad in (dict(rec, commit="b" * 40), dict(rec, line="go"), dict(rec, reviews="")):
        json.dump(bad, open(go, "w"))
        assert decide(world, "pk-tl-N1B")[0] == "refuse"
        assert guard(world)[0] == "refuse"
    os.remove(go)
    assert decide(world, "pk-tl-N1B")[0] == "refuse"
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))


def test_admission_with_seed2_off(world):
    assert decide(world, "pk-tl-N1B", bal=G.EXPECT_TOTAL + 29.9)[0] == "wait"
    assert decide(world, "pk-tl-N1B", bal=None)[0] == "wait"
    assert decide(world, "pk-tl-N1B", free=7.9)[0] == "wait"
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))
    kind, argv = decide(world, "pk-tl-N1B", bal=G.EXPECT_TOTAL + 30, guard_up=False)   # the admitting arm cannot wait for a guard
    assert kind == "exec"
    rec = json.load(open(os.path.join(world["ld"], "launch.json")))
    assert rec["deadline_utc"] == S2.iso(NOW + 168 * 3600) and rec["commit"] == COMMIT and rec["cap_units"] == 540.0
    assert all(os.path.isdir(a["state_dir"]) for a in world["arms"].values())     # the guard's stop can reach every arm
    assert argv[argv.index("--deadline-utc") + 1] == rec["deadline_utc"]
    assert argv[argv.index("--max-hours") + 1] == "26" and argv[argv.index("--gpu") + 1] == "L4"
    assert argv[argv.index("--store-campaign") + 1] == "ladder-N1B"
    # the command is seed 2's but for the names, the jobs file and the hours
    ref = S2.runner_argv("pk-b2s2-N2", COMMIT, rec["deadline_utc"])
    diff = {i for i, (x, y) in enumerate(zip(argv, ref)) if x != y}
    assert len(argv) == len(ref) and {ref[i - 1] for i in diff} == {"--jobs", "--name", "--art", "--state-dir", "--max-hours", "--store-campaign"}


def test_only_a_slots_first_arm_admits_and_a_launched_ladder_without_a_record_refuses(world):
    home(world, "pk-tl-N1B")
    assert "not been admitted" in decide(world, "pk-tl-S1B")[1]
    assert decide(world, "pk-tl-N2B")[0] == "refuse"


def test_an_invalid_record_is_never_replaced(world):
    p = os.path.join(world["ld"], "launch.json")
    open(p, "w").write(json.dumps({"at_utc": S2.iso(NOW), "deadline_utc": S2.iso(NOW + 999), "commit": COMMIT}))
    before = open(p).read()
    assert decide(world, "pk-tl-N1B")[0] == "refuse" and guard(world)[0] == "refuse"
    assert open(p).read() == before


# ---------------------------------------------------------------- order, time, stops

def test_an_arm_waits_for_the_arm_before_it_in_its_slot(world):
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    assert "not home" in decide(world, "pk-tl-S1B")[1]
    home(world, "pk-tl-N1B")
    assert decide(world, "pk-tl-S1B")[0] == "exec"
    assert decide(world, "pk-tl-NT")[0] == "wait"
    open(os.path.join(world["arms"]["pk-tl-S1B"]["state_dir"], "HALTED"), "w").write("{}")
    assert decide(world, "pk-tl-NT")[0] == "halt"
    assert [G.ARMS[n]["before"] for n in ("pk-tl-N1B", "pk-tl-S1B", "pk-tl-NT", "pk-tl-N2B", "pk-tl-S2B")] == \
        [None, "pk-tl-N1B", "pk-tl-S1B", None, None]


def test_an_arm_that_no_longer_fits_before_the_deadline_halts(world):
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    home(world, "pk-tl-N1B")
    home(world, "pk-tl-S1B")
    assert decide(world, "pk-tl-NT", now=NOW + 53 * 3600)[0] == "exec"
    k, (kind, _) = decide(world, "pk-tl-NT", now=NOW + 55 * 3600)       # 55 + 114 hours is past 168
    assert (k, kind) == ("halt", "LADDER_NO_TIME")
    k, (kind, _) = decide(world, "pk-tl-N2B", now=NOW + 131 * 3600)
    assert (k, kind) == ("halt", "LADDER_NO_TIME")
    # the registered order fits the week at the expected rates: slot 1 ends at 151 hours
    assert sum(G.EXPECT_HOURS[L.RUNGS[a]["rung"]] for a in L.SLOTS[1]) == 151.0
    assert G.EXPECT_TOTAL == 466 < G.GUARD_STOP_AT < G.CAP_UNITS


def test_a_stop_reaches_arms_that_have_not_started(world, tmp_path):
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    home(world, "pk-tl-N1B")
    assert decide(world, "pk-tl-S1B", stop=True)[0] == "halt"
    open(os.path.join(world["arms"]["pk-tl-N2B"]["state_dir"], "STOP_REQUEST"), "w").write("units\n")
    assert decide(world, "pk-tl-N2B")[0] == "halt"
    log = tmp_path / "guard.jsonl"
    assert G.guard_stop_asked(str(log)) is False
    log.write_text(json.dumps({"balance": 1.0}) + "\n")
    assert G.guard_stop_asked(str(log)) is False
    log.write_text(json.dumps({"balance": 1.0}) + "\n" + json.dumps({"stop_requested": True}) + "\n")
    assert G.guard_stop_asked(str(log)) is True
    log.write_text("not json\n")
    assert G.guard_stop_asked(str(log)) is True


def test_a_launched_arm_runs_again_on_its_own_record_unless_it_is_halted(world):
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    d = world["arms"]["pk-tl-N1B"]["state_dir"]
    json.dump({"launches": 2}, open(os.path.join(d, "state.json"), "w"))
    kind, argv = decide(world, "pk-tl-N1B", seed2=LIVE, bal=None, guard_up=False, free_l4=0, stop=True, overlap="short")
    assert kind == "exec" and argv[argv.index("--name") + 1] == "pk-tl-N1B"
    # a pending stop is the runner's to take (its confirmed stop ends the session); HALTED is final for the gate
    open(os.path.join(d, "STOP_REQUEST"), "w").write("units")
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    open(os.path.join(d, "HALTED"), "w").write("{}")
    assert decide(world, "pk-tl-N1B") == ("halt", ("LADDER_HALTED", "this arm holds HALTED: the gate runs nothing; a person reads it"))


def test_a_start_reserves_its_l4_and_the_loser_of_an_admission_race_starts_as_an_admitted_arm(world, tmp_path):
    # seed 2 off, one L4 visibly free: the first start takes it; the next gate counts the pending start and waits
    assert decide(world, "pk-tl-N1B", free_l4=1)[0] == "exec"
    assert os.path.exists(os.path.join(world["arms"]["pk-tl-N1B"]["state_dir"], "START_PENDING"))
    assert decide(world, "pk-tl-N2B", free_l4=1)[0] == "wait"
    assert decide(world, "pk-tl-N2B", free_l4=2)[0] == "exec"
    assert decide(world, "pk-tl-S2B", free_l4=2)[0] == "wait"          # two pending now
    assert decide(world, "pk-tl-S2B", free_l4=2, now=NOW + G.PENDING_S + 1)[0] == "exec"     # a reservation lapses
    assert G.pending_others("pk-tl-S2B", world["arms"], NOW, [("pk-tl-N1B", "L4")]) == 1  # a listed session is counted by the listing
    # an arm whose admission was written by another arm does not start on that pass
    rec = json.load(open(os.path.join(world["ld"], "launch.json")))
    assert rec["first"] == "pk-tl-N1B"
    real = G.load_record
    calls = {"n": 0}
    def racing(path, commit=None):          # the record appears between this gate's first look and its write
        calls["n"] += 1
        return ("absent", None) if calls["n"] == 1 else real(path, commit)
    G.load_record = racing
    try:
        for n in ("pk-tl-N2B", "pk-tl-S2B"):
            pth = os.path.join(world["arms"][n]["state_dir"], "START_PENDING")
            if os.path.exists(pth):
                os.remove(pth)
        os.remove(os.path.join(world["arms"]["pk-tl-N1B"]["state_dir"], "START_PENDING"))
        kind, why = decide(world, "pk-tl-N2B", free_l4=3, guard_up=False)
    finally:
        G.load_record = real
    assert kind == "wait" and "admitted the ladder first" in why
    assert decide(world, "pk-tl-N2B", free_l4=3, guard_up=False)[0] == "wait"     # as an admitted arm it needs the guard


def test_the_cap_holds_by_wall_time_with_the_guard_blind_and_locks_exclude(tmp_path):
    worst = (G.WEEK_HOURS + G.MAX_HOURS["2B"] * 2) * G.L4_UNITS_PER_HOUR
    assert round(worst, 2) == 519.92 and worst < G.CAP_UNITS
    a = G.take_lock(str(tmp_path / "x" / "gate.lock"), wait=False)
    assert a is not None and G.take_lock(str(tmp_path / "x" / "gate.lock"), wait=False) is None
    a.close()
    assert G.take_lock(str(tmp_path / "x" / "gate.lock"), wait=False) is not None
    assert G.scan(str(tmp_path / "no_proc")) is None and G.scan(str(tmp_path)) is None
    assert "process table" in G.seed2_off(path=drivers(tmp_path), cmdlines=lambda: None, rows=[])


def test_the_guard_starts_after_seed2_with_the_overlap_counted_against_its_bound(world):
    assert guard(world) == ("wait", "the ladder has not been admitted")
    assert decide(world, "pk-tl-N1B", seed2=LIVE)[0] == "exec"
    assert guard(world, seed2=LIVE, now=NOW + 3600)[0] == "wait"
    assert guard(world, bal=None, now=NOW + 3600)[0] == "wait"
    kind, argv = guard(world, bal=512.5, now=NOW + 75 * 3600)           # seed 2 ends about 75 hours in
    assert kind == "exec" and argv[2] == "tools/colab_chat/base2_units_guard.py"
    assert argv[argv.index("--campaign") + 1] == "token-ladder"
    assert argv[argv.index("--stop-at") + 1] == "374.5"                # 520 less 75 hours of one L4 (145.5)
    assert argv[argv.index("--start-balance") + 1] == "512.5"
    assert argv[argv.index("--log") + 1] == os.path.join(G.GUARD["dir"], "guard.jsonl")
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--state-dir"] == [a["state_dir"] for a in G.ARMS.values()]
    # a restart reuses the record: the same start and bound, whatever the balance and the hour then
    assert guard(world, bal=100.0, now=NOW + 99 * 3600)[1] == argv
    # seed 2 is asked before every start of the guard, a restart too
    assert guard(world, seed2=LIVE, bal=None, now=NOW + 99 * 3600)[0] == "wait"
    # the record is held to this admission and to the registered bound
    gp = os.path.join(world["gd"], "guard_start.json")
    good = json.load(open(gp))
    for bad in (dict(good, stop_at=5000.0), dict(good, commit="b" * 40), dict(good, admitted_utc=S2.iso(NOW + 1)),
                dict(good, at_utc=S2.iso(NOW - 3600)), dict(good, start_balance=None)):
        json.dump(bad, open(gp, "w"))
        assert guard(world)[0] == "refuse", bad
    json.dump(good, open(gp, "w"))
    assert guard(world)[1] == argv
    assert G.guard_running(lambda: [argv]) and not G.guard_running(lambda: [S2.guard_argv(1.0)])
    late = dict(good, at_utc=S2.iso(NOW + 167 * 3600), stop_at=round(520 - 1.94 * 167, 2))
    json.dump(dict(late, at_utc=S2.iso(NOW + 168 * 3600), stop_at=round(520 - 1.94 * 168, 2)), open(gp, "w"))
    assert guard(world)[0] == "exec"          # 194 units still to go at the deadline: a bound above zero is run
    assert G.guard_start_problem({"start_balance": 5.0, "stop_at": -1.0}, json.load(open(os.path.join(world["ld"], "launch.json"))), COMMIT)
    open(gp, "w").write("{not json")
    assert guard(world)[0] == "refuse"
