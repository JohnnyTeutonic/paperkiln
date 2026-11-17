"""The token ladder's start gate (tools/colab_chat/ladder_launch.py). Nothing of the ladder starts or is admitted while seed 2
is live; "seed 2 off" is a positive reading; an arm's deadline is fixed at its first start and passed again at every restart,
so the cap holds by wall time; a slot's arms run one after another; a start reserves its L4; a stop reaches arms that have
not started. Nothing here opens a session, reads a balance or touches a real folder. Claude Fable 5.2, 17 Nov 2026."""
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
H = 3600
LIVE = "seed 2's section has not ended"
N1B, N2B, S1B, S2B, NT = ("pk-tl-" + a for a in ("N1B", "N2B", "S1B", "S2B", "NT"))


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
    (ld / "GO.json").write_text(json.dumps({"line": "Jonathan, direct chat, 17 Nov 2026: \"a fresh go\"", "commit": COMMIT,
                                            "reviews": "reviews/x.md", "at_utc": S2.iso(NOW - 60)}))
    return {"arms": arms, "ld": str(ld), "tmp": tmp_path}


def decide(world, name, seed2=None, free_l4=3, bal=900.0, guard_up=True, stop=False, free=20.0, now=NOW, rows=()):
    return G.decide_arm(name, COMMIT, now, Slots(free_l4), lambda: list(rows) if rows is not None else None, lambda: bal,
                        launch_dir=world["ld"], seed2=lambda now=None, rows=None: seed2, guard_up=lambda: guard_up,
                        stop_asked=lambda: stop, free=lambda: free, arms=world["arms"], clock=lambda: now,
                        ended=lambda now: None if seed2 else "home")


def guard(world, seed2=None, now=NOW):
    return G.decide_guard(now, COMMIT, launch_dir=world["ld"], seed2=lambda now=None, rows=None: seed2,
                          ended=lambda now: None if seed2 else "home")


def home(world, name, launches=1):
    d = world["arms"][name]["state_dir"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"launches": launches, "campaign_home": True, "store_release_final_done": True}, f)


def launched(world, name, n=1):
    json.dump({"launches": n}, open(os.path.join(world["arms"][name]["state_dir"], "state.json"), "w"))


def own(world, name):
    return json.load(open(os.path.join(world["arms"][name]["state_dir"], G.OWN_COPY)))


# ---------------------------------------------------------------- seed 2: a positive reading, and no overlap

S2_AT = S2.WINDOW_END - 5 * 24 * H            # a seed-2 admission inside its window
S2_DL = S2_AT + S2.MAX_HOURS * H


def seed2_record(deadline=None, **kw):
    rec = {"at_utc": S2.iso(S2_AT), "deadline_utc": S2.iso(S2_DL if deadline is None else deadline), "start_balance": 463.19,
           "first": "pk-b2s2-S2", "commit": "c" * 40, "min_balance": S2.MIN_BALANCE, "guard_stop_at": S2.GUARD_STOP_AT,
           "max_hours": S2.MAX_HOURS}
    rec.update(kw)
    return rec


def seed2_world(tmp, home_n2=False, home_s2=False, record=None, own=None):
    record = seed2_record() if record is None else record
    arms = {n: {"state_dir": str(tmp / f"s2_{i}")} for i, n in enumerate(S2.ARMS)}
    for (n, a), is_home in zip(arms.items(), (home_n2, home_s2)):
        os.makedirs(a["state_dir"], exist_ok=True)
        st = {"launches": 1, "campaign_home": True, "store_release_final_done": True} if is_home else {"launches": 1}
        json.dump(st, open(os.path.join(a["state_dir"], "state.json"), "w"))
        json.dump(record if own is None else own, open(os.path.join(a["state_dir"], S2.OWN_COPY), "w"))
    rec = tmp / "s2_launch.json"
    rec.write_text(json.dumps(record))
    ld = tmp / "tl_launch"
    ld.mkdir(exist_ok=True)
    for f in ld.glob("SEED2_ENDED.json"):
        f.unlink()
    return dict(seed2_arms=arms, seed2_launch=str(rec), launch_dir=str(ld))


def test_seed2_is_off_only_on_a_positive_end_with_no_process_and_no_session(tmp_path):
    quiet = dict(cmdlines=lambda: [["python3", "x.py"]], rows=[("pk-tl-N1B", "L4")])
    t = S2_AT + 10 * H
    assert S2.record_problem(seed2_record()) is None
    w = seed2_world(tmp_path)
    assert "has not ended" in G.seed2_off(now=t, **quiet, **w)
    assert "has not ended" in G.seed2_off(now=t, **quiet, **seed2_world(tmp_path, home_n2=True))      # one arm home is not an end
    # a HALTED seed-2 arm is not an end: it may be resumed
    w = seed2_world(tmp_path)
    open(os.path.join(w["seed2_arms"]["pk-b2s2-N2"]["state_dir"], "HALTED"), "w").write("{}")
    assert "has not ended" in G.seed2_off(now=t, **quiet, **w)
    os.remove(os.path.join(w["seed2_arms"]["pk-b2s2-N2"]["state_dir"], "HALTED"))
    # end 1: both arms home with the last release made
    both = seed2_world(tmp_path, home_n2=True, home_s2=True)
    assert G.seed2_section_ended(t, both["launch_dir"], both["seed2_arms"], both["seed2_launch"]) == "home"
    assert G.seed2_off(now=t, **quiet, **both) is None
    # end 2: an hour past seed 2's deadline, proven by a valid record and the arms' own copies of it
    w = seed2_world(tmp_path)
    assert "has not ended" in G.seed2_off(now=S2_DL + 3599, **quiet, **w)
    assert G.seed2_off(now=S2_DL + H, **quiet, **w) is None
    assert G.seed2_section_ended(S2_DL + H, w["launch_dir"], w["seed2_arms"], w["seed2_launch"]) == "deadline"
    late = S2_DL + 10 * H
    for bad in ({"deadline_utc": S2.iso(S2_AT)},                               # a bare deadline proves nothing
                seed2_record(deadline=S2_AT + 3 * H),                          # a deadline that is not the admission plus its hours
                seed2_record(min_balance=1.0)):
        assert G.seed2_proven_deadline(*[seed2_world(tmp_path, record=bad)[k] for k in ("seed2_arms", "seed2_launch")]) is None
        assert "has not ended" in G.seed2_off(now=late, **quiet, **seed2_world(tmp_path, record=bad))
    # a launched arm whose own copy holds another deadline, or none, could be resumed under it: no proof
    other = seed2_record(at_utc=S2.iso(S2_AT + 24 * H), deadline=S2_AT + 24 * H + S2.MAX_HOURS * H)
    assert S2.record_problem(other) is None
    assert "has not ended" in G.seed2_off(now=late, **quiet, **seed2_world(tmp_path, own=other))
    assert "has not ended" in G.seed2_off(now=late, **quiet, **seed2_world(tmp_path, own={"x": 1}))
    # end 3: the lead's note, complete and not from the future
    w = seed2_world(tmp_path)
    lead = os.path.join(w["launch_dir"], "SEED2_ENDED.json")
    json.dump({"by": "the lead"}, open(lead, "w"))
    assert "has not ended" in G.seed2_off(now=t, **quiet, **w)
    json.dump({"by": "the lead", "why": "48.3 closed", "at_utc": S2.iso(t + 2 * H)}, open(lead, "w"))
    assert "has not ended" in G.seed2_off(now=t, **quiet, **w)
    json.dump({"by": "the lead", "why": "48.3 closed", "at_utc": S2.iso(t)}, open(lead, "w"))
    # the note alone is not enough: each seed-2 arm's own folder must show it stopped
    assert "has not ended" in G.seed2_off(now=t, **quiet, **w)
    n2, s2 = (a["state_dir"] for a in w["seed2_arms"].values())
    json.dump({"kind": "LEAD_STOP", "stop_confirmed": False}, open(os.path.join(n2, "HALTED"), "w"))
    json.dump({"kind": "DEADLINE", "stopped": True}, open(os.path.join(s2, "DEADLINE"), "w"))
    assert "has not ended" in G.seed2_off(now=t, **quiet, **w)                 # one arm's stop is not confirmed
    json.dump({"kind": "LEAD_STOP", "stop_confirmed": True}, open(os.path.join(n2, "HALTED"), "w"))
    assert G.seed2_arm_stopped(n2) and G.seed2_arm_stopped(s2)
    assert G.seed2_off(now=t, **quiet, **w) is None
    assert G.seed2_section_ended(t, w["launch_dir"], w["seed2_arms"], w["seed2_launch"]) == "lead"
    # and still live with a seed-2 process, a seed-2 session, or anything unreadable
    both = seed2_world(tmp_path, home_n2=True, home_s2=True)
    live = [["python3", "-u", "runner_multi.py", "--name", "pk-b2s2-S2"]]
    assert "pk-b2s2-S2" in G.seed2_off(now=t, cmdlines=lambda: live, rows=[], **both)
    sd = both["seed2_arms"]["pk-b2s2-N2"]["state_dir"]
    assert "pk-b2s2-N2" in G.seed2_off(now=t, cmdlines=lambda: [["python3", "r.py", "--state-dir", sd]], rows=[], **both)
    assert "process table" in G.seed2_off(now=t, cmdlines=lambda: None, rows=[], **both)
    assert "did not answer" in G.seed2_off(now=t, cmdlines=lambda: [], rows=None, **both)
    assert "pk-b2s2-N2" in G.seed2_off(now=t, cmdlines=lambda: [], rows=[("pk-b2s2-N2", "L4")], **both)
    assert G.scan(str(tmp_path / "no_proc")) is None and G.scan(str(tmp_path / "tl_launch")) is None


def test_nothing_starts_and_nothing_is_admitted_while_seed2_is_live(world):
    for name in G.ARMS:
        kind, why = decide(world, name, seed2=LIVE)
        assert kind == "wait"
    assert "nothing of the ladder starts beside it" in decide(world, N1B, seed2=LIVE)[1]
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))
    assert not any(os.path.exists(os.path.join(a["state_dir"], G.OWN_COPY)) for a in world["arms"].values())
    # after admission too: an arm that has not started waits, and the guard waits
    assert decide(world, N1B)[0] == "exec"
    assert decide(world, S1B, seed2=LIVE)[0] == "wait" and decide(world, NT, seed2=LIVE)[0] == "wait"
    assert guard(world, seed2=LIVE)[0] == "wait"
    src = open(G.__file__, encoding="utf-8").read()
    assert "overlap_problem" not in src and "seed2_cap_left" not in src and "SEED2_PURCHASED" not in src     # no path beside seed 2


def test_the_gate_never_writes_to_seed2s_names():
    src = open(G.__file__, encoding="utf-8").read()
    assert "os.kill" not in src and "STOP_REQUEST\"), \"w\"" not in src and "[\"enabled\"] =" not in src
    for a in G.ARMS.values():
        assert "seed2" not in a["state_dir"] and a["state_dir"] not in [x["state_dir"] for x in S2.ARMS.values()]
        assert a["campaign"] not in [x["campaign"] for x in S2.ARMS.values()] and not a["campaign"].startswith("base2")
    assert G.GUARD["dir"] != S2.GUARD["dir"] and G.GUARD["campaign"] != S2.GUARD["campaign"]
    assert not set(G.ARMS) & set(S2.ARMS)
    assert not G.LAUNCH_DIR.startswith(S2.LAUNCH_DIR) and G.SEED2_LAUNCH == S2.LAUNCH_DIR + "/launch.json"
    assert set(G.ARMS) == {N1B, N2B, S1B, S2B, NT}


# ---------------------------------------------------------------- GO, admission

def test_no_go_file_or_another_commits_or_no_dated_line_refuses(world):
    go = os.path.join(world["ld"], "GO.json")
    rec = json.load(open(go))
    for bad in (dict(rec, commit="b" * 40), dict(rec, line="go"), dict(rec, line="someone: \"go\" and more text to be long"),
                dict(rec, reviews="")):
        json.dump(bad, open(go, "w"))
        assert decide(world, N1B)[0] == "refuse"
        assert guard(world)[0] == "refuse"
    os.remove(go)
    assert decide(world, N1B)[0] == "refuse"
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))


def test_admission_and_the_first_start_record(world):
    assert decide(world, N1B, bal=None)[0] == "wait"
    assert decide(world, N1B, free=7.9)[0] == "wait"
    assert decide(world, N1B, free_l4=0)[0] == "wait"
    assert decide(world, N1B, rows=None)[0] == "wait"                    # the listing did not answer
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))
    kind, argv = decide(world, N1B, guard_up=False)                      # the admitting arm cannot wait for a guard
    assert kind == "exec"
    rec = json.load(open(os.path.join(world["ld"], "launch.json")))
    assert rec["deadline_utc"] == S2.iso(NOW + 168 * H) and rec["commit"] == COMMIT
    assert (rec["cap_units"], rec["guard_stop_at"], rec["worst_hours"], rec["worst_units"]) == (580.0, 560.0, 292, 566.5)
    assert all(os.path.isdir(a["state_dir"]) for a in world["arms"].values())     # the guard's stop can reach every arm
    o = own(world, N1B)
    assert o["first_start_utc"] == S2.iso(NOW) and o["arm_deadline_utc"] == S2.iso(NOW + 26 * H) and o["max_hours"] == 26
    assert argv[argv.index("--deadline-utc") + 1] == o["arm_deadline_utc"]        # the ARM's deadline, not the week's
    assert argv[argv.index("--max-hours") + 1] == "26" and argv[argv.index("--gpu") + 1] == "L4"
    assert argv[argv.index("--store-campaign") + 1] == "ladder-N1B"
    assert argv[argv.index("--jobs") + 1] == "tools/colab_chat/ladder_N1B.json"
    # the command is seed 2's but for the names, the jobs file, the hours and the deadline's value
    ref = S2.runner_argv("pk-b2s2-N2", COMMIT, o["arm_deadline_utc"])
    diff = {i for i, (x, y) in enumerate(zip(argv, ref)) if x != y}
    assert len(argv) == len(ref) and {ref[i - 1] for i in diff} == {"--jobs", "--name", "--art", "--state-dir", "--max-hours", "--store-campaign"}
    # the other slots' first arms start on the same admission, once the guard runs
    assert decide(world, S1B, guard_up=False) == ("wait", "the ladder's units guard is not running yet")
    assert decide(world, S1B, now=NOW + 600)[0] == "exec" and decide(world, NT, now=NOW + 900)[0] == "exec"
    assert own(world, NT)["arm_deadline_utc"] == S2.iso(NOW + 900 + 132 * H)


def test_only_a_slots_first_arm_admits_and_a_launched_ladder_without_a_record_refuses(world):
    home(world, N1B)
    assert "not been admitted" in decide(world, N2B)[1]
    assert decide(world, S1B)[0] == "refuse"


def test_an_invalid_record_is_never_replaced(world):
    p = os.path.join(world["ld"], "launch.json")
    open(p, "w").write(json.dumps({"at_utc": S2.iso(NOW), "deadline_utc": S2.iso(NOW + 999), "commit": COMMIT}))
    before = open(p).read()
    assert decide(world, N1B)[0] == "refuse" and guard(world)[0] == "refuse"
    assert open(p).read() == before


# ---------------------------------------------------------------- the arm deadline: fixed once, passed at every restart

def test_a_restart_passes_the_first_starts_deadline_and_can_never_extend_the_arm(world):
    assert decide(world, N1B)[0] == "exec"
    assert decide(world, NT, now=NOW + 60)[0] == "exec"
    first = own(world, NT)
    launched(world, NT)
    # restarts at 10, 100 and 131.9 hours with its session up: the same deadline every time, whatever else is true then
    for h in (10, 100, 131.9):
        kind, argv = decide(world, NT, now=NOW + int(h * H), seed2=LIVE, bal=None, guard_up=False, free_l4=0, stop=True,
                            rows=[(NT, "L4")])
        assert kind == "exec"
        assert argv[argv.index("--deadline-utc") + 1] == first["arm_deadline_utc"] == S2.iso(NOW + 60 + 132 * H)
        assert argv[argv.index("--max-hours") + 1] == "132"
        assert own(world, NT) == first                                    # written once, never rewritten
    # the same restarts with its session gone (a replacement would be made): the same deadline again
    for h in (10, 100, 131.9):
        kind, argv = decide(world, NT, now=NOW + int(h * H))
        assert kind == "exec" and argv[argv.index("--deadline-utc") + 1] == first["arm_deadline_utc"]
        assert own(world, NT) == first
    # the runner takes the earlier of the passed deadline and now + --max-hours: a restart cannot outlive the first start + limit
    import runner_multi as R
    dl = S2.parse_iso(first["arm_deadline_utc"])
    assert R.resolve_deadline(132, dl, NOW + 100 * H) == dl
    with pytest.raises(ValueError):
        R.resolve_deadline(132, dl, dl + 1)                               # past it, no session may be created
    # past the arm's deadline no session is made. With no session of its name listed, the gate halts the arm
    past = NOW + 60 + 132 * H
    kind, (hk, why) = decide(world, NT, now=past)
    assert (kind, hk) == ("halt", "LADDER_ARM_DEADLINE") and "no session of its name is listed" in why
    # with its session still listed, or the listing not answering, the runner is run to stop it: started after its deadline
    # it creates nothing, stops the session the folder recorded and writes DEADLINE (runner_multi.py's own expired start)
    for rows in ([(NT, "L4")], None):
        kind, argv = decide(world, NT, now=past + 6 * H, rows=rows, seed2=LIVE, bal=None, guard_up=False, free_l4=0)
        assert kind == "exec" and argv[argv.index("--deadline-utc") + 1] == first["arm_deadline_utc"]
        assert S2.parse_iso(argv[argv.index("--deadline-utc") + 1]) < past + 6 * H      # the expired deadline: nothing is created
    # a low arm likewise: N1B's 26 hours from its first start
    launched(world, N1B, 3)
    assert decide(world, N1B, now=NOW + 25 * H)[1][decide(world, N1B, now=NOW + 25 * H)[1].index("--deadline-utc") + 1] == S2.iso(NOW + 26 * H)
    assert decide(world, N1B, now=NOW + 26 * H)[0] == "halt"


def test_a_tampered_or_missing_first_start_record_refuses(world):
    assert decide(world, N1B)[0] == "exec"
    launched(world, N1B)
    p = os.path.join(world["arms"][N1B]["state_dir"], G.OWN_COPY)
    good = json.load(open(p))
    for bad in (dict(good, arm_deadline_utc=S2.iso(NOW + 168 * H)), dict(good, first_start_utc=S2.iso(NOW + 100 * H)),
                dict(good, name=NT), dict(good, at_utc=S2.iso(NOW + 5)), {k: v for k, v in good.items() if k != "arm_deadline_utc"}):
        json.dump(bad, open(p, "w"))
        assert decide(world, N1B, now=NOW + H)[0] == "refuse", bad
    os.remove(p)
    assert decide(world, N1B, now=NOW + H)[0] == "refuse"
    json.dump(good, open(p, "w"))
    assert decide(world, N1B, now=NOW + H)[0] == "exec"
    # an arm that starts late in the week is cut by the week's deadline, not by its own limit
    home(world, N1B)
    kind, argv = decide(world, N2B, now=NOW + 100 * H)
    assert kind == "exec" and own(world, N2B)["arm_deadline_utc"] == S2.iso(NOW + 154 * H)


def test_the_cap_holds_by_wall_time_alone(world):
    # every arm's hour limit, summed: no restart can add to it (the test above), so this is the most the ladder can bill
    assert G.MAX_HOURS == {"1B": 26, "2B": 54, "T": 132} and G.WORST_HOURS == 26 * 2 + 54 * 2 + 132 == 292
    assert G.WORST_UNITS == 566.5 and G.CAP_UNITS - G.WORST_HOURS * G.L4_UNITS_PER_HOUR >= 10      # at least 10 units under the cap
    assert G.CAP_UNITS == 580.0 and G.GUARD_STOP_AT == 560.0
    assert G.EXPECT_HOURS["T"] == 94.0 and abs(L.STEPS["T"] / 3300 - 94.0) < 0.1                 # 310,000 steps at base 2's rate
    for rung, h in G.EXPECT_HOURS.items():
        assert G.MAX_HOURS[rung] >= 1.4 * h - 0.01                         # 40% over the expectation
    # each slot's limits fit the week, with the chat campaign after the last arm
    for slot in L.SLOTS.values():
        assert sum(G.MAX_HOURS[L.RUNGS[a]["rung"]] for a in slot) + G.CHAT_HOURS <= G.WEEK_HOURS
    assert sum(G.EXPECT_HOURS[L.RUNGS[a]["rung"]] for a in L.LAUNCHED_ARMS) == 207.0


# ---------------------------------------------------------------- units: the balance must carry the worst case

def test_a_start_needs_the_worst_case_of_what_is_left_and_the_top_rung_never_holds_the_low_arms_back(world):
    low = (26 * 2 + 54 * 2) * 1.94 + 20          # 330.4: the four low arms and the margin
    full = 292 * 1.94 + 20                       # 586.48: everything and the margin
    assert abs(G.units_needed(N1B, world["arms"], NOW, None) - low) < 1e-6
    assert abs(G.units_needed(NT, world["arms"], NOW, None) - full) < 1e-6
    assert abs(G.units_needed(None, world["arms"], NOW, None) - full) < 1e-6
    assert decide(world, N1B, bal=low - 0.5)[0] == "wait"
    assert decide(world, N1B, bal=low + 0.5)[0] == "exec"
    assert "no units are bought" in decide(world, NT, bal=full - 0.5)[1]     # the top rung waits; it is not admitted on hope
    assert decide(world, S1B, bal=low + 0.5)[0] == "exec"                    # and does not hold a low arm back
    assert decide(world, NT, bal=full + 0.5)[0] == "exec"
    # once arms have started they count to their own deadlines, and finished arms drop out
    t = NOW + 20 * H
    home(world, N1B)
    left = (6 + 54 + 54 + 112) * 1.94 + 20       # S1B's last 6 h, N2B and S2B whole, NT's last 112 h
    assert abs(G.units_needed(N2B, world["arms"], t, json.load(open(os.path.join(world["ld"], "launch.json")))) - left) < 0.01
    assert decide(world, N2B, now=t, bal=left - 1)[0] == "wait" and decide(world, N2B, now=t, bal=left + 1)[0] == "exec"


# ---------------------------------------------------------------- order, time, stops, reservations

def test_an_arm_waits_for_the_arm_before_it_in_its_slot(world):
    assert decide(world, N1B)[0] == "exec"
    assert "not home" in decide(world, N2B)[1]
    home(world, N1B)
    assert decide(world, N2B)[0] == "exec"
    assert decide(world, S1B)[0] == "exec"
    open(os.path.join(world["arms"][S1B]["state_dir"], "HALTED"), "w").write("{}")
    assert decide(world, S2B)[0] == "halt"
    assert decide(world, NT)[0] == "exec"                                 # the top rung depends on no other arm
    assert [G.ARMS[n]["before"] for n in (N1B, N2B, S1B, S2B, NT)] == [None, N1B, None, S1B, None]


def test_an_arm_whose_limit_and_chat_campaign_no_longer_fit_halts(world):
    assert decide(world, N1B)[0] == "exec"
    home(world, N1B)
    assert decide(world, N2B, now=NOW + 102 * H)[0] == "exec"            # 102 + 54 + 12 = 168
    os.remove(os.path.join(world["arms"][N2B]["state_dir"], G.OWN_COPY))
    os.remove(os.path.join(world["arms"][N2B]["state_dir"], "START_PENDING"))
    k, (kind, _) = decide(world, N2B, now=NOW + 102 * H + 1)
    assert (k, kind) == ("halt", "LADDER_NO_TIME")


def test_the_top_rung_must_start_within_twenty_four_hours_of_admission(world):
    assert decide(world, N1B)[0] == "exec"
    assert decide(world, NT, now=NOW + 24 * H + 1)[0] == "halt"          # 132 + 12 hours no longer fit
    os.remove(os.path.join(world["arms"][NT]["state_dir"], "HALTED")) if os.path.exists(os.path.join(world["arms"][NT]["state_dir"], "HALTED")) else None
    assert decide(world, NT, now=NOW + 24 * H)[0] == "exec"            # exactly 24 hours in, it still fits


def test_a_stop_reaches_arms_that_have_not_started_and_a_torn_log_line_is_not_a_stop(world, tmp_path):
    assert decide(world, N1B)[0] == "exec"
    home(world, N1B)
    assert decide(world, N2B, stop=True)[0] == "halt"
    open(os.path.join(world["arms"][S1B]["state_dir"], "STOP_REQUEST"), "w").write("units\n")
    assert decide(world, S1B)[0] == "halt"
    log = tmp_path / "guard.jsonl"
    assert G.guard_stop_asked(str(log)) is False
    log.write_text(json.dumps({"balance": 1.0}) + "\n" + '{"balance": 2.0, "sp')          # a torn last line
    assert G.guard_stop_asked(str(log)) is False
    log.write_text(json.dumps({"stop_requested": True}) + "\n" + "not json\n")
    assert G.guard_stop_asked(str(log)) is True


def test_a_listed_session_is_resumed_and_a_lost_one_is_replaced_only_under_every_start_condition(world):
    assert decide(world, N1B)[0] == "exec"
    d = world["arms"][N1B]["state_dir"]
    up = [(N1B, "L4")]
    # a session of its name is listed though no launch was counted (a crash between the create and the save): it is this
    # arm's, so the runner is run for it; no second slot is asked
    kind, argv = decide(world, N1B, now=NOW + H, free_l4=0, bal=None, rows=up)
    assert kind == "exec" and argv[argv.index("--name") + 1] == N1B
    launched(world, N1B, 2)
    # launched before, and its session is gone: a replacement is a new allocation, asked like any start
    t = NOW + 2 * H
    assert decide(world, N1B, now=t)[0] == "exec"
    os.remove(os.path.join(d, "START_PENDING"))
    assert "no replacement session beside it" in decide(world, N1B, now=t, seed2=LIVE)[1]
    assert decide(world, N1B, now=t, free_l4=0)[0] == "wait"
    assert decide(world, N1B, now=t, guard_up=False)[0] == "wait"
    assert decide(world, N1B, now=t, free=1.0)[0] == "wait"
    assert decide(world, N1B, now=t, bal=None)[0] == "wait"
    need = G.units_needed(N1B, world["arms"], t, json.load(open(os.path.join(world["ld"], "launch.json"))))
    assert abs(need - ((24 + 54 + 26 + 54) * 1.94 + 20)) < 0.01           # its own 24 hours left, and the three unstarted low arms
    assert decide(world, N1B, now=t, bal=need - 1)[0] == "wait" and decide(world, N1B, now=t, bal=need + 1)[0] == "exec"
    assert decide(world, N1B, now=t, rows=None)[0] == "wait"              # the listing did not answer: unknown, so nothing
    assert decide(world, N1B, now=t, stop=True)[0] == "halt"              # a stop asked, and no session to stop: none is made
    os.remove(os.path.join(d, "HALTED")) if os.path.exists(os.path.join(d, "HALTED")) else None
    # a pending stop with its session up is the runner's to take; with no session the gate halts
    open(os.path.join(d, "STOP_REQUEST"), "w").write("units")
    assert decide(world, N1B, now=t, rows=up)[0] == "exec"
    assert decide(world, N1B, now=t)[0] == "halt"
    os.remove(os.path.join(d, "STOP_REQUEST"))
    go = os.path.join(world["ld"], "GO.json")
    saved = open(go).read()
    os.remove(go)
    assert decide(world, N1B, now=t)[0] == "refuse" and decide(world, N1B, now=t, rows=up)[0] == "exec"
    open(go, "w").write(saved)
    # HALTED with its stop confirmed is final for the gate, session listed or not
    json.dump({"kind": "LEAD_STOP", "stop_confirmed": True}, open(os.path.join(d, "HALTED"), "w"))
    for rows in (up, (), None):
        assert decide(world, N1B, now=t, rows=rows) == ("halt", ("LADDER_HALTED", "this arm holds HALTED: the gate runs nothing; a person reads it"))
    # HALTED with its stop NOT confirmed: with the session listed, or the listing silent, the runner is run to finish the stop
    json.dump({"kind": "STORE_STALL", "stop_confirmed": False}, open(os.path.join(d, "HALTED"), "w"))
    assert decide(world, N1B, now=t, rows=up, seed2=LIVE, bal=None, free_l4=0)[0] == "exec"
    assert decide(world, N1B, now=t, rows=None)[0] == "exec"
    assert decide(world, N1B, now=t, rows=())[0] == "halt"                # the listing answers without it: nothing to stop


def test_a_start_reserves_its_l4_until_its_session_is_listed_or_it_has_ended(world):
    assert decide(world, N1B, free_l4=1)[0] == "exec"
    assert os.path.exists(os.path.join(world["arms"][N1B]["state_dir"], "START_PENDING"))
    assert decide(world, S1B, free_l4=1)[0] == "wait"
    assert decide(world, S1B, free_l4=2)[0] == "exec"
    assert decide(world, NT, free_l4=2)[0] == "wait"                      # two pending now
    arms = world["arms"]
    assert G.pending_others(NT, arms, []) == 2
    assert G.pending_others(NT, arms, [(N1B, "L4")]) == 1                 # a listed session is counted by the listing
    launched(world, S1B)
    assert G.pending_others(NT, arms, [(N1B, "L4")]) == 1                 # launched and not yet listed: still held
    home(world, S1B)
    assert G.pending_others(NT, arms, [(N1B, "L4")]) == 0                 # ended: freed
    open(os.path.join(arms[N1B]["state_dir"], "HALTED"), "w").write("{}")
    assert G.pending_others(NT, arms, []) == 0


def test_the_loser_of_an_admission_race_starts_as_an_admitted_arm(world):
    assert decide(world, N1B)[0] == "exec"
    real, calls = G.load_record, {"n": 0}

    def racing(path, commit=None):          # the record appears between this gate's first look and its write
        calls["n"] += 1
        return ("absent", None) if calls["n"] == 1 else real(path, commit)
    os.remove(os.path.join(world["arms"][N1B]["state_dir"], "START_PENDING"))
    G.load_record = racing
    try:
        kind, why = decide(world, S1B, guard_up=False)
    finally:
        G.load_record = real
    assert kind == "wait" and "admitted the ladder first" in why
    assert not os.path.exists(os.path.join(world["arms"][S1B]["state_dir"], G.OWN_COPY))
    assert decide(world, S1B, guard_up=False)[0] == "wait"                # as an admitted arm it needs the guard


def test_the_guard_is_base2s_guard_on_the_ladders_folders_with_the_admissions_balance(world, tmp_path):
    assert guard(world) == ("wait", "the ladder has not been admitted")
    assert decide(world, N1B, bal=640.25)[0] == "exec"
    kind, argv = guard(world)
    assert kind == "exec" and argv[2] == "tools/colab_chat/base2_units_guard.py"
    assert argv[argv.index("--campaign") + 1] == "token-ladder" and argv[argv.index("--stop-at") + 1] == "560"
    assert argv[argv.index("--start-balance") + 1] == "640.25"
    assert argv[argv.index("--log") + 1] == os.path.join(G.GUARD["dir"], "guard.jsonl")
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--state-dir"] == [a["state_dir"] for a in G.ARMS.values()]
    assert guard(world, now=NOW + 99 * H)[1] == argv                      # a restart: the same start and bound
    assert guard(world, seed2=LIVE)[0] == "wait"                          # seed 2 is asked before every start
    assert G.guard_running(lambda: [argv]) and not G.guard_running(lambda: [S2.guard_argv(1.0)])
    assert G.guard_running(lambda: None) is False
    a = G.take_lock(str(tmp_path / "x" / "gate.lock"), wait=False)
    assert a is not None and G.take_lock(str(tmp_path / "x" / "gate.lock"), wait=False) is None
    a.close()
    assert G.take_lock(str(tmp_path / "x" / "gate.lock"), wait=False) is not None


def test_an_arm_that_recorded_its_deadline_and_never_counted_a_launch_halts_when_it_passes(world):
    # Colab had no L4 for the arm's whole limit: its first-start record exists, state.json counts no launch
    assert decide(world, N1B)[0] == "exec"
    assert decide(world, S1B)[0] == "exec"
    d = world["arms"][S1B]["state_dir"]
    pending = os.path.join(d, "START_PENDING")
    before = open(pending).read()
    assert decide(world, S1B, now=NOW + 25 * H)[0] == "exec"                 # inside its deadline it is still tried, on that deadline
    assert own(world, S1B)["arm_deadline_utc"] == S2.iso(NOW + 26 * H)
    open(pending, "w").write(before)
    kind, (hk, why) = decide(world, S1B, now=NOW + 26 * H)
    assert (kind, hk) == ("halt", "LADDER_ARM_DEADLINE") and "before it counted a launch" in why
    assert open(pending).read() == before                                    # no fresh reservation is written past the deadline
    # once it holds HALTED (the gate's main loop writes it), its reservation is freed and the arm after it halts
    open(os.path.join(d, "HALTED"), "w").write("{}")
    assert G.pending_others(NT, world["arms"], [(N1B, "L4")]) == 0
    assert decide(world, S2B, now=NOW + 27 * H)[0] == "halt"


def test_an_arm_that_has_never_started_asks_colab_for_nothing_while_seed2s_section_runs(world):
    calls = {"n": 0}

    def rows_fn():
        calls["n"] += 1
        return []
    def run(name, ended, **kw):
        return G.decide_arm(name, COMMIT, NOW, Slots(3), rows_fn, lambda: 900.0, launch_dir=world["ld"],
                            seed2=lambda now=None, rows=None: None, guard_up=lambda: True, stop_asked=lambda: False,
                            free=lambda: 20.0, arms=world["arms"], clock=lambda: NOW, ended=lambda now: ended, **kw)
    for name in G.ARMS:
        assert run(name, None) == ("wait", "seed 2's section has not ended: nothing of the ladder starts beside it")
    assert calls["n"] == 0
    assert G.decide_guard(NOW, COMMIT, launch_dir=world["ld"], rows_fn=rows_fn, ended=lambda now: None)[0] in ("wait", "refuse")
    assert calls["n"] == 0
    os.rename(os.path.join(world["ld"], "GO.json"), os.path.join(world["ld"], "GO.off"))
    assert run(N1B, None)[0] == "refuse" and calls["n"] == 0                 # no GO: refused, still with no call
    os.rename(os.path.join(world["ld"], "GO.off"), os.path.join(world["ld"], "GO.json"))
    assert run(N1B, "home")[0] == "exec" and calls["n"] == 1
    # an arm that has started is another matter: its session must be looked for, whatever seed 2 is doing
    assert run(N1B, None)[0] == "exec" and calls["n"] == 2
