"""The token ladder's start gate (tools/colab_chat/ladder_launch.py): no arm and no guard starts while seed 2 is live, the
admission is written once with the registered bounds, a slot's arms run one after another, and a stop reaches arms that have
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
    return {"arms": arms, "ld": str(ld), "tmp": tmp_path}


def decide(world, name, seed2=None, free_l4=3, bal=800.0, guard_up=True, stop=False, free=20.0, now=NOW):
    return G.decide_arm(name, COMMIT, now, Slots(free_l4), lambda: [], lambda: bal, launch_dir=world["ld"],
                        seed2=lambda rows=None: seed2, guard_up=lambda: guard_up, stop_asked=lambda: stop,
                        free=lambda: free, arms=world["arms"], clock=lambda: now)


def home(world, name, launches=1):
    d = world["arms"][name]["state_dir"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"launches": launches, "campaign_home": True, "store_release_final_done": True}, f)


# ---------------------------------------------------------------- seed 2 first

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


def test_nothing_starts_or_is_admitted_while_seed2_is_live(world):
    for name in G.ARMS:
        kind, why = decide(world, name, seed2="seed 2 is live")
        assert kind == "wait" and "seed 2" in why
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))
    assert G.decide_guard(NOW, COMMIT, launch_dir=world["ld"], seed2=lambda rows=None: "seed 2 is live")[0] == "wait"


def test_the_gate_never_writes_to_seed2s_names():
    src = open(G.__file__, encoding="utf-8").read()
    assert "STOP_REQUEST\"), \"w\"" not in src and "os.kill" not in src and "enabled\"] =" not in src
    for a in G.ARMS.values():
        assert "seed2" not in a["state_dir"] and a["state_dir"] not in [x["state_dir"] for x in S2.ARMS.values()]
    assert G.GUARD["dir"] != S2.GUARD["dir"] and G.GUARD["campaign"] != S2.GUARD["campaign"]
    assert not set(G.ARMS) & set(S2.ARMS)


# ---------------------------------------------------------------- GO, admission

def test_no_go_file_or_another_commits_or_another_line_refuses(world):
    go = os.path.join(world["ld"], "GO.json")
    rec = json.load(open(go))
    for bad in (dict(rec, commit="b" * 40), dict(rec, line="go"), dict(rec, reviews="")):
        json.dump(bad, open(go, "w"))
        assert decide(world, "pk-tl-N1B")[0] == "refuse"
        assert G.decide_guard(NOW, COMMIT, launch_dir=world["ld"], seed2=lambda rows=None: None)[0] == "refuse"
    os.remove(go)
    assert decide(world, "pk-tl-N1B")[0] == "refuse"
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))


def test_admission_needs_room_for_three_and_the_whole_cap_in_balance(world):
    assert decide(world, "pk-tl-N1B", free_l4=2)[0] == "wait"
    assert "bought before admission" in decide(world, "pk-tl-N1B", bal=759.9)[1]
    assert decide(world, "pk-tl-N1B", bal=None)[0] == "wait"
    assert decide(world, "pk-tl-N1B", free=7.9)[0] == "wait"
    assert not os.path.exists(os.path.join(world["ld"], "launch.json"))
    kind, argv = decide(world, "pk-tl-N1B", guard_up=False)     # the admitting arm does not wait for the guard
    assert kind == "exec"
    rec = json.load(open(os.path.join(world["ld"], "launch.json")))
    assert rec["deadline_utc"] == S2.iso(NOW + 168 * 3600) and rec["start_balance"] == 800.0 and rec["commit"] == COMMIT
    assert all(os.path.isdir(a["state_dir"]) for a in world["arms"].values())     # the guard's stop can reach every arm
    assert argv[argv.index("--jobs") + 1] == "tools/colab_chat/ladder_N1B.json"
    assert argv[argv.index("--deadline-utc") + 1] == rec["deadline_utc"]
    assert argv[argv.index("--max-hours") + 1] == "26" and argv[argv.index("--gpu") + 1] == "L4"
    assert argv[argv.index("--store-campaign") + 1] == "ladder-N1B"
    # the command is seed 2's but for the names, the jobs file and the hours
    ref = S2.runner_argv("pk-b2s2-N2", COMMIT, rec["deadline_utc"])
    diff = {i for i, (x, y) in enumerate(zip(argv, ref)) if x != y}
    assert len(argv) == len(ref) and {ref[i - 1] for i in diff} == {"--jobs", "--name", "--art", "--state-dir", "--max-hours", "--store-campaign"}
    # the second and third slots start on the same record, once the guard runs
    assert decide(world, "pk-tl-S1B", guard_up=False)[0] == "wait"
    assert decide(world, "pk-tl-NT", free_l4=0)[0] == "wait"
    kind, argv = decide(world, "pk-tl-NT", bal=0.0)
    assert kind == "exec" and argv[argv.index("--deadline-utc") + 1] == rec["deadline_utc"]


def test_only_a_slots_first_arm_admits_and_a_launched_ladder_without_a_record_refuses(world):
    assert decide(world, "pk-tl-N2B")[0] == "wait"
    home(world, "pk-tl-N1B")
    assert "not been admitted" in decide(world, "pk-tl-N2B")[1]
    assert decide(world, "pk-tl-S1B")[0] == "refuse"


def test_an_invalid_record_is_never_replaced(world):
    p = os.path.join(world["ld"], "launch.json")
    open(p, "w").write(json.dumps({"at_utc": S2.iso(NOW), "deadline_utc": S2.iso(NOW + 999), "commit": COMMIT}))
    before = open(p).read()
    assert decide(world, "pk-tl-N1B")[0] == "refuse"
    assert open(p).read() == before


# ---------------------------------------------------------------- order, time, stops

def test_an_arm_waits_for_the_arm_before_it_in_its_slot(world):
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    assert "not home" in decide(world, "pk-tl-N2B")[1]
    home(world, "pk-tl-N1B")
    assert decide(world, "pk-tl-N2B")[0] == "exec"
    assert decide(world, "pk-tl-S2B")[0] == "wait"
    os.makedirs(world["arms"]["pk-tl-N2B"]["state_dir"], exist_ok=True)
    open(os.path.join(world["arms"]["pk-tl-N2B"]["state_dir"], "HALTED"), "w").write("{}")
    assert decide(world, "pk-tl-S2B")[0] == "halt"
    assert [G.ARMS[n]["before"] for n in ("pk-tl-N1B", "pk-tl-N2B", "pk-tl-S2B", "pk-tl-S1B", "pk-tl-ST", "pk-tl-NT")] == \
        [None, "pk-tl-N1B", "pk-tl-N2B", None, "pk-tl-S1B", None]


def test_an_arm_that_no_longer_fits_before_the_deadline_halts(world):
    assert decide(world, "pk-tl-S1B")[0] == "exec"
    home(world, "pk-tl-S1B")
    assert decide(world, "pk-tl-ST", now=NOW + 53 * 3600)[0] == "exec"
    k, (kind, _) = decide(world, "pk-tl-ST", now=NOW + 55 * 3600)       # 55 + 114 hours is past 168
    assert (k, kind) == ("halt", "LADDER_NO_TIME")
    # the registered order fits the week at the expected rates
    for slot in L.SLOTS.values():
        assert sum(G.EXPECT_HOURS[L.RUNGS[a]["rung"]] for a in slot) <= 168 - 24
    assert sum(G.EXPECT_UNITS[L.RUNGS[a]["rung"]] for a in L.FIRST_SEED_ARMS) == 662 < G.GUARD_STOP_AT < G.CAP_UNITS


def test_a_stop_reaches_arms_that_have_not_started(world, tmp_path):
    assert decide(world, "pk-tl-N1B")[0] == "exec"
    home(world, "pk-tl-N1B")
    assert decide(world, "pk-tl-N2B", stop=True) == ("halt", ("LADDER_UNITS_STOP", "a stop was asked of the ladder before this arm launched"))
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


def test_a_launched_arm_runs_again_on_its_own_record(world):
    assert decide(world, "pk-tl-NT")[0] == "exec"
    d = world["arms"]["pk-tl-NT"]["state_dir"]
    json.dump({"launches": 2}, open(os.path.join(d, "state.json"), "w"))
    kind, argv = decide(world, "pk-tl-NT", seed2="unreadable", bal=None, guard_up=False, free_l4=0, stop=True)
    assert kind == "exec" and argv[argv.index("--name") + 1] == "pk-tl-NT"


def test_the_guard_is_base2s_guard_on_the_ladders_folders(world):
    assert G.decide_guard(NOW, COMMIT, launch_dir=world["ld"], seed2=lambda rows=None: None) == ("wait", "the ladder has not been admitted")
    assert decide(world, "pk-tl-N1B", bal=812.5)[0] == "exec"
    kind, argv = G.decide_guard(NOW, COMMIT, launch_dir=world["ld"], seed2=lambda rows=None: None)
    assert kind == "exec" and argv[2] == "tools/colab_chat/base2_units_guard.py"
    assert argv[argv.index("--campaign") + 1] == "token-ladder" and argv[argv.index("--stop-at") + 1] == "740"
    assert argv[argv.index("--start-balance") + 1] == "812.5"
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--state-dir"] == [a["state_dir"] for a in G.ARMS.values()]
    assert G.guard_running(lambda: [argv]) and not G.guard_running(lambda: [S2.guard_argv(1.0)])
