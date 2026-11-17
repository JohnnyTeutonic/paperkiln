"""The token ladder's jobs files (tools/colab_chat/make_ladder.py): each arm is base 2's arm N or S with the steps, the
marks, the seed and the names changed, and nothing else. Claude Fable 5.2, 17 Nov 2026."""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "colab_chat"))
import make_base2 as M  # noqa: E402
import make_ladder as L  # noqa: E402

ARMS = sorted(L.RUNGS)


def test_files_are_what_the_generator_writes_and_base2s_are_untouched():
    assert L.main(["--check"]) == 0
    assert M.main(["--check"]) == 0


def test_the_arms_and_their_rungs():
    assert set(L.FIRST_SEED_ARMS) == {"N1B", "S1B", "N2B", "S2B", "NT", "ST"}
    assert set(L.RUNGS) == set(L.FIRST_SEED_ARMS) | {a + "s50" for a in L.FIRST_SEED_ARMS}
    assert L.STEPS == {"1B": 60000, "2B": 122000, "T": 325000}
    # the low rungs end at base 2's own marks, so they have read what N and S had read there
    assert {60000, 122000} <= set(M.MARKS["N"])
    assert L.marks(60000) == [10000, 58500, 60000]
    assert L.marks(122000) == [10000, 119000, 122000]
    assert L.marks(325000) == [10000, 317000, 325000]
    assert L.marks(250000)[1:] == M.MARKS["N"][-2:]   # the late-mark rule gives base 2's own 244,000


def _renamed(obj, name, recipe):
    """obj with the arm's names put back to the recipe arm's."""
    t = json.dumps(obj)
    for a, b in ((f"pt_base2_{name}", f"pt_base2_{recipe}"), (f"base2_{name}", f"base2_{recipe}"),
                 (f"ckpt-{name}", f"ckpt-{recipe}"), (f"rung_{name}_s", f"rung_{recipe}_s"),
                 (f"BASE2_{name}", f"BASE2_{recipe}")):
        t = t.replace(a, b)
    return json.loads(t)


@pytest.mark.parametrize("name", ARMS)
def test_an_arm_differs_from_its_recipe_in_steps_marks_seed_names_and_turn_only(name):
    r = L.RUNGS[name]
    got, ref = L.arm(name), M.arm(r["recipe"])
    assert set(got) - set(ref) <= {"store_release_offset_min"}
    # top-level keys other than the comment, the record, the turn and the jobs are the recipe arm's
    for k in set(ref) | set(got):
        if k in ("_comment", "_base2", "jobs", "store_release_offset_min"):
            continue
        assert got[k] == ref[k], k
    assert got.get("store_release_offset_min", 0) == r["offset_min"]
    g = _renamed(got, name, r["recipe"])
    gj = {j["name"]: j for j in g["jobs"]}
    rj = {j["name"]: j for j in ref["jobs"]}
    assert gj["pins"] == rj["pins"]
    gt, rt = gj[f"pt_base2_{r['recipe']}"], rj[f"pt_base2_{r['recipe']}"]
    gs, rs = copy.deepcopy(gt.pop("spec")), copy.deepcopy(rt.pop("spec"))
    assert gt.pop("snapshot_steps") == L.marks(r["steps"]) and rt.pop("snapshot_steps") == M.MARKS[r["recipe"]]
    assert gt == rt
    assert gs["train"].pop("steps") == r["steps"] and rs["train"].pop("steps") == M.STEPS[r["recipe"]]
    assert gs["train"].pop("seed") == r["seed"] and rs["train"].pop("seed") == 1
    assert gs == rs                      # shape, corpus, rate, warm-up, cosine floor, tail fraction, batch, precision
    assert gs["train"]["lr_schedule"] == {"warmup_steps": 4000, "decay": "cosine", "min_lr_frac": 0.1}
    assert gs["data"]["anneal"]["from_fraction"] == M.TAIL_FROM


@pytest.mark.parametrize("name", ARMS)
def test_every_mark_has_its_rung_and_the_rung_is_base2s_at_that_step(name):
    r = L.RUNGS[name]
    got = L.arm(name)
    job = f"pt_base2_{name}"
    mk = L.marks(r["steps"])
    panels = [(p, f"{M.DATA}/panel_{p}.ids") for p in M.PANELS]
    rungs = [j for j in got["jobs"] if j["name"].startswith("rung_")]
    assert [j["name"] for j in rungs] == [f"rung_{name}_s{m}" for m in mk]
    for j, m in zip(rungs[:-1], mk[:-1]):
        assert j == M.rung(job, m, name, M.SHAPE_D1, panels, seed=r["seed"])
    assert rungs[-1] == M.rung(job, mk[-1], name, M.SHAPE_D1, panels, of=False, after=[job], seed=r["seed"])
    assert got["_base2"]["positions"] == r["steps"] * 16 * 1024
    assert got["_base2"]["first_tail_step"] == int(0.85 * r["steps"]) + 1


def test_a_rung_at_base2s_own_steps_is_base2s_arm(monkeypatch):
    """The construction's own check: at 250,000 steps, seed 1 and slot 1, the ladder's arm is make_base2's arm N."""
    monkeypatch.setitem(L.RUNGS, "NX", {"recipe": "N", "rung": "X", "steps": 250000, "seed": 1, "slot": None,
                                        "offset_min": 0, "launched_by_this_registration": False})
    monkeypatch.setattr(L, "marks", lambda steps: list(M.MARKS["N"]))
    got = _renamed(L.arm("NX"), "NX", "N")
    ref = M.arm("N")
    for k in ("_comment", "_base2"):
        got.pop(k), ref.pop(k)
    assert got == ref
    assert list(got) == list(ref)


def test_slots_release_turns_and_what_is_launched():
    assert L.SLOTS == {1: ["N1B", "N2B"], 2: ["S1B", "S2B"], 3: ["NT"]}
    assert L.LAUNCHED_ARMS == ["N1B", "N2B", "S1B", "S2B", "NT"]
    # seed 2's arms held the turns 0 and 20
    seed2 = {M.arm(a).get("store_release_offset_min", 0) for a in ("N2", "S2")}
    assert seed2 == {0, 20} and sorted(L.SLOT_OFFSET_MIN.values()) == [0, 20, 40]      # seed 2 has ended before any ladder arm
    for name, r in L.RUNGS.items():
        assert L.arm(name).get("store_release_offset_min", 0) == r["offset_min"]
        assert r["launched_by_this_registration"] == (name in L.LAUNCHED_ARMS)
    assert [L.RUNGS[a]["offset_min"] for a in L.LAUNCHED_ARMS] == [0, 0, 20, 20, 40]
    assert not M.block_streams(M.SECOND_SEED) & M.block_streams(1)
    assert all(L.RUNGS[a + "s50"]["seed"] == M.SECOND_SEED for a in L.FIRST_SEED_ARMS)
    # three writers on the repository at the most, as base 2 budgeted: 3 x 24 + 3 x 12 commits an hour under the hub's 128
    assert 3 * (M.COURIER_COMMIT_BUDGET + M.HOME_COMMIT_BUDGET) <= M.HUB_COMMITS_PER_HOUR


def test_store_branches_and_jobs_are_named_for_the_arm_alone():
    seen = set()
    for name in ARMS:
        got = L.arm(name)
        tr = next(j for j in got["jobs"] if j["name"] == f"pt_base2_{name}")
        assert tr["store_branch"] == f"ckpt-{name}" and tr["model_name"] == f"base2_{name}"
        assert tr["store_branch"] not in seen
        seen.add(tr["store_branch"])
        assert got["store_inputs"] == M.arm(L.RUNGS[name]["recipe"])["store_inputs"]
    assert not seen & {f"ckpt-{a}" for a in ("N", "S", "C", "N2", "S2")}
