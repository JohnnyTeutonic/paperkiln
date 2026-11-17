"""E9's D arms are the token ladder's S1B (or S1Bs50) with only the scenario stream swapped.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(HERE, "..", "tools", "colab_chat")
sys.path.insert(0, CC)
import make_base2 as M   # noqa: E402
import make_e9 as E9     # noqa: E402
import make_ladder as L  # noqa: E402


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path, x


def test_every_leaf_is_the_reference_arms_but_for_names_and_corpus():
    for name, a in E9.ARMS.items():
        got = E9.arm(name)
        ref = L.arm(a["ref"])
        g = dict(leaves({k: v for k, v in got.items() if k not in ("_comment", "_base2")}))
        r = dict(leaves({k: v for k, v in ref.items() if k not in ("_comment", "_base2")}))
        g.pop(("store_release_offset_min",), None)
        r.pop(("store_release_offset_min",), None)
        assert set(g) == set(r), name
        subs = E9.corpus_subs(a["base"])
        changed = 0
        for p, rv in r.items():
            gv = g[p]
            if p == ("store_release_offset_min",):
                continue
            if gv == rv:
                continue
            changed += 1
            assert isinstance(rv, str), (name, p)
            assert gv == subs.get(rv, rv.replace(a["ref"], name)), (name, p, rv, gv)
            assert a["ref"] in rv or rv in subs, (name, p, rv)
        assert changed > 0
        # the corpus is the level's, and nothing of S's corpus remains
        text = json.dumps(got)
        for v in subs:
            assert json.dumps(v) not in text, (name, v)
        assert a["ref"] not in json.dumps({k: v for k, v in got.items() if k not in ("_base2", "_comment")}), name


def test_training_spec_steps_marks_seed_and_schedule_are_s1b_s():
    for name, a in E9.ARMS.items():
        got, ref = E9.arm(name), L.arm(a["ref"])
        tg = next(j for j in got["jobs"] if "spec" in j)
        tr = next(j for j in ref["jobs"] if "spec" in j)
        assert tg["spec"]["train"] == tr["spec"]["train"], name
        assert tg["spec"]["model"] == tr["spec"]["model"] if "model" in tr["spec"] else True
        assert tg["snapshot_steps"] == tr["snapshot_steps"]
        for k in ("steps", "marks", "seed", "positions", "shape", "params", "first_tail_step", "last_main_step"):
            assert got["_base2"][k] == ref["_base2"][k], (name, k)
        assert got["_base2"]["corpus"] == a["base"] and got["_base2"]["e9_level"] == E9.LEVEL[a["base"]]
        assert len(got["jobs"]) == len(ref["jobs"])


def test_names_branches_and_turns_are_unique_and_not_the_ladders():
    seen = set()
    for name in E9.ARMS:
        got = E9.arm(name)
        tr = next(j for j in got["jobs"] if "spec" in j)
        assert tr["store_branch"] == f"ckpt-{name}" and tr["name"] == f"pt_base2_{name}"
        assert tr["store_branch"] not in seen
        seen.add(tr["store_branch"])
    ladder_branches = {next(j for j in L.arm(n)["jobs"] if "spec" in j)["store_branch"] for n in L.RUNGS}
    assert not (seen & ladder_branches)
    first = [E9.arm(n).get("store_release_offset_min", 0) for n in ("D2", "D3", "D4")]
    assert sorted(first) == [0, 20, 40]


def test_files_are_current_and_the_ladders_unchanged():
    assert E9.main(["--check"]) == 0
    assert L.main(["--check"]) == 0
    assert M.main(["--check"]) == 0 if hasattr(M, "main") else True
