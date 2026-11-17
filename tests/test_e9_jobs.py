"""E9's D arms are the token ladder's S1B (or S1Bs50) with only the scenario stream swapped.

The allowed differences are an explicit table of JSON paths with their exact old and new values (ALLOWED); every
other leaf must be equal in value and type, the release turn and the metadata are asserted per arm, and the
ladder's reference files are compared with their bytes at the ladder's pretraining commit.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(HERE, "..", "tools", "colab_chat")
sys.path.insert(0, CC)
import make_base2 as M   # noqa: E402
import make_e9 as E9     # noqa: E402
import make_ladder as L  # noqa: E402

LADDER_COMMIT = "672501bb"
D = "/content/chat/data"
P = "/content/chat/pt_base2_"


def allowed():
    """(path, old, new) templates: {r} the reference arm, {n} the D arm, {c} its corpus."""
    t = [(("store_inputs", 1, "prefix"), "corpus/S", "corpus/{c}"),
         (("store_inputs", 1, "files", 0), "S.main.ids", "{c}.main.ids"),
         (("store_inputs", 1, "files", 1), "S.tail.ids", "{c}.tail.ids"),
         (("jobs", 0, "cmd", 3), "arm-S", "arm-{c}"),
         (("jobs", 1, "name"), "pt_base2_{r}", "pt_base2_{n}"),
         (("jobs", 1, "model_name"), "base2_{r}", "base2_{n}"),
         (("jobs", 1, "store_branch"), "ckpt-{r}", "ckpt-{n}"),
         (("jobs", 1, "spec", "name"), "base2_{r}", "base2_{n}"),
         (("jobs", 1, "spec", "data", "corpus"), D + "/S.main.ids", D + "/{c}.main.ids"),
         (("jobs", 1, "spec", "data", "anneal", "corpus"), D + "/S.tail.ids", D + "/{c}.tail.ids"),
         (("jobs", 1, "spec", "out_dir"), P + "{r}", P + "{n}")]
    for j, step in ((2, 10000), (3, 58500)):
        t += [(("jobs", j, "name"), "rung_{r}_s%d" % step, "rung_{n}_s%d" % step),
              (("jobs", j, "cmd", 9), P + "{r}/rung_%d/model.safetensors" % step, P + "{n}/rung_%d/model.safetensors" % step),
              (("jobs", j, "cmd", 11), P + "{r}", P + "{n}"),
              (("jobs", j, "cmd", 15), "pt_base2_{r}", "pt_base2_{n}"),
              (("jobs", j, "cmd", 17), "BASE2_{r}", "BASE2_{n}"),
              (("jobs", j, "rung_of", "job"), "pt_base2_{r}", "pt_base2_{n}")]
    t += [(("jobs", 4, "name"), "rung_{r}_s60000", "rung_{n}_s60000"),
          (("jobs", 4, "after", 0), "pt_base2_{r}", "pt_base2_{n}"),
          (("jobs", 4, "cmd", 13), P + "{r}/model.safetensors", P + "{n}/model.safetensors"),
          (("jobs", 4, "cmd", 15), P + "{r}", P + "{n}"),
          (("jobs", 4, "cmd", 19), "pt_base2_{r}", "pt_base2_{n}"),
          (("jobs", 4, "cmd", 21), "BASE2_{r}", "BASE2_{n}")]
    return t


OFFSETS = {"D2": 0, "D3": 20, "D4": 40, "D4ops": 0, "D2s50": 40, "D3s50": 40, "D4s50": 40}
REFS = {"D2": "S1B", "D3": "S1B", "D4": "S1B", "D4ops": "S1B", "D2s50": "S1Bs50", "D3s50": "S1Bs50", "D4s50": "S1Bs50"}
LEVELS = {"D2": "L2", "D3": "L3", "D4": "L4", "D4ops": "L4-ops"}


def leaves(x, path=()):
    if isinstance(x, dict):
        if not x:
            yield path, "{}"
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        if not x:
            yield path, "[]"
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path, x


def test_the_arm_set_is_the_registered_one():
    assert set(E9.ARMS) == set(OFFSETS)
    for n, a in E9.ARMS.items():
        assert a["ref"] == REFS[n]


def test_only_the_allowed_paths_change_and_exactly_as_allowed():
    for name, a in E9.ARMS.items():
        got, ref = E9.arm(name), L.arm(REFS[name])
        assert list(got)[:2] == ["_comment", "_base2"]
        g = dict(leaves({k: v for k, v in got.items() if k not in ("_comment", "_base2", "store_release_offset_min")}))
        r = dict(leaves({k: v for k, v in ref.items() if k not in ("_comment", "_base2", "store_release_offset_min")}))
        assert list(g) == list(r), name                      # same paths, same order
        c = name[:-len(L.SECOND)] if name.endswith(L.SECOND) else name
        table = {p: (o.format(r=REFS[name]), w.format(n=name, c=c)) for p, o, w in allowed()}
        changed = {p for p in r if not (type(g[p]) is type(r[p]) and g[p] == r[p])}
        assert changed == set(table), (name, sorted(changed ^ set(table), key=str))
        for p, (old, new) in table.items():
            assert r[p] == old and g[p] == new, (name, p, r[p], g[p])


def test_release_turns_and_metadata_per_arm():
    for name in E9.ARMS:
        got, ref = E9.arm(name), L.arm(REFS[name])
        assert got.get("store_release_offset_min", 0) == OFFSETS[name], name
        mg, mr = got["_base2"], ref["_base2"]
        c = name[:-len(L.SECOND)] if name.endswith(L.SECOND) else name
        assert set(mg) == set(mr) | {"e9_level", "e9_ref_arm"}
        for k in mr:
            want = {"arm": name, "corpus": c, "slot": None}.get(k, mr[k])
            assert mg[k] == want and type(mg[k]) is type(want), (name, k)
        assert mg["e9_level"] == LEVELS[c] and mg["e9_ref_arm"] == REFS[name]


def test_files_are_current_and_the_ladders_are_the_pinned_commits():
    assert E9.main(["--check"]) == 0
    assert L.main(["--check"]) == 0
    root = os.path.normpath(os.path.join(HERE, ".."))
    for ref in set(REFS.values()):
        rel = "tools/colab_chat/ladder_%s.json" % ref
        pinned = subprocess.run(["git", "show", "%s:%s" % (LADDER_COMMIT, rel)], cwd=root, check=True,
                                capture_output=True).stdout.decode("utf-8")
        assert pinned == M.text(L.arm(ref)), ref
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            assert f.read() == pinned, ref
