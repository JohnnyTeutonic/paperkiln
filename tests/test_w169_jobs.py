"""The 169M base's jobs file (tools/colab_chat/make_w169.py): base 2's arm N with the shape, the peak rate, the names, the
rungs' time limits and two store halts changed, and nothing else. Claude Opus 5.5, 26 Nov 2026."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "colab_chat"))
import make_base2 as M  # noqa: E402
import make_w169 as W  # noqa: E402


def _renamed(obj):
    t = json.dumps(obj)
    for a, b in (("pt_base2_W169", "pt_base2_N"), ("base2_W169", "base2_N"), ("ckpt-W169", "ckpt-N"),
                 ("rung_W169_s", "rung_N_s"), ("BASE2_W169", "BASE2_N"), (W.SHAPE_FILE, M.SHAPE_D1)):
        t = t.replace(a, b)
    return json.loads(t)


def test_files_are_what_the_generator_writes_and_base2s_are_untouched():
    assert W.main(["--check"]) == 0
    assert M.main(["--check"]) == 0


def test_shape_and_parameters():
    assert W.PARAMS == M.params(1024, 12, 2752) == 168588288
    s = W.spec()
    assert s["arch"]["custom"] == {"d": 1024, "layers": 12, "heads": 16, "d_ff": 2752}
    assert s["arch"]["custom"]["d"] % s["arch"]["custom"]["heads"] == 0
    assert {k: v for k, v in s["arch"].items() if k != "custom"} == {k: v for k, v in M.arch_d1().items() if k != "custom"}


def test_spec_differs_from_N_in_shape_rate_and_names_only():
    s, n = W.spec(), M.spec("N")
    assert s["train"]["lr"] == 0.0003 and n["train"]["lr"] == 0.0005
    s2 = json.loads(json.dumps(s))
    s2["arch"], s2["train"]["lr"] = n["arch"], n["train"]["lr"]
    s2["name"], s2["out_dir"] = n["name"], n["out_dir"]
    assert s2 == n
    assert s["train"]["steps"] == 250000 and s["train"]["batch"] * s["train"]["accum"] == M.ROWS == 16
    assert s["data"] == n["data"]


def test_arm_differs_from_N_where_registered_only():
    got, ref = W.arm(), M.arm("N")
    assert set(got) == set(ref)
    for k in set(ref) | set(got):
        if k in ("_comment", "_base2", "jobs", "store_stall_min", "store_no_progress_min"):
            continue
        assert got[k] == ref[k], k
    assert got["store_stall_min"] == 240 and got["store_no_progress_min"] == 420
    assert ref.get("store_stall_min") == 120 and ref.get("store_no_progress_min") == 300
    g = _renamed(got)
    gj = {j["name"]: j for j in g["jobs"]}
    rj = {j["name"]: j for j in ref["jobs"]}
    assert set(gj) == set(rj)
    assert gj["pins"] == rj["pins"]
    gt, rt = gj["pt_base2_N"], rj["pt_base2_N"]
    gs, rs = gt.pop("spec"), rt.pop("spec")
    assert gt == rt
    assert gs["train"]["lr"] == 0.0003
    for name in gj:
        if name.startswith("rung_"):
            a, b = gj[name], rj[name]
            if "timeout_s" in b:
                assert a["timeout_s"] == W.RUNG_TIMEOUT_S and b["timeout_s"] == M.RUNG_TIMEOUT_S
                a = dict(a, timeout_s=b["timeout_s"])
            if a["cmd"][0] == "timeout":
                assert a["cmd"][3] == str(W.RUNG_TIMEOUT_S)
                a = dict(a, cmd=[*a["cmd"][:3], b["cmd"][3], *a["cmd"][4:]])
            assert a == b, name


def test_marks_positions_and_record():
    a = W.arm()
    r = a["_base2"]
    assert r["marks"] == M.MARKS["N"] == [10000, 60000, 122000, 244000, 250000]
    assert r["positions"] == 250000 * 16 * 1024 == 4096000000
    assert r["params"] == 168588288 and r["corpus"] == "N" and r["seed"] == 1
    rungs = [j for j in a["jobs"] if j["name"].startswith("rung_")]
    assert [int(j["name"].rsplit("_s", 1)[1]) for j in rungs] == r["marks"]
    assert all(W.SHAPE_FILE in j["cmd"] for j in rungs)


def test_rung_shape_file_matches_the_training_arch():
    rs, s = W.rung_shape(), W.spec()
    assert rs["arch"] == s["arch"] and rs["data"]["T"] == s["data"]["T"] == 1024
    assert rs["_shape"]["params"] == W.PARAMS
