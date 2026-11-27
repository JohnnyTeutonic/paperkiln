"""The transfer programme's Stage 3(a) jobs files (tools/colab_chat/make_tp.py) against the token ladder's reference rungs:
only the registered differences. Claude Opus 5.5, 28 Nov 2026."""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools", "colab_chat")
sys.path.insert(0, TOOLS)
import make_base2 as M  # noqa: E402
import make_ladder as L  # noqa: E402
import make_tp as P  # noqa: E402


def test_files_on_disk_are_current():
    assert P.main(["--check"]) == 0
    assert L.main(["--check"]) == 0


def test_only_rendered_runs_have_files():
    have = sorted(f for f in os.listdir(TOOLS) if f.startswith("tp_R") and f.endswith(".json"))
    assert have == sorted(f"tp_{r}.json" for r in P.BUNDLES)


def test_only_the_registered_differences():
    for run in P.BUNDLES:
        ref_name = P.RUNS[run]
        got = json.load(open(os.path.join(TOOLS, f"tp_{run}.json")))
        ref = json.load(open(os.path.join(TOOLS, f"ladder_{ref_name}.json")))
        names = [j["name"] for j in got["jobs"]]
        assert names == ["pins", "tp_data", f"pt_tp_{run}", f"rung_TP_{run}_s60000"], names
        g = {j["name"]: j for j in got["jobs"]}
        r = {j["name"]: j for j in ref["jobs"]}
        assert g["pins"] == r["pins"]
        # the training job: the reference rung's but for names, after, marks and the two corpus files
        t, rt = copy.deepcopy(g[f"pt_tp_{run}"]), copy.deepcopy(r[f"pt_base2_{ref_name}"])
        assert t["spec"]["data"].pop("corpus") == "/content/chat/data/TP.main.ids"
        assert t["spec"]["data"]["anneal"].pop("corpus") == "/content/chat/data/TP.tail.ids"
        assert rt["spec"]["data"].pop("corpus") == "/content/chat/data/N.main.ids"
        assert rt["spec"]["data"]["anneal"].pop("corpus") == "/content/chat/data/N.tail.ids"
        assert t.pop("after") == ["pins", "tp_data"] and rt.pop("after") == ["pins"]
        assert t.pop("snapshot_steps") == P.MARKS and rt.pop("snapshot_steps") == [10000, 58500, 60000]
        assert (t.pop("name"), t.pop("model_name"), t.pop("store_branch")) == (f"pt_tp_{run}", f"tp_{run}", f"ckpt-tp-{run}")
        for k in ("name", "model_name", "store_branch"):
            rt.pop(k)
        assert t["spec"].pop("name") == f"tp_{run}" and t["spec"].pop("out_dir") == f"/content/chat/pt_tp_{run}"
        rt["spec"].pop("name")
        rt["spec"].pop("out_dir")
        assert t == rt
        assert t["spec"]["train"]["steps"] == 60000 and t["spec"]["train"]["seed"] == L.RUNGS[ref_name]["seed"]
        # the end rung: the reference end rung with the run's names
        e = json.dumps(g[f"rung_TP_{run}_s60000"])
        assert e == json.dumps(r[f"rung_{ref_name}_s60000"]).replace(f"pt_base2_{ref_name}", f"pt_tp_{run}") \
            .replace(f"BASE2_{ref_name}", f"BASE2_TP_{run}").replace(f"rung_{ref_name}_", f"rung_TP_{run}_")
        # the data job: pinned bundle, the run's name, after pins, fails the campaign on failure
        d = g["tp_data"]
        assert d["after"] == ["pins"] and d["halt_on_failure"] is True
        b = f"tp_bundle_{P.BUNDLES[run][:12]}.tar.gz"
        assert f"sha256sum /content/chat/upload/{b} " in d["cmd"][2] and f"grep -q '^{P.BUNDLES[run]} '" in d["cmd"][2]
        assert d["cmd"][2].endswith(f"tp_data_vm.sh {run} {{out}} {b}")
        # the switches and inputs: the reference's but for the release turn
        for k in ("store_inputs", "max_parallel", "max_parallel_light", "store_push_cap_gb", "store_release_every_min",
                  "store_stall_halt", "store_stall_min", "store_unverified_halt_min", "store_no_progress_min",
                  "store_no_progress_launches"):
            assert got.get(k) == ref.get(k), k
        assert got["store_release_offset_min"] == 40


def test_marks_are_saves_and_log_spaced():
    assert all(m % M.CHECKPOINT_EVERY == 0 for m in P.MARKS)
    assert P.MARKS == sorted(set(P.MARKS)) and P.MARKS[0] == 1000 and P.MARKS[-1] == 60000
    assert len(P.MARKS) >= 13


def test_trainer_is_the_ladders():
    # no source outside the campaign's three files differs from the ladder's launch commit
    out = subprocess.run(["git", "-C", HERE, "diff", "--name-status", "672501bb1d09f707fdd20a5c7280120940734327", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    rows = [r.split("\t") for r in out if r.strip()]
    own = {"tools/colab_chat/make_tp.py", "tests/test_tp_jobs.py"} | {f"tools/colab_chat/tp_{r}.json" for r in P.BUNDLES}
    assert all(s == "A" and p in own for s, p in rows), rows
