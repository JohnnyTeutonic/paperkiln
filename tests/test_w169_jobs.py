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
        if k in ("_comment", "_base2", "jobs", "store_stall_min", "store_no_progress_min", "max_parallel_light"):
            continue
        assert got[k] == ref[k], k
    assert got["store_stall_min"] == 240 and got["store_no_progress_min"] == 420
    assert ref.get("store_stall_min") == 120 and ref.get("store_no_progress_min") == 300
    g = _renamed(got)
    gj = {j["name"]: j for j in g["jobs"]}
    rj = {j["name"]: j for j in ref["jobs"]}
    assert set(gj) == set(rj) | {"loss_guard", "diverge_10k"}
    gj.pop("loss_guard"), gj.pop("diverge_10k")
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


def test_whole_file_against_N_by_path_in_order():
    """The generated file against N's file on disk, path by path, jobs in order: only the allowlisted paths differ."""
    got = _renamed(json.load(open(os.path.join(ROOT, "tools", "colab_chat", "w169_W169.json"))))
    ref = json.load(open(os.path.join(ROOT, "tools", "colab_chat", "base2_N.json")))
    names = [j["name"] for j in got["jobs"]]
    assert names == [j["name"] for j in ref["jobs"]] + ["loss_guard", "diverge_10k"] and len(set(names)) == len(names)
    extra = got["jobs"][-2:]
    assert all(j["halt_on_failure"] and j["light"] and W.GUARD_SCRIPT in j["cmd"] for j in extra)
    assert extra[1]["after"] == ["rung_W169_s10000".replace("W169", "N")]   # renamed back above
    got["jobs"] = got["jobs"][:-2]
    assert got.pop("max_parallel_light") == 2 and ref.pop("max_parallel_light") == 1
    allowed = {("_comment",), ("_base2",), ("store_stall_min",), ("store_no_progress_min",),
               ("jobs", "pt_base2_N", "spec", "arch", "custom"), ("jobs", "pt_base2_N", "spec", "train", "lr")}
    for name in names:
        if name.startswith("rung_"):
            allowed |= {("jobs", name, "timeout_s"), ("jobs", name, "cmd", 3)}
    diffs = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                walk(a.get(k), b.get(k), path + (k,))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                key = x["name"] if isinstance(x, dict) and "name" in x and path == ("jobs",) else i
                walk(x, y, path + (key,))
        elif a != b:
            diffs.append(path)
    walk(got, ref, ())
    assert diffs and all(any(d[:len(p)] == p for p in allowed) for d in diffs), [d for d in diffs if not any(d[:len(p)] == p for p in allowed)]


def test_embedded_spec_and_record():
    a, n = W.arm(), M.arm("N")
    t = next(j for j in a["jobs"] if j["name"] == "pt_base2_W169")
    assert t["spec"] == W.spec()
    rb, nb = a["_base2"], n["_base2"]
    changed = {k for k in set(rb) | set(nb) if rb.get(k) != nb.get(k)}
    assert changed == {"arm", "recipe_of", "seed", "shape", "params", "tokens_per_parameter"}, changed
    assert rb["recipe_of"] == "N" and rb["seed"] == 1 and rb["shape"] == "W169" and rb["params"] == W.PARAMS


def test_guard_loss_and_diverge(tmp_path):
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("g", os.path.join(ROOT, "tools", "colab_chat", "w169_guard_vm.py"))
    g = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(g)
    chat = tmp_path / "chat"
    (chat / "job").mkdir(parents=True)
    ev = chat / "job" / "events.jsonl"
    ev.write_text('{"event": "step", "step": 1, "loss": 9.1}\n{"event": "step", "step": 2, "loss": 8.7}\n')
    (chat / "STATUS_job").write_text("FINISHED")
    assert g.main(["loss", "--chat", str(chat), "--job", "job", "--out", str(tmp_path / "o1"), "--interval", "0"]) == 0
    ev.write_text('{"event": "step", "step": 1, "loss": 9.1}\n{"event": "step", "step": 2, "loss": null}\n')
    (chat / "STATUS_job").write_text("RUNNING")
    assert g.main(["loss", "--chat", str(chat), "--job", "job", "--out", str(tmp_path / "o2"), "--interval", "0"]) == 1
    assert json.load(open(tmp_path / "o2" / "guard.json"))["first"]["step"] == 2
    # a resume rewrites the stream shorter (truncate_events_after); a null loss written after that must still halt
    import threading
    import time as _t
    long = "".join(f'{{"event": "step", "step": {s}, "loss": 5.0}}\n' for s in range(1, 201))
    ev.write_text(long)
    (chat / "STATUS_job").write_text("RUNNING")
    rc = {}
    th = threading.Thread(target=lambda: rc.setdefault("rc", g.main(["loss", "--chat", str(chat), "--job", "job", "--out",
                                                                  str(tmp_path / "o4"), "--interval", "0.05"])))
    th.start()
    _t.sleep(0.5)
    ev.write_text('{"event": "step", "step": 1, "loss": 5.0}\n')
    _t.sleep(0.5)
    with open(ev, "a") as f:
        f.write('{"event": "step", "step": 2, "loss": null}\n')
    th.join(10)
    assert rc.get("rc") == 1 and json.load(open(tmp_path / "o4" / "guard.json"))["first"]["step"] == 2
    # a partial last line is not read until it is complete
    ev.write_text('{"event": "step", "step": 1, "loss": 9.1}\n{"event": "step", "step": 2, "lo')
    assert g.scan(str(ev), 0)[0] == []
    rung = tmp_path / "rung.json"
    for fw, rc in ((3.70, 0), (g.N_FINEWEB_10K + g.MARGIN - 1e-9, 0), (g.N_FINEWEB_10K + g.MARGIN + 1e-6, 1), (None, 1)):
        rung.write_text(json.dumps({"panels": {"fineweb": fw}}))
        assert g.main(["diverge", "--rung", str(rung), "--out", str(tmp_path / "o3")]) == rc, fw
    assert g.N_FINEWEB_10K == 3.750180445611477


def test_rehearsal_runs_W169s_spec_on_Ns_inputs():
    r, a = W.rehearsal(), W.arm()
    t = next(j for j in r["jobs"] if j["name"] == W.REH_JOB)
    s, w = t["spec"], W.spec()
    assert s["arch"] == w["arch"] and s["data"] == w["data"] and s["train"]["lr"] == w["train"]["lr"]
    assert s["train"]["steps"] == 1000 and s["train"]["checkpoint_every"] == 250
    assert t["pod_loss_drill_step"] == 250 and t["snapshot_steps"] == [750] and t["store_branch"] == "ckpt-W169reh"
    assert r["store_inputs"] == a["store_inputs"]
    assert r["store_release_grace_s"] == 600 and r["store_push_cap_gb"] == 40
    rungs = [j for j in r["jobs"] if j["name"].startswith("rung_")]
    assert [j["name"] for j in rungs] == ["rung_W169reh_s750", "rung_W169reh_s1000"]
    assert rungs[0]["timeout_s"] == W.RUNG_TIMEOUT_S and rungs[0].get("exclusive")
    assert rungs[1]["cmd"][:4] == ["timeout", "-k", "60", str(W.RUNG_TIMEOUT_S)] and "--export-gguf" in rungs[1]["cmd"]
    assert all(W.SHAPE_FILE in j["cmd"] for j in rungs)
    assert os.path.exists(os.path.join(ROOT, "tools", "colab_chat", "w169_watch.py"))


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
