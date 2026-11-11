"""runner_multi.py's --ckpt-store hf option (base 2), end to end on the fake colab CLI.

1. Without the option the runner makes exactly the calls it made at the trainer commit 5506e6df: the committed
   runner and this one drive tests/runner_multi_equiv/fakecolab.py through run_eq.py's scenarios (several reclaim
   patterns) and a rung campaign, and every colab call, the log, state.json and the artifact tree must be equal.
2. With the option (fakecolab_store.py: the fake VM plus the real restore and courier on a "dir:" store): no
   checkpoint of a training job crosses the home link in either direction, the job resumes on the replacement VM
   from the store's generation, the mark snapshot reaches the rung job on the new VM and comes home from the store,
   the token is sent as a file, appears in no call, log or state, and is removed from the VM before the stop.
Run on ext4 (pytest's tmp_path), not OneDrive. Each scenario moves 46 MB files, so the file takes a few minutes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
EQ = ROOT / "tests" / "runner_multi_equiv"
sys.path.insert(0, str(EQ))
sys.path.insert(0, str(CC))
os.environ.setdefault("FAKE_ROOT", str(EQ))  # fakecolab reads it at import; only blob() and BIG are used here
import fakecolab  # noqa: E402
import hf_store  # noqa: E402
import run_eq  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the fake colab CLI is a POSIX shell wrapper")
BASE = "5506e6dfd5ec630b4b49fae0592329cc06799070"  # the trainer commit the branch was cut from
TOKEN = "hf_FAKEtokenFAKEtokenFAKEtoken0123"


@pytest.fixture(scope="module")
def old_runner(tmp_path_factory):
    """tools/colab_chat as committed at BASE (the runner and the modules it imports), in a folder of its own."""
    d = tmp_path_factory.mktemp("old")
    r = subprocess.run(f"git -C '{ROOT}' archive {BASE} tools/colab_chat | tar -x -C '{d}'", shell=True, capture_output=True)
    if r.returncode != 0 or not (d / "tools" / "colab_chat" / "runner_multi.py").exists():
        pytest.skip(f"commit {BASE[:8]} is not in this checkout")
    return str(d / "tools" / "colab_chat" / "runner_multi.py")


def same(old, new, work, scenario, jobs=run_eq.JOBS):
    a = run_eq.run(old, str(work / "w"), scenario, jobs=jobs)  # the same folder for both: exec code carries paths
    b = run_eq.run(str(CC / "runner_multi.py"), str(work / "w"), scenario, jobs=jobs)
    shutil.rmtree(work, ignore_errors=True)  # each run leaves several hundred MB
    assert [a["rc"], b["rc"]] == [0, 0], (a["stderr"], b["stderr"])
    assert a["calls"].count("\n") > 50
    assert a["calls"] == b["calls"]
    assert a["log"] == b["log"]
    for r in (a, b):  # the two wall-clock fields of a rung snapshot's record
        for e in (r["state"].get("rung_snapshots") or {}).values():
            e.update({k: bool(e[k]) for k in ("fetched", "deleted") if k in e})
    assert a["state"] == b["state"]
    assert a["art"] == b["art"]
    return a


@pytest.mark.parametrize("reclaim", [[6, 14], [4, 9, 17], [3, 5, 8, 11], []])
def test_without_the_option_every_call_is_as_before(old_runner, tmp_path, reclaim):
    a = same(old_runner, None, tmp_path, {"fail_new": [1], "reclaim_at": reclaim})
    assert "store" not in a["log"] and "store_token" not in a["calls"]


def rung(name, step, **kw):
    return {"name": name, "after": "data", "rung_of": {"job": "pt", "step": step}, "light": True, "cmd": ["r"],
            "outputs": ["rung.json"], **kw}


def test_without_the_option_a_rung_campaign_is_as_before(old_runner, tmp_path):
    jobs = {"jobs": [{"name": "data", "cmd": ["x"], "outputs": ["out.txt"]},
                     {"name": "pt", "model_name": "m", "after": "data", "snapshot_steps": [100, 200], "spec": {}},
                     rung("rung_100", 100), rung("rung_200", 200)]}
    a = same(old_runner, None, tmp_path, {"reclaim_at": [9], "rung_start_at": 11}, jobs=jobs)
    assert "rung snapshot at step 100 downloaded" in a["log"] and "rung snapshots uploaded from home" in a["log"]


def test_without_the_option_a_pick_campaign_is_as_before(old_runner, tmp_path):
    import run_picks
    jobs = run_picks.jobs([0.5, 0.5, 0.2], hold_pick=25)  # run_picks.py's scenario A: reclaimed with every post home
    a = same(old_runner, None, tmp_path, {"reclaim_at": [24]}, jobs=jobs)
    assert "post jobs their pick can never choose are not rerun" in a["log"]


# ---------------------------------------------------------------- with the store
STORE_JOBS = {"jobs": [
    {"name": "data", "cmd": ["x"], "outputs": ["out.txt"]},
    {"name": "pt", "model_name": "m", "after": "data", "snapshot_steps": [100], "store_branch": "ckpt-N",
     "env": {"MICROTORCH_CHECKPOINT_TXN": "1"}, "spec": {}},
    rung("rung_100", 100),
]}


def run_store(work, scenario, jobs=STORE_JOBS, extra=()):
    layout, fake = work / "repo" / "tools" / "colab_chat", work / "fake"
    layout.mkdir(parents=True)
    fake.mkdir()
    (fake / "fake.json").write_text(json.dumps({"alive": False, "launches": 0, "name": None, **scenario}))
    for f in os.listdir(CC):
        if f.endswith(".py") or f.endswith(".sh"):
            shutil.copyfile(CC / f, layout / f)
    (layout / "eqjobs.json").write_text(json.dumps(jobs))
    col, colpy = work / "colab", work / "colpy"
    col.write_text(f"#!/bin/sh\nexec {sys.executable} {EQ}/fakecolab_store.py \"$@\"\n")
    colpy.write_text(f"#!/bin/sh\nshift\nexec {sys.executable} {EQ}/fakecolab_store.py adopt \"$@\"\n")
    col.chmod(0o755)
    colpy.chmod(0o755)
    tok = work / "token"
    tok.write_text(TOKEN + "\n")
    tok.chmod(0o600)
    wrap = work / "wrap.py"
    wrap.write_text(run_eq.WRAP)
    env = dict(os.environ, FAKE_ROOT=str(fake), FAKE_JOBS=str(layout / "eqjobs.json"), FAKE_STORE_TOOLS=str(layout),
               CUDA_VISIBLE_DEVICES="")
    args = [sys.executable, str(wrap), str(layout / "runner_multi.py"), str(col), str(colpy), "--commit", "abc123",
            "--jobs", "tools/colab_chat/eqjobs.json", "--name", "pk-store", "--art", str(work / "art"),
            "--state-dir", str(work / "state"), "--max-hours", "0.2", *extra]
    p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=900)
    calls = (fake / "calls.log").read_text() if (fake / "calls.log").exists() else ""
    return p, calls


STORE_ARGS = ["--ckpt-store", "hf", "--boot-script", "tools/colab_chat/boot_base2.sh"]


def test_with_the_store_nothing_of_a_checkpoint_crosses_the_home_link(tmp_path):
    work = tmp_path
    repo = work / "store"
    # Status calls: 1-2 the data job; 3 pt starts; 4, 5 pt reaches steps 100, 200; the VM is reclaimed at the 6th.
    # The rung job is held until the 8th, so it has not run when the first VM goes.
    p, calls = run_store(work, {"reclaim_at": [6], "rung_start_at": 8},
                         extra=[*STORE_ARGS, "--store-repo", f"dir:{repo}", "--store-token-file", str(work / "token")])
    log = (work / "state" / "runner.log").read_text()
    assert p.returncode == 0, (p.stderr[-2000:], log[-2000:])
    state = json.loads((work / "state" / "state.json").read_text())
    vm_log = (work / "fake" / "store_vm.log").read_text()
    # Two launches; the job went home finished.
    assert state["launches"] == 2 and all(state["jobs"][n]["home"] for n in ("data", "pt", "rung_100"))
    assert state["jobs"]["pt"]["status"] == "FINISHED" and state["jobs"]["rung_100"]["status"] == "FINISHED"
    # No checkpoint relay and no checkpoint upload: neither in the log nor among the calls.
    assert "pt: relayed checkpoint" not in log and "checkpoint uploaded for resume" not in log
    assert "rung snapshots uploaded from home" not in log and "rung snapshot at step" not in log
    ups = [json.loads(ln)[-1] for ln in calls.splitlines() if ln.startswith('["upload"')]
    assert sorted(set(ups)) == ["/content/chat/upload/store_token"] and len(ups) == 2  # the token, once a launch
    downs = [json.loads(ln)[-2] for ln in calls.splitlines() if ln.startswith('["download"')]
    assert not any("relay_" in d or "/pt/rung_" in d or d.endswith("state.txt") or "optim" in d for d in downs)
    assert any("pt_m.safetensors" in d for d in downs)  # the final weights still come home, as before (in parts)
    # The runner read the pushed steps from the store, and the replacement VM resumed from the store's generation.
    assert "pt: the store holds its generation at step 100" in log and "pt: the store holds its generation at step 200" in log
    assert "launch 1: pt store_step None placed False" in vm_log and "launch 2: pt store_step 200 placed True" in vm_log
    assert "generation g200 of ckpt-N verified and placed for the resume" in vm_log
    assert state["jobs"]["pt"]["relayed_step"] >= 200
    final = work / "art" / "pt" / "m.safetensors"
    assert hashlib.sha256(final.read_bytes()).digest() == hashlib.sha256(fakecolab.blob(("pt", 300), fakecolab.BIG)).digest()
    # The mark: pulled from the store for the rung job on the new VM (the weights of step 100, made on the first),
    # and fetched home from the store, never from the VM.
    w100 = hashlib.sha256(fakecolab.blob(("pt", 100), fakecolab.BIG)).hexdigest()
    assert "mark snapshot rung_100 verified and placed for rung_100" in vm_log
    assert json.loads((work / "art" / "rung_100" / "rung.json").read_text())["weights_sha256"] == w100
    assert "pt: mark snapshot at step 100 fetched from the store" in log
    home = work / "art" / "pt" / "rung_snapshots" / "rung_100" / "model.safetensors"
    assert hashlib.sha256(home.read_bytes()).hexdigest() == w100 == state["store_marks"]["pt/rung_100"]["sha256"]
    # The store at the end: two generations, the last the run's end, and the mark.
    store = hf_store.open_store({"repo": f"dir:{repo}"})
    assert store.steps("ckpt-N") == [200, 300] and store.mark_steps("pt") == [100]
    # The token: its path in two upload calls and nowhere else; its text in no call, log, state or exec code; and
    # no file on the VM holds it when the session is stopped.
    assert log.count("store: config and token file sent") == 2
    assert log.rstrip().splitlines()[-2].endswith("store: token removed from the VM")
    assert calls.rstrip().splitlines()[-1].startswith('["stop"')
    for text in (calls, log, json.dumps(state), vm_log, p.stdout, p.stderr, (work / "state" / "exec.py").read_text()):
        assert TOKEN not in text
    held = [str(q) for q in (work / "fake" / "vm").rglob("*") if q.is_file() and q.stat().st_size < 4096
            and TOKEN.encode() in q.read_bytes()]
    assert held == []
    cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert cfg["repo"] == f"dir:{repo}" and cfg["branches"] == {"pt": "ckpt-N"} and not os.path.exists(cfg["token_file"])
    assert cfg["reclaim"] is True and cfg["min_push_interval_s"] == 0 and cfg["repo_type"] == "dataset"


def test_the_rehearsal_hook_stops_the_session_between_saves(tmp_path):
    """BASE2_PLAN.md's rehearsal: "pod_loss_drill_step" on the job ends the session once the store holds that step;
    the token is taken off the VM first; the next session resumes from the store."""
    jobs = json.loads(json.dumps(STORE_JOBS))
    jobs["jobs"][1]["pod_loss_drill_step"] = 200
    work = tmp_path
    p, calls = run_store(work, {}, jobs=jobs, extra=[*STORE_ARGS, "--store-repo", f"dir:{work / 'store'}",
                                                     "--store-token-file", str(work / "token")])
    log = (work / "state" / "runner.log").read_text()
    assert p.returncode == 0, (p.stderr[-2000:], log[-2000:])
    lines = [ln[15:] for ln in log.splitlines()]
    i = lines.index("pt: pod loss drill: the pod ends at the provider now; relayed 200")
    assert lines[i + 1] == "store: token removed from the VM"
    state = json.loads((work / "state" / "state.json").read_text())
    assert state["launches"] == 2 and state["pod_loss_drill"]["relayed_step"] == 200 and state["jobs"]["pt"]["home"]
    vm_log = (work / "fake" / "store_vm.log").read_text()
    assert "launch 2: pt store_step 200 placed True" in vm_log
    assert lines.count("store: token removed from the VM") == 2 and TOKEN not in calls + log


def test_the_option_needs_its_repo_token_and_boot(tmp_path):
    tok = tmp_path / "token"
    tok.write_text(TOKEN)
    tok.chmod(0o600)

    def refused(*extra):
        work = tmp_path / f"w{len(os.listdir(tmp_path))}"
        work.mkdir()
        p, calls = run_store(work, {}, extra=list(extra))
        assert p.returncode == 2 and calls == "", p.stderr[-500:]
        assert TOKEN not in p.stderr
        return p.stderr

    assert "no default store" in refused(*STORE_ARGS, "--store-token-file", str(tok))
    assert "no default token path" in refused(*STORE_ARGS, "--store-repo", "someone/base2-store")
    assert "inputs repo" in refused(*STORE_ARGS, "--store-repo", hf_store.REFUSED_REPOS[0], "--store-token-file", str(tok))
    assert "boot_base2.sh" in refused("--ckpt-store", "hf", "--store-repo", "someone/base2-store", "--store-token-file", str(tok))
    assert "need --ckpt-store hf" in refused("--boot-script", "tools/colab_chat/boot_base2.sh")
    assert "need --ckpt-store hf" in refused("--store-repo", "someone/base2-store")
    tok.chmod(0o644)
    assert "readable by others" in refused(*STORE_ARGS, "--store-repo", "someone/base2-store", "--store-token-file", str(tok))
