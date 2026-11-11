"""runner_multi.py's --ckpt-store hf option (base 2), end to end on the fake colab CLI.

1. Without the option the runner makes exactly the calls it made at the trainer commit 5506e6df: the committed
   runner and this one drive tests/runner_multi_equiv/fakecolab.py through run_eq.py's scenarios (several reclaim
   patterns), a rung campaign and a pick campaign, and every colab call, the log, state.json and the artifact tree
   must be equal.
2. With the option (fakecolab_store.py: the fake VM plus the real restore and courier on a "dir:" store): no
   checkpoint of a training job crosses the home link in either direction, the job resumes on the replacement VM
   from the store's generation, the mark snapshot reaches the rung job on the new VM and comes home from the store,
   the token is sent as a file, moved to a tmpfs by the next call, appears in no call, log or state, and is removed
   from the VM before every stop.
3. What must not happen: sessions created while the store cannot be read or after two failed boots; a start from
   step 0 that nobody admitted, or below a step home has seen pushed; a job counted home before its final
   generation and marks are safe; a store that stopped receiving going unnoticed.
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
STORE_ARGS = ["--ckpt-store", "hf", "--boot-script", "tools/colab_chat/boot_base2.sh"]


def run_store(work, scenario, jobs=STORE_JOBS, extra=(), hours="0.2", fresh=True):
    """One runner run on the fake VM with the store. fresh=False: again on the same folder (a watchdog's restart)."""
    layout, fake = work / "repo" / "tools" / "colab_chat", work / "fake"
    if fresh:
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
        (work / "wrap.py").write_text(run_eq.WRAP)
    env = dict(os.environ, FAKE_ROOT=str(fake), FAKE_JOBS=str(layout / "eqjobs.json"), FAKE_STORE_TOOLS=str(layout),
               CUDA_VISIBLE_DEVICES="", PK_STORE_TOKEN_DIR=str(work / "shm"))  # the VM's tmpfs, for the fake
    args = [sys.executable, str(work / "wrap.py"), str(layout / "runner_multi.py"), str(work / "colab"), str(work / "colpy"),
            "--commit", "abc123", "--jobs", "tools/colab_chat/eqjobs.json", "--name", "pk-store", "--art", str(work / "art"),
            "--state-dir", str(work / "state"), "--max-hours", hours, *extra]
    p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=900)
    calls = (fake / "calls.log").read_text() if (fake / "calls.log").exists() else ""
    return p, calls


def store_args(work, *more, repo=None):
    return [*STORE_ARGS, "--store-repo", f"dir:{repo or work / 'store'}", "--store-token-file", str(work / "token"), *more]


def log_of(work):
    return (work / "state" / "runner.log").read_text()


def state_of(work):
    return json.loads((work / "state" / "state.json").read_text())


def news(calls):
    return sum(1 for ln in calls.splitlines() if ln.startswith('["new"'))


def test_with_the_store_nothing_of_a_checkpoint_crosses_the_home_link(tmp_path):
    work = tmp_path
    repo = work / "store"
    # Status calls: 1-2 the data job; 3 pt starts; 4, 5 pt reaches steps 100, 200; the VM is reclaimed at the 6th.
    # The rung job is held until the 8th, so it has not run when the first VM goes.
    p, calls = run_store(work, {"reclaim_at": [6], "rung_start_at": 8}, extra=store_args(work, "--store-first-start"))
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-2000:])
    state = state_of(work)
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
    assert "first start admitted" in vm_log
    assert "generation g200 of ckpt-N verified and installed for the resume" in vm_log
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
    # The store at the end: two generations, the last the run's end, written under the two launches' epochs, and
    # the mark. Home was told the job is home only with these in place.
    store = hf_store.open_store({"repo": f"dir:{repo}"})
    assert store.steps("ckpt-N") == [200, 300] and store.mark_steps("pt") == [100]
    m200, m300 = (store.read_manifest("ckpt-N", f"gens/g{s}") for s in (200, 300))
    assert (m200["epoch"], m300["epoch"]) == (1, 2) and m200["boot_id"] != m300["boot_id"]
    assert m300["campaign"] == "pk-store" and m300["job"] == "pt" and not state.get("durability_pending")
    # What the VM was told at the second launch: the epoch, the step home had seen, and no first start any more.
    cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert cfg["repo"] == f"dir:{repo}" and cfg["branches"] == {"pt": "ckpt-N"} and cfg["campaign"] == "pk-store"
    assert cfg["epoch"] == 2 and cfg["min_step"] == {"pt": 200} and cfg["first_start"] == []
    assert cfg["reclaim"] is False and cfg["min_push_interval_s"] == 0 and cfg["repo_type"] == "dataset"
    # The token: its path in two upload calls and nowhere else; its text in no call, log, state or exec code; in
    # the tmpfs from the call after the upload; and no file anywhere when the session is stopped.
    assert log.count("store: config and token file sent") == 2
    assert log.rstrip().splitlines()[-2].endswith("store: courier ended and token removed from the VM (before the session stops)")
    assert calls.rstrip().splitlines()[-1].startswith('["stop"')
    lines = calls.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('["upload"'):
            assert lines[i + 1].startswith('["exec"')  # the very next call moves it off the disk
    for text in (calls, log, json.dumps(state), vm_log, p.stdout, p.stderr, (work / "state" / "exec.py").read_text()):
        assert TOKEN not in text
    held = [str(q) for q in [*(work / "fake" / "vm").rglob("*"), *(work / "shm").rglob("*")] if q.is_file()
            and q.stat().st_size < 4096 and TOKEN.encode() in q.read_bytes()]
    assert held == [] and cfg["token_file"] == str(work / "shm" / "token") and not os.path.exists(cfg["token_file"])


def test_the_rehearsal_hook_stops_the_session_between_saves(tmp_path):
    """BASE2_PLAN.md's rehearsal: "pod_loss_drill_step" on the job ends the session once the store holds that step;
    the courier is ended and the token taken off the VM first; the next session resumes from the store."""
    jobs = json.loads(json.dumps(STORE_JOBS))
    jobs["jobs"][1]["pod_loss_drill_step"] = 200
    work = tmp_path
    p, calls = run_store(work, {}, jobs=jobs, extra=store_args(work, "--store-first-start"))
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-2000:])
    lines = [ln[15:] for ln in log.splitlines()]
    i = lines.index("pt: pod loss drill: the pod ends at the provider now; relayed 200")
    assert lines[i + 1] == "store: courier ended and token removed from the VM (before the drill's stop)"
    state = state_of(work)
    assert state["launches"] == 2 and state["pod_loss_drill"]["relayed_step"] == 200 and state["jobs"]["pt"]["home"]
    vm_log = (work / "fake" / "store_vm.log").read_text()
    assert "launch 2: pt store_step 200 placed True" in vm_log
    assert sum("token removed from the VM" in ln for ln in lines) == 2 and TOKEN not in calls + log
    assert any("before an old session is stopped" in ln for ln in lines)  # launch() asks too, before each stop


def test_no_session_while_the_store_cannot_be_read(tmp_path):
    """An outage, seen from home before any session exists: nothing is created, the wait grows, the runner stays up
    until its deadline. (The store's path is a file here, so every read fails.)"""
    work = tmp_path
    broken = work / "not_a_store"
    broken.write_text("x")
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start", repo=broken), hours="0.0008")
    log = log_of(work)
    assert p.returncode == 0 and news(calls) == 0, log[-1500:]
    assert "store: not readable from home" in log and "no session is created; next try in 120 s (failure 1)" in log
    assert log.count("store: not readable from home") == 1  # inside the wait the store is not even asked again
    st = state_of(work)
    assert st["launches"] == 0 and st["store_preflight_failures"] == 1


def test_no_session_after_two_failed_boots_at_one_commit(tmp_path):
    """A boot that fails must not become a loop of sessions under the watchdog: the first two failures each cost
    one session, the third start creates none. The token is gone from the VM each time."""
    work = tmp_path
    for n in (1, 2):
        p, calls = run_store(work, {"boot_fail": True}, extra=store_args(work, "--store-first-start"), fresh=n == 1)
        assert p.returncode == 0 and news(calls) == n
        assert f"store: boot failure {n} of 2 at abc123" in log_of(work)
    assert state_of(work)["store_boot_failures"] == {"commit": "abc123", "n": 2}
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"), fresh=False)
    assert p.returncode == 1 and "2 boots failed at abc123" in p.stderr and news(calls) == 2
    assert not list((work / "shm").glob("token*"))
    # A boot that never reports a status is ended at the limit and counted the same way.
    work2 = tmp_path / "hang"
    work2.mkdir()
    p, calls = run_store(work2, {"boot_hang": True}, extra=store_args(work2, "--store-first-start", "--store-boot-limit-min", "0.0001"))
    log = log_of(work2)
    assert p.returncode == 0 and news(calls) == 1 and "min without a status; counted as a failed boot" in log
    assert state_of(work2)["store_boot_failures"]["n"] == 1 and calls.rstrip().splitlines()[-1].startswith('["stop"')


def test_no_start_from_zero_that_nobody_admitted(tmp_path):
    # A first start without --store-first-start: no session.
    work = tmp_path / "a"
    work.mkdir()
    p, calls = run_store(work, {}, extra=store_args(work), hours="0.0008")
    assert news(calls) == 0 and "--store-first-start was not given" in log_of(work)
    # Home has seen step 300 pushed and the branch now lists nothing (removed, renamed, another repo): the runner
    # halts before any session, and --store-first-start does not override it.
    work = tmp_path / "b"
    (work / "state").mkdir(parents=True)
    (work / "state" / "state.json").write_text(json.dumps({"endpoint": None, "launches": 3, "jobs": {
        "pt": {"relayed_step": 300, "status": "RUNNING", "home": False}}}))
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"))
    assert p.returncode == 3 and news(calls) == 0
    halted = json.loads((work / "state" / "HALTED").read_text())
    assert halted["kind"] == "STORE_HISTORY_MISSING" and halted["home_saw"] == 300 and halted["store_lists"] is None
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"), fresh=False)
    assert "HALTED marker present" in p.stderr and news(calls) == 0
    # The VM side of the same rule, when home's check is passed by a branch that lists enough but the VM is sent a
    # higher minimum: the restore refuses, the boot fails, the token is removed (tests/test_ckpt_courier.py).


def test_a_stalled_store_is_said_loudly_and_can_halt_the_arm(tmp_path):
    """The courier stops pushing after step 100 while the job trains on. The runner says so on every tick once the
    limit has passed; with --store-stall-halt it halts instead of training on with nothing saved."""
    work = tmp_path / "log"
    work.mkdir()
    p, calls = run_store(work, {"courier_off_from": 5}, extra=store_args(work, "--store-first-start", "--store-stall-min", "0.00001",
                                                                        "--store-settle-min", "0.00001"))
    log = log_of(work)
    assert "pt: STORE STALLED: no new generation in the store for" in log and "(step 100) while the job runs" in log
    # It trained to the end with nothing newer in the store: finished, but never counted home, and halted.
    assert p.returncode == 3 and "pt: finished, durability pending: final generation g300 not whole in the store" in log
    st = state_of(work)
    assert st["jobs"]["pt"]["home"] is False and not st.get("campaign_home") and "pt" in st["durability_pending"]
    assert json.loads((work / "state" / "HALTED").read_text())["kind"] == "STORE_DURABILITY_PENDING"
    assert "all jobs finished and home" not in log and calls.rstrip().splitlines()[-1].startswith('["stop"')
    work = tmp_path / "halt"
    work.mkdir()
    p, calls = run_store(work, {"courier_off_from": 5}, extra=store_args(work, "--store-first-start", "--store-stall-min", "0.00001",
                                                                        "--store-stall-halt"))
    assert p.returncode == 3 and json.loads((work / "state" / "HALTED").read_text())["kind"] == "STORE_STALLED"
    assert "token removed from the VM (before the session stops)" in log_of(work)
    assert calls.rstrip().splitlines()[-1].startswith('["stop"')


def test_a_hung_store_call_at_home_is_ended(tmp_path):
    """store_call(): the child is ended at the limit and the loop goes on (in process, no fake VM)."""
    import time
    import runner_multi as rm
    rm.STORE = {"call_limit": 1}
    rm._STORE.clear()
    t0 = time.time()
    with pytest.raises(hf_store.StoreError, match="over 1 s: ended"):
        rm.store_call(lambda: time.sleep(60))
    assert time.time() - t0 < 15
    assert rm.store_call(lambda: 5) == 5
    rm.STORE = None


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
