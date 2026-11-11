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
COMMIT = "abc123" + "0" * 34  # with the store the runner takes --commit as a full SHA


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


def run_store(work, scenario, jobs=STORE_JOBS, extra=(), hours="0.2", fresh=True, commit=COMMIT, more_env=None):
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
    env.update(more_env or {})
    args =[sys.executable, str(work / "wrap.py"), str(layout / "runner_multi.py"), str(work / "colab"), str(work / "colpy"),
            "--commit", commit, "--jobs", "tools/colab_chat/eqjobs.json", "--name", "pk-store", "--art", str(work / "art"),
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
    assert cfg["min_push_interval_s"] == 0 and cfg["repo_type"] == "dataset"
    # A VM is told nothing about release (it never releases), and it is told its commit budget.
    assert "reclaim" not in cfg and "release_grace_s" not in cfg and cfg["store_commit_budget_per_hour"] == 30
    # Home commits nothing without --store-release-storage, and every commit on the store is a generation or a mark.
    sent = [json.loads(ln)["message"] for ln in (repo / "commits.jsonl").read_text().splitlines()]
    assert sorted({m.split(":")[0] for m in sent}) == ["generation", "mark"] and "store_releases" not in state
    assert len(sent) == 4                                   # the mark and three saves: one commit each
    # The token: its path in two upload calls and nowhere else; its text in no call, log, state or exec code; in
    # the tmpfs from the call after the upload; and no file anywhere when the session is stopped.
    assert log.count("store: config and token file sent") == 2
    tail = [ln[15:] for ln in log.rstrip().splitlines()[-3:]]
    assert tail[0] == "store: courier ended and token removed from the VM (before the session stops)"
    assert tail[1].startswith("store: session files home in sessions/launch02 (before the session stops)")
    assert tail[2] == "session stopped; runner exiting"
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
        assert f"store: boot failure {n} of 2 at {COMMIT}" in log_of(work)
    assert state_of(work)["store_boot_failures"] == {"commit": COMMIT, "n": 2}
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"), fresh=False)
    assert p.returncode == 1 and f"2 boots failed at {COMMIT}" in p.stderr and news(calls) == 2
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

    def refused(*extra, **kw):
        work = tmp_path / f"w{len(os.listdir(tmp_path))}"
        work.mkdir()
        p, calls = run_store(work, {}, extra=list(extra), **kw)
        assert p.returncode == 2 and calls == "", p.stderr[-500:]
        assert TOKEN not in p.stderr
        return p.stderr

    assert "no default store" in refused(*STORE_ARGS, "--store-token-file", str(tok))
    assert "no default token path" in refused(*STORE_ARGS, "--store-repo", "someone/base2-store")
    assert "inputs repo" in refused(*STORE_ARGS, "--store-repo", hf_store.REFUSED_REPOS[0], "--store-token-file", str(tok))
    # The hub does not tell case apart in a repo's name, so neither does the refusal.
    assert "inputs repo" in refused(*STORE_ARGS, "--store-repo", hf_store.REFUSED_REPOS[0].lower(), "--store-token-file", str(tok))
    # The commit is taken as a full SHA: a short one is refused before anything is created.
    for short in ("abc123", "5b1d5187", "main"):
        assert "full 40-character SHA" in refused(*STORE_ARGS, "--store-repo", "someone/base2-store", "--store-token-file", str(tok),
                                                  commit=short)
    assert "boot_base2.sh" in refused("--ckpt-store", "hf", "--store-repo", "someone/base2-store", "--store-token-file", str(tok))
    assert "need --ckpt-store hf" in refused("--boot-script", "tools/colab_chat/boot_base2.sh")
    assert "need --ckpt-store hf" in refused("--store-repo", "someone/base2-store")
    tok.chmod(0o644)
    assert "readable by others" in refused(*STORE_ARGS, "--store-repo", "someone/base2-store", "--store-token-file", str(tok))


# ---------------------------------------------------------------- the rehearsal's own needs (second reviews, 12 Nov 2026)
def reh_jobs(drill=None, **top):
    """STORE_JOBS as the rehearsal's files are: a training job whose spec exports safetensors only, with its step
    count and a tail from half way (the fake trainer ends at step 300, so the main part is steps 1 to 150)."""
    jobs = json.loads(json.dumps(STORE_JOBS))
    jobs["jobs"][1]["spec"] = {"train": {"steps": 300}, "data": {"anneal": {"from_fraction": 0.5}},
                               "export": {"formats": ["safetensors"]}}
    if drill:
        jobs["jobs"][1]["pod_loss_drill_step"] = drill
    return {**jobs, **top}


def stat_lines(text, what):
    return [json.loads(ln[5:]) for ln in text.splitlines() if ln.startswith("stat ") and json.loads(ln[5:])["what"] == what]


def test_the_rehearsal_comes_home_without_a_gguf_with_its_session_files_and_its_drill_evidence(tmp_path):
    """Three things the second reviews found missing, on one run of the rehearsal's shape.
    1. A job that exports no gguf comes home: the runner fetches what "export.formats" declares and no more, and
       events.jsonl whatever else happened. (The fake trainer writes no gguf for such a spec; before, the runner
       asked for one for ever, the job was never home and the session billed to the deadline.) The job is counted
       home through store_durable, on the same fake.
    2. Before each stop, the session's own record comes home, named by session: courier.log with each push's bytes
       and seconds and each restore pull's, boot.log, courier_status.json, store_restore/*.json.
    3. The drill's record: the trainer's training step, from its own event stream, and the courier's pushed step,
       read from the VM just before the stop; between_pushes only because the first is above the second; the side
       of the tail switch."""
    work = tmp_path / "a"
    work.mkdir()
    # "trainer_ahead": the trainer is 37 steps past its last save whenever it is looked at.
    p, calls = run_store(work, {"trainer_ahead": 37}, jobs=reh_jobs(drill=200), extra=store_args(work, "--store-first-start"))
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-3000:])
    state = state_of(work)
    lines = [ln[15:] for ln in log.splitlines()]
    # 1. Home, and the session stopped: no wait for a file the job never writes.
    assert "all jobs finished and home" in lines and calls.rstrip().splitlines()[-1].startswith('["stop"')
    assert state["launches"] == 2 and all(state["jobs"][n]["home"] for n in ("data", "pt", "rung_100"))
    assert state.get("campaign_home") and not state.get("durability_pending")
    asked = [ln for ln in calls.splitlines() if ".gguf" in ln or ".ir.json" in ln]
    assert asked == [], asked                                            # nothing but what the spec declares
    assert "pt: final outputs downloaded" in lines
    art = work / "art" / "pt"
    assert sorted(f for f in os.listdir(art) if not f.startswith("rung_")) == ["events.jsonl", "m.safetensors", "pt_spec.json",
                                                                              "pt_train.log", "result.json"]
    assert hashlib.sha256((art / "m.safetensors").read_bytes()).digest() == hashlib.sha256(fakecolab.blob(("pt", 300), fakecolab.BIG)).digest()
    assert json.loads((art / "events.jsonl").read_text().splitlines()[-1])["step"] == 300   # the losses' record
    # 3. The drill's record.
    rec = json.loads((work / "state" / "drill_record.json").read_text())
    ev = rec["evidence"]
    assert rec["job"] == "pt" and ev["vm_answered"] is True
    assert (ev["training_step"], ev["pushed_step"], ev["saved_step"]) == (237, 200, 200)
    assert rec["between_pushes"] is True and rec["steps_lost"] == 37
    assert (rec["last_main_step"], rec["drill_side"], rec["side"], rec["side_missed"]) == (150, "tail", "tail", False)
    assert rec["stop_confirmed"] is True and rec["store_step_after_stop"] == 200
    for k in ("read_at_home", "vm_clock", "training_step_at", "pushed_at", "courier_status_at"):
        assert isinstance(ev[k], float) and abs(ev[k] - rec["at"]) < 600, k         # each reading carries its time
    assert rec["at"] <= ev["read_at_home"] <= rec["stop_returned_at"]
    assert state["pod_loss_drill"] == rec
    # The order at the drill: the courier is ended and the token removed, the session's files come home, the VM is
    # read, and only then is the session stopped.
    i = lines.index("pt: pod loss drill: the pod ends at the provider now; relayed 200")
    assert lines[i + 1] == "store: courier ended and token removed from the VM (before the drill's stop)"
    assert lines[i + 2].startswith("store: session files home in sessions/launch01 (before the drill's stop): ")
    assert lines[i + 3] == ("pt: drill evidence: training step 237, pushed step 200, between pushes True, side tail "
                            "(the drill step is on the tail side)")
    assert lines[i + 4].startswith("pt: drill record written (drill_record.json): ")
    cl = calls.splitlines()
    stops = [k for k, ln in enumerate(cl) if ln.startswith('["stop"')]
    assert len(stops) == 4                    # before each of the two sessions is created, at the drill, at the end
    assert all(cl[stops[1] - k].startswith('["exec"') for k in (1, 2, 3))   # token, files home, the read; then the stop
    # 2. The sessions' own records, at home.
    s1, s2 = work / "state" / "sessions" / "launch01", work / "state" / "sessions" / "launch02"
    for d in (s1, s2):
        assert {"courier.log", "boot.log", "courier_status.json", "store_restore__pt.json", "index.json"} <= set(os.listdir(d))
    i1, i2 = (json.loads((d / "index.json").read_text()) for d in (s1, s2))
    assert (i1["when"], i1["launch"], i1["session"]) == ("before the drill's stop", 1, "pk-store")
    assert (i2["when"], i2["launch"]) == ("before the session stops", 2) and all(f["whole"] for f in i2["files"].values())
    c1, c2 = (s1 / "courier.log").read_text(), (s2 / "courier.log").read_text()
    pushes = stat_lines(c1, "push")
    assert [(x["kind"], x["step"]) for x in pushes] == [("mark", 100), ("generation", 100), ("generation", 200)]
    assert all(x["bytes"] > fakecolab.BIG and x["seconds"] >= 0 for x in pushes)     # bytes and seconds of every push
    pulls = stat_lines(c2, "pull")
    assert [(x["kind"], x["step"]) for x in pulls] == [("generation", 200), ("mark", 100)]
    assert all(x["bytes"] > fakecolab.BIG and x["seconds"] >= 0 for x in pulls)      # and of every pull of the restore
    assert stat_lines(c2, "push")[-1]["step"] == 300
    st1 = json.loads((s1 / "courier_status.json").read_text())
    assert st1["jobs"]["pt"]["pushed_step"] == 200 and st1["http_429"] == 0 and "releases" not in st1
    assert st1["pushed_bytes"] == st1["bytes_since_release"] and st1["commit_budget_per_hour"] == 30
    r2 = json.loads((s2 / "store_restore__pt.json").read_text())
    assert r2["store_step"] == 200 and r2["placed"] and r2["pull_bytes"] > fakecolab.BIG and r2["pull_seconds"] >= 0
    assert "boot_base2 (fake) launch 2" in (s2 / "boot.log").read_text()
    for d in (s1, s2):
        for f in os.listdir(d):
            assert TOKEN.encode() not in (d / f).read_bytes()
    # What the first session's courier had pushed is what the second one counted on from (the storage guard).
    cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert cfg["pushed_bytes_start"] == st1["bytes_since_release"] > 3 * fakecolab.BIG and cfg["push_cap_bytes"] == 0
    assert cfg["mark_ns"] == "" and cfg["branches"] == {"pt": "ckpt-N"}

    # A drill meant for the main part that lands in the tail says so. The drill step is 100 (main: the last main
    # step is 150) and the trainer is 60 steps past its save when the runner looks: step 160, in the tail.
    work = tmp_path / "b"
    work.mkdir()
    p, calls = run_store(work, {"trainer_ahead": 60}, jobs=reh_jobs(drill=100), extra=store_args(work, "--store-first-start"))
    assert p.returncode == 0, (p.stderr[-2000:], log_of(work)[-3000:])
    rec = json.loads((work / "state" / "drill_record.json").read_text())
    assert (rec["evidence"]["training_step"], rec["evidence"]["pushed_step"]) == (160, 100)
    assert (rec["drill_side"], rec["side"], rec["side_missed"], rec["between_pushes"]) == ("main", "tail", True, True)
    assert "side tail (the drill step is on the main side); SIDE MISSED" in log_of(work)

    # A trainer that is AT its pushed step is not between pushes, whatever the drill meant: the record says false.
    work = tmp_path / "c"
    work.mkdir()
    p, calls = run_store(work, {}, jobs=reh_jobs(drill=200), extra=store_args(work, "--store-first-start"))
    assert p.returncode == 0, (p.stderr[-2000:], log_of(work)[-3000:])
    rec = json.loads((work / "state" / "drill_record.json").read_text())
    assert (rec["evidence"]["training_step"], rec["evidence"]["pushed_step"]) == (200, 200)
    assert rec["between_pushes"] is False and rec["steps_lost"] == 0 and rec["side_missed"] is False


def test_the_push_cap_halts_the_campaign_by_name(tmp_path):
    """The storage guard. The jobs file carries the cap ("store_push_cap_gb"; the flag overrides it). The courier
    counts what it pushes; at the cap it stops pushing and says why; the runner reads that on its next tick, halts
    with STORE_PUSH_CAP, brings the session's record home and stops the session."""
    work = tmp_path
    cap = 1.5 * fakecolab.BIG / 1e9          # passed by the mark and the first generation; the second is not pushed
    p, calls = run_store(work, {}, jobs=reh_jobs(store_push_cap_gb=cap), extra=store_args(work, "--store-first-start"))
    log = log_of(work)
    assert p.returncode == 3, (p.stderr[-2000:], log[-3000:])
    halted = json.loads((work / "state" / "HALTED").read_text())
    assert halted["kind"] == "STORE_PUSH_CAP" and halted["job"] == "pt" and halted["cap_bytes"] == int(cap * 1e9)
    assert halted["bytes_since_release"] > 2 * fakecolab.BIG and "at the last release from home" in halted["why"]
    store = hf_store.open_store({"repo": f"dir:{work / 'store'}"})
    assert store.steps("ckpt-N") == [100] and store.mark_steps("pt") == [100]      # nothing went up past the cap
    state = state_of(work)
    assert state["store_pushed_bytes"] == halted["bytes_since_release"] and state["jobs"]["pt"]["home"] is False
    assert not state.get("campaign_home") and "all jobs finished and home" not in log
    assert calls.rstrip().splitlines()[-1].startswith('["stop"')
    s1 = work / "state" / "sessions" / "launch01"
    assert "PUSH CAP" in (s1 / "courier.log").read_text() and (s1 / "COURIER_CAPPED").exists()
    assert json.loads((s1 / "courier_status.json").read_text())["capped"]["cap"] == int(cap * 1e9)
    cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert cfg["push_cap_bytes"] == int(cap * 1e9) and cfg["pushed_bytes_start"] == 0
    # The marker stops a restart, as every halt does.
    p, calls2 = run_store(work, {}, jobs=reh_jobs(store_push_cap_gb=cap), extra=store_args(work, "--store-first-start"), fresh=False)
    assert "HALTED marker present" in p.stderr and news(calls2) == news(calls) == 1
    # The flag overrides the jobs file: with the guard off the same campaign comes home.
    work2 = tmp_path / "off"
    work2.mkdir()
    p, calls = run_store(work2, {}, jobs=reh_jobs(store_push_cap_gb=cap),
                         extra=store_args(work2, "--store-first-start", "--store-push-cap-gb", "0"))
    assert p.returncode == 0 and state_of(work2)["campaign_home"], log_of(work2)[-2000:]


def test_storage_is_released_from_home_and_the_guards_count_is_carried_never_reset(tmp_path):
    """Fix round three, on the rehearsal's own sequence, through the real HfBackend over the stand-in hub (the
    runner at home and the courier on the fake VM both): session one pushes a mark and two generations; the drill
    loses the VM; session two resumes, and its next save removes the first generation in its own commit.

    1. Commits. A save is one commit. The VM makes no release, no lock and no prune commit, and is sent no release
       setting. The runner, with --store-release-storage, releases on its timer and once more at the end; the lock
       and its removal are the only commits home makes.
    2. Release from home frees exactly what no branch names: the removed generation's optimiser (its weights are
       the mark's, and stay).
    3. The storage guard's count (the rehearsal of 12 Nov 2026: state.json read 0 during the second session,
       because the VM's release freed part of what had been pushed and the count started again from nothing). The
       count is what the store was measured to hold for the campaign, plus what was pushed since; the first
       session's count is what the second starts from; it is never 0 while the store holds generations."""
    pytest.importorskip("huggingface_hub")
    work = tmp_path
    hub = work / "hub"
    env = {"HUB_STANDIN_ROOT": str(hub),
           "PYTHONPATH": f"{ROOT / 'tests' / 'hub_standin_site'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)}
    extra = [*STORE_ARGS, "--store-repo", "someone/base2-store", "--store-token-file", str(work / "token"),
             "--store-first-start", "--store-release-storage", "--store-release-grace-s", "0",
             "--store-release-every-min", "0.000001", "--store-home-commit-budget-per-hour", "400"]
    p, calls = run_store(work, {"trainer_ahead": 37}, jobs=reh_jobs(drill=200, store_push_cap_gb=100), extra=extra, more_env=env)
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-3000:])
    state = state_of(work)
    assert state["launches"] == 2 and state["campaign_home"]
    sent = [json.loads(ln) for ln in (hub / "commit_log.jsonl").read_text().splitlines()]
    assert all(x["ok"] for x in sent)
    # 1. The branch: three saves, three commits; the third removes g100 itself. Main: the mark, and home's locks.
    branch = [x for x in sent if x["branch"] == "ckpt-N"]
    assert [x["message"] for x in branch] == [f"generation: gens/g{s}" for s in (100, 200, 300)]
    assert [x["deletes"] for x in branch] == [[], [], ["gens/g100/"]]
    kinds = [x["message"].split(":")[0] for x in sent if x["branch"] == "main"]
    assert kinds.count("mark") == 1 and set(kinds) == {"mark", "lock", "unlock"}
    assert kinds.count("lock") == kinds.count("unlock") >= 3
    assert all(x["message"] == "lock: release by home:pk-store" for x in sent if x["message"].startswith("lock"))
    assert not (hub / "branches" / "main" / "locks" / "release.json").exists()
    vm_cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert "reclaim" not in vm_cfg and "release_grace_s" not in vm_cfg and vm_cfg["store_commit_budget_per_hour"] == 30
    s1, s2 = work / "state" / "sessions" / "launch01", work / "state" / "sessions" / "launch02"
    c1, c2 = (s1 / "courier.log").read_text(), (s2 / "courier.log").read_text()
    assert "storage release" not in c1 + c2 and not stat_lines(c1 + c2, "release")
    assert stat_lines(c2, "push")[-1]["pruned"] == [100]
    # 2. What home's releases did: one of them freed the removed generation's optimiser, 1,000 bytes, and nothing else.
    rel = [json.loads(ln.split("store: stat ", 1)[1]) for ln in log.splitlines() if "store: stat " in ln]
    rel = [r for r in rel if r["what"] == "release"]
    assert len(rel) == state["store_releases"]["count"] == kinds.count("lock") and rel[-1]["final"] is True
    assert [r["bytes"] for r in rel if r["bytes"]] == [1000] and state["store_releases"]["bytes"] == 1000
    assert state["store_releases"]["objects"] == 1 and state["store_releases"]["failed"] == 0
    objects = json.loads((hub / "lfs.json").read_text())
    w = {s: hashlib.sha256(fakecolab.blob(("pt", s), fakecolab.BIG)).hexdigest() for s in (100, 200, 300)}
    assert set(w.values()) <= set(objects) and len(objects) == 5          # three weights (g100's is the mark's), two optimisers
    store = hf_store.open_store({"repo": "someone/base2-store", "token_file": str(work / "token")},
                                api=__import__("hub_standin").DiskApi(str(hub)))
    assert store.steps("ckpt-N") == [200, 300] and store.mark_steps("pt") == [100]
    assert store.verify_by_download("ckpt-N", 300) and store.verify_by_download("ckpt-N", 200)
    store.pull_mark("pt", 100, str(work / "mark100"))                     # the mark's weights outlived its generation
    # 3. The count. Every release measured the store; with anything stored, none left the campaign counted at nothing.
    big = fakecolab.BIG
    holding = [r for r in rel if r["stored_bytes"]]
    assert len(holding) >= 3 and all(r["counted_bytes"] > 2 * big for r in holding)
    freeing = next(r for r in rel if r["bytes"])
    assert freeing["counted_bytes"] == 3 * big + 2 * 1000 and freeing["unnamed_bytes"] == 0   # g200, g300 and the mark
    # Session two started from what was counted when session one ended (the mark and two generations), and its
    # courier counted on from there.
    assert vm_cfg["pushed_bytes_start"] == 3 * big + 2 * 1000 and vm_cfg["push_cap_bytes"] == 100 * 10 ** 9
    st2 = json.loads((s2 / "courier_status.json").read_text())
    assert st2["bytes_since_release"] >= 3 * big and st2["measure_seq"] >= 1 and st2["capped"] is None
    measures = [m for m in stat_lines(c1 + c2, "measure") if m["store_bytes"]]
    assert measures and all(m["bytes_since_release"] > 2 * big for m in measures)
    assert state["store_pushed_bytes"] == rel[-1]["counted_bytes"] == 3 * big + 2 * 1000 > 0
    for text in (calls, log, json.dumps(state), c1, c2, p.stdout, p.stderr):
        assert TOKEN not in text


def test_a_stop_that_is_not_confirmed_is_said_loudly_and_the_runner_exits_non_zero(tmp_path):
    """`colab stop` can fail. Its return code is read and the stop tried again; when it keeps failing and the
    listing still holds the session, STOP_UNRESOLVED names the session and the runner exits 4. A stop that fails
    where the listing answers without the session had nothing to stop. A later confirmed stop closes the record."""
    work = tmp_path / "a"
    work.mkdir()
    # The boot fails, so the runner goes straight to its end; every stop fails and the session stays up.
    p, calls = run_store(work, {"boot_fail": True, "stop_fail": True}, extra=store_args(work, "--store-first-start"))
    log = log_of(work)
    assert p.returncode == 4, (p.stderr[-1500:], log[-2500:])
    rec = json.loads((work / "state" / "STOP_UNRESOLVED").read_text())
    assert rec["kind"] == "STOP_UNRESOLVED" and rec["session"] == "pk-store" and rec["endpoint"] == "fake-endpoint-1"
    assert rec["return_code"] == 1 and rec["tries"] == 3 and "colab stop -s pk-store" in rec["do"]
    assert "STOP_UNRESOLVED: the stop of session pk-store" in log and "IT MAY STILL BE BILLING" in log
    assert log.rstrip().splitlines()[-1].endswith("SESSION STOP NOT CONFIRMED (STOP_UNRESOLVED); runner exiting")
    cl = calls.splitlines()
    # Before the session was created the stop failed too, the listing answered without the session, and the
    # launch went on: one stop, one listing.
    first_new = next(k for k, ln in enumerate(cl) if ln.startswith('["new"'))
    assert [ln[:6] for ln in cl[:first_new]].count('["stop') == 1
    assert "the listing answers and does not hold pk-store: nothing to stop" in log
    # At the end: three stops, each followed by a listing that still shows the session.
    end = [ln for ln in cl[first_new:] if ln.startswith('["stop"') or ln.startswith('["sessions"')]
    assert [ln[:6] for ln in end[-6:]] == ['["stop', '["sess', '["stop', '["sess', '["stop', '["sess']
    assert json.loads((work / "fake" / "fake.json").read_text())["alive"] is True     # it is, in fact, still up
    assert log.count("is not known to be gone (try") == 3
    # The same state folder started again with a provider that answers: its end stops the session and closes the record.
    fj = work / "fake" / "fake.json"
    d = json.loads(fj.read_text())
    d["stop_fail"] = False
    fj.write_text(json.dumps(d))
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"), fresh=False)
    assert p.returncode == 0, (p.stderr[-1500:], log_of(work)[-1500:])
    assert not (work / "state" / "STOP_UNRESOLVED").exists() and (work / "state" / "STOP_RESOLVED").exists()
    assert json.loads(fj.read_text())["alive"] is False
    # A stop that fails twice and then succeeds is confirmed: no record, exit 0.
    work = tmp_path / "b"
    work.mkdir()
    p, calls = run_store(work, {"boot_fail": True, "stop_fail": 3}, extra=store_args(work, "--store-first-start"))
    assert p.returncode == 0, (p.stderr[-1500:], log_of(work)[-2500:])
    assert not (work / "state" / "STOP_UNRESOLVED").exists() and log_of(work).count("is not known to be gone (try") == 2
    assert log_of(work).rstrip().splitlines()[-1].endswith("session stopped; runner exiting")
    assert json.loads((work / "fake" / "fake.json").read_text())["alive"] is False
    # A failed stop of the drill's forced loss ends the run the same way, with the drill's record saying so.
    work = tmp_path / "c"
    work.mkdir()
    fail_from = {"trainer_ahead": 37, "stop_fail_from_status": 5}
    p, calls = run_store(work, fail_from, jobs=reh_jobs(drill=200), extra=store_args(work, "--store-first-start"))
    assert p.returncode == 4, (p.stderr[-1500:], log_of(work)[-2500:])
    rec = json.loads((work / "state" / "drill_record.json").read_text())
    assert rec["stop_confirmed"] is False and rec["between_pushes"] is True
    assert json.loads((work / "state" / "STOP_UNRESOLVED").read_text())["session"] == "pk-store"
    assert news(calls) == 1                                               # no second session beside the first


def test_a_new_attempt_never_resumes_from_an_earlier_attempts_branches(tmp_path):
    """A rehearsal run again under the same name and jobs file: the earlier attempt's ckpt-* branches carry the
    same campaign, job and spec, so the VM would resume from them without a word. A state folder that has never
    launched refuses branches that hold generations, before any session, and names --store-campaign; that flag
    gives the new attempt its own branches and marks. The first attempt's own state folder goes on as before."""
    repo = tmp_path / "shared_store"
    first = tmp_path / "first"
    first.mkdir()
    p, calls = run_store(first, {}, jobs=reh_jobs(), extra=store_args(first, "--store-first-start", repo=repo))
    assert p.returncode == 0, (p.stderr[-1500:], log_of(first)[-2500:])
    assert state_of(first)["store_attempt"]["campaign"] == "pk-store"
    store = hf_store.open_store({"repo": f"dir:{repo}"})
    assert store.steps("ckpt-N") == [200, 300] and store.mark_steps("pt") == [100]
    # A second attempt: a fresh state folder, the same name, jobs and store.
    second = tmp_path / "second"
    second.mkdir()
    p, calls = run_store(second, {}, jobs=reh_jobs(), extra=store_args(second, "--store-first-start", repo=repo))
    assert p.returncode == 1 and news(calls) == 0, (p.stderr[-1500:], log_of(second)[-1500:])
    for text in (p.stderr, log_of(second)):
        assert "ckpt-N holds generations up to g300" in text and "an earlier attempt's" in text and "--store-campaign" in text
    assert "store_attempt" not in state_of(second)
    # The same state folder with a campaign name of its own: its own branch and marks, from step 0, to the end.
    p, calls = run_store(second, {}, jobs=reh_jobs(), fresh=False,
                         extra=store_args(second, "--store-first-start", "--store-campaign", "try 2", repo=repo))
    assert p.returncode == 0, (p.stderr[-1500:], log_of(second)[-2500:])
    vm_log = (second / "fake" / "store_vm.log").read_text()
    assert "launch 1: pt store_step None placed False" in vm_log and "first start admitted" in vm_log
    cfg = json.loads((second / "fake" / "vm" / "chat" / "store.json").read_text())
    assert cfg["branches"] == {"pt": "ckpt-N-try-2"} and cfg["mark_ns"] == "try-2." and cfg["campaign"] == "try 2"
    assert store.steps("ckpt-N-try-2") == [200, 300] and store.read_manifest("ckpt-N-try-2", "gens/g300")["campaign"] == "try 2"
    assert sorted(os.listdir(repo / "main" / "marks")) == ["pt", "try-2.pt"]
    assert store.read_manifest("ckpt-N", "gens/g300")["campaign"] == "pk-store"    # the first attempt's, untouched
    assert state_of(second)["campaign_home"] and state_of(second)["store_marks"]["pt/rung_100"]["sha256"]
    # The first attempt's own state folder is that attempt: a restart of it is not refused. (Everything is home, so
    # the restart only has to get past the preflight; a short deadline ends it.)
    st = json.loads((first / "state" / "state.json").read_text())
    st["jobs"]["pt"].update(home=False, relayed_step=0)
    st["endpoint"] = None
    st.pop("campaign_home", None)
    (first / "state" / "state.json").write_text(json.dumps(st))
    p, calls = run_store(first, {}, jobs=reh_jobs(), extra=store_args(first, "--store-first-start", repo=repo), fresh=False,
                         hours="0.01")
    assert "an earlier attempt's" not in p.stderr + log_of(first) and news(calls) == 2
