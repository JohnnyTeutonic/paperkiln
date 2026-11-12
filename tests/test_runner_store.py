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


def run_store(work, scenario, jobs=STORE_JOBS, extra=(), hours="0.2", fresh=True, commit=COMMIT, more_env=None, wrap=None):
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
        (work / "wrap.py").write_text(wrap or run_eq.WRAP)
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
    # The store at the end: three generations, the last the run's end, written under the two launches' epochs, and
    # the mark. Home was told the job is home only with these in place.
    store = hf_store.open_store({"repo": f"dir:{repo}"})
    assert store.steps("ckpt-N") == [100, 200, 300] and store.mark_steps("pt") == [100]
    m200, m300 = (store.read_manifest("ckpt-N", f"gens/g{s}") for s in (200, 300))
    assert (m200["epoch"], m300["epoch"]) == (1, 2) and m200["boot_id"] != m300["boot_id"]
    assert m300["campaign"] == "pk-store" and m300["job"] == "pt" and not state.get("durability_pending")
    # What the VM was told at the second launch: the epoch, the step home had seen, and no first start any more.
    cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert cfg["repo"] == f"dir:{repo}" and cfg["branches"] == {"pt": "ckpt-N"} and cfg["campaign"] == "pk-store"
    assert cfg["epoch"] == 2 and cfg["min_step"] == {"pt": 200} and cfg["first_start"] == []
    assert cfg["min_push_interval_s"] == 0 and cfg["repo_type"] == "dataset"
    # A VM is told nothing about release (it never releases), and it is told its commit budget.
    assert "reclaim" not in cfg and "release_grace_s" not in cfg and cfg["store_commit_budget_per_hour"] == 24
    assert cfg["courier_wait_s"] == 5400                    # the courier's final delivery window, which the boot reads
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
                                                                        "--store-settle-min", "0.00001",
                                                                        "--store-courier-wait-s", "0"))
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
    assert st1["pushed_bytes"] == st1["bytes_since_release"] and st1["commit_budget_per_hour"] == 24
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


def hub_env(hub, **more):
    return {"HUB_STANDIN_ROOT": str(hub), **more,
            "PYTHONPATH": f"{ROOT / 'tests' / 'hub_standin_site'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)}


def earlier_campaign(hub, tmp):
    """What the repo holds before a fresh attempt of the rehearsal's part b: main's seed weights; an earlier
    attempt's branch with two generations and its mark (no ledgers: it ran on the code before them); and an object
    that attempt uploaded and never committed. Returns (a store on it, the bytes of each of those three)."""
    sys.path.insert(0, str(ROOT / "tests"))
    import hub_standin
    import store_standin as S
    old = hf_store.Store(hf_store.HfBackend("someone/base2-store", "dataset", TOKEN, api=hub_standin.DiskApi(str(hub))),
                         log=lambda m: None, delay=0, mark_ns="base2-reh-b1.")
    seed = tmp / "seed"
    seed.mkdir()
    (seed / "tc_b1_r1.safetensors").write_bytes(b"seed weights " * 500)
    old.push_set("main", "seed/weights", str(seed))
    branch = "ckpt-Rb-base2-reh-b1"
    for step in (100, 200):
        S.make_gen(str(tmp / "oldjob"), step, seed=5)
        old.push_generation(branch, "pt_reh_b", step, str(tmp / "oldjob" / "ckpt_gen" / f"g{step}"))
    mark = tmp / "oldmark"
    mark.mkdir()
    (mark / "model.safetensors").write_bytes(S.state_at(100, 5)[0])
    (mark / "state.txt").write_text("100\n{}\n")
    old.push_mark("pt_reh_b", 100, str(mark))
    (hub / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": 1}))
    left = tmp / "left"
    left.mkdir()
    (left / "never_committed.safetensors").write_bytes(b"an upload whose commit never came " * 100)
    with pytest.raises(hf_store.RateLimited):
        old.push_set(branch, "x", str(left))
    (hub / "limit.json").unlink()
    sizes = {"seed": 13 * 500, "kept": sum(len(b) for s in (100, 200) for b in S.state_at(s, 5)), "orphan": 34 * 100}
    return old, sizes


def test_storage_is_released_from_home_and_the_guards_count_is_carried_never_reset(tmp_path):
    """Fix rounds three and four, on the rehearsal's own sequence, through the real HfBackend over the stand-in hub
    (the runner at home and the courier on the fake VM both), as a FRESH ATTEMPT BESIDE AN EARLIER ONE on the same
    repo: --store-campaign gives it its own branch and marks, and the earlier attempt's branch, mark and leftovers
    are there throughout. Session one pushes a mark and two generations; the drill loses the VM; session two
    resumes and trains on to step 500, removing g100 and g200 on the way, each in a later save's commit.

    1. Commits. A save is one commit. The VM makes no release, no lock and no prune commit, and is sent no release
       setting. The runner, with --store-release-storage, releases on its timer and once more at the end; the lock
       and its removal are the only commits home makes.
    2. Release from home frees exactly what the removed generations' ledgers name and nothing names any more: g100's
       optimiser (its weights are the mark's, and stay) and g200's two files. Nothing of the earlier attempt: not
       its generations, not its mark, not the object it uploaded and never committed, which is reported and left.
    3. The storage guard's count (the rehearsal of 12 Nov 2026: state.json read 0 during the second session,
       because the VM's release freed part of what had been pushed and the count started again from nothing). The
       count is what the store was measured to hold for THIS campaign, each object once (a mark's weights are its
       generation's), nothing of main's seed although the branch's tree names it, nothing of the earlier attempt;
       plus what was pushed since. The first session's count is what the second starts from; never 0 while the
       store holds generations.
    4. The drill session's watch.jsonl comes home with its other files before the stop."""
    pytest.importorskip("huggingface_hub")
    work = tmp_path
    hub = work / "hub"
    old, there = earlier_campaign(hub, tmp_path)
    before = {"steps": old.steps("ckpt-Rb-base2-reh-b1"), "tree": old.b.tree("ckpt-Rb-base2-reh-b1"),
              "objects": json.loads((hub / "lfs.json").read_text()), "n": len((hub / "commit_log.jsonl").read_text().splitlines())}
    assert before["steps"] == [100, 200] and len(before["objects"]) == 6          # the seed, four of two generations, the orphan
    jobs = reh_jobs(drill=200, store_push_cap_gb=100)
    jobs["jobs"][1]["spec"]["train"]["steps"] = 500
    jobs["jobs"].insert(1, {"name": "watch", "after": "data", "cmd": ["w"], "outputs": ["watch.jsonl"]})
    extra = [*STORE_ARGS, "--store-repo", "someone/base2-store", "--store-token-file", str(work / "token"),
             "--store-first-start", "--store-campaign", "base2-reh-b2", "--store-release-storage",
             "--store-release-grace-s", "600", "--store-release-every-min", "0.000001",
             "--store-home-commit-budget-per-hour", "400"]
    p, calls = run_store(work, {"trainer_ahead": 37}, jobs=jobs, extra=extra,
                         more_env=hub_env(hub, FAKE_STEPS="500", FAKE_VM_LEDGER_AGE_S="100000"))
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-3000:])
    state = state_of(work)
    assert state["launches"] == 2 and state["campaign_home"]
    sent = [json.loads(ln) for ln in (hub / "commit_log.jsonl").read_text().splitlines()][before["n"]:]
    assert all(x["ok"] for x in sent)
    mine, ns = "ckpt-N-base2-reh-b2", "base2-reh-b2."
    # 1. The branch: its creation, then five saves, five commits; g100 goes in g400's commit and g200 in g500's (two
    # generations verified by their listings stand above each), with its ledger. Main: the mark, and home's locks.
    branch = [x for x in sent if x["branch"] == mine]
    assert [x["message"] for x in branch] == ["create_branch", *[f"generation: gens/g{s}" for s in (100, 200, 300, 400, 500)]]
    assert [[d for d in x["deletes"] if d.startswith("gens/")] for x in branch[1:]] == [[], [], [], ["gens/g100/"], ["gens/g200/"]]
    assert "pruned/g100.json" in branch[4]["adds"] and "pruned/g200.json" in branch[5]["adds"]
    assert {x["branch"] for x in sent} == {mine, "main"}                          # nothing was written to the earlier attempt's
    kinds = [x["message"].split(":")[0] for x in sent if x["branch"] == "main"]
    assert kinds.count("mark") == 1 and set(kinds) == {"mark", "lock", "unlock"}
    assert kinds.count("lock") == kinds.count("unlock") >= 4
    assert all(x["message"] == "lock: release by home:pk-store" for x in sent if x["message"].startswith("lock"))
    assert not (hub / "branches" / "main" / "locks" / "release.json").exists()
    vm_cfg = json.loads((work / "fake" / "vm" / "chat" / "store.json").read_text())
    assert "reclaim" not in vm_cfg and "release_grace_s" not in vm_cfg and vm_cfg["store_commit_budget_per_hour"] == 24
    assert vm_cfg["courier_wait_s"] == 5400 and vm_cfg["branches"] == {"pt": mine} and vm_cfg["mark_ns"] == ns
    s1, s2 = work / "state" / "sessions" / "launch01", work / "state" / "sessions" / "launch02"
    c1, c2 = (s1 / "courier.log").read_text(), (s2 / "courier.log").read_text()
    assert "storage release" not in c1 + c2 and not stat_lines(c1 + c2, "release")
    assert [x["pruned"] for x in stat_lines(c2, "push")] == [[], [100], [200]]
    assert [(v["step"], v["whole"]) for v in stat_lines(c1 + c2, "verify")] == [(s, True) for s in (100, 200, 300, 400, 500)]
    # 2. What home's releases did.
    big = fakecolab.BIG
    rel = [json.loads(ln.split("store: stat ", 1)[1]) for ln in log.splitlines() if "store: stat " in ln]
    rel = [r for r in rel if r["what"] == "release"]
    assert len(rel) == state["store_releases"]["count"] == kinds.count("lock") and rel[-1]["final"] is True
    assert sum(r["bytes"] for r in rel) == state["store_releases"]["bytes"] == big + 2 * 1000       # g100's optimiser, g200's two
    assert sum(r["objects"] for r in rel) == state["store_releases"]["objects"] == 3 and state["store_releases"]["failed"] == 0
    assert all((r["orphan_objects"], r["orphan_bytes"]) == (1, there["orphan"]) for r in rel)       # the earlier upload: seen, left
    objects = json.loads((hub / "lfs.json").read_text())
    assert set(before["objects"]) <= set(objects)                                 # everything that was there is there
    w = {s: hashlib.sha256(fakecolab.blob(("pt", s), fakecolab.BIG)).hexdigest() for s in (100, 200, 300, 400, 500)}
    assert {w[100], w[300], w[400], w[500]} <= set(objects) and w[200] not in objects
    assert len(objects) == len(before["objects"]) + 7                             # four weights (g100's is the mark's), three optimisers
    assert old.steps("ckpt-Rb-base2-reh-b1") == before["steps"] and old.b.tree("ckpt-Rb-base2-reh-b1") == before["tree"]
    assert old.verify_by_download("ckpt-Rb-base2-reh-b1", 200) and old.verify_by_download("ckpt-Rb-base2-reh-b1", 100)
    old.pull_mark("pt_reh_b", 100, str(work / "oldmark_back"))
    store = hf_store.open_store({"repo": "someone/base2-store", "token_file": str(work / "token"), "mark_ns": ns},
                                api=__import__("hub_standin").DiskApi(str(hub)))
    assert store.steps(mine) == [300, 400, 500] and store.mark_steps("pt") == [100]
    assert all(store.verify_by_download(mine, s) for s in (300, 400, 500))
    store.pull_mark("pt", 100, str(work / "mark100"))                             # the mark's weights outlived its generation
    assert "seed/weights/tc_b1_r1.safetensors" in store.b.tree(mine)              # the branch started from main's tree
    # 3. The count. Every release measured the store; with a generation of this campaign stored, none left it counted
    # at nothing; none counted main's seed, the earlier attempt or its orphan.
    holding = [r for r in rel if r["counted_bytes"]]
    assert len(holding) >= 4 and all(r["counted_bytes"] >= big + 1000 for r in holding)
    assert all(r["stored_bytes"] >= r["counted_bytes"] + there["seed"] + there["kept"] + there["orphan"] for r in rel)
    assert rel[-1]["counted_bytes"] == 4 * big + 3 * 1000 == state["store_pushed_bytes"]     # g300 to g500 and the mark
    assert rel[-1]["stored_bytes"] == rel[-1]["counted_bytes"] + there["seed"] + there["kept"] + there["orphan"]
    # Session two started from what was counted when session one ended: two generations, the mark's weights being
    # g100's own (once, not twice), and its courier counted on from there.
    assert vm_cfg["pushed_bytes_start"] == 2 * big + 2 * 1000 and vm_cfg["push_cap_bytes"] == 100 * 10 ** 9
    st2 = json.loads((s2 / "courier_status.json").read_text())
    assert st2["bytes_since_release"] >= 3 * big and st2["measure_seq"] >= 1 and st2["capped"] is None
    measures = [m for m in stat_lines(c1 + c2, "measure") if m["store_bytes"]]
    assert measures and all(m["bytes_since_release"] >= big for m in measures)
    # 4. The watcher's samples of the session the drill lost.
    assert (s1 / "watch__watch.jsonl").read_text() == "watch output watch.jsonl launch 1\n"
    assert json.loads((s1 / "index.json").read_text())["files"]["watch/watch.jsonl"]["whole"] is True
    for text in (calls, log, json.dumps(state), c1, c2, p.stdout, p.stderr):
        assert TOKEN not in text


BOOM_WRAP = run_eq.WRAP.replace("rm.main()\n", """def boom(*a, **kw):
    raise RuntimeError("the boot could not be launched")
rm.boot_launch_code = boom
rm.main()
""")


def test_a_failure_in_the_first_launch_still_removes_the_token_and_stops_the_session(tmp_path):
    """The second review (12 Nov 2026): the first launch() ran before the runner's try/finally, so a failure after
    the session had been created (here: after the token has been handed over and adopted, when the boot is
    launched) left a paid session up with the token on it. It is inside the clean-up now: the token goes, the
    session's files are asked for, the session is stopped with confirmation, and the runner exits non-zero."""
    assert "rm.boot_launch_code = boom" in BOOM_WRAP
    work = tmp_path / "run"
    work.mkdir()
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"), wrap=BOOM_WRAP)
    log = log_of(work)
    assert p.returncode == 1 and "RuntimeError: the boot could not be launched" in p.stderr, (p.stderr[-1500:], log[-1500:])
    cl = calls.splitlines()
    assert news(calls) == 1 and cl[-1].startswith('["stop"')                    # one session was created, and stopped
    assert json.loads((work / "fake" / "fake.json").read_text())["alive"] is False
    ups = [k for k, ln in enumerate(cl) if ln.startswith('["upload"') and "store_token" in ln]
    assert len(ups) == 1                                                         # the token had reached the VM
    lines = [ln[15:] for ln in log.splitlines()]
    i = lines.index("store: config and token file sent (dir:%s, branches ['ckpt-N'], epoch 1)" % (work / "store"))
    assert lines[i + 1] == "store: courier ended and token removed from the VM (before the session stops)"
    assert lines[-1] == "session stopped; runner exiting" and not (work / "state" / "runner.pid").exists()
    held = [str(q) for q in [*(work / "fake" / "vm").rglob("*"), *(work / "shm").rglob("*")] if q.is_file()
            and q.stat().st_size < 4096 and TOKEN.encode() in q.read_bytes()]
    assert held == [] and not (work / "state" / "STOP_UNRESOLVED").exists()
    for text in (calls, log, p.stdout, p.stderr):
        assert TOKEN not in text


PREFLIGHT_BOOM_WRAP = run_eq.WRAP.replace("rm.main()\n", """def boom(*a, **kw):
    raise RuntimeError("the store could not be asked")
rm.store_preflight = boom
rm.main()
""")


def test_a_failure_before_any_session_exists_ends_cleanly(tmp_path):
    """The other side of the first launch's clean-up: the failure comes before a session was ever created. The
    clean-up runs with nothing to clean: no session is created, the stop finds none and is not an alarm (no
    STOP_UNRESOLVED, exit 1 for the failure itself, not 4), and the pid file goes."""
    work = tmp_path
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start"), wrap=PREFLIGHT_BOOM_WRAP)
    log = log_of(work)
    assert p.returncode == 1 and "RuntimeError: the store could not be asked" in p.stderr, (p.stderr[-1500:], log[-1500:])
    assert news(calls) == 0 and not any(ln.startswith('["upload"') for ln in calls.splitlines())
    assert not (work / "state" / "STOP_UNRESOLVED").exists() and not (work / "state" / "HALTED").exists()
    assert not (work / "state" / "runner.pid").exists()
    lines = [ln[15:] for ln in log.splitlines()]
    assert lines[-1] == "session stopped; runner exiting" and "store: session files NOT brought home" not in log
    assert not (work / "state" / "sessions").exists() and not (work / "fake" / "vm").exists()
    for text in (calls, log, p.stdout, p.stderr):
        assert TOKEN not in text


def test_a_last_generation_that_is_not_whole_halts_the_campaign_at_once(tmp_path):
    """The run's last generation arrives short on the hub (its commit is accepted). It is written once, so nothing
    will mend it. The courier names it; the runner's own check, the same one made from home (the listing, the
    manifest, a byte of each large file), finds it in the store and not whole and halts at once, by name, instead
    of waiting out the settle window (95 minutes of an idle session) for something that cannot come."""
    pytest.importorskip("huggingface_hub")
    work = tmp_path
    hub = work / "hub"
    hub.mkdir()
    (hub / "damage.json").write_text(json.dumps({"path": "gens/g300/model.safetensors", "bytes": 1000}))
    extra = [*STORE_ARGS, "--store-repo", "someone/base2-store", "--store-token-file", str(work / "token"), "--store-first-start"]
    p, calls = run_store(work, {}, jobs=reh_jobs(), extra=extra, more_env=hub_env(hub))
    log = log_of(work)
    assert p.returncode == 3, (p.stderr[-2000:], log[-3000:])
    halted = json.loads((work / "state" / "HALTED").read_text())
    assert halted["kind"] == "STORE_FINAL_GENERATION_NOT_WHOLE" and (halted["job"], halted["step"], halted["branch"]) == ("pt", 300, "ckpt-N")
    assert halted["checks"] == 3 and "(check 1 of 3: " in log and "(check 2 of 3: " in log   # read at three ticks before the halt
    assert len(halted["seen"]) == 3 and all("model.safetensors is listed at 1000 bytes, the manifest says" in c["saw"] for c in halted["seen"])
    assert state_of(work)["durability_pending"]["pt"]["final_checks"] == halted["seen"]         # kept in state.json as they were made
    assert "is not whole" in halted["why"] and "durability pending" not in log        # at once: no settle window
    state = state_of(work)
    assert state["jobs"]["pt"]["home"] is False and not state.get("campaign_home")
    assert calls.rstrip().splitlines()[-1].startswith('["stop"') and json.loads((work / "fake" / "fake.json").read_text())["alive"] is False
    s1 = work / "state" / "sessions" / "launch01"
    marker = [json.loads(ln) for ln in (s1 / "COURIER_NOT_WHOLE").read_text().splitlines()]
    assert [(m["job"], m["step"], m["last"]) for m in marker] == [("pt", 300, True)]
    st1 = json.loads((s1 / "courier_status.json").read_text())["jobs"]["pt"]
    assert st1["not_whole"] == [300] and st1["verified"] == [100, 200] and st1["to_verify"] == []
    sys.path.insert(0, str(ROOT / "tests"))
    store = hf_store.open_store({"repo": "someone/base2-store", "token_file": str(work / "token")},
                                api=__import__("hub_standin").DiskApi(str(hub)))
    assert store.steps("ckpt-N") == [100, 200, 300] and store.verify_by_download("ckpt-N", 200) and store.verify_by_download("ckpt-N", 100)
    assert store.pull_generation("ckpt-N", str(work / "back"))[0] == 200              # what a person resumes from
    for text in (calls, log, json.dumps(state), p.stdout, p.stderr):
        assert TOKEN not in text


def test_the_runner_refuses_a_grace_or_a_settle_window_that_is_too_short(tmp_path):
    """--store-release-grace-s under the store's minimum (a restore may be reading what was just removed), and a
    --store-settle-min shorter than the courier's final delivery window and a tick, are refused before anything is
    created."""
    work = tmp_path
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage",
                                                   "--store-release-grace-s", "599"))
    assert p.returncode == 2 and "--store-release-grace-s is at least 600" in p.stderr and news(calls) == 0
    for settle, wait in (("89", None), ("10", "900")):
        p, calls = run_store(work, {}, fresh=False, extra=store_args(
            work, "--store-first-start", "--store-settle-min", settle, *(["--store-courier-wait-s", wait] if wait else [])))
        assert p.returncode == 2 and "--store-settle-min is at least" in p.stderr and news(calls) == 0, p.stderr[-800:]
    runner = (CC / "runner_multi.py").read_text()
    assert '"--store-settle-min", type=float, default=95' in runner and "STORE_COURIER_WAIT_S = 5400" in runner
    assert '"--store-commit-budget-per-hour", type=int, default=24' in runner
    assert '"--store-home-commit-budget-per-hour", type=int, default=12' in runner
    assert 3 * 24 + 3 * 12 == 108 < 128                                          # three arms' ceilings, under the hub's limit


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
    assert store.steps("ckpt-N") == [100, 200, 300] and store.mark_steps("pt") == [100]
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
    assert store.steps("ckpt-N-try-2") == [100, 200, 300] and store.read_manifest("ckpt-N-try-2", "gens/g300")["campaign"] == "try 2"
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


# ---------------------------------------------------------------- the arms' round (third reviews and the rerun, 12 Nov 2026)
def load_runner(home):
    """runner_multi.py as a module in this process, with its state folder: for the functions that need no VM."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner_multi_inproc", CC / "runner_multi.py")
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    os.makedirs(home, exist_ok=True)
    rm.HOME, rm.NAME = str(home), "pk-inproc"
    return rm


class St:
    def __init__(self):
        self.d, self.jobs, self.saved = {}, {}, 0

    def job(self, n):
        return self.jobs.setdefault(n, {"home": False, "relayed_step": 0})

    def save(self):
        self.saved += 1


class FinalStore:
    """What store_durable asks of the store, with the final generation's check answering as a test says."""

    def __init__(self, answers):
        self.answers, self.asked, self.b = list(answers), 0, self

    def tree(self, branch):
        return {}

    def steps(self, branch, tree):
        return [100, 200, 300]

    def why_not_whole(self, branch, step):
        self.asked += 1
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return [a, None if a else f"reading {self.asked}: optim.safetensors is not listed"]

    def read_manifest(self, branch, prefix):
        return {"job": "pt"}

    def mark_steps(self, n):
        return []


def durable(tmp_path, answers):
    rm = load_runner(tmp_path / "home")
    rm.STORE = {"settle_min": 95, "branch_suffix": ""}
    store = FinalStore(answers)
    rm.store_open = lambda jobs: store
    rm.store_call = lambda fn, limit=None: fn()
    j = {"name": "pt", "spec": {"train": {"steps": 300}}}
    return rm, store, St(), j


def test_one_bad_listing_does_not_halt_a_finished_run(tmp_path):
    """The third review: a finished run's last generation that read "in the store and not whole" ONCE halted the
    campaign at once, and one listing that is transiently wrong could do that to four days of training. The check
    is now made again, three completed checks in all, before STORE_FINAL_GENERATION_NOT_WHOLE: a repeat that reads
    whole is whole, a repeat that cannot be read leaves the job pending, and three that differ halt as before."""
    # The reviews of the arms' round: the three checks ran inside one bounded store call, which a slow store could
    # cut short (the count then began again at the next tick), and they spanned a minute. They are one a tick now
    # (eight minutes in all), each kept in state.json as it is made, with what it saw.
    # One bad reading, then the store's state: durable, nothing halted. One store read a tick.
    rm, store, st, j = durable(tmp_path / "a", [False, True])
    assert rm.store_durable(st, [j], j) is False and store.asked == 1          # not durable yet, and not halted
    kept = st.d["durability_pending"]["pt"]["final_checks"]
    assert [c["saw"] for c in kept] == ["reading 1: optim.safetensors is not listed"] and st.saved >= 1
    assert rm.store_durable(st, [j], j) is True and store.asked == 2
    log = (tmp_path / "a" / "home" / "runner.log").read_text()
    assert "the final generation g300 read as in the store and not whole (check 1 of 3: reading 1: optim" in log
    assert "the final generation g300 reads whole after 1 reading(s) that differed" in log
    assert not (tmp_path / "a" / "home" / "HALTED").exists() and "durability pending" not in log
    assert not st.d.get("durability_pending")
    # Twice bad, then whole.
    rm, store, st, j = durable(tmp_path / "b", [False, False, True])
    assert [rm.store_durable(st, [j], j) for _ in range(3)] == [False, False, True]
    assert store.asked == 3 and not (tmp_path / "b" / "home" / "HALTED").exists()
    # A check that cannot be read in between: pending, and the differing one is NOT forgotten.
    import hf_store as H
    refused = H.StoreError("HTTP 403 on the byte read of ckpt-pt:gens/g300/model.safetensors (from the object store, "
                           "after the hub's redirect; refused)")
    rm, store, st, j = durable(tmp_path / "c", [False, refused, refused, refused, refused, refused, False, False, True])
    assert rm.store_durable(st, [j], j) is False and rm.store_durable(st, [j], j) is False and store.asked == 2
    pend = st.d["durability_pending"]["pt"]
    assert (pend["unread"], pend["http_status"]) == (1, 403) and "store not read (StoreError: HTTP 403" in pend["why"][0]
    assert len(pend["final_checks"]) == 1                                       # kept across the unread one
    # The third reviews' other point: the same refusal every time is said apart at home, with its status.
    for k in range(2, 6):
        assert rm.store_durable(st, [j], j) is False and st.d["durability_pending"]["pt"]["unread"] == k
    log = (tmp_path / "c" / "home" / "runner.log").read_text()
    loud = [ln for ln in log.splitlines() if "STORE CHECK UNREAD" in ln]
    assert len(loud) == 1 and "g300 could not be read from home 5 times running (HTTP status: 403)" in loud[0]
    assert rm.store_durable(st, [j], j) is False and len(st.d["durability_pending"]["pt"]["final_checks"]) == 2
    with pytest.raises(SystemExit) as e:                                        # the third that differs, six ticks after the first
        rm.store_durable(st, [j], j)
    halted = json.loads((tmp_path / "c" / "home" / "HALTED").read_text())
    assert e.value.code == 3 and halted["kind"] == "STORE_FINAL_GENERATION_NOT_WHOLE" and halted["checks"] == 3
    assert [c["saw"].split(":")[0] for c in halted["seen"]] == ["reading 1", "reading 7", "reading 8"]
    # Three completed checks that differ, tick after tick: the halt, by name, with what each saw.
    rm, store, st, j = durable(tmp_path / "d", [False, False, False, True])
    assert rm.store_durable(st, [j], j) is False and rm.store_durable(st, [j], j) is False
    with pytest.raises(SystemExit) as e:
        rm.store_durable(st, [j], j)
    halted = json.loads((tmp_path / "d" / "home" / "HALTED").read_text())
    assert e.value.code == 3 and store.asked == 3 and halted["kind"] == "STORE_FINAL_GENERATION_NOT_WHOLE"
    assert (halted["job"], halted["step"], halted["checks"]) == ("pt", 300, 3) and len(halted["seen"]) == 3
    assert all(c["saw"] and c["at"] for c in halted["seen"])
    assert rm.STORE_FINAL_CHECKS == 3 and not hasattr(rm, "STORE_FINAL_RECHECK_S")
    # The settle window's halt carries the count of unread checks and the status.
    rm, store, st, j = durable(tmp_path / "f", [refused, refused])
    assert rm.store_durable(st, [j], j) is False
    st.d["durability_pending"]["pt"]["since"] -= 96 * 60
    with pytest.raises(SystemExit):
        rm.store_durable(st, [j], j)
    halted = json.loads((tmp_path / "f" / "home" / "HALTED").read_text())
    assert halted["kind"] == "STORE_DURABILITY_PENDING" and (halted["unread_checks_running"], halted["http_status"]) == (2, 403)


def test_the_runner_repeats_the_couriers_unread_report_at_home(tmp_path):
    """A check the courier could not read ten times running is in its status ("check_unread") and in
    CHAT/COURIER_CHECK_UNREAD. The runner reads the status on every tick and says it in its own log, once for each
    generation, with the status the courier met; the marker comes home with the session's files."""
    rm = load_runner(tmp_path / "home")
    rm.STORE = {}
    st = St()

    def vm(count, step="300"):
        return {"courier": {"jobs": {"pt": {"check_unread": {step: {"count": count, "http_status": 403, "error": "StoreError, HTTP 403"}}}}}}

    rm.store_guard(st, [], vm(9))
    assert not (tmp_path / "home" / "runner.log").exists() and "store_check_unread_said" not in st.d
    rm.store_guard(st, [], vm(10))
    rm.store_guard(st, [], vm(11))
    rm.store_guard(st, [], {"courier": {"jobs": {"pt": {"check_unread": {}}}}})
    loud = [ln for ln in (tmp_path / "home" / "runner.log").read_text().splitlines() if "STORE CHECK UNREAD" in ln]
    assert len(loud) == 1 and "pt: STORE CHECK UNREAD on the VM" in loud[0] and "g300 10 times running (StoreError, HTTP 403)" in loud[0]
    rm.store_guard(st, [], vm(10, "400"))
    assert st.d["store_check_unread_said"] == ["pt/g300", "pt/g400"]
    assert "'check_unread'" in rm.STORE_VM_READ and "'COURIER_CHECK_UNREAD'" in rm.BRING_HOME


def test_a_store_that_takes_pushes_and_verifies_none_halts_the_arm_when_the_registration_says_so(tmp_path):
    """The reviews of the arms' round: checks that stay unread, or generations that are not whole every time, were
    reported and nothing more; with the arms' push cap at 0 the store then grows by a generation a save for days
    and training bills with nothing that counts. The courier's status carries "unverified_since"; with
    --store-unverified-halt-min (or the jobs file's "store_unverified_halt_min"; off by default) the runner halts
    the campaign by name when that is older than the limit, by the VM's own clock. --store-stall-halt remains what
    covers pushes that do not arrive. The hub's refusal of a token is named as an authentication failure, on the
    VM and at home, and no address's query string reaches the log or the state."""
    rm = load_runner(tmp_path / "home")
    st = St()

    def vm(minutes, **more):
        return {"at": 50_000.0, "courier": {"jobs": {"pt": {
            "unverified_since": 50_000.0 - minutes * 60, "to_verify": [300, 400], "not_whole": [200], "pushed_step": 400,
            "check_unread": {"400": {"count": 60, "http_status": 403, "error": "StoreError, HTTP 403 from the object store"}},
            "auth_failure": None, **more}}}}

    rm.STORE = {}                                        # the default: reported, never halted
    rm.store_guard(st, [], vm(600))
    assert not (tmp_path / "home" / "HALTED").exists()
    rm.STORE = {"unverified_halt_min": 30}
    rm.store_guard(st, [], vm(29))
    rm.store_guard(st, [], {"at": 50_000.0, "courier": {"jobs": {"pt": {"unverified_since": None}}}})   # verified: nothing open
    assert not (tmp_path / "home" / "HALTED").exists()
    with pytest.raises(SystemExit) as e:
        rm.store_guard(st, [], vm(31))
    halted = json.loads((tmp_path / "home" / "HALTED").read_text())
    assert e.value.code == 3 and halted["kind"] == "STORE_UNVERIFIED" and (halted["job"], halted["minutes"], halted["limit_min"]) == ("pt", 31, 30)
    assert halted["to_verify"] == [300, 400] and halted["not_whole"] == [200] and halted["check_unread"]["400"]["http_status"] == 403
    runner = (CC / "runner_multi.py").read_text()
    assert '"--store-unverified-halt-min", type=float, default=None' in runner
    assert 'doc.get("store_unverified_halt_min")' in runner and 'doc.get("store_stall_halt")' in runner
    assert "'unverified_since'" in rm.STORE_VM_READ and "'auth_failure'" in rm.STORE_VM_READ
    # The hub refuses the courier's token: said as that, once for each run of refusals.
    rm2 = load_runner(tmp_path / "home2")
    rm2.STORE = {}
    st2 = St()
    au = {"since": 7.0, "count": 3, "http_status": 401, "during": "pt: generation g500 not pushed"}
    for _ in range(3):
        rm2.store_guard(st2, [], vm(5, auth_failure=au))
    rm2.store_guard(st2, [], vm(5, auth_failure={**au, "since": 9000.0}))
    log = (tmp_path / "home2" / "runner.log").read_text()
    loud = [ln for ln in log.splitlines() if "STORE AUTHENTICATION FAILURE on the VM" in ln]
    assert len(loud) == 2 and "HTTP 401, during pt: generation g500 not pushed" in loud[0]
    # And home's own: by name once, with what mends it; the object store's 403 is not one.
    import hf_store as H
    rm2.store_auth(st2, "pt", H.StoreError("HTTP 403 on the byte read of b:p (from the object store, after the hub's redirect; refused)"))
    assert "store_auth_failure" not in st2.d
    for _ in range(2):
        rm2.store_auth(st2, "pt", Exception("401 Client Error. Repository Not Found for url: https://huggingface.co/api/datasets/x/tree/ckpt-N"))
    log = (tmp_path / "home2" / "runner.log").read_text()
    home = [ln for ln in log.splitlines() if "STORE AUTHENTICATION FAILURE at home" in ln]
    assert len(home) == 1 and "HTTP 401" in home[0] and "replace the token file" in home[0] and st2.d["store_auth_failure"]["http_status"] == 401
    signed = ConnectionError("Max retries exceeded with url: /repos/ab/cd/0123?X-Amz-Signature=SECRETSIG&X-Amz-Date=1 (Caused by x)")
    assert rm2.store_err(signed) == "ConnectionError: Max retries exceeded with url: /repos/ab/cd/0123?[query removed] (Caused by x)"


OWED_WRAP = run_eq.WRAP.replace("rm.main()\n", """import os
rm.STORE_FINAL_RELEASE_WAIT_S, rm.STORE_FINAL_RELEASE_NAP_S = float(os.environ["OWED_WAIT_S"]), 1
rm.STORE_HOME_ESSENTIAL_WAIT_S = float(os.environ["OWED_ESSENTIAL_S"])
rm.main()
""")


def full_home_budget(work, n=12):
    """HOME/store_commits.json as an hour leaves it in which home has spent its whole commit budget."""
    import time
    (work / "state").mkdir()
    (work / "state" / "store_commits.json").write_text(json.dumps(
        {"commits": [[time.time(), "lock"]] * n, "blocked_until": 0, "refused": 0, "http_429": 0, "reported_at": None,
         "essential_since": None}))


def test_the_release_at_the_campaigns_end_waits_for_room_after_the_session_has_stopped(tmp_path):
    """The rerun of 12 Nov 2026 (PROTOCOL 47.2): home's commit budget had no room for the lock when the campaign
    ended, the release did not run, and six ledgers stayed in the store with one quiet line in the log. The
    release at the end is essential now and is owed: when it cannot run in the loop it is made after the session
    has been stopped (the wait bills nothing), as soon as the budget's essential wait is over."""
    work = tmp_path
    full_home_budget(work)
    assert "rm.STORE_FINAL_RELEASE_WAIT_S" in OWED_WRAP
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), wrap=OWED_WRAP,
                         more_env={"OWED_WAIT_S": "60", "OWED_ESSENTIAL_S": "3"})
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-3000:])
    state = state_of(work)
    assert state["campaign_home"] and "store_release_final_owed" not in state
    lines = [ln[15:] for ln in log.splitlines()]
    held = [i for i, ln in enumerate(lines) if ln.startswith("store: release: the lock was not taken (budget: room in ")]
    owed = [i for i, ln in enumerate(lines) if ln.startswith("store: the release at the campaign's end is owed (the lock was not taken (budget: room in ")]
    home = [i for i, ln in enumerate(lines) if ln.startswith("store: session files home in sessions/")]
    ran = [i for i, ln in enumerate(lines) if ln.startswith("store: stat ") and json.loads(ln[12:])["what"] == "release"]
    assert len(held) == 1 and len(owed) == 1 and len(ran) == 1 and json.loads(lines[ran[0]][12:])["final"] is True   # one line a state, not a try
    assert held[0] < home[-1] < owed[0] < ran[0] and lines[-1] == "session stopped; runner exiting"   # tried, stopped, made
    assert calls.rstrip().splitlines()[-1].startswith('["stop"')                # no call to a VM after the stop
    assert state["store_releases"]["count"] == 1 and "DID NOT RUN" not in log and state["store_release_final_done"] is True
    assert state["store_releases"]["held"] >= 2                                # tried more than once, said once
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert sent[-2:] == ["lock: release by home:pk-store", "unlock: release"] and sent.count("unlock: release") == 1
    assert not (work / "state" / "store_release_by_hand.json").exists() and TOKEN not in log


def test_a_release_at_the_campaigns_end_that_cannot_run_says_so_and_names_the_command(tmp_path):
    """The same end with no room inside the bounded wait: ONE line says the release did not run, why, and the
    command that makes it by hand; the command works as written."""
    work = tmp_path
    full_home_budget(work)
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), wrap=OWED_WRAP,
                         more_env={"OWED_WAIT_S": "2", "OWED_ESSENTIAL_S": "900"})
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-3000:])          # the campaign is home all the same
    state = state_of(work)
    assert state["campaign_home"] and state["store_release_final_owed"].startswith("the lock was not taken (budget: room in ")
    lines = [ln[15:] for ln in log.splitlines()]
    said = [ln for ln in lines if "THE RELEASE AT THE CAMPAIGN'S END DID NOT RUN" in ln]
    cfg = work / "state" / "store_release_by_hand.json"
    layout = work / "repo" / "tools" / "colab_chat"
    command = f"python3 {layout / 'hf_store.py'} release --config {cfg} --grace 7200 --owner hand:pk-store"
    assert len(said) == 1 and ("By hand: " + command + " (it prints") in said[0] and "budget: room in " in said[0]
    assert '"released": null means the lock was held and NOTHING was released' in said[0]
    assert lines[-2] == said[0] and lines[-1] == "session stopped; runner exiting"
    assert len([ln for ln in lines if ln.startswith("store: release: the lock was not taken")]) == 1   # one line a state
    import stat as stat_mod
    assert stat_mod.S_IMODE(cfg.stat().st_mode) == 0o600 and not list((work / "state").glob("store_release_by_hand.json.tmp*"))
    assert calls.rstrip().splitlines()[-1].startswith('["stop"') and "store_releases" in state and state["store_releases"]["count"] == 0
    by_hand = json.loads(cfg.read_text())
    assert by_hand == {"repo": f"dir:{work / 'store'}", "repo_type": "dataset", "token_file": str(work / "token"), "mark_ns": "",
                       "store_commit_budget_per_hour": 0}
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert not any(m.startswith(("lock", "unlock")) for m in sent)
    # The debt is in state.json. The same state folder started again: every job is home, so NO session is created
    # (Colab too), no VM is asked anything, and the release is made (the budget's essential wait is long over).
    n_calls = len(calls.splitlines())
    p2, calls2 = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), fresh=False,
                           more_env={"OWED_WAIT_S": "60", "OWED_ESSENTIAL_S": "3"})
    log2 = log_of(work)[len(log):]
    assert p2.returncode == 0, (p2.stderr[-2000:], log2[-3000:])
    assert news(calls2) == 1 and ("every job is home in this state folder: no session is created; the listing answers and holds no "
                                  "session of this name: no token to remove, nothing to stop; the release at the campaign's end is owed") in log2
    later = calls2.splitlines()[n_calls:]
    assert later and all(ln.startswith('["sessions"') for ln in later)          # the listing was asked, and nothing else
    state2 = state_of(work)
    assert state2["store_release_final_done"] is True and "store_release_final_owed" not in state2 and state2["store_releases"]["count"] == 1
    assert "DID NOT RUN" not in log2 and "NOT brought home" not in log2 and log2.rstrip().endswith("session stopped; runner exiting")
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert sent[-2:] == ["lock: release by home:pk-store", "unlock: release"]
    # A third start has nothing left to do, and commits nothing.
    p3, calls3 = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), fresh=False,
                           more_env={"OWED_WAIT_S": "60", "OWED_ESSENTIAL_S": "3"})
    assert p3.returncode == 0 and news(calls3) == 1
    assert len((work / "store" / "commits.jsonl").read_text().splitlines()) == len(sent)
    state, log = state2, log_of(work)
    # The by-hand command, as printed, works (here after the fact: nothing is left to release).
    args = command.split()
    q = subprocess.run([sys.executable, *args[1:]], capture_output=True, text=True, timeout=120)
    assert q.returncode == 0 and json.loads(q.stdout)["released"] == 0, (q.stdout, q.stderr)
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert sent[-2:] == ["lock: release by hand:pk-store", "unlock: release"]
    for text in (calls, log, json.dumps(state), cfg.read_text(), p.stdout, p.stderr, q.stdout, q.stderr):
        assert TOKEN not in text


# ---------------------------------------------------------------- the arms' round, second pass (its two reviews, 12 Nov 2026)
DIE_WRAP = run_eq.WRAP.replace("rm.main()\n", """import os, hf_store
at = os.environ.get("DIE_AT")
if at == "before":          # the runner dies as the release's call begins
    def die(self, *a, **kw):
        os.kill(os.getppid(), 9)
        os._exit(1)
    hf_store.Store.release_report = die
elif at == "during":        # ... inside it: the lock is taken, the round has run, the lock is not removed
    def die_in(self, mine, done=None):
        os.kill(os.getppid(), 9)
        os._exit(1)
    hf_store.Store._unlock = die_in
elif at == "orphan":        # ... SIGKILL of the runner ALONE: the release's child is still inside its store call
    def linger(self, mine, done=None):
        import time
        open(os.environ["DIE_PIDFILE"], "w").write(str(os.getpid()))
        os.kill(os.getppid(), 9)
        time.sleep(600)
    hf_store.Store._unlock = linger
elif at == "after":         # ... after it has succeeded, before the success is written to state.json
    real = rm.store_call
    def call(fn, limit=None):
        r = real(fn, limit)
        if isinstance(r, dict) and r.get("held") is False:
            os.kill(os.getpid(), 9)
        return r
    rm.store_call = call
rm.main()
""")


def ended(pid):
    """True when the process is gone, or dead and not yet reaped."""
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


@pytest.mark.parametrize("at", ["before", "during", "orphan", "after"])
def test_a_runner_that_dies_at_the_campaigns_end_leaves_its_release_as_a_debt_the_next_start_pays(tmp_path, at):
    """The reviews of the arms' round: the release at the campaign's end was written to state.json as owed only
    after it had been attempted, and nothing at start-up read it; a runner killed before, during or just after
    the release left six ledgers' worth of storage with no word, and on Colab the restart created a new session
    to find that everything was home. Now the debt is on disk before the call; a start on a state folder in which
    every job is home creates NO session, stops the one the dead runner left (taking the token off it first), and
    pays the debt; a lock its dead self left on main is its own and is taken again."""
    work = tmp_path
    args = store_args(work, "--store-first-start", "--store-release-storage")
    pidfile = work / "release_child.pid"
    p, calls = run_store(work, {}, extra=args, wrap=DIE_WRAP, more_env={"DIE_AT": at, "DIE_PIDFILE": str(pidfile)})
    assert p.returncode == -9, (p.returncode, p.stderr[-1500:], log_of(work)[-2000:])
    if at == "orphan":
        # The second reviews of the arms' round: only the runner was killed. Its release, in a forked child, went on
        # by itself, holding the lock and the token, while a new runner could take that lock again by its name.
        # The child dies with its parent now (a parent-death signal), and the lock's record names its pid, so the
        # new runner takes the lock again because that process is gone, not because of the name.
        import time
        child = int(pidfile.read_text())
        try:
            for _ in range(50):
                if ended(child):
                    break
                time.sleep(0.1)
            assert ended(child), "the release's child outlived the runner that was killed"
        finally:
            if not ended(child):
                os.kill(child, 9)
    state = state_of(work)
    assert all(state["jobs"][n]["home"] for n in ("data", "pt", "rung_100")) and not state.get("campaign_home")
    assert state["store_release_final_owed"] == "not tried yet" and "store_release_final_done" not in state   # on disk before the call
    assert json.loads((work / "fake" / "fake.json").read_text())["alive"] is True                    # the dead runner's session bills
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert sent.count("lock: release by home:pk-store") == (0 if at == "before" else 1)
    assert sent.count("unlock: release") == (1 if at == "after" else 0)
    first = log_of(work)
    p2, calls2 = run_store(work, {}, extra=args, fresh=False)
    log = log_of(work)[len(first):]
    assert p2.returncode == 0, (p2.stderr[-2000:], log[-3000:])
    assert news(calls2) == 1                                                       # no second session, on Colab
    assert "every job is home in this state folder: no session is created; the session an earlier runner left is stopped; " \
           "the release at the campaign's end is owed" in log
    lines = [ln[15:] for ln in log.splitlines()]
    i_tok = lines.index("store: courier ended and token removed from the VM (before the session stops)")
    i_owed = next(i for i, ln in enumerate(lines) if ln.startswith("store: the release at the campaign's end is owed"))
    i_ran = next(i for i, ln in enumerate(lines) if ln.startswith("store: stat ") and json.loads(ln[12:])["what"] == "release")
    assert i_tok < i_owed < i_ran and lines[-1] == "session stopped; runner exiting"
    assert calls2.rstrip().splitlines()[-1].startswith('["stop"') and json.loads((work / "fake" / "fake.json").read_text())["alive"] is False
    state = state_of(work)
    assert state["store_release_final_done"] is True and "store_release_final_owed" not in state and state["campaign_home"]
    assert "DID NOT RUN" not in log and not (work / "state" / "store_release_by_hand.json").exists()
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert sent.count("lock: release by home:pk-store") == {"before": 1, "during": 2, "orphan": 2, "after": 2}[at]
    assert sent[-1] == "unlock: release" and not (work / "store" / "main" / "locks" / "release.json").exists()
    assert ("one left by home:pk-store itself is taken again (its process is gone)" in log) == (at in ("during", "orphan"))
    held = [str(q) for q in [*(work / "fake" / "vm").rglob("*"), *(work / "shm").rglob("*")] if q.is_file()
            and q.stat().st_size < 4096 and TOKEN.encode() in q.read_bytes()]
    assert held == []
    for text in (calls2, log_of(work), json.dumps(state), p2.stdout, p2.stderr):
        assert TOKEN not in text


def test_a_stop_that_is_not_confirmed_does_not_drop_the_release_that_is_owed(tmp_path):
    """The reviews of the arms' round: with STOP_UNRESOLVED the owed release was skipped without a word. It is not
    tried then (a session may still be up), and it is not dropped: the one line and its by-hand config are written,
    and the debt stays in state.json."""
    work = tmp_path
    full_home_budget(work)
    p, calls = run_store(work, {"stop_fail_from_status": 1}, extra=store_args(work, "--store-first-start", "--store-release-storage"),
                         wrap=OWED_WRAP, more_env={"OWED_WAIT_S": "60", "OWED_ESSENTIAL_S": "900"})
    log = log_of(work)
    assert p.returncode == 4 and (work / "state" / "STOP_UNRESOLVED").exists(), (p.stderr[-1500:], log[-2500:])
    lines = [ln[15:] for ln in log.splitlines()]
    said = [ln for ln in lines if "THE RELEASE AT THE CAMPAIGN'S END DID NOT RUN" in ln]
    assert len(said) == 1 and "the session's stop is not confirmed (STOP_UNRESOLVED), so the release was not tried; " in said[0]
    assert "hf_store.py release --config" in said[0] and (work / "state" / "store_release_by_hand.json").exists()
    assert lines[-2] == said[0] and lines[-1] == "SESSION STOP NOT CONFIRMED (STOP_UNRESOLVED); runner exiting"
    assert not any(ln.startswith("store: the release at the campaign's end is owed") for ln in lines)   # nothing was tried
    state = state_of(work)
    assert state["store_release_final_owed"].startswith("the lock was not taken (budget") and state["store_releases"]["held"] == 1


def owed(tmp_path, home="home", debt="the lock was not taken (another releaser holds the lock)"):
    rm = load_runner(tmp_path / home)
    rm.STORE = {"reclaim": True, "release_grace_s": 7200, "repo": "someone/base2-store", "repo_type": "dataset",
                "token_file": str(tmp_path / "a token"), "mark_ns": ""}
    rm.STORE_FINAL_RELEASE_WAIT_S, rm.STORE_FINAL_RELEASE_NAP_S = 3, 1
    st = St()
    st.d = {"store_release_final_owed": debt}
    return rm, st


def test_the_owed_release_keeps_to_its_bound_spends_no_commits_on_a_ledger_that_will_not_read_and_always_says_so(tmp_path):
    """The reviews of the arms' round. (1) "At most 20 minutes" was not one: a try that began near the end of the
    wait ran for its own limit, 15 minutes or more. Each try is now given what is left of the wait as its limit.
    (2) A round that ran and met a ledger it could not read was run again every 30 seconds, a lock and an unlock
    each time. It is run once more and no further, and not at all again for a ledger that is not well formed.
    (3) The line that says the release did not run is written on every way out, names such a ledger by branch and
    path, quotes its paths, and says what exit 0 with "released": null means; the config is mode 600."""
    import shlex
    import stat as stat_mod
    import time
    # (1) Every try is as slow as it is allowed to be.
    rm, st = owed(tmp_path, "a")
    limits = []

    def slow(st_, jobs, final=False, limit=None):
        limits.append(limit)
        time.sleep(min(limit, 900))
        return "held"

    rm.store_release = slow
    t0 = time.time()
    rm.store_release_owed(st, [])
    assert time.time() - t0 < 3 + 1 and limits and all(0 < x <= 3 for x in limits) and len(limits) == 1
    log = (tmp_path / "a" / "runner.log").read_text()
    assert log.count("THE RELEASE AT THE CAMPAIGN'S END DID NOT RUN") == 1
    # Quick tries: several inside the wait, each with a smaller limit, and still the one line.
    rm, st = owed(tmp_path, "b")
    limits = []
    rm.store_release = lambda st_, jobs, final=False, limit=None: limits.append(limit) or "held"
    t0 = time.time()
    rm.store_release_owed(st, [])
    assert time.time() - t0 < 4 and 2 <= len(limits) <= 4 and limits == sorted(limits, reverse=True)
    assert (tmp_path / "b" / "runner.log").read_text().count("DID NOT RUN") == 1
    # A try that succeeds ends it without the line.
    rm, st = owed(tmp_path, "c")

    def made(st_, jobs, final=False, limit=None):
        st_.d.pop("store_release_final_owed")
        st_.d["store_release_final_done"] = True
        return "done"

    rm.store_release = made
    rm.store_release_owed(st, [])
    assert "DID NOT RUN" not in (tmp_path / "c" / "runner.log").read_text() and not (tmp_path / "c" / "store_release_by_hand.json").exists()
    # A failure of the try itself: the line all the same.
    rm, st = owed(tmp_path, "d")

    def boom(st_, jobs, final=False, limit=None):
        raise RuntimeError("the store could not be opened")

    rm.store_release = boom
    rm.store_release_owed(st, [])
    log = (tmp_path / "d" / "runner.log").read_text()
    assert "stopped on an error (RuntimeError)" in log and log.count("DID NOT RUN") == 1
    # (2) A ledger that was not read: one more round, then no more.
    unread = "the round ran and 1 ledger(s) could not be read: ckpt-N:pruned/g100.json (not read (ConnectionError))"
    rounds = []

    def ledger_round(st_, jobs, final=False, limit=None):   # as store_release leaves state.json after such a round
        rounds.append(1)
        st_.d["store_release_final_rounds"] = st_.d.get("store_release_final_rounds", 0) + 1
        return "ledgers"

    rm, st = owed(tmp_path, "e", debt=unread)
    rm.store_release = ledger_round
    rm.store_release_owed(st, [])
    assert len(rounds) == 2 and st.d["store_release_final_rounds"] == 2 == rm.STORE_FINAL_LEDGER_ROUNDS
    # The second reviews of the arms' round: the count was this process's own, so every new start had its rounds
    # again (a watchdog that starts a finished runner again spent a lock and an unlock each time). It is in
    # state.json: the loop's own round counts, and so do the rounds of every earlier start.
    for home, before, want in (("f", 1, 1), ("f2", 2, 0), ("f3", 5, 0)):
        rm, st = owed(tmp_path, home, debt=unread)
        st.d["store_release_final_rounds"] = before
        del rounds[:]
        rm.store_release = ledger_round
        rm.store_release_owed(st, [])
        assert len(rounds) == want, home
        line = [ln for ln in (tmp_path / home / "runner.log").read_text().splitlines() if "DID NOT RUN" in ln]
        assert len(line) == 1 and ("2 rounds ran and met a ledger that could not be read, and no more are made" in line[0]) == (want == 0)
    # A ledger that is not well formed is kept with its branch's head, and gets NO new round on any start until that
    # head moves or a person removes the key: the start reads the head (a read, no commit), writes the line, ends.
    bad = "the round ran and 1 ledger(s) could not be read: ckpt-N:pruned/g100.json (not well formed)"

    class Heads:
        def __init__(self, head):
            self.head_now, self.b, self.asked = head, self, 0

        def head(self, branch):
            self.asked += 1
            if isinstance(self.head_now, Exception):
                raise self.head_now
            return self.head_now

    for home, head_now, want in (("g", "ckpt-N@7", 0), ("h", ConnectionError("unreachable"), 0), ("h2", "ckpt-N@8", 2)):
        rm, st = owed(tmp_path, home, debt=bad)
        st.d.update(store_release_final_malformed={"ckpt-N": "ckpt-N@7"}, store_release_final_rounds=1)
        store = Heads(head_now)
        rm.store_open = lambda jobs, store=store: store
        rm.store_call = lambda fn, limit=None: fn()
        del rounds[:]
        rm.store_release = ledger_round
        rm.store_release_owed(st, [])
        assert len(rounds) == want and store.asked == 1, home
        line = [ln for ln in (tmp_path / home / "runner.log").read_text().splitlines() if "DID NOT RUN" in ln]
        assert len(line) == 1 and "ckpt-N:pruned/g100.json (not well formed)" in line[0]
        assert "A ledger that is not well formed stops every release, this command's too" in line[0]
        assert ("no new round is made until that head moves or store_release_final_malformed is removed" in line[0]) == (want == 0)
        assert ("store_release_final_malformed" in st.d) == (want == 0)             # the head moved: the key is gone, a round ran
    # A KeyboardInterrupt in the wait: the line all the same, and the interrupt goes on.
    rm, st = owed(tmp_path, "k")

    def interrupted(st_, jobs, final=False, limit=None):
        raise KeyboardInterrupt

    rm.store_release = interrupted
    with pytest.raises(KeyboardInterrupt):
        rm.store_release_owed(st, [])
    assert (tmp_path / "k" / "runner.log").read_text().count("DID NOT RUN") == 1
    runner = (CC / "runner_multi.py").read_text()
    assert "end = time.monotonic() + STORE_FINAL_RELEASE_WAIT_S" in runner and "limit=end - time.monotonic()" in runner
    # (3) The stop not confirmed: nothing is tried, the line is written. Paths with a space are quoted.
    rm, st = owed(tmp_path / "with space", "i")
    rounds = []
    rm.store_release = lambda st_, jobs, final=False, limit=None: rounds.append(1) or "held"
    rm.store_release_owed(st, [], stopped=False)
    home = tmp_path / "with space" / "i"
    line = [ln for ln in (home / "runner.log").read_text().splitlines() if "DID NOT RUN" in ln]
    assert rounds == [] and len(line) == 1 and "the session's stop is not confirmed (STOP_UNRESOLVED)" in line[0]
    cmd = line[0].split("By hand: ", 1)[1].split(" (it prints", 1)[0]
    cfg = home / "store_release_by_hand.json"
    assert shlex.split(cmd) == ["python3", str(CC / "hf_store.py"), "release", "--config", str(cfg), "--grace", "7200",
                                "--owner", "hand:pk-inproc"]
    assert f"'{cfg}'" in cmd                                                    # quoted, because of the space
    assert '"released": null means the lock was held and NOTHING was released' in line[0]
    assert stat_mod.S_IMODE(cfg.stat().st_mode) == 0o600 and not list(home.glob("*.tmp*"))
    assert json.loads(cfg.read_text()) == {"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tmp_path / "with space" / "a token"),
                                           "mark_ns": "", "store_commit_budget_per_hour": 0}
    # No debt: nothing is tried and nothing is written.
    rm, st = owed(tmp_path, "j")
    st.d = {"store_release_final_done": True, "campaign_home": True}
    rm.store_release = boom
    rm.store_release_owed(st, [])
    assert not (tmp_path / "j" / "runner.log").exists()
    # A campaign that is home with nothing recorded (a runner that died between the two) owes it.
    st.d = {"campaign_home": True}
    assert rm.store_release_debt(st) == "the campaign is home and its last release is not recorded as made"


def test_a_malformed_ledger_at_the_campaigns_end_is_named_and_costs_one_round(tmp_path):
    """End to end: a ledger on the campaign's branch that is not a ledger. The release at the campaign's end runs
    its round in the loop, deletes nothing, and is not run again after the stop: one lock and one unlock in all.
    The line names the ledger by branch and path."""
    work = tmp_path
    bad = work / "store" / "ckpt-N" / "pruned"
    bad.mkdir(parents=True)
    (bad / "g50.json").write_text("{not a ledger")
    p, calls = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), wrap=OWED_WRAP,
                         more_env={"OWED_WAIT_S": "30", "OWED_ESSENTIAL_S": "900"})
    log = log_of(work)
    assert p.returncode == 0, (p.stderr[-2000:], log[-3000:])
    lines = [ln[15:] for ln in log.splitlines()]
    said = [ln for ln in lines if "THE RELEASE AT THE CAMPAIGN'S END DID NOT RUN" in ln]
    assert len(said) == 1 and "ckpt-N:pruned/g50.json (not well formed)" in said[0] and "repair or remove it first" in said[0]
    assert len([ln for ln in lines if ln.startswith("store: release: 1 ledger(s) could not be read (ckpt-N:pruned/g50.json (not well formed))")]) == 1
    sent = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert sent.count("lock: release by home:pk-store") == 1 and sent.count("unlock: release") == 1
    state = state_of(work)
    assert state["campaign_home"] and "not well formed" in state["store_release_final_owed"] and state["store_releases"]["unread"] == 1
    # The second reviews of the arms' round: every later start of this folder made another round (a lock and an
    # unlock) and found the same file. The ledger is in state.json with its branch's head: a start now reads that
    # head, makes NO commit, writes the line and exits 0.
    head = hf_store.open_store({"repo": f"dir:{work / 'store'}"}).b.head("ckpt-N")
    assert state["store_release_final_malformed"] == {"ckpt-N": head} and state["store_release_final_rounds"] == 1
    for _ in range(2):
        before = log_of(work)
        p2, calls2 = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), fresh=False,
                               more_env={"OWED_WAIT_S": "30", "OWED_ESSENTIAL_S": "900"})
        log2 = log_of(work)[len(before):]
        assert p2.returncode == 0 and news(calls2) == 1, (p2.stderr[-1500:], log2[-2000:])
        said2 = [ln for ln in log2.splitlines() if "DID NOT RUN" in ln]
        assert len(said2) == 1 and "no new round is made until that head moves or store_release_final_malformed is removed" in said2[0]
        assert "ckpt-N:pruned/g50.json (not well formed)" in said2[0]
        assert [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()] == sent   # no commit
    # A person removes the bad file and the key: the next start makes the release.
    (bad / "g50.json").unlink()
    state = state_of(work)
    state.pop("store_release_final_malformed")
    (work / "state" / "state.json").write_text(json.dumps(state))
    p3, calls3 = run_store(work, {}, extra=store_args(work, "--store-first-start", "--store-release-storage"), fresh=False,
                           more_env={"OWED_WAIT_S": "30", "OWED_ESSENTIAL_S": "900"})
    assert p3.returncode == 0 and news(calls3) == 1
    state = state_of(work)
    assert state["store_release_final_done"] is True and not any(k in state for k in (
        "store_release_final_owed", "store_release_final_rounds", "store_release_final_malformed"))
    after = [json.loads(ln)["message"] for ln in (work / "store" / "commits.jsonl").read_text().splitlines()]
    assert after == [*sent, "lock: release by home:pk-store", "unlock: release"]


# ---------------------------------------------------------------- the arms' round, third pass (the second reviews, 12 Nov 2026)
def test_a_state_folder_has_one_runner_one_name_one_set_of_jobs_and_acts_only_on_its_own_session(tmp_path):
    """The second reviews of the arms' round. A finished state folder started again took ANY session listed under
    its --name for "the one an earlier runner left", removed its token and stopped it; the pid file was read and
    then written, so two starts could both pass; nothing tied the folder to a name or to the jobs it ran. Now:
    the folder's lock is taken atomically and held; state.json records the name and a fingerprint of the jobs, and
    a start under another name, or with other jobs on a folder that has jobs home, is refused; at start-up only a
    session at the endpoint this folder recorded is stopped and stripped, another one under the name is left
    alone and said, and a listing that does not answer is said too, with what was not done."""
    import fcntl
    work = tmp_path
    args = store_args(work, "--store-first-start")
    p, calls = run_store(work, {}, extra=args)
    assert p.returncode == 0, (p.stderr[-1500:], log_of(work)[-2000:])
    state = state_of(work)
    assert state["store_name"] == "pk-store" and len(state["store_jobs_sha256"]) == 64 and state["endpoint"] == "fake-endpoint-1"
    def again(extra=(), **kw):                           # one more start: (the process, the calls it made, its log)
        before, n_calls = log_of(work), len((work / "fake" / "calls.log").read_text().splitlines())
        q, c = run_store(work, {}, extra=[*args, *extra], fresh=False, **kw)
        return q, c.splitlines()[n_calls:], log_of(work)[len(before):]

    # (a) Another runner holds the folder: refused before any call.
    with open(work / "state" / "runner.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        q, new, log = again()
        assert q.returncode != 0 and "another runner holds this state folder (runner.lock); refusing" in q.stderr and new == [] and log == ""
    # (b) Another --name on this folder: refused before any call. (A finished folder under a live arm's name would
    # otherwise strip and stop that arm's session.)
    q, new, log = again(extra=["--name", "pk-other"])
    assert q.returncode != 0 and "belongs to the session name 'pk-store'; refusing --name 'pk-other'" in q.stderr and new == []
    # (c) Other jobs under the same names on a folder that has jobs home: refused.
    jobs_file = work / "repo" / "tools" / "colab_chat" / "eqjobs.json"
    kept = jobs_file.read_text()
    changed = json.loads(kept)
    changed["jobs"][1]["spec"] = {"train": {"steps": 999}}
    jobs_file.write_text(json.dumps(changed))
    q, new, log = again()
    assert q.returncode != 0 and "this state folder ran other jobs" in q.stderr and "Use a new --state-dir" in q.stderr and new == []
    jobs_file.write_text(kept)
    # (d) A session IS listed under the name, at another endpoint than the one this folder recorded: another
    # runner's. It is left alone: no exec (no token removed), no stop.
    fake = work / "fake" / "fake.json"
    d = json.loads(fake.read_text())
    fake.write_text(json.dumps({**d, "alive": True, "launches": d["launches"] + 1}))
    q, new, log = again()
    assert q.returncode == 0, (q.stderr[-1500:], log[-1500:])
    assert ("a session is listed under this name at fake-endpoint-2, which is NOT the endpoint this folder recorded "
            "(fake-endpoint-1): it is another runner's and is left alone (no token removed, no stop)") in log
    assert new and all(ln.startswith('["sessions"') for ln in new) and json.loads(fake.read_text())["alive"] is True
    # (e) The folder's own session (the recorded endpoint): its token is removed and it is stopped.
    state = state_of(work)
    state["endpoint"] = "fake-endpoint-2"
    (work / "state" / "state.json").write_text(json.dumps(state))
    q, new, log = again()
    assert q.returncode == 0 and "the session an earlier runner left is stopped" in log
    assert "store: courier ended and token removed from the VM (before the session stops)" in log
    assert new[-1].startswith('["stop"') and json.loads(fake.read_text())["alive"] is False
    # (f) The listing does not answer: nothing is done to any VM, and that is SAID; not a confirmed stop.
    col = work / "colab"
    real_col = col.read_text()
    col.write_text("#!/bin/sh\nif [ \"$1\" = sessions ]; then exit 1; fi\n" + real_col.split("\n", 1)[1])
    q, new, log = again()
    assert q.returncode == 4 and new == [], (q.returncode, new, log[-1500:])
    assert "THE LISTING DID NOT ANSWER" in log and "its token was NOT removed and it was NOT stopped" in log
    assert "colab stop -s pk-store" in log and log.rstrip().endswith("SESSION STOP NOT CONFIRMED (STOP_UNRESOLVED); runner exiting")
    col.write_text(real_col)


class FakeTime:
    """runner_multi's `time`, on a clock a test moves."""

    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def time(self):
        return self.t

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def strftime(self, fmt):
        return "11-12 00:00:00"


RAN = {"held": False, "objects": 0, "bytes": 0, "stored_bytes": 0, "orphan_bytes": 0, "orphan_objects": 0, "pending_bytes": 0,
       "ledgers": 0, "ledgers_done": 0, "unknown_sizes": 0, "ledgers_unread": 0, "gens": {}, "marks": {}, "pending": {}}


def test_a_timed_release_that_did_not_run_is_tried_again_soon_and_a_lost_measurement_is_sent_again(tmp_path):
    """The second reviews of the arms' round: a timed release that was held, failed or met a ledger it could not
    read set its next turn a full interval away, and the storage guard's count is only set back by a release that
    ran: two missed turns carry a healthy arm past its cap. Now a held one (nothing was sent) is tried at the next
    tick; one that failed or met an unread ledger once more, ten minutes later, and then at its regular turn; and
    a measurement that did not reach the VM is kept and sent again at each tick."""
    rm = load_runner(tmp_path / "home")
    clock = rm.time = FakeTime()
    rm.STORE = {"reclaim": True, "release_every_min": 120, "release_grace_s": 7200, "call_limit": 300, "mark_ns": "", "branch_suffix": ""}
    answers, asked, vm_ok, sent = [], [], [True], []

    class Store:
        def release_report(self, owner, grace=None, essential=False, takeover=None, now=None):
            asked.append((clock.t, essential, takeover, now))
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return a

    def exec_py(code, timeout=200):
        sent.append(clock.t)
        return (0, "MEASURE_OK") if vm_ok[0] else (1, "")

    rm.store_open = lambda jobs: Store()
    rm.store_call = lambda fn, limit=None: fn()
    rm.store_vm_read = lambda jobs: {"courier": {"pushed_bytes": 10, "boot_id": "b"}}
    rm.exec_py = exec_py
    st, jobs = St(), [{"name": "pt", "spec": {}}]
    t0 = clock.t
    rm.store_release(st, jobs)                           # the first tick: the timer starts, a full interval ahead
    assert asked == [] and st.d["store_release_next"] == t0 + 7200
    clock.t = t0 + 7200
    held = {"held": True, "objects": 0, "bytes": 0, "why": "another releaser holds the lock"}
    answers[:] = [held, held, dict(RAN)]
    for k in range(3):                                   # held twice: tried at each next tick, not two hours later
        rm.store_release(st, jobs)
        clock.t += 240
    assert [a[0] for a in asked] == [t0 + 7200, t0 + 7440, t0 + 7680] and st.d["store_releases"]["count"] == 1
    assert all(a[1] is False and a[2] == 900 and a[3] == a[0] for a in asked)        # timed; its call's limit; the runner's clock
    assert st.d["store_release_next"] == t0 + 7680 + 7200 and "store_measure_pending" not in st.d and len(sent) == 1
    log = (tmp_path / "home" / "runner.log").read_text()
    assert log.count("the lock was not taken (another releaser holds the lock); nothing was sent; tried again at the next tick") == 1
    # A round that ran and met a ledger it could not read, and one that failed: once more in ten minutes, then
    # at the regular turn. The budget's timed allowance is not what holds this back: the runner's own count is.
    del asked[:]
    clock.t = due = st.d["store_release_next"]
    unread = {**RAN, "ledgers_unread": 1, "unread": ["ckpt-N:pruned/g100.json (not read (ConnectionError))"], "unread_heads": {}}
    answers[:] = [unread, ConnectionError("reset"), dict(RAN)]
    rm.store_release(st, jobs)
    assert st.d["store_release_next"] == due + 600 == due + rm.STORE_RELEASE_RETRY_S and st.d["store_release_spent"] == 1
    clock.t = due + 240
    rm.store_release(st, jobs)                           # not yet
    clock.t = due + 600
    rm.store_release(st, jobs)                           # the one early retry: it fails
    assert [a[0] for a in asked] == [due, due + 600] and st.d["store_release_next"] == due + 7200   # then its regular turn
    assert "store_release_final_rounds" not in st.d      # a timed round is not the end's
    clock.t = due + 7200
    rm.store_release(st, jobs)
    assert st.d["store_releases"]["count"] == 2 and st.d["store_release_spent"] == 0
    # A measurement that does not reach the VM is kept, and sent again at each tick until it arrives.
    vm_ok[0] = False
    del sent[:]
    clock.t = due2 = st.d["store_release_next"]
    answers[:] = [dict(RAN)]
    rm.store_release(st, jobs)
    assert st.d["store_measure_pending"]["at_pushed"] == 10 and st.d["store_measure_pending"]["seq"] == 3 and len(sent) == 1
    for k in (1, 2):
        clock.t = due2 + 240 * k
        rm.store_release(st, jobs)                       # not due: no release, the measurement again
    assert len(sent) == 3 and "store_measure_pending" in st.d
    vm_ok[0] = True
    clock.t = due2 + 720
    rm.store_release(st, jobs)
    assert len(sent) == 4 and "store_measure_pending" not in st.d
    clock.t = due2 + 960
    rm.store_release(st, jobs)
    assert len(sent) == 4
    log = (tmp_path / "home" / "runner.log").read_text()
    assert log.count("the measurement did not reach the VM") == 1 and log.count("the measurement has reached the VM") == 1


def test_three_runners_whose_timers_start_together_all_release_and_no_arm_reaches_its_cap(tmp_path):
    """The second reviews' test: three arms on one repo (the stand-in hub, the real HfBackend), three runners with
    one budget each, their release timers started in the same tick. At every turn arm N's round holds the lock
    while S and C try, as a round that reads every tree and ledger does for minutes. Before the retry S and C were
    held at every turn and never measured: their guards' counts only grew. Now they run in the tick the lock is
    free. Over twelve hours of saves no arm's count reaches the cap the formula gives; and at the campaign's end
    the three end releases, started in the same tick against the same lock, are all made."""
    pytest.importorskip("huggingface_hub")
    sys.path.insert(0, str(ROOT / "tests"))
    import hub_standin
    import make_base2 as M
    import store_standin as S
    root = tmp_path / "hub"
    clock = FakeTime()
    tok = tmp_path / "tok"
    tok.write_text(TOKEN + "\n")
    tok.chmod(0o600)

    def backend():
        return hf_store.HfBackend("someone/base2-store", "dataset", TOKEN, api=hub_standin.DiskApi(str(root)))

    class Arm:
        def __init__(self, k, name):
            self.k, self.name, self.steps, self.pushed, self.base, self.peak, self.gen_bytes = k, name, [], 0, 0, 0, 0
            self.rm = load_runner(tmp_path / name)
            self.rm.time, self.rm.NAME = clock, f"pk-{name}"
            self.rm.STORE = {"reclaim": True, "release_every_min": 120, "release_grace_s": 7200, "call_limit": 300, "mark_ns": "",
                             "branch_suffix": "", "repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tok)}
            self.st, self.jobs = St(), [{"name": f"pt_{name}", "store_branch": f"ckpt-{name}", "spec": {}}]
            budget = hf_store.CommitBudget(12, str(tmp_path / name / "store_commits.json"), clock=clock.time)
            self.home = hf_store.Store(backend(), log=lambda m: None, delay=0, budget=budget, clock=clock.time)
            self.courier = hf_store.Store(backend(), log=lambda m: None, delay=0, clock=clock.time)
            self.rm.store_open = lambda jobs: self.home
            self.rm.store_call = lambda fn, limit=None: fn()
            self.rm.store_vm_read = lambda jobs: {"courier": {"pushed_bytes": self.pushed, "boot_id": "b"}}
            self.rm.exec_py = self.measure

        def measure(self, code, timeout=200):            # the VM takes the measurement, as the courier does
            m = self.st.d["store_measure_pending"]
            self.base = m["bytes"] - m["at_pushed"]
            return 0, "MEASURE_OK"

        def save(self, step):                            # the courier: one commit, pruning behind two verified
            d = tmp_path / f"gen_{self.name}"
            S.make_gen(str(d), step, seed=self.k)
            head = self.courier.head(f"ckpt-{self.name}")["commit"]
            man = self.courier.push_generation(f"ckpt-{self.name}", f"pt_{self.name}", step, str(d / "ckpt_gen" / f"g{step}"), None, head,
                                               prune={"keep": 2, "verified": self.steps[-2:]})
            self.gen_bytes = sum(f["bytes"] for f in man["files"].values())
            self.pushed += self.gen_bytes
            self.steps.append(step)

        def release(self, final=False):
            return self.rm.store_release(self.st, self.jobs, final=final)

        @property
        def counted(self):                               # the courier's count: the measurement, and what it pushed since
            return self.base + self.pushed

    n, s, c = arms = [Arm(k, name) for k, name in enumerate("NSC")]
    ending, inside = [False], []
    real = n.home.b.lfs_objects

    def while_n_holds_the_lock():                        # N's round is under way: the two others try now
        if not inside:
            inside.append(1)
            try:
                for other in (s, c):
                    other.release(final=ending[0])
            finally:
                inside.pop()
        return real()

    n.home.b.lfs_objects = while_n_holds_the_lock
    ticks, save_every = 180, 2                           # twelve hours of 240 s ticks; a save every 480 s
    for tick in range(ticks):
        clock.t += 240
        (root / "now.txt").parent.mkdir(exist_ok=True)
        (root / "now.txt").write_text(str(clock.t))
        for a in arms:
            if tick % save_every == 0:
                a.save(100 * (tick // save_every + 1))
            a.peak = max(a.peak, a.counted)
        for a in arms:
            a.release()
    # Every turn ran for every arm: five turns in twelve hours (the first is one interval after the start).
    assert [a.st.d["store_releases"]["count"] for a in arms] == [5, 5, 5]
    assert [a.st.d["store_releases"]["held"] for a in arms] == [0, 5, 5]              # S and C met N's lock at each turn
    assert all(a.st.d["store_releases"]["failed"] == 0 and "unread" not in a.st.d["store_releases"] for a in arms)
    assert sum(a.st.d["store_releases"]["objects"] for a in arms) > 100                # and they freed what the prunes removed
    # No arm's count reached the cap the formula gives for these settings (in generations of this size); without a
    # release that ran an arm would have counted all ninety of its saves.
    gens = M.gens_kept_max() + int((7200 + M.RELEASE_SKEW_S) / 480) + 1 + int((1 + M.ARM_MISSED_RELEASES) * 7200 / 480) + 2
    assert gens == 68 and len(n.steps) == 90
    for a in arms:
        assert a.peak < M.PUSH_CAP_HEADROOM * gens * a.gen_bytes, (a.name, a.peak / a.gen_bytes)
        assert a.peak / a.gen_bytes < 40                                               # about 5 kept, 16 removed, 15 pushed since
    # Home's commits: a lock and an unlock a turn, inside each runner's own budget of 12.
    assert all(a.home.budget.count() <= 2 for a in arms)
    # The campaign ends for all three in the same tick, N's round first: S and C meet its lock, owe their release,
    # and make it in the wait after their stop.
    clock.t += 240
    (root / "now.txt").write_text(str(clock.t))
    ending[0] = True
    assert n.release(final=True) == "done"
    assert [a.st.d.get("store_release_final_done") for a in arms] == [True, None, None]
    assert all("the lock was not taken (another releaser holds the lock)" in a.st.d["store_release_final_owed"] for a in (s, c))
    for a in (s, c):
        a.rm.store_release_owed(a.st, a.jobs)
    assert all(a.st.d.get("store_release_final_done") is True and "store_release_final_owed" not in a.st.d for a in arms)
    assert not any("DID NOT RUN" in (tmp_path / a.name / "runner.log").read_text() for a in arms)
    assert not (root / "branches" / "main" / "locks" / "release.json").exists()
    log = [json.loads(ln) for ln in (root / "commit_log.jsonl").read_text().splitlines()]
    locks = [x["message"] for x in log if x["branch"] == "main"]
    assert all(locks.count(f"lock: release by home:pk-{a.name}") == 6 for a in arms) and locks.count("unlock: release") == 18
    assert all(x["ok"] for x in log)
    # Every branch still holds its three newest generations, whole.
    for a in arms:
        assert a.courier.steps(f"ckpt-{a.name}") == a.steps[-3:] and a.courier.verify_by_download(f"ckpt-{a.name}", a.steps[-1])


def test_refusals_are_named_by_who_refused_and_what_comes_home_holds_no_signed_address(tmp_path):
    """The second reviews of the arms' round. Home's preflight and mark fetch did not go through the naming of an
    authentication failure, and any 401 or 403 was the token's. A session's logs came home with only the token's
    text replaced: boot.log can hold a signed address (the library's warning for a download it resumes). And a
    campaign opened again by a rung reset kept "the end's release is made", so its second end owed none."""
    import base64
    rm = load_runner(tmp_path / "home")
    rm.STORE = {"inputs": [], "branch_suffix": "", "call_limit": 300}
    st, jobs = St(), [{"name": "pt", "spec": {}, "snapshot_steps": [100]}]
    st.d = {"jobs": {}, "launches": 1}
    hub = Exception("401 Client Error. Repository Not Found for url: https://huggingface.co/api/datasets/someone/base2-store/tree/main")
    rm.store_open = lambda jobs: (_ for _ in ()).throw(hub)
    assert rm.store_preflight(st, jobs) is False
    log = (tmp_path / "home" / "runner.log").read_text()
    assert "preflight: STORE AUTHENTICATION FAILURE at home: the hub refused the token (HTTP 401)" in log
    # A mark's download refused by the CDN after the hub's redirect: said as that; nothing about the token.
    rm2 = load_runner(tmp_path / "home2")
    rm2.STORE = {"inputs": [], "branch_suffix": "", "call_limit": 300}
    st2 = St()
    st2.d = {"jobs": {}, "launches": 1}
    rm2.store_open = lambda jobs: object()
    cdn = hf_store.StoreError("HfHubHTTPError: 403 Forbidden: Cannot access content at: https://cdn-lfs.example/o/1?[query removed] "
                              "[HTTP 403] [from the object store]")
    rm2.store_call = lambda fn, limit=None: (_ for _ in ()).throw(cdn)
    rm2.store_fetch_mark(st2, jobs)
    log2 = (tmp_path / "home2" / "runner.log").read_text()
    assert "marks: the refusal (HTTP 403) is the object store's, not the hub's: the token is not the cause" in log2
    assert "AUTHENTICATION FAILURE" not in log2 and "replace the token file" not in log2 and "store_auth_failure" not in st2.d
    # The session's files: no signed address in a log that comes home.
    boot = (b"Downloading model.safetensors\nError while downloading from https://cdn-lfs.example/repos/ab/cd/0123?"
            b"X-Amz-Signature=SECRETSIG&X-Amz-Date=1: read timed out. Trying to resume download...\nboot done\n")
    status = json.dumps({"jobs": {}}).encode()
    files = {"boot.log": [len(boot), base64.b64encode(boot).decode()], "courier_status.json": [len(status), base64.b64encode(status).decode()]}
    rm2.exec_py = lambda code, timeout=200: (0, "BRINGHOME " + json.dumps(files))
    assert rm2.store_bring_home(st2, "a test") is True
    home = tmp_path / "home2" / "sessions" / "launch01"
    text = (home / "boot.log").read_text()
    assert "SECRETSIG" not in text and "https://cdn-lfs.example/repos/ab/cd/0123?[query removed]" in text and text.endswith("boot done\n")
    assert (home / "courier_status.json").read_bytes() == status
    index = json.loads((home / "index.json").read_text())["files"]
    assert index["boot.log"]["whole"] is True and index["boot.log"]["bytes_on_vm"] == len(boot)
    # A rung reset opens the campaign again: its end owes its release again, with a whole allowance of rounds.
    rm3 = load_runner(tmp_path / "home3")
    rm3.ART = str(tmp_path / "art")
    jobs3 = [{"name": "pt", "spec": {}}, {"name": "rung_100", "rung_of": {"job": "pt", "step": 100}, "cmd": ["x"]}]
    snap = tmp_path / "art" / "pt" / "rung_snapshots" / "rung_100"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"w")
    (snap / "state.txt").write_text("100\n")
    (tmp_path / "art" / "rung_100").mkdir()
    (tmp_path / "art" / "rung_100" / "rung_100_train.log").write_text("rung lost: no snapshot\n")
    (tmp_path / "home3" / "rung_reset.json").write_text(json.dumps({"rungs": ["rung_100"]}))
    st3 = St()
    st3.d = {"jobs": {"pt": {"home": True, "status": "FINISHED"}, "rung_100": {"home": True, "status": "FAILED"}},
             "campaign_home": True, "store_release_final_done": True, "store_release_final_rounds": 2,
             "store_release_final_malformed": {"ckpt-N": "7"}, "store_release_final_owed": "x"}
    rm3.apply_rung_resets(st3, jobs3)
    assert st3.d["jobs"]["rung_100"]["home"] is False
    assert not any(k in st3.d for k in ("campaign_home", "store_release_final_done", "store_release_final_rounds",
                                        "store_release_final_malformed", "store_release_final_owed"))
