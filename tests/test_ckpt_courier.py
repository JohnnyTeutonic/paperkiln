"""tools/colab_chat/ckpt_courier.py with a store on local disk and a stand-in trainer (tests/store_standin.py).

The courier: a save and a prune during staging never tear the pushed generation; a step behind the store's is
refused; a mark snapshot is pushed once, survives the rung job dropping its weights, and is never removed; a failed
push leaves training alone, is retried, and then sends the newest generation; the store keeps two generations.
The restore: jobs_vm.py's own restore() takes the optimiser file placed for it (BASE2_PLAN.md work item 5), a
damaged newest generation falls back to the one before, and an unreadable store stops the boot.
The rehearsal runs the real boot_base2.sh, jobs_vm.py and courier daemon over two "VM" folders: the first is killed
between saves, the store's newest generation is then damaged, and the second must resume from the one before it and
end on the bytes of an uninterrupted run. With MTSTUDIO_BIN set, the same resume is run with the real trainer on CPU.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(ROOT / "tests"))
import ckpt_courier as cc  # noqa: E402
import hf_store  # noqa: E402
import store_standin as S  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the courier runs on the VM (Linux)")
TOKEN = "hf_FAKEtokenFAKEtokenFAKEtoken0123"
JOBS = {"pt": {"name": "pt", "snapshot_steps": [200], "spec": {}},
        "rung_200": {"name": "rung_200", "rung_of": {"job": "pt", "step": 200}, "cmd": ["r"]}}
BR = {"pt": "ckpt-N"}


class Flaky(hf_store.DirBackend):
    """A store whose commits fail on demand, with a token to keep out of every log."""
    token = TOKEN

    def __init__(self, root):
        super().__init__(root)
        self.fail_commits, self.fail_reads, self.commits = 0, False, 0

    def commit(self, branch, adds, deletes, message):
        if self.fail_commits:
            self.fail_commits -= 1
            raise ConnectionError(f"503 from the hub (Authorization: Bearer {TOKEN})")
        self.commits += 1
        super().commit(branch, adds, deletes, message)

    def tree(self, branch):
        if self.fail_reads:
            raise ConnectionError("hub unreachable")
        return super().tree(branch)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def setup(tmp_path, **kw):
    chat = tmp_path / "chat"
    (chat / "pt").mkdir(parents=True)
    backend = Flaky(str(tmp_path / "repo"))
    logs, clock = [], Clock()
    store = hf_store.Store(backend, log=logs.append)
    c = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, **kw)
    return chat, backend, store, c, logs, clock


def fresh(tmp_path):
    return hf_store.Store(hf_store.DirBackend(str(tmp_path / "repo")), log=lambda m: None)


def events(chat, upto):
    with open(chat / "pt" / "events.jsonl", "a") as f:
        for s in upto:
            f.write(json.dumps({"event": "step", "step": s}) + "\n")
            if s % 100 == 0:
                f.write(json.dumps({"event": "checkpoint", "step": s}) + "\n")


def test_pushes_each_new_generation_once_and_keeps_two(tmp_path):
    chat, backend, store, c, logs, _ = setup(tmp_path)
    c.tick()  # nothing saved yet
    assert backend.commits == 0
    for step in (100, 200, 300, 400):
        S.make_gen(str(chat / "pt"), step)
        events(chat, range(step - 99, step + 3))  # the trainer is two steps past the save when the courier looks
        c.tick()
        c.tick()  # nothing new: no second push
    assert store.steps("ckpt-N") == [300, 400]
    pushes = [ln for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines() if '"generation: ' in ln]
    assert len(pushes) == 4
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["pushed_step"] == 400 and st["pushes"] == 4 and st["failures"] == 0 and st["marks"] == [200]
    # What was pushed resumes: files whole, and the events record cut at the checkpoint's own event.
    step, rejected = fresh(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"))
    assert (step, rejected) == (400, [])
    ev = [json.loads(ln) for ln in (tmp_path / "r" / "g400" / "events.jsonl").read_text().splitlines()]
    assert ev[-1] == {"event": "checkpoint", "step": 400} and max(e["step"] for e in ev) == 400
    assert not os.listdir(chat / "pt" / "courier_stage")  # nothing left staged
    assert not any(TOKEN in ln for ln in logs)


def test_a_save_during_staging_never_tears_the_pushed_generation(tmp_path, monkeypatch):
    """While g100 is being staged the trainer commits g200 and g300, which removes g100 from disk. Either the
    staged files are whole g100 (hard links outlive the removal) or the tick is skipped; never a mix."""
    chat, backend, store, c, logs, _ = setup(tmp_path)
    out = str(chat / "pt")
    S.make_gen(out, 100)
    real, n = cc.link_or_copy, []

    def racing(src, dst):
        n.append(src)
        if len(n) == 2:  # after the first file is linked: two saves, and the trainer prunes g100
            S.make_gen(out, 200)
            S.make_gen(out, 300)
        real(src, dst)

    monkeypatch.setattr(cc, "link_or_copy", racing)
    c.tick()
    assert not os.path.isdir(os.path.join(out, "ckpt_gen", "g100"))
    assert store.steps("ckpt-N") == [] and any("changed while it was staged" in ln for ln in logs)
    c.tick()  # the next pass takes the live generation, whole
    assert store.steps("ckpt-N") == [300]
    assert fresh(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r")) == (300, [])
    # A copy instead of links (another filesystem), raced the same way, gives the same guarantee.
    chat2, backend2, store2, c2, logs2, _ = setup(tmp_path / "b")
    out2 = str(chat2 / "pt")
    S.make_gen(out2, 100)
    n.clear()

    def racing_copy(src, dst):
        n.append(src)
        if len(n) == 3:
            S.make_gen(out2, 200)
            S.make_gen(out2, 300)
        import shutil
        shutil.copyfile(src, dst)

    monkeypatch.setattr(cc, "link_or_copy", racing_copy)
    c2.tick()
    c2.tick()
    for s in store2.steps("ckpt-N"):
        man = store2.read_manifest("ckpt-N", f"gens/g{s}")
        d = tmp_path / "b" / f"chk{s}"
        store2.pull_set("ckpt-N", f"gens/g{s}", str(d))
        hf_store.check_generation(str(d), man["step"])
    assert store2.steps("ckpt-N")[-1] == 300


def test_a_step_behind_the_store_is_refused(tmp_path):
    chat, backend, store, c, logs, _ = setup(tmp_path)
    for step in (100, 200, 300):
        S.make_gen(str(chat / "pt"), step)
        c.tick()
    # A VM that holds an older checkpoint than the store (a stale VM, or a run restarted in place): nothing pushed.
    chat2 = tmp_path / "vm2" / "chat"
    (chat2 / "pt").mkdir(parents=True)
    S.make_gen(str(chat2 / "pt"), 100, seed=7)  # other bytes: it must not replace anything
    c2 = cc.Courier(store, str(chat2), JOBS, BR, log=logs.append)
    before = backend.commits
    c2.tick()
    c2.tick()
    assert backend.commits == before and store.steps("ckpt-N") == [200, 300]
    st = json.loads((chat2 / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["refused_step"] == 100 and st["pushed_step"] == 300
    assert sum("is behind the store's g300; refused" in ln for ln in logs) == 1
    # The same step as the store's is not pushed again either; the next one is.
    S.make_gen(str(chat2 / "pt"), 300)
    c2.tick()
    assert backend.commits == before
    S.make_gen(str(chat2 / "pt"), 400)
    c2.tick()
    assert store.steps("ckpt-N") == [300, 400]


def test_a_mark_is_pushed_once_and_never_removed(tmp_path):
    chat, backend, store, c, logs, clock = setup(tmp_path)
    out = chat / "pt"
    m100 = S.make_gen(str(out), 100)[0]
    # jobs_vm.take_snapshot's copy at a step that is not in the job's own list still counts as a mark.
    (out / "rung_100").mkdir()
    (out / "rung_100" / "model.safetensors").write_bytes(m100)
    (out / "rung_100" / "state.txt").write_text("100\n{}\n")
    backend.fail_commits = 99  # the store is down
    c.tick()
    assert backend.commits == 0
    # The rung job scores the snapshot and drops its weights before any push got through.
    (out / "rung_100" / "model.safetensors").unlink()
    m200 = S.make_gen(str(out), 200)[0]  # step 200 is in snapshot_steps: its mark comes from the generation itself
    backend.fail_commits = 0
    clock.t += 3600
    c.tick()
    c.tick()
    assert store.mark_steps("pt") == [100, 200]
    got = tmp_path / "got"
    store.pull_mark("pt", 100, str(got / "100"))
    store.pull_mark("pt", 200, str(got / "200"))
    assert (got / "100" / "model.safetensors").read_bytes() == m100
    assert (got / "200" / "model.safetensors").read_bytes() == m200
    marks = lambda: [ln for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines() if '"mark: ' in ln]  # noqa: E731
    assert len(marks()) == 2
    # jobs_vm's own snapshot of step 200 appears later; later ticks and a restarted courier push nothing again.
    (out / "rung_200").mkdir()
    (out / "rung_200" / "model.safetensors").write_bytes(m200)
    (out / "rung_200" / "state.txt").write_text("200\n{}\n")
    c.tick()
    c2 = cc.Courier(store, str(chat), JOBS, BR, log=logs.append)
    c2.tick()
    assert len(marks()) == 2
    # Many generations later the marks are still there: the prune works on the ckpt branch only.
    for step in (300, 400, 500, 600):
        S.make_gen(str(out), step)
        c2.tick()
    assert store.steps("ckpt-N") == [500, 600] and store.mark_steps("pt") == [100, 200]
    assert not any('"branch": "main"' in ln and '"deletes": ["' in ln
                   for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines())


def test_a_failed_push_is_retried_and_then_sends_the_newest(tmp_path):
    chat, backend, store, c, logs, clock = setup(tmp_path)
    out = str(chat / "pt")
    S.make_gen(out, 100)
    backend.fail_commits = 3
    c.tick()
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["failures"] == 1 and st["pushed_step"] is None and st["live_step"] == 100
    assert "[token]" in st["last_error"] and TOKEN not in json.dumps(st) and not any(TOKEN in ln for ln in logs)
    assert any("training goes on" in ln for ln in logs)
    c.tick()  # inside the wait: no call to the store at all
    assert backend.fail_commits == 2
    S.make_gen(out, 200)  # training went on meanwhile
    S.make_gen(out, 300)
    clock.t += 31
    c.tick()  # second failure, a longer wait
    assert backend.fail_commits == 1
    clock.t += 31
    c.tick()
    assert backend.fail_commits == 1
    clock.t += 30
    c.tick()  # third failure
    clock.t += 121
    c.tick()  # the store is back: the newest generation goes, not the one that first failed
    assert store.steps("ckpt-N") == [300]
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["pushed_step"] == 300 and st["failures"] == 0 and st["last_error"] is None
    # A store that cannot even be read: no push is tried, nothing raised, and the first contact is retried.
    chat2, backend2, store2, c2, logs2, clock2 = setup(tmp_path / "b")
    S.make_gen(str(chat2 / "pt"), 100)
    backend2.fail_reads = True
    c2.tick()
    assert backend2.commits == 0 and any("store not read" in ln for ln in logs2)
    backend2.fail_reads = False
    clock2.t += 31
    c2.tick()
    assert store2.steps("ckpt-N") == [100]


def test_min_interval_spaces_pushes_but_not_the_last(tmp_path):
    chat, backend, store, c, logs, clock = setup(tmp_path, min_interval=600)
    out = str(chat / "pt")
    for step in (100, 200, 300):
        S.make_gen(out, step)
        clock.t += 100
        c.tick()
    assert store.steps("ckpt-N") == [100]  # 200 and 300 fell inside the interval
    (chat / "STATUS_pt").write_text("FINISHED\n")
    assert not c.idle()
    c.tick()
    assert store.steps("ckpt-N") == [100, 300] and c.idle()  # a job that ended: its last generation goes at once


def test_bounded_ends_a_hung_call_and_redacts(tmp_path):
    assert cc.bounded(lambda: {"a": 1}, 20) == {"a": 1}
    t0 = time.time()
    with pytest.raises(hf_store.StoreError, match="ended"):
        cc.bounded(lambda: time.sleep(60), 0.5)
    assert time.time() - t0 < 10

    def boom():
        raise ConnectionError(f"Bearer {TOKEN}")

    with pytest.raises(hf_store.StoreError) as e:
        cc.bounded(boom, 20, TOKEN)
    assert "[token]" in str(e.value) and TOKEN not in str(e.value)
    # The daemon's default path: every store call in a child with a limit.
    chat, backend, store, c, logs, _ = setup(tmp_path, push_timeout=60)
    S.make_gen(str(chat / "pt"), 100)
    c.tick()
    assert store.steps("ckpt-N") == [100]


# ---------------------------------------------------------------- restore
def jobs_vm_at(chat):
    spec = importlib.util.spec_from_file_location("jobs_vm_under_test", CC / "jobs_vm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.C = str(chat)
    return m


def pushed_store(tmp_path, steps):
    chat, backend, store, c, logs, _ = setup(tmp_path / "vm1")
    for step in steps:
        S.make_gen(str(chat / "pt"), step)
        events(chat, range(step - 99, step + 1))
        c.tick()
    return backend, store


def test_restore_places_a_whole_generation_and_jobs_vm_takes_the_optimiser(tmp_path, capsys):
    """Work item 5: jobs_vm.py's restore(), unchanged, keeps the optimiser file and the AdamW step count."""
    backend, store = pushed_store(tmp_path, [100, 200, 300])
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    logs = []
    rec = cc.restore(store, str(chat), JOBS, BR, log=logs.append)["pt"]
    assert rec["store_step"] == 300 and rec["placed"] and rec["rejected"] == []
    up = chat / "upload" / "pt"
    assert sorted(os.listdir(up)) == ["READY", "events.jsonl", "model.safetensors.part00", "optim.safetensors.part00", "state.txt"]
    m, o = S.state_at(300)
    assert (up / "model.safetensors.part00").read_bytes() == m and (up / "optim.safetensors.part00").read_bytes() == o
    # The mark the unfinished rung job needs came with it.
    assert (chat / "pt" / "rung_200" / "model.safetensors").read_bytes() == S.state_at(200)[0]
    assert (chat / "pt" / "rung_200" / "state.txt").read_text().startswith("200\n")
    jv = jobs_vm_at(chat)
    assert not jv.awaiting_upload("pt")
    jv.restore("pt")
    out = chat / "pt"
    assert (out / "model.safetensors").read_bytes() == m and (out / "optim.safetensors").read_bytes() == o
    lines = (out / "state.txt").read_text().split("\n")
    assert lines[0] == "300" and json.loads(lines[1])["adamw_t"] == [300]  # not popped: this is a warm resume
    assert "cold resume" not in capsys.readouterr().out
    assert (out / "events.jsonl").read_text().splitlines()[-1] == json.dumps({"event": "checkpoint", "step": 300})
    # The boot runs again on the same VM (the manager died): this VM holds step 300 already, nothing is placed.
    S.make_gen(str(out), 400)
    os.remove(up / "READY")
    rec = cc.restore(store, str(chat), JOBS, BR, log=logs.append)["pt"]
    assert rec["placed"] is False and rec["local_step"] == 400 and not (up / "READY").exists()
    # A job that has ended is left alone.
    (chat / "STATUS_pt").write_text("FINISHED\n")
    assert cc.restore(store, str(chat), JOBS, BR, log=logs.append) == {}


def test_restore_falls_back_when_the_newest_generation_is_damaged(tmp_path):
    backend, store = pushed_store(tmp_path, [100, 200, 300])
    bad = tmp_path / "vm1" / "repo" / "ckpt-N" / "gens" / "g300" / "model.safetensors"
    bad.write_bytes(bad.read_bytes()[:-1] + b"?")
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    logs = []
    rec = cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=logs.append)["pt"]
    assert rec["store_step"] == 200 and [s for s, _ in rec["rejected"]] == [300] and rec["placed"]
    assert (chat / "upload" / "pt" / "optim.safetensors.part00").read_bytes() == S.state_at(200)[1]
    assert (chat / "upload" / "pt" / "state.txt").read_text().startswith("200\n")
    assert json.loads((chat / "store_restore" / "pt.json").read_text())["store_step"] == 200
    # The courier of the new VM pushes over the damaged generation when training reaches that step again.
    S.make_gen(str(chat / "pt"), 300)
    st2 = fresh(tmp_path / "vm1")
    c = cc.Courier(st2, str(chat), JOBS, BR, log=logs.append)
    c.tick()
    assert fresh(tmp_path / "vm1").pull_generation("ckpt-N", str(tmp_path / "r")) == (300, [])


def test_restore_never_starts_from_zero_on_a_store_it_cannot_trust(tmp_path):
    backend, store = pushed_store(tmp_path, [100, 200])
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    backend.fail_reads = True
    with pytest.raises(ConnectionError):
        cc.restore(store, str(chat), JOBS, BR, log=lambda m: None, tries=2, delay=0)
    assert not (chat / "upload" / "pt" / "READY").exists()
    backend.fail_reads = False
    for s in (100, 200):
        p = tmp_path / "vm1" / "repo" / "ckpt-N" / "gens" / f"g{s}" / "optim.safetensors"
        p.write_bytes(b"damaged")
    with pytest.raises(hf_store.StoreError, match="no generation verifies"):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=lambda m: None, tries=2, delay=0)
    assert not (chat / "upload" / "pt" / "READY").exists()
    # An empty branch is a first launch: nothing placed, no error.
    rec = cc.restore(fresh(tmp_path / "empty"), str(chat), JOBS, BR, log=lambda m: None)["pt"]
    assert rec["store_step"] is None and rec["placed"] is False
    # The command the boot runs exits 1 on the same failure, with one line and no traceback.
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'vm1' / 'repo'}", "branches": BR}))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": list(JOBS.values())}))
    r = subprocess.run([sys.executable, str(CC / "ckpt_courier.py"), "restore", "--config", str(cfg), "--jobs", str(jobs),
                        "--chat", str(chat), "--retry-delay", "0"], capture_output=True, text=True)
    assert r.returncode == 1 and "no generation verifies" in r.stdout and "Traceback" not in r.stdout + r.stderr


# ---------------------------------------------------------------- the rehearsal: two VM folders, the real boot
STEPS, EVERY, MARK = 200, 10, 20
RUNG = """import hashlib, json, os, sys
snap, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
w = open(os.path.join(snap, "model.safetensors"), "rb").read()
json.dump({"weights_sha256": hashlib.sha256(w).hexdigest()}, open(os.path.join(out, "rung.json"), "w"))
os.remove(os.path.join(snap, "model.safetensors"))
"""


def make_vm(tmp_path, name, repo):
    root = tmp_path / name
    chat, pk = root / "chat", root / "pk"
    (chat / "upload").mkdir(parents=True)
    (pk / "build_cuda").mkdir(parents=True)
    os.symlink(ROOT / "tools", pk / "tools")
    S.write_trainer(str(pk / "build_cuda" / "mtstudio"), sys.executable)
    (root / "rung.py").write_text(RUNG)
    (chat / "store.json").write_text(json.dumps({"repo": f"dir:{repo}", "branches": {"pt": "ckpt-N"}}))
    tok = chat / "upload" / "store_token"
    tok.write_text(TOKEN)
    jobs = {"max_parallel": 1, "store_inputs": [{"prefix": "seed/vocab", "to": "chat/upload"}], "jobs": [
        {"name": "pt", "model_name": "m", "snapshot_steps": [MARK], "env": {"MICROTORCH_CHECKPOINT_TXN": "1"},
         "spec": {"train": {"steps": STEPS, "checkpoint_every": EVERY, "seed": 0}}},
        {"name": "rung_mark", "after": "pt", "rung_of": {"job": "pt", "step": MARK}, "light": True,
         "cmd": [sys.executable, str(root / "rung.py"), f"{chat}/pt/rung_{MARK}", "{out}"], "outputs": ["rung.json"]}]}
    (root / "jobs.json").write_text(json.dumps(jobs))
    return root


def boot(root, sleep):
    env = {**os.environ, "PK_BOOT_ROOT": str(root), "PK_STORE_TOKEN_DIR": str(root / "shm"), "JOBS_TICK": "0.2",
           "PK_COURIER_ARGS": "--tick 0.1 --push-timeout 0", "PK_RESTORE_ARGS": "--retry-delay 0",
           "STANDIN_STEP_SLEEP": str(sleep), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(["bash", str(CC / "boot_base2.sh"), "0000000", str(root / "jobs.json")], env=env,
                          capture_output=True, text=True, timeout=120)


def vm_pids(root):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                if str(root).encode() in open(f"/proc/{d}/cmdline", "rb").read():
                    out.append(int(d))
            except OSError:
                pass
    return out


def kill_vm(root):
    for _ in range(3):
        for pid in vm_pids(root):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.2)


def wait(cond, what, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def test_rehearsal_two_vms_through_the_real_boot(tmp_path):
    repo = tmp_path / "repo"
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "vocab.gguf").write_bytes(b"vocab bytes")
    fresh(tmp_path).push_set("main", "seed/vocab", str(seed))
    store = fresh(tmp_path)
    vm1, vm2 = make_vm(tmp_path, "vm1", repo), make_vm(tmp_path, "vm2", repo)
    try:
        # Session one: the boot pulls the inputs, finds no generation, starts the courier and the job.
        r = boot(vm1, 0.03)
        assert r.returncode == 0, read(vm1 / "chat" / "boot.log")
        assert (vm1 / "chat" / "upload" / "vocab.gguf").read_bytes() == b"vocab bytes"
        assert "holds no generation" in read(vm1 / "chat" / "boot.log")
        wait(lambda: (store.steps("ckpt-N") or [0])[-1] >= 60, "session one's pushes")
        time.sleep(0.1)  # somewhere between two saves
        kill_vm(vm1)
        assert read(vm1 / "chat" / "STATUS_pt").strip() == "RUNNING"  # it never finished
        steps = store.steps("ckpt-N")
        assert len(steps) == 2 and store.mark_steps("pt") == [MARK]
        prev, newest = steps
        lost_at = int(read(vm1 / "chat" / "pt" / "state.txt").split("\n")[0])
        # The newest generation is damaged in the store before the next session.
        bad = repo / "ckpt-N" / "gens" / f"g{newest}" / "optim.safetensors"
        bad.write_bytes(bad.read_bytes()[:-1] + b"?")
        # Session two, a new VM folder with an empty disk.
        r = boot(vm2, 0.005)
        assert r.returncode == 0, read(vm2 / "chat" / "boot.log")
        chat = vm2 / "chat"
        rec = json.loads(read(chat / "store_restore" / "pt.json"))
        assert rec["store_step"] == prev and [s for s, _ in rec["rejected"]] == [newest] and rec["placed"]
        wait(lambda: read(chat / "STATUS").strip() == "FINISHED", "session two's end")
        assert read(chat / "STATUS_pt").strip() == "FINISHED" and read(chat / "STATUS_rung_mark").strip() == "FINISHED"
        ev = [json.loads(ln) for ln in read(chat / "pt" / "events.jsonl").splitlines()]
        resumes = [e for e in ev if e["event"] == "resume"]
        assert resumes == [{"event": "resume", "step": prev, "optimizer": "restored", "from": "flat"}]
        # One record of the run: every step once, in order, the join at the restored step.
        assert [e["step"] for e in ev if e["event"] == "step"] == list(range(1, STEPS + 1))
        # The end state equals an uninterrupted run's, weights and optimiser (the stand-in's bytes chain both).
        m, o = S.state_at(STEPS)
        assert (chat / "pt" / "m.safetensors").read_bytes() == m
        assert (chat / "pt" / "optim.safetensors").read_bytes() == o
        # The rung on the new VM scored the mark pulled from the store: the weights of step MARK, from session one.
        assert json.loads(read(chat / "rung_mark" / "rung.json"))["weights_sha256"] == S.sha(S.state_at(MARK)[0])
        wait(lambda: "courier done" in read(chat / "courier.log"), "the courier's last push")
        end = store.steps("ckpt-N")  # two generations, the last one the run's end, and the mark from session one
        assert len(end) == 2 and end[-1] == STEPS and store.mark_steps("pt") == [MARK]
        assert fresh(tmp_path).pull_generation("ckpt-N", str(tmp_path / "final")) == (STEPS, [])
        cs = json.loads(read(chat / "courier_status.json"))["jobs"]["pt"]
        assert cs["pushed_step"] == STEPS and cs["failures"] == 0
        # The token: adopted into the tmpfs stand-in with mode 600, gone from upload/, in no log, removed at stop.
        for vm in (vm1, vm2):
            tf = json.loads(read(vm / "chat" / "store.json"))["token_file"]
            assert tf == str(vm / "shm" / "token") and os.stat(tf).st_mode & 0o777 == 0o600
            assert not (vm / "chat" / "upload" / "store_token").exists()
            for f in ("boot.log", "courier.log", "jobs.log", "courier_status.json", "pt_train.log"):
                assert TOKEN not in read(vm / "chat" / f)
            g = subprocess.run([sys.executable, str(CC / "hf_store.py"), "forget-token", "--config",
                                str(vm / "chat" / "store.json")], capture_output=True, text=True)
            assert g.stdout.strip() == "TOKEN_GONE" and not os.path.exists(tf)
        assert TOKEN not in read(repo / "commits.jsonl")
        print(f"rehearsal: lost at step {lost_at}; store held {steps}; g{newest} damaged; resumed at {prev}; "
              f"finished at {STEPS}; store now {store.steps('ckpt-N')}, marks {store.mark_steps('pt')}")
    finally:
        kill_vm(vm1)
        kill_vm(vm2)


def test_boot_fails_when_an_input_does_not_verify(tmp_path):
    repo = tmp_path / "repo"
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "vocab.gguf").write_bytes(b"vocab bytes")
    fresh(tmp_path).push_set("main", "seed/vocab", str(seed))
    (repo / "main" / "seed" / "vocab" / "vocab.gguf").write_bytes(b"vocab bytez")
    vm = make_vm(tmp_path, "vm", repo)
    try:
        r = boot(vm, 0)
        assert r.returncode != 0
        assert read(vm / "chat" / "STATUS").strip() == "BOOT_FAILED"
        assert "sha256 or size differs" in read(vm / "chat" / "boot.log")
        assert not (vm / "chat" / "jobs_vm.pid").exists() and not (vm / "chat" / "STATUS_pt").exists()
    finally:
        kill_vm(vm)


# ---------------------------------------------------------------- the real trainer on CPU
def test_resume_through_the_store_equals_a_straight_run_real_trainer(tmp_path):
    """40 steps straight, and 20 steps, a pull from the store on a new folder, then 20 more: the same weights,
    optimiser and exported model, byte for byte, at full precision, and the resume event reads optimizer: restored."""
    import test_checkpoint_txn as T
    if T.BIN is None:
        pytest.skip("set MTSTUDIO_BIN to a built mtstudio (CPU)")
    cpu = {"CUDA_VISIBLE_DEVICES": ""}
    T.setup(tmp_path, state_tokens=True)
    control = tmp_path / "control"
    r = T.run(T.spec(tmp_path, control, steps=40, every=10, state_tokens=True), cpu)
    assert r.returncode == 0, r.stderr[-800:]
    # Session one: ended as its step-20 generation is committed (the process dies at the last publication stage).
    chat1 = tmp_path / "vm1" / "chat"
    chat1.mkdir(parents=True)
    r = T.run(T.spec(tmp_path, chat1 / "pt", steps=40, every=10, state_tokens=True), {**cpu, "MICROTORCH_CKPT_CRASH": "prune:20"})
    assert r.returncode == 137, r.stderr[-800:]
    store = fresh(tmp_path)
    jobs = {"pt": {"name": "pt", "spec": {}}}
    c = cc.Courier(store, str(chat1), jobs, BR, log=lambda m: None)
    c.tick()
    assert store.steps("ckpt-N") == [20]
    # Session two: a new folder; the boot's restore, jobs_vm.py's restore, then the trainer again.
    chat2 = tmp_path / "vm2" / "chat"
    chat2.mkdir(parents=True)
    rec = cc.restore(fresh(tmp_path), str(chat2), jobs, BR, log=lambda m: None)["pt"]
    assert rec["store_step"] == 20 and rec["placed"]
    jobs_vm_at(chat2).restore("pt")
    r = T.run(T.spec(tmp_path, chat2 / "pt", steps=40, every=10, state_tokens=True), cpu)
    assert r.returncode == 0, r.stderr[-800:]
    res = T.events(chat2 / "pt", "resume")
    assert len(res) == 1 and res[0]["step"] == 20 and res[0]["optimizer"] == "restored"
    assert T.finals(chat2 / "pt") == T.finals(control)
    a, b = T.steps_of(control), T.steps_of(chat2 / "pt")
    assert {s: b[s] for s in range(21, 41)} == {s: a[s] for s in range(21, 41)}
    # The courier of session two pushes what the resumed trainer saves, and the store's newest resumes again.
    c2 = cc.Courier(fresh(tmp_path), str(chat2), jobs, BR, log=lambda m: None)
    c2.tick()
    assert fresh(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r")) == (40, [])
