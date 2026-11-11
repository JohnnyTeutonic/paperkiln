"""tools/colab_chat/ckpt_courier.py with a store on local disk and a stand-in trainer (tests/store_standin.py).

The courier: a save and a prune during staging never tear the pushed generation; a step behind the store's is
refused and a step in the store is never written again; two couriers on one branch: the one of the older epoch, or
the one that finds another boot's newer generation, stops for good; a mark snapshot is staged before any network
call, pushed before its generation, once, survives the rung job dropping its weights, and is never removed; the
prune runs only behind two generations this VM verified and never at or above an unpushed mark; a failed push
leaves training alone, is retried, and then sends the newest generation.
The restore: a verified generation is installed as the job's live trainer generation by one rename; a kill during
the install leaves nothing a trainer or a later boot would trust; a transport failure is not corruption; missing
history, another identity and an unadmitted first start are refused.
The rehearsal runs the real boot_base2.sh, jobs_vm.py and courier daemon over two "VM" folders: the first is killed
between saves, the store's newest generation is then damaged, and the second must resume from the one before it and
end on the bytes of an uninterrupted run. With MTSTUDIO_BIN set, the same resume is run with the real trainer on CPU.
"""
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
JOBS = {"pt": {"name": "pt", "snapshot_steps": [200], "spec": {"train": {"steps": 900}}},
        "rung_200": {"name": "rung_200", "rung_of": {"job": "pt", "step": 200}, "cmd": ["r"]}}
BR = {"pt": "ckpt-N"}
CFG = {"campaign": "base2-test", "epoch": 1, "first_start": ["pt"]}


class Flaky(hf_store.DirBackend):
    """A store whose commits and reads fail on demand, with a token to keep out of every log."""
    token = TOKEN

    def __init__(self, root):
        super().__init__(root)
        self.fail_commits, self.fail_reads, self.commits, self.fail_marks = 0, False, 0, False

    def commit(self, branch, adds, deletes, message, parent=None):
        if self.fail_commits or (self.fail_marks and branch == "main"):
            self.fail_commits = max(0, self.fail_commits - 1)
            raise ConnectionError(f"503 from the hub (Authorization: Bearer {TOKEN})")
        self.commits += 1
        super().commit(branch, adds, deletes, message, parent=parent)

    def tree(self, branch):
        if self.fail_reads:
            raise ConnectionError("hub unreachable")
        return super().tree(branch)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def setup(tmp_path, cfg=CFG, **kw):
    chat = tmp_path / "chat"
    (chat / "pt").mkdir(parents=True)
    backend = Flaky(str(tmp_path / "repo"))
    logs, clock = [], Clock()
    store = hf_store.Store(backend, log=logs.append, delay=0)
    c = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=cfg, **kw)
    return chat, backend, store, c, logs, clock


def fresh(tmp_path):
    return hf_store.Store(hf_store.DirBackend(str(tmp_path / "repo")), log=lambda m: None, delay=0)


def events(chat, upto):
    with open(chat / "pt" / "events.jsonl", "a") as f:
        for s in upto:
            f.write(json.dumps({"event": "step", "step": s}) + "\n")
            if s % 100 == 0:
                f.write(json.dumps({"event": "checkpoint", "step": s}) + "\n")


def commit_lines(tmp_path, what):
    return [ln for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines() if f'"{what}: ' in ln]


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
    assert len(commit_lines(tmp_path, "generation")) == 4
    doc = json.loads((chat / "courier_status.json").read_text())
    st = doc["jobs"]["pt"]
    assert st["pushed_step"] == 400 and st["pushes"] == 4 and st["failures"] == 0 and st["marks"] == [200]
    assert st["verified"] == [100, 200, 300, 400] and st["fenced"] is None
    assert doc["epoch"] == 1 and doc["boot_id"] == c.boot and doc["huggingface_hub"] == hf_store.hub_version()
    # What was pushed carries who wrote it and for what, and resumes: files whole, the record cut at the checkpoint.
    man = fresh(tmp_path).read_manifest("ckpt-N", "gens/g400")
    assert (man["campaign"], man["job"], man["epoch"], man["boot_id"]) == ("base2-test", "pt", 1, c.boot)
    assert man["spec_sha256"] == hf_store.spec_hash(JOBS["pt"])
    step, rejected = fresh(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"), expect=cc.identity(CFG, str(chat), JOBS["pt"]))
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
        assert fresh(tmp_path / "b").verify_by_download("ckpt-N", s)
    assert store2.steps("ckpt-N")[-1] == 300


def test_a_step_behind_the_store_is_refused_and_no_step_is_written_twice(tmp_path):
    chat, backend, store, c, logs, _ = setup(tmp_path)
    for step in (100, 200, 300):
        S.make_gen(str(chat / "pt"), step)
        c.tick()
    # A VM that holds an older checkpoint than the store (a run restarted in place): nothing pushed.
    chat2 = tmp_path / "vm2" / "chat"
    (chat2 / "pt").mkdir(parents=True)
    S.make_gen(str(chat2 / "pt"), 100, seed=7)  # other bytes: they must not replace anything
    c2 = cc.Courier(store, str(chat2), JOBS, BR, log=logs.append, cfg={**CFG, "epoch": 2})
    before = backend.commits
    c2.tick()
    c2.tick()
    assert backend.commits == before and store.steps("ckpt-N") == [200, 300]
    st = json.loads((chat2 / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["refused_step"] == 100 and st["pushed_step"] == 300
    assert sum("is behind the store's g300; refused" in ln for ln in logs) == 1
    # The same step as the store's is not written again, even with other bytes; the next one is.
    S.make_gen(str(chat2 / "pt"), 300, seed=7)
    c2.tick()
    assert backend.commits == before
    assert (tmp_path / "repo" / "ckpt-N" / "gens" / "g300" / "model.safetensors").read_bytes() == S.state_at(300)[0]
    S.make_gen(str(chat2 / "pt"), 400, seed=7)
    c2.tick()
    assert store.steps("ckpt-N") == [200, 300, 400]  # one generation verified on this VM so far: nothing pruned yet
    S.make_gen(str(chat2 / "pt"), 500, seed=7)
    c2.tick()
    assert store.steps("ckpt-N") == [400, 500]


def test_fencing_two_couriers_on_one_branch(tmp_path):
    """The driver lost contact with a VM and made another. Both train and both hold the token. The old one must
    stop writing: at once when it sees a generation of a higher epoch, and also when a boot of its own epoch has
    pushed past what it knew. The new one goes on, and nothing the old one holds reaches the branch again."""
    chat1, backend, store, old, logs, _ = setup(tmp_path)
    S.make_gen(str(chat1 / "pt"), 100)
    old.tick()
    S.make_gen(str(chat1 / "pt"), 200)
    old.tick()
    assert store.steps("ckpt-N") == [100, 200]
    # The replacement (epoch 2) resumes from the store's 200 and pushes 300 while the old VM is still training.
    chat2 = tmp_path / "vm2" / "chat"
    (chat2 / "pt").mkdir(parents=True)
    logs2 = []
    new = cc.Courier(hf_store.Store(backend, log=logs2.append, delay=0), str(chat2), JOBS, BR, log=logs2.append,
                     cfg={**CFG, "epoch": 2})
    S.make_gen(str(chat2 / "pt"), 300)
    new.tick()
    assert store.steps("ckpt-N") == [100, 200, 300]
    # The old VM reaches 300 and then 400 with its own bytes. It reads the head before pushing and stops.
    S.make_gen(str(chat1 / "pt"), 300, seed=9)
    S.make_gen(str(chat1 / "pt"), 400, seed=9)
    before = backend.commits
    old.tick()
    st = json.loads((chat1 / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["fenced"]["head_epoch"] == 2 and st["fenced"]["head_step"] == 300
    assert any("FENCED" in ln and "epoch 2" in ln for ln in logs)
    assert json.loads((chat1 / "COURIER_FENCED").read_text().splitlines()[0])["job"] == "pt"
    S.make_gen(str(chat1 / "pt"), 500, seed=9)
    (chat1 / "pt" / "rung_500").mkdir()
    (chat1 / "pt" / "rung_500" / "model.safetensors").write_bytes(b"old vm weights")
    (chat1 / "pt" / "rung_500" / "state.txt").write_text("500\n{}\n")
    old.tick()
    old.tick()
    assert backend.commits == before and store.mark_steps("pt") == [200]  # neither generations nor marks any more
    assert old.idle() is False  # its job has not ended; once it has, a fenced courier has nothing left to do
    (chat1 / "STATUS_pt").write_text("FAILED\n")
    assert old.idle() is True
    # The new one is not disturbed, and every generation above the fork is its own.
    S.make_gen(str(chat2 / "pt"), 400)
    new.tick()
    assert new.s["pt"]["fenced"] is None
    for s in (300, 400):
        assert fresh(tmp_path).read_manifest("ckpt-N", f"gens/g{s}")["boot_id"] == new.boot
    assert (tmp_path / "repo" / "ckpt-N" / "gens" / "g400" / "model.safetensors").read_bytes() == S.state_at(400)[0]
    # Same epoch (the runner's state was lost and the count restarted): the one that finds a foreign boot's newer
    # generation since it last looked is the one that stops.
    chat3, backend3, store3, a, logs3, _ = setup(tmp_path / "same")
    chat4 = tmp_path / "same" / "vm2" / "chat"
    (chat4 / "pt").mkdir(parents=True)
    b = cc.Courier(hf_store.Store(backend3, log=logs3.append, delay=0), str(chat4), JOBS, BR, log=logs3.append, cfg=CFG)
    S.make_gen(str(chat3 / "pt"), 100)
    a.tick()
    S.make_gen(str(chat4 / "pt"), 200)
    b.tick()  # b's first look: it sees a's 100 as history and pushes 200
    S.make_gen(str(chat3 / "pt"), 300)
    a.tick()  # a finds b's 200, which it never knew
    assert a.s["pt"]["fenced"] and b.s["pt"]["fenced"] is None and store3.steps("ckpt-N") == [100, 200]
    # A start that finds a higher epoch already in the store is fenced before its first push.
    chat5 = tmp_path / "vm5" / "chat"
    (chat5 / "pt").mkdir(parents=True)
    stale = cc.Courier(store, str(chat5), JOBS, BR, log=logs.append, cfg={**CFG, "epoch": 1})
    S.make_gen(str(chat5 / "pt"), 900)
    before = backend.commits
    stale.tick()
    assert stale.s["pt"]["fenced"] and backend.commits == before


def test_a_commit_that_loses_the_race_is_refused_by_the_store(tmp_path, monkeypatch):
    """The head is read, then another writer commits before this push lands: the conditional commit fails, nothing
    is written, and the next pass reads the head again (and is fenced by what it finds)."""
    chat, backend, store, c, logs, clock = setup(tmp_path)
    S.make_gen(str(chat / "pt"), 100)
    c.tick()
    S.make_gen(str(chat / "pt"), 200)
    real = store.push_generation
    other = tmp_path / "other"

    def late(branch, job, step, folder, identity=None, parent=None):
        S.make_gen(str(other), 150, seed=3)  # the other writer's commit lands between the read and the push
        fresh(tmp_path).push_generation(branch, job, 150, str(other / "ckpt_gen" / "g150"), {"epoch": 1, "boot_id": "other"})
        return real(branch, job, step, folder, identity, parent)

    monkeypatch.setattr(store, "push_generation", late)
    c.tick()
    assert store.steps("ckpt-N") == [100, 150] and any("moved" in ln for ln in logs)
    monkeypatch.undo()
    clock.t += 31
    c.tick()
    assert store.steps("ckpt-N") == [100, 150] and c.s["pt"]["fenced"]


def test_marks_are_staged_first_pushed_first_once_and_never_removed(tmp_path):
    chat, backend, store, c, logs, clock = setup(tmp_path)
    out = chat / "pt"
    m100 = S.make_gen(str(out), 100)[0]
    # jobs_vm.take_snapshot's copy at a step that is not in the job's own list still counts as a mark.
    (out / "rung_100").mkdir()
    (out / "rung_100" / "model.safetensors").write_bytes(m100)
    (out / "rung_100" / "state.txt").write_text("100\n{}\n")
    backend.fail_commits = 99  # the store is down
    c.tick()
    assert backend.commits == 0 and c.staged_marks("pt") == [100]
    # The rung job scores the snapshot and drops its weights before any push got through.
    (out / "rung_100" / "model.safetensors").unlink()
    m200 = S.make_gen(str(out), 200)[0]  # step 200 is in snapshot_steps: its mark comes from the generation itself
    backend.fail_commits = 0
    clock.t += 3600
    c.tick()
    c.tick()
    assert store.mark_steps("pt") == [100, 200]
    got = tmp_path / "got"
    store.pull_mark("pt", 100, str(got / "100"), expect={"campaign": "base2-test", "job": "pt"})
    store.pull_mark("pt", 200, str(got / "200"))
    assert (got / "100" / "model.safetensors").read_bytes() == m100
    assert (got / "200" / "model.safetensors").read_bytes() == m200
    with pytest.raises(hf_store.IdentityError):
        store.pull_mark("pt", 100, str(got / "x"), expect={"campaign": "another", "job": "pt"})
    assert len(commit_lines(tmp_path, "mark")) == 2
    # The mark of step 200 went up before the generation of step 200.
    order = [json.loads(ln)["message"] for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines()]
    assert order.index("mark: marks/pt/rung_200") < order.index("generation: gens/g200")
    # jobs_vm's own snapshot of step 200 appears later; later ticks and a restarted courier push nothing again.
    (out / "rung_200").mkdir()
    (out / "rung_200" / "model.safetensors").write_bytes(m200)
    (out / "rung_200" / "state.txt").write_text("200\n{}\n")
    c.tick()
    c2 = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, cfg=CFG)
    c2.tick()
    assert len(commit_lines(tmp_path, "mark")) == 2
    # Many generations later the marks are still there: the prune works on the ckpt branch only.
    for step in (300, 400, 500, 600):
        S.make_gen(str(out), step)
        c2.tick()
    assert store.steps("ckpt-N") == [500, 600] and store.mark_steps("pt") == [100, 200]
    assert not any('"branch": "main"' in ln and '"deletes": ["' in ln
                   for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines())


def test_a_mark_is_staged_while_a_push_is_in_flight(tmp_path):
    """A push can take minutes. A mark that appears meanwhile, and whose weights the rung job then drops, is linked
    into the staging folder by the waiting parent, not after the push."""
    chat, backend, store, c, logs, _ = setup(tmp_path, push_timeout=60)
    out = chat / "pt"
    S.make_gen(str(out), 100)
    real = store.push_generation

    def slow(*a, **kw):
        time.sleep(3)  # in the child: the parent runs stage_all_marks about once a second meanwhile
        return real(*a, **kw)

    store.push_generation = slow
    (out / "rung_70").mkdir()

    def appear():  # the snapshot arrives, and is dropped, during the push
        if not (out / "rung_70" / "state.txt").exists():
            (out / "rung_70" / "model.safetensors").write_bytes(b"weights at 70")
            (out / "rung_70" / "state.txt").write_text("70\n{}\n")
        elif 70 in c.staged_marks("pt") and (out / "rung_70" / "model.safetensors").exists():
            (out / "rung_70" / "model.safetensors").unlink()

    staging = c.stage_all_marks
    c.stage_all_marks = lambda: (appear(), staging())
    c.tick()
    assert store.steps("ckpt-N") == [100] and not (out / "rung_70" / "model.safetensors").exists()
    got = tmp_path / "got"
    fresh(tmp_path).pull_mark("pt", 70, str(got))
    assert (got / "model.safetensors").read_bytes() == b"weights at 70"


def test_prune_waits_for_verified_generations_and_spares_an_unpushed_marks_generation(tmp_path):
    chat, backend, store, c, logs, clock = setup(tmp_path)
    out = str(chat / "pt")
    # Three generations another VM left; this courier has verified none of them.
    other = cc.Courier(store, str(tmp_path / "vm0" / "chat"), JOBS, BR, log=lambda m: None, cfg={**CFG, "epoch": 0})
    for step in (10, 20, 30):
        S.make_gen(str(tmp_path / "vm0" / "chat" / "pt"), step)
        other.tick()
    assert store.steps("ckpt-N") == [20, 30]
    S.make_gen(out, 100)
    c.tick()
    assert store.steps("ckpt-N") == [20, 30, 100]  # one generation verified here: the two before it are not removed
    S.make_gen(out, 150)
    c.tick()
    assert store.steps("ckpt-N") == [100, 150] and c.s["pt"]["verified"] == {100, 150}
    # Step 200 is a mark step and main refuses commits: the mark stays staged, the generation still goes up, and
    # from then on nothing at or above 200 is pruned, so the mark's weights stay recoverable from the store.
    backend.fail_marks = True
    S.make_gen(out, 200)
    c.tick()
    assert c.staged_marks("pt") == [200] and store.mark_steps("pt") == []
    for step in (300, 400, 500):
        clock.t += 3600
        S.make_gen(out, step)
        c.tick()
    assert store.steps("ckpt-N") == [200, 300, 400, 500]
    assert json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]["staged_marks"] == [200]
    backend.fail_marks = False
    clock.t += 3600
    S.make_gen(out, 600)
    c.tick()
    assert store.mark_steps("pt") == [200] and store.steps("ckpt-N") == [500, 600]
    # A restore's download counts as a verification on the VM that made it, and only there.
    chat2 = tmp_path / "vm2" / "chat"
    chat2.mkdir(parents=True)
    cfg2 = {**CFG, "epoch": 2, "min_step": {"pt": 600}, "first_start": []}
    cc.restore(fresh(tmp_path), str(chat2), JOBS, BR, log=lambda m: None, cfg=cfg2)
    c2 = cc.Courier(fresh(tmp_path), str(chat2), JOBS, BR, log=lambda m: None, cfg=cfg2)
    assert c2.s["pt"]["verified"] == {600}
    S.make_gen(str(chat2 / "pt"), 700)
    c2.tick()
    assert fresh(tmp_path).steps("ckpt-N") == [600, 700]


def test_release_runs_only_when_the_config_asks_and_with_its_grace(tmp_path):
    chat, backend, store, c, logs, _ = setup(tmp_path)
    asked = []
    store.release = lambda owner, grace=None: asked.append((owner, grace)) or 0
    for step in (100, 300, 400):
        S.make_gen(str(chat / "pt"), step)
        c.tick()
    assert store.steps("ckpt-N") == [300, 400] and asked == []  # pruned, and release is off unless asked for
    chat2, backend2, store2, c2, logs2, _ = setup(tmp_path / "b", cfg={**CFG, "release_grace_s": 60})
    store2.reclaim_on = True
    store2.release = lambda owner, grace=None: asked.append((owner, grace)) or 3
    for step in (100, 300, 400):
        S.make_gen(str(chat2 / "pt"), step)
        c2.tick()
    assert asked == [(c2.boot, 60.0)] and any("storage release: 3 objects deleted" in ln for ln in logs2)


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
    S.make_gen(out, 300)  # training went on meanwhile
    S.make_gen(out, 400)
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
    assert store.steps("ckpt-N") == [400]
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["pushed_step"] == 400 and st["failures"] == 0 and st["last_error"] is None
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
    for step in (100, 300, 400):
        S.make_gen(out, step)
        clock.t += 100
        c.tick()
    assert store.steps("ckpt-N") == [100]  # 300 and 400 fell inside the interval
    (chat / "STATUS_pt").write_text("FINISHED\n")
    assert not c.idle()
    c.tick()
    assert store.steps("ckpt-N") == [100, 400] and c.idle()  # a job that ended: its last generation goes at once


def test_every_store_call_of_the_daemon_is_bounded(tmp_path):
    chat, backend, store, c, logs, _ = setup(tmp_path, push_timeout=60)
    S.make_gen(str(chat / "pt"), 100)
    c.tick()
    assert store.steps("ckpt-N") == [100] and c.s["pt"]["verified"] == {100}
    # A push that hangs is ended at the limit, counted as a failure, and the daemon goes on.
    chat2, backend2, store2, c2, logs2, _ = setup(tmp_path / "b", push_timeout=1)
    S.make_gen(str(chat2 / "pt"), 100)
    store2.push_generation = lambda *a, **kw: time.sleep(60)
    t0 = time.time()
    c2.tick()
    assert time.time() - t0 < 20 and c2.s["pt"]["failures"] == 1 and any("over 1 s: ended" in ln for ln in logs2)
    assert not hf_store._CHILDREN
    # One courier per VM: the pid file is taken exclusively.
    (chat / "courier.pid").write_text(str(os.getpid() + 0))
    cc.take_pid(str(chat))  # our own pid: taken over
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (chat / "courier.pid").write_text(str(p.pid))
        with pytest.raises(SystemExit, match="another courier"):
            cc.take_pid(str(chat))
    finally:
        p.kill()
        p.wait()


# ---------------------------------------------------------------- restore
def pushed_store(tmp_path, steps):
    chat, backend, store, c, logs, _ = setup(tmp_path / "vm1")
    for step in steps:
        S.make_gen(str(chat / "pt"), step)
        events(chat, range(step - 99, step + 1))
        c.tick()
    return backend, store


def cfg2(**kw):
    return {**CFG, "epoch": 2, "first_start": [], **kw}


def test_restore_installs_a_whole_generation_as_the_trainers_own(tmp_path):
    backend, store = pushed_store(tmp_path, [100, 200, 300])
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    logs = []
    rec = cc.restore(store, str(chat), JOBS, BR, log=logs.append, cfg=cfg2(min_step={"pt": 300}))["pt"]
    assert rec["store_step"] == 300 and rec["placed"] and rec["rejected"] == [] and rec["verified_step"] == 300
    out = chat / "pt"
    m, o = S.state_at(300)
    # The trainer's layout: the generation whole under ckpt_gen, `live` naming it, the flat names links through it.
    assert os.readlink(out / "ckpt_gen" / "live") == "g300"
    assert sorted(os.listdir(out / "ckpt_gen" / "g300")) == ["MANIFEST.json", "model.safetensors", "optim.safetensors", "state.txt"]
    hf_store.check_generation(str(out / "ckpt_gen" / "g300"), 300)
    assert all(os.path.islink(out / f) for f in cc.FLAT)
    assert (out / "model.safetensors").read_bytes() == m and (out / "optim.safetensors").read_bytes() == o
    lines = (out / "state.txt").read_text().split("\n")
    assert lines[0] == "300" and json.loads(lines[1])["adamw_t"] == [300]  # the optimiser's step count comes with it
    assert (out / "events.jsonl").read_text().splitlines()[-1] == json.dumps({"event": "checkpoint", "step": 300})
    assert not (chat / "upload" / "pt").exists() and cc.local_step(str(chat), "pt") == 300
    assert not (chat / "store_restore" / "pt").exists()
    # The mark the unfinished rung job needs came with it.
    assert (out / "rung_200" / "model.safetensors").read_bytes() == S.state_at(200)[0]
    assert (out / "rung_200" / "state.txt").read_text().startswith("200\n")
    # jobs_vm.py's restore(), unchanged, leaves an installed generation alone (there is no upload for the job).
    import importlib.util
    spec = importlib.util.spec_from_file_location("jobs_vm_under_test", CC / "jobs_vm.py")
    jv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jv)
    jv.C = str(chat)
    (chat / "upload" / "pt").mkdir(parents=True)  # the runner makes the folder; nothing is in it
    assert not jv.awaiting_upload("pt")
    jv.restore("pt")
    assert os.readlink(out / "ckpt_gen" / "live") == "g300" and (out / "optim.safetensors").read_bytes() == o
    assert json.loads((out / "state.txt").read_text().split("\n")[1])["adamw_t"] == [300]
    # The boot runs again on the same VM (the manager died): this VM holds step 400, whole; nothing is installed.
    S.make_gen(str(out), 400)
    rec = cc.restore(store, str(chat), JOBS, BR, log=logs.append, cfg=cfg2())["pt"]
    assert rec["placed"] is False and rec["local_step"] == 400 and os.readlink(out / "ckpt_gen" / "live") == "g400"
    # A local state number without a whole generation behind it counts for nothing: the store's is installed.
    with open(out / "ckpt_gen" / "g400" / "optim.safetensors", "ab") as f:
        f.write(b"torn")
    assert cc.local_step(str(chat), "pt") == 0
    rec = cc.restore(store, str(chat), JOBS, BR, log=logs.append, cfg=cfg2())["pt"]
    assert rec["placed"] and os.readlink(out / "ckpt_gen" / "live") == "g300"
    # A job that has ended is left alone.
    (chat / "STATUS_pt").write_text("FINISHED\n")
    assert cc.restore(store, str(chat), JOBS, BR, log=logs.append, cfg=cfg2()) == {}


def test_a_kill_during_install_leaves_nothing_to_trust(tmp_path, monkeypatch):
    """The boot dies between the generation's rename into place and the `live` pointer's move. A trainer started
    on that folder finds no checkpoint to resume, the next boot sees a VM at step 0 and installs again."""
    backend, store = pushed_store(tmp_path, [100, 200])
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    real = os.symlink

    def killed(src, dst, *a, **kw):
        if str(dst).endswith("live.install"):
            raise KeyboardInterrupt("the VM's boot is killed here")
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(os, "symlink", killed)
    with pytest.raises(KeyboardInterrupt):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=lambda m: None, cfg=cfg2())
    monkeypatch.undo()
    out = chat / "pt"
    assert (out / "ckpt_gen" / "g200").is_dir() and not os.path.lexists(out / "ckpt_gen" / "live")
    assert not any(os.path.lexists(out / f) for f in cc.FLAT)  # no flat checkpoint a trainer could resume from
    assert cc.local_step(str(chat), "pt") == 0
    rec = cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=lambda m: None, cfg=cfg2())["pt"]
    assert rec["placed"] and rec["local_step"] == 0 and os.readlink(out / "ckpt_gen" / "live") == "g200"
    assert (out / "optim.safetensors").read_bytes() == S.state_at(200)[1]
    # The same with an older local generation live: a kill leaves the old one live and whole, never a mix.
    chat3 = tmp_path / "vm3" / "chat"
    chat3.mkdir(parents=True)
    S.make_gen(str(chat3 / "pt"), 100)
    monkeypatch.setattr(os, "symlink", killed)
    with pytest.raises(KeyboardInterrupt):
        cc.restore(fresh(tmp_path / "vm1"), str(chat3), JOBS, BR, log=lambda m: None, cfg=cfg2())
    monkeypatch.undo()
    assert cc.local_step(str(chat3), "pt") == 100
    assert (chat3 / "pt" / "model.safetensors").read_bytes() == S.state_at(100)[0]
    assert cc.restore(fresh(tmp_path / "vm1"), str(chat3), JOBS, BR, log=lambda m: None, cfg=cfg2())["pt"]["placed"]
    assert cc.local_step(str(chat3), "pt") == 200


def test_restore_falls_back_only_on_a_verification_error(tmp_path):
    backend, store = pushed_store(tmp_path, [100, 200, 300])
    bad = tmp_path / "vm1" / "repo" / "ckpt-N" / "gens" / "g300" / "model.safetensors"
    bad.write_bytes(bad.read_bytes()[:-1] + b"?")
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    logs = []
    # Home has seen 300 pushed; 300 is listed and fails its hash: the fallback to 200 is allowed and recorded.
    rec = cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=logs.append, cfg=cfg2(min_step={"pt": 300}))["pt"]
    assert rec["store_step"] == 200 and [s for s, _ in rec["rejected"]] == [300] and rec["placed"]
    assert (chat / "pt" / "optim.safetensors").read_bytes() == S.state_at(200)[1]
    assert json.loads((chat / "store_restore" / "pt.json").read_text())["store_step"] == 200
    # A store that stops answering in the middle is not a damaged generation: the restore fails, nothing is
    # installed, and no older generation is taken.
    chat3 = tmp_path / "vm3" / "chat"
    chat3.mkdir(parents=True)
    bad.write_bytes(S.state_at(300)[0])
    real = backend.fetch
    backend.fetch = lambda br, path, dest: (_ for _ in ()).throw(ConnectionError("reset")) if "optim" in path else real(br, path, dest)
    store.tries = 2
    with pytest.raises(ConnectionError):
        cc.restore(store, str(chat3), JOBS, BR, log=logs.append, cfg=cfg2())
    assert cc.local_step(str(chat3), "pt") == 0 and not (chat3 / "pt" / "ckpt_gen").exists()
    backend.fetch = real
    assert cc.restore(store, str(chat3), JOBS, BR, log=logs.append, cfg=cfg2())["pt"]["store_step"] == 300


def test_restore_never_starts_from_zero_on_a_store_it_cannot_trust(tmp_path):
    backend, store = pushed_store(tmp_path, [100, 200])
    chat = tmp_path / "vm2" / "chat"
    chat.mkdir(parents=True)
    quiet = lambda m: None  # noqa: E731
    backend.fail_reads = True
    with pytest.raises(ConnectionError):
        cc.restore(store, str(chat), JOBS, BR, log=quiet, cfg=cfg2())
    backend.fail_reads = False
    # Minimum step: home has seen 200 pushed. A branch that is empty, renamed or behind is not a first start.
    for br, why in (({"pt": "ckpt-renamed"}, "no generation"), ({"pt": "ckpt-N"}, "up to g200")):
        with pytest.raises(hf_store.IdentityError, match=why):
            cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, br, log=quiet, cfg=cfg2(min_step={"pt": 200 if "renamed" in br["pt"] else 300}))
    # An empty branch with no step seen at home still needs the first start admitted, job by job.
    with pytest.raises(hf_store.IdentityError, match="first start is not admitted"):
        cc.restore(fresh(tmp_path / "empty"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())
    rec = cc.restore(fresh(tmp_path / "empty"), str(chat), JOBS, BR, log=quiet, cfg=cfg2(first_start=["pt"]))["pt"]
    assert rec["store_step"] is None and rec["placed"] is False
    # Identity: another campaign's branch, another spec, other inputs are refused, with no fallback.
    with pytest.raises(hf_store.IdentityError, match="another campaign"):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2(campaign="base2-S"))
    other = {**JOBS, "pt": {**JOBS["pt"], "spec": {"train": {"steps": 901}}}}
    with pytest.raises(hf_store.IdentityError, match="another spec_sha256"):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), other, BR, log=quiet, cfg=cfg2())
    (chat / "store_inputs.json").write_text(json.dumps({"inputs_sha256": "f" * 64}))
    with pytest.raises(hf_store.IdentityError, match="another inputs_sha256"):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())
    (chat / "store_inputs.json").unlink()
    assert not (chat / "pt" / "ckpt_gen").exists()
    # No generation verifies: refused.
    for s in (100, 200):
        p = tmp_path / "vm1" / "repo" / "ckpt-N" / "gens" / f"g{s}" / "optim.safetensors"
        p.write_bytes(b"damaged")
    with pytest.raises(hf_store.StoreError, match="no generation verifies"):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())
    assert cc.local_step(str(chat), "pt") == 0
    # The command the boot runs exits 1 on the same failure, with one line and no traceback, inside its limit.
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'vm1' / 'repo'}", "branches": BR, **cfg2()}))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": list(JOBS.values())}))
    r = subprocess.run([sys.executable, str(CC / "ckpt_courier.py"), "restore", "--config", str(cfg), "--jobs", str(jobs),
                        "--chat", str(chat), "--retry-delay", "0", "--limit", "60"], capture_output=True, text=True)
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


def make_vm(tmp_path, name, repo, **cfg):
    """A VM folder as the runner leaves it before the boot: the store config with the token already in the tmpfs."""
    root = tmp_path / name
    chat, pk, shm = root / "chat", root / "pk", root / "shm"
    (chat / "upload").mkdir(parents=True)
    (pk / "build_cuda").mkdir(parents=True)
    shm.mkdir(mode=0o700)
    os.symlink(ROOT / "tools", pk / "tools")
    S.write_trainer(str(pk / "build_cuda" / "mtstudio"), sys.executable)
    (root / "rung.py").write_text(RUNG)
    (shm / "token").write_text(TOKEN)
    (shm / "token").chmod(0o600)
    (chat / "store.json").write_text(json.dumps({"repo": f"dir:{repo}", "branches": {"pt": "ckpt-N"}, "campaign": "rehearsal",
                                                 "token_file": str(shm / "token"), **cfg}))
    jobs = {"max_parallel": 1, "store_inputs": [{"prefix": "seed/vocab", "to": "chat/upload"}], "jobs": [
        {"name": "pt", "model_name": "m", "snapshot_steps": [MARK], "env": {"MICROTORCH_CHECKPOINT_TXN": "1"},
         "spec": {"train": {"steps": STEPS, "checkpoint_every": EVERY, "seed": 0}}},
        {"name": "rung_mark", "after": "pt", "rung_of": {"job": "pt", "step": MARK}, "light": True,
         "cmd": [sys.executable, str(root / "rung.py"), f"{chat}/pt/rung_{MARK}", "{out}"], "outputs": ["rung.json"]}]}
    (root / "jobs.json").write_text(json.dumps(jobs))
    return root


def boot(root, sleep, **env):
    env = {**os.environ, "PK_BOOT_ROOT": str(root), "PK_STORE_TOKEN_DIR": str(root / "shm"), "JOBS_TICK": "0.2",
           "PK_COURIER_ARGS": "--tick 0.1 --push-timeout 30", "PK_RESTORE_ARGS": "--retry-delay 0",
           "PK_BOOT_LIMIT_S": "120", "STANDIN_STEP_SLEEP": str(sleep), "CUDA_VISIBLE_DEVICES": "", **env}
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


def seeded(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "vocab.gguf").write_bytes(b"vocab bytes")
    fresh(tmp_path).push_set("main", "seed/vocab", str(seed))
    return tmp_path / "repo"


def test_rehearsal_two_vms_through_the_real_boot(tmp_path):
    repo = seeded(tmp_path)
    store = fresh(tmp_path)
    vm1 = make_vm(tmp_path, "vm1", repo, epoch=1, first_start=["pt"])
    try:
        # Session one: the boot pulls the inputs, finds no generation (a first start, admitted), starts the courier.
        r = boot(vm1, 0.03)
        assert r.returncode == 0, read(vm1 / "chat" / "boot.log")
        assert (vm1 / "chat" / "upload" / "vocab.gguf").read_bytes() == b"vocab bytes"
        assert "first start admitted" in read(vm1 / "chat" / "boot.log")
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
        # Session two, a new VM folder with an empty disk; the runner has seen `newest` pushed and sends epoch 2.
        vm2 = make_vm(tmp_path, "vm2", repo, epoch=2, min_step={"pt": newest}, first_start=[])
        r = boot(vm2, 0.005)
        assert r.returncode == 0, read(vm2 / "chat" / "boot.log")
        chat = vm2 / "chat"
        rec = json.loads(read(chat / "store_restore" / "pt.json"))
        assert rec["store_step"] == prev and [s for s, _ in rec["rejected"]] == [newest] and rec["placed"]
        wait(lambda: read(chat / "STATUS").strip() == "FINISHED", "session two's end")
        assert read(chat / "STATUS_pt").strip() == "FINISHED" and read(chat / "STATUS_rung_mark").strip() == "FINISHED"
        ev = [json.loads(ln) for ln in read(chat / "pt" / "events.jsonl").splitlines()]
        resumes = [e for e in ev if e["event"] == "resume"]
        assert resumes == [{"event": "resume", "step": prev, "optimizer": "restored", "from": "generation"}]
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
        final = fresh(tmp_path)
        assert final.pull_generation("ckpt-N", str(tmp_path / "final")) == (STEPS, [])
        man = final.read_manifest("ckpt-N", f"gens/g{STEPS}")
        assert (man["campaign"], man["job"], man["epoch"]) == ("rehearsal", "pt", 2)
        assert man["inputs_sha256"] == json.loads(read(chat / "store_inputs.json"))["inputs_sha256"]
        cs = json.loads(read(chat / "courier_status.json"))
        assert cs["jobs"]["pt"]["pushed_step"] == STEPS and cs["jobs"]["pt"]["failures"] == 0 and cs["epoch"] == 2
        # The token: it stayed in the tmpfs stand-in at mode 600 while the session worked, is in no log, and was
        # removed by the boot's own last step once the courier had pushed the last generation and ended.
        wait(lambda: "TOKEN_GONE" in read(chat / "boot.log"), "the orderly removal of the token")
        assert os.listdir(vm2 / "shm") == [] and not (chat / "upload" / "store_token").exists()
        for vm in (vm1, vm2):
            for f in ("boot.log", "courier.log", "jobs.log", "courier_status.json", "pt_train.log", "store.json"):
                assert TOKEN not in read(vm / "chat" / f)
        assert TOKEN not in read(repo / "commits.jsonl")
        print(f"rehearsal: lost at step {lost_at}; store held {steps}; g{newest} damaged; resumed at {prev}; "
              f"finished at {STEPS}; store now {store.steps('ckpt-N')}, marks {store.mark_steps('pt')}")
    finally:
        kill_vm(vm1)
        kill_vm(tmp_path / "vm2")


def test_boot_failure_removes_the_token_and_starts_nothing(tmp_path):
    """Each way a boot can fail ends as BOOT_FAILED with every copy of the token gone (the adopted file, a
    temporary, one still in upload/) and no job started: an input that does not verify, missing history, a first
    start that was not admitted, and a boot that runs past its limit."""
    repo = seeded(tmp_path)

    def failed(vm, why, **env):
        (vm / "shm" / "token.tmp77").write_text(TOKEN)
        (vm / "chat" / "upload" / "store_token").write_text(TOKEN)
        try:
            r = boot(vm, 0, **env)
            wait(lambda: read(vm / "chat" / "STATUS").strip() == "BOOT_FAILED", f"BOOT_FAILED ({why})", 60)
            assert r.returncode != 0
            assert why in read(vm / "chat" / "boot.log") + read(vm / "chat" / "courier.log")
            wait(lambda: os.listdir(vm / "shm") == [], "the token's removal", 30)
            assert not (vm / "chat" / "upload" / "store_token").exists()
            assert not (vm / "chat" / "jobs_vm.pid").exists() and not (vm / "chat" / "STATUS_pt").exists()
            assert TOKEN not in read(vm / "chat" / "boot.log")
        finally:
            kill_vm(vm)

    failed(make_vm(tmp_path, "a", repo, epoch=1), "first start is not admitted")
    failed(make_vm(tmp_path, "b", repo, epoch=3, min_step={"pt": 500}), "home has seen step 500")
    slow = make_vm(tmp_path, "c", repo, epoch=1, first_start=["pt"])
    (slow / "pk" / "build_cuda" / "mtstudio").unlink()  # the build step runs; its clone is replaced by a hang
    (slow / "bin").mkdir()
    (slow / "bin" / "git").write_text("#!/bin/sh\nexec sleep 300\n")
    (slow / "bin" / "git").chmod(0o755)
    failed(slow, "boot over 3 s", PK_BOOT_LIMIT_S="3", PATH=f"{slow / 'bin'}:{os.environ['PATH']}")
    (repo / "main" / "seed" / "vocab" / "vocab.gguf").write_bytes(b"vocab bytez")
    failed(make_vm(tmp_path, "d", repo, epoch=1, first_start=["pt"]), "sha256 or size differs")


# ---------------------------------------------------------------- the real trainer on CPU
def test_resume_through_the_store_equals_a_straight_run_real_trainer(tmp_path):
    """40 steps straight, and 20 steps, a pull from the store on a new folder, then 20 more: the same weights,
    optimiser and exported model, byte for byte, at full precision, and the resume event reads optimizer: restored.
    The trainer resumes from the generation the restore installed, after validating it by its own manifest."""
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
    c = cc.Courier(store, str(chat1), jobs, BR, log=lambda m: None, cfg=CFG)
    c.tick()
    assert store.steps("ckpt-N") == [20]
    # Session two: a new folder; the boot's restore installs the generation, then the trainer starts.
    chat2 = tmp_path / "vm2" / "chat"
    chat2.mkdir(parents=True)
    rec = cc.restore(fresh(tmp_path), str(chat2), jobs, BR, log=lambda m: None, cfg=cfg2(min_step={"pt": 20}))["pt"]
    assert rec["store_step"] == 20 and rec["placed"]
    r = T.run(T.spec(tmp_path, chat2 / "pt", steps=40, every=10, state_tokens=True), cpu)
    assert r.returncode == 0, r.stderr[-800:]
    res = T.events(chat2 / "pt", "resume")
    assert len(res) == 1 and res[0]["step"] == 20 and res[0]["optimizer"] == "restored"
    assert T.finals(chat2 / "pt") == T.finals(control)
    a, b = T.steps_of(control), T.steps_of(chat2 / "pt")
    assert {s: b[s] for s in range(21, 41)} == {s: a[s] for s in range(21, 41)}
    # The courier of session two pushes what the resumed trainer saves, and the store's newest resumes again.
    c2 = cc.Courier(fresh(tmp_path), str(chat2), jobs, BR, log=lambda m: None, cfg=cfg2())
    c2.tick()
    assert fresh(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r")) == (40, [])
