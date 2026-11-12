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

    def commit(self, branch, adds, deletes, message, parent=None, **kw):
        if self.fail_commits or (self.fail_marks and branch == "main"):
            self.fail_commits = max(0, self.fail_commits - 1)
            raise ConnectionError(f"503 from the hub (Authorization: Bearer {TOKEN})")
        self.commits += 1
        super().commit(branch, adds, deletes, message, parent=parent, **kw)

    def tree(self, branch, rev=None):
        if self.fail_reads:
            raise ConnectionError("hub unreachable")
        return super().tree(branch, rev=rev)


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


def test_pushes_each_new_generation_once_and_keeps_two_verified_and_the_newest(tmp_path):
    chat, backend, store, c, logs, _ = setup(tmp_path)
    c.tick()  # nothing saved yet
    assert backend.commits == 0
    for step in (100, 200, 300, 400):
        S.make_gen(str(chat / "pt"), step)
        events(chat, range(step - 99, step + 3))  # the trainer is two steps past the save when the courier looks
        c.tick()
        c.tick()  # nothing new: no second push
    assert store.steps("ckpt-N") == [200, 300, 400]   # two verified when g400 went up, and g400 itself
    assert [v["step"] for v in stats(logs, "verify")] == [100, 200, 300, 400] and all(v["whole"] for v in stats(logs, "verify"))
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
    assert backend.commits == before and store.steps("ckpt-N") == [100, 200, 300]
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
    assert store.steps("ckpt-N") == [100, 200, 300, 400]  # nothing verified on this VM when g400 went up: nothing pruned
    S.make_gen(str(chat2 / "pt"), 500, seed=7)
    c2.tick()
    assert store.steps("ckpt-N") == [100, 200, 300, 400, 500]  # one generation verified on this VM (g400): not two
    S.make_gen(str(chat2 / "pt"), 600, seed=7)
    c2.tick()
    assert store.steps("ckpt-N") == [400, 500, 600]  # two, g400 and g500: what is behind them goes


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


def test_a_head_of_lower_epoch_never_fences_the_newer_courier(tmp_path):
    """The other ordering (the review of 12 Nov 2026 reproduced it): the replacement's courier has read the head,
    and THEN the stale VM pushes a generation the replacement never knew. The old rule (another boot pushed above
    what I knew, whatever its epoch) fenced the replacement and left the branch to the stale VM. The epoch decides:
    a head of lower epoch never fences; the stale courier is the one that stops, when it next reads the head."""
    chat1, backend, store, old, logs, _ = setup(tmp_path)
    for step in (100, 200):
        S.make_gen(str(chat1 / "pt"), step)
        old.tick()
    chat2 = tmp_path / "vm2" / "chat"
    (chat2 / "pt").mkdir(parents=True)
    logs2 = []
    new = cc.Courier(hf_store.Store(backend, log=logs2.append, delay=0), str(chat2), JOBS, BR, log=logs2.append,
                     cfg={**CFG, "epoch": 2})
    S.make_gen(str(chat2 / "pt"), 200)     # the replacement's boot restored g200; its courier reads the head once
    new.tick()
    assert new.s["pt"]["known"] and new.s["pt"]["head"] == 200
    S.make_gen(str(chat1 / "pt"), 300, seed=9)
    old.tick()                              # the stale VM is still alive and pushes g300, epoch 1
    assert store.steps("ckpt-N") == [100, 200, 300]
    S.make_gen(str(chat2 / "pt"), 400)
    new.tick()                              # the replacement reaches g400: the head is another boot's, above what it knew
    assert new.s["pt"]["fenced"] is None and not (chat2 / "COURIER_FENCED").exists()
    assert fresh(tmp_path).read_manifest("ckpt-N", "gens/g400")["boot_id"] == new.boot
    S.make_gen(str(chat1 / "pt"), 400, seed=9)
    S.make_gen(str(chat1 / "pt"), 500, seed=9)
    before = backend.commits
    old.tick()                              # the stale VM reads a head of epoch 2 and stops for good
    assert old.s["pt"]["fenced"]["head_epoch"] == 2 and backend.commits == before
    assert any("FENCED" in ln and "epoch 2" in ln for ln in logs) and not any("FENCED" in ln for ln in logs2)
    S.make_gen(str(chat2 / "pt"), 500)
    new.tick()
    assert store.steps("ckpt-N")[-1] == 500
    for s in (400, 500):                    # everything above the stale VM's last push is the replacement's
        man = fresh(tmp_path).read_manifest("ckpt-N", f"gens/g{s}")
        assert (man["epoch"], man["boot_id"]) == (2, new.boot)
    assert (tmp_path / "repo" / "ckpt-N" / "gens" / "g500" / "model.safetensors").read_bytes() == S.state_at(500)[0]
    # A step the stale VM wrote first is not written twice, and does not fence either: the replacement goes past it.
    chat3, backend3, store3, old3, logs3, _ = setup(tmp_path / "same_step")
    S.make_gen(str(chat3 / "pt"), 100)
    old3.tick()
    chat4 = tmp_path / "same_step" / "vm2" / "chat"
    (chat4 / "pt").mkdir(parents=True)
    new3 = cc.Courier(hf_store.Store(backend3, log=logs3.append, delay=0), str(chat4), JOBS, BR, log=logs3.append,
                      cfg={**CFG, "epoch": 2})
    S.make_gen(str(chat4 / "pt"), 100)
    new3.tick()
    S.make_gen(str(chat3 / "pt"), 300, seed=9)
    old3.tick()
    S.make_gen(str(chat4 / "pt"), 300)
    new3.tick()
    assert new3.s["pt"]["fenced"] is None and any("g300 is already in the store; not written again" in ln for ln in logs3)
    S.make_gen(str(chat4 / "pt"), 400)
    new3.tick()
    assert fresh(tmp_path / "same_step").read_manifest("ckpt-N", "gens/g400")["epoch"] == 2
    # The boot rule still holds between writers of ONE epoch (tests above), and an epoch-less head (a generation
    # written by hand) falls under it too.
    assert new.foreign("pt", {"step": 900, "epoch": None, "boot_id": "someone"}) is not None
    assert new.foreign("pt", {"step": 900, "epoch": 1, "boot_id": "someone"}) is None
    assert new.foreign("pt", {"step": 900, "epoch": 3, "boot_id": "someone"}) is not None


def test_a_commit_that_loses_the_race_is_refused_by_the_store(tmp_path, monkeypatch):
    """The head is read, then another writer commits before this push lands: the conditional commit fails, nothing
    is written, and the next pass reads the head again (and is fenced by what it finds)."""
    chat, backend, store, c, logs, clock = setup(tmp_path)
    S.make_gen(str(chat / "pt"), 100)
    c.tick()
    S.make_gen(str(chat / "pt"), 200)
    real = store.push_generation
    other = tmp_path / "other"

    def late(branch, job, step, folder, identity=None, parent=None, prune=None):
        S.make_gen(str(other), 150, seed=3)  # the other writer's commit lands between the read and the push
        fresh(tmp_path).push_generation(branch, job, 150, str(other / "ckpt_gen" / "g150"), {"epoch": 1, "boot_id": "other"})
        return real(branch, job, step, folder, identity, parent, prune=prune)

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
    assert store.steps("ckpt-N") == [400, 500, 600] and store.mark_steps("pt") == [100, 200]
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
    assert store.steps("ckpt-N") == [10, 20, 30]   # three: the one a commit adds never counts in that commit
    S.make_gen(out, 100)
    c.tick()
    assert store.steps("ckpt-N") == [10, 20, 30, 100]  # nothing verified here when g100 went up
    S.make_gen(out, 150)
    c.tick()
    assert store.steps("ckpt-N") == [10, 20, 30, 100, 150]  # one generation verified here (g100, by its listing)
    S.make_gen(out, 180)
    c.tick()                                        # two verified here, g100 and g150: what is behind g100 goes
    assert store.steps("ckpt-N") == [100, 150, 180] and c.s["pt"]["verified"] == {100, 150, 180}
    # Step 200 is a mark step and main refuses commits: the mark stays staged, the generation still goes up, and
    # from then on nothing at or above 200 is pruned, so the mark's weights stay recoverable from the store.
    backend.fail_marks = True
    S.make_gen(out, 200)
    c.tick()
    assert c.staged_marks("pt") == [200] and store.mark_steps("pt") == [] and store.steps("ckpt-N") == [150, 180, 200]
    for step in (300, 400, 500):
        clock.t += 3600
        S.make_gen(out, step)
        c.tick()
    assert store.steps("ckpt-N") == [200, 300, 400, 500]   # without the mark's hold g200 would have gone with g500
    assert json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]["staged_marks"] == [200]
    backend.fail_marks = False
    clock.t += 3600
    S.make_gen(out, 600)
    c.tick()
    assert store.mark_steps("pt") == [200] and store.steps("ckpt-N") == [400, 500, 600]
    # A restore's download counts as a verification on the VM that made it, and only there.
    chat2 = tmp_path / "vm2" / "chat"
    chat2.mkdir(parents=True)
    cfg2 = {**CFG, "epoch": 2, "min_step": {"pt": 600}, "first_start": []}
    cc.restore(fresh(tmp_path), str(chat2), JOBS, BR, log=lambda m: None, cfg=cfg2)
    c2 = cc.Courier(fresh(tmp_path), str(chat2), JOBS, BR, log=lambda m: None, cfg=cfg2)
    assert c2.s["pt"]["verified"] == {600}
    S.make_gen(str(chat2 / "pt"), 700)
    c2.tick()
    assert fresh(tmp_path).steps("ckpt-N") == [400, 500, 600, 700]   # one verified on this VM: nothing removed yet
    S.make_gen(str(chat2 / "pt"), 800)
    c2.tick()
    assert fresh(tmp_path).steps("ckpt-N") == [600, 700, 800] and c2.s["pt"]["verified"] == {600, 700, 800}


def test_a_save_is_one_commit_and_the_courier_never_releases(tmp_path):
    """The hub limits commit creation per repository (128 an hour; the rehearsal of 12 Nov 2026 met it). A save
    was a push, a prune, and with release on a lock and an unlock: up to five commits. It is one: the commit that
    adds the generation removes the pruned ones. No VM releases storage, whatever its config says (the keys an
    older runner sent are ignored): release is the runner's, from home."""
    chat, backend, store, c, logs, _ = setup(tmp_path, cfg={**CFG, "reclaim": True, "release_grace_s": 60})
    asked = []
    store.release_report = lambda *a, **kw: asked.append("release") or {"held": False, "objects": 0, "bytes": 0}
    store.prune = lambda *a, **kw: asked.append("prune") or []
    for step in (100, 300, 400, 500):
        S.make_gen(str(chat / "pt"), step)
        c.tick()
        c.tick()
    assert store.steps("ckpt-N") == [300, 400, 500] and asked == [] and backend.commits == 4
    sent = [json.loads(ln) for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines()]
    assert [x["message"] for x in sent] == [f"generation: gens/g{s}" for s in (100, 300, 400, 500)]
    # The commit that adds g500 removes g100, behind g300 and g400, each verified by a listing after its own commit;
    # g400 did not count in the commit that added it. The same commit writes g100's ledger.
    assert [x["deletes"] for x in sent] == [[], [], [], ["gens/g100/"]] and {x["branch"] for x in sent} == {"ckpt-N"}
    assert "pruned/g100.json" in sent[-1]["adds"] and not any("pruned/" in a for x in sent[:-1] for a in x["adds"])
    ledger = json.loads((tmp_path / "repo" / "ckpt-N" / "pruned" / "g100.json").read_text())
    m100, o100 = S.state_at(100)
    assert ledger["step"] == 100 and ledger["format"] == hf_store.PRUNED_FORMAT and abs(ledger["at"] - time.time()) < 60
    assert {S.sha(m100), S.sha(o100)} <= {o["sha256"] for o in ledger["objects"]}
    assert [p["pruned"] for p in stats(logs, "push")] == [[], [], [], [100]]
    assert any("generations [100] removed from the store in the same commit" in ln for ln in logs)
    st = json.loads((chat / "courier_status.json").read_text())
    assert "releases" not in st and st["commit_budget_per_hour"] == 0 and st["jobs"]["pt"]["verified"] == [100, 300, 400, 500]
    src = (CC / "ckpt_courier.py").read_text()
    assert "release_report(" not in src and ".release(" not in src and ".prune(" not in src
    assert "A VM never releases storage" in " ".join(cc.__doc__.split())


def stats(logs, what):
    return [json.loads(ln[5:]) for ln in logs if ln.startswith("stat ") and json.loads(ln[5:])["what"] == what]


def test_the_log_carries_what_each_transfer_cost_and_the_push_cap_stops_the_courier(tmp_path):
    """courier.log is what comes home from a stopped VM. It holds, as `stat` lines: every push with its bytes and
    seconds, every measurement taken over from a release at home, and a count of failures that name HTTP 429. The
    push cap: counted are what the store held for the campaign at the last release from home and the bytes pushed
    since, across sessions; at the cap the courier stops pushing, says why, and leaves a marker. A release never
    sets the count to zero: it replaces it by what the store was measured to hold."""
    gen = sum(len(b) for b in S.state_at(100))                         # model and optimiser; state and manifest on top
    CAP = 6500 + 2 * gen                                               # reached by the second push, not by the first
    cfg = {**CFG, "push_cap_bytes": CAP, "pushed_bytes_start": 6500}
    chat, backend, store, c, logs, clock = setup(tmp_path, cfg=cfg)
    assert c.since_release == 6500 and c.cap == CAP                    # what the sessions before this one had counted
    out = str(chat / "pt")
    S.make_gen(out, 100)
    c.tick()
    S.make_gen(out, 300)
    c.tick()
    pushes = stats(logs, "push")
    assert [(p["kind"], p["step"]) for p in pushes] == [("generation", 100), ("generation", 300)]
    assert all(p["bytes"] > 6000 and p["seconds"] >= 0 and p["job"] == "pt" for p in pushes)
    assert pushes[-1]["bytes_since_release"] == c.since_release == 6500 + sum(p["bytes"] for p in pushes) >= c.cap
    assert gen < pushes[0]["bytes"] < gen + 1000
    # At the cap: the next generation is not pushed, the status and a marker say why, nothing is retried.
    S.make_gen(out, 400)
    before = backend.commits
    c.tick()
    c.tick()
    st = json.loads((chat / "courier_status.json").read_text())
    assert backend.commits == before and store.steps("ckpt-N") == [100, 300]
    assert st["capped"]["cap"] == CAP and st["capped"]["bytes_since_release"] == c.since_release
    assert "at the last release from home, and what was pushed since" in st["capped"]["why"] and st["push_cap_bytes"] == CAP
    assert json.loads((chat / "COURIER_CAPPED").read_text().splitlines()[0])["cap"] == CAP
    assert len((chat / "COURIER_CAPPED").read_text().splitlines()) == 1 and sum("PUSH CAP" in ln for ln in logs) == 1
    assert c.idle() is False                                           # its job has not ended
    (chat / "STATUS_pt").write_text("FINISHED\n")
    assert c.idle() is True                                            # a capped courier holds nothing open
    # A courier started again on the same boot (the daemon died) is still capped and keeps its count.
    again = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=cfg)
    assert again.capped and again.since_release == c.since_release
    # Without a release from home nothing is ever taken off the count: it is every byte this campaign has pushed.
    chat3, backend3, store3, c3, logs3, _ = setup(tmp_path / "c", cfg={**cfg, "pushed_bytes_start": 0, "push_cap_bytes": 10 ** 9})
    for step in (100, 300, 400, 500):
        S.make_gen(str(chat3 / "pt"), step)
        c3.tick()
    assert c3.since_release == c3.pushed == sum(p["bytes"] for p in stats(logs3, "push")) > 4 * 6000

    # The real sequence of the rehearsal (12 Nov 2026). Session one pushes; its count goes home; the VM is lost.
    # Session two's courier starts from that count and counts on. A release from home then frees part of what was
    # pushed and hands over what the store still holds. The count becomes that plus what was pushed since: never
    # zero while the store holds generations. (It read 0 in state.json during the second session: the VM's own
    # release freed 0.56 GB of 3.6 GB pushed and the count started again from nothing.)
    chat1, _, store1, c1, logs1, _ = setup(tmp_path / "s1", cfg={**CFG, "push_cap_bytes": 10 ** 9})
    for step in (100, 300):
        S.make_gen(str(chat1 / "pt"), step)
        c1.tick()
    carried = json.loads((chat1 / "courier_status.json").read_text())["bytes_since_release"]
    assert carried == c1.since_release == sum(p["bytes"] for p in stats(logs1, "push")) > 2 * gen
    cfg2 = {**CFG, "epoch": 2, "push_cap_bytes": carried + 5 * gen, "pushed_bytes_start": carried, "first_start": []}
    chat2 = tmp_path / "s2" / "chat"
    (chat2 / "pt").mkdir(parents=True)
    logs2 = []
    c2 = cc.Courier(store1, str(chat2), JOBS, BR, log=logs2.append, cfg=cfg2)       # a new VM, the same store
    assert c2.since_release == carried and c2.pushed == 0
    S.make_gen(str(chat2 / "pt"), 400)
    c2.tick()
    p400 = stats(logs2, "push")[-1]["bytes"]
    assert c2.since_release == carried + p400 == stats(logs2, "push")[-1]["bytes_since_release"]

    def measure(seq, held, at_pushed, boot=None):
        (chat2 / "store_measure.json").write_text(json.dumps({"seq": seq, "boot_id": boot or c2.boot, "bytes": held,
                                                             "at_pushed": at_pushed, "at": time.time()}))

    held = 2 * gen + 700                                               # what the branch names after the release
    measure(1, held, at_pushed=c2.pushed)
    S.make_gen(str(chat2 / "pt"), 500)
    c2.tick()
    p500 = stats(logs2, "push")[-1]["bytes"]
    assert c2.since_release == held + p500 > 0 and c2.measure_seq == 1
    m = stats(logs2, "measure")
    assert len(m) == 1 and m[0]["store_bytes"] == held and m[0]["counted_before"] == carried + p400
    assert m[0]["bytes_since_release"] == held                         # taken over before the push of that tick
    st2 = json.loads((chat2 / "courier_status.json").read_text())
    assert st2["bytes_since_release"] == held + p500 and st2["pushed_bytes"] == p400 + p500 and st2["measure_seq"] == 1
    # A measurement of another boot, or one already taken, changes nothing; a courier started again on the same
    # boot keeps the count; what was pushed between the runner's read and its measurement is counted twice.
    measure(2, 1, at_pushed=0, boot="another-boot")
    c2.tick()
    measure(1, 1, at_pushed=0)
    c2.tick()
    assert c2.since_release == held + p500 and c2.measure_seq == 1
    again2 = cc.Courier(store1, str(chat2), JOBS, BR, log=logs2.append, cfg=cfg2)
    assert again2.since_release == held + p500 and again2.measure_seq == 1 and again2.pushed == p400 + p500
    measure(2, held, at_pushed=p400)                                   # the runner had read the count before g500
    again2.tick()
    assert again2.since_release == held + p500 and again2.capped is None
    # With measurements the cap is not reached by a long run whose store stays small; without them it is.
    for step in (600, 700, 800, 900, 1000, 1100):
        S.make_gen(str(chat2 / "pt"), step)
        measure(step, held, at_pushed=again2.pushed)
        again2.tick()
    assert again2.capped is None and store1.steps("ckpt-N") == [900, 1000, 1100] and again2.since_release < held + 2 * gen
    assert sum(p["bytes"] for p in stats(logs2, "push")) > 5 * gen     # far more than the cap's room was pushed
    # HTTP 429: counted when a failure names it, and the count is in the log and the status.
    chat4, backend4, store4, c4, logs4, clock4 = setup(tmp_path / "d")
    S.make_gen(str(chat4 / "pt"), 100)
    store4.push_generation = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("429 Client Error: Too Many Requests"))
    c4.tick()
    clock4.t += 31
    c4.tick()
    assert [s["count"] for s in stats(logs4, "http_429")] == [1, 2]
    assert json.loads((chat4 / "courier_status.json").read_text())["http_429"] == 2


def test_no_store_call_is_started_after_sigterm(tmp_path):
    """The runner ends the courier before it removes the token and stops the VM. From the signal on, the courier
    starts no store call: the pass in hand ends where it is, and nothing new holds the token."""
    chat, backend, store, c, logs, _ = setup(tmp_path, push_timeout=30)
    S.make_gen(str(chat / "pt"), 100)
    c.tick()
    assert store.steps("ckpt-N") == [100]
    S.make_gen(str(chat / "pt"), 300)
    calls = []
    real_call = cc.bounded

    def counting(fn, timeout, token=None, while_waiting=None):
        calls.append(1)
        if len(calls) == 1:
            c.stopping = True          # SIGTERM arrives while the first call of this pass is in flight
        return real_call(fn, timeout, token, while_waiting=while_waiting)

    cc.bounded = counting
    try:
        c.tick()
        c.tick()
    finally:
        cc.bounded = real_call
    assert len(calls) == 1 and store.steps("ckpt-N") == [100]          # the head read ran; the push never started
    assert c.s["pt"]["failures"] == 0 and c.s["pt"]["last_error"] is None   # a stop is not a failure of the store
    with pytest.raises(cc.Stopping):
        c.call(lambda: 1)
    # The daemon itself: a SIGTERM during a slow push ends the push's child and the daemon, and no later call starts.
    root = tmp_path / "daemon"
    chat2 = root / "chat"
    (chat2 / "pt").mkdir(parents=True)
    cfg = root / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{root / 'repo'}", "branches": BR, **CFG}))
    jobs = root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": list(JOBS.values())}))
    S.make_gen(str(chat2 / "pt"), 100)
    slow = root / "slow.py"
    slow.write_text("import sys, time\n"
                    f"sys.path.insert(0, {str(CC)!r})\n"
                    "import hf_store, ckpt_courier\n"
                    "real = hf_store.Store.push_generation\n"
                    "def push(self, *a, **kw):\n"
                    f"    open({str(root / 'pushing')!r}, 'a').write('x')\n"
                    "    time.sleep(120)\n"
                    "hf_store.Store.push_generation = push\n"
                    "sys.exit(ckpt_courier.main(sys.argv[1:]))\n")
    p = subprocess.Popen([sys.executable, str(slow), "run", "--config", str(cfg), "--jobs", str(jobs), "--chat", str(chat2),
                          "--tick", "0.1", "--push-timeout", "100"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(lambda: (root / "pushing").exists(), "the slow push")
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=20) == 0
    finally:
        if p.poll() is None:
            p.kill()
    assert (root / "pushing").read_text() == "x"                        # one push was started, and none after the signal
    assert not (chat2 / "courier.pid").exists()


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


def hub_courier(tmp_path, per_hour, push_timeout=None, cfg=CFG, essential_wait=hf_store.ESSENTIAL_WAIT_S):
    """A courier on the real HfBackend over the stand-in hub (tests/hub_standin.py), with a commit budget, one
    clock for the courier, the budget and the hub, and a trainer that saves when the test says."""
    import hub_standin
    root, chat = tmp_path / "hub", tmp_path / "chat"
    (chat / "pt").mkdir(parents=True)
    tf = tmp_path / "hub.token"
    tf.write_text(TOKEN + "\n")
    tf.chmod(0o600)
    logs, clock = [], Clock()
    budget = hf_store.CommitBudget(per_hour, str(chat / "store_commits.json"), clock=clock, essential_wait=essential_wait)
    store = hf_store.open_store({"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tf)},
                                api=hub_standin.DiskApi(str(root)), log=logs.append, delay=0, budget=budget, clock=clock)
    c = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=cfg, push_timeout=push_timeout)

    def at(t):
        clock.t = t
        (root / "now.txt").write_text(str(t))

    at(clock.t)
    return chat, root, store, c, logs, clock, at


def hub_log(root):
    p = root / "commit_log.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def hub_fresh(root):
    import hub_standin
    return hf_store.Store(hf_store.HfBackend("someone/base2-store", "dataset", TOKEN, api=hub_standin.DiskApi(str(root))),
                          log=lambda m: None, delay=0)


def test_a_courier_under_the_hubs_commit_limit_waits_out_the_429_and_loses_no_generation(tmp_path):
    """Part a of the rehearsal, 12 Nov 2026: another writer on the repo (the watcher of then) had used the hub's
    hourly limit, and the courier's commits were refused with HTTP 429 for 25 minutes. Here the hub refuses with
    the library's own error and a Retry-After. The courier sends nothing more until that wait is over (it used to
    try again after 60, 120, 240 and 480 s, each a refused commit), training goes on, and then the mark and the
    NEWEST generation go up and resume. Every store call runs in a child, as on a VM. The hub counts a branch's
    creation as a commit, and so does the courier's budget."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=30, push_timeout=60)
    t0 = clock.t
    (root / "limit.json").write_text(json.dumps({"per_window": 7, "retry_after": "auto"}))
    other = hub_fresh(root)
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "watch.json").write_text("{}")
    for i in range(4):                                   # the other writer takes five of the hour's seven: its
        at(t0 - 1500 + i)                                # branch and four commits, 25 minutes ago
        other.push_set("base2-build", f"watch/{i}", str(watch))
    out = str(chat / "pt")
    at(t0 + 10)
    S.make_gen(out, 100)
    c.tick()                                             # the courier's branch and g100: the sixth and the seventh
    assert store.steps("ckpt-N") == [100] and [x["ok"] for x in hub_log(root)] == [True] * 7
    assert store.budget.count() == 2 and c.s["pt"]["verified"] == {100}
    at(t0 + 510)
    S.make_gen(out, 200)                                 # a mark step: the mark goes first, and is the one refused
    c.tick()
    n429 = stats(logs, "http_429")
    assert len(n429) == 1 and n429[0]["count"] == 1
    refused = [x for x in hub_log(root) if not x["ok"]]
    assert len(refused) == 1 and store.steps("ckpt-N") == [100] and store.mark_steps("pt") == []
    wait = 3600 - 1500 - 510                             # the hub's Retry-After: until its hour has room
    assert store.budget.wait_s() == wait and any(f"retry in {wait} s" in ln for ln in logs)
    assert c.staged_marks("pt") == [200]                 # staged: the rung job may drop its copy
    for k, step in enumerate((300, 400, 500), start=2):  # training goes on; nothing is sent
        at(t0 + 10 + 500 * k)
        S.make_gen(out, step)
        c.tick()
        c.tick()
    assert len(hub_log(root)) == 8 and len(stats(logs, "http_429")) == 1    # not one more refused commit
    st = json.loads((chat / "courier_status.json").read_text())
    assert st["http_429"] == 1 and st["jobs"]["pt"]["live_step"] == 500 and st["jobs"]["pt"]["pushed_step"] == 100
    assert st["jobs"]["pt"]["failures"] == 0             # a wait for room is not a failure of the generation's push
    at(t0 + 510 + wait + 1)                              # the hub's hour has room again
    S.make_gen(out, 600)
    c.tick()
    assert store.mark_steps("pt") == [200] and store.steps("ckpt-N") == [100, 600]   # the mark, and the newest
    assert c.staged_marks("pt") == [] and all(x["ok"] for x in hub_log(root)[8:])
    for k, step in enumerate((700, 800), start=1):
        at(t0 + 510 + wait + 1 + 500 * k)
        S.make_gen(out, step)
        c.tick()
    assert store.steps("ckpt-N") == [600, 700, 800] and hub_log(root)[-1]["deletes"] == ["gens/g100/"]
    assert hub_fresh(root).pull_generation("ckpt-N", str(tmp_path / "got"), expect=cc.identity(CFG, str(chat), JOBS["pt"])) == (800, [])
    hub_fresh(root).pull_mark("pt", 200, str(tmp_path / "mark"))
    assert (tmp_path / "mark" / "model.safetensors").read_bytes() == S.state_at(200)[0]
    # The courier's own calls on the hub: its branch, five commits the hub took, and the one it refused.
    mine = [x for x in hub_log(root) if x["branch"] != "base2-build"]
    assert [(x["message"], x["ok"]) for x in mine] == [
        ("create_branch", True), ("generation: gens/g100", True), ("mark: marks/pt/rung_200", False),
        ("mark: marks/pt/rung_200", True), ("generation: gens/g600", True), ("generation: gens/g700", True),
        ("generation: gens/g800", True)]
    assert not any(TOKEN in ln for ln in logs)
    assert sum(ln.startswith("create_branch ckpt-N") for ln in (root / "calls.log").read_text().splitlines()) == 1


def test_a_branch_creation_is_counted_and_its_429_is_read_and_a_long_retry_after_is_capped(tmp_path):
    """The rehearsal's logs show the hub refusing /branch/ with the commit limit's 429. The courier's first contact
    creates its branch through the budget; the 429 is counted, read and waited out like a commit's, and a later
    courier process on the same VM does not ask again. A Retry-After of hours is honoured for 30 minutes."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=30)
    t0 = clock.t
    (root / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": 86400}))
    S.make_gen(str(chat / "pt"), 100)
    c.tick()
    assert [s["count"] for s in stats(logs, "http_429")] == [1] and not (root / "branches" / "ckpt-N").exists()
    assert any("store not read" in ln and "the creation of ckpt-N" in ln for ln in logs)
    assert store.budget.wait_s() == hf_store.MAX_429_WAIT_S == 1800 and store.budget.count() == 1
    assert any("retry in 1800 s" in ln for ln in logs) and [x["message"] for x in hub_log(root)] == ["create_branch"]
    at(t0 + 1700)
    c.tick()
    assert len(hub_log(root)) == 1                       # nothing is asked inside the wait
    (root / "limit.json").unlink()
    at(t0 + 1801)
    c.tick()
    assert store.steps("ckpt-N") == [100] and [x["message"] for x in hub_log(root)][1:] == ["create_branch", "generation: gens/g100"]
    assert store.budget.count() == 3                     # the refused creation, the creation, the commit
    # A second process of this VM (the courier started again; the data job's separate calls) asks for no branch.
    import hub_standin
    budget2 = hf_store.CommitBudget(30, str(chat / "store_commits.json"), clock=clock)
    again = hf_store.open_store({"repo": "someone/base2-store", "token_file": str(tmp_path / "hub.token")},
                                api=hub_standin.DiskApi(str(root)), log=logs.append, delay=0, budget=budget2)
    again.ensure_branch("ckpt-N")
    c2 = cc.Courier(again, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=CFG)
    S.make_gen(str(chat / "pt"), 300)
    c2.tick()
    assert again.steps("ckpt-N") == [100, 300]
    assert sum(ln.startswith("create_branch") for ln in (root / "calls.log").read_text().splitlines()) == 2
    assert json.loads((chat / "store_commits.json.branches").read_text()) == {"repo": "someone/base2-store", "branches": ["ckpt-N"]}


def test_a_generation_that_arrives_damaged_never_counts_and_two_whole_ones_always_remain(tmp_path):
    """The first review of the commit budget (12 Nov 2026): the commit that added a generation counted it as
    verified at once and removed down to the one before it; a generation that arrived short then left ONE whole
    generation in the store. Now a generation counts only after a listing taken after its own commit shows it
    whole, and the commit that adds it never relies on it. Here g300's weights arrive short on the hub in g300's
    own commit. Whatever is pushed after, two whole generations that this VM verified are in the store at every
    moment, and a restore skips the bad one."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100)
    out = str(chat / "pt")
    ident = cc.identity(CFG, str(chat), JOBS["pt"])

    def whole():
        fresh_store = hub_fresh(root)
        return [s for s in fresh_store.steps("ckpt-N") if fresh_store.verify_by_download("ckpt-N", s, expect=ident)]

    for step in (100, 200):
        S.make_gen(out, step)
        c.tick()
    assert c.s["pt"]["verified"] == {100, 200} and whole() == [100, 200]
    (root / "damage.json").write_text(json.dumps({"path": "gens/g300/model.safetensors", "bytes": 100}))
    S.make_gen(out, 300)
    c.tick()
    # The commit was accepted and the generation is listed, short. It is not whole, and is said so.
    assert store.steps("ckpt-N") == [100, 200, 300] and (root / "branches" / "ckpt-N" / "gens" / "g300" / "model.safetensors").stat().st_size == 100
    assert c.s["pt"]["verified"] == {100, 200} and c.s["pt"]["not_whole"] == {300} and whole() == [100, 200]
    assert stats(logs, "verify")[-1] == {"what": "verify", "job": "pt", "step": 300, "whole": False}
    assert any("g300 is NOT WHOLE in the store after its own commit" in ln for ln in logs)
    assert hub_log(root)[-1]["deletes"] == []            # its commit removed nothing: g100 and g200 both stand
    step, rejected = hub_fresh(root).pull_generation("ckpt-N", str(tmp_path / "r1"), expect=ident)
    assert step == 200 and [s for s, _ in rejected] == [300]      # a restore gives the bad one up for the one before
    # The pushes after it: nothing is ever removed on g300's account.
    seen = []
    for step in (400, 500, 600):
        S.make_gen(out, step)
        c.tick()
        seen.append((store.steps("ckpt-N"), whole()))
        assert len(whole()) >= 2 and len([s for s in whole() if s in c.s["pt"]["verified"]]) >= 2
    assert seen == [([100, 200, 300, 400], [100, 200, 400]),      # two verified before g400: g100, g200; nothing below g100
                    ([200, 300, 400, 500], [200, 400, 500]),      # g200 and g400: g100 goes
                    ([400, 500, 600], [400, 500, 600])]           # g400 and g500: g200 and the bad g300 go
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["not_whole"] == [300] and st["verified"] == [100, 200, 400, 500, 600] and st["failures"] == 0
    # A listing that cannot be read leaves the generation uncounted until it is read: asked again before the next push.
    real = store.listed_whole
    store.listed_whole = lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("hub unreachable"))
    S.make_gen(out, 700)
    c.tick()
    assert store.steps("ckpt-N") == [500, 600, 700] and 700 not in c.s["pt"]["verified"] and list(c.s["pt"]["to_verify"]) == [700]
    store.listed_whole = real
    at(clock.t + 31)
    S.make_gen(out, 800)
    c.tick()
    assert {700, 800} <= c.s["pt"]["verified"] and c.s["pt"]["to_verify"] == {} and store.steps("ckpt-N") == [600, 700, 800]


def test_the_last_generations_check_is_asked_again_until_it_is_read_and_a_damaged_one_is_named(tmp_path):
    """The second reviews (12 Nov 2026). (1) A check that could not be read is not a verdict. Here the listing
    after g300's commit is good and its manifest cannot be fetched (the path the earlier test went round: it made
    the whole check raise). The step waits in to_verify; it is not "not whole". (2) g300 is the run's LAST
    generation: nothing new is ever pushed after it. The courier asks again on every pass all the same, does not
    end while the check is unread, and ends once it reads whole. (3) A last generation that arrives damaged, or
    whose object is gone though the listing shows its pointer, is named: status, log and CHAT/COURIER_NOT_WHOLE,
    which the runner's own check turns into a halt. Every store call runs in a child, as on a VM."""
    pytest.importorskip("huggingface_hub")
    import requests
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100, push_timeout=60)
    out = str(chat / "pt")
    for step in (100, 200):
        S.make_gen(out, step)
        c.tick()
    api, real = store.b.api, store.b.api.hf_hub_download

    class Asked:                                         # counted in a file: the calls are made in children
        def __len__(self):
            return len(read(tmp_path / "asked"))

    asked = Asked()

    def no_manifest_of_g300(**kw):
        if kw["filename"] == "gens/g300/manifest.json":
            with open(tmp_path / "asked", "a") as f:
                f.write("x")
            raise requests.ConnectionError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out.")
        return real(**kw)

    api.hf_hub_download = no_manifest_of_g300
    S.make_gen(out, 300)
    (chat / "STATUS_pt").write_text("FINISHED\n")        # the run has ended: g300 is its last generation
    c.tick()
    s = c.s["pt"]
    assert store.steps("ckpt-N") == [100, 200, 300] and s["head"] == 300 and len(asked) > 0
    assert list(s["to_verify"]) == [300] and s["not_whole"] == set() and s["verified"] == {100, 200}
    assert not (chat / "COURIER_NOT_WHOLE").exists() and not any("NOT WHOLE" in ln for ln in logs)
    assert any("its check could not be read" in ln for ln in logs) and not stats(logs, "verify")[2:]
    assert c.idle() is False                             # ended, pushed, and not left: the check is still owed
    n = len(asked)
    c.tick()                                             # within 30 s: not asked again yet
    assert len(asked) == n
    at(clock.t + 31)
    c.tick()                                             # unread a second time: still no verdict, still not idle
    assert len(asked) > n and list(s["to_verify"]) == [300] and c.idle() is False and s["not_whole"] == set()
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["to_verify"] == [300] and st["not_whole"] == [] and st["failures"] == 0
    # A courier started again on this boot still owes the check.
    again = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=CFG, push_timeout=60)
    assert list(again.s["pt"]["to_verify"]) == [300] and again.idle() is False
    api.hf_hub_download = real
    at(clock.t + 31)
    c.tick()                                             # read: whole
    assert s["to_verify"] == {} and s["verified"] == {100, 200, 300} and c.idle() is True
    assert stats(logs, "verify")[-1] == {"what": "verify", "job": "pt", "step": 300, "whole": True}
    assert len(hub_log(root)) == 5                       # the branch, the mark and three saves: the checks were reads

    # (3a) The last generation arrives short.
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path / "b", per_hour=100, push_timeout=60)
    out = str(chat / "pt")
    for step in (100, 200):
        S.make_gen(out, step)
        c.tick()
    (root / "damage.json").write_text(json.dumps({"path": "gens/g300/optim.safetensors", "bytes": 7}))
    S.make_gen(out, 300)
    (chat / "STATUS_pt").write_text("FINISHED\n")
    c.tick()
    s = c.s["pt"]
    assert s["not_whole"] == {300} and s["to_verify"] == {} and s["verified"] == {100, 200}
    marker = [json.loads(ln) for ln in (chat / "COURIER_NOT_WHOLE").read_text().splitlines()]
    assert [(m["job"], m["step"], m["last"]) for m in marker] == [("pt", 300, True)]
    assert any("It is the run's last generation: the runner halts the campaign." in ln for ln in logs)
    assert c.idle() is True                              # nothing more this VM can do: written once, not pushed again
    assert hub_fresh(root).pull_generation("ckpt-N", str(tmp_path / "r"))[0] == 200     # the two before it stand
    # (3b) The commit lands whole and its object is gone afterwards: the listing shows the pointer, size and hash;
    # only the byte read sees it.
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path / "c", per_hour=100, push_timeout=60)
    out = str(chat / "pt")
    S.make_gen(out, 100)
    c.tick()
    (root / "lose.json").write_text(json.dumps({"path": "gens/g200/optim.safetensors"}))
    S.make_gen(out, 200)
    c.tick()
    s = c.s["pt"]
    assert store.whole_in_listing("ckpt-N", 200, store.b.tree("ckpt-N")) is True       # what the listing alone says
    assert s["not_whole"] == {200} and s["verified"] == {100}
    assert json.loads((chat / "COURIER_NOT_WHOLE").read_text().splitlines()[0])["last"] is False
    assert any("probe 404 from the object store: gens/g200/optim.safetensors" in ln for ln in (root / "calls.log").read_text().splitlines())
    seen = []
    for step in (300, 400, 500, 600):                    # training goes on: g200 never counts for a removal
        S.make_gen(out, step)
        c.tick()
        seen.append(store.steps("ckpt-N"))
    assert seen == [[100, 200, 300], [100, 200, 300, 400],     # behind g100 alone, then behind g100 and g300: nothing
                    [300, 400, 500], [400, 500, 600]]          # behind g300 and g400; behind g400 and g500
    assert s["verified"] == {100, 300, 400, 500, 600} and s["not_whole"] == {200}


def test_a_check_that_stays_unread_is_reported_apart_with_its_http_status(tmp_path, monkeypatch):
    """The third reviews. The byte read of a pushed generation is redirected to the object store, which answers
    403: what one does for a key it will not show, and for a signed address it refuses. That is no verdict, so the
    generation stays unread; but it read "unread" for ever, and the courier's log said only "(StoreError)". Now the
    first line names the HTTP status, and a check that stays unread UNREAD_REPORT_AFTER times running for the same
    generation is reported apart: log, stat line, status and CHAT/COURIER_CHECK_UNREAD. It is never called not
    whole, and it counts once it is read. Every store call runs in a child, as on a VM."""
    pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(cc, "UNREAD_REPORT_AFTER", 3)
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100, push_timeout=60)
    out = str(chat / "pt")
    (root / "probe.json").write_text(json.dumps({"status": 403}))
    S.make_gen(out, 100)
    c.tick()
    s = c.s["pt"]
    first = [ln for ln in logs if "its check could not be read" in ln]
    assert first == ["pt: g100 is pushed; its check could not be read (StoreError, HTTP 403 from the object store); it does not count yet and is asked again"]
    assert list(s["to_verify"]) == [100] and s["unread"][100]["count"] == 1 and not stats(logs, "check_unread")
    at(clock.t + 31)
    c.tick()
    assert s["unread"][100]["count"] == 2 and not (chat / "COURIER_CHECK_UNREAD").exists()
    at(clock.t + 31)
    c.tick()                                             # the third running: reported apart
    assert stats(logs, "check_unread") == [{"what": "check_unread", "job": "pt", "step": 100, "count": 3, "http_status": 403,
                                            "error": "StoreError, HTTP 403 from the object store"}]
    marker = [json.loads(ln) for ln in (chat / "COURIER_CHECK_UNREAD").read_text().splitlines()]
    assert [(m["job"], m["step"], m["count"], m["http_status"]) for m in marker] == [("pt", 100, 3, 403)]
    loud = [ln for ln in logs if "CHECK UNREAD" in ln]
    assert len(loud) == 1 and "g100 could not be read 3 times running (StoreError, HTTP 403 from the object store)" in loud[0]
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["check_unread"]["100"]["count"] == 3 and st["check_unread"]["100"]["http_status"] == 403
    assert st["to_verify"] == [100] and st["not_whole"] == [] and st["verified"] == []
    assert not (chat / "COURIER_NOT_WHOLE").exists() and not any("NOT WHOLE" in ln for ln in logs)   # no verdict
    (chat / "STATUS_pt").write_text("FINISHED\n")
    assert c.idle() is False                             # still owed
    at(clock.t + 31)
    c.tick()                                             # a fourth: counted, not said again
    assert s["unread"][100]["count"] == 4 and len([ln for ln in logs if "CHECK UNREAD" in ln]) == 1
    for text in logs:
        assert "SIGNED" not in text and TOKEN not in text                            # never the signed address
    (root / "probe.json").unlink()
    at(clock.t + 31)
    c.tick()                                             # read: whole, and the count is gone
    assert s["verified"] == {100} and s["to_verify"] == {} and s["unread"] == {}
    assert json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]["check_unread"] == {}
    # A failure that names no status says so.
    store.listed_whole = lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("hub unreachable"))
    (chat / "STATUS_pt").unlink()
    S.make_gen(out, 200)
    c.tick()
    assert any("g200 is pushed; its check could not be read (StoreError, no HTTP status)" in ln for ln in logs)


def test_a_store_that_stays_unread_costs_a_bounded_number_of_checks_and_is_seen_at_home(tmp_path, monkeypatch):
    """The reviews of the arms' round. With every byte read refused, to_verify grew by one a save and reverify asked
    for every owed step on every pass, before each push: the calls a pass grew without bound over four days. Now a
    pass asks for the REVERIFY_PER_PASS newest and says so once. The status carries what home needs to bound the
    state in time: "unverified_since", the oldest push since which no check of a push read whole. The unread
    counts in the status are taken up by a courier started again on the same boot. A mark's stat line carries
    "ledger_upload" too (None: a mark's commit writes no ledger)."""
    pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(cc, "REVERIFY_PER_PASS", 2)
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100)
    out = str(chat / "pt")
    s = c.s["pt"]
    S.make_gen(out, 100)
    c.tick()
    assert s["verified"] == {100} and s["unverified_since"] is None                       # read whole: nothing is open
    asked = []
    real = store.listed_whole

    def refused(branch, step, files=None, probe=True):
        asked.append(step)
        raise hf_store.StoreError("HTTP 403 on the byte read of ckpt-N:gens/g%d/model.safetensors (from the object store, "
                                  "after the hub's redirect; refused)" % step)

    store.listed_whole = refused
    t0 = None
    for step in (200, 300, 400, 500, 600):
        at(clock.t + 31)
        del asked[:]
        S.make_gen(out, step)
        c.tick()
        t0 = t0 or s["unverified_since"]
        owed_before = [x for x in (200, 300, 400, 500, 600) if x < step]
        assert asked == sorted(owed_before, reverse=True)[:2] + [step]                  # the two newest owed, then its own
    assert sorted(s["to_verify"]) == [200, 300, 400, 500, 600] and s["verified"] == {100}
    assert s["unverified_since"] == t0 and t0 is not None                              # from the first push that was not read
    assert len([ln for ln in logs if "newest are asked in each pass" in ln]) == 1
    assert store.steps("ckpt-N") == [100, 200, 300, 400, 500, 600]                      # nothing verified: nothing removed
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["unverified_since"] == t0 and st["to_verify"] == [200, 300, 400, 500, 600] and st["auth_failure"] is None
    assert st["check_unread"]["600"]["count"] == 1 and st["check_unread"]["500"]["count"] == 2
    marks = [p for p in stats(logs, "push") if p["kind"] == "mark"]
    assert len(marks) == 1 and "ledger_upload" in marks[0] and marks[0]["ledger_upload"] is None
    # A courier started again on this boot goes on from those counts.
    again = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=CFG)
    assert again.s["pt"]["unread"][500]["count"] == 2 and again.s["pt"]["unverified_since"] == t0
    del asked[:]
    at(clock.t + 31)
    again.tick()
    assert asked == [600, 500] and again.s["pt"]["unread"][500]["count"] == 3
    # The store answers again: the newest are read first, the older ones after them, and the state closes.
    store.listed_whole = real
    for _ in range(3):
        at(clock.t + 31)
        again.tick()
    assert again.s["pt"]["to_verify"] == {} and again.s["pt"]["unverified_since"] is None
    assert again.s["pt"]["verified"] >= {200, 300, 400, 500, 600}


def test_the_hubs_refusal_of_the_token_is_named_as_an_authentication_failure(tmp_path):
    """The reviews of the arms' round: a token that expires or is revoked mid-campaign stops every push and every
    check, and the reports read like any other failure, or like missing bytes. A 401 or 403 from the hub (not the
    object store's after a redirect) is named: loudly once, as a stat line, and in the status until a call succeeds."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100, push_timeout=60)
    out = str(chat / "pt")
    S.make_gen(out, 100)
    c.tick()
    (root / "fail.json").write_text(json.dumps({"mode": "auth"}))        # every call: 401 from the hub
    S.make_gen(out, 200)
    at(clock.t + 31)
    c.tick()
    s = c.s["pt"]
    loud = [ln for ln in logs if "AUTHENTICATION FAILURE" in ln]
    assert len(loud) == 1 and "the hub refused this VM's token (HTTP 401)" in loud[0]
    assert s["auth_failure"]["http_status"] == 401 and s["auth_failure"]["count"] >= 1
    assert [a["http_status"] for a in stats(logs, "auth_failure")] == [401]
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["auth_failure"]["http_status"] == 401 and st["pushed_step"] == 100
    at(clock.t + 700)
    c.tick()                                             # still refused: counted, said once
    assert len([ln for ln in logs if "AUTHENTICATION FAILURE" in ln]) == 1 and s["auth_failure"]["count"] >= 2
    (root / "fail.json").unlink()
    at(clock.t + 700)
    c.tick()
    assert s["auth_failure"] is None and s["head"] == 200 and s["verified"] == {100, 200}
    for text in logs:
        assert TOKEN not in text


def test_a_courier_that_died_between_a_commit_and_its_status_has_its_last_push_checked(tmp_path):
    """The second reviews of the arms' round: a courier that dies after its commit has landed and before its
    status is written leaves a generation in the store that is in none of its lists. The courier started again on
    that boot found `step == head`, pushed nothing and checked nothing: the push was never verified, and the timer
    the runner halts on ("unverified_since") was never opened. It is checked at the first contact now, and the
    timer is opened until that check reads whole. A courier with no earlier status on its boot (one the boot
    starts: a new boot id) takes up nothing and checks nothing it did not push or restore."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100)
    out = str(chat / "pt")
    for step in (100, 200):
        S.make_gen(out, step)
        c.tick()
    assert c.s["pt"]["verified"] == {100, 200}
    # The commit of g300 lands; the courier dies before its check and before its status.
    S.make_gen(out, 300)
    ident = {**cc.identity(CFG, str(chat), JOBS["pt"]), "epoch": 1, "boot_id": c.boot}
    store.push_generation("ckpt-N", "pt", 300, c.stage_generation("pt", 300, c.live("pt")[1]), ident, store.head("ckpt-N")["commit"])
    st = json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]
    assert st["pushed_step"] == 200 and st["verified"] == [100, 200] and st["unverified_since"] is None
    real = store.listed_whole
    store.listed_whole = lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("hub unreachable"))
    again = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=CFG)
    assert again.resumed is True
    again.tick()
    s = again.s["pt"]
    assert s["head"] == 300 and list(s["to_verify"]) == [300] and s["unverified_since"] is not None   # owed, and the timer runs
    assert any("g300, the newest generation in the store, has no check on record from this VM" in ln for ln in logs)
    assert json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]["unverified_since"] == s["unverified_since"]
    assert len(hub_log(root)) == 5                       # the branch, the mark, three saves: g300 was not written again
    store.listed_whole = real
    at(clock.t + 31)
    again.tick()
    assert s["verified"] == {100, 200, 300} and s["to_verify"] == {} and s["unverified_since"] is None
    assert stats(logs, "verify")[-1] == {"what": "verify", "job": "pt", "step": 300, "whole": True}
    # A courier of another boot (the boot writes a new boot id on every run) reads no earlier status: it has
    # verified what its restore downloaded, takes nothing else for checked and checks nothing it did not push.
    chat2 = tmp_path / "chat2"
    (chat2 / "pt").mkdir(parents=True)
    (chat2 / "store_restore").mkdir()
    (chat2 / "boot_id").write_text("c" * 32)
    (chat2 / "store_restore" / "pt.json").write_text(json.dumps({"verified_step": 200, "boot_id": "c" * 32}))
    S.make_gen(str(chat2 / "pt"), 300)
    logs2 = []
    c2 = cc.Courier(store, str(chat2), JOBS, BR, log=logs2.append, clock=clock, cfg={**CFG, "epoch": 2})
    c2.tick()
    assert c2.resumed is False and c2.s["pt"]["verified"] == {200} and c2.s["pt"]["to_verify"] == {}
    assert "boot_base2.sh writes a new boot id on every run" in " ".join(cc.__doc__.split())
    # A push that the OBJECT STORE refuses (an upload's signed address) is not the token's failure.
    import requests

    def s3_refuses(**kw):
        r = requests.Response()
        r.status_code, r.url = 403, "https://s3.example/presigned/put/abc?X-Amz-Signature=SECRETSIG"
        e = requests.HTTPError("403 Client Error: Forbidden for url: " + r.url)
        e.response = r
        raise e

    store.b.api.create_commit = s3_refuses
    S.make_gen(out, 400)
    at(clock.t + 31)
    again.tick()
    assert s["auth_failure"] is None and not any("AUTHENTICATION FAILURE" in ln for ln in logs)
    assert any("generation g400 not pushed" in ln and "[refused by the object store, not by the hub: the token is not the cause]" in ln
               and "SECRETSIG" not in ln for ln in logs)


def test_the_upload_first_fallback_is_said_loudly_once_and_in_every_stat_line(tmp_path):
    """The third reviews: a library that does not take preupload_lfs_files as called sends the commit in the old
    order, with the ledger stamped before the upload, and nothing said so. Now every push that wrote a ledger says
    in its stat line which order ran, a fallback has a stat line of its own, and the log says it loudly, once."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100)
    out = str(chat / "pt")
    for step in (100, 200, 300, 400):
        S.make_gen(out, step)
        c.tick()
    gens = [p for p in stats(logs, "push") if p["kind"] == "generation"]
    assert [(p["step"], p["pruned"], p["ledger_upload"]) for p in gens] == [
        (100, [], None), (200, [], None), (300, [], None), (400, [100], "upload_first")]
    assert not stats(logs, "upload_fallback") and not any("UPLOAD-FIRST" in ln for ln in logs)

    def refused(**kw):
        raise TypeError("preupload_lfs_files() got an unexpected keyword argument 'additions'")

    store.b.api.preupload_lfs_files = refused
    for step in (500, 600):
        S.make_gen(out, step)
        c.tick()
    gens = [p for p in stats(logs, "push") if p["kind"] == "generation"][4:]
    assert [(p["step"], p["pruned"]) for p in gens] == [(500, [200]), (600, [300])]
    assert all(p["ledger_upload"].startswith("fallback (TypeError: preupload_lfs_files() got an unexpected") for p in gens)
    fb = stats(logs, "upload_fallback")
    assert [(f["job"], f["step"], f["pruned"]) for f in fb] == [("pt", 500, [200]), ("pt", 600, [300])]
    loud = [ln for ln in logs if "UPLOAD-FIRST DID NOT RUN" in ln]
    assert len(loud) == 1 and "at g500" in loud[0] and "TypeError" in loud[0] and "BEFORE that upload" in loud[0]
    assert store.steps("ckpt-N") == [400, 500, 600] and c.s["pt"]["verified"] >= {400, 500, 600}   # the saves went on


def test_a_step_removed_while_its_check_was_unread_is_never_called_not_whole(tmp_path):
    """The third review: g300's check cannot be read while g400 and g500 verify; g600's commit then removes g300
    with what else is behind g400. The courier went on asking for g300, found it missing from the listing, called
    it NOT WHOLE and wrote CHAT/COURIER_NOT_WHOLE for a generation it had removed itself. A removed generation is
    owed no check: it leaves to_verify in the commit that removes it; and one a restarted courier still carries is
    recognised by the listing (gone, and behind the newest pushed step)."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100)
    out = str(chat / "pt")
    real = store.listed_whole

    def g300_unread(branch, step, files=None, probe=True):
        if step == 300:
            raise ConnectionError("hub unreachable")
        return real(branch, step, files, probe)

    store.listed_whole = g300_unread
    s = c.s["pt"]
    for step in (100, 200, 300, 400, 500):
        S.make_gen(out, step)
        c.tick()
    assert s["verified"] == {100, 200, 400, 500} and list(s["to_verify"]) == [300] and store.steps("ckpt-N") == [200, 300, 400, 500]
    S.make_gen(out, 600)
    c.tick()                                             # behind g400 and g500: g200 and the unread g300 go
    assert hub_log(root)[-1]["deletes"] == ["gens/g200/", "gens/g300/"] and store.steps("ckpt-N") == [400, 500, 600]
    assert s["to_verify"] == {} and s["unread"] == {}
    assert any("g300 was removed in this commit with its check still unread; no check is owed for it" in ln for ln in logs)
    store.listed_whole = real
    for _ in range(2):
        at(clock.t + 31)
        c.tick()
    assert s["not_whole"] == set() and not (chat / "COURIER_NOT_WHOLE").exists() and not any("NOT WHOLE" in ln for ln in logs)
    # A courier started again on this boot carries g200 as owed, from a status written before the removal.
    st = json.loads((chat / "courier_status.json").read_text())
    st["jobs"]["pt"]["to_verify"] = [200]
    (chat / "courier_status.json").write_text(json.dumps(st))
    again = cc.Courier(store, str(chat), JOBS, BR, log=logs.append, clock=clock, cfg=CFG)
    assert list(again.s["pt"]["to_verify"]) == [200]
    again.tick()
    assert again.s["pt"]["to_verify"] == {} and again.s["pt"]["not_whole"] == set() and not (chat / "COURIER_NOT_WHOLE").exists()
    assert any("g200 is no longer in the store (a later save removed it before its check was read)" in ln for ln in logs)
    # A generation that IS listed and differs is still named, as before.
    (root / "damage.json").write_text(json.dumps({"path": "gens/g700/model.safetensors", "bytes": 9}))
    S.make_gen(out, 700)
    again.tick()
    assert again.s["pt"]["not_whole"] == {700} and (chat / "COURIER_NOT_WHOLE").exists()


def test_after_a_session_is_replaced_the_branch_holds_four_and_the_next_save_removes_two_with_their_ledgers(tmp_path):
    """The rerun of 12 Nov 2026 (PROTOCOL 47.2), and intended: a new VM has verified ONE generation, the one its
    restore downloaded, and does not take the lost VM's word for the others. Its first save therefore removes
    nothing and the branch holds four; once that save is read back whole it has two, and its second save removes
    two generations in one commit, each with its own ledger, which a release from home then follows."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=100)
    for step in (100, 200, 300, 400):
        S.make_gen(str(chat / "pt"), step)
        c.tick()
    assert store.steps("ckpt-N") == [200, 300, 400]
    # The session is lost. The new VM: another boot, the next epoch, a restore that downloaded g400.
    chat2 = tmp_path / "chat2"
    (chat2 / "pt").mkdir(parents=True)
    (chat2 / "store_restore").mkdir()
    (chat2 / "boot_id").write_text("b" * 32)
    (chat2 / "store_restore" / "pt.json").write_text(json.dumps({"verified_step": 400, "boot_id": "b" * 32}))
    logs2 = []
    c2 = cc.Courier(store, str(chat2), JOBS, BR, log=logs2.append, clock=clock, cfg={**CFG, "epoch": 2})
    assert c2.s["pt"]["verified"] == {400}
    S.make_gen(str(chat2 / "pt"), 500)
    c2.tick()
    assert store.steps("ckpt-N") == [200, 300, 400, 500]                     # four: one verified generation removes nothing
    assert hub_log(root)[-1]["deletes"] == [] and c2.s["pt"]["verified"] == {400, 500} and not c2.s["pt"]["fenced"]
    S.make_gen(str(chat2 / "pt"), 600)
    c2.tick()
    last = hub_log(root)[-1]
    assert last["message"] == "generation: gens/g600" and last["deletes"] == ["gens/g200/", "gens/g300/"]   # two, in the one commit
    assert {"pruned/g200.json", "pruned/g300.json"} <= set(last["adds"]) and store.steps("ckpt-N") == [400, 500, 600]
    assert [p["pruned"] for p in stats(logs2, "push") if p["kind"] == "generation"] == [[], [200, 300]]
    assert stats(logs2, "push")[-1]["ledger_upload"] == "upload_first"
    led = {s: json.loads((root / "branches" / "ckpt-N" / "pruned" / f"g{s}.json").read_text()) for s in (200, 300)}
    for s, d in led.items():
        m, o = S.state_at(s)
        assert d["step"] == s and d["branch"] == "ckpt-N" and d["at"] == clock.t
        assert sorted((x["path"], x["sha256"], x["bytes"]) for x in d["objects"]) == sorted(
            [(f"gens/g{s}/model.safetensors", S.sha(m), len(m)), (f"gens/g{s}/optim.safetensors", S.sha(o), len(o))])
    # Home follows the ledgers in one round (these two and g100's, from the first session): g200's weights are the
    # mark's and stay; g100's two objects, g200's optimiser and g300's two go.
    home = hub_fresh(root)
    r = home.release_report("home", grace=600, now=clock.t + 10 ** 5)
    assert (r["objects"], r["ledgers"], r["ledgers_done"], r["ledgers_unread"]) == (5, 3, 3, 0)
    objects = json.loads((root / "lfs.json").read_text())
    assert S.sha(S.state_at(200)[0]) in objects and S.sha(S.state_at(200)[1]) not in objects
    assert all(home.verify_by_download("ckpt-N", s) for s in (400, 500, 600))
    assert "FOUR" in cc.__doc__ and "removes two in one commit" in cc.__doc__


def test_the_budget_is_strict_while_an_essential_commit_may_wait_and_bounded_when_it_may_not(tmp_path):
    """With saves far more frequent than the budget allows the courier waits, without counting a failure, says each
    wait once, logs the hour's count, and sends the newest generation when there is room. While the wait an
    essential commit is allowed (essential_wait) is as long as the budget's hour, no rolling hour holds more than
    the budget. With the real allowance, 15 minutes, a generation goes at the latest 15 minutes after it first
    asked, whatever the count: that is what lets a run's last generation arrive inside the courier's final window."""
    pytest.importorskip("huggingface_hub")
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path, per_hour=4, essential_wait=3600)
    t0 = clock.t
    out = str(chat / "pt")
    for k in range(1, 41):                               # a save every 300 s for 200 minutes: 12 an hour
        at(t0 + 300 * k)
        S.make_gen(out, 100 * k)
        c.tick()
        at(t0 + 300 * k + 20)
        c.tick()
    (chat / "STATUS_pt").write_text("FINISHED\n")
    ticks = 0
    while not c.idle():                                  # the daemon's loop at the end of the job
        at(clock.t + 60)
        c.tick()
        ticks += 1
        assert ticks < 200
    sent = hub_log(root)
    times = [x["at"] for x in sent]
    assert all(x["ok"] for x in sent) and len(sent) >= 12
    assert max(sum(1 for u in times if t - 3600 < u <= t) for t in times) == 4      # never a fifth in an hour
    assert store.steps("ckpt-N")[-1] == 4000 == store.head_step("ckpt-N")              # the last save is in the store
    assert store.mark_steps("pt") == [200]
    pushed = [p["step"] for p in stats(logs, "push") if p["kind"] == "generation"]
    assert pushed == sorted(pushed) and pushed[-1] == 4000 and len(pushed) < 20        # the newest each time, not every one
    st = json.loads((chat / "courier_status.json").read_text())
    assert st["jobs"]["pt"]["failures"] == 0 and st["http_429"] == 0 and st["commit_budget_per_hour"] == 4
    assert st["commits_last_hour"] <= 4
    waits = stats(logs, "budget_wait")
    assert waits and all(w["seconds"] > 0 for w in waits)
    assert len(waits) == len({w["waits"] for w in waits}) <= 41                          # said once for each that waits
    hours = stats(logs, "commits")
    assert len(hours) >= 3 and all(h["commits"] <= 4 and h["budget"] == 4 for h in hours)
    step, rejected = store.pull_generation("ckpt-N", str(tmp_path / "got"))
    assert (step, rejected) == (4000, [])

    # The real allowance. The budget is spent when the job ends; its last generation waits 15 minutes at the most.
    chat, root, store, c, logs, clock, at = hub_courier(tmp_path / "b", per_hour=4)
    t0 = clock.t
    out = str(chat / "pt")
    for k in range(1, 5):                                # the branch, g100, the mark of step 200 and g200 spend the
        at(t0 + 60 * k)                                  # budget; g300 and g400 wait, and are overtaken
        S.make_gen(out, 100 * k)
        c.tick()
    assert store.budget.count() == 4 and store.budget.wait_s() > 0
    S.make_gen(out, 500)                                 # the run's last save
    (chat / "STATUS_pt").write_text("FINISHED\n")
    ended = clock.t
    while not c.idle():
        at(clock.t + 20)
        c.tick()
        assert clock.t - ended < 3600
    assert clock.t - ended <= hf_store.ESSENTIAL_WAIT_S + 60 < 1800     # inside the 5,400 s the boot gives the courier
    assert store.steps("ckpt-N")[-1] == 500 and store.mark_steps("pt") == [200]
    assert json.loads((chat / "courier_status.json").read_text())["jobs"]["pt"]["failures"] == 0
    assert all(w["seconds"] <= hf_store.ESSENTIAL_WAIT_S for w in stats(logs, "budget_wait"))


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


def test_a_kill_after_the_pointer_has_moved_is_completed_by_the_next_boot(tmp_path, monkeypatch):
    """The boot dies after `live` names the new generation and before the flat names follow. The generation is
    live and whole, so the next boot installs nothing; it must still put the flat names right, or model.safetensors,
    optim.safetensors and state.txt stay a mix of two steps for anything that reads them (jobs_vm.take_snapshot
    copies the flat weights and state)."""
    backend, store = pushed_store(tmp_path, [100, 200])
    quiet = lambda m: None  # noqa: E731
    live_link = os.path.join("ckpt_gen", "live")

    def vm(name):
        """A VM that holds step 100, whole, with flat names that are files of their own (an older layout)."""
        chat = tmp_path / name / "chat"
        chat.mkdir(parents=True)
        out = chat / "pt"
        S.make_gen(str(out), 100)
        for f in cc.FLAT:
            data = (out / f).read_bytes()
            os.remove(out / f)
            (out / f).write_bytes(data)
        return chat, out

    real = os.replace
    # 1. Killed at the commit point itself: the pointer has moved, no flat name has.
    chat, out = vm("vm2")

    def after_commit(a, b):
        real(a, b)
        if str(b).endswith(live_link):
            raise KeyboardInterrupt("the boot is killed right after the commit point")
    monkeypatch.setattr(os, "replace", after_commit)
    with pytest.raises(KeyboardInterrupt):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())
    monkeypatch.setattr(os, "replace", real)
    assert os.readlink(out / "ckpt_gen" / "live") == "g200" and cc.local_step(str(chat), "pt") == 200
    assert (out / "state.txt").read_text().startswith("100\n") and not os.path.islink(out / "model.safetensors")  # the mix
    rec = cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())["pt"]
    assert rec["placed"] is False and rec["local_step"] == 200           # nothing to install: the generation is live
    assert all(os.readlink(out / f) == os.path.join("ckpt_gen", "live", f) for f in cc.FLAT)
    assert (out / "state.txt").read_text().startswith("200\n")
    assert ((out / "model.safetensors").read_bytes(), (out / "optim.safetensors").read_bytes()) == S.state_at(200)
    # 2. Killed between two flat names: the weights' name is the new link, the optimiser's and the state's are old.
    chat, out = vm("vm3")

    def between_links(a, b):
        if str(a).endswith("optim.safetensors.link"):
            raise KeyboardInterrupt("the boot is killed between two flat names")
        return real(a, b)
    monkeypatch.setattr(os, "replace", between_links)
    with pytest.raises(KeyboardInterrupt):
        cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())
    monkeypatch.setattr(os, "replace", real)
    assert (out / "model.safetensors").read_bytes() == S.state_at(200)[0]     # through the new pointer
    assert (out / "optim.safetensors").read_bytes() == S.state_at(100)[1]     # the old file: a mix of two steps
    assert all(os.path.lexists(out / f) for f in cc.FLAT)                     # and no name was ever missing
    assert cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())["pt"]["placed"] is False
    assert not os.path.lexists(out / "optim.safetensors.link")
    assert ((out / "model.safetensors").read_bytes(), (out / "optim.safetensors").read_bytes()) == S.state_at(200)
    assert (out / "state.txt").read_text().startswith("200\n")
    # The stand-in trainer resumes from the live generation on such a folder, with its optimiser.
    assert cc.local_step(str(chat), "pt") == 200


def test_replacing_a_damaged_generation_that_live_names_never_leaves_the_pointer_on_it(tmp_path, monkeypatch):
    """This VM's live generation is g200 and it is damaged; the store's newest is g200 too. The install replaces
    the folder `live` names. The pointer is taken away first, so at no kill point does `live` name a folder that
    is being removed, is gone, or is being put back: a boot killed anywhere leaves a VM with no checkpoint, and the
    next boot installs."""
    backend, store = pushed_store(tmp_path, [100, 200])
    quiet = lambda m: None  # noqa: E731
    g200 = os.path.join("ckpt_gen", "g200")

    def vm(name):
        chat = tmp_path / name / "chat"
        chat.mkdir(parents=True)
        out = chat / "pt"
        S.make_gen(str(out), 100)
        S.make_gen(str(out), 200)
        with open(out / "ckpt_gen" / "g200" / "optim.safetensors", "ab") as f:
            f.write(b"torn")
        assert os.readlink(out / "ckpt_gen" / "live") == "g200" and cc.local_step(str(chat), "pt") == 0
        return chat, out

    def killed_restore(chat):
        with pytest.raises(KeyboardInterrupt):
            cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())
        monkeypatch.undo()

    def nothing_to_trust(chat, out):
        assert not os.path.lexists(out / "ckpt_gen" / "live")               # the pointer names nothing
        assert cc.local_step(str(chat), "pt") == 0
        assert not any(os.path.exists(out / f) for f in cc.FLAT)            # the flat names lead nowhere

    def next_boot(chat, out):
        rec = cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())["pt"]
        assert rec["placed"] and rec["local_step"] == 0 and os.readlink(out / "ckpt_gen" / "live") == "g200"
        hf_store.check_generation(str(out / "ckpt_gen" / "g200"), 200)
        assert ((out / "model.safetensors").read_bytes(), (out / "optim.safetensors").read_bytes()) == S.state_at(200)
        assert cc.local_step(str(chat), "pt") == 200

    # 1. Killed as the damaged folder is about to be removed (and so at any point while it is being removed).
    chat, out = vm("vm2")
    real_rmtree = cc.shutil.rmtree

    def no_rmtree(path, *a, **kw):
        if str(path).endswith(g200):
            raise KeyboardInterrupt("the boot is killed while the damaged folder is removed")
        return real_rmtree(path, *a, **kw)
    monkeypatch.setattr(cc.shutil, "rmtree", no_rmtree)
    killed_restore(chat)
    assert (out / "ckpt_gen" / "g200").is_dir()
    nothing_to_trust(chat, out)
    next_boot(chat, out)
    # 2. Killed after the damaged folder is gone and before the verified one is renamed into its place.
    chat, out = vm("vm3")
    real_rename = os.rename

    def no_rename(a, b):
        if str(b).endswith(g200):
            raise KeyboardInterrupt("the boot is killed before the rename into place")
        return real_rename(a, b)
    monkeypatch.setattr(os, "rename", no_rename)
    killed_restore(chat)
    assert not (out / "ckpt_gen" / "g200").exists()
    nothing_to_trust(chat, out)
    next_boot(chat, out)
    # 3. Killed after the rename and before the pointer: the folder is whole and nothing names it yet.
    chat, out = vm("vm4")
    real_symlink = os.symlink

    def no_pointer(src, dst, *a, **kw):
        if str(dst).endswith("live.install"):
            raise KeyboardInterrupt("the boot is killed before the commit point")
        return real_symlink(src, dst, *a, **kw)
    monkeypatch.setattr(os, "symlink", no_pointer)
    killed_restore(chat)
    hf_store.check_generation(str(out / "ckpt_gen" / "g200"), 200)
    nothing_to_trust(chat, out)
    next_boot(chat, out)
    # Without a kill: the damaged live generation is replaced and the boot goes on.
    chat, out = vm("vm5")
    next_boot(chat, out)
    # A damaged folder of that step which `live` does NOT name (live is the older, whole g100): the pointer stays on
    # g100 until the commit point, as before.
    chat = tmp_path / "vm6" / "chat"
    chat.mkdir(parents=True)
    out = chat / "pt"
    S.make_gen(str(out), 100)
    (out / "ckpt_gen" / "g200").mkdir()
    (out / "ckpt_gen" / "g200" / "junk").write_text("half a generation")
    monkeypatch.setattr(os, "symlink", no_pointer)
    killed_restore(chat)
    assert os.readlink(out / "ckpt_gen" / "live") == "g100" and cc.local_step(str(chat), "pt") == 100
    assert cc.restore(fresh(tmp_path / "vm1"), str(chat), JOBS, BR, log=quiet, cfg=cfg2())["pt"]["placed"]
    assert cc.local_step(str(chat), "pt") == 200


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
        assert len(steps) == 3 and store.mark_steps("pt") == [MARK]   # two verified by their listings, and the newest
        _, prev, newest = steps
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
        end = store.steps("ckpt-N")  # three generations, the last one the run's end, and the mark from session one
        assert len(end) == 3 and end[-1] == STEPS and store.mark_steps("pt") == [MARK]
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


def test_the_token_is_removed_when_the_courier_does_not_end(tmp_path):
    """When the manager has ended, the boot's last step gives the courier PK_COURIER_WAIT_S to push the last of it
    and then removes the token whether or not the courier has ended (the review of 12 Nov 2026: the wait had no end,
    and the token's removal came after it). Here the courier makes one pass at its start and then sleeps for longer
    than the test, so it never reaches its own end."""
    repo = seeded(tmp_path)
    vm = make_vm(tmp_path, "vm1", repo, epoch=1, first_start=["pt"])
    chat = vm / "chat"
    try:
        r = boot(vm, 0, PK_COURIER_ARGS="--tick 600 --push-timeout 30", PK_COURIER_WAIT_S="3")
        assert r.returncode == 0, read(chat / "boot.log")
        wait(lambda: read(chat / "STATUS").strip() == "FINISHED", "the manager's end")
        pid = int(read(chat / "courier.pid"))
        assert os.path.exists(f"/proc/{pid}") and (vm / "shm" / "token").exists()   # the courier is up and holds the token
        wait(lambda: "TOKEN_GONE" in read(chat / "boot.log"), "the token's removal", 60)
        assert "the courier had not ended after 3 s" in read(chat / "boot.log")
        assert os.listdir(vm / "shm") == [] and not (chat / "upload" / "store_token").exists()

        def gone():
            try:
                return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
            except OSError:
                return True
        wait(gone, "the courier's end", 30)
        assert TOKEN not in read(chat / "boot.log") + read(chat / "courier.log")
    finally:
        kill_vm(vm)
    boot_text = (CC / "boot_base2.sh").read_text()
    assert "WAIT=${PK_COURIER_WAIT_S:-$(python3 -c" in boot_text and ".get('courier_wait_s', 5400)" in boot_text
    assert "|| echo 5400)}" in boot_text and "-lt \\$end" in boot_text


def test_the_last_generation_delayed_by_the_budget_still_arrives_before_the_token_goes(tmp_path):
    """One number governs the end of a run: the courier's final delivery window (store.json's "courier_wait_s",
    which the runner sends; 5,400 s by default). When the jobs have ended the boot waits that long for the courier
    and then takes the token away. The courier's own waits fit inside it: a commit that finds the budget spent
    waits at most its allowance (15 minutes; 6 s here) and then goes. So a run whose budget is spent at its end
    still delivers its last generation, the courier ends by itself, and the token goes after it. (Before: the
    budget's wait could be an hour and the window was 30 minutes.) With a window shorter than the wait, the same
    run shows the bounded failure: the token goes on time and the last generation is not in the store."""
    repo = seeded(tmp_path)
    store = fresh(tmp_path)
    slow = {"store_commit_budget_per_hour": 3, "commit_essential_wait_s": 6}
    vm = make_vm(tmp_path, "vm1", repo, epoch=1, first_start=["pt"], courier_wait_s=60, **slow)
    chat = vm / "chat"
    try:
        r = boot(vm, 0.03)                               # 200 steps and 20 saves in about six seconds
        assert r.returncode == 0, read(chat / "boot.log")
        wait(lambda: read(chat / "STATUS").strip() == "FINISHED", "the manager's end")
        ended = time.time()
        wait(lambda: "TOKEN_GONE" in read(chat / "boot.log"), "the token's removal", 60)
        took = time.time() - ended
        log = read(chat / "courier.log")
        assert store.steps("ckpt-N")[-1] == STEPS and store.mark_steps("pt") == [MARK]      # the last generation arrived
        assert "courier done" in log and "the courier had not ended" not in read(chat / "boot.log")
        waits = [json.loads(ln.split("stat ", 1)[1]) for ln in log.splitlines() if '"what": "budget_wait"' in ln]
        assert waits and all(w["seconds"] <= 6 for w in waits)                               # it had to wait, and did
        assert took < 45                                                                     # well inside the window of 60
        assert json.loads(read(chat / "courier_status.json"))["jobs"]["pt"]["pushed_step"] == STEPS
        assert os.listdir(vm / "shm") == [] and TOKEN not in log + read(chat / "boot.log")
    finally:
        kill_vm(vm)
    # The same run with a wait the window cannot hold: the token goes at the window's end, the generation does not arrive.
    (tmp_path / "b").mkdir()
    repo2 = seeded(tmp_path / "b")
    vm2 = make_vm(tmp_path / "b", "vm1", repo2, epoch=1, first_start=["pt"], courier_wait_s=3,
                  store_commit_budget_per_hour=3, commit_essential_wait_s=600)
    try:
        r = boot(vm2, 0.03)
        assert r.returncode == 0, read(vm2 / "chat" / "boot.log")
        wait(lambda: "TOKEN_GONE" in read(vm2 / "chat" / "boot.log"), "the token's removal", 60)
        assert "the courier had not ended after 3 s" in read(vm2 / "chat" / "boot.log") and os.listdir(vm2 / "shm") == []
        assert (fresh(tmp_path / "b").steps("ckpt-N") or [0])[-1] < STEPS
    finally:
        kill_vm(vm2)
    # The runner's side of the same number (runner_multi.py): it sends it, and never settles for less.
    runner = (CC / "runner_multi.py").read_text()
    assert "STORE_COURIER_WAIT_S = 5400" in runner and '"courier_wait_s": STORE.get("courier_wait_s", STORE_COURIER_WAIT_S)' in runner
    assert "a.store_settle_min * 60 < a.store_courier_wait_s + TICK" in runner
    assert hf_store.ESSENTIAL_WAIT_S + hf_store.MAX_429_WAIT_S + 1800 <= 5400   # both waits and a full push limit fit


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
