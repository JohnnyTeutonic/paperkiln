"""tools/colab_chat/hf_store.py against a store on local disk (DirBackend) and a recording stand-in for
huggingface_hub.HfApi: a push is one commit that holds the manifest; a pull refuses a wrong hash or a missing file;
a resume takes the newest generation that verifies, gives one up only on a verification error (a transport error
is retried and raised), and refuses another campaign's generation or history missing below the step home has seen;
a generation is written once and against the branch head it was computed on; removal happens only behind two
generations the caller has verified, never below an unpushed mark, never on `main`; storage is released by
difference over every branch, under a lock, sparing young objects; a store call has a wall-time limit; the token
comes from a file, is refused when others can read it, and is absent from every output. No network: nothing here
reaches Hugging Face. base2_seed.py's push-once rule is tested at the end."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
sys.path.insert(0, str(ROOT / "tests"))
import base2_seed  # noqa: E402
import hf_store  # noqa: E402
import store_standin as S  # noqa: E402

TOKEN = "hf_FAKEtokenFAKEtokenFAKEtoken0123"
IDENT = {"campaign": "base2-N", "job": "pt", "spec_sha256": "s" * 64, "inputs_sha256": "i" * 64}


def dstore(tmp_path, **kw):
    return hf_store.open_store({"repo": f"dir:{tmp_path / 'repo'}"}, delay=0, **kw)


def commits(tmp_path):
    return [json.loads(ln) for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines()]


def staged_gen(tmp_path, step, seed=0):
    """A generation as the courier stages it: the trainer's four files and events.jsonl in one folder."""
    out = tmp_path / f"job{seed}"
    S.make_gen(str(out), step, seed)
    g = out / "ckpt_gen" / f"g{step}"
    (g / "events.jsonl").write_text(json.dumps({"event": "checkpoint", "step": step}) + "\n")
    return str(g)


def push_gens(tmp_path, store, steps, branch="ckpt-N", ident=None):
    for s in steps:
        store.push_generation(branch, "pt", s, staged_gen(tmp_path, s), ident)


def test_push_is_one_commit_with_the_manifest(tmp_path):
    store = dstore(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ids").write_bytes(b"a" * 1000)
    (src / "b.ids").write_bytes(b"b" * 10)
    man = store.push_set("main", "corpus/N", str(src))
    c = commits(tmp_path)
    assert len(c) == 1 and c[0]["branch"] == "main" and c[0]["deletes"] == []
    assert c[0]["adds"] == ["corpus/N/a.ids", "corpus/N/b.ids", "corpus/N/manifest.json"]
    assert man["format"] == hf_store.FORMAT and man["files"]["a.ids"] == {"bytes": 1000, "sha256": S.sha(b"a" * 1000)}
    got = tmp_path / "got"
    assert dstore(tmp_path).pull_set("main", "corpus/N", str(got))["files"] == man["files"]
    assert (got / "a.ids").read_bytes() == b"a" * 1000 and (got / "b.ids").read_bytes() == b"b" * 10


def test_pull_refuses_a_wrong_hash_and_a_missing_file(tmp_path):
    store = dstore(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ids").write_bytes(b"a" * 1000)
    (src / "b.ids").write_bytes(b"b" * 10)
    store.push_set("main", "corpus/N", str(src))
    repo = tmp_path / "repo" / "main" / "corpus" / "N"
    (repo / "a.ids").write_bytes(b"a" * 999 + b"x")  # same size, other bytes
    got = tmp_path / "got"
    with pytest.raises(hf_store.VerifyError, match="sha256 or size differs"):
        dstore(tmp_path).pull_set("main", "corpus/N", str(got))
    assert not (got / "a.ids").exists() and not (got / "a.ids.pull.tmp").exists()
    (repo / "a.ids").write_bytes(b"a" * 1000)
    (repo / "b.ids").unlink()
    with pytest.raises(hf_store.VerifyError, match="missing from the store"):
        dstore(tmp_path).pull_set("main", "corpus/N", str(got))
    # A file already there with the right bytes is kept; a wrong one under the name is replaced by the right one.
    (repo / "b.ids").write_bytes(b"b" * 10)
    (got / "b.ids").write_bytes(b"wrong")
    dstore(tmp_path).pull_set("main", "corpus/N", str(got))
    assert (got / "b.ids").read_bytes() == b"b" * 10
    (repo / "manifest.json").write_text("{}")
    with pytest.raises(hf_store.VerifyError, match="bad manifest"):
        dstore(tmp_path).pull_set("main", "corpus/N", str(tmp_path / "got2"))
    # A set that is not there at all is Missing, a StoreError that is not a VerifyError (base2_corpus.store_has).
    with pytest.raises(hf_store.Missing) as e:
        dstore(tmp_path).read_manifest("main", "corpus/none")
    assert not isinstance(e.value, hf_store.VerifyError)


def test_a_generation_is_written_once_whole_and_against_its_head(tmp_path):
    store = dstore(tmp_path)
    g = staged_gen(tmp_path, 100)
    with pytest.raises(hf_store.StoreError, match="ckpt-"):
        store.push_generation("main", "pt", 100, g)
    with open(os.path.join(g, "optim.safetensors"), "ab") as f:
        f.write(b"x")
    with pytest.raises(hf_store.VerifyError, match="optim.safetensors differs"):
        store.push_generation("ckpt-N", "pt", 100, g)
    with pytest.raises(hf_store.VerifyError):
        store.push_generation("ckpt-N", "pt", 200, staged_gen(tmp_path, 100, seed=1))  # the folder is another step
    assert not (tmp_path / "repo" / "commits.jsonl").exists()
    # A step in the store is never written again, by this writer or another.
    push_gens(tmp_path, store, [100], ident={**IDENT, "epoch": 1, "boot_id": "a"})
    with pytest.raises(hf_store.ConflictError, match="written once"):
        dstore(tmp_path).push_generation("ckpt-N", "pt", 100, staged_gen(tmp_path, 100, seed=3))
    assert (tmp_path / "repo" / "ckpt-N" / "gens" / "g100" / "model.safetensors").read_bytes() == S.state_at(100)[0]
    # A push names the head it read; a commit by anyone in between makes the store refuse it.
    head = store.head("ckpt-N")
    assert head["step"] == 100 and head["epoch"] == 1 and head["boot_id"] == "a" and head["steps"] == [100]
    dstore(tmp_path).push_generation("ckpt-N", "pt", 150, staged_gen(tmp_path, 150, seed=4), None, head["commit"])
    with pytest.raises(hf_store.ConflictError, match="moved"):
        store.push_generation("ckpt-N", "pt", 200, staged_gen(tmp_path, 200), None, head["commit"])
    assert store.steps("ckpt-N") == [100, 150]


def test_resume_takes_the_newest_generation_that_verifies(tmp_path):
    store = dstore(tmp_path)
    assert store.pull_generation("ckpt-N", str(tmp_path / "r0")) == (None, [])
    push_gens(tmp_path, store, [100, 200, 300])
    assert store.steps("ckpt-N") == [100, 200, 300] and store.head_step("ckpt-N") == 300
    step, rejected = dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r1"))
    assert (step, rejected) == (300, [])
    m, o = S.state_at(300)
    assert (tmp_path / "r1" / "g300" / "model.safetensors").read_bytes() == m
    assert (tmp_path / "r1" / "g300" / "optim.safetensors").read_bytes() == o
    # The newest generation's optimiser is damaged in the store: the one before it is taken, whole.
    bad = tmp_path / "repo" / "ckpt-N" / "gens" / "g300" / "optim.safetensors"
    bad.write_bytes(bad.read_bytes()[:-1] + b"?")
    step, rejected = dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r2"))
    assert step == 200 and [s for s, _ in rejected] == [300]
    assert not (tmp_path / "r2" / "g300").exists()
    assert (tmp_path / "r2" / "g200" / "optim.safetensors").read_bytes() == S.state_at(200)[1]
    # A manifest that agrees with damaged files (both rewritten) still fails the trainer's own manifest.
    g200 = tmp_path / "repo" / "ckpt-N" / "gens" / "g200"
    (g200 / "model.safetensors").write_bytes(b"other weights")
    man = json.loads((g200 / "manifest.json").read_text())
    man["files"]["model.safetensors"] = {"bytes": 13, "sha256": S.sha(b"other weights")}
    (g200 / "manifest.json").write_text(json.dumps(man))
    step, rejected = dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r3"))
    assert step == 100 and [s for s, _ in rejected] == [300, 200]
    # None left that verifies: the resume is refused, never a start from step 0.
    (tmp_path / "repo" / "ckpt-N" / "gens" / "g100" / "state.txt").write_text("100\n{}\n")
    with pytest.raises(hf_store.StoreError, match="no generation verifies"):
        dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r4"))


class Shaky(hf_store.DirBackend):
    """A store whose reads fail in transport `fail` times, counting the fetches of each path."""

    def __init__(self, root, fail=0):
        super().__init__(root)
        self.fail, self.fetched = fail, []

    def fetch(self, branch, path, dest):
        if self.fail and path.endswith("optim.safetensors"):
            self.fail -= 1
            raise ConnectionError("connection reset by peer")
        self.fetched.append(path)
        super().fetch(branch, path, dest)


def test_a_transport_error_is_not_corruption(tmp_path):
    """A failed download is tried again and, when it keeps failing, raised as itself: the newest generation is not
    given up for an older one, and what was already fetched and verified is not fetched again."""
    push_gens(tmp_path, dstore(tmp_path), [100, 200])
    logs = []
    b = Shaky(str(tmp_path / "repo"), fail=2)
    store = hf_store.Store(b, log=logs.append, tries=4, delay=0)
    assert store.pull_generation("ckpt-N", str(tmp_path / "r")) == (200, [])
    assert sum("fetch of ckpt-N:gens/g200/optim.safetensors failed" in ln for ln in logs) == 2
    assert not any(p.startswith("gens/g100/") for p in b.fetched)  # the older generation was never touched
    assert len([p for p in b.fetched if p == "gens/g200/model.safetensors"]) == 1
    b2 = Shaky(str(tmp_path / "repo"), fail=99)
    store2 = hf_store.Store(b2, log=logs.append, tries=3, delay=0)
    with pytest.raises(ConnectionError):
        store2.pull_generation("ckpt-N", str(tmp_path / "r2"))
    assert not any(p.startswith("gens/g100/") for p in b2.fetched)
    # The next try keeps the files that verified: only the one that failed is fetched.
    b2.fail, b2.fetched = 0, []
    assert store2.pull_generation("ckpt-N", str(tmp_path / "r2")) == (200, [])
    assert b2.fetched == ["gens/g200/optim.safetensors", "gens/g200/state.txt"]


def test_resume_refuses_missing_history_and_another_identity(tmp_path):
    store = dstore(tmp_path)
    # Home has seen step 300 pushed and the branch is empty, missing or renamed: refused, not a first start.
    with pytest.raises(hf_store.IdentityError, match="home has seen step 300"):
        store.pull_generation("ckpt-N", str(tmp_path / "r"), min_step=300)
    push_gens(tmp_path, store, [100, 200], ident=IDENT)
    with pytest.raises(hf_store.IdentityError, match="up to g200"):
        dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"), min_step=300)
    # At or above the minimum it resumes; a damaged newest one still falls back, because the history is there.
    assert dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"), expect=IDENT, min_step=200)[0] == 200
    man = store.read_manifest("ckpt-N", "gens/g200", fresh=True)
    assert {k: man[k] for k in IDENT} == IDENT and man["step"] == 200
    bad = tmp_path / "repo" / "ckpt-N" / "gens" / "g200" / "model.safetensors"
    bad.write_bytes(bad.read_bytes()[:-1] + b"?")
    assert dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r1"), expect=IDENT, min_step=200)[0] == 100
    # A generation of another campaign, job, spec or inputs is refused outright: no fallback to an older one.
    for k in IDENT:
        with pytest.raises(hf_store.IdentityError, match=f"another {k}"):
            dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / f"x{k}"), expect={**IDENT, k: "other"})
    assert not dstore(tmp_path).verify_by_download("ckpt-N", 100, expect={**IDENT, "job": "pt2"})
    assert dstore(tmp_path).verify_by_download("ckpt-N", 100, expect=IDENT)
    assert not dstore(tmp_path).verify_by_download("ckpt-N", 200, expect=IDENT)


def test_prune_only_behind_two_verified_generations(tmp_path):
    store = dstore(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ids").write_bytes(b"a")
    store.push_set("main", "corpus/N", str(src))
    (src / "state.txt").write_text("100\n{}\n")
    (src / "model.safetensors").write_bytes(S.state_at(100)[0])
    assert store.push_mark("pt", 100, str(src)) is True
    assert store.push_mark("pt", 100, str(src)) is False  # once
    push_gens(tmp_path, store, [100, 200, 300, 400, 500])
    main_before = store.b.tree("main")
    # A listing alone never counts: with nothing verified, or one generation verified, nothing is removed.
    assert store.prune("ckpt-N") == [] and store.prune("ckpt-N", verified=[500]) == []
    assert store.steps("ckpt-N") == [100, 200, 300, 400, 500]
    # Verified 300 and 500: everything below 300 goes; the unverified 400 between them stays.
    assert store.prune("ckpt-N", verified=[300, 500]) == [100, 200]
    assert store.steps("ckpt-N") == [300, 400, 500]
    assert store.b.tree("main") == main_before and store.mark_steps("pt") == [100]
    c = commits(tmp_path)[-1]
    assert c["adds"] == [] and c["deletes"] == ["gens/g100/", "gens/g200/"]      # each generation as its folder
    assert sorted(os.listdir(tmp_path / "repo" / "ckpt-N" / "gens")) == ["g300", "g400", "g500"]
    for branch in ("main", "marks", "ckpt"):
        with pytest.raises(hf_store.StoreError):
            store.prune(branch, verified=[300, 500])
    with pytest.raises(hf_store.StoreError, match="at least two"):
        store.prune("ckpt-N", keep=1, verified=[500])
    assert dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"))[0] == 500


def test_prune_never_removes_the_last_verified_generation_or_below_an_unpushed_mark(tmp_path):
    """Newer generations that only look whole do not count. 300 is the newest one the caller verified that is still
    whole; it and the verified one before it stay whatever is listed above them."""
    store = dstore(tmp_path)
    push_gens(tmp_path, store, [100, 200, 300, 400, 500])
    gens = tmp_path / "repo" / "ckpt-N" / "gens"
    (gens / "g500" / "model.safetensors").write_bytes(b"short")  # verified once, damaged since: not whole now
    store = dstore(tmp_path)
    assert store.prune("ckpt-N", verified=[200, 300, 500]) == [100]
    assert store.steps("ckpt-N") == [200, 300, 400, 500]
    step, rejected = dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"))
    assert step == 400 and [s for s, _ in rejected] == [500]
    # A mark staged at step 200 and not yet in the store: nothing at or above 200 is removed, whatever is verified.
    push_gens(tmp_path, store, [600, 700])
    assert store.prune("ckpt-N", verified=[600, 700], protect=[200]) == []
    assert store.steps("ckpt-N") == [200, 300, 400, 500, 600, 700]
    assert store.prune("ckpt-N", verified=[600, 700], protect=[400]) == [200, 300]
    assert store.prune("ckpt-N", verified=[600, 700]) == [400, 500]


def test_bounded_calls(tmp_path):
    assert hf_store.bounded(lambda: {"a": 1}, 20) == {"a": 1}
    assert hf_store.bounded(lambda: 7, 0) == 7  # no limit: in this process
    t0 = time.time()
    with pytest.raises(hf_store.StoreError, match="ended"):
        hf_store.bounded(lambda: time.sleep(60), 0.5)
    assert time.time() - t0 < 10 and not hf_store._CHILDREN
    # A result far larger than a pipe's buffer comes back whole (the parent reads while the child writes).
    big = hf_store.bounded(lambda: ["x" * 100] * 20000, 30)
    assert len(big) == 20000

    def boom():
        raise ConnectionError(f"Bearer {TOKEN}")

    with pytest.raises(hf_store.StoreError) as e:
        hf_store.bounded(boom, 20, TOKEN)
    assert "[token]" in str(e.value) and TOKEN not in str(e.value) and "ConnectionError" in str(e.value)
    # A store error keeps its class across the child, so a verification error still reads as one.
    with pytest.raises(hf_store.VerifyError, match="differs"):
        hf_store.bounded(lambda: (_ for _ in ()).throw(hf_store.VerifyError("sha256 differs")), 20)
    with pytest.raises(hf_store.IdentityError):
        hf_store.bounded(lambda: (_ for _ in ()).throw(hf_store.IdentityError("another job")), 20)
    seen = []
    hf_store.bounded(lambda: time.sleep(2.5), 20, while_waiting=lambda: seen.append(1))
    assert len(seen) >= 2


def test_token_file_rules(tmp_path):
    tf = tmp_path / "token"
    tf.write_text(TOKEN + "\n")
    tf.chmod(0o644)
    with pytest.raises(hf_store.StoreError, match="readable by others"):
        hf_store.read_token(str(tf))
    tf.chmod(0o600)
    assert hf_store.read_token(str(tf)) == TOKEN
    with pytest.raises(hf_store.StoreError, match="not found"):
        hf_store.read_token(str(tmp_path / "none"))
    # No default store and no default token path; the pinned inputs repo is refused outright.
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"repo_type": "dataset"}))
    with pytest.raises(hf_store.StoreError, match="no default store"):
        hf_store.load_config(str(cfg))
    with pytest.raises(hf_store.StoreError, match="no default token path"):
        hf_store.open_store({"repo": "someone/base2-store"})
    for repo in hf_store.REFUSED_REPOS:
        with pytest.raises(hf_store.StoreError, match="inputs repo"):
            hf_store.open_store({"repo": repo, "token_file": str(tf)}, api=FakeApi())
    assert "chat-screens-inputs" in " ".join(hf_store.REFUSED_REPOS)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"  # set when hf_store is imported


def test_adopt_and_forget_token(tmp_path, monkeypatch):
    cfg, up, shm = tmp_path / "store.json", tmp_path / "upload_token", tmp_path / "shm"
    cfg.write_text(json.dumps({"repo": "someone/base2-store"}))
    with pytest.raises(hf_store.StoreError, match="no token was uploaded"):
        hf_store.adopt_token(str(cfg), str(up), str(shm))
    # A failure half way leaves no temporary copy behind.
    up.write_text(TOKEN)
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        hf_store.adopt_token(str(cfg), str(up), str(shm))
    monkeypatch.undo()
    assert os.listdir(shm) == []
    hf_store.adopt_token(str(cfg), str(up), str(shm))
    tf = json.loads(cfg.read_text())["token_file"]
    assert tf == str(shm / "token") and not up.exists() and os.stat(tf).st_mode & 0o777 == 0o600
    assert hf_store.read_token(tf) == TOKEN and TOKEN not in cfg.read_text()
    hf_store.adopt_token(str(cfg), str(up), str(shm))  # a second boot on the same VM: kept
    # Forgetting removes the adopted file, a temporary left by a killed adoption, and the upload copy.
    (shm / "token.tmp123").write_text(TOKEN)
    chat = tmp_path / "chat"
    (chat / "upload").mkdir(parents=True)
    (chat / "upload" / "store_token").write_text(TOKEN)
    assert hf_store.forget_token(str(cfg), chat=str(chat)) is True
    assert os.listdir(shm) == [] and not (chat / "upload" / "store_token").exists()
    assert hf_store.forget_token(str(cfg)) is True  # nothing left to remove is still "gone"


def test_an_adoption_that_fails_after_the_move_leaves_nothing_the_clean_up_cannot_find(tmp_path, monkeypatch):
    """The token has been renamed into the tmpfs folder and its upload removed; the config's write that would name
    it then fails. No record says where the token is. forget_token() finds it all the same: by the folder the
    adoption wrote into the config before it moved anything, and by the tmpfs folder it always searches."""
    def setup(name):
        d = tmp_path / name
        d.mkdir()
        cfg, up, shm = d / "store.json", d / "upload_token", d / "shm"
        cfg.write_text(json.dumps({"repo": "someone/base2-store"}))
        up.write_text(TOKEN)
        return cfg, up, shm

    real = os.replace
    state = {"moved": False}

    def fail_config_after_move(a, b):
        if str(b).endswith("token"):
            state["moved"] = True
        elif state["moved"] and str(b).endswith("store.json"):
            raise OSError("disk full")
        return real(a, b)

    # 1. The folder is the one the VM's environment names (or the default): the glob finds the token.
    cfg, up, shm = setup("env")
    monkeypatch.setenv("PK_STORE_TOKEN_DIR", str(shm))
    monkeypatch.setattr(os, "replace", fail_config_after_move)
    with pytest.raises(OSError, match="disk full"):
        hf_store.adopt_token(str(cfg), str(up), str(shm))
    monkeypatch.setattr(os, "replace", real)
    left = json.loads(cfg.read_text())
    assert (shm / "token").read_text() == TOKEN and not up.exists() and "token_file" not in left   # moved, and unrecorded
    assert str(shm / "token") in hf_store.token_copies(str(cfg))
    assert hf_store.forget_token(str(cfg)) is True and os.listdir(shm) == []
    # 2. Another folder, which no environment names: the config's "token_dir", written before the move, finds it.
    state["moved"] = False
    cfg, up, shm = setup("named")
    monkeypatch.setenv("PK_STORE_TOKEN_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(os, "replace", fail_config_after_move)
    with pytest.raises(OSError, match="disk full"):
        hf_store.adopt_token(str(cfg), str(up), str(shm))
    monkeypatch.setattr(os, "replace", real)
    assert json.loads(cfg.read_text()) == {"repo": "someone/base2-store", "token_dir": str(shm)}
    assert (shm / "token").exists() and hf_store.forget_token(str(cfg)) is True and os.listdir(shm) == []
    # 3. No config at all (it was never written, or is damaged): the tmpfs folder is searched all the same.
    cfg, up, shm = setup("noconfig")
    shm.mkdir()
    (shm / "token").write_text(TOKEN)
    (shm / "token.tmp9").write_text(TOKEN)
    cfg.write_text("{not json")
    monkeypatch.setenv("PK_STORE_TOKEN_DIR", str(shm))
    assert hf_store.main(["forget-token", "--config", str(cfg)]) == 0 and os.listdir(shm) == []
    # The boot's own last step and its failure path both go through this clean-up; the runner's exec at home
    # searches the same folder (runner_multi.store_forget_token).
    boot = (ROOT / "tools" / "colab_chat" / "boot_base2.sh").read_text()
    assert boot.count("forget-token --config") >= 2
    runner = (ROOT / "tools" / "colab_chat" / "runner_multi.py").read_text()
    assert "glob.glob(d + '/token*')" in runner


def test_forget_token_ends_the_courier_and_its_children_first(tmp_path):
    """A process group standing for the courier and a push in flight: both are gone before the file is."""
    chat = tmp_path / "chat"
    chat.mkdir()
    cfg = tmp_path / "store.json"
    tf = tmp_path / "token"
    tf.write_text(TOKEN)
    cfg.write_text(json.dumps({"repo": "someone/base2-store", "token_file": str(tf)}))
    code = ("import os, signal, subprocess, sys, time\n"
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            "open(sys.argv[1], 'w').write(str(c.pid))\n"
            "time.sleep(300)\n")
    p = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "child.pid")], start_new_session=True)
    (chat / "courier.pid").write_text(str(p.pid))
    t0 = time.time()
    while not (tmp_path / "child.pid").exists() or not (tmp_path / "child.pid").read_text():
        assert time.time() - t0 < 20
        time.sleep(0.05)
    child = int((tmp_path / "child.pid").read_text())
    assert hf_store.forget_token(str(cfg), chat=str(chat)) is True
    p.wait(timeout=10)  # stop_courier has reaped it already, so the exit status is not ours to read
    assert not os.path.exists(f"/proc/{p.pid}") and not tf.exists()
    t0 = time.time()
    while os.path.exists(f"/proc/{child}") and open(f"/proc/{child}/stat").read().rsplit(")", 1)[1].split()[0] != "Z":
        assert time.time() - t0 < 10, "the child outlived the courier"
        time.sleep(0.05)


# ---------------------------------------------------------------- the Hugging Face backend, on a recording stand-in
try:  # the library's own classes, as the hub raises them; stand-ins of the same names only without the library
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
except ImportError:
    class RevisionNotFoundError(Exception):
        pass

    class EntryNotFoundError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass


class FakeFile:
    def __init__(self, path, data, lfs=True):
        self.path, self.size = path, len(data)
        self.lfs = {"sha256": S.sha(data), "size": len(data)} if lfs else None


class FakeFolder:
    def __init__(self, path):
        self.path, self.tree_id = path, "t"


class FakeLfs:
    def __init__(self, sha, at, by="oid", size=None):
        self.oid, self.file_oid = (sha, "blob" + sha[:8]) if by == "oid" else ("blob" + sha[:8], sha)
        self.pushed_at = datetime.datetime.fromtimestamp(at, datetime.timezone.utc)
        self.size = size


class Ref:
    def __init__(self, name):
        self.name = name


class FakeApi:
    """Records every call; every argument must come by keyword, as the pinned library's signatures take them.
    Branches are {path: bytes}; LFS objects are {sha256: pushed_at}."""

    def __init__(self, lfs_by="oid"):
        self.calls, self.branches, self.lfs, self.fail, self.lfs_by = [], {"main": {}}, {}, None, lfs_by
        self.heads, self.now = {"main": 0}, 1_000_000.0
        self.snaps, self.sizes, self.after = {}, {}, {}

    def _rec(self, name, **kw):
        self.calls.append((name, kw))
        if self.fail:
            raise self.fail

    def _done(self, name):
        """A test's hook that runs once, right after the named call has answered (another writer acting then)."""
        hook = self.after.pop(name, None)
        if hook:
            hook()

    def _files(self, revision):
        """A branch's files now, or, for a commit id "<branch>@<n>" that is no longer the head, as that commit left
        them."""
        if revision in self.branches:
            return self.branches[revision]
        branch, _, n = str(revision).partition("@")
        if branch in self.branches and n.isdigit():
            if int(n) == self.heads[branch] or revision not in self.snaps:
                return self.branches[branch]
            return self.snaps[revision]
        raise RevisionNotFoundError(str(revision))

    def create_branch(self, *, repo_id, branch, exist_ok=False, repo_type=None, token=None):
        self._rec("create_branch", repo=repo_id, branch=branch, repo_type=repo_type, token=token, exist_ok=exist_ok)
        self.branches.setdefault(branch, {})
        self.heads.setdefault(branch, 0)

    def list_repo_refs(self, *, repo_id, repo_type=None, token=None):
        self._rec("list_repo_refs", repo=repo_id, repo_type=repo_type, token=token)
        return type("Refs", (), {"branches": [Ref(b) for b in self.branches]})()

    def repo_info(self, *, repo_id, revision=None, repo_type=None, token=None):
        self._rec("repo_info", repo=repo_id, revision=revision, repo_type=repo_type, token=token)
        if revision not in self.branches:
            raise RevisionNotFoundError(revision)
        sha = f"{revision}@{self.heads[revision]}"
        self._done("repo_info")
        return type("Info", (), {"sha": sha})()

    def list_repo_tree(self, *, repo_id, recursive=False, revision=None, repo_type=None, token=None):
        self._rec("list_repo_tree", repo=repo_id, revision=revision, repo_type=repo_type, token=token, recursive=recursive)
        files = dict(self._files(revision))
        self._done("list_repo_tree")
        return [FakeFolder("gens")] + [FakeFile(p, b, lfs=p.endswith(".safetensors")) for p, b in sorted(files.items())]

    def create_commit(self, *, repo_id, operations, commit_message, revision=None, parent_commit=None, repo_type=None, token=None):
        ops = list(operations)
        self._rec("create_commit", repo=repo_id, revision=revision, repo_type=repo_type, token=token, message=commit_message,
                  parent=parent_commit, ops=[(type(o).__name__, o.path_in_repo) for o in ops])
        if parent_commit is not None and parent_commit != f"{revision}@{self.heads[revision]}":
            raise RuntimeError("412 Precondition Failed: parent commit is not the head")
        self.snaps[f"{revision}@{self.heads[revision]}"] = dict(self.branches[revision])  # the commit being replaced
        for o in ops:
            if type(o).__name__ == "CommitOperationAdd":
                data = open(o.path_or_fileobj, "rb").read()
                self.branches[revision][o.path_in_repo] = data
                if o.path_in_repo.endswith(".safetensors"):
                    self.lfs.setdefault(S.sha(data), self.now)
                    self.sizes[S.sha(data)] = len(data)
            elif o.is_folder:  # the library's own reading of a path that ends in "/"
                for p in [p for p in self.branches[revision] if p.startswith(o.path_in_repo)]:
                    del self.branches[revision][p]
            else:
                del self.branches[revision][o.path_in_repo]
        self.heads[revision] += 1

    def hf_hub_download(self, *, repo_id, filename, revision=None, local_dir=None, repo_type=None, token=None):
        self._rec("hf_hub_download", repo=repo_id, filename=filename, revision=revision, repo_type=repo_type, token=token)
        files = self._files(revision)
        if filename not in files:
            raise EntryNotFoundError(filename)
        p = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(files[filename])
        return p

    def list_lfs_files(self, *, repo_id, repo_type=None, token=None):
        self._rec("list_lfs_files", repo=repo_id, repo_type=repo_type, token=token)
        return [FakeLfs(o, at, self.lfs_by, self.sizes.get(o)) for o, at in sorted(self.lfs.items())]

    def permanently_delete_lfs_files(self, *, repo_id, lfs_files, rewrite_history=True, repo_type=None, token=None):
        gone = [f.oid if self.lfs_by == "oid" else f.file_oid for f in lfs_files]
        self._rec("permanently_delete_lfs_files", repo=repo_id, oids=gone, rewrite_history=rewrite_history,
                  repo_type=repo_type, token=token)
        for o in gone:
            del self.lfs[o]


def hstore(tmp_path, api, **kw):
    pytest.importorskip("huggingface_hub")
    tmp_path.mkdir(parents=True, exist_ok=True)
    tf = tmp_path / "token"
    tf.write_text(TOKEN + "\n")
    tf.chmod(0o600)
    logs = []
    cfg = {"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tf), **kw}
    return hf_store.open_store(cfg, log=logs.append, api=api, delay=0), logs


def test_hf_backend_calls(tmp_path):
    """The calls the library receives, every argument by keyword: one create_commit per push with every file and
    the manifest, on the branch, against the head that was read, with the token and the repo type; downloads by
    revision; the listing parsed; a path that is not there read as Missing; a moved head refused."""
    import huggingface_hub
    assert huggingface_hub.__version__ == hf_store.HUB_VERSION, "the home library is not the pinned version"
    assert f"HUB={hf_store.HUB_VERSION}" in (ROOT / "tools" / "colab_chat" / "boot_base2.sh").read_text()
    api = FakeApi()
    store, _ = hstore(tmp_path, api)
    store.b.ensure_branch("ckpt-N")
    head = store.head("ckpt-N")
    assert head["commit"] == "ckpt-N@0" and head["step"] is None
    store.push_generation("ckpt-N", "pt", 100, staged_gen(tmp_path, 100), {**IDENT, "epoch": 2, "boot_id": "b1"}, head["commit"])
    kw = [c[1] for c in api.calls if c[0] == "create_commit"][0]
    assert kw["revision"] == "ckpt-N" and kw["repo"] == "someone/base2-store" and kw["repo_type"] == "dataset"
    assert kw["token"] == TOKEN and kw["parent"] == "ckpt-N@0"
    assert kw["ops"] == [("CommitOperationAdd", f"gens/g100/{f}") for f in
                         sorted(["MANIFEST.json", "events.jsonl", "manifest.json", "model.safetensors",
                                 "optim.safetensors", "state.txt"])]
    assert store.b.tree("nope") is None and store.b.head("nope") is None  # a branch that does not exist yet
    tree = store.b.tree("ckpt-N")
    assert "gens" not in tree and tree["gens/g100/model.safetensors"]["sha256"] == S.sha(S.state_at(100)[0])
    assert tree["gens/g100/state.txt"]["sha256"] is None  # not an LFS file: size only
    head = hstore(tmp_path, api)[0].head("ckpt-N")
    assert (head["step"], head["epoch"], head["boot_id"], head["commit"]) == (100, 2, "b1", "ckpt-N@1")
    step, _ = hstore(tmp_path, api)[0].pull_generation("ckpt-N", str(tmp_path / "r"), expect=IDENT, min_step=100)
    assert step == 100 and not any(d.startswith(".hfstage") for d in os.listdir(tmp_path / "r" / "g100"))
    with pytest.raises(hf_store.Missing):
        store.read_manifest("main", "corpus/none")
    with pytest.raises(RuntimeError, match="412"):  # the hub's refusal of a stale parent reaches the caller
        store.push_generation("ckpt-N", "pt", 200, staged_gen(tmp_path, 200), None, "ckpt-N@0")
    assert store.steps("ckpt-N") == [100] and store.b.branches() == ["ckpt-N", "main"]
    assert all(c[1]["token"] == TOKEN and c[1]["repo_type"] == "dataset" for c in api.calls)
    assert ("create_branch", "main") not in [(c[0], c[1].get("branch")) for c in api.calls]
    # create_branch: once for a branch, by the first call that needs it, never again before a commit (it was sent
    # before every commit; a commit that names its parent is on a branch that exists).
    assert [c[1]["branch"] for c in api.calls if c[0] == "create_branch"] == ["ckpt-N"]
    api.calls.clear()
    fresh_store, _ = hstore(tmp_path, api)
    head = fresh_store.head("ckpt-N")
    for step in (200, 300):
        fresh_store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step, seed=7), None,
                                    fresh_store.head("ckpt-N")["commit"])
    assert head["step"] == 100 and "create_branch" not in [c[0] for c in api.calls]
    fresh_store.push_set("ckpt-new", "x", str(tmp_path / "job0" / "ckpt_gen" / "g100"))      # no parent: created, once
    fresh_store.push_set("ckpt-new", "y", str(tmp_path / "job0" / "ckpt_gen" / "g100"))
    assert [c[1]["branch"] for c in api.calls if c[0] == "create_branch"] == ["ckpt-new"]


def release_setup(tmp_path, lfs_by="oid"):
    api = FakeApi(lfs_by)
    store, logs = hstore(tmp_path, api, reclaim=True)
    mark = tmp_path / "mark"
    mark.mkdir(exist_ok=True)
    (mark / "model.safetensors").write_bytes(S.state_at(100)[0])
    (mark / "state.txt").write_text("100\n{}\n")
    store.push_mark("pt", 100, str(mark))
    push_gens(tmp_path, store, [100, 200, 300])
    api.branches["base2-build"] = {"build/blob.safetensors": b"build bytes"}  # a branch no config names
    api.lfs[S.sha(b"build bytes")] = api.now
    api.heads["base2-build"] = 0
    store.push_generation("ckpt-S", "pt2", 200, staged_gen(tmp_path, 200))  # another arm holds g200's bytes too
    return api, store, logs


def test_release_by_difference_under_a_lock(tmp_path):
    """After a prune, the bytes to free are every LFS object minus every object named in the tree of every branch.
    A mark keeps its generation's weights, another branch keeps what it names, an object not yet in any tree
    (another writer's upload before its commit) is spared while it is young, and a held lock means no release."""
    api, store, logs = release_setup(tmp_path)
    assert store.prune("ckpt-N", verified=[200, 300]) == [100]
    m100, o100 = S.state_at(100)
    uncommitted = S.sha(b"another arm's upload, not committed yet")
    api.lfs[uncommitted] = api.now  # young
    api.calls.clear()
    # Too early: everything is younger than the grace period, nothing is deleted.
    assert store.release("boot-a", now=api.now + 60) == 0
    assert "permanently_delete_lfs_files" not in [c[0] for c in api.calls]
    # Later: only the removed generation's optimiser has no name left; the uncommitted upload is now old too.
    api.lfs[uncommitted] = api.now + hf_store.RELEASE_GRACE_S  # pushed a moment ago, at the later time
    api.calls.clear()
    assert store.release("boot-a", now=api.now + hf_store.RELEASE_GRACE_S + 60) == 1
    dele = dict(api.calls)["permanently_delete_lfs_files"]
    assert dele["oids"] == [S.sha(o100)] and dele["rewrite_history"] is False
    for kept in (S.sha(m100), S.sha(S.state_at(200)[0]), S.sha(b"build bytes"), uncommitted):
        assert kept in api.lfs
    # The lock: taken by a conditional commit on main, removed afterwards; every branch's tree was read.
    cc = [c[1] for c in api.calls if c[0] == "create_commit"]
    assert cc[0]["ops"] == [("CommitOperationAdd", hf_store.LOCK)] and cc[0]["parent"] is not None
    assert cc[-1]["ops"] == [("CommitOperationDelete", hf_store.LOCK)] and hf_store.LOCK not in api.branches["main"]
    assert {c[1]["revision"] for c in api.calls if c[0] == "list_repo_tree"} >= {"main", "ckpt-N", "ckpt-S", "base2-build"}
    # Another releaser holds a fresh lock: nothing is listed or deleted. A stale lock is taken over.
    api.branches["main"][hf_store.LOCK] = json.dumps({"owner": "boot-b", "at": api.now}).encode()
    api.calls.clear()
    assert store.release("boot-a", now=api.now + 10) is None
    assert "list_lfs_files" not in [c[0] for c in api.calls]
    assert store.release("boot-a", now=api.now + hf_store.LOCK_STALE_S + 10) == 0
    # A lock commit that loses the race on main (the head moved) is not a lock.
    api.fail = None
    real = api.create_commit

    def racing(**kw):
        if kw["operations"] and getattr(kw["operations"][0], "path_in_repo", "") == hf_store.LOCK and kw.get("parent_commit"):
            api.heads["main"] += 1  # someone commits to main between the read and the lock
        return real(**kw)

    api.create_commit = racing
    assert store.release("boot-a", now=api.now + 3 * hf_store.LOCK_STALE_S) is None


def test_release_refuses_when_it_cannot_tell_the_objects_apart(tmp_path):
    """The library's two id fields: whichever holds the sha256, the named objects are kept; when neither does,
    nothing is deleted."""
    api, store, _ = release_setup(tmp_path, lfs_by="file_oid")
    assert store.prune("ckpt-N", verified=[200, 300]) == [100]
    assert store.release("a", now=api.now + hf_store.RELEASE_GRACE_S + 60) == 1
    assert S.sha(S.state_at(100)[0]) in api.lfs and S.sha(S.state_at(100)[1]) not in api.lfs
    api2, store2, _ = release_setup(tmp_path / "b")
    assert store2.prune("ckpt-N", verified=[200, 300]) == [100]
    real = api2.list_lfs_files
    api2.list_lfs_files = lambda **kw: [type("O", (), {"oid": "x" + f.oid[1:], "file_oid": "y", "pushed_at": f.pushed_at})()
                                        for f in real(**kw)]
    with pytest.raises(hf_store.StoreError, match="release refused"):
        store2.release("a", now=api2.now + hf_store.RELEASE_GRACE_S + 60)
    assert "permanently_delete_lfs_files" not in [c[0] for c in api2.calls]
    assert hf_store.LOCK not in api2.branches["main"]  # the lock is given back on the refusal too
    # A prune by itself releases nothing: release is a separate, opt-in call.
    api3, store3, _ = release_setup(tmp_path / "c")
    api3.calls.clear()
    assert store3.prune("ckpt-N", verified=[200, 300]) == [100]
    assert not {"list_lfs_files", "permanently_delete_lfs_files"} & {c[0] for c in api3.calls}


def test_the_release_lock_reads_the_head_first_and_is_released_only_by_its_owner(tmp_path):
    """Two releasers. The order is: read main's head, look for the lock in the tree of that commit, commit the lock
    with that commit as parent. So a second releaser's lock that lands after the head was read refuses the first
    one's commit, wherever in between it lands (the review of 12 Nov 2026: the old order listed the tree, then read
    the head, and a lock committed between the two was overwritten). And a lock is removed only by the release
    that took it."""
    api, store, logs = release_setup(tmp_path)
    other, other_logs = hstore(tmp_path / "other", api, reclaim=True)
    later = api.now + 10
    took = []
    # 1. B's lock lands right after A has read the head, before A lists the tree.
    api.after["repo_info"] = lambda: took.append(other._lock("boot-b", later))
    api.calls.clear()
    assert store.release("boot-a", now=later) is None
    assert took[0] and json.loads(api.branches["main"][hf_store.LOCK])["nonce"] == took[0]["nonce"]
    assert "list_lfs_files" not in [c[0] for c in api.calls]            # A released nothing
    locks = [c[1] for c in api.calls if c[0] == "create_commit" and c[1]["ops"] == [("CommitOperationAdd", hf_store.LOCK)]]
    assert len(locks) == 2 and locks[0]["parent"] == locks[1]["parent"]  # both named the head they read; the hub took one
    other._unlock(took[0])
    assert hf_store.LOCK not in api.branches["main"]
    # 2. B's lock lands after A has listed the tree and found no lock (the old code read the head only now).
    api.after["list_repo_tree"] = lambda: took.append(other._lock("boot-b", later))
    assert store.release("boot-a", now=later) is None
    assert took[1] and json.loads(api.branches["main"][hf_store.LOCK])["nonce"] == took[1]["nonce"]
    other._unlock(took[1])
    # 3. With no one else, the release goes through and reports what it freed.
    assert store.prune("ckpt-N", verified=[200, 300]) == [100]
    rep = store.release_report("boot-a", now=api.now + hf_store.RELEASE_GRACE_S + 60)
    assert (rep["held"], rep["objects"], rep["bytes"]) == (False, 1, len(S.state_at(100)[1]))
    # What the store then holds, for the storage guard at home: every object left; the bytes each branch's tree
    # names; the marks by their folder; and nothing unnamed, since the one unnamed object was old and is gone.
    sizes = {s: sum(len(b) for b in S.state_at(s)) for s in (200, 300)}
    assert rep["tree_bytes"]["ckpt-N"] == sizes[200] + sizes[300] and rep["tree_bytes"]["ckpt-S"] == sizes[200]
    assert rep["mark_bytes"] == {"pt": len(S.state_at(100)[0])} and rep["unnamed_bytes"] == 0
    assert rep["stored_bytes"] == sum(api.sizes[o] for o in api.lfs if o in api.sizes)
    again = other.release_report("boot-b", now=api.now + hf_store.RELEASE_GRACE_S + 60)
    assert (again["held"], again["objects"], again["bytes"]) == (False, 0, 0) and again["stored_bytes"] == rep["stored_bytes"]
    # An object no tree names that is too young to delete is still stored, and is counted as such.
    young = b"an upload that is not committed yet"
    api.lfs[S.sha(young)], api.sizes[S.sha(young)] = api.now + hf_store.RELEASE_GRACE_S, len(young)
    assert other.release_report("boot-b", now=api.now + hf_store.RELEASE_GRACE_S + 60)["unnamed_bytes"] == len(young)
    del api.lfs[S.sha(young)]
    # 4. Ownership. A holds the lock, it goes stale, B takes it over; A's unlock then leaves B's lock where it is.
    mine = store._lock("boot-a", later)
    theirs = other._lock("boot-b", later + hf_store.LOCK_STALE_S + 1)
    assert mine and theirs and mine["nonce"] != theirs["nonce"]
    store._unlock(mine)
    assert json.loads(api.branches["main"][hf_store.LOCK])["nonce"] == theirs["nonce"]
    assert any("another releaser's" in ln for ln in logs)
    other._unlock(theirs)
    assert hf_store.LOCK not in api.branches["main"]
    # 5. Main moves under the unlock (a mark, an input): the conditional delete is refused, read again, and done.
    mine = store._lock("boot-a", later)
    api.after["list_repo_tree"] = lambda: api.create_commit(repo_id="r", operations=[], commit_message="a mark", revision="main")
    store._unlock(mine)
    assert hf_store.LOCK not in api.branches["main"]
    # 6. A lock that cannot be read is never taken for a stale one.
    api.branches["main"][hf_store.LOCK] = json.dumps({"owner": "boot-b", "at": later, "nonce": "n"}).encode()
    real = api.hf_hub_download
    api.hf_hub_download = lambda **kw: (_ for _ in ()).throw(ConnectionError("reset"))
    assert store.release("boot-a", now=later + 10 * hf_store.LOCK_STALE_S) is None
    api.hf_hub_download = real
    assert json.loads(api.branches["main"][hf_store.LOCK])["owner"] == "boot-b"
    # What the code says of itself: the lock is between releasers, and a small grace needs one writer.
    for text in (hf_store.Store.release_report.__doc__, hf_store.__doc__):
        assert "one writer at a time" in " ".join(text.split())
    assert "NOT SAFE BESIDE ANOTHER WRITER WITH A SMALL GRACE" in hf_store.Store.release_report.__doc__


def test_marks_have_a_namespace_and_the_refused_repo_is_refused_in_any_case(tmp_path):
    """A second attempt under another campaign name keeps its marks apart from the first one's (the runner's
    --store-campaign sets "mark_ns"); without one the paths are as they were."""
    mark = tmp_path / "mark"
    mark.mkdir()
    (mark / "model.safetensors").write_bytes(b"w")
    (mark / "state.txt").write_text("100\n{}\n")
    plain = dstore(tmp_path)
    second = hf_store.open_store({"repo": f"dir:{tmp_path / 'repo'}", "mark_ns": "reh2."}, delay=0)
    assert plain.push_mark("pt", 100, str(mark)) is True
    assert second.mark_steps("pt") == [] and second.push_mark("pt", 100, str(mark)) is True
    assert plain.mark_steps("pt") == [100] == second.mark_steps("pt") and plain.mark_steps("reh2.pt") == [100]
    assert sorted(os.listdir(tmp_path / "repo" / "main" / "marks")) == ["pt", "reh2.pt"]
    second.pull_mark("pt", 100, str(tmp_path / "got"))
    tf = tmp_path / "token"
    tf.write_text(TOKEN)
    tf.chmod(0o600)
    for repo in ("johnnyteutonic/chat-screens-inputs", "JOHNNYTEUTONIC/Chat-Screens-Inputs"):
        assert hf_store.refused_repo(repo)
        with pytest.raises(hf_store.StoreError, match="inputs repo"):
            hf_store.open_store({"repo": repo, "token_file": str(tf)}, api=FakeApi())
    assert not hf_store.refused_repo("johnnyteutonic/base2-store")


def test_cli_never_prints_the_token(tmp_path, monkeypatch, capsys):
    api = FakeApi()
    api.fail = RuntimeError(f"401 Unauthorized: Bearer {TOKEN} rejected")
    tf = tmp_path / "token"
    tf.write_text(TOKEN)
    tf.chmod(0o600)
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": "someone/base2-store", "token_file": str(tf)}))
    real_open = hf_store.open_store
    monkeypatch.setattr(hf_store, "open_store", lambda c, **kw: real_open(c, api=api, **kw))
    src = tmp_path / "src"
    src.mkdir()
    (src / "a").write_bytes(b"a")
    for limit in ("0", "20"):  # in this process, and through the bounded child
        rc = hf_store.main(["push", "--config", str(cfg), "--branch", "main", "--prefix", "x", "--dir", str(src), "--limit", limit])
        out = capsys.readouterr()
        assert rc == 1 and "hf_store push failed" in out.err and "[token]" in out.err
        assert TOKEN not in out.err and TOKEN not in out.out
    assert hf_store.redact(f"x {TOKEN} y", TOKEN) == "x [token] y"


def test_cli_on_a_dir_store(tmp_path):
    """The command line the boot and the data job use, in a process of its own."""
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'repo'}", "branches": {"pt": "ckpt-N"}}))
    src = tmp_path / "src"
    src.mkdir()
    (src / "N.main.ids").write_bytes(b"\x01\x02" * 500)
    tool = [sys.executable, str(ROOT / "tools" / "colab_chat" / "hf_store.py")]
    run = lambda *a: subprocess.run([*tool, *a], capture_output=True, text=True)  # noqa: E731
    assert run("push", "--config", str(cfg), "--branch", "main", "--prefix", "corpus/N", "--dir", str(src)).returncode == 0
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [], "store_inputs": [{"prefix": "corpus/N", "to": "chat/data"}]}))
    rec = tmp_path / "inputs.json"
    r = run("inputs", "--config", str(cfg), "--jobs", str(jobs), "--root", str(tmp_path / "vm"), "--record", str(rec), "--limit", "60")
    assert r.returncode == 0 and "1 files" in r.stdout, r.stderr
    assert (tmp_path / "vm" / "chat" / "data" / "N.main.ids").read_bytes() == b"\x01\x02" * 500
    d = json.loads(rec.read_text())
    assert d["sets"] == {"main:corpus/N": {"N.main.ids": S.sha(b"\x01\x02" * 500)}}
    assert d["inputs_sha256"] == hf_store.inputs_hash(d["sets"])
    store = hf_store.open_store(str(cfg))
    assert hf_store.preflight(store, ["ckpt-N"], json.loads(jobs.read_text())["store_inputs"]) == {"ckpt-N": None}
    (tmp_path / "repo" / "main" / "corpus" / "N" / "N.main.ids").write_bytes(b"\x01\x03" * 500)
    (tmp_path / "vm" / "chat" / "data" / "N.main.ids").unlink()
    r = run("inputs", "--config", str(cfg), "--jobs", str(jobs), "--root", str(tmp_path / "vm"))
    assert r.returncode == 1 and "sha256 or size differs" in r.stderr and "Traceback" not in r.stderr
    (tmp_path / "repo" / "main" / "corpus" / "N" / "N.main.ids").write_bytes(b"short")
    with pytest.raises(hf_store.VerifyError, match="not in the store as its manifest says"):
        hf_store.preflight(hf_store.open_store(str(cfg)), ["ckpt-N"], json.loads(jobs.read_text())["store_inputs"])
    assert json.loads(run("steps", "--config", str(cfg), "--branch", "ckpt-N").stdout)["head"] is None
    assert run("prune", "--config", str(cfg), "--branch", "main").returncode == 1
    # A prune by hand verifies by download: the damaged newest generation does not count as one of the two.
    push_gens(tmp_path, store, [100, 200, 300, 400])
    bad = tmp_path / "repo" / "ckpt-N" / "gens" / "g400" / "optim.safetensors"
    bad.write_bytes(bad.read_bytes()[:-1] + b"?")
    out = json.loads(run("prune", "--config", str(cfg), "--branch", "ckpt-N").stdout)
    assert out == {"branch": "ckpt-N", "verified": [200, 300], "removed": [100]}


# ---------------------------------------------------------------- the hub's commit limit (the rehearsal of 12 Nov 2026)
class Tick:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_a_save_is_one_commit_that_adds_the_generation_and_removes_the_pruned_ones(tmp_path):
    """The hub limits commit creation per repository, and a save used to cost a push and then a prune. Now the
    commit that adds a generation removes the ones behind two verified generations: the one being added and the
    newest the caller verified before that is whole in the listing. The rules are prune()'s."""
    store = dstore(tmp_path)
    push_gens(tmp_path, store, [100, 200, 300])
    n0 = len(commits(tmp_path))
    man = store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), IDENT, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [200, 300], "protect": []})
    c = commits(tmp_path)
    assert len(c) == n0 + 1 and man["pruned"] == [100, 200] and man["step"] == 400
    assert c[-1]["message"] == "generation: gens/g400" and c[-1]["deletes"] == ["gens/g100/", "gens/g200/"]
    assert "gens/g400/manifest.json" in c[-1]["adds"] and store.steps("ckpt-N") == [300, 400]
    assert "pruned" not in fresh_manifest(tmp_path, 400)             # the store's manifest is the generation's own
    # One generation is not two: with nothing verified but the one being added, nothing is removed.
    man = store.push_generation("ckpt-N", "pt", 500, staged_gen(tmp_path, 500), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": []})
    assert man["pruned"] == [] and store.steps("ckpt-N") == [300, 400, 500]
    # A generation verified once and damaged since is not whole in the listing and does not count: the two are
    # then 400 and the one being added, so only what is below 400 goes.
    (tmp_path / "repo" / "ckpt-N" / "gens" / "g500" / "model.safetensors").write_bytes(b"short")
    store = dstore(tmp_path)
    man = store.push_generation("ckpt-N", "pt", 600, staged_gen(tmp_path, 600), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [400, 500]})
    assert man["pruned"] == [300] and store.steps("ckpt-N") == [400, 500, 600]
    # Nothing at or above a mark that is staged and not yet pushed.
    man = store.push_generation("ckpt-N", "pt", 700, staged_gen(tmp_path, 700), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [400, 600], "protect": [400]})
    assert man["pruned"] == [] and store.steps("ckpt-N") == [400, 500, 600, 700]
    # Without `prune` a push removes nothing, as before.
    man = store.push_generation("ckpt-N", "pt", 800, staged_gen(tmp_path, 800), None, store.head("ckpt-N")["commit"])
    assert man["pruned"] == [] and store.steps("ckpt-N") == [400, 500, 600, 700, 800]
    # A commit that loses the race is refused whole: neither the generation nor the removal happens.
    head = store.head("ckpt-N")["commit"]
    dstore(tmp_path).push_generation("ckpt-N", "pt", 850, staged_gen(tmp_path, 850, seed=2), None, head)
    with pytest.raises(hf_store.ConflictError, match="moved"):
        store.push_generation("ckpt-N", "pt", 900, staged_gen(tmp_path, 900), None, head,
                              prune={"keep": 2, "verified": [700, 800]})
    assert store.steps("ckpt-N") == [400, 500, 600, 700, 800, 850]
    with pytest.raises(hf_store.StoreError, match="at least two"):
        store.push_generation("ckpt-N", "pt", 900, staged_gen(tmp_path, 900, seed=3), None, None, prune={"keep": 1, "verified": [800]})
    # The hub's calls: ONE create_commit, conditional on the head that was read, the generation's files as adds and
    # each pruned generation as one folder delete (the pinned library reads the trailing slash as a folder).
    api = FakeApi()
    hs, _ = hstore(tmp_path / "h", api)
    for step in (100, 200, 300):
        hs.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path / "h", step))
    api.calls.clear()
    head = hs.head("ckpt-N")
    man = hs.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path / "h", 400), None, head["commit"],
                             prune={"keep": 2, "verified": [200, 300]})
    cc = [c[1] for c in api.calls if c[0] == "create_commit"]
    assert len(cc) == 1 and cc[0]["parent"] == head["commit"] == "ckpt-N@3" and man["pruned"] == [100, 200]
    assert [o for o in cc[0]["ops"] if o[0] == "CommitOperationDelete"] == [("CommitOperationDelete", "gens/g100/"),
                                                                           ("CommitOperationDelete", "gens/g200/")]
    assert len([o for o in cc[0]["ops"] if o[0] == "CommitOperationAdd"]) == 6
    assert sorted({p.split("/")[1] for p in api.branches["ckpt-N"]}) == ["g300", "g400"]
    assert "create_branch" not in [c[0] for c in api.calls]
    from huggingface_hub import CommitOperationDelete
    assert CommitOperationDelete(path_in_repo="gens/g100/").is_folder is True
    assert CommitOperationDelete(path_in_repo="gens/g100/manifest.json").is_folder is False


def fresh_manifest(tmp_path, step):
    return json.loads((tmp_path / "repo" / "ckpt-N" / "gens" / f"g{step}" / "manifest.json").read_text())


def test_the_commit_budget_counts_a_rolling_hour_and_keeps_room_for_what_matters(tmp_path):
    clock, slept = Tick(), []

    def sleep(s):
        slept.append(s)
        clock.t += s

    ledger = str(tmp_path / "chat" / "store_commits.json")
    b = hf_store.CommitBudget(8, ledger, clock=clock, sleep=sleep)
    assert b.reserve == 2 and b.count() == 0 and b.wait_s() == 0 and b.report() is None      # the report's hour starts
    t0 = clock.t
    for _ in range(6):                                   # what is not essential stops at the budget less its reserve
        b.admit("prune", essential=False)
        clock.t += 60
    with pytest.raises(hf_store.BudgetSpent, match="room in 3240 s") as e:
        b.admit("lock", essential=False)
    assert e.value.wait_s == 3240 == b.wait_s(essential=False) and b.count() == 6          # nothing was counted
    b.admit("generation")                                # an essential commit may use the whole budget
    b.admit("mark")
    with pytest.raises(hf_store.BudgetSpent):
        b.admit("generation")
    assert b.count() == 8 and b.wait_s() == t0 + 3600 - clock.t
    # A second handle on the same file, and a bounded() child, count in the same place: one count a machine.
    b2 = hf_store.CommitBudget(8, ledger, clock=clock, sleep=sleep)
    assert b2.count() == 8 and b2.wait_s() == b.wait_s()
    clock.t = t0 + 3600 + 1                              # the first commit is an hour old: room for one essential
    assert b.wait_s() == 0 and b.wait_s(essential=False) > 0
    hf_store.bounded(lambda: b.admit("generation") or 1, 20)
    assert b.count() == 8 and b2.count() == 8
    # A lock is taken only with room for its unlock; the unlock is then counted whatever the count is.
    clock.t = t0 + 6601                                  # only the child's commit is younger than an hour
    assert b.count() == 1
    for _ in range(4):
        b.admit("files", essential=False)
    with pytest.raises(hf_store.BudgetSpent):
        b.admit("lock", essential=False, need=2)         # five sent, six allowed: room for one, not for two
    b.admit("files", essential=False, need=1)
    for _ in range(3):
        b.admit("unlock", force=True)
    assert b.count() == 9                                # forced past the budget: an admitted lock is always removed
    # An essential commit with patience waits for its turn inside the call, in short sleeps, then goes.
    wait = b.wait_s()
    assert wait > 0
    with pytest.raises(hf_store.BudgetSpent):
        b.admit("generation", patience=wait - 10)
    was = clock.t
    b.admit("generation", patience=wait + 10)
    assert wait <= clock.t - was <= wait + 10 and max(slept) <= 5
    # After a 429 nothing passes until the hub's wait is over, at least a minute.
    clock.t += 7200
    assert b.count() == 0 and b.wait_s() == 0
    b.block(1234)
    assert b.wait_s() == 1234 and b.wait_s(essential=False) == 1234
    with pytest.raises(hf_store.BudgetSpent, match="room in 1234 s"):
        b.admit("generation")
    clock.t += 1234
    b.admit("generation")
    b.block(0)
    assert b.wait_s() == hf_store.MIN_429_WAIT_S
    # The hour's report: the commits of the last hour, and what was refused and limited since the last report.
    rep = b.report()
    assert rep["commits"] == 1 and rep["budget"] == 8 and rep["refused"] >= 5 and rep["http_429"] == 2
    assert b.report() is None and b.report(0, peek=True)["refused"] == 0
    # Without a path the count is the process's own, in a file that goes with it.
    own = hf_store.CommitBudget(3)
    own.admit("generation")
    assert own.count() == 1 and os.path.exists(own.path) and own.path != ledger


def test_every_commit_of_a_store_passes_the_budget(tmp_path):
    """A push (a generation, a mark, an input set) is essential; a prune by itself and the release lock are not,
    and are refused with nothing sent while the budget is within its reserve; the lock needs room for its unlock."""
    clock = Tick()
    budget = hf_store.CommitBudget(6, str(tmp_path / "ledger.json"), clock=clock)
    store = hf_store.Store(hf_store.DirBackend(str(tmp_path / "repo")), log=lambda m: None, delay=0, budget=budget)
    push_gens(tmp_path, store, [100, 200, 300, 400])                                     # four commits
    assert budget.count() == 4 == len(commits(tmp_path)) and budget.reserve == 2
    with pytest.raises(hf_store.BudgetSpent):
        store.prune("ckpt-N", verified=[300, 400])                                       # not essential: in the reserve
    assert store.release_report("home", grace=0)["held"] is True                         # the lock: no room for two
    assert len(commits(tmp_path)) == 4 and store.steps("ckpt-N") == [100, 200, 300, 400]
    mark = tmp_path / "mark"
    mark.mkdir()
    (mark / "model.safetensors").write_bytes(S.state_at(100)[0])
    (mark / "state.txt").write_text("100\n{}\n")
    assert store.push_mark("pt", 100, str(mark)) is True                                 # essential: goes
    store.push_generation("ckpt-N", "pt", 500, staged_gen(tmp_path, 500), None, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [300, 400]})                     # and the save's own prune with it
    assert budget.count() == 6 and store.steps("ckpt-N") == [400, 500]
    with pytest.raises(hf_store.BudgetSpent, match="generation not sent"):
        store.push_generation("ckpt-N", "pt", 600, staged_gen(tmp_path, 600), None, store.head("ckpt-N")["commit"])
    assert len(commits(tmp_path)) == 6
    # An hour on: the release takes its lock and gives it back, two commits, and measures what the branches hold.
    clock.t += 3601
    rep = store.release_report("home", grace=0)
    assert rep["held"] is False and rep["objects"] == 0 and rep["tree_bytes"]["ckpt-N"] > 0 and rep["unnamed_bytes"] == 0
    assert [c["message"].split(":")[0] for c in commits(tmp_path)[-2:]] == ["lock", "unlock"] and budget.count() == 2
    # With no budget a Store commits as it always did.
    plain = dstore(tmp_path / "plain")
    push_gens(tmp_path / "plain", plain, [100, 200, 300])
    assert plain.budget is None and plain.prune("ckpt-N", verified=[200, 300]) == [100]
    # A config's budget: its size, and one ledger beside the config for every process that reads it.
    cfg = tmp_path / "vm" / "store.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'repo2'}", "store_commit_budget_per_hour": 5}))
    s2 = hf_store.open_store(str(cfg))
    assert s2.budget.per_hour == 5 and s2.budget.path == str(cfg.parent / "store_commits.json")
    assert hf_store.open_store({"repo": f"dir:{tmp_path / 'repo2'}"}).budget is None
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ids").write_bytes(b"a")
    tool = [sys.executable, str(ROOT / "tools" / "colab_chat" / "hf_store.py"), "push", "--config", str(cfg), "--branch", "main",
            "--dir", str(src), "--limit", "60"]
    for i in range(2):                                  # the data job's separate processes share the count
        r = subprocess.run([*tool, "--prefix", f"set{i}", "--commit-budget", "2"], capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(r.stderr.strip().splitlines()[-1][5:])["commits"] == i + 1, r.stderr
    r = subprocess.run([*tool, "--prefix", "set2", "--commit-budget", "2"], capture_output=True, text=True)
    assert r.returncode == 1 and "BudgetSpent" in r.stderr and "room in" in r.stderr
    assert not (tmp_path / "repo2" / "main" / "set2").exists() and s2.budget.count() == 2


def hub_store(tmp_path, **kw):
    """The real HfBackend over the on-disk stand-in for the hub (tests/hub_standin.py)."""
    import hub_standin
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "hub"
    tf = tmp_path / "hub.token"
    tf.write_text(TOKEN + "\n")
    tf.chmod(0o600)
    logs = []
    store = hf_store.open_store({"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tf)},
                                api=hub_standin.DiskApi(str(root)), log=logs.append, delay=0, **kw)
    return store, root, logs


def hub_commits(root):
    p = root / "commit_log.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def test_the_hubs_429_is_read_with_its_retry_after(tmp_path):
    """The hub refuses a commit past its limit with HTTP 429, as the library raises it (HfHubHTTPError with the
    response). The store reads Retry-After, raises RateLimited with it, and lets nothing through its budget for at
    least that long; the class and the wait survive a bounded() child; a 412 stays what it was."""
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import HfHubHTTPError
    clock = Tick()
    budget = hf_store.CommitBudget(50, str(tmp_path / "ledger.json"), clock=clock)
    store, root, _ = hub_store(tmp_path, budget=budget)
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").write_text(json.dumps({"per_window": 2, "retry_after": 1234}))
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ids").write_bytes(b"a")
    store.push_set("main", "set0", str(src))
    store.push_set("main", "set1", str(src))
    with pytest.raises(hf_store.RateLimited) as e:
        store.push_set("main", "set2", str(src))
    assert e.value.retry_after == 1234 and "HTTP 429" in str(e.value) and TOKEN not in str(e.value)
    assert "Too Many Requests" in str(e.value)                                  # the first line of the hub's own text
    assert [c["ok"] for c in hub_commits(root)] == [True, True, False]
    assert budget.wait_s() == 1234 and budget.wait_s(essential=False) == 1234
    with pytest.raises(hf_store.BudgetSpent, match="room in 1234 s"):           # not sent: the hub sees no fourth try
        store.push_set("main", "set2", str(src))
    assert len(hub_commits(root)) == 3
    # Through a bounded() child (how the courier and the runner call): the class and the wait come back.
    clock.t += 1300
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").write_text(json.dumps({"per_window": 2, "retry_after": "auto"}))
    with pytest.raises(hf_store.RateLimited) as e:
        hf_store.bounded(lambda: bool(store.push_set("main", "set2", str(src))), 30, TOKEN)
    assert e.value.retry_after == 3600 - 1300 == budget.wait_s()
    # A 429 that names no wait: at least a minute.
    clock.t += 2300
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": None}))
    with pytest.raises(hf_store.RateLimited) as e:
        store.push_set("main", "set2", str(src))
    assert e.value.retry_after == hf_store.MIN_429_WAIT_S == budget.wait_s()
    # With room again the same set goes, and what the refusals left behind is nothing.
    clock.t += 3600
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").unlink()
    store.push_set("main", "set2", str(src))
    assert sorted(os.listdir(root / "branches" / "main")) == ["set0", "set1", "set2"]
    # A stale parent is the library's own error still (the courier reads the head again); it is not a rate limit.
    with pytest.raises(HfHubHTTPError, match="412"):
        store.commit("main", {}, ["set0/a.ids"], "x: y", parent="main@0")
    assert hf_store.rate_limited(RuntimeError("429 Client Error: Too Many Requests")) is None
    assert hf_store.rate_limited(ValueError("line 429 of a file")) is None
    # A store without a budget reads the 429 just the same.
    plain, root2, _ = hub_store(tmp_path / "p")
    (root2 / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": 77}))
    with pytest.raises(hf_store.RateLimited) as e:
        plain.push_set("main", "set0", str(src))
    assert e.value.retry_after == 77


def test_release_through_the_real_backend_frees_only_what_no_branch_names(tmp_path):
    """The save's one commit and the release from home, through HfBackend over the stand-in hub, whose object
    store keeps what a removed generation held until it is deleted for good. A release frees exactly the removed
    generations' objects, reports the bytes, measures what is left, and what is left still resumes."""
    pytest.importorskip("huggingface_hub")
    store, root, _ = hub_store(tmp_path)
    store.b.ensure_branch("ckpt-N")
    for step in (100, 200, 300):
        store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step), IDENT, store.head("ckpt-N")["commit"])
    store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), IDENT, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [200, 300]})
    log = hub_commits(root)
    assert len(log) == 4 and log[-1]["deletes"] == ["gens/g100/", "gens/g200/"] and store.steps("ckpt-N") == [300, 400]
    objects = json.loads((root / "lfs.json").read_text())
    assert len(objects) == 8                                         # the removed generations' bytes are still stored
    gen = {s: sum(len(b) for b in S.state_at(s)) for s in (100, 200, 300, 400)}
    held = store.release_report("home", grace=3600)                  # too young: nothing freed, and said so
    assert (held["objects"], held["bytes"], held["unnamed_bytes"]) == (0, 0, gen[100] + gen[200])
    rep = store.release_report("home", grace=0, now=time.time() + 5)
    assert (rep["held"], rep["objects"], rep["bytes"]) == (False, 4, gen[100] + gen[200])
    assert rep["stored_bytes"] == gen[300] + gen[400] == rep["tree_bytes"]["ckpt-N"] and rep["unnamed_bytes"] == 0
    assert set(json.loads((root / "lfs.json").read_text())) == {S.sha(b) for s in (300, 400) for b in S.state_at(s)}
    assert [c["message"].split(":")[0] for c in hub_commits(root)[4:]] == ["lock", "unlock", "lock", "unlock"]
    assert not (root / "branches" / "main" / "locks" / "release.json").exists()
    step, rejected = hub_store(tmp_path)[0].pull_generation("ckpt-N", str(tmp_path / "r"), expect=IDENT)
    assert (step, rejected) == (400, [])
    assert hub_store(tmp_path)[0].verify_by_download("ckpt-N", 300, expect=IDENT)
    calls = (root / "calls.log").read_text().splitlines()
    assert sum(ln.startswith("create_branch") for ln in calls) == 1  # once, for the branch; never before a commit


# ---------------------------------------------------------------- base2_seed.py
def test_seed_pushes_once_and_never_overwrites(tmp_path):
    w, v, eng = tmp_path / "tc_b1_r1.safetensors", tmp_path / "vocab.gguf", tmp_path / "engine"
    w.write_bytes(b"weights" * 100)
    v.write_bytes(b"vocab")
    eng.mkdir()
    for f in ("banks.py", "engine.py", "frames.py", "generate.py", "validate.py", "test_scenario_engine.py", "README.md"):
        (eng / f).write_text(f"# {f}\n")
    rec = tmp_path / "record.json"
    args = ["--store-repo", f"dir:{tmp_path / 'repo'}", "--weights", str(w), "--vocab", str(v), "--engine-dir", str(eng),
            "--record", str(rec)]
    assert base2_seed.main(args + ["--dry-run"]) == 0 and not rec.exists() and not (tmp_path / "repo").exists()
    assert base2_seed.main(args) == 0
    r = json.loads(rec.read_text())
    assert r["result"] == {"seed/weights": "pushed", "seed/vocab": "pushed", "seed/scenario_engine": "pushed"}
    assert sorted(r["sets"]["seed/scenario_engine"]) == ["banks.py", "engine.py", "frames.py", "generate.py", "validate.py"]
    assert r["sets"]["seed/weights"]["tc_b1_r1.safetensors"]["sha256"] == S.sha(b"weights" * 100)
    assert len(commits(tmp_path)) == 3
    got = tmp_path / "got"
    dstore(tmp_path).pull_set("main", "seed/weights", str(got))
    assert (got / "tc_b1_r1.safetensors").read_bytes() == b"weights" * 100
    assert base2_seed.main(args) == 0 and len(commits(tmp_path)) == 3  # the same bytes: nothing sent
    assert json.loads(rec.read_text())["result"]["seed/weights"] == "present"
    w.write_bytes(b"other")
    assert base2_seed.main(args) == 1 and len(commits(tmp_path)) == 3  # other bytes: refused, nothing overwritten
    # No default repo or token: both must be given.
    with pytest.raises(SystemExit):
        base2_seed.main(["--weights", str(w), "--record", str(rec)])
    assert base2_seed.main(["--store-repo", "someone/base2-store", "--vocab", str(v), "--record", str(rec)]) == 1
