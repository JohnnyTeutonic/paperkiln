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
    assert c["deletes"] == ["gens/g100/", "gens/g200/"]                           # each generation as its folder
    assert c["adds"] == ["pruned/g100.json", "pruned/g200.json"]                  # and its ledger, in the same commit
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

    def preupload_lfs_files(self, *, repo_id, additions, token=None, repo_type=None, revision=None, create_pr=None,
                            num_threads=5, free_memory=True, gitignore_content=None):
        self._rec("preupload_lfs_files", repo=repo_id, revision=revision, repo_type=repo_type, token=token,
                  paths=[o.path_in_repo for o in additions])
        self._done("preupload_lfs_files")

    @property
    def http_session(self):
        """The ranged one-byte GET of HfBackend.probe: 206, or the object store's 404 after a redirect for an LFS
        file whose object is gone, or the hub's own 404 for a path that is not there."""
        api = self

        class Session:
            def get(self, url, headers=None, timeout=None, stream=False, allow_redirects=True):
                import io
                import urllib.parse
                import requests
                revision, path = url.split("/resolve/", 1)[1].split("/", 1)
                revision = urllib.parse.unquote(revision)    # the library quotes it: a commit id is "<branch>@<n>" here
                api._rec("probe", repo="someone/base2-store", revision=revision, filename=path, range=headers.get("Range"),
                         repo_type="dataset" if "/datasets/" in url else "model",
                         token=str(headers.get("authorization") or headers.get("Authorization")).replace("Bearer ", ""))
                resp = requests.Response()
                resp.url, resp.raw = url, io.BytesIO(b"")
                files = api._files(revision)
                if path not in files:
                    resp.status_code = 404
                    resp.headers["X-Error-Code"] = "EntryNotFound"
                elif path.endswith(".safetensors") and S.sha(files[path]) not in api.lfs:
                    first = requests.Response()
                    first.status_code = 302
                    resp.status_code, resp.history, resp.url = 404, [first], "https://cdn-lfs.example/" + S.sha(files[path])
                else:
                    resp.status_code, resp.raw = 206, io.BytesIO(files[path][:1])
                    resp.headers["Content-Range"] = f"bytes 0-0/{len(files[path])}"
                return resp
        return Session()

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
    clock = kw.pop("clock", None)  # the time a removed generation's ledger carries (the fake hub's own clock)
    cfg = {"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tf), **kw}
    return hf_store.open_store(cfg, log=logs.append, api=api, delay=0, **({"clock": clock} if clock else {})), logs


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
    store, logs = hstore(tmp_path, api, clock=lambda: api.now)
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
    """A release deletes only what a commit once named and a later commit removed: the objects in a removed
    generation's ledger, once the ledger is older than the grace, that no branch's tree names and no other ledger
    lists. A mark keeps its generation's weights, another branch keeps what it names, and an object in no tree and
    no ledger (another writer's upload before its commit) is NEVER deleted, however old. A held lock means no
    release."""
    api, store, logs = release_setup(tmp_path)
    assert store.prune("ckpt-N", verified=[200, 300]) == [100]
    m100, o100 = S.state_at(100)
    ledger = json.loads(api.branches["ckpt-N"]["pruned/g100.json"])    # written by the commit that removed g100
    assert ledger["step"] == 100 and ledger["at"] == api.now and ledger["format"] == hf_store.PRUNED_FORMAT
    assert sorted((o["path"], o["sha256"], o["bytes"]) for o in ledger["objects"]) == [
        ("gens/g100/model.safetensors", S.sha(m100), len(m100)), ("gens/g100/optim.safetensors", S.sha(o100), len(o100))]
    uncommitted = S.sha(b"another arm's upload, not committed yet")
    api.lfs[uncommitted] = api.now - 30 * 86400                        # a month old, and in no tree and no ledger
    api.sizes[uncommitted] = 39
    api.calls.clear()
    # Too early: the ledger is younger than the grace period, nothing is deleted.
    early = store.release_report("boot-a", now=api.now + 60)
    assert early["objects"] == 0 and early["pending"] == {"ckpt-N": {S.sha(o100): len(o100)}} and early["ledgers_done"] == 0
    assert "permanently_delete_lfs_files" not in [c[0] for c in api.calls]
    with pytest.raises(hf_store.StoreError, match="under the minimum of 600 s"):
        store.release_report("boot-a", grace=599, now=api.now + 10 ** 6)
    # Later: only the removed generation's optimiser goes. Its weights are the mark's; the old upload is nobody's.
    api.calls.clear()
    rep = store.release_report("boot-a", now=api.now + hf_store.RELEASE_GRACE_S + 600)
    assert (rep["objects"], rep["bytes"]) == (1, len(o100)) and rep["pending"] == {} and rep["ledgers_done"] == 1
    assert (rep["orphan_objects"], rep["orphan_bytes"]) == (1, 39)
    dele = dict(api.calls)["permanently_delete_lfs_files"]
    assert dele["oids"] == [S.sha(o100)] and dele["rewrite_history"] is False
    for kept in (S.sha(m100), S.sha(S.state_at(200)[0]), S.sha(b"build bytes"), uncommitted):
        assert kept in api.lfs
    # The lock: taken by a conditional commit on main, removed afterwards; every branch's tree was read. The
    # removal's commit also records the ledger that is finished with (its objects gone, or named by a tree).
    cc = [c[1] for c in api.calls if c[0] == "create_commit"]
    assert cc[0]["ops"] == [("CommitOperationAdd", hf_store.LOCK)] and cc[0]["parent"] is not None
    assert cc[-1]["ops"] == [("CommitOperationAdd", hf_store.RELEASED), ("CommitOperationDelete", hf_store.LOCK)]
    assert hf_store.LOCK not in api.branches["main"] and json.loads(api.branches["main"][hf_store.RELEASED]) == {"ckpt-N": [100]}
    listed = {c[1]["revision"] for c in api.calls if c[0] == "list_repo_tree"}       # each at the commit read as its head
    assert {r.split("@")[0] for r in listed} >= {"main", "ckpt-N", "ckpt-S", "base2-build"} and all("@" in r for r in listed)
    # The branch's next save removes the finished ledger in its own commit; the release after that drops the record.
    store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), None, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [200, 300]})
    assert "pruned/g100.json" not in api.branches["ckpt-N"] and store.steps("ckpt-N") == [200, 300, 400]
    assert store.release("boot-a", now=api.now + hf_store.RELEASE_GRACE_S + 600) == 0
    assert hf_store.RELEASED not in api.branches["main"]
    # By hand, with no VM writing: --orphans also takes what is in no tree and no ledger.
    assert store.release_report("boot-a", now=api.now + hf_store.RELEASE_GRACE_S + 600, orphans=True)["objects"] == 1
    assert uncommitted not in api.lfs and S.sha(m100) in api.lfs
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
    assert store.release("a", now=api.now + hf_store.RELEASE_GRACE_S + 600) == 1
    assert S.sha(S.state_at(100)[0]) in api.lfs and S.sha(S.state_at(100)[1]) not in api.lfs
    api2, store2, _ = release_setup(tmp_path / "b")
    assert store2.prune("ckpt-N", verified=[200, 300]) == [100]
    real = api2.list_lfs_files
    api2.list_lfs_files = lambda **kw: [type("O", (), {"oid": "x" + f.oid[1:], "file_oid": "y", "pushed_at": f.pushed_at})()
                                        for f in real(**kw)]
    with pytest.raises(hf_store.StoreError, match="release refused"):
        store2.release("a", now=api2.now + hf_store.RELEASE_GRACE_S + 600)
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
    rep = store.release_report("boot-a", now=api.now + hf_store.RELEASE_GRACE_S + 600)
    assert (rep["held"], rep["objects"], rep["bytes"]) == (False, 1, len(S.state_at(100)[1]))
    # What the store then holds, for the storage guard at home, each object by its sha256 so that it counts once:
    # what each branch names under gens/, the marks by their folder, nothing pending and no orphan.
    files = {s: {S.sha(b): len(b) for b in S.state_at(s)} for s in (100, 200, 300)}
    assert rep["gens"]["ckpt-N"] == {**files[200], **files[300]} and rep["gens"]["ckpt-S"] == files[200]
    assert rep["gens"]["base2-build"] == {} and "main" not in rep["gens"]
    m100 = S.state_at(100)[0]
    assert rep["marks"] == {"pt": {S.sha(m100): len(m100)}} and rep["pending"] == {} and rep["orphan_objects"] == 0
    # Two campaigns that hold the same bytes each count them; one campaign counts an object once, by whatever path.
    assert hf_store.campaign_bytes(rep, ["ckpt-N"], ["pt"]) == sum(files[200].values()) + sum(files[300].values()) + len(m100)
    assert hf_store.campaign_bytes(rep, ["ckpt-S"], []) == sum(files[200].values())
    assert hf_store.campaign_bytes(rep, ["ckpt-N", "ckpt-S"], ["pt"]) == hf_store.campaign_bytes(rep, ["ckpt-N"], ["pt"])
    # A size the store does not give is not taken for zero: the total is unknown, and said to be.
    assert rep["unknown_sizes"] == 1 and rep["stored_bytes"] is None      # the build branch's object carries none
    api.sizes[S.sha(b"build bytes")] = len(b"build bytes")
    again = other.release_report("boot-b", now=api.now + hf_store.RELEASE_GRACE_S + 600)
    assert (again["held"], again["objects"], again["bytes"], again["unknown_sizes"]) == (False, 0, 0, 0)
    assert again["stored_bytes"] == sum(api.sizes[o] for o in api.lfs)
    # An object no tree and no ledger names is still stored and is reported, never deleted, whatever its age.
    young = b"an upload that is not committed yet"
    api.lfs[S.sha(young)], api.sizes[S.sha(young)] = api.now - 10 ** 7, len(young)
    orphan = other.release_report("boot-b", now=api.now + hf_store.RELEASE_GRACE_S + 600)
    assert (orphan["objects"], orphan["orphan_objects"], orphan["orphan_bytes"]) == (0, 1, len(young)) and S.sha(young) in api.lfs
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
    # What the code says of itself: a release never takes an upload from under its commit; only --orphans can.
    assert "is never deleted by a release" in " ".join(hf_store.__doc__.split())
    assert "WITH NO VM WRITING" in hf_store.Store.release_report.__doc__
    assert "THE BUDGET IS PER WRITER, NOT PER REPOSITORY" in hf_store.__doc__


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


def fresh_manifest(tmp_path, step):
    return json.loads((tmp_path / "repo" / "ckpt-N" / "gens" / f"g{step}" / "manifest.json").read_text())


def test_a_save_is_one_commit_that_adds_the_generation_and_removes_the_pruned_ones(tmp_path):
    """The hub limits commit creation per repository, and a save used to cost a push and then a prune. Now the
    commit that adds a generation removes the ones behind the two newest generations the caller has verified that
    are whole in the listing. The generation being added never counts (whether it arrived whole is known only
    from a listing after its commit), so one more generation stays until the next save. The same commit writes the
    ledger of each generation it removes. The rules are prune()'s."""
    store = dstore(tmp_path)
    push_gens(tmp_path, store, [100, 200, 300])
    n0 = len(commits(tmp_path))
    man = store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), IDENT, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [200, 300], "protect": []})
    c = commits(tmp_path)
    assert len(c) == n0 + 1 and man["pruned"] == [100] and man["step"] == 400      # behind g200 and g300; not g200
    assert c[-1]["message"] == "generation: gens/g400" and c[-1]["deletes"] == ["gens/g100/"]
    assert "gens/g400/manifest.json" in c[-1]["adds"] and "pruned/g100.json" in c[-1]["adds"]
    assert store.steps("ckpt-N") == [200, 300, 400] and hf_store.ledger_steps(store.b.tree("ckpt-N")) == [100]
    assert "pruned" not in fresh_manifest(tmp_path, 400)             # the store's manifest is the generation's own
    assert store.listed_whole("ckpt-N", 400, man["files"]) and not store.listed_whole("ckpt-N", 400, {"x": 1})
    # With the two before it verified and the new one NOT counted, the same call cannot remove g200 however often it
    # is made: nothing verified but one generation, nothing removed.
    man = store.push_generation("ckpt-N", "pt", 500, staged_gen(tmp_path, 500), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [400]})
    assert man["pruned"] == [] and store.steps("ckpt-N") == [200, 300, 400, 500]
    # A generation verified once and damaged since is not whole in the listing and does not count: the two are
    # then g300 and g400, so only what is below g300 goes.
    (tmp_path / "repo" / "ckpt-N" / "gens" / "g500" / "model.safetensors").write_bytes(b"short")
    store = dstore(tmp_path)
    assert not store.listed_whole("ckpt-N", 500)
    man = store.push_generation("ckpt-N", "pt", 600, staged_gen(tmp_path, 600), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [300, 400, 500]})
    assert man["pruned"] == [200] and store.steps("ckpt-N") == [300, 400, 500, 600]
    # Nothing at or above a mark that is staged and not yet pushed.
    man = store.push_generation("ckpt-N", "pt", 700, staged_gen(tmp_path, 700), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [400, 600], "protect": [300]})
    assert man["pruned"] == [] and store.steps("ckpt-N") == [300, 400, 500, 600, 700]
    # Without `prune` a push removes nothing, as before.
    man = store.push_generation("ckpt-N", "pt", 800, staged_gen(tmp_path, 800), None, store.head("ckpt-N")["commit"])
    assert man["pruned"] == [] and store.steps("ckpt-N") == [300, 400, 500, 600, 700, 800]
    # A commit that loses the race is refused whole: neither the generation nor the removal nor a ledger happens.
    head = store.head("ckpt-N")["commit"]
    dstore(tmp_path).push_generation("ckpt-N", "pt", 850, staged_gen(tmp_path, 850, seed=2), None, head)
    with pytest.raises(hf_store.ConflictError, match="moved"):
        store.push_generation("ckpt-N", "pt", 900, staged_gen(tmp_path, 900), None, head,
                              prune={"keep": 2, "verified": [700, 800]})
    assert store.steps("ckpt-N") == [300, 400, 500, 600, 700, 800, 850]
    assert hf_store.ledger_steps(store.b.tree("ckpt-N")) == [100, 200]
    with pytest.raises(hf_store.StoreError, match="at least two"):
        store.push_generation("ckpt-N", "pt", 900, staged_gen(tmp_path, 900, seed=3), None, None, prune={"keep": 1, "verified": [800]})
    # The hub's calls: ONE create_commit, conditional on the head that was read, the generation's files and the
    # ledger as adds and each pruned generation as one folder delete (the pinned library reads the trailing slash
    # as a folder).
    api = FakeApi()
    hs, _ = hstore(tmp_path / "h", api)
    for step in (100, 200, 300):
        hs.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path / "h", step))
    api.calls.clear()
    head = hs.head("ckpt-N")
    man = hs.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path / "h", 400), None, head["commit"],
                             prune={"keep": 2, "verified": [200, 300]})
    cc = [c[1] for c in api.calls if c[0] == "create_commit"]
    assert len(cc) == 1 and cc[0]["parent"] == head["commit"] == "ckpt-N@3" and man["pruned"] == [100]
    assert [o for o in cc[0]["ops"] if o[0] == "CommitOperationDelete"] == [("CommitOperationDelete", "gens/g100/")]
    adds = [o[1] for o in cc[0]["ops"] if o[0] == "CommitOperationAdd"]
    assert len(adds) == 7 and "pruned/g100.json" in adds
    assert sorted({p.split("/")[1] for p in api.branches["ckpt-N"] if p.startswith("gens/")}) == ["g200", "g300", "g400"]
    assert "create_branch" not in [c[0] for c in api.calls]
    assert hs.listed_whole("ckpt-N", 400, man["files"])              # the listing after the commit: sizes and LFS hashes
    api.branches["ckpt-N"]["gens/g400/optim.safetensors"] = b"other bytes of some length"
    assert not hs.listed_whole("ckpt-N", 400, man["files"])          # a file of another hash is not whole
    from huggingface_hub import CommitOperationDelete
    assert CommitOperationDelete(path_in_repo="gens/g100/").is_folder is True
    assert CommitOperationDelete(path_in_repo="gens/g100/manifest.json").is_folder is False


def test_the_commit_budget_counts_a_rolling_hour_and_keeps_room_for_what_matters(tmp_path):
    clock, slept = Tick(), []

    def sleep(s):
        slept.append(s)
        clock.t += s

    ledger = str(tmp_path / "chat" / "store_commits.json")
    # essential_wait as long as the hour: the strict count, without the allowance tested further down.
    b = hf_store.CommitBudget(8, ledger, clock=clock, sleep=sleep, essential_wait=3600)
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
    # A second handle on the same file, and a bounded() child, count in the same place: one count a writer.
    b2 = hf_store.CommitBudget(8, ledger, clock=clock, sleep=sleep, essential_wait=3600)
    assert b2.count() == 8 and b2.wait_s() == b.wait_s()
    clock.t = t0 + 3600 + 1                              # the first commit is an hour old: room for one essential
    assert b.wait_s() == 0 and b.wait_s(essential=False) > 0
    hf_store.bounded(lambda: b.admit("generation") or 1, 20)
    assert b.count() == 8 and b2.count() == 8
    # A lock is taken only with room for its unlock, and both are counted at once (the review of 12 Nov 2026: room
    # was checked for two and one was counted, so another process could take the unlock's place). The unlock that
    # was counted then is not counted again; an unlock tried a second time is, whatever the count.
    clock.t = t0 + 6601                                  # only the child's commit is younger than an hour
    assert b.count() == 1
    for _ in range(3):
        b.admit("files", essential=False)
    b.admit("lock", essential=False, need=2)             # four sent, six allowed: room for two
    assert b.count() == 6
    b.admit("unlock", essential=False, reserved=True)
    assert b.count() == 6
    with pytest.raises(hf_store.BudgetSpent):
        b.admit("lock", essential=False, need=2)
    with pytest.raises(hf_store.BudgetSpent):
        b.admit("files", essential=False)
    for _ in range(3):
        b.admit("unlock", essential=False, force=True)
    assert b.count() == 9                                # forced past the budget: an admitted lock is always removed
    # An essential commit with patience waits for its turn inside the call, in short sleeps, then goes.
    wait = b.wait_s()
    assert wait > 0
    with pytest.raises(hf_store.BudgetSpent):
        b.admit("generation", patience=wait - 10)
    was = clock.t
    b.admit("generation", patience=wait + 10)
    assert wait <= clock.t - was <= wait + 10 and max(slept) <= 5
    # After a 429 nothing passes until the hub's wait is over: at least a minute, at most half an hour, and a
    # reserved or forced unlock waits it out too (it used to go through).
    clock.t += 7200
    assert b.count() == 0 and b.wait_s() == 0
    b.block(1234)
    assert b.wait_s() == 1234 and b.wait_s(essential=False) == 1234
    for kw in ({}, {"essential": False, "reserved": True}, {"essential": False, "force": True}):
        with pytest.raises(hf_store.BudgetSpent, match="room in 1234 s"):
            b.admit("generation", **kw)
    clock.t += 1234
    b.admit("generation")
    b.block(0)
    assert b.wait_s() == hf_store.MIN_429_WAIT_S == 60
    clock.t += 60
    b.block(86400)                                       # a day: honoured for half an hour
    assert b.wait_s() == hf_store.MAX_429_WAIT_S == 1800
    b.block(float("inf"))
    assert b.wait_s() == 1800
    clock.t += 1800
    # The hour's report: the commits of the last hour, and what was refused and limited since the last report.
    rep = b.report()
    assert rep["commits"] == 1 and rep["budget"] == 8 and rep["refused"] >= 8 and rep["http_429"] == 4
    assert b.report() is None and b.report(0, peek=True)["refused"] == 0
    # Without a path the count is the process's own, in a file that goes with it.
    own = hf_store.CommitBudget(3)
    own.admit("generation")
    assert own.count() == 1 and os.path.exists(own.path) and own.path != ledger


def test_an_essential_commit_waits_fifteen_minutes_for_the_budget_and_then_goes(tmp_path):
    """The second review (12 Nov 2026): a budget wait could reach an hour and a 429's any length, while the boot
    takes the token away a fixed time after the jobs end. An essential commit now waits for room at most
    ESSENTIAL_WAIT_S from the moment it first asked, then passes whatever the count is, and the hub answers."""
    clock = Tick()
    b = hf_store.CommitBudget(4, str(tmp_path / "ledger.json"), clock=clock)
    assert b.essential_wait == hf_store.ESSENTIAL_WAIT_S == 900
    t0 = clock.t
    for _ in range(4):
        b.admit("generation")
    assert b.wait_s() == 900                             # the hour's room is 3,600 s away; the allowance is 900
    with pytest.raises(hf_store.BudgetSpent, match="room in 900 s"):
        b.admit("generation")
    clock.t = t0 + 600
    assert b.wait_s() == 300
    assert b.wait_s(essential=False) > 2900              # what is not essential has no allowance
    clock.t = t0 + 900
    assert b.wait_s() == 0
    b.admit("generation")                                # past the budget, on purpose: five in the hour
    assert b.count() == 5
    assert b.wait_s() == 900                             # and the next one's own wait starts now
    clock.t = t0 + 1000
    with pytest.raises(hf_store.BudgetSpent, match="room in 800 s"):
        b.admit("mark")
    # A wait that ended because room came is forgotten: a later one starts from its own first asking.
    clock.t = t0 + 3601 + 900
    assert b.wait_s() == 0
    b.admit("generation")
    # The worst case at the end of a run: a 429's wait (30 minutes at most) and then the budget's (15), inside the
    # 90 minutes the boot gives the courier, with a whole push limit to spare.
    assert hf_store.MAX_429_WAIT_S + hf_store.ESSENTIAL_WAIT_S + 1800 <= 5400
    # The 429's wait is read from seconds or from an HTTP date.
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    def refusal(**headers):
        resp = requests.Response()
        resp.status_code = 429
        resp.headers.update(headers)
        return HfHubHTTPError("429 Client Error: Too Many Requests", response=resp)

    assert hf_store.rate_limited(refusal(**{"Retry-After": "120"})) == 120
    assert hf_store.rate_limited(refusal(**{"Retry-After": "Thu, 12 Nov 2026 05:10:00 GMT"}), now=1794460200.0 - 600) == \
        __import__("email.utils").utils.parsedate_to_datetime("Thu, 12 Nov 2026 05:10:00 GMT").timestamp() - (1794460200.0 - 600)
    when = __import__("email.utils").utils.formatdate(time.time() + 500, usegmt=True)
    assert 495 <= hf_store.rate_limited(refusal(**{"Retry-After": when})) <= 501
    assert hf_store.rate_limited(refusal(**{"Retry-After": "Thu, 12 Nov 1998 05:10:00 GMT"})) == 0     # a date gone by
    assert hf_store.rate_limited(refusal(**{"Retry-After": "soon"})) == 0 and hf_store.rate_limited(refusal()) == 0
    assert hf_store.rate_limited(refusal(**{"RateLimit": '"api";r=0;t=77'})) == 77
    assert hf_store.capped_429_wait(10 ** 9) == 1800 and hf_store.capped_429_wait(None) == 60 and hf_store.capped_429_wait("x") == 60


CONTEND = """import sys, time
sys.path.insert(0, {cc!r})
import hf_store
b = hf_store.CommitBudget(int(sys.argv[2]), sys.argv[1], essential_wait=3600)
while time.time() < float(sys.argv[3]):
    time.sleep(0.001)
got = 0
for _ in range(int(sys.argv[4])):
    try:
        b.admit("generation", essential=sys.argv[5] == "essential", need=int(sys.argv[6]))
        got += int(sys.argv[6])
    except hf_store.BudgetSpent:
        pass
print(got)
"""


def test_the_budget_holds_with_several_processes_at_once_and_with_a_ledger_that_is_wrong(tmp_path):
    """One file, several writers' processes at the same moment: exactly the budget passes, however they interleave
    (and a lock's two are taken together or not at all). A ledger that is not what it should be never ends the
    caller and never opens the budget wide: it is moved aside, and the new one counts half the budget as used. A
    clock that jumps back does not stretch a wait past its bounds; one that jumps forward ages the count."""
    ledger = str(tmp_path / "vm" / "store_commits.json")
    script = tmp_path / "contend.py"
    script.write_text(CONTEND.format(cc=str(ROOT / "tools" / "colab_chat")))
    start = time.time() + 1.5
    procs = [subprocess.Popen([sys.executable, str(script), ledger, "24", str(start), "10", "essential", "1"],
                              stdout=subprocess.PIPE, text=True) for _ in range(8)]
    got = [int(p.communicate(timeout=120)[0]) for p in procs]
    assert sum(got) == 24 and max(got) <= 10, got       # eighty asked at once; the budget passed, no more
    b = hf_store.CommitBudget(24, ledger)
    assert b.count() == 24
    ledger2 = str(tmp_path / "home" / "store_commits.json")
    start = time.time() + 1.5
    procs = [subprocess.Popen([sys.executable, str(script), ledger2, "9", str(start), "5", "lock", "2"],
                              stdout=subprocess.PIPE, text=True) for _ in range(6)]
    got = [int(p.communicate(timeout=120)[0]) for p in procs]
    b2 = hf_store.CommitBudget(9, ledger2)
    assert b2.reserve == 2 and sum(got) == 6 == b2.count() and all(g % 2 == 0 for g in got), got    # three locks, each with its unlock

    # A ledger that is wrong, each way. None raises; each is moved aside; the count starts at half the budget.
    clock = Tick()
    bad = [b"{not json", b"[1, 2, 3]", json.dumps({"commits": "many"}).encode(), json.dumps({"commits": [["soon", "x"]]}).encode(),
           json.dumps({"commits": [[1.0, "x"]], "blocked_until": "later"}).encode(),
           b'{"commits": [], "blocked_until": Infinity}', b'{"commits": [[NaN, "x"]]}',
           json.dumps({"commits": [], "refused": -3}).encode(), json.dumps({"commits": [], "essential_since": [1]}).encode(),
           b"\x00\xff\xfe binary"]
    for i, raw in enumerate(bad):
        p = tmp_path / f"bad{i}" / "store_commits.json"
        p.parent.mkdir()
        p.write_bytes(raw)
        bb = hf_store.CommitBudget(8, str(p), clock=clock)
        assert bb.count() == 4, raw                      # half of eight: neither empty nor closed
        assert bb.wait_s() == 0 and bb.wait_s(essential=False) == 0
        bb.admit("generation")
        assert bb.count() == 5 and json.loads(p.read_text())["commits"][-1][1] == "generation"
        aside = [f for f in os.listdir(p.parent) if ".bad" in f]
        assert len(aside) == 1 and (p.parent / aside[0]).read_bytes() == raw, raw
        clock.t += 3601
        assert bb.count() == 0                           # and what it assumed ages out like anything else
    # A ledger that cannot be written at all (its folder is a file): counted in memory, nothing raised.
    (tmp_path / "blocker").write_text("x")
    mem = hf_store.CommitBudget(2, str(tmp_path / "blocker" / "store_commits.json"), clock=clock)
    mem.admit("generation")
    mem.admit("generation")
    assert mem.count() == 2 and mem.report(0, peek=True)["commits"] == 2
    with pytest.raises(hf_store.BudgetSpent):
        mem.admit("generation")
    mem.note_branch("r", "ckpt-N")
    assert mem.knows_branch("r", "ckpt-N") is False      # nowhere to remember it: asked again, never an error

    # The clock. Set back: times that are now in the future are brought to now, so a wait is never longer than its
    # own bound. Set forward: the count ages early, which the hub's own 429 then answers.
    clock = Tick()
    cb = hf_store.CommitBudget(4, str(tmp_path / "clock" / "store_commits.json"), clock=clock, essential_wait=3600)
    for _ in range(4):
        cb.admit("generation")
    cb.block(1000)
    clock.t -= 86400                                     # a day back
    assert cb.count() == 4 and cb.wait_s() == 3600       # not a day and an hour: the hour, from now
    stored = json.loads(open(cb.path).read())
    assert all(c[0] == clock.t for c in stored["commits"]) and stored["blocked_until"] <= clock.t + hf_store.MAX_429_WAIT_S
    clock.t += 3600
    assert cb.count() == 0 and cb.wait_s() == 0
    for _ in range(4):
        cb.admit("generation")
    clock.t += 10 * 86400                                # ten days forward
    assert cb.count() == 0 and cb.wait_s() == 0


def test_every_commit_of_a_store_passes_the_budget(tmp_path):
    """A push (a generation, a mark, an input set) is essential; a prune by itself and the release lock are not,
    and are refused with nothing sent while the budget is within its reserve; the lock is counted with its unlock."""
    clock = Tick()
    budget = hf_store.CommitBudget(6, str(tmp_path / "ledger.json"), clock=clock)
    store = hf_store.Store(hf_store.DirBackend(str(tmp_path / "repo")), log=lambda m: None, delay=0, budget=budget)
    push_gens(tmp_path, store, [100, 200, 300, 400])                                     # four commits
    assert budget.count() == 4 == len(commits(tmp_path)) and budget.reserve == 2
    with pytest.raises(hf_store.BudgetSpent):
        store.prune("ckpt-N", verified=[300, 400])                                       # not essential: in the reserve
    assert store.release_report("home", grace=600)["held"] is True                       # the lock: no room for two
    assert len(commits(tmp_path)) == 4 and store.steps("ckpt-N") == [100, 200, 300, 400]
    mark = tmp_path / "mark"
    mark.mkdir()
    (mark / "model.safetensors").write_bytes(S.state_at(100)[0])
    (mark / "state.txt").write_text("100\n{}\n")
    assert store.push_mark("pt", 100, str(mark)) is True                                 # essential: goes
    store.push_generation("ckpt-N", "pt", 500, staged_gen(tmp_path, 500), None, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [300, 400]})                     # and the save's own prune with it
    assert budget.count() == 6 and store.steps("ckpt-N") == [300, 400, 500]
    with pytest.raises(hf_store.BudgetSpent, match="generation not sent"):
        store.push_generation("ckpt-N", "pt", 600, staged_gen(tmp_path, 600), None, store.head("ckpt-N")["commit"])
    assert len(commits(tmp_path)) == 6
    # An hour on: the release takes its lock and gives it back, two commits counted together, and measures.
    clock.t += 3601
    rep = store.release_report("home", grace=600)
    assert rep["held"] is False and rep["objects"] == 0 and rep["gens"]["ckpt-N"] and rep["orphan_objects"] == 0
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
    assert (s2.budget.window, s2.budget.essential_wait) == (3600, 900)
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
    cfg = {"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tf), "mark_ns": kw.pop("mark_ns", "")}
    store = hf_store.open_store(cfg, api=hub_standin.DiskApi(str(root)), log=logs.append, delay=0, **kw)
    return store, root, logs


def hub_commits(root):
    p = root / "commit_log.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def test_the_hubs_429_is_read_with_its_retry_after(tmp_path):
    """The hub refuses a commit past its limit with HTTP 429, as the library raises it (HfHubHTTPError with the
    response). The store reads Retry-After, raises RateLimited with it, and lets nothing through its budget for at
    least that long, at most half an hour; the class and the wait survive a bounded() child; a 412 stays what it
    was. A branch's creation is refused by the same limit, and is read and counted the same way."""
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
    # Through a bounded() child (how the courier and the runner call): the class and the wait come back. The hub
    # asks for 2,300 s here; half an hour is honoured.
    clock.t += 1300
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").write_text(json.dumps({"per_window": 2, "retry_after": "auto"}))
    with pytest.raises(hf_store.RateLimited) as e:
        hf_store.bounded(lambda: bool(store.push_set("main", "set2", str(src))), 30, TOKEN)
    assert e.value.retry_after == 1800 == budget.wait_s() == hf_store.MAX_429_WAIT_S
    # A 429 that names no wait: at least a minute.
    clock.t += 2300
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": None}))
    with pytest.raises(hf_store.RateLimited) as e:
        store.push_set("main", "set2", str(src))
    assert e.value.retry_after == hf_store.MIN_429_WAIT_S == budget.wait_s()
    # A branch's creation: through the budget, refused by the same limit, read the same way, nothing created.
    clock.t += 60
    n = budget.count()
    with pytest.raises(hf_store.RateLimited, match="the creation of ckpt-N") as e:
        store.ensure_branch("ckpt-N")
    assert e.value.retry_after == 60 and budget.count() == n + 1 and budget.wait_s() == 60
    assert not (root / "branches" / "ckpt-N").exists() and hub_commits(root)[-1]["message"] == "create_branch"
    with pytest.raises(hf_store.BudgetSpent):
        store.ensure_branch("ckpt-N")
    # With room again the same set goes, and what the refusals left behind is nothing.
    clock.t += 3600
    (root / "now.txt").write_text(str(clock.t))
    (root / "limit.json").unlink()
    store.push_set("main", "set2", str(src))
    assert sorted(os.listdir(root / "branches" / "main")) == ["set0", "set1", "set2"]
    # The branch: created once, remembered beside the ledger, so no other process of this writer asks again; and
    # it starts from main's tree, as on the hub.
    store.ensure_branch("ckpt-N")
    store.ensure_branch("ckpt-N")
    other, _, _ = hub_store(tmp_path, budget=hf_store.CommitBudget(50, str(tmp_path / "ledger.json"), clock=clock))
    other.push_set("ckpt-N", "x", str(src))             # no parent named: it would have created the branch
    calls = (root / "calls.log").read_text().splitlines()
    assert sum(ln.startswith("create_branch ckpt-N") for ln in calls) == 2 and sum("create_branch" == c["message"] and c["ok"] for c in hub_commits(root)) == 1
    assert sorted(os.listdir(root / "branches" / "ckpt-N")) == ["set0", "set1", "set2", "x"]
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


def test_the_data_jobs_separate_processes_create_their_branch_once(tmp_path):
    """base2_data.sh pushes each set with a process of its own (hf_store.py push). Each used to ask the hub to
    create the build branch again, a call the hub counts against the repository's commit limit. The first one's
    answer is kept beside the ledger, and the count is one for them all."""
    pytest.importorskip("huggingface_hub")
    root = tmp_path / "hub"
    tok = tmp_path / "token"
    tok.write_text(TOKEN + "\n")
    tok.chmod(0o600)
    cfg = tmp_path / "chat" / "store.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tok)}))
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"a")
    site = ROOT / "tests" / "hub_standin_site"
    env = {**os.environ, "HUB_STANDIN_ROOT": str(root), "PYTHONPATH": f"{site}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)}
    for i in range(3):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "hf_store.py"), "push", "--config", str(cfg),
                            "--branch", "base2-build", "--prefix", f"build/s{i}", "--dir", str(src), "--limit", "60",
                            "--commit-budget", "30", "--commit-wait", "900"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and TOKEN not in r.stdout + r.stderr, r.stderr
        assert json.loads(r.stderr.strip().splitlines()[-1][5:])["commits"] == i + 2        # the branch, then one a set
    calls = (root / "calls.log").read_text().splitlines()
    assert sum(ln.startswith("create_branch") for ln in calls) == 1 and sum(ln.startswith("create_commit") for ln in calls) == 3
    assert [c["message"] for c in hub_commits(root)] == ["create_branch", "files: build/s0", "files: build/s1", "files: build/s2"]
    assert json.loads((cfg.parent / "store_commits.json.branches").read_text())["branches"] == ["base2-build"]


def gen_objects(*steps):
    return {S.sha(b): len(b) for s in steps for b in S.state_at(s)}


def test_release_through_the_real_backend_never_takes_an_upload_from_under_its_commit(tmp_path):
    """The reviews' second blocking point. The library uploads a commit's large files first and commits after; the
    stand-in hub now does the same, keeps an uploaded object whatever the commit's answer, and refuses a commit that
    names an object it does not hold. The release from home deleted "every object no tree names that is older than
    the grace": an upload whose commit was still to come (refused with 429, retried later) was exactly that.

    Now a release deletes only what a removed generation's ledger names. So: an object uploaded long ago and not
    committed survives every timed release, and its retried commit succeeds; a release that runs BETWEEN a commit's
    upload and its commit changes nothing for that commit; and the by-hand `--orphans`, in the same place, shows
    what the old rule did: the commit is refused, because its object is gone."""
    pytest.importorskip("huggingface_hub")
    import hub_standin
    from huggingface_hub.errors import HfHubHTTPError
    store, root, _ = hub_store(tmp_path)
    home, _, _ = hub_store(tmp_path)                    # the releaser: another process, another handle
    far = time.time() + 30 * 86400                      # a release a month from now: everything is old by then
    store.ensure_branch("ckpt-N")
    for step in (100, 200, 300):
        store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step), IDENT, store.head("ckpt-N")["commit"])
    store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), IDENT, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [200, 300]})
    log = [c for c in hub_commits(root) if c["message"] != "create_branch"]
    assert len(log) == 4 and log[-1]["deletes"] == ["gens/g100/"] and "pruned/g100.json" in log[-1]["adds"]
    assert store.steps("ckpt-N") == [200, 300, 400] and len(json.loads((root / "lfs.json").read_text())) == 8
    # 1. Uploaded, refused, old: g500's objects are in the object store and in no commit.
    (root / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": 30}))
    g500 = staged_gen(tmp_path, 500)
    with pytest.raises(hf_store.RateLimited):
        store.push_generation("ckpt-N", "pt", 500, g500, IDENT, store.head("ckpt-N")["commit"],
                              prune={"keep": 2, "verified": [200, 300, 400]})
    (root / "limit.json").unlink()
    assert set(gen_objects(500)) <= set(json.loads((root / "lfs.json").read_text())) and store.steps("ckpt-N") == [200, 300, 400]
    young = home.release_report("home", grace=600)                        # g100's ledger is a moment old
    assert (young["objects"], young["pending"]) == (0, {"ckpt-N": gen_objects(100)})
    assert (young["orphan_objects"], young["orphan_bytes"]) == (2, sum(gen_objects(500).values()))
    rep = home.release_report("home", grace=600, now=far)
    assert (rep["held"], rep["objects"], rep["bytes"]) == (False, 2, sum(gen_objects(100).values()))   # g100's two, by its ledger
    assert (rep["orphan_objects"], rep["orphan_bytes"]) == (2, sum(gen_objects(500).values()))         # g500's: reported, left
    assert set(json.loads((root / "lfs.json").read_text())) == set(gen_objects(200, 300, 400, 500))
    # ... and its retried commit succeeds, a month later, with the same bytes.
    man = store.push_generation("ckpt-N", "pt", 500, g500, IDENT, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [200, 300, 400]})
    assert man["pruned"] == [200] and store.listed_whole("ckpt-N", 500, man["files"])
    assert hub_store(tmp_path)[0].verify_by_download("ckpt-N", 500, expect=IDENT)
    # 2. A timed release between the upload and the commit of g600: the commit goes through, whole.
    api = store.b.api
    ran = []
    api.after_upload = lambda branch, ops: ran.append(home.release_report("home", grace=600, now=far))
    man = store.push_generation("ckpt-N", "pt", 600, staged_gen(tmp_path, 600), IDENT, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [300, 400, 500]})
    assert len(ran) == 1 and ran[0]["held"] is False and ran[0]["objects"] == 2       # g200's, by its ledger; not g600's
    assert ran[0]["orphan_objects"] == 2 and store.listed_whole("ckpt-N", 600, man["files"])
    assert hub_store(tmp_path)[0].verify_by_download("ckpt-N", 600, expect=IDENT)
    # 3. The same moment with --orphans (the old rule, kept for a hand with no VM writing): the commit is refused.
    ran.clear()
    api.after_upload = lambda branch, ops: ran.append(home.release_report("home", grace=600, now=far, orphans=True))
    with pytest.raises(HfHubHTTPError, match="does not exist"):
        store.push_generation("ckpt-N", "pt", 700, staged_gen(tmp_path, 700), IDENT, store.head("ckpt-N")["commit"])
    api.after_upload = None
    assert ran[0]["objects"] >= 2 and store.steps("ckpt-N") == [400, 500, 600]         # nothing of g700 was committed
    # The ledgers that are finished with: recorded on main by the unlock, removed by the branch's next save.
    assert json.loads((root / "branches" / "main" / "locks" / "released.json").read_text()) == {"ckpt-N": [200, 300]}
    store.push_generation("ckpt-N", "pt", 700, staged_gen(tmp_path, 700, seed=1), IDENT, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [400, 500, 600]})
    assert hf_store.ledger_steps(store.b.tree("ckpt-N")) == [400] and store.steps("ckpt-N") == [500, 600, 700]
    assert not (root / "branches" / "main" / "locks" / "release.json").exists()
    step, rejected = hub_store(tmp_path)[0].pull_generation("ckpt-N", str(tmp_path / "r"), expect=IDENT)
    assert (step, rejected) == (700, [])
    calls = (root / "calls.log").read_text().splitlines()
    assert sum(ln.startswith("create_branch") for ln in calls) == 1  # once, for the branch; never before a commit
    assert isinstance(api, hub_standin.DiskApi)


def test_three_campaigns_on_one_repo_are_each_counted_for_what_is_theirs(tmp_path):
    """The storage guard's measurement, on the stand-in hub, where a new branch starts from main's tree as on the
    real one: main holds the seed weights (every branch's tree then names them), three campaigns each keep
    generations and a mark, one has removed a generation that is not released yet, and an upload that nobody
    committed lies about. Each campaign is counted for the objects under gens/ on its own branch, its own marks
    and its own ledgers' objects, each object once. Before: every branch's whole tree (main's copy with it) and
    every unnamed object of the repo, for each campaign."""
    pytest.importorskip("huggingface_hub")
    seedw = b"seed weights of base A " * 300
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "tc_b1_r1.safetensors").write_bytes(seedw)
    hub_store(tmp_path)[0].push_set("main", "seed/weights", str(seed))
    stores, want = {}, {}
    for k, name in enumerate(("N", "S", "C")):
        st, root, _ = hub_store(tmp_path, mark_ns=f"arm{name}.")
        stores[name] = st
        branch = f"ckpt-{name}"
        st.ensure_branch(branch)
        assert "seed/weights/tc_b1_r1.safetensors" in st.b.tree(branch)          # inherited from main, as on the hub
        steps = [100, 200, 300]
        for s in steps:
            st.push_generation(branch, "pt", s, staged_gen(tmp_path / name, s, seed=k), None, st.head(branch)["commit"])
        mark = tmp_path / name / "mark"
        mark.mkdir()
        (mark / "model.safetensors").write_bytes(S.state_at(100, k)[0])           # the mark's weights are g100's
        (mark / "state.txt").write_text("100\n{}\n")
        st.push_mark("pt", 100, str(mark))
        want[name] = {S.sha(b): len(b) for s in steps for b in S.state_at(s, k)}
    # Campaign N removes g100 (its ledger is young): the optimiser is pending, the weights are still its mark's.
    stores["N"].push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path / "N", 400, seed=0), None,
                                stores["N"].head("ckpt-N")["commit"], prune={"keep": 2, "verified": [200, 300]})
    want["N"].update({S.sha(b): len(b) for b in S.state_at(400, 0)})
    # An upload nobody committed.
    (root / "limit.json").write_text(json.dumps({"per_window": 0, "retry_after": 1}))
    with pytest.raises(hf_store.RateLimited):
        stores["S"].push_generation("ckpt-S", "pt", 400, staged_gen(tmp_path / "S", 400, seed=1), None, stores["S"].head("ckpt-S")["commit"])
    (root / "limit.json").unlink()
    orphan = sum(len(b) for b in S.state_at(400, 1))
    rep = hub_store(tmp_path)[0].release_report("home", grace=600)
    assert rep["held"] is False and rep["objects"] == 0 and rep["unknown_sizes"] == 0
    counted = {n: hf_store.campaign_bytes(rep, [f"ckpt-{n}"], [f"arm{n}.pt"]) for n in "NSC"}
    assert counted == {n: sum(want[n].values()) for n in "NSC"}                    # each object once: the mark is g100's weights
    assert rep["pending"] == {"ckpt-N": {S.sha(S.state_at(100, 0)[1]): len(S.state_at(100, 0)[1])}}
    assert (rep["orphan_objects"], rep["orphan_bytes"]) == (2, orphan)
    assert rep["stored_bytes"] == sum(counted.values()) + len(seedw) + orphan      # the three, main's seed, the orphan: all of it
    assert all(S.sha(seedw) not in rep["gens"][f"ckpt-{n}"] for n in "NSC")        # main's copy is nobody's generation
    assert counted["S"] == counted["C"] < counted["N"]                             # another's pending and orphans are not mine
    # What the first version of the measurement would have said for S: its tree with main's copy, and everybody's leftovers.
    old = sum(e["bytes"] for e in stores["S"].b.tree("ckpt-S").values() if e.get("sha256")) + rep["pending_bytes"] + orphan
    assert old > counted["S"] + len(seedw)


# ---------------------------------------------------------------- the second reviews (12 Nov 2026)
def test_an_unread_check_is_not_a_damaged_generation_and_a_missing_object_is_seen(tmp_path):
    """listed_whole has three answers. True: whole. False: a COMPLETED comparison that differs. An error: something
    could not be read, which is no verdict. The reviews: whole_in_listing swallowed the manifest fetch's error, so
    a good listing followed by an unreadable manifest read as "not whole" for good; and the listing's size and hash
    of an LFS file come from its pointer, so a pointer whose object is gone read as whole."""
    pytest.importorskip("huggingface_hub")
    import requests
    api = FakeApi()
    store, _ = hstore(tmp_path, api)
    man = store.push_generation("ckpt-N", "pt", 100, staged_gen(tmp_path, 100), IDENT)
    api.calls.clear()
    assert store.listed_whole("ckpt-N", 100, man["files"]) is True
    probes = [c[1] for c in api.calls if c[0] == "probe"]
    assert sorted(p["filename"] for p in probes) == ["gens/g100/model.safetensors", "gens/g100/optim.safetensors"]   # the two large files
    assert all(p["range"] == "bytes=0-0" and p["token"] == TOKEN and p["revision"] == "ckpt-N@1" for p in probes)   # at the commit
    # 1. The listing is good and the manifest cannot be fetched: an error, in this process and through a child.
    real = api.hf_hub_download

    def no_manifest(**kw):
        if kw["filename"].endswith("manifest.json"):
            raise requests.ConnectionError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out.")
        return real(**kw)

    api.hf_hub_download = no_manifest
    with pytest.raises(requests.ConnectionError):
        store.listed_whole("ckpt-N", 100, man["files"])
    with pytest.raises(hf_store.StoreError, match="ConnectionError"):
        hf_store.bounded(lambda: store.listed_whole("ckpt-N", 100, man["files"]), 30, TOKEN)
    assert store.whole_in_listing("ckpt-N", 100, store.b.tree("ckpt-N")) is False     # the prune's own reading stays the cautious one
    api.hf_hub_download = real
    # The listing cannot be read: an error. A branch that is not there: an error.
    api.fail = requests.ConnectionError("reset")
    with pytest.raises(requests.ConnectionError):
        store.listed_whole("ckpt-N", 100)
    api.fail = None
    with pytest.raises(hf_store.StoreError, match="not listed"):
        store.listed_whole("ckpt-nowhere", 100)
    # A byte that cannot be read (the hub answers 503): an error.
    session = FakeApi.http_session.fget(api)
    broken = type("S", (), {"get": lambda self, url, **kw: _resp(503)})()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeApi, "http_session", property(lambda self: broken))
        with pytest.raises(Exception) as e:
            store.listed_whole("ckpt-N", 100)
        assert "503" in str(e.value) and not isinstance(e.value, AssertionError)
        with pytest.raises(Exception):
            store.b.probe("ckpt-N", "gens/g100/model.safetensors")
    assert session is not None and store.listed_whole("ckpt-N", 100) is True
    # 2. Completed comparisons that differ: False, each.
    assert store.listed_whole("ckpt-N", 200) is False                                    # not listed at all
    assert store.listed_whole("ckpt-N", 100, {**man["files"], "state.txt": {"bytes": 1, "sha256": "0" * 64}}) is False
    keep = dict(api.branches["ckpt-N"])
    api.branches["ckpt-N"]["gens/g100/manifest.json"] = b"{not json"
    assert store.listed_whole("ckpt-N", 100) is False                                    # read, and not a manifest
    api.branches["ckpt-N"] = dict(keep)
    api.branches["ckpt-N"]["gens/g100/optim.safetensors"] = b"short"
    assert store.listed_whole("ckpt-N", 100) is False                                    # a file of another size
    api.branches["ckpt-N"] = dict(keep)
    # 3. The pointer is listed, with its size and hash, and the object is gone: the listing alone says whole; the
    # byte read says no.
    gone = S.sha(S.state_at(100)[1])
    at = api.lfs.pop(gone)
    assert store.whole_in_listing("ckpt-N", 100, store.b.tree("ckpt-N")) is True and store.listed_whole("ckpt-N", 100, probe=False) is True
    assert store.listed_whole("ckpt-N", 100, man["files"]) is False
    assert store.b.probe("ckpt-N", "gens/g100/optim.safetensors") is False and store.b.probe("ckpt-N", "gens/g100/model.safetensors") is True
    assert store.b.probe("ckpt-N", "gens/g100/nothing.bin") is False                     # the hub's own "no such file"
    api.lfs[gone] = at
    # The same through the stand-in hub: a commit that lands and whose object is lost after it.
    hs, root, _ = hub_store(tmp_path / "hub")
    hs.ensure_branch("ckpt-N")
    (root / "lose.json").write_text(json.dumps({"path": "gens/g200/optim.safetensors"}))
    m2 = hs.push_generation("ckpt-N", "pt", 200, staged_gen(tmp_path / "hub", 200), IDENT, hs.head("ckpt-N")["commit"])
    assert hs.steps("ckpt-N") == [200] and hs.whole_in_listing("ckpt-N", 200, hs.b.tree("ckpt-N")) is True
    assert hs.listed_whole("ckpt-N", 200, m2["files"]) is False and not hs.verify_by_download("ckpt-N", 200)
    (root / "fail.json").write_text(json.dumps({"mode": "transport"}))
    with pytest.raises(requests.ConnectionError):
        hs.listed_whole("ckpt-N", 200, m2["files"])
    (root / "fail.json").unlink()
    # The calls probe() and the late ledger rely on are in the pinned library as they are used here.
    import inspect
    import huggingface_hub
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
    params = inspect.signature(huggingface_hub.HfApi.preupload_lfs_files).parameters
    assert {"repo_id", "additions", "token", "repo_type", "revision"} <= set(params)
    assert huggingface_hub.hf_hub_url(repo_id="o/n", filename="gens/g1/model.safetensors", repo_type="dataset", revision="ckpt-N") == \
        "https://huggingface.co/datasets/o/n/resolve/ckpt-N/gens/g1/model.safetensors"
    assert build_hf_headers(token="hf_x")["authorization"] == "Bearer hf_x" and callable(get_session) and callable(hf_raise_for_status)


def _resp(code):
    import io
    import requests
    r = requests.Response()
    r.status_code, r.url, r.raw = code, "https://huggingface.co/datasets/someone/base2-store/resolve/ckpt-N/x", io.BytesIO(b"")
    return r


def test_release_with_objects_that_several_ledgers_or_a_mark_name(tmp_path):
    """The rehearsal's interrupted run and its control train the same steps from the same seed: their generations
    are the same bytes, on two branches. And a mark names its generation's weights. The first ledger rule ("no
    other ledger names it") never freed what both runs had removed, and left both ledgers unfinished for ever.

    Now an object goes when EVERY ledger that names it is older than the grace (and a margin for the two clocks)
    and no tree names it; it goes once, and each ledger is finished when all it names is gone or named elsewhere.
    A ledger that cannot be read stops all deletion in that round. Ledgers are read at the commit the release saw
    as their branch's head, so a save that removes one in the meantime changes nothing."""
    api = FakeApi()
    store, logs = hstore(tmp_path, api, clock=lambda: api.now)
    t0 = api.now
    for branch in ("ckpt-R", "ckpt-Rc"):                               # the run and its control: the same bytes
        for step in (100, 200, 300):
            store.push_generation(branch, "pt", step, staged_gen(tmp_path / branch, step))
    mark = tmp_path / "mark"
    mark.mkdir()
    m100, o100 = S.state_at(100)
    (mark / "model.safetensors").write_bytes(m100)
    (mark / "state.txt").write_text("100\n{}\n")
    store.push_mark("pt", 100, str(mark))
    m200, o200 = S.state_at(200)
    grace, skew = 600, hf_store.RELEASE_SKEW_S
    assert skew == 120
    # The run removes g100 and g200 at t0; the control still names them.
    assert store.prune("ckpt-R", keep=2, verified=[300, 200]) == [100]
    store.push_generation("ckpt-R", "pt", 400, staged_gen(tmp_path / "ckpt-R", 400), None, store.head("ckpt-R")["commit"],
                          prune={"keep": 2, "verified": [300, 200]})
    assert store.prune("ckpt-R", keep=2, verified=[300, 400]) == [200]
    r = store.release_report("home", grace=grace, now=t0 + 10 ** 5)
    assert r["objects"] == 0 and r["pending"] == {} and r["ledgers_done"] == 2          # all of it is named by the control's tree
    assert json.loads(api.branches["main"][hf_store.RELEASED]) == {"ckpt-R": [100, 200]}
    # The control removes its g100 and g200 later, at t0 + 5,000. Now two ledgers name each object.
    api.now = t0 + 5000
    assert store.prune("ckpt-Rc", keep=2, verified=[200, 300]) == [100]
    store.push_generation("ckpt-Rc", "pt", 400, staged_gen(tmp_path / "ckpt-Rc", 400), None, store.head("ckpt-Rc")["commit"],
                          prune={"keep": 2, "verified": [200, 300]})
    assert store.prune("ckpt-Rc", keep=2, verified=[300, 400]) == [200]
    assert hf_store.ledger_steps(store.b.tree("ckpt-R")) == [100, 200] == hf_store.ledger_steps(store.b.tree("ckpt-Rc"))
    # One ledger of each pair is old, the other young: nothing shared is deleted, and it waits in both.
    api.calls.clear()
    r = store.release_report("home", grace=grace, now=t0 + 5000 + grace + skew - 1)     # inside the margin for the clocks
    want = {S.sha(o100): len(o100), S.sha(m200): len(m200), S.sha(o200): len(o200)}     # m100 is the mark's: not pending
    assert r["objects"] == 0 and r["pending"] == {"ckpt-R": want, "ckpt-Rc": want} and r["ledgers_done"] == 0
    assert r["pending_bytes"] == sum(want.values()) and "permanently_delete_lfs_files" not in [c[0] for c in api.calls]
    assert hf_store.campaign_bytes(r, ["ckpt-R", "ckpt-Rc"], ["pt"]) == sum(want.values()) + len(m100) + sum(
        len(b) for s in (300, 400) for b in S.state_at(s))                              # each object once for the campaign
    # A ledger that cannot be read: nothing at all is deleted in that round, though every readable ledger is old.
    good = api.branches["ckpt-Rc"]["pruned/g200.json"]
    api.branches["ckpt-Rc"]["pruned/g200.json"] = b"\x00 not a ledger"
    late = t0 + 10 ** 6
    r = store.release_report("home", grace=grace, now=late)
    assert (r["objects"], r["ledgers_unread"]) == (0, 1) and S.sha(o100) in api.lfs
    assert S.sha(o100) in r["pending"]["ckpt-R"] and any("nothing is deleted in this round" in ln for ln in logs)
    api.branches["ckpt-Rc"]["pruned/g200.json"] = good
    # Every ledger old and read: each object goes once, and all four ledgers are finished. The mark's weights stay.
    api.calls.clear()
    r = store.release_report("home", grace=grace, now=late)
    assert (r["objects"], r["bytes"], r["ledgers_unread"]) == (3, sum(want.values()), 0) and r["pending"] == {}
    assert sorted(dict(api.calls)["permanently_delete_lfs_files"]["oids"]) == sorted(want)
    assert S.sha(m100) in api.lfs and r["ledgers_done"] == 4
    assert json.loads(api.branches["main"][hf_store.RELEASED]) == {"ckpt-R": [100, 200], "ckpt-Rc": [100, 200]}
    assert store.release_report("home", grace=grace, now=late)["objects"] == 0
    # A ledger its branch removes while the release is between the listing and the read: read all the same, at
    # the commit the release saw as the head.
    api2 = FakeApi()
    s2, _ = hstore(tmp_path / "two", api2, clock=lambda: api2.now)
    for step in (100, 200, 300):
        s2.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path / "two", step))
    assert s2.prune("ckpt-N", verified=[200, 300]) == [100]
    seen = []

    def courier_removes_the_ledger():
        if not seen and api2.calls[-1][1].get("revision", "").startswith("ckpt-N@"):
            seen.append(1)
            api2.create_commit(repo_id="r", operations=[type("CommitOperationDelete", (), {"path_in_repo": "pruned/g100.json", "is_folder": False})()],
                               commit_message="generation: gens/g400", revision="ckpt-N")
        elif not seen:
            api2.after["list_repo_tree"] = courier_removes_the_ledger

    api2.after["list_repo_tree"] = courier_removes_the_ledger
    r = s2.release_report("home", grace=grace, now=api2.now + 10 ** 5)
    assert seen and "pruned/g100.json" not in api2.branches["ckpt-N"]
    assert (r["objects"], r["ledgers_unread"]) == (2, 0)                                 # g100's two files, by the ledger as it was
    fetched = [c[1] for c in api2.calls if c[0] == "hf_hub_download" and c[1]["filename"] == "pruned/g100.json"]
    assert fetched and all(c["revision"].startswith("ckpt-N@") for c in fetched)         # by commit, not by the moving name


def test_a_ledgers_time_is_the_moment_its_commit_is_sent(tmp_path):
    """The grace a reader of a removed generation is owed runs from the removal. The ledger's time was taken before
    the new generation was hashed and uploaded, which can take half an hour: the grace could be over before the
    removal existed. The large files are now uploaded first (the library's preupload_lfs_files), the ledger is
    written, and the commit is sent."""
    api = FakeApi()
    store, _ = hstore(tmp_path, api, clock=lambda: api.now)
    for step in (100, 200, 300):
        store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step))
    t0 = api.now
    api.after["preupload_lfs_files"] = lambda: setattr(api, "now", api.now + 1500)      # the upload takes 25 minutes
    api.calls.clear()
    store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), None, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [200, 300]})
    order = [c[0] for c in api.calls if c[0] in ("preupload_lfs_files", "create_commit")]
    assert order == ["preupload_lfs_files", "create_commit"]
    pre = dict(api.calls)["preupload_lfs_files"]
    assert "gens/g400/model.safetensors" in pre["paths"] and "pruned/g100.json" not in pre["paths"] and pre["token"] == TOKEN
    assert ("CommitOperationAdd", "pruned/g100.json") in dict(api.calls)["create_commit"]["ops"]
    assert json.loads(api.branches["ckpt-N"]["pruned/g100.json"])["at"] == t0 + 1500   # after the upload, not before
    # So a release 600 s after the upload BEGAN deletes nothing: the removal is 600 s old only 1,500 s later.
    assert store.release_report("home", grace=600, now=t0 + 1500 + 600)["objects"] == 0
    assert store.release_report("home", grace=600, now=t0 + 1500 + 600 + hf_store.RELEASE_SKEW_S + 1)["objects"] == 2
    # A save that removes nothing is one plain create_commit, as before.
    api.calls.clear()
    store.push_generation("ckpt-N", "pt", 500, staged_gen(tmp_path, 500), None, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": [300]})
    assert [c[0] for c in api.calls if c[0] in ("preupload_lfs_files", "create_commit")] == ["create_commit"]
    # A library that does not take the upload call as written: the commit still goes, with its ledger.
    api.preupload_lfs_files = lambda **kw: (_ for _ in ()).throw(TypeError("unexpected keyword argument"))
    man = store.push_generation("ckpt-N", "pt", 600, staged_gen(tmp_path, 600), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [300, 400, 500]})
    assert man["pruned"] == [200, 300] and hf_store.ledger_steps(store.b.tree("ckpt-N")) == [200, 300]
    # A store on local disk writes it under its commit lock; a standalone prune stamps it the same way.
    clock = Tick()
    d = hf_store.Store(hf_store.DirBackend(str(tmp_path / "repo")), log=lambda m: None, delay=0, clock=clock)
    push_gens(tmp_path / "d", d, [100, 200, 300])
    clock.t += 777
    assert d.prune("ckpt-N", verified=[200, 300]) == [100]
    assert json.loads((tmp_path / "repo" / "ckpt-N" / "pruned" / "g100.json").read_text())["at"] == clock.t


def test_the_budget_in_memory_keeps_its_history_and_a_bad_record_of_finished_ledgers_is_ignored(tmp_path):
    """Two smaller points of the second reviews. (1) When the ledger file stops being writable the count goes on
    in memory: from the history that was in the file, with the edit that was under way applied once, and ageing as
    it does in the file (it used to start from nothing, could apply the edit twice, and never aged). (2)
    main:locks/released.json is written by the release alone, and is checked all the same: one that is not
    {branch: [steps]} is said in the log and ignored, and the save goes on without removing any ledger."""
    if os.geteuid() == 0:
        pytest.skip("a folder cannot be made unwritable for root")
    clock = Tick()
    folder = tmp_path / "vm"
    folder.mkdir()
    b = hf_store.CommitBudget(6, str(folder / "store_commits.json"), clock=clock, essential_wait=3600)
    for _ in range(3):
        b.admit("generation")
        clock.t += 100
    os.chmod(folder, 0o555)                              # the disk goes read-only under the courier
    try:
        b.admit("generation")                            # counted once, on top of the three that were in the file
        assert b.count() == 4 and b._mem is not None
        other = hf_store.CommitBudget(6, str(folder / "store_commits.json"), clock=clock, essential_wait=3600)
        assert other.count() == 3                        # a new process still reads what the file held
        b.admit("generation")
        b.admit("generation")
        with pytest.raises(hf_store.BudgetSpent):
            b.admit("generation")
        clock.t += 3501                                  # the three from the file are an hour old; they age in memory too
        assert b.count() == 3 and b.wait_s() == 0
        b.admit("generation")
        clock.t += 3601
        assert b.count() == 0
    finally:
        os.chmod(folder, 0o755)
    # (2)
    logs = []
    store = hf_store.Store(hf_store.DirBackend(str(tmp_path / "repo")), log=logs.append, delay=0)
    push_gens(tmp_path, store, [100, 200, 300])
    assert store.prune("ckpt-N", verified=[200, 300]) == [100]
    locks = tmp_path / "repo" / "main" / "locks"
    locks.mkdir(parents=True)
    for bad in ('{"ckpt-N": 5}', '{"ckpt-N": ["100"]}', '[100]', '{"ckpt-N": [true]}', "{not json", "\x00\xff"):
        (locks / "released.json").write_bytes(bad.encode("latin-1"))
        logs.clear()
        step = max(store.steps("ckpt-N")) + 100
        man = store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step), None, store.head("ckpt-N")["commit"],
                                    prune={"keep": 2, "verified": store.steps("ckpt-N")[-2:]})
        assert man["step"] == step and 100 in hf_store.ledger_steps(store.b.tree("ckpt-N")), bad
        assert any("is not well formed; ignored" in ln for ln in logs), bad
    (locks / "released.json").write_text(json.dumps({"ckpt-N": [100], "ckpt-other": []}))
    step = max(store.steps("ckpt-N")) + 100
    store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step), None, store.head("ckpt-N")["commit"],
                          prune={"keep": 2, "verified": store.steps("ckpt-N")[-2:]})
    assert 100 not in hf_store.ledger_steps(store.b.tree("ckpt-N"))                      # a good record is followed


# ---------------------------------------------------------------- the arms' round (third reviews and the rerun, 12 Nov 2026)
def _reply(code, body=b"", content_range=None, redirected=False):
    """A reply to the one-byte ranged read, as requests hands it over; after the hub's redirect when asked."""
    r = _resp(code)
    r.raw = __import__("io").BytesIO(body)
    if content_range:
        r.headers["Content-Range"] = content_range
    if redirected:
        first = _resp(302)
        r.history, r.url = [first], "https://cdn-lfs.example/objects/abc?X-Amz-Signature=SIGNED"
    return r


def test_the_byte_read_is_held_against_the_listed_size_and_made_at_the_listings_commit(tmp_path):
    """The third reviews. (1) The byte read took a 416 for "there" whatever the file's size, and any successful
    reply for the file's bytes. Now it is given the listed size: a 416 is "there" only for a file listed as empty,
    a reply that names another whole size (Content-Range) is "not there", and a 416 that cannot be held against a
    size is no answer. (2) The listing was taken at the branch's name and the bytes were read at the branch's name
    a moment later. Now the head is read once, and the listing, the manifest and every byte are read at that
    commit. (3) A 403 is no verdict, from the hub or from the object store; its error names the status and where
    it came from, never the signed address, and the status survives a bounded() child."""
    pytest.importorskip("huggingface_hub")
    api = FakeApi()
    store, _ = hstore(tmp_path, api)
    man = store.push_generation("ckpt-N", "pt", 100, staged_gen(tmp_path, 100), IDENT)
    size = man["files"]["model.safetensors"]["bytes"]
    path = "gens/g100/model.safetensors"

    def answers(reply):
        seen = []

        def get(self, url, **kw):
            seen.append(url)
            return reply
        return type("S", (), {"get": get})(), seen

    def probe(reply, **kw):
        session, _ = answers(reply)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(FakeApi, "http_session", property(lambda self: session))
            return store.b.probe("ckpt-N", path, **kw)

    # (1) 416: there only when the file is listed as empty.
    assert probe(_reply(416), size=0) is True
    assert probe(_reply(416, content_range="bytes */0"), size=0) is True
    assert probe(_reply(416, content_range="bytes */0"), size=size) is False         # an empty object for a file that is not
    for kw in ({"size": size}, {}, {"size": 0, "content_range": "bytes */7"}):
        cr = kw.pop("content_range", None)
        with pytest.raises(hf_store.StoreError, match="HTTP 416") as e:               # no answer: asked again
            probe(_reply(416, content_range=cr), **kw)
        assert hf_store.http_status(e.value) == 416
    # A successful reply is held against the listed size where it names one.
    assert probe(_reply(206, b"x", f"bytes 0-0/{size}"), size=size) is True
    assert probe(_reply(206, b"x", f"bytes 0-0/{size - 1}"), size=size) is False    # the bytes of another object
    assert probe(_reply(206, b"x", "bytes 0-0/*"), size=size) is True               # it names none: one byte is the evidence
    assert probe(_reply(200, b"x"), size=size) is True                              # the range was ignored: a byte all the same
    assert probe(_reply(206, b""), size=size) is False and probe(_reply(200, b""), size=0) is True
    assert hf_store.range_total(_reply(206, b"x", "bytes 0-0/12")) == 12 and hf_store.range_total(_reply(206)) is None
    # listed_whole passes each file's listed size, so a 416 for a large file is never "whole".
    session, _ = answers(_reply(416))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeApi, "http_session", property(lambda self: session))
        with pytest.raises(hf_store.StoreError, match="HTTP 416"):
            store.listed_whole("ckpt-N", 100, man["files"])
    session, _ = answers(_reply(206, b"x", "bytes 0-0/5"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeApi, "http_session", property(lambda self: session))
        assert store.listed_whole("ckpt-N", 100, man["files"]) is False
    # (3) 403 and 401: no verdict, from either side; 404 after the redirect is "gone", as before.
    for code, redirected, where in ((403, True, "from the object store"), (403, False, "from the hub"), (401, False, "from the hub")):
        with pytest.raises(hf_store.StoreError) as e:
            probe(_reply(code, redirected=redirected), size=size)
        text = str(e.value)
        assert f"HTTP {code} on the byte read of ckpt-N:{path}" in text and where in text and hf_store.http_status(e.value) == code
        assert "cdn-lfs" not in text and "SIGNED" not in text and TOKEN not in text  # not the signed address
    assert probe(_reply(404, redirected=True), size=size) is False
    session, _ = answers(_reply(403, redirected=True))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeApi, "http_session", property(lambda self: session))
        with pytest.raises(hf_store.StoreError) as e:
            hf_store.bounded(lambda: store.listed_whole("ckpt-N", 100, man["files"]), 30, TOKEN)
    assert hf_store.http_status(e.value) == 403 and "from the object store" in str(e.value)
    # An error whose text names no status carries its response's across the pipe; one without a status names none.
    class Refused(Exception):
        response = type("R", (), {"status_code": 418})()

    def refuse():
        raise Refused("the hub said no")
    with pytest.raises(hf_store.StoreError) as e:
        hf_store.bounded(refuse, 30, TOKEN)
    assert hf_store.http_status(e.value) == 418 and hf_store.http_status(ConnectionError("reset by peer")) is None
    # (2) One commit for all three reads.
    api.calls.clear()
    assert store.listed_whole("ckpt-N", 100, man["files"]) is True
    head = "ckpt-N@1"
    assert [c[1]["revision"] for c in api.calls if c[0] == "list_repo_tree"] == [head]
    assert [c[1]["revision"] for c in api.calls if c[0] == "hf_hub_download"] == [head]
    assert [c[1]["revision"] for c in api.calls if c[0] == "probe"] == [head, head]
    # The branch moves between the listing and the byte read (a later save removes a file under the same name):
    # the check is of the commit it listed, where the bytes are.

    def a_commit_lands():
        api.create_commit(repo_id="r", operations=[type("CommitOperationDelete", (), {"path_in_repo": "gens/g100/optim.safetensors",
                                                                                      "is_folder": False})()],
                          commit_message="generation: gens/g200", revision="ckpt-N")

    api.after["list_repo_tree"] = a_commit_lands
    assert store.listed_whole("ckpt-N", 100, man["files"]) is True
    assert api.heads["ckpt-N"] == 2 and "gens/g100/optim.safetensors" not in api.branches["ckpt-N"]
    assert store.listed_whole("ckpt-N", 100, man["files"]) is False                 # asked again, at the new head: a file is missing
    # The stand-in hub answers the ranged read at a commit id too, and names the whole size.
    hs, root, _ = hub_store(tmp_path / "hub")
    hs.ensure_branch("ckpt-N")
    m2 = hs.push_generation("ckpt-N", "pt", 200, staged_gen(tmp_path / "hub", 200), IDENT, hs.head("ckpt-N")["commit"])
    assert hs.listed_whole("ckpt-N", 200, m2["files"]) is True
    asked = [ln for ln in (root / "calls.log").read_text().splitlines() if ln.startswith("probe ")]
    assert asked and all(ln.startswith("probe ckpt-N@1:gens/g200/") for ln in asked)
    (root / "probe.json").write_text(json.dumps({"status": 403}))
    with pytest.raises(hf_store.StoreError, match="HTTP 403") as e:
        hs.listed_whole("ckpt-N", 200, m2["files"])
    assert "SIGNED" not in str(e.value)


def test_release_refuses_a_round_in_which_a_tree_cannot_be_read(tmp_path):
    """The third review: release read `tree(...) or {}`, so a branch whose listing was not given counted as a
    branch that names nothing, and what it names could be deleted from under it. Here a run and its control hold
    the same bytes; the run has removed g100 and its ledger is old; the control still names g100. With the
    control's tree unread the round must be refused, not run as if the control were empty. A head that is not
    found is the same case, and so is a main that cannot be listed when the lock is looked for."""
    pytest.importorskip("huggingface_hub")
    api = FakeApi()
    store, logs = hstore(tmp_path, api, clock=lambda: api.now)
    for branch in ("ckpt-R", "ckpt-Rc"):
        for step in (100, 200, 300):
            store.push_generation(branch, "pt", step, staged_gen(tmp_path / branch, step))
    assert store.prune("ckpt-R", keep=2, verified=[200, 300]) == [100]
    late = api.now + 10 ** 6
    objects = dict(api.lfs)
    real_tree, real_info = api.list_repo_tree, api.repo_info

    def no_tree_of_the_control(**kw):
        if str(kw.get("revision")).startswith("ckpt-Rc"):
            raise RevisionNotFoundError(str(kw.get("revision")))
        return real_tree(**kw)

    api.list_repo_tree = no_tree_of_the_control
    api.calls.clear()
    with pytest.raises(hf_store.StoreError, match="release refused: the tree of ckpt-Rc could not be read"):
        store.release_report("home", grace=600, now=late)
    assert api.lfs == objects and "permanently_delete_lfs_files" not in [c[0] for c in api.calls]
    assert hf_store.LOCK not in api.branches["main"]                                    # the lock was removed all the same
    api.list_repo_tree = real_tree

    def no_head_of_the_control(**kw):
        if kw.get("revision") == "ckpt-Rc":
            raise RevisionNotFoundError("ckpt-Rc")
        return real_info(**kw)

    api.repo_info = no_head_of_the_control
    with pytest.raises(hf_store.StoreError, match="release refused: the tree of ckpt-Rc"):
        store.release_report("home", grace=600, now=late)
    assert api.lfs == objects and hf_store.LOCK not in api.branches["main"]
    api.repo_info = real_info
    # Main cannot be listed: the lock is not taken on the word of a listing that was not given.

    def no_tree_of_main(**kw):
        if str(kw.get("revision")).startswith("main"):
            raise RevisionNotFoundError(str(kw.get("revision")))
        return real_tree(**kw)

    api.list_repo_tree = no_tree_of_main
    n = len([c for c in api.calls if c[0] == "create_commit"])
    r = store.release_report("home", grace=600, now=late)
    assert r["held"] is True and r["why"] == "main could not be listed"
    assert len([c for c in api.calls if c[0] == "create_commit"]) == n and api.lfs == objects
    api.list_repo_tree = real_tree
    # Every tree read: the round runs, and the control's tree keeps what it names.
    r = store.release_report("home", grace=600, now=late)
    assert (r["held"], r["objects"]) == (False, 0) and api.lfs == objects and r["ledgers_done"] == 1


def test_the_release_at_a_campaigns_end_is_essential_and_waits_for_room(tmp_path):
    """The rerun of 12 Nov 2026 (PROTOCOL 47.2): home's budget was 8 an hour and a timed lock may use six of it;
    three timed releases in the hour left the release at the campaign's end no room, and six ledgers stayed. That
    release is essential now: it may use the whole budget, it is owed the budget's essential wait and then goes
    whatever the count is, and a release that is not made says why. With the new default of 12: four timed releases
    an hour, and one runner's worst hour (ten commits) fits."""
    def home(per_hour, where):
        clock = Tick()
        budget = hf_store.CommitBudget(per_hour, str(tmp_path / where / "store_commits.json"), clock=clock)
        store = hf_store.Store(hf_store.DirBackend(str(tmp_path / where / "repo")), log=lambda m: None, delay=0, budget=budget)
        return store, budget, clock

    store, budget, clock = home(8, "eight")
    for _ in range(3):                                   # the rehearsal's hour: a timed release every twenty minutes
        assert store.release_report("home", grace=600)["held"] is False
        clock.t += 1200
    clock.t -= 1200 - 60                                 # a minute after the third: the campaign ends
    assert budget.count() == 6
    r = store.release_report("home", grace=600)          # as it was: a timed lock has no room
    assert r["held"] is True and r["why"].startswith("budget: room in ")
    r = store.release_report("home", grace=600, essential=True)
    assert r["held"] is False and budget.count() == 8    # the end's release: the whole budget is its to use
    # With the budget full an essential release waits: held, with the seconds; never longer than the essential wait.
    r = store.release_report("home", grace=600, essential=True)
    assert r["held"] is True and r["why"].startswith("budget: room in ")
    assert 0 < int(r["why"].split()[-2]) <= hf_store.ESSENTIAL_WAIT_S
    clock.t += hf_store.ESSENTIAL_WAIT_S - 2
    assert store.release_report("home", grace=600, essential=True)["held"] is True
    clock.t += 3
    assert store.release_report("home", grace=600, essential=True)["held"] is False and budget.count() == 10   # sent whatever the count
    # The default of 12 (runner_multi.py): nine for timed locks, so four an hour; the end's release on top; ten in all.
    store, budget, clock = home(12, "twelve")
    for _ in range(4):
        assert store.release_report("home", grace=600)["held"] is False
        clock.t += 60
    assert store.release_report("home", grace=600)["held"] is True and budget.count() == 8
    assert store.release_report("home", grace=600, essential=True)["held"] is False and budget.count() == 10
    assert not os.path.exists(tmp_path / "twelve" / "repo" / "main" / "locks" / "release.json")
    # The fourth reviews: a lock refused at the budget's own door, when the commit is admitted (another process
    # took the room after the look), was reported as "the lock's commit was refused", as if it had been sent, and
    # the end's release counted a round for it. Nothing was sent: it is the budget's "held", with its seconds.
    assert store.lock_owner() is None
    real_admit = budget.admit

    def no_room(*a, **kw):
        raise hf_store.BudgetSpent("the commit budget of 12 an hour has no room (lock not sent); room in 77 s")

    budget.admit = no_room
    n = len(commits(tmp_path / "twelve"))
    r = store.release_report("home", grace=600, essential=True)
    assert r["held"] is True and r["why"] == "budget: room in 77 s" and len(commits(tmp_path / "twelve")) == n
    budget.admit = real_admit
    # And the owner a lock on main names can be read (what a call that was ended at its limit is told by).
    mine = store._lock("home:pk-N", time.time(), essential=True, takeover=900)
    assert mine and store.lock_owner() == "home:pk-N"
    store._unlock(mine)
    assert store.lock_owner() is None


def test_which_upload_order_ran_comes_back_with_the_push(tmp_path):
    """The third reviews: when the library does not take preupload_lfs_files as it is called, the commit falls back
    to the old order (the ledger stamped before the upload) and nothing said so. The push now hands back which
    order ran, for the courier to log (tests/test_ckpt_courier.py)."""
    pytest.importorskip("huggingface_hub")
    api = FakeApi()
    store, _ = hstore(tmp_path, api, clock=lambda: api.now)
    for step in (100, 200, 300):
        man = store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step), None, None if step == 100 else store.head("ckpt-N")["commit"],
                                    prune={"keep": 2, "verified": []})
        assert man["pruned"] == [] and man["ledger_upload"] is None           # no ledger was written: neither order ran
    man = store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [200, 300]})
    assert man["pruned"] == [100] and man["ledger_upload"] == "upload_first"

    def refused(**kw):
        raise TypeError(f"preupload_lfs_files() got an unexpected keyword argument (token {TOKEN})")

    api.preupload_lfs_files = refused
    man = store.push_generation("ckpt-N", "pt", 500, staged_gen(tmp_path, 500), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [300, 400]})
    assert man["pruned"] == [200] and man["ledger_upload"].startswith("fallback (TypeError: preupload_lfs_files() got an unexpected")
    assert TOKEN not in man["ledger_upload"] and hf_store.ledger_steps(store.b.tree("ckpt-N")) == [100, 200]
    # A store on local disk has no upload phase: nothing to say.
    d = dstore(tmp_path)
    push_gens(tmp_path / "d", d, [100, 200, 300])
    man = d.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path / "d", 400), None, d.head("ckpt-N")["commit"],
                            prune={"keep": 2, "verified": [200, 300]})
    assert man["pruned"] == [100] and man["ledger_upload"] is None
    assert "FOUR" in hf_store.__doc__ and "3 x 12" in hf_store.__doc__


# ---------------------------------------------------------------- the arms' round, second pass (its two reviews, 12 Nov 2026)
def test_a_refusal_is_never_read_as_absence_and_no_signed_address_is_kept(tmp_path):
    """The reviews of the arms' round. (1) probe() read the library's error class before the status, so a 401 or
    403 that the library happened to class as "entry not found" was an absence, a verdict. The status is looked
    at first now. (2) A transport error after the hub's redirect carries the signed address ("Max retries exceeded
    with url: /path?X-Amz-..."); what crosses bounded()'s pipe, and what no_query() is given, keeps the path and
    drops the query. (3) auth_failure() tells the hub's refusal of the token from the object store's 403."""
    pytest.importorskip("huggingface_hub")
    api = FakeApi()
    store, _ = hstore(tmp_path, api)
    man = store.push_generation("ckpt-N", "pt", 100, staged_gen(tmp_path, 100), IDENT)
    size = man["files"]["model.safetensors"]["bytes"]
    for code in (401, 403):
        r = _reply(code)
        r.headers["X-Error-Code"] = "EntryNotFound"          # the library raises EntryNotFoundError for this header
        session = type("S", (), {"get": lambda self, url, r=r, **kw: r})()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(FakeApi, "http_session", property(lambda self: session))
            with pytest.raises(hf_store.StoreError, match=f"HTTP {code} on the byte read") as e:
                store.b.probe("ckpt-N", "gens/g100/model.safetensors", size=size)
            assert hf_store.auth_failure(e.value)                # from the hub: the token's refusal
            with pytest.raises(hf_store.StoreError, match=f"HTTP {code}"):
                store.listed_whole("ckpt-N", 100, man["files"])  # unread, never "not whole"
    r404 = _reply(404)
    r404.headers["X-Error-Code"] = "EntryNotFound"
    session = type("S", (), {"get": lambda self, url, **kw: r404})()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeApi, "http_session", property(lambda self: session))
        assert store.b.probe("ckpt-N", "gens/g100/model.safetensors", size=size) is False   # the hub's own absence stays one
    # (3)
    assert hf_store.auth_failure(hf_store.StoreError("HTTP 403 on the byte read of b:p (from the hub; refused)"))
    assert not hf_store.auth_failure(hf_store.StoreError("HTTP 403 on the byte read of b:p (from the object store, after the hub's redirect; refused)"))
    assert hf_store.auth_failure(Exception("401 Client Error. Repository Not Found for url: https://huggingface.co/api/datasets/x"))
    assert not hf_store.auth_failure(Exception("503 Server Error")) and not hf_store.auth_failure(ConnectionError("reset"))
    # (2)
    signed = ("HTTPSConnectionPool(host='cdn-lfs.example', port=443): Max retries exceeded with url: "
              "/repos/ab/cd/0123?X-Amz-Algorithm=AWS4&X-Amz-Signature=SECRETSIG (Caused by ReadTimeoutError)")
    assert hf_store.no_query(signed) == ("HTTPSConnectionPool(host='cdn-lfs.example', port=443): Max retries exceeded with url: "
                                         "/repos/ab/cd/0123?[query removed] (Caused by ReadTimeoutError)")
    assert hf_store.no_query("for url: https://cdn-lfs.example/o/1?sig=SECRETSIG&x=1.") == "for url: https://cdn-lfs.example/o/1?[query removed]"
    assert hf_store.no_query("no address here? none") == "no address here? none"

    def transport():
        raise ConnectionError(signed)
    with pytest.raises(hf_store.StoreError) as e:
        hf_store.bounded(transport, 30, TOKEN)
    assert "SECRETSIG" not in str(e.value) and "/repos/ab/cd/0123?[query removed]" in str(e.value)
    # What a completed "not whole" saw is kept for the caller's record.
    assert store.why_not_whole("ckpt-N", 100, man["files"]) == [True, None]
    assert store.why_not_whole("ckpt-N", 200) == [False, "at ckpt-N@1: the manifest is not listed"]
    api.branches["ckpt-N"]["gens/g100/optim.safetensors"] = b"short"
    whole, saw = store.why_not_whole("ckpt-N", 100)
    assert whole is False and "optim.safetensors is listed at 5 bytes, the manifest says" in saw


def test_a_release_names_the_ledger_that_did_not_read_and_a_runner_takes_its_own_lock_again(tmp_path):
    """The reviews of the arms' round. (1) A release that met a ledger it could not read said how many, not which:
    a ledger that is not well formed never reads, stops every release of the repo, and a person has to find it.
    The result names each by branch and path, and says which kind. (2) A runner that died inside a release left
    its lock on main for half an hour, and the runner started again in its place could not release. A lock that
    carries the releaser's own name is taken again when the caller says it is the only one of that name."""
    pytest.importorskip("huggingface_hub")
    api = FakeApi()
    store, logs = hstore(tmp_path, api, clock=lambda: api.now)
    for step in (100, 200, 300, 400):
        store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step))
    assert store.prune("ckpt-N", verified=[200, 300, 400]) == [100, 200]
    late = api.now + 10 ** 6
    api.branches["ckpt-N"]["pruned/g200.json"] = b"{not a ledger"
    real = api.hf_hub_download

    def g100_times_out(**kw):
        if kw["filename"] == "pruned/g100.json":
            raise ConnectionError("read timed out")
        return real(**kw)

    api.hf_hub_download = g100_times_out
    r = store.release_report("home:pk", grace=600, now=late)
    assert (r["held"], r["objects"], r["ledgers_unread"]) == (False, 0, 2)
    assert r["unread"] == ["ckpt-N:pruned/g100.json (not read (ConnectionError))", "ckpt-N:pruned/g200.json (not well formed)"]
    # The third reviews of the arms' round: a ledger that is not well formed is told by its OWN identity, its path
    # and the sha256 of its bytes, not by its branch's head, which every save of a training arm moves.
    assert r["malformed"] == {"ckpt-N:pruned/g200.json": S.sha(b"{not a ledger")}
    assert store.file_sha256("ckpt-N", "pruned/g200.json") == S.sha(b"{not a ledger")
    assert store.file_sha256("ckpt-N", "pruned/g999.json") is None
    api.hf_hub_download = real
    r = store.release_report("home:pk", grace=600, now=late)
    assert r["unread"] == ["ckpt-N:pruned/g200.json (not well formed)"] and r["objects"] == 0   # it stops the round by itself
    api.branches["ckpt-N"]["pruned/g200.json"] = json.dumps({"format": hf_store.PRUNED_FORMAT, "at": "yesterday", "objects": []}).encode()
    assert store.release_report("home:pk", grace=600, now=late)["unread"] == ["ckpt-N:pruned/g200.json (not well formed)"]
    del api.branches["ckpt-N"]["pruned/g200.json"]
    # (2) The releaser dies with the lock taken. The lock's record names the process that took it.
    mine = store._lock("home:pk", late)
    assert mine and hf_store.LOCK in api.branches["main"]
    assert (mine["pid"], mine["host"]) == (os.getpid(), hf_store.HOST)
    r = store.release_report("home:pk", grace=600, now=late + 60)
    assert r["held"] is True and r["why"] == "another releaser holds the lock"            # as before: waited out
    assert store.release_report("home:other", grace=600, now=late + 60, takeover=900)["held"] is True   # not another name's
    # The name alone is not enough (the second reviews of the arms' round): the process that took the lock is
    # alive here (it is this test), and the lock is younger than a release call may last: it may still be running.
    r = store.release_report("home:pk", grace=600, now=late + 60, takeover=900)
    assert r["held"] is True and "own earlier release may still be running (pid %d on %s" % (os.getpid(), hf_store.HOST) in r["why"]
    assert hf_store.LOCK in api.branches["main"]
    # Older than a release call may last: over, whatever the pid says. Taken again.
    r = store.release_report("home:pk", grace=600, now=late + 901, takeover=900)
    assert (r["held"], r["objects"], r["unread"]) == (False, 2, []) and hf_store.LOCK not in api.branches["main"]
    assert any("one left by home:pk itself is taken again (its process is past the 900 s" in ln for ln in logs)
    # Young, and its process is gone from this host: taken again at once. On another host, or with no pid: not.
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    for rec, taken in (({"pid": gone.pid, "host": hf_store.HOST}, True), ({"pid": gone.pid, "host": "elsewhere"}, False), ({}, False)):
        api.branches["main"][hf_store.LOCK] = json.dumps({"owner": "home:pk", "at": late + 2000, "nonce": "n", **rec}).encode()
        r = store.release_report("home:pk", grace=600, now=late + 2060, takeover=900)
        assert (r["held"] is False) == taken, rec
        api.branches["main"].pop(hf_store.LOCK, None)
    assert any("itself is taken again (its process is gone)" in ln for ln in logs)
    # The third reviews: "may last" is the limit of the call that TOOK the lock, which its record carries; a later
    # call with a short limit (the last second of the owed wait) does not take a young lock of a live process.
    mine = store._lock("home:pk", late + 5000, takeover=900)
    assert mine["limit"] == 900.0
    r = store.release_report("home:pk", grace=600, now=late + 5060, takeover=1)
    assert r["held"] is True and "may still be running" in r["why"]
    assert store.release_report("home:pk", grace=600, now=late + 5901, takeover=1)["held"] is False
    # An unlock is sent at most twice; a lock that is left is said, and is not removed by anyone else's nonce.
    real_commit, sent = api.create_commit, []

    def no_unlock(**kw):
        if kw["commit_message"].startswith("unlock"):
            sent.append(1)
            raise RuntimeError("412 Precondition Failed: main moved")
        return real_commit(**kw)

    api.create_commit = no_unlock
    r = store.release_report("home:pk", grace=600, now=late + 9000)
    assert r["held"] is False and len(sent) == 2 == hf_store.UNLOCK_TRIES and hf_store.LOCK in api.branches["main"]
    assert any("release lock not removed (RuntimeError" in ln and "it expires in 1800 s" in ln for ln in logs)
    api.create_commit = real_commit


def test_a_store_call_dies_with_the_process_that_made_it(tmp_path):
    """The second reviews of the arms' round: bounded() forked a child that outlived a parent killed with SIGKILL:
    a dead runner's release went on, holding the token and the lock, while the runner started in its place took
    that lock again by its name. The child now carries a parent-death signal."""
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "caller.py"
    script.write_text(
        "import os, sys, time\n"
        f"sys.path.insert(0, {str(ROOT / 'tools' / 'colab_chat')!r})\n"
        "import hf_store\n"
        "def call():\n"
        f"    open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(600)\n"
        "hf_store.bounded(call, 900)\n")
    p = subprocess.Popen([sys.executable, str(script)])
    child = None
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            time.sleep(0.1)
        child = int(pidfile.read_text())
        assert child != p.pid and hf_store._alive(child)
        p.kill()                                         # SIGKILL: the parent runs no clean-up of its own
        p.wait()
        def ended(pid):                                  # gone, or dead and not yet reaped
            try:
                return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
            except OSError:
                return True

        gone = False
        for _ in range(50):
            if ended(child):
                gone = True
                break
            time.sleep(0.1)
        assert gone, "the store call's child outlived its parent"
    finally:
        p.kill()
        if child and hf_store._alive(child):
            os.kill(child, 9)
    # The third reviews: prctl answers a failure by its return value, which nothing read; the comment said the
    # call's limit would still end the child, and the limit is kept by the parent that is then gone. On Linux a
    # signal that could not be set fails the call, before the function runs.
    ran = tmp_path / "ran"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hf_store, "_set_pdeathsig", lambda: -1)
        with pytest.raises(hf_store.StoreError, match="parent-death signal could not be set"):
            hf_store.bounded(lambda: ran.write_text("x") and 1, 30)
    assert not ran.exists()
    assert hf_store.bounded(lambda: ran.write_text("x") and 1, 30) == 1 and ran.exists()     # set: the call is made
    assert hf_store._LIBC and hf_store._LIBC[0] is not None                                # libc was loaded before the fork


def test_no_signed_address_is_kept_and_a_refusal_is_the_hubs_only_when_it_is_known_to_be(tmp_path, capsys):
    """The second reviews of the arms' round. (1) Query strings were stripped where an error crossed bounded() or
    went through the runner's store_err, and not where the store itself writes: fetch()'s retry line (in the child,
    before the pipe), retry(), the CLI's failure text, the upload fallback's text. A download that fails after the
    hub's redirect carries the signed address in its error; none of those keeps it now. (2) Any 401 or 403 that did
    not say "object store" was named the hub's refusal of the token. It is the hub's only when the reply is known
    to be: no redirect before it and the hub's own host. An object store's or a CDN's is said as that."""
    pytest.importorskip("huggingface_hub")
    import requests
    api = FakeApi()
    store, logs = hstore(tmp_path, api)
    signed = ("HTTPSConnectionPool(host='cdn-lfs.example', port=443): Max retries exceeded with url: "
              "/repos/ab/cd/0123?X-Amz-Algorithm=AWS4&X-Amz-Signature=SECRETSIG (Caused by ReadTimeoutError)")

    def redirected_download_fails(**kw):
        raise requests.ConnectionError(signed)

    api.hf_hub_download = redirected_download_fails
    with pytest.raises(requests.ConnectionError):
        store.fetch("main", "marks/pt/rung_100/model.safetensors", str(tmp_path / "m"))
    kept = [ln for ln in logs if "fetch of main:marks/pt/rung_100/model.safetensors failed" in ln]
    assert len(kept) == 3 and all("SECRETSIG" not in ln and "/repos/ab/cd/0123?[query removed]" in ln for ln in kept)
    said = []
    with pytest.raises(requests.ConnectionError):
        hf_store.retry(lambda: redirected_download_fails(), tries=2, delay=0, log=said.append, what="pull", token=TOKEN)
    assert len(said) == 1 and "SECRETSIG" not in said[0] and "?[query removed]" in said[0]
    store.log(f"anything at all: https://cdn-lfs.example/o/1?sig=SECRETSIG and {TOKEN}")
    assert "SECRETSIG" not in logs[-1] and TOKEN not in logs[-1]
    cfg = tmp_path / "cli.json"
    cfg.write_text(json.dumps({"repo": "someone/base2-store", "token_file": str(tmp_path / "token")}))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hf_store.HfBackend, "tree", lambda self, branch, rev=None: (_ for _ in ()).throw(requests.ConnectionError(signed)))
        assert hf_store.main(["steps", "--config", str(cfg), "--branch", "ckpt-N"]) == 1
    err = capsys.readouterr().err
    assert "hf_store steps failed: ConnectionError" in err and "SECRETSIG" not in err and TOKEN not in err
    api.hf_hub_download = FakeApi.hf_hub_download.__get__(api)
    for step in (100, 200, 300):
        store.push_generation("ckpt-N", "pt", step, staged_gen(tmp_path, step), None, None if step == 100 else store.head("ckpt-N")["commit"])
    api.preupload_lfs_files = lambda **kw: (_ for _ in ()).throw(TypeError("bad call for https://cdn-lfs.example/up?sig=SECRETSIG"))
    man = store.push_generation("ckpt-N", "pt", 400, staged_gen(tmp_path, 400), None, store.head("ckpt-N")["commit"],
                                prune={"keep": 2, "verified": [200, 300]})
    assert man["ledger_upload"].startswith("fallback (TypeError") and "SECRETSIG" not in man["ledger_upload"]

    # (2)
    def http_error(code, url, redirected=False):
        r = _reply(code, redirected=redirected)
        if not redirected:
            r.url = url
        e = requests.HTTPError(f"{code} Client Error: refused for url: {r.url}")
        e.response = r
        return e

    hub = http_error(403, "https://huggingface.co/api/datasets/someone/base2-store/commit/ckpt-N")
    cdn = http_error(403, "https://cdn-lfs.example/objects/abc", redirected=True)
    s3 = http_error(403, "https://s3.example/presigned/put/abc")                 # an upload's signed address: no redirect, another host
    assert (hf_store.reply_side(hub), hf_store.reply_side(cdn), hf_store.reply_side(s3)) == ("hub", "object store", "object store")
    assert hf_store.auth_failure(hub) and not hf_store.auth_failure(cdn) and not hf_store.auth_failure(s3)
    assert hf_store.reply_side(Exception("403 Forbidden")) is None and not hf_store.auth_failure(Exception("403 Client Error: Forbidden"))
    assert not hf_store.auth_failure(http_error(404, "https://huggingface.co/api/datasets/x"))
    # The third reviews: the host of the FINAL reply decides; a redirect that ends back on the hub (a renamed repo)
    # is the hub's refusal, not an object store's.
    back = http_error(403, "https://huggingface.co/api/datasets/someone/renamed/tree/main")
    back.response.history = [_resp(301)]
    assert hf_store.reply_side(back) == "hub" and hf_store.auth_failure(back)
    # The fourth reviews: the byte read wrote "from the object store" for any redirected reply, and those words then
    # decided. It goes by the final host too.
    for final, side in (("https://huggingface.co/datasets/someone/renamed/resolve/x/f", "from the hub"),
                        ("https://cdn-lfs.example/objects/abc", "from the object store")):
        r = _reply(403, redirected=True)
        r.url = final
        session = type("S", (), {"get": lambda self, url, r=r, **kw: r})()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(FakeApi, "http_session", property(lambda self: session))
            with pytest.raises(hf_store.StoreError) as got:
                store.b.probe("ckpt-N", "gens/g100/model.safetensors", size=1)
        assert side in str(got.value) and hf_store.auth_failure(got.value) == (side == "from the hub")
    # The side crosses bounded()'s pipe, in words.
    for e, side in ((hub, "hub"), (s3, "object store")):
        def refuse(e=e):
            raise e
        with pytest.raises(hf_store.StoreError) as got:
            hf_store.bounded(refuse, 30, TOKEN)
        assert hf_store.reply_side(got.value) == side and hf_store.auth_failure(got.value) == (side == "hub")
    class Bare(Exception):
        response = type("R", (), {"status_code": 403, "url": "https://s3.example/put/abc?sig=SECRETSIG", "history": []})()

    def bare():
        raise Bare("refused")
    with pytest.raises(hf_store.StoreError) as got:
        hf_store.bounded(bare, 30, TOKEN)
    assert "[HTTP 403] [from the object store]" in str(got.value) and "SECRETSIG" not in str(got.value)


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
