"""tools/colab_chat/hf_store.py against a store on local disk (DirBackend) and a recording stand-in for
huggingface_hub.HfApi: a push is one commit that holds the manifest; a pull refuses a wrong hash or a missing file;
a resume takes the newest generation that verifies and falls back to the one before it; removal keeps the newest
two whole generations, never the newest valid one, never `main`; the token comes from a file, is refused when others
can read it, and is absent from every output. No network: nothing here reaches Hugging Face.
base2_seed.py's push-once rule is tested at the end."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
sys.path.insert(0, str(ROOT / "tests"))
import base2_seed  # noqa: E402
import hf_store  # noqa: E402
import store_standin as S  # noqa: E402

TOKEN = "hf_FAKEtokenFAKEtokenFAKEtoken0123"


def dstore(tmp_path, **kw):
    return hf_store.open_store({"repo": f"dir:{tmp_path / 'repo'}", **kw})


def commits(tmp_path):
    return [json.loads(ln) for ln in (tmp_path / "repo" / "commits.jsonl").read_text().splitlines()]


def staged_gen(tmp_path, step, seed=0):
    """A generation as the courier stages it: the trainer's four files and events.jsonl in one folder."""
    out = tmp_path / f"job{seed}"
    S.make_gen(str(out), step, seed)
    g = out / "ckpt_gen" / f"g{step}"
    (g / "events.jsonl").write_text(json.dumps({"event": "checkpoint", "step": step}) + "\n")
    return str(g)


def push_gens(tmp_path, store, steps, branch="ckpt-N"):
    for s in steps:
        store.push_generation(branch, "pt", s, staged_gen(tmp_path, s))


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


def test_generation_push_refuses_a_torn_folder_and_a_wrong_branch(tmp_path):
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


def test_prune_keeps_the_newest_two_and_never_touches_main(tmp_path):
    store = dstore(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ids").write_bytes(b"a")
    store.push_set("main", "corpus/N", str(src))
    (src / "state.txt").write_text("100\n{}\n")
    (src / "model.safetensors").write_bytes(S.state_at(100)[0])
    assert store.push_mark("pt", 100, str(src)) is True
    assert store.push_mark("pt", 100, str(src)) is False  # once
    push_gens(tmp_path, store, [100, 200])
    assert store.prune("ckpt-N") == []  # two: nothing to remove
    push_gens(tmp_path, store, [300, 400, 500])
    main_before = store.b.tree("main")
    assert store.prune("ckpt-N") == [100, 200, 300]
    assert store.steps("ckpt-N") == [400, 500]
    assert not (tmp_path / "repo" / "ckpt-N" / "gens" / "g300").exists()
    assert store.b.tree("main") == main_before and store.mark_steps("pt") == [100]
    c = commits(tmp_path)[-1]
    assert c["adds"] == [] and all(p.startswith("gens/g") for p in c["deletes"]) and len(c["deletes"]) == 18
    for branch in ("main", "marks", "ckpt"):
        with pytest.raises(hf_store.StoreError):
            store.prune(branch)
    with pytest.raises(hf_store.StoreError, match="at least two"):
        store.prune("ckpt-N", keep=1)
    # Both kept generations still verify.
    assert dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"))[0] == 500


def test_prune_never_removes_the_newest_valid_generation(tmp_path):
    """Generations that are not whole in the listing do not count towards the two: 300 is the newest whole one,
    and it and 200 stay however many newer, damaged ones there are."""
    store = dstore(tmp_path)
    push_gens(tmp_path, store, [100, 200, 300, 400, 500])
    gens = tmp_path / "repo" / "ckpt-N" / "gens"
    (gens / "g500" / "model.safetensors").write_bytes(b"short")  # size differs from the manifest
    (gens / "g400" / "optim.safetensors").unlink()                # a file is missing
    store = dstore(tmp_path)
    assert store.prune("ckpt-N") == [100]
    assert store.steps("ckpt-N") == [200, 300, 400, 500]
    step, rejected = dstore(tmp_path).pull_generation("ckpt-N", str(tmp_path / "r"))
    assert step == 300 and [s for s, _ in rejected] == [500, 400]


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


def test_adopt_and_forget_token(tmp_path):
    cfg, up, shm = tmp_path / "store.json", tmp_path / "upload_token", tmp_path / "shm"
    cfg.write_text(json.dumps({"repo": "someone/base2-store"}))
    with pytest.raises(hf_store.StoreError, match="no token was uploaded"):
        hf_store.adopt_token(str(cfg), str(up), str(shm))
    up.write_text(TOKEN)
    hf_store.adopt_token(str(cfg), str(up), str(shm))
    tf = json.loads(cfg.read_text())["token_file"]
    assert tf == str(shm / "token") and not up.exists() and os.stat(tf).st_mode & 0o777 == 0o600
    assert hf_store.read_token(tf) == TOKEN and TOKEN not in cfg.read_text()
    hf_store.adopt_token(str(cfg), str(up), str(shm))  # a second boot on the same VM: kept
    assert hf_store.forget_token(str(cfg), [str(up)]) is True and not os.path.exists(tf)
    assert hf_store.forget_token(str(cfg)) is True  # nothing left to remove is still "gone"


# ---------------------------------------------------------------- the Hugging Face backend, on a recording stand-in
class RevisionNotFoundError(Exception):
    pass


class FakeFile:
    def __init__(self, path, data, lfs=True):
        self.path, self.size = path, len(data)
        self.lfs = {"sha256": S.sha(data), "size": len(data)} if lfs else None


class FakeFolder:
    def __init__(self, path):
        self.path, self.tree_id = path, "t"


class FakeLfs:
    def __init__(self, oid):
        self.oid = oid


class FakeApi:
    """Records every call with its keyword arguments; branches are {path: bytes}."""

    def __init__(self):
        self.calls, self.branches, self.lfs, self.fail = [], {"main": {}}, set(), None

    def _rec(self, name, **kw):
        self.calls.append((name, kw))
        if self.fail:
            raise self.fail

    def create_branch(self, repo, *, branch, exist_ok=False, repo_type=None, token=None):
        self._rec("create_branch", repo=repo, branch=branch, repo_type=repo_type, token=token, exist_ok=exist_ok)
        self.branches.setdefault(branch, {})

    def list_repo_tree(self, repo, *, recursive=False, revision=None, repo_type=None, token=None):
        self._rec("list_repo_tree", repo=repo, revision=revision, repo_type=repo_type, token=token, recursive=recursive)
        if revision not in self.branches:
            raise RevisionNotFoundError(revision)
        files = self.branches[revision]
        return [FakeFolder("gens")] + [FakeFile(p, b, lfs=p.endswith(".safetensors")) for p, b in sorted(files.items())]

    def create_commit(self, repo, operations, *, commit_message, revision=None, repo_type=None, token=None):
        ops = list(operations)
        self._rec("create_commit", repo=repo, revision=revision, repo_type=repo_type, token=token, message=commit_message,
                  ops=[(type(o).__name__, o.path_in_repo) for o in ops])
        for o in ops:
            if type(o).__name__ == "CommitOperationAdd":
                data = open(o.path_or_fileobj, "rb").read()
                self.branches[revision][o.path_in_repo] = data
                if o.path_in_repo.endswith(".safetensors"):
                    self.lfs.add(S.sha(data))
            else:
                del self.branches[revision][o.path_in_repo]

    def hf_hub_download(self, repo, filename, *, revision=None, local_dir=None, repo_type=None, token=None):
        self._rec("hf_hub_download", repo=repo, filename=filename, revision=revision, repo_type=repo_type, token=token)
        p = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(self.branches[revision][filename])
        return p

    def super_squash_history(self, repo, *, branch=None, commit_message=None, repo_type=None, token=None):
        self._rec("super_squash_history", repo=repo, branch=branch, repo_type=repo_type, token=token)

    def list_lfs_files(self, repo, *, repo_type=None, token=None):
        self._rec("list_lfs_files", repo=repo, repo_type=repo_type, token=token)
        return [FakeLfs(o) for o in sorted(self.lfs)]

    def permanently_delete_lfs_files(self, repo, lfs_files, *, rewrite_history=True, repo_type=None, token=None):
        gone = [f.oid for f in lfs_files]
        self._rec("permanently_delete_lfs_files", repo=repo, oids=gone, rewrite_history=rewrite_history,
                  repo_type=repo_type, token=token)
        self.lfs -= set(gone)


def hstore(tmp_path, api, **kw):
    pytest.importorskip("huggingface_hub")
    tf = tmp_path / "token"
    tf.write_text(TOKEN + "\n")
    tf.chmod(0o600)
    logs = []
    cfg = {"repo": "someone/base2-store", "repo_type": "dataset", "token_file": str(tf), **kw}
    return hf_store.open_store(cfg, log=logs.append, api=api), logs


def test_hf_backend_calls(tmp_path):
    """The calls the real library receives: one create_commit per push with every file and the manifest, on the
    branch, with the token passed by keyword and the repo type; downloads by revision; the listing parsed."""
    api = FakeApi()
    store, _ = hstore(tmp_path, api)
    push_gens(tmp_path, store, [100])
    kinds = [c[0] for c in api.calls]
    assert kinds == ["create_branch", "create_commit"]
    kw = api.calls[1][1]
    assert kw["revision"] == "ckpt-N" and kw["repo"] == "someone/base2-store" and kw["repo_type"] == "dataset"
    assert kw["token"] == TOKEN
    assert kw["ops"] == [("CommitOperationAdd", f"gens/g100/{f}") for f in
                         sorted(["MANIFEST.json", "events.jsonl", "manifest.json", "model.safetensors",
                                 "optim.safetensors", "state.txt"])]
    assert store.b.tree("nope") is None  # a branch that does not exist yet
    tree = store.b.tree("ckpt-N")
    assert "gens" not in tree and tree["gens/g100/model.safetensors"]["sha256"] == S.sha(S.state_at(100)[0])
    assert tree["gens/g100/state.txt"]["sha256"] is None  # not an LFS file: size only
    step, _ = hstore(tmp_path, api)[0].pull_generation("ckpt-N", str(tmp_path / "r"))
    assert step == 100 and not any(d.startswith(".hfstage") for d in os.listdir(tmp_path / "r" / "g100"))
    assert all(c[1]["token"] == TOKEN and c[1]["repo_type"] == "dataset" for c in api.calls)
    assert ("create_branch", "main") not in [(c[0], c[1].get("branch")) for c in api.calls]


def test_hf_prune_releases_storage_but_never_a_marks_bytes(tmp_path):
    """A mark's weights are the same bytes as its generation's. Removing the generation must not delete that
    object: only objects no branch's listing still names are released, with history left alone on main."""
    api = FakeApi()
    store, logs = hstore(tmp_path, api, reclaim=True, branches={"pt": "ckpt-N", "pt2": "ckpt-S"})
    mark = tmp_path / "mark"
    mark.mkdir()
    (mark / "model.safetensors").write_bytes(S.state_at(100)[0])
    (mark / "state.txt").write_text("100\n{}\n")
    store.push_mark("pt", 100, str(mark))
    push_gens(tmp_path, store, [100, 200, 300])
    store.push_generation("ckpt-S", "pt2", 200, staged_gen(tmp_path, 200))  # another branch holds g200's bytes too
    api.calls.clear()
    assert store.prune("ckpt-N") == [100]
    names = [c[0] for c in api.calls]
    assert names.index("create_commit") < names.index("super_squash_history") < names.index("permanently_delete_lfs_files")
    squash = dict(api.calls)["super_squash_history"]
    assert squash["branch"] == "ckpt-N"
    dele = dict(api.calls)["permanently_delete_lfs_files"]
    m100, o100 = S.state_at(100)
    assert dele["oids"] == [S.sha(o100)] and dele["rewrite_history"] is False  # the optimiser only: the mark keeps the weights
    assert S.sha(m100) in api.lfs and S.sha(S.state_at(200)[0]) in api.lfs
    assert any("released" in ln for ln in logs)
    # Without "reclaim" the generations leave the branch and nothing else is called.
    api2 = FakeApi()
    store2, _ = hstore(tmp_path, api2)
    push_gens(tmp_path, store2, [100, 200, 300])
    api2.calls.clear()
    assert store2.prune("ckpt-N") == [100]
    assert "super_squash_history" not in [c[0] for c in api2.calls]
    # A failed release is logged and the prune still stands.
    api3 = FakeApi()
    store3, logs3 = hstore(tmp_path, api3, reclaim=True)
    push_gens(tmp_path, store3, [100, 200, 300])
    api3.super_squash_history = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"403 for {TOKEN}"))
    assert store3.prune("ckpt-N") == [100] and store3.steps("ckpt-N") == [200, 300]
    assert any("storage not released" in ln for ln in logs3) and not any(TOKEN in ln for ln in logs3)


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
    rc = hf_store.main(["push", "--config", str(cfg), "--branch", "main", "--prefix", "x", "--dir", str(src)])
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
    r = run("inputs", "--config", str(cfg), "--jobs", str(jobs), "--root", str(tmp_path / "vm"))
    assert r.returncode == 0 and "1 files" in r.stdout, r.stderr
    assert (tmp_path / "vm" / "chat" / "data" / "N.main.ids").read_bytes() == b"\x01\x02" * 500
    (tmp_path / "repo" / "main" / "corpus" / "N" / "N.main.ids").write_bytes(b"\x01\x03" * 500)
    (tmp_path / "vm" / "chat" / "data" / "N.main.ids").unlink()
    r = run("inputs", "--config", str(cfg), "--jobs", str(jobs), "--root", str(tmp_path / "vm"))
    assert r.returncode == 1 and "sha256 or size differs" in r.stderr and "Traceback" not in r.stderr
    assert json.loads(run("steps", "--config", str(cfg), "--branch", "ckpt-N").stdout)["head"] is None
    assert run("prune", "--config", str(cfg), "--branch", "main").returncode == 1


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
