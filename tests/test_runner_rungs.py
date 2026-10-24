"""runner_multi.py's rung-snapshot backstop, in process: what prepare() puts back on a VM (a RunPod volume that
already holds a snapshot gets nothing; a job resuming before the step makes it again), the events.jsonl beside
it, a reset job's stale FAILED status removed, no VM call for a campaign without rung jobs, the home copy's
deletion rule, and apply_rung_resets touching only the requested job. The VM is a local directory; exec runs
the generated code there. tests/runner_multi_equiv/run_rungs.py covers the same end to end on the fake CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import runner_multi as rm  # noqa: E402


class LocalVM:
    """A transport whose /content is a local directory (persistent: a RunPod volume)."""
    persistent, whole_files = True, True

    def __init__(self, root):
        self.root, self.calls = str(root), 0

    def exec(self, code, timeout=200):
        self.calls += 1
        p = subprocess.run([sys.executable, "-c", code.replace("/content/", self.root + "/")],
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr


def rung(name, pt, step):
    return {"name": name, "after": "data", "rung_of": {"job": pt, "step": step}, "cmd": ["r"], "outputs": ["rung.json"]}


JOBS = [{"name": "data", "cmd": ["x"], "outputs": []},
        {"name": "pt", "snapshot_steps": [100, 200], "spec": {}},
        {"name": "pt2", "snapshot_steps": [100, 500], "spec": {}},
        rung("r_pt_100", "pt", 100), rung("r_pt_200", "pt", 200), rung("r_pt2_100", "pt2", 100),
        rung("r_pt2_500", "pt2", 500)]


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)


def setup(tmp_path, monkeypatch, state):
    art, home, vm = tmp_path / "art", tmp_path / "state", tmp_path / "vm"
    home.mkdir()
    (home / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(rm, "ART", str(art))
    monkeypatch.setattr(rm, "HOME", str(home))
    t = LocalVM(vm)
    monkeypatch.setattr(rm, "T", t)
    return art, home, vm / "chat", t


def snap(art, pt, step, body=b"w"):
    write(art / pt / "rung_snapshots" / f"rung_{step}" / "model.safetensors", body)
    write(art / pt / "rung_snapshots" / f"rung_{step}" / "state.txt", f"{step}\n{{}}\n")


def test_rung_inputs(tmp_path, monkeypatch):
    st0 = {"endpoint": None, "launches": 1, "jobs": {
        "pt": {"relayed_step": 0, "status": "FINISHED", "home": True},
        "pt2": {"relayed_step": 300, "status": "RUNNING", "home": False},
        "r_pt2_100": {"relayed_step": 0, "status": None, "home": False, "rung_reset": 1.0}}}
    art, home, chat, t = setup(tmp_path, monkeypatch, st0)
    for pt, s in (("pt", 100), ("pt", 200), ("pt2", 100), ("pt2", 500)):
        snap(art, pt, s)
    write(art / "pt" / "events.jsonl", '{"event": "eval", "step": 100}\n')
    write(art / "pt2" / "state.txt", "300\n{}\n")  # pt2 resumes from its relayed step 300
    write(art / "pt2" / "events.jsonl", "{}\n")
    write(chat / "pt" / "rung_100" / "model.safetensors", b"w")  # the volume holds pt's step-100 snapshot
    write(chat / "pt" / "rung_100" / "state.txt", "100\n")
    write(chat / "pt2" / "events.jsonl", "{}\n")
    write(chat / "STATUS_r_pt2_100", "FAILED\n")
    st = rm.State()
    up = rm.rung_inputs(st, JOBS)
    dst = sorted(d for _, _, d in up)
    assert dst == ["/content/chat/pt/events.jsonl", "/content/chat/pt/rung_200/model.safetensors",
                   "/content/chat/pt/rung_200/state.txt", "/content/chat/pt2/rung_100/model.safetensors",
                   "/content/chat/pt2/rung_100/state.txt"]  # pt2 step 500: pt2 resumes at 300 and makes it again
    assert all(os.path.exists(src) for _, src, _ in up) and len({k for k, _, _ in up}) == len(up)
    assert not (chat / "STATUS_r_pt2_100").exists() and "rung_reset" not in st.job("r_pt2_100")
    log = (home / "runner.log").read_text()
    assert "pt: rung_100 snapshot already on this VM (volume); nothing uploaded" in log
    assert t.calls == 1


def test_no_rung_jobs_no_call(tmp_path, monkeypatch):
    art, home, chat, t = setup(tmp_path, monkeypatch, {"endpoint": None, "launches": 1, "jobs": {}})
    st = rm.State()
    assert rm.rung_inputs(st, JOBS[:3]) == [] and t.calls == 0
    rm.fetch_rung_snapshot(st, JOBS[:3], {"pt": "RUNNING"}, False)
    assert t.calls == 0 and "rung_snapshots" not in st.d


def test_fetch_and_delete(tmp_path, monkeypatch):
    jobs = JOBS[:4]
    art, home, chat, t = setup(tmp_path, monkeypatch, {"endpoint": None, "launches": 1, "jobs": {}})
    monkeypatch.setattr(rm, "download_file", lambda src, dst, timeout=1800: subprocess.run(
        ["cp", src.replace("/content/", str(tmp_path / "vm") + "/"), dst]).returncode)
    write(chat / "pt" / "rung_100" / "model.safetensors", b"weights")
    write(chat / "pt" / "rung_100" / "state.txt", "100\n")
    st = rm.State()
    r = {"pt": "RUNNING", "r_pt_100": "WAITING"}
    rm.fetch_rung_snapshot(st, jobs, r, True)  # a relay's tick: nothing fetched
    assert t.calls == 0
    rm.fetch_rung_snapshot(st, jobs, r, False)
    assert (art / "pt" / "rung_snapshots" / "rung_100" / "model.safetensors").read_bytes() == b"weights"
    assert st.d["rung_snapshots"]["pt/rung_100"]["bytes"] == 7
    rm.fetch_rung_snapshot(st, jobs, r, False)  # home already: the VM is not asked for it again
    assert t.calls == 1
    st.job("r_pt_100").update(home=True, status="FINISHED")
    rm.fetch_rung_snapshot(st, jobs, {"pt": "FINISHED", "r_pt_100": "FINISHED"}, False)
    assert not (art / "pt" / "rung_snapshots" / "rung_100").exists() and st.d["rung_snapshots"]["pt/rung_100"]["deleted"]


def test_apply_rung_resets(tmp_path, monkeypatch):
    jobs = {"r_pt_100": {"relayed_step": 0, "status": "FAILED", "home": True, "failed_seen": 2,
                         "fetched": ["../r_pt_100_train.log"]},
            "r_pt_200": {"relayed_step": 0, "status": "FAILED", "home": True, "failed_seen": 2},
            "r_pt2_100": {"relayed_step": 0, "status": "FAILED", "home": True, "failed_seen": 2},
            "pt": {"relayed_step": 300, "status": "FINISHED", "home": True}}
    art, home, chat, t = setup(tmp_path, monkeypatch, {"endpoint": "e", "launches": 2, "jobs": jobs,
                                                         "campaign_home": True})
    for n in ("r_pt_100", "r_pt_200"):
        write(art / n / f"{n}_train.log", "rung lost: no snapshot\n")
    write(art / "r_pt2_100" / "r_pt2_100_train.log", "Traceback: a scoring failure\n")
    snap(art, "pt", 100)
    snap(art, "pt2", 100)
    (home / rm.RESET_REQUEST).write_text(json.dumps({"rungs": ["r_pt_100", "r_pt_200", "r_pt2_100"]}))
    st = rm.State()
    assert [x[0] for x in rm.lost_rungs(st.d, JOBS)] == ["r_pt_100", "r_pt_200"]
    rm.apply_rung_resets(st, JOBS)
    d = json.loads((home / "state.json").read_text())
    assert d["jobs"]["r_pt_100"]["home"] is False and d["jobs"]["r_pt_100"]["status"] is None
    assert d["jobs"]["r_pt_100"]["rung_reset"]
    for n in ("r_pt_200", "r_pt2_100", "pt"):  # no home snapshot; not "rung lost"; not a rung job
        assert d["jobs"][n] == jobs[n]
    assert "campaign_home" not in d and not (home / rm.RESET_REQUEST).exists()
    assert (art / "r_pt_100" / "r_pt_100_train.lost.log").exists() and not (art / "r_pt_100" / "r_pt_100_train.log").exists()
    log = (home / "runner.log").read_text()
    assert "r_pt_200: rung reset refused" in log and "r_pt2_100: rung reset refused" in log
