"""runner_multi.py: a data job whose outputs have all landed is home and FINISHED at once, and a state dir whose
job has every output home but is unmarked (written before that) is treated as home when it is adopted, so a
replacement VM does not run the job again (LOCAL_LLM_WORK E5.3: post_e1_rec_MC7_r2's summary.json and log were
home at 18:20 on 26 Oct, the VM was lost before the next tick, and two relaunches reran it, 1.5 L4-hours each)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import runner_multi as rm  # noqa: E402


class FakeColab:
    """The Colab transport's shape for fetch_outputs: one file per call (whole_files False)."""
    whole_files = False
    persistent = False


def setup(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    monkeypatch.setattr(rm, "ART", str(art))
    monkeypatch.setattr(rm, "HOME", str(tmp_path))
    monkeypatch.setattr(rm, "T", FakeColab())
    return art


def fake_download(art, calls):
    def dl(src, dst, timeout=1800):
        calls.append(src)
        Path(dst).write_text("x")
        return 0
    return dl


def test_home_and_finished_in_the_call_that_lands_the_last_file(tmp_path, monkeypatch):
    art = setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(rm, "download_file", fake_download(art, calls))
    st = rm.State()
    st.job("post")["status"] = "FINISHED"
    outs = ["summary.json", "sub/report.json"]
    for k in range(3):  # three files (two outputs and the log), one per tick
        assert not st.job("post")["home"]
        rm.fetch_outputs(st, "post", outs)
    js = st.job("post")
    assert js["home"] and js["status"] == "FINISHED"  # no fourth tick needed
    assert len(calls) == 3
    assert (art / "post" / "sub" / "report.json").exists() and (art / "post" / "post_train.log").exists()
    saved = json.loads((tmp_path / "state.json").read_text())["jobs"]["post"]
    assert saved["home"] and saved["status"] == "FINISHED"
    rm.fetch_outputs(st, "post", outs)  # a later call downloads nothing again
    assert len(calls) == 3


def test_a_failed_download_is_retried_and_does_not_mark_home(tmp_path, monkeypatch):
    art = setup(tmp_path, monkeypatch)
    calls = []
    ok = fake_download(art, calls)
    fails = {"n": 1}

    def flaky(src, dst, timeout=1800):
        if src.endswith("_train.log") and fails["n"]:
            fails["n"] -= 1
            return 1
        return ok(src, dst, timeout)
    monkeypatch.setattr(rm, "download_file", flaky)
    st = rm.State()
    rm.fetch_outputs(st, "j", ["a.json"])
    rm.fetch_outputs(st, "j", ["a.json"])  # the log fails once
    assert not st.job("j")["home"]
    rm.fetch_outputs(st, "j", ["a.json"])
    assert st.job("j")["home"] and st.job("j")["status"] == "FINISHED"


def jobs():
    return [{"name": "gate", "cmd": ["x"], "outputs": ["gate.json"]},
            {"name": "bad", "cmd": ["y"], "outputs": ["bad.json"]},
            {"name": "partial", "cmd": ["z"], "outputs": ["p.json"]},
            {"name": "train", "spec": {}}]


def landed(art, st, name, outputs, files=True, status="RUNNING"):
    js = st.job(name)
    js.update(status=status, fetched=rm.output_files(name, outputs))
    if files:
        for f in rm.output_files(name, outputs):
            p = art / name / (Path(f).name if f.startswith("../") else f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")


def test_adoption_marks_home_jobs_finished(tmp_path, monkeypatch):
    art = setup(tmp_path, monkeypatch)
    st = rm.State()
    landed(art, st, "gate", ["gate.json"])                       # every file home, state unmarked
    landed(art, st, "bad", ["bad.json"], status="FAILED")       # failures are never turned into FINISHED
    landed(art, st, "partial", ["p.json"], files=False)          # the state says fetched, the disk disagrees
    st.save()
    st = rm.State()  # a restarted runner adopting the state dir
    assert rm.adopt_home(st, jobs()) == ["gate"]
    assert st.job("gate")["home"] and st.job("gate")["status"] == "FINISHED"
    assert not st.job("bad")["home"] and st.job("bad")["status"] == "FAILED"
    assert not st.job("partial")["home"]
    assert "outputs already home" in (tmp_path / "runner.log").read_text()
    assert rm.adopt_home(st, jobs()) == []  # idempotent


def test_prepare_marks_the_adopted_job_finished_on_the_new_vm(tmp_path, monkeypatch):
    """prepare() on a replacement VM writes STATUS_gate = FINISHED (jobs_vm.py then never runs it), where
    before the fix the job was not home and ran again."""
    art = setup(tmp_path, monkeypatch)
    st = rm.State()
    landed(art, st, "gate", ["gate.json"])
    st.d["booted"] = False
    st.save()
    sent = []

    def fake_exec(code, timeout=200):
        sent.append(code)
        return 0, "DIRS_OK\nHOME_OK\nPENDING_OK\nINPUTS_OK\nBOOTED\n"
    monkeypatch.setattr(rm, "exec_py", fake_exec)
    assert rm.prepare(st, "c0ffee", "jobs.json", jobs(), [])
    home_code = next(c for c in sent if "HOME_OK" in c)
    finished = home_code.split("for n in ", 1)[1].split(":", 1)[0]
    assert "'gate'" in finished
    assert "'bad'" not in finished and "'partial'" not in finished
    assert st.d["booted"]
