"""runner_multi.py: a data job that finished on the VM before the deadline has its outputs brought home before the
session is stopped (26 Nov 2026: the SAT and SATW CUDA smoke's checks job exited 0 on the VM 25 s before the deadline,
the loop had last read it as RUNNING, and its result file was lost with the session). Offline: no VM, no network."""
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


JOBS = [{"name": "train"},                                              # a training job: never fetched here
        {"name": "checks", "cmd": ["x"], "outputs": ["smoke_vm.json"]},
        {"name": "later", "cmd": ["x"], "outputs": ["a.json", "b.json"]},
        {"name": "running", "cmd": ["x"], "outputs": ["c.json"]}]


def setup(tmp_path, monkeypatch, fail=()):
    art = tmp_path / "art"
    art.mkdir()
    monkeypatch.setattr(rm, "ART", str(art))
    monkeypatch.setattr(rm, "HOME", str(tmp_path))
    monkeypatch.setattr(rm, "T", FakeColab())
    monkeypatch.setattr(rm, "log", lambda m: None)
    calls = []

    def dl(src, dst, timeout=1800):
        calls.append(src)
        if any(src.endswith(f) for f in fail):
            return 1
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("x")
        return 0
    monkeypatch.setattr(rm, "download_file", dl)
    return art, calls


STATUS = {"vm": "RUNNING", "train": "FINISHED", "checks": "FINISHED", "later": "FINISHED", "running": "RUNNING"}


def test_finished_data_jobs_come_home_whole_at_the_deadline(tmp_path, monkeypatch):
    art, calls = setup(tmp_path, monkeypatch)
    st = rm.State()
    st.job("checks")["status"] = "RUNNING"          # as the loop last read it, a tick before
    got = rm.deadline_fetch(st, JOBS, status=dict(STATUS))
    assert got == ["checks", "later"]
    assert (art / "checks" / "smoke_vm.json").exists() and (art / "checks" / "checks_train.log").exists()
    assert (art / "later" / "a.json").exists() and (art / "later" / "b.json").exists()
    assert st.job("checks")["home"] and st.job("checks")["status"] == "FINISHED"
    assert not st.job("running")["home"] and not st.job("train")["home"]
    assert not any("c.json" in c or "/train/" in c for c in calls)
    assert len(calls) == 5                           # every file of both jobs in this one call, not one a tick
    saved = json.loads((tmp_path / "state.json").read_text())["jobs"]
    assert saved["checks"]["home"] and saved["later"]["home"]


def test_a_job_already_home_is_not_fetched_again_and_a_silent_vm_fetches_nothing(tmp_path, monkeypatch):
    art, calls = setup(tmp_path, monkeypatch)
    st = rm.State()
    st.job("checks")["home"] = True
    monkeypatch.setattr(rm, "vm_status", lambda jobs: None)
    assert rm.deadline_fetch(st, JOBS) == [] and calls == []
    monkeypatch.setattr(rm, "vm_status", lambda jobs: dict(STATUS))      # the status is read from the VM when none is given
    assert rm.deadline_fetch(st, JOBS) == ["later"]
    assert not any("smoke_vm.json" in c for c in calls)


def test_a_failed_download_ends_that_job_without_looping_and_the_next_job_is_still_tried(tmp_path, monkeypatch):
    art, calls = setup(tmp_path, monkeypatch, fail=("smoke_vm.json",))
    st = rm.State()
    got = rm.deadline_fetch(st, JOBS, status=dict(STATUS))
    assert got == ["later"] and not st.job("checks")["home"]
    assert sum("smoke_vm.json" in c for c in calls) == 1


def test_the_time_limit_bounds_the_fetch(tmp_path, monkeypatch):
    art, calls = setup(tmp_path, monkeypatch)
    st = rm.State()
    t = [0.0]

    def clock():
        t[0] += 100.0                                # every look at the clock costs 100 s
        return t[0]
    got = rm.deadline_fetch(st, JOBS, status=dict(STATUS), limit_s=250, clock=clock)
    assert got == [] or got == ["checks"]            # the limit ends the work; nothing runs past it
    assert len(calls) <= 2


def test_a_transport_with_a_deadline_is_given_the_fetch_time_and_no_more(tmp_path, monkeypatch):
    art, calls = setup(tmp_path, monkeypatch)

    class WithDeadline(FakeColab):
        deadline = 1000.0
    monkeypatch.setattr(rm, "T", WithDeadline())
    rm.deadline_fetch(rm.State(), JOBS, status=dict(STATUS), limit_s=300, clock=lambda: 1000.0)
    assert rm.T.deadline == 1300.0


def test_the_loop_calls_it_at_the_deadline_before_the_stop():
    src = (ROOT / "tools" / "colab_chat" / "runner_multi.py").read_text()
    main = src[src.index("def main("):]
    call = main.index("deadline_fetch(st, jobs)")
    assert main.index("while not home_already and time.time() < deadline:") < call < main.index('stop_session(st, "at the runner\'s end")')
    guard = main[main.rindex("if (not home_already", 0, call):call]
    for cond in ("not expired", "time.time() >= deadline", 'st.d.get("endpoint")', 'not st.d.get("campaign_home")'):
        assert cond in guard, cond
    assert rm.DEADLINE_FETCH_S == 300
