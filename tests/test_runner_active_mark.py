"""runner_multi.py, A32 launch round 10 (GPT round 10, B2): with home markers on, the active mark is saved before the first launch
(pod creation, uploads, prepare), so a runner that dies there leaves it set (the dead-man's evidence), and an exception there is
recorded as the run's terminal event by the finaliser, which clears the mark. A blocked start never sets it."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import runner_multi as rm  # noqa: E402


class FakeT:
    whole_files = False
    persistent = False

    def over_budget(self, st):
        return None

    def stop(self, st):
        pass

    def release(self, st):
        pass


def run_main(tmp_path, monkeypatch, launch, extra=()):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"home_markers": True, "jobs": [{"name": "gate", "cmd": ["x"], "outputs": ["gate.json"]}]}))
    state = tmp_path / "state"
    monkeypatch.setattr(rm, "make_transport", lambda a: FakeT())
    monkeypatch.setattr(rm, "listed", lambda: None)
    monkeypatch.setattr(rm, "launch", launch)
    monkeypatch.setattr(sys, "argv", ["runner_multi.py", "--commit", "abc", "--jobs", str(jobs), "--name", "t", "--art",
                                      str(tmp_path / "art"), "--state-dir", str(state), "--backend", "colab", *extra])
    return state


def test_the_first_launch_runs_inside_the_active_mark_and_its_exception_is_recorded(tmp_path, monkeypatch):
    seen = {}

    def launch(st, commit, jobs_rel, jobs, uploads):
        seen["disk"] = json.loads((tmp_path / "state" / "state.json").read_text())   # what a death here would leave
        raise RuntimeError("pod creation failed")
    state = run_main(tmp_path, monkeypatch, launch)
    with pytest.raises(RuntimeError):
        rm.main()
    assert "active" in seen["disk"]   # saved before the launch: a death during create, upload or prepare leaves it set
    d = json.loads((state / "state.json").read_text())
    assert d["terminals"][-1]["reason"] == "exception: RuntimeError"   # recorded by the finaliser
    assert "active" not in d   # cleared with the recorded end


def test_a_blocked_start_never_sets_the_mark_or_launches(tmp_path, monkeypatch):
    called = []
    state = run_main(tmp_path, monkeypatch, lambda *a: called.append(1), extra=("--deadline-utc", "2020-01-01T00:00:00Z"))
    rm.main()
    d = json.loads((state / "state.json").read_text())
    assert not called and "active" not in d and d["terminals"][-1]["reason"] == "blocked"
