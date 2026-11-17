"""A32 before C1 (GPT round 10's N2 and the re-run's P7(i); the lead's rulings P3 and P7 of 15 Nov 2026).

N2: an arm trained from a start made in its own campaign waits for that SFT1500's watch to end healthy, so an unhealthy start trains
no arm; the runner relays the watch's readouts while it runs. C0's jobs file, already run, keeps its rendering.
P7(i): the runner saves its active mark inside the lifecycle try, so a failure at the save itself meets the finaliser.
Claude Fable 5.2, 17 Nov 2026. CPU only, a few seconds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import b2cue_jobs as J  # noqa: E402
import runner_multi as rm  # noqa: E402


def start_of(j):
    a = [j["after"]] if isinstance(j.get("after"), str) else (j.get("after") or [])
    got = [x for x in a if x.startswith("sft_")]
    return got[0] if got else None


def trained(jobs):
    return [j for j in jobs if "spec" in j and start_of(j)]


@pytest.mark.parametrize("v", [v for v in J.variants() if v[0] != "c0"], ids=lambda v: os.path.basename(J.out_name(*v)))
def test_every_arm_from_a_campaign_made_start_waits_on_its_watch(v):
    jobs = J.render(*v)["jobs"]
    names = [j["name"] for j in jobs]
    for j in trained(jobs):
        w = f"watch_{start_of(j)}"
        assert w in names and w in j["after"] and not j.get("after_terminal"), j["name"]
        assert j["spec"]["train"]["init_from"].endswith(f"/{start_of(j)}/{start_of(j)}.safetensors"), j["name"]
    for j in jobs:   # an arm from an uploaded start has no watch to wait on
        if "spec" in j and not start_of(j) and not j["name"].startswith(("sft_", "repro_")):
            assert "/upload/" in j["spec"]["train"]["init_from"], j["name"]
            assert not any(a.startswith("watch_") for a in j["after"]), j["name"]


def test_c1a_c1b_and_c2_have_such_arms():
    for c, base in (("c1a", "S"), ("c1b", "N60"), ("c2", "N2"), ("c2", "S2")):
        jobs = J.render(c, 0.3)["jobs"]
        got = [j["name"] for j in trained(jobs) if start_of(j) == f"sft_{base}"]
        assert f"P_{base}_1" in got and f"N_{base}_125" in got, (c, base)
    assert any(j["name"] == "N_N2_213" and "barrier" in j["after"] and "watch_sft_N2" in j["after"] for j in J.render("c2", 0.3)["jobs"])


def test_c0s_jobs_file_keeps_the_rendering_it_ran_with():
    committed = subprocess.run(["git", "show", "f216598c:tools/colab_chat/b2cue_c0.json"], cwd=ROOT, capture_output=True, text=True)
    if committed.returncode != 0:
        pytest.skip("the launch commit of C0 is not in this clone")
    assert json.dumps(J.render("c0"), indent=1) + "\n" == committed.stdout


def run_vm(tmp, watch_rc):
    """The real jobs_vm.py over stand-in commands that keep a rendered arm's own prerequisites."""
    arm = next(j for j in J.render("c1a", 0.3)["jobs"] if j["name"] == "P_S_1")
    C = tmp / "chat"
    C.mkdir()
    jobs = {"jobs": [{"name": "gate", "cmd": ["true"]}, {"name": "sft_S", "cmd": ["true"], "after": "gate"},
                     {"name": "watch_sft_S", "light": True, "after": ["gate"], "after_terminal": True,
                      "cmd": ["sh", "-c", f"while [ \"$(cat {C}/STATUS_sft_S 2>/dev/null)\" != FINISHED ]; do sleep 0.1; done; exit {watch_rc}"]},
                     {"name": "P_S_1", "cmd": ["sh", "-c", "echo ran > {out}/ran"], "after": arm["after"]},
                     {"name": "score_P_S_1", "light": True, "cmd": ["true"], "after": "P_S_1"}],
            "max_parallel": 2, "max_parallel_light": 2}
    (tmp / "jobs.json").write_text(json.dumps(jobs))
    env = {**os.environ, "JOBS_CHAT_DIR": str(C), "JOBS_REPO": str(ROOT), "JOBS_TICK": "0.1", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "jobs_vm.py"), str(tmp / "jobs.json")], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return C, {j["name"]: (C / f"STATUS_{j['name']}").read_text().strip() for j in jobs["jobs"]}


def test_an_unhealthy_watch_trains_no_arm(tmp_path):
    C, st = run_vm(tmp_path, 1)
    assert st["sft_S"] == "FINISHED" and st["watch_sft_S"] == "FAILED"
    assert st["P_S_1"] == "FAILED" and st["score_P_S_1"] == "FAILED" and not (C / "P_S_1" / "ran").exists()


def test_a_healthy_watch_lets_the_arm_run(tmp_path):
    C, st = run_vm(tmp_path, 0)
    assert set(st.values()) == {"FINISHED"} and (C / "P_S_1" / "ran").exists()


# ---------------------------------------------------------------- the runner's live WATCH relay

class St:
    def __init__(self):
        self.d = {"jobs": {}}
        self.saves = 0

    def job(self, n):
        return self.d["jobs"].setdefault(n, {"relayed_step": 0, "status": None, "home": False})

    def save(self):
        self.saves += 1


def test_watch_lines_come_once_each_in_step_order_then_the_health():
    js = {}
    w = {"job": "sft_S", "health": "running", "at": {"188": {"sft_loss": 3.1, "val_loss": 3.2}}}
    assert rm.watch_lines(js, w) == ["WATCH sft_S step 188: sft 3.1 val 3.2"]
    assert rm.watch_lines(js, w) == []
    w["at"].update({"750": {"sft_loss": 2.5, "val_loss": 2.6}, "375": {"sft_loss": 2.8, "val_loss": 2.9}})
    assert [x.split()[3] for x in rm.watch_lines(js, w)] == ["375:", "750:"]
    w.update({"health": "stop", "health_stop": "non-finite SFT loss at steps [800]"})
    assert rm.watch_lines(js, w) == ["WATCH sft_S health stop: non-finite SFT loss at steps [800]"]
    assert rm.watch_lines(js, w) == [] and rm.watch_lines({}, None) == []
    assert rm.watch_lines({}, {"job": "sft_S", "health": "ok", "at": {}}) == ["WATCH sft_S health ok"]


def test_a_running_watch_is_relayed_each_tick_and_a_miss_is_silent(tmp_path, monkeypatch):
    st, logged, vm = St(), [], {}

    def download(src, dst, timeout=0):
        if "body" not in vm:
            return 1
        Path(dst).write_text(vm["body"])
        return 0
    monkeypatch.setattr(rm, "ART", str(tmp_path))
    monkeypatch.setattr(rm, "download_file", download)
    monkeypatch.setattr(rm, "log", logged.append)
    rm.relay_watch(st, "watch_sft_S")
    assert logged == [] and st.saves == 0
    vm["body"] = json.dumps({"job": "sft_S", "health": "running", "at": {"188": {"sft_loss": 3.0, "val_loss": 3.1}}})
    rm.relay_watch(st, "watch_sft_S")
    rm.relay_watch(st, "watch_sft_S")
    assert logged == ["WATCH sft_S step 188: sft 3.0 val 3.1"] and st.saves == 1
    vm["body"] = "{half a file"
    rm.relay_watch(st, "watch_sft_S")
    assert len(logged) == 1
    assert not (tmp_path / "watch_sft_S" / "watch.json").exists()   # the home copy lands only through fetch_outputs


def test_the_final_download_shows_whatever_the_relay_missed(tmp_path, monkeypatch):
    st, logged = St(), []
    final = {"job": "sft_S", "health": "ok", "at": {s: {"sft_loss": 1.0, "val_loss": 1.0} for s in ("188", "375", "750")}}

    def download(src, dst, timeout=0):
        Path(dst).write_text(json.dumps(final) if src.endswith("watch.json") else "log")
        return 0

    class T:
        whole_files = True
    monkeypatch.setattr(rm, "ART", str(tmp_path))
    monkeypatch.setattr(rm, "T", T)
    monkeypatch.setattr(rm, "download_file", download)
    monkeypatch.setattr(rm, "log", logged.append)
    st.job("watch_sft_S")["watch_relayed"] = ["188"]
    rm.fetch_outputs(st, "watch_sft_S", ["watch.json"])
    assert [x for x in logged if x.startswith("WATCH")] == ["WATCH sft_S step 375: sft 1.0 val 1.0", "WATCH sft_S step 750: sft 1.0 val 1.0",
                                                           "WATCH sft_S health ok"]
    assert st.job("watch_sft_S")["home"] is True


# ---------------------------------------------------------------- P7(i)

class FakeT:
    whole_files = False
    persistent = False

    def over_budget(self, st):
        return None

    def stop(self, st):
        pass

    def release(self, st):
        pass


def test_a_failure_saving_the_active_mark_meets_the_finaliser(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"home_markers": True, "jobs": [{"name": "gate", "cmd": ["x"], "outputs": ["gate.json"]}]}))
    state = tmp_path / "state"
    launched = []
    monkeypatch.setattr(rm, "make_transport", lambda a: FakeT())
    monkeypatch.setattr(rm, "listed", lambda: None)
    monkeypatch.setattr(rm, "launch", lambda *a: launched.append(1))
    real = rm.State.save
    hit = []

    def save(self):
        if "active" in self.d and not hit:   # the mark's own save fails once
            hit.append(1)
            raise OSError("disk full")
        real(self)
    monkeypatch.setattr(rm.State, "save", save)
    monkeypatch.setattr(sys, "argv", ["runner_multi.py", "--commit", "abc", "--jobs", str(jobs), "--name", "t", "--art",
                                      str(tmp_path / "art"), "--state-dir", str(state), "--backend", "colab"])
    with pytest.raises(OSError):
        rm.main()
    d = json.loads((state / "state.json").read_text())
    assert hit and not launched
    assert d["terminals"][-1]["reason"] == "exception: OSError" and "active" not in d   # recorded and cleared, never left dangling


# ---------------------------------------------------------------- a replacement pod never reruns a home watch

class Fresh:
    whole_files = False
    persistent = False   # a fresh volume: nothing of the lost pod is on it


def replacement(tmp_path, monkeypatch, watch_status, sft_home=True):
    art = tmp_path / "art"
    art.mkdir()
    monkeypatch.setattr(rm, "ART", str(art))
    monkeypatch.setattr(rm, "HOME", str(tmp_path))
    monkeypatch.setattr(rm, "T", Fresh())
    monkeypatch.setattr(rm, "HOME_MARKERS", True)
    arm = next(j for j in J.render("c1a", 0.3)["jobs"] if j["name"] == "P_S_1")
    js = [{"name": "gate", "cmd": ["x"], "outputs": ["gate.json"]}, {"name": "sft_S", "spec": {}, "after": "gate"},
          {"name": "watch_sft_S", "light": True, "after": ["gate"], "after_terminal": True,
           "cmd": ["python3", "tools/colab_chat/b2cue_early.py", "--watch", "--job", "sft_S"], "outputs": ["watch.json"]},
          {"name": "P_S_1", "spec": {}, "after": arm["after"]}]
    st = rm.State()
    for n, status in (("gate", "FINISHED"), ("sft_S", "FINISHED"), ("watch_sft_S", watch_status)):
        st.job(n).update(home=(n != "sft_S" or sft_home), status=status if (n != "sft_S" or sft_home) else "RUNNING", home_marked=True)
        (art / n).mkdir()
    (art / "sft_S" / "sft_S.safetensors").write_text("w")
    st.d["booted"] = False
    st.save()
    sent = []

    def fake_exec(code, timeout=200):
        sent.append(code)
        return 0, "DIRS_OK HOME_OK PENDING_OK INPUTS_OK BOOTED"
    logged = []
    monkeypatch.setattr(rm, "exec_py", fake_exec)
    monkeypatch.setattr(rm, "upload_file", lambda src, dst: None)
    monkeypatch.setattr(rm, "log", logged.append)
    assert rm.prepare(st, "c0ffee", "jobs.json", js, [])
    return next(c for c in sent if "HOME_OK" in c), logged


def marked(code, status):
    """The job names prepare() writes STATUS_<job> = status for (its two loops are `for n in [...]: ... write('STATUS\\n')`)."""
    line = next(x for x in code.split("\n") if f"write('{status}" in x)
    return set(eval(line[line.index("["):line.index("]") + 1]))


def test_a_healthy_home_watch_is_marked_finished_on_a_replacement_pod_not_run_again(tmp_path, monkeypatch):
    code, logged = replacement(tmp_path, monkeypatch, "FINISHED")
    assert "watch_sft_S" in marked(code, "FINISHED") and "sft_S" in marked(code, "FINISHED")
    assert "gate" not in marked(code, "FINISHED")   # the gate, a data job the arm needs, still runs again, as before
    assert not any("rerun" in x and "watch_sft_S" in x for x in logged)


def test_an_unhealthy_home_watch_stays_failed_so_no_arm_trains(tmp_path, monkeypatch):
    code, _ = replacement(tmp_path, monkeypatch, "FAILED")
    assert "watch_sft_S" in marked(code, "FAILED") and "watch_sft_S" not in marked(code, "FINISHED")


def test_a_home_watch_whose_sft1500_is_not_home_runs_again_with_it(tmp_path, monkeypatch):
    """Explore review of 17 Nov 2026, point 4: the SFT1500's final download failed, its watch landed, the pod and volume were lost.
    The SFT1500 trains again; its watch must run again too, or the arms would start from a start nobody watched."""
    code, logged = replacement(tmp_path, monkeypatch, "FINISHED", sft_home=False)
    assert "watch_sft_S" not in marked(code, "FINISHED") and "sft_S" not in marked(code, "FINISHED")
    assert "watch_sft_S" not in code[code.index("HOME_"):] if "HOME_" in code else True   # no home marker: the barrier waits for it
    js = rm.State().job("watch_sft_S")
    assert js["home"] is False and js["fetched"] == [] and js["status"] is None   # its new watch.json will be fetched
    assert any("the watch runs again" in x for x in logged)


def test_watched_home_reads_the_watchs_own_job():
    class S:
        def __init__(self, home):
            self.h = home

        def job(self, n):
            return {"home": self.h.get(n, False)}
    by = {"sft_S": {}, "watch_sft_S": {}}
    w = {"cmd": ["python3", "b2cue_early.py", "--watch", "--job", "sft_S", "--out", "{out}"]}
    assert rm.watched_home(S({"sft_S": True}), w, by) is True and rm.watched_home(S({}), w, by) is False
    assert rm.watched_home(S({"sft_X": True}), {"cmd": ["x", "--job", "sft_X"]}, by) is False     # not a job of this campaign
    assert rm.watched_home(S({"sft_S": True}), {"cmd": ["x"]}, by) is False                       # names no job: run it again
