"""A31's whole jobs file run by the real jobs_vm.py on a stand-in trainer (tests/fake_mtstudio_a31.py), with the real gate, early
readouts, selection and score binding, then the real runner_multi.py over the finished pod directory (round-2 and round-3
reviews, 13 Nov 2026). Claude Opus 5.5,
13 Nov 2026. CPU only, about two minutes. One sweep run fails at step 200 and one arm trips the 25% health stop; the test checks
that every job ends, the failed sweep run is ineligible and nothing else, the selection promotes exactly the registered candidates
and the rest end as not promoted, the early readouts of a failed run's lost snapshots end FAILED, the health stop is recorded,
no sweep job reads the final requests, and every final score file carries its weights' hash.

Needs the lane's readout.py (uploaded to the pod as a31_readout.py); skipped when the lane's tree is not mounted."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import a31_jobs as J  # noqa: E402

LANE = Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/cue_selective/a31")
pytestmark = pytest.mark.skipif(not (LANE / "readout.py").exists(), reason="the lane's readout.py is not mounted")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_set(up, name, n, prefix):
    ts = [{"id": f"{prefix}{i}", "family": "f", "heldout_family": True} for i in range(n)]
    (up / f"a31_{name}_triplets.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ts))
    return ts


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a31sched")
    C = tmp / "chat"
    up = C / "upload"
    up.mkdir(parents=True)
    shutil.copy(LANE / "readout.py", up / "a31_readout.py")
    dev, hd = write_set(up, "dev", 8, "d"), write_set(up, "heldout_dev", 4, "h")
    (up / "a31_early_req.jsonl").write_text("".join(json.dumps({"r": i}) + "\n" for i in range(5 * 12)))
    (up / "a31_early_req.manifest.json").write_text(json.dumps(
        {"requests": 60, "sets": {"dev": {"triplets": 8, "first_request": 0}, "heldout_dev": {"triplets": 4, "first_request": 40}}}))
    (up / "a31_readout_req.jsonl").write_text("".join(json.dumps({"r": i}) + "\n" for i in range(25)))
    (up / "s71.safetensors").write_text("s71 weights")
    fake = tmp / "mtstudio"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fake_mtstudio_a31.py'} \"$@\"\n")
    dev_test = tmp / "test_twin_device"
    dev_test.write_text("#!/bin/sh\necho 'twin loss: device vs host (device op set ON)'\n")
    for p in (fake, dev_test):
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    text = json.dumps(J.render())
    text = text.replace(J.BIN, str(fake)).replace(J.DEVICE_TEST, str(dev_test)).replace("/content/chat", str(C))
    jobs = json.loads(text)
    (tmp / "jobs.json").write_text(json.dumps(jobs))
    gate = json.loads(json.dumps(J.gate_spec()).replace("/content/chat", str(C)))
    (up / "a31_gate_spec.json").write_text(json.dumps(gate))
    (up / "s71_spec.json").write_text(json.dumps(json.loads(json.dumps(J.s71_spec()).replace("/content/chat", str(C)))))
    env = {**os.environ, "JOBS_CHAT_DIR": str(C), "JOBS_MTSTUDIO": str(fake), "JOBS_REPO": str(ROOT), "JOBS_TICK": "0.2",
           "FAKE_FAIL": "Dm_0p3", "FAKE_HIGH_VAL": "N_1_0p1", "FAKE_SLOW": "N_1_0p1", "FAKE_CKPT_PAUSE": "0.5",
           "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "jobs_vm.py"), str(tmp / "jobs.json")], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    status = {j["name"]: (C / f"STATUS_{j['name']}").read_text().strip() for j in jobs["jobs"]}
    return {"tmp": tmp, "C": C, "jobs": jobs, "status": status, "log": r.stdout}


class LocalVM:
    """runner_multi.py's transport over a local directory standing for the pod's /content (a RunPod volume): exec runs the
    generated code there, downloads copy whole files. Records what the runner asks of it."""
    persistent, whole_files = True, True

    def __init__(self, root):
        self.root, self.released, self.stopped = str(root), 0, 0

    def path(self, p):
        return p.replace("/content/", self.root + "/")

    def exec(self, code, timeout=200):
        p = subprocess.run([sys.executable, "-c", code.replace("/content/", self.root + "/")], capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def download_whole(self, src, dst, timeout=0):
        if not os.path.exists(self.path(src)):
            return 1
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(self.path(src), dst)
        return 0

    def get(self, src, dst, timeout=0):
        return self.download_whole(src, dst, timeout)

    def over_budget(self, st):
        return None

    def refresh(self, st):
        pass

    def attach(self, st):
        pass

    def gone(self, st):
        return False

    def stop(self, st):
        self.stopped += 1

    def release(self, st):
        self.released += 1


@pytest.fixture(scope="module")
def home(campaign):
    """The real runner_multi.py main loop over the finished pod directory: every job's files brought home."""
    import importlib
    import runner_multi as RM
    importlib.reload(RM)
    tmp = campaign["tmp"]
    state, art = tmp / "runner_state", tmp / "art"
    state.mkdir()
    (state / "state.json").write_text(json.dumps({"endpoint": "local", "launches": 1, "booted": True, "jobs": {}}))
    T = LocalVM(tmp)
    mp = pytest.MonkeyPatch()
    mp.setattr(RM, "make_transport", lambda a: T)
    mp.setattr(RM, "listed", lambda: "local")
    mp.setattr(RM, "TICK", 0.01)
    mp.setattr(sys, "argv", ["runner_multi.py", "--commit", "x" * 40, "--jobs", str(tmp / "jobs.json"), "--name", "t", "--art", str(art),
                             "--state-dir", str(state), "--backend", "runpod", "--deadline-utc", "2099-01-01T00:00:00Z"])
    RM.main()
    mp.undo()
    return {"art": art, "state": json.loads((state / "state.json").read_text()), "T": T}


def test_every_job_ends(campaign):
    assert set(campaign["status"].values()) <= {"FINISHED", "FAILED"}, campaign["status"]
    assert campaign["status"]["gate"] == "FINISHED"
    assert json.loads((campaign["C"] / "gate" / "gate.json").read_text())["pass"]


def test_a_failed_sweep_run_is_ineligible_and_the_selection_runs(campaign):
    st, C = campaign["status"], campaign["C"]
    assert st["Dm_0p3"] == "FAILED" and st["select"] == "FINISHED"
    dec = json.loads((C / "select" / "decisions.json").read_text())
    m = dec["arms"]["M"]["table"]
    assert not m["0.3"]["complete"] and "status FAILED" in m["0.3"]["why"]
    assert not m["1.0"]["eligible"] and m["1.0"]["complete"]          # held-out loss 4.1 > 1.02 x 4.0
    assert dec["arms"]["M"]["chosen"] == 0.1 and dec["arms"]["N"]["chosen"] == 0.1   # ties go to the smaller lambda
    # the failed run's snapshots after step 200 never exist: those early readouts end FAILED (rung lost), the 128 one ran
    assert st["early_Dm_0p3_128"] == "FINISHED" and st["early_Dm_0p3_256"] == "FAILED" and st["early_Dm_0p3_512"] == "FAILED"


def test_promotion_runs_exactly_the_selected_arms(campaign):
    st = campaign["status"]
    for sd in J.SEEDS:
        assert st[f"P_{sd}"] == "FINISHED" and st[f"score_P_{sd}"] == "FINISHED"
        for x in J.GRID:
            for arm in ("M", "N"):
                n = f"{arm}_{sd}_{J.lname(x)}"
                if n == "N_1_0p1":
                    continue                         # the health-stopped run: test_the_health_stop_kills_the_run
                want = "FINISHED" if x == 0.1 else "FAILED"
                assert st[n] == want and st[f"score_{n}"] == want, n
    assert "not promoted" in campaign["log"]


def test_the_health_stop_kills_the_run(campaign):
    """End to end: the early job finds the live trainer by its `run SPEC` command line, records the stop, then signals it; the
    training job ends FAILED and its final scoring never runs."""
    C, st = campaign["C"], campaign["status"]
    e = json.loads((C / "early_N_1_0p1_256" / "early.json").read_text())
    assert "more than 10% above the reference" in e["health_stop"] and e["kill"] == "SIGTERM sent" and e["stopped_pid"]
    assert st["N_1_0p1"] == "FAILED" and st["score_N_1_0p1"] == "FAILED"
    assert not (C / "N_1_0p1" / "result.json").exists()
    ok = json.loads((C / "early_N_50_0p1_256" / "early.json").read_text())
    assert not ok.get("health_stop") and not ok.get("health_unread") and ok["early"]["dev"]["n"] == 8


def test_scores_are_bound_after_scoring_and_the_sweep_never_reads_the_final_requests(campaign):
    C = campaign["C"]
    final_req = sha(C / "upload" / "a31_readout_req.jsonl")
    for n in ("P_1", "N_50_0p1", "M_1_0p1"):
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["weights_sha256"] == sha(C / n / f"{n}.safetensors") and b["requests_sha256"] == final_req
        assert b["scores_sha256"] == sha(C / f"score_{n}" / "scores.jsonl") and b["gguf_sha256"] == sha(C / n / f"{n}.gguf")
    for n in ("D0", "Dm_0p1", "Dn_1p0"):
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["requests_sha256"] == sha(C / "upload" / "a31_early_req.jsonl") and "gguf_sha256" not in b
    s71 = json.loads((C / "score_S71" / "score.json").read_text())
    assert s71["weights_sha256"] == sha(C / "upload" / "s71.safetensors") and s71["requests_sha256"] == final_req


def test_the_runner_brings_every_job_home_and_releases_the_storage(campaign, home):
    st, art = home["state"], home["art"]
    assert st.get("campaign_home") is True
    assert all(j.get("home") for j in st["jobs"].values()) and set(st["jobs"]) == {j["name"] for j in campaign["jobs"]["jobs"]}
    assert (art / "D0" / "D0.safetensors").exists() and not (art / "D0" / "D0.gguf").exists()   # a safetensors-only job came home
    assert (art / "P_1" / "P_1.gguf").exists() and (art / "P_1" / "events.jsonl").exists()
    assert (art / "score_P_1" / "score.json").exists() and (art / "early_P_1_256" / "early.json").exists()
    assert (art / "select" / "decisions.json").exists() and (art / "gate" / "gate.json").exists()
    assert home["T"].released == 1
