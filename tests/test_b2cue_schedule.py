"""A32's (b2cue) jobs files run by the real jobs_vm.py on a stand-in trainer (tests/fake_mtstudio_b2cue.py), with the real gate,
early readouts, SFT1500 watch, selection, barrier and score binding; then the real runner_multi.py over the finished C0 pod directory.
Claude Opus 5.5, 15 Nov 2026, after A31's tests/test_a31_schedule.py. CPU only, a few minutes.

C0: one sweep run fails at step 200; the test checks that every job ends, the failed run is ineligible, the run over 1.02 x D0's
held-out loss is ineligible, the selection names the registered lambda, the watch records the SFT1500's readouts at 188, 375 and 750,
and the runner brings every job home and writes the home markers.
C1a at lambda 0.3: one N arm trips the 25% health stop and one P0 run's loss turns non-finite; a stand-in for the runner marks jobs home
as they end; the barrier lifts only then and the extra seeds run after it; every score file is bound to its weights and request file.

Needs the lane's readout.py and b2cue_measures.py (uploaded to the pod); skipped when the lane's tree is not mounted. answer_tokens.py
is replaced by a stub (every token answer-deciding), since the stand-in's ids are not a real vocabulary's."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import b2cue_jobs as J  # noqa: E402

LANE = Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/cue_selective")
pytestmark = pytest.mark.skipif(not (LANE / "a31" / "readout.py").exists(), reason="the lane's tree is not mounted")
EOT = "<|endoftext|>"
FAMS = ("verify", "reverse_lookup", "running_total")
STUB_AT = '''
class Vocab:
    def __init__(self, *a):
        pass
def masks(vocab, ida, idb, ta, tb, ka, kb, stats=None):
    return [1] * len(ida), [1] * len(idb)
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trips(prefix, n, device="new_person", op="qualifier"):
    out = []
    for i in range(n):
        out.append({"id": f"{prefix}{i}", "family": FAMS[i % 3], "heldout_family": True, "device": device, "nuisance": {"op": op},
                    "keys": {"A": "p:x", "B": "p:y", "N": "p:x"},
                    "A": [f"ctx A {prefix}{i}", "reply a", EOT], "B": [f"ctx B {prefix}{i}", "reply b", EOT],
                    "N": [f"ctx N {prefix}{i}", "reply a", EOT]})
    return out


def write_req(up, base, sets):
    """sets: {name: triplets}; writes b2cue_<name>.jsonl, the request file (five lines a triplet) and its manifest."""
    info, n = {}, 0
    for name, ts in sets.items():
        (up / f"b2cue_{name}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ts))
        info[name] = {"triplets": len(ts), "first_request": n}
        n += 5 * len(ts)
    (up / f"{base}.jsonl").write_text("".join(json.dumps({"r": i}) + "\n" for i in range(n)))
    (up / f"{base}.manifest.json").write_text(json.dumps({"requests": n, "sets": info}))


def setup(tmp, campaign, lam=None):
    C = tmp / "chat"
    up = C / "upload"
    up.mkdir(parents=True)
    shutil.copy(LANE / "a31" / "readout.py", up / "b2cue_readout.py")
    shutil.copy(LANE / "b2cue" / "b2cue_measures.py", up / "b2cue_measures.py")
    (up / "answer_tokens.py").write_text(STUB_AT)
    for f in ("vocab.gguf", "llama3_bpe.py", "S74.jsonl", "P0.jsonl", "train_twins.jsonl", "gate_twins.jsonl", "sft_human.jsonl"):
        (up / f).write_text("stand-in")
    for b in ("N250", "S", "N60", "N2", "S2"):
        (up / f"base_{b}.safetensors").write_text(f"base {b}")
    write_req(up, "b2cue_selection_req", {"S_hf_train": trips("a", 9), "S_hf_held": trips("b", 6, "speaker", "qualifier") + trips("c", 3, "speaker", "carrier"),
                                          "S_hf_held_nq": trips("d", 3, "speaker", "carrier"), "S_in": trips("e", 3)})
    write_req(up, "b2cue_early_req", {"early_in": trips("f", 3), "early_hf": trips("g", 3)})
    write_req(up, "b2cue_confirmation_req", {"C_hf_train": trips("h", 6)})
    fake = tmp / "mtstudio"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fake_mtstudio_b2cue.py'} \"$@\"\n")
    dev_test = tmp / "test_twin_device"
    dev_test.write_text("#!/bin/sh\necho 'twin loss: device vs host (device op set ON)'\n")
    for p in (fake, dev_test):
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    sub = lambda text: text.replace(J.BIN, str(fake)).replace(J.DEVICE_TEST, str(dev_test)).replace("/content/chat", str(C))  # noqa: E731
    jobs = json.loads(sub(json.dumps(J.render(campaign, lam))))
    (tmp / "jobs.json").write_text(json.dumps(jobs))
    (up / "start_N250.safetensors").write_text("start N250")
    for name, spec in J.staged_specs(campaign).items():   # the launch's staged specs: the gate's, the uploaded start's
        (up / name).write_text(sub(json.dumps(spec)))
    return C, jobs, fake


def run_jobs(tmp, C, jobs, fake, extra_env, mark_home=False):
    env = {**os.environ, "JOBS_CHAT_DIR": str(C), "JOBS_MTSTUDIO": str(fake), "JOBS_REPO": str(ROOT), "JOBS_TICK": "0.2",
           "FAKE_CKPT_PAUSE": "0.3", "B2CUE_POLL_S": "0.5", "CUDA_VISIBLE_DEVICES": "", **extra_env}
    stop = threading.Event()

    def marker():   # a stand-in for runner_multi.py's home markers: a job counts home once it has ended
        while not stop.is_set():
            for j in jobs["jobs"]:
                st = C / f"STATUS_{j['name']}"
                if st.exists() and st.read_text().strip() in ("FINISHED", "FAILED") and not (C / f"HOME_{j['name']}").exists():
                    (C / f"HOME_{j['name']}").write_text("1")
            time.sleep(0.5)
    th = threading.Thread(target=marker, daemon=True)
    if mark_home:
        th.start()
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "jobs_vm.py"), str(tmp / "jobs.json")], env=env,
                       capture_output=True, text=True, timeout=1800)
    stop.set()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return {j["name"]: (C / f"STATUS_{j['name']}").read_text().strip() for j in jobs["jobs"]}, r.stdout


@pytest.fixture(scope="module")
def c0(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b2cue_c0")
    C, jobs, fake = setup(tmp, "c0")
    status, log = run_jobs(tmp, C, jobs, fake, {"FAKE_FAIL": "Dn_0p3"})
    return {"tmp": tmp, "C": C, "jobs": jobs, "status": status, "log": log}


def test_c0_every_job_ends_and_the_gate_passes(c0):
    assert set(c0["status"].values()) <= {"FINISHED", "FAILED"}, c0["status"]
    g = json.loads((c0["C"] / "gate" / "gate.json").read_text())
    assert g["pass"], g["fails"]
    assert set(g["shapes"]) == {"b2cue_gate_D1"} and g["shapes"]["b2cue_gate_D1"]["runs"]["cuda_am"]


def test_c0_selection_applies_the_registered_rule(c0):
    st, C = c0["status"], c0["C"]
    assert st["Dn_0p3"] == "FAILED" and st["select"] == "FINISHED"
    d = json.loads((C / "select" / "decisions.json").read_text())
    assert d["outcome"] == "selected" and d["chosen"] == 0.1, d
    assert not d["runs"]["0.3"]["complete"] and "status FAILED" in d["runs"]["0.3"]["why"]
    assert d["runs"]["1.0"]["complete"] and not d["runs"]["1.0"]["eligible"]   # 4.1 > 1.02 x 4.0
    assert d["D0"]["n_pool"] == 9 + 3 + 3   # the speaker qualifier triplets are out of the pool


def test_c0_watch_relays_the_sft1500_readouts(c0):
    w = json.loads((c0["C"] / "watch_sft_N250" / "watch.json").read_text())
    assert set(w["at"]) == {"188", "375", "750"} and w["complete"] and not w.get("health_stop")


def test_c0_sweep_scores_read_the_selection_file_only(c0):
    C = c0["C"]
    for n in ("D0", "Dn_0p1", "Dn_1p0"):
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["requests_sha256"] == sha(C / "upload" / "b2cue_selection_req.jsonl") and "gguf_sha256" not in b
    e = json.loads((C / "early_Dn_0p1_256" / "early.json").read_text())
    assert e["scored"] and e["early"]["early_hf"]["n"] == 3 and not e.get("health_stop")


class LocalVM:
    persistent, whole_files = True, True

    def __init__(self, root):
        self.root, self.released = str(root), 0

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
        pass

    def release(self, st):
        self.released += 1


def test_c0_runner_brings_every_job_home_and_marks_it(c0):
    import importlib
    import runner_multi as RM
    importlib.reload(RM)
    tmp = c0["tmp"]
    # the pod's /content is tmp: the jobs file names tmp/chat for /content/chat
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
    st = json.loads((state / "state.json").read_text())
    assert st.get("campaign_home") is True and all(j.get("home") for j in st["jobs"].values())
    assert all(j.get("home_marked") for j in st["jobs"].values())
    assert (art / "select" / "decisions.json").exists() and (art / "sft_N250" / "sft_N250.safetensors").exists()


@pytest.fixture(scope="module")
def c1a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b2cue_c1a")
    C, jobs, fake = setup(tmp, "c1a", 0.3)
    status, log = run_jobs(tmp, C, jobs, fake, {"FAKE_HIGH_VAL": "N_N250_1", "FAKE_SLOW": "N_N250_1", "FAKE_NAN": "P0_N250_50"},
                           mark_home=True)
    return {"tmp": tmp, "C": C, "jobs": jobs, "status": status, "log": log}


def test_c1a_every_job_ends(c1a):
    assert set(c1a["status"].values()) <= {"FINISHED", "FAILED"}, c1a["status"]


def test_c1a_health_stops(c1a):
    C, st = c1a["C"], c1a["status"]
    e = json.loads((C / "early_N_N250_1_256" / "early.json").read_text())
    assert "more than 10% above the reference" in e["health_stop"] and e["kill"] == "SIGTERM sent"
    assert st["N_N250_1"] == "FAILED" and not e["scored"]
    p0 = json.loads((C / "early_P0_N250_50_128" / "early.json").read_text())
    assert "non-finite" in p0["health_stop"]


def test_c1a_barrier_lifts_after_every_other_job_is_home(c1a):
    C = c1a["C"]
    b = json.loads((C / "barrier" / "barrier.json").read_text())
    others = set(b["jobs"])
    assert "P_N250_175" not in others and "score_P_N250_1" in others and "sft_S" in others
    lifted = (C / "barrier" / "barrier.json").stat().st_mtime
    for n in others:
        assert (C / f"HOME_{n}").exists()
    for n in ("P_N250_175", "N_N250_213"):
        first = min(p.stat().st_mtime for p in (C / n).iterdir())
        assert first >= lifted - 1, n
        assert c1a["status"][n] == "FINISHED"


def test_c1a_scores_are_bound(c1a):
    C = c1a["C"]
    conf = sha(C / "upload" / "b2cue_confirmation_req.jsonl")
    for n in ("P_N250_1", "NA_N250_50", "M_N250_125", "N_S_1", "N_N250_213"):
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["weights_sha256"] == sha(C / n / f"{n}.safetensors") and b["requests_sha256"] == conf
    s = json.loads((C / "score_start_N250" / "score.json").read_text())
    assert s["weights_sha256"] == sha(C / "upload" / "start_N250.safetensors")
    spk = json.loads((C / "score_spk_P_N250_1" / "score.json").read_text())
    assert spk["requests_sha256"] == sha(C / "upload" / "b2cue_selection_req.jsonl")
    na = json.loads((C / "NA_N250_1_spec.json").read_text())
    assert na["train"]["twin_loss"]["answer_mask"] is True and na["train"]["twin_loss"]["lambda"] == 0.3 == na["train"]["twin_loss"]["mu"]
