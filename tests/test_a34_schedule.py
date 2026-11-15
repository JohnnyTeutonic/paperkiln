"""A34's whole jobs file run by the real jobs_vm.py on a stand-in trainer (tests/fake_mtstudio_a34.py), with the real gate, reference
builds, early readouts, selection and score binding, then the real runner_multi.py over the finished pod directory. Claude Opus 5.5,
15 Nov 2026 (A31's test_a31_schedule.py, adapted). CPU only, a few minutes. The stand-in makes Fd_0p3 fail the hold and Fd_1p0 and
Fd_3p0 pass both conditions, so the rule must choose 1.0; P_50r fails at step 200, so its floor runs (F_50, NF_50) must never start
and its reference must not be built. Checks: every job ends; the gate passes on the floor spec; each reference is built from its own
run's snapshots and the start, with nine knots; the selection chooses the registered floor; exactly F_<s>_1p0 and NF_<s>_1p0 run,
at the seed whose reference exists; each floor run read its own seed's reference; no sweep job reads the final requests; the final
scores are bound to their weights; the runner brings every job home.

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
sys.path.insert(0, str(ROOT / "tests"))
import a34_jobs as J  # noqa: E402
from test_a31_schedule import LocalVM, write_set  # noqa: E402

LANE = Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/cue_selective/a31")
pytestmark = pytest.mark.skipif(not (LANE / "readout.py").exists(), reason="the lane's readout.py is not mounted")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a34sched")
    C = tmp / "chat"
    up = C / "upload"
    up.mkdir(parents=True)
    shutil.copy(LANE / "readout.py", up / "a31_readout.py")
    write_set(up, "dev", 8, "d")
    write_set(up, "heldout_dev", 4, "h")
    (up / "a31_early_req.jsonl").write_text("".join(json.dumps({"r": i}) + "\n" for i in range(5 * 12)))
    (up / "a31_early_req.manifest.json").write_text(json.dumps(
        {"requests": 60, "sets": {"dev": {"triplets": 8, "first_request": 0}, "heldout_dev": {"triplets": 4, "first_request": 40}}}))
    (up / "a31_readout_req.jsonl").write_text("".join(json.dumps({"r": i}) + "\n" for i in range(25)))
    (up / "a34_own_req.jsonl").write_text("".join(json.dumps({"id": f"g{g}", "row": r}) + "\n" for g in range(3) for r in ("AA", "BB")))
    (up / "a34_gate_ref.jsonl").write_text('{"g": 0}\n')
    (up / "tc_sft_b1_r1.safetensors").write_text("start weights")
    fake = tmp / "mtstudio"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fake_mtstudio_a34.py'} \"$@\"\n")
    dev_test = tmp / "test_twin_device"
    dev_test.write_text("#!/bin/sh\necho 'twin loss: device vs host (device op set ON)'\necho '    floor: a'\necho '    floor: b'\n")
    for p in (fake, dev_test):
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    text = json.dumps(J.render())
    text = text.replace(J.BIN, str(fake)).replace(J.DEVICE_TEST, str(dev_test)).replace("/content/chat", str(C))
    jobs = json.loads(text)
    (tmp / "jobs.json").write_text(json.dumps(jobs))
    (up / "a34_gate_spec.json").write_text(json.dumps(json.loads(json.dumps(J.gate_spec()).replace("/content/chat", str(C)))))
    env = {**os.environ, "JOBS_CHAT_DIR": str(C), "JOBS_MTSTUDIO": str(fake), "JOBS_REPO": str(ROOT), "JOBS_TICK": "0.2",
           "FAKE_FAIL": "P_50r", "FAKE_CKPT_PAUSE": "0.5", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "jobs_vm.py"), str(tmp / "jobs.json")], env=env,
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    status = {j["name"]: (C / f"STATUS_{j['name']}").read_text().strip() for j in jobs["jobs"]}
    return {"tmp": tmp, "C": C, "jobs": jobs, "status": status, "log": r.stdout}


@pytest.fixture(scope="module")
def home(campaign):
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


def test_every_job_ends_and_the_gate_passes(campaign):
    assert set(campaign["status"].values()) <= {"FINISHED", "FAILED"}, campaign["status"]
    g = json.loads((campaign["C"] / "gate" / "gate.json").read_text())
    assert g["pass"], g["fails"]


def test_references_are_built_from_their_own_runs(campaign):
    C, st = campaign["C"], campaign["status"]
    assert st["ref_start"] == st["ref_D0r"] == st["ref_P_1r"] == "FINISHED"
    assert st["P_50r"] == "FAILED" and st["ref_P_50r"] == "FAILED"
    for run in ("D0r", "P_1r"):
        rec = json.loads((C / f"ref_{run}" / "receipt.json").read_text())
        assert rec["job"] == run and rec["steps"] == [0, *J.REF_STEPS, 1024]
        for k in J.REF_STEPS:
            assert rec["weights_sha256"][str(k)] == sha(C / run / f"rung_{k}" / "model.safetensors")
        assert rec["weights_sha256"]["1024"] == sha(C / run / f"{run}.safetensors")
        assert rec["weights_sha256"]["0"] == sha(C / "upload" / "tc_sft_b1_r1.safetensors")
        lines = [json.loads(x) for x in (C / f"ref_{run}" / "floor_ref.jsonl").read_text().splitlines()]
        assert [x["g"] for x in lines] == [0, 1, 2] and all(x["steps"] == rec["steps"] and x["n_a"] == 2 for x in lines)


def test_the_rule_chooses_the_smallest_floor_that_holds_and_retains(campaign):
    C = campaign["C"]
    dec = json.loads((C / "select" / "decisions.json").read_text())
    t = dec["table"]
    assert t["0.3"]["eligible"] and not t["0.3"]["hold"]
    assert t["1.0"]["hold"] and t["1.0"]["retain"] and t["3.0"]["hold"] and t["3.0"]["retain"]
    assert dec["chosen"] == 1.0 and dec["flags"] == []
    assert dec["decisions"]["a34"] == {"F_0p3": False, "NF_0p3": False, "F_1p0": True, "NF_1p0": True, "F_3p0": False, "NF_3p0": False}
    assert t["1.0"]["parity_step1"] == 1e-4


def test_exactly_the_chosen_floor_runs_where_its_reference_exists(campaign):
    st, C = campaign["status"], campaign["C"]
    for arm in ("F", "NF"):
        assert st[f"{arm}_1_1p0"] == "FINISHED" and st[f"score_{arm}_1_1p0"] == "FINISHED"
        assert st[f"{arm}_50_1p0"] == "FAILED"            # its reference run failed: never started
        assert not (C / f"{arm}_50_1p0" / "events.jsonl").exists()
        for f in ("0p3", "3p0"):
            for sd in (1, 50):
                assert st[f"{arm}_{sd}_{f}"] == "FAILED"
    for n in ("F_1_1p0", "NF_1_1p0"):
        spec = json.loads((C / f"{n}_spec.json").read_text())
        assert spec["data"]["twin_floor_ref"] == str(C / "ref_P_1r" / "floor_ref.jsonl")
        assert spec["train"]["twin_loss"]["floor"] == 1.0 and spec["train"]["seed"] == 1
    assert json.loads((C / "Fd_1p0_spec.json").read_text())["data"]["twin_floor_ref"] == str(C / "ref_D0r" / "floor_ref.jsonl")
    assert st["M_1r"] == "FINISHED" and st["score_M_1r"] == "FINISHED" and st["score_P_1r"] == "FINISHED"


def test_scores_are_bound_and_the_sweep_never_reads_the_final_requests(campaign):
    C = campaign["C"]
    final_req, early_req = sha(C / "upload" / "a31_readout_req.jsonl"), sha(C / "upload" / "a31_early_req.jsonl")
    for n in ("P_1r", "M_1r", "F_1_1p0", "NF_1_1p0"):
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["weights_sha256"] == sha(C / n / f"{n}.safetensors") and b["requests_sha256"] == final_req
        assert b["gguf_sha256"] == sha(C / n / f"{n}.gguf")
    for n in ("D0r", "Fd_0p3", "Fd_1p0", "Fd_3p0"):
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["requests_sha256"] == early_req and "gguf_sha256" not in b
    for j in campaign["jobs"]["jobs"]:
        if j["name"].startswith(("Fd_", "D0r", "early_", "ref_", "select")) or j["name"] in ("score_D0r",) or \
                j["name"].startswith("score_Fd_"):
            assert "a31_readout_req" not in json.dumps(j), j["name"]


def test_the_runner_brings_every_job_home(campaign, home):
    st, art = home["state"], home["art"]
    assert st.get("campaign_home") is True
    assert all(j.get("home") for j in st["jobs"].values()) and set(st["jobs"]) == {j["name"] for j in campaign["jobs"]["jobs"]}
    assert (art / "ref_P_1r" / "floor_ref.jsonl").exists() and (art / "ref_start" / "start_own.jsonl").exists()
    assert (art / "select" / "decisions.json").exists() and (art / "F_1_1p0" / "F_1_1p0.gguf").exists()
    assert home["T"].released == 1
