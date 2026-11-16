"""A35's jobs file: S71's chat recipe on base C, with nothing changed but the base. Claude Fable 5.2, 16 Nov 2026.

The rendering is checked against the file on disk, against A's SFT1500 recipe and S71's spec as A31 and A32 render them, and for the
absence of any twin term or twin data. Then the whole schedule runs under the real jobs_vm.py on A32's stand-in trainer
(tests/fake_mtstudio_b2cue.py): every job ends, the reproduction record is written, the watch relays the SFT1500 readouts, and each
arm's score.json binds its weights, its GGUF-less stand-in aside, and the early request file. CPU only, under a minute."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import a31_jobs as A31  # noqa: E402
import a35_jobs as J  # noqa: E402
import b2cue_jobs as B  # noqa: E402


def by_name():
    return {j["name"]: j for j in J.render()["jobs"]}


def test_file_on_disk_is_the_rendering():
    assert (ROOT / "tools" / "colab_chat" / "a35_c.json").read_text() == json.dumps(J.render(), indent=1) + "\n"


def test_job_list_and_order():
    names = [j["name"] for j in J.render()["jobs"]]
    assert names == ["repro_A", "gate", "sft_C", "watch_sft_C",
                     "S75_1", "early_S75_1_128", "early_S75_1_256", "early_S75_1_512",
                     "S75_2", "early_S75_2_128", "early_S75_2_256", "early_S75_2_512",
                     "score_S75_1", "score_S75_2"]
    heavy = [j["name"] for j in J.render()["jobs"] if "spec" in j]
    assert heavy == ["repro_A", "sft_C", "S75_1", "S75_2"]          # the heavy pool runs in list order, one at a time
    assert J.render()["max_parallel"] == 1


def test_sft1500_is_a_recipe_from_c():
    s = by_name()["sft_C"]["spec"]
    assert s["train"]["init_from"] == "/content/chat/upload/base_C.safetensors"
    assert s["arch"] == B.A_ARCH                                     # C continued A's weights: A's shape
    want = json.loads(json.dumps(B.SFT1500))
    got = json.loads(json.dumps(s))
    got.pop("arch")
    got["train"].pop("init_from")
    assert got == want
    t = s["train"]
    assert (t["steps"], t["lr"], t["batch"], t["seed"], t["lr_schedule"]["warmup_steps"]) == (1500, 0.0003, 16, 1, 100)
    assert s["data"]["sft_examples"].endswith("/sft_human.jsonl")


def test_arms_are_s71s_spec_with_only_the_start_the_seed_and_the_packet_path():
    jobs = by_name()
    for sd in (1, 2):
        s = jobs[f"S75_{sd}"]["spec"]
        ref = A31.spec(sd)                                           # S71's spec as A31 renders it (plain: no twin term)
        assert "twin_loss" not in s["train"] and "twin_examples" not in s["data"]
        diff = {(k, kk) for k in ("arch", "data", "train", "export") for kk in set(s[k]) | set(ref[k]) if s[k].get(kk) != ref[k].get(kk)}
        assert diff == {("data", "sft_examples"), ("train", "init_from")}, diff
        assert s["data"]["sft_examples"] == "/content/chat/upload/S71.jsonl"
        assert s["train"]["init_from"] == "/content/chat/sft_C/sft_C.safetensors"
        assert s["train"]["seed"] == sd and s["train"]["steps"] == 1024 and s["train"]["lr"] == 0.0001
        assert jobs[f"S75_{sd}"]["init_job"] == "sft_C" and set(jobs[f"S75_{sd}"]["after"]) == {"gate", "sft_C"}


def test_no_job_names_a_twin_file_or_a_sealed_request_file():
    text = json.dumps(J.render())
    for word in ("train_twins", "twin_loss", "answer_mask", "confirmation", "selection_req", "S74", "S73", "readout_req"):
        assert word not in text, word
    for sd in (1, 2):
        cmd = " ".join(by_name()[f"score_S75_{sd}"]["cmd"])
        assert "--in /content/chat/upload/a31_early_req.jsonl" in cmd and "a31_bind.py" in cmd
        assert f"--gguf /content/chat/S75_{sd}/S75_{sd}.gguf" in cmd
    for sd in (1, 2):
        for st in (128, 256, 512):
            assert "--no-score" in by_name()[f"early_S75_{sd}_{st}"]["cmd"]


def test_gate_is_the_reproduction_check_and_never_blocks():
    g = by_name()["gate"]
    assert "--repro" in g["cmd"] and g["after"] == ["repro_A"] and g["after_terminal"] is True and g["outputs"] == ["repro.json"]
    assert "after" not in by_name()["repro_A"]                      # no cycle: the first job waits on nothing
    r = by_name()["repro_A"]["spec"]
    assert r["train"]["steps"] == 50 and r["train"]["init_from"].endswith("/base_A.safetensors") and "eval_at" not in r["train"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a35")
    C = tmp / "chat"
    up = C / "upload"
    up.mkdir(parents=True)
    for f in ("vocab.gguf", "S71.jsonl", "sft_human.jsonl", "base_C.safetensors", "base_A.safetensors"):
        (up / f).write_text(f"stand-in {f}")
    # the stand-in trainer's step loss is 3.0 - 0.5 * step / steps: the reference events of the 50-step reproduction match it
    (up / "sft_b1_r1_events.jsonl").write_text("".join(json.dumps({"event": "step", "step": s, "loss": 3.0 - 0.5 * s / 50}) + "\n" for s in range(1, 51)))
    (up / "a31_early_req.jsonl").write_text("".join(json.dumps({"r": i}) + "\n" for i in range(20)))
    fake = tmp / "mtstudio"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fake_mtstudio_b2cue.py'} \"$@\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    jobs = json.loads(json.dumps(J.render()).replace(B.BIN, str(fake)).replace("/content/chat", str(C)))
    (tmp / "jobs.json").write_text(json.dumps(jobs))
    env = {**os.environ, "JOBS_CHAT_DIR": str(C), "JOBS_MTSTUDIO": str(fake), "JOBS_REPO": str(ROOT), "JOBS_TICK": "0.2",
           "FAKE_CKPT_PAUSE": "0.3", "B2CUE_POLL_S": "0.5", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "jobs_vm.py"), str(tmp / "jobs.json")], env=env,
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return {"C": C, "status": {j["name"]: (C / f"STATUS_{j['name']}").read_text().strip() for j in jobs["jobs"]}}


def test_schedule_every_job_finishes(ran):
    assert set(ran["status"].values()) == {"FINISHED"}, ran["status"]


def test_schedule_reproduction_record(ran):
    g = json.loads((ran["C"] / "gate" / "repro.json").read_text())
    assert g["pass"] and g["steps"] == 50 and g["job"] == "repro_A" and not g["failed_steps"]


def test_schedule_watch_and_early(ran):
    w = json.loads((ran["C"] / "watch_sft_C" / "watch.json").read_text())
    assert set(w["at"]) == {"188", "375", "750"} and w["complete"] and not w.get("health_stop")
    for sd in (1, 2):
        e = json.loads((ran["C"] / f"early_S75_{sd}_256" / "early.json").read_text())
        assert not e["scored"] and not e.get("health_stop")


def test_schedule_arms_start_from_the_sft1500_and_are_bound(ran):
    C = ran["C"]
    req = sha(C / "upload" / "a31_early_req.jsonl")
    for sd in (1, 2):
        n = f"S75_{sd}"
        spec = json.loads((C / f"{n}_spec.json").read_text())
        assert spec["train"]["init_from"] == str(C / "sft_C" / "sft_C.safetensors") and spec["train"]["seed"] == sd
        assert "twin_loss" not in spec["train"]
        b = json.loads((C / f"score_{n}" / "score.json").read_text())
        assert b["weights_sha256"] == sha(C / n / f"{n}.safetensors") and b["requests_sha256"] == req
