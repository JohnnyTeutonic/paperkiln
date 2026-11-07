"""The stronger base's packet: jobs files, data pins, source filter, restore guard, and the runner's failure path.

No pod and no GPU: the runner runs against tests/runner_multi_equiv/fakecolab.py, the checkpoint path has its own file
(test_checkpoint_txn.py). POSIX only (the fake VM uses sh)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(ROOT / "tests" / "runner_multi_equiv"))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX paths and sh")

import make_tc_b1 as M  # noqa: E402
import tc_b1_pins as P  # noqa: E402


def load(n):
    return json.load(open(CC / n))


def test_race_arms_differ_in_shape_alone():
    a = next(j for j in load("tc_b1_r1.json")["jobs"] if j["name"] == "pt_b1_r1")["spec"]
    b = next(j for j in load("tc_b2_l12.json")["jobs"] if j["name"] == "pt_b2_l12")["spec"]
    assert a["data"] == b["data"]
    assert a["train"]["steps"] == b["train"]["steps"] == 60000
    assert a["train"]["batch"] * a["train"]["accum"] == b["train"]["batch"] * b["train"]["accum"] == 16
    drop = ("batch", "accum", "checkpoint_every")
    assert {k: v for k, v in a["train"].items() if k not in drop} == {k: v for k, v in b["train"].items() if k not in drop}
    assert b["arch"]["custom"] == {"d": 768, "layers": 12, "heads": 12, "d_ff": 2048}
    assert b["arch"]["state_tokens"] == a["arch"]["state_tokens"]
    assert (b["train"]["batch"], b["train"]["accum"]) == (8, 2)
    assert 20000 * 16 * 1024 == 327680000


def test_race_arm_gates_on_its_probe_and_the_same_pins():
    jobs = {j["name"]: j for j in load("tc_b2_l12.json")["jobs"]}
    assert "l12_probe" in jobs["pt_b2_l12"]["after"] and jobs["l12_probe"]["halt_on_failure"]
    assert jobs["data"]["cmd"][-1] == "b1" and jobs["data"]["env"]["TC_B1_KEEP_MIX_TEXT"] == "1"
    assert jobs["rung_l12_s20000"].get("keep_home_snapshot") and "cell_b2_l12" in " ".join(jobs["rung_l12_s20000"]["cmd"])
    assert jobs["sft_b2_l12"]["spec"]["train"]["accum"] == 2 and jobs["sft_b2_l12"]["init_job"] == "pt_b2_l12"
    pins = load("tc_b1_pins.json")
    assert all(pins["mix"].values())


def test_probe_bars():
    import tc_b2_probe as Q
    ok = {"peak_pool_mib": 15000, "peak_smi_mib": 18000, "s_per_step": 2.1, "params": 97539840}
    assert Q.verdict(ok) == []
    assert Q.verdict({**ok, "peak_smi_mib": 18100})
    assert Q.verdict({**ok, "s_per_step": 2.31})
    assert Q.verdict({**ok, "params": 72761088})
    assert Q.verdict({"peak_pool_mib": None, "peak_smi_mib": None, "s_per_step": None})


def race_state(home, name="pk-tc-b2-l12", pod=None, pods=True):
    os.makedirs(home, exist_ok=True)
    st = {"runpod": {"pod": pod, "volume": {"id": "v1", "dc": "X", "gb": 80},
                     "ledger": {"pods": [{"id": "p", "name": name, "usd_per_hr": 0.59, "start": 0, "stop": 3600}] if pods else [],
                                "volumes": [{"id": "v1", "gb": 80, "created": 0, "name": f"{name}-vol"}]}}}
    json.dump(st, open(os.path.join(home, "state.json"), "w"))


def test_stop_releases_only_its_own_volume(tmp_path, monkeypatch):
    import runpod_api as api
    import tc_b2_stop as X
    home = str(tmp_path / "s")
    race_state(home)
    deleted = []
    monkeypatch.setattr(api, "list_volumes", lambda: [{"id": "v1", "name": "pk-tc-b2-l12-vol"}, {"id": "other", "name": "x"}])
    monkeypatch.setattr(api, "delete_volume", lambda i: deleted.append(i) or True)
    X.stop(home, "pk-tc-b2-l12", "lost at the cut-off", sleep=lambda s: None)
    assert deleted == []  # stop never releases
    st = X.release_volume(home, "pk-tc-b2-l12", "the lead's word")
    assert deleted == ["v1"] and st["runpod"]["volume"] is None
    assert json.load(open(os.path.join(home, "HALTED")))["kind"] == "RACE_ARM_STOPPED"


def test_stop_keeps_the_volume_unless_told_and_refuses_a_stranger(tmp_path, monkeypatch):
    import runpod_api as api
    import tc_b2_stop as X
    monkeypatch.setattr(api, "delete_volume", lambda i: pytest.fail("no delete"))
    home = str(tmp_path / "s")
    race_state(home)
    X.stop(home, "pk-tc-b2-l12", "held", sleep=lambda s: None)
    with pytest.raises(SystemExit):           # the wrong arm's name for this directory
        X.release_volume(home, "pk-tc-b1-r1", "typo")
    other = str(tmp_path / "o")
    race_state(other, name="pk-c12-sfull-r1")
    with pytest.raises(SystemExit):
        X.stop(other, "pk-tc-b2-l12", "no", sleep=lambda s: None)
    assert not os.path.exists(os.path.join(other, "HALTED"))
    nopod = str(tmp_path / "n")               # a volume made, no pod ever up
    race_state(nopod, pods=False)
    X.stop(nopod, "pk-tc-b2-l12", "race off", sleep=lambda s: None)


def test_pause_and_release(tmp_path):
    import runner_multi as rm
    import tc_b2_stop as X
    home = tmp_path / "h"
    race_state(str(home))
    rm.HOME = str(home)
    (home / "runner.log").write_text("")

    class St:
        jobs = {"rung_l12_s20000": {"home": False}, "rung_l12_s30000": {"home": False}}

        def job(self, n):
            return self.jobs[n]

    j = {"name": "pt_b2_l12", "pause_rungs": ["rung_l12_s20000", "rung_l12_s30000"]}
    st = St()
    assert rm.pause_at_rung(j, st) is None
    st.jobs["rung_l12_s20000"]["home"] = True
    assert rm.pause_at_rung(j, st) == "rung_l12_s20000"
    assert json.load(open(home / "HALTED"))["kind"] == "CUTOFF_REACHED"
    X.release_rung(str(home), "pk-tc-b2-l12", "rung_l12_s20000", "verdict a")
    assert not (home / "HALTED").exists() and (home / "HALTED.rung_l12_s20000").exists()
    assert rm.pause_at_rung(j, st) is None
    st.jobs["rung_l12_s30000"]["home"] = True
    assert rm.pause_at_rung(j, st) == "rung_l12_s30000"


def test_the_arms_pause_at_the_reads_and_keep_whole_checkpoints():
    for name, job, tag in (("tc_b1_r1.json", "pt_b1_r1", "b1"), ("tc_b2_l12.json", "pt_b2_l12", "l12")):
        jobs = {j["name"]: j for j in load(name)["jobs"]}
        assert jobs[job]["pause_rungs"] == [f"rung_{tag}_s20000", f"rung_{tag}_s30000"]
        assert jobs[job]["snapshot_full_steps"] == [20000, 30000]
        assert "overlap_check" in jobs[job]["after"]
        for s in (10000, 20000, 30000):
            r = jobs[f"rung_{tag}_s{s}"]
            assert r["halt_on_failure"] and r["keep_home_snapshot"] and "model.gguf" in r["outputs"] and "--export-gguf" in r["cmd"]
        assert not jobs[f"rung_{tag}_s40000"].get("halt_on_failure")
    assert "overlap" in {j["name"] for j in load("tc_b1_r1.json")["jobs"]}
    assert "overlap" not in {j["name"] for j in load("tc_b2_l12.json")["jobs"]}
    assert next(j for j in load("tc_b2_l12.json")["jobs"] if j["name"] == "data")["after"] == "l12_probe"


def test_overlap_gate_names_both_hashes(tmp_path):
    import tc_b2_overlap as O
    want = load("tc_b1_pins.json")["mix"]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(O, "manifest_text", lambda d: "the manifest")
    good = {"mix": want, "decisions": {"b2": {"go": True}}, "packet_sha256": O.PACKET_SHA256,
            "worker_sha256": O.sha256(str(CC / "codex_overlap_worker.py")),
            "manifest_sha256": hashlib.sha256(b"the manifest").hexdigest()}
    d = tmp_path / "d.json"
    args = ["check", "--data", "x", "--decision", str(d), "--out", str(tmp_path / "o")]
    d.write_text(json.dumps(good))
    assert O.main(args) == 0
    for bad in ({"mix": {k: "0" for k in want}}, {"decisions": {"b2": {"go": False}}}, {"worker_sha256": "0"}, {"packet_sha256": "0"},
                {"manifest_sha256": "0"}):
        d.write_text(json.dumps({**good, **bad}))
        with pytest.raises(SystemExit):
            O.main(args)
    monkey.undo()


def test_overlap_manifest_refuses_text_or_ids_that_are_not_the_jobs(tmp_path):
    import tc_b2_overlap as O
    d = tmp_path
    (d / "mix_b1.main.txt").write_text("m")
    (d / "mix_b1.tail.txt").write_text("t")
    (d / "mix_b1.main.ids").write_text("i")
    (d / "mix_b1.tail.ids").write_text("j")
    (d / "mix_b1.ledger.json").write_text(json.dumps({"parts": {"main": {"sha256": "0"}, "tail": {"sha256": "0"}}}))
    with pytest.raises(SystemExit):
        O.manifest(str(d))
    assert all(load("tc_b1_pins.json")["mix_text"].values())


def overlap_fixture(tmp_path, monkeypatch):
    import tc_b2_overlap as O
    d = tmp_path
    sha = {}
    for name, body in (("mix_b1.main.txt", "m"), ("mix_b1.tail.txt", "t"), ("mix_b1.main.ids", "i"), ("mix_b1.tail.ids", "j")):
        (d / name).write_text(body)
        sha[name] = hashlib.sha256(body.encode()).hexdigest()
    (d / "mix_b1.ledger.json").write_text(json.dumps({"parts": {"main": {"sha256": sha["mix_b1.main.txt"]}, "tail": {"sha256": sha["mix_b1.tail.txt"]}}}))
    monkeypatch.setattr(O, "pins", lambda: {k: sha[k] for k in ("mix_b1.main.ids", "mix_b1.tail.ids")})
    monkeypatch.setattr(O, "text_pins", lambda: {k: sha[k] for k in ("mix_b1.main.txt", "mix_b1.tail.txt")})
    return O, sha


def test_overlap_manifest_on_a_correct_mix_and_on_each_wrong_pin(tmp_path, monkeypatch):
    O, sha = overlap_fixture(tmp_path, monkeypatch)
    rows = O.manifest(str(tmp_path))
    assert [r["name"] for r in rows] == ["main", "tail"]
    assert rows[0]["text"].endswith("mix_b1.main.txt") and rows[0]["text_sha256"] == sha["mix_b1.main.txt"]
    assert rows[1]["ids_sha256"] == sha["mix_b1.tail.ids"]
    monkeypatch.setattr(O, "text_pins", lambda: {"mix_b1.main.txt": "0", "mix_b1.tail.txt": sha["mix_b1.tail.txt"]})
    with pytest.raises(SystemExit):
        O.manifest(str(tmp_path))
    O, sha = overlap_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(O, "pins", lambda: {"mix_b1.main.ids": sha["mix_b1.main.ids"], "mix_b1.tail.ids": "0"})
    with pytest.raises(SystemExit):
        O.manifest(str(tmp_path))


def test_overlap_scan_runs_the_real_worker_on_a_correct_mix(tmp_path, monkeypatch):
    O, sha = overlap_fixture(tmp_path, monkeypatch)
    packet = tmp_path / "p.json"
    packet.write_text(json.dumps({"version": 1, "key": "00" * 16, "targets": []}))
    monkeypatch.setattr(O, "PACKET_SHA256", O.sha256(str(packet)))
    out = tmp_path / "o"
    assert O.main(["scan", "--data", str(tmp_path), "--packet", str(packet), "--out", str(out)]) == 0
    r = json.load(open(out / "overlap_receipt.json"))
    assert r["status"] == "COMPLETE_REQUIRES_CODEX_REVIEW"
    assert r["identity"]["manifest_sha256"] == hashlib.sha256(O.manifest_text(str(tmp_path)).encode()).hexdigest()


def test_exclusive_rung_past_its_limit_fails_and_training_carries_on(tmp_path):
    chat = tmp_path / "chat"
    (chat / "pt").mkdir(parents=True)
    (chat / "pt" / "model.safetensors").write_text("w")
    fake = tmp_path / "fake_mtstudio"
    fake.write_text("#!/bin/sh\nout=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['out_dir'])\" $2)\nmkdir -p $out\n"
                    "echo 5 > $out/state.txt\ni=0\nwhile [ $i -lt 40 ]; do i=$((i+1)); sleep 0.1; echo $i > $out/ticks; done\n")
    fake.chmod(0o755)
    jobs = {"max_parallel": 1, "max_parallel_light": 1, "jobs": [
        {"name": "pt", "spec": {}, "snapshot_steps": [5], "model_name": "m"},
        {"name": "rung", "light": True, "exclusive": True, "timeout_s": 1, "rung_of": {"job": "pt", "step": 5}, "cmd": ["sleep", "30"]}]}
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    env = {**os.environ, "JOBS_CHAT_DIR": str(chat), "JOBS_REPO": str(tmp_path), "JOBS_MTSTUDIO": str(fake), "JOBS_TICK": "0.05"}
    subprocess.run([sys.executable, str(CC / "jobs_vm.py"), str(jf)], env=env, check=True, timeout=25, capture_output=True)
    assert (chat / "STATUS_rung").read_text().strip() == "FAILED"
    assert (chat / "STATUS_pt").read_text().strip() == "FINISHED"
    assert (chat / "pt" / "ticks").read_text().strip() == "40"


def test_exclusive_rung_stops_training_and_a_failed_rung_never_goes_home(tmp_path):
    """The real jobs_vm.py with a fake trainer: the trainer sits stopped while the rung runs, and carries on after it."""
    chat = tmp_path / "chat"
    chat.mkdir()
    fake = tmp_path / "fake_mtstudio"
    fake.write_text("#!/bin/sh\nout=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['out_dir'])\" $2)\nmkdir -p $out\n"
                    "echo 5 > $out/state.txt\ni=0\nwhile [ $i -lt 60 ]; do i=$((i+1)); sleep 0.1; echo $i > $out/ticks; done\n")
    fake.chmod(0o755)
    probe = tmp_path / "rung.py"
    probe.write_text("import glob, os, sys, time\n"
                     f"p = [d for d in os.listdir('/proc') if d.isdigit() and b'fake_mtstudio' in open(f'/proc/{{d}}/cmdline','rb').read()]\n"
                     "state = [open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()[0] for d in p]\n"
                     f"open({str(tmp_path / 'seen')!r}, 'w').write(' '.join(state))\n"
                     f"sys.exit({0})\n")
    jobs = {"max_parallel": 1, "max_parallel_light": 1, "jobs": [
        {"name": "pt", "spec": {}, "snapshot_steps": [5], "model_name": "m"},
        {"name": "rung", "light": True, "exclusive": True, "timeout_s": 30, "rung_of": {"job": "pt", "step": 5}, "cmd": ["python3", str(probe)]}]}
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    (chat / "pt").mkdir()
    (chat / "pt" / "model.safetensors").write_text("w")
    env = {**os.environ, "JOBS_CHAT_DIR": str(chat), "JOBS_REPO": str(tmp_path), "JOBS_MTSTUDIO": str(fake), "JOBS_TICK": "0.05"}
    subprocess.run([sys.executable, str(CC / "jobs_vm.py"), str(jf)], env=env, check=True, timeout=60, capture_output=True)
    assert "T" in (tmp_path / "seen").read_text().split()      # stopped while the rung ran
    assert (chat / "pt" / "ticks").read_text().strip() == "60"  # and it finished afterwards
    assert (chat / "STATUS_pt").read_text().strip() == "FINISHED"


def test_relay_copies_one_generation_through_a_commit(tmp_path):
    """The relay's real exec code on a generation layout, with a commit landing during the copy."""
    import re as _re
    import runner_multi as rm
    out = tmp_path / "chat" / "pt"
    gens = out / "ckpt_gen"
    for step in (125, 250):
        g = gens / f"g{step}"
        g.mkdir(parents=True)
        (g / "model.safetensors").write_text(f"m{step}")
        (g / "optim.safetensors").write_text(f"o{step}")
        (g / "state.txt").write_text(f"{step}\n{{}}\n")
    (gens / "live").symlink_to("g125")
    (out / "events.jsonl").write_text("{}\n")
    captured = {}

    class T:
        def exec(self, code, timeout=200):
            captured["code"] = code
            return 0, ""

    rm.T = T()
    rm.HOME, rm.ART = str(tmp_path), str(tmp_path / "art")

    class St:
        def job(self, n):
            return {"relayed_step": 0}

    rm.relay(St(), "pt", 0)
    code = captured["code"].replace("/content/chat/", str(tmp_path / "chat") + "/")
    # The commit lands after the generation is named and before the copy: g125 outlives it.
    code = code.replace("s0 = step()\n", "s0 = step()\nos.unlink(live); os.symlink('g250', live)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert _re.search(r"SNAP 125", r.stdout)
    snap = tmp_path / "chat" / "relay_pt"
    assert (snap / "model.safetensors").read_text() == "m125" and (snap / "optim.safetensors").read_text() == "o125"
    assert (snap / "state.txt").read_text().startswith("125")


def test_fetch_retries_transient_errors_only(monkeypatch):
    sys.path.insert(0, str(ROOT / "corpus" / "fetchers"))
    import urllib.error
    import human_dialogue as H
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 504, "Gateway Timeout", {}, None)
        return "bytes"

    assert H.with_retries(flaky, "u") == "bytes" and len(calls) == 3
    calls.clear()

    def down():
        calls.append(1)
        raise urllib.error.HTTPError("u", 504, "Gateway Timeout", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        H.with_retries(down, "u")
    assert len(calls) == 6                                       # bounded
    calls.clear()

    def gone():
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        H.with_retries(gone, "u")
    assert len(calls) == 1                                       # a client error stands at once


def test_reset_failed_only_on_a_stopped_arm_that_never_trained(tmp_path):
    import tc_b2_stop as X
    home = tmp_path / "h"
    race_state(str(home), name="pk-tc-b1-r1")
    st = json.load(open(home / "state.json"))
    st["jobs"] = {"cuda_gate": {"status": "FINISHED", "home": True, "relayed_step": 0}, "data": {"status": "FAILED", "home": False, "relayed_step": 0, "failed_seen": 2},
                  "pt_b1_r1": {"status": "FAILED", "home": False, "relayed_step": 0}}
    (home / "state.json").write_text(json.dumps(st))
    with pytest.raises(SystemExit):                              # not halted
        X.reset_failed(str(home), "pk-tc-b1-r1", "go")
    (home / "HALTED").write_text(json.dumps({"kind": "JOB_FAILED_STATE_KEPT", "job": "data"}))
    out = X.reset_failed(str(home), "pk-tc-b1-r1", "Codex's go")
    assert out["reset_jobs"] == ["data", "pt_b1_r1"] and out["jobs"]["cuda_gate"]["home"] is True
    assert out["jobs"]["data"] == {"relayed_step": 0, "status": None, "home": False}
    assert (home / "HALTED.failed").exists() and not (home / "HALTED").exists()
    race_state(str(tmp_path / "t"), name="pk-tc-b1-r1")
    st = json.load(open(tmp_path / "t" / "state.json"))
    st["jobs"] = {"pt_b1_r1": {"status": "FAILED", "home": False, "relayed_step": 2500}}
    (tmp_path / "t" / "state.json").write_text(json.dumps(st))
    (tmp_path / "t" / "HALTED").write_text(json.dumps({"kind": "JOB_FAILED_STATE_KEPT", "job": "pt_b1_r1"}))
    with pytest.raises(SystemExit):                              # never for an arm that has trained
        X.reset_failed(str(tmp_path / "t"), "pk-tc-b1-r1", "go")


def test_a_failed_job_can_run_again_after_a_reset(tmp_path):
    """The runner against the fake CLI: halted on a failed job, reset by the tool, run again on the same state, and home."""
    import run_eq
    jobs = {"jobs": [{"name": "good", "cmd": ["x"], "outputs": ["out.txt"]},
                     {"name": "bad", "cmd": ["z"], "outputs": ["x.json"], "after": "good", "fake_fail": True, "halt_on_failure": True}]}
    r = run_fake(tmp_path, jobs, {})
    assert r["rc"] == 3 and r["state"]["jobs"]["bad"]["home"] is False
    assert (tmp_path / "w" / "state" / "HALTED").exists()
    st = json.load(open(tmp_path / "w" / "state" / "state.json"))
    st["jobs"]["bad"] = {"relayed_step": 0, "status": None, "home": False}
    st["reset_jobs"] = ["bad"]
    st["booted"] = False
    json.dump(st, open(tmp_path / "w" / "state" / "state.json", "w"))
    os.replace(tmp_path / "w" / "state" / "HALTED", tmp_path / "w" / "state" / "HALTED.failed")
    jobs["jobs"][1].pop("fake_fail")
    r = run_eq.run(str(CC / "runner_multi.py"), str(tmp_path / "w"), {}, jobs=jobs, fresh=False)
    assert r["rc"] == 0, r["stderr"][-400:]
    assert r["state"]["jobs"]["bad"]["home"] is True and r["state"]["reset_jobs"] == []
    assert "jobs reset on the volume" in r["log"]


def test_files_are_what_the_generator_writes():
    s0, r1 = load("tc_b1_s0.json"), load("tc_b1_r1.json")
    assert s0 == json.loads(json.dumps(M.stage0()))
    assert r1 == json.loads(json.dumps(M.b1()[0]))
    assert load("tc_b1_probe_spec.json") == json.loads(json.dumps(M.probe_spec()))
    assert load("tc_b2_l12.json") == json.loads(json.dumps(M.l12()[0]))
    assert load("tc_b2_probe_spec.json") == json.loads(json.dumps(M.l12_probe_spec()))


def test_b_spec_is_section_9():
    s = next(j for j in load("tc_b1_r1.json")["jobs"] if j["name"] == "pt_b1_r1")
    assert s["model_name"] == "tc_b1_r1" and s["relay_every_steps"] == 2500
    assert s["snapshot_steps"] == [1000, 10000, 20000, 30000, 40000, 50000]
    t, d, a = s["spec"]["train"], s["spec"]["data"], s["spec"]["arch"]
    assert t["steps"] * t["batch"] * d["T"] == 983040000
    assert (t["lr"], t["batch"], t["accum"], t["checkpoint_every"], t["eval_every"], t["seed"]) == (0.0005, 16, 1, 125, 1000, 1)
    assert t["lr_schedule"] == {"warmup_steps": 2000, "decay": "cosine", "min_lr_frac": 0.1}
    assert d["anneal"] == {"corpus": "/content/chat/data/mix_b1.tail.ids", "from_fraction": 0.85}
    assert a["custom"] == {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240}
    assert a["state_tokens"] == {"k": 4, "mask": "causal"}
    sfull = next(j for j in load("carry12_sfull_r1.json")["jobs"] if j["name"] == "pt_c12_Sfull_r1")
    assert sfull["spec"]["arch"] == a  # strict load from every S-lineage recipe


@pytest.mark.parametrize("name", ["tc_b1_s0.json", "tc_b1_r1.json", "tc_b2_l12.json"])
def test_jobs_are_well_formed(name):
    jobs = load(name)["jobs"]
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)
    for j in jobs:
        after = j.get("after") or []
        for a in [after] if isinstance(after, str) else after:
            assert a in names
        if j.get("rung_of"):
            assert j["rung_of"]["step"] in next(x for x in jobs if x["name"] == j["rung_of"]["job"])["snapshot_steps"]
            assert "--drop-snapshot" not in j["cmd"]
        if "spec" in j:
            assert j["env"]["MICROTORCH_CHECKPOINT_TXN"] == "1", j["name"]
            for s in j.get("snapshot_steps", []):
                assert s % j["spec"]["train"]["checkpoint_every"] == 0
            assert j.get("halt_on_failure") or j.get("halt_drill")
    assert all(j.get("halt_on_failure") for j in jobs if j["name"] in ("data", "cuda_gate"))


def test_stage0_has_every_drill():
    jobs = {j["name"]: j for j in load("tc_b1_s0.json")["jobs"]}
    assert jobs["pt_s0"]["pod_loss_drill_step"] == 250
    assert jobs["pt_s0"]["spec"]["train"]["checkpoint_every"] == 125
    assert jobs["s0_torn"].get("halt_drill") and not jobs["s0_torn"].get("halt_on_failure")
    assert jobs["s0_torn"]["after"] == "s0_torn_setup"
    assert jobs["data"]["cmd"][-1] == "stage0" and "b1" == next(
        j for j in load("tc_b1_r1.json")["jobs"] if j["name"] == "data")["cmd"][-1]
    assert jobs["pt_s0"]["spec"]["data"]["anneal"]["from_fraction"] == 0.85


def test_rungs_kept_at_home_for_the_cpu_lambada():
    j = {x["name"]: x for x in load("tc_b1_r1.json")["jobs"]}
    assert j["rung_b1_s10000"].get("keep_home_snapshot") and j["rung_b1_s20000"].get("keep_home_snapshot")
    assert not j["rung_b1_s40000"].get("keep_home_snapshot")


def test_sft_recipe_is_sfull_sft1500():
    t = json.load(open(ROOT / "experiments/tiny_chat_long_2026-10/specs/sft_T1024.json"))
    s = next(j for j in load("tc_b1_r1.json")["jobs"] if j["name"] == "sft_b1_r1")
    assert s["init_job"] == "pt_b1_r1"
    assert s["spec"]["data"] == t["data"]
    assert s["spec"]["train"] == {**t["train"], "batch": 16, "accum": 1, "seed": 1}


def test_data_script_ratios():
    text = (CC / "base_b1_data.sh").read_text()
    assert "\r" not in text
    subprocess.run(["bash", "-n", str(CC / "base_b1_data.sh")], check=True)
    for key in ("MAIN=", "TAIL="):
        line = next(ln for ln in text.splitlines() if ln.startswith(key))
        vals = dict(x.split("=") for x in line.split("=", 1)[1].strip('"').split(","))
        assert abs(sum(float(v) for v in vals.values()) - 1.0) < 2e-6, line
        assert set(vals) == {"fineweb", "human", "soda", "ultrachat", "everyday"}
    assert "msc" in text and "--allow-unstated-licence" in text
    assert "dialogue_oasst_v1" not in text and "simplewiki" not in text


def test_pins_refuse_and_record(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "m.ids").write_text("m")
    want = hashlib.sha256(b"a").hexdigest()
    pins = {"files": {"a.txt": want}, "mix": {"m.ids": None}}
    assert P.check(pins, str(tmp_path), "stage0")[0] == []
    assert P.check(pins, str(tmp_path), "b1")[0]  # B refuses without Stage 0's hashes
    pins["mix"]["m.ids"] = hashlib.sha256(b"m").hexdigest()
    assert P.check(pins, str(tmp_path), "b1")[0] == []
    (tmp_path / "m.ids").write_text("n")
    assert P.check(pins, str(tmp_path), "b1")[0]
    (tmp_path / "a.txt").write_text("b")
    assert P.check(pins, str(tmp_path), "stage0")[0]


def test_filter_drops_held_out_text_and_keeps_the_rest(tmp_path):
    sys.path.insert(0, str(ROOT / "corpus" / "fetchers"))
    import _common as C
    a = " ".join(f"word{i}" for i in range(8))
    b = " ".join(f"word{i}" for i in range(8, 20))
    c = " ".join(f"later{i}" for i in range(8))
    d = " ".join(f"later{i}" for i in range(8, 20))
    shared = f"{a} {b}"
    held = [{"role": "user", "content": a}, {"role": "assistant", "content": b},
            {"role": "user", "content": c}, {"role": "assistant", "content": d}]
    val = tmp_path / "val.txt"
    val.write_text(C.render_dialogue(held), encoding="utf-8")  # the real format: markers between the turns
    sft = tmp_path / "sft.jsonl"
    sft.write_text(json.dumps({"messages": [{"role": "user", "content": "only sft text " + " ".join(f"x{i}" for i in range(14))}]}) + "\n")
    src = tmp_path / "human_msc_v1" / "dialogues.jsonl"
    src.parent.mkdir()
    across = f"{' '.join(b.split()[-6:])} {' '.join(c.split()[:7])}"  # a gram over two exchanges of one val dialogue
    rows = [[{"role": "user", "content": shared}, {"role": "assistant", "content": "fine"}],
            [{"role": "user", "content": across}, {"role": "assistant", "content": "that too"}],
            [{"role": "user", "content": "unrelated talk here"}, {"role": "assistant", "content": "yes indeed it is"}],
            [{"role": "user", "content": "only sft text " + " ".join(f"x{i}" for i in range(14))},
             {"role": "assistant", "content": "ok"}]]
    src.write_text("".join(json.dumps({"messages": m, "uri": str(i)}) + "\n" for i, m in enumerate(rows)))
    out = tmp_path / "f"
    subprocess.run([sys.executable, str(CC / "tc_b1_filter.py"), "--repo", str(ROOT), "--heldout", str(val), str(sft),
                    "--out", str(out), "--source", f"msc={src}"], check=True, capture_output=True)
    led = json.load(open(out / "filter.ledger.json"))["sources"]["msc"]
    assert (led["rows_in"], led["rows_kept"], led["dropped_heldout_13gram"]) == (4, 1, 3)
    assert led["gate"] == "licence-unstated" and "not stated" in led["licence"] and led["model_generated"] is False
    assert "unrelated talk here" in (out / "msc.txt").read_text()


def test_fineweb_keeps_each_document_once_and_never_a_val_document(tmp_path):
    sep = "<|endoftext|>\n"
    (tmp_path / "s0").mkdir()
    (tmp_path / "s1").mkdir()
    val = tmp_path / "val.txt"
    val.write_text(f"panel doc\n{sep}", encoding="utf-8")
    a = tmp_path / "s0" / "t.txt"
    a.write_text(f"one\n{sep}two\n{sep}one\n{sep}", encoding="utf-8")
    b = tmp_path / "s1" / "t.txt"
    b.write_text(f"two\n{sep}three\n{sep}panel doc\n{sep}", encoding="utf-8")
    out = tmp_path / "f.txt"
    subprocess.run([sys.executable, str(CC / "tc_b1_fineweb.py"), str(out), "--val", str(val), str(a), str(b)], check=True,
                   capture_output=True)
    assert out.read_text() == f"one\n{sep}two\n{sep}three\n{sep}"
    led = json.load(open(str(out) + ".ledger.json"))
    assert (led["kept"], led["dropped_equal_val"], led["dropped_repeat"]) == (3, 1, 2)
    assert led["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_restore_never_writes_through_a_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_CHAT_DIR", str(tmp_path))
    sys.modules.pop("jobs_vm", None)
    import jobs_vm
    out = tmp_path / "pt"
    gen = out / "ckpt_gen" / "g125"
    gen.mkdir(parents=True)
    for f in ("model.safetensors", "optim.safetensors", "state.txt"):
        (gen / f).write_text("generation")
        (out / f).symlink_to(Path("ckpt_gen/live") / f)
    (out / "ckpt_gen" / "live").symlink_to("g125")
    up = tmp_path / "upload" / "pt"
    up.mkdir(parents=True)
    (up / "model.safetensors.part00").write_text("uploaded")
    (up / "state.txt").write_text("250\n{}\n")
    jobs_vm.restore("pt")
    assert (out / "model.safetensors").read_text() == "uploaded" and not (out / "model.safetensors").is_symlink()
    moved = [p for p in out.iterdir() if p.name.startswith("ckpt_gen.superseded.")]
    assert len(moved) == 1 and (moved[0] / "g125" / "model.safetensors").read_text() == "generation"
    assert not (out / "optim.safetensors").exists() and "adamw_t" not in (out / "state.txt").read_text()


def run_fake(tmp_path, jobs, scenario):
    import run_eq
    return run_eq.run(str(CC / "runner_multi.py"), str(tmp_path / "w"), scenario, jobs=jobs)


def test_failed_halting_job_never_becomes_home(tmp_path):
    jobs = {"jobs": [{"name": "data", "cmd": ["x"], "outputs": ["out.txt"]},
                     {"name": "pt", "model_name": "m", "after": "data", "relay_every_steps": 100, "spec": {}},
                     {"name": "bad", "after": "pt", "cmd": ["z"], "outputs": ["x.json"], "fake_fail": True,
                      "halt_on_failure": True}]}
    r = run_fake(tmp_path, jobs, {})
    st = r["state"]
    assert r["rc"] == 3
    assert st["jobs"]["bad"]["status"] == "FAILED" and st["jobs"]["bad"]["home"] is False
    assert "campaign_home" not in st
    marker = json.load(open(tmp_path / "w" / "state" / "HALTED"))
    assert marker["job"] == "bad" and marker["kind"] == "JOB_FAILED_STATE_KEPT"
    assert "HALTED written" in r["log"]
    assert '"stop"' in r["calls"]  # the VM ended


def test_failed_job_without_the_flag_still_goes_home(tmp_path):
    jobs = {"jobs": [{"name": "bad", "cmd": ["z"], "outputs": ["x.json"], "fake_fail": True}]}
    r = run_fake(tmp_path, jobs, {})
    assert r["rc"] == 0 and r["state"]["jobs"]["bad"]["home"] is True and r["state"].get("campaign_home") is True


def test_halted_marker_stops_a_restart(tmp_path):
    jobs = {"jobs": [{"name": "bad", "cmd": ["z"], "outputs": ["x.json"], "fake_fail": True, "halt_on_failure": True}]}
    run_fake(tmp_path, jobs, {})
    import run_eq
    r = run_eq.run(str(CC / "runner_multi.py"), str(tmp_path / "w"), {}, jobs=jobs, fresh=False)
    assert r["rc"] != 0 and "HALTED marker present" in r["stderr"]


def test_pod_loss_drill_relaunches_and_resumes_once(tmp_path):
    jobs = {"jobs": [{"name": "data", "cmd": ["x"], "outputs": ["out.txt"]},
                     {"name": "pt", "model_name": "m", "after": "data", "relay_every_steps": 100, "spec": {},
                      "pod_loss_drill_step": 100, "halt_on_failure": True}]}
    r = run_fake(tmp_path, jobs, {})
    st = r["state"]
    assert r["rc"] == 0, r["stderr"]
    assert st["pod_loss_drill"]["job"] == "pt" and st["pod_loss_drill"]["relayed_step"] >= 100
    assert st["launches"] == 2 and r["log"].count("pod loss drill") == 1
    assert st["jobs"]["pt"]["home"] is True and st.get("campaign_home") is True


class Stub:
    """The transport as the halt drill sees it."""

    def __init__(self, refuse=True, present=True):
        self.refuse, self.present, self.stopped = refuse, present, 0

    def stop(self, st):
        self.stopped += 1

    def release(self, st):
        import runpod_api
        if self.refuse:
            raise runpod_api.Refused("volume release requires all campaign jobs home and no active pod")

    def volume_present(self, st):
        return self.present


@pytest.mark.parametrize("refuse,present,ok", [(True, True, True), (False, True, False), (True, False, False)])
def test_halt_drill(tmp_path, monkeypatch, refuse, present, ok):
    import runner_multi as rm
    (tmp_path / "h").mkdir()
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(rm, "HOME", str(tmp_path / "h"))
    monkeypatch.setattr(rm, "ART", str(tmp_path / "a"))
    monkeypatch.setattr(rm, "T", Stub(refuse, present))

    (tmp_path / "a" / "s0_torn").mkdir()
    (tmp_path / "a" / "s0_torn" / "s0_torn_train.log").write_text("checkpoint: no whole generation under x; refusing to resume")

    class St:
        d = {}

        def job(self, n):
            return {"home": False, "status": "FAILED"}

    if ok:
        rm.halt_drill(St(), "s0_torn")
    else:
        with pytest.raises(SystemExit):
            rm.halt_drill(St(), "s0_torn")
        assert (tmp_path / "h" / "HALTED").exists()
    rec = json.load(open(tmp_path / "h" / "halt_drill.json"))
    assert rec["release_refused"] is refuse and rec["volume_present"] is present and rec["home_before"] is False
    assert rec["refusal_in_log"] is True and rec["status_seen"] == "FAILED"


def test_pins_sources_mode_ignores_the_mix(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    pins = {"files": {"a.txt": hashlib.sha256(b"a").hexdigest()}, "mix": {"m.ids": None}}
    assert P.check(pins, str(tmp_path), "sources")[0] == []
