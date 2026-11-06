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


def test_files_are_what_the_generator_writes():
    s0, r1 = load("tc_b1_s0.json"), load("tc_b1_r1.json")
    assert s0 == json.loads(json.dumps(M.stage0()))
    assert r1 == json.loads(json.dumps(M.b1()[0]))
    assert load("tc_b1_probe_spec.json") == json.loads(json.dumps(M.probe_spec()))


def test_b_spec_is_section_9():
    s = next(j for j in load("tc_b1_r1.json")["jobs"] if j["name"] == "pt_b1_r1")
    assert s["model_name"] == "tc_b1_r1" and s["relay_every_steps"] == 2500
    assert s["snapshot_steps"] == [10000, 20000, 30000, 40000, 50000]
    t, d, a = s["spec"]["train"], s["spec"]["data"], s["spec"]["arch"]
    assert t["steps"] * t["batch"] * d["T"] == 983040000
    assert (t["lr"], t["batch"], t["accum"], t["checkpoint_every"], t["eval_every"], t["seed"]) == (0.0005, 16, 1, 125, 1000, 1)
    assert t["lr_schedule"] == {"warmup_steps": 2000, "decay": "cosine", "min_lr_frac": 0.1}
    assert d["anneal"] == {"corpus": "/content/chat/data/mix_b1.tail.ids", "from_fraction": 0.85}
    assert a["custom"] == {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240}
    assert a["state_tokens"] == {"k": 4, "mask": "causal"}
    sfull = next(j for j in load("carry12_sfull_r1.json")["jobs"] if j["name"] == "pt_c12_Sfull_r1")
    assert sfull["spec"]["arch"] == a  # strict load from every S-lineage recipe


@pytest.mark.parametrize("name", ["tc_b1_s0.json", "tc_b1_r1.json"])
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
    assert not j["rung_b1_s30000"].get("keep_home_snapshot")


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
    shared = " ".join(f"word{i}" for i in range(20))
    val = tmp_path / "val.txt"
    val.write_text(f"<|user|>\n{shared}\n<|endoftext|>\n", encoding="utf-8")
    sft = tmp_path / "sft.jsonl"
    sft.write_text(json.dumps({"messages": [{"role": "user", "content": "only sft text " + " ".join(f"x{i}" for i in range(14))}]}) + "\n")
    src = tmp_path / "human_msc_v1" / "dialogues.jsonl"
    src.parent.mkdir()
    rows = [[{"role": "user", "content": shared}, {"role": "assistant", "content": "fine"}],
            [{"role": "user", "content": "unrelated talk here"}, {"role": "assistant", "content": "yes indeed it is"}],
            [{"role": "user", "content": "only sft text " + " ".join(f"x{i}" for i in range(14))},
             {"role": "assistant", "content": "ok"}]]
    src.write_text("".join(json.dumps({"messages": m, "uri": str(i)}) + "\n" for i, m in enumerate(rows)))
    out = tmp_path / "f"
    subprocess.run([sys.executable, str(CC / "tc_b1_filter.py"), "--repo", str(ROOT), "--heldout", str(val), str(sft),
                    "--out", str(out), "--source", f"msc={src}"], check=True, capture_output=True)
    led = json.load(open(out / "filter.ledger.json"))["sources"]["msc"]
    assert (led["rows_in"], led["rows_kept"], led["dropped_heldout_13gram"]) == (3, 1, 2)
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

    class St:
        d = {}

        def job(self, n):
            return {"home": False}

    if ok:
        rm.halt_drill(St(), "s0_torn")
    else:
        with pytest.raises(SystemExit):
            rm.halt_drill(St(), "s0_torn")
        assert (tmp_path / "h" / "HALTED").exists()
    rec = json.load(open(tmp_path / "h" / "halt_drill.json"))
    assert rec["release_refused"] is refuse and rec["volume_present"] is present and rec["home_before"] is False
