"""Base 2's jobs files: the three arms as BASE2_PLAN.md section 3 gives them, the corpus build, the rehearsal, and the
trainer they build.

No VM and no GPU. The last tests run each arm's spec for one step at T 32 through a CPU build of the trainer
(BASE2_MTSTUDIO, default build_cpu/mtstudio in this tree) and are skipped without it or without the 16k vocabulary
(PK_VOCAB16); arm C's also needs base A's final weights (BASE2_A_WEIGHTS). POSIX only."""
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX paths")

import base2_pins as P  # noqa: E402
import make_base2 as M  # noqa: E402

TRAINER = "5506e6dfd5ec630b4b49fae0592329cc06799070"   # the source base A trained on and the probe measured
MTSTUDIO = Path(os.environ.get("BASE2_MTSTUDIO", ROOT / "build_cpu" / "mtstudio"))
VOCAB = Path(os.environ.get("PK_VOCAB16", "/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"))
A_WEIGHTS = Path(os.environ.get("BASE2_A_WEIGHTS", "/mnt/c/ml_artifacts/tiny_chat/b1_r1/pt_b1_r1/tc_b1_r1.safetensors"))
ARMS = ("N", "S", "C")
D1 = {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 512, "layers": 18, "heads": 8, "d_ff": 1536},
      "state_tokens": {"k": 4, "mask": "causal"}}
A = {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
     "state_tokens": {"k": 4, "mask": "causal"}}


def load(n):
    return json.load(open(CC / n))


def jobs(arm):
    return {j["name"]: j for j in load(f"base2_{arm}.json")["jobs"]}


def train(arm):
    return jobs(arm)[f"pt_base2_{arm}"]


def test_files_are_what_the_generator_writes_and_check_sees_a_change(tmp_path, monkeypatch, capsys):
    for name, obj in M.files().items():
        assert (CC / name).read_text() == M.text(obj), name
    assert M.main(["--check"]) == 0
    assert set(M.files()) == {"base2_data.json", "base2_N.json", "base2_S.json", "base2_C.json", "base2_rehearsal.json",
                              "base2_rehearsal_b.json", "base2_rung_D1.json", "base2_load_probe.json"}
    monkeypatch.setattr(M, "HERE", str(tmp_path))
    assert M.main(["--check"]) == 1          # nothing written yet
    assert M.main([]) == 0 and M.main(["--check"]) == 0
    s = json.loads((tmp_path / "base2_S.json").read_text())
    s["jobs"][1]["spec"]["train"]["lr"] = 0.0003
    (tmp_path / "base2_S.json").write_text(M.text(s))
    assert M.main(["--check"]) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["different"] == ["base2_S.json"]


def test_parameter_counts():
    assert M.params(512, 18, 1536) == 69751296 == M.PARAMS["D1"]
    assert M.params(768, 8, 2240) == 72761088 == M.PARAMS["A"]
    assert M.params(768, 12, 2048) == 97539840          # the race's 12 x 768 arm, the count its probe recorded
    for arm in ARMS:
        assert load(f"base2_{arm}.json")["_base2"]["params"] == (72761088 if arm == "C" else 69751296)


@pytest.mark.parametrize("arm", ["N", "S"])
def test_n_and_s_are_section_3(arm):
    j = train(arm)
    s = j["spec"]
    assert s["arch"] == D1
    assert s["train"] == {"steps": 250000, "lr": 0.0005, "batch": 8, "accum": 2, "clip": 1.0, "eval_every": 1000,
                          "eval_windows": 32, "checkpoint_every": 500, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
                          "precision": "bf16", "attention_kernel": "flash",
                          "lr_schedule": {"warmup_steps": 4000, "decay": "cosine", "min_lr_frac": 0.1}}
    assert s["data"] == {"corpus": f"/content/chat/data/{arm}.main.ids", "vocab": "/content/chat/upload/vocab_llama3_bpe.gguf",
                         "vocab_cap": 16384, "T": 1024, "max_tokens": 0,
                         "anneal": {"corpus": f"/content/chat/data/{arm}.tail.ids", "from_fraction": 0.85}}
    assert "init_from" not in s["train"] and "init_file" not in j and "init_job" not in j     # from scratch
    assert j["env"] == {"MICROTORCH_CHECKPOINT_TXN": "1"} and j["store_branch"] == f"ckpt-{arm}"
    assert j["model_name"] == f"base2_{arm}" and j["halt_on_failure"]


def test_n_and_s_differ_in_the_corpus_alone():
    n, s = json.dumps(load("base2_N.json")), json.dumps(load("base2_S.json"))
    swapped = (n.replace("N.main.ids", "S.main.ids").replace("N.tail.ids", "S.tail.ids").replace("corpus/N", "corpus/S")
               .replace("base2_N", "base2_S").replace("ckpt-N", "ckpt-S").replace("arm-N", "arm-S")
               .replace("rung_N_", "rung_S_").replace("BASE2_N", "BASE2_S"))
    a, b = json.loads(swapped), json.loads(s)
    # The registration sets the arms' release turns off from each other (N 0, the runner's default and not written;
    # S 20): the one store switch in which they differ.
    assert "store_release_offset_min" not in a and b.pop("store_release_offset_min") == 20
    for d in (a, b):
        d.pop("_comment")
        d["_base2"].pop("arm")
        d["_base2"].pop("corpus")
    assert a == b


def test_c_is_base_a_continued_on_s_corpus():
    j = train("C")
    s = j["spec"]
    assert s["arch"] == A
    assert s["arch"] == json.load(open(ROOT / M.SHAPE_A))["arch"]          # base A's own shape file, read by its rungs
    b1 = next(x for x in load("tc_b1_r1.json")["jobs"] if x["name"] == "pt_b1_r1")["spec"]
    assert s["arch"] == b1["arch"] and (b1["train"]["batch"], b1["train"]["accum"]) == (16, 1)
    assert s["train"] == {"steps": 330000, "lr": 0.00025, "batch": 16, "accum": 1, "clip": 1.0, "eval_every": 1000,
                          "eval_windows": 32, "checkpoint_every": 500, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
                          "precision": "bf16", "attention_kernel": "flash",
                          "lr_schedule": {"warmup_steps": 2000, "decay": "cosine", "min_lr_frac": 0.1}}
    assert s["train"]["lr"] == b1["train"]["lr"] / 2                        # half of A's peak
    # S's corpus and S's seed. C reads S's windows only while both read the main part: S turns to its tail at step
    # 212,501 and C at 280,501 (the sampling test below shows 16 x 1 and 8 x 2 draw the same windows until then).
    assert s["data"] == train("S")["spec"]["data"] and s["train"]["seed"] == train("S")["spec"]["train"]["seed"] == 1
    meta, smeta = load("base2_C.json")["_base2"], load("base2_S.json")["_base2"]
    assert meta["same_windows_as_S_through_step"] == smeta["last_main_step"] == 212500
    assert (smeta["first_tail_step"], meta["first_tail_step"]) == (212501, 280501)
    assert "same_windows_as_S_through_step" not in smeta
    assert [m for m in M.MARKS["C"] if m <= meta["same_windows_as_S_through_step"]] == [10000, 60000, 122000]
    assert j["init_file"] == "tc_b1_r1.safetensors" and "init_from" not in s["train"]
    assert j["store_branch"] == "ckpt-C" and j["env"] == {"MICROTORCH_CHECKPOINT_TXN": "1"}
    pins = load("base2_pins.json")
    assert list(pins["a_weights"]) == [j["init_file"]] and ("a_weights", "upload") in P.MODES["arm-C"][0]


def test_every_arm_reads_sixteen_rows_a_step_and_flash_fits_its_heads():
    for arm in ARMS:
        s = train(arm)["spec"]
        assert s["train"]["batch"] * s["train"]["accum"] == 16 and s["data"]["T"] == 1024
        c = s["arch"]["custom"]
        assert c["d"] % c["heads"] == 0 and c["d"] // c["heads"] <= 64     # mtstudio: flash supports head widths up to 64


def test_marks_positions_and_the_tail():
    assert M.MARKS["N"] == M.MARKS["S"] == [10000, 60000, 122000, 244000, 250000]
    assert M.MARKS["C"] == [10000, 60000, 122000, 244000, 305000, 330000]
    pos = {10000: 163840000, 60000: 983040000, 122000: 1998848000, 244000: 3997696000, 250000: 4096000000,
           305000: 4997120000, 330000: 5406720000}
    for arm in ARMS:
        j, meta = train(arm), load(f"base2_{arm}.json")["_base2"]
        every = j["spec"]["train"]["checkpoint_every"]
        assert j["snapshot_steps"] == M.MARKS[arm] == meta["marks"]
        assert all(m % every == 0 for m in j["snapshot_steps"])
        assert j["snapshot_steps"][-1] == j["spec"]["train"]["steps"] == meta["steps"]
        assert meta["mark_positions"] == {str(m): pos[m] for m in M.MARKS[arm]}
        assert meta["positions"] == pos[meta["steps"]] and meta["positions_per_step"] == 16384
        # mtstudio reads the tail when `step > int(from_fraction * steps)`: the main part through that step.
        last_main = int(j["spec"]["data"]["anneal"]["from_fraction"] * meta["steps"])
        assert last_main == meta["last_main_step"] == {"N": 212500, "S": 212500, "C": 280500}[arm]
        assert meta["first_tail_step"] == last_main + 1
    src = (ROOT / "tools" / "mtstudio.cpp").read_text()
    assert "step > static_cast<int>(s.anneal_from * s.steps)" in src
    assert 60000 * 16384 == 983040000          # the first billion is base A's whole exposure, step for step


@pytest.mark.parametrize("name", ["base2_N.json", "base2_S.json", "base2_C.json", "base2_rehearsal.json",
                                  "base2_rehearsal_b.json", "base2_data.json"])
def test_jobs_are_well_formed(name):
    f = load(name)
    js = f["jobs"]
    names = [j["name"] for j in js]
    assert len(set(names)) == len(names)
    # One trainer and one rung at a time; a rehearsal's watcher has the second light slot.
    assert f["max_parallel"] == 1 and f["max_parallel_light"] == (2 if "rehearsal" in name else 1)
    for j in js:                                  # every rung has a time limit, whichever way it runs
        if j["name"].startswith("rung_"):
            limited = j["cmd"][:4] == ["timeout", "-k", "60", "3600"]
            assert bool(j.get("exclusive")) != limited, j["name"]
            assert (j.get("timeout_s") == 3600) == bool(j.get("exclusive"))
    for j in js:
        assert ("cmd" in j) != ("spec" in j)
        after = j.get("after") or []
        for a in [after] if isinstance(after, str) else after:
            assert a in names
        if j.get("rung_of"):
            pt = next(x for x in js if x["name"] == j["rung_of"]["job"])
            assert j["rung_of"]["step"] in pt["snapshot_steps"]
            assert j["light"] and (j.get("exclusive") or "rehearsal" in name)   # the arms' mark rungs hold training
            assert f"/content/chat/{pt['name']}/rung_{j['rung_of']['step']}/model.safetensors" in j["cmd"]
            assert "--drop-snapshot" not in j["cmd"]
        if "spec" in j:
            assert j["env"]["MICROTORCH_CHECKPOINT_TXN"] == "1" and j["halt_on_failure"], j["name"]
            assert j["store_branch"].startswith("ckpt-")
            assert j["spec"]["out_dir"] == f"/content/chat/{j['name']}"       # jobs_vm.py sets it so in any case
            assert j["spec"]["train"]["precision"] == "bf16" and j["spec"]["train"]["attention_kernel"] == "flash"
        for a in j.get("cmd", []):
            if a.startswith("tools/") or a.startswith("corpus/") or a.startswith("experiments/"):
                assert (ROOT / a).exists(), a
    branches = [j["store_branch"] for j in js if "spec" in j]
    assert len(set(branches)) == len(branches)
    for item in f["store_inputs"]:
        assert set(item) <= {"branch", "prefix", "to", "files"} and item["branch"] == "main"
        assert not item["to"].startswith("/")                              # relative to the boot's root


@pytest.mark.parametrize("arm", ARMS)
def test_an_arm_checks_its_pins_first_and_scores_every_mark(arm):
    js = jobs(arm)
    order = list(js)
    pt = f"pt_base2_{arm}"
    assert order[:2] == ["pins", pt]
    assert js["pins"]["halt_on_failure"] and js["pins"]["cmd"][2:6] == ["/content/chat/data", f"arm-{arm}", "--upload",
                                                                         "/content/chat/upload"]
    assert js[pt]["after"] == ["pins"]
    marks = M.MARKS[arm]
    assert [n for n in order if n.startswith("rung_")] == [f"rung_{arm}_s{m}" for m in marks]
    shape = M.SHAPE_A if arm == "C" else M.SHAPE_D1
    for m in marks:
        r = js[f"rung_{arm}_s{m}"]
        cmd = r["cmd"]
        at = 4 if m == marks[-1] else 0           # the end rung runs under coreutils' timeout
        assert cmd[at:at + 3] == ["python3", "tools/colab_chat/tiny_chat_run.py", "rung"]
        assert cmd[cmd.index("--spec") + 1] == shape and cmd[cmd.index("--step") + 1] == str(m)
        assert cmd[cmd.index("--job") + 1] == pt and cmd[cmd.index("--pt-dir") + 1] == f"/content/chat/{pt}"
        panels = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--panel"]
        assert panels == [f"{p}=/content/chat/data/panel_{p}.ids" for p in ("fineweb", "dialogue", "scenario")]
        assert cmd[cmd.index("--carry5-items") + 1] == "corpus/eval/carry5_v1.jsonl" and "--export-gguf" in cmd
        assert cmd[cmd.index("--score-batch-tokens") + 1] == "4096"
    # The end is the run's own last checkpoint, read once the run has ended: no snapshot to lose.
    end = js[f"rung_{arm}_s{marks[-1]}"]
    assert "rung_of" not in end and end["after"] == [pt] and not end.get("exclusive")
    assert f"/content/chat/{pt}/model.safetensors" in end["cmd"]
    # jobs_vm.py times only a rung that holds training, so the end rung carries its own limit.
    assert end["cmd"][:4] == ["timeout", "-k", "60", "3600"] and "timeout_s" not in end
    # The rung's shape is the arm's: its panels and Carry-5 load the arm's weights strictly.
    assert json.load(open(ROOT / shape))["arch"] == js[pt]["spec"]["arch"]
    assert json.load(open(ROOT / shape))["data"]["T"] == 1024


def test_the_rung_command_is_one_the_tool_takes_and_the_pins_job_runs(tmp_path):
    # Every flag of the rung command is one tiny_chat_run.py's `rung` takes (its parser is built inside main()).
    text = (CC / "tiny_chat_run.py").read_text()
    rung_parser = text[text.index('sub.add_parser("rung")'):]
    rung_parser = rung_parser[:rung_parser.index("sub.add_parser(", 10)]
    for flag in [x for x in jobs("N")["rung_N_s10000"]["cmd"] if x.startswith("--")]:
        assert f'"{flag}"' in rung_parser, flag
    # The pins job as the arm runs it: it refuses while the committed pins hold nulls, and says which.
    d = tmp_path / "data"
    d.mkdir()
    cmd = [x.replace("/content/chat/data", str(d)).replace("/content/chat/upload", str(d)).replace("{out}", str(tmp_path / "o"))
           for x in jobs("N")["pins"]["cmd"]]
    p = subprocess.run([sys.executable, *cmd[1:]], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "N/N.main.ids: missing" in p.stderr
    assert json.load(open(tmp_path / "o" / "pins.json"))["arm-N"]["differences"]


def test_store_inputs_name_what_each_file_needs():
    def sets(name):
        return {(i["prefix"], i["to"]): i.get("files") for i in load(name)["store_inputs"]}
    panels = {("corpus/panels", "chat/data"): None, ("corpus/panel_scenario", "chat/data"): ["panel_scenario.ids"]}
    vocab = {("seed/vocab", "chat/upload"): None}
    assert sets("base2_N.json") == {**vocab, ("corpus/N", "chat/data"): ["N.main.ids", "N.tail.ids"], **panels}
    assert sets("base2_S.json") == {**vocab, ("corpus/S", "chat/data"): ["S.main.ids", "S.tail.ids"], **panels}
    assert sets("base2_C.json") == {**vocab, ("seed/weights", "chat/upload"): None,
                                    ("corpus/S", "chat/data"): ["S.main.ids", "S.tail.ids"], **panels}
    assert sets("base2_data.json") == {**vocab, ("seed/scenario_engine", "chat/upload/scenario_engine"): None}
    assert sets("base2_rehearsal.json") == vocab == sets("base2_rehearsal_b.json")
    # The data script pushes the sets the arms pull, under the same names.
    text = (CC / "base2_data.sh").read_text()
    for prefix in ("corpus/N", "corpus/S", "corpus/panels", "corpus/panel_scenario", "corpus/rehearsal"):
        assert f"push_set main {prefix} " in text, prefix
    assert 'ENGINE=${BASE2_ENGINE_DIR:-/content/chat/upload/scenario_engine}' in text
    assert 'STORE=${BASE2_STORE_CONFIG:-/content/chat/store.json}' in text
    if (CC / "base2_seed.py").exists():
        seed = (CC / "base2_seed.py").read_text()
        for prefix in ("seed/weights", "seed/vocab", "seed/scenario_engine"):
            assert f'"{prefix}"' in seed, prefix
    if (CC / "hf_store.py").exists():
        store = (CC / "hf_store.py").read_text()
        assert 'json.load(open(jobs_path)).get("store_inputs", [])' in store and 'item.get("files")' in store


def test_data_file_builds_the_corpus_then_loads_it():
    js = {j["name"]: j for j in load("base2_data.json")["jobs"]}
    assert list(js) == ["data", "load_probe"]
    assert js["data"]["cmd"] == ["bash", "tools/colab_chat/base2_data.sh", "{out}", "/content/chat/upload/vocab_llama3_bpe.gguf", "all"]
    assert js["data"]["halt_on_failure"] and js["data"]["light"]
    assert {"base2_stage1.json", "base2_stage2.json", "base2_stage3.json", "fineweb_edu_shard2_v1.sha256",
            "fineweb_edu_shard5_v1.sha256"} <= set(js["data"]["outputs"])
    # The probe goes through the data script, which first pulls S's ids and the panels and checks them against the
    # store's manifests: a VM on which all three stages were skipped holds no ids.
    assert js["load_probe"]["after"] == "data" and js["load_probe"]["cmd"] == [
        "bash", "tools/colab_chat/base2_data.sh", "/content/chat/data", "/content/chat/upload/vocab_llama3_bpe.gguf",
        "loadprobe", "{out}"]
    text = (CC / "base2_data.sh").read_text()
    probe = text[text.index("\nloadprobe() {"):text.index("\nrun_stage() {")]
    assert probe.index("pull_set main corpus/S") < probe.index("PINS \"$OUT\" build-S") < probe.index("B2 loadprobe --spec $CC/base2_load_probe.json")
    # corpus/S cannot be a boot input of this file: it does not exist when the build's first VM boots.
    assert not any(i["prefix"].startswith("corpus/") for i in load("base2_data.json")["store_inputs"])
    s = load("base2_load_probe.json")
    assert s["arch"] == D1 and s["train"]["steps"] == 20 and s["export"] == {"formats": []}
    assert s["data"] == train("S")["spec"]["data"]                         # the real ids, both parts
    assert s["train"]["checkpoint_every"] > 20 and s["train"]["eval_every"] > 20 and "out_dir" not in s
    assert (s["train"]["batch"], s["train"]["accum"], s["train"]["precision"]) == (8, 2, "bf16")


@pytest.mark.parametrize("name,job,drill,mark,held", [("base2_rehearsal.json", "pt_reh", 400, 200, False),
                                                      ("base2_rehearsal_b.json", "pt_reh_b", 1200, 1800, True)])
def test_rehearsal_loses_a_session_on_each_side_of_the_tail_switch(name, job, drill, mark, held):
    f = load(name)
    js = {j["name"]: j for j in f["jobs"]}
    rung = next(n for n in js if n.startswith("rung_"))
    assert list(js) == ["data", "watch", job, rung, f"{job}_control"]
    assert js["data"]["cmd"][-1] == "rehearsal" and js[job]["after"] == ["data"]
    s = js[job]["spec"]
    t = s["train"]
    every = t["checkpoint_every"]
    assert s["arch"] == D1 and s["data"]["T"] == 1024
    assert (t["steps"], every, t["batch"], t["accum"], t["precision"], t["attention_kernel"]) == \
        (2000, 200, 8, 2, "bf16", "flash")
    assert f["_base2"]["checkpoint_every"] == every == M.REH_EVERY == t["eval_every"]   # each mark has its validation loss
    # A main and a tail corpus of different sizes; the main part through step 1,000, the tail from 1,001.
    assert s["data"]["corpus"] == "/content/chat/data/rehearsal.main.ids"
    assert s["data"]["anneal"] == {"corpus": "/content/chat/data/rehearsal.tail.ids", "from_fraction": 0.5}
    last_main = int(0.5 * 2000)
    assert f["_base2"]["last_main_step"] == last_main == 1000 and f["_base2"]["first_tail_step"] == 1001
    # The forced stop. The runner fires it at its first tick after the store holds the drill step's generation. At
    # the probe's 1.053 s a step (PROBE_RESULT.md) a tick of 240 s is 228 steps, and a push that takes up to two
    # minutes is 114 more: the trainer is then past the drill step and at most `late` steps past it.
    assert js[job]["pod_loss_drill_step"] == drill == f["_base2"]["pod_loss_drill_step"]
    assert drill % every == 0 and drill in range(every, 2000, every)       # a step the trainer saves, so one the store gets
    late = int((M.TICK_S + 120) / M.STEP_S) + 1
    assert late == 342 and M.STEP_S == 1.053 and M.TICK_S == 240
    assert drill * M.STEP_S > M.TICK_S               # the run is still going when the first tick after it can fire
    # Between pushes: the next save is `every` steps on and the stop can come at any step of that interval and the
    # next; the trainer is above the pushed step from the first step after the save.
    if drill < last_main:
        assert drill + late < last_main              # the stop, and the generation resumed from, before the switch
    else:
        assert drill > last_main and drill + late < 2000   # after it, and steps left to resume into
    assert sum("pod_loss_drill_step" in j for j in f["jobs"]) == 1     # the runner keeps one drill a state folder
    # The mark and its rung. Part a's rung runs beside training, so the next VM scores the mark from the store's
    # copy when the stop finds it unfinished; part b's holds training as the arms' rungs do, after the resume.
    assert js[job]["snapshot_steps"] == [mark] and js[rung]["rung_of"] == {"job": job, "step": mark}
    assert mark % every == 0                         # jobs_vm.take_snapshot copies a checkpoint: a mark is a save step
    assert bool(js[rung].get("exclusive")) == held and js[rung]["after"] == "data" and M.SHAPE_D1 in js[rung]["cmd"]
    if not held:
        assert mark < drill                          # taken before the stop
    else:
        assert mark > drill + late                   # taken by the VM that resumed
    # The control: the same 2,000 steps, and not before the interrupted run and its rung have both ended.
    c = js[f"{job}_control"]
    assert c["after"] == [job, rung] and c["after_terminal"] and c["store_branch"] != js[job]["store_branch"]
    same = dict(c["spec"], name=s["name"], out_dir=s["out_dir"])
    assert same == s
    assert "snapshot_steps" not in c and "pod_loss_drill_step" not in c
    # Both training jobs export weights only; the runner fetches what the spec declares (runner_multi.fetch_final).
    assert s["export"] == {"formats": ["safetensors"]} == c["spec"]["export"]
    # The watcher: the training step against the courier's pushed step every 30 s, in a file on the VM. It never
    # opens the store (its commits, 120 an hour, used the repository's whole limit in part a's first run). It ends
    # itself after two hours, and coreutils' timeout five minutes later whatever it is doing.
    w = js["watch"]["cmd"]
    assert w[:4] == ["timeout", "-k", "30", "7500"] and w[6] == "drill-watch"
    assert w[w.index("--job") + 1] == job and w[w.index("--interval") + 1] == "30"
    assert w[w.index("--limit-s") + 1] == "7200" and "--push-limit-s" not in w and "--config" not in w
    assert not any("store.json" in x for x in w) and not js["watch"].get("halt_on_failure")
    assert js["watch"]["outputs"] == ["watch_summary.json", "watch.jsonl"]     # home with the job's outputs


def test_the_storage_guard_fits_the_rehearsal_and_the_arms_registered_caps():
    """The courier stops pushing once the bytes counted for the campaign reach the cap. Counted are what the store
    held at the last release from home and what was pushed since. A release leaves what a ledger younger than its
    grace period names, so a removed generation stays stored, and counted, for up to the grace and one release
    interval; on top come what the branches keep (at the most FOUR generations each: two verified and the newest
    between two saves, and one more for the save after a session is replaced, when the new VM has verified only
    the generation it restored; the rerun of 12 Nov 2026 held four and removed two in one commit), the mark, and
    what is pushed before the next measurement. The save interval keeps that under the cap; half the interval
    does not."""
    for name in ("base2_rehearsal.json", "base2_rehearsal_b.json"):
        f = load(name)
        assert f["store_push_cap_gb"] == 28 == M.REH_PUSH_CAP_GB
        assert f["_base2"]["store_release_grace_s"] == 600 == M.REH_RELEASE_GRACE_S
        assert f["store_release_every_min"] == 20 == M.REH_RELEASE_EVERY_MIN
    import hf_store
    assert M.REH_RELEASE_GRACE_S >= hf_store.MIN_RELEASE_GRACE_S          # the store refuses a shorter one
    gen_gb = 3 * M.PARAMS["D1"] * 4 / 1e9             # weights and two AdamW moments, 4 bytes a parameter
    assert 0.83 < gen_gb < 0.85

    def counted(every):                               # GB, at the most, that the guard can count in the rehearsal
        save_s, release_s = every * M.STEP_S, M.REH_RELEASE_EVERY_MIN * 60
        kept = 2 * M.gens_kept_max() * gen_gb + gen_gb / 3   # five generations on each of the two branches, and the mark
        removed = (int((M.REH_RELEASE_GRACE_S + release_s) / save_s) + 1) * gen_gb    # removed, not yet deletable
        since = (int(release_s / save_s) + 2) * gen_gb                               # pushed since the measurement,
        return kept + removed + since                                                # one of them counted twice
    # The rule (the reviews of the arms' round): three, plus one for each session loss between two verified saves.
    # Four after one loss, as the rerun saw; five after two in a row, which is what the sums take. Not a bound:
    # while no check reads whole nothing is removed at all, and only the cap or the runner's halt ends that.
    assert (M.gens_kept_max(0), M.gens_kept_max(1), M.gens_kept_max(2), M.gens_kept_max()) == (3, 4, 5, 5) and M.ARM_SESSION_LOSSES == 2
    assert 22 < counted(M.REH_EVERY) < 22.2
    assert counted(M.REH_EVERY) < 0.8 * M.REH_PUSH_CAP_GB
    assert counted(100) > M.REH_PUSH_CAP_GB
    # The same sum, as make_base2.push_cap_gb, for one branch; and what it suggests for the arms, whose files carry
    # 0 until the lead sets it: kept + removed-and-not-yet-deletable within the grace + pushed since the release,
    # and every mark, held under 0.8 of the cap.
    # The second reviews of the arms' round: the guard's count is what the last release that RAN measured plus all
    # pushed since. Removed and still stored at that release: what is younger than the grace PLUS the release's
    # 120 s skew. Pushed since: through ARM_MISSED_RELEASES (2) turns in a row that did not run.
    assert M.RELEASE_SKEW_S == hf_store.RELEASE_SKEW_S == 120 and M.ARM_MISSED_RELEASES == 2
    one = M.push_cap_gb("D1", M.REH_EVERY, M.STEP_S, 1, M.REH_RELEASE_EVERY_MIN, M.REH_RELEASE_GRACE_S)
    assert (one["kept"], one["removed"], one["since"], one["marks"]) == (5, 4, 19, 1)
    assert M.push_cap_gb("D1", M.REH_EVERY, M.STEP_S, 1, M.REH_RELEASE_EVERY_MIN, M.REH_RELEASE_GRACE_S, missed_releases=0)["since"] == 7
    r = M.rates()
    assert r["generations_kept"] == {"rule": "three plus one for each session loss between two verified saves", "losses_taken": 2, "kept": 5}
    # The registered settings (12 Nov 2026): a release every 60 minutes, the grace of 7,200 s, offsets 0, 20, 40.
    assert r["settings"]["C"] == {"release_every_min": 60, "release_grace_s": 7200, "release_offset_min": 40} and r["missed_releases_taken"] == 2
    caps = r["push_cap_sum"]
    assert {a: (c["kept"], c["removed"], c["since"], c["marks"]) for a, c in caps.items()} == {
        "N": (5, 14, 22, 5), "S": (5, 14, 22, 5), "C": (5, 19, 29, 6)}
    assert {a: (c["gen_gb"], c["counted_gb"]) for a, c in caps.items()} == {"N": (0.837, 35.7), "S": (0.837, 35.7), "C": (0.873, 48.0)}
    assert r["suggested_store_push_cap_gb"] == {"N": 45, "S": 45, "C": 60} and (r["counted_gb_three_arms"], r["caps_gb_three_arms"]) == (119.4, 150)
    # The second launch review (Explore B1): at the sums' caps a home outage of 4 to 5 hours capped every arm (only
    # a release from home lowers a courier's count). The lead's caps are 200, 200 and 250 GB: an outage of about 31
    # hours (N, S) or 27 (C) at the worst, 650 GB in all against the account's 1 TB. Printed by --rates.
    assert {a: load(f"base2_{a}.json")["store_push_cap_gb"] for a in ARMS} == r["registered_store_push_cap_gb"] == {
        "N": 200, "S": 200, "C": 250}
    assert r["registered_caps_gb_three_arms"] == 650 < r["account_storage_gb"] == 1000
    out_h = r["home_outage_hours_to_cap"]
    assert {a: (o["at_release_gb"], o["gb_an_hour"], o["hours_to_cap"], o["hours_to_cap_worst"]) for a, o in out_h.items()} == {
        "N": (15.6, 5.72, 32.2, 31.2), "S": (15.6, 5.72, 32.2, 31.2), "C": (21.0, 8.03, 28.5, 27.5)}
    # Explore's arithmetic at the old caps: about 5 hours for N and 4.9 for C from the last release.
    old = M.home_outage_hours("D1", 500, M.STEP_S, 5, 60, 7200, 45)
    assert 5.0 < old["hours_to_cap"] < 5.2 and 4.8 < M.home_outage_hours("A", 500, M.STEP_S_A, 6, 60, 7200, 60)["hours_to_cap"] < 5.0
    for a, c in caps.items():
        assert abs(c["counted_gb"] - ((c["kept"] + c["removed"] + c["since"]) * c["gen_gb"] + c["marks"] * c["gen_gb"] / 3)) < 0.1
        assert c["counted_gb"] <= M.PUSH_CAP_HEADROOM * c["suggested_cap_gb"] < c["counted_gb"] + 1
    # Other settings are read without editing a file: make_base2.py --rates --release-every-min M --grace-s S.
    assert M.rates(60, 7200)["suggested_store_push_cap_gb"] == {"N": 45, "S": 45, "C": 60}
    assert M.rates(60, 3600)["suggested_store_push_cap_gb"] == {"N": 39, "S": 39, "C": 51}
    assert M.rates(30, 1800)["suggested_store_push_cap_gb"] == {"N": 24, "S": 24, "C": 30}
    assert M.rates(120, 7200, missed_releases=0, losses=1)["suggested_store_push_cap_gb"]["N"] < 40
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert M.main(["--rates", "--release-every-min", "60", "--grace-s", "3600"]) == 0
    got = json.loads(out.getvalue())
    assert got["settings"]["N"] == {"release_every_min": 60.0, "release_grace_s": 3600.0, "release_offset_min": 0} and got["caps_gb_three_arms"] == 129
    assert got["commits_an_hour_three_arms"]["steady_state_average"] == 29.1
    runner = (CC / "runner_multi.py").read_text()
    assert '"--store-release-grace-s", type=float, default=None' in runner and 'doc.get("store_release_grace_s") or 7200' in runner


def test_the_registrations_store_switches_are_set_in_make_base2_and_the_files_do_not_move_until_one_does():
    """The second reviews of the arms' round: the arms' cap, release interval, grace and halts are the
    registration's to set, and a value edited into a jobs file by hand fails --check. They are parameters of
    arm(): ARM_STORE for all arms, ARM_STORE_BY_ARM for one. A switch that is the runner's own default is not
    written; runner_multi.py reads each key when its flag is absent. The values are the lead's registration (12 Nov
    2026): a release every 60 minutes with the default grace of 7,200 s; offsets 0, 20, 40; caps 200, 200, 250 GB
    (after the second launch review); the stall halt on at 120 minutes; the unverified halt at 90; the no-progress
    halt at 300 (260 before the sixth launch review)."""
    import hf_store
    assert M.ARM_STORE == {"store_push_cap_gb": 0, "store_release_every_min": 60, "store_release_grace_s": 7200,
                           "store_stall_halt": True, "store_stall_min": 120, "store_unverified_halt_min": 90,
                           "store_release_offset_min": 0, "store_no_progress_min": 300, "store_no_progress_launches": 4}
    assert M.ARM_STORE_BY_ARM == {"N": {"store_push_cap_gb": 200, "store_release_offset_min": 0},
                                  "S": {"store_push_cap_gb": 200, "store_release_offset_min": 20},
                                  "C": {"store_push_cap_gb": 250, "store_release_offset_min": 40}}
    # Each file carries them, the grace (the runner's own default) and N's offset of 0 left out.
    for arm, cap, off in (("N", 200, None), ("S", 200, 20), ("C", 250, 40)):
        f = load(f"base2_{arm}.json")
        assert (f["store_push_cap_gb"], f["store_release_every_min"], f["store_stall_halt"], f["store_stall_min"],
                f["store_unverified_halt_min"], f["store_no_progress_min"], f.get("store_release_offset_min"),
                f["store_no_progress_launches"]) == (cap, 60, True, 120, 90, 300, off, 4)
        assert "store_release_grace_s" not in f
    # The registered no-progress limits (the fifth launch review's counters): 300 minutes in which Colab lists the
    # session, and the 4th launch, without a new generation; any new generation resets both. The sixth review's
    # slow restore: boot 9, a held rung 65, a gap of two ticks 8, a save interval 75, a second rung 65, boot 9, and
    # the courier's documented waits (15 for its budget, 30 for a 429): 276, under 300.
    assert 9 + 65 + 8 + 75 + 65 + 9 == 231 and 231 + 15 + 30 == 276 < M.ARM_STORE["store_no_progress_min"]
    assert M.arm("N", store={"store_no_progress_launches": 5})["store_no_progress_launches"] == 5
    assert 'doc.get("store_no_progress_launches")' in (CC / "runner_multi.py").read_text()
    keys = ("store_release_grace_s", "store_stall_halt", "store_stall_min", "store_unverified_halt_min", "store_release_offset_min",
            "store_no_progress_min", "store_no_progress_launches")
    # The no-progress halt (the fourth reviews): a switch like the others; 0 leaves the runner's own default, which
    # with the stall halt on is the stall's minutes, the boot limit and 15 more: 210 at a stall of 120.
    assert M.arm("N", store={"store_no_progress_min": 240})["store_no_progress_min"] == 240
    src = (CC / "runner_multi.py").read_text()
    assert 'doc.get("store_no_progress_min")' in src and "stall_min + a.store_boot_limit_min + 15" in src
    assert '"--store-boot-limit-min", type=float, default=75' in src and 120 + 75 + 15 == 210
    # The release offset (the third reviews): a per-arm switch like the cap; the runner reads it from the jobs file.
    assert M.arm("S", store={"store_release_offset_min": 20})["store_release_offset_min"] == 20
    assert 'doc.get("store_release_offset_min")' in (CC / "runner_multi.py").read_text()
    for arm in ARMS:
        assert M.arm(arm) == load(f"base2_{arm}.json")                                                  # --check stays equal
        assert set(keys) - set(M.arm(arm)) == {"store_release_grace_s", *(["store_release_offset_min"] if arm == "N" else [])}
    assert M.main(["--check"]) == 0
    set_ = {"store_push_cap_gb": 45, "store_release_every_min": 60, "store_release_grace_s": 3600, "store_stall_halt": True,
            "store_stall_min": 120, "store_unverified_halt_min": 60}
    doc = M.arm("N", store=set_)
    assert {k: doc[k] for k in set_} == set_ and doc["jobs"] == M.arm("N")["jobs"]
    assert list(doc).index("store_release_grace_s") == list(doc).index("store_release_every_min") + 1    # beside the others
    with pytest.raises(ValueError, match="unknown store switch"):
        M.arm("N", store={"store_push_cap": 45})
    was = dict(M.ARM_STORE_BY_ARM["C"])
    try:                                                 # the per-arm table is what a cap is set in
        M.ARM_STORE_BY_ARM["C"] = {**was, "store_push_cap_gb": 70}
        assert M.arm("C")["store_push_cap_gb"] == 70 and M.arm("N")["store_push_cap_gb"] == 200
        assert M.rates()["settings"]["C"]["release_every_min"] == 60
    finally:
        M.ARM_STORE_BY_ARM["C"] = was
    # The runner reads every one of them from the jobs file when its flag is absent.
    runner = (CC / "runner_multi.py").read_text()
    for key in ("store_push_cap_gb", "store_release_every_min", "store_release_grace_s", "store_stall_halt", "store_stall_min",
                "store_unverified_halt_min"):
        assert f'doc.get("{key}"' in runner, key
    assert '"--store-stall-min", type=float, default=None' in runner
    import ckpt_courier
    for doc in (hf_store.__doc__, ckpt_courier.__doc__, hf_store.Store.push_generation.__doc__):
        assert "four" in doc.lower()                                     # the docstrings say four, not only three
    # The rehearsal's files are not the arms': they keep their own settings.
    for name in ("base2_rehearsal.json", "base2_rehearsal_b.json"):
        f = load(name)
        assert (f["store_push_cap_gb"], f["store_release_every_min"]) == (28, 20) and not any(k in f for k in keys)


def test_three_arms_on_one_repo_stay_far_under_the_hubs_commit_limit():
    """The hub refuses commits past 128 an hour for a repository, every writer together (the rehearsal of
    12 Nov 2026), and the three arms share one repo for four days. As first built an arm spent about five commits
    a save (push, prune, lock, release, unlock): about 105 an hour for the three. Now: one commit a save, one a
    mark, and two for each release an arm's runner makes from home. Computed here from the arms' jobs files and
    the probe's step times, and held under 40 an hour."""
    step_s = {"N": M.STEP_S, "S": M.STEP_S, "C": M.STEP_S_A}
    assert (M.STEP_S, M.STEP_S_A) == (1.053, 0.783)                    # PROBE_RESULT.md: D1 at 8 x 2, A at 16 x 1, mixed
    per = {}
    for arm in ARMS:
        f = load(f"base2_{arm}.json")
        assert f["store_release_every_min"] == M.ARM_RELEASE_EVERY_MIN == 60      # registered 12 Nov 2026 (was 120)
        assert sum("spec" in j for j in f["jobs"]) == 1 and f["max_parallel"] == 1       # one training job a file
        t = train(arm)["spec"]["train"]
        per[arm] = M.commits_per_hour(f, step_s[arm], f["store_release_every_min"])
        save_min = t["checkpoint_every"] * step_s[arm] / 60
        assert abs(per[arm]["generations"] - 60 / save_min) < 1e-9
        assert per[arm]["release"] == 2.0                              # a lock and its removal every hour
        assert per[arm]["marks"] < 0.1                                 # five or six marks in about three days
    assert 8.7 < 500 * 1.053 / 60 < 8.8 and 6.5 < 500 * 0.783 / 60 < 6.6   # a save every 8.8 minutes, and every 6.5
    total = sum(p["total"] for p in per.values())
    assert round(total, 1) == 29.1, total
    assert total <= M.ARMS_COMMITS_PER_HOUR_MAX == 40 < M.HUB_COMMITS_PER_HOUR == 128
    # As first built: five commits a save and one a mark.
    old = sum(5 * p["generations"] + p["marks"] for p in per.values())
    assert 100 < old < 128 and old > 2.5 * M.ARMS_COMMITS_PER_HOUR_MAX
    # The 29.1 is an average of the planned work, not a bound, and its docstring says what it leaves out. A bad
    # hour, everything at once for each of the three arms: every save and mark sent twice; the session lost once,
    # so that its new VM creates the branch again (tried twice); and home's worst hour, a timed release and the
    # release at the campaign's end, each a lock and an unlock sent four times (5 + 5).
    doc = " ".join(M.commits_per_hour.__doc__.split())
    for left_out in ("not a bound", "sent again after a refusal", "creation of a branch", "unlock tried again",
                     "campaign's end", "data build", "seed", "by hand", "probe's step times"):
        assert left_out in doc, left_out
    worst = sum(M.commits_per_hour_bad_model(load(f"base2_{arm}.json"), step_s[arm], M.ARM_RELEASE_EVERY_MIN) for arm in ARMS)
    saves = sum(p["generations"] + p["marks"] for p in per.values())
    # An unlock is sent at most twice (it was four times until the third reviews of the arms' round): a release is
    # three commits at the worst.
    import hf_store
    assert M.UNLOCK_TRIES == hf_store.UNLOCK_TRIES == 2
    assert round(saves, 1) == 23.1 and M.HOME_RELEASE_COMMITS_WORST == 3 and M.home_commits_worst(120) == 6
    assert abs(worst - (2 * saves + 3 * 2 + 3 * 6)) < 1e-9 and round(worst, 1) == 70.2
    # 70.2 is a MODELLED BAD HOUR FROM AVERAGE RATES (fractional saves and marks), and 96 is this test's threshold
    # for it: no code enforces either. What the code enforces is each writer's ceiling, 108 summed, which an
    # essential commit passes after 15 minutes; only the hub's 429 bounds the repository.
    assert worst < M.ARMS_BAD_HOUR_TEST_THRESHOLD == 96 and not hasattr(M, "ARMS_COMMITS_PER_HOUR_WORST")
    assert "MODELLED BAD HOUR FROM AVERAGE RATES, not a worst rolling hour" in " ".join(M.commits_per_hour_bad_model.__doc__.split())
    # A ROLLING HOUR in whole commits, everything at once (the reviews of the arms' round): the saves that fit in
    # an hour rounded up (7, 7, 10) and one mark an arm, each sent twice; the branch created again, tried twice; at
    # home a timed release, the end's release and the one more round it is allowed when a ledger did not read
    # (three end releases in the one hour), each a lock and an unlock sent four times.
    # Home's part as the runner now stands (the second reviews): a timed turn and its ONE early retry when its
    # round failed or met an unread ledger (a held round sends nothing), never past the budget; and the two rounds
    # that are all the end's release makes when a ledger does not read, counted across every start.
    roll = {arm: M.commits_rolling_hour(load(f"base2_{arm}.json"), step_s[arm], M.ARM_RELEASE_EVERY_MIN) for arm in ARMS}
    assert {a: (x["saves"], x["marks"], x["sent_twice"], x["branch"], x["home"], x["total"]) for a, x in roll.items()} == {
        "N": (7, 1, 16, 2, 16, 34), "S": (7, 1, 16, 2, 16, 34), "C": (10, 1, 22, 2, 16, 40)}
    assert all((x["home_timed_cap_by_the_budget"], x["home_final_rounds"]) == (10, 6) for x in roll.values())   # printed
    # The fourth reviews of the arms' round: the model counted a runner's timed commits by its turns (a turn and its
    # early retry: 6), and with the turns anchored a late round, its retry, the next turn and that turn's retry can
    # share an hour. It now takes what the BUDGET allows, which a count of turns cannot understate: a timed lock is
    # admitted while the hour's count is at most 12 - 3 - 2 = 7, and its round sends at most three: 10. With the
    # end's two rounds: 16, whatever the interval and the offsets.
    assert M.HOME_BUDGET_RESERVE == 3 == hf_store.CommitBudget(M.HOME_COMMIT_BUDGET).reserve and M.home_timed_cap() == 10
    assert (M.home_commits_rolling(120), M.home_commits_rolling(60), M.home_commits_rolling(30)) == (16, 16, 16)
    assert M.home_commits_rolling(120, unlock_tries=1) == 13 and M.FINAL_LEDGER_ROUNDS == 2
    runner = (CC / "runner_multi.py").read_text()
    assert "STORE_FINAL_LEDGER_ROUNDS = 2" in runner and "STORE_RELEASE_RETRY_S = 600" in runner
    rolling = sum(x["total"] for x in roll.values())
    assert rolling == 2 * (7 + 7 + 10 + 3) + 3 * 2 + 3 * 16 == 108 < M.HUB_COMMITS_PER_HOUR
    # 108 of the hub's 128 is the hour in which every save is sent twice, every arm loses a session, and every
    # runner's home sends all its budget allows its timed releases and both of its end's rounds. Equal to the summed
    # ceilings (108); only the hub's 429 bounds the repository.
    assert M.rates()["commits_an_hour_three_arms"] == {
        "steady_state_average": 29.1, "modelled_bad_hour_from_average_rates": 70.2,
        "rolling_hour_whole_commits_everything_at_once": 108, "rolling_hour_without_the_offsets": 108,
        "rolling_hour_if_every_unlock_lands_at_once": 99, "timed_release_turns_in_one_hour": 3,
        "per_writer_ceilings_summed": 108, "hub_limit": 128}
    assert M.rates(60, 7200)["commits_an_hour_three_arms"]["steady_state_average"] == 29.1
    # The arms' release turns set off from each other (0, a third and two thirds of the interval; anchored, so the
    # offsets last): at two hours only two arms have a turn in any one hour; at one hour every arm still has one,
    # so the hour's figure does not move, and what the offsets buy there is rounds that do not meet each other's lock.
    assert (M.timed_turns_in_an_hour(120, [0, 0, 0]), M.timed_turns_in_an_hour(120, [0, 40, 80])) == (3, 2)
    assert (M.timed_turns_in_an_hour(60, [0, 0, 0]), M.timed_turns_in_an_hour(60, [0, 20, 40])) == (3, 3)
    c120 = M.rates(120, 7200, offsets_min=[0, 40, 80])["commits_an_hour_three_arms"]
    assert (c120["rolling_hour_whole_commits_everything_at_once"], c120["rolling_hour_without_the_offsets"]) == (108, 108)
    assert c120["timed_release_turns_in_one_hour"] == 2          # what the offsets change: the turns that share an hour
    for grace in (7200, 3600):
        c60 = M.rates(60, grace, offsets_min=[0, 20, 40])
        assert c60["settings"]["S"] == {"release_every_min": 60, "release_grace_s": grace, "release_offset_min": 20}
        assert (c60["commits_an_hour_three_arms"]["rolling_hour_whole_commits_everything_at_once"],
                c60["commits_an_hour_three_arms"]["rolling_hour_without_the_offsets"]) == (108, 108)
    import io
    import contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert M.main(["--rates", "--release-every-min", "120", "--release-offsets-min", "0,40,80"]) == 0   # nothing is written
    assert json.loads(out.getvalue())["commits_an_hour_three_arms"]["rolling_hour_whole_commits_everything_at_once"] == 108
    # Each writer's own ceiling (per writer, not per repository): the defaults of the courier and of
    # runner_multi.py. Summed over three arms they stay under the hub's limit, and the steady state is inside each.
    # Home's budget was 8 until the rerun of 12 Nov 2026, where the release at the campaign's end found no room.
    import ckpt_courier
    assert ckpt_courier.COMMIT_BUDGET == M.COURIER_COMMIT_BUDGET == 24 and M.HOME_COMMIT_BUDGET == 12
    runner = (CC / "runner_multi.py").read_text()
    assert '"--store-commit-budget-per-hour", type=int, default=24' in runner
    assert '"--store-home-commit-budget-per-hour", type=int, default=12' in runner
    assert 3 * M.COURIER_COMMIT_BUDGET + 3 * M.HOME_COMMIT_BUDGET == 108 < M.HUB_COMMITS_PER_HOUR
    assert all(p["generations"] + p["marks"] < M.COURIER_COMMIT_BUDGET / 2 for p in per.values())
    assert all(p["release"] <= M.HOME_COMMIT_BUDGET * 3 / 4 for p in per.values())
    # One runner's worst hour at home fits its budget, and a timed lock (nine of the twelve) leaves the end's
    # release its two.
    assert M.home_commits_worst(M.ARM_RELEASE_EVERY_MIN) <= M.HOME_COMMIT_BUDGET
    # The rehearsal, alone on the repo: a save every 3.5 minutes and a release every twenty: six commits an hour
    # of the nine the runner's budget lets a timed lock have (a quarter is kept back), and the end's two on top.
    for name in ("base2_rehearsal.json", "base2_rehearsal_b.json"):
        r = M.commits_per_hour(load(name), M.STEP_S, M.REH_RELEASE_EVERY_MIN)
        assert 17 < r["generations"] < 17.2 and r["release"] == 6 and r["total"] < 26
        assert r["generations"] + r["marks"] < M.COURIER_COMMIT_BUDGET and r["release"] <= M.HOME_COMMIT_BUDGET * 3 / 4
        assert r["release"] + 2 <= M.HOME_COMMIT_BUDGET


def test_the_two_rehearsals_run_one_after_the_other_and_share_only_the_corpus():
    """Part a to its end, then part b: storage release runs with a grace of 1,800 s, which is safe with one writer on
    the store and not with two. In the store they share the corpus, main:corpus/rehearsal, which part a builds and
    part b finds and pulls (tests/test_base2_data.py runs both through base2_data.sh); nothing else: their training
    jobs, and so their marks, and their branches are apart, and apart from the arms'."""
    a, b = load("base2_rehearsal.json"), load("base2_rehearsal_b.json")
    assert a["_base2"]["after"] is None and b["_base2"]["after"] == "base2_rehearsal.json"
    for f in (a, b):
        assert f["_base2"]["runs"] == "alone on the store: part a to its end, then part b"
        assert f["_base2"]["shared_with_the_other_part"] == ["main:corpus/rehearsal"]
        data = next(j for j in f["jobs"] if j["name"] == "data")
        assert data["cmd"] == ["bash", "tools/colab_chat/base2_data.sh", "{out}", M.V, "rehearsal"]
    sh = (CC / "base2_data.sh").read_text()
    reh = sh[sh.index("\nrehearsal() {"):sh.index("\nloadprobe() {")]
    assert "if in_store main corpus/rehearsal; then" in reh and "pull_set main corpus/rehearsal" in reh
    assert "push_set main corpus/rehearsal" in reh
    ja, jb = {j["name"] for j in a["jobs"] if "spec" in j}, {j["name"] for j in b["jobs"] if "spec" in j}
    assert not ja & jb                               # marks are kept by job name on the store's main branch
    ba, bb = {j["store_branch"] for j in a["jobs"] if "spec" in j}, {j["store_branch"] for j in b["jobs"] if "spec" in j}
    assert not ba & bb and not (ba | bb) & {"ckpt-N", "ckpt-S", "ckpt-C"}
    sa = next(j for j in a["jobs"] if j["name"] == "pt_reh")["spec"]
    sb = next(j for j in b["jobs"] if j["name"] == "pt_reh_b")["spec"]
    assert dict(sb, name=sa["name"], out_dir=sa["out_dir"]) == sa      # one recipe, stopped at two places
    # The watcher's records are apart too: watch/<job>/<VM start>.
    assert a["_base2"]["watch"] != b["_base2"]["watch"]


def test_drill_watch_records_live_against_pushed(tmp_path):
    import base2_corpus as B
    if not (CC / "hf_store.py").exists():
        pytest.skip("hf_store.py is the store agent's file")
    chat = tmp_path / "chat"
    gen = chat / "pt_reh" / "ckpt_gen"
    (gen / "g350").mkdir(parents=True)
    os.symlink("g350", gen / "live")
    (chat / "pt_reh" / "state.txt").write_text("350\n{}\n")
    # The trainer's own event stream: it saved at 350 and is at step 387 now. A torn last line is skipped.
    with open(chat / "pt_reh" / "events.jsonl", "w") as f:
        for s in range(340, 388):
            f.write(json.dumps({"event": "step", "step": s, "loss": 3.0}) + "\n")
            if s == 350:
                f.write(json.dumps({"event": "checkpoint", "step": 350, "bytes": 1}) + "\n")
        f.write(json.dumps({"event": "eval", "step": 400, "val_loss": 3.0}) + "\n")      # not a training step
        f.write('{"event": "step", "step": 38')
    (chat / "courier_status.json").write_text(json.dumps({"jobs": {"pt_reh": {"branch": "ckpt-R", "pushed_step": 300}}}))
    (chat / "STATUS_pt_reh").write_text("FINISHED\n")
    cfg = tmp_path / "store.json"
    cfg.write_text(json.dumps({"repo": f"dir:{tmp_path / 'store'}"}))
    assert B.main(["drill-watch", "--config", str(cfg), "--chat", str(chat), "--job", "pt_reh", "--out", str(tmp_path / "o"),
                   "--interval", "0"]) == 0
    s = json.load(open(tmp_path / "o" / "watch_summary.json"))
    assert (s["last"]["live_step"], s["last"]["pushed_step"], s["last"]["live_minus_pushed"]) == (350, 300, 50)
    # The training step is the trainer's own, from its events, not a saved generation's.
    assert (s["last"]["training_step"], s["last"]["training_minus_pushed"]) == (387, 87)
    assert abs(s["last"]["training_step_at"] - os.path.getmtime(chat / "pt_reh" / "events.jsonl")) < 1e-3
    # The record is the watch job's own file on the VM, one line a sample; the runner brings it home with the job's
    # outputs. NOTHING goes to the store: a commit every 30 s is 120 an hour, and the hub allows the repository 128.
    assert s["store_commits"] == 0 and s["samples"] == 1 and "prefix" not in s
    lines = [json.loads(ln) for ln in open(tmp_path / "o" / "watch.jsonl")]
    assert len(lines) == 1 and lines[0] == s["last"] and lines[0]["status"] == "FINISHED"
    assert lines[0]["live_step"] == 350 and lines[0]["pushed_step"] == 300 and lines[0]["training_step"] == 387
    assert not (tmp_path / "store").exists()                         # the store named by --config was never opened
    # No courier yet, no checkpoint yet, no event yet: recorded as unknown, never guessed.
    none = B.watch_sample(str(tmp_path / "nowhere"), "pt_reh")
    assert none["live_minus_pushed"] is None and none["training_step"] is None and none["training_minus_pushed"] is None
    # The watcher cannot commit: it does not import the store, whatever a jobs file of before passes it.
    import hf_store
    real = hf_store.Store.push_set
    hf_store.Store.push_set = lambda self, *a, **kw: pytest.fail("the watcher pushed")
    hf_store.Store.commit, real_commit = (lambda self, *a, **kw: pytest.fail("the watcher committed")), hf_store.Store.commit
    try:
        (chat / "STATUS_pt_reh").write_text("RUNNING\n")             # three samples, then its own time limit ends it
        assert B.main(["drill-watch", "--config", str(cfg), "--chat", str(chat), "--job", "pt_reh", "--out", str(tmp_path / "o2"),
                       "--interval", "0.2", "--limit-s", "0.5", "--push-limit-s", "1"]) == 0
        assert B.main(["drill-watch", "--chat", str(chat), "--job", "pt_reh", "--out", str(tmp_path / "o3"),
                       "--interval", "0", "--limit-s", "0"]) == 0     # as the regenerated jobs files call it
    finally:
        hf_store.Store.push_set, hf_store.Store.commit = real, real_commit
    s2 = json.load(open(tmp_path / "o2" / "watch_summary.json"))
    assert s2["samples"] >= 3 and s2["samples"] == len(open(tmp_path / "o2" / "watch.jsonl").read().splitlines()) and s2["last"]["training_step"] == 387
    src = (CC / "base2_corpus.py").read_text()
    watch = src[src.index("def drill_watch("):src.index("def scn_ledger(")]
    assert "hf_store" not in watch and "push_set" not in watch and "open_store" not in watch


def test_rung_shape_for_d1():
    s = load("base2_rung_D1.json")
    assert s["arch"] == D1 and s["data"]["T"] == 1024 and s["data"]["vocab_cap"] == 16384
    assert "anneal" not in s["data"] and "out_dir" not in s and s["_shape"]["params"] == 69751296
    assert (s["train"]["batch"], s["train"]["accum"]) == (8, 2)


def test_trainer_sources_are_the_measured_commit():
    """Tooling commits only on this branch: the trainer the arms build is the one the probe measured."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    paths = ["src", "include", "third_party", "tools/mtstudio.cpp", "CMakeLists.txt"]
    p = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", TRAINER + "^{commit}"], capture_output=True)
    if p.returncode != 0:
        pytest.skip("the trainer commit is not in this clone")
    d = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", TRAINER, "--", *paths], capture_output=True, text=True)
    assert d.returncode == 0 and d.stdout.strip() == "", d.stdout
    u = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *paths], capture_output=True, text=True)
    assert u.stdout.strip() == "", u.stdout


# ---------------------------------------------------------------- the CPU build of the trainer
needs_trainer = pytest.mark.skipif(not (MTSTUDIO.exists() and VOCAB.exists()), reason="needs a CPU mtstudio and the 16k vocabulary")


def toy_ids(path, n=20000, seed=1):
    rng = random.Random(seed)
    with open(path, "wb") as f:
        f.write(b"PKIDS001" + struct.pack("<IIQ", 2, 16384, n))
        f.write(struct.pack(f"<{n}H", *[rng.randrange(300, 16000) for _ in range(n)]))


def run_trainer(tmp, spec, name):
    spec = json.loads(json.dumps(spec))
    ids = tmp / "toy.ids"
    if not ids.exists():
        toy_ids(ids)
    spec["data"].update(corpus=str(ids), vocab=str(VOCAB))
    if "anneal" in spec["data"]:
        spec["data"]["anneal"]["corpus"] = str(ids)
    spec["out_dir"] = str(tmp / name)
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MICROTORCH_CHECKPOINT_TXN": "1"}
    r = subprocess.run([str(MTSTUDIO), "run", str(p)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    assert "cuda" not in (r.stdout + r.stderr).lower().replace("cuda_visible", "")
    return [json.loads(ln) for ln in open(tmp / name / "events.jsonl")], r.stdout


@needs_trainer
@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_loads_in_the_trainer_with_its_parameter_count(arm, tmp_path):
    """The arm's own spec, with T 32, one step and a toy corpus in place of the real ones and nothing else changed."""
    if arm == "C" and not A_WEIGHTS.exists():
        pytest.skip("base A's final weights are not on this machine")
    j = train(arm)
    s = json.loads(json.dumps(j["spec"]))
    s["data"]["T"] = 32
    s["train"].update(steps=1, eval_every=1000, checkpoint_every=1000)
    s["export"] = {"formats": []}
    if arm == "C":
        assert P.sha256(str(A_WEIGHTS)) == load("base2_pins.json")["a_weights"][j["init_file"]]
        s["train"]["init_from"] = str(A_WEIGHTS)       # what jobs_vm.py's spec_for() writes for "init_file"
    ev, out = run_trainer(tmp_path, s, arm)
    model = next(e for e in ev if e["event"] == "model")
    want = 72761088 if arm == "C" else 69751296
    assert want in model.values(), model
    assert model.get("precision") == "bf16" and model.get("attention_kernel") == "flash"
    c = j["spec"]["arch"]["custom"]
    assert f"d={c['d']} layers={c['layers']} heads={c['heads']}" in out
    assert f"batch={j['spec']['train']['batch']} accum={j['spec']['train']['accum']}" in out
    steps = [e for e in ev if e["event"] == "step"]
    assert steps and steps[-1]["step"] == 1 and steps[-1]["loss"] == steps[-1]["loss"] and steps[-1]["loss"] < 20
    init = [e for e in ev if e["event"] == "init"]
    if arm == "C":
        # A strict load of base A's final weights into C's shape; a continued model starts near A's loss, not at ln(16384).
        assert init and init[0]["from"] == str(A_WEIGHTS)
    else:
        assert not init
    assert any(e["event"] == "state_tokens" for e in ev) or "state_tokens: k 4" in out


@needs_trainer
def test_the_last_checkpoint_is_at_the_last_step_and_the_tail_starts_at_its_step(tmp_path):
    """The end rung reads <job>/model.safetensors with state.txt at the run's last step; a small shape shows the trainer
    leaves them so under the transactional checkpoint, and that the tail corpus comes in after from_fraction of the steps."""
    s = json.loads(json.dumps(train("N")["spec"]))
    s["arch"]["custom"] = {"d": 64, "layers": 2, "heads": 2, "d_ff": 128}
    s["data"]["T"] = 32
    s["train"].update(steps=20, eval_every=1000, checkpoint_every=10)
    s["train"]["lr_schedule"]["warmup_steps"] = 2
    s["export"] = {"formats": ["safetensors"]}
    ev, _ = run_trainer(tmp_path, s, "small")
    out = tmp_path / "small"
    assert open(out / "state.txt").readline().strip() == "20"
    assert (out / "model.safetensors").exists() and (out / "optim.safetensors").exists()
    assert [e["step"] for e in ev if e["event"] == "checkpoint"] == [10, 20]
    an = [e for e in ev if e["event"] == "anneal_data"]
    assert len(an) == 1 and an[0]["from_step"] == 17              # 0.85 x 20


def region_ids(path, n, token, seed):
    """Half a constant token, half random ids: a window in the first half is soon easy, one in the second never is,
    so a step loss says which windows the step drew."""
    rng = random.Random(seed)
    ids = [token] * (n // 2) + [rng.randrange(300, 16000) for _ in range(n - n // 2)]
    with open(path, "wb") as f:
        f.write(b"PKIDS001" + struct.pack("<IIQ", 2, 16384, n))
        f.write(struct.pack(f"<{n}H", *ids))
    return ids


def swap_halves(src_ids, path):
    n = len(src_ids)
    ids = src_ids[n // 2:] + src_ids[:n // 2]
    with open(path, "wb") as f:
        f.write(b"PKIDS001" + struct.pack("<IIQ", 2, 16384, n))
        f.write(struct.pack(f"<{n}H", *ids))


def sampling_run(tmp, name, batch, accum, main, tail):
    s = json.loads(json.dumps(train("N")["spec"]))
    s["arch"]["custom"] = {"d": 64, "layers": 2, "heads": 2, "d_ff": 128}
    s["data"].update(T=32, corpus=str(main), vocab=str(VOCAB))
    s["data"]["anneal"] = {"corpus": str(tail), "from_fraction": 0.5}
    s["train"].update(steps=12, batch=batch, accum=accum, lr=0.003, eval_every=1000, checkpoint_every=1000)
    s["train"]["lr_schedule"]["warmup_steps"] = 1
    s["export"] = {"formats": []}
    s["out_dir"] = str(tmp / name)
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(s))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    r = subprocess.run([str(MTSTUDIO), "run", str(p)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    ev = [json.loads(ln) for ln in open(tmp / name / "events.jsonl")]
    assert [e["from_step"] for e in ev if e["event"] == "anneal_data"] == [6]
    return [e["loss"] for e in ev if e["event"] == "step"]


def test_one_window_a_row_from_one_generator_in_the_source():
    """What the sampling test below measures, as the trainer source has it: inside the micro-step loop each row takes
    one draw, so 2 micro-steps of 8 rows and 1 of 16 take the same 16 draws in the same order; and a resume replays
    start_step x accum x batch draws."""
    src = (ROOT / "tools" / "mtstudio.cpp").read_text()
    loop = src[src.index("for (int k = 0; !grpo && !po && k < s.accum; ++k) {"):]
    loop = loop[:loop.index('mark("data");')]
    assert loop.count("rng()") == 2 and "for (int b = 0; b < s.batch; ++b) {" in loop    # one for SFT rows, one for windows
    assert "const size_t at = rng() % lim;" in loop
    assert "for (int i = 0; i < start_step * s.accum * s.batch; ++i) rng();" in src


@needs_trainer
def test_eight_by_two_and_sixteen_by_one_read_the_same_windows(tmp_path):
    """Arm C trains at 16 x 1 and arm S at 8 x 2, on one corpus with one seed. The same windows give the same loss at
    every step, in the main part and after the tail switch; the control, the same run on a corpus whose halves are
    swapped, shows that these losses do tell windows apart."""
    main = region_ids(tmp_path / "main.ids", 40000, 500, 1)
    tail = region_ids(tmp_path / "tail.ids", 24000, 900, 2)     # another size, as the arms' tails are
    swap_halves(main, tmp_path / "main_swapped.ids")
    swap_halves(tail, tmp_path / "tail_swapped.ids")
    a = sampling_run(tmp_path, "r8x2", 8, 2, tmp_path / "main.ids", tmp_path / "tail.ids")
    b = sampling_run(tmp_path, "r16x1", 16, 1, tmp_path / "main.ids", tmp_path / "tail.ids")
    c = sampling_run(tmp_path, "control", 16, 1, tmp_path / "main_swapped.ids", tmp_path / "tail_swapped.ids")
    assert len(a) == len(b) == len(c) == 12
    same = max(abs(x - y) for x, y in zip(a, b))
    other_main = max(abs(x - y) for x, y in zip(b[1:6], c[1:6]))
    other_tail = max(abs(x - y) for x, y in zip(b[6:], c[6:]))
    record = {"8x2": a, "16x1": b, "control": c, "max_difference_8x2_16x1": same,
              "max_difference_control_main": other_main, "max_difference_control_tail": other_tail}
    print(json.dumps(record))
    if os.environ.get("BASE2_SAMPLING_RECORD"):
        Path(os.environ["BASE2_SAMPLING_RECORD"]).write_text(json.dumps(record, indent=1))
    assert same < 2e-3, (a, b)
    assert other_main > 50 * max(same, 1e-4) and other_tail > 50 * max(same, 1e-4), (b, c)


def guard_reads(*values):
    it = iter(values)
    return lambda: next(it)


def test_the_units_guard_asks_every_arm_to_stop_at_the_bound_and_delivers_to_each_folder(tmp_path, capsys):
    """The launch reviews of 12 Nov 2026: the units bound had no observer. base2_units_guard.py reads the balance as
    ccu_read.py does, logs every reading, and at the bound asks every listed runner to stop (STOP_REQUEST). The
    second launch review: delivery is per folder, each delivery logged, and retried at every later reading for a
    folder not yet delivered to; a listed folder that does not exist is said loudly (stdout and ALARM beside the
    log); the request is logged before any file is written. Here with a stubbed reader."""
    import base2_units_guard as G
    dirs = [tmp_path / f"arm_{a}" for a in ARMS]
    for d in dirs[:2]:
        d.mkdir()                                       # arm C's folder does not exist yet
    log = tmp_path / "g" / "guard.jsonl"
    argv = ["--campaign", "base2", "--stop-at", "640", "--once", "--log", str(log),
            *[x for d in dirs for x in ("--state-dir", str(d))]]
    for value in ({"currentBalance": 1000.0, "consumptionRateHourly": 5.82, "eligibleGpus": 3},
                  {"currentBalance": 700.5, "consumptionRateHourly": 5.82}):
        assert G.main(argv, read=guard_reads(value)) == 0
    assert not any((d / "STOP_REQUEST").exists() for d in dirs[:2]) and not (tmp_path / "g" / "ALARM").exists()
    capsys.readouterr()
    assert G.main(argv, read=guard_reads({"currentBalance": 359.0})) == 0     # 641 spent: at the bound
    text = (dirs[0] / "STOP_REQUEST").read_text()
    assert text.startswith("units: 641.00 spent since a balance of 1000.00, at or over the bound of 640.00")
    assert (dirs[1] / "STOP_REQUEST").read_text() == text
    alarm = (tmp_path / "g" / "ALARM").read_text()
    assert f"the state folder {dirs[2]} does not exist: that arm is not asked" in alarm
    assert f"ALARM (base2_units_guard): the stop is requested and the state folder {dirs[2]}" in capsys.readouterr().out
    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    at = next(i for i, r in enumerate(recs) if r.get("stop_requested"))
    assert [r.get("delivered") for r in recs[at + 1:]] == [str(dirs[0]), str(dirs[1])]   # logged after the request
    # The runners take theirs; arm C's folder appears; the next reading delivers to it alone, and the alarm goes.
    for d in dirs[:2]:
        os.replace(d / "STOP_REQUEST", d / "STOP_REQUEST.taken")
    dirs[2].mkdir()
    assert G.main(argv, read=guard_reads({"currentBalance": 340.0})) == 0
    assert not any((d / "STOP_REQUEST").exists() for d in dirs[:2]) and (dirs[2] / "STOP_REQUEST").read_text() == text
    assert not (tmp_path / "g" / "ALARM").exists()
    assert G.main(argv, read=guard_reads({"currentBalance": 330.0})) == 0     # every folder delivered: nothing more
    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert sum(1 for r in recs if r.get("stop_requested")) == 1 and sum(1 for r in recs if "delivered" in r) == 3
    assert all(r["campaign"] == "base2" for r in recs) and all(r["start"] == 1000.0 for r in recs if "start" in r)
    # A crash after the request was logged and before any file: the next reading delivers (the request stands).
    log3, d3 = tmp_path / "g3" / "guard.jsonl", tmp_path / "arm3"
    d3.mkdir()
    log3.parent.mkdir()
    log3.write_text(json.dumps({"at": 1.0, "campaign": "c3", "balance": 300.0, "start": 1000.0, "spent": 700.0,
                                "stop_at": 640.0, "stop_requested": True}) + "\n")
    G.main(["--campaign", "c3", "--stop-at", "640", "--once", "--log", str(log3), "--state-dir", str(d3)],
           read=guard_reads({"currentBalance": 299.0}))
    assert (d3 / "STOP_REQUEST").read_text().startswith("units: 700.00 spent since a balance of 1000.00")


def test_the_units_guard_is_one_campaigns_takes_only_good_readings_and_alarms_when_it_cannot_read(tmp_path, capsys):
    """The second launch review: an old log for another campaign supplied the start and an already-fired latch; a
    reader that exited non-zero, or printed a non-finite balance, was taken; and failed readings went on for ever
    with nothing but a log line. Now the log is one campaign's (refused otherwise); a reading counts only from a zero
    exit and a finite "currentBalance"; four failed readings in a row write ALARM, which the next good one clears."""
    import base2_units_guard as G
    d = tmp_path / "arm"
    d.mkdir()
    log = tmp_path / "g" / "guard.jsonl"
    argv = ["--campaign", "base2", "--stop-at", "640", "--once", "--log", str(log), "--state-dir", str(d)]
    G.main(argv, read=guard_reads({"currentBalance": 1000.0}))
    with pytest.raises(SystemExit) as e:
        G.main(["--campaign", "other", *argv[2:]], read=guard_reads({"currentBalance": 1.0}))
    assert "holds a record of campaign 'base2', not 'other'" in str(e.value)
    for n in range(3):                                   # three failed readings: no alarm yet
        G.main(argv, read=guard_reads({"error": 401}))
        assert not (tmp_path / "g" / "ALARM").exists(), n
    capsys.readouterr()
    G.main(argv, read=guard_reads({"error": "exit 1"}))                       # the fourth
    assert "4 readings of the balance have failed in a row" in (tmp_path / "g" / "ALARM").read_text()
    assert "ALARM (base2_units_guard): 4 readings" in capsys.readouterr().out
    G.main(argv, read=guard_reads({"currentBalance": 990.0}))                 # a good one clears it
    assert not (tmp_path / "g" / "ALARM").exists()
    G.main(argv, read=guard_reads({"currentBalance": float("nan")}))          # not finite: a failed reading
    assert json.loads(log.read_text().splitlines()[-1])["error"] == "no finite balance"
    # The real reader: only a zero exit with a finite balance; never the reply's text.
    stub = tmp_path / "ccu_stub.py"
    for code, want in (("import json\nprint(json.dumps({'currentBalance': 12.5, 'note': 'secret-text', 'ok': True}))\n",
                        {"currentBalance": 12.5}),
                       ("import json, sys\nprint(json.dumps({'currentBalance': 12.5}))\nsys.exit(3)\n", {"error": "exit 3"}),
                       ("import sys\nprint('Traceback: secret-text')\nsys.exit(2)\n", {"error": "exit 2"}),
                       ("print('[1, 2]')\n", {"error": "not an object"}),
                       ("print('{\"currentBalance\": \"12\"}')\n", {"error": "no finite balance"}),
                       ("print('{\"currentBalance\": Infinity}')\n", {"error": "no finite balance"})):
        stub.write_text(code)
        assert G.read_ccu(sys.executable, str(stub)) == want, code
    assert G.CCU_READ.endswith("AI_ML/chat_screens_claude/cloud_eval/ccu_read.py") and G.ALARM_AFTER == 4
    assert G.CLI_PY == "~/.local/share/uv/tools/google-colab-cli/bin/python"
    with pytest.raises(SystemExit):
        G.main(["--campaign", "x", "--stop-at", "0", "--once", "--state-dir", str(d), "--log", str(tmp_path / "x.jsonl")],
               read=guard_reads({}))


def test_the_units_guard_delivers_past_a_folder_it_cannot_write_and_retries_it(tmp_path, capsys, monkeypatch):
    """The third launch review (GPT 3): a request that could not be written into one folder (or whose delivery could
    not be logged) raised out of the loop: the later folders were never asked, and no ALARM said so. Each folder is
    tried on its own; the ones that fail are named in ALARM and tried again at the next reading, a failed reading
    included."""
    import base2_units_guard as G
    dirs = [tmp_path / f"arm_{a}" for a in ARMS]
    for d in dirs:
        d.mkdir()
    log = tmp_path / "g" / "guard.jsonl"
    argv = ["--campaign", "base2", "--stop-at", "640", "--once", "--log", str(log),
            *[x for d in dirs for x in ("--state-dir", str(d))]]
    G.main(argv, read=guard_reads({"currentBalance": 1000.0}))
    os.chmod(dirs[0], 0o500)                             # arm N's folder cannot be written
    real_append = G.append

    def append(path, rec):                               # and arm S's delivery cannot be logged, once
        if rec.get("delivered") == str(dirs[1]) and not append.failed:
            append.failed = True
            raise OSError("disk full")
        return real_append(path, rec)

    append.failed = False
    monkeypatch.setattr(G, "append", append)
    try:
        capsys.readouterr()
        G.main(argv, read=guard_reads({"currentBalance": 350.0}))               # 650 spent: at the bound
        alarm = (tmp_path / "g" / "ALARM").read_text()
        assert f"could not be delivered to {dirs[0]} (PermissionError)" in alarm
        assert f"could not be delivered to {dirs[1]} (OSError)" in alarm
        assert "ALARM (base2_units_guard): the stop is requested and it could not be delivered to" in capsys.readouterr().out
        assert not (dirs[0] / "STOP_REQUEST").exists() and (dirs[2] / "STOP_REQUEST").exists()   # C was asked all the same
        assert (dirs[1] / "STOP_REQUEST").exists()                              # written, its record not
        recs = [json.loads(ln) for ln in log.read_text().splitlines()]
        assert [r["delivered"] for r in recs if "delivered" in r] == [str(dirs[2])]
    finally:
        os.chmod(dirs[0], 0o700)
    # The next reading fails (the balance cannot be read): the deliveries outstanding are made all the same.
    G.main(argv, read=guard_reads({"error": 401}))
    assert (dirs[0] / "STOP_REQUEST").exists()
    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert sorted(r["delivered"] for r in recs if "delivered" in r) == sorted(map(str, dirs))
    assert (dirs[1] / "STOP_REQUEST").read_text() == (dirs[2] / "STOP_REQUEST").read_text()  # not written twice over
    G.main(argv, read=guard_reads({"currentBalance": 340.0}))                    # a good reading: nothing outstanding
    assert not (tmp_path / "g" / "ALARM").exists()

