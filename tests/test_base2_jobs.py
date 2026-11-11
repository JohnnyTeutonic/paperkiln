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
    # The watcher: the training step against the courier's pushed step, to the store every 30 s; every push ended
    # after 60 s, the watcher after two hours, and coreutils' timeout five minutes later whatever it is doing.
    w = js["watch"]["cmd"]
    assert w[:4] == ["timeout", "-k", "30", "7500"] and w[6] == "drill-watch"
    assert w[w.index("--job") + 1] == job and w[w.index("--interval") + 1] == "30"
    assert w[w.index("--limit-s") + 1] == "7200" and w[w.index("--push-limit-s") + 1] == "60"
    assert w[w.index("--config") + 1] == "/content/chat/store.json" and not js["watch"].get("halt_on_failure")


def test_the_storage_guard_fits_the_rehearsal_and_is_off_for_the_arms():
    """The courier stops pushing at 12 GB pushed since a release last freed storage. Release spares objects younger
    than its grace period, so for the first grace period of pushing (and the minute or two until the next push
    prunes and releases) nothing can be freed and everything pushed counts. The save interval keeps that under the
    cap; the intervals the rehearsal had before (50) and the one first proposed (100) do not."""
    for name in ("base2_rehearsal.json", "base2_rehearsal_b.json"):
        f = load(name)
        assert f["store_push_cap_gb"] == 12 == M.REH_PUSH_CAP_GB
        assert f["_base2"]["store_release_grace_s"] == 1800 == M.REH_RELEASE_GRACE_S
    gen_gb = 3 * M.PARAMS["D1"] * 4 / 1e9             # weights and two AdamW moments, 4 bytes a parameter
    assert 0.83 < gen_gb < 0.85

    def counted(every):                               # GB pushed before a release can first free anything
        window = M.REH_RELEASE_GRACE_S + 2 * every * M.STEP_S     # the grace, then up to two more saves
        return (int(window / (every * M.STEP_S)) + 1) * gen_gb + gen_gb / 3     # and the mark's weights
    assert counted(M.REH_EVERY) < 0.8 * M.REH_PUSH_CAP_GB
    assert counted(100) > M.REH_PUSH_CAP_GB and counted(50) > 2 * M.REH_PUSH_CAP_GB
    # The arms carry the field, off: the lead sets it before they launch.
    for arm in ARMS:
        assert load(f"base2_{arm}.json")["store_push_cap_gb"] == 0


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
    assert s["pushed"] == 1 and s["prefix"].startswith("watch/pt_reh/")
    stored = json.load(open(tmp_path / "store" / "base2-build" / s["prefix"] / "watch.json"))
    assert stored["live_step"] == 350 and stored["pushed_step"] == 300 and stored["status"] == "FINISHED"
    assert stored["training_step"] == 387
    # No courier yet, no checkpoint yet, no event yet: recorded as unknown, never guessed.
    none = B.watch_sample(str(tmp_path / "nowhere"), "pt_reh")
    assert none["live_minus_pushed"] is None and none["training_step"] is None and none["training_minus_pushed"] is None
    # A push that hangs is ended at its limit and the watcher goes on to its end: it cannot keep the job open.
    import hf_store
    import time
    real = hf_store.Store.push_set
    hf_store.Store.push_set = lambda self, *a, **kw: time.sleep(600)
    try:
        t0 = time.time()
        assert B.main(["drill-watch", "--config", str(cfg), "--chat", str(chat), "--job", "pt_reh", "--out", str(tmp_path / "o2"),
                       "--interval", "0", "--push-limit-s", "1"]) == 0
        assert time.time() - t0 < 30
    finally:
        hf_store.Store.push_set = real
    s2 = json.load(open(tmp_path / "o2" / "watch_summary.json"))
    assert s2["pushed"] == 0 and s2["samples"] == 1 and s2["last"]["training_step"] == 387


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
