#!/usr/bin/env python3
"""The jobs files of the stronger base (AI_ML/tiny_chat_benchmarks/stronger_base_spec.md, sections 9 to 11, in the author's
research tree). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/make_tc_b1.py          # writes tc_b1_s0.json, tc_b1_r1.json, tc_b1_probe_spec.json, bench specs

Stage 0 builds the corpus and proves the checkpoint path on a pod: saves every 125, a provider pod loss and warm resume, a
kill during a write, a rung job beside training, one retrieval, and the failed-job path up to a refused volume release. B
reuses the same data job, which then refuses unless the mix equals Stage 0's hashes.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V = "/content/chat/upload/vocab_llama3_bpe.gguf"
TXN = {"MICROTORCH_CHECKPOINT_TXN": "1"}
SHAPE = "experiments/tiny_chat_long_2026-10/specs/cell_Sfull.json"
SHAPE_L12 = "experiments/tiny_chat_long_2026-10/specs/cell_b2_l12.json"
RUNGS = [10000, 20000, 30000, 40000, 50000]
PAUSES = [20000, 30000]   # the race's reads: both arms stop at each until released (PROTOCOL.md section 30)
SMOKE = 1000              # a rung early on, so a rung fault or a limit shows here, at about $1, not at step 10,000
RUNG_TIMEOUT_S = 3600     # a rung runs alone (jobs_vm.py "exclusive"); past this it has failed
# Carry-5's forward chunks on a rung (PROTOCOL.md section 32): well under the 16,384 Carry-12 registered, because a rung runs beside a stopped
# trainer's memory, and Carry-5's contexts are short enough that a forward still holds many requests.
RUNG_SCORE_BATCH = 4096
# ember's /score for the final benchmarks: the commit tools/colab_chat/bench_v2.json gated (580e1089), under its new id
# after the history rewrite of 3 Nov (C:/ml_artifacts/purge_20261103/ember.cpp.commit-map).
EMBER_SCORE = "cc6635bee8c89b8b0304edeca633b1903b5d9291"


def arch():
    return {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
            "state_tokens": {"k": 4, "mask": "causal"}}


def b_spec():
    """Section 9, word for word."""
    return {"name": "tc_b1_r1", "arch": arch(),
            "data": {"corpus": "/content/chat/data/mix_b1.main.ids", "vocab": V, "vocab_cap": 16384, "T": 1024, "max_tokens": 0,
                     "anneal": {"corpus": "/content/chat/data/mix_b1.tail.ids", "from_fraction": 0.85}},
            "train": {"steps": 60000, "lr": 0.0005, "batch": 16, "accum": 1, "clip": 1.0, "eval_every": 1000, "eval_windows": 32,
                      "checkpoint_every": 125, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
                      "lr_schedule": {"warmup_steps": 2000, "decay": "cosine", "min_lr_frac": 0.1}},
            "export": {"formats": ["safetensors", "gguf"]}, "out_dir": "/content/chat/pt_b1_r1"}


def arch_l12():
    """October's shape, pk_base_v1 [BP], with state tokens: 97,539,840 parameters."""
    return {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 12, "heads": 12, "d_ff": 2048},
            "state_tokens": {"k": 4, "mask": "causal"}}


def l12_spec():
    """B's spec with the shape changed and nothing else that reaches the arithmetic. 8 x 2 is 16 rows a step, drawn one by one from
    the same stream as 16 x 1 (mtstudio's loop takes one rng() per row), so the same windows land in the same steps and the
    positions match step for step: 16,384 a step, 327.68M at step 20,000, 983.04M at 60,000. checkpoint_every is I/O only: 100 keeps
    a save about every 3.5 minutes at this rate."""
    s = b_spec()
    s["name"], s["out_dir"], s["arch"] = "tc_b2_l12", "/content/chat/pt_b2_l12", arch_l12()
    s["train"].update(batch=8, accum=2, checkpoint_every=100)
    return s


def l12_probe_spec():
    """60 steps of the real shape, no eval, no save, on the sample text boot_jobs.sh fetches: it needs only a tokenised window."""
    s = l12_spec()
    s["name"] = "probe"
    s.pop("out_dir")
    s["data"] = {"corpus": "corpus/data/fineweb_edu_sample_v1/pretrain_train.txt", "vocab": V, "vocab_cap": 16384, "T": 1024, "max_tokens": 2000000}
    s["train"].update(steps=60, eval_every=1000, checkpoint_every=1000)
    s["train"]["lr_schedule"]["warmup_steps"] = 10
    s["export"] = {"formats": []}
    return s


def l12_cell():
    s = l12_spec()
    s.pop("out_dir")
    s["data"].pop("anneal")
    s["data"]["corpus"] = "/content/chat/data/mix_s35.ids"
    s["_shape"] = {"note": "the rung job's shape: arch and T only reach it"}
    return s


def s0_spec():
    s = b_spec()
    s["name"], s["out_dir"] = "tc_b1_s0", "/content/chat/pt_s0"
    s["train"].update(steps=1000, eval_every=250)
    s["train"]["lr_schedule"]["warmup_steps"] = 100
    return s


def probe_spec():
    s = b_spec()
    s["name"] = "probe"
    s.pop("out_dir")
    s["train"].update(steps=150, eval_every=1000, checkpoint_every=50)
    s["train"]["lr_schedule"]["warmup_steps"] = 20
    s["export"] = {"formats": ["safetensors"]}
    return s


def torn_spec():
    s = probe_spec()
    s["data"] = {"corpus": "/content/chat/data/panel_fineweb.ids", "vocab": V, "vocab_cap": 16384, "T": 1024, "max_tokens": 0}
    s["train"].update(steps=2, eval_every=1000, checkpoint_every=1)
    return s


def sft_spec():
    """tiny_chat_run.cmd_post's SFT spec for S-full's SFT1500, with arch and T unchanged; init_job sets train.init_from."""
    t = json.load(open(os.path.join(ROOT, "experiments/tiny_chat_long_2026-10/specs/sft_T1024.json")))
    return {"name": "tc_sft_b1_r1", "arch": arch(), "data": t["data"],
            "train": {**t["train"], "batch": 16, "accum": 1, "seed": 1}, "export": t["export"]}


DECISION = {"home": "/mnt/c/ml_artifacts/tiny_chat/b2_overlap/overlap_decision.json", "vm": "/content/chat/upload/overlap_decision.json"}


def overlap_jobs(scan):
    """Codex's counts-only scan of the final mix, once, on arm A; a Codex-reviewed decision naming both mix hashes, then training on both."""
    j = []
    if scan:
        j.append({"name": "overlap", "after": "data", "light": True, "halt_on_failure": True,
                  "cmd": ["python3", "tools/colab_chat/tc_b2_overlap.py", "scan", "--data", "/content/chat/data", "--packet",
                          "/content/chat/upload/targets.keyed.json", "--out", "{out}"],
                  "outputs": ["overlap_receipt.json", "manifest.json"]})
    j.append({"name": "overlap_check", "after": "data", "light": True, "halt_on_failure": True, "gate_upload": DECISION,
              "promoted_by": {"files": [DECISION["vm"]], "round": "b2", "key": "go"},
              "cmd": ["python3", "tools/colab_chat/tc_b2_overlap.py", "check", "--data", "/content/chat/data", "--decision", DECISION["vm"],
                      "--out", "{out}"],
              "outputs": ["overlap_check.json"]})
    return j


def cuda_gate():
    return {"name": "cuda_gate", "halt_on_failure": True,
            "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "cuda_gate", "--out", "{out}"],
            "outputs": ["cuda_gate.json", "cuda_gate.log"]}


def precheck(train, rung_job, jobs_file, after):
    """The whole read path before step 1, on the arm's own shape and mix (tc_b2_precheck.py). Training waits on it as well as on the overlap go."""
    return {"name": "precheck", "after": after, "halt_on_failure": True,
            "cmd": ["python3", "tools/colab_chat/tc_b2_precheck.py", "--out", "{out}", "--jobs", f"tools/colab_chat/{jobs_file}", "--train", train,
                    "--rung", rung_job, "--ember", EMBER_SCORE],
            "outputs": ["precheck.json", "precheck_train.json", "previous.json", "rung/rung.json", "parity/parity_summary.json", "train.log", "rung.log", "parity.log"]}


def data(mode, after=None):
    return {"name": "data", "light": True, "halt_on_failure": True, **({"after": after} if after else {}),
            # Keeps the mix's text: Codex's counts-only scan of the final mix reads it (inbox, 7 Nov). Stage 0 needed no scan.
            "env": {} if mode == "stage0" else {"TC_B1_KEEP_MIX_TEXT": "1"},
            "cmd": ["bash", "tools/colab_chat/base_b1_data.sh", "{out}", V, mode],
            "outputs": ["data.ledger.json", "human_decontam.json", "mix_b1.ledger.json", "filter.ledger.json",
                        "fineweb_b1.ledger.json", "ids_b1.json", "tc_b1_pins_observed.json",
                        "long/human_dialogue_v1.ledger.json", "lever2/dialogue_pt_v1.ledger.json",
                        "human/human_dialogue_v1.ledger.json"]}


def rung(job, step, tag, keep=False, shape=SHAPE, halt=False, exclusive=False, batch=0):
    """keep: a rung the race reads: its GGUF and weights stay at home, and a failure stops the arm, because the read needs it."""
    j = {"name": f"rung_{tag}_s{step}", "after": "data", "light": True, "rung_of": {"job": job, "step": step},
         "env": {"MICROTORCH_POOL_MB": 2048},
         "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "rung", "--out", "{out}", "--spec", shape, "--export-gguf",
                 "--weights", f"/content/chat/{job}/rung_{step}/model.safetensors", "--pt-dir", f"/content/chat/{job}",
                 "--step", str(step), "--job", job, "--cell", tag.upper(), "--seed", "1", "--share", "30",
                 "--panel", "fineweb=/content/chat/data/panel_fineweb.ids",
                 "--panel", "dialogue=/content/chat/data/panel_dialogue.ids",
                 "--carry5-items", "corpus/eval/carry5_v1.jsonl", *(["--score-batch-tokens", str(batch)] if batch else [])],
         "outputs": ["rung.json"]}
    if exclusive:
        j["exclusive"], j["timeout_s"] = True, RUNG_TIMEOUT_S
    if halt:
        j["halt_on_failure"] = True
    if keep:
        j["keep_home_snapshot"] = True
        j["halt_on_failure"] = True
        j["outputs"] = ["rung.json", "model.gguf"]
    return j


def stage0():
    return {"_comment": "Stage 0 of tc_b1: section 10, step 2, of the stronger base spec. Cap $2, one RunPod L4. Reviewed and "
                        "admitted on its own; B has its own admission after this.",
            "max_parallel": 1, "max_parallel_light": 2,
            "jobs": [
                cuda_gate(), data("stage0"),
                {"name": "pt_s0", "model_name": "tc_b1_s0", "after": ["data", "cuda_gate"], "relay_every_steps": 250,
                 "snapshot_steps": [250], "pod_loss_drill_step": 250, "halt_on_failure": True, "env": TXN, "spec": s0_spec()},
                rung("pt_s0", 250, "s0"),
                {"name": "s0_probe", "after": ["pt_s0", "rung_s0_s250"], "after_terminal": True, "halt_on_failure": True,
                 "env": TXN,
                 "cmd": ["python3", "tools/colab_chat/tc_b1_s0_probe.py", "--out", "{out}", "--spec",
                         "tools/colab_chat/tc_b1_probe_spec.json"],
                 "outputs": ["probe.json"]},
                {"name": "s0_torn_setup", "after": "s0_probe", "light": True, "halt_on_failure": True,
                 "cmd": ["python3", "tools/colab_chat/tc_b1_s0_probe.py", "--out", "{out}", "--torn", "/content/chat/s0_torn"],
                 "outputs": ["torn.json"]},
                {"name": "s0_torn", "model_name": "tc_b1_torn", "after": "s0_torn_setup", "halt_drill": True, "env": TXN,
                 "spec": torn_spec()},
            ]}


def b1():
    tags = [f"b1_s{s}" for s in RUNGS] + ["b1_final", "b1_sft", "sfull_sft1500"]
    links = [(f"/content/chat/pt_b1_r1/rung_{s}/model.safetensors", f"b1_s{s}") for s in RUNGS]
    links += [("/content/chat/pt_b1_r1/tc_b1_r1.safetensors", "b1_final"), ("/content/chat/sft_b1_r1/tc_sft_b1_r1.safetensors", "b1_sft")]
    link = " && ".join(f"ln -sf {a} /content/chat/upload/{t}.safetensors" for a, t in links)
    jobs = [cuda_gate(), data("b1"), *overlap_jobs(True), precheck("pt_b1_r1", f"rung_b1_s{SMOKE}", "tc_b1_r1.json", ["data", "cuda_gate"]),
            {"name": "pt_b1_r1", "model_name": "tc_b1_r1", "after": ["data", "cuda_gate", "overlap", "precheck", "overlap_check"], "relay_every_steps": 2500,
             "snapshot_steps": [SMOKE, *RUNGS], "snapshot_full_steps": PAUSES, "pause_rungs": [f"rung_b1_s{s}" for s in PAUSES],
             "halt_on_failure": True, "env": TXN, "spec": b_spec()}]
    jobs += [rung("pt_b1_r1", SMOKE, "b1", halt=True, exclusive=True, batch=RUNG_SCORE_BATCH)]
    jobs += [rung("pt_b1_r1", s, "b1", keep=s <= PAUSES[-1], exclusive=True, batch=RUNG_SCORE_BATCH) for s in RUNGS]
    jobs += [{"name": "sft_b1_r1", "model_name": "tc_sft_b1_r1", "after": "pt_b1_r1", "init_job": "pt_b1_r1",
              "relay_every_steps": 500, "halt_on_failure": True, "env": TXN, "spec": sft_spec()},
             {"name": "items", "after": "sft_b1_r1", "light": True, "halt_on_failure": True,
              "cmd": ["bash", "tools/colab_chat/bench_vm.sh", "items", "{out}"], "outputs": ["items_build.json", "fetch.log"]},
             {"name": "bench_b1", "after": ["items", "sft_b1_r1"], "halt_on_failure": True,
              "cmd": ["sh", "-c", f"{link} && env BENCH_SCORE_COMMIT={EMBER_SCORE} EMBER_DETERMINISTIC=1 bash "
                      f"tools/colab_chat/bench_vm.sh ember {{out}} 100 1 {' '.join(tags)}"],
              "outputs": ["results.tar.gz"]}]
    return {"_comment": "The race's 8x768 arm (PROTOCOL section 30): 60,000 steps at T 1,024, k 4, stopping at each read. Not admitted.",
            "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}, tags


def l12():
    """The race's 12x768 arm, pod B. The same corpus pins, schedule and seed as tc_b1_r1.json; its own SFT stage at 8 x 2."""
    tags = [f"l12_s{s}" for s in RUNGS] + ["l12_final", "l12_sft", "sfull_sft1500"]
    links = [(f"/content/chat/pt_b2_l12/rung_{s}/model.safetensors", f"l12_s{s}") for s in RUNGS]
    links += [("/content/chat/pt_b2_l12/tc_b2_l12.safetensors", "l12_final"),
              ("/content/chat/sft_b2_l12/tc_sft_b2_l12.safetensors", "l12_sft")]
    link = " && ".join(f"ln -sf {a} /content/chat/upload/{t}.safetensors" for a, t in links)
    sft = sft_spec()
    sft["name"], sft["arch"] = "tc_sft_b2_l12", arch_l12()
    sft["train"].update(batch=8, accum=2)
    jobs = [cuda_gate(),
            {"name": "l12_probe", "after": "cuda_gate", "halt_on_failure": True,
             "cmd": ["python3", "tools/colab_chat/tc_b2_probe.py", "--out", "{out}", "--spec", "tools/colab_chat/tc_b2_probe_spec.json"],
             "outputs": ["probe.json"]},
            data("b1", after="l12_probe"), *overlap_jobs(True), precheck("pt_b2_l12", f"rung_l12_s{SMOKE}", "tc_b2_l12.json", ["data", "cuda_gate"]),
            {"name": "pt_b2_l12", "model_name": "tc_b2_l12", "after": ["data", "cuda_gate", "l12_probe", "overlap", "precheck", "overlap_check"],
             "relay_every_steps": 2500, "snapshot_steps": [SMOKE, *RUNGS], "snapshot_full_steps": PAUSES,
             "pause_rungs": [f"rung_l12_s{s}" for s in PAUSES], "halt_on_failure": True, "env": TXN, "spec": l12_spec()}]
    jobs += [rung("pt_b2_l12", SMOKE, "l12", shape=SHAPE_L12, halt=True, exclusive=True, batch=RUNG_SCORE_BATCH)]
    jobs += [rung("pt_b2_l12", s, "l12", keep=s <= PAUSES[-1], shape=SHAPE_L12, exclusive=True, batch=RUNG_SCORE_BATCH) for s in RUNGS]
    jobs += [{"name": "sft_b2_l12", "model_name": "tc_sft_b2_l12", "after": "pt_b2_l12", "init_job": "pt_b2_l12",
              "relay_every_steps": 500, "halt_on_failure": True, "env": TXN, "spec": sft},
             {"name": "items", "after": "sft_b2_l12", "light": True, "halt_on_failure": True,
              "cmd": ["bash", "tools/colab_chat/bench_vm.sh", "items", "{out}"], "outputs": ["items_build.json", "fetch.log"]},
             {"name": "bench_l12", "after": ["items", "sft_b2_l12"], "halt_on_failure": True,
              "cmd": ["sh", "-c", f"{link} && env BENCH_SCORE_COMMIT={EMBER_SCORE} EMBER_DETERMINISTIC=1 bash "
                      f"tools/colab_chat/bench_vm.sh ember {{out}} 100 1 {' '.join(tags)}"],
              "outputs": ["results.tar.gz"]}]
    return {"_comment": "The race's 12x768 arm (PROTOCOL section 30). Not admitted. It continues past the verdict at 327.68M positions only "
                        "if it wins, and then only inside its cap.", "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}, tags


def bench_spec(tag, a=None):
    return {"name": tag, "arch": a or arch(), "data": {"vocab": V, "vocab_cap": 16384, "T": 1024},
            "train": {"steps": 1, "lr": 0.001, "batch": 16, "seed": 1}, "export": {"formats": []}, "out_dir": f"/content/eval/{tag}"}


def write(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def main():
    write(os.path.join(HERE, "tc_b1_s0.json"), stage0())
    jobs, tags = b1()
    write(os.path.join(HERE, "tc_b1_r1.json"), jobs)
    write(os.path.join(HERE, "tc_b1_probe_spec.json"), probe_spec())
    for t in tags:
        write(os.path.join(ROOT, f"experiments/benchmarks_2026-10-26/specs/{t}.spec.json"), bench_spec(t))
    jobs, tags = l12()
    write(os.path.join(HERE, "tc_b2_l12.json"), jobs)
    write(os.path.join(HERE, "tc_b2_probe_spec.json"), l12_probe_spec())
    write(os.path.join(ROOT, "experiments/tiny_chat_long_2026-10/specs/cell_b2_l12.json"), l12_cell())
    for t in tags:
        if t != "sfull_sft1500":
            write(os.path.join(ROOT, f"experiments/benchmarks_2026-10-26/specs/{t}.spec.json"), bench_spec(t, arch_l12()))


if __name__ == "__main__":
    main()
