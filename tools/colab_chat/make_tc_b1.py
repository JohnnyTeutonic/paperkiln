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
RUNGS = [10000, 20000, 30000, 40000, 50000]
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


def cuda_gate():
    return {"name": "cuda_gate", "halt_on_failure": True,
            "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "cuda_gate", "--out", "{out}"],
            "outputs": ["cuda_gate.json", "cuda_gate.log"]}


def data(mode):
    return {"name": "data", "light": True, "halt_on_failure": True,
            "cmd": ["bash", "tools/colab_chat/base_b1_data.sh", "{out}", V, mode],
            "outputs": ["data.ledger.json", "human_decontam.json", "mix_b1.ledger.json", "filter.ledger.json",
                        "fineweb_b1.ledger.json", "ids_b1.json", "tc_b1_pins_observed.json",
                        "long/human_dialogue_v1.ledger.json", "lever2/dialogue_pt_v1.ledger.json"]}


def rung(job, step, tag, keep=False):
    j = {"name": f"rung_{tag}_s{step}", "after": "data", "light": True, "rung_of": {"job": job, "step": step},
         "env": {"MICROTORCH_POOL_MB": 2048},
         "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "rung", "--out", "{out}", "--spec", SHAPE,
                 "--weights", f"/content/chat/{job}/rung_{step}/model.safetensors", "--pt-dir", f"/content/chat/{job}",
                 "--step", str(step), "--job", job, "--cell", "B1", "--seed", "1", "--share", "30",
                 "--panel", "fineweb=/content/chat/data/panel_fineweb.ids",
                 "--panel", "dialogue=/content/chat/data/panel_dialogue.ids",
                 "--carry5-items", "corpus/eval/carry5_v1.jsonl"],
         "outputs": ["rung.json"]}
    if keep:
        j["keep_home_snapshot"] = True
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
    jobs = [cuda_gate(), data("b1"),
            {"name": "pt_b1_r1", "model_name": "tc_b1_r1", "after": ["data", "cuda_gate"], "relay_every_steps": 2500,
             "snapshot_steps": RUNGS, "halt_on_failure": True, "env": TXN, "spec": b_spec()}]
    jobs += [rung("pt_b1_r1", s, "b1", keep=s <= 20000) for s in RUNGS]
    jobs += [{"name": "sft_b1_r1", "model_name": "tc_sft_b1_r1", "after": "pt_b1_r1", "init_job": "pt_b1_r1",
              "relay_every_steps": 500, "halt_on_failure": True, "env": TXN, "spec": sft_spec()},
             {"name": "items", "after": "sft_b1_r1", "light": True, "halt_on_failure": True,
              "cmd": ["bash", "tools/colab_chat/bench_vm.sh", "items", "{out}"], "outputs": ["items_build.json", "fetch.log"]},
             {"name": "bench_b1", "after": ["items", "sft_b1_r1"], "halt_on_failure": True,
              "cmd": ["sh", "-c", f"{link} && env BENCH_SCORE_COMMIT={EMBER_SCORE} EMBER_DETERMINISTIC=1 bash "
                      f"tools/colab_chat/bench_vm.sh ember {{out}} 100 1 {' '.join(tags)}"],
              "outputs": ["results.tar.gz"]}]
    return {"_comment": "B: fresh 8x768, 60,000 steps at T 1,024, k 4 (spec section 9). Cap $21, cumulative with Stage 0 $23. Not "
                        "admitted: needs Stage 0's hashes, measured costs and its own review.",
            "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}, tags


def bench_spec(tag):
    return {"name": tag, "arch": arch(), "data": {"vocab": V, "vocab_cap": 16384, "T": 1024},
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


if __name__ == "__main__":
    main()
