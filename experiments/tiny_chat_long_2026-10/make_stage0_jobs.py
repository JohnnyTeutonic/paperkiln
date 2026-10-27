#!/usr/bin/env python3
"""Writes tools/colab_chat/carry12_stage0_v1.json: the Carry-12 programme's pre-freeze VM work on one RunPod L4
(PREREGISTRATION_DRAFT.md section 11 items 3-6), and prints the runner's --upload arguments.

    python experiments/tiny_chat_long_2026-10/make_stage0_jobs.py

Jobs, one GPU job at a time (CPU data jobs in the light pool beside them):
- cuda_gate: every CUDA leg (tools/colab_cuda_validate.sh, state tokens and llama_attnres included) on this GPU.
- decontam_fineweb (light): Carry-12 v1's contamination scan (tools/carry12_build.py decontam, the laptop scan's
  rules) over the VM's FineWeb-Edu shard, corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt (3.46 GB, ~8 MB/s).
  Taskmaster-3 is on the laptop and was scanned there (carry12_v1_decontam_taskmaster3.json).
- data (light): the T 1,024 data job (section 11 item 5): E1's sources, split, 35% mix and panels, with Carry-12
  in the decontamination beside Carry-5 (--carry12) and the SFT set at T 1,024, plus cell D's set
  (--long-sft 10:4: each SFT row of 10+ trained turns four times). The mix is a token stream: mtstudio draws
  windows of T, so mix_s35.ids serves every cell at any T.
- data_long3 (light, after data): the same with data option 3's four long sources (Taskmaster-3, ABCD, CCPE-M,
  PersuasionForGood; MSC excluded) added: counts only (its ledger comes home), for Jonathan's ruling on option 3
  (section 11 item 7). Nothing trains on it.
- probes (Stage 0 at T 1,024, section 11 item 6): 300 steps alone on the GPU for B (60m_L8, 16 x 1,024, and the
  8 x 2 fallback), S-gist (B + k 4 state slots, turn_state), P16 (60m_L16, 8 x 2) and X (60m_L16 + llama_attnres,
  8 x 2), on the data job's mix_s35.ids. D shares B's rate.
- cost_* (section 10 and item 3: the measured cost of a Carry-12 post job): tools/carry12_l0.py run --cost-run on
  the E1 at 60M picks of the core's two shapes (60m_L8 for B, D and S-gist; 60m_L16 for P16 and X) and on
  pick_e1_60m_L32, served at context 1,024 (--ctx 1024, the F32 GGUF exported from the weights on the VM), the
  post job's Carry-12 configuration: forced choice in gold, self, near, twin and noctx by mtstudio score on the
  GPU, self history to the target at --max-new 48 by ember's CUDA server (ember 7526469, EMBER_GEMM_TF32=0,
  EMBER_DETERMINISTIC=1) after its 20-item spot-check, no near_self, every reply through the parity gate. The
  picks were trained at T 512, so beyond position 512 their replies run to the 48-token cap more often than a
  T 1,024 model's would: an upper bound on generation time. Receipts are smoke (never L0 results).
At home: copy ART/decontam_fineweb/*.json, ART/data/data.ledger.json (as carry12_data_t1024_ledger.json),
ART/data_long3/data.ledger.json (as carry12_data_long3_ledger.json), ART/cuda_gate/cuda_gate.json and every
ART/probe60_*/summary.json and ART/cost_*/*/l0.json into experiments/tiny_chat_long_2026-10/stage0/, then
`python experiments/tiny_chat_long_2026-10/stage0_total.py` writes the measured total (stage0/total.json).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
JOBS = os.path.join(ROOT, "tools", "colab_chat", "carry12_stage0_v1.json")
REL = "tools/colab_chat/carry12_stage0_v1.json"
SPECS = "experiments/tiny_chat_long_2026-10/specs"
VOCAB = "/content/chat/upload/vocab_llama3_bpe.gguf"
SOURCES = "oasst,convai2,wow,bst,taskmaster,multiwoz,self_dialogue,topical_chat,dolly,quac,empathetic,dailydialog,no_robots"
LONG = "taskmaster3,abcd,ccpe,p4g"
EMBER = "7526469695ee3f07f2b5d6e60568cd3e09529c84"
PICKS = {  # tag: home folder (WSL path), each with weights (not optimiser state), spec.json at T 512
    "pick_e1_60m_L8": "/mnt/c/ml_artifacts/tiny_chat/e1_60m_L8/pick_e1_60m_L8",
    "pick_e1_60m_L16": "/mnt/c/ml_artifacts/tiny_chat/e1_60m_L16/pick_e1_60m_L16",
    "pick_e1_60m_L32": "/mnt/c/ml_artifacts/tiny_chat/e1_60m/pick_e1_60m_L32",
}
PROBES = ["probe60_B_T1024", "probe60_B_T1024_acc2", "probe60_Sgist_T1024", "probe60_P16_T1024", "probe60_X_T1024"]


def data_cmd(extra_sources="", human_dirs=None, fetch=None):
    cmd = ["python3", "tools/colab_chat/tiny_chat_data.py", "--out", "{out}", "--vocab", VOCAB,
           "--fineweb", "corpus/data/fineweb_edu_shard0_v1", "--shares", "35", "--mix-bytes", "800000000",
           "--max-epochs", "4", "--T", "1024", "--seed", "1"]
    for d in human_dirs or []:
        cmd += ["--human-dir", d]
    cmd += ["--human-fetch", fetch or f"python3 corpus/fetchers/human_dialogue.py {SOURCES} --allow-sa --allow-nc "
                                      "--out-dir {human}",
            "--sources", SOURCES + ("," + extra_sources if extra_sources else ""), "--require-pinned", "--carry12",
            "--long-sft", "10:4"]
    return cmd


def main():
    jobs = [{"name": "cuda_gate", "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "cuda_gate", "--out", "{out}"],
             "outputs": ["cuda_gate.json", "cuda_gate.log"]},
            {"name": "decontam_fineweb", "light": True,
             "cmd": ["python3", "tools/carry12_build.py", "decontam", "--out", "{out}/carry12_v1_decontam_fineweb_shard0.json",
                     "--state", "{out}/carry12_v1_decontam_fineweb_shard0.state.json",
                     "corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt"],
             "outputs": ["carry12_v1_decontam_fineweb_shard0.json", "carry12_v1_decontam_fineweb_shard0.state.json"]},
            {"name": "data", "light": True, "cmd": data_cmd(), "outputs": ["data.ledger.json", "human_decontam.json"]}]
    for p in PROBES:
        jobs.append({"name": p, "after": ["data", "cuda_gate"],
                     "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "probe", "--out", "{out}", "--spec",
                             f"{SPECS}/{p}.json", "--corpus", "/content/chat/data/mix_s35.ids", "--steps", "300",
                             "--cuda-gate", "/content/chat/cuda_gate", "--data-job", REL],
                     "outputs": ["summary.json"]})
    # Option 3's counts after the probes (which run alone on the VM), beside the first cost job.
    jobs.append({"name": "data_long3", "light": True, "after": ["data"] + PROBES, "after_terminal": True,
                 "cmd": data_cmd(LONG, ["{out}/long", "/content/chat/data/human"],
                                 f"python3 corpus/fetchers/human_dialogue.py {LONG} --out-dir {{human}}"),
                 "outputs": ["data.ledger.json", "human_decontam.json"]})
    uploads = []
    for tag, d in PICKS.items():
        uploads += [f"{tag}.model.safetensors={d}/model.safetensors", f"{tag}.spec.json={d}/spec.json"]
        # after_terminal: a probe that fails (e.g. B's 16 x 1,024 out of memory) must not cancel the cost runs.
        jobs.append({"name": f"cost_{tag}", "after": ["cuda_gate"] + PROBES, "after_terminal": True,
                     "cmd": ["python3", "tools/carry12_l0.py", "run", "--weights",
                             f"/content/chat/upload/{tag}.model.safetensors", "--spec-json",
                             f"/content/chat/upload/{tag}.spec.json", "--tag", tag, "--out", "{out}", "--gen", "ember",
                             "--ember-bin", "auto", "--ember-commit", EMBER, "--mtstudio",
                             "/content/pk/build_cuda/mtstudio", "--vocab", VOCAB, "--gpu-layers", "-1", "--spot", "20",
                             "--cost-run", "--ctx", "1024", "--max-new", "48", "--no-near-self"],
                     "outputs": [f"{tag}/l0.json", f"{tag}/eval/summary.json", f"{tag}/eval/summary.md",
                                 f"{tag}/eval/{tag}/gen_engine.json", f"{tag}/eval.log"]})
    uploads.append("vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf")
    doc = {"_comment": "Carry-12 pre-freeze VM work (experiments/tiny_chat_long_2026-10/make_stage0_jobs.py, which "
                       "says what each job is and what to do at home): the CUDA gate, the FineWeb-Edu shard's "
                       "Carry-12 contamination scan, the T 1,024 data job (Carry-12 in the decontamination, cell D's "
                       "SFT set), option 3's counts, Stage 0 probes at T 1,024 of B, S-gist, P16 and X, and the "
                       "measured cost of a Carry-12 post job on the 60m_L8, 60m_L16 and 60m_L32 picks. One GPU job "
                       "at a time; nothing trains a Carry-12 cell. --upload arguments: "
                       + " ".join(f"--upload {u}" for u in uploads),
           "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}
    with open(JOBS, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(" ".join(f"--upload {u}" for u in uploads))


if __name__ == "__main__":
    main()
