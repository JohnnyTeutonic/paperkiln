# General-text chat model: results (20 Oct 2026)

Plan: `PLAN.md`. Receipts: `receipts/`.

## What ran

- **Base:** d384 L6 H6 llama (14.6M parameters), 47,500 steps at batch
  16 x 256 (~195M tokens) on FineWeb-Edu shard 0 (80%) and TinyChat (20%),
  vocabulary `vocab_fineweb_tinychat_bpe8k_v1`. Best held-out val loss 3.437
  (`receipts/fw_base_result.json`). Colab L4, about 15 hours.
- **fwsft:** the base + 500 SFT steps on the TinyChat sample dialogues (val
  2.63).
- **fwmix:** the base + 1,500 SFT steps on 48,000 examples, half
  smol-smoltalk Q&A dialogues (Apache-2.0) and half TinyChat (val 2.52 on its
  own held-out mix).
- Both SFTs ran on the base's VM after the base finished, and were
  re-exported locally with paperkiln master (rms eps 1e-5, ggml dims,
  embedded IR), f32 and q8_0_all.

Compared with the live model, `chat_sftfinal` (TinyChat base, 12,000 steps,
then SFT), served by ember's frozen release c843421 on CUDA.

## Results

Chat (`tools/chat_eval.py`, 100 frozen TinyChat prompts, temperature 0.7):

| model | clean stop | repeats | ref F1 | ref bits/byte |
|---|---:|---:|---:|---:|
| live | 1.00 | 0.00 | 0.162 | 0.565 |
| fwsft | 1.00 | 0.10 | 0.085 | 0.838 |
| fwmix | 1.00 | 0.06 | 0.089 | 0.842 |

Simple-QA probe (`tools/qa_eval.py`, 60 frozen questions, greedy):

| model | arith | fact | science | turn | all |
|---|---:|---:|---:|---:|---:|
| live | 0.00 | 0.00 | 0.00 | 0.20 | 0.03 |
| fwsft | 0.00 | 0.00 | 0.07 | 0.30 | 0.07 |
| fwmix | 0.00 | 0.08 | 0.07 | 0.10 | 0.07 |

## Reading

- Neither FineWeb model replaces the live model. Both are worse at chat (half
  the reference F1, higher reference bits/byte, 6-10% repeating replies).
- The QA gain is not knowledge. fwmix has learnt the *form* of an answer
  ("The capital of France is ...", "The closest star to the Earth is the Sun")
  but not its content: most answers loop on the question ("the capital of
  France is the capital of France ..."), and its three fact/science hits sit
  inside loops or are lenient containment matches (`receipts/qa_fw_vs_live.md`).
- ember gates: fwsft f32 40/40 greedy vs mtstudio on CPU, CUDA and WASM,
  q8_0_all 35/40, context-contract harness 13/13.

## Capacity session (20-21 Oct 2026)

One Colab L4 session (tools/colab_chat/runner_multi.py, capacity_v2.json),
after the training-speed work (d384 1.14 -> 0.109 s/step on the L4).

- **MoE vs dense at equal tokens.** From fw_base, 10,000 steps (41M tokens)
  of FineWeb-Edu text the base never read, same seed, data and warmup +
  cosine schedule. Dense continuation (ctrl): best val 3.3735. E=8 top-2
  renorm upcycle (moe, 70M parameters, ~2x FFN compute per token): best val
  3.3516. A real but small gain at this length. ctrl's rerun on a second VM
  reproduced 3.3734705448150635 exactly.
- **ctrl_sft** (ctrl + the 48k QA+chat mix, 1,500 steps; SFT val 2.410 vs
  fwmix 2.52). QA columns (QA all to turn): simple_qa_v1. Chat columns (ref F1 onward):
  tinychat_hf_prompts_v1, 100 TinyChat prompts, not chat_prompts_v1:

| model | QA all | fact | science | turn | ref F1 | prompt overlap | repeats |
|---|---:|---:|---:|---:|---:|---:|---:|
| live | 0.03 | 0.00 | 0.00 | 0.20 | 0.162 | 0.150 | 0.00 |
| fwmix | 0.07 | 0.08 | 0.07 | 0.10 | 0.089 | 0.134 | 0.06 |
| ctrl_sft | 0.12 | 0.12 | 0.13 | 0.20 | 0.123 | 0.185 | 0.05 |

  The first correct content ("Carbon dioxide gas is a gas...", "Plants need
  sunlight to grow"), but capitals and numbers stay wrong and greedy replies
  loop; several hits are lenient containment matches inside loops
  (`receipts/qa_capacity_ctrlsft.md`). Progress, not a model to ship.
- moe_sft, big_sft (chat_fw_big, d768 L12 from scratch) and pk_base_v1's SFT
  are compared in the next section.
- Training and serving templates are identical (8,986/8,986 multi-turn SFT
  examples rebuild to /chat's token ids; the SFT loader appends and trains
  <|endoftext|>), so the gap is data and capacity, not serialisation.

## What it says

At 14.6M parameters and ~195M tokens the model has room for the register of
general text and the shape of a Q&A turn, not for facts. Q&A needs capacity:
a larger dense model (d and d_ff multiples of 256, so Q4_K applies), or the
FineWeb base upcycled to MoE (8 experts, top-2), which adds capacity at the
same per-token compute. Repetition in answers is a decoding and
preference-training target (the judged DPO stage), not a data one.

## The general chat models side by side (25 Oct 2026)

One RunPod L4 campaign (`tools/colab_chat/general_chat_v1.json`, paperkiln 0ad646d5, ember
aa50a61 built on the pod, TF32 off; receipts `receipts/general_chat_1025/`, artifacts
`C:/ml_artifacts/general_chat/`). The winner rule (PLAN.md, "Choosing the winner", and its
amendment A1) was fixed before any of these numbers was read; `winner.py` applies it.

### pk_sft_v1: pk_base_v1 + lever 1 at T 512

- **Data:** lever 1's 16k-vocabulary set as conversation rows (`tools/lever1_conversations.py`):
  one row per dialogue, its longest kept per-turn row parsed back, every shorter row checked as a
  prefix; 313,606 dialogues carrying all 463,695 replies of the per-turn file (sha256 b7a43fb5...).
  `train.sft_loss` {"turns": "all"}: 313,606 rows, none over T or cut, 80.2M trained tokens.
- **Run:** init pk_base_v1 (1251f204...), 18,620 steps x 16 (one pass in expectation over the
  297,926 training rows), lr 1e-4 (big_sft's), warmup 200, cosine to 0.1, BF16 GEMMs + flash
  attention, 2.47 h at 17.1k tokens/s.
- **Held-out loss** (512 of the 15,680 held-out dialogues): 1.731 (step 2,000), 1.656, 1.608,
  1.573, 1.551, 1.533, 1.520, 1.513, 1.508 (step 18,000); still falling at the end.

### Results (evals2, the run of record; evals1, without pk_sft_v1, matched it exactly)

| model | params | simple-QA | fact | science | arith | turn | frozen ref F1 | frozen ref bits/byte | frozen repeats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| live (chat_sftfinal_v3) | 14.6M | 2/60 | 0.00 | 0.00 | 0.00 | 0.20 | 0.162 | 0.565 | 0.00 |
| ctrl_sft | 14.6M | 7/60 | 0.12 | 0.13 | 0.00 | 0.20 | 0.122 | 0.831 | 0.05 |
| moe_sft | 70M (E8 top-2) | 7/60 | 0.12 | 0.13 | 0.00 | 0.20 | 0.100 | 0.815 | 0.02 |
| big_sft | 91M | 11/60 | 0.08 | 0.20 | 0.20 | 0.40 | 0.102 | 0.737 | 0.03 |
| **pk_sft_v1** | 97.5M | **21/60** | **0.44** | **0.33** | 0.00 | **0.50** | 0.155 | 0.600 | 0.01 |

`chat_prompts_v1` per source (temperature 0.7, 64 new tokens):

| model | ultrachat clean stop | length stops | protocol fails | repeats | ref F1 | ref bits/byte (cut, dropped) | everyday clean stop |
|---|---:|---:|---:|---:|---:|---|---:|
| live | 1.000 | 0.000 | 0 | 0.014 | 0.013 | 2.703 (68, 22) | 1.000 |
| ctrl_sft | 0.600 | 0.400 | 0 | 0.453 | 0.062 | 1.098 (70, 18) | 1.000 |
| moe_sft | 0.520 | 0.480 | 0 | 0.547 | 0.058 | 1.100 (70, 18) | 1.000 |
| big_sft | 0.307 | 0.693 | 0 | 0.547 | 0.080 | 0.892 (70, 18) | 1.000 |
| pk_sft_v1 | 0.117 | 0.883 | 0 | 0.558 | 0.122 | 0.656 (36, 9) | 1.000 |

The ultrachat bits/byte of pk_sft_v1 (T 512) cover more of each long reference than the T 256
models' (36 cut and 9 dropped against 70 and 18), so that column compares only roughly. Full
tables, including ctrl_sft's per-source numbers for ember: `receipts/general_chat_1025/evals2/tables.md`.

### The winner

- **Registered rule: live.** Every other model fails the clean-stop gate on ultrachat, and no
  model ever ends a reply on a role marker: every non-clean stop is the 64-token budget cutting a
  long answer (pk_sft_v1 averages 45 words there).
- **Amendment A1 (gate on protocol failures only; written after the 23 Oct run, so post hoc):
  pk_sft_v1**, 21/60 against big_sft's 11/60, a lead of 10 questions where the rule needs 4.
- Jonathan picks which verdict stands. pk_sft_v1 is the model the preference stage is prepared on.

### Parity and cost

- **Greedy parity** (contract point 1; `tools/colab_chat/chat_parity_vm.py`, ember's own
  `tools/chat_parity.py`, 40 frozen prompts, 60 new tokens): ember aa50a61 against `mtstudio chat`
  (CPU, full recompute, FP32) on pk_sft_v1.gguf (4d53c7ff...): CUDA (TF32 off) and CPU paths
  each 40/40 on reply text, token count, stop reason and prompt tokens against llama3_bpe.py, 40
  distinct replies (`receipts/general_chat_1025/parity/`).
- **Cost:** one RunPod L4 (EU-RO-1, $0.59/h), 15:40-21:02, $3.19 (cap $8). Uploads 46 min (2.1 GB);
  training 2.47 h; evals1 9 min, evals2 14 min; parity 44 min, almost all of it mtstudio's CPU
  reference.
- evals2 first stopped when chat_eval's reference-scoring runs (each a full checkpoint, 1.4 GB
  at d768) filled the 40 GB volume; it was rerun from scratch after the space was freed, and
  chat_eval now deletes those runs (paperkiln a77885d3).

### Reading

- pk_sft_v1 is the first paperkiln model to answer common facts: Paris, Rome, London, the
  Pacific, Jupiter, the Sun, honey, gravity, sunlight. It still gets numbers wrong (a spider
  "has 4 legs"), scores 0/10 on arithmetic, invents details (Rome "in the Italian Alps"), and
  loops in long answers (repeats 0.56 on ultrachat). Several hits for every model are lenient
  containment matches inside wrong or looping replies (`receipts/general_chat_1025/evals2/qa_replies.md`).
- On the frozen TinyChat prompts it is close to the live model in reference F1 (0.155 vs 0.162)
  and reference bits/byte (0.600 vs 0.565), with 1% repeats: the first FineWeb-line model that
  does not lose chat quality to gain Q&A.
- The step from big_sft (11/60) to pk_sft_v1 (21/60) comes at about the same size, so it is data:
  0.98B tokens with 27-47% dialogue against ~0.5B tokens of FineWeb, then 300k dialogues of SFT
  against 48k examples. MoE at equal tokens (moe_sft) bought nothing over ctrl_sft here.
- Repetition in long answers is the preference stage's target; its repeat gate starts from
  pk_sft_v1's 0.01 (frozen) and 0.56 (ultrachat).
- Reproducibility: the four earlier models' simple-QA replies are identical in evals1 and
  evals2, and 239 of 240 equal the 23 Oct Colab run's (TF32 on there, off here; one moe_sft
  reply differs). Sampled chat replies differ between evals1 and evals2 in 2 of 1,100 (one
  moe_sft frozen-prompt reply, one big_sft chat_prompts_v1 reply), which moves moe_sft's frozen
  reference F1 from 0.101 to 0.100; every other summary field is equal.
