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
- Still running: moe_sft, and a d768 L12 dense model (~97M parameters,
  60,000 steps x 8,192 tokens) from scratch with its SFT.
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
