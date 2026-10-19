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

## What it says

At 14.6M parameters and ~195M tokens the model has room for the register of
general text and the shape of a Q&A turn, not for facts. Q&A needs capacity:
a larger dense model (d and d_ff multiples of 256, so Q4_K applies), or the
FineWeb base upcycled to MoE (8 experts, top-2), which adds capacity at the
same per-token compute. Repetition in answers is a decoding and
preference-training target (the judged DPO stage), not a data one.
