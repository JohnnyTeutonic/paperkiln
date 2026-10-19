# TinyChat chat model: results (18-19 Oct 2026)

*Descriptive, not a pre-registered study. Receipts in `receipts/`; weights
and GGUFs in `C:\ml_artifacts\chat_base\` (outside the repo). Served by
ember.cpp (BPE tokenizer, greedy parity 40/40 with mtstudio).*

## The chain

| Stage | What | Result |
|---|---|---|
| Base | llama-tiny d384, 6 layers, 6 heads, T 256, 14.65M params; TinyChat (starhopp3r, pinned) as plain dialogue; byte BPE 8192; 12,000 steps x 16 x 256 (49M tokens), AdamW lr 1e-3 constant; Colab L4 | val loss 3.72 (step 250) -> 1.99 (best 1.989); 2,526 tokens/s over the run |
| SFT | 24,000 examples in the served template (`<|user|>U<|assistant|>A<|endoftext|>`), masked to the reply, 1500 steps, lr 3e-4; dialogues carrying source role markers dropped (ledger 0003) | val loss 1.90 at step 1000, 1.94 at 1500 |
| Preference | SPIN pilot on the step-1500 preview (reference reply chosen, own sample rejected; DPO beta 0.1 + NLL 0.2), lr 3e-5 / 1e-4 / 1e-5 | **retired**: held-out pair accuracy 0.89-0.91, but more off-topic replies, and at 1e-4 repetition and collapse onto one opening (ember side-by-side) |

The base ran through three engine builds as they landed: 6.4 s/step, then
device blocked attention (4.6 s from step 5000), then the full device op set
with residency, deferral and cuBLAS FP32 (1.82 s from step 6500). Each switch
resumed from a checkpoint with its optimizer state, and step losses were
unchanged in every same-sitting comparison.

## Chat quality on the 200 frozen TinyChat prompts (first 100 used)

Sampled by ember at temperature 0.7, one reply per prompt, the same seeds for
every model (`tools/chat_eval.py --ember-bin`).

| Model | Clean stop | Repeats | Words | Prompt overlap | Ref F1 | Ref bits/byte |
|---|---:|---:|---:|---:|---:|---:|
| base step 1500, no SFT (mtstudio sampler) | 0.36 | 0.19 | 47.8 | - | - | 1.250 |
| SFT on base 1500 | 1.00 | 0.00 | 13.6 | 0.085 | 0.109 | 0.745 |
| SFT on base 5250 | 1.00 | 0.00 | 14.0 | 0.112 | 0.148 | 0.620 |
| **SFT on base 12000 (final)** | **1.00** | **0.00** | 14.1 | **0.155** | **0.164** | **0.565** |

Prompt overlap and reference F1 are content-word proxies for topic and
relevance, not judges. Each base step bought topicality: replies engage with
the message ("I understand why you feel that way, can you tell me more about
what is happening?") where the early preview drifted.

## Simple QA probe (`corpus/eval/simple_qa_v1.jsonl`, greedy)

| Model | Facts (25) | Science (15) | Arithmetic (10) | Turns (10) |
|---|---:|---:|---:|---:|
| SFT on base 1500 | 0.00 | 0.00 | 0.00 | 0.20 |
| SFT on base 5250 | 0.00 | 0.00 | 0.00 | 0.30 |
| SFT on base 12000 | 0.00 | 0.00 | 0.00 | 0.20 |

No TinyChat model answers a factual or arithmetic question ("What is the
capital of France?" -> "Then I am surprised by how many different types of
instruments..."): TinyChat is emotional small talk with no facts. That is the
gap the FineWeb-Edu base (`experiments/chat_general/PLAN.md`) is for.

## What carries forward

- The serving chain works end to end: paperkiln trains, GGUF exports carry
  the BPE tokenizer, ember serves with token parity.
- Preference optimisation on SPIN pairs is not used again for chat; the next
  preference stage is judge-ranked (`tools/posttrain/chat_judge.py`) and must
  pass the repeat gate and a side-by-side.
- Licence: TinyChat is CC-BY-NC-4.0, and these models inherit it.
