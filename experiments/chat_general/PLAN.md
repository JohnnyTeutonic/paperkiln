# General-text chat model: plan (19 Oct 2026)

*For the coordinator (session b1) and Jonathan. Nothing launches until the
current Colab session (pk-chat) has finished and stopped: one Colab session
at a time.*

## Goal

A paperkiln model that holds a simple conversation **and** answers simple
questions. TinyChat alone cannot give the second: it is 60M tokens of
emotional small talk with no facts or arithmetic. So the base learns from
general text (FineWeb-Edu, Jonathan's choice via b1), then chat SFT on
TinyChat, then a judged preference stage.

## What compute buys today

Measured on a Colab L4 at the chat shape (d384, 6 layers, T 256, batch 16)
with the 18 Oct engine (device op set, residency, deferral, cuBLAS FP32):
**1.82 s per step = about 2,250 tokens/s** while sharing the GPU with an SFT
run. Batch does not buy throughput (batch 64 costs 4.9x batch 16), so tokens/s
is roughly fixed per model size until more host work moves to the device.

| Option | Params | Tokens/s (est.) | Tokens in 24 L4-hours | Tokens per param |
|---|---:|---:|---:|---:|
| **A. d384 L6 (today's shape)** | 14.6M | ~2,250 | **~195M** | ~13 |
| B. d512 L8 | ~33M | ~1,000 (est., cost/token ~2.2x) | ~85M | ~3 |

B would be badly under-trained in a night; A lands near the compute-optimal
ratio (~20 tokens/param). **Recommendation: A**, as the first general-text
base. A bigger shape follows when the next engine pass lifts tokens/s, and
the same pipeline then scales up unchanged.

## Data

- **FineWeb-Edu `sample/10BT`**, pinned revision `87f09149` (registry
  `fineweb_edu_v1`, odc-by). 14 parquet shards, 28.5 GB, ~750M GPT-2 tokens
  each. **One shard (`000_00000.parquet`, 2.15 GB) covers 195M of our
  tokens with room to spare.** New fetcher `corpus/fetchers/fineweb_edu.py`:
  pinned shard, sha256-pinned on first fetch, documents split train/val by a
  hash of their id, `pretrain_train.txt` with `<|endoftext|>` between documents.
- **Vocabulary:** a new byte-level BPE (8192 entries, same specials) trained
  on a mix of FineWeb-Edu (80%) and TinyChat (20%), so the general text and
  the chat SFT both tokenise well. Bytes per token reported on both held-out
  sets; registered as `vocab_fineweb_tinychat_bpe8k_v1`.
- The base never sees TinyChat dialogue; that is SFT's job.

## Runs (one Colab session at a time)

1. **Probe, 15 min:** tokens/s of option A alone on an L4 with the FineWeb
   vocabulary, to fix the step count. If it is well above 2,250, the step
   count rises to fill the same hours.
2. **Base:** option A, ~195M tokens = ~47,500 steps at batch 16 x 256,
   ~24 L4-hours, over two nights. The existing runner (checkpoint every 250
   steps, relayed home, resume on reclaim) carries it across reclaims and
   nights.
3. **SFT:** the same 24k TinyChat examples in the chat template, 1500 steps
   (~45 min on the L4 at today's speed).
4. **Preference (RLAIF):** Claude (`claude-opus-5-5`) as judge through the
   logged-in Claude Code CLI in print mode, no API key handled. Measured cost
   with a trimmed system prompt and no tools: **$0.0117 per call**. Listwise:
   one call ranks a prompt's 4 sampled replies, run twice with the order
   reversed, and only agreeing ranks are kept. **1,000 prompts = 2,000 calls,
   about $25**, needing Jonathan's OK. Best vs worst pairs, length-matched;
   DPO+NLL at a gentle rate. The arm fails if it repeats more than its SFT
   start (`chat_eval.py` repeat gate) or loses the ember side-by-side.

## How it is judged

- `tools/chat_eval.py` on the 200 frozen TinyChat prompts: clean stop,
  repeats, reference bits/byte, prompt overlap, reference F1.
- **New: a frozen simple-QA probe** (`corpus/eval/simple_qa_v1.jsonl`, ~60
  questions with short answers, written for this and frozen before any model
  sees it): exact-match and contains-answer rates. This is the number that
  says whether general text bought Q&A.
- Held-out FineWeb bits/byte, and the ember side-by-side.

## Cost and time

- Colab: ~24 L4-hours for the base + ~1 hour for probe and SFT. Check the
  units against the Pro+ allowance before launch.
- Judge: ~$25 per 1,000 prompts (Jonathan's OK first).
- My work before launch: fetcher + vocab + QA probe + listwise judge tool,
  about 3 hours, none of it touching Colab.
