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
   reversed, and only agreeing ranks are kept. About $0.03 per prompt.
   **Budget: $100 total (Jonathan, 19 Oct 2026)**, enforced by
   `chat_judge.py --max-usd 100` over the cache's whole recorded spend: a
   200-prompt pilot (~$6) measures cost and agreement, then ~3,000 prompts
   (~1,700-2,000 pairs; up to twice that with `--pairs-per-prompt 2`). Best vs
   worst pairs, length-matched;
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
- Judge: $100 cap, enforced by the tool.
- My work before launch: fetcher + vocab + QA probe + listwise judge tool,
  about 3 hours, none of it touching Colab.

## Choosing the winner (rule fixed 25 Oct 2026, before any result was read)

Fixed before reading any number from moe_sft, big_sft, pk_base_v1's SFT or
the 23 Oct Colab run (`C:/ml_artifacts/chat_eval_1023`, which finished but was
never read). The winner is the model the preference stage starts from.

- **Candidates:** the live model (chat_sftfinal), ctrl_sft, moe_sft and
  big_sft (chat_fw_big); then pk_base_v1's SFT on lever 1's set at T 512.
- **Measurements of record:** one run of `tools/colab_chat/chat_eval_vm.sh`
  (one paperkiln commit, one ember build, every candidate in the same run)
  on a cloud L4: chat_eval on the 100 frozen TinyChat prompts
  (`tinychat_hf_prompts_v1`) and on the 200 prompts of `chat_prompts_v1` per
  source, and qa_eval on `simple_qa_v1` (60 questions, greedy). When
  pk_base_v1's SFT arrives, the run is repeated with all five models, and
  that run decides.
- **Gate, the turn protocol:** clean stop at least 0.90 on the frozen
  prompts and on each source of `chat_prompts_v1`. A model below it cannot
  win.
- **Primary, Q&A:** simple-QA `all`. The gated model with the most correct
  answers wins if it leads every other gated model by at least 4 of the 60
  questions (0.067; at these rates one standard error is ~0.045).
- **Tie** (another gated model within 3 questions of the top): among the
  tied models, the lowest reference bits per byte on `chat_prompts_v1`'s
  ultrachat source (Q&A references, not the everyday greetings repeated in
  training) wins. If two are within 0.01 bits/byte, the lower repeat
  fraction on that source wins, then the smaller model.
- **Comparability:** bits per byte do not depend on the vocabulary, but
  chat_eval cuts each reference to fit the model's T, so a T 512 model scores
  more of each long reference than a T 256 one; the counts of cut and dropped
  references are reported beside every bits-per-byte figure. Sampling skips
  a prompt when prompt + 64 new tokens exceeds 256 tokens for every model
  (`chat_sample_ember.py --ctx` default), so the sampled prompt sets match
  up to tokenisation.
- The other metrics (reference F1, prompt overlap, repeats, words) are
  reported, not used. The preference stage's own gates (PLAN above: no more
  repetition than its SFT start, the ember side-by-side) are unchanged.

**Amendment A1 (25 Oct 2026, written after reading the 23 Oct run, before the
run of record).** The 23 Oct run shows the gate above measures reply length,
not the turn protocol: on `chat_prompts_v1`'s ultrachat source every reply
that did not stop cleanly hit the 64-token budget (`stop` "length": ctrl_sft
27, moe_sft 38, big_sft 52 of 75), and no reply of any model ended on a
generated role marker (`stop` "turn"). Under A1 the gate counts protocol
failures only: a reply fails it when it ends on a generated role marker or
contains one; a reply cut by the budget does not. The gate is then a
protocol-failure share of at most 0.10 on each prompt set and source. Both
verdicts are reported: the rule as first registered, and under A1. A1 was
written knowing the 23 Oct simple-QA numbers too, so it is a post-hoc
change, and Jonathan chooses which verdict stands.
