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
   logged-in Claude Code CLI in print mode, no API key handled, run from an
   empty directory with the rubric as the whole system prompt and no tools.
   Listwise: one call ranks a prompt's 4 sampled replies, run twice with the
   order reversed, and only agreeing ranks are kept. Measured on the 128-token
   pilot: $0.024 per call, $0.048 per prompt.
   **Budget: $200 total (Jonathan, 28 Oct 2026)**, enforced by
   `chat_judge.py --max-usd 200` over the cache's whole recorded spend. Two
   200-prompt pilots measure cost and agreement (128 and 256 new tokens;
   RESULTS.md, "Preference pilot"), then the full run takes every eligible
   smol-smoltalk test prompt (3,021 at 256 new tokens and T 512) if the
   256-token pilot's candidates mostly finish (more than half of its runs
   end on end-of-text; Jonathan, 28 Oct 2026). Best vs worst pairs where
   both orders agree, length-matched; DPO+NLL at a gentle rate (below).

   **Sampling the candidates** (`tools/colab_chat/pref_pilot_v1.json` at 128
   new tokens, `pref_pilot_v2.json` at 256, `pref_sample_vm.py`,
   `tools/posttrain/sample_candidates.py`): the winner, pk_sft_v1 (Jonathan's
   pick, 27 Oct 2026), on ember's CUDA server with `EMBER_GEMM_TF32=0` and
   `EMBER_DETERMINISTIC=1` (a recorded seed draws the same reply again on the
   same GPU and build), `EMBER_SPEC` and `EMBER_DRAFT` unset. The 128-token
   pilot ran at ember main 7526469; the 256-token pilot and the full run run
   at ember main 9d993f48, whose engine is b49eb5c's (conversation schema v2
   with paperkiln/1 byte-identical, ec3201b's tiled split-K decode attention
   gated in default and deterministic mode). Both carry e57c0fb's
   `utf8_valid`: /chat answers a reply cut inside a UTF-8 character with
   U+FFFD, where fe01b35 and bace837 answer HTTP 500; generated ids are
   unchanged. aa50a61 (the tiny-chat default) and fe01b35 are ancestors, and
   no gate holds sampling to that line: TF32 is off by the same switch, and
   pk_sft_v1's greedy parity gate (aa50a61) and its IF-tiny run (fe01b35) are
   records of those builds, not pins. Replies holding U+FFFD follow
   sample_candidates.py's rule `trim_cut_tail_drop_other`: a reply cut by the
   token budget inside a character is a cut reply, trimmed to its last whole
   character; any other U+FFFD (or a cut with no whole character before it)
   drops that run from the candidates (the run stays in the record), and a
   prompt left with fewer than two distinct candidates is skipped as
   `fewer_than_two_valid`. The full run continues the 256-token pilot's
   candidate file (`pref_sample_vm.py` RESUME_PREFIX), so its first prompts
   are the pilot's and their verdicts are already in the cache.

   **The DPO arm and its gate (fixed 28 Oct 2026, before the full run's pairs
   or any DPO number existed).**
   - *Pairs:* the full run's (256-token candidates only), one per prompt, best
     vs worst where both orders agree, within 1.5x in words. A reply gets an
     end-of-text only if it stopped on one (`chosen_stop` / `rejected_stop`
     from `chat_judge.py pairs`; mtstudio trains a cut reply as the prefix it
     is). The last 5% of the file are mtstudio's held-out pairs.
   - *Training (pk_dpo_v1):* mtstudio `provenance_po` from pk_sft_v1, the
     reference pk_sft_v1 frozen; weight_mode uniform; beta 0.1; NLL 0.2 on the
     chosen reply; AdamW, lr 2e-6, warmup 20 steps, cosine to 0.1; batch 8
     pairs; two passes in expectation over the training pairs; FP32; the
     final checkpoint.
   - *Gate* (pk_dpo_v1 against pk_sft_v1, both in one `chat_eval_vm.sh` run on
     one ember build): (1) protocol failures at most 0.10 on every prompt set
     and source (amendment A1); (2) repeats no higher than pk_sft_v1's on the
     frozen TinyChat prompts and on `chat_prompts_v1`'s ultrachat source;
     (3) simple-QA at most 3 questions below pk_sft_v1's (one standard error
     is about 2.7); (4) ember parity: greedy 40/40 against `mtstudio chat` on
     CUDA (TF32 off) and CPU; (5) **the judged side-by-side**
     (`tools/posttrain/side_by_side.py`): one reply from each model to each of
     the first 300 eligible smol-smoltalk val-split prompts (held out from
     the SFT and from the pairs), temperature 0.7, seed 71 + prompt index,
     256 new tokens, on ember's CUDA server in deterministic mode, ranked by
     the same judge in both orders; a prompt is won when both orders agree,
     a tie otherwise. pk_dpo_v1 wins the side-by-side when its share of the
     decided prompts exceeds 0.5 with the Wilson 95% lower bound above 0.5.
   - pk_dpo_v1 replaces pk_sft_v1 only if all five hold. Otherwise the arm is
     recorded as a negative, with every number, and pk_sft_v1 stays.

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
