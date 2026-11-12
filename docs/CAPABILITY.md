# What the current chat models can do: the measured record

**Version 2, 13 November 2026.** Written by Claude Opus 5.5 from stored receipts. Nothing was rerun
for this page. A later version replaces these numbers in place and takes the next version number.

This page answers one question: what do paperkiln's chat models measurably do now? It covers only
what has been measured, says which instrument measured it, and compares models only where the
protocol was the same. The full record is the programme registry,
[`TINY_CHAT_RESEARCH_PROGRAMME.md`](https://github.com/JohnnyTeutonic/paperkiln/blob/36523b27c93223a48cb10503adfc8bc6bf0a8c23/experiments/tiny_chat_programme/TINY_CHAT_RESEARCH_PROGRAMME.md)
on branch `claude/chat-screens` (sections 4, 7, 9 and 10b), and the protocol it cites.

**How to read it**
- **Three kinds of statement are kept apart.** Established fact is a count from a receipt or a
  registered verdict. A reading is an assistant's interpretation and names its producer and date.
  Jonathan's reads are marked as his.
- **Strict format and content are different results.** tinychat-eval's exact checks score an
  answer-and-format contract: "The meeting is on Thursday." fails a question that asked for the day
  alone. They appear in their own table, never merged with judged or rule-scored content.
- **Where receipts live.** "Workspace" paths are in the project's private research workspace, under
  `AI_ML/chat_screens_claude/` unless another folder is named; "PROTOCOL §n" is that folder's
  `PROTOCOL.md`. Run folders are under `C:/ml_artifacts/chat_screens_claude/runs/` on the project
  machine. Neither is in this repository. The registry above carries every verdict in public.
- **One seed is a screen, not a claim.** Most rows are one training seed. Two seeds exist only for
  S67's packet, S71's packet, S72's packet and the open-base arm of A29.

## 1. The current best model: S71

**By the programme's own criteria, S71 is the current best model of our own.**
- **Only `ready`.** It is the only model to read `ready` under its registration (A26, PROTOCOL
  §41.2). Every leg held: identity, ended role-play, stance, role-play breaks, loops, `cue_v001`,
  13-turn J1, told recall and attribution.
- **Highest relevance on our base.** On the 532-turn wide J1 it reads 31.7 points, the highest of any
  model on base A. The registry's §10b states that "S71 stays the candidate recipe and the best
  model read". After A28 it also states that "S71's recipe stays the best own-base recipe, at a
  2-point seed spread".

**What qualifies that.**
- **Seed 2 missed three bars.** At seed 2 the same packet reads `wide_only` (A28, §44.2): it misses
  the ended twin (3 of 8), role-play breaks (15 of 48) and the loop bar (3 of 212). So `ready` is
  one seed's reading.
- **Not separated from S70.** On the wide J1, S71 against S70 is +1.1 points (−0.4 to +2.6).
- **Not the best on strict format.** It passes 1 of tinychat-eval's 11 strict cases, against 4 for S70
  and 3 for S71s2 (section 4.1).
- **An open base scored higher, and is out of scope.** S71's packet on SmolLM2-135M-Instruct read
  higher (A29). That model is an open base, and the programme does not adopt an open base as its
  product (Jonathan's ruling of 11 Nov 2026, PROTOCOL §48). It stands here as a diagnostic, not as
  a candidate.

### Lineage

| Stage | What was done | Identity (SHA-256) | Receipt |
|---|---|---|---|
| Vocabulary | 16,384-token BPE in the Llama 3 scheme | `vocab.gguf` `0b880aa073a7e27c0770d5ecc5215b55d6d415080d20bd3dc7b169ecabadf198` | run `input_manifest.json` |
| Shape | Llama-style, 8 layers × 768, 12 heads, d_ff 2,240 SwiGLU, context 1,024, tied embedding, 4 per-turn state tokens (k = 4, causal); 72,761,088 unique parameters | | registry §2; S71 `result.json` (`params`) |
| Pretraining: base A (`pt_b1_r1`) | Arm A of a registered race against 12 × 768 (B2, §30), chosen by the frozen rule (§34). 60,000 steps × 16 rows × T 1,024 = 983.04M positions; lr 5e-4, warm-up 2,000, cosine to 10%, annealing tail from 0.85; seed 1; RunPod L4; trainer numerical source `5506e6df` | weights `4b4afcc212e62a43bd767e3600ae461afdca63beb18fc046db7954b516ffdf97`; GGUF `5328dc840b8a58d686a51ea4ffef0c15ba1d49a9826c8358979659129d09af3e` | workspace `AI_ML/tiny_chat_benchmarks/b1_stage1/receipts/a_winner_continuation_20261108/final_home_20261109.json`; PROTOCOL §29, §30, §34, §35 |
| Pretraining corpus `mix_b1` | 6.6 GB by bytes: FineWeb-Edu shards 0 and 1; UltraChat; SoDa; everyday dialogue; the pretraining split of 13 human dialogue sets; six long dialogue sets. Main part 85% at 73/27, tail 15% at 53/47. Every dialogue source filtered against the frozen evaluation sets | `mix_b1.main.ids` `ea8b4242a7ead30959fd1cf5ab17b51e2fe8a0160e7a966a83b3da22466e3b91`; `mix_b1.tail.ids` `0585df71706e872d2e3144fde315c94aa3a10cecf8b11db2d25f42efb0ebfc94` | workspace `AI_ML/tiny_chat_benchmarks/b1_stage1/stage0/tc_b1_pins_observed.json`; PROTOCOL §29 |
| SFT1500 (`B1SFT`, `tc_sft_b1_r1`) | S-full's SFT1500 recipe and data unchanged: 1,500 steps, lr 3e-4, batch 16, T 1,024, warm-up 100 then cosine to 0.1, fresh AdamW, seed 1, `sft_human.jsonl` | data `4785e1494d34b81970d947e3488897afcdd36d19a8625b6670dfefcc69b42ecb`; weights `86de03a5b1455ebe13792ac0be944b3eb14b851567ffb89249ee4fae2e1c8dbd`; GGUF `156cb84f7c27b34c5d17e70829976874ff623c90d52cbd7abcc8d3a9f4a708dc` | PROTOCOL §35.2, §36 ("The start") |
| S71's SFT | Packet `S71.jsonl`: S2's rows with repeated replies masked (71.8% of sampled targets), the K2 pool (23.9%) and persona pool v2 (4.3%), small-talk and greeting replies unsupervised. 1,024 steps, lr 1e-4, warm-up 64, cosine to 10%, batch 16, clip 1.0, fresh AdamW, seed 1, trainer `1ccb693f`; run `a26-s71-20261111` on a Colab L4 | packet `37afd630ca0a4a6977e2431fb9d5de6ebfcf6a62171eca66c071c24812a2de05`; weights `ab49046a163acebbf9d9d4b545804b3497e06b5bdc4a20724434f8c2115b7dde`; GGUF `01ffd913d643e45441c661fafd59a53dfb833df85f4d6f9ce3f5a2eaeee63874`; train log `7057d5fe3e5648b6d86e8a47fda87967d98e1a80ae46283e0c629f8e064ee3f9` | run folder `a26-s71-20261111/` (`input_manifest.json`, `S71/manifest.json`, `S71/spec_S71.json`); PROTOCOL §41, §41.2 |
| S71 at seed 2 (S71s2) | The same packet and recipe at seed 2; run `a28-s71s2-20261111` | weights `5c3408119be2763b7a95eb8c164ac1b8a91b1322ffbfc4a71b790c23158d6df0`; GGUF `b763fc2fbabc9fb2356d5ead78013b8e4f27066c87a15c3078de08eb38c88c70` | run folder `a28-s71s2-20261111/S71/manifest.json`; PROTOCOL §44.2 |

**Where S71's training text comes from.** Part of the packet was written by Claude Opus 5.5
subagents: the K2 pool and both persona pools. Other rows come from programmatic generators. Claude
models also wrote `cue_v001` and the persona instruments, so a shared phrasing style may carry part
of any gain on them. The 13-turn J1 (tinychat-eval's cases, judged by Gemini) and chatflow's told
recall come from instruments that model family did not write. That bounds the confound without
removing it (PROTOCOL §37.3, §41).

## 2. The instruments

| Instrument | What it scores | Kind | Denominator | Protocol | Defined in |
|---|---|---|---|---|---|
| tinychat-eval release v002, dev exact checks | answer-and-format contract on basic Q&A, immediate and delayed recall, correction, missing information and instructions | strict format | 11 scored cases of the 13 dev cases (2 are judged only). Reports count 22 checks: each case twice, with identical greedy repeats | suite `a28bbe95…`, harness `evaluate.py` `364ffe89…`; temperature 0, top-k 1, seed 123, at most 32 tokens; history generated by the model | [tinychat-eval](https://github.com/JohnnyTeutonic/tinychat-eval) README |
| the same, `semantic_content` | whether the expected value is mentioned, and negation on missing-information cases | mention, not correctness | 9 cases with an expected value; 2 missing-information cases | as above | tinychat-eval `evaluate.py` |
| J1, 13 turns | relevance and coherence of every turn of five open conversations from that suite | judged (Gemini `gemini-3.1-pro-preview`, temperature 0, rubric `be533789…`), 0–4 | 13 turns; relevance as points of 52 | per turn, full history visible to the judge | registry §3 |
| wide J1 | relevance on scripted persona, contrast and chatflow conversations, eight families | judged, J1's call and rubric | 532 planned turns; level is the mean of eight family means, shown as points (mean × 13) | 95% frame bootstrap, 10,000 draws; replies are stored greedy replies | PROTOCOL §42; `rel_wide2.py` |
| `cue_v001` | content-tied acknowledgement, activity-tied suggestion, correction across 7 frames, "just X" format; component 5 request discrimination | deterministic rules, sealed set | components 1–4: 174 items; component 5: 120 | greedy on the model's own history, CPU | PROTOCOL §37; `cue_v001/README.md` |
| `persona_v003` | who it is (name, AI status), stance on personal questions, role-play kept until the user ends it, loops | rule-scored labels | identity 92 turns, stance 60, role-play 48, loop count over 212 turns | greedy, own history, two replays | PROTOCOL §38.2, §41 |
| `contrast_v002`, ended twin | after the user ends a role-play, does the reply return to the assistant | rule-scored | 8 items | as above | PROTOCOL §24.3, §26 |
| chatflow told recall | recall of a fact the user told earlier, on the model's own history | rule-scored labels | 108 items | greedy | PROTOCOL §26, §36 |
| attribution | the user's fact answered as the assistant's, or a recall question answered with the assistant's identity | label count | count over own-history finals | greedy | PROTOCOL §36 |
| knowledge guard | PIQA accuracy, LAMBADA accuracy, bits per byte | log-probability scoring through ember `/score` | PIQA 205 items and LAMBADA 199 items of `bench_v1` (`978548d1…`); 200 LAMBADA segments for bits per byte | | run folder `eval/<model>/knowledge/` |
| `bench_v1`, full | HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, LAMBADA, BLiMP | log-probability | 28,548 items | ember `/score`; used for bases and SFT1500 | PROTOCOL §29 |

**Limits that apply to every judged number.**
- **One judge.** On 7 Nov 2026's 34 sheet turns, J1 read 0.44 points above a blind three-rater
  ensemble (`analysis/rater_ensemble.md`).
- **Stable on retest, lenient against Jonathan.** On the wide J1's 60 retest turns the judge was
  identical (weighted kappa 1.00). Against Jonathan's blind ratings of S-full it is lenient:
  on-topic but wrong answers get 1–2 where he gave 0 (registry §4b).
- **13 turns cannot rank models of this level.** S67's packet read 39 and 28 at two seeds on 13
  turns, and 26.4 and 25.9 on the wide J1 (PROTOCOL §42.2).

## 3. S71: what is measured

Seed 1 is the model; seed 2 is shown beside it as the only measure of seed spread.

| Dimension | S71 (seed 1) | S71s2 (seed 2) | Instrument and denominator | Receipt |
|---|---|---|---|---|
| Relevance, wide | 31.7 points; mean 2.44 of 4 (95% 2.30 to 2.58); 532 of 532 turns judged | 29.7; 2.28 (2.15 to 2.42); 532 of 532 | wide J1 | `analysis/rel_wide2_read_s71.json`; `analysis/rel_wide4_read.json` |
| Relevance by family, mean of 4 | chat 3.34, correction 2.85, recall 2.56, role 2.44, names 2.44, identity 2.31, abstain 1.90, follow-up 1.66 | chat 3.00, correction 2.67, recall 2.49, role 2.31, identity 2.25, abstain 2.20, names 2.06, follow-up 1.28 | wide J1 families | `analysis/rel_wide4_read.md` |
| Relevance, 13 turns | 36 of 52 points; 8 of 13 turns at 3 or more | 32 of 52; 6 of 13 | J1 | `analysis/a26_score.json`, `a28_score.json` |
| Coherence | mean 3.15 of 4 | 3.00 | J1, 13 turns | as above |
| Told recall | 62 of 108 | 58 of 108 | chatflow, own history | as above |
| Correction | `cue_v001` component 3: 21 of 42 | 26 of 42 | `cue_v001` | as above |
| Content-tied replies | `cue_v001` 106 of 174 (acknowledgement 42 of 48, suggestion 31 of 42, correction 21 of 42, format 12 of 42); request discrimination 75 of 120 | 118 of 174 (43, 35, 26, 14); 78 of 120 | `cue_v001` | as above |
| Identity | 80 of 92 correct, 0 denying it is an AI, 3 wrong names | 81, 0, 2 | `persona_v003` | as above |
| Stance on personal questions | 35 of 60 engaged | 37 of 60 | `persona_v003` | as above |
| Role-play | 6 of 48 turns break to the assistant; 17 stay in character | 15 break; 14 in character | `persona_v003` | as above |
| Ending a role-play | 5 of 8 return to the assistant | 3 of 8 | `contrast_v002` ended twin | as above |
| Loops | 0 of 212 turns say "I'm an AI" twice; 0 of 212 repeat a sentence | 3 of 212 | `persona_v003` | as above; PROTOCOL §41.2, §44.2 |
| Attribution errors | 5 | 4 | chatflow labels | as above |
| Strict format | 1 of 11 cases (`name_now`); the receipt counts 2 of 22 checks, each case twice | 3 of 11 (`name_now`, `colour_corrected`, `format_only`); receipt 6 of 22 | tinychat-eval release v002 dev | run folders `eval/S71/release_dev/report.json`, `eval/S71s2/release_dev/report.json` |
| Knowledge guard | PIQA 0.546 (95% 0.478 to 0.613) of 205; LAMBADA accuracy 0.166 (0.121 to 0.224) of 199; 1.616 bits per byte on 200 segments | 0.551; 0.161; 1.617 | ember `/score` on `bench_v1` subsets | run folders `eval/<model>/knowledge/` |

The strict-format counts and the strict-format rows of section 4.1 were counted by Claude Opus 5.5
on 13 Nov 2026 from the stored reports. No registered bar reads them. Every other number in the table
is a score file's or a registered verdict's.

**What the replies show (Claude Opus 5.5's reading, 13 Nov 2026, of S71's stored release_dev
replies).** Content and format come apart:
- **Right content, wrong format.** "The meeting is on Thursday." and "You left the keys in the
  drawer." carry the right value and fail the exact check.
- **Wrong content.** The delayed pet's name comes back as "Bruno" where the user gave another name.
  "2 + 3" gets a sentence about three plus three.
- **Abstention fails.** Both missing-information questions get a "No…" reply. One of them, "No,
  you're talking about me. I'm Ember, an AI assistant.", is an identity answer, not an abstention.

**What the verdicts read in S71's replies** (Claude Opus 5.5, 11 Nov 2026, PROTOCOL §41.2, §43.2,
§45):
- the six role-play breaks are name questions asked in character and answered as Ember;
- the fall in stance from S70 is truthful AI answers to personal questions, which the rubric reads
  as recitals;
- on own-history recall finals S71 gives the right value 127 times and a wrong one 128 times.

## 4. Comparisons within one protocol

### 4.1 Strict format: tinychat-eval release v002, dev split

Same suite, harness, decoding settings, split, repeat count and history mode (generated) for every
row. tinychat-eval's `compare` refuses a pair that differs in suite, harness, settings, split or
repeat count. The server engine is not part of that check. Every base-A row ran on the archived ember
CPU build, server binary `b9aa22dc…`. The 3 Nov 2026 S-full run names its server only by identity
(`archived-ember-cpu-server-Sfull-20261103`).

**The unit is the distinct case.** The dev split has 13 cases. The 2 open conversations are judged
only, so 11 carry an exact check. Each case runs twice, and greedy decoding makes the two repeats
identical in every report here (13 of 13 repeat pairs). A report's summary therefore counts 22
"checks" that are 11 cases counted twice. The table gives distinct cases first; the receipt's own
count of 22 stays beside them, so the numbers can be traced to `report.json`.

| Model | Lineage | Cases passing, of 11 | Which | Receipt checks, of 22 | Expected value mentioned, of 9 | Report |
|---|---|---:|---|---:|---:|---|
| S-full SFT1500 (3 Nov baseline) | S-full base (163.84M positions), SFT1500 | 0 | | 0 | 2 | tinychat-eval `runs/v002-baseline-generated-20261103/` |
| B1SFT | base A, SFT1500 | 0 | | 0 | 0 | `a21-start/eval/B1SFT/release_dev/` |
| S64 | B1SFT, S30's packet | 1 | `format_only` | 2 | | `a21-s64-20261109/` |
| S66 | B1SFT, de-canned | 0 | | 0 | | `a22-s66-20261110/` |
| S67 | B1SFT, K2 pool, seed 1 | 3 | `name_now`, `colour_corrected`, `format_only` | 6 | 6 | `a22-s67-20261110/` |
| S67s2 | the same, seed 2 | 1 | `colour_corrected` | 2 | 5 | `a23-s67s2-20261110/` |
| S68 | S67 with A13's identity rows | 1 | `colour_corrected` | 2 | | `a23-s68-20261110/` |
| S69 | S67 with persona pool v1 | 4 | `name_now`, `colour_corrected`, `meeting_corrected`, `format_only` | 8 | | `a24-s69-20261110/` |
| S70 | S69, filler unsupervised | 4 | as S69 | 8 | 5 | `a25-s70-20261110/` |
| **S71** | S70's construction, persona pool v2, seed 1 | **1** | `name_now` | **2** | **4** | `a26-s71-20261111/` |
| S71s2 | the same, seed 2 | 3 | `name_now`, `colour_corrected`, `format_only` | 6 | 5 | `a28-s71s2-20261111/` |
| S72 | S71 with the thread pool, seed 1 | 2 | `object_now`, `colour_corrected` | 4 | 5 | `a27-s72-20261111/` |
| S72s2 | the same, seed 2 | 3 | `name_now`, `object_now`, `colour_corrected` | 6 | 5 | `a27-s72s2-20261111/` |

Blank cells were not counted. "Mentioned" counts cases whose expected value appears at both repeats.

- **What the base-A models do not pass.** No model of base A's lineage (B1SFT and every screen from
  A21 on) passes a strict delayed-recall, basic-Q&A or missing-information case.
- **Earlier models did.** Three earlier models on S-full's base pass the delayed-recall case
  `pet_delayed` with "Basil": S12 (A9) and S24 and S26 (A11). Eighteen runs on S-full's base pass both
  dev missing-information cases, each with the reply "No.". The 18 runs include those three and S5,
  S6, S8 to S10, S13, S15, S16 and S25. Every missing-information case in the suite has the gold
  answer "no", so a constant "No." passes them, and those passes do not show that the model tracked
  what it was told (tinychat-eval audit item B1, 13 Nov 2026). Source: the 71 `release_dev` reports
  under the run folders, the same suite and harness throughout.
- **Against tinychat-eval's v0.0.2 criteria.** The criteria were accepted on 3 Nov 2026 as
  prospective engineering targets, not calibrated estimates. The holdout split, which they also need,
  has never been run, so no model has passed v0.0.2.
  - **The headline floor has no defined denominator.** `dev_exact_min` is 0.25, and neither
    `release_gate_v002.json` nor `GATE_ACCEPTANCE_V002.json` says what it divides by. It could be
    the 11 scored cases, the 13 dev cases, or the 22 or 26 repeated checks (tinychat-eval audit item
    B2, 13 Nov 2026).
  - **Most models clear or miss it on every reading.** S71's 1 case reads 0.09 of 11 and 0.08 of 13,
    a miss on every reading. S69's and S70's 4 cases read 0.36 and 0.31, a pass on every reading.
  - **Models at 3 cases depend on the reading.** S67, S71s2 and S72s2 read 0.27 of 11 (a pass) and
    0.23 of 13 (a miss). Which applies is not defined, and this page does not choose.
  - **The family floors read the same either way.** S71 passes 1 of 2 immediate-recall cases, at the
    0.5 floor. It passes 0 of 2 delayed-recall cases (floor 0.25) and 0 of 2 correction cases (floor
    0.5). S70 passes 2 of 2 correction cases and 0 of 2 delayed. Relevance at 3 or more on the 13
    judged turns: S71 8 of 13 and S70 7 of 13, both below 0.7.
  - (Arithmetic against the criteria files: Claude Opus 5.5, 13 Nov 2026.)
- **The 3 Nov pilot is a different suite.** The pilot that first read S-full at 0 of 11 cases (its
  receipt says 0 of 22 checks) used tinychat-eval's pilot suite (`suite.json`, `cfff40f8…`;
  `runs/sfull-baseline-20261103/`). It is not compared with any row here. The same S-full model on the
  release suite also passes 0 of 11 (first row).

### 4.2 Relevance on 13 turns and told recall, across lineages

Same judge, rubric, mode and 13 turns; chatflow told recall on the same 108 items. S-full's chatflow
rows came from one CPU evaluation on a text-identical greedy runner (PROTOCOL §36). The 13-turn J1
cannot rank models at this level (section 2), so this table shows a change of lineage, not a
ranking among close models.

| Model | J1 points of 52 | Told recall of 108 | Receipt |
|---|---:|---:|---|
| S-full SFT1500 | 19 | 7 | `analysis/a21_score.json` (interaction block) |
| B1SFT (base A, SFT1500) | 29 | 16 | `analysis/a26_score.json` |
| S71 | 36 | 62 | `analysis/a26_score.json` |
| S71s2 | 32 | 58 | `analysis/a28_score.json` |

### 4.3 Relevance on 532 turns (wide J1)

All rows were judged on the same 532 planned turns with the same judge, request and plan
(`plan_v1.json`). Base A's SFT1500 and the S-full lineage were not read on this instrument.

| Model | Base | Level (points) | 95% (mean of 4) | Receipt |
|---|---|---:|---|---|
| S66 | A | 27.0 | 1.94 to 2.21 | `analysis/rel_wide2_read.json` |
| S67 | A | 26.4 | 1.89 to 2.16 | as above |
| S67s2 | A | 25.9 | 1.86 to 2.12 | as above |
| S70 | A | 30.6 | 2.23 to 2.48 | as above |
| **S71** | A | **31.7** | 2.30 to 2.58 | `analysis/rel_wide2_read_s71.json` |
| S71s2 | A | 29.7 | 2.15 to 2.42 | `analysis/rel_wide4_read.json` |
| S72 | A | 29.1 | | `analysis/rel_wide3_read.json` |
| S72s2 | A | 30.2 | | as above |
| O1, SmolLM2-135M-Instruct as released | open | 24.5 | 1.75 to 2.02 | `analysis/rel_wide5_read.json` |
| O3, SmolLM2-360M-Instruct as released | open | 28.3 | 2.04 to 2.31 | as above |
| O2, S71's packet on SmolLM2-135M, seed 1 | open | 37.0 | 2.72 to 2.97 | as above |
| O2s2, the same, seed 2 | open | 36.4 | 2.68 to 2.92 | as above |

**Registered contrasts, with 95% intervals in points.**
- **The persona pools helped.** S71 is above S67's packet at both seeds: +5.3 (+3.4 to +7.2) and
  +5.8 (+3.7 to +7.9). On the five families the persona pools did not target it is also above at
  both seeds.
- **S71 and S70 are level.** S71 against S70 is +1.1 (−0.4 to +2.6), not separated.
- **The second seed reads lower.** S71s2 against S71 is −2.0 (−3.7 to −0.3).
- **The thread pool did not help.** S72 against S71 is −2.6 (−4.2 to −1.0), below, and S72s2 against
  S71 is −1.5 (−3.4 to +0.3), not separated (A27, `relevance_cost`).
- **The open base read higher.** O2 against S71 is +5.3 (+3.2 to +7.4) and O2s2 against S71 is +4.7
  (+2.6 to +6.9): above at both seeds (A29, `base_up`).

**Where the open-base rows differ.**
- **Different engine.** They were generated by `transformers` in bf16 on a GPU, and our models by
  ember on the CPU.
- **Different reply cap.** The cap counts each model's own tokens, and SmolLM2's 49,152-token
  vocabulary fits more text into 32 tokens.
- **O1 and O3 were truncated.** They reach the cap on most turns.

A29 changed the base, size, tokeniser, architecture, pretraining, the authors' instruction tuning,
trainer and engine together (PROTOCOL §45).

### 4.4 Identity, stance, role-play and loops

`persona_v003` and `contrast_v002`, all on base A except O2 and O2s2.

| Model | Identity correct of 92 | Stance engaged of 60 | Role-play breaks of 48 | Ended twin of 8 | Loops of 212 | Source |
|---|---:|---:|---:|---:|---:|---|
| B1SFT | 1 | | | 0 | | PROTOCOL §36.2 |
| S67 | 0 | 35 | 0 | 0 | 0 | §38.3, §41 |
| S68 | 87 | 3 | 30 | 4 | 13 | §38.3, §41 |
| S69 | 78 | 42 | 18 | 3 | 23 | §39.2, §41 |
| S70 | 77 | 45 | 17 | 3 | 33 | §40.2, §41 |
| **S71** | **80** | **35** | **6** | **5** | **0** | §41.2 |
| S71s2 | 81 | 37 | 15 | 3 | 3 | §44.2 |
| S72 | 75 | 34 | 15 | 2 | 1 | §43.2 |
| S72s2 | 78 | 34 | 10 | 2 | 2 | §43.2 |
| O2 | 83 | 34 | 14 | 5 | 4 | §45.3 |
| O2s2 | 80 | 40 | 14 | 2 | 0 | §45.3 |

Bars at A26: identity at least 80% correct with at most 5% wrong names; ended twin at least 5;
stance at least 30; breaks at most 12; loops at most 2.
- **S70's identity count is inflated.** Its count of 77 includes looped answers ("I'm an AI, and I'm
  an AI") that the identity scorer counts as correct (§40.2).
- **S67 has no persona.** Its 0 breaks come with no identity at all: its packet carries no identity
  rows.

### 4.5 Content-tied replies: `cue_v001`

Components 1 to 4, of 174, all greedy on base A (PROTOCOL §37 to §44):
- B1SFT 33, S64 54, S65 48, S66 57;
- S67 111 and S67s2 107; S68 115; S69 87; S70 107;
- **S71 106** and S71s2 118;
- S72 120 and S72s2 113.

The two seeds of S67's packet differ by 4 items and those of S71's by 12. A 15-item gain is the
registered screen bar.

**Higher `cue_v001` does not mean higher relevance.**
- S72 has the highest `cue_v001` score and reads below S71 on the wide J1.
- The K2 pool lifted `cue_v001` by about 50 items without moving relevance overall (PROTOCOL
  §42.2).

### 4.6 Knowledge benchmarks

**Bases and SFT1500, full `bench_v1`.** Point values, ember `/score` at `cc6635be`. Source: workspace
`AI_ML/tiny_chat_benchmarks/b1_stage1/receipts/final_base_eval_20261109/bench_b1_summary.md`.

| Checkpoint | HellaSwag (norm) | ARC-E | ARC-C | PIQA | WinoGrande | LAMBADA acc | BLiMP |
|---|---:|---:|---:|---:|---:|---:|---:|
| S-full SFT1500 | 0.267 | 0.359 | 0.187 | 0.556 | 0.500 | 0.132 | 0.682 |
| Base A, step 60,000 | 0.288 | 0.427 | 0.201 | 0.605 | 0.512 | 0.234 | 0.751 |
| Base A, SFT1500 | 0.293 | 0.431 | 0.214 | 0.609 | 0.507 | 0.208 | 0.710 |

**Paired comparison.** Base A against S-full's base gives paired 95% intervals above zero on LAMBADA
accuracy (+9.53 percentage points), ARC-Easy (+5.47), PIQA (+3.97) and BLiMP (+2.76). HellaSwag is
not paired, because S-full's HellaSwag run is incomplete (`FINAL_RESULT.md`, GPT/Codex, 9 Nov 2026).

**LAMBADA target bits per byte, lower is stronger.**
- S-full base 1.4004, and S-full after SFT1500 1.493;
- base A 1.0378;
- October's 12 × 768 models: pk_base_v1 0.970, pk_sft_v1 1.014.

Sources: PROTOCOL §29, §35.

**The SFT'd chat models, small guard.** These figures come from the 205-item PIQA and 199-item
LAMBADA subsets and the 200-segment bits-per-byte read in section 3. They compare only with each
other, not with the table above:
- PIQA: S67 0.556, S70 0.571, S71 0.546, S71s2 0.551;
- bits per byte: 1.610, 1.620, 1.616, 1.617.

The intervals overlap.

None of these numbers measures chat.

## 5. Baselines and negative results, kept in view

- **The 3 Nov 2026 S-full pilot.**
  - **What it read.** tinychat-eval's first baseline read S-full SFT1500 on the pilot suite:
    26 episodes, all repeat pairs identical, 0 of its 11 scored cases (the receipt's 0 of 22 strict
    checks counts each case twice).
  - **What the replies held.** Values survived in replies of the wrong format ("Nora", "blue"), and
    the delayed cat Basil came back as Ted. Jonathan's blind ratings gave relevance 0 on all four
    rated cases (tinychat-eval `calibration/`; registry §3).
  - **What it is a baseline for.** It describes the lineage before base A: a weaker base
    (163.84M positions against base A's 983.04M) and SFT1500 only, with none of S71's SFT. It is the baseline, not a reading of current capability.
- **Waves 1 to 3 (4 Nov 2026):** 0 of 11 dev exact passes for every screen (registry §4).
- **A21 (9 Nov):** `told_only` and a base `guard`. S34's and S30's recipes did not lift base A's
  SFT1500 on J1.
- **A22 (10 Nov):** `guard`.
  - De-canning (S66) moved nothing.
  - The K2 pool (S67) moved `cue_v001` by +54 and lost identity entirely (0 of 92).
- **A23 (10 Nov):** `identity_gap`. Restoring the old identity rows (S68) brought identity back. The
  same rows taught it to decline 42 of 60 personal questions and to break 30 of 48 role-play turns.
- **A24 (10 Nov):** `relevance_cost`. Persona pool v1 (S69) lost 24 `cue_v001` items.
- **A25 (10 Nov):** `decline`. S70 broke 17 of 48 role-play turns, and 33 of its 212 turns looped
  "I'm an AI".
- **A27 (11 Nov):** `relevance_cost`.
  - The thread pool (S72) read below S71 at seed 1 on the wide J1.
  - Told recall fell to 32 and 38 of 108, through false "you didn't tell me" declines.
- **A28 (11 Nov):** `wide_only`.
  - **What failed at seed 2.** S71's packet at seed 2 missed A26's bars on the ended twin, role-play
    breaks and loops.
  - **What held.** Its relevance lead over S67's packet held.
  - **Claude Fable 5.2's reading, 11 Nov 2026.** S71 at seed 1 was a favourable draw.
- **A29 (11 Nov):** `base_up`.
  - **The registered rule.** It reads relevance as limited by our base, on this evidence.
  - **Claude Opus 5.5's reading, 11 Nov 2026.** "The base was the limit on relevance"; the cause is
    the whole package, not one ingredient.
  - **What followed.** Base 2 was started for this reason (section 7).
- **Superseded.** A22's reading of S67's 39 J1 points as the highest on record is superseded. That
  score was one draw: 28 at seed 2, and 26.4 on the wide J1 (registry §9).

## 6. What has not been measured

- **Salience.** No rubric exists for it (registry §3).
- **Free conversation.** Every judged number is on scripted suites with greedy replies. The
  protocol records Jonathan's live chats with S54 (7 Nov 2026) and with base A and its SFT1500
  (9 Nov 2026), and none with S71.
- **Replies at serving temperature.** S71's sampled replies at about 0.7 are not on record.
  Best-of-16 on `cue_v001` is recorded for models up to S70 (PROTOCOL §37.3, §38.3, §40.2), not for
  S71.
- **The tinychat-eval holdout split.** It has never been run.
- **Human ratings.** None exist after S-full's calibration set.
- **Retrieval utility.** `rag_eval.py` has not been run on any model of base A's lineage. The
  context sweep has been run on S-full only.
- **Seed spread.** It is known only from two seeds of three packets.
- **Speed and memory use while serving.** These are not part of this record.

## 7. Base 2

A deeper base, 18 × 512, is registered in three arms (PROTOCOL §48, `base2/BASE2_PLAN.md`). On
13 Nov 2026 its corpus build was relaunched after a source host's expired certificate (§48.2). It
has produced no model, and this page claims nothing for it.

## 8. Readings on this page

- **Which model is current best.** "S71 is the current best model of our own by the programme's
  criteria; it is not the best on strict format, and its `ready` holds at one seed of two."
  Claude Opus 5.5, 13 Nov 2026, from registry §10b and the verdicts cited.
- **The strict-format counts and the v0.0.2 arithmetic.** Claude Opus 5.5, 13 Nov 2026, counted
  from stored reports. They are not registered outcomes.
- **Cited readings keep their producers and dates.** Claude Opus 5.5, 10 and 11 Nov 2026; Claude
  Fable 5.2, 11 Nov 2026; GPT/Codex, 9 Nov 2026.
- **Jonathan's reads.** None of S71 is recorded in the protocol.
