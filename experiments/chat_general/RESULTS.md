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
- **The verdict that stands: pk_sft_v1** (Jonathan, 27 Oct 2026). The preference stage runs on it.

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

## Preference pilot on pk_sft_v1 (27 Oct 2026)

The pilot of PLAN.md's preference step on the winner, pk_sft_v1: candidates sampled on a RunPod L4, then
ranked at home by Claude (`claude-opus-5-5`) through the Claude Code CLI. Receipts: the candidates in
`C:/ml_artifacts/pref_pilot/sample/` (`candidates.jsonl`, sha256 c969def7...), the verdicts and pairs in
`judge/` (`judge_cache.jsonl`, the cache the full run reuses; `pilot_pairs.jsonl`; `pilot_report.json`).

### Sampling

- **Run:** root `scripts/colab_driver.json` entry pk-pref-pilot (`tools/colab_chat/pref_pilot_v1.json`,
  paperkiln 799aa7d0), one RunPod L4 (EUR-IS-1, $0.59/h), 12:50-13:32, **$0.41** (cap $2): uploads 20 min,
  the job 17 min (data fetch, ember build, sampling). ember main 7526469 built on the pod (117 s), CUDA, `EMBER_GEMM_TF32=0`,
  `EMBER_DETERMINISTIC=1` (its /health reports it); pk_sft_v1.gguf 4d53c7ff..., T 512.
- **Prompts:** the first 200 eligible smol-smoltalk test-split first turns (the filters drop 38 test-split
  prompts in all for overlap with the train split, none for the eval sets or markers); 5 skipped because
  prompt + 128 new tokens exceed 512. **195 requests, each with 4 distinct candidates (780).**
- **Replies:** 685 of the 780 runs stop at the 128-token budget, 95 on end-of-text; median 96 words per
  candidate. 3 runs ended inside a UTF-8 character and were trimmed to their last whole character (rule
  `trim_cut_tail_drop_other`); none was dropped, no candidate holds U+FFFD, and no request failed.
- **The rule on a CPU build** of 7526469 with pk_sft_v1 (`receipts/pref_pilot_1027/`): 48 greedy requests
  that draw Japanese text, each reply equal to the vocabulary's decode of the returned ids and none answered
  HTTP 500; at the budgets that cut a character, 52 sampled runs, 7 trimmed, no U+FFFD in any candidate.

### Judging

`bash experiments/chat_general/judge_pilot.sh C:/ml_artifacts/pref_pilot/sample/candidates.jsonl 200`
(paperkiln cf63b95b), each prompt ranked twice, the candidates in order and reversed, one call at a time.

| calls | failures | cost | per call (median, range) | per prompt | pairs | pairs per prompt | position disagreements | length-mismatched |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 390 | 0 | $9.40 | $0.0238 (0.0176-0.0325) | $0.048 | 125 | 0.641 | 57 of 195 (0.292) | 13 |

- The two orders agree on the best reply for 156 prompts, on the worst for 170, on the whole ranking for
  117. The candidate shown first is ranked best in 118 of 390 calls (0.30; 0.25 by chance).
- 112 of the 125 pairs have both replies cut by the 128-token budget; 8 pair a finished chosen reply with a
  cut rejected one, 2 the reverse, 3 are both finished.
- A call costs about twice PLAN.md's measured $0.0117: the 128-token candidates are longer than the ones it
  was measured on.
- The CLI now runs in an empty directory of its own (`chat_judge.py`, paperkiln cf63b95b). Started from the
  microtorch root it loaded both AGENTS.md files and project memory into each call: 15,610 prompt tokens and
  $0.13 a call, against 1,177 tokens and $0.012 for the same probe from an empty directory, and the judge
  saw the workspace's instructions. Every pilot verdict was made from the empty directory. Three probe
  calls outside the cache cost $0.15.

### The full run

- At the pilot's rate, 3,000 prompts cost **$144.66** and give about 1,920 pairs. The cache already holds
  $9.40, so the run passes the $100 cap (`--max-usd 100` stops it there): under the cap, about 1,880 more
  prompts (about 2,075 in all, about 1,330 pairs).
- Nine in ten pairs would train DPO on two replies the budget cut. More new tokens per candidate (T 512
  leaves room) or a preference for finished replies changes that, and the cost per call with it.

## Conversation schema v2: system prompts and tool turns (27 Oct 2026)

Design: LOCAL_LLM_STACK.md section 3, point 3g (paperkiln/2). Code: `tools/chat_template.py` and
`include/microtorch/chat_template.hpp` (paperkiln 22f8fe6d), mtstudio's SFT path and `/chat`,
`tools/chat_v2_vocab.py` (vocabulary and embedding rows), `tools/chat_v2_data.py` (the mix),
`tools/chat_v2_eval.py` (tool-call set, parity prompts), `tools/iftiny_eval.py --placement`.

### The corpus

`tools/chat_v2_data.py build` (ledger `C:/ml_artifacts/chat_v2/data/chat_v2_mix_v1.jsonl.ledger.json`;
mix sha256 f5933a46...): 135,525 conversations at T 512, each validated under paperkiln/2, decontaminated
with corpus/decontam.py's rules and against the IF-tiny and turn-distance questions (4 words or more).

| source | licence | rows | rows mtstudio cuts at T 512 | dropped |
|---|---|---:|---:|---|
| smol-smoltalk @f73fe857, rows with a system message, train split (openhermes-50k, smol-summarize-5k/20k, smollm-rewrite-30k, explore-instruct-rewrite) | Apache-2.0 | 55,873 | 3,011 | 13,194 over T, 5 IF-tiny, 1 decontam |
| no_robots @e6f9a4ac, Chat rows with a system message (human-written) | CC BY-NC 4.0 | 776 | 3 | 4 IF-tiny |
| glaive-function-calling-v2 @e7f4b645 (synthetic), rows with tool definitions; 40k drawn | Apache-2.0 | 38,880 | 3,367 | 1,929 + 1 decontam, 1,120 over T |
| lever 1's conversation rows (pk_sft_v1's set), seeded 40k replay | as lever 1 | 39,996 | 0 | 4 IF-tiny |

- With a system message: 95,529 rows; with tool definitions 38,880; with tool calls 30,740 (40,245 tool
  results). Lever 1 (pk_sft_v1) read openhermes-50k with its system messages dropped
  (`smol_smoltalk.py`'s `turns()`; 18,285 of its rows carry one) and none of the other four sources; the
  same smol-smoltalk revision holds 70,484 rows with a system message.
- glaive parse (112,960 rows): 60,284 with calls, 14,980 with definitions the reply does not use, 34,594
  without definitions (left out: generic synthetic Q&A), 3,098 refused (2,386 malformed `<functioncall>`,
  398 ending on a tool result, 188 structure, 116 arguments that are not JSON, 10 holding a control string).
- The tool half's training data is synthetic (glaive). The general chat track allows it; the tiny-chat
  programme (human-written only) does not use this mix.

### Design of the measurement (written before any result was read)

Exploratory, not pre-registered: no prediction is registered, no threshold decides anything, and the
numbers are reported as they come.

- **Question.** Does pk_sft_v2 comply with an instruction given as a system message as often as with the
  same instruction inside the user turn?
- **Models.** pk_sft_v2 (pk_sft_v1 + one pass of the v2 mix under paperkiln/2) and pk_sft_v1 (v1: its
  system message folds into the first user message, so its "system" condition is user text with "\n\n").
- **Items.** IF-tiny v1's 574 items (15 families; held out: no training row holds an IF-tiny or
  turn-distance question). Conditions per item, the same instruction and question text:
  user = no system message, the instruction, a space and the bare question (the item's own prompt for 464
  items; for numbered_list, repeat_phrase and repeat_word, whose prompt is the instruction alone, the bare
  question is appended so that both placements carry the same text); system = the instruction as the
  system message, the bare question as the user message; bare = the bare question (the base rate). Also
  the turn-distance set (1,344 conversations: instr_user vs instr_system at d = 0, 1, 2, 4, 84 per cell).
- **Generation.** Greedy, 64 new tokens, `mtstudio chat --kv-cache` on the pod's CPU; every server prompt
  checked against the model's own template rendered at home (`prompt_check`).
  *Amendment (27 Oct 2026, 20:15, before any result was read):* `mtstudio chat` generated at about 17 s a
  prompt on the pod (200 of pk_sft_v1's 2,158 prompts in 59 minutes), ten hours a model. Generation moves to
  ember's CUDA server at b49eb5c (the first ember with paperkiln/2; `EMBER_GEMM_TF32=0`,
  `EMBER_DETERMINISTIC=1`), as for IF-tiny on 26 Oct, and `mtstudio score` re-scores every pk_sft_v2 reply
  that ends on a stop token (its argmax must equal the reply at every position). Nothing else changes.
- **Quantities.** Per model: compliance (micro over items, macro over families) per condition; system
  minus user as a paired difference over the 574 items with a normal-approximation 95% interval and the
  discordant counts; system minus bare and user minus bare; per family. Turn distance: system minus user
  per distance (`analyze_if.py`). Between models: pk_sft_v2's system condition against pk_sft_v1's.
  Families with a bare rate near 1 (end with a full stop) carry little.
- **Tool half.** `corpus/eval/tool_calls_v1.jsonl`: 200 held-out glaive requests whose gold reply makes a
  call (call made, function name, arguments as JSON, arguments equal to the gold's) and 100 whose gold
  reply makes none (false calls). glaive is templated, so held-out rows resemble training rows: this
  measures the format, not generalisation.

### The run

- **Campaign** pk-chat-v2 (root `scripts/colab_driver.json`; jobs `tools/colab_chat/chat_v2_v1.json`), one
  RunPod L4 (EUR-IS-1, $0.59/h): launch 1 at paperkiln 1a8db67d, 18:45-20:38 (uploads 16 min, the SFT
  67 min at 16.2k tokens/s); launch 2 at 88551a59, 20:40-21:05, for the evals on the same volume.
  **$1.35** (2.28 L4-hours; cap $5). The first launch's evals, generated with `mtstudio chat` on the pod,
  ran at about 17 s a prompt and were stopped (the amendment above); its evalv2 had also failed on a port
  both eval jobs took. Their replacements, evalv1e and evalv2e, ran with ember.
- **pk_sft_v2**: pk_sft_v1 (embedding rows 16384-16391 the mean of the old rows; `v2init_extend.json`) +
  8,000 steps x 16 of the v2 mix under paperkiln/2 (135,525 rows, none over T, 20,721 later exchanges cut
  at T 512; 6,776 held out), lr 5e-5, warmup 200, cosine to 0.1, BF16 GEMMs + flash attention. Held-out loss
  1.299 (step 2,000), 1.274, 1.261, 1.257 (step 8,000). Files `C:/ml_artifacts/chat_v2/run/sft/`:
  pk_sft_v2.gguf sha256 2363aa42..., pk_sft_v2.safetensors 6e17512c..., q8_0_all e88a77ce...
- **Receipts** `receipts/chat_v2_1027/`: the mix ledger, sft_spec, sft_result, the SFT's eval events,
  `eval/ember/` (IF-tiny with the placement set and turn distance for both models, the tool-call set for
  pk_sft_v2, ember's build record) and `tables.md` (`v2_placement.py`).

### Checks

- Every server prompt equals the model's own template rendered at home: 2,158/2,158 per model, 798 with a
  system message (paperkiln/1's fold for pk_sft_v1, paperkiln/2's system turn for pk_sft_v2).
- **ember vs mtstudio on paperkiln/2 at scale:** `mtstudio score` re-scored all 1,864 pk_sft_v2 replies that
  ended on end-of-text; its argmax equals ember's reply at every one of the 30,504 positions.
- pk_sft_v1's IF-tiny numbers equal 26 Oct's (macro 0.123, micro 0.129; ember b49eb5c here, fe01b35 then).

### Results

Instruction placement (574 items; micro over items [Wilson 95%], macro over the 15 families):

| model | user | system | bare | macro user | macro system | macro bare | system - user [95%] | system only / user only |
|---|---|---|---|---|---|---|---|---|
| pk_sft_v1 (system folded into the user turn) | 0.124 [0.099, 0.153] | 0.139 [0.113, 0.170] | 0.065 [0.047, 0.088] | 0.118 | 0.133 | 0.062 | +0.016 [-0.006, +0.037] | 25 / 16 |
| pk_sft_v2 (system turn) | 0.160 [0.133, 0.193] | 0.138 [0.112, 0.168] | 0.065 [0.047, 0.088] | 0.153 | 0.133 | 0.062 | -0.023 [-0.054, +0.009] | 36 / 49 |

- **pk_sft_v2 complies with the instruction in the user turn slightly more often than with the same
  instruction as a system message**: -0.023, with an interval that includes 0; without end_full_stop
  (bare rate 0.47) -0.032 [-0.063, -0.001] over 534 items. Both placements beat the bare question: system
  +0.073 [+0.045, +0.102], user +0.096 [+0.069, +0.123].
- **Against pk_sft_v1**, v2 training moved the user placement (+0.037 [+0.012, +0.061], 36 items gained, 15
  lost) and left the system placement where the fold had it (-0.002 [-0.029, +0.025]). IF-tiny macro
  0.158 against 0.123.
- **By family** (pk_sft_v2, user / system / bare): the system turn carries avoid_word (0.12 / 0.35 / 0.12)
  and yes_no (0.00 / 0.25 / 0.00); the user turn carries include_word (0.42 / 0.12 / 0.00) and
  repeat_phrase (0.65 / 0.20 / 0.30). Lower case, upper case, quoted, repeat_word and number_only stay at 0
  in every condition for both models. 40 items or fewer per family: these are single-family readings.
- **Length:** 116 of pk_sft_v2's user-placement replies and 25 of its system-placement replies run to the
  64-token budget.
- **Turn distance** (84 per cell; system - user, paired): pk_sft_v2 -0.012, +0.036, +0.071, +0.012 at
  d = 0, 1, 2, 4, every interval including 0; compliance at d = 4 is 0.70 (user) and 0.71 (system) against
  pk_sft_v1's 0.55 and 0.52, with nohint at 0.64 (pk_sft_v2) and 0.48 (pk_sft_v1): most of the rise comes from the compliant history,
  as on 26 Oct.

Tool calls (pk_sft_v2, `corpus/eval/tool_calls_v1.jsonl`, greedy, 96 new tokens):

| call made | right function | arguments parse as JSON | arguments equal to the gold's | false calls |
|---|---|---|---|---|
| 198/200 | 198/200 | 195/200 | 172/200 | 2/100 |

298 of 300 replies end on end-of-text. The set is held-out glaive rows, and glaive is templated: this
measures the call format on familiar tools, not tool use in general.

### Reading

- Exploratory, as designed: one model, one seed, 574 items, no registered prediction.
- The system turn works: an instruction there lifts compliance over the bare question by about as much as
  the folded system message did in pk_sft_v1. It does not yet carry an instruction better than the user
  turn: the point estimate favours the user turn by 2-3 points. The training mix puts task instructions in
  system turns (rewrite, summarise, personas, tool definitions) but few format constraints of IF-tiny's
  kind anywhere, and none of its questions.
- Compliance stays low in absolute terms (0.14-0.16 against the bare 0.065; Qwen2.5-0.5B-Instruct 0.50 on
  26 Oct): v2 gives the model a place for an instruction; following one is the preference and data work's
  job.
- The tool half works on its own format: calls are well formed, named and argued (172/200 exact) and rarely
  made unasked.
