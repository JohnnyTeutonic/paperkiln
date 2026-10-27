# Tiny chat, long carry (Carry-12): pre-registration DRAFT

*DRAFT, 24 Oct 2026. NOT FROZEN, NOT LICENSED, NOTHING LAUNCHED. This file plans the programme that follows
E5/E1 (`../tiny_chat_2026-10/PREREGISTRATION.md` and its amendments, which this draft does not change). It
freezes only after E5, E1 and the recursion arms have reported (section 11 lists what must be known), as a new
file `PREREGISTRATION.md` in this folder, licensed by the commit that adds it with the frozen Carry-12 v1, its
pin, the evaluation tool and the jobs files, each exercised end to end on CPU at toy size. Until then every
number below is a planning number and every rule is open to change without an amendment.*

*The base is 60M (Jonathan's change of tactics, 26 Oct 2026; `../tiny_chat_2026-10/TACTICS_2026-10-26.md`): at 20M,
five interventions left Carry-5's carry-over cost unchanged and size is the one lever with a signal, so the
long-gap hypotheses, Jonathan's cross-layer HX among them, are tested where the models have more capacity. Numbers
marked † were set for the 60M base on 26 Oct 2026; section 13 lists the 20M planning values they replace. The freeze
rule is unchanged: this file freezes only after E5, E1 (its 60M stage included) and the recursion arms report.
Jonathan approved the 60M core on 26 Oct 2026 (section 10: 104-139 L4 VM-hours, $61-82, with self history played to
the target only and no near_self; every registered test kept, HX included; S-full and R conditional), under that
freeze rule. On 27 Oct 2026 he gave the go on the measured total (section 10), left data option 3 untaken (section 11
item 7) and added cell T, template SFT with explicit recall targets (sections 0, 6 and 8), on an outside reviewer's
suggestion.*

Receipts in this folder: `carry12_v1_build.json` (counts and balance), `carry12_v1_budget.json` (context
lengths), `carry12_v1_decontam.json` (contamination), `dialogue_lengths_v1.json` (the human corpora in turns
and tokens), `STAGE_L0.md` (Stage L0's frozen registration), `IR_SPEC_DRAFT.md` (spec text for ir.hpp
review), `CROSSLAYER_LITCHECK.md` (cross-layer arm, by another agent), `stage0/` (the pre-freeze campaign's
receipts and E1 at 60M's measured walls, `stage0/e1_60m_walls_t512.json`), `carry12_tsft_v1_build.json` and
`carry12_tsft_v1_disjoint.json` (cell T's template set: counts and hashes; its disjointness from Carry-12 v1 and
Carry-5 v1). Tools: `tools/carry12_build.py`, `tools/carry12_eval.py`, `tools/carry12_l0.py`,
`tools/dialogue_lengths.py`, `tools/carry12_tsft_build.py` (cell T's template set); the pre-freeze campaign
`tools/colab_chat/carry12_stage0_v1.json` (written by `make_stage0_jobs.py`) and its total, `stage0_total.py`; the
cells' specs and jobs files, `make_jobs.py` (`specs/cell_*.json`, `specs/sft_*.json`,
`tools/colab_chat/carry12_<cell>_r<seed>.json`), and section 9's `analyze.py`.

## 0. Objective and constraints

- **Objective (Jonathan, 24 Oct 2026):** "We do want them to get some recall to carry a conversation for, I
  don't know, at least a dozen turns." The ability measured: using a fact from turns 1-4 at turns 10-14.
- **Human-written data only, no distillation** (as E5/E1, section 0). Templates written by us are not model
  text; they are evaluation material and never enter training (section 4 says why they stay out), with one
  registered exception. **Cell T** (Jonathan's ruling, 27 Oct 2026, on an outside reviewer's suggestion of SFT with
  explicit recall targets) trains on a template set written for it under option 5's conditions (section 4): our own
  templates, not model-generated dialogue, sharing no template, entity or value with Carry-12 v1 or Carry-5 v1
  (checked: `carry12_tsft_v1_disjoint.json`, section 6), with T scored on Carry-12 v1, which is built from other
  templates. No other cell sees a template.
- **Measurement before mechanism**, as before: the long-carry gap is measured on E5/E1's own models first
  (Stage L0, no training), then the interventions are deltas against a long-context base.

## 1. What the receipts say now

- **Carry-12 v1 (frozen, sha256 `de24f2c7...`):** 1,452 items (recall 480, correction 360, topic 360,
  load 120, instruction 72, knowledge 60), 12-14 turns, target turns 10-14 (balanced), fact distances 6-13
  (recall, topic, load) and 1-11 (correction). Section 2.
- **Context:** with the gold (canned) history every item fits T 512 (max 353 tokens, median 269). With self
  history at 24 tokens per earlier reply, 4 of 1,452 targets pass 512 (max 515); at 48 tokens all pass 512
  and none passes 1,024 (max 827; the whole 14-turn conversation, max 869). **Recommended T = 1,024.**
  Section 3.
- **Long human dialogue is scarce, not cut.** Of the 13 sources' 170,572 dialogues, 6,429 (3.8%) reach 12
  turns (2.90M of 39.9M tokens), 2,814 reach 14 and 306 reach 20; 5,161 of the 6,429 are Taskmaster. The
  first 12 turns of 85% of them fit T 512 and of 99.4% fit T 1,024. In the SFT split, 668 dialogues reach
  12 turns. A word-overlap proxy finds turns 10+ revisiting content last seen in turns 1-4 in 9.9% of late
  turns (3,192 of 32,324). Section 4.

## 2. The benchmark: Carry-12

**Generator** `tools/carry12_build.py` (imports Carry-5's slot, topic and knowledge templates, value lists,
acknowledgements, openers, closings and question-answer pairs from `tools/carry5_build.py`; seed 20261024).
Every item is a scripted conversation: 12-14 user turns, a canned short assistant reply per turn (the gold
history), one target turn whose reply is scored, four forced-choice candidates in a shared answer frame.
Turns that hold no event are natural small talk from 46 hand-written single exchanges and 14 two-turn arcs
(weather, cooking, a cold, a jigsaw ...), never model text. The build refuses small talk that names any list
value or raises a slot's subject, plural forms included (names, pets, homes, work, ages, favourites and their
categories, trips, cakes, books, gardens, parties, music and learning, relatives), and per item refuses
small talk before the target holding a word of that item's candidates (knowledge: also its accepted
answers; instruction: its four forms), so small talk never primes a candidate ("two people" before a
tricycle question).

**Balance and artefacts (the v1 review of the draft).** The gold position is exactly balanced in every
family (a shuffled position sequence independent of the cells; the draft had 132/112/127/109 in recall);
candidate values are drawn from per-slot shuffled cycles, so a value is gold on average a quarter of the
times it is a candidate (recall, correction, topic: means 0.248; load, 20 items per slot, 0.20); no two items
of a family share a twin, a user-turn sequence or a near rendering (the draft had 24 topic items with a
duplicated near rendering). The instruction family is length-neutral: Carry-5's four brevity kinds made the
compliant answer the shortest candidate on every item (72 of 72 in the draft), so v1 has two format kinds
whose four candidates are the same correct sentence in four forms: `opener` (start every answer with Sure /
Okay / Right / Well) and `closer` (end it with "of course" / "I think" / "you know" / "for sure"), the
required form rotating (each form gold 9 times). In tokens the gold is uniquely the shortest candidate on 0
of 72 instruction items and on 2-7% of the other families' (`carry12_v1_budget.json`); the build refuses any
family where it is uniquely the shortest or the longest on more than 35%.

| family | items | cells | distance (target - fact) |
|---|---:|---|---|
| recall | 480 | 6 slots x fact turn 1-4 x target 10-14 x 0-3 distractors (early window 1-4 or late window 5..target-1, half each) | 6-13 |
| correction | 360 | 6 slots x (plant, fix) in {(1,3), (2,4)} early or {(1,7), (3,9)} late x target 10-14 x 0-2 distractors | 1-11 |
| topic | 360 | 6 topics x plant 1-4 x target 10-14 x 0-2 distractors in the digression | 6-13 |
| load | 120 | 6 slots x plant 1-4 x target 10-14; two facts of other slots in turns 1-4 | 6-13 |
| instruction | 72 | 2 format kinds x instruction turn 1-3 x target 10/12/14, four times (form rotating) | 7-13 |
| knowledge | 60 | Carry-5's 60 questions at targets 10-14 after small talk | - |

Small-talk turns before the target: 5-12 per memory item. **Controls on every item:** `twin` (the statements
and the question in one user turn), `noctx` (the question alone; every value is gold and distractor equally
often, so it is chance by construction), and new, `near`: the same event turns (plants, distractors,
corrections, load facts) and the question with the small talk removed. So Y decomposes into the cost of
splitting into turns (twin -> near) and the cost of distance (near -> full).

**Contamination** (`carry12_v1_decontam.json`, carry5_build's scan: decontam.py's 13-grams, user-turn
equality and 8-12-word spans, plus template sentences, fact sentences and item leaks), over the 13 tiny-chat
sources, Simple English Wikipedia, the local FineWeb-Edu sample and the four long sources built on the laptop
(MSC, ABCD, CCPE-M, PersuasionForGood; section 4): 19 corpora, 419,812 rows. **No item leaks**, no 13-gram or
span hits; 45 items' fact sentences occur somewhere (everyday sentences such as "I work as a chef."; 26 of
them in MSC, whose personas are ConvAI2's); 7 Carry-12 user turns occur as whole training user turns ("Hi
there! How are you today?", "I work as a chef.", "I work as an accountant.", "My favourite colour is blue.",
"Thanks, that is all for now.", "What is my name?", "Can you tell me something interesting?"). The long
programme's data job (`tools/colab_chat/tiny_chat_data.py --carry12`, which refuses a Carry-12 file other than
the pinned one) adds Carry-12's user turns, gold replies and twins to the frozen sets beside Carry-5 and drops
dialogues by decontam.py's rules, counted in its ledger. **Taskmaster-3** (`human_taskmaster3_v1`, its pinned
bytes, on the laptop): 16,675 rows, no item leak, no 13-gram, prompt or span hit, one fact sentence ("I am 41
years old"; `stage0/carry12_v1_decontam_taskmaster3.json`). **The VM's FineWeb-Edu shard**
(`fineweb_edu_shard0_v1/pretrain_train.txt`, 3.46 GB, the source of every mix's FineWeb documents; pre-freeze
campaign job `decontam_fineweb`, `stage0/carry12_v1_decontam_fineweb_shard0.json`): 728,756 rows, no item leak, no
13-gram, prompt or span hit; 28 items' fact sentences occur in 44 rows (everyday sentences such as "I am 25 years
old"). **The T 1,024 data job** (`stage0/carry12_data_t1024_ledger.json`) drops 53 of 170,572 dialogues by the
prompt rule with Carry-5 and Carry-12 as targets (E1's data job: 52 with Carry-5 alone); its two panels are E1's
byte for byte (sha256 c1aa6d42... and 7881aa1c...), and mix_s35.ids holds 184,064,335 tokens (E1's: 184,060,090).

**Scoring: `tools/carry12_eval.py`** (built on carry5_eval's internals, which stay unchanged because the
running jobs use them; tests in `tests/test_carry12_eval.py`, end to end on CPU with a toy model at T 1,024
through `mtstudio chat` and ember). Forced choice (raw summed log-prob, `acc`; per token, `acc_norm`) in six
history modes: gold, self, near, near_self (the near rendering with the model's own replies), twin and noctx;
a rendering beyond the model's context is `over_ctx` (0 credit in `acc`, out of `acc_fit`, counted). Greedy
generation through ember's server after a spot-check against `mtstudio chat` (ember only on N/N identical
items, otherwise mtstudio for the whole run, as the tiny-chat post job), `EMBER_GEMM_TF32=0`,
`EMBER_DETERMINISTIC=1` (both set by carry12_eval for every ember server) at ember **ec3201b** (main; the main
session's pin, 27 Oct 2026; `carry12_l0.py --ember-commit`'s default). Its reasons: fe01b35's deterministic decode
(bit-identical run to run on one GPU model and build, ember df43463), 7526469's valid UTF-8 on every text path (a
reply cut inside a character at `--max-new` is no HTTP 500), and ec3201b's tiled split-K decode attention (per-layer
attention at 385 keys about 37 µs to 6 µs; 1.69-1.70x end to end at 60M on 245-430-token Carry-12 contexts; ids
360/360 against 7526469, deterministic mode bit-stable run to run, verify rows bit-identical to plain decode;
`ember.cpp/experiments/decode_occupancy_2026-10-27/`). Its deterministic logits differ from 7526469's by the new
fixed reduction order; `EMBER_DECODE_ATTN=split` on ec3201b reproduces 7526469 bit for bit. Every run's GPU model
is recorded, and all of a cell's generations are on L4s. Every
generated reply is scored again by `mtstudio score` (the parity gate); self history is played to the target
(`--self-scope target`, Stage L0) or the whole conversation. Curves by distance, target turn, interference,
distractor window, correction kind and family x distance, the self conversation by turn to 14, the HL0
slopes, and the pools and decomposition below (Y12, twin -> near, near -> gold, gold -> self). **The S cells'
state-token template (section 6):** for a model whose GGUF declares `paperkiln.state_tokens.k`, carry12_eval
inserts the state block after every `<|endoftext|>` that closes an exchange into its pre-tokenised forced-choice
and gate prompts (`prompt_ids`, `tools/state_tokens.py`'s rule; `mtstudio score` validates the blocks of
pre-tokenised prompts but inserts none), and checks ember's prompt ids against the same layout; `mtstudio chat`
and ember's /chat insert the blocks themselves. Tested end to end on a toy state-token model: the gold request
equals `mtstudio score` on the same prompt as text. carry5_eval (the post job's Carry-5) inserts them by the same
rule (`prompt_ids` and `state_template`, defined once in carry5_eval and imported by carry12_eval) only for a model
whose GGUF declares state tokens, and refuses a spec with `arch.state_tokens` whose GGUF does not; every other model's
prompts are byte-identical to its output before the change (`tests/test_carry5_state.py`: the whole Carry-5 request
list, gold, twin, noctx, self and gate rows, hashes as before on `pick_e1_L16`'s GGUF and on a toy vocabulary), and on
a toy state-token model its pre-tokenised gold and twin requests score what `mtstudio score` scores on the same text.

**Frozen v1:** `corpus/eval/carry12_v1.jsonl`, sha256
`de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797`, pinned in
`corpus/hashes/carry12_v1.sha256`; `python tools/carry12_build.py check` rebuilds it byte for byte
(`tests/test_carry12.py`).

## 3. Context length: T = 1,024

From `carry12_v1_budget.json` (the 16k Llama 3-scheme vocabulary, the served template; a rendering's length
equals the sum of its pieces, checked on 15 whole renderings):

| rendering | median | p99 | max | over 512 | over 1,024 |
|---|---:|---:|---:|---:|---:|
| gold history to the target, plus the reply | 269 | 339 | 353 | 0 | 0 |
| self history, 24 tokens per earlier reply | 423 | 510 | 515 | 4 | 0 |
| self history, 48 tokens per earlier reply | 687 | 822 | 827 | 1,452 | 0 |
| whole conversation, 48-token replies | 788 | 864 | 869 | 1,452 | 0 |
| twin | 40 | 67 | 73 | 0 | 0 |
| near | 65 | 289 | 336 | 0 | 0 |

(Near's long tail is the instruction family, whose compliant question-answer turns are all events.)

The canned replies are short (median 9 tokens); human chit-chat replies run 11-25 tokens at the median
(`dialogue_lengths_v1.json`), so 48 tokens per reply is a generous cap for self history. **T = 2,048 buys
nothing measurable here**: no Carry-12 item and only 38 of 6,429 long human dialogues need it for 12 turns.

**Cost of T = 1,024** (planning model; tokens per step fixed at 16,384, so batch 16 x 1,024). Attention's
share of training FLOPs at T 512 is 5% (20m_L2) to 21% (20m_L32); doubling T doubles that share. From
amendment 2's probes (A2.1), assuming the attention kernels run at the same efficiency as the rest or at half
of it:

| shape | tok/s at 512 (probe) | at 1,024 (est.) | at 2,048 (est.) | stored softmax per step at 512 / 1,024 / 2,048 |
|---|---:|---:|---:|---|
| 20m_L4 | 32,202 | 28,100-29,800 | 22,400-26,000 | 1.25 / 2.5 / 5.0 GiB |
| 20m_L8 | 27,948 | 23,000-25,000 | 17,100-20,600 | 1.75 / 3.5 / 7.0 GiB |
| 20m_L16 | 25,383 | 19,600-21,600 | 13,500-16,700 | 2.5 / 5.0 / 10.0 GiB |
| 20m_L32 | 21,856 | 16,200-18,000 | 10,600-13,300 | 3.0 / 6.0 / 12.0 GiB |
| 60m_L4 | 13,803 | 13,070-13,430 † | | 2.1 / 4.3 / 8.5 GiB † |
| 60m_L8 | 13,799 | 12,740-13,250 † | 9,600-11,100 | 3.0 / 6.0 / 12.0 GiB |
| 60m_L16 | 12,517 | 11,220-11,830 † | | 2.0 / 4.0 / 8.0 GiB † |
| 60m_L32 | 10,308 | 8,790-9,490 † | | 3.0 / 6.0 / 12.0 GiB † |
| 60m_X (60m_L16 + llama_attnres) | 8,360 † (0.668 x L16) | 7,500-7,900 † | | as 60m_L16 + 15% † |

† 60M rows: the attention term's share of training FLOPs at T 512 (2.8% at 60m_L4 to 8.6% at 60m_L32), doubled at
1,024, at the same efficiency as the rest or at half of it; X at X5's measured ratio to plain L16 on one L4 (0.668,
A5.12), which is conservative at d 512, where depth attention's cost per sublayer is a smaller share of the layer
than at d 320. Stored softmax per micro-batch at 16,384 tokens per step (batch 16 x 1,024 for the 60M shapes at
L <= 8; 8 x 1,024 with 2-step accumulation at L >= 16).

- mtstudio materialises each layer's attention probabilities ([heads x tokens, T] floats, kept for the
  backward pass; `src/cuda_ops.cu` `attn_heads_fwd`), so that memory grows with T at fixed tokens per step. At
  1,024 every probed shape stays inside the L4 (60m_L8 about 18.5 GiB against its 15.9 at 512); at 2,048 the
  60M shapes would not fit without accumulation or a fused attention kernel.
- Per run: about 10-21% more VM time at 20M for pretraining at 1,024 (20m_L8 1.8-2.0 h for 163.84M tokens).
- The probes are not FLOP-bound (20m_L32 is 1.55x 20m_L2 per token at about the same FLOPs), so this model is
  a guide: **Stage 0 of this programme probes every cell at T 1,024 before any cost is final.**
- ember serving: f32 KV is 2 x d x 4 bytes per layer per token, 28 KiB at 20m_L8: 28 MiB at 1,024 tokens.

## 4. Long human dialogue: what there is, and what can be done

`dialogue_lengths_v1.json` (`tools/dialogue_lengths.py`; a turn is one exchange, normalised as the data job
does; tokens in the template; "fits T" is chat_data's SFT rule, whole exchanges within T + 1):

| source | dialogues | turns p50 / p90 / max | reach 12 | reach 14 | reach 20 | tokens to turn 12 (p50) | first 12 fit 512 / 1,024 |
|---|---:|---|---:|---:|---:|---:|---|
| taskmaster | 28,791 | 9 / 13 / 89 | 5,161 | 2,572 | 286 | 337 | 4,970 / 5,158 |
| topical_chat | 8,605 | 10 / 11 / 26 | 773 | 122 | 11 | 662 | 112 / 738 |
| multiwoz | 8,292 | 7 / 10 / 22 | 348 | 95 | 6 | 470 | 265 / 348 |
| quac | 11,567 | 6 / 9 / 12 | 86 | 0 | 0 | 433 | 74 / 86 |
| dailydialog | 10,235 | 3 / 6 / 17 | 37 | 14 | 0 | 437 | 28 / 37 |
| convai2 | 8,951 | 7 / 8 / 25 | 23 | 11 | 3 | 354 | 23 / 23 |
| bst | 4,819 | 6 / 6 / 13 | 1 | 0 | 0 | 459 | 1 / 1 |
| self_dialogue | 24,133 | 10 / 10 / 10 | 0 | 0 | 0 | - | - |
| wow, oasst, empathetic, dolly, no_robots | 65,179 | 1-4 at the median | 0 | 0 | 0 | - | - |
| **13 sources** | **170,572** | | **6,429** | **2,814** | **306** | | **5,473 / 6,391** |

- Every self-dialogue conversation in the pinned corpus has at most 20 utterances (10 turns); the survey's
  "depth" counts utterances, so its 15.4 is 7.7 turns here.
- **The SFT split** (the data job's 10% by content hash): 668 dialogues reach 12 turns, 306 reach 14.
  Under the E5/E1 SFT rule at T 512, 90,167 of 92,636 SFT turns are trained and 5,708 of the 7,015 turns at
  position 10 or later; at T 1,024, 6,978 of 7,015. So T 1,024 recovers the last fifth of the late-turn
  signal, and the late-turn signal is small whatever T is: about 7,000 turns.
- **Whether late turns use early content** (`long_gap_echo`, a word-overlap proxy: turns 10+ that use a
  content word last seen in turns 1-4 and absent from every turn between): 9.9% of the 32,324 late turns in
  12+-turn dialogues (taskmaster 9.8%, topical_chat 11.3%, multiwoz 5.0%, quac 25.2%, dailydialog 20.4%).
  Natural long dialogue mostly talks about the recent past; overlap is not proof of a dependency, so this
  bounds the signal from above in spirit, not exactly.

**Is there enough to train on at T 1,024?** For pretraining exposure to long dialogue, yes at the level the
corpus allows: 2.9M tokens of 12+-turn dialogue, about 7% of the dialogue pool and whole at 1,024. For
supervised long-carry signal, little: a few hundred long SFT dialogues and of the order of a few hundred
late turns that revisit early content.

**Options, under the human-only rule** (concatenating related dialogues to fake one long conversation is
not allowed, and nor is any rearrangement that presents separate exchanges as one continuous conversation):
1. **Weight what exists (cell D below):** oversample the SFT dialogues of 10+ turns and train them whole at
   T 1,024. Cheap (SFT only) and within the rule. In the T 1,024 data job 3,698 of the 16,932 SFT rows hold 10 or
   more trained turns, so D's set has 28,026 rows (`stage0/carry12_data_t1024_ledger.json`, `long_sft`); by source
   (the SFT split of `dialogue_lengths_v1.json`, 3,702 dialogues of 10+ turns): Taskmaster 1,320, Self-Dialogue
   1,297 (every one of its dialogues has at most 10 turns, so these are exactly 10), Topical-Chat 875 (all of its
   SFT dialogues), MultiWOZ 113, QuAC 76, DailyDialog 14, ConvAI2 7.
2. **Train at T 1,024 throughout:** the 85% pretraining split's 12+-turn dialogues then sit whole in windows
   instead of across window boundaries (the E5/E1 mixes pack documents into 512-token windows).
3. **Add genuinely long human sources, by the survey's rules** (`corpus/HUMAN_DIALOGUE_SURVEY.md` section 9,
   fetcher group `long` in `corpus/fetchers/human_dialogue.py`, each a new corpus id). Pass: Taskmaster-3
   (`human_taskmaster3_v1`, CC BY 4.0, one worker writing both sides; its "auto template" dialogues, whose
   agent turns are the collectors' template lines, are dropped), ABCD (MIT), CCPE-M (CC BY 4.0) and
   PersuasionForGood (Apache-2.0); Multi-Session Chat (`human_msc_v1`, crowdworker-written, no licence stated by
   its authors) is held behind `--allow-unstated-licence`. Excluded: Conversation Chronicles and LoCoMo
   (LLM-written), DuLeMon (Chinese). Measured on the laptop (`dialogue_lengths_long_laptop_v1.json`; corpora in
   `C:/ml_artifacts/human_dialogue/hd_long/`): MSC
   3,989 dialogues, all 12+ turns (2,306 reach 20), ABCD 242, CCPE-M 119, PersuasionForGood 25; with these four
   the 12+-turn count is 10,804 of 184,093 dialogues (6,815 without MSC), and the SFT split's 12+ dialogues go
   from 668 to 1,151; Taskmaster-3 is estimated at about 740 (one of its 20 files), measured by the prepared
   Colab job `tools/colab_chat/human_dialogue_long_v1.json`. Rulings (25 Oct 2026, the main session under
   Jonathan's delegation): **MSC is excluded**: its authors state no data licence (the paper, the ParlAI
   archive and the project page parl.ai/projects/msc, checked 25 Oct 2026, carry none), and an unclear licence
   excludes a source. Recorded for a later version that finds a licence: an MSC episode (the same two personas
   continuing) counts as one conversation, and the ConvAI2 dialogues its first sessions repeat (3,987 of 3,989)
   are dropped. The long sources are therefore Taskmaster-3, ABCD, CCPE-M and PersuasionForGood. Taskmaster-3
   (`human_taskmaster3_v1`, pinned 25 Oct 2026, 16,675 dialogues, 5.37M tokens on the 16k vocabulary): its
   splits are per turn, not per dialogue (the train shards name 22,967 of its 23,757 dialogues), so a
   dialogue is held out only when its id appears in dev/test and never in train (2); 6,856 "auto template"
   dialogues drop. It adds 902 dialogues of 12+ turns (124 of 14+, none of 20+; median 10 turns), so the
   12+-turn count is 7,717 without MSC (`dialogue_lengths_long_v1.json`: the four long sources pooled hold
   1,288 of 12+ turns). **Not taken** (Jonathan, 27 Oct 2026; section 11 item 7): B trains on W512's sources, so HT
   stays a contrast of T alone.
4. **Packing unrelated dialogues into one window is allowed only as separate documents** (an
   `<|endoftext|>` between, as the mixes do now): it is ordinary packing, teaches nothing about long carry,
   and must never be presented as one conversation (no shared template run, no state tokens across the join).
5. **Hand-written templated long dialogues as training data** (human-written templates, not model text) meet the
   letter of the rule but would teach the benchmark's own templates, so they run only as a separate, registered
   arm with disjoint templates and value lists, scored only on a Carry-12 built from other templates. **Taken for
   one cell, T** (Jonathan's ruling, 27 Oct 2026; section 6), on those conditions: `tools/carry12_tsft_build.py`'s
   templates, entities and values share nothing with Carry-12 v1 or Carry-5 v1, and T is scored on Carry-12 v1.

## 5. Stage L0: the gap on E5/E1's models (no training)

Registered and frozen in `STAGE_L0.md` (before any L0 scoring). Every E5 and E1 pick (20M; 60M if run; one
SFT model per cell comes home) is scored on Carry-12 v1 at T 512, where it was trained: forced choice with
gold, near, twin and noctx history on every model (`tools/carry12_l0.py run --gen none`, on the laptop CPU),
and, on Colab, self and near_self history at `--max-new 24` played to the target (4 items over 512 are
reported as overflow; `colab-jobs` prepares the jobs file). Read: the validity gate, and HL0 on the E5 and
E1 winners' picks; everything else is descriptive. This needs no new training and answers the first
question: does E5/E1's best model already carry facts 10 turns, and how much does distance cost?

## 6. Cells

All new cells train at **T = 1,024** at **60M non-embedding parameters†**, E1's recipe otherwise (163.84M
tokens, AdamW, cosine, the same SFT recipe at 16 x 1,024 rows per step so SFT tokens per step stay 16,384),
seeds 1, 2, 3 paired by seed. Rung logging is not registered here.
- **SHAPE = 60m_L\*†**, the depth E1 at 60M favours, by the rule of the size arm (`../tiny_chat_2026-10/
  PREREGISTRATION.md` amendment 6, A6.1, `size/make_size_jobs.py depth`): the 60M depth with the highest mean Y over
  seeds 1-3 among cells passing the validity gate; depths within 1 point tied, the one with the higher A2.1 probe rate
  winning. One rule, so the size arm and this programme share one 60M shape. It fires once all twelve E1 at 60M runs
  are home.
- **The mix: 35%**, E1's level, on which every 60M run trained, so W512 is E1 at 60M's own L\* cell and HT reads T
  alone. E5's data bar came out at 0% (H5 CONTRADICTED: Carry-5's Y at 35% minus 0% is -0.046, 0/3 seeds above
  zero). The mix stays at 35% (decided 27 Oct 2026 by the main session; Jonathan delegated the choice): at 0% the
  pretraining mix holds no dialogue, so option 2's long dialogues whole in T 1,024 windows and S-gist's state
  slots (trained in pretraining dialogue documents) would reach the model through SFT alone, and the design is
  for long carry, which the bar's short-gap Y does not measure. Stage L0 points the same way descriptively (one
  pick each: E5's 0% pick D -0.066, its 17% pick -0.039; the 35% and 70% picks are read at the freeze). The bar's
  result is reported beside every Carry-12 table.
- **Learning rates†:** every cell trains at E1 at 60M's T 512 choice for its plain shape (`stage0/lr_choice.json`:
  60m_L4 and 60m_L8 5e-4, 60m_L16 and 60m_L32 1e-3); D, T, S-gist and S-full at B's, X at P16's (Kimi's convention, as
  A5.2). No T 1,024 screens: the rate held fixed keeps HT a contrast of T alone, and tokens per step do not change.

| cell | what | parameters vs B | status† |
|---|---|---|---|
| W512 | E1 at 60M's L\* cell itself (T 512): its three pretrained runs, re-SFT'd by E1's own SFT on B's machines and scored on Carry-12 by forced choice (its per-seed SFT models stayed on E1's VMs; Stage L0 scored the pick only) | = | 1.83 h per seed (section 10) |
| **B** | 60m_L\* at T 1,024 | = | core: every contrast below is against it |
| D | B's pretrained weights; SFT with 10+-turn dialogues oversampled x4 (`sft_human_long4.jsonl`: the T 1,024 data job's SFT rows, each row of 10+ trained turns four times, `--long-sft 10:4`; B's 1,500 steps) | = | core (data option 1, SFT only) |
| T | B's pretrained weights; SFT on B's rows plus 4,224 template dialogues with explicit recall targets, 20% of the rows (`sft_human_tsft.jsonl`, `--template-sft`; B's 1,500 steps) | = | core (Jonathan's ruling, 27 Oct 2026; data option 5) |
| S-gist | B + k = 4 state slots, `turn_state` mask (IR_SPEC_DRAFT section 1) | + 4d (1,536-4,352)† | core: E4, per-turn state, finished turns dropped |
| S-full | B + k = 4 state slots after every exchange, plain causal mask | + 4d† | conditional: only if HS1 (S-gist vs B) is SUPPORTED (S-gist loses), to place the cost |
| P16 | plain 60m_L16 at T 1,024 (= B if L\* = 16)† | = | core when L\* is not 16 |
| X | 60m_L16 + full attention residuals (`llama_attnres`) | +16,384 (0.02%)† | core: Jonathan's cross-layer hypothesis |
| R | the recursion arm's winner at 60M | E1-R's matching | conditional†: only if a recursion contrast at 20M is SUPPORTED |

**Why S-gist stays and S-full and R do not, at this budget†.** S-gist is the one cell built for long carry
(per-turn state, a cache about 5x smaller at 14 turns) and no arm has measured it, so it stays in the core and is
read against B directly. S-full's role is to split S-gist's cost between the insertion and the mask; it is needed
only if S-gist loses, so it runs only then (about 27-33 L4 VM-hours). R carried the recursion arm's result to long
carry, and at 20M fixed recursion and MoR have shown no advantage so far (`INTERIM_2026-10-25.md`, n 1-2 per cell).
R runs only if E1-R's HR2 (or HR2-J) or the MoR block's H-MoR1 is SUPPORTED, with its design (the winner's topology
recursing a 60M stack) and its cost (about 39 VM-hours for an R2-like cell plus its own screens) put to Jonathan
then.

**T: template SFT with explicit recall targets (Jonathan's ruling, 27 Oct 2026, on an outside reviewer's
suggestion; section 0's one exception).** B's pretrained weights; SFT at B's recipe for B's 1,500 steps of 16,384
tokens, so the SFT token budget is B's and the contrast with B is the data, not more steps; on B's machines like D,
seeds 1-3, paired by seed. Its SFT set (`sft_human_tsft.jsonl`, the data job's `--template-sft`) is B's 16,932 human
rows once each plus the template set's 4,224 dialogues, shuffled with the data job's seed: the templates are 20% of
T's rows (4,224 of 21,156) and about 21% of its tokens (1.01M beside the SFT split's 3.89M at T 1,024, section 4).
- **Why 20%.** Section 4 counts a few hundred natural late turns that revisit early content in the whole SFT split;
  4,224 dialogues with one carry target each give an order of magnitude more long-gap supervision, which a test of
  "explicit recall targets" needs. Four rows in five stay human, so every human row is still seen about 1.2 times
  in T's 24,000 row draws (1,500 steps of 16 rows, the last 5% of the file held out; B: about 1.5), and the checks
  of section 12 can see what the share costs general chat; above a third, the human rows would fall under one pass
  each. One proportion is registered; there is no dose curve.
- **The template set** (`tools/carry12_tsft_build.py`, seed 20261027, sha256 `3dbe77b5...`, pinned in
  `corpus/hashes/carry12_tsft_v1.sha256`; counts in `carry12_tsft_v1_build.json`). Every dialogue has one target turn
  whose correct reply needs a fact, a correction or a standing instruction from N turns back, N = 1-12 (352
  dialogues at each N), then a closing turn: 3-16 turns, 240 tokens on average (max 468), all whole at T 1,024.
  - recall (2,160): a fact about the user in one of six slots of the set's own (boat name, street, gym locker
    number, car, team, home language; 14-30 values each) at turn 1-3, asked N turns later; 0-3 same-slot facts
    about other people (eight relations of its own: flatmate, grandad, nephew ...) before the target.
  - correction (1,440): the fact planted at turn 1-3, corrected 1-4 turns later ("I mixed that up earlier. My boat
    is named X rather than Y."), asked N turns after the correction; 0-2 distractors.
  - instruction (624): a standing format instruction with a question at turn 1-3, in forms of its own (start every
    reply with Indeed / Certainly / Gladly / Absolutely, or end it with "no doubt" / "as ever" / "in fact" / "all in
    all"; Carry-12's are Sure / Okay / Right / Well and "of course" / "I think" / "you know" / "for sure"), every reply
    from then on complying, the target question N turns later.
  - Between the events: the set's own hand-written small talk (67 single exchanges, 16 two-turn arcs, 5 openers, 3
    closings), naming no value and raising no slot's subject. **No turn comes from the human dialogue pool:**
    splicing human exchanges into a templated conversation would present separate exchanges as one conversation
    (section 4's rule), and the pool is not scrubbed of Carry-12's values.
  - Carry-12's topic and load families have no counterpart (no digression; one user fact per dialogue), so the
    transfer reading of section 8 has an untrained side.
- **Disjointness, checked** (`carry12_tsft_v1_disjoint.json`, `python tools/carry12_tsft_build.py disjoint`):
  decontam.py's rules (word 13-grams, prompts, spans) with the data job's targets (the frozen eval sets, Carry-5 and
  Carry-12) drop 0 of 4,224 dialogues (6 simple-QA answer flags on everyday words such as "cold" and "light", flagged
  and kept as the data job keeps them); carry5_build's scan of Carry-12 v1 and of Carry-5 v1 finds no 13-gram,
  prompt, span, template sentence (any sentence of 3+ words), fact sentence or item leak; no value is shared,
  whole or as a content word with its plural (the set's 190 values against Carry-12's 790); no word of the sets'
  entity and slot-value lists (names, pet names, cities, jobs, favourites, topic values, animals, ages 18-79) or of
  Carry-12's instruction forms occurs anywhere in the set's text; and no user turn, reply or raw template is equal.
  The check catches planted overlaps (`tests/test_carry12_tsft.py`). The data job rebuilds the set, refuses bytes
  other than the pin and refuses the set if its own decontamination drops any dialogue.
- B, D and T also read the panels at their SFT weights (post `--sft-panels`): their pretrained weights are one, so
  their pretraining panels are equal by construction. B's machines end with the chat-prompt checks of B, D and T
  side by side (`chat_c12_r<seed>`: `tools/colab_chat/chat_eval_vm.sh`, tools/chat_eval.py on the frozen
  tinychat_hf_prompts_v1 and chat_prompts_v1, tools/qa_eval.py on simple_qa_v1).

**E4: per-turn state tokens (S-full, S-gist).** After every `<|endoftext|>` that closes an exchange, k = 4
learned state slots (rows of a `state_embd` [4, d] tensor, outside the vocabulary). Under the `turn_state`
mask a turn's text sees its own tokens and the state blocks of earlier turns; a state block sees its own
turn's text and every earlier state block (gisting's mask applied after every turn, with RMT-style chaining
through attention). Trained from scratch in pretraining dialogue documents and SFT, full backpropagation
through each window, no loss at state positions. At inference a finished turn's rows are dropped and its 4
state rows kept: after n turns the cache holds 4n state rows plus the current turn (about 5x smaller than
the full cache on a 14-turn Carry-12 conversation at 48-token replies). Dropping the rows is bit-exact on
ember's CPU path and rounding-level on CUDA (tiled attention), so S-gist's numbers do not depend on whether
the serving engine evicts. A reply cut short (here, by `--max-new 48`) is closed with `<|endoftext|>` and
the state block before the next user turn, and the training serialiser does the same, so self-history
conversations keep the segment layout training saw. k = 4: gisting found its models insensitive to k from 1
to 10, and RMT's gain faded beyond 5 memory tokens. Spec text, kernels and gates: `IR_SPEC_DRAFT.md`
sections 1 and 5.

**X: cross-layer connections (Jonathan's long-held intuition; `CROSSLAYER_LITCHECK.md`).** A mechanism that
learns, per token, which earlier activations to strengthen beyond the fixed residual sum. Full Attention
Residuals (Kimi 2603.15031, paperkiln's `nn::AttnResStack` with `block_size` 0) at sublayer level on the llama
base: the embedding and every attention and FFN output are sources (33 at 20m_L16), each sublayer's input
is a softmax over depth with a zero-initialised pseudo-query against the RMSNormed sources, so at
initialisation it equals the plain residual model. Positioned against DenseFormer (static weights, null at
436M in Kimi's ablation), Hyper-Connections (parallel streams, +26% activation memory), MUDDFormer (weights
generated per position from the current state, four streams for Q, K, V and the residual; the closest to
Jonathan's idea, but no published comparison with AttnRes and a new four-stream wiring) and value residual
learning (first-layer values into every layer). At **60m_L16†** (d 512, 8 heads, d_ff 1,760; 33 sources, 32
pseudo-queries of d 512; the depth the literature check chose, since depth attention needs depth, at the 60M base)
against P16, paired by seed, P16's learning rate (conservative against the arm, as Kimi). The mechanism is arm
X5's, unchanged (`../tiny_chat_2026-10/PREREGISTRATION.md` amendment 5): `llama_attnres` (paperkiln c48ca962,
keramos family `llama_attnres`), its CUDA kernels gated on an L4 and ember's gate passed at e115fd3 on CPU, CUDA and
WASM (A5.12). The X cell's post jobs generate with ember after the per-run spot-check, which is the gate at d 512.
Cost from X5's measured ratio, **0.668x plain L16's tokens/s†** (A5.12), not the 0.3x planned before the fused
`ops::depth_attend` existed. **X5's result at 20M is the context, not a prior:** HX5-main was EQUIVALENT on Carry-5
(D -0.010, 90% [-0.030, +0.010]), X5 lowered every loss by 0.03-0.05 nats and sat on the plain runs' loss-to-Y line,
and Carry-5 has no long gaps. HX tests the gap interaction, which Carry-5 could not.

**Step 2, registered as conditional (not costed in the core):**
- *on X's main effect* (section 8, HX-main SUPPORTED): a separate value route (one more pseudo-query per layer
  for the value input, the cheapest multiway step), and a recursion x attention-residual cell with every pass
  as a source and per-pass queries (IR_SPEC_DRAFT section 3's `"tensor_index": "application"`);
- *EXPLORATORY* (no hypothesis tested; no such combination found anywhere): MoR x attention residuals, MoR
  choosing how many passes and depth attention choosing what each pass reads (IR_SPEC_DRAFT section 4, a
  row-presence mask on the bank). Runs only if R is MoR and X has a main effect.

## 7. Metrics

**Primary, per run: Y12 = forced-choice accuracy (raw log-prob) with SELF history on Carry-12's primary
families minus that on their single-turn twins.** Primary families: recall, correction, topic, load and
knowledge (1,380 items). Instruction is out of Y here and reported by forced choice and by generation-scored
compliance; v1's instruction family is length-neutral (section 2), so its forced choice is not confounded
as Carry-5's was (A2.5), and whether it joins Y12 is decided at the freeze. Self history at `--max-new 48`,
**played to the target turn only† (`--self-scope target`, Stage L0's setting)**: Y12 scores the target, so the
turns after it never enter it; the self conversation after the target turn is not generated. The post jobs run
forced choice in five modes†: gold, self, near, twin and noctx (no near_self).

**Validity gate (as A2.5).** A cell's Y12 is interpreted only if, on its means over seeds, (i) twin accuracy
over the primary families is at least chance + 10 points (0.35), and (ii) twin minus question-alone accuracy
over the context families (recall, correction, topic, load: 1,320 items) is at least 10 points. A contrast
with a cell failing either is FLOOR.

**Secondary (descriptive unless named in a hypothesis):** Y12 with gold history; the decomposition
near - twin (splitting) and full - near (distance), with gold history† (the self-history decomposition needs
near_self, which the post jobs do not run); self minus gold at the target (error propagation); accuracy by distance (6-13), by
interference (0-3), by distractor window (early or late), by correction kind (early or late) and held turns,
by load; Carry-5's Y on the same model (the short-gap reference); the two panel losses (FineWeb-Edu and
dialogue val, 512 windows of 1,024 tokens), for B, D and T also at the SFT weights; S12 instruction compliance at
turns 10-14; for B, D and T the chat-prompt checks (clean stops, repetition, reference bits per byte on
tinychat_hf_prompts_v1 and chat_prompts_v1; simple-QA accuracy); tokens/s and peak memory; KV rows per turn in
serving (S cells); Y12 per L4-hour.

**SESOI:** 3 points for pairwise contrasts and for interaction contrasts; 1 point per turn of distance for
slopes on distance; 0.02 nats pairwise for panel losses.

## 8. Hypotheses, with directions (draft wording)

- **HL0, the distance cost (Stage L0, on E5/E1's winner, gold history):** accuracy falls with distance:
  acc(full) - acc(near) < 0 over the context families, beyond the SESOI (MT-Eval 2401.16745: distance to the
  relevant content drives degradation). Also descriptive: Y12 against Carry-5's Y on the same models.
- **HT, training at T 1,024 (B vs W512, gold history, which both fit):** Y12_gold(B) - Y12_gold(W512) > 0.
  Reason: whole long dialogues in pretraining windows and the last fifth of late SFT turns (section 4).
  Counter **HT-0:** EQUIVALENT within 3 points (long carry is limited by the scarcity of long-gap signal in
  the data, not by the window). Falsified by EQUIVALENT or CONTRADICTED.
- **HD, long-dialogue weighting (D vs B):** Y12(D) - Y12(B) > 0. Counter **HD-0:** EQUIVALENT (only ~10% of
  natural late turns revisit early content, so oversampling them teaches recency, not recall).
- **HT-SFT, template SFT with explicit recall targets (T vs B):** Y12(T) - Y12(B) > 0 beyond the SESOI (the template
  rows give the long-gap supervision natural dialogue lacks). Counter **HT-SFT-0:** EQUIVALENT within 3 points (the
  templates teach their own format, not carry: the model learns to answer the set's own slots and forms, and
  Carry-12's other templates see no gain). Falsified by EQUIVALENT or CONTRADICTED. **Transfer reading** (registered,
  secondary): T's gain over B in Y (self history minus twin, per pool, by item counts) on the Carry-12 families whose
  structure T's templates have (recall and correction, 840 items) and on those they lack (topic and load, 480 items),
  each by the verdict rule, and their difference. A SUPPORTED gain on topic and load reads as transfer beyond the
  trained structure; a SUPPORTED gain on recall and correction with topic and load EQUIVALENT reads as a gain that
  stays with the trained structure. Its limit: all of Carry-12's templates are new to T, so even the trained
  families test transfer across templates, slots and values; the reading separates family structure only, and
  knowledge and instruction are outside it (instruction is reported by compliance). **Beside HT-SFT**, as
  degradation checks (SUPPORTED = T worse than B beyond the SESOI; reported, not conditions on HT-SFT): Carry-5's Y
  (3 points), the dialogue and FineWeb-Edu panels at the SFT weights (0.02 nats), and, descriptively, the chat-prompt
  checks of B, D and T and S12 instruction compliance.
- **HS1, per-turn state against the base (S-gist vs B)†:** Y12(S-gist) - Y12(B) < 0 beyond the SESOI
  (compression into 4 rows per turn loses detail; gisting's failures were exactly verbatim details, and
  Carry-12's answers are verbatim values). Alternative **HS1-alt** (RMT: memory tokens learn compact
  storage; a bottleneck may force fact retention): the difference is at least -3 points (EQUIVALENT or
  positive), which would make the 5x smaller cache free. Descriptive beside it: the retention ratio
  (acc(S-gist) - acc(noctx)) / (acc(B) - acc(noctx)) by distance, and the KV rows saved.
- **HS0, insertion alone (S-full vs B), conditional†:** S-full runs only if HS1 is SUPPORTED (S-gist scores below B beyond the SESOI). Then: EQUIVALENT
  within 3 points (predicted; the slots carry no information under the plain mask), and the mask's own cost,
  Y12(S-gist) - Y12(S-full), is reported beside it. A SUPPORTED HS0 means the slots act as extra computation or
  sinks.
- **HR, recursion carried to long carry (R vs B), conditional†:** only if a recursion contrast at 20M (HR2,
  HR2-J or H-MoR1) is SUPPORTED. Then the sign of Y12(R) - Y12(B) equals that contrast's sign, a sign-transfer
  prediction like E1's 60M rule.
- **HX, Jonathan's cross-layer hypothesis (X vs P16 at 60m_L16†; CROSSLAYER_LITCHECK section 4):** registered prediction
  **H-gap**: the gap interaction
  D_X = [Y_gold,far(X) - Y_gold,far(P16)] - [Y_near(X) - Y_near(P16)] > SESOI, far = recall items (distances
  6-13, all 6+ turns back), near = the same items' near rendering; and the gain's slope on distance in the
  correction family (distances 1-11) is positive. Default counter-hypothesis (literature-led): the gain is a
  general language-model improvement: D_X inside +-3 points, and at matched FineWeb panel loss X sits on
  the plain cells' loss-to-Y line (the isoplot test, as HR3; at 60M the line runs through the plain 60M T 1,024
  cells, B and P16, and is flagged as thin†). Second counter: no main effect at 60M
  (**HX-main**, Y12(X) - Y12(P16), EQUIVALENT; X5's Carry-5 result at 20M is this pattern). Secondary, descriptive: the gain by distractor count; depth
  weights at the answer position against the mean over other assistant positions, with the embedding and
  early sources singled out (the direct test of "selective strengthening"); routing entropy per layer
  (collapse); tokens/s.

## 9. Analysis (`analyze.py` in this folder)

One number per (cell, seed); per-seed paired contrasts, one-sample t over seeds (n = 3, critical 4.303);
verdicts SUPPORTED, CONTRADICTED, EQUIVALENT (90% interval inside +-SESOI), SIGNIFICANT_BELOW_SESOI,
NOT_RESOLVED; the extension rule (seeds 4-5 once, for a primary contrast NOT_RESOLVED with one sign on all
three seeds); no peeking (`--partial` labels output DESCRIPTIVE, PARTIAL). W512's seeds are E1 at 60M's seeds
1-3†, so HT pairs by seed.

`analyze.py --art ART [--partial] [--lstar L] [--r-sign +1|-1] [--e1-home DIR]` reads every
`post_c12_<cell>_r<seed>/summary.json` the cells' jobs bring home and uses E5/E1's `decide()` (the verdict rule and
the extension flag) unchanged. Per cell: the validity gate of section 7 on means over seeds (a contrast with a failing
cell is FLOOR), Y12, Y12_gold, the decomposition, Carry-5's Y, the panels (and, for B, D and T, the panels at the SFT
weights) and instruction compliance. Registered: HT (Y12_gold, B - W512, with HT-0's reading), HD (D - B, HD-0),
HT-SFT (T - B, HT-SFT-0) with its transfer reading (recall and correction against topic and load) and the degradation
checks beside it (Carry-5's Y, the SFT-weight panels, and the chat-prompt checks read from B's machines'
`chat_c12_r<seed>` outputs, extracted or in `chat_eval_results.tar.gz`), HS1 (S-gist - B, direction negative, with HS1-alt: holds
on EQUIVALENT or CONTRADICTED) and its retention ratio by distance over the context families, HS0 and the mask's cost
when S-full is home, HR when R is home (`--r-sign` gives the 20M contrast's sign; SUPPORTED = the sign transferred),
HX's D_X on recall (gold far minus near, X - P16), the slope on distance of the gold-history gain in correction
(1 point per turn), H-gap (both SUPPORTED), HX-main and the isoplot residual (the line through B and P16 per seed,
flagged thin; none when L\* = 16, where P16 is B). Descriptive: the panel contrasts and W512's reproduction of E1's SFT
(its `sft_weights_sha256` against E1's home post summaries). Tested on synthetic summaries
(`tests/test_carry12_analyze.py`, 14 tests: every contrast and verdict, FLOOR, the extension rule and seeds 4-5, the
refusal of an incomplete core, T included, and `--partial`, the conditional cells, L\* = 16, HT-SFT SUPPORTED and
EQUIVALENT with each transfer reading, the degradation checks SUPPORTED and EQUIVALENT, missing SFT panels, and the
chat checks read from a tarball and from folders) and on the toy cells' own post summaries (section 11).

## 10. Budget (planning numbers; Stage 0's probes at T 1,024 replace them)

Per run at T 1,024†: pretraining at the 60M shape's T 1,024 rate (section 3's lower bound) times 1.08 for in-run
evaluations and checkpoints (E1 at 60M ran 6-11% under its probe rate); SFT at the same rate (0.6-0.8 h); post jobs
**3.5 h†**: Carry-5 with panels and base-model Carry-5 as E1 at 60M measured them (0.78-2.88 h, median 1.36), plus
Carry-12 with ember generation at `--max-new` 48, self history played to the target only and no near_self, costed
at **2.1 h†** and **not yet measured**. (Played to the target, about 17,400 replies per run against about 24,700
for the whole conversation plus near_self: 30% fewer, and one forced-choice mode fewer.) At $0.59 per L4-hour
(RunPod).

| block (60M, T 1,024)† | runs | L4 VM-hours by L\* = 4 / 8 / 16 / 32 |
|---|---:|---:|
| Stage L0 (forced choice on the E5/E1 picks, 60M included; self history on the winner's) | - | 4 |
| Stage 0 (CUDA gate, the T 1,024 data job, probes of B, S-gist, X and P16; no screens) | 4 | 1.5 |
| B (3 seeds; PT 3.8 / 3.9 / 4.4 / 5.6 h each) | 3 | 23.3 / 23.7 / 25.5 / 29.7 |
| D (SFT and post only, from B's weights) | 3 | 12.1 / 12.1 / 12.4 / 12.9 |
| S-gist (B's cost + 5% for the slots) | 3 | 24.0 / 24.4 / 26.2 / 30.6 |
| P16 (0 when L\* = 16) | 3 | 25.5 / 25.5 / 0 / 25.5 |
| X (60m_L16 at 0.668x: PT 6.6 h, SFT 1.0 h, post 3.5 h) | 3 | 33.0 |
| data and gate jobs (one per machine) | | 1.5 |
| **core** | 15-18 | **125 / 126 / 104 / 139** |
| | | **$74 / $74 / $61 / $82** |
| conditional: S-full (if HS1 is SUPPORTED) | 3 | 24.0-30.6 |
| conditional: R (if a 20M recursion contrast is SUPPORTED) | 3 | about 36, plus its screens |

- **Approved (Jonathan, 26 Oct 2026):** the core at 60M with these two cuts (self history to the target only, no
  near_self), every registered test kept, HX included; S-full and R conditional as above. The approval stands under
  the freeze rule: nothing launches before this file freezes, after E5, E1 (its 60M stage included) and the
  recursion arms report. The core is above 80 VM-hours at every L\*, which is why it went to him (section 8 rule 5's
  form).
- **Go (Jonathan, 27 Oct 2026):** on the measured total below, 158.8 L4 VM-hours, $93.68 (the core and W512's
  re-run), plus cell T, with a RunPod top-up of about $120 to cover the caps. With T, its SFT panels and the
  chat-prompt checks the total is **178.5 VM-hours, $105.29**, 1.42 times the approved planning number for L\* = 8
  (126), under the 1.5x rule. The twelve machines' caps now sum to $141 (below): the top-up covers the estimate, and
  the caps once the account's balance before it is at least $21.
- **The Carry-12 post evaluation was the largest unmeasured cost** (15 runs x 3.5 h, about 53 VM-hours of the core
  above). If the measured total is above 1.5 times these numbers, it goes back to Jonathan before launch; it goes
  to him in any case for the go of section 11 item 6.
- **The measurement (pre-freeze campaign `pk-c12-s0`, one RunPod L4, paperkiln 8ba23af2, $10 cap, launched 27 Oct
  2026; `tools/colab_chat/carry12_stage0_v1.json`).** Stage 0's probes at T 1,024 (300 steps alone on the GPU): B
  at 60m_L8 in both layouts (16 x 1,024, and 8 x 2 if that does not fit), S-gist, P16 and X. The Carry-12 part of a
  post job: `carry12_l0.py run --cost-run` on E1 at 60M's picks of the core's two shapes (60m_L8 for B, D and
  S-gist; 60m_L16 for P16 and X) and on `pick_e1_60m_L32`, served at context 1,024, in the post job's configuration
  (forced choice in gold, self, near, twin and noctx by CUDA `mtstudio score`; self history to the target at
  `--max-new` 48 by ember's CUDA server after the 20-item spot-check; no near_self; every reply through the parity
  gate). The generation time is an upper bound twice over: the picks were trained at T 512, so their replies past
  position 512 run to the cap more often than a T 1,024 model's, and the cost runs used ember 7526469, whose decode
  attention ec3201b (the pin, section 2) runs 1.69-1.70x faster end to end at 60M on Carry-12 contexts. **The total** (`stage0_total.py`, `stage0/total.json`): per
  run, E1 at 60M's measured walls at T 512 (`stage0/e1_60m_walls_t512.json`: pretraining 3.4-3.6 h at 60m_L8 and
  4.0-4.1 h at 60m_L16; SFT 0.67 h; the rest of the post job, panels and both Carry-5 runs, 2.4-2.6 h at 60m_L8 and
  3.1 h at 60m_L16, medians over seeds) with pretraining and SFT scaled by the probes' T 512 / T 1,024 step-time
  ratio, plus the measured Carry-12 part and its ember build, plus 0.37 h per machine for boot, CUDA gate and data
  job (pk-tc-s150-L8, 26 Oct 2026), on 12 machines (D follows B on B's); X's post job at P16's measured times (X's
  own generation is not measured).
- **The measured total: 153.3 L4 VM-hours, $90** (`stage0/total.json`), 1.22 times the approved planning number
  for L\* = 8 (126, $74), under the 1.5x rule; with W512's re-run, 158.8 VM-hours, $93.68, on which Jonathan gave
  item 6's go on 27 Oct 2026. Cell T and the checks beside it come after the table.

  | cell (3 seeds each) | T 1,024 tok/s (probe) | PT h | SFT h | rest of post h | Carry-12 post h | machine h | VM-hours |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | B (60m_L8, 16 x 1,024; 18.95 GiB peak) | 12,511 | 3.96 | 0.74 | 2.53 | 2.58 | 0.37 | 30.5 |
  | D (B's weights) | (B's) | 0 | 0.74 | 2.53 | 2.58 | 0 | 17.5 |
  | S-gist (k 4, turn_state; 19.2 GiB) | 12,230 | 4.05 | 0.76 | 2.53 | 2.58 | 0.37 | 30.8 |
  | P16 (60m_L16, 8 x 2) | 11,021 | 4.60 | 0.76 | 3.07 | 2.66 | 0.37 | 34.4 |
  | X (60m_L16 + llama_attnres, 8 x 2) | 8,115 (0.736 x P16) | 6.25 | 1.03 | 3.07 | 2.66 | 0.37 | 40.1 |
  | **core** | | | | | | | **153.3 ($90)** |

  Against the planning table above: pretraining is close to plan (B 3.96 h against 3.9 h; the probes ran about 2% under
  section 3's lower bounds at L8 and L16, and X 3% above its upper bound); the Carry-12 post part measured 2.58 h
  at 60m_L8 and 2.66 h at 60m_L16 (9,179 and 9,586 s, 29,040 replies each, ember spot-check 20/20, parity gate
  17,424/17,424) against 2.1 h planned; most of the excess is the rest of the post job, which E1 at 60M measured at
  2.5-3.1 h against the 1.4 h the plan left for it, and 0.37 h per machine that the plan did not count. Of the
  Carry-12 part, 2.2-2.3 h is `mtstudio score` (forced choice in five modes and the parity gate) and 13-15 minutes
  is ember's generation, so ember ec3201b changes the total by minutes, and batching `mtstudio score` is the lever
  on the post jobs. The 60m_L32 cost run (`pick_e1_60m_L32`) is descriptive and not part of the total.
- **Cell T and the checks beside it, from the measured rates** (`make_jobs.py --table`). T is D's job on another SFT
  file (B's weights, B's SFT steps, the same post job), so it costs D's measured 5.85 h per seed, plus 0.05 h for its
  panels at the SFT weights (W512's allowance for two panels): 5.90 h, **17.7 VM-hours, $10.44** over three seeds.
  B's and D's SFT-weight panels (0.05 h each) and the chat-prompt checks of B, D and T (0.5 h per machine, an
  allowance, not measured) add 0.60 h per B machine, 1.8 VM-hours, $1.06. Summed over the twelve machines' estimates:
  **178.5 VM-hours, $105.29.**
- **The cells' jobs files** (`make_jobs.py` in this folder; `tools/colab_chat/carry12_<B|Sgist|P16|X>_r<1|2|3>.json`,
  one RunPod L4 each, twelve machines): CUDA gate, the T 1,024 data job (the pre-freeze campaign's), a gate that refuses
  until this folder's `PREREGISTRATION.md` is committed (and checks the pre-freeze CUDA gate, design.json and every
  rate against `lr_choice.json`), the pretraining run on the probed spec (`specs/cell_*.json`, the Stage 0 probes'
  arch, data and train), and the post job: SFT at 16,384 tokens per step (`specs/sft_T1024.json`; D's
  `sft_T1024_long4.json`; T's `sft_T1024_tsft.json`), panels, Carry-5, then Carry-12, generation at ember ec3201b
  with `EMBER_DETERMINISTIC=1`. B's machine also runs D, T (its data job adds `--template-sft`), W512 (E1's
  pretrained weights uploaded, E1's SFT on E1's T 512 data job, Carry-12 forced choice: 1.83 h per seed, 5.5
  VM-hours, $3.24) and, last, the chat-prompt checks of B, D and T; B, D and T read the panels at their SFT weights
  too (`--sft-panels`). Caps, ceil(1.3 x estimate x $0.59): **B's machines (B, D, T, W512, chat checks; 24.4 h)
  $19**, S-gist's (10.3 h) $8, P16's (11.5 h) $9, X's (13.4 h) $11; **$141 over the twelve**. Each runner's deadline
  is 23.5 h, or ceil(1.3 x estimate) when that is longer: 32 h on B's machines.
- **Two cheaper layouts, Jonathan's call at the freeze:** when L\* is 32, X on B's shape instead of 60m_L16 (P16
  dropped, about -28 VM-hours, at the cost of the shape the literature check chose and a different X/plain throughput
  ratio, measured by the probe); when L\* is 4 or 8, X stays at 60m_L16 because depth attention needs depth.
- Wall-clock: one seed per L4 (A5.10's pattern), about 10-13 h per machine and 24 h on B's (B, then D, T, W512 and
  the chat checks); with the twelve machines at once the core comes home in about a day of runs.

## 11. What must be known before freezing

1. **E5's data bar** (the dialogue share) and its verdicts; **E1's 60M results and L\*** (A6.1's depth rule over
   all twelve E1 at 60M runs†) and whether any E cell sat on the FLOOR (if the 60M L\* cell is FLOOR on Carry-5, the
   long programme waits). Known: the bar is 0% and the mix stays at 35% (section 6); L\* is 60m_L8
   (`../tiny_chat_2026-10/size/depth_choice.json`).
2. **The recursion arm** (E1-R and MoR): HR2/HR2-J/H-MoR1 verdicts, which decide whether R runs at all† and, if it
   does, its design and HR's direction. Known: the MoR block is complete (H_MoR1 NOT_RESOLVED, +0.019, 2/3 seeds;
   `../tiny_chat_2026-10/analysis_2026-10-26.json`). Open: E1-R's last two seed-3 runs (R4, MC7), then HR2 and HR2-J.
3. **Stage L0's numbers**, read under `STAGE_L0.md` (frozen before any L0 scoring: the commands, the
   models, and that only the gate's quantities and HL0 are read), on the 60M picks†, and the measured cost of a
   Carry-12 post job (section 10). Known: the registered picks are scored (`l0/pick_e5_s0`, `pick_e1_L16` and
   `pick_e1_60m_L32`; every receipt so far passes the validity gate); the descriptive picks score on the laptop's
   CPU. The measured cost of a Carry-12 post job is home (section 10: 2.58 h at 60m_L8, 2.66 h at 60m_L16).
4. **Carry-12 v1** is frozen with its pin, `tools/carry12_eval.py` and tests. Known: Taskmaster-3 is scanned (no
   item leak, no drop-rule hit; section 2), the FineWeb-Edu shard is scanned (no item leak, no drop-rule hit;
   section 2), and the T 1,024 data job drops dialogues by Carry-12's rules (53 of 170,572). Nothing open.
5. **Engine work, each through IR_SPEC_DRAFT's gates.** Known: the `turn_state` mask and kernels in mtstudio, the
   extended embedding and the early-stop rule in the serialiser (`nn::insert_state_ids`, `tools/state_tokens.py`;
   `state_build_receipt.json`; the CUDA leg in every CUDA gate); in ember the mask with every row kept, then
   eviction, and the early-stop rule (ember 2cb4d2a and e5f4470 on CPU, gate 2 bit-equal 40/40 with the cut-reply
   prompt; aadeed2 and 210f3b8 on CUDA, greedy 40/40, logits within 6.2e-7; 3b86e08 in WebAssembly);
   `llama_attnres` in mtstudio, its export and ember's gate (arm X5, A5.1 and A5.12†); the S cells' template in
   carry12_eval (section 2; end to end on a toy state-token model in `tests/test_carry12_eval.py`); the T 1,024 data
   job (`tiny_chat_data.py --T 1024 --carry12 --long-sft 10:4`, tested on CPU, run in the pre-freeze campaign:
   `stage0/carry12_data_t1024_ledger.json`). Settled by the main session (27 Oct 2026): the S cells' Carry-5 is
   scored on the layout they train on. carry5_eval inserts the state blocks only for a model whose GGUF declares
   state tokens (the family `llama_state`; section 2), and `tests/test_carry5_state.py` holds every other model's
   prompts byte-identical to carry5_eval's output before the change (the whole Carry-5 request list on
   `pick_e1_L16`'s GGUF and on a toy vocabulary), so E5, E1, the size arm and the cells B, D, P16 and X are scored as
   before; on a toy state-token model its requests score what `mtstudio score` scores on the same text. D's threshold stays at 10+ trained turns: the rule is a turn count, not a source list, and it takes in
   Self-Dialogue's 1,297 ten-turn dialogues and every Topical-Chat dialogue (section 4, option 1); 11+ would leave
   out all of Self-Dialogue and most of Topical-Chat (its p90 is 11 turns) and shrink D's long rows by about half.
6. **Stage 0 at T 1,024:** probes replace section 10's estimates; Jonathan's go on the measured total. Known: the
   probes and the measured total, 153.3 L4 VM-hours, $90 (section 10), and W512's re-run on B's machines (E1 at
   60M's per-seed SFT models are not home; HT pairs by seed), 5.5 VM-hours, $3.24: 158.8 VM-hours, $93.68.
   Jonathan gave the go on that total on 27 Oct 2026, with cell T, and a RunPod top-up of about $120; with T and the
   checks beside it, 178.5 VM-hours, $105.29, and the caps sum to $141. Open: the top-up itself.
7. **Data option 3 (new long human sources): not taken** (Jonathan, 27 Oct 2026). B's data stays W512's, so HT stays a
   contrast of T alone. What the four licensed long sources (Taskmaster-3, ABCD, CCPE-M, PersuasionForGood) would
   have added to the T 1,024 data job, counted by the pre-freeze campaign (`stage0/carry12_data_long3_ledger.json`;
   nothing trains on it): 196,779 dialogues against 170,572 (56 dropped by the decontamination against 53),
   pretraining dialogue text 178.4 MB against 149.5 MB (the 35% mix still takes two copies), SFT rows 19,461 against
   16,932, of which 5,351 hold 10+ turns against 3,698 (D's set 35,514 rows against 28,026).
8. **S-full's trigger** (settled by the main session, 27 Oct 2026): S-full runs only if HS1 is SUPPORTED, that
   is, S-gist scores below B beyond the SESOI, the case in which S-full places S-gist's cost between the
   insertion and the mask (section 6: "only if S-gist loses"). `analyze.py` applies this one reading.
9. **Cell T** (Jonathan's ruling, 27 Oct 2026): the template set is built, pinned and checked disjoint from Carry-12
   v1 and Carry-5 v1 (section 6); the data job's `--template-sft`, the cell on B's machines, its SFT-weight panels,
   the chat-prompt checks and HT-SFT in `analyze.py` are tested (`tests/test_carry12_tsft.py`, 6 tests;
   `tests/test_carry12_analyze.py`, 14; `tests/test_carry12_jobs.py`, 5) and exercised end to end at toy size (below).
   Nothing open.

For the freeze itself (the header's licence: the jobs files exercised end to end on CPU): the cells' jobs files
(section 10: pretraining, SFT and the post job with Carry-12 after Carry-5, generation at ember ec3201b) and section
9's `analyze.py` exist and ran end to end on CPU at toy size on 28 Oct 2026 (`tests/test_carry12_cells_e2e.py`,
4/4, on a clean ext4 clone at paperkiln 6d6af447; receipt `stage0/toy_e2e_2026-10-28.json`): for each of the twelve
machine files, its own pretraining spec and post command lines with the VM's paths mapped to the toy's (width d 64,
2 or 4 layers, a 700-token vocabulary whose text includes every eighth template dialogue, a few steps, 2,048 tokens
per step and 16 panel windows through two test-only overrides, Carry-5's every 26th item and Carry-12's every
110th), seeds 1-3: the T 1,024 data job with Carry-12, D's long set and T's template rows (the pinned set rebuilt,
none dropped by the decontamination, every dialogue whole), E1's T 512 data job, pretraining with `mtstudio run`
(state slots in S-gist, attention residuals in X), SFT and export, the panels (B, D and T also at the SFT weights),
Carry-5 and base-model Carry-5, then Carry-12. 21 post jobs (T's from B's weights with B's steps, on its own SFT
file): every ember spot-check identical at ec3201b on CPU, S-gist and X included (Carry-5 2/2 items, Carry-12
14/14), every parity-gate reply identical (Carry-5 3,600 of 3,600, Carry-12 2,790 of 2,790), no rendering over the
context, a rerun of a finished post job a no-op; the chat checks (the chat job's NAME@JOB models through
tools/chat_eval.py and ember's CPU server, 3 prompts) on B, D and T for each seed; `analyze.py`'s registered path on
the 21 summaries and the chat checks (the toy models sit at chance, so every registered contrast, HT-SFT and its
transfer reading included, reads FLOOR, the gate path). What the freeze commit adds:
`PREREGISTRATION.md` in this folder, which every jobs file's gate requires, and the runner entries printed in the
jobs files' comments, their `--commit` set to that commit.

## 12. Threats, stated before data

- **Templated proxy:** Carry-12 measures carrying templated facts through templated small talk; a gain is not
  a claim of general long-conversation quality. Gold-history scores bound the self-history reading.
- **The base changes with T:** B differs from W512 in window length for pretraining and SFT together; HT reads
  the bundle.
- **Teaching to the test format (cell T):** with no template, slot, entity or value shared, T's rows still share
  Carry-12's task shape (a fact, turns later a question about it, the answer in a short frame), so a gain can be the
  shape learnt rather than carry. HT-SFT-0 names this outcome; the transfer reading separates the family structures
  T trained on from those it did not; the twin and near decompositions show where a gain sits; and T's instruction
  family is reported by compliance, not counted in Y12.
- **Natural-dialogue performance (cell T):** T draws 20% fewer human SFT rows than B and spends them on templated
  dialogue, so a Carry-12 gain can be bought with general chat. Carry-5's Y, the dialogue and FineWeb-Edu panels at
  the SFT weights and the chat-prompt checks (clean stops, repetition, reference bits per byte, simple QA) are
  reported beside HT-SFT for B, D and T, so that price is visible; one template share is registered, with no dose
  curve.
- **One budget** (163.84M tokens), three seeds, one SFT recipe, as E5/E1.
- **State-slot positions consume window positions** (about 4 per exchange; a few per cent of dialogue
  windows), so the S cells see slightly fewer real tokens; reported.
- **AttnRes cost and convergence:** a gain at this budget may be faster convergence rather than capacity
  (the literature check's failure modes); the isoplot test and the panel loss bound the reading.
- **Embedding confound** where shapes differ (X and P16 at 60m_L16 against B at 60m_L\*; R if it runs), as E1-R
  states.
- **Data per parameter†:** 163.84M tokens is 2.7 tokens per non-embedding parameter at 60M (8.2 at 20M), so a null
  at 60M can be under-training as well as an absent effect; amendment 6's size slope on Carry-5 reads the same
  budget across 20M, 60M and 150M.

## 13. The 20M planning values the † numbers replace

| item | 20M plan (24 Oct 2026) | 60M plan† |
|---|---|---|
| base shape B | `SHAPE`, E1's 20M winner | 60m_L\*, A6.1's rule on E1 at 60M |
| mix | `BAR`, E5's data bar | 35% (E1's level; the bar is 0%, section 6) |
| X and P16 | 20m_L16 (d 320), +10,240 query parameters (0.04% of all, 0.05% of non-embedding) | 60m_L16 (d 512), +16,384 (0.02%) |
| X's throughput | 0.3x P16 (unfused AttnRes in the literature) | 0.668x (X5's probe, A5.12) |
| state slots | + 4d at 20M | + 4d = 1,536-4,352 |
| S-full | core, positive control | conditional on HS1 SUPPORTED |
| HS1 | S-gist vs S-full | S-gist vs B |
| R | core, 20m_L4 recursion, about 14 VM-hours | conditional on a SUPPORTED 20M recursion contrast, about 39 |
| learning rates | screens for B, S-gist and X at 3 rates | E1 at 60M's T 512 choice, no screens |
| Stage 0 | 13 jobs, 6-9 VM-hours | 4 probes, 1.5 VM-hours |
| Carry-12 generation in the post jobs | self history over the whole conversation, plus near_self | self history to the target only, no near_self |
| post job per run | 0.5 h | 3.5 h (Carry-5 measured at 60M; Carry-12 unmeasured, 2.1 h) |
| core budget | 77-89 VM-hours | 104-139 VM-hours, $61-82 (approved by Jonathan, 26 Oct 2026) |
