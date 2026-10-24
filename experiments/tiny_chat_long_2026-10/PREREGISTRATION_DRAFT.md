# Tiny chat, long carry (Carry-12): pre-registration DRAFT

*DRAFT, 24 Oct 2026. NOT FROZEN, NOT LICENSED, NOTHING LAUNCHED. This file plans the programme that follows
E5/E1 (`../tiny_chat_2026-10/PREREGISTRATION.md` and its amendments, which this draft does not change). It
freezes only after E5, E1 and the recursion arms have reported (section 11 lists what must be known), as a new
file `PREREGISTRATION.md` in this folder, licensed by the commit that adds it with the frozen Carry-12 v1, its
pin, the evaluation tool and the jobs files, each exercised end to end on CPU at toy size. Until then every
number below is a planning number and every rule is open to change without an amendment.*

Receipts in this folder: `carry12_v1_build.json` (counts and balance), `carry12_v1_budget.json` (context
lengths), `carry12_v1_decontam.json` (contamination), `dialogue_lengths_v1.json` (the human corpora in turns
and tokens), `STAGE_L0.md` (Stage L0's frozen registration), `IR_SPEC_DRAFT.md` (spec text for ir.hpp
review), `CROSSLAYER_LITCHECK.md` (cross-layer arm, by another agent). Tools: `tools/carry12_build.py`,
`tools/carry12_eval.py`, `tools/carry12_l0.py`, `tools/dialogue_lengths.py`.

## 0. Objective and constraints

- **Objective (Jonathan, 24 Oct 2026):** "We do want them to get some recall to carry a conversation for, I
  don't know, at least a dozen turns." The ability measured: using a fact from turns 1-4 at turns 10-14.
- **Human-written data only, no distillation** (as E5/E1, section 0). Templates written by us are not model
  text; they are evaluation material and never enter training (section 4 says why they stay out).
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
programme's data job adds Carry-12 to the frozen sets beside Carry-5 and drops dialogues by those rules,
counted. The VM's FineWeb-Edu shard and Taskmaster-3 are not on the laptop; their scans are VM data-job steps
before Stage 0.

**Scoring: `tools/carry12_eval.py`** (built on carry5_eval's internals, which stay unchanged because the
running jobs use them; tests in `tests/test_carry12_eval.py`, end to end on CPU with a toy model at T 1,024
through `mtstudio chat` and ember). Forced choice (raw summed log-prob, `acc`; per token, `acc_norm`) in six
history modes: gold, self, near, near_self (the near rendering with the model's own replies), twin and noctx;
a rendering beyond the model's context is `over_ctx` (0 credit in `acc`, out of `acc_fit`, counted). Greedy
generation through ember's server after a spot-check against `mtstudio chat` (ember only on N/N identical
items, otherwise mtstudio for the whole run, as the tiny-chat post job), `EMBER_GEMM_TF32=0`, every
generated reply scored again by `mtstudio score` (the parity gate); self history played to the target
(`--self-scope target`, Stage L0) or the whole conversation. Curves by distance, target turn, interference,
distractor window, correction kind and family x distance, the self conversation by turn to 14, the HL0
slopes, and the pools and decomposition below (Y12, twin -> near, near -> gold, gold -> self). Still to add
for the S cells: the state-token template (section 6).

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
| 60m_L8 | 13,799 | 12,000-12,800 | 9,600-11,100 | 3.0 / 6.0 / 12.0 GiB |

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
1. **Weight what exists (cell D below):** oversample the SFT dialogues of 10+ turns (Taskmaster,
   Topical-Chat, MultiWOZ, QuAC, DailyDialog, ConvAI2) and train them whole at T 1,024. Cheap (SFT only) and
   within the rule.
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
   Colab job `tools/colab_chat/human_dialogue_long_v1.json`. Jonathan's rulings needed: whether an MSC episode
   (the same two personas continuing, the workers possibly changing between sessions) is one conversation;
   MSC's unstated licence; dropping the ConvAI2 dialogues MSC's first sessions repeat (3,987 of 3,989).
4. **Packing unrelated dialogues into one window is allowed only as separate documents** (an
   `<|endoftext|>` between, as the mixes do now): it is ordinary packing, teaches nothing about long carry,
   and must never be presented as one conversation (no shared template run, no state tokens across the join).
5. **Hand-written templated long dialogues as training data** (human-written templates, not model text) would
   meet the letter of the rule but teach the benchmark's own templates; if ever wanted, it is a separate,
   registered arm with disjoint templates and value lists, scored only on a Carry-12 built from other
   templates. Not proposed here.

## 5. Stage L0: the gap on E5/E1's models (no training)

Registered and frozen in `STAGE_L0.md` (before any L0 scoring). Every E5 and E1 pick (20M; 60M if run; one
SFT model per cell comes home) is scored on Carry-12 v1 at T 512, where it was trained: forced choice with
gold, near, twin and noctx history on every model (`tools/carry12_l0.py run --gen none`, on the laptop CPU),
and, on Colab, self and near_self history at `--max-new 24` played to the target (4 items over 512 are
reported as overflow; `colab-jobs` prepares the jobs file). Read: the validity gate, and HL0 on the E5 and
E1 winners' picks; everything else is descriptive. This needs no new training and answers the first
question: does E5/E1's best model already carry facts 10 turns, and how much does distance cost?

## 6. Cells

All new cells train at **T = 1,024** with the E5 data bar and the E1 shape winner (placeholders `BAR`,
`SHAPE` until E5/E1 report), E1's recipe otherwise (163.84M tokens, AdamW, cosine, the same SFT recipe at
16 x 1,024 rows per step so SFT tokens per step stay 16,384), seeds 1, 2, 3 paired by seed. Rung logging is
not registered here.

| cell | what | parameters vs B | why |
|---|---|---|---|
| W512 | E5/E1's winner cell itself (T 512): existing runs, Stage L0 scores | = | the T contrast for free |
| **B** | the base at T 1,024 | = | every contrast below is against it |
| D | B's pretrained weights; SFT with 10+-turn dialogues oversampled x4 | = | data option 1 (SFT only) |
| S-full | B + k = 4 state slots after every exchange, plain causal mask | + k d | positive control: the insertion alone |
| S-gist | B + k = 4 state slots, `turn_state` mask (IR_SPEC_DRAFT section 1) | + k d | E4: per-turn state, finished turns dropped |
| R | the recursion arm's winner (fixed or MoR at 20m_L4), at T 1,024 | E1-R's matching | carry the E1-R result to long carry |
| P16 | plain 20m_L16 at T 1,024 (= B if SHAPE is 20m_L16) | = | the cross-layer arm's control |
| X | 20m_L16 + full attention residuals (`llama_attnres`) | +10.6K (0.05%) | Jonathan's cross-layer hypothesis |

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
learning (first-layer values into every layer). At 20m_L16 (the shape the literature check chose: depth
attention needs depth) against P16, paired by seed, P16's learning rate (conservative against the arm, as
Kimi). A port is needed: the `attnres` family is GPT-2 blocks only. ember already runs the ops, but they
have been gated only on GPT-2 blocks, so the first trained tiny `llama_attnres` model passes an ember gate
on CPU, CUDA and WASM (the recursion gate's files) before any X run relies on it (IR_SPEC_DRAFT sections 2
and 5). mtstudio composes depth attention from tape ops; unfused AttnRes has
been reported at 0.2-0.5x training throughput, so tokens/s is reported and the cell is costed at 0.3x.

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
as Carry-5's was (A2.5), and whether it joins Y12 is decided at the freeze. Self history at `--max-new 48`, the whole conversation played (it
fits T 1,024: section 3).

**Validity gate (as A2.5).** A cell's Y12 is interpreted only if, on its means over seeds, (i) twin accuracy
over the primary families is at least chance + 10 points (0.35), and (ii) twin minus question-alone accuracy
over the context families (recall, correction, topic, load: 1,320 items) is at least 10 points. A contrast
with a cell failing either is FLOOR.

**Secondary (descriptive unless named in a hypothesis):** Y12 with gold history; the decomposition
near - twin (splitting) and full - near (distance), gold and self; accuracy by distance (6-13), by
interference (0-3), by distractor window (early or late), by correction kind (early or late) and held turns,
by load; Carry-5's Y on the same model (the short-gap reference); the two panel losses (FineWeb-Edu and
dialogue val, 512 windows of 1,024 tokens); S12 instruction compliance at turns 10-14; tokens/s and peak
memory; KV rows per turn in serving (S cells); Y12 per L4-hour.

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
- **HS1, the state mask's cost (S-gist vs S-full):** Y12(S-gist) - Y12(S-full) < 0 beyond the SESOI
  (compression into 4 rows per turn loses detail at 20M; gisting's failures were exactly verbatim details,
  and Carry-12's answers are verbatim values). Alternative **HS1-alt** (RMT: memory tokens learn compact
  storage; a bottleneck may force fact retention): the difference is at least -3 points (EQUIVALENT or
  positive). Descriptive beside it: the retention ratio (acc(S-gist) - acc(noctx)) / (acc(S-full) -
  acc(noctx)) by distance, and the KV rows saved.
- **HS0, insertion alone (S-full vs B):** EQUIVALENT within 3 points (predicted; the slots carry no
  information under the plain mask). A SUPPORTED difference means the slots act as extra computation or
  sinks, and HS1 is then read against S-full only.
- **HR, recursion carried to long carry (R vs B):** the sign of Y12(R) - Y12(B) equals the sign of the
  recursion arm's registered contrast on Carry-5 (HR2, or H-MoR1 if MoR), a sign-transfer prediction like
  E1's 60M rule. Its direction is fixed when E1-R reports, before this file freezes.
- **HX, Jonathan's cross-layer hypothesis (X vs P16; CROSSLAYER_LITCHECK section 4):** registered prediction
  **H-gap**: the gap interaction
  D_X = [Y_gold,far(X) - Y_gold,far(P16)] - [Y_near(X) - Y_near(P16)] > SESOI, far = recall items (distances
  6-13, all 6+ turns back), near = the same items' near rendering; and the gain's slope on distance in the
  correction family (distances 1-11) is positive. Default counter-hypothesis (literature-led): the gain is a
  general language-model improvement: D_X inside +-3 points, and at matched FineWeb panel loss X sits on
  the plain cells' loss-to-Y line (the isoplot test, as HR3). Second counter: no main effect at 20M
  (**HX-main**, Y12(X) - Y12(P16), EQUIVALENT). Secondary, descriptive: the gain by distractor count; depth
  weights at the answer position against the mean over other assistant positions, with the embedding and
  early sources singled out (the direct test of "selective strengthening"); routing entropy per layer
  (collapse); tokens/s.

## 9. Analysis (as E5/E1 section 7, `analyze.py` style, to be written and tested on synthetic summaries)

One number per (cell, seed); per-seed paired contrasts, one-sample t over seeds (n = 3, critical 4.303);
verdicts SUPPORTED, CONTRADICTED, EQUIVALENT (90% interval inside +-SESOI), SIGNIFICANT_BELOW_SESOI,
NOT_RESOLVED; the extension rule (seeds 4-5 once, for a primary contrast NOT_RESOLVED with one sign on all
three seeds); no peeking (`--partial` labels output DESCRIPTIVE, PARTIAL). W512's seeds are E5/E1's seeds 1-3,
so HT pairs by seed.

## 10. Budget (planning numbers; Stage 0's probes at T 1,024 replace them)

Per run at T 1,024: pretraining at the shape's estimated rate (section 3), SFT about 0.3 h with the device
masked cross-entropy, post jobs about 0.5 h (Carry-12 with ember generation of 14-turn conversations, Carry-5,
panels).

| block | runs | L4 VM-hours |
|---|---:|---:|
| Stage L0 (forced choice on ~24 E5/E1 SFT models, self history on the winner's 3) | - | 4 |
| Stage 0 (probes for B, S-gist, X, R at 1,024; lr screens for B, S-gist, X at 3 rates) | 13 | 6-9 |
| B, S-full, S-gist (3 seeds each, 20m_L8-like cost ~2.9 h) | 9 | 26 |
| D (SFT only, from B's weights) | 3 | 2 |
| R (fixed or MoR at 20m_L4, ~4.8 h) | 3 | 14 |
| P16 (0 if SHAPE is 20m_L16) | 3 | 0-9 |
| X (depth attention at 0.3x throughput: ~7.5 h pretraining + 0.8 h) | 3 | 25 |
| **core** | | **77-89** |

At about 4.8 compute units per L4 hour, 370-430 units. Over 80 VM-hours, section 8 rule 5's form applies:
Jonathan decides before launch. Order if cut: L0, then B with D and the X pair, then the S cells, then R.
X's cost falls by up to two thirds if mtstudio gets a device depth-attention path first.

## 11. What must be known before freezing

1. **E5's data bar** (the dialogue share) and its verdicts; **E1's shape winner** (20M; 60M if run) and
   whether any E cell sat on the FLOOR (if the winner is FLOOR on Carry-5, the long programme waits).
2. **The recursion arm** (E1-R and MoR): HR1/HR2/H-MoR1 verdicts, which fix R's design and HR's direction.
3. **Stage L0's numbers**, read under `STAGE_L0.md` (frozen before any L0 scoring: the commands, the
   models, and that only the gate's quantities and HL0 are read).
4. **Carry-12 v1** is frozen with its pin, `tools/carry12_eval.py` and tests; still open: the VM-side
   decontamination against the FineWeb-Edu shard and Taskmaster-3.
5. **Engine work, each through IR_SPEC_DRAFT's gates:** the `turn_state` mask and kernels in mtstudio, the
   extended embedding and the early-stop rule (`<|endoftext|>` and the state block after a reply that did
   not end with one, in the serialiser and in ember); in ember the mask with every row kept first, then
   eviction, gated bit-equal on CPU and on CUDA by greedy 40/40 with logits within ROUNDING (gate 2);
   `llama_attnres` in mtstudio and its export, and the ember gate on its first trained tiny model; the
   T 1,024 data job (the same sources and split at T 1,024).
6. **Stage 0 at T 1,024:** probes replace section 10's estimates; Jonathan's go on the measured total.
7. Jonathan's ruling on data option 3 (new long human sources), if wanted before the freeze.

## 12. Threats, stated before data

- **Templated proxy:** Carry-12 measures carrying templated facts through templated small talk; a gain is not
  a claim of general long-conversation quality. Gold-history scores bound the self-history reading.
- **The base changes with T:** B differs from W512 in window length for pretraining and SFT together; HT reads
  the bundle.
- **One budget** (163.84M tokens), three seeds, one SFT recipe, as E5/E1.
- **State-slot positions consume window positions** (about 4 per exchange; a few per cent of dialogue
  windows), so the S cells see slightly fewer real tokens; reported.
- **AttnRes cost and convergence:** a gain at this budget may be faster convergence rather than capacity
  (the literature check's failure modes); the isoplot test and the panel loss bound the reading.
- **Embedding confound** where shapes differ (R at 20m_L4 against B at SHAPE), as E1-R states.
