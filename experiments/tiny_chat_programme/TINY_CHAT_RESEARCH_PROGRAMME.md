# Tiny-chat research programme

The durable record of the tiny-chat research question, its experiments and its results. Kept by the Claude
screening lane (Claude Opus 5.5 since 4 Nov 2026); the stack's ownership rules are in the root workspace's
`LOCAL_LLM_STACK.md`. Each entry states what was measured, with hashes and where the receipts are. Superseded
readings stay in place, marked as superseded, with the reason.

## 1. Research question

> Given a compute-constrained language model of about this size, what levels of conversational coherence,
> salience and relevance can be achieved, and which architecture, training-data distribution and optimisation
> procedure achieve the strongest results?

Set by Jonathan on 4 Nov 2026. This is an empirical programme: it does not assume that the model's size imposes a
known ceiling. The guiding rule is to measure the failure, isolate the mechanism, change the smallest relevant
variable, and test whether the improvement transfers.

## 2. The model

| | |
|---|---|
| Architecture | Llama-style decoder: 8 layers, d_model 768, FFN 2,240 (SwiGLU), 12 heads (12 KV heads), context 1,024 |
| Parameters | 72,761,088 unique: 60.17M in the 8 layers, 12.58M in the tied token embedding (16,384-token BPE vocabulary), plus norms and a 3,072-value state embedding. Source: the native PT result and the safetensors header (Codex's check, `reviews/codex_sfull_cost_and_parameter_check_20261104.json` in the root workspace). The GGUF export stores the tied embedding twice, as `token_embd` and `output.weight`, so its tensor total of 85.3M is export storage, not trainable parameters. |
| Trainer | paperkiln `mtstudio`, numerical source `1ccb693f55d7…` |
| Base for every screen | `sfull_sft1500.safetensors`: the preserved S-full baseline (pretraining PT10000, then SFT1500), owned by Codex |
| SFT recipe (arm A) | 256 updates × 16 rows, lr 3e-5, loss on assistant segments only, EOS supervised; spec `arm_a_lr3e5.json` (`66c718e0`) |
| Serving and evaluation | the archived ember CPU server build `ember_publication_build_20261030`, greedy |

## 3. Measurement

Five quantities are kept apart. None is a proxy for another, and fluency is none of them.

| Quantity | Instrument | Scale |
|---|---|---|
| Format compliance | Exact-match checks in tinychat-eval and the screening instruments; the A5 copy probe | pass/fail |
| Semantic correctness | tinychat-eval `semantic_content` flags (mention and negation, not correctness) and the semantic Gemini rubric | flags; 0-4 |
| Coherence | Gemini judge, `integrations.py` RUBRIC | 0-4 |
| Relevance | Gemini judge, same rubric | 0-4 |
| Salience | **No established rubric yet** (see below) | none |

**The established judge rubric** (tinychat-eval `integrations.py`, file `3be76306…`; RUBRIC text `be533789…`;
SEMANTIC_RUBRIC text `8eaf649a…`), quoted exactly:

> Use scores 0-4 with quoted evidence: 0=absent/broken, 1=severely deficient, 2=mixed, 3=adequate, 4=strong. Judge
> independently: fluency: grammatical, readable language (not factual correctness). coherence: internally
> consistent and compatible with preceding conversation. relevance: answers the actual last question, rather than
> merely sounding plausible.

- **Configuration:** model `gemini-3.1-pro-preview` (the v0.0.2 gates); temperature 0; JSON schema output with
  evidence of at most 500 characters retained; at most 20 calls per run.
- **Conversational evidence:** the judge sees the full message history as well as the final reply. A per-turn mode
  scores every turn of the five open-conversation cases.
- **Run receipts:** every run is reserved exclusively, so a second write to the same file is refused.

**Two discrepancies with the programme brief, recorded before any new score series exists:**
1. **The brief describes a 1-4 scale; the established rubric is 0-4.** Every judged score in this programme uses
   0-4 with the anchors above. No 1-4 series is created.
2. **The brief names salience; the established rubric has no salience attribute.** It scores fluency, coherence
   and relevance. Salience stays unmeasured until a rubric is written for it.

   A salience rubric is a new instrument. It needs its own version and hash, and calibration against human
   ratings before use. It is never back-filled into the existing series.

   Proposed definition, not yet adopted: salience asks whether the reply takes up the information the conversation
   made important for this turn, for example a fact the user gave or a correction they made. A plausible reply that
   ignores it does not.

**Calibration.** `calibration/human_final_reply_ratings_v002_sfull.json` in tinychat-eval holds Jonathan's blind
ratings, made on 3 Nov 2026, of the S-full baseline's final replies on the five judged cases:

| Case | Fluency | Coherence | Relevance |
|---|---:|---:|---:|
| open_hobby | 3 | 2 | 0 |
| name_now | 2 | 1 | 0 |
| pet_delayed | 4 | 2 | 0 |
| meeting_corrected | 3 | 2 | 0 |

`open_friend` is not yet rated. These ratings are the calibration set. Before any judge-based claim:
- the Gemini judgments of the identical transcripts are compared with them;
- disagreements of 2 or more points are listed;
- the same transcripts are judged twice, to measure the judge's instability.

The judge is an instrument, not ground truth. In paired comparisons, candidate identities are hidden from the judge
and the order of presentation is counterbalanced.

**Judge run J1 (4 Nov 2026), fixed before any call.** Jonathan corrected the brief the same day: salience is not
part of the rubric, and relevance is the programme's priority.
- **Calibration:** S-full's calibration transcript (`f7d4f4d3…`), judged twice, per-turn, on the five gate cases
  (13 turns). The final-turn scores are compared with Jonathan's four ratings. Reported: each disagreement of 2 or
  more points, and test-retest agreement between the two runs.
- **Screens:** the release-suite dev reports of S0, S2 and S4 (seeds 1-3), S1 and S3 (seed 1, wave 1) and S5
  (seeds 1-3, as they are evaluated), each judged once, per-turn.
- **Configuration:** `integrations.py judge --per-turn --model gemini-3.1-pro-preview --max-calls 20 --limit 20`,
  with the root workspace `.env` as the designated env file.
- **Primary reading:** relevance per turn. Per model: the mean, and the fraction of turns at 3 or more (the v0.0.2
  gate's statistic, threshold 0.7). Each screen is paired with S0 at the same seed, case by case and turn by turn.
- **Also reported:** coherence the same way. Fluency is reported but is never a target.
- **Not claimed:** no screen is called better on the judge unless its relevance difference from S0 has the same
  sign at every seed. The calibration disagreements are stated beside every judged number.
- **Receipts:** one exclusive JSON per run in `C:/ml_artifacts/chat_screens_claude/judge/`.

**Task instruments (frozen):**

| Instrument | SHA-256 | Items | Use |
|---|---|---|---|
| tinychat-eval `suite.json`, `suite_release_v002.json` (dev splits) | in tinychat-eval | 11 dev cases with exact checks | regression and dev passes |
| `interrogative_heldout_v001.json` | `59382d03…` | 96 (68 non-constant, 28 yes/no) | held-out interrogative wording; the wave-3 and A4 bars |
| `interrogative_transfer_v002.json` | `d13952a0…` | 116 (92 constrained, 24 foils) | position, polarity and order counterbalanced; exploratory |
| `interrogative_trainside_v003.json` | `5e0862ec…` | 96, each v001's twin with training-side wording | in-distribution control (A6) |
| RCA probe (`rca/rca_probe.py`) | in the root workspace | 30 recall + relevance pairs | "Your" against "My", value mentions |
| A3 margin probe (`a3_probe.py`) | `57f90465…` | v001's 50 bare items | content against format margins |
| A5 format probe (`format_probe.py`) | in the root workspace (`e864a464`) | 64 copy + 30 × 4 ladder | format without content; first-token competition |

The protocol, scorers and probes are in the root workspace at `AI_ML/chat_screens_claude/` (`PROTOCOL.md`, versions
and amendments A1-A5). The session driver and receipts are here, in `experiments/chat_screens_2026-11-04/`.

## 4. Experiments

Each entry gives the hypothesis, what would falsify it, the result, and where the evidence is. Seeds are the
trainer's `train.seed`. Training at this recipe is bit-deterministic across sessions: seed 1's S0 and S2 reproduced
exactly between waves.

### Wave 1 (4 Nov, done). Three data changes against S0
- **Hypothesis:** an answer-first (S1), second-person (S2) or de-templated (S3) data change gives exact dev passes
  that S0 lacks.
- **Falsifier:** no exact dev gain at seed 1.
- **Result:** 0 of 11 dev exact passes for every model, so no screen is promising.
  S2 moved the RCA target: it prefers "Your …" in 21 of 30 recall items against S0's 13, and its replies mention
  the value in 25 of 30 against S0's 15.
- **Evidence:** `WAVE1_RESULTS.md`; receipts `w1-20261104`.

### Wave 2 (4 Nov, done). S0, S2 and S4 (bare-answer rows) at three seeds
- **Result:** 0 of 11 dev exact passes everywhere. S2's second-person effect replicates at all three seeds. S4
  converts no format at any seed.
- **Outcome:** the futility rule paused the lane until Jonathan set the next design.
- **Evidence:** `WAVE2_RESULTS.md`, `wave2_score.*`; receipts `w2-s1/s2/s3-20261104`.

### Wave 3 (4 Nov, scoring). S5: interrogative coverage
- **Hypothesis (Jonathan's):** zero exact passes come from missing training coverage of interrogative forms.
- **Bar (A2 r1):** S5 beats S2 by at least 10 of v001's 68 non-constant items at 2 of 3 seeds, with the RCA guard
  held.
- **Seed 1:** 1 of 68 (S2: 0); 0 of 28 yes/no items.
  - Recall values are mentioned in 32 of 60 episodes (S2: 30), always inside a sentence.
  - "No" is mentioned in 34 of 36 unknown-fact episodes (S2: 0), never bare.
  - Corrections (2 of 40) and arithmetic (0 of 36) fail on content.
  - The exact-pass bar is at the floor, mostly through format.
- **Seeds 2 and 3:** rerun after an operational failure. Seed 3 exited 0. Seed 2's model is a hash-verified rescue
  copy (root `PROTOCOL.md` §12). Evaluation is under way.
- **Exposure audit (A3):** grid rows carry only a few per cent of supervised tokens, and most are drawn less than
  once per run. S5 barely trained its target.

### A3 (CPU diagnostics, under way)
- **Done:** the exposure audit, the content and format margin probe, and transfer set v002.
- **S2 seed 1's margins:** `format_A` median −10.9 log-probability units, positive on 3 of 50 items. Bare answers
  are strongly disfavoured even when instructed.
- **Paired S5 against S2 contrast:** `a3_compare.py`, with its test fixed before S5's probe existed.

### A4 (designed, reviewed, conditional). Template diversity at matched exposure
- **Arms:** S6, S7 and S8 use 10, 30 and 57 training wordings. Supervised text and row draws are identical across
  arms.
- **Bar:** D_high beats D_low by at least 10 of 68 at both seeds. A floor rule calls a seed uninformative when both
  arms pass at most 2.
- **Launch condition:** S5 shows a format shift against S2 on the A3 probe.
- **Cost:** 65 L4-minutes worst case per seed; seed 2 waits for next week's budget.

### A5 (CPU, running). Format with the content problem removed
- **Question:** can the model follow a bare-output instruction when there is nothing to retrieve, and where does
  that break as retrieval is added?
- **Items:** 64 copy items; a 30-item ladder (copy, same-turn fact, earlier-turn fact, constraint first);
  first-token competition against twelve openers.
- **Decision rules,** fixed before results (root `PROTOCOL.md` §14):
  1. Copy fails: train format alone.
  2. Copy works and retrieval fails: progressive curriculum.
  3. Constraint first gains at least 5 of 30: placement goes into the design.
  4. "Your" or "You" wins on recall but not on copy: interference from S2's convention.
- **Result:** section 5.

### A6-A10 (4-5 Nov, done). Results against their registered bars

Full records are in root `AI_ML/chat_screens_claude/PROTOCOL.md`. Each experiment ran at seed 1, or at seeds 1-2
where stated, and is exploratory.
- **A6 (§15):** S5 is 0 of 68 on v003, the training-side twin, at all three seeds. The policy was not learned even
  where it was trained.
- **A4 (§13, §17): uninformative (floor) at both seeds.** S8 and S6 each pass 1 of 68. The format margin moves with
  dose: S6 − S2 +6.57, S8 − S2 +7.41, and diversity adds only +0.84.
- **A7 (§16-17): FAILS.** The contrastive arm S9 does not beat S10, and at seed 2 its format margin is significantly
  below the control's.
- **A8 (§17): FAILS.** S11 copies 17 and 13 of 64, against a bar of 32.
- **A9, dose (§18): primary FAILS** (S12 7 and S13 6 of 68, against 10). **Co-primary PASSES:** the median bare-format
  margin is +2.89 and +1.06, against −11.95 for the S2-data control. Guards hold.
  - Format data, not steps, moves bare-answer learning: the control's probe loss stays at about 5.2 for 1,024
    steps.
  - The remaining failures are mostly wrong values.
- **A10, scenario diversity (§19): primary FAILS.** On the frozen misfire probe, S15 has 58 correct and S16 64,
  with p 0.38, against a bar of +21.
  - S15 halves the misfires in the scenario types it trains, 12 against 24, but misfires more where neither arm
    trains, 24 against 15.
  - Guards hold, and paperkiln/ember inference parity passes, single- and multi-turn.
  - At this dose, diversity buys scenario-specific conventions, not a transferable operation rule.
  - That dose was about 2% of sampled supervised tokens. Every later intervention is sized by sampled tokens, not rows.
- **A11, human conversation at 30% of tokens (§20, verdict §20.4): primary FAILS.** S25 67 against S24 68 of 111
  decided, p 1, against a bar of +21. S25 ties S26, the matched grid half.
  - S27, human data alone, is strongest in scenario types the grid never saw (16 / 9 against 6 / 29) and weakest on
    instruction families e and f (16 of 40 against 24). It follows no format (v001 0).
  - S25 fails the relevance guard (−0.54) and more than doubles persona invention (45.4% against 19.7%).
- **A12, instruction diversity at 68, 272 and 1,088 phrasings (§21, verdict §21.2): primary FAILS by two items.** S23
  against S21 is 42 to 26 of 90, +16, p 0.0025, against a bar of +18.
  - Monotone on held-out items: 28, 35, 47 of 120. Constraint satisfaction 61, 73, 86; operation choice 65, 66, 71.
  - The registered reading is wording transfer, not composition (held-out gain 0.158, reference gain 0.225).
  - Persona invention falls 7.9%, 4.6%, 0.7%.
- **A14, hybrids (§22, verdict §22.3): both FAIL on precision alone.** S28 and S29 have 18 and 17 of 40 against a limit
  of 20, and pass naturalness (22, 16) and the ambiguity guard (60, 62).
  - They have the highest misfire-probe totals so far, 78 and 79 correct of 140, mostly from family b (16 and 17 of 20).
  - Identity: 94.7% and 98.0% of identity turns answer as an assistant on `persona_v002`, on wordings left out of
    training, with invention at 2.6% and 1.3%.
- **A15, hybrid plus instruction diversity (§23-23.1): running.** S30 and S31 on a five-condition bar that a constant-form
  model cannot pass.
- **Misfire probe baselines** (correct of 140): S12 74, S14 57, S2 58, S0 54. The best degenerate responder gets
  55.

## 4b. Judge run J1: results (4 Nov 2026)

Receipts are in `C:/ml_artifacts/chat_screens_claude/judge/`. The item table is `analysis/judge_j1_items.json` (root
workspace), from `judge_items.py` and `judge_compare.py`.

**Calibration against Jonathan's ratings (S-full, final turns).**
- **Test-retest:** relevance was identical on 11 of 13 turns, and every turn was within 1.
- **Leniency:** 4 of 12 scores differ from Jonathan's by 2 or more, all with Gemini higher.
- **Pattern:** Gemini gives relevance 1-2 to on-topic but wrong answers, which Jonathan scores 0. Examples:
  - "Its name is Ted." (the cat is Basil): 2.
  - "Your meeting is on Tuesday." after a correction to Thursday: 2.

**Mean relevance per model** (0-4, 13 turns; fraction of turns at 3 or more):

| Seed | S0 | S2 | S4 | Seed 1 only |
|---|---|---|---|---|
| 1 | 1.92 (6/13) | 1.54 | 1.69 | S1 1.31, S3 1.31, S5 1.69 |
| 2 | 1.31 | 1.85 | 2.08 | |
| 3 | 1.31 | 1.23 | 1.85 | |

- **Registered verdict:** no screen beats S0, since none has the same sign at every seed. Every model is far below
  the v0.0.2 gate (70% of turns at 3 or more), under a judge already lenient by 1-2 points on wrong answers.
- **S4:** its paired mean differences against S0 are −0.23, +0.77 and +0.54. That is an exploratory directional
  signal only.
- **Noise:** S0's own seed spread (1.31 to 1.92) is as large as any screen's effect.

**Item level.** The means hide a consistent trade, not noise. Across 7 runs containing S2's rows (S2, S4 and S5),
the same items move in the same direction at every seed:
- `pet_delayed` turn 0 goes from 1 to 4. "My cat is named Basil." is now answered "Thanks, so your cat is named
  Basil.", where S0 echoed it.
- `pet_delayed` turn 2 goes from 3 to 0 or 1. "Today is quite sunny." is now answered "Thanks, so you live in a
  small town." (S5: "…you live in the park.").
- `name_now` turn 0: "My name is Nora." gives "Thanks, so you are a native English language."
- `open_hobby` turn 1 goes from 4 to 1: "You could try a small amount of bread."

The second-person data taught an *unconditional* acknowledgement template, fired by the surface form "user
statement". Other findings:
- **Fail in every model:** `open_friend` turns 0 and 1, `name_now` turn 0, and the final exact turns.
- **A separate limit:** in `open_friend`'s "explain with an example", every model degenerates into repetition loops
  ("a good friend, a friend, a friend…"). No format intervention addresses this.
- **Gemini against the exact checks:** no disagreement. No reply passed an exact check, and Gemini gave no failed
  final turn 3 or more.

**Corpus link** (counted directly; the full audit is in section 5b). S2's contrast class, non-personal statements
that should *not* be restated in the second person, has exactly **2 distinct pairs**:
- "It is quite windy this evening." → "Sounds blustery." (120 copies);
- "The train was late again today." → "That is annoying." (109 copies).

Against them are 71 distinct personal facts, acknowledged with 6 openers. Two memorised pairs cannot define a
category, so the condition was never taught; only the template and two lookups were.

**Calibration set calib_v1** (`analysis/calibration_v1/`, `15b19e8f`, frozen): 12 real replies across the five
classes (fluent but irrelevant; on-topic but failing; correct but misformatted; correct and formatted; incoherent).
It has a blind form for Jonathan and a separate key. Gemini judges it with the unchanged rubric only after the
human ratings exist.

**calib_v1 results (4 Nov).** Jonathan's blind ratings were recorded (`f7c8baa1`) before Gemini judged the set
(judge version `J1-calib-v1`: same rubric, schema, model and settings; `calib_judge.py`).

| | Exact agreement | Within 1 | Mean (Gemini − human) |
|---|---|---|---|
| Fluency | 11/12 | 12/12 | +0.08 |
| Coherence | 8/12 | 11/12 | −0.08 |
| Relevance | 8/12 | 10/12 | +0.42 |

- **Gemini is lenient on on-topic content that is wrong or nonsensical:**
  - "You could try baking a small piece of bread with a twist of a stick.": human 1, Gemini 4.
  - "Your meeting is on Tuesday." after the correction to Thursday: human 0, Gemini 2.
- **It agrees with the human** on plainly off-topic replies and on correct, well-formatted ones.
- **On correct content in the wrong format it is slightly stricter:** human 3, Gemini 2 or 3.
- **The human's relevance tracks semantic correctness,** with a one-point cost for a format violation.
- **Selection note:** item c1_07 was placed in class D (correct and appropriate) on the strength of Gemini's earlier
  4; the human rating shows that placement was judge-biased. Class labels for open-ended items need a human anchor.
- **Consequence for the relevance target:** judged relevance can rise on on-topic wrong answers without the model
  becoming more correct. Every judged relevance result is reported beside the semantic and exact checks. A gate
  claim needs agreement with human anchors on that candidate's own transcripts.

## 5. Results of the current diagnostic (A5)

Pending: filled in from `C:/ml_artifacts/chat_screens_claude/format_probe/{S0,S2,S5}-s1/summary.json` when the run
ends.

## 6. Depth × diversity (proposed, unexecuted, not authorised for training)

**Question.** Does more depth, at approximately constant parameters, improve generalisation of conversational
operations to unfamiliar scenario and instruction combinations? The estimand is the interaction on a held-out
transfer measure R:

Δ = (R_16,diverse − R_16,narrow) − (R_8,diverse − R_8,narrow).

A depth effect alone would not show how depth helps, and a positive Δ would not show that operations occupy
separate layers. Proposed by Jonathan and GPT on 5 Nov 2026.

**Assessment (Claude Opus 5.5's read, 5 Nov 2026).** Defer, with a cheaper experiment first.
- **For:** the prior is reasonable at this scale.
  - Petty et al. (NAACL 2024) matched parameters by shrinking the FFN, at sizes including 41M. Deeper transformers
    generalised more compositionally after fine-tuning, with gains that shrink quickly, and not only through better
    language modelling.
  - MobileLLM (Liu et al., ICML 2024) found deep-and-thin better than wide-and-shallow below a billion parameters.
- **Against:** the failures measured so far are explained by data at fixed architecture.
  - A9 moved the median bare-format margin from −11.95 (control) to +2.89 with no architectural change.
  - On the A5 ladder, constraint-first items stay at 0 of 30 in every model. Training data that never states the
    constraint first explains that as well as depth does. A10's counterbalanced data tests it.
  - Most remaining v001 failures are wrong values, a recall limit even on training-side wording: v003 13 of 68 for
    S12.
  - Cost is dominated by pretraining from scratch, and the interaction needs several independently pretrained
    bases per arm.
  - **A depth sweep already exists for another outcome.** The October from-scratch programme's E1 pretrained
    depths 2 to 32 at about 20M parameters, and the 4 to 32 cells at 60M, on human-written data, at three seeds
    each (microtorch `experiments/tiny_chat_2026-10/`, `analysis_2026-10-28.json`, `TACTICS_2026-10-26.md`). Depth
    did not change Carry-5's context carry-over cost at 20M; model size was the one lever with a signal. Carry-5
    measures carry-over, not operation selection, so E1 does not answer this question. It is evidence that depth at
    fixed size has been a weak lever in this codebase, and its per-seed spread is the nearest available estimate of
    between-base variance.
- **The binding constraint: base-to-base variance is the error term for Δ.** Depth must be compared across
  independently pretrained replicates, at least three per architecture. SFT seeds on one base per architecture
  confound depth with that base's idiosyncrasies. No estimate of between-base variance exists yet for these
  instruments.

**Architectures.**

| Arm | Layers | d_model | FFN | Heads × dim | In the layers | Unique total (tied embedding) | Forward FLOPs per token at T = 1,024 (approx.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 8 | 768 | 2,240 | 12 × 64 | 60,174,336 | 72,761,088 | about 171M |
| Deeper | 16 | 560 | 1,632 | 10 × 56 | 63,956,480 | 73,134,320 (+0.5%) | about 183M (+7%) |

- **Three matchings, which are not equivalent.**
  - **Parameter matching:** the pair is matched on unique total parameters. It is not matched on parameters inside
    the layers (+6.3% for the deeper arm), because its tied embedding is narrower. A registered design picks one and
    reports both; layer parameters (non-embedding N) are the usual scaling quantity. A 16-layer variant matched on
    layer parameters would need a narrower FFN.
  - **Token matching:** the same pretraining corpus, tokenizer, data order and token count, then identical SFT
    mixtures and supervised-token counts.
  - **FLOP matching:** at matched tokens the deeper arm costs about 7% more forward compute: 2N, plus the attention term
    4·L·T·d, plus the tied output projection 2·V·d, which is smaller for the narrower arm. On an L4 its wall time is likely higher still, from sequential depth and narrower matrix products.
    Report FLOPs and wall time per arm. A FLOP-matched reading needs the baseline trained on proportionally more
    tokens, as a separate arm.
- **Optimiser:** the best learning rate depends on shape without μP. Both arms get a short learning-rate sweep, or
  a registered argument that the shared rate is near-optimal for both, with sensitivity reported.

**Alternative comparisons.**
- **A depth sweep:** at least three depths at matched layer parameters, for example 6, 8, 12 and 16. It gives a
  dose-response curve instead of one pair whose width and head dimension also change.
- **Weight-shared (looped) depth:** 8 layers applied twice is 16 sequential stages at the baseline's parameters. It
  isolates "more sequential computation" from "parameters allocated differently". MobileLLM's layer-sharing
  variant is the closest precedent. It needs trainer support, which is not built and not to be built merely for
  this.

**Instruments (frozen, pre-registered).**
- A compositional held-out split. Scenario × instruction cells form a grid, the training data covers a registered
  subset (a Latin-square design), and R is measured on the complement. Unseen wording alone does not measure
  composition.
- The misfire probe v001.
- The v003 against v001 gap: trained against held-out wording.
- The A5 ladder, including constraint-first items.
- J1's 0-4 fluency, coherence and relevance, with human calibration. Salience needs a new, versioned instrument;
  the J1 rubric has none.
- Factual recall, separated from operation choice, and the knowledge smoke.
- Not as a sole endpoint: training loss or any single aggregate score.

**First, cheaper experiments, in order.**
1. **A10 itself (S15 against S16, §4).** Score S15 separately on scenario and instruction combinations it trained
   on and on held-out ones. That gives the composition gap at 8 layers.
2. **Score E1's existing bases on operation selection, before any new pretraining.** E1's 20M and 60M bases at
   each depth and seed already exist. Give each the same small SFT (S15's mixture) and score it on the misfire
   probe and the compositional split. That is a depth sweep with three independent bases per depth, for SFT cost
   only. It tests whether depth changes operation selection at all, and it estimates between-base variance for the
   73M power calculation. A new small proxy sweep is needed only if E1's bases cannot take this SFT (tokenizer,
   template or context differences, checked first).

**Trigger.** All three of the following.
- A10 shows S15 learning its trained combinations (low in-distribution misfire), while a held-out-combination gap
  persists after one further diversity dose at 8 layers.
- The E1-base sweep, or the proxy sweep, shows Δ > 0 across seeds.
- The cost of one 73M base is verified (§6 Cost, ledger) and fits an authorised envelope.

**Retirement.**
- **Unnecessary:** S15 closes the held-out-combination gap, so diversity at 8 layers suffices.
- **Premature:** the remaining failures are content recall or knowledge. The stronger-base question then comes
  first.
- **Recipe problem first:** S15 and S16 both fail in-distribution.
- **Shelved:** the proxy shows no depth effect across seeds.

**Cost.**
- Extrapolating the resumed pretraining receipt (steps 7,675 to 10,000 in 3.08 L4-hours), one base is about
  13 L4-hours. This estimate is unverified (§9).
- The deeper arm costs about 7% more FLOPs, and likely more in wall time.
- Three bases per architecture is roughly 80-100 L4-hours: a multi-day campaign, beyond any screening wave. The SFT
  arms on top are cheap (about 15 minutes each).
- The S-full campaign's closed pod total was 18.97 hours and US$11.19. It includes phases other than pretraining
  (Codex's check, 4 Nov), so the cost of one base remains unverified.

**Already settled from the original step 1.**
- A6: S5 failed even its trained wording.
- A9: a larger dose of format data fixed the format margin but not value recall.
- The original trigger ("an in-distribution failure that a larger dose of the same data does not fix") has therefore
  fired for recall only. Recall is a base-capability question as much as a depth question.

**Training-procedure factors, in order:** uniform mixing, the current default. Then a sequential curriculum
(deferred from A10 as S17, to be compared with a matched uniform mixture at identical exposure). Then preference
optimisation, only if SFT fails to establish format transfer. No tokenizer change, speculative architecture or
reinforcement learning without evidence that the simpler explanations fail.

## 6b. Typed conversational state registers (proposed, unexecuted, not authorised for training)

**Origin.** Jonathan's aim, 5 Nov 2026, refined with GPT the same day: supervise the model's existing k = 4 per-turn
state tokens with explicit, typed dialogue state. Recall, role-tracking and identity should then come from trained
state rather than template matching.
- **ι (identity):** the assistant and the user. The prior-art lessons recommend splitting it, or binding role to
  filler.
- **φ (facts):** values, corrections and absences.
- **τ (mode):** role-play state and speaker attribution.
- **κ (standing instructions).**

The registers hold persistent state; the current instruction stays in the token stream.

**Central question.** Does explicit, semantically typed state improve the model's use of persistent conversational
information across unfamiliar situations? The experiment must also separate a state-representation failure from a
response-policy failure.

**Endpoints, kept separate.**
- **Decodability:** a low-capacity probe recovers the state.
- **Necessity:** disrupting a register impairs generated behaviour, against matched equal-norm perturbations.
- **Selectivity:** disrupting a register disproportionately impairs its own capability (a capability × register damage
  matrix).

A probe result alone is never called functional memory.

**Factors.**
- **Typing:** a stable against a variable role-to-slot assignment, with identical heads, targets, loss weight and
  exposure. The variable assignment must not be inferable by the model.
- **Masking regime:** causal (raw turns visible, so a bypass exists) against turn_state with eviction (the registers
  are the only memory of finished turns). These are different memory tasks. The primary contrast fixes the mask, and
  a result under eviction is not evidence about ordinary causal conversation.

**Smallest controlled matrix (one mask, identical exposure).**
- A: no auxiliary loss.
- A′: the same heads and weight with shuffled, uninformative targets.
- B: variable assignment.
- C: stable assignment.

A′ − A isolates supervision as such, B − A′ isolates informative state, and C − B isolates typing.

**Known facts** (`AI_ML/chat_screens_claude/analysis/state_token_feasibility.md`, 5 Nov):
- `state_embd` is 4 distinct trained vectors, and the slot index is the id minus V.
- State-position outputs are unused in training and inference, so the heads are training-only, with no change to
  export, the GGUF or Ember.
- Open values are predicted as a first token through the tied embedding, plus "absent".

**Prior art** (`analysis/state_token_prior_art.md`): the plain idea is ADJACENT (SituationSupervision, Findings of ACL
2023; Compressed Context Memory, ICLR 2024; the dialogue-state-tracking line). The typed combination is OPEN. Two 2026
preprints cited there remain unverified.

**Stages.**
- **Stage 0:** a CPU probing diagnostic of existing checkpoints, decodability only, with numpy-to-trainer logit parity
  required.
- **Stage 1:** a data-only pilot, if justified.
- **Stage 2:** the matrix above.

**Stage 0 result** (`analysis/state_probe_results.md`, 5 Nov; numpy-to-trainer parity at most 9.5e-6). Closed state is
decodable from the slots at 94-98%, with no slot specialisation. Seen values reach 69-85%, and held-out values 14-42%
against chance 12.5%, so the value code does not generalise. Latest-after-correction (49-58%) and ownership (54-66%) sit
near chance. Reply attention to the slots is 4-11% against a uniform 12%.

**Model scale (Jonathan, 6 Nov 2026).** If the state tokens perform, a separate paper needs them tested at several model
scales with more rigour. E1's bases at 20M, 60M and 150M are on disk, at several seeds, and carry no `state_embd`. They
save pretraining only: each scale still needs a GPU SFT run with state tokens added before Stage 0 can be repeated there.

Each stage needs its own registration, independent review and Jonathan's authorisation. The specification is
`analysis/TYPED_STATE_SPEC.md`. A trainer change stays on `claude/chat-screens` and is announced in the
LOCAL_LLM_STACK inbox before it lands. Paperkiln master and any `src/llama.cpp` change are Codex's. Kept separate
from §6 (depth).

## 7. Capability frontier (to be filled as configurations are evaluated)

One row per evaluated configuration. No single weighted score is used for selection. A Pareto-efficient row is one
that cannot gain on one target without losing another or costing more. The best configuration tested is not a
demonstrated optimum.

| Configuration | Params (total / non-emb.) | PT tokens | SFT rows × updates | Template diversity | L4 time | v001 exact (68) | Format (A5 copy) | Coherence | Relevance | Salience | Seeds | Regressions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-full baseline | 72.76M / 60.17M | Codex record | SFT1500 | n/a | n/a | not run | pending | human: 1-2 | human: 0 | unmeasured | 1 | n/a |
| S2 (seed 1) | same | same | arm A, 256 × 16 | n/a | about 10 min | 0 | pending | not judged | not judged | unmeasured | 3 | none (wave 2) |
| S5 (seed 1) | same | same | arm A, 256 × 16 | train side, 1 pool | about 10 min | 1 | pending | not judged | not judged | unmeasured | 3 (scoring) | scoring |

## 8. Compute ledger (Colab L4, from session creation to certified cleanup)

| Run | L4 minutes | Outcome |
|---|---:|---|
| w1-20261104 (S0-S3, seed 1) | 41 | exit 0 |
| w2-s1/s2/s3 (S0, S2, S4) | 120 | three exit 0 |
| w3-s1 (S2, S5) | 26.5 | exit 0 |
| w3-s2, w3-s3 (S5) | about 46 | failed before training (CLI stall); cancelled, certified stopped |
| w3b-s2, w3b-s3 (S5) | 51.8 | exit 2 (rescued) and exit 0 |
| w5-s1 (S8, S9, S10, S11, seed 1) | 51.1 | exit 0 |
| w5-s2 (S8, S9, S10, S11, seed 2) | 59.0 | exit 0 |
| w6-s1 (S6, seed 1) | 20.0 | exit 0 |
| w6-s2 (S6, seed 2) | 20.7 | exit 2 before training (CLI stalls); certified stopped |
| w6b-s2 (S6, seed 2) | 0 | stopped at preflight (git fetch); no session |
| w6c-s2 (S6, seed 2) | 23.9 | exit 0 |
| a9-s1 (S12, S13, S14; one session each) | 128.9 | three exit 0 |
| a10-s1 (S15, S16; one session each) | 67.0 | two exit 0 |
| a11b-s1 (S24, S25, S26, S27-r2; one session each) | 193.5 | four exit 0; S27's first session was refused at assignment (0 L4) |
| a12-s1 (S21, S22, S23; one session each) | 140.4 | three exit 0 |
| a14-s1 (S28, S29; one session each) | 105.3 | S29 exit 0; S28 exit 2 (log archive timeout; model and train check verified) |
| **4-5 Nov** | **about 1,095 (18.3 h)** | the week's 6 h, increased by Jonathan's redistribution of budget to this week (4 Nov); A15's two sessions, from 6 Nov, are not yet counted |

CPU evaluation runs under the shared CPU budget (`scripts/cpu_budget.py`, session `claude`, at most 4 slots).

## 9. Retractions and superseded readings

- **Model size.** The first version of this registry (`baff7c47`) gave 85.3M parameters and called "70M"
  an undercount. That reading is superseded: 85.3M counted the GGUF's duplicated tied embedding. The model has
  72,761,088 unique parameters (section 2, Codex's correction, 4 Nov). The same version's depth arm, 16 × 576
  (77.1M unique), was not parameter-matched; 16 × 560 replaces it.
- **Wave-3 incident (4 Nov):** the reading "a fresh VM may not answer its first exec" is superseded. The review
  found that the VM answered and the local Colab CLI stalled (`reviews/launch_review_6_w3_rerun.md`).
- **A4 figures:** the earlier figures (grid about 14% of supervised tokens; 487, 961 and 2,049 masked templates)
  are superseded. The grid is 12.0%, and the arms have 10, 30 and 57 wordings (`reviews/launch_review_5_wave4.md`).

## 10. Programme end goal (Jonathan, 4 Nov 2026)

1. **Clear the bare-answer and format bottleneck.** Jonathan identified the training corpus as its cause; the SFT
   corpus audit and A5/A6 confirmed it.
2. **Train full runs, score them above all on relevance** (the 0-4 Gemini judge with human anchors, beside the
   semantic and exact checks), and iterate.
3. **Demonstrate utility, not just function.** A working small chatbot is not the end point. The model must do
   something useful; the first candidate is connecting it to a basic retrieval-augmented setup (RAG) over
   documents.
4. **Benchmark it** against comparable small models at scale, reporting where it stands. The aim is an honest
   comparison, not a claim to beat larger systems.

### Design principle: early readout (Jonathan, 4 Nov 2026)

Every training design reports its decisive metrics at a pre-registered cut-off without waiting for training to
finish, wherever possible. GPU runs may be used whenever they are faster. The mechanisms, none needing a trainer
change:
1. **Probe rows at the start of the held-out tail.** The trainer's SFT evaluation (the first `eval_examples`
   held-out rows, every `eval_every` steps; held-out rows never receive gradient) then reports a format-learning
   curve in train.log. The probe items are distinct from the frozen instruments.
2. **The session driver streams train.log while polling,** and applies pre-registered rules: report at the cut-off,
   stop early if the curve is flat.
3. **For long runs, one mid-run checkpoint is pulled at a registered cut-off** and scored at home with the A3 and A5
   probes.

The cut-off and the rules are registered with the bar.

**Gates for the combined run** (Jonathan, 4 Nov: "eventually we will combine the best of both worlds"), all
required together:
- relevance (Gemini 0-4, read against the human anchors);
- format (v001, A5);
- utility (the RAG instrument, `AI_ML/tiny_chat_utility/`);
- knowledge (the benchmark suite, `AI_ML/tiny_chat_benchmarks/`), as a regression gate against S0 and October's
  models.

A run that wins on format but loses knowledge fails.

## 10b. Next authorised step and its stopping rule

Score A15 (S30 and S31) under `AI_ML/chat_screens_claude/PROTOCOL.md` §23.1 and apply its continuation rules. Any later
training run has its design, hypothesis and bar written here and in the root protocol, and independently reviewed,
before training.

**Stopping rule:** no GPU training without a registered design, review and budget. An intervention that does not
beat its control at its registered bar is recorded here as a negative and not retried with relaxed criteria.
