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

## 6. Capacity × diversity: the smallest discriminating design (proposed, not authorised)

The question is whether compositional generalisation, here held-out instruction wording combined with retrieval,
is limited by data coverage or by depth and capacity. A full factorial sweep is not the first step.

**Step 1, CPU only, no new training: an in-distribution control (amendment A6, running).** Evaluate S2 and S5 at
three seeds on instrument v003. Each v003 item is its v001 twin with training-side wordings; values, facts and
answers are unchanged. A4's arms get the same control, built with their own pools, if they are trained. Reading
rules are in root `PROTOCOL.md` §15.
- If the models pass the training-side items but not v001, they can represent and execute the policy, and the
  failure is transfer to new wording: a data-coverage limit, which A4 tests directly.
- If they also fail the training-side items, the policy is not learned even where it was trained. That points to
  dose, optimisation or capacity, and more template diversity alone will not help.

Step 1 decides whether depth is worth paying for.

**Step 2, only if step 1 shows an in-distribution failure that a larger dose of the same data does not fix: depth
at matched parameters.**

| Arm | Layers | d_model | FFN | Heads × dim | In the layers | Unique total (tied embedding) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8 | 768 | 2,240 | 12 × 64 | 60,174,336 | 72,761,088 |
| Deeper | 16 | 560 | 1,632 | 10 × 56 | 63,956,480 | 73,134,320 (+0.5%) |

The pair is matched on unique parameters. The deeper arm puts more of them in the layers, because its embedding is
narrower. Its total is computed by the same accounting that reproduces the baseline's native count exactly: layers,
tied embedding, final norm, and a state embedding of 4 × d_model.

- **Matched start:** a newly initialised deeper model is never compared with the extensively pretrained baseline.
  Both arms are pretrained from scratch on the same corpus, tokenizer, token count, optimiser and schedule. Then
  both get the identical SFT mixture (S2 with the A4 D_high grid) and the identical evaluation. At least two seeds.
- **Exposure accounting:** pretraining tokens, SFT tokens and FLOPs are reported per arm.
- **Cost:** the pretraining pair dominates and exceeds a two-L4-hour wave.
  - The S-full base campaign's closed pod total was 18.97 hours and US$11.19. That total includes phases other than
    pretraining, and excludes separately allocated storage (Codex's check, 4 Nov).
  - The final pretraining receipt's 3.08 hours cover only the resumed steps 7,675 to 10,000, not the whole of
    pretraining.
  - The exact cost of pretraining one base therefore remains unverified. Step 2 needs its own registered design, a cost estimate from that record and separate
  authorisation.
- **A larger-parameter control** comes only if the matched-depth comparison is null and step 1 still shows an
  in-distribution failure.

**Training-procedure factors, in order:** uniform mixing (the current default), then a sequential curriculum if A5
rule 2 fires, then preference optimisation only if SFT fails to establish format transfer. No tokenizer change,
speculative architecture or reinforcement learning without evidence that the simpler explanations fail.

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
| **Week of 4 Nov** | **about 285 (4.75 h of 6)** | |

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

## 10b. Next authorised step and its stopping rule

Finish A5 and apply its rules; score wave 3; run step 1 of section 6 on CPU. Then exactly one of:
- A4 seed 1, if its launch condition holds (65 L4-minutes); or
- a registered A5 follow-up (format-only arm or ladder curriculum). Its design, hypothesis and bar are written here
  and in the root protocol, and independently reviewed, before training, within the remaining weekly L4 budget.

**Stopping rule:** no GPU training without a registered design, review and budget. An intervention that does not
beat its control at its registered bar is recorded here as a negative and not retried with relaxed criteria.
