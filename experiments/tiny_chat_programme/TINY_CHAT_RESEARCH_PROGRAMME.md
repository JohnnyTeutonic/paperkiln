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
| Base, screens to A17 | `sfull_sft1500.safetensors`: the preserved S-full baseline (pretraining PT10000, then SFT1500), owned by Codex. A19 continues from S34 and S30, on the same lineage. A18 uses October's 20M bases, A20 uses pk_sft_v1 (12 × 768) and A21 to A27 use B1's winner A after SFT1500 (8 × 768, k = 4); each has its shape in `PROTOCOL.md` |
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

### A6-A20 (4-7 Nov). Results against their registered bars

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
  - S23 has the highest misfire-probe total of any arm, 84 correct and 33 misfires of 140, with nine items sharing
    target frames with its packet (§22.2).
- **A14, hybrids (§22, verdict §22.3): both FAIL on precision alone.** S28 and S29 have 18 and 17 of 40 against a limit
  of 20, and pass naturalness (22, 16) and the ambiguity guard (60, 62).
  - They have 78 and 79 correct of 140 on the misfire probe. Their lead over S24 is all in family b (16 and 17 of 20);
    outside b they have 62 each against S24's 67.
  - Identity behaves close to a constant reply. On `persona_v002`, 94.7% and 98.0% of plain identity turns answer as an
    assistant, but 1 and 0 of 48 role-play turns stay in character, and 7 and 5 of 12 user-name questions get the
    model's own identity (S24: 0, with 8 recalled).
- **A15, hybrid plus instruction diversity (§23-23.2): both FAIL, on precision and discrimination.** S30 and S31 have 18
  and 17 of 40 on precision against a limit of 20, and 0 and 2 twin pairs against a limit of 4. Naturalness and the
  ambiguity guard hold. By §23.1's rule no further phrasing or weighting variant is run.
- **A16, paired contrast data (§24-24.3): all three FAIL.** S32, S33 and S34 have 12, 6 and 16 of the 59 context pairs of
  `contrast_v001` against a bar of 17. S34 meets the McNemar leg (13 and 2, p 0.0074) and the strata leg and misses the count by
  one pair. Every guard holds and parity passes.
  - Rule 3 fires: S32 and S34 reach 19 and 22 of 100 and fail the context legs, so they are recorded as a copy reflex, with no
    credit.
  - Paired data moved identity and role, and neither recall nor correction: S34's 16 are I 11, R 4, U 1, A 0, C 0.
    `contrast_v002` takes S34 to 8 of 59, because declines had scored as correct.
  - The passage test with the right passage is 0 of 11 for all three.
- **A17, does S30 read a correct note (§25-25.3, CPU, no L4): B FAILS on reach, spread and content, and a read is shown.** Reach is
  +2 of 82, p 0.815, against +15; content 8 against 10; collateral −2 holds. Continuation rule 3 applies.
  - Ranked read 62 of 114 frames, none the opposite way. Produced read 21 of 114. Selection about none, 0.038.
  - The read is availability in context: condition D, the only one with both values in the prompt, has a mean margin of −0.59
    nats, positive on 52 of 114. The role flag is flat.
  - S30 was never trained on notes, so this does not rule out a scratchpad. A read-only training arm is the next test, on a
    separate registration.
- **A19, mixed rows against a sham (§27-27.1), seed 2: not a pass. Rule 3 fires, because a guard failed.** S51 (M1) against S52
  (C0), both from S34, on `chatflow_v001`'s 108 told-value items.
  - **Primary fails** on reach (+7, 37 against 30, bar +13), the frame sign test (0.174) and template (3.81 and −8.33 points per
    item, bar 7). Role, spread, discrimination, selection, parent and preference hold.
  - **The guard:** the ended twin on `plain_name` is 6 of 8 against a floor of 7. It fails by one item, and ROLE_LEAK totals are
    equal at 4.
  - **The claim tier is not reached.** On the 51 items the cue table fails, M1 has 13, C0 12 and S34 12: a gain of 1 against a bar
    of 7.
  - **Rows were learned.** On M1's held-out mixed rows, fact in a role is 0.903 against C0's 0.483, like in a role 0.989 against
    0.000, name in a role 0.956 against 0.433.
  - **Reported:** job gains +11 of the net +7, on items asking a job 17 of 42 against 7. Corrections fall from 7 to 3 of 16. J1
    relevance 2.00 against 2.08. Pairs split 6 to 0, p 0.031.
  - **Next by rule:** no seed 3 until the cause is read, and M2 waits. Exploratory, one seed, one checkpoint, on S-full's base: a
    failed bar here is a fact about this dose on this checkpoint. Whether mixed rows work on a stronger base stays open.
- **A18, depth for width at matched parameters (§26-26.3), ten bases at about 20M: unresolved, rule 8.** S35 to S44 in four
  sessions, each exit 0 with cleanup certified. The floor gate read clear, mean y59 9.667 on the first three arms. Score files:
  `analysis/a18_score.json`, `analysis/a18_gate.json`.
  - **Primary, on five cell means of `contrast_v001` y59:** slope 0.7 pairs per doubling, 95% [-1.58, 2.98], t 0.977 on 3 df, p
    0.401. T1, T2 and T3 do not hold, with leave-end slopes of -0.45 without L2 and 1.5 without L32. T4 holds, b37 0.7. The
    interval is not inside (−2, +2), so no null is excluded.
  - **Cells, L2, L4, L8, L16, L32:** mean y59 8.0, 13.5, 12.0, 13.5, 11.5; mean y37 1.5, 3.0, 4.5, 7.0, 3.0.
  - **Reported:** pure error s 1.97, 95% [1.22, 4.84]; lack of fit F(3,5) 2.63; permutation p 0.262; interior contrast t 2.55, p
    0.051, which does not hold; cluster bootstrap on b [-0.45, 1.85].
  - **Guards hold.** The rate flag does not fire, mean R(2,048) 0.539 at L32 against 0.419 over L2 and L8, so S49 and S50 don't open.
  - **Next by rule:** recorded with the curve and s. No extension exists at 20M inside this registration.
  - **The verdict's reading (Claude Opus 5.5, 7 Nov 2026, exploratory, two bases a cell):** this design neither finds a depth trend
    at 20M on this recipe nor excludes one of 2 pairs a doubling. It speaks to raw 20M bases without SFT1500, does not rule out
    depth at 73M, and reads as conditional on those bases.
- **A20, S34's and S30's recipes on pk_sft_v1 (§28-28.1), one seed: `guard`, rule 6. No arm moved; S54 reads `guard` and S55
  `flat`.** Runs `a20-s54-20261107` and `a20-s55-20261107`, S54 at 2,048 steps and S55 at 1,024, both exit 0 with cleanup
  certified. Score file: `analysis/a20_score.json`.
  - **J1 relevance, of 52 points:** S54 32, +4 on S34 and +5 on S52; S55 36, +3 on S30. The bar is +9.
  - **Told of 108:** S54 49, +21 on S34 and +19 on S52; S55 37, +6 on S30. The bar is 20, and a move on S54 must hold against both
    comparators. S55's pairs fall by 4.
  - **The guard:** the ended twin on `plain_name` is 5 of 8 on S54, against 8 of 8 on S34. The verdict read the three failures
    (`cv_R_13b`, `cv_R_15b`, `cv_R_16b`), and the label stands. S55's guards hold.
  - **Reported:** `contrast_v002` context pairs 14 against 8 and 7 against 1; `contrast_v001` y59 21 and 7; L3 on own wording, own
    history, 0.823 against 0.448 and 0.354 against 0.708; J1 coherence 2.462 and 3.077; PIQA 0.595 and 0.590; passage 0 on both.
  - **Next by rule:** the guard is the recorded cost and B's two-by-two stays unchanged. Neither arm's seed 2 opens, and neither does S45.
  - **The verdict's reading (Claude Opus 5.5, 7 Nov 2026, exploratory, one seed):** these arms do not show that a stronger base moves
    relevance, and a flat result does not say the base does not matter. S54's told line rests on one seed and a bundled change of
    base, depth, parameters, k, T, 8×2 and corpus, so nothing here attributes it.
- **B1, a stronger pretrained base (§29-35): the race chose A, 8×768.** Stage 0 was not passed, one bar missed (§30). The race
  of 8×768 against 12×768 to step 20,000 chose A (§34). A trained on to step 60,000, 983.04M positions, and reads 1.0378 LAMBADA
  target bits per byte. Its SFT1500 (§35) beats S-full's SFT1500 on every benchmark set but WinoGrande, which is level. That SFT1500
  is A21's start. This work spends no L4-hours of §8. Costs are in `AI_ML/tiny_chat_benchmarks/b1_ledger.md`.
- **A21, S34's and S30's recipes and their 15:15 mixture on A's SFT1500 (§36-36.2), one seed: recipe `told_only`, rule 10; base
  `guard`.** No arm moved J1 against the start, B1SFT. Told recall passed its rule on S62 and S63. Runs `a21-s61-r2-20261109`,
  `a21-s62-r2-20261109`, `a21-s63-20261109` and `a21-s64-20261109`, all whole, validation PASS, cleanup certified. Score file:
  `analysis/a21_score.json`.
  - **J1 relevance, of 52 points:** start 29; S61 26, S62 26, S63 32, S64 27. Against the start that is −3, −3 and +3. The bar is +9.
    S61 is −2 on S34 and −1 on S52, and S64 −6 on S30.
  - **Told of 108:** start 16; S61 33, S62 36, S63 42, S64 38. S62 (+20) and S63 (+26) pass the told rule in full. S61's gain is
    17, with 11 lures.
  - **Mixture:** S63 is +6 J1 points on each pure arm, `some`, unresolved. Neither `up` nor `down`.
  - **The guard:** the ended twin on `plain_name` is S61 5 of 8 against S34's bar of 7, and S64 4 of 8 against S30's 5. The
    verdict read the replies. S61's failures are role persistence. S64's are identity drift ("I don't have a name"), not role leak.
  - **Reported:** attribution start 15 against 27, 28, 46 and 27, mostly RECITAL ("what am I" answered with the assistant's
    identity); L3 on own wording, own history, S61 0.677 against S34's 0.448 and S64 0.729 against S30's 0.708; interaction −12
    and −16 J1 points.
  - **Next by rule:** `told_only` is not a move on the bar. No arm goes to seed 2. The base guard is the recorded cost.
  - **The verdict's reading (Claude Opus 5.5, 9 Nov 2026, exploratory, one seed):** the recipes bring a model to about the start's
    J1 level, and A's start is already there. The first-person "my" failure of the start (142 of 270 recall finals open "I" or
    "my") turns to second person in every arm (200 to 222 open "you"). The remaining form is identity recital. Continued SFT on
    S2's rows is not separated from the recipes, and seed noise on A's base is unmeasured.
- **A22, de-canned SFT rows and content-dependent rows on A's SFT1500 (§37-37.3), one seed: `guard`, rule 7.** S67 (K2, S2's
  rows de-canned and the K2 pool) moved `cue_v001` against S66 and tripped two guards, identity and the ended twin. S66 (K1, S30's
  packet de-canned) did not move against S64 and tripped attribution. S65 (K0, S14's packet) is reported only. All three arms
  start from B1SFT at 1,024 steps on S64's recipe. Runs `a22-s65-20261110`, `a22-s66-20261110` and `a22-s67-20261110`, all on
  Colab L4s, each exit 0 with cleanup certified and parity passed; dose passed on all three. Score file: `analysis/a22_score.json`.
  - **`cue_v001`, components 1 to 4, passes of 174:** start 33, S64 54, S65 48, S66 57, S67 111. S66 against S64 is +3 (27
    wins, 24 losses, p 0.39); S67 against S66 is +54 (61 wins, 7 losses, p < 0.001). The bar is +15 with a one-sided sign test
    at p < 0.05.
  - **Guards:** S66's attribution is 39 against S64's 27, past the +8 limit. S67's identity is 0 correct of 92, with 44 wrong
    names, and its ended twin 0 of 8 against S66's 5. Told and J1 hold on both.
  - **J1 relevance, of 52 points:** start 29, S64 27, S65 26, S66 29, S67 39. S67's 39 is above the 38 on record (§7, §9).
  - **Told of 108, own history:** start 16, S64 38, S65 42, S66 40, S67 63. On fixed history S64 41, S65 36, S66 37, S67 50.
  - **Reported:** component 5 of 120: start 18, S64 35, S65 42, S66 55, S67 66. Canned share: S64 0.411, S65 0.031, S66
    0.001, S67 0.002. S67 against S65, both without identity rows: cue +63 (71 wins, 8 losses), told 63 against 42, J1 39
    against 26. S65 against the start: cue +15 (p 0.018), told 42 against 16, J1 26 against 29. PIQA S65 0.561, S66 0.576, S67
    0.556; bits per byte 1.631, 1.558, 1.610.
  - **Next by rule:** the guards were read in the replies first (§37.3). Nothing goes to seed 2. Jonathan's line of 9 Nov covers
    neither seed 2 nor a new arm; either needs its own registration and his dated line.
  - **The verdict's reading of the guards (Claude Opus 5.5, 10 Nov 2026):** S67's identity and ended-twin trips are one missing
    persona. The identity rows come only with S30's additions, which the K2 pool replaced, so S67 has no identity by
    construction: a gap in A22's packet design, not an effect of the pool's content. The same absence takes S67's attribution
    to 0. S66's attribution is the same default from the other side: with the stock acknowledgements masked, the identity line
    is the next most frequent reply it falls back on. S67's cue gain holds against S65, which also lacks identity rows.
  - **The verdict's reading (Claude Opus 5.5, 10 Nov 2026, exploratory, one seed):** H1, de-canning, does not lift relevance at
    this dose; it removes the copying without supplying what the copies displaced, and cannot tell repetition from volume. H2,
    the K2 pool, moves every relevance measure A22 reads, but a move names the pool, not one property of it. The pool and
    `cue_v001` were each written blind by a Claude Opus 5.5 subagent, so a shared phrasing style may carry part of the cue
    gain; J1 and told recall, from instruments that model family did not write, move too, which bounds that confound without
    removing it. Part of S67's acknowledgement passes are restatements, which the keyword rule passes. Not shown: seed noise on
    `cue_v001`, whether the gain survives restoring the identity rows, which of the pool's five kinds carries it, and whether
    it holds on S-full's base.
- **A23, S67's packet with the identity rows restored, and S67 at a second seed, on A's SFT1500 (§38-38.3): `identity_gap`,
  rule 7.** S67s2 (S67's packet as it is, at seed 2) moved against S66. S68 (S67's packet with S30's A13 identity rows
  restored) held every leg but the ended twin. Both arms start from B1SFT at 1,024 steps on S64's recipe. Runs
  `a23-s68-20261110` and `a23-s67s2-20261110`, both on Colab L4s, each exit 0 with cleanup certified and parity passed; dose
  passed on both (first loss 2.56 and 3.29, falls 0.78 and 0.86). Score file: `analysis/a23_score.json`.
  - **`cue_v001`, components 1 to 4, passes of 174:** S67 111, S68 115, S67s2 107. S68 against S67 is +4 (22 wins, 18 losses),
    not down. S67s2 against S66 is +50 (61 wins, 11 losses, p < 0.001), past A22's bar of +15: A22's move holds at a second
    seed of S67. S67s2 against S67, the first seed-noise reading for one packet, is −4 (15 wins, 19 losses).
  - **S68's legs:** identity 87 correct of 92, 0 denying, 1 wrong name (holds); ended twin 4 of 8 against a bar of 5 (fails);
    J1 34 against S67's 39 and told 61 against 63 (hold); attribution 21 against S64's 27 + 8 (holds).
  - **§38.2's stance and role-play gate fails on both counts,** so S68's packet does not go forward under any verdict.
    persona_v003, read with `score_a24.py`'s reader: S68 engages on 3 of 60 stance turns (42 dead declines) and breaks to the
    assistant on 30 of 48 role-play turns, against gates of at least 30 and at most 12. S67 reads 35 and 0, S67s2 31 and 0.
  - **J1 relevance, of 52 points (mean, coherence):** S67 39 (3.00, 3.15), S68 34 (2.62, 2.77), S67s2 28 (2.15, 2.46).
  - **Told of 108, own history:** S67 63, S68 61, S67s2 56. On fixed history S68 54, S67 50, S67s2 50.
  - **Reported:** components 1 to 4: S68 40, 36, 27, 12; S67s2 39, 32, 24, 12. Component 5 of 120: S67 66, S68 66, S67s2 70.
    Canned share 0.002 on all three. PIQA S68 0.551, S67s2 0.556; bits per byte 1.611, 1.607. S68 against S66: cue +58 (69
    wins, 11 losses), J1 34 against 29, attribution 21 against 39, ended twin 4 against 5. Best-of-16 lines are in §38.3.
  - **Next by rule:** rule 7, `identity_gap`. Under §38.2's gate, the next arm replaces A13's rows with rows that tell the
    cases apart: A24 (§39), under Jonathan's line of 10 Nov 2026.
  - **The verdict's reading (Claude Opus 5.5, 10 Nov 2026, exploratory, two seeds of S67 and one of S68):** The K2 pool's cue
    gain is robust to the seed: S67 at seeds 1 and 2 gives 111 and 107, a 4-item difference, against a gain of 50 to 54 over
    S66. J1 is not robust to the seed at 13 turns: the same packet reads 39 at seed 1 and 28 at seed 2, and told recall 63 and
    56. S67's "highest J1 on record" is one draw from a distribution at least 11 points wide, and J1's guard margin of 9 is
    narrower than that seed difference. A13's identity rows trade stance and role-play for identity, as the stored models
    predicted; the trade sits in the identity rows, not in the K2 pool. Not shown: whether S68's ended twin (4 against 5) is a
    real shortfall or noise, with one seed of S68; whether any dose of A13's rows avoids the trade, with one dose tested. The
    shared-author confound of §37.3 still applies to the cue gain.
- **A24, a persona pool that tells identity from preference, on S67's packet (§39-39.2), one seed: `relevance_cost`, rule 5.**
  S69 (S67's packet with the decontaminated persona pool, written blind by a Claude Opus 5.5 subagent) fell on `cue_v001`
  against S67. Identity, the ended twin and role-play also miss their bars, and stance holds. It starts from B1SFT at 1,024
  steps on S64's recipe. Run `a24-s69-20261110` on a Colab L4, exit 0 with cleanup certified and parity passed; dose passed
  (first loss 2.84, fall 0.76). Score file: `analysis/a24_score.json`.
  - **`cue_v001`, passes of 174:** S69 87 against S67's 111, −24 (15 wins, 39 losses, p < 0.001). Components 1 to 4: 20, 34,
    19, 14 against S67's 42, 36, 20, 13.
  - **Legs:** identity 78 correct of 92, 0 denying, 5 wrong names against a limit of 4 (fails); ended twin 3 of 8 against 5
    (fails); stance 42 engaged of 60 against 30 (holds); role-play 18 breaks to the assistant of 48 against 12, 15 in
    character (fails); J1 34 against S67's 39 and told 61 against 63 (hold); attribution 7 against S64's 27 + 8 (holds).
  - **Beside the comparators:** who it is, stance engaged, breaks: S67 0, 35, 0; S68 87, 3, 30; S69 78, 42, 18. Recitals on
    stance turns 1, 12 and 11.
  - **Reported:** J1 mean 2.62, coherence 2.85. Told on fixed history 53 against S67's 50. Component 5 69 of 120. Canned share
    0.001. PIQA 0.566, bits per byte 1.620. S69 against S68: cue −28 (7 wins, 35 losses), J1 34 against 34, told 61 against
    61, attribution 7 against 21. The preference probe's margin on S69 is reported in §39.2.
  - **Next by rule:** rule 5, `relevance_cost`, with the replies read (§39.2). A25 (§40) was registered and launched on the
    cue drop's suspected source before the persona blocks were read, under Jonathan's line of 10 Nov 2026.
  - **The labels read (Claude Opus 5.5, 10 Nov 2026):** the cue drop is in acknowledgements, 20 against 42; the lost replies
    are content-free acknowledgements with a question back ("That sounds lovely. How long did it take?"), the form of the
    pool's small-talk replies. The role-play breaks are in-role identity questions answered as Ember ("What's your name?" →
    "My name is Ember, and I'm an AI."): decontamination removed most of the pool's in-role identity examples, 79 of 240
    role-play conversations remained, and the model generalises the sincere rule into role-play. The five wrong names are
    human names on name questions, in the base's prior. The ended twin, 3 of 8, is the same in-role and out-of-role boundary.
  - **The verdict's reading (Claude Opus 5.5, 10 Nov 2026, exploratory, one seed):** the tension Jonathan named is largely
    resolved by the data. S69 is the first model to hold its identity (78 of 92) and engage on stance (42 of 60) together;
    models with A13's rows had one without the other, and attribution falls to 7. The readout moved by class: the probe
    margin's class gap widened to 11.5 nats and its intercept fell by about half against S68's, consistent with the twin pairs
    moving the slope, within the fixed-continuation limits of §39. Two costs remain: the filler's content-free
    acknowledgements, which A25 tests, and role-play identity, where the decontaminated pool has too little in-role identity
    data in wording the model can generalise from. Not shown: anything past one seed (J1 and told recall moved by up to 11 and
    7 between S67's seeds); whether the role-play gap closes with in-role data in other wording; whether identity's one wrong
    name past the limit is a real shortfall.
- **A25, supervision by turn role on S69's construction (§40-40.2), one seed: `decline`, rule 6.** S70 (S69's packet with the
  persona pool's 2,036 small-talk and greeting replies set unsupervised, their text kept as context) breaks to the assistant
  on 17 of 48 role-play turns against at most 12. The ended twin, 3 of 8 against 5, is also below (rule 7 would fire next).
  Every relevance leg holds. It starts from B1SFT at 1,024 steps on S64's recipe. Run `a25-s70-20261110` on a Colab L4, exit
  0 with cleanup certified, hash verified and parity passed; evaluation on five Colab CPU sessions, all exit 0 with cleanup
  certified; `cue_v001` read 294 of 294 cases locally; J1 judged 13 of 13 turns (Gemini, per turn). Dose passed (first loss
  2.84, fall 0.75). Score file: `analysis/a25_score.json`.
  - **Legs:** identity 77 correct of 92, 0 denying, 4 wrong names (holds); ended twin 3 of 8 against 5 (fails); stance 45
    engaged of 60 against 30 (holds); role-play 17 breaks to the assistant of 48 against 12, 18 in character (fails);
    `cue_v001` against S67 −4 (15 wins, 19 losses, p down 0.30) and against S67s2 0 (18 wins, 18 losses), not down against
    both (holds); J1 33 against the two seeds' mean less 9, 24.5 (holds); told 61 against 39.5 (holds); attribution 7 against
    S64's 27 + 8 (holds).
  - **Against S69, the mask-only contrast:** cue +20 (28 wins, 8 losses, p < 0.001); acknowledgements 39 against 20,
    suggestion 33 against 34, correction 21 against 19, format 14 against 14; component 5 68 of 120 against 69. Told 61
    against 61, J1 33 against 34, attribution 7 against 7, ended twin 3 against 3, identity 77 against 78, stance 45 against
    42, role-play breaks 17 against 18.
  - **Reported:** `cue_v001` 107 of 174 (39, 33, 21, 14), against S67's 111 and S67s2's 107. J1 mean 2.54, coherence 2.92.
    Told on fixed history 54 against S67's 50 and S69's 53. Canned share 0.002. PIQA 0.571, bits per byte 1.620. Probe
    margin, identity − stance: +14.6 on identity, +3.4 on preference, +7.3 on user, +8.0 on ambiguous questions, against
    S69's +14.7, +3.2, +7.2 and +7.9 (§40.2; in no score file). Sampled targets 2,012,169. S70 against S68: cue −8 (11
    wins, 19 losses); against S64 +53 (69, 16); against the start +74 (79, 5). Best-of-16: 146 of 174 against S67's 147
    (samples passing 72%, 69%, 44%, 26%).
  - **Next by rule:** rule 6, `decline`, with the replies read (§40.2). No next arm follows from the table; a next arm needs
    Jonathan's dated line.
  - **The labels read (Claude Opus 5.5, 10 Nov 2026):** all 17 breaks are items S69 also broke: 13 name or identity
    questions asked in character ("What's your name?" → "My name is Ember, and I'm an AI."), one the character's own name
    with the tail attached, and four personal questions answered as an AI. 33 of S70's 212 persona replies contain "I'm an
    AI, and I'm an AI" (S69 22, S68 none); it scores as correct on identity questions, so the 77 overstates the identity
    reply. Its source is in the pool: the pool's name answers end in a fixed tail, ", and I'm an AI." (128 uses behind six
    openings), which the model learned as a frame. The mask did what it was built for: the acknowledgements came back, and
    nothing else moved beyond a few items.
  - **The verdict's reading (Claude Opus 5.5, 10 Nov 2026, exploratory, one seed):** turn-role supervision recovered
    relevance without losing the persona. S70 holds S67's cue level at two seeds, J1, told recall and attribution, and keeps
    S69's identity and stance: the first model with relevance, identity and stance together. The recovery names the mask
    and its 1.0% loss in sampled targets together. What remains is the pool's identity answer, not the turn roles: the
    role-play breaks and the loop share one source, the fixed tail on every name answer, and the breaks sit where in-role
    identity data was decontaminated away (§39.2). Not shown: anything past one seed; whether varied identity wording
    without the fixed tail, and in-role identity data in wording apart from persona_v003's, close the role-play gap and the
    loop; whether the ended twin's 3 of 8 is the same boundary or another failure.
- **A26, persona pool v2 on S70's construction (§41-41.2), one seed: `ready`, rule 4.** S71 (S70's construction with
  persona pool v2: 63 varied identity sentences and no fixed tail, 520 role-play conversations with in-role identity and
  personal questions, exits in plain words) holds every leg. It starts from B1SFT at 1,024 steps on S64's recipe. Run
  `a26-s71-20261111` on a Colab L4, exit 0 with cleanup certified, hash verified, validation passed; evaluation on five Colab
  CPU sessions, all exit 0 with cleanup certified; `cue_v001` read 294 of 294 cases locally; J1 judged 13 of 13 turns. Dose
  passed (first loss 2.66, fall 0.87). Score file: `analysis/a26_score.json`.
  - **Legs:** identity 80 correct of 92, 0 denying, 3 wrong names (holds); ended twin 5 of 8 against 5 (holds); stance 35
    engaged of 60 against 30 (holds); role-play 6 breaks of 48 against 12, 17 in character (holds); loop 0 of 212 against 2
    (holds); `cue_v001` against S67 −5 (14 wins, 19 losses, p down 0.24) and against S67s2 −1 (16, 17) (holds); J1 36
    against 24.5 (holds); told 62 against 39.5 (holds); attribution 5 against 35 (holds).
  - **Against S70:** cue −1 (18 wins, 19 losses); told 62 against 61; ended twin 5 against 3; J1 36 against 33; identity 80
    against 77; stance 35 against 45; role-play breaks 6 against 17; loop 0 against 33.
  - **Next by rule:** rule 4, `ready`: S71's packet is the candidate recipe. A27 (§43) builds on it under Jonathan's line of
    11 Nov 2026.
  - **The labels read (Claude Opus 5.5, 11 Nov 2026):** the six remaining breaks are name questions asked in character and
    answered as Ember. The stance fall is ten turns S70 engaged that S71 answers truthfully as an AI, which the rubric reads
    as recitals; several of S70's engaged answers there carried the loop or an invented human life.
  - **The verdict's reading (Claude Opus 5.5, 11 Nov 2026, exploratory, one seed):** without v1's fixed tail the loop is
    gone, and with in-role identity data the breaks fall from 17 to 6, relevance holding on every measure. The arm changed
    both at once, so §41's two hypotheses are not separated. Not shown: anything past one seed; transfer to in-role wording
    unlike the pool's.
- **Wide J1, second reading (§42-42.3), exploratory: 532 judged turns per model.** S66 27.0 points on the 13-turn scale,
  S67 26.4, S67s2 25.9, S70 30.6, S71 31.7. Turn-role supervision (S67's packet against S66): not separated at either
  seed. The persona arms (S70 against S67's packet): above at both seeds; S71 against S67's packet: above at both seeds,
  also on the five untargeted families; S71 against S70: not separated. Judge retest 60 of 60 identical. Score files:
  `analysis/rel_wide2_read.json`, `analysis/rel_wide2_read_s71.json`.
  - **Reading (Claude Opus 5.5, 11 Nov 2026):** the 13-turn J1 could not rank these models (S67's 39 reads 26.4 on 532
    turns). The K2 pool moved relevance between families (recall and correction up, chat and identity down), not overall;
    the persona pools won back chat and identity and kept the K2 gains. Followup and abstain are the weakest families of
    every model.
- **A27, a thread pool on S71's packet (§43-43.2), two seeds: `relevance_cost`, rule 6.** S72 (S71's packet with a blind
  thread pool of 2,697 conversations: topic follow-ups, names, told/untold/lure triples) at seeds 1 and 2, each from B1SFT at
  1,024 steps on S64's recipe. Runs `a27-s72-20261111` and `a27-s72s2-20261111` on Colab L4s, exit 0 with cleanup certified,
  hash verified; evaluation on Colab CPU sessions, all exit 0; `cue_v001` 294 of 294 each; J1 13 of 13 each. Dose passed
  (3.24 and 0.85; 2.63 and 0.83). Score files: `analysis/a27_score.json`, `analysis/rel_wide3_read.json`.
  - **Primary, wide J1 against S71's 31.7 points:** S72 29.1 (−2.6, −4.2 to −1.0, below); S72s2 30.2 (−1.5, −3.4 to +0.3,
    not separated): split between seeds. The three targeted families: not separated at either seed; the other five: below
    at both seeds.
  - **Guards:** told 32 and 38 against a bar above 42 (fail at both); ended twin 2 and 2 against 5 (fail at both); role-play
    breaks 15 and 10 against 12 (fail at seed 1); identity 75 and 78 of 92, stance 34 and 34, loop 1 and 2, attribution 6
    and 3 hold; cue +14 and +7 against S71 (120 and 113 of 174, the highest read).
  - **Next by rule:** rule 6, `relevance_cost`: the replies are read. S71's packet stays the candidate recipe.
  - **The labels read (Claude Opus 5.5, 11 Nov 2026):** told recall fell to false declines (recall finals answered "not
    told" rise from 0 to 25 and 20; correct values fall from 127 to 89 at both seeds). The pool's ", then." tag (950 of its
    7,944 supervised replies) appears on 30 and 29 of the arms' wide turns and none of S71's, and its paraphrases change
    the user's fact where S71 echoes it.
  - **The verdict's reading (Claude Opus 5.5, 11 Nov 2026, exploratory, two seeds against one of S71):** the pool cost
    relevance outside its targets and did not buy it inside them. Not separated: which of the pool's three kinds did the
    harm, and the share due to diluting the base's parts.
- **A29, S71's packet on an open base (§45-45.3), two seeds: `base_up`, rule 3.** S71's packet and recipe on SmolLM2-135M-Instruct
  (O2, O2s2), with the released 135M and 360M models reported (O1, O3), replayed greedily on S71's evaluation suites and judged on the
  wide J1. One Colab L4 session, `a29-20261111-r2`, exit 0 with cleanup certified (a first session, `a29-20261111`, ran out of GPU
  memory at O2's training and stopped cleanly; §45.2). Score files: `analysis/a29_score.json`, `analysis/rel_wide5_read.json`.
  - **Primary, wide J1 against S71's 31.7 points:** O2 37.0 (+5.3, +3.2 to +7.4); O2s2 36.4 (+4.7, +2.6 to +6.9): above at both
    seeds. Identity 83 and 80 of 92 (holds at both). Released: O1 24.5, O3 28.3, both below S71 and mostly at the reply cap.
  - **Reported:** role-play breaks 14 and 14 of 48 (S71's 6); stance 34 and 40; loop 4 and 0; follow-up family 2.41 and 2.19
    (S71's 1.66), abstain 2.38 and 2.22 (1.90).
  - **The labels read (Claude Opus 5.5, 11 Nov 2026):** the gain is not reply length (largest where O2's reply is no longer than
    S71's); it is recall of names given earlier, correct facts in follow-ups and reactions that keep the user's sentence.
  - **The verdict's reading (Claude Opus 5.5, 11 Nov 2026, exploratory):** the base was the limit on relevance; the package (base,
    tokeniser, trainer, engine) changed together, so no one ingredient is named. Role-play does not carry over.
- **A28, S71's packet at seed 2 (§44-44.2), one seed: `wide_only`, rule 6.** Run `a28-s71s2-20261111`, exit 0, cleanup certified.
  Wide J1 29.7 points: above S67's packet at both of its seeds (+3.3, +3.8), below S71 at seed 1 (−2.0, −3.7 to −0.3), level with
  S70. A26's legs at seed 2: identity 81 of 92, stance 37, told 58, attribution 4 and cue hold; the ended twin (3 of 8), role-play
  (15 breaks of 48) and the loop (3 of 212) miss. Score files: `analysis/a28_score.json`, `analysis/rel_wide4_read.json`.
  - **Reading (Claude Fable 5.2, 11 Nov 2026):** §41.2's `ready` is one seed's. Two seeds of S71's packet differ by 2.0 wide J1
    points and 9 role-play breaks; A27's arms sit inside that spread on relevance (their told recall does not), and A29's arms sit
    6.7 to 7.3 points above S71s2, outside it.
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

A18 (root `PROTOCOL.md` §26), authorised by Jonathan on 6 Nov, ran depth for width on October's 20M bases, the cheap first test
this section names. Its verdict is rule 8, unresolved (§26.3), and it does not rule out depth at 73M. The 2×2 below stays proposed.

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

**Stage 0b result** (`analysis/state_ablation_results.md`, 6 Nov; 320 forced-choice items on S-full, S24, S25 and S23).
The replies do not depend on slot content; recall comes from the raw turns. Masking all slots costs nothing on S-full,
S24 or S25, and improves S23's recall by 0.36 nats, so there they act as a distractor. With raw turns hidden, the slots
add nothing beyond the empty floor (only S25's interval clears zero, narrowly), and replacing them with another
conversation's slots changes the margin by at most 0.03 nats. Together with Stage 0, the state is written but not read.
A head that supervises writing alone could make the slots more decodable and leave them inert, so sufficiency with raw
turns hidden and the donor test are the endpoints that matter.

**Scope of that result.** Stage 0b shows the existing unsupervised tokens are largely bypassed under the tested inference
conditions. It does not falsify supervised typed registers, which remain a separate hypothesis from visible scratchpads, with
the endpoints above and matched typed and untyped controls.

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
| S23 (highest relevance on S-full's lineage) | same | same | 1,024 steps, 30% instruction, 1,088 phrasings | 1,088 phrasings | 47.5 min | 2 | 1 of 64 | 2.46 | 2.92 | unmeasured | 1 | A12's primary +16 against +18 |
| S30 (practical baseline, not confirmed) | same | same | 1,024 steps, 30% hybrid | 1,088-phrasing generator | 59.3 min | 0 | 1 of 64 | 2.85 | 2.54 | unmeasured | 1 | fails precision and discrimination |
| S34 (most context pairs on S-full's lineage, 16 of 59) | same | same | 2,048 steps, 30% contrast | not varied | 73.8 min | not scored | not scored | 2.46 | 2.15 | unmeasured | 1 | copy reflex recorded; 8 of 59 on `contrast_v002` |
| S51, M1 (from S34) | same | same | 1,024 more steps, mixed rows | not varied | 67.5 min with S52 | not scored | not scored | 2.46 | 2.00 | unmeasured | 1 | corrections 3 of 16 against C0's 7; ended twin 6 of 8 |
| S54 (most context pairs, 21 of 59; 49 told of 108) | 97.54M total | 983.04M positions | pk_sft_v1, then 2,048 steps, S34's packet, k = 0 | not varied | 80.2 min | not scored | not scored | 2.46 | 2.46 | unmeasured | 1 | ended twin 5 of 8, a guard; 14 of 59 on `contrast_v002` |
| S55 (highest coherence on pk_sft_v1) | as S54 | as S54 | pk_sft_v1, then 1,024 steps, S30's packet, k = 0 | as S30 | 47.3 min | not scored | not scored | 3.08 | 2.77 | unmeasured | 1 | told pairs 4 below S30; 7 of 59 on `contrast_v002` |
| B1SFT, A's SFT1500 (A21's start) | 72.76M / 60.17M | 983.04M positions | SFT1500 | n/a | RunPod, not §8 | not scored | not scored | 2.54 | 2.23 | unmeasured | 1 | who-it-is 1 of 92; told 16 of 108 |
| S63 (most told recall, 42 of 108) | as B1SFT | as B1SFT | B1SFT, then 2,048 steps, the 15:15 mixture | not varied | 68.1 min | not scored | not scored | 2.92 | 2.46 | unmeasured | 1 | attribution 46 against the start's 15 |
| S67 (highest relevance on record at one seed, J1 39 of 52; 28 at seed 2; highest coherence in this table; `cue_v001` 111 of 174, 107 at seed 2) | as B1SFT | as B1SFT | B1SFT, then 1,024 steps, S2's rows de-canned and the K2 pool | K2 pool, no reply used more than twice | 42.9 min; seed 2 51.1 | not scored | not scored | 3.15; seed 2 2.46 | 3.00; seed 2 2.15 | unmeasured | 2 | identity 0 of 92 and ended twin 0 of 8, both guards, at both seeds; its packet has no identity rows |
| S69 (first model with identity and stance together: who it is 78 of 92, stance 42 of 60) | as B1SFT | as B1SFT | B1SFT, then 1,024 steps, S67's packet with the persona pool | persona pool, decontaminated, 1,945 conversations | 47.4 min | not scored | not scored | 2.85 | 2.62 | unmeasured | 1 | `cue_v001` 87 of 174 against S67's 111; 5 wrong names, past the limit; role-play breaks 18 of 48; ended twin 3 of 8 |
| S70 (first model with relevance, identity and stance together, by A25's reading: `cue_v001` 107 of 174, who it is 77 of 92, stance 45 of 60; verdict `decline`, not `ready`) | as B1SFT | as B1SFT | B1SFT, then 1,024 steps, S69's packet with the persona pool's small-talk and greeting replies unsupervised | persona pool, decontaminated; 2,036 filler replies kept as context | 39.8 min | not scored | not scored | 2.92 | 2.54 | unmeasured | 1 | role-play breaks 17 of 48, past the gate; ended twin 3 of 8; "I'm an AI, and I'm an AI" in 33 of 212 persona replies, so who it is overstates the identity reply |
| S71 (first `ready`: identity 80 of 92, role-play breaks 6 of 48, no loop, relevance held; highest wide J1, 31.7 points on 532 turns, level with S70) | as B1SFT | as B1SFT | B1SFT, then 1,024 steps, S70's construction with persona pool v2 | persona pool v2, decontaminated; 63 identity sentences, 520 role-play conversations | 40.1 min | not scored | not scored | 3.15 | 2.77 | unmeasured | 1 | stance 35 of 60, from S70's 45: truthful AI answers read as recitals |

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
| a15-s1 (S30, S31; one session each) | 114.4 | two exit 0 |
| **4-6 Nov, through A15** | **about 1,210 (20.2 h)** | the week's 6 h, increased by Jonathan's redistribution of budget to this week (4 Nov) |
| a16-s1 (S32, S33, S34; one session each) | 168.7 | three exit 0; a verdict, §24.3 |
| A17 (S30, CPU) | none | no L4; a verdict, §25.3 |
| a19-s2 (S51, S52, seed 2; one session) | 67.5 | exit 0; a verdict, §27.1; evaluation on Colab CPU runtimes, no L4 |
| a18-s1 to a18-s4 (S35 to S44; four sessions) | 292.9 (4.88 h) | four exit 0; a verdict, §26.3; sessions 83.4, 83.1, 67.2 and 59.2 |
| a20-s54-20261107 (S54) | 80.2 | exit 0; a verdict, §28.1 |
| a20-s55-20261107 (S55) | 47.3 | exit 0; a verdict, §28.1; A20 in all 127.5 (2.12 h) |
| **4-7 Nov, through A20's and A18's verdicts** | **about 1,866 (31.1 h)** | §8's 6 h is past; Jonathan's dated line of 7 Nov stands for up to 12 L4-hours in this period, shared by A18, A19 and A20 (§26, §28), which spent 8.1 |
| a21-s61-20261109, a21-s62-20261109 | 9.4, 3.0 | failed before training (GitHub 504 at `configure`); cleanup certified, nothing home |
| a21-s61-r2-20261109 (S61) | 80.2 | exit 0; rerun under Jonathan's dated line of 9 Nov; a verdict, §36.2 |
| a21-s62-r2-20261109 (S62) | 68.4 | exit 0; rerun under the same line; a verdict, §36.2 |
| a21-s63-20261109 (S63) | 68.1 | exit 0; a verdict, §36.2 |
| a21-s64-20261109 (S64) | 48.6 | exit 0; a verdict, §36.2; A21 in all 277.7 (4.63 h), inside its registered 375 and Jonathan's dated line of 9 Nov past §8's weekly 6 |
| a22-s65, a22-s66, a22-s67, first runs and `-r2` reruns (10 Nov) | 0 | all six refused by Colab at GPU assignment; no session created, cleanup certified, folders kept as `a22nc-…` receipts (§37.2) |
| a22-s65-20261110 (S65) | 56.8 | exit 0; reported only; a verdict, §37.3 |
| a22-s66-20261110 (S66) | 44.6 | exit 0; a verdict, §37.3 |
| a22-s67-20261110 (S67) | 42.9 | exit 0; a verdict, §37.3; A22 in all 144.3 (2.41 h), inside its registered 225 and Jonathan's dated line of 9 Nov past §8's weekly 6 |
| a23-s68-20261110 (S68) | 46.2 | exit 0; a verdict, §38.3 |
| a23-s67s2-20261110 (S67 at seed 2) | 51.1 | exit 0; a verdict, §38.3; A23 in all 97.3 (1.62 h), inside its registered 150 under Jonathan's line of 10 Nov |
| a24-s69-20261110 (S69) | 47.4 | exit 0 at its first launch; no launch refused; a verdict, §39.2; inside its registered 75 under Jonathan's line of 10 Nov |
| a25-s70-20261110 (S70) | 39.8 | exit 0 at its first launch (`a25_retry_s70.log`, one entry); a verdict, §40.2; inside its registered 75 under Jonathan's line of 10 Nov; evaluation on five Colab CPU sessions, no L4 |
| a26-s71-20261111 (S71) | 40.1 | exit 0; a verdict, §41.2; inside its registered 75 under Jonathan's line of 10 Nov; evaluation on five Colab CPU sessions, no L4 |
| a27-s72-20261111 (S72) | 46.3 | exit 0; a verdict, §43.2; inside its registered 75 under Jonathan's line of 11 Nov; evaluation on Colab CPU sessions, no L4 |
| a27-s72s2-20261111 (S72 at seed 2) | 46.1 | exit 0; a verdict, §43.2; inside its registered 75 under Jonathan's line of 11 Nov |
| a28-s71s2-20261111 (S71 at seed 2) | 39.1 | exit 0; a verdict, §44.2; by the driver log; inside its registered 75 under Jonathan's line of 11 Nov |
| a29-20261111 (A29, first session) | 6.6 | exit 2; out of GPU memory at O2's training; cleanup certified; nothing placed (§45.2) |
| a29-20261111-r2 (A29: O1, O3, O2, O2s2) | 37.1 | exit 0; a verdict, §45.3; inside its registered 90 under Jonathan's line of 11 Nov |

By session creation to the last certified stop, `score_a19.py compute` reads 1,650.2 minutes for the lane on its own count, with
a18-s1, a20-s54 and a20-s55 inside it. A18's sessions 2 to 4 add 209.5, so that count stands at 1,859.7 (31.0 h).

CPU evaluation runs under the shared CPU budget (`scripts/cpu_budget.py`, session `claude`). From A19 it runs on Colab CPU runtimes
through `cloud_eval/`, at 0.08 units an hour per session, with no L4.

## 9. Retractions and superseded readings

- **Model size.** The first version of this registry (`baff7c47`) gave 85.3M parameters and called "70M"
  an undercount. That reading is superseded: 85.3M counted the GGUF's duplicated tied embedding. The model has
  72,761,088 unique parameters (section 2, Codex's correction, 4 Nov). The same version's depth arm, 16 × 576
  (77.1M unique), was not parameter-matched; 16 × 560 replaces it.
- **Wave-3 incident (4 Nov):** the reading "a fresh VM may not answer its first exec" is superseded. The review
  found that the VM answered and the local Colab CLI stalled (`reviews/launch_review_6_w3_rerun.md`).
- **Pretraining cost:** the 130-L4-hour figure for ten times S-full's tokens is superseded. It matched S-full's
  checkpoint-burdened rate. At the clean rate of about 40M positions an L4-hour the figure is about 41 hours
  (`AI_ML/tiny_chat_benchmarks/base_receipt_comparison.md` §6.6).
- **Where S-full's knowledge gap sits:** the receipts comparison of 6 Nov could not order the SFT recipe against pretraining
  exposure. Gate 0 (`cloud_eval/receipts/d0cpu5/decision_compare.json`) places 82% of the LAMBADA target-bits gap in the base:
  1.4004 for S-full's base, 1.493 after SFT1500, 0.970 for pk_base_v1. It locates the gap and names no cause.
- **Most context pairs:** S34's 16 of 59 on `contrast_v001` no longer tops the programme. S54 has 21 (`analysis/a20_score.json`), on a
  different base and shape. S34 stays the highest on S-full's lineage.
- **Highest relevance and coherence:** S23's 38 of 52 J1 points (2.92) no longer top the programme, and neither does S55's
  coherence of 3.08. S67 has 39 points (3.00) and coherence 3.15 (`analysis/a22_score.json`), on A's SFT1500. S23 stays the
  highest relevance on S-full's lineage.
- **S67's J1 record:** A22's reading of S67's 39 points as the highest J1 on record is superseded as a property of the packet.
  It is one draw: the same packet at seed 2, S67s2, reads 28 points (2.15) and told 56 against 63 (`analysis/a23_score.json`).
  By A23's reading (Claude Opus 5.5, 10 Nov 2026), J1 at 13 turns is not robust to the seed, and its guard margin of 9 is
  narrower than that seed difference, so a single-seed J1 move of that size cannot be read as a lever. `cue_v001` held across
  the two seeds, 111 and 107.
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
4. **Benchmark it** against comparable small models at scale, reporting where it stands. The aim is a fair
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

**The stronger base is trained and its recipe screen is read.** B1's race chose A, 8×768, trained to 983.04M positions (§34,
§35). A21 read S34's and S30's recipes and their mixture on A's SFT1500: `told_only` and a base `guard` (§36.2). No arm goes to
seed 2. A22 read de-canning and the K2 pool on the same start: `guard` (§37.3). S67, the K2 pool, moved `cue_v001` and tripped
identity and the ended twin; S66, de-canning, did not move. A23 read S67 at seed 2 and S67's packet with A13's identity rows
restored: `identity_gap` (§38.3). S67's cue gain held at seed 2 (+50 against S66), and S68 brought identity back while
failing §38.2's stance and role-play gate. A24 replaced those rows with a persona pool: `relevance_cost` (§39.2). S69 reads
identity (78 of 92) and stance (42 of 60) together and falls 24 `cue_v001` items against S67. A25 trained S70, S69's
construction with the persona pool's small-talk and greeting replies unsupervised: `decline` (§40.2). S70 holds every
relevance leg against S67's two seeds and keeps S69's identity and stance, and breaks to the assistant on 17 of 48 role-play
turns; its ended twin, 3 of 8, is also below. A26 trained S71, S70's construction with persona pool v2: `ready` (§41.2),
the first model to hold identity, stance, role-play, no loop and relevance together. The wide J1 (§42) read 532 turns per
model: S71 is above S67's packet at both seeds and level with S70. A27 (§43), under Jonathan's line of 11 Nov 2026, added
a blind thread pool (topic follow-ups, names, told/untold/lure) to S71's packet at two seeds: `relevance_cost` (§43.2).
S72 reads below S71 on the wide J1 at seed 1 and not separated at seed 2, and told recall falls to false declines. S71
stays the candidate recipe and the best model read. Nothing goes to seed 2 without a new registration and Jonathan's dated line. The A-against-B interaction stays unestimated: §36 names its two routes and their costs, and either needs Jonathan's
dated line and Codex's admission. Jonathan's cumulative cap for the stronger base and its follow-ons is $80 (§31).

Beside it, each registration's continuation table decides:
- **A19:** rule 3 holds, so seed 3 waits until the cause of the guard is read, and M2 waits as registered.
- **A18:** rule 8, unresolved, recorded with the curve and s. No extension exists at 20M inside this registration, and S49 and S50 don't open.
- **A28:** rule 6, `wide_only`. The replies are read (§44.2); S71's recipe stays the best own-base recipe, at a 2-point seed spread.
- **A29:** rule 3, `base_up`. On this evidence our base limits relevance; the base question goes to Jonathan (§45.3).
- **A27:** rule 6, `relevance_cost`. The replies are read (§43.2). No next arm follows from the table.
- **A26:** rule 4, `ready`. S71's packet is the candidate recipe; it was A27's base (§41.2, §43).
- **A25:** rule 6, `decline`. The replies are read (§40.2). No next arm follows from the table; a next arm needs Jonathan's
  dated line. S70 at seed 2 is not opened.
- **A24:** rule 5, `relevance_cost`. The replies are read (§39.2); A25 tested the acknowledgement drop's suspected source. S69 at
  seed 2 is not opened.
- **A23:** rule 7, `identity_gap`. §38.2's gate fails on both counts, so S68's packet does not go forward under any verdict; A24
  is the next arm it names.
- **A22:** rule 7, `guard`. The guards were read in the replies (§37.3); nothing goes to seed 2. A23 is the registration that
  followed, under Jonathan's line of 10 Nov.
- **A21:** rule 10, `told_only`; base `guard`. Not a move on the bar; the replies are read in §36.2. No arm goes to seed 2.
- **A20:** rule 6, `guard`. The guard is the recorded cost. Neither arm's seed 2 opens, and neither does S45.
- **A17:** a read-only training arm stays the next test of a note, on a separate registration.

S30 is the practical baseline, not a confirmed winner: promotion needs a sealed evaluation set, and none is built. Seed-1 screens are
triage, 30% is a reference dose, and a candidate is replicated before it is promoted. An intervention on S-full's base reads as
conditional on that base, not as failed in general (Jonathan, 7 Nov 2026). Any later training run has its design, hypothesis and bar
written here and in the root protocol, and independently reviewed, before training.

**Stopping rule:** no GPU training without a registered design, review and budget. An intervention that does not
beat its control at its registered bar is recorded here as a negative and not retried with relaxed criteria.
