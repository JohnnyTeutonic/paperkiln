# 0001 — Post-training charter, and the corpus selection ledger

*Status: **APPROVED by the owner, 16 Oct 2026.** The owner delegated
the open judgement calls; they are recorded in §12. The decision rules
here are frozen in the D4 sense. Each study gets its own
`PREREGISTRATION.md` (the licence anchor) that fixes the numbers this
charter leaves to a calibration pilot. Decisions and results are appended
as dated sections.*

---

## 1. Scope and non-goals

**In scope for v1.** Five post-training methods, each a readable forward
and a hand-derived backward on the paperkiln tape:

| Method | What it is here | Spec key |
|---|---|---|
| SFT | next-token loss on completions, with prompt tokens masked out | `train.objective: "sft"` |
| Rejection sampling, then SFT | sample N, score with a verifier, keep the accepted ones, SFT on them | `train.objective: "rejection_sft"` |
| Verifiable-reward RL | GRPO: group-relative advantage, KL to a frozen reference | `train.objective: "grpo"` |
| Preference optimisation | DPO with a per-pair weight taken from the pair's evidence record | `train.objective: "provenance_po"` |
| Reward modelling | **the verifiers themselves** (§6); no learned reward model in v1 | — |

**Out of scope for v1:** PPO and any learned value head or critic.
Multi-turn RL. Tool-use RL. Constitutional or self-critique loops. Learned
reward models. Online judge calls inside a training step. No dependency
on trl, verl, OpenRLHF or PyTorch, except an optional HF parity check
(as `tools/hf_export_verify.py` already does).

**The scale this runs at, stated up front.** paperkiln models are 0.95M
to about 10M parameters, with word-level vocabularies capped at 4,096. A
token is a run of letters, digits and apostrophes, so `42` is one token
and `1234` is `<unk>` unless the vocabulary contains it. Three
consequences:

1. GSM8K, HumanEval-class code, and open-domain instruction following
   are out of reach, and **no v1 claim is scoped to them**. The verifiable
   task is synthetic and lives inside the vocabulary (§9, `tinyarith`).
2. A completion's answer is often one token. Then GRPO at the answer
   position is close to a contextual bandit over the vocabulary. That
   makes the stack exactly checkable (§4), and it bounds what any
   finding says about long-horizon credit assignment: nothing.
3. Every finding's `scope` carries the parameter range, vocabulary,
   task generator version, and the word "synthetic".

## 2. The provenance thesis

**Claim.** At small scale, preference data whose labels carry mechanical
evidence (exact match, unit test) produces policies that differ
measurably from preference data labelled by a judge alone. The
difference runs towards *less reward hacking*. Reward hacking means a
policy that scores well on its training signal and badly on the truth.

**What would make the claim trivial, and how the design avoids it.**
Judge labels are sometimes wrong, so a judge-trained policy being worse
on truth says only that wrong labels hurt. Nobody needs an experiment
for that. The claim has content only if the harm comes from the
*structure* of judge error (systematic, exploitable bias), not its
*rate*. Every comparison therefore includes a **noise-matched arm**:
exact-match labels flipped at random at the judge's measured error
rate. The thesis is supported only if the judge arm hacks more than the
noise-matched arm. If it does not, the finding is "judge labels act like
random label noise at this scale". That is a publishable negative and
the thesis is in trouble (§5, F-PT5a).

**Why the judge is a synthetic, biased judge in the primary arms.** A
competent LLM judge grades two-number arithmetic close to perfectly.
The "judge-only" arm would then equal the exact-match arm, and the
experiment would measure nothing. The primary judge is
`SyntheticJudgeVerifier`: deterministic, receipted, with a published
error profile. It gives credit for the format cue ("the answer is"),
for answers within ±k of the truth, and for confident phrasing, at
rates set by its parameters. One **calibration arm** runs a real API
judge on the fixed audit set (§3.5). The real judge's measured error
profile, published as a number, either licenses the synthetic profile
or replaces it in the study's pre-registration. The claim is scoped to
"judges with this error profile", never "LLM judges".

Everything else in this charter is scaffolding to test §2. §8 adds a
second, parallel claim about corpora.

## 3. The measurement plan

All metrics come from the events stream (D1). Each one is computed on
**frozen held-out sets** (§9.5) by **frozen verifiers**, identical for
every arm, and never by the verifier that arm trained on.

| Metric | Definition | Where |
|---|---|---|
| `truth_acc` | Fraction of held-out completions the `ExactMatchVerifier` accepts. Temperature 1.0, no top-k or top-p, a fixed number of samples per prompt. | `eval` events |
| `judge_truth_gap` | Mean reward from the **frozen evaluation judge** minus `truth_acc`, on the same completions. Every arm is scored by the same judge, so the gap measures how far a policy fools that judge, not how well it fits its own training signal. | `eval` |
| `hack_frac` | Fraction of completions that pass the `FormatVerifier` and fail the `ExactMatchVerifier`. The reward-hacking signal. | `step` (training batch) + `eval` (probe set) |
| `kl_ref` | Exact per-token KL(π ‖ π_ref), summed over the vocabulary at every position of sampled completions and averaged. The vocabulary is ≤ 4,096, so the exact sum is cheap and no k1/k3 estimator is needed. | `step` |
| `entropy` | Exact per-position policy entropy, averaged | `step` |
| `ess` | Kish effective sample size, (Σw)² / Σw², of the pair weights in the batch. Σw is logged alongside it. | `step` (provenance_po) |
| `reward_mean` | Mean training reward | `step` |

One detector per verifier (each verifier ships its own; §6):

- **ExactMatch:** cannot be hacked by construction on the answer token.
  The detector checks for *extraction* hacks: several numbers in one
  completion, or an answer outside the parse template. It reports the
  fraction of accepted completions that parse more than one candidate
  answer.
- **Format:** `hack_frac` itself.
- **Judge (synthetic or real):** `judge_truth_gap`, plus the bias
  decomposition (fraction of accepted-wrong completions carrying the
  format cue, fraction off by at most k).
- **UnitTest:** fraction of accepted programs that also pass a
  held-out adversarial test (it catches special-casing of the visible
  tests).

### 3.1 Statistics, fixed before data

- **Seeds.** At least 5 per cell. Every primary comparison in §5 uses
  **10**, because runs at this scale are cheap. With a paired t on
  df = 4 the critical value is 2.776, so five seeds detect only effects
  of about 1.7 seed-SDs at 80% power. That is too coarse for the
  primary question.
- **Pairing.** Arms share seeds, prompt order and evaluation samples.
  Effects are paired differences.
- **What every comparison reports:** mean paired difference, SE, t,
  df, the per-seed values (the seed spread, as `SEED_LOTTERY.md`), and
  the sign count.
- **"Inside seed noise"** is a registered outcome, not a failure. If
  |mean difference| < SD of per-seed differences, or t is below the
  two-tailed critical value, the row reads "not resolved at n seeds"
  with the interval. It is published as such.
- **Decision rules** are written as code in each study's `analyze.py`,
  with a `smoke_analyze.py` that drives every branch, positive branches
  included (D3), before data exists.

### 3.2 Calibration before licence

Some numbers cannot be fixed responsibly before seeing the base model:
the `tinyarith` difficulty that puts base accuracy in a band, the
synthetic judge's error rates, and the `hack_frac` threshold for
"monitor fires". Each is set by a **calibration pilot whose rule is
written here**. The pilot's runs are excluded from every confirmatory
analysis:

- **Task difficulty:** choose the smallest operand range at which the
  SFT base model's `truth_acc` lies in [0.25, 0.60] on the dev split,
  mean over 3 seeds. Below that, RL has no signal. Above it, there is
  no headroom.
- **Judge profile:** set the synthetic judge's false-accept rate on
  wrong completions to the rate the real judge shows on the audit set,
  rounded up to the nearest 0.05. If the real judge shows under 0.05,
  use the pre-declared fallback grid {0.10, 0.25} and say so. The study
  is then about judges *worse* than the real one, and the scope says
  this.
- **Monitor threshold:** the base model's `hack_frac` on the probe set,
  plus 3 SD over seeds.

## 4. Mechanisms, their checks, and their falsifiers

Each mechanism carries its falsifier in its own header, as in `srd.hpp`.
A flag or a mode in the same file turns the mechanism off or scrambles
it, so the null run uses the same code path.

| Mechanism | Gradcheck / parity | Falsifier shipped in the module |
|---|---|---|
| `loss_mask` (SFT) | FD gradcheck of masked CE w.r.t. logits. Masked CE with an all-ones mask is **bit-identical** to `ops::cross_entropy`: same kernel and same reduction order, so the claim is exact equality, not tolerance. | `mask_mode: "invert"` trains on prompts only. If held-out answer accuracy matches normal SFT, the mask was not doing what we think. |
| `rejection_sft` | Round trip: sampler JSONL → loader → identical token ids and masks. HF export round trip of the SFT'd model via `hf_export_verify.py`. | `accept: "random"` keeps a random subset of the same size. If accuracy matches verifier-filtered SFT, the filter carries no signal at this scale. |
| GRPO | (1) FD gradcheck of the surrogate w.r.t. logits, at fixed samples and advantages. (2) **Two-action bandit with a known optimum.** With std-normalisation off, the KL-regularised optimum is π\* ∝ π_ref · exp(r/β). Train on the bandit and require |π − π\*| < 1e-3. (3) With G = 1 the advantage is identically 0 and the gradient is exactly the KL term (unit test). | `advantage: "shuffled"` permutes advantages within a group. If reward still climbs, the improvement did not come from the advantage. |
| `provenance_po` | FD gradcheck of weighted DPO w.r.t. policy logits. With all weights 1 it is bit-identical to unweighted DPO. Two parity anchors: weights in {0,1} equal DPO on the kept subset; β → large with π = π_ref gives the analytic gradient. | `weights: "shuffled"` permutes weights across pairs, keeping Σw and ESS. If it matches true weighting, provenance carries no information (F-PT5b). |
| reward-hacking monitor | Smoke: a fabricated stream with a known hack rate must make the monitor fire, and a clean stream must not (D3's "can it fire" rule). | §5, F-PT4. |

**GRPO choices, stated.**

- **GRPO, not PPO.** PPO needs a value network (a second model with its
  own failure modes and tuning). At this scale and with verifiable
  rewards, a group-relative baseline gives the variance reduction PPO's
  critic exists for. That baseline is what the recent
  reasoning-model work used.
- **Advantage:** A_i = (r_i − mean_G r) / (std_G r + ε), with ε = 1e-6
  and std-normalisation a spec switch (`grpo.std_norm`, default true).
  A group with identical rewards gives A = 0 and is logged as a
  "flat group" count. A batch of flat groups yields no policy
  gradient, and the step event says so.
- **One gradient step per sampled batch in v1.** The importance ratio
  is then exactly 1 at the point of differentiation, and PPO-style
  clipping does nothing. It is omitted, and the omission is documented.
  Multi-epoch reuse (where clipping matters) is v2.
- **Length normalisation:** sum over tokens divided by a constant (the
  maximum completion length), not by each sequence's own length. The
  per-sequence form biases the gradient by length (the critique of
  "Dr. GRPO", Liu et al. 2025). It is a spec switch, so the comparison
  can be run.
- **Sampling** is pure temperature sampling, with no top-k or top-p:
  truncation makes the sampled distribution differ from π, and the
  gradient would be off-policy without saying so.

**Provenance weighting, stated.**

- **Default weights:** `exact_match` 1.0, `unit_test` 1.0, `human` 1.0,
  `format` 0.5, `judge` 0.3, missing `evidence_kind` or `evidence_uri`
  0.0.
- **A pair's weight is the minimum** of the weights of the two
  verdicts that ordered it.
- **The `audited` flag does not zero mechanical evidence.** An exact
  match or a unit test is reproducible: re-running the verifier *is*
  the audit. Audit status changes the weight only for `judge` evidence.
  An unaudited judge verdict gets 0.3, and an audit that overturns it
  sets the pair to 0.
- **The weight table is frozen at licence.** Choosing 0.3 after seeing
  results would be a forking path, so sensitivity to the table is a
  *separate*, pre-registered sweep, never a tuning loop.

## 5. The studies and their falsifiers

Each study gets its own directory, `experiments/posttrain_<name>/`.
`PREREGISTRATION.md`, `analyze.py`, `smoke_analyze.py` and the sweep
files go in at the licence commit, before any run.

**Study PT2 — filter provenance (rejection sampling, then SFT).** Same
prompts and same base model. The arms filter the same samples with
different verifiers, at an equal number of accepted examples:

- E: exact-match
- J: synthetic judge
- R: exact-match with acceptance flipped at J's error rate (the
  noise-matched arm)
- X: a random accept, as a floor

SFT each, 10 seeds.

- *Prediction:* `judge_truth_gap`(J) > `judge_truth_gap`(R) ≥
  `judge_truth_gap`(E).
- **F-PT2:** if J − R is not resolved, the J-specific harm is not
  established at this scale. Registered as a negative.

**Study PT4 — the hacking monitor.**

- *Open channel:* GRPO with reward = `FormatVerifier` only (credit for
  "the answer is <number>" whatever the number).
- *Closed channel:* GRPO with reward = exact match.
- Both run 10 seeds from the same SFT base.

**F-PT4** (two parts, both required):

- (a) the open run's `hack_frac` crosses the calibrated threshold, and
  its `truth_acc` does not rise with `reward_mean`;
- (b) the closed run's `hack_frac` stays under the threshold.

If (a) fails, the monitor cannot see a hack we built on purpose, and it
is not shipped as a monitor. Both runs are registered whatever happens.

**Study PT5 — provenance-weighted preference optimisation (the
contribution).** Same prompt set and **the same number of pairs, N, in
every arm**. Six arms, 10 seeds each (60 runs):

| Arm | Pairs |
|---|---|
| E | exact-match ordered |
| J | synthetic-judge ordered |
| R | exact-match ordered, then flipped at random at J's measured error rate |
| Mu | 50/50 E and J, unweighted |
| Mw | 50/50 E and J, provenance weights (§4) |
| Md | 50/50 E and J, judge pairs dropped (weight 0). The "just throw judge data away" baseline, at lower ESS. |

- **F-PT5a (the thesis):** `hack_frac`(J) − `hack_frac`(R) > 0,
  resolved at 10 seeds. If not, judge error acts as plain noise here,
  and §2 is published as not supported at this scale.
- **F-PT5b (the mechanism):** Mw must beat Mu on `hack_frac` **and** must
  not lose to Md on `truth_acc` by more than its seed SD. If Mw ≈ Mu,
  weighting does nothing. If Md ≥ Mw, dropping is simpler and better,
  and graded weights are not justified. Either outcome retires the
  default weight table and is registered. The `weights: "shuffled"`
  control (§4) runs as a seventh arm only if F-PT5b passes.
- **Secondary, descriptive:** `kl_ref`, `entropy` trajectories, and the
  ESS log.

**Study PT6 — the post-training Atlas.** This changes the brief's design.
A Plackett–Burman screen uses two-level factors. "Objective" has four
levels, and it and "data provenance" are the *treatments* under study,
not nuisance factors to screen. So:

- **Treatments:** a full crossing of objective {rejection_sft, grpo,
  provenance_po} × reward/label source {exact, judge, mixed}. SFT is the
  shared base, not an arm. GRPO with "mixed" means a mixed reward. That
  makes 9 cells × 5 seeds.
- **Nuisance screen:** a PB12 screen *within* each objective, over its
  two-level hyperparameters (lr, β or G, KL coefficient, temperature, N,
  model size, task difficulty), run at 5 seeds (60 runs per objective).
  Its purpose is to choose the levels the treatment crossing runs at.
  Then a token-matched factorial on the survivors, as in Stage 3.
- **Licensing:** PT6 is licensed only after PT2, PT4 and PT5 report.
  Its design depends on which mechanisms survive.

## 6. The verifier interface

`include/microtorch/posttrain/verifier.hpp`, as specified in the brief,
with these decisions:

- **Every `VerifierResult` has non-empty `rationale`, `evidence_kind`
  and `evidence_uri`.** The constructor enforces it and throws on
  violation, in tests too. `reward == -1` (abstain) also requires a
  rationale.
- **Every verifier carries a version,** and `name()` returns
  `"<kind>:<version>"`. A completion's receipt is meaningful only
  against the version that issued it.
- **`LLMJudgeVerifier` does no networking.** The C++ class reads a
  verdict cache, a JSONL file keyed by sha256(prompt, completion,
  judge model id, rubric version). The cache is written by
  `tools/posttrain/judge_sidecar.py`, which makes the API calls under
  an explicit call budget. A cache miss returns an **abstain with the
  rationale "no verdict on record"**; it never fabricates a reward. A
  cached verdict without a rationale is refused. This keeps network
  code out of the trainer (D1's sidecar pattern), makes every judge
  verdict a file on disk, and makes judge-labelled runs replayable
  without the API.
- **`SyntheticJudgeVerifier`** sets `evidence_kind: "judge"` and an
  `evidence_uri` of the form `synthetic-judge:v1:<parameter hash>`. Its
  parameters are in the receipt.
- **`UnitTestVerifier` is Linux-only.** It runs the program with
  fork/exec, `setrlimit` (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC,
  RLIMIT_FSIZE), a wall-clock kill, captured stdout and stderr, and a
  scratch working directory. **Network isolation is not guaranteed in
  v1**: `unshare` needs privileges we cannot assume, and the docs say
  so. On other platforms it abstains with that rationale. It ships with
  tests (timeout, memory cap, output capture, nonzero exit, fork bomb
  held by NPROC). **No v1 study uses it**, because no paperkiln model
  can write code (§1).
- **`HumanAuditVerifier` is evaluation-only in v1.** It reads audited
  JSONL. The trainer refuses records with `evidence_kind: "human"`
  unless `train.allow_human_labels` is set: a human audit sample is too
  small to train on and too valuable to spend.

## 7. Prior art: what is taken, what is refused

**DPO** (Rafailov et al. 2023, "Direct Preference Optimization").
*Taken:* the closed-form, reference-anchored preference loss. It needs
no reward model and no sampling inside the loss, which makes it the
readable base for §4's weighting. *Refused:* treating every pair as
equally true. DPO has no notion of where a label came from, and that
gap is this charter's entry point.

**KTO** (Ethayarajh et al. 2024). *Taken:* the observation that
unpaired accept/reject signals suffice. That is what a verifier
naturally emits, and it argues for keeping per-completion records
rather than only pairs (§6 does). *Refused for v1:* the prospect-theory
value function adds two asymmetry hyperparameters whose effect would be
confounded with provenance. A KTO variant can reuse the same weights in
v2.

**ORPO** (Hong et al. 2024). *Taken:* nothing structural. *Refused:*
merging SFT and preference into one reference-free loss removes the
reference model, and with it `kl_ref`, which is one of our primary
measurements of drift. A method that cannot report its distance from
the base cannot take part in a reward-hacking study.

**GRPO** (Shao et al. 2024, the DeepSeekMath paper). *Taken:* the
group-relative advantage, no critic, verifiable rewards. *Refused:*
per-sequence length normalisation and unexamined std-normalisation,
since both are known to bias the gradient (Liu et al. 2025). Both are
kept only as spec switches whose effect can be measured, with the
analytic-bandit test (§4) as the arbiter of correctness.

**RLAIF / judge-as-reward** (Bai et al. 2022, constitutional AI; Lee et
al. 2023, RLAIF). *Taken:* the judge as one evidence source, with its
rationale recorded. *Refused:* the judge as the *only* source and as an
unlogged one. Here a judge verdict is a cached file with model id,
rubric version and rationale, and it has lower default weight than
mechanical evidence. Over-optimisation against a proxy is the
documented failure (Gao, Schulman and Hilton 2023), and
`judge_truth_gap` is how we watch for it.

**Rejection sampling, then SFT** (STaR, Zelikman et al. 2022; RAFT,
Dong et al. 2023; ReST, Gulcehre et al. 2023; Llama 2's rejection
sampling stage). *Taken:* all of it. It is the cheapest end-to-end path
and surfaces every interface problem first, so it is built before RL.
*Refused:* discarding the rejected samples and the verifier's reasons.
Every completion, accepted or not, is a receipted record (§6), so the
same samples feed DPO pairs and the audit.

*Citation discipline (D5):* the references above are author, year and
title only. They will be checked against the fetched papers before
anything outside this repository cites them.

## 8. Corpus provenance and the selection ledger

**Thesis.** Every training corpus in paperkiln must be traceable to a
documented selection decision, with the alternatives that were rejected
and the reason for each rejection recorded in the same place.

**Claim.** At solo-builder scale, corpus choice dominates architecture
choice, and neither is currently measured against a shared standard —
so this repo will measure both.

### 8.1 What the repo records

The full account is in `corpus/ledger/0000-legacy-unknown.md`:

- **One corpus is behind every research finding.** Atlas Stages 2–3, the
  sparse-attention and transfer studies, and the highway pilot all use
  `tinystories_slice_v1`: the first 5,000 TinyStories train stories,
  written on 12 Jul 2026 as a `transformer_cpp` smoke-test file. It has no
  end-of-story token and has CRLF line endings. The trainer's default
  `data.max_tokens` means the studies read only its first 400,001 tokens,
  41% of it. Its choice as a narrative corpus is on record:
  `transformer_cpp` chose TinyStories for its fit to a word-level
  vocabulary. Why the smoke-test slice became the research corpus is not
  on record.
- **The vocabulary is the first 4,096 entries of `chat7b.gguf`.** It was
  built from TinyStories-Instruct by a whitespace tokenizer that keeps
  punctuation attached. Under mtstudio's tokenizer, 19.1% of trained
  tokens map to `<unk>` (21.3% in the held-out tail), and 1,503 of the
  4,095 non-`<unk>` entries can never occur. Measured on 9 Oct 2026 for
  the transfer paper (`experiments/transfer_s2/posthoc/token_coverage.py`)
  and replicated in registry row `C0-legacy-vocab-coverage`. Why this
  vocabulary was paired with this corpus is not on record. Lanes are
  compared on identical batches, so the shared vocabulary does not by
  itself reverse a paired comparison. It does bound what the absolute
  losses mean.
- **The bytes are reproducible.** `corpus/fetchers/tinystories.py`
  rebuilds the slice from `roneneldan/TinyStories@f54c09fd`, with a
  sha256 equal to the retained historical file's (also given in the
  transfer paper's data note). `corpus/fetchers/vocab_chat7b.py` rebuilds
  the vocabulary from a committed token list. Every registry row trained
  on them carries a `corpus` field naming both.
- **TinyChat v1/v2** do have a recorded rationale: the
  `tools/get_tinychat_data.py` docstring says DailyDialog taught
  register but not answers, while the synthetic world taught both.
  They get a real ledger entry. The alternatives considered and the
  measured baseline are partly missing, and the entry says so.
- **`get_tinystories_data.py`** builds a *frequency* vocabulary from the
  corpus itself (14 Sep 2026), so the quickstart path and the research
  path use different tokenisers. This is a live confound, and the
  tokeniser factor in §8.3 exists for it.

### 8.2 Corrections to the follow-up brief, before anything is built

1. **Validation loss is not comparable across corpora.** Each corpus's
   own held-out loss measures a different distribution, so "corpus A
   reaches 3.4 and B reaches 3.9" says nothing about which is better.
   Corpus effects are measured on a **fixed evaluation panel**: the
   same held-out sets for every arm, chosen per target capability
   (narrative, dialogue, arithmetic, instruction). The headline claim
   is scoped per panel member.
2. **Nats per token are not comparable across tokenisers.** Every
   corpus comparison reports **bits per byte** of the evaluation text:
   total nats over the sequence divided by its UTF-8 byte count and by
   ln 2. Out-of-vocabulary bytes are charged through a fixed `<unk>`
   cost, declared in the pre-registration, so a tokeniser cannot win by
   mapping hard text to `<unk>`.
3. **Comparing with Stage 2 needs the same unit.** Stage 2's
   architecture effects are in nats per token on one tokeniser. They
   convert to bits per byte with that tokeniser's tokens-per-byte on
   the same text, and the conversion is published with its inputs.
4. **"The same corpus with the vocabulary shuffled" is not a negative
   control.** Permuting token ids is an isomorphism for a model trained
   from scratch: the model learns the same thing up to its random
   initialisation. The meaningful bad baselines are **word order
   shuffled within each document** (keeps unigrams, destroys syntax)
   and **documents sampled from a unigram model of the corpus**. Both
   go in the registry.
5. **A corpus is only a fair arm if it is plausible for the target.**
   Training on dialogue and testing on stories is trivially bad and
   proves nothing about selection. For each panel member the
   pre-registration lists which corpora are candidate arms.

### 8.3 The corpus study (C1), outline for its own pre-registration

- **Factors:** corpus (§9.4 candidates plus the two shuffled
  baselines) × token budget {matched, 4× matched} × tokeniser {the
  corpus's own frequency vocabulary at 4,096, a shared fixed
  vocabulary} × model size {d=128 L=2, d=256 L=2}. 5 seeds per cell.
- **Primary outcome:** bits per byte on each panel member.
- **Headline:** the spread of bits per byte across plausible corpora,
  at fixed budget, tokeniser and size (range and SD over corpora, each
  with seed SE), compared with the largest Stage 2 architecture effect
  in the same unit. Whichever is larger is the finding. If the corpus
  spread is inside seed noise, that is the finding, and it argues
  against spending more time on data at this scale.
- **Size:** about 6 corpora × 2 × 2 × 2 × 5 = 240 runs. A measured pilot
  fixes the GPU-hour estimate before the owner approves budget (as in
  AGENTS.md §2).
- **Order:** after post-training Phase 3, as the follow-up specifies.

## 9. Corpus infrastructure (Phases A–H), decisions only

- **`corpus/registry.jsonl`** has the follow-up's fields, plus
  `fetcher` (path), `revision` (pinned dataset commit or content hash),
  `unk_rate_at_cap` and `tokens_per_byte`. Status adds **`candidate`**
  to active/deprecated/retired: a corpus with a card but no measured
  effect is a candidate, not active. `sha256` is computed by the fetcher
  from the bytes it wrote, never typed in.
- **Ledger:** `corpus/ledger/NNNN-<id>.md`, answering the follow-up's
  seven questions. Question 7, the baseline the next corpus must beat,
  is a number in bits per byte on a named panel member, written before
  training. `0000-legacy-unknown.md` holds §8.1.
- **Cards:** `corpus/cards/<id>.md`. The "measured effect" section
  reads `PENDING — not a card yet` until C1 reports.
- **Fetchers:** `corpus/fetchers/<id>.py`, behind one `make corpus`
  target. Each pins a revision, writes to a deterministic path, is
  idempotent, verifies its sha256 and exits non-zero on a mismatch. The
  existing `get_tinystories_data.py` and `get_tinychat_data.py` become
  thin callers of the fetchers, so earlier results still reproduce.
  Licences are read off the pinned source at fetch time and copied into
  the card. They are not taken from this charter.
- **Candidates (v1):**

  | Candidate | Role |
  |---|---|
  | TinyStories | existing, narrative |
  | TinyChat v2 | existing, dialogue |
  | **`tinyarith`** | ours: a seeded synthetic generator of arithmetic word problems in the TinyStories world, with numbers as single in-vocabulary tokens. The verifiable corpus for PT2–PT5. |
  | **TinyStories-Instruct** | proposed: the instruction-conditioned variant in the same Hugging Face dataset, if the pinned revision carries it. Instruction following inside the same world and vocabulary, which is the only kind a 4,096-word model can learn. |
  | **Simple English Wikipedia** | proposed: real expository text with a different register. It tests whether the narrative findings are a TinyStories artefact; its coverage under a 4,096 vocabulary is the measured price. |
  | the two shuffled baselines | §8.2 item 4 |
  | **a code corpus** | registered as `retired-at-scale`, with a ledger entry: a word-level 4,096 vocabulary cannot represent code, so the `UnitTestVerifier` is tested on fixtures, not trained against |

- **Evaluation sets (`corpus/eval/`):**
  - `tinyarith_eval_v1` (verifiable)
  - `tinyarith_hackprobe_v1`: prompts on which "the answer is <wrong
    number>" passes the format verifier, so a format-gamer scores high
    and truth scores low. The negative control for PT4.
  - `format_v1`
  - `tinystories_heldout_v1`
  - `instruct_heldout_v1`

  Each set is frozen at first use, has a sha256 in its card, and is
  versioned on any change (v1 is kept). Each card carries a measured
  n-gram contamination line against every training corpus.
- **Quality measurements (Phase H):** exact duplicate rate; MinHash
  near-duplicate rate over word 5-gram shingles (about 100 lines, no
  dependency, with a test against a hand-built duplicate set); length
  percentiles; n-gram perplexity (Kneser–Ney trigram trained on one
  half, scored on the other); tokeniser coverage; eval n-gram overlap.
  Each one is a function with a unit test, in `tools/corpus_quality.py`.

## 10. Registry entries this charter creates

These are added to `atlas/findings.jsonl` with `status: "pending"`
before anything runs. Each one names this charter as its receipt until
its own study directory exists.

| id | Pending claim |
|---|---|
| PT2-filter-provenance | Judge-filtered rejection-SFT shows a larger `judge_truth_gap` than noise-matched and exact-filtered SFT |
| PT4-hack-monitor | `hack_frac` detects an open format-reward channel and stays low when the channel is closed |
| PT5-judge-vs-noise | The provenance thesis: judge-ordered pairs produce more reward hacking than noise-matched pairs at equal N |
| PT5-provenance-weighting | Provenance-weighted DPO beats unweighted mixing on `hack_frac` without losing to judge-dropping on `truth_acc` |
| C0-legacy-vocab-coverage | Measurement: `<unk>` rate and tokens per byte of the chat7b cap-4096 vocabulary on the S1/T-series TinyStories slice |
| C1-corpus-vs-architecture | At fixed budget, tokeniser and size, the spread over plausible corpora in bits per byte is larger (or smaller) than the largest Stage 2 architecture effect |
| C1-tokeniser-confound | At a fixed corpus, the choice of vocabulary moves bits per byte on the panel |

## 11. Order of work after approval

1. **Corpus Phases A, B and F:** registry, ledger (legacy entry first),
   fetchers with hashes, and the C0 measurement.
2. **Post-training Phase 1:** the verifier interface and its tests.
   **Corpus Phase E:** freeze the evaluation sets, including
   `tinyarith`.
3. **Phase 2:** the C++ sampler (an mtstudio subcommand using the same
   `next()` sampler as chat, so training-time and sampling-time
   policies are the same code; ember.cpp is not used for policy
   samples), the receipted JSONL, the audit sheet and its round trip.
   Then the PT2 licence and run.
4. **Phase 3:** SFT `loss_mask` and `rejection_sft`. **Corpus Phase C
   and H:** cards and quality measurements.
5. **Phase 4:** GRPO and the monitor. PT4 licence and run.
6. **Phase 5:** `provenance_po`. PT5 licence and run.
7. **Corpus Phase D (C1), then PT6.**

Every step: tests first; the event stream extended with new keys only
(D1); a registry row whatever the result.

## 12. Approval and decisions (16 Oct 2026)

The owner approved this charter and delegated the open judgement calls.
They are settled as follows.

1. **Weight table.** The §4 defaults are frozen as written (exact match,
   unit test and human 1.0; format 0.5; judge 0.3; unattributed 0.0).
   Sensitivity to the table is a separate, pre-registered sweep, and only
   if F-PT5b passes.
2. **Audit status.** `audited` changes the weight of `judge` evidence only,
   as §4 states. Mechanical evidence is reproducible, so re-running the
   verifier is its audit.
3. **The real-judge calibration arm is kept.** It runs once, on the frozen
   audit set, through `judge_sidecar.py`:
   - under a hard cap of 600 calls;
   - with credentials only from an explicitly designated environment
     file, never by searching the machine for keys;
   - with the model id recorded in every verdict's `evidence_uri`.

   If no designated key is available when PT2 is licensed, the §3.2
   fallback grid {0.10, 0.25} applies, and the scope says so.
4. **Compute.** All GPU work runs on Colab, where the owner has given free
   rein, through the paths in `AGENTS.md` §3. The owner's own GPU is not
   used. Local CPU runs only short tests and runs of the smallest models
   (d=128). The C1 pilot still measures GPU hours before C1's
   pre-registration, so the receipt states the cost. That measurement is
   for the record, not for approval.
5. **Corpus registry statuses** add `candidate` and `retired-at-scale` to
   active, deprecated and retired.
6. **Step 1 of §11 is done.**
   - `corpus/` holds the registry (5 corpora, 1 vocabulary, 6
     candidates), ledgers 0000–0002, pinned fetchers behind `make
     corpus`, hashes, and rendered placeholder cards.
   - `tools/corpus_quality.py` has 13 tests.
   - `C0-legacy-vocab-coverage` is supported.
   - The research corpus is rebuilt byte-exact from the public revision.

## 13. Calibration result (17 Oct 2026)

`experiments/posttrain_calib/` (licence `ce5d2f7`; execution clarification
1 recorded there) chose **tinyarith R = 20**. Its mean dev `truth_acc` over
three seeds is 0.386 (0.397, 0.384, 0.375; SD 0.011), inside [0.25, 0.60].
- R = 10 falls below the band and is erratic across seeds (0.174, 0.021,
  0.354).
- R = 50 falls below the band (0.177).

Every confirmatory post-training study uses R = 20 unless its
pre-registration says otherwise.
