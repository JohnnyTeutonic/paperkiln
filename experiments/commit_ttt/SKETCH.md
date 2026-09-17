# commit_ttt: atomic-commit fast-weight edits. Scoping, mechanism, pre-registration sketch

*Sketch, 17 September 2026. Not a pre-registration and not a licence anchor.
Nothing here has been run. The purpose is to decide whether the idea is
worth a pre-registration, and if so to fix the shape of one before any data
exists. House rules apply: journals only (TMLR class), no arXiv, twelve
seeds, matched FLOPs and parameters, about 40 L4 GPU-hours for the whole
study, strength over speed.*

## 0. The idea in one paragraph

A language model adapts at test time by editing a small set of fast weights.
Every k tokens a candidate edit dW is computed from the tokens just read and
held in a staging buffer, not applied. The next m tokens arrive and are
predicted with the live weights as usual. Once they have arrived, the same m
tokens are scored under the frozen weights W0 and under the staged weights
W_live + dW. The edit is committed into the live forward pass only if the
staged loss on those m tokens beats the comparison loss by a margin;
otherwise it is discarded and the buffer is reset. Two things follow by
construction: the live weights are always W0 plus a sum of edits that each
passed a held-out test, and W0 is recoverable bit-exactly at any moment. The
question the study asks is whether that construction buys anything an
always-commit TTT, a periodic reset, or a two-model mixture does not, at the
same FLOPs.

## 1. Literature scoping by hand

Searches were run on 17 September 2026 with WebSearch, WebFetch on arxiv.org
abstract and HTML pages, the Semantic Scholar graph API (forward citations of
TTT-E2E and RW-TTT), the OpenReview notes search API, and one Google Scholar
result page (which loaded). Vocabularies: (a) test-time training with
validation, accept/reject, rollback, safe adaptation; (b) fast weights with
gating or verification; (c) online learning with never-worse-than-baseline
guarantees (safe policy improvement, conservative updates, hedging, expert
mixtures); (d) speculative and transactional framings of weight updates.
The forward-citation list for TTT-E2E on Semantic Scholar had 42 entries; RW-TTT
had none. The Niu et al. survey of test-time intelligence (2609.01679, 2
September 2026) was read for any method that verifies a parameter update
against a held-out loss before committing it; its only relevant sentence is
that full-parameter updates "require reliable feedback and rollback controls",
and it names no such method.

One correction to the brief: the DeltaBox paper is arXiv 2605.22781, not
2605.22791 (that id is Gated DeltaNet-2, Hatamizadeh et al., 21 May 2026, a
linear-attention erase/write gate paper, and is also distinct).

### 1.1 The table

| # | Paper | Year | Quoted sentence | Verdict and reason |
|---|---|---|---|---|
| 1 | Sun et al., Learning to (Learn at Test Time): RNNs with Expressive Hidden States, arXiv 2407.04620 | 2024 | "The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning." | Distinct. Every inner step is applied. The inner learning rate is input-dependent and learned, which is a soft, unconditional gate, not a test. |
| 2 | Tandon et al., End-to-End Test-Time Training for Long Context, arXiv 2512.23675 | 2025 | "our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights." | Distinct. Unconditional commit, meta-learned initialisation. It is the strongest named comparator for the always-commit arm. |
| 3 | Feng et al., In-Place Test-Time Training, arXiv 2604.06169 | 2026 | "In-Place TTT treats the final projection matrix of the ubiquitous MLP blocks as its adaptable fast weights, enabling a 'drop-in' enhancement for LLMs without costly retraining from scratch." | Distinct, but it fixes our Rule A: the fast weight is the FFN down-projection and the write is chunk-wise. Their write is unconditional; ours is the same write behind a test. |
| 4 | Yang et al., RW-TTT: Batched Serving for Request-Owned Test-Time Training State, arXiv 2605.28053 | 2026 | "RW-TTT ... tags each decode step with its owner, version, and READ/WRITE effect, batches only compatible phases, and commits updates only to the owner." | ADJACENT. Commit, version and rollback exist as serving primitives; the commit condition is "the WRITE group succeeded", never "the edit improved held-out loss". No quality gate. No forward citations yet. |
| 5 | He et al., DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback, arXiv 2605.22781 | 2026 | "a sandbox should only duplicate the changes between consecutive checkpoints." | Distinct. Transactional rollback of filesystem and process state, not weights. Only the vocabulary overlaps. |
| 6 | Trust-Gated Fast-Weight Updates for TTT-E2E LLMs, GitHub (Deep-Learning-130), public since 1 August 2026, no paper | 2026 | "The Phase 1 kill-gate: does a benign-looking stream actually corrupt fast weights? has not yet been run." | ADJACENT, closest in mechanism. A gate intercepts `inner_loop_step`, checks "anchor consistency" against a frozen reference and a cumulative drift budget, and rolls back to a ring-buffered checkpoint in O(1). Framed as a poisoning defence. The check is consistency with the anchor, not held-out loss against the anchor; no results; scaffold only. It shows someone else has the rollback plumbing in mind, and it is not a quality claim. |
| 7 | Xie, SoftModel: A Neural Model That Grows Its Own Topology, arXiv 2608.16409 | 2026 | "all training and growth happen in a speculative working state, and promotion to the committed version, the one that serves, requires strictly surpassing the incumbent's score on the most recent slice of a held-out stream" | ADJACENT, closest in principle. This is our acceptance rule stated for a different object: structural growth proposals in a continual-learning system, not per-chunk fast-weight edits in a language model. Gradient steps happen in the working state without per-step adjudication. No LM or transformer fast-weight experiment, no matched-FLOPs baselines visible, single author. We should cite it as the nearest statement of the rule. |
| 8 | Press et al., RDumb, arXiv 2306.05401 (NeurIPS 2023); Lim et al., When and Where to Reset Matters for Long-Term Test-Time Adaptation, arXiv 2603.03796 (ICLR 2026) | 2023, 2026 | RDumb: "eventually all but one state-of-the-art methods collapse and perform worse than a non-adapting model ... a simple baseline, 'RDumb', that periodically resets the model to its pretrained state ... performs better or on par with the previously proposed state-of-the-art." | ADJACENT. Reset-to-pretrained is the existing safety device for continual TTA and must be a baseline arm here. Neither paper tests each update before applying it. Vision classification, not LM. |
| 9 | Behrouz et al., Titans: Learning to Memorize at Test Time, arXiv 2501.00663 (NeurIPS 2025) | 2025 | Surprise is "its gradient with respect to the input, where larger gradients indicate data different from past data", with "a data-dependent forget gate". | Distinct. Learned write and forget gates, applied every step; no held-out test, no recoverability claim. |
| 10 | Chaudhary, Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity, arXiv 2510.21908 | 2025 | "Hebbian plasticity is sharply gated around salient events." | Distinct. Neuromodulated (learned) Hebbian gate on small tasks; the gate is a learned scalar, not a verification. Useful for the Hebbian rule form. |
| 11 | Moradi et al., VDS-TTT, arXiv 2505.19475; Zhu et al., Self-Guided TTT, arXiv 2607.09415; Yuan et al., EASE-TTT, arXiv 2606.06906 | 2025, 2026 | S-TTT: "before adaptation, the model identifies the evidence spans it should learn from, and the standard language-modeling training objective is applied only to those selected spans." | ADJACENT family: input-side selection (which tokens or samples to train on), not output-side acceptance (whether to keep the resulting edit). The two are complementary and can be combined; neither of these tests the edit. |
| 12 | Anonymous, Rollback-Safe Long-Horizon Generation: Auditing and Transactional State Commit for Narrative Agents, OpenReview, TMLR under review | 2026 | "text, extracted events, and state deltas are staged, audited, gated, and then either committed atomically or preserved as a rejected trace." | ADJACENT framing, distinct object (agent memory state, not weights). It tells us TMLR is currently seeing the staged/audited/atomic-commit vocabulary, so our framing will read as familiar rather than novel; the novelty must sit in the object and the measurement. |
| 13 | Song et al., Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training, arXiv 2607.00368 | 2026 | The paper uses "future-token loss" as a proxy metric and finds "LoRA updates improve loss metrics while generated recall remains zero." | ADJACENT. The quantity we gate on (loss on tokens after the update window) is their post-hoc evaluation proxy. They do not gate on it. Their finding is also a threat to us: loss gains may not translate to behaviour; we claim only loss. |
| 14 | Laroche et al., Safe Policy Improvement with Baseline Bootstrapping, arXiv 1712.06924 (ICML 2019) | 2019 | SPI trains a policy "guaranteed to perform at least as well as the baseline policy used to collect the data." | Distinct field, the conceptual ancestor. The frozen model is our baseline policy; our guarantee is per verification window and empirical, theirs is a PAC bound. Cite for lineage only. |
| 15 | Kumar et al., RG-TTA: Regime-Guided Meta-Control for TTA in Streaming Time Series, arXiv 2603.27814 | 2026 | The controller "gates checkpoint reuse from a regime memory, loading stored specialist models only when they demonstrably outperform the current model (loss improvement >= 30%)." | ADJACENT. A loss-improvement threshold gates which stored checkpoint serves; it does not gate each fresh update, and the domain is forecasting. |
| 16 | Zhao et al., On Pitfalls of Test-Time Adaptation (ICML 2023); Realistic Evaluation of TTA: Unsupervised Hyperparameter Selection, arXiv 2407.14231 | 2023, 2024 | "selecting appropriate hyper-parameters, especially for model selection, is exceedingly difficult due to online batch dependency." | Distinct, but it names the reason our gate is possible in LMs and not in vision TTA: next-token loss on the held-out window is a label-free, exact, online validation signal. Vision TTA has no such signal. |

Not in the table, checked and distinct: LaCT (Test-Time Training Done Right, 2505.23884), TTT-NTP (2606.21803), Reinforced Fast Weights (2602.16704), Fast Weight Attention for Continual Learning (2608.27763), Elastic TTT (2604.07350, a Fisher prior that shapes the update rather than deciding it), Memoir (2607.20792), Learning What to Remember (2608.01672), OLOR weight rollback (2401.10962, a regulariser named rollback), EVA-0 (2605.18867), Test-Time Learning for LLMs (2505.20633, perplexity-weighted sample selection), Fast-weight Product Key Memory (2601.00671, learned gate between episodic and static). None tests an edit on held-out tokens before applying it.

### 1.2 The theoretical foil that must be a baseline

Prediction with expert advice under log-loss gives a guarantee stronger than
ours for free: a Bayesian mixture over two predictors, the frozen model and
an always-commit TTT model, with weights proportional to the product of past
likelihoods, has cumulative log-loss at most that of the better predictor
plus ln 2 nats over the whole stream (Cesa-Bianchi and Lugosi 2006, the
mixture bound). It never touches weights, needs no margin, and its
guarantee is cumulative rather than per window. Its cost is one extra
forward per token forever, and it does not produce a single deployable set
of weights. This is the arm the commit-gated method has to justify itself
against, and the sketch treats it as a primary baseline, not an ablation.
No paper found applies this mixture to TTT of language models as a safety
baseline; that absence is itself a small contribution if we report it.

### 1.3 Verdict

The idea is still OPEN in its exact form. Nothing found evaluates a
staged fast-weight edit of a language model on held-out tokens against the
frozen model and commits or discards it on that comparison, with exact
recoverability, and reports the result against always-commit TTT at matched
FLOPs, a periodic-reset baseline and a two-expert mixture. The nearest prior
statements of the rule are SoftModel (held-out reality gate, for structural
growth) and the trust-gated TTT-E2E repository (anchor-consistency gate, for
poisoning defence, unrun). The transactional vocabulary is TAKEN by RW-TTT,
DeltaBox and the TMLR narrative-agent submission and should be used
sparingly; "commit" and "rollback" are fine as verbs, "transactional" as a
title word is not.

The claim that survives, and the only one worth a pre-registration: a
held-out-window commit test on fast-weight edits gives a language model
adaptation whose worst-seed prequential regression against the frozen model
is zero within noise on a stream designed to make naive TTT regress, while
retaining a stated fraction of naive TTT's gain where TTT helps, at an
amortised inference cost below that of the Bayes mixture. The mechanism's
existence is not the contribution; the measurement of when the gate pays,
and against which cheap alternative, is.

## 2. Mechanism specification

### 2.1 Notation

W0: frozen pretrained weights. A: the accumulator of committed edits for
the adapted matrices (same shape as those matrices, initialised to zero).
W_live = W0 + A. dW: the staged candidate edit. U_j = tokens [t_j, t_j + k):
the j-th update window. V_j = tokens [t_j + k, t_j + k + m): the j-th
verification window. T: the model's attention context length. L(W; V | c):
mean next-token cross-entropy in nats over the tokens of V under weights W,
teacher-forced, with preceding context c.

Verification windows are held out from the update they judge: V_j is
disjoint from U_j by construction. V_j may and does become part of U_{j+1}.
The token is held out relative to the edit it scores, not from the stream.

### 2.2 Adaptation rules

Rule A, outer-product write on the FFN down-projection (the In-Place TTT
object). For the top n_A layers (n_A = 1 primary, 2 as ablation), the fast
matrix is W_down (d_ff by d). The candidate edit is one gradient step of the
next-token loss over U_j restricted to W_down:

    dW = -eta_A * sum_{t in U_j} g_t h_t^T

where h_t is the FFN hidden activation at position t and g_t is the gradient
of the token-t loss with respect to the W_down output. This is a sum of k
rank-one outer products, so "Hebbian outer product" and "one gradient step on
this matrix" are the same object here; the backward pass is only through the
layers above the adapted one, which for the top layer is the final norm and
the unembedding. Rank of dW is at most k.

Rule B, low-rank gradient TTT on a LoRA adapter. LoRA rank r = 8 on W_q,
W_v and W_down in every layer, B initialised to zero so the adapter is
exactly inert at the start. Candidate edit: s plain SGD steps (no momentum,
no Adam state, so that discard has no hidden state to reset) on the
next-token loss over U_j with learning rate eta_B; s = 1 primary. Staged
parameters are a copy of the live adapter after the steps; commit copies
staged to live; discard drops the copy.

Rule A is the primary rule because it has no optimiser state and its
verification is exact against a dense accumulator; Rule B is the second rule
because it is what most current LLM TTT practice uses.

### 2.3 The staging and verification loop

At chunk boundary j (every k tokens, k a multiple of the eval granularity):

1. Compute dW from U_j using W_live (Rule A or B). Hold it in `stage`.
2. Continue prequential prediction of the next m tokens V_j with W_live. The
   per-token losses of W_live on V_j, call them l_live, are a by-product of
   this prediction and cost nothing extra.
3. When V_j has fully arrived, compute per-token losses on V_j under W0
   (l_frozen) and under W_live + dW (l_staged), both with the same preceding
   context as the live prediction used.
4. Decide, in this order:
   - rollback: if mean(l_live - l_frozen) > delta, set A = 0 (live returns to
     frozen), and discard `stage`. The live model had drifted below frozen
     on the current window.
   - commit: else if mean(l_staged - l_live) < -delta, set A = A + dW.
   - discard: otherwise leave A unchanged.
5. Reset `stage`. Advance t_j.

The rollback branch is what makes "never worse than frozen on the
verification window" a property of the live weights and not only of the
edit. Without it the invariant holds for each edit at the time it was
committed but says nothing about a stale edit later. With it, after every
decision the live weights are within delta of the best of {frozen, previous
live, staged} on V_j.

The margin. Primary: a paired signal-to-noise rule, commit iff
mean(d) < -z * sd(d) / sqrt(m) with d = l_staged - l_live over the m tokens,
z fixed at 1.0. Robustness: a fixed margin delta in nats per token from
{0, 0.005, 0.01, 0.02}. The margin, k, m, eta and z are all fixed on pilot
seeds (Section 3.7) and never on panel seeds.

Commit granularity. Global (all adapted matrices as one edit) is primary.
Per-layer commit for Rule A with n_A = 2 is an ablation: it costs one extra
staged forward per layer, so it is affordable only at n_A = 2.

### 2.4 What "the same preceding context" means, and the two verification variants

The loss on V_j depends on the tokens before it. Two variants are defined and
both are run on the nano testbed, where compute is not the constraint:

- exact: every candidate (frozen, staged) is evaluated by a fresh forward
  over the full window of T tokens ending at the end of V_j, reading the loss
  on the last m positions. Cost: 2T token-forwards per decision. This is the
  primary variant for the licensed claim because it is the one the lemma
  below is about.
- cached: the candidate is evaluated over the m tokens of V_j only, attending
  to the live model's existing KV cache for the preceding context (a prefix
  computed under other weights). Cost: 2m token-forwards per decision. This
  is the deployable approximation; TTT-E2E-style methods already accept
  stale-prefix KV. It is reported as a secondary arm, and the difference in
  commit decisions between the two variants is itself a reported quantity.

### 2.5 Cost accounting (in token-forward equivalents, one backward counted as two forwards)

Per k tokens of stream, with the primary settings k = 256, m = 64, T = 256:

| arm | update | verification | total per token |
|---|---|---|---|
| frozen | 0 | 0 | 1.00 |
| always-commit, Rule A, 1 step, top layer | fwd already done; bwd through top layer only, about 0.3 k | 0 | about 1.3 |
| always-commit, Rule B, 1 step, all layers | 2k (full bwd) | 0 | 3.00 |
| commit-gated, Rule A, exact | 0.3 k | 2T | 1.3 + 2T/k = 3.3 |
| commit-gated, Rule A, cached | 0.3 k | 2m | 1.3 + 2m/k = 1.8 |
| commit-gated, Rule B, exact | 2k | 2T | 5.00 |
| Bayes mixture (frozen + always-commit B) | 2k | 0, but a frozen forward every token | 4.00 |
| periodic reset (RDumb analogue, reset every R chunks) | as always-commit | 0 | as always-commit |

The point the table makes before any data: with exact verification at
k = 256 the gate is not cheaper than the Bayes mixture for Rule B, and only
cheaper for Rule A. With cached verification it is cheaper for both. So the
cost argument for the gate depends on the cached variant being faithful,
which is why the agreement between exact and cached decisions is a
pre-registered secondary quantity. If k is reduced to 64 the exact variant
costs 2T/k = 8 extra forwards per token and the cost argument is lost
entirely; k = 64 is therefore an ablation, not a primary setting.

Matched-FLOPs always-commit. The gated arm spends 2T (exact) or 2m (cached)
extra token-forwards per chunk on verification. The matched always-commit
arm spends the same budget on more update steps instead: for Rule B exact,
s_matched = 1 + 2T / (2k) = 2 steps at k = T; for Rule A exact,
s_matched = 1 + 2T / (0.3 k), which at k = T = 256 is about 7 steps, capped at
4 with the remainder disclosed as unspent. Both the natural (1-step) and the
matched always-commit arms are run; the matched one is the licensed
comparator.

### 2.6 The exactness guarantee

Lemma 1 (exact recoverability). Let W_live = W0 + A with A stored separately
from W0 and W0 never written after load. Then for any sequence of commit,
discard and rollback decisions, setting A = 0 restores the frozen forward
pass bit-for-bit, and the forward under W_live at any time equals the
forward under W0 + A for the stored A. Proof: W0 is never mutated; the
forward reads W0 + A, which is computed fresh; A = 0 gives W0 exactly, up to
the floating-point addition W0 + 0 = W0, which is exact. For Rule B, A is
the LoRA product and B = 0 gives the zero matrix exactly.

Lemma 2 (window guarantee, exact variant). Let the decision at chunk j be
made by the rule in 2.3 with margin delta >= 0 and exact verification. Then
immediately after the decision, L(W_live; V_j | c_j) <= L(W0; V_j | c_j) +
delta, and if a commit occurred, L(W_live; V_j | c_j) <= L(W_live_before;
V_j | c_j) - delta. Proof: if the rollback branch fired, W_live = W0 and the
first inequality holds with equality. Otherwise mean(l_live - l_frozen) <=
delta, and a commit only replaces W_live by a candidate with a strictly
lower mean loss on V_j by at least delta, so both inequalities hold.

What Lemma 2 does not say, and the paper must say plainly: the guarantee is
teacher-forced, on the window used for the decision, at the moment of the
decision. Tokens of V_j were predicted by the previous live weights; tokens
after V_j will be predicted by weights whose test was V_j. The prequential
loss, which is the quantity that matters to a user, is not covered by the
lemma. Whether the window guarantee translates into a prequential
guarantee-in-practice is exactly the empirical question, and it is why
the primary hypothesis is empirical and has a numeric threshold.

## 3. Falsifiers and pre-registration sketch

Written in the style of experiments/transfer_s2/PREREGISTRATION.md: every
clause below would be fixed before a panel run, the analysis script frozen at
the anchor commit, and any change after the anchor recorded as a dated
amendment. The RESULTS_STAGE2.md lesson is applied throughout: a selection
rule must never select on the data it judges. Here that means (i) all
hyperparameters are fixed on pilot seeds that never enter the panel, (ii) the
primary metric is prequential, so a commit decision cannot influence the
loss recorded for the tokens it was made on, and (iii) the segment labels of
the synthetic stream are generator-side ground truth, never inferred from
model behaviour.

### 3.1 Metrics (all in nats per token, per seed, per stream)

- M1, prequential loss delta vs frozen: mean over the stream of
  (loss under the arm's live weights at prediction time) minus (loss under
  W0), same tokens, same context.
- M2, worst-seed regression: the maximum over twelve seeds of M1 on the
  mixed stream. This is the number the safety claim is about.
- M3, commit fraction: commits / decisions; also rollback fraction.
- M4, verification-window loss vs update-window loss: for each committed
  edit, mean loss under the staged weights on V_j (held out) and on U_j
  (the edit's own training tokens). The gap is a direct measure of how much
  a naive TTT step overfits its own chunk, and it is what the gate is
  filtering on.
- M5, gain retained: on the helpful segments, (frozen - gated) / (frozen -
  always-commit-matched), per seed.
- M6, exact-vs-cached decision agreement: fraction of chunks where the two
  verification variants make the same decision.
- M7, amortised token-forward equivalents per token, counted, not estimated.

### 3.2 Arms (matched parameters throughout: the same W0 per seed; the same adapted matrices and LoRA rank across arms)

frozen; always-commit-1 (natural); always-commit-matched (Section 2.5);
commit-gated-exact (primary); commit-gated-cached; periodic-reset (always-
commit-1 with A = 0 every R chunks, R fixed on pilot); Bayes-mix (frozen and
always-commit-1, log-loss mixture weights with a uniform prior). Each arm
runs under Rule A (primary) and Rule B.

### 3.3 Streams

S-mixed, the stream the guarantee has to bite on. Generated per seed from
TinyStories-format text with three kinds of segments interleaved at random
in fixed proportions (segment labels recorded by the generator):

- helpful segments (about 50 percent of tokens): a synthetic in-context
  key-value mapping. A block of about 2000 tokens in which a set of 16 novel
  pseudo-word tokens is bound to existing concrete nouns by short definition
  sentences at the top and then used consistently in TinyStories-style
  sentences. The frozen model cannot know the bindings; a model that learns
  them from the first few hundred tokens predicts the rest better. This is
  the segment where adaptation should pay and the gate should commit.
- adversarial segments (about 25 percent): bursts of about 512 tokens
  where naive TTT is expected to overfit: (i) a single sentence repeated
  verbatim 20 times, then normal text resumes; (ii) tokens drawn uniformly
  at random from the vocabulary; (iii) a helpful-style block whose bindings
  are silently re-randomised every 200 tokens, so that what was learned is
  stale by the time it is used. Segment (i) is the repetition trap; (ii) is
  the noise trap; (iii) is the staleness trap and is the one aimed at the
  gate itself, since a verification window inside (iii) can pass an edit
  that the next window will punish.
- neutral segments (about 25 percent): held-out TinyStories text.

S-id, distribution-shift control, in-distribution: 128K tokens of held-out
TinyStories. Prediction: adaptation helps little, the gate mostly discards,
M1 for the gated arm is within noise of zero. If the gate commits often here
and M1 goes positive, the gate is passing noise.

S-shift, distribution-shift control, out-of-distribution: 128K tokens of a
different children's-text corpus or simple encyclopaedic prose (fixed at
anchor time). Prediction: adaptation helps throughout, the gate commits
often, and gated retains most of always-commit's gain. If the gate blocks
adaptation that always-commit shows to be real, it is too conservative.

Stream length: 128K tokens each; 12 seeds; stream seed and model seed tied
(seed s trains W0_s and draws the streams for seed s).

### 3.4 Primary hypothesis and falsifier

Manipulation check, fixed now (the analogue of transfer_s2's reachability
condition). The trap must fire: on S-mixed, always-commit-1 must show
M1 > +0.02 nats per token on the adversarial segments in at least 9 of 12
seeds. If it does not, the stream does not test the gate, the panel is
reported as a protocol failure, and no claim about the gate is made. The
pilot's job is to find an eta at which the trap fires without making
always-commit useless on the helpful segments; eta is then frozen.

H1 (safety, primary). On S-mixed, commit-gated-exact has
M2 <= +0.005 nats per token, that is, the worst of twelve seeds regresses
against frozen by no more than 0.005 over the whole stream, while
always-commit-matched has M2 > +0.02.

Falsifier, fixed now. If any seed shows the gated arm more than 0.005 nats
per token worse than frozen on S-mixed, H1 is falsified: the window
guarantee does not carry to prequential loss at this k and m, and the paper
becomes the receipts-backed negative "held-out verification of fast-weight
edits does not protect prequential loss on adversarial streams", with M4
and the staleness-trap breakdown as the mechanism. Both outcomes are
bankable; neither is chosen in advance.

H2 (benefit, secondary). On the helpful segments of S-mixed and on S-shift,
gated retains at least 50 percent of always-commit-matched's gain over
frozen (M5 >= 0.5) in at least 9 of 12 seeds. Falsifier: M5 < 0.5 in 4 or
more seeds means the gate is too conservative to be useful, and the paper
reports the safety-versus-benefit trade-off as its result, with the margin
sweep as the descriptive picture.

H3 (against the cheap alternative, secondary). On S-mixed, gated-cached is
within 0.01 nats per token of Bayes-mix on M1 and costs no more than half
its counted token-forwards per token (M7). Falsifier: if Bayes-mix beats
gated by more than 0.01 at any cost, or if the cost ratio is not met, then
the mixture is the better safety device and the paper says so; the gate's
remaining claim is deployability as a single weight set, which is an
engineering property and is not what the paper would be about.

H4 (against reset). Gated beats periodic-reset on M1 over S-mixed in at
least 9 of 12 seeds. Falsifier: reset is as good, and RDumb's lesson
transfers to LM TTT; the paper reports it.

Headline claimable only if the manipulation check passes and H1 and H2 both
clear. H3 and H4 shape the discussion; they cannot rescue H1.

Predictions written before running. H1 holds for Rule A at k = 256, m = 64;
H1 fails for Rule B at the same settings because a full-network LoRA step
on a 512-token repetition burst moves further than the top-layer outer
product and one window cannot catch the staleness trap; M5 lands near 0.6
to 0.8; Bayes-mix is within 0.01 of gated on M1 and H3 is decided by cost;
periodic reset is clearly worse on helpful segments because it throws away
bindings mid-block. The staleness trap (iii) is where the gated arm's
regressions, if any, will be.

### 3.5 Pilot and selection rule (calibration, not evidence)

Pilot seeds 0 to 2, never in the panel. Grid: eta_A in {3 values}, eta_B in
{3 values}, k in {128, 256}, m in {32, 64}, z in {0.5, 1.0, 2.0}, R for reset
in {8, 32} chunks. Selection rule, fixed now: eta is the largest rate at
which the manipulation check fires in 3 of 3 pilot seeds; (k, m, z) is the
tuple with the lowest median pilot M1 on S-mixed among tuples whose pilot M2
<= 0.005; ties go to the larger k (cheaper). If no tuple meets M2 <= 0.005 on
the pilot, the panel still runs at the tuple with the lowest pilot M2, and
this is disclosed, since the transfer_s2 Amendment 1 lesson is that a rule
is not rewritten after seeing what it chose.

### 3.6 Fallback clauses, fixed now

- If pretraining at the nano size yields a W0 whose always-commit gain on
  helpful segments is below 0.02 nats (TTT inert at this scale), the study
  moves to the 23M size before the panel and says so; if inert there too,
  the study stops and reports "TTT inert below 25M on this stream", which
  licenses nothing about the gate.
- If exact and cached decisions agree on fewer than 90 percent of chunks
  (M6), the cached variant is reported as a distinct method, not an
  approximation, and H3 is evaluated on the exact variant's cost, which will
  likely fail it.
- If the Bayes-mix arm cannot be run at matched parameters for a
  technical reason, H3 is dropped and its absence is a stated limitation,
  not replaced by a weaker foil.

### 3.7 Threats to validity

- The stream is synthetic and designed to make the gate bite. That is the
  point of S-mixed and is disclosed; S-id and S-shift are the natural-text
  controls, and the paper claims nothing about natural adversarial streams
  it did not run.
- Loss is not behaviour (Song et al. 2607.00368). We claim next-token loss
  and nothing about recall or generation.
- The verification window can be gamed by the staleness trap by design;
  that is the study's hardest case and it is labelled as such rather than
  averaged away: segment-wise M1 is reported alongside the stream mean.
- Pilot-selected hyperparameters could favour Rule A over Rule B if the
  grid is unequal. Both grids have the same size and the same rule.
- Twelve seeds tie stream and model seed; a seed's stream draw and its W0
  vary together. A second panel with streams re-drawn under fixed W0 is a
  robustness check, not a licensed arm.
- Numerics: all arms of a seed run on the same GPU type (L4) and the same
  code commit; the frozen loss is computed once per stream and reused by
  every arm, so M1 differences are never confounded by re-evaluation noise.

## 4. Implementation plan

### 4.1 Testbed recommendation

PyTorch, nano Llama (RMSNorm, RoPE, SwiGLU, tied embeddings), 4 layers,
d = 384, 6 heads, d_ff = 1024, an 8K SentencePiece BPE trained on
TinyStories so that embeddings do not dominate the parameter count: about
7.1M non-embedding plus 3.1M embedding, about 10M total. A second size
(6 layers, d = 512, about 23M) is the fallback of 3.6 and, budget allowing,
a scale point. T = 256. Pretraining: 300M tokens of TinyStories per seed at
a fixed schedule, twelve seeds, about 25 minutes each on an L4 (the wall
clock is kernel-launch bound at this size, not FLOP bound).

paperkiln is not the right first vehicle. To run this study it would need:
(1) a fast-weight accumulator A stored separately from base weights with a
forward that reads W0 + A without materialising (or a materialise-and-
restore path with a checksum); (2) a gradient restricted to a single named
matrix, with backward only through the layers above it; (3) LoRA modules
with zero-initialised B and copy/restore of adapter state; (4) a prequential
streaming evaluator with teacher forcing, per-token loss emission, segment
labels and decision events in the events.jsonl format the other experiments
use; (5) a two-model mixture predictor for the Bayes arm; (6) GPU execution
(memory says local-GPU training is retired and CUDA is behind the GGUF
exporter on the roadmap). Items 1 to 4 are a week of C++ each side of
debugging. The right use of paperkiln is a second-implementation
reproduction of the licensed result after the PyTorch panel lands, which
doubles as the numerics-bridge check the house protocol likes.

### 4.2 Files to write (all under experiments/commit_ttt/)

- `PREREG_DRAFT.md`, then `PREREGISTRATION.md` at the anchor commit.
- `model.py`: nano Llama with `FastDown` (W_down plus accumulator A and a
  staging slot) and `LoRAPair`; a `with_weights(mode in {frozen, live,
  staged})` context so that all three evaluations share one code path.
- `tokenizer/`: the 8K BPE, trained once, committed.
- `data.py`: TinyStories shard loader; `streams.py`: generator for S-mixed,
  S-id, S-shift with segment labels, seeded.
- `pretrain.py`: twelve-seed pretraining, checkpoint per seed, receipt.
- `ttt_stream.py`: the online loop, one process per (seed, arm, rule,
  stream), emitting `events.jsonl` (one line per chunk: segment label,
  l_live, l_frozen, l_staged, decision, counted FLOPs) and `result.json`.
- `sweep_pilot.json`, `sweep_panel.json`: argv-sharded run lists for the
  colab-sweep skill (script takes `--shard I N`, is resumable per run,
  prints a done-marker, zips results).
- `analyze.py`: frozen at the anchor; prints M1 to M7 per seed and per
  segment, the manipulation check, and H1 to H4 verdicts with the thresholds
  in the text, deterministic.
- `receipts/`: events and results copied in immediately after each session,
  since WSL /tmp and Colab VMs are volatile.

### 4.3 Loop pseudocode

```
load W0[seed]; A = 0; stage = None; kv = empty
frozen_losses = precomputed per stream (one pass under W0, cached to disk)
for chunk j in stream:
    U = tokens[t : t+k]
    # prequential prediction of U under live weights; record l_live for U
    l_live_U = forward(W0 + A, U, kv); kv.extend(U)
    if stage is not None:
        # U's first m tokens are V_{j-1}; score the candidates there
        V = U[:m]
        l_frozen = frozen_losses[V]                      # exact: recompute over T; cached: reuse
        l_staged = forward(W0 + A + stage, V, context)   # exact: full T recompute; cached: kv prefix
        d_live_frozen = mean(l_live_U[:m] - l_frozen)
        d = l_staged - l_live_U[:m]
        if d_live_frozen > delta:        decision = ROLLBACK; A = 0
        elif mean(d) < -z*sd(d)/sqrt(m): decision = COMMIT;   A = A + stage
        else:                            decision = DISCARD
        log(j, segment[j], l_live_U, l_frozen, l_staged, decision, flops)
        stage = None
    # propose the next edit from this chunk
    stage = propose(W0 + A, U)      # Rule A: -eta * sum g h^T on W_down; Rule B: s SGD steps on LoRA copy
    t += k
```

Arms differ only in the decision line: always-commit sets decision = COMMIT
unconditionally and skips the two candidate forwards; periodic-reset adds
A = 0 every R chunks; Bayes-mix keeps two models and mixes probabilities;
frozen skips everything. The frozen per-token losses are computed once per
stream and shared by every arm so that M1 is never confounded by
re-evaluation.

### 4.4 Compute estimate (L4 GPU-hours)

- Pretraining: 12 seeds x 25 min = 5 h.
- Frozen pass per stream: negligible.
- Pilot: 3 seeds x 2 rules x grid (3 eta x 2 k x 2 m x 3 z = 36) on S-mixed
  only, gated arm, plus always-commit at 3 eta: about 3 x 2 x 40 = 240 runs
  x about 1 minute (128K tokens, launch bound) = 4 h.
- Panel: 12 seeds x 2 rules x 7 arms x 3 streams = 504 runs x about 1
  minute, with the gated-exact arm at about 3 minutes: roughly 10 h.
- Ablations (k = 64, n_A = 2 per-layer commit, fixed-nat margins): about
  4 h.
- Total about 23 h, leaving room under the 40 h ceiling for the 23M size
  as a scale point (about 12 h more) or for a re-drawn-stream robustness
  panel. Colab L4 reclaims sessions at about 80 minutes; every run is
  resumable and receipts are copied out per shard.

## 5. Where it dies

1. The trap does not fire at nano scale. A 10M model with a small eta may
   barely move on a 512-token burst, so always-commit does not regress and
   the manipulation check fails; raising eta until it does may make TTT
   useless on the helpful segments, so no eta satisfies both. What this
   still licenses: a clean measurement of the overfit gap M4 (verification-
   window versus update-window loss) as a function of eta and k at this
   scale, which is the quantity every TTT paper implicitly relies on and
   none of the ones above reports. That is a small methods result, not a
   TMLR paper on its own, and the sketch says so.

2. The window cannot see the staleness trap. The gate passes edits that a
   64-token window rewards and the next window punishes, H1 fails on
   segment (iii), and the guarantee turns out to be about the window and
   not about the stream. What this still licenses: the pre-registered
   negative, with a concrete statement of the failure mode (verification
   staleness) and its rate, plus the observation that a two-expert Bayes
   mixture has the guarantee the gate lacks. That is bankable as the
   "receipts-backed negative" in the transfer_s2 sense, and it has a clear
   lesson for anyone proposing verified TTT.

3. The Bayes mixture dominates. Gated and mixture land within noise on M1,
   and at exact verification the mixture is cheaper (Section 2.5); with
   cached verification the gate is cheaper but its decisions diverge from
   exact (M6 low), so the cost advantage belongs to a different method.
   What this still licenses: the recommendation that anyone wanting
   safe TTT should run the mixture, with numbers, and the finding that the
   only remaining argument for weight-space commitment is deployment as a
   single weight set. That is worth a short, well-measured paper; it is not
   the paper the idea was pitched as, and choosing it in advance would be
   selecting the outcome.

A fourth, smaller way: the result holds for Rule A and fails for Rule B, as
predicted in 3.4. That is not a death; it narrows the claim to
top-layer outer-product edits and says LoRA-style full-network TTT needs a
longer verification window than one chunk affords, which is a finding.

## 6. What this sketch does not decide

Venue beyond "TMLR class" (read recent TMLR issues first, per house rule);
whether the 23M size is a scale point or a fallback; the exact second
corpus for S-shift; whether the paper leads with the safety result or with
the measurement of the overfit gap M4. Those are the author's calls after a
pilot, and none of them changes a threshold above.
