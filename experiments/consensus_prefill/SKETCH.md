# consensus_prefill: multi-lane consensus prefill for the first token. SKETCH

*Drafted 17 September 2026. This is a scoping pass, a mechanism
specification and a pre-registration SKETCH. It is not the licence
anchor; nothing here licenses a claim. The anchor, if this study is
pursued, is the commit that renames a finished PREREGISTRATION.md next to
its frozen analysis script, nominated by Jonathan after reading it. House
protocol as in transfer_s1 and transfer_s2: hypotheses and thresholds
fixed before data, receipts for every run, analysis run verbatim from the
anchor commit, amendments dated in place.*

*Scope note: the automated scoper passed this idea and a hand check
passed it; this document is the deeper hand scoping requested after
that, plus the design. Sources were reached through web search and the
arXiv abstract pages directly, because the arXiv export API was
refusing us on the day. Every quoted sentence below is from the
paper's abstract page unless marked "listing snippet".*

## 0. The idea in one paragraph

Time-to-first-token (TTFT) is bounded below by the exact prefill of the
prompt. Run several cheap, differently approximated prefill lanes in
parallel over the same prompt inside the same model (context-truncated
lanes: sliding window, window plus sink tokens, strided token subset,
and a small-model drafter as the classical comparator). Each lane yields
a first-token distribution. If a quorum of lanes agrees (argmax
agreement and a calibrated distribution-distance test), emit the first
token provisionally and begin drafting; the exact full prefill runs
regardless, and when it lands its first-token distribution is compared
with what was emitted. On disagreement the emitted token is corrected
(greedy: replaced; sampling: the speculative-decoding acceptance rule).
The final output is therefore the exact model's output. The claim is a
reduction of average provisional TTFT on the agreeing fraction of
prompts, at zero final-output error, at a compute overhead that
vanishes with prompt length because a context-truncated lane's cost for
the LAST position is independent of prompt length (Section 3.1).

## 1. Literature scoping by hand (17 September 2026)

Three vocabularies were searched: (a) TTFT reduction and speculative or
approximate prefill with verification; (b) quorums of cheap
approximations gated by agreement with an exact fallback (cascades,
confidence-gated early exit, conformal deferral); (c) lossless
speculation proofs and multi-draft acceptance, for the sampling case.
Engineering documentation (vLLM chunked prefill and prefix caching,
TensorRT-LLM disaggregated serving, llama.cpp prompt-processing
discussions) was read for the systems baseline.

### 1.1 The table

Verdicts: TAKEN means the same claim exists; ADJACENT means the
mechanism or the gate is shared but the object (phase, losslessness,
multiplicity) differs; distinct means it is background or a tool.

| # | Work | What it does (quoted) | Phase / lossless / multi-lane | Verdict |
|---|---|---|---|---|
| 1 | SpecPrefill, arXiv 2502.02789 (ICML 2025) | "a training-free framework that improves the time-to-first-token (TTFT) for LLM inference by speculating locally important tokens using a lightweight model" (listing snippet); "< 5% accuracy loss" | prefill / lossy, no verification / single draft | distinct (lossy pruning; a baseline for us) |
| 2 | KV Prediction, arXiv 2410.08391 (Apple) | "A small auxiliary model is used to process the prompt and produce an approximation of the KV cache used by a base model" (listing snippet) | prefill / lossy / single auxiliary | distinct (a baseline) |
| 3 | LazyLLM, arXiv 2407.14057 | "selectively computes the KV for tokens important for the next token prediction in both the prefilling and decoding stages" (listing snippet) | prefill / lossy / single | distinct |
| 4 | MInference 1.0, arXiv 2407.02490 (NeurIPS 2024) | "reduces 95% of FLOPs in the attention computation to significantly accelerate the pre-filling stage" (listing snippet) | prefill / lossy dynamic sparse / single | distinct (note: attention-only savings; see 3.1) |
| 5 | CLAA, arXiv 2602.16054 and Cross-Family Speculative Prefill, arXiv 2603.02631 | CLAA "reduces Time-to-First-Token (TTFT) by up to 39% compared to the Full KV Cache baseline"; cross-family "retains 90~100% of full-prompt baseline performance" | prefill / lossy / single | distinct (the 2026 state of lossy TTFT work; the bar our lossless scheme is compared against on speed) |
| 6 | MagicDec, arXiv 2408.11049 | "We leverage draft model with sparse KV cache to address the KV bottleneck ... reduce latency without compromising accuracy" | DECODE / lossless (dense verification) / single self-drafter | ADJACENT: the nearest mechanism (sparse-KV self-draft, dense verify) but applied after prefill, never to the first token; target prefill stays exact and on the critical path |
| 7 | Windowed-MTP, arXiv 2607.21535 | "We apply a StreamingLLM-style sliding window plus attention sink to the draft's attention only ... lossless by construction: the full-attention target still decides every accepted token" | decode / lossless / single draft head | ADJACENT: window plus sink as the drafter's attention, our lane 1 exactly, but on the MTP head during decode; the target prefill is untouched |
| 8 | SparseSpec-L, arXiv 2607.27735; Vegas, arXiv 2602.07223; SpecAttn | SparseSpec-L "generates lightweight drafts directly from the target model using a dynamically sparsified and recallable KV cache" and verifies against the full cache | decode / lossless / single self-drafter | ADJACENT: same-model sparse drafting with exact verification is now a crowded 2026 vein for decoding; nobody in it touches the first token or runs several drafters and gates on their agreement |
| 9 | Distributed Speculative Inference, arXiv 2405.14105 (ICLR 2025) | "DSI leverages speculation parallelism (SP), a novel type of task parallelism, to orchestrate target and drafter instances that overlap in time" | decode / lossless / multiple drafter INSTANCES (same drafter, staggered) | ADJACENT: the systems template for "exact pass in the background while drafts are emitted"; ours is DSI's overlap applied at the prefill step with heterogeneous self-drafters |
| 10 | Multi-model speculative classification, arXiv 2503.18076 | "When majority worker models agree on a label, it is accepted as the final label ... In cases of disagreement, the judge model intervenes" | classification / LOSSY when workers agree (judge never checks) / multiple separate models | ADJACENT: the quorum gate, but the expensive model is skipped on agreement, so agreement errors are final; ours always runs the exact pass |
| 11 | Semantic-agreement cascades, arXiv 2509.21837 (EMNLP 2025 Industry); Conformal Cascade, arXiv 2607.25018 | "semantic agreement, meaning-level consensus between ensemble outputs, as a training-free signal for reliable deferral"; CC "accept when the calibrated set collapses to a single answer, defer otherwise" with a "distribution-free, finite-sample accuracy guarantee" | answer level / lossy on accept / multiple separate models | ADJACENT: agreement-as-deferral-signal is established at the answer level with separate models; Conformal Cascade's calibration is worth importing for our threshold (Section 3.3) |
| 12 | Multi-draft speculative sampling: Khisti et al. arXiv 2410.18234 (ICLR 2025); SpecTr arXiv 2310.15141; Global Resolution arXiv 2511.15898 | "a token-level draft selection scheme takes a list of valid tokens as input and produces an output token whose distribution matches that of the target model ... decomposed into a two-step solution: ... importance sampling ... [then] (single-draft) speculative sampling" | decode / lossless / multiple drafts | distinct (the tool that makes the sampling case lossless with several lanes) |
| 13 | Speculative Pre-Positioning, arXiv 2606.29565 | "when a confidence gate fires, is answered from a cached distribution in one near-constant vocabulary scan with no decode, at a cost only of energy and a rare, bounded false accept" | first token of the NEXT request / lossy (bounded false accept) / single | ADJACENT: a confidence-gated fast first token, but session-state specific and explicitly lossy |
| 14 | Speculative Speculative Decoding (Saguaro), arXiv 2603.03251 | "While a verification is ongoing, the draft model predicts likely verification outcomes and prepares speculations pre-emptively for them" | decode / lossless / single | ADJACENT: overlapping speculation with an in-flight verification; the same shape as our lanes drafting while the exact prefill runs |
| 15 | Leviathan et al. 2023 (arXiv 2211.17192), Chen et al. 2023 (arXiv 2302.01318) | "a novel modified rejection sampling scheme which preserves the distribution of the target model within hardware numerics" | decode / lossless / single | distinct (the lemma we apply to token 1) |

Also read and set aside: vLLM chunked prefill and automatic prefix
caching (orthogonal: they change what "the prompt" is; the scheme
composes with a prefix cache by treating the uncached suffix as the
prompt); Stream2LLM arXiv 2604.16395 (overlaps context streaming with
prefill; a scheduling result, no approximation); PEARL arXiv 2408.11850
("pre-verify" the first draft token during drafting; decode phase);
LayerSkip, SpecEE, HiSpec (early-exit self-speculation by depth, not by
context; a possible extra lane, Section 3.1); Draft-based Approximate
Inference arXiv 2506.08373 (SpecKV, SpecPC; lossy); Speculative
Cascades arXiv 2405.19261 (deferral rule inside the verification step;
a lossy-by-design cost-quality trade); Revisiting Lossy Verification
arXiv 2607.26627 (the analysis of what lossy acceptance does to the
distribution; useful for the paper's discussion of why we refuse it).

### 1.2 Verdict

Not TAKEN. No paper found runs approximate prefill lanes inside the
target model to produce a provisional FIRST token, gates on
inter-lane agreement, and verifies against the exact prefill so that
the final output is the exact model's. The three ingredients each exist
separately and are each well established:

- sparse-KV self-drafting with dense verification (MagicDec,
  Windowed-MTP, SparseSpec-L, Vegas, SpecAttn): decode phase only, one
  drafter, no gate other than the verifier;
- agreement between cheap models as a deferral signal (2503.18076,
  semantic cascades, Conformal Cascade): answer level, separate models,
  lossy on accept;
- overlap of drafting with an in-flight exact computation (DSI, Saguaro).

The exact claim that survives is narrower than the idea as first
stated, and it is this:

> For a fixed target model and a fixed gate rate, agreement across
> several differently context-truncated self-lanes is a better
> calibrated accept signal for the first token than the confidence of
> any single such lane at matched lane compute, and the resulting
> provisional-emit, exact-verify scheme lowers provisional TTFT on the
> gated fraction with the final output equal to the exact model's.

The first clause is the scientific content and the primary falsifier
(Section 4). The second clause on its own is not novel enough to carry a
paper: a single-lane version (window-plus-sink draft of token 1,
verified by the exact prefill) is MagicDec's mechanism moved one step
earlier, and a reviewer would say so. The quorum has to earn its place
by being better calibrated than a single lane's max-probability at the
same cost, or the paper collapses to the single-lane scheme plus a
measurement (Section 6).

One more thing the scoping changed. The idea as posed lists "a low-rank
or top-k attention approximation" among the lanes. Those lanes do not
reduce prefill cost in the regime where it matters (Section 3.1): they
save attention FLOPs only, and attention is a minority of prefill FLOPs
until the prompt is several times longer than the model width. The lanes
that are actually cheap are the ones that reduce the number of
token-layer evaluations, and for the first token that means
context-truncated lanes whose cost is independent of prompt length.
That is a sharper mechanism than "cheap attention", and it is what the
design below uses.

## 2. What already exists in our stack

- paperkiln: `ops::swa_attention(q, k, v, scale, window, sinks, seq_len)`
  (include/microtorch/ops.hpp, line 149) and `nn::SlidingWindowAttention`
  (nn.hpp, line 110), with the bitwise equivalence pin against full causal
  attention at window >= T and sinks = 0 (tests/test_swa.cpp). Exact and
  window-plus-sink lanes are therefore spec-expressible for TRAINING; for
  this study the lanes are inference-time approximations of one exactly
  trained model, so the training-side op matters only if we want a model
  that was trained windowed as an extra control (Section 4.5, threat 5).
- paperkiln export: mtstudio writes `<name>.safetensors` and `<name>.gguf`
  with the vocabulary embedded; `tools/hf_export.py` converts a finished
  run to an HF-layout folder that `AutoModelForCausalLM.from_pretrained`
  opens, and `tools/hf_export_verify.py` pins argmax-identical greedy
  continuations between transformers and `mtstudio sample --topk 1`,
  which is the same parity standard ember.cpp serving is held to. This
  is the path for the PyTorch prototype (Section 5.1).
- ember.cpp (local dir tinyllama.cpp): CPU inference for Llama-family
  GGUF and safetensors, batched prefill (`forward_device_batch_prefill`,
  `enable_prefill_chunking`), a `KVCache` with per-sequence batch slots,
  and a scalar CPU attention loop that runs `for t_hist in [0,
  history_len)` per query per head (cpu_attention.cpp lines 115 to 200).
  A windowed or strided lane is a change of the loop bounds plus a
  per-token true-position for RoPE, which the batched path already
  tracks as `global_seq_pos`. `TINYLLAMA_TOPK_DEBUG=1` already dumps
  top-10 (id, logit) at every position, which is most of the lane
  logging we need. Word-level tiny models from paperkiln serve directly.

## 3. Mechanism specification

### 3.1 The lanes and what they cost

Notation: L layers, width d, prompt length T, window w, sinks s. Per
token per layer, a Llama block costs about 24 d^2 FLOPs in projections
and MLP (SwiGLU at d_ff = 8d/3) plus about 4 T d in attention over T
keys. Attention therefore dominates only when T > 6 d, which for a
1B-class model (d = 2048) is T > 12k. At T <= 4096 an approximation that
only thins attention leaves 75% or more of the prefill cost in place.
This is why the lanes below all reduce the NUMBER of token-layer
evaluations and not merely the attention pattern.

The key observation for the first token: we need the output at ONE
position, the last. Under windowed attention of width w, the last
position at layer L-1 depends on positions [T-w, T) at layer L-2, which
depend on [T-2w, T) at layer L-3, and so on. The dependency cone of the
last position has L(L+1)/2 * w token-layer evaluations in total, and it
does not grow with T. Sink tokens (the first s positions) are in every
window and, being causal-first, need only themselves at every layer:
L * s more. So:

| lane | definition | token-layer cost | cost at L = 22, w = 128, s = 4 (TinyLlama) | note |
|---|---|---|---|---|
| E, exact | full causal prefill | L * T | 22 T (90k at T = 4096) | the baseline and the verifier |
| C(w), cone-windowed | the windowed-attention model's output at the last position, computed on its cone only | L(L+1)/2 * w + L * s | 32.4k, independent of T | break-even against E at T = (L+1) w / 2 = 1472; at T = 4096 it is 36% of E |
| C(w') | same at a second window | as above | w' = 64: 16.3k | the second member of the quorum |
| P(R), truncated prompt | the exact model run on sinks plus the last R tokens with true positions, full causal attention among them | L (R + s) | R = 512: 11.4k | a different approximation from C: the same tokens, full attention among them |
| S(k, w), strided | sinks plus every k-th token plus the last w, true positions, full causal among kept tokens | L (T/k + w + s) | k = 8, w = 128: 14.2k at T = 4096 | the only lane that sees the middle of the prompt; SpecPrefill-style position preservation without the draft-model scoring |
| D, small-model drafter | exact prefill of a smaller model in the same family | L_D * T at width d_D | Qwen2.5-0.5B against 1.5B: about 1/3 of E | the classical comparator (KV Prediction and SpecPrefill class); grows with T |
| X, depth early exit (optional) | full-width prefill, exit after L' layers with the final norm and head | L' * T | not cost-free at T = 4096 (grows with T) | LayerSkip-style; included only in the PyTorch agreement table, not in the wall-clock claim |

A quorum of {C(128), C(64), P(512), S(8, 128)} costs about 74k
token-layers at T = 4096 against E's 90k, and is T-independent apart
from S. At T = 1024 the same quorum costs more than E (74k against
22.5k), so THE SCHEME ONLY PAYS IN COMPUTE AT LONG PROMPTS; at short
prompts it is a latency-for-compute trade. That is stated up front and
the compute-overhead ratio is a registered metric, not a footnote.

Wall-clock, where the win actually exists. On a GPU the exact prefill
of a 1B model at T = 4096 is one fused compute-bound pass of about 0.3
s on an L4; the lanes are smaller passes and would run concurrently on
separate streams, but they contend for the same SMs and a 36% FLOP lane
is not a 36% wall-clock lane when the exact kernel is already at full
occupancy. On a CPU (ember.cpp, llama.cpp-class engines, the on-device
regime KV Prediction and SpecPrefill motivate themselves by) prompt
processing runs at hundreds of tokens per second for a 1B model, an
exact 4096-token prefill takes seconds to tens of seconds, and FLOPs
track wall-clock much more closely. The registered wall-clock claim
(Section 4.2, H4) is therefore for ember.cpp on CPU. The GPU numbers are
measured and reported descriptively; no wall-clock claim is registered
for the GPU.

Two implementation notes. First, the cone lane C(w) is computed most
simply as a suffix run of length L * w with a banded causal mask of
width w, sink columns always visible, and TRUE position ids for RoPE;
the last position's output is then exactly the windowed model's (proof:
position p at layer l is correct if its keys [p-w, p] are correct at
layer l-1, and by induction every position in [T-(L-l)w, T) is correct
at layer l, with layer 0 inputs being embeddings). The suffix run costs
about twice the cone (L^2 w against L^2 w / 2); the C++ version can do
the triangle. Second, all lanes use the SAME weights and the SAME
engine as the exact pass, so "exact" is well defined as that engine's
own full prefill, and losslessness is relative to it, bit for bit under
greedy decoding.

### 3.2 The agreement test

Each lane i returns a first-token distribution q_i over the vocabulary
(the full softmax; we keep the top-64 (id, logit) pairs plus the log
partition, which suffices for the tests below). Two gates, both
registered, both evaluated from the same stored table:

- G-arg(m of n): the argmax of at least m of the n lanes coincides.
  Emit that token. Registered configurations: 3 of 4 and 4 of 4.
- G-dist(tau): G-arg holds AND the maximum pairwise total variation
  distance among the agreeing lanes' top-64 distributions is below tau.
  Total variation rather than KL because KL between truncated softmaxes
  is unstable in the tail and because TV is the quantity that bounds
  the acceptance probability in the sampling case (Section 3.4).

Threshold calibration is done ONCE on a held-out calibration set of
prompts disjoint from every evaluation fold, by the conformal recipe:
choose tau as the largest value such that the empirical precision
P(exact argmax = vote | gate fires) on the calibration set is at least
1 - alpha with alpha = 0.02, using the finite-sample quantile
correction of split conformal prediction (Conformal Cascade uses the
same construction for its deferral rule; we cite it and do not claim
the construction). The calibration set is 1024 prompts per model per
length band, drawn before any lane is run, and its receipts are
committed with the anchor. No threshold is ever tuned on an evaluation
fold.

The single-lane comparator (baseline B1, Section 4.3) gets the same
treatment: for each lane alone, the max-probability threshold is
conformally calibrated to the same target precision, and its gate rate
is compared with the quorum's. This is the matched comparison the
primary hypothesis rests on.

### 3.3 The correction protocol

Greedy decoding (temperature 0, top-k 1). The vote token v is emitted
provisionally at time t_lane. Drafting continues from the best lane's
KV cache (the lane with the highest probability on v), producing draft
tokens v_2, v_3, ... until the exact prefill lands at t_exact. The exact
pass then (i) compares its argmax x_1 with v; if x_1 = v the provisional
token is committed; if not, v is retracted and x_1 is emitted; (ii) if
committed, verifies the draft continuation in one target pass exactly as
ordinary greedy speculative decoding does (accept the longest prefix that
matches the exact argmaxes, then one bonus token). Decoding then
proceeds from the exact KV cache. Every token the user finally holds is
the exact model's greedy token. What the user SEES may include a
retracted token; the retraction rate equals 1 minus the gate precision
and is a registered metric, because a scheme that retracts 5% of first
tokens is not shippable however lossless its final output.

Sampling (temperature > 0). Let p be the exact first-token
distribution and let q be the drafting distribution, defined as the
distribution of the lane that won the vote (or the uniform mixture of
the agreeing lanes; both are registered, the mixture is the default
because it is smoother). Emit x ~ q provisionally. When p arrives,
accept x with probability min(1, p(x) / q(x)); on rejection, retract x
and emit y ~ norm(max(0, p - q)). This is the Leviathan and Chen
acceptance rule applied to token 1 with the lane mixture as the draft.
If we want to use every lane's SAMPLE rather than one mixture, the
optimal multi-draft acceptance (Khisti et al.) applies; it is not the
default because its importance-sampling step needs p and so cannot be
run before the exact pass lands, whereas the single-mixture rule can
emit before p is known. The gate under sampling is G-dist only (an
argmax quorum is the wrong object when the argmax is not what is
emitted).

Sinks, positions and caches. Every lane keeps its own KV cache; the
exact pass writes the one cache decoding uses. Lanes are never written
into the exact cache and the exact cache is never read by a lane.
Draft continuation from a lane cache is the only place lane state
outlives the gate, and it is discarded after verification.

### 3.4 The losslessness lemma

Lemma (final-output losslessness). Let p be the exact model's
first-token distribution given the prompt, q the drafting distribution,
and G an event (the gate) that is a measurable function of the lanes'
outputs alone and therefore independent of the randomness used to
sample from p. Define the committed first token X as: if G fails, X ~ p;
if G holds, X is the output of the speculative acceptance rule applied
to x ~ q against p. Then X ~ p. For greedy decoding the same holds with
p a point mass. Subsequent tokens are produced from the exact KV cache
conditioned on X (with any draft continuation verified by the standard
rule), so the joint law of the output equals the exact model's.

Proof sketch. On {G fails} the law is p by construction. On {G holds}
the single-draft acceptance rule yields law p for any draft q (Leviathan
et al. 2023, Theorem 1; Chen et al. 2023, Theorem 1). G is independent
of the sampling randomness by hypothesis, so the mixture over G of two
p-distributed tokens is p. The continuation claim is the standard
speculative-decoding argument applied from the exact cache. The only
assumption with teeth is that the gate is a function of lane outputs
alone; a gate that peeked at p would break independence. Hardware
numerics: losslessness is bitwise for greedy within one engine, and
"within hardware numerics" for sampling, as in the cited proofs.

The lemma is unremarkable, and that is the point: the mechanism's
correctness is inherited, and the whole scientific content lives in
whether the gate is well calibrated and fires often (Section 4).

### 3.5 Cost accounting

Registered accounting for every prompt: FLOPs of each lane, FLOPs of the
exact pass, their sum, and the ratio rho = sum(lanes) / exact. Under
DSI-style overlap the total compute is exact plus lanes; there is no
setting in which the scheme uses less compute than the exact prefill.
Wall-clock: TTFT_prov (request to provisional emit, gated prompts only),
TTFT_comm (request to committed first token, all prompts; on gated
prompts this is the exact pass's finish plus contention), and
TTFT_exact (the exact-only baseline with ALL threads, the strongest
baseline, not a crippled one). Thread accounting on CPU: the lanes and
the exact pass share a fixed thread budget; the split is a registered
parameter (default: lanes on one quarter of threads, exact on the
rest), and the serial variant (lanes first, then exact on all threads)
is also reported so the contention cost is visible.

## 4. Falsifiers and pre-registration sketch

### 4.1 Sampling unit and seeds

The models are fixed and the randomness is over prompts. The sampling
unit is a PROMPT FOLD: each evaluation set is partitioned once, before
any lane is run, into twelve disjoint folds of equal size stratified by
length band and prompt source; every headline number is computed per
fold and reported as the fold median with the twelve-fold bootstrap
2.5th and 97.5th percentiles; paired comparisons (quorum against a
single lane) are per fold, twelve pairs, reported as a paired t with
df = 11 (two-tailed critical 2.201) and a sign count. Twelve is the
house number and the folds play the role seeds play in the training
studies. For the paperkiln arm, where we train the models, the unit is
the training SEED: twelve seeds of one small configuration, and the
statistic is the per-seed gate precision, so that the agreement
structure is shown to be a property of the mechanism rather than of one
checkpoint.

### 4.2 Hypotheses with thresholds (drafted; to be frozen at the anchor)

Evaluated on the natural prompt set at T >= 1024 unless stated;
"quorum" means G-dist with the calibrated tau over lanes {C(128),
C(64), P(512), S(8, 128)}, 3 of 4.

- H1, quorum precision. P(exact argmax = vote | gate fires) >= 0.98,
  with fold-bootstrap 2.5th percentile >= 0.95. Under sampling, the
  first-token acceptance probability on gated prompts >= 0.90.
- H2, gate rate. The gate fires on >= 0.50 of natural prompts, and on
  >= 0.30 of the NON-TRIVIAL stratum (exact first-token entropy above
  its median for that length band). The second number is the one that
  matters; the first alone can be met by formulaic prompts.
- H3, the quorum earns its place (PRIMARY). At matched precision (both
  calibrated to 0.98 on the calibration set), the quorum's gate rate
  exceeds the best single lane's confidence-gated rate by a factor
  >= 1.25 on the non-trivial stratum, paired over the twelve folds,
  t >= 2.201 two-tailed and >= 10 of 12 folds in the same direction.
  Equivalently, at matched gate rate, precision higher by >= 0.02
  absolute. Both readings are computed; the rate reading is primary.
- H4, wall-clock on CPU. In ember.cpp at T = 4096 with the default
  thread split, mean TTFT_prov on gated prompts <= 0.5 * TTFT_exact,
  and mean TTFT_comm over ALL prompts <= 1.15 * TTFT_exact. The second
  clause is the price: if overlap costs more than 15% on the committed
  token, the scheme is a UI trick and not a latency result.
- H5, compute. rho <= 1.0 at T = 4096 and rho decreasing in T over the
  band (the T-independence of the cone lanes is a prediction that can
  fail if the implementation is not the cone).

Headline "consensus prefill lowers TTFT losslessly and the quorum is
what makes it safe" is claimable only if H1, H3 and H4 all clear. H1
and H4 without H3 is the single-lane paper. H3 without H4 is a
calibration result with no latency claim.

### 4.3 Baselines

- B0, exact prefill, all threads. The bound being beaten.
- B1, single-lane self-speculative first token with a conformally
  calibrated max-probability gate, one baseline per lane. This is
  MagicDec's mechanism moved to token 1 and is the comparator H3 is
  about. If B1 with C(128) alone matches the quorum, the multi-lane
  element is dead.
- B2, SpecPrefill-style pruned prefill (top fraction of tokens by a
  small model's attention, positions preserved), reported LOSSY: its
  first-token accuracy against exact and its TTFT, so the reader sees
  what losslessness costs relative to the lossy state of the art.
- B3, small-model drafter D with a calibrated confidence gate (KV
  Prediction and SpecPrefill class drafter, made lossless by our
  verification). If D beats the self-lanes at matched cost, the
  interesting object is the drafter, not the lanes.
- B4, context-free predictor: a unigram over first tokens estimated on
  the calibration set, plus the trivial lane P(64). If the quorum's
  gated precision is not clearly above these on the non-trivial
  stratum, the test is too easy and the prompt set is replaced
  (fallback clause 2).

### 4.4 Prompt sets, metrics, fallback clauses

Prompt sets, lengths {128, 256, 512, 1024, 2048, 4096} tokens where the
model's context allows:

- N, natural continuation: PG-19 and WikiText-103 documents cut at a
  random token; code files from a permissively licensed sample cut at a
  random line (code first tokens are often deterministic; kept because
  that is the easy regime and it is reported as such).
- Q-tail, document then question: long document followed by a
  question whose answer is the first token; the answer's evidence is
  near the end for half the prompts and at controlled depth for the
  other half.
- Q-head, question then document: the question precedes the document,
  so the first token depends on content the tail lanes cannot see.
  This is the designed contrast for the correlated-lanes threat and is
  where S(8, 128) has to earn its inclusion.
- Y, synthetic needle: the first token IS a value planted at controlled
  depth; agreement should collapse when the value sits outside every
  lane's view, and precision must not fall with it (the gate must
  refuse rather than agree on a wrong token).

Calibration set: 1024 prompts per model per length band drawn from N
and Q-tail only, disjoint from all folds.

Metrics: gate precision and rate (overall and by entropy stratum), the
full pairwise lane-agreement matrix conditional on exact entropy and
on prompt set, per-lane accuracy and calibration curve, retraction
rate, sampling acceptance probability and a losslessness pin (10,000
sampled first tokens on ten fixed prompts, TV distance to p below
0.01, chi-square not rejecting), rho, TTFT_prov, TTFT_comm,
TTFT_exact, and thread-split sensitivity.

Fallback clauses, fixed now:

1. If TinyLlama's 2048 context caps the band, the 4096 cell runs on
   Qwen2.5-0.5B or Llama-3.2-1B (whichever ember.cpp loads cleanly at
   4096; the Qwen parity tooling exists in paperkiln but the ember.cpp
   Qwen path is unverified); if neither, the 4096 cell is PyTorch CPU
   wall-clock only, labelled as a weaker proxy.
2. If B4 is within 0.02 of the quorum's precision on N, the natural set
   is too easy for the first token; the primary reading moves to
   Q-tail and Q-head and N is reported as descriptive.
3. If the calibration set cannot reach precision 0.98 at any tau with
   gate rate >= 0.10, the precision target is reported as unreachable
   for that model and the study reports the precision-rate curve with
   no H1 claim.
4. If the CPU thread split makes TTFT_comm exceed 1.15 x exact at every
   split, H4 fails and the wall-clock section reports the serial
   variant only.

### 4.5 Threats

1. Correlated lanes. Every tail lane deletes the same region (the
   middle of the prompt), so agreement among them is not independent
   evidence. Q-head and Y are designed to expose this; the pairwise
   agreement matrix conditional on failure is reported; S(8, 128) is
   the decorrelating lane and its marginal contribution to precision
   is a registered secondary number.
2. Trivial first token. Formulaic starts make agreement cheap. The
   entropy stratification and B4 guard this; H2 and H3 are stated on
   the non-trivial stratum.
3. FLOPs versus wall-clock. Addressed by registering wall-clock on CPU
   only, with the strongest baseline, and reporting the GPU
   descriptively. A reviewer will ask why anyone cares about CPU
   prefill; the answer is the on-device regime that the lossy TTFT
   literature already invokes, and that ember.cpp is that regime.
4. Provisional tokens are visible. A retraction is a user-visible
   event even though the final output is exact; retraction rate is a
   registered metric and the paper says plainly that "lossless" is a
   statement about the final output.
5. Windowed-at-inference is not what the model was trained for. The
   cone lane's quality depends on how far the model's attention
   actually reaches; a model trained windowed (paperkiln can, via
   swa_attention) would agree with itself trivially. That is a
   control, not a headline: a windowed-trained paperkiln model's lanes
   should show near-perfect agreement, and if they do not the cone
   implementation is wrong. Registered as a pin, not a result.
6. Prompt caching. In production the prefix is often cached and the
   uncached suffix is short, where the cone lanes are not cheaper than
   exact (Section 3.1). The claim is scoped to uncached prompts of
   T >= 1024 and says so.

### 4.6 Predictions written before running (to be frozen)

H1 passes at T >= 1024 on N and Q-tail; fails on Q-head at any tau that
fires. H2 passes on N, and reaches about 0.3 on the non-trivial stratum,
marginal. H3 is the one I would bet against at even odds: the tail
lanes are correlated, so the quorum's gain over C(128) alone is likely
small (+0.01 precision or a 1.1x rate); S(8, 128) helps on Q-head but
fires rarely there. H4 passes on CPU at 4096 (TTFT_prov near 0.35 x
exact for a 1B model) and would not pass on an L4. H5 passes if the
cone is implemented as the cone.

## 5. Implementation plan

### 5.1 Phase A, PyTorch measurement of agreement structure (no C++)

Everything in Section 4 except wall-clock is computed from ONE stored
table per model: for every prompt, the exact top-64 (id, logit) and log
partition, the same for each lane, and the exact first-token entropy.
Gates, thresholds, baselines B1, B3, B4 and all hypotheses but H4 are
then evaluated offline from the table, so the calibration and every
reading are reproducible from receipts without re-running a model.

Models: (i) a paperkiln model trained on TinyStories at T = 1024, L = 6,
d = 256, twelve seeds, exported through `tools/hf_export.py` and pinned
with `hf_export_verify.py` (the mechanism arm, house seeds); (ii)
TinyLlama-1.1B (context 2048) and Qwen2.5-0.5B or Llama-3.2-1B (context
to 4096) from HF, the public arm, prompt folds as the unit.

Lanes in transformers: custom 4D attention masks with true
`position_ids`. C(w) as the suffix run with a banded mask and sink
columns; P(R) and S(k, w) as token-subset inputs with full causal
masks; D as a second model; X as `output_hidden_states` plus the final
norm and head. A unit pin: C(w) with w >= T must reproduce the exact
logits to float tolerance, and S(k, w) with k = 1 must too; both are
the cone's equivalent of the swa bitwise pin.

Script sketch: `experiments/consensus_prefill/lanes.py` (produces the
table), `calibrate.py` (conformal tau on the calibration set, writes
the threshold receipt), `analyze.py` (frozen at the anchor: folds,
gates, baselines, hypotheses, plots). Colab L4, `supervisor.py` lanes.

### 5.2 Phase B, ember.cpp (only if Phase A clears H1 and H3)

1. Lane execution: a `LanePlan {window, sinks, stride, tail}` applied in
   `attention_batch_cpu` by restricting `t_hist` to sinks and
   [pos - window, pos] (cone) or by feeding a token subset with true
   positions (P, S), each lane with its own `KVCache`.
2. Agreement gate: lane logits to top-64 plus log partition, the two
   gates, the calibrated tau loaded from the Phase A receipt.
3. Deferred exact pass: OpenMP thread partition (lanes on a subset,
   exact on the rest) or serial for the first measurement; timestamps
   at request, provisional emit, exact finish, committed emit.
4. Correction: greedy replace-and-reverify; sampling acceptance rule
   with the residual resample; the 10,000-sample losslessness pin.
5. `--lanes` CLI, a `TTFT` JSONL log, and a `tools/reproduce.py` row.

### 5.3 Compute estimate

Phase A: public arm, two models, about 15k prompts each, six passes per
prompt at up to 4096 tokens, about 5 GPU-hours per model on an L4 (a 1B
model prefills 4096 tokens in about 0.3 s; lanes are cheaper). Paperkiln
arm: twelve seeds at L = 6, d = 256, T = 1024, about 20 minutes each on
an L4, 4 GPU-hours, plus 2 GPU-hours of tables. GPU descriptive
wall-clock, 2 GPU-hours. Total about 18 GPU-hours, half the 40-hour
budget, leaving room for one re-run. Phase B: CPU only, about 1000
prompts at T = 4096 for a 1B model at tens of seconds per exact
prefill, 10 to 20 CPU-hours, run unattended on the local machine or a
Colab CPU runtime; no GPU.

## 6. Where it dies

1. H3 fails: agreement across tail lanes is no better calibrated than
   one lane's confidence at matched cost, because the lanes are
   correlated. What survives: the single-lane self-speculative first
   token on CPU is still a lossless TTFT result (MagicDec's mechanism at
   token 1, which nobody has measured), and the agreement matrix
   conditional on entropy and prompt structure is a clean small
   measurement of how far a model's first token actually depends on its
   context. That is a short TMLR-class negative with a systems
   footnote, not a flagship.
2. H2 fails on the non-trivial stratum: the gate fires only when the
   first token was formulaic, so the win exists on prompts nobody
   needed it for. What survives: a characterisation of context-truncated
   first-token prediction (when it fails and how the entropy of the
   exact first token predicts that), which is a useful table for
   anyone building lossy prefill methods, and a stated boundary on where
   speculative prefill of any kind can help.
3. H4 fails: on CPU the lanes' contention or the serial lane time eats
   the gain, or the exact prefill at T <= 4096 for a 1B model is already
   fast enough that the saving is tens of milliseconds. What survives:
   the T-independent cone-cost analysis and the break-even length
   (L+1) w / 2 as an analytic result, with the measured FLOP ratios, and
   a plain statement that the regime where this pays is longer
   prompts than a 40-GPU-hour study can reach with a lossless verifier.

A fourth, quieter death: a paper appears in the next three months doing
sparse self-speculation on the first token with a verifier. The
decoding vein (items 6 to 8 in the table) is moving fast and the step to
token 1 is small. If that happens the quorum-calibration question (H3)
is still open and the study narrows to it; if that paper also gates on
agreement, the idea is TAKEN and the file is closed with this note.

## 7. Open decisions for Jonathan

- Whether to pursue Phase A at all given the month's priority (the
  agentic project to about 11 September is done; this would sit behind
  the transfer methodology paper's final pass).
- The public model for the 4096 cell (TinyLlama caps at 2048).
- The gate default (3 of 4 argmax plus TV, or TV only).
- Whether the paperkiln twelve-seed arm is worth 6 GPU-hours or whether
  the public arm with folds is enough for a first look; the house rule
  says seeds where training is involved, and it is involved.
