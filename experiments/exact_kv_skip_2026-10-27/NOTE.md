# Exactness-certified KV tile skipping: measurement and prior art (27 Oct 2026)

`LOCAL_LLM_WORK.md` D1 (Jonathan, 27 Oct 2026). No kernel was built. Everything below runs on the CPU
through the keramos numpy IR reference (`spec/tools/ir_numpy.py`).

## Result

On our four picks, at 64 to 512 keys, **no key-free bound certifies a meaningful share of KV tiles.** The
largest certified share in any model, tiling, context bin, bound and certificate was 0.011% of keys
(pick_e1_60m_L32, message tiles). In every configuration the metadata costs more reads than the skips
save.

Two things make the outcome independent of how good a bound is:
- **Oracle ceiling.** Even an oracle that knows the true softmax weights and the exact output-error vector
  can skip only 0.45-3.1% of keys within ROUNDING (1e-5) at 385-512 keys.
- **Bit-exactness.** No closed tile ever reached the order-free bit-exact regime: every weight
  underflowing to +0 in f32, 87-104 nats below the row max. The half-ulp criterion admits at most 0.29%
  of keys.

Projected decode-time saving: none. A perfect, free bound would cap the saving at about 0.5% of a
pick_e1_L16 decode step at 384 keys.

The attention is not sparse in the sense exactness needs:
- 91-97% of 32-key tiles have their largest logit within 10 nats of the row maximum.
- 98-99.8% have it within 15 nats.
- Certification needs about 11-17 nats plus the bound's own looseness.
- That looseness is 10-25 nats for every key-free bound we tried.

**Recommendation:** do not build the kernel for these models.

## Setup

**Models.** Each pick's `model.gguf` with its embedded IR document, f32. All have ctx 512, head_dim 64,
softmax scale 0.125, no window and no GQA.

| pick | path | shape |
|---|---|---|
| pick_e1_L16 (20M) | `C:/ml_artifacts/tiny_chat/e1/pick_e1_L16` | d 320, 16 layers x 5 heads |
| pick_e5_s0 | `C:/ml_artifacts/tiny_chat/e5/pick_e5_s0` | d 448, 8 x 7 |
| pick_e1_60m_L32 (60M) | `C:/ml_artifacts/tiny_chat/e1_60m/pick_e1_60m_L32` | d 384, 32 x 6 |
| pk_sft_v1 (97.5M) | `C:/ml_artifacts/general_chat/sft/pk_sft_v1.gguf` | d 768, 12 x 12 |

Contexts of 1,024 tokens were not run: no pick allows more than 512.

**Contexts.** 15 per model, rendered with the paperkiln/1 template. Each exchange is
`<|user|>U<|assistant|>A<|endoftext|>`.

| source | contexts | tokens |
|---|---|---|
| Carry-12 v1 (`corpus/eval/carry12_v1.jsonl`, gold replies), single | 4: rows 0, 400, 800, 1200 | 268-281 (12 exchanges) |
| Carry-12 v1, two conversations chained | 2: rows 100+101, 900+901 | 512 (22 exchanges, cut at 512) |
| DailyDialog, the corpus file's last 400 dialogues, longest first | 3 | 512, 512, 374 |
| BST, the corpus file's last 400 dialogues, longest first | 3 | 415, 366, 332 |
| `lever1_conv_v1.jsonl` tail, longest first | 3 | 511, 471, 495 |

The `lever1_conv_v1.jsonl` tail is pk_sft_v1's SFT file; mtstudio holds out a file's tail. Its rows
there are single exchanges.

The tiny-chat SFT file (`sft_human.jsonl`) is not on this machine. So the DailyDialog and BST
dialogues are human dialogue from the corpus, not a verified held-out split.

**Decode positions.** Causal attention at position p over keys 0..p is the decode step at p. One
reference prefill per context therefore gives every decode step:
- q, k and v are captured post-RoPE at every layer;
- every position, layer and head is analysed in float64 over those f32 tensors;
- results are binned by key count n = p + 1: <=64, 65-128, 129-256, 257-384, 385-512.

**Tilings:**
- fixed32, fixed64 and fixed128;
- **turn**: each `<|endoftext|>` closes a tile;
- **message**: each `<|user|>` or `<|assistant|>` opens one.

The query's own (open) tile is always read. Only closed tiles before it are candidates. Their share of
keys ("closed share") is 0.50-0.97 for fixed tiles. For turn tiles it is 0.57-0.72, because the
single-exchange lever1 contexts have no closed turn.

### The criterion (part b)

Let δ be the dense softmax mass of the skipped set T, μ_T the value average over T, and o_S the output
of the read set S alone. The dense output is o = (1 - δ) o_S + δ μ_T, so

    o - o_S = δ (μ_T - o_S),   and per coordinate   |o_i - o_S,i| <= δ · max_{j in T} |v_j,i - o_S,i|.

This is the omitted-mass identity, the same one as in 2602.06317. It is a criterion on the output
vector, not on the weight: a tile is safe when its mass times its values' spread around the output is
small. IR_SPEC 4.3 gives the ROUNDING test, per tensor:

    |a - b| <= 1e-5 · max(1, max|ref|)

The measured terms have these sizes. The value spread R (per-dimension range of v over keys 0..p,
largest dimension) averages 6.5 (pick_e1_L16), 7.4 (60M) and 8.6 (pk_sft_v1); max|o| is at most 5-7.

With Z_S >= 1, one n-key tile is certified when n·e^{U - m}·R / Z_S <= 1e-5·max(1, max|o|), where U is
the bound on the tile's largest logit and m is the row maximum. For n = 32 this needs **U at least
about 17 - ln Z_S nats below m**. ln Z_S lies between 0 and ln n (at most 6.2 here), so the requirement
is between about 11 and 17 nats, and the dense kernel's own rounding still has to share the tolerance.

Bit-exactness is far stricter:
- **Order-free regime:** every weight of the tile rounds to +0 in f32, so adding the tile changes no
  partial sum in any order. That needs every logit at least 103.97 nats under m (87.34 with
  flush-to-zero).
- **"Cannot move the last bit" criterion:** |o_i - o_S,i| <= ½ ulp(o_i) for every i. This is necessary
  for bit equality but not sufficient, because intermediate sums round too.

**Oracle quantities recorded:**
- each tile's mass;
- tiles below masses 2^-24 through 1e-4;
- tiles whose omission alone stays within ROUNDING;
- the largest ascending-mass skip set within ROUNDING, using the exact error vector;
- the same set within half an ulp of every coordinate;
- underflow tiles;
- the tile-max gap histogram;
- for contrast, skip sets at tail-mass budgets 1e-4 to 5e-2 (the approximate regime where Quest and
  top-k work).

### Bounds and certificates (parts c and d)

All bounds are upper bounds on the scaled q·k of every key in a tile, computed from per-tile metadata
without reading the tile's keys:

| bound | definition | metadata per tile |
|---|---|---|
| **norm** | \|q\| · max\|k\| (Cauchy-Schwarz) | 1 float |
| **pairnorm** (ours) | Σ over RoPE pairs of \|q_c\| · max\|k_c\|; invariant to RoPE rotation | 32 floats |
| **box** (Quest) | Σ_i max(q_i kmin_i, q_i kmax_i) | 128 floats |
| **ball** (ours here; cluster bounds as in Louver and ArkVale) | q·c + \|q\|·r, with c the key centroid and r the largest distance from it | 65 floats |
| **pairmin** (ours) | Σ over pairs of min(the pair's box term, the pair's norm term); never looser than box | 160 floats |
| **exact** (control) | the true tile maximum | reads the keys, saving V only (BLASST's regime) |

Certificates. U_t is a tile's bound and n_t its key count; the bound on the skipped set's mass is
Σ_t n_t e^{U_t - m}/Z_S.

| certificate | read order | value term |
|---|---|---|
| greedy_hull | tiles by descending U until the rest certify | per-dimension range of all values so far (one running min/max per head) |
| greedy_tile | the same | per-tile value min/max around o_S (128 more floats per tile) |
| twophase_hull | own tile and the top-U tile, then skip the lowest-U tiles whose summed bound fits | as greedy_hull |

Tolerance 1e-5 throughout, plus greedy_hull at 1e-6 to leave room for the kernel's own rounding. Scale:
a valid lower bound on max(1, max|ref|).

**Cost model (d).** One row is one key plus its value, 2·64 floats. Dense reads n rows. A certified
kernel reads:
- the own tile;
- the unskipped tiles;
- each closed tile's metadata.

The net saving is 1 - rows/n. It is a share of KV reads, not of wall time (see the projection below).

## Measurements

### Headline: share of keys skippable at 385-512 keys (%)

The oracle columns are the ceiling for any bound, using exact weights and the exact error. The last two
columns are the maxima over every tiling, bin and bound.

| model | oracle fixed32 | fixed64 | fixed128 | turn | message | best key-free certified skip | oracle half-ulp |
|---|---|---|---|---|---|---|---|
| pick_e1_L16 | 1.09 | 0.74 | 0.48 | 0.64 | 0.91 | 0.0008 | 0.03 |
| pick_e5_s0 | 1.93 | 1.25 | 0.73 | 0.98 | 1.55 | 0.0008 | 0.02 |
| pick_e1_60m_L32 | 3.08 | 2.40 | 1.79 | 1.94 | 2.51 | 0.0113 | 0.29 |
| pk_sft_v1 | 1.28 | 0.78 | 0.45 | 0.87 | 1.27 | 0.0002 | 0.01 |

Oracle ROUNDING skip by context length (fixed32 / turn / message, %):

| model | <=128 keys | 129-256 | 257-384 |
|---|---|---|---|
| pick_e1_L16 | 0.25 / 0.25 / 0.42 | 0.61 / 0.52 / 0.80 | 0.95 / 0.62 / 0.93 |
| pick_e5_s0 | 0.59 / 0.45 / 0.86 | 1.12 / 0.83 / 1.39 | 1.83 / 1.05 / 1.74 |
| pick_e1_60m_L32 | 1.24 / 1.04 / 1.70 | 2.35 / 1.86 / 2.60 | 2.89 / 1.97 / 2.67 |
| pk_sft_v1 | 0.44 / 0.35 / 0.64 | 0.69 / 0.56 / 0.93 | 1.04 / 0.70 / 1.10 |

**Underflow tiles:** 0 in every cell, with denormals and with flush-to-zero alike. Across all models
and bins, 99.96-100% of closed 32-key tiles have their maximum within 30 nats of the row maximum.

The oracle grows with context (0.4-1 point per doubling) and with model depth: the 60M model at 32
layers is highest. It is not large anywhere.

**Net saving of the certified kernels at 385-512 keys** (every model alike, because nothing is
skipped):

| tiles | norm | pairnorm | ball | box | pairmin | box + value ranges | pairmin + value ranges |
|---|---|---|---|---|---|---|---|
| fixed32 | -0.0% | -0.8% | -1.5% | -3.0% | -3.8% | -6.0% | -6.8% |
| turn | -0.0% | -0.5% | -1.0% | -1.9% | -2.4% | -3.8% | -4.3% |

The exact-maximum control skips 0.2-1.2% of keys but loses 30-48% of reads, because it must read every
closed key to know its bound.

### Why: gaps, bound looseness and the approximate regime (fixed32, 385-512 keys)

| model | tiles within 5 / 10 / 15 nats of the row max | tiles with mass < 1e-5 | bound above the tile's true max, nats: norm / pairnorm / box / ball / pairmin | keys skippable at tail mass 1e-4 / 1e-3 / 1e-2 / 5e-2 |
|---|---|---|---|---|
| pick_e1_L16 | 0.67 / 0.95 / 0.995 | 1.6% | 17.6 / 21.3 / 13.0 / 10.7 / 12.4 | 3.2 / 9.0 / 24 / 44% |
| pick_e5_s0 | 0.57 / 0.92 / 0.992 | 2.8% | 20.4 / 24.6 / 15.2 / 12.2 / 14.4 | 5.8 / 15 / 33 / 54% |
| pick_e1_60m_L32 | 0.63 / 0.91 / 0.982 | 4.2% | 19.3 / 23.5 / 15.9 / 13.2 / 15.1 | 6.4 / 14 / 28 / 48% |
| pk_sft_v1 | 0.59 / 0.94 / 0.996 | 1.8% | 15.6 / 19.0 / 12.2 / 10.2 / 11.5 | 4.0 / 11 / 29 / 50% |

Turn tiles look the same: within 15 nats 0.979-0.994, looseness within 0.5 nats of fixed32.

Two separate obstacles, either one sufficient:

1. **The tails are fat.** A 32-key tile whose largest logit sits 15 nats under the row maximum is in the
   bottom 0.2-1.8% of tiles.
2. **Key-free bounds over 32 keys in 64 dimensions** overshoot the true maximum by 10-25 nats. The
   tightest is the ball, then pairmin. A tile would have to be about 25-30 nats under the row maximum
   to certify, and essentially none are (0.04% beyond 30 nats).

The last column shows where approximate methods live. Dropping 1% of the softmax mass would remove
24-33% of keys, and 5% would remove about half. That regime is what Quest, top-k and BLASST exploit,
and it is roughly three orders of magnitude of mass away from ROUNDING.

### Heads: local heads exist, and still do not certify (fixed32, 385-512 keys)

| model | heads | oracle skip per head: median / 90th pct / max | heads with oracle > 5% / > 20% | heads where a key-free certificate skips > 0.1% | oracle by depth (first / middle / last third) |
|---|---|---|---|---|---|
| pick_e1_L16 | 16x5 | 0.03% / 3.1% / 17.6% | 5 / 0 | 0 | 0.7 / 0.8 / 1.7% |
| pick_e5_s0 | 8x7 | 0.45% / 4.0% / 20.3% | 5 / 1 | 0 | 3.0 / 1.7 / 1.5% |
| pick_e1_60m_L32 | 32x6 | 0.42% / 7.1% / 49.9% | 27 / 7 | 2 | 4.9 / 2.3 / 2.3% |
| pk_sft_v1 | 12x12 | 0.10% / 3.0% / 26.3% | 11 / 2 | 0 | 2.5 / 0.3 / 1.0% |

A few heads are local:
- the strongest are early-layer heads: 60M 1.4, 1.2 and 2.2; pk_sft_v1 1.8 and 0.3; e5 0.4;
- there are some late ones: 60M 16.5, L16 7.0 and 14.3;
- on those heads the oracle can drop 18-50% of keys.

The median head is near zero. The local heads have large \|q\|, so the norm-based bounds are loosest
exactly there: at fixed32 the norm bound sits 20-56 nats above the true tile maximum on the probed heads
(for example 26.6 and 55.7 on 60M 1.2 and 2.2, and 31.3 and 37.3 on L16 11.4 and 11.3).

**Follow-up probe** (`kvskip_local_probe.py`, `receipts/local_probe.json`). The five most local heads
per model on the five 512-token contexts, with smaller tiles:

| head | tiles | oracle | best certified skip | net saving |
|---|---|---|---|---|
| pk_sft_v1 1.8 | fixed4 | 68% | 29% (pairmin, 3.6 nats loose) | -2.5% |
| pk_sft_v1 1.8 | fixed8 | 57% | 11% | -4.9% |
| pick_e1_60m_L32 1.4 | fixed4 | 84% | 32% | **+1.2%** |
| pick_e1_60m_L32 1.4 | fixed8 | | | -6.2% |
| pick_e1_60m_L32 16.5 | fixed4 | 57% | 18% | -13% |
| each other probed head | fixed4 | | 0-6% | -25% to -31% |

The one net gain is a single head of 192, and its metadata at 4 keys per tile is 1.25 rows per 4 keys.
Pairmin's looseness falls to 3.6-12 nats at 4 keys per tile, against 7-23 at 32. Smaller tiles tighten
the bound, but the metadata grows faster than the skips.

### Projected decode-time saving

At 384 keys the CUDA decode attention kernel takes 35.6 µs per layer on an L4. This is from
`ember.cpp/experiments/verify_2026-10-27/README.md`: 2 chunks of 256 keys, 5 heads, so 10 thread
blocks. For pick_e1_L16 that is 0.57 ms per step over 16 layers. Plain decode at chat lengths runs at
738 tok/s (1.35 ms per token), so at 384 keys attention is of the order of 40% of a step.

| case | keys skipped | effect on decode time |
|---|---|---|
| any measured key-free certificate | <= 0.011% | +0% to +9% more attention reads; a small slowdown |
| a perfect, free bound (the oracle) | 1.1% (L16) to 3.1% (60M) | at most about 0.5% of a pick_e1_L16 step and about 1.3% of a 60M step |

One more fact bears on the time. The K and V a pick_e1_L16 layer reads at 384 keys are
384 x 320 x 2 x 4 B = 0.98 MB, which is 3.3 µs at the L4's 300 GB/s. The measured 35.6 µs is about 11x
that floor. So the kernel is bound by latency and occupancy (10 blocks on 58 SMs, each walking 256 keys
in turn), not by bytes, and skipping bytes is the wrong lever there.

Smaller split-K chunks (e.g. 32-64 keys: 30-60 blocks) attack the 35.6 µs directly. That is a
hypothesis for ember to time on Colab; it was not measured here.

## Part 2. Prior art

Read for this note:
- **By hand:** BLASST's Algorithm 1 and Appendix B; the abstracts of EntmaxKV, Perturbed Retrieval,
  Attention-Mass Condensation and Certified Continuation.
- **By a delegated literature pass, summarised here:** the method sections of Quest, EntmaxKV,
  vAttention, Louver, Calver and Perturbed Retrieval.

The synthesis tool (`tools/synthesis`) was not run: its generate and scope stages call paid models and
it has no offline mode.

| work | what it does | guarantee, against what | bound read without the keys? | what we would add |
|---|---|---|---|---|
| Quest (Tang et al., ICML 2024, arXiv 2406.10774) | per-page key channel min/max, upper bound Σ max(q_i kmin_i, q_i kmax_i); top-K pages | none on the output: the bound only ranks pages | yes | the bound turned into an output certificate (our "box") |
| BLASST (Yuan et al., arXiv 2512.12087, v3 Apr 2026) | FlashAttention-style skip of a block when local max minus running max < ln λ; prefill and decode (1.48x decode at 73% sparsity, B200) | exact arithmetic: ‖y - ŷ‖ <= δ V_max < \|S\| B_c λ V_max (App. B, Z >= 1); the renormalisation term is called O(δ² V_max) and dropped | no: QK^T is computed first (Alg. 1, lines 6-9); decode saves V loads only | a decision made without reading the keys; the identity above (constant 2 in the norm form, or per-coordinate ranges); f32 rounding |
| EntmaxKV (Duarte, Couceiro, Treviso, arXiv 2605.21649) | Quest-style page scores, then sparse α-entmax | ‖o - õ‖ <= 2Bδ (tight); under entmax, box-selected pages contain the whole support, so the output is exact | yes | the softmax case, which they call "always a dense-tail approximation" |
| Certified Continuation (Ashok and Lua, NeurIPS 2026, OpenReview qas58udFE9) | upper bound on omitted mass, lower bound on retained mass, carried across decode steps; fail-closed to dense | deterministic, exact arithmetic only; output bound 11-13x looser than measured error; Qwen2.5-0.5B, 32K, zero violations, 2.8-7.0x fewer rows per head | not stated in the abstract; the PDF sits behind OpenReview's browser check and was **not read** | finite precision, a defined tolerance, short contexts, turn tiles |
| Attention-Mass Condensation (Ruiz Williams, arXiv 2602.06317) | the exact omitted-mass identity; a greedy-margin condition | a sufficient condition for an unchanged greedy token; 0 of 60 sparse runs equal to dense through 128 tokens | n/a | nothing: our certificate uses its identity |
| vAttention (Desai et al., ICLR 2026, arXiv 2510.05688) | sink, window, top-k, then random sampling of the rest | Pr(‖err‖ > ε‖SDPA‖) <= δ | n/a | a deterministic certificate |
| Certified Sparse Attention via Perturbed Retrieval (Li, arXiv 2610.24991, 18 Oct 2026) | Exp(1)-perturbed scores, Horvitz-Thompson estimate; uncertified heads fall back to dense | probabilistic per head and step; the candidate scan has deterministic, outward-rounded score bounds | no: the 4-bit scan "reads every key" | a key-free bound; a deterministic certificate |
| Calver (arXiv 2605.20868) | runtime-certified quantised attention, fail-closed ladder to dense FP16 | deterministic bound on quantisation error (E_key <= 2 V_max TV) | n/a (nothing skipped) | the template for metadata written at KV-write time with fail-closed dispatch |
| Louver (Dehghankar and Asudeh, arXiv 2605.06763) | sparse attention as half-space range search; per-cluster ball bounds per subspace | no false negatives for q·k >= τ; τ from sampling, no output bound | yes | an output certificate |
| ArkVale (NeurIPS 2024) | page digests (cuboid or sphere) to recall evicted pages | its sound sphere is loose under RoPE, so it uses a non-enclosing radius and gives the guarantee up (as VestigeKV 2609.03949 reports) | yes | |
| LeOPArd (ISCA 2022, arXiv 2204.03227), Token-Picker (2407.15131) | bit-serial q·k with a conservative margin; LeOPArd learns its thresholds in training | the threshold decision is exact, the output is not | partly | they already co-design thresholds at training time |
| InfLLM, Landmark Attention, MInference, RetrievalAttention, DuoAttention, ShadowKV, ClusterKV, PQCache, MagicPIG, Double Sparsity, SparQ, Loki, HashAttention, TokenSelect, RocketKV, Twilight, Tactic, Top-Theta, NSA, MoBA | representatives, landmarks, low rank, clusters, PQ, ANN, LSH sampling, calibrated or trained selection | none deterministic against dense (MagicPIG: an unbiased estimator) | mostly yes | |

### Already done

- **Key-free per-page upper bounds on q·k:** Quest (box), ArkVale (sphere), Louver (ball).
- **Tail mass times value size as an output bound:** BLASST (δ·V_max, with the renormalisation dropped)
  and EntmaxKV (2Bδ, tight).
- **The exact identity o - o_S = δ(μ_T - o_S):** 2602.06317.
- **Running-max block skipping:** BLASST and the SpargeAttn family; these read the keys.
- **Deterministic fail-closed mass certificates with dense fallback:** Certified Continuation, in exact
  arithmetic.
- **Probabilistic certificates with rounding-aware bounds:** Perturbed Retrieval.
- **Exact sparse decoding:** under entmax only (EntmaxKV).

### What remains ours

- A certificate against a **stated floating-point tolerance**. That means the IR's ROUNDING test with a
  valid lower bound on max(1, max\|ref\|), and the half-ulp and underflow conditions for bit equality.
  Every bound above is in exact arithmetic.
- **A conformance suite and a bit-exact engine.** In those, "passes the dense suite" is a gate with a
  pass/fail answer, not a KL figure.
- **Turn-structured caches.** Turn tiles cut at `<|endoftext|>`. No published scheme found uses them.
  They certified no better than fixed tiles here.
- **Training from scratch, so the bound can be co-designed.** No published work trains for certificate
  tightness. LeOPArd, NSA and MoBA train selection or thresholds instead.
- **A measured negative at small scale.** Exact certification is out of reach in the 20-100M,
  <= 512-token regime. The measured reasons are fat tails and bounds 10-25 nats loose.

## Synthesis

The strongest combination of published pieces:
1. Quest's box per tile, tightened by our pairmin and with the ball as an alternative (the lower of the
   bounds).
2. Per-tile value min/max written at KV-write time (Calver's practice for error metadata).
3. BLASST's online-softmax running statistics, taken over the tiles actually read.
4. The exact omitted-mass identity (2602.06317), giving the per-coordinate certificate
   |o_i - o_S,i| <= Σ_t (n_t e^{U_t - m_S}/Z_S) · max(vmax_t,i - o_S,i, o_S,i - vmin_t,i).
5. Perturbed Retrieval's outward rounding, so that the bound itself holds in f32.
6. Fail-closed dispatch to dense, from Certified Continuation and Calver.

**What the combination guarantees** is a deterministic, key-free, per-query bound on the output error
that is valid in f32, checked against the conformance tolerance. The sparse kernel then passes the dense
suite by construction, on every query rather than on average. No single piece gives this:
- Quest certifies nothing;
- BLASST reads the keys and ignores rounding;
- EntmaxKV certifies only under entmax;
- vAttention and Perturbed Retrieval certify only in probability;
- Certified Continuation certifies mass in exact arithmetic.

This note's `greedy_tile` certificate is that combination, less the outward rounding. It certifies
almost nothing on our models, so the synthesis is sound but has nothing to act on here.

**What co-design would have to achieve.** Certification needs closed tiles 11-17 nats under the row
maximum plus the bound's looseness. Today almost no tiles meet that: 0.4-1.8% are 15 nats under, and
looseness at 32 keys is 10-15 nats for the ball. Three routes would each change the arithmetic, and none
is tested here:
- **α-entmax attention trained from scratch.** EntmaxKV's exact-support result then applies, and the
  needed gap falls from about 15 nats to the entmax support width, about 2 score units at α = 1.5. The
  bound's looseness still dominates. The IR would also need an entmax op (D2), and its quality at
  20-100M is unknown.
- **QK-norm with a learned temperature.** This caps \|q\|·\|k\|, so norm bounds would no longer lose the
  20-56 nats they lose on the sharp heads.
- **A regulariser that widens inter-turn logit gaps.**

Each is a training-side bet with an unknown effect on quality. None would show in a decode-time gain
until contexts are long enough that bytes, not occupancy, set the attention time.

## Recommendation

1. **Do not build the kernel** for the 20-100M picks at <= 512 tokens. The ceiling with a perfect free
   bound is under 1.5% of a decode step, and every real key-free certificate is a net loss.
2. **Attack the 35.6 µs through occupancy first** (ember): split-K chunks of 32-64 keys instead of 256.
   The arithmetic above says the kernel is about 11x off its bandwidth floor at 384 keys. It needs timing
   on Colab.
3. **Revisit the idea only together with a model change**: entmax attention, or QK-norm with a learned
   temperature, at contexts of 2K or more. Before that, read Certified Continuation's method section. If
   its mass bound is key-free, the finite-precision certificate here is an extension of that work.

## Files

| file | contents |
|---|---|
| `kvskip_measure.py` | the measurement: contexts, reference capture, oracle, bounds, certificates. Resumable, with one checkpoint per model (`np.savez`, written atomically about every 2.5 minutes, per layer) |
| `kvskip_report.py` | the cost model and per-cell summaries |
| `kvskip_tables.py` | this note's tables and `receipts/tables.json` |
| `kvskip_local_probe.py` | the small-tile probe on the most local heads |
| `receipts/tables.json` | every cell (model x tiling x bin): oracle shares, mass thresholds, gap CDF, bound looseness, tail-mass skips, every bound x certificate's skip share and net saving, per-head values, the contexts |
| `receipts/tables_full.txt` | the full printed tables, including fixed64, fixed128 and message at every bin |
| `receipts/local_probe.json` | per probed head and tile size: oracle, gap, per-bound looseness, skip and net saving |

Reproduce from this folder:

    python kvskip_measure.py MODEL --ckpt ck_MODEL.npz
    python kvskip_tables.py receipts/tables.json ck_*.npz
    python kvskip_local_probe.py local_probe.json receipts/tables.json MODEL ...

Each job runs through `scripts/cpu_budget.py`, one core per model chain. At one core a model takes
15-50 minutes.
