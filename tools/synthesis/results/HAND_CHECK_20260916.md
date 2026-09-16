# Hand check of the top 45 OPEN ideas, 16 September 2026

The scoper's OPEN is a filter verdict. This is the human pass over the
45 highest-rated OPEN ideas from the first full run (318 scoped: 117
OPEN, 129 ADJACENT, 72 TAKEN; 37 unscoped when the API credit ran out).
Reader: Claude, from memory of the literature, with two suspicions
verified against arXiv. Jonathan has not yet read these.

## Calibration findings from this pass

Two of the 45 are taken by papers the scoper did not retrieve:

- **PID-controlled expert capacity** (moe-router x feedback-control):
  DeepSeek's *Auxiliary-Loss-Free Load Balancing Strategy for
  Mixture-of-Experts* (arXiv 2408.15664, Aug 2024) adds a per-expert bias
  to the router logits, updated from the load error, with no auxiliary
  loss. That is a proportional (and effectively integral) controller.
  TAKEN. The scoper missed it because no query said "load balancing".
- **Hebbian fast-weight FFN overlay** (ffn x hebbian-fast-weights):
  *Meta-Learning Fast Weight Language Models* (Clark et al., arXiv
  2212.02475) puts a fast-weight layer on the LM's FFN side; *Fast-weight
  Product Key Memory* (2601.00671) is the 2026 version. TAKEN.

Both are now negative controls in `pipeline.py`, and `vocab.py` carries
the field's own terms for every deficit (`FIELD_TERMS`), used as extra
backstop queries. Rerun `scope` on the OPEN set once credit exists.

Several more are adjacent on my reading and were called OPEN by the
judge; they are marked below. The false-OPEN rate at the top of the
list is material: treat OPEN as "worth a hand search", nothing more.

## Verdicts

Format: name; my verdict; reason.

**Survive a hand check (worth a proper scoping pass and, if that holds, a pre-registration):**

1. **Atomic-commit fast-weight edits** (weights-at-inference x
   transactions-and-rollback). Stage a candidate weight edit from the
   stream, validate it on the last m tokens against the frozen weights,
   commit only if it helps, else discard. The invariant is the point:
   test-time adaptation that can never do worse than the frozen model.
   Nearest: TTT layers (Sun et al. 2024), "Test-Time Training Provably
   Improves Transformers as In-context Learners" (2503.11842). Neither
   verifies before committing. Feasible at nano scale; the falsifier is
   clean (worst-case regression versus always-commit TTT).
2. **Multi-lane consensus prefill** (prefill x consensus-protocols).
   Several cheap prefill lanes (window, sink, low-rank) vote on the first
   token; emit on quorum; exact prefill runs as arbiter and corrects on
   disagreement. This is the lossless-speculative-prefill direction with
   the verification made cheap by agreement. Nearest: KV Prediction for
   TTFT (2410.08391), SpecPrefill, LazyLLM (all lossy or single-draft).
   ember.cpp is the natural testbed; the metric (TTFT at zero
   first-token error) is one Jonathan named as a target.

**Maybe (cheap to test, modest splash):**

3. **Delta-rule fast-weight router with in-context error feedback**
   (moe-router x delta-rule-memory). A router that learns within a
   sequence from realised loss attributable to the chosen expert.
   Nearest: HeRo history-aware routing (2609.08189), which is for depth
   skipping. Open-ish; MoE noise at nano scale is the risk.
4. **Test-time position scalars** (positional-encoding x
   test-time-training). Online gradient steps on a handful of RoPE or
   ALiBi scalars from the stream's own next-token loss. Very feasible;
   the length-extrapolation field is crowded (DAPE, CARoPE, GAPE, LaMPE)
   so the splash is low even if it works.
5. **Predictive-coding expert routing** (route on the residual not
   explained by the previous layer's expert). Adjacent to "Share First,
   Route What Remains" (2608.10392), which routes the remainder after a
   shared expert. Probably a follow-up.

**Adjacent or taken on my reading (judge said OPEN):**

- Periodic bidirectional residual denoising: Bottlenecked Transformers
  (2505.16950) moved from the KV cache to the residual stream; the
  residual at past positions is what the cache is computed from. Adjacent.
- TTT gate on residual; Transactional block commit; Per-token TTT
  residual gate: all forms of a gated residual (highway networks, 2015)
  with a new trigger. Adjacent.
- Fast-weight residual capacity buffer: fast weight programmers
  (Schlag et al. 2021; "Going Beyond Linear Transformers with Recurrent
  Fast Weight Programmers" 2106.06295). Adjacent, judge's 0.9 confidence
  notwithstanding.
- Delta-rule FFN memory edits: Fast-weight Product Key Memory
  (2601.00671). Taken or adjacent.
- Conformal abstention mass in attention: attention sinks already give
  abstention (StreamingLLM; "attention is off by one"); the conformal
  threshold is decoration on a null key. Adjacent.
- Mixture-density attention abstention; Mixture-density next-token
  head; Multi-horizon mixture-density heads: mixture of softmaxes
  (Yang et al. 2017) and multi-token prediction (Gloeckle et al. 2024).
  Adjacent.
- CBF-constrained expert routing; Thompson-sampled expert selection;
  dual-process router override; Kalman-fused router logits: the
  no-starvation capability already exists by construction in
  expert-choice routing (Zhou et al. 2022) and loss-free balancing
  (2408.15664); the rest are control laws on the same bias. Adjacent.
- Kalman-gated block re-entry; hebbian fast-weight depth gate: adaptive
  computation time and mixture of depths with a new halting signal.
  Adjacent.
- submodular content-aware sparse mask: content-based sparse attention
  (Routing Transformer, Reformer, SeerAttention, FlexPrefill); the
  (1-1/e) guarantee is for an objective chosen to make it hold. Adjacent.
- spectral prefill attention: FNet, Hyena, long convolutions. Adjacent.
- RG coarse-graining tokeniser: hierarchical and byte-patch tokenisers
  (MEGABYTE, Byte Latent Transformer, H-Net). Adjacent.
- Delta-rule content-adaptive positions; delta-rule position memory
  beyond L; Hebbian position cache; Hebbian position refresh; sparse
  dictionary positions; RG multiscale positions; MDN positional decoder;
  Kalman-fused positional estimate; graph-propagated position state;
  TTT content positions: the positional-encoding column is where the
  generator found the most OPENs (13) because the mechanisms are exotic
  there, but the target (length extrapolation, content-dependent
  position) is one of the most crowded problems in the field. None of
  these is a splash; at most one is a workshop paper.
- Redundant-code cache repair: an ECC framing on a learned drift
  detector; the parity is decoration. Weak.
- Graph-message-passing FFN memory: for model editing; nearest HoReN
  (2605.08143), memory networks. Adjacent.
- spectral cross-sequence normalisation; Thompson LR bandit; IB
  auxiliary future head; PID retokenisation; Multiresolution residual
  channels; fast-weight sampling memory (a repetition penalty with decay):
  small ideas; adjacent or tuning tricks.

## What this pass says about the tool

- The generator's splash and feasibility self-ratings are usable as a
  first sort, and inflated by about one point on splash.
- The judge is honest about its evidence (the quoted phrases were real)
  but its recall depends entirely on what the queries retrieved. The two
  confirmed misses were both cases where the field has a name for the
  problem that neither the deficit text nor the mechanism name contains.
  `FIELD_TERMS` is the fix; it needs a rerun to be measured.
- The yield after a hand check is two ideas worth a scoping pass out of
  45 OPENs, from 318 scoped, from 575 enumerated. That is the honest
  rate for this vocabulary. A second vocabulary pass (more components
  from outside the transformer block: data pipeline, tokenizer training,
  distillation, serving; more mechanisms from optimisation and
  statistics) is the cheapest way to raise it.
