"""The vocabulary of the synthesis search.

The move that produced "diffusion inside the KV cache" (16 Sep 2026) was
not a random pairing of two words. It was:

  1. name a COMPONENT of the transformer system and a DEFICIT it has by
     construction (the KV cache is append-only: nothing written at step
     100 can be revised by what is learnt at step 900);
  2. find a MECHANISM from elsewhere in ML, mathematics, neuroscience or
     systems whose NATIVE OPERATION negates that deficit (masked
     denoising rewrites entries from bidirectional context);
  3. state the INVARIANT that keeps the system valid after the graft
     (outputs stay causal because edits only affect future reads);
  4. state the FALSIFIER at nano scale.

This file encodes steps 1 and 2 as data so the pairing can be enumerated
instead of waited for. Deficits carry the operations that would negate
them; mechanisms carry the operations they perform natively. A candidate
is any (component, deficit, mechanism) with a non-empty intersection.
The language model then does step 3 and 4 for each candidate, and the
scoper checks the literature.

Everything here is a prior, not a claim. Add rows freely; the scoper is
what separates the open from the taken.
"""

# Operations. A deficit is negated by some of these; a mechanism performs
# some of these natively. Keep the set small so intersections are dense.
OPS = [
    "rewrite",        # revise stored state in place from later context
    "erase",          # remove or overwrite stored state
    "consolidate",    # compress many entries into fewer, keeping what predicts
    "denoise",        # iterative refinement of a noisy/partial estimate
    "calibrate",      # make scores mean what they say (probabilities)
    "verify",         # check a cheap proposal against an exact criterion
    "speculate",      # act on a cheap guess, correct later
    "parallelise",    # replace a sequential dependency with a parallel one
    "select-set",     # choose a subset with interactions, not top-k of scores
    "adapt",          # change at test time from the input stream
    "forget",         # decay or evict by a learned/derived rule
    "feedback",       # closed-loop control from a measured error
    "allocate",       # distribute a budget (precision, compute, memory) unevenly
    "abstain",        # produce "no answer" as a first-class output
    "multi-horizon",  # optimise beyond the next step
    "share",          # reuse state across requests, layers or heads
    "compress",       # reduce size with a fidelity guarantee
    "sparsify",       # keep a structured subset
    "retrieve",       # content-addressed lookup
    "replay",         # revisit past inputs or states on a schedule
    "aggregate-interactions",  # combine scores with pairwise interaction terms
    "transport",      # move mass between distributions optimally
    "resample",       # importance-weighted rebirth of candidates
    "schedule",       # decide when, not what
    "route",          # decide where
    "quantise",       # discretise with awareness of downstream error
    "structure",      # impose a graph/tree/order on unstructured state
]

# Components of a transformer language-model system, training and
# inference, each with the deficits it has BY CONSTRUCTION. A deficit is
# a sentence a reviewer would accept as true of the vanilla system, plus
# the ops that would negate it.
COMPONENTS = {
    "kv-cache": {
        "what": "the per-layer key/value entries written once per token during prefill and decode",
        "deficits": [
            ("append-only: an entry written at step t is never revised by later context", ["rewrite", "denoise", "consolidate"]),
            ("uniform precision and cost per entry regardless of how much it is attended to", ["allocate", "quantise", "compress"]),
            ("no principled forgetting: eviction is a heuristic bolted on at inference", ["forget", "consolidate"]),
            ("private to one request: identical prefixes are recomputed", ["share", "retrieve"]),
        ],
    },
    "attention-scores": {
        "what": "the softmax-normalised query-key similarities that mix values",
        "deficits": [
            ("sums to one: attention cannot abstain, so a token with nothing relevant still mixes something", ["abstain", "structure"]),
            ("heads are scored independently; no term for redundancy or complementarity between heads", ["aggregate-interactions", "select-set"]),
            ("similarity is bilinear; no notion of transport cost between query and key sets", ["transport", "structure"]),
        ],
    },
    "moe-router": {
        "what": "the linear gate that scores experts per token and picks top-k",
        "deficits": [
            ("experts are scored independently, so top-k cannot express that two experts are redundant or complementary", ["aggregate-interactions", "select-set"]),
            ("router scores are not calibrated: a 0.9 gate value does not mean the expert is right nine times in ten", ["calibrate", "verify"]),
            ("routing is open-loop: no feedback from the expert's downstream error to the gate at inference", ["feedback", "adapt"]),
            ("load balancing is imposed by an auxiliary loss that fights the quality objective", ["transport", "schedule"]),
        ],
    },
    "residual-stream": {
        "what": "the additive skip pathway every block writes into",
        "deficits": [
            ("additive only: a block can add a direction but never erase what an earlier block wrote", ["erase", "rewrite"]),
            ("one width for every depth and every token, however much or little the token needs", ["allocate", "adapt"]),
        ],
    },
    "positional-encoding": {
        "what": "the function of index that tells attention where tokens are",
        "deficits": [
            ("a fixed function of the integer index: position is never a function of content or of what the model has already learnt about the sequence", ["adapt", "structure"]),
            ("extrapolation beyond the training length degrades because the encoding was never trained there", ["adapt", "calibrate"]),
        ],
    },
    "tokeniser": {
        "what": "the fixed subword vocabulary and merge rules",
        "deficits": [
            ("static: granularity cannot change at inference for easy or hard spans", ["adapt", "allocate", "compress"]),
            ("frequency-driven merges, not prediction-driven: the vocabulary is chosen before the model exists", ["feedback", "multi-horizon"]),
        ],
    },
    "output-head": {
        "what": "the unembedding and logits over the vocabulary",
        "deficits": [
            ("after preference post-training the logits are no longer calibrated probabilities", ["calibrate", "verify"]),
            ("no first-class abstention: the model must emit some token", ["abstain"]),
            ("a one-token horizon: the head is trained to predict only the next token", ["multi-horizon", "speculate"]),
        ],
    },
    "sampler": {
        "what": "the decoding loop that turns logits into tokens",
        "deficits": [
            ("strictly sequential: one token per forward pass", ["parallelise", "speculate", "denoise"]),
            ("no rollback: a committed token is never revised", ["rewrite", "resample", "verify"]),
            ("temperature and top-k are global knobs, not per-position decisions", ["adapt", "allocate", "schedule"]),
        ],
    },
    "prefill": {
        "what": "the forward pass over the prompt that builds the cache before the first token",
        "deficits": [
            ("must complete exactly before any token is emitted; time-to-first-token is bounded below by it", ["speculate", "verify", "parallelise"]),
            ("quadratic in prompt length with no reuse of structure across prompts", ["compress", "share", "sparsify"]),
        ],
    },
    "speculative-drafting": {
        "what": "the small draft model whose proposals the target verifies",
        "deficits": [
            ("operates on tokens only: nothing else in the system (cache, routing, layers) is speculated on", ["speculate", "verify"]),
            ("the draft length is fixed rather than chosen from the draft's own confidence", ["calibrate", "schedule", "adapt"]),
        ],
    },
    "layer-schedule": {
        "what": "the fixed sequence of blocks every token passes through",
        "deficits": [
            ("every token gets the same depth regardless of difficulty", ["allocate", "adapt", "schedule"]),
            ("no loop: a block cannot be re-applied when its own output says it should", ["feedback", "denoise", "replay"]),
        ],
    },
    "ffn": {
        "what": "the per-token feed-forward block, read as a key-value memory",
        "deficits": [
            ("dense: every key is scored for every token", ["sparsify", "retrieve", "route"]),
            ("its memory is fixed at training time and never edited by the stream", ["adapt", "rewrite", "erase"]),
        ],
    },
    "normalisation": {
        "what": "per-token normalisation of activations",
        "deficits": [
            ("statistics are per token; nothing is normalised across the sequence or across the batch at inference", ["structure", "share"]),
        ],
    },
    "optimiser-state": {
        "what": "the per-parameter moment estimates kept during training",
        "deficits": [
            ("per-parameter and structureless: the moments know nothing of the module they belong to", ["structure", "compress"]),
            ("discarded at the end of training although it encodes curvature the model could use later", ["share", "retrieve", "quantise"]),
        ],
    },
    "lr-schedule": {
        "what": "the predetermined learning-rate curve",
        "deficits": [
            ("open-loop: it never reads the loss curve it is producing", ["feedback", "adapt", "schedule"]),
        ],
    },
    "training-objective": {
        "what": "next-token cross-entropy",
        "deficits": [
            ("single horizon: no term rewards a representation for what it will predict ten tokens on", ["multi-horizon", "consolidate"]),
            ("no calibration term: probabilities are a by-product, not a target", ["calibrate"]),
        ],
    },
    "data-order": {
        "what": "the random order in which training examples arrive",
        "deficits": [
            ("no replay: an example seen once is never revisited on the model's own terms", ["replay", "schedule", "resample"]),
            ("the mixture is fixed in advance rather than chosen from the model's current errors", ["feedback", "adapt", "allocate"]),
        ],
    },
    "weights-at-inference": {
        "what": "the frozen parameters at serving time",
        "deficits": [
            ("static: nothing in the input stream changes the parameters", ["adapt", "rewrite"]),
        ],
    },
    "attention-topology": {
        "what": "the fixed mask deciding which positions may attend to which",
        "deficits": [
            ("the pattern is fixed before the content is seen", ["adapt", "structure", "sparsify"]),
            ("causal masking forbids revising a representation once its later context arrives", ["rewrite", "denoise"]),
        ],
    },
    "gradient": {
        "what": "the single backward pass that produces the update",
        "deficits": [
            ("noisy and unverified: no cheap check that the step will reduce the loss", ["denoise", "verify", "resample"]),
        ],
    },
    "checkpoints": {
        "what": "the discrete snapshots saved during training",
        "deficits": [
            ("used only for resumption; the trajectory between snapshots is thrown away", ["consolidate", "retrieve", "share"]),
        ],
    },
    "context-window": {
        "what": "the hard maximum number of positions",
        "deficits": [
            ("uniform cost per position however little each contributes", ["allocate", "compress", "forget"]),
        ],
    },
    "multi-head": {
        "what": "the fixed set of parallel attention heads",
        "deficits": [
            ("heads are always all on; there is no routing among heads", ["route", "select-set", "sparsify"]),
        ],
    },
    "serving-batch": {
        "what": "the set of concurrent requests sharing a GPU",
        "deficits": [
            ("requests are independent: no state, cache or computation is shared across them", ["share", "retrieve", "consolidate"]),
        ],
    },
}

# Mechanisms from OUTSIDE the component's home, with their native ops and a
# one-line statement of what they do. Sources: other parts of ML, maths,
# statistics, neuroscience, systems, signal processing.
MECHANISMS = {
    "masked-diffusion-denoising": ("iteratively refine masked or noised positions from bidirectional context", ["denoise", "rewrite", "parallelise"]),
    "delta-rule-memory": ("erase the old value at a key before writing the new one (fast-weight associative memory)", ["erase", "rewrite", "adapt"]),
    "choquet-integral": ("aggregate scores under a fuzzy measure with pairwise interaction terms", ["aggregate-interactions", "select-set"]),
    "submodular-maximisation": ("greedy selection of a set with diminishing returns and a guarantee", ["select-set", "sparsify"]),
    "conformal-prediction": ("distribution-free prediction sets with a coverage guarantee", ["calibrate", "abstain", "verify"]),
    "proper-scoring-rules": ("train probabilities to be calibrated by scoring them on outcomes", ["calibrate"]),
    "speculate-and-verify": ("act on a cheap proposal, then verify against the exact computation and roll back", ["speculate", "verify", "parallelise"]),
    "online-self-distillation": ("a slow copy teaches a fast copy from the same stream", ["consolidate", "share", "adapt"]),
    "ema-averaging": ("an exponential average of a trajectory as a better estimate than its endpoint", ["consolidate", "denoise"]),
    "contrastive-objective": ("pull positives together, push negatives apart", ["structure", "multi-horizon"]),
    "energy-based-refinement": ("descend an energy over candidates rather than sample once", ["denoise", "verify", "resample"]),
    "optimal-transport-sinkhorn": ("match two sets under a cost with balanced marginals", ["transport", "route", "select-set"]),
    "product-quantisation": ("split vectors into subspaces and quantise each with a codebook", ["compress", "quantise", "retrieve"]),
    "hopfield-retrieval": ("content-addressed recall by energy descent over stored patterns", ["retrieve", "denoise"]),
    "sequential-monte-carlo": ("keep a population of hypotheses, weight them, resample", ["resample", "speculate", "multi-horizon"]),
    "test-time-training": ("update a small set of parameters on the incoming stream with a self-supervised loss", ["adapt", "rewrite"]),
    "kalman-filtering": ("fuse a prediction with a measurement in proportion to their uncertainties", ["denoise", "feedback", "calibrate"]),
    "quantisation-aware-training": ("train through the rounding so the model expects it", ["quantise", "allocate"]),
    "experience-replay": ("revisit stored past experiences on a prioritised schedule", ["replay", "schedule", "consolidate"]),
    "evolutionary-population": ("mutate, evaluate, select over a population", ["resample", "select-set"]),
    "feedback-control": ("drive a measured error to zero with proportional, integral and derivative terms", ["feedback", "schedule", "adapt"]),
    "sparse-coding": ("represent an input as a sparse combination of dictionary atoms", ["sparsify", "structure", "compress"]),
    "spectral-transform": ("work in a frequency basis where long-range structure is local", ["structure", "compress", "parallelise"]),
    "graph-message-passing": ("update node states from neighbours over an explicit graph", ["structure", "rewrite"]),
    "knn-retrieval": ("look up nearest stored neighbours and mix them in", ["retrieve", "share"]),
    "sleep-replay-consolidation": ("offline reactivation that compresses episodic traces into a stable store", ["consolidate", "replay", "forget"]),
    "homeostatic-plasticity": ("scale activity to keep a target statistic constant", ["feedback", "allocate"]),
    "predictive-coding": ("propagate prediction errors, not signals, between levels", ["feedback", "denoise", "multi-horizon"]),
    "bandits-thompson": ("choose actions by sampling from a posterior over their value", ["schedule", "route", "calibrate"]),
    "bayesian-model-averaging": ("weight hypotheses by posterior mass instead of picking one", ["aggregate-interactions", "calibrate"]),
    "error-correcting-codes": ("add structured redundancy so corruption can be detected and repaired", ["verify", "denoise", "compress"]),
    "generational-garbage-collection": ("age-partition memory and collect young objects often, old ones rarely", ["forget", "consolidate", "schedule"]),
    "branch-prediction": ("guess the outcome of a control decision from history and speculatively proceed", ["speculate", "verify", "schedule"]),
    "cache-replacement-arc": ("adaptive replacement policies that balance recency and frequency", ["forget", "allocate", "adapt"]),
    "arithmetic-coding": ("compress a stream to its model's entropy", ["compress", "verify"]),
    "learned-index": ("replace a lookup structure with a model that predicts position", ["retrieve", "structure", "compress"]),
    "transactions-and-rollback": ("group operations so they commit atomically or not at all", ["verify", "rewrite", "speculate"]),
    "jit-tracing": ("record a hot path once and specialise it", ["share", "compress", "schedule"]),
    "information-bottleneck": ("keep what predicts the target, discard the rest, explicitly", ["consolidate", "compress", "forget"]),
    "minimum-description-length": ("prefer the representation that shortens the total code", ["compress", "select-set", "structure"]),
    "renormalisation-group": ("coarse-grain and ask what survives the change of scale", ["consolidate", "structure", "compress"]),
    "control-barrier-functions": ("constrain actions so a safety set is never left", ["verify", "abstain", "feedback"]),
    "importance-sampling": ("reweight samples from the wrong distribution to estimate the right one", ["resample", "calibrate", "allocate"]),
    "hashing-lsh": ("bucket by locality-sensitive hashes so similar items collide", ["retrieve", "sparsify", "route"]),
    "curriculum-self-paced": ("order examples by the learner's own current loss", ["schedule", "replay", "allocate"]),
    "mixture-density-heads": ("predict a mixture of distributions rather than a point", ["calibrate", "multi-horizon", "abstain"]),
    "wavelet-multiresolution": ("represent a signal at several resolutions at once", ["structure", "allocate", "compress"]),
    "consensus-protocols": ("agree on a value among unreliable parties by voting rounds", ["verify", "aggregate-interactions", "abstain"]),
    "hebbian-fast-weights": ("outer-product updates that store recent associations for a few steps", ["adapt", "rewrite", "forget"]),
    "dual-process-gating": ("a fast reflex path with a slow deliberate path that overrides it on demand", ["route", "schedule", "abstain"]),
}

INVARIANTS = [
    "outputs remain causal (nothing emitted depends on a token after it)",
    "exactness is recoverable: any approximation is verified and the exact result can be restored",
    "matched FLOPs and parameters against the baseline",
    "no retraining of the base model (graft trains alone) OR from-scratch at matched budget, stated which",
    "streaming: works token by token without seeing the whole sequence",
]

# What this team can actually test. Feasibility is scored against this.
TESTBED = """paperkiln (C++ trainer, gpt2/llama nano presets, exact / sliding-window / sink attention lanes, checkpoint/resume, events.jsonl receipts) on Colab L4s, three concurrent sessions, budget about 40 GPU-hours per study; PyTorch nano-scale scripts also acceptable. Twelve seeds, pre-registered falsifier, matched-FLOPs baseline. No cluster, no >1B models, no RLHF-scale post-training."""


# Search backstops. The 16 Sep 2026 calibration failure: the query writer
# searched by mechanism name ("diffusion", "denoising") and missed the
# paper that does the same thing under "cache consolidation". So every
# scope also runs deterministic fielded queries built from the component's
# aliases and the negating ops' synonyms, with no mechanism word at all.
ALIASES = {
    "kv-cache": ["KV cache", "key-value cache"],
    "attention-scores": ["attention weights", "softmax attention"],
    "moe-router": ["mixture of experts", "MoE routing"],
    "residual-stream": ["residual stream", "residual connections"],
    "positional-encoding": ["positional encoding", "position embeddings"],
    "tokeniser": ["tokenizer", "subword vocabulary"],
    "output-head": ["language model logits", "output distribution"],
    "sampler": ["decoding", "autoregressive sampling"],
    "prefill": ["prefill", "time to first token"],
    "speculative-drafting": ["speculative decoding", "draft model"],
    "layer-schedule": ["adaptive depth", "layer skipping"],
    "ffn": ["feed-forward network", "MLP layers"],
    "normalisation": ["layer normalization", "normalization layer"],
    "optimiser-state": ["optimizer state", "Adam moments"],
    "lr-schedule": ["learning rate schedule", "learning rate"],
    "training-objective": ["next-token prediction", "training objective"],
    "data-order": ["data ordering", "curriculum"],
    "weights-at-inference": ["test-time adaptation", "inference-time weight update"],
    "attention-topology": ["attention mask", "sparse attention pattern"],
    "gradient": ["gradient estimate", "gradient noise"],
    "checkpoints": ["checkpoint averaging", "training trajectory"],
    "context-window": ["long context", "context length"],
    "multi-head": ["attention heads", "head pruning"],
    "serving-batch": ["LLM serving", "batched inference"],
}

OP_SYNONYMS = {
    "rewrite": ["rewrite", "revise", "refine", "update in place", "non-causal"],
    "erase": ["erase", "overwrite", "unlearn"],
    "consolidate": ["consolidation", "consolidate", "summarize", "distill"],
    "denoise": ["denoising", "refinement", "iterative refinement"],
    "calibrate": ["calibration", "calibrated"],
    "verify": ["verification", "verify", "lossless"],
    "speculate": ["speculative", "speculation"],
    "parallelise": ["parallel", "non-autoregressive"],
    "select-set": ["subset selection", "submodular", "diversity"],
    "adapt": ["adaptive", "test-time", "online"],
    "forget": ["forgetting", "eviction", "decay"],
    "feedback": ["feedback", "closed-loop", "control"],
    "allocate": ["allocation", "budget", "adaptive precision"],
    "abstain": ["abstention", "abstain", "reject option"],
    "multi-horizon": ["multi-token prediction", "lookahead", "future"],
    "share": ["sharing", "reuse", "shared"],
    "compress": ["compression", "compress"],
    "sparsify": ["sparse", "sparsification", "pruning"],
    "retrieve": ["retrieval", "nearest neighbor", "lookup"],
    "replay": ["replay", "rehearsal", "revisit"],
    "aggregate-interactions": ["interaction", "non-additive", "pairwise"],
    "transport": ["optimal transport", "Sinkhorn", "assignment"],
    "resample": ["resampling", "particle", "population"],
    "schedule": ["schedule", "scheduling", "when to"],
    "route": ["routing", "gating"],
    "quantise": ["quantization", "quantized", "low-bit"],
    "structure": ["structured", "graph", "hierarchical"],
}
