# Positioning map: paperkiln, keramos and ember against prior work (26 Oct 2026)

*A map, not a verdict. Each claim is stated as the receipts in this workspace support it, then set
against the nearest work found. OPEN means targeted searches found nothing that does it; that is a
search result, not a proof, and the searches are listed with each row. ADJACENT means close work exists
and the difference can be stated in one sentence. TAKEN means the capability exists elsewhere; it may
still be useful here, but it is not a novelty claim. This builds on `keramos/IR_PRIOR_ART.md` (25 Oct),
which surveys ONNX, StableHLO, TOSA, NNEF, TVM, Core ML, LiteRT, torch.export, GGUF and the serving
engines property by property; it is not repeated here, only extended with 2025-2026 work found today.*

## 1. What is open

Five claims survive the search. Each is a combination whose parts exist separately; the combination
is what nobody was found doing.

| # | Claim (as the receipts support it) | Status | Nearest prior work | The precise difference |
|---|---|---|---|---|
| O1 | Decode state as normative spec text: KV rows, window and sinks, eviction, recurrent state, snapshot and rewind, "prefill = token-by-token decode" as a conformance condition, with each statement classed EXACT or ROUNDING, checked by a black-box suite against several engines (keramos v1.2: 370 cases; numpy 362/0/8, the clean-room Rust 330/0/40, mtstudio 139/0/231) | OPEN | ONNX `Attention` (past_key concatenation) and ORT's `GroupQueryAttention` contrib op (cache *layout* independent of step split); TVM `KVState` (a C++ API); llama.cpp `test-backend-ops` (per-op NMSE, no state); "The Illusion of Equivalence" ([2604.15409](https://arxiv.org/abs/2604.15409)): cache-on vs cache-off divergence measured in FP16 on three HF models, gone in FP32 | Prior work either has state as runtime code or measures its divergence after the fact. Nobody states what a decode step must keep and what it must equal, per op, and tests engines against that statement |
| O2 | Routing causality as a servability rule: `depth_route` (Mixture-of-Recursions) decided per token from the prefix, served with per-application caches, on CPU, CUDA, WASM and WebGPU | OPEN | Mixture-of-Depths ([2404.02258](https://arxiv.org/abs/2404.02258)) notes top-k is non-causal; MoR ([2507.10524](https://arxiv.org/abs/2507.10524)) treats leakage as a training choice | No engine or format found that serves MoR; no format that states the causality constraint (search today found no MoR in llama.cpp or vLLM; repeat in the trackers before printing) |
| O3 | A spec-completeness experiment by clean room: a third engine written by an agent from the spec text alone, with a read log as the isolation receipt; 276/0/8 at v1.1.4, 23 ranked findings, all answered in revision 1.2 | OPEN as applied to an ML model specification; the method is ADJACENT | Wasm's "two or more VMs" phase rule; Watt's mechanisation finding Wasm spec errors (CPP 2018); SpecBench ([awesomepapers 2605.30314](https://awesomepapers.io/papers/2605.30314)): agents find gaps in RFCs by reading, best 44.4%; FDE-Bench ([2609.27571](https://arxiv.org/abs/2609.27571)): 63 behaviours no spec stated; the microservice-generation study ([2508.20119](https://arxiv.org/abs/2508.20119)) fixed its specs when generated code exposed them | Prior work finds gaps by reading or by one generation loop. Here an implementation built from text alone is scored by a conformance suite, so each gap is a failing or undecided case, not an opinion. It is the same author's spec and agent, so it measures completeness, not independent authorship, and must be reported that way |
| O4 | Paper to a verified, spec-conformant artefact: one Claude call returns the architecture in IR terms with a verbatim quote and location per value; a deterministic verifier asserts only quote-checked values and refuses tier-3 ops; the chain trains, exports a GGUF with the IR embedded, and gates ember against the trainer (GoLU, ReLU², Primer's norm, DyT: 10/10 or 40/40 greedy) | OPEN as a combination; parts ADJACENT | Paper2Code / PaperCoder (ICLR 2026, [2504.17192](https://arxiv.org/abs/2504.17192)) and SciReplicate-Bench ([2504.00255](https://arxiv.org/abs/2504.00255), 39% best) produce code; PaperBench ([2504.01848](https://arxiv.org/abs/2504.01848), 21%) grades replications by rubric; Genesys ([2506.20249](https://arxiv.org/abs/2506.20249)) and ASI-Arch search architectures as PyTorch code with no export contract; HyperPIE ([2312.10638](https://arxiv.org/abs/2312.10638)) extracts hyperparameters without verbatim provenance; LangExtract does grounded extraction in general | The output is a validated document two engines run identically, and every value is either quote-verified or refused. Limits to state with it: the 40-paper benchmark is labelled by the same author; four papers have gone end to end; reply quality is not a goal; a quote that is valid but meant another model (Primer, 22 Oct) is not caught by the verifier |
| O5 | Multi-turn carry-over measured on 5M-150M models trained from scratch on human-written text only, against single-turn twins, with log-probability scoring (Carry-5, Carry-12) | OPEN as a measurement; the mechanisms tested (state tokens, attention residuals, recursion) are TAKEN | `experiments/tiny_chat_litcheck_2026-10-23.md`; nanochat ([repo](https://github.com/karpathy/nanochat)) trains end to end but fine-tunes on SmolTalk, which is LLM-generated, with no carry-over test; μLMs ([2604.19642](https://arxiv.org/abs/2604.19642), 8-30M) are evaluated on single openers; BabyLM's Interaction track learns from a teacher LLM; MT-Eval and MultiChallenge measure recollection on large models; Beyond Goldfish Memory ([2107.07567](https://arxiv.org/abs/2107.07567)) is human multi-session data, not a tiny-model probe | The scale and the probe are the contribution. This is the live programme (section 11 item 1 of `LOCAL_LLM_STACK.md`) and needs no new direction here |

Searches behind this section (26 Oct 2026, web and arXiv): conformance suites for prefill/decode
equivalence in LLM engines; declarative GGUF or JSON compute-graph proposals; mechanised (Lean, Coq)
proofs of KV-cache equivalence; MoR serving in llama.cpp and vLLM; agents implementing a specification
from text and conformance-testing it; LLM extraction of architectures with verbatim evidence; tiny
chat models on human-only data with multi-turn memory evaluation. The only declarative-graph proposal
for GGUF remains llama.cpp discussion [#21496](https://github.com/ggml-org/llama.cpp/discussions/21496)
(April 2026: Qwen2.5, Qwen3, Gemma2; no SSM or hybrid models; no numerics; no maintainer reply). No
mechanised proof of decode = prefill was found; the nearest are pen-and-paper induction arguments in
KV-injection papers ([2607.27090](https://arxiv.org/abs/2607.27090),
[2604.03270](https://arxiv.org/abs/2604.03270)) and TorchLean ([2602.22631](https://arxiv.org/abs/2602.22631)),
which formalises networks in Lean with IEEE binary32 semantics but, as far as its abstract states, no
KV cache.

## 2. What is adjacent

| # | Claim | Nearest prior work | The precise difference |
|---|---|---|---|
| A1 | **Bit-level parity of one model across trainer, CPU, CUDA, WASM and WebGPU from one spec.** What the receipts show today: token-level parity everywhere (greedy N/N, top-1 at every position), logits within ROUNDING tolerance across backends (CUDA vs CPU up to 5.5e-5 on Qwen2.5), bit identity on specific legs only (dequantisation on CPU and WASM; prefill = decode for recurrent ops; eviction on = off; CUDA run to run under `EMBER_DETERMINISTIC=1`). The spec itself classes softmax attention ROUNDING across libms (IR_SPEC section 5) | DetLLM ([NeurIPS 2026 listing](https://neurips.cc/virtual/2026/170003)): integer-only Qwen3-0.6B, one logit hash across A100, H100 and AMD, Intel and Apple CPUs; no browser, no trainer. ReproFuse ([2609.25624](https://arxiv.org/abs/2609.25624)): linear layers bitwise across Ampere, Ada and Hopper. Gensyn RepOps and OPEN-1B ([2609.17380](https://arxiv.org/abs/2609.17380)): training steps replayable bitwise on heterogeneous hardware; RepOps is proprietary. bitgpu ([README](https://cdn.jsdelivr.net/npm/bitgpu@0.19.1/README.md)): WebGPU 1-bit Qwen3 models gated bit-exact against a reference forward on one machine | Nobody found gives f32 bit identity from a written spec across a trainer and four engine backends including the browser. It is not shown here either: it is direction 2 below. WGSL's own spec allows reassociation, fusion "at least as accurate", and subnormal flushing ([WGSL 15.7](https://www.w3.org/TR/WGSL/#floating-point-evaluation)), so the WebGPU leg cannot be EXACT by the spec as written |
| A2 | Deterministic training and inference end to end | Same hardware: PyTorch deterministic algorithms; DASH (ICLR 2026, [2601.21824](https://arxiv.org/abs/2601.21824)); vLLM and SGLang batch-invariant modes; Thinking Machines (Sept 2025); LLM-42 (Microsoft). Across hardware: OPEN-1B/RepOps, DetLLM | paperkiln reproduced a run's best validation loss exactly on a fresh VM (log, 20 Oct; record the GPU type with it), and its fused head and BF16 options are bitwise the old step when off. The difference is that one author's trainer and engine agree to the token with stated tolerances; determinism as such is taken |
| A3 | A WGSL interpreter of the model document in the browser, covering recurrent state, MoR, state tokens and `expr` compiled to WGSL (gated on SwiftShader against ember CPU: 40/40 on 21 documents, Pythia-160m, Mamba-130m) | ONNX Runtime Web's WebGPU provider and transformers.js interpret ONNX graphs; WebLLM ([2412.15803](https://arxiv.org/abs/2412.15803)) compiles each model through TVM; bitgpu reads a manifest for one family; Burn runs one Rust model on wgpu, CUDA and CPU | The families it runs in the browser (SSM scans, depth routing, turn-state masks) were not found in any browser engine. Real-GPU speed is not yet measured |
| A4 | `expr`: a closed grammar for runtime-defined ops with fixed numerics and gradient rules, as a safe alternative to `trust_remote_code` | ONNX functions; StableHLO `composite`; Halide, Tensor Comprehensions, MLIR OpDSL | Closed and executable without code, with gradient rules; prior forms are compositions or opaque escapes (IR_PRIOR_ART section 4) |
| A5 | `recurrence`: a declared state, update and readout in a closed grammar, with prefill as the sequential fold and gradient checks as conformance; four hand-written ops re-expressed (three bit for bit, selective_scan within ROUNDING) | ONNX `Scan`; the Keras RNN-cell contract; flash-linear-attention's naive/recurrent/chunk forms tested equal; gradcheck tooling | No portable format found that derives decode semantics from a declared update and makes the gradient check part of conformance |
| A6 | Third-party checkpoints (Qwen2.5/3, Gemma 3, GPT-2, Pythia, Mistral SWA, Mamba-1) served through the same document, judged against a float64 evaluation of it | ORT GenAI's model builder; MLC-LLM; llama.cpp's 168 per-architecture builders; #21496 | One document per family instead of one builder per engine; closeness judged against float64 of the same IR rather than against one engine |
| A7 | Attention Residuals served from the IR | llama.cpp `kimi-k3.cpp`, vLLM and SGLang serve Kimi's AttnRes, each with a hand-written builder | Served from a definition rather than a builder; the mechanism itself is Kimi's |

## 3. What is taken (useful here, not a novelty claim)

- A from-scratch C++/CUDA trainer with its own autograd: llm.c (GPT-2/3 miniseries in C/CUDA),
  tinygrad, Burn, candle. paperkiln's breadth of families is its own; the category is not new.
- An end-to-end from-scratch chat model on one node: nanochat.
- Byte-identical k-quants and GGUF export: a compatibility property (gguf-py and other readers
  reimplement them); valuable for interoperability, not a claim.
- Determinism on one hardware class, training and inference (A2).
- Cross-hardware bitwise reproducibility as a goal: Gensyn (Verde, RepOps, REE), OPEN-1B, DetLLM,
  ReproFuse. The open part is the spec-defined f32 profile including the browser (A1).
- Verifiable inference without determinism: TOPLOC (ICML 2025, [2501.16007](https://arxiv.org/abs/2501.16007)),
  locality-sensitive hashes of activations tolerant of GPU and reordering differences.
- Paper to code: Paper2Code, PaperBench, SciReplicate-Bench, ReproRepo, AI Scientist; architecture
  search by agents: Genesys, ASI-Arch.
- Per-turn state tokens (RMT, gisting), role embeddings (ISE, ASIDE), turn-level positions (CoPE),
  depth vs width at 100M+ (MobileLLM, Gemstones): mechanisms, per the litcheck.
- A model-exchange graph with a written spec, reference interpreter and tests: ONNX, StableHLO, TOSA.
  The decode-state layer on top of it is O1.

## 4. Three directions

Chosen to use what is built, to run on a laptop CPU plus small RunPod or Colab bills, and to give a
result that is useful whether or not it becomes a paper. Systems and specification work, not new ML
bets.

### Direction 1: point keramos at engines you did not write

**What is built or shown.** keramos adapters for outside engines: llama.cpp (through
llama-cpp-python's low-level bindings: state save and restore, sequence removal for rewind, logits),
HF transformers (its `Cache` classes), ONNX Runtime GenAI, and vLLM on one L4. Run the suite's `trace/`
cases on the third-party models it already holds (tp_qwen3, tp_gemma3, tp_mistral, tp_gptneox,
tp_mamba1) and on the paperkiln models llama.cpp reads (llama, MoE). The output is a table per engine
and family: prefill = decode (bit identical, within ROUNDING, or token flips), chunked prefill, rewind
into a sliding window, SSM snapshot and restore, cache on and off. Every deviation gets a minimal
reproducer and, where it is a defect, an upstream issue.

**Why it is new.** 2604.15409 measured cache-on vs cache-off divergence in one engine, in FP16, for
three models. Nothing found measures several engines against a written statement of what decode must
equal, per family, including sliding windows and SSM state. It turns O1 from "our engines conform" into
"here is what conformance reveals about the engines people use", which is the evidence an outside
reader needs and the missing third-party consumer that IR_PRIOR_ART section 4 names as the adoption
risk.

**First milestone (two to four days).** A llama-cpp-python adapter answering the suite's load,
prefill, decode, snapshot and truncate methods for tp_qwen3 and tp_mistral on the CPU; the trace cases
run, one table, every SKIP named.

**Cost.** CPU work through `scripts/cpu_budget.py` (two slots). The vLLM leg: a few L4-hours on RunPod,
under $5.

**What makes it publishable or valuable.** For the TOPLAS or TOMS paper scoped in IR_PRIOR_ART
section 7, a measured comparison of production engines against the spec is the section that shows the
spec says something engines leave unstated. If every engine agrees in FP32, that is also a result: the
first written account of why, with the FP16 and chunked cases where they part. Without a paper: a
`pip`-installable checker an engine author can run, and upstream bug reports with reproducers, which are
public and citable work.

### Direction 2: one f32 model, the same bits everywhere, including the browser

**What is built or shown.** An `exact` execution profile in which one tiny model (the live 14.6M
chat model, or pk_llama) gives bit-identical logits on five executors: paperkiln's forward on the
CPU, ember CPU, ember CUDA, ember WebAssembly and ember WebGPU. Three things stand in the way, and each
is a contained piece of work:
1. **Transcendentals.** Softmax's `exp` is libm's today, which differs by an ulp across platforms, so
   attention is ROUNDING. Specify an `exp` (and `tanh`, `erf`, `log1p`) as a fixed sequence of f32 adds
   and multiplies (range reduction plus a fixed polynomial), written once in the spec and once per
   engine.
2. **CUDA.** nvcc contracts to FMA by default; the profile compiles its kernels with `--fmad=false` or
   uses `__fadd_rn`/`__fmul_rn`, with sequential ascending reductions and no cuBLAS.
3. **WebGPU.** WGSL lets an implementation reassociate, fuse and flush subnormals, so f32 shader
   arithmetic carries no bitwise guarantee. Either measure whether real browser stacks (Chrome/Dawn on
   NVIDIA, Intel, Apple) keep the bits anyway, or carry the reductions in u32 (integer arithmetic is
   exact in WGSL), as DetLLM does on CPUs and GPUs.

Each workaround has a cost in speed; measuring that cost per backend is part of the result.

**Why it is new.** DetLLM gets cross-vendor identity by leaving floating point for integers, without a
browser or a trainer. ReproFuse covers NVIDIA linear layers. RepOps is proprietary and covers CPU,
NVIDIA and Apple GPUs. bitgpu gates WebGPU against a reference on one machine. Not found: f32 bit
identity defined by an open spec, across a trainer and four engine backends, with the browser GPU among
them. The WGSL finding (the spec forbids an EXACT f32 class there) is itself a statement nobody has
written down for ML.

**First milestone (three to five days).** The spec'd `exp` as an EXACT op in keramos (numpy
reference, ember CPU, ember WASM); the live model's logits bit-identical across those three on the 40
frozen prompts. Then the CUDA leg on one L4.

**Cost.** CPU work for the first milestone. CUDA: one or two L4-hours (about $1). Real-GPU WebGPU: the
headless-Chrome runner already planned in section 11 item 7, a few dollars per GPU type.

**What makes it publishable or valuable.** A TOMS-style paper: an EXACT profile for language-model
inference, its definition, its cost on each backend, and the WGSL result. Without a paper: a public
page where a visitor's own GPU computes a reply and prints its logit hash, which equals the hash the
server and the trainer compute. That is a demonstration of verifiable inference (the problem TOPLOC and
Gensyn work on) that anyone can check in a browser, and a portfolio piece that explains itself.

### Direction 3: clean rooms as a measuring instrument for specifications

**What is built or shown.** Repeat the clean-room experiment that produced the Rust engine: three or
four more implementations from the spec text alone, each with its own read log, varying the language
(TypeScript, Go, C) and the agent (Claude, and Codex working on a copy outside the repository, as
AGENTS.md requires for any Codex run). Run each at revision 1.1.4, where the Rust engine found 23 gaps, and at the
current revision. Record every gap each run hits. With several independent finding sets, capture-recapture
estimation (the Lincoln-Petersen family, used for decades to estimate defects left after software
inspections) gives an estimate of how many gaps remain undiscovered, per revision.

**Why it is new.** SpecBench asks agents to find gaps by reading RFCs; FDE-Bench and the
microservice study found unstated behaviours while generating code against one spec. Capture-recapture
is established for human inspections. Not found: independent from-text implementations scored by a
black-box conformance suite, used to estimate a specification's remaining gaps and to show the count
falling across revisions. keramos already has everything this needs: the suite, the adapter contract,
the read-log discipline and a first finding set.

**First milestone (two days of agent time).** One TypeScript clean room at v1.1.4, receipts and a
findings list; overlap with the Rust engine's 23 findings counted. A TypeScript engine is also the
natural host-side reference for the browser legs of direction 2.

**Cost.** Agent tokens only; no GPU. Tens of dollars.

**What makes it publishable or valuable.** An empirical software-engineering paper: N implementations,
the gap counts and their overlap, the estimate of what remains, and the drop between revisions. The
journal is chosen after checking scope and history, not from this note. Without a paper: a written
clean-room protocol any spec author can reuse, and a keramos that has been implemented five times from
its text, which is the strongest answer to "is this a spec or one engine's internal format".

### Also considered

- **Executable architecture cards.** The chain's artefacts on a page: each paper's IR document, a
  click from every value to its LaTeX quote, the model running in the browser. Open as a combination
  (O4), mostly a portfolio piece; the extraction claims need an independently labelled truth set
  before they carry a paper.
- **The Lean core** (IR_PRIOR_ART section 5, option C): structural theorems (eviction preserves
  outputs, recurrent prefill = decode, MoR depths prefix-determined) without a floating-point model. No
  mechanisation was found; it fits a Lean-experienced author and the Journal of Automated Reasoning
  route already checked there.
- **`recurrence` in WGSL**: closes A3's last refusal; engineering, not positioning.

## 5. Notes on the evidence

- The candidate claim "bit-level parity across trainer, CPU, CUDA, WASM and WebGPU" is stated in this
  map as what the receipts show (A1): token parity everywhere, bit identity on named legs. Direction 2
  is the work that would make the stronger sentence true.
- Several 2026 sources (DetLLM, ReproFuse, OPEN-1B, SpecBench, FDE-Bench) were read at abstract or
  listing level only. Read the full texts before citing them in a submission.
- MoR serving (O2) and browser SSM serving (A3) are absences from searches of engine docs and the web,
  not from the engines' trackers.
