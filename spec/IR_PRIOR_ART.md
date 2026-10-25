# The paperkiln IR against prior art: where it can lead

*Survey of 25 October 2026. Sources were fetched on that date. Claims about other systems are cited.
"UNVERIFIED" marks a claim we could not confirm from a primary source, and every claim of absence
("no system does X") is the result of a search, not a proof. The rows on ACM, Elsevier and IEEE
journals rest on search-engine extracts because those sites refused direct fetches; check them in a
browser before acting on them.*

## 0. Verdict

The model-exchange problem is solved for stateless graphs. ONNX, StableHLO, TOSA and NNEF all give a
versioned graph and a closed op set, and ONNX and StableHLO add a reference interpreter and a test
suite. ONNX functions and the StableHLO `composite` op already let a new op carry its definition.
None of them specifies the parts that make a language model a language model:

- what a decode step must keep (the KV cache with window, sinks and eviction, or a recurrent state);
- that prefill and token-by-token decode compute the same function;
- when rows may be dropped, rewound or snapshotted;
- that per-token routing is causal enough to decide at T = 1;
- how numerics are classed (bit-exact against within-tolerance) per op.

In every surveyed system these live in engine code, vendor ops or runtime APIs. This is the gap.
paperkiln's IR states them normatively, and two engines are gated against one reference
interpreter. The way to lead is to make that layer the artefact: a standalone spec, a conformance
suite, a primitive-definition layer for stateful ops, and a mechanised core. Adding more families is
not the way.

## 1. What our IR is, precisely

Normative text: `IR_SPEC.md` (IR version 1), with the schema `ir.schema.json`, the validator
`tools/ir_validate.py` and the conformance suite `conformance/`. Reference interpreter:
`tools/ir_numpy.py` with `tools/ir_expr.py`. Implementations: paperkiln (exporter and trainer; CPU
and CUDA) and ember.cpp (`graph_ir.cpp` and `graph_ir_cuda.cu`; CPU, CUDA, WASM, and WebGPU for llama
and MoE). Contracts and gates: `LOCAL_LLM_STACK.md` sections 3 to 5 and 10.

| # | Property | What the spec says | Evidence today | Limits today |
|---|---|---|---|---|
| P1 | Portable, versioned graph | JSON document: `ir_version`, `family_hint`, `model`, `tokenizer`, `inputs`, `embed`, `block {repeat, state, nodes}`, `head`, `outputs`, `modes`. Stored in GGUF as `paperkiln.ir` plus `paperkiln.ir_version`; a `.ir.json` sidecar is allowed | 9 worked documents in `ir/`; `tools/ir_check.py` checks the GGUF against the IR | `ir_version` is still 0 after 16 commits that added semantics. No JSON Schema. `ir_check` checks names and agreement, not semantics; semantic validation lives in each engine's loader |
| P2 | Normative numerics, exactness classes | Per op: order of accumulation, f32 rounding after each primitive, no FMA, tie rules (MoE top-k and MoR routing go to the lower index). Class EXACT (bit-identical) or ROUNDING (tolerance `\|a-b\| <= tol * max(1, max\|ref\|)`: 1e-5 CPU, 5e-5 CUDA). Official checkpoints are judged against a float64 evaluation of the same IR | Bit-identity gates on EXACT changes (`scale` vs the diagonal linear; eviction on vs off). Logit tolerances and greedy 40/40 across families and backends | `LOCAL_LLM_STACK.md` section 4 point 5 records that paperkiln's vendored transformer_core builds with `-ffast-math`, so the reference's reduction order is not pinned there. The normative wording is stricter than one reference build enforces |
| P3 | Decode state as typed slots | `kv` {heads, kv_heads, head_dim, max_len, window, sinks}; `evict: turn_state`; `kv_rows: active`; `recurrent` {op, tensors [{name, shape, init}]}. Prefix reuse, rewind and snapshot rules are stated. Eviction's exactness is classed per kernel: EXACT on sequential ascending sums, ROUNDING on tiled flash kernels | ember: eviction on = off bit for bit, 26 vs 132 rows held; recurrent snapshots and `resume_point` | CUDA legs for state tokens and MoR not yet gated |
| P4 | Prefill = decode | Every op has the same semantics at T and at T = 1. Recurrent prefill is the sequential update, so the state matches bit for bit; a chunked prefill is ROUNDING against it. MoR depths are a function of the prefix | ember and mtstudio: decode = prefill on recurrent, MoR (depth traces 40/40) and state-token models | For softmax attention the equality is semantic: the values are ROUNDING unless the kernel sums in the same order |
| P5 | Per-layer attributes; tied-block recursion | `{"per_layer": [...]}` on any block node or slot attribute. Block `weight_index`: application i binds `blk.{m_i}`, with its own state per application | gemma3 local/global; three recursion models, 2.7e-7, 40/40 | Device run of `weight_index` pending |
| P6 | Data-dependent control | `depth_route`: causal token-choice MoR; nested depths; skipped steps write no rows; written-rows mask per application. Five binding constraints were agreed with ember on 23 Oct | ember ceadd42: log-probs within 2.3e-7, every depth equal, 9 malformed IRs refused | `kv_from` (recursive KV sharing) reserved; no device kernel |
| P7 | Masks over token ids; extra embedding rows | `mask: turn_state` with codes c(p) and s(p) from the ids alone. `state_weight` rows outside the vocabulary are never logits. The template contract binds the IR to the chat template | ember e5f4470: template rule agrees with the serialiser on 3,000/3,000 random sequences; /chat 40/40 | Defined over `paperkiln/1` only |
| P8 | Closed-grammar runtime ops | `expr`: a fixed grammar, f32 per primitive, constant folding in f32, gradient rules for max/min/where/rowmax/abs/relu at ties | C++ vs numpy 1.19e-7 on 80 cases; ember 46/46; GoLU and Primer norms travelled from paper to chat with no human edits | Elementwise and row-wise only. Recurrences, convolutions and scans go to `needs_new_primitive` |
| P9 | Third-party graphs | `tools/gguf_ir.py` builds the IR from llama.cpp metadata (Qwen2/3, Gemma 3, GPT-2, Pythia, Mistral SWA, Mamba-1); ember's C++ builder is equal field by field | Against HF: Qwen2.5 40/40, Pythia-160m 40/40, Gemma-3-270m top-1 everywhere, etc. (section 5.3) | WASM not gated for third-party families |
| P10 | Reference interpreter + engines | numpy reference; paperkiln; ember on CPU, CUDA, WASM and (llama/MoE) WebGPU | Greedy-parity contract (section 3.1) | Same author for both engines. ember vendors paperkiln's `expr` parser and `chat_template.hpp`, so independence is partial. No per-op fixture package exists: `ir.hpp`'s "fixtures committed per IR version" is not yet a suite |
| P11 | Training from the IR | `expr` defines gradients | paperkiln trains from its own modules and exports the IR per module | paperkiln does not build a trainable model from an IR document. Today the IR is a serving contract with training-aware ops, not yet one source for both |
| P12 | Paper to IR with provenance | `papers/llm_extract.py`: one call per paper; a verbatim quote and location for every value; a deterministic verifier; abstains rather than guesses | 40 papers: 82 correct / 9 abstain / 1 wrong (rule-based: 72 / 20 / 0) | The one error was a quote that was valid but meant the wrong model (Primer's norm) |

## 2. Prior art, property by property

**Matrix** (F full, P partial, – none):

| System | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10/11 |
|---|---|---|---|---|---|---|---|---|---|---|
| ONNX (+ ORT contrib ops, GenAI) | F | P | P | P (cache layout only) | P | P | – | P | F | P (training deprecated) |
| StableHLO | F | P | – | – | P | P | – | P | F | – |
| TOSA | F | P (strongest) | P (experimental) | – | – | P (experimental) | – | – | F | – |
| torch.export / AOTInductor | P | – | P | – | – | P | – | – | – | P |
| jaxpr | P | – | – | – | P | P | – | P | – | F (in JAX) |
| TVM Relax / MLC-LLM | P | – | P (runtime API) | – | P | P | – | – | – | – |
| NNEF | F | – | P | – | P | – | – | P | P | – |
| TFLite / LiteRT | F | – | P (convention) | – | – | P | – | – | – | – |
| Core ML / MIL | P | – | P | – | – | P | – | – | – | – |
| ExecuTorch | P | – | P (custom ops) | – | – | P | – | – | – | – |
| GGUF + llama.cpp | P | P (test tolerances) | P (code) | – | P | P (code) | – | – | P (test-backend-ops) | – |
| HF transformers | – (code is the spec) | – | P (Cache classes) | – (no guarantee) | P | P | – | – | – | F (in PyTorch) |
| vLLM / SGLang / TRT-LLM / MAX | – | – | P (runtime) | – | P | P | – | – | – | – |
| tinygrad | – | – | – | – | – | – | – | P | – | F |

**P1, the graph.**
- ONNX has a versioned IR plus versioned opsets
  ([Versioning.md](https://raw.githubusercontent.com/onnx/onnx/main/docs/Versioning.md)).
- StableHLO promises 5 years of backward and 2 years of forward compatibility
  ([compatibility.md](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md)).
- GGUF, the format our IR rides in, says a compute graph is "a future extension and still needs to
  be discussed" ([gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)). llama.cpp
  keeps one C++ graph builder per architecture (168 files under `src/models/`), and its guide to
  adding a model checks by running the tools, not against HF
  ([HOWTO-add-model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)).
- The one proposal for a declarative graph in GGUF, discussion
  [#21496](https://github.com/ggml-org/llama.cpp/discussions/21496) (April 2026), covers Qwen2.5,
  Qwen3 and Gemma2 only, with no SSM or hybrid models, and had no maintainer response when fetched.
- HF positions transformers as "the source of truth" for model definitions
  ([May 2025](https://huggingface.co/blog/transformers-model-definition),
  [v5](https://huggingface.co/blog/transformers-v5)). A new architecture there needs
  `trust_remote_code=True`, which is arbitrary code execution
  ([vLLM docs](https://docs.vllm.ai/en/v0.9.1/design/huggingface_integration.html)).
- The only 2023-2026 academic proposal for an LLM-specific architecture description we found is a
  MOF metamodel poster with no semantics or execution
  ([CIbSE 2026](https://sol.sbc.org.br/index.php/cibse/article/view/42481)).

**P2, numerics.**
- StableHLO: "does not provide guarantees about numerical accuracy"; `precision_config` is
  "underspecified"; reductions run as "an implementation-defined full binary tree"
  ([spec](https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md)).
- ONNX's IR has no numerics section. The few normative numeric rules include TopK's lower-index tie
  rule ([TopK](https://onnx.ai/onnx/operators/onnx__TopK.html)). Backend tests default to rtol 1e-3
  and atol 1e-7
  ([loader](https://raw.githubusercontent.com/onnx/onnx/main/onnx/backend/test/loader/__init__.py)).
- NNEF defines semantics "supposing infinite arithmetics"
  ([NNEF 1.0.4](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.4.html)).
- TOSA is the strongest ML precedent ([TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)):
  - integer operations "must match exactly";
  - floating-point operations meet per-op ULP bounds against an fp64 reference that follows the
    pseudocode's order;
  - dot products get a statistical bound;
  - a C reference model is the golden reference.
  It bounds error. It does not require bit-identical floats or forbid FMA.
- llama.cpp's `test-backend-ops` compares each backend with the CPU reference by normalised MSE
  (default 1e-7), with exact index agreement for top-k
  ([source](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)). This
  is a per-op tolerance, not judged against float64, and has no named classes.
- Outside ML the templates are explicit:
  - WebAssembly specifies IEEE round-to-nearest-even, allows implementation-dependent results under
    relaxed SIMD (fused or unfused `madd`), and pins every choice in a **deterministic profile**
    ([numerics](https://webassembly.github.io/spec/core/exec/numerics.html));
  - Vulkan/SPIR-V lists per-instruction precision ("correctly rounded" add and multiply, 2.5 ULP
    divide) and allows contraction unless `NoContraction` is set
    ([spirvenv](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)).

  Our EXACT/ROUNDING classes are the same idea applied to LLM ops.

**P3 and P4, decode state and prefill = decode.**
- ONNX's Attention op (opset 23 to 25) takes `past_key`/`past_value` and concatenates them. Its page
  warns this "only works correctly" without padding
  ([Attention](https://onnx.ai/onnx/operators/onnx__Attention.html)).
- The richest cache semantics found are a vendor op, Microsoft's `GroupQueryAttention`
  ([ContribOperators](https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/ContribOperators.md)):
  - an eviction formula for a window-sized cache;
  - "the resulting layout is independent of how the tokens were split into steps";
  - rewind "is exact when L(T-k) == L(T)-k".

  That invariance covers the cache layout, not the values.
- Decode wiring for ORT GenAI lives outside the graph, in `genai_config.json`
  ([config](https://onnxruntime.ai/docs/genai/reference/config.html)).
- TVM's `KVState` runtime object has `ForkSequence`, `PopN`,
  `EnableSlidingWindowForSeq(window, attn_sink_size)` and RNN state
  ([kv_state.h](https://apache.googlesource.com/tvm/+/HEAD/src/runtime/vm/kv_state.h)): a C++ API
  with no exactness statements.
- Core ML stateful models (`read_state`, `coreml_update_state`) have no window or eviction semantics,
  and "execution precision varies"
  ([stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)).
- LiteRT exports `prefill_{N}` and `decode` signatures by naming convention
  ([litert-torch](https://raw.githubusercontent.com/google-ai-edge/litert-torch/main/litert_torch/generative/README.md)).
- torch.export returns cache writes as `BUFFER_MUTATION` outputs
  ([graph_signature.py](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/export/graph_signature.py)).
- llama.cpp's `llama_memory_i`, its SWA enum and `--keep`, vLLM's hybrid KV cache manager
  ([design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html)) and HF's `Cache`
  layers ([cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py))
  are all code.
- HF states the opposite of our contract: KV caching "will change the logits"
  ([#25420](https://github.com/huggingface/transformers/issues/25420)).
- Thinking Machines' batch-invariance work requires the attention reduction order to be identical
  whether or not keys come from the cache
  ([blog, Sept 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
  vLLM and SGLang now ship deterministic modes
  ([vLLM](https://docs.vllm.ai/en/stable/features/batch_invariance/),
  [SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)). These are kernel properties
  within one engine and one hardware class, not a cross-engine contract.
- We found no mechanised proof, and no formal statement, of prefill = decode anywhere.

**P6, routing.**
- Mixture-of-Depths states that "the top-k operation is non-causal"
  ([2404.02258](https://arxiv.org/html/2404.02258v1) section 3.5).
- MoR ([2507.10524](https://arxiv.org/html/2507.10524v3)) discusses expert-choice leakage and
  token-choice balance as training choices.
- We found no engine or format that serves MoR, and no format that states a routing-causality
  constraint. Our `depth_route` constraint (1) is, as far as we found, the first statement of it as
  a servability rule.

**P7, id-defined masks.** Every format takes the mask as an arbitrary tensor. Gisting exists only
as the paper's training mask ([2304.08467](https://arxiv.org/abs/2304.08467)). We found no
engine that serves gist-style masks with eviction of finished turns (an UNVERIFIED absence).

**P8, the extension mechanism.** This is prior art, covered in section 4(a). Our `expr`'s
distinction is that it is closed and safe: no code runs, every primitive's numerics are fixed, and
gradient rules come with it. ONNX functions, StableHLO composites and tinygrad's `CUSTOM` are
compositions or opaque escapes.

**P9 and P10, reference interpreter, conformance and independent engines.**
- ONNX's `ReferenceEvaluator` is "a pure python implementation of ONNX specifications", and "the
  official spec overrides" it ([reference](https://onnx.ai/onnx/api/reference.html)).
- The StableHLO interpreter covers 91 of 96 ops
  ([status](https://openxla.org/stablehlo/interpreter_status)).
- TOSA has a golden reference model and a conformance generator.
- Differential-testing work on DL frameworks (CRADLE, NNSmith, LEMON, Audee, TitanFuzz) uses ad hoc
  tolerances. NNSmith's code defaults to rtol 1e-2 and atol 1e-3
  ([difftest.py](https://raw.githubusercontent.com/ise-uiuc/nnsmith/main/nnsmith/difftest.py)).
- MLPerf's "99% / 99.9% of FP32" is a task-level accuracy class
  ([rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)).
- No LLM runtime has a conformance suite for decode state.

**P11, training.**
- ONNX's runtime training packages are deprecated (last release 1.19.2;
  [ORT 1.20.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0)).
- jaxpr, tinygrad and PyTorch train, but each format is internal to its framework.
- TOSA, StableHLO, NNEF, Core ML and GGUF are inference formats.

**P12, extraction.**
- Paper2Code ([2504.17192](https://arxiv.org/abs/2504.17192)), PaperBench
  ([2504.01848](https://arxiv.org/abs/2504.01848v1); best agent 21.0%) and ResearchCodeBench
  ([2506.02314](https://arxiv.org/abs/2506.02314)) all produce code.
- HyperPIE extracts hyperparameters without verbatim provenance
  ([2312.10638](https://arxiv.org/abs/2312.10638v2)).
- Exact-substring grounding with abstention exists in general-purpose extraction
  ([LangExtract](https://github.com/google/langextract)) and in medicine. One medical study found
  valid quotes in 83-91% of runs, but only 48-79% of evidence-bearing predictions semantically
  supported ([medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.03.03.26346690v1)). That is
  our Primer failure in general form.
- Papers with Code's methods pages now redirect to Hugging Face. The July 2025 sunset date is from
  secondary sources (UNVERIFIED).

**Attention residuals and hyper-connections in practice.** Since our `llama_attnres` work, Kimi's
Attention Residuals ([2603.15031](https://arxiv.org/abs/2603.15031)) have reached production engines:
- llama.cpp `src/models/kimi-k3.cpp`
  ([source](https://github.com/ggml-org/llama.cpp/blob/master/src/models/kimi-k3.cpp));
- vLLM, with fused kernels ([preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview));
- SGLang, where AttnRes "breaks standard layer plumbing"
  ([blog](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)).

Each engine has one hand-written builder per model plus hyperparameter keys. This is the concrete
case for a definitional IR: the same mechanism was written three times, with no shared definition.

## 3. Practitioner reality: "nobody uses ONNX for LLMs"

The claim is false as stated and true where Jonathan works.

**Where ONNX is used.** ONNX is the shipped LLM path on Microsoft client devices and in the browser:
- Foundry Local runs "all model execution" through ONNX Runtime
  ([architecture](https://learn.microsoft.com/en-us/azure/foundry-local/concepts/foundry-local-architecture));
- Windows ML went generally available on ONNX Runtime in September 2025
  ([blog](https://blogs.windows.com/windowsdeveloper/2025/09/23/windows-ml-is-generally-available-empowering-developers-to-scale-local-ai-across-windows-devices));
- transformers.js uses ONNX Runtime in the browser
  ([docs](https://huggingface.co/docs/transformers.js/index));
- onnxruntime-genai is actively released (v0.17.0, September 2026).

**Where it is not.**
- No server engine (vLLM, SGLang, TensorRT-LLM) takes ONNX as input.
- On the Hugging Face Hub (25 Oct 2026) there are 2,428 text-generation ONNX repositories against
  40,722 GGUF ([ONNX](https://huggingface.co/models?library=onnx&pipeline_tag=text-generation),
  [GGUF](https://huggingface.co/models?library=gguf&pipeline_tag=text-generation)).
- Across all tasks: safetensors 1,437,798; GGUF 214,567; ONNX 58,600; MLX 27,206.
- Where ONNX does serve LLMs, decode depends on vendor contrib ops and an external config.

Standard ONNX alone does not carry an LLM's decode semantics. That, not adoption, is the argument
to make in print.

## 4. The framing: an architecture spec independent of any engine

The GPT reviewer's framing has four parts:
- a research architecture should be representable independently of the framework it was first
  built in;
- a paper publishes a compact, executable spec beside its code and weights;
- a new primitive carries its definition, not an opaque name;
- the extractor is one producer of specs and the engines are two consumers, so the language
  outlives both.

**Already prior art (cite it, do not claim it):**
- "Independent of the framework" is the founding purpose of ONNX, NNEF and StableHLO.
- "Definitions, not names": ONNX functions are operators "whose semantics is formally expressed via
  expansion into a sub-graph", and runtimes inline the body when they lack a kernel
  ([IR.md](https://raw.githubusercontent.com/onnx/onnx/main/docs/IR.md),
  [AddFunctionBody](https://raw.githubusercontent.com/onnx/onnx/main/docs/AddFunctionBody.md)).
  For StableHLO `composite`, "the semantics of the op are implemented by the decomposition", with a
  version field for semantic changes ([spec](https://openxla.org/stablehlo/spec)).
- Algorithm-as-definition is the Halide, Tensor Comprehensions and MLIR OpDSL tradition. PyTorch's
  decompositions are definitions, but inside PyTorch only.
- A recurrence with a body is ONNX `Scan` and `Loop`
  ([Scan](https://onnx.ai/onnx/operators/onnx__Scan.html)), JAX `lax.scan` and `associative_scan`,
  and the Keras RNN-cell contract (`call(input_t, states_t) -> (output_t, states_t+1)`,
  `get_initial_state`; [Keras](https://keras.io/api/layers/recurrent_layers/rnn/)).
- One op in recurrent and chunked forms, tested equal: flash-linear-attention ships `naive`,
  `fused_recurrent` and `chunk` per op, and tests outputs, gradients and final state against each
  other at about 5e-3 ([test_gla.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/tests/ops/test_gla.py)).
- Gradient checking is standard tooling (`torch.autograd.gradcheck`, `jax.test_util.check_grads`).
  `torch.library.opcheck` explicitly does not check gradient correctness
  ([torch.library](https://docs.pytorch.org/docs/2.14/library.html)).

**Open, as far as we found:**
1. A portable format whose primitive definitions declare decode-state semantics normatively (init,
   per-token update, readout), with "prefill = sequential update" as a conformance condition that
   independent engines check. ONNX `Scan` has no decode interface. Engines hand-write the update op:
   MAX has `selective_scan_fwd` and a separate `selective_scan_update`
   ([MAX](https://max.modular.com/api/mojo/state_space/selective_scan_ops)), and llama.cpp has
   `ggml_ssm_scan`.
2. A gradient check as a conformance condition of the op format, rather than an optional test
   utility.
3. Spec-plus-two-engines as a publication norm. The NeurIPS checklist asks only that "the paper
   should describe the architecture fully", in prose
   ([checklist](https://neurips.cc/public/guides/PaperChecklist)). NAS-Bench encodings are
   executable but have closed vocabularies ([NAS-Bench-101](https://arxiv.org/abs/1902.09635)), and
   HF's `config.json` resolves to a Python class. No venue asks for an executable, engine-neutral
   architecture spec.
4. A closed, safe definition language as the alternative to `trust_remote_code`.

The framing is new as a combination and as a norm. Its core, the definitional layer, must be
presented as extending ONNX functions and StableHLO composites into decode state and numerics
classes. One risk should be stated plainly: a language "outlives both engines" only when someone
other than its author produces or consumes it. Until then it is a format, and the paper should not
claim a standard.

**(c) Where `expr` stops, and what a primitive-definition layer needs.** `expr` covers stateless
maps of [T, w] rows. Every hand-written tier-3 op in `ir.hpp` (`linear_attention`, `ssm_scan`,
`selective_scan`, `conv1d_causal`) has the same shape:
- a fixed-size state;
- an initial value;
- a per-token update that reads the state and the token's inputs;
- a readout;
- prefill defined as the update applied for t = 0..T-1.

A `recurrence` primitive would carry that shape in the spec:
- **Declaration.** State tensors with shapes, dtype and init; inputs and outputs per token; the
  attributes.
- **Update and readout.** Written in a closed grammar: `expr` extended with indexed tensors and two
  or three contraction forms (an einsum-like `sum over n` in ascending index order, and an
  outer-product update). Every primitive keeps `expr`'s numerics rule. That one extension is enough
  to restate all four existing ops.
- **Decode semantics by construction.** Decode is one update. Prefill is the sequential fold, and
  its result is the normative value. An engine may run a chunked or parallel form (SSD
  ([2405.21060](https://arxiv.org/abs/2405.21060)), fla's `chunk`), which is classed ROUNDING
  against the fold, as `ir.hpp` already says for `linear_attention`. Snapshot and rewind follow from
  the declaration: a snapshot is the state tensors.
- **Backward.** Derived by the trainer's autograd through the update (backpropagation through time
  over the fold) or supplied as a second definition. Either way, a finite-difference gradient check
  on fixtures is part of the op's conformance, as `expr` already did (14 expressions, worst relative
  error 1.06e-4).
- **Acceptance gate.** Re-express `linear_attention`, `ssm_scan`, `selective_scan` and
  `conv1d_causal` as `recurrence` nodes. Both engines must reproduce the hand-written ops' outputs:
  bit for bit where the declared order is the hand-written order, ROUNDING otherwise. Then the
  extractor's `needs_new_primitive` becomes "a recurrence, verified", and LLM-drafted primitives
  (the 23 Oct "arXiv for LLMs" phase) enter through this gate rather than through engine code.

Nothing surveyed combines a portable serialisation, normative decode semantics derived from a
declared update, and a gradient check as conformance. Each piece exists separately: ONNX `Scan`,
the Keras cell, fla, and gradcheck.

**(d) What decoupling the format from both engines takes.** Today the IR is tied to GGUF and to
llama.cpp in six ways:
1. Weights are addressed by llama.cpp GGUF names (`blk.{i}.attn_q.weight`).
2. Layout is GGUF's [out, in], and "each tensor's storage type is its GGUF type".
3. `family_hint` equals `general.architecture`.
4. `paperkiln.state_tokens.k` and `paperkiln.chat_template` are GGUF keys outside the document.
5. The tokenizer is `{"source": "gguf"}`.
6. The version is a GGUF uint32 that has stayed 0.

Needed:
- **A standalone document** (`*.ir.json`, already allowed as a sidecar) that is complete on its own.
  It should hold the state-token count, a reference to the template record, and a tokenizer
  descriptor or digest.
- **A binding table.** It maps the document's own parameter names (with shapes and layout) to a
  container: GGUF names as today, or safetensors or HF names. The GGUF embedding becomes one binding
  among several.
- **Engine-neutral naming.** Families become informative labels with no semantics. The current
  `ir.hpp` already treats them as hints, but its families section mixes normative graphs with
  llama.cpp names.
- **Versioning.**
  - A semantic version for the core.
  - Namespaced, separately versioned op sets or profiles (`core`, `state`, `recursion`, `chat`),
    after ONNX opset domains and TOSA profiles.
  - Each engine declares the profiles it implements and refuses others by name, as it already
    refuses unknown ops.
  - A canonical serialisation and a hash, so a paper can cite the exact spec it used.
- **A JSON Schema plus a semantic validator** shared by every consumer, replacing per-loader checks.

## 5. Ways to push further, evaluated

| Candidate | Novelty against prior art | Value | Effort | Risk |
|---|---|---|---|---|
| A. Decode-state semantics with exactness classes as a conformance suite (prefill = decode, eviction, rewind, snapshot, routing causality) | High: the gap in sections 2 and 4. TOSA and Wasm give the numerics template; nobody covers state | High: engine authors (llama.cpp, MLX, WebGPU ports) re-derive this per model; HF says caching changes logits | Moderate: a fixture format, a state-trace runner and the existing gates repackaged | Low technically. Adoption is the risk |
| B. A `recurrence` primitive-definition layer, with gradient checks as conformance (section 4c) | High as a combination; parts are prior art | High: turns tier 3 from engine code into spec text, which is the reviewer's framing made real | Moderate: grammar extension, one interpreter path per engine, re-expressing four ops | Medium: chunked fast paths stay hand-written, and the readout's fixed order may cost device speed |
| C. Lean mechanisation of the core | High: no mechanised semantics of an LLM IR with decode state exists. The nearest are TorchLean, which as far as its abstract states models no KV cache ([2602.22631](https://arxiv.org/abs/2602.22631)), and Lean-MLIR ([ITP 2024](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.9)) | High for a journal paper; moderate for practitioners | Moderate if scoped (below); high for floating point | Lean IEEE support is immature (FloatSpec has about 270 `sorry`s; [reservoir](https://reservoir.lean-lang.org/@Beneficial-AI-Foundation/FloatSpec)). Avoid it by the scoping below |
| D. One IR for training and serving | Moderate: jaxpr and tinygrad train, but in-framework; ONNX training is deprecated | High for the framing: "train in one engine, serve in another from the spec" | Moderate: an IR-to-module builder in paperkiln (each op already has a differentiable primitive) | Low |
| E. Spec-first mechanisms (state tokens, attention residuals, MoR done this way on 23-24 Oct) | Low as process (Wasm's phases are the model); high as ML practice | High as evidence: the second engine caught a template-rule divergence (ember e5f4470) | Already done; needs writing up | Low |
| F. `expr` as a safe extension mechanism | Low to moderate (ONNX functions, composites, einops) | Moderate: the safe alternative to `trust_remote_code` | Done on CPU | Low |
| G. The IR as a search space (servable by construction) | Moderate to high: STAR/LFM2 optimises cache size but hand-ports each winner to runtimes ([2511.23404](https://arxiv.org/abs/2511.23404)); Genesys and ASI-Arch search PyTorch code with no export contract | High if it worked | High: compute-heavy, and AI/ML is deprioritised for solo work | High |
| H. Paper to IR with verified quotes | Moderate: grounding with abstention exists; per-value provenance for architectures does not | Moderate: a producer for the format | Done; needs a larger, independent truth set | Quote-valid-but-wrong is unsolved |

**How C should be scoped.** State the EXACT class as equality under any interpretation of the
primitives, provided the evaluation trees are identical. Then the structural theorems need no
floating-point model at all, because they show the two computations build the same expression tree:
- turn_state eviction preserves every output, since evicted rows are never visible and the sum over
  the remaining keys is the same ascending fold;
- recurrent prefill equals token-by-token decode;
- MoR depths are prefix-determined, and inactive rows are untouched;
- sliding-window decode needs only sinks + window rows.

ROUNDING then becomes a separate, stated assumption. This mirrors Mirage's split between exact
algebraic equivalence and a separate numeric check ([2405.05751](https://arxiv.org/html/2405.05751)),
and WasmCert's mechanise-then-run-the-suite method
([WasmCert-Coq](https://github.com/WasmCert/WasmCert-Coq)).

**Ranking.**
1. **A, the conformance suite with decode-state semantics.** It is the clearest gap. It reuses gates
   that already pass. It is what an outside engine author would use.
2. **B, the `recurrence` definition layer.** It is the reviewer's framing turned into a testable
   claim: four existing primitives re-expressed and reproduced bit for bit is a clean result. It
   also opens the "arXiv for LLMs" phase safely. Built as `IR_SPEC.md` section 6.10 (revision
   1.1): three of the four reproduce the hand-written ops bit for bit, selective_scan within ROUNDING
   (log1p), and gradient cases are part of the suite.
3. **C, the Lean core scoped to structural theorems.** No prior work exists on it, and it fits
   Jonathan's Lean experience. Floating point stays out of scope.

D should be done alongside B, because the reviewer's "train in one engine, serve in another"
depends on it. E is the paper's evidence, not a separate project.

## 6. What makes the IR a credible primary artefact

In order of importance:
1. **A standalone, versioned spec.** Move the normative text out of a C++ header comment into a
   spec document with numbered sections. Declare v1 with a changelog: v0 has absorbed 16 commits of
   semantics. Split it into profiles. Resolve the `-ffast-math` tension: either the reference build
   pins the order, or the affected ops are classed ROUNDING.
2. **The conformance suite as a package.** It needs:
   - per-op fixtures: inputs, expected f32 bit patterns, class and tolerance;
   - state traces: prefill, decode, rewind into an evicted segment, snapshot restore, MoR depth
     traces;
   - negative tests: malformed IRs that must be refused (ember already has nine for MoR);
   - whole-model gates;
   - an engine-neutral runner (CLI protocol or a small C ABI).

   This is what Wasm's `testsuite` and TOSA's conformance tests are to their specs.
3. **A schema and a shared validator** (JSON Schema plus semantic checks), so a third consumer does
   not have to reverse-engineer two loaders.
4. **Decoupling from GGUF** (section 4d): a binding table, engine-neutral names, profiles, a
   canonical hash.
5. **A third implementation that does not share code.** ember vendors paperkiln's `expr` parser and
   `chat_template.hpp`, so today's two engines are not independent in the Wasm sense; Wasm's phase 4
   requires "two or more Web VMs"
   ([phases](https://github.com/WebAssembly/meetings/blob/main/process/phases.md)). Two options:
   - a PyTorch or JAX consumer, which is also the bridge to practitioners;
   - cheaper: a clean-room build from the spec text alone by a fresh agent with no code access. Its
     conformance failures measure what the spec leaves unsaid. This is a spec-completeness
     experiment, not independent authorship, and must be reported as such.
6. **Governance, written down.** The working process already matches Wasm's phases: a draft spec
   text (`IR_SPEC_DRAFT.md`), review by the other engine, the reference, two engines passing, then
   normative. Writing it as the change policy costs an hour.

Items 1 to 3 matter most. Without them, a reviewer sees an engine's internal format, and "spec"
claims will not survive.

## 7. Publication angle

Constraints: journals only; no conferences; no arXiv or preprints; no TMLR or Synthese. Venue facts
below were checked on 25 Oct 2026; the ACM rows rest on search extracts of the cited pages.

- **ACM TOPLAS** fits best: "foundations to design, implementation, and applications", with no hard
  page limit (contact the Editor-in-Chief above about 50 pages)
  ([guidelines](https://dl.acm.org/journal/toplas/author-guidelines)). A spec with semantics, a
  reference interpreter, a conformance suite and two implementations is standard material there.
- **ACM TOMS** is the strongest home for the numerics and conformance angle
  ([guidelines](https://dl.acm.org/journal/toms/author-guidelines)). Its optional Replicated
  Computational Results review publishes an independent replication beside the paper
  ([RCR](https://www.cs.kent.ac.uk/projects/toms/RCR.html)). The risk is scope: lead with numerics,
  not language models.
- **Journal of Automated Reasoning** suits a separate paper on the Lean mechanisation ("formal proof
  assistants", "verification of hardware or software";
  [scope](https://link.springer.com/journal/10817/aims-and-scope)). LLM use is described in the
  Methods section ([guidelines](https://link.springer.com/journal/10817/submission-guidelines)).
- **Fees.** ACM is fully open access from 1 January 2026. An author without an ACM Open institution
  pays US$1,450, or US$950 as a member ([ACM OA](https://www.acm.org/publications/openaccess); the
  figures come from a search extract).
- **Not yet.**
  - JOSS needs six months of public history and evidence of research use
    ([submitting](https://joss.readthedocs.io/en/latest/submitting.html)).
  - JMLR's MLOSS track needs an active user community, and excludes software already published
    elsewhere ([mloss-info](https://www.jmlr.org/mloss/mloss-info.html)).
  - No ML-systems journal exists. MLSys is a conference.
- **AI disclosure.** JOSS requires one, and ACM and JAR expect a description. State the split as it
  is: Jonathan designed the IR and its contracts; Claude wrote scripting and code under his direction.

**Evidence a TOPLAS or TOMS paper would need.** Most of it exists:
- the v1 spec;
- the conformance suite, with counts of fixtures and traces;
- coverage results: paperkiln families and the seven third-party families against HF, across CPU,
  CUDA and WASM;
- an expressiveness comparison with ONNX, StableHLO and TOSA on the decode-state properties (the
  matrix in section 2, made rigorous);
- the spec-first case studies (state tokens, MoR, attention residuals) with the divergences the
  second engine caught. This is the analogue of Watt's mechanisation finding spec errors
  ([CPP 2018](https://popl18.sigplan.org/details/CPP-2018/21/Mechanising-and-Verifying-the-WebAssembly-Specification));
- the interpreter's overhead against ember's tuned llama path;
- if B is done, the four primitives re-expressed and reproduced.

The Lean core supports either paper, or stands alone at JAR.

## 8. Not verified

- The ACM, Elsevier and IEEE journal pages were not read directly.
- Liquid AI STAR's operator details: abstract only.
- MAX's KV-cache manager methods: search snippets only.
- The Papers with Code sunset date.
- Several absences (gist-mask serving, a derived chunked form from a recurrence, DenseFormer
  serving) are search results, not proofs.
- Kimi K3's use of Attention Residuals ([2607.24653](https://arxiv.org/pdf/2607.24653)) and the
  llama.cpp file are cited from fetched sources dated after mid-2026 and were not cross-checked
  further.
