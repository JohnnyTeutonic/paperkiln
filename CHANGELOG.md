# Changelog

- Materialise computed embedding tables before the conditional CPU fallback. Actual L4 reproduces the old deferred-table error and passes all seven fixed cases; ordinary leaf tables and invalid IDs remain controls.

- Added independent full-network Mamba1 CPU finite-difference checks for every parameter in two fixed tiny fixtures; merged constructor replay and 64 parser/negative tests pass. Composed CUDA, quality and residency remain separate gates.
## Watchdog-owned API judging continuation (31 October 2026)

Add a frozen, fail-stop campaign wrapper for the existing training-stage allowance.
It replays ledger/cache provenance on restart, checkpoints progress in bounded
batches and leaves HALTED on unknown charges, failed rankings or budget stops.
No automated recovery, profile tuning or allowance increase. Clean checkpoint
restarts reuse only verified API rankings. Fifteen offline lifecycle tests join
the 90 transport/CLI/sidecar tests; no campaign launch is included.

## Prospective Anthropic judge accounting and calibration (31 October 2026)

Add a separate exact-model Messages API transport with durable cache/ledger binding,
raw receipts, all-token cost accounting, historical spend and held-out reserves.
A frozen 48-attempt calibration returned 47 valid rankings; one network-unknown
charge keeps its full ceiling under an explicit audited recovery. Independent
receipt replay and 90 focused tests pass. This establishes operational readiness
for reviewed continuation, not CLI equivalence or a model-quality result.
[Calibration and limits](experiments/chat_general/receipts/anthropic_api_20261031/CALIBRATION.md).
## Preserve GPT2 export aggregate initialization (31 October 2026)

Append the optional training-recipe field after existing Gpt2ExportConfig
members, preserving positional initialization. A legacy-name runtime regression
and all 14 focused CLI/export checks pass on the corrected source. Earlier raw
CPU receipts remain unchanged.

## Experimental train-only GPT2 attention precision (31 October 2026)

Add per-instance FP32/default or BF16 attention GEMMs for the exact two-block
GPT2 LM/SFT subset. Unsupported combinations fail before outputs; inference
constructors and exported math stay FP32 with separate training-recipe metadata.
Preserved default logits/full gradients, six-step checkpoint and GGUF are bit
identical. CPU model/op gates, 14 CLI tests, NumPy/Ember inference and eight VRAM
tests pass. CUDA model validation and speed/quality evidence remain open.
[Evidence](docs/receipts/gpt2_attention_precision_20261031/RESULTS.md).

## Deferred GRPO CLI host readers (31 October 2026)

Materialize logits before `sample_pure` and reference `log_softmax_rows` host
loops, with debug host-read assertions. The actual CLI helpers pass independent
probability/ID/RNG references in a real L4 NoGrad/deferred step; both original
numeric failures are retained and landed logs independently replay. Source and
arithmetic are otherwise unchanged. This is a reader-correctness result, not
full GPU GRPO training validation or a model-quality gain. Historical run-mode
applicability remains unresolved where runtime metadata were not retained.
[Receipt](experiments/grpo_host_reads_20261031/receipts/l4_gate_v1/replay.json).

## Mamba causal-convolution CUDA subset (31 October 2026)

Add ordered forward/backward device kernels for `ops::conv1d_causal`, retaining
CPU fallback, sequence boundaries and selective gradients. Actual L4 CUDA plus
DEVCHECK passes 120 strict cases and the existing CUDA-selected Mamba regression;
landed raw logs independently replay. All float32 reductions retain CPU order.
Selective scan and a parallel scan remain open; no end-to-end speed claim.
[Receipt](experiments/conv1d_cuda_20261031/receipts/l4_gate_v1/replay.json).

## CPU host LayerNorm indexing (31 October 2026)

Replace checked per-element host access with checked contiguous pointers within
LayerNorm only, preserving accumulation order and device dispatch. The fixed
one-thread i9-13905H comparison measured 3.66x over 16 operation cases and 1.279x
on tiny d16/L2/T16 teacher-forced scoring; both block-bootstrap intervals exclude
1. All 321 operation cases, three whole-model gradient/update cases and every
score/logit artifact stayed bit-identical. CPU CTest: 53/53 passed. This is a scoped
CPU measurement, not a general training or GPU claim.
[Receipt and replay](docs/receipts/cpu_layernorm_20261031/validation.json).

## Tied-head Flex Hugging Face export (31 October 2026)

Plain GPT2- and Llama-flavour Flex now export either separate or explicitly tied
token/output heads. Actual trained models are saved, reloaded through Transformers,
and checked for full-row logits, greedy IDs, tensor inventory and shared parameters.
Modified flavour combinations remain refused. CPU validation and raw replay evidence:
[receipt](docs/receipts/hf_tied_flex_20261031/validation.json).

- Audited remaining host Matrix work with current tiny CPU scoring: no production optimization justified; existing CUDA zero-fill avoidance distinguished from the still-open historical CUDA follow-up. Source, profiler and raw-report evidence: `docs/receipts/host_matrix_cpu_audit_20261031/validation.json`.

## Existing Flex compatibility work integrated (31 October 2026)

Validated the preserved workspace implementation of exact GELU, tied Flex heads
and embedding expressions before positional tables. CPU gradient and full-row
Keramos/Ember parity checks pass; HF mapping and GPU validation remain separate.
[Source, provenance and validation](docs/receipts/author_flex_20261031/validation.json).
Original working files were preserved; two test defects were corrected without
weakening their assertions.


## Studio evaluation probes (31 October 2026)

Optional bounded needle-retrieval and behavioural literal contracts now run after
training against the actual exported final weights. Studio configures the stage
and displays immutable receipts; `mtstudio probes` supports manual replay.
CPU model, HTTP, browser and identity checks are recorded in
[the validation receipt](docs/receipts/studio_probes_20261031/validation.json).
These are configured diagnostic contracts, not semantic-quality benchmark scores.


Notable changes to microtorch. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are tagged `v*`.

## [Unreleased]

### September–October 2026

- **Non-causal backward for the channelwise ELU mixer** (31 Oct). The legacy-named
  `KimiLinearAttention` now differentiates its full-sequence sums; the autograd wrapper
  captures the forward mode and preserves independent stacked sequences and selective
  gradients. Empty/mismatched geometry refuses before slicing/division; denominator-clamp
  derivatives are guarded in both modes. Forward arithmetic is unchanged. This operator
  is not the published Kimi Linear/KDA architecture. An independent dense float64 oracle,
  270 component oracle finite differences, 18 production-float directional differences,
  batching/selective-gradient/metamorphic checks and causal regressions pass. Very negative
  ELU features retain the existing float-cancellation limitation; tests explicitly distinguish
  smooth-expression AD from finite differences of the rounded float forward there.
  [CPU source-bound receipt](docs/receipts/kimi_noncausal_20261031.json).

- **Studio LLM extraction and evidence review** (31 Oct). With an explicitly configured
  `ANTHROPIC_API_KEY`, Studio defaults to the source-verified LLM extractor, with a per-job
  spend cap, background status, cancellation and original extraction download. The worker
  re-verifies raw replies against source before atomic publication; failures never fall back
  to the rule extractor. Evidence, unresolved values and unsupported mechanisms are reviewed
  without silently approximating them in the coarse form builder. Owned workers stop if Studio
  exits; paid routes reject foreign origins. Offline worker, HTTP and browser checks use no API.
  Validation: `docs/receipts/studio_extractor_20261031/validation.json`.

- **HS1 full-history parity limitation disclosed** (30 Oct). A complete audit found
  six nonidentical reply checks among174,240, despite passing engine spot checks.
  Exact affected items and retained flags are recorded; raw scorer argmax vectors
  and margins were not retained, so the cause is unresolved. The registered
  numerical verdict stays unchanged, but full-history parity is not all-pass;
  S-full remains unlaunched pending parity and funding review. No thresholds,
  histories or canonical analysis outputs changed.

- **Original 150M pick L0 evaluation home** (30 Oct). No new training or reselection.
  All1,452 items and8,712 forced-choice credits independently audited; original model
  inputs/pinned sources verified. The single-pick descriptive gold-minus-near cost
  is -4.697 points (95% item-level CI[-7.069,-2.325]), with the validity gate passing.
  Recorded generation gates pass22,956 replies. The documented lost partial run is
  distinguished from its authorized unchanged-recipe recovery ($6.973 cumulative).
  `experiments/tiny_chat_long_2026-10/RESULTS.md` retains evidence and limitations;
  the newly triggered S-full control still awaits review, so the programme stays open.

- **Registered five-seed HS1 extension complete** (30 Oct). All four new B/S-gist
  runs are home. S-gist minus B is SUPPORTED at -12.668 Y12 points (90% CI
  [-16.353,-8.983]); the original three-seed core stays unchanged. Independent
  source/input/report/statistical audits and 72,600 raw-credit replays pass.
  The S-full condition is now true, but no control campaign has launched and the
  broader programme remains open. Results and exact cache-resume exception:
  `experiments/tiny_chat_long_2026-10/RESULTS.md`.

- **Standalone BF16 attention primitive gated on L4** (31 Oct). All 32 fixed dense/causal FP32/BF16
  cases pass unchanged stage/end bounds; an independent NumPy 1.26.4 raw replay reproduces every
  metric and leaves evidence bytes unchanged. This validates the scoped six-product primitive
  with FP32 softmax/backward and the explicit BF16 surrogate, not production wiring, model loss
  or a speed adoption. Raw binaries remain outside Git; hashes and scope:
  `docs/receipts/bf16_attention_20261031/RESULTS.md`.


- **Four missing evaluation hash manifests added** (30 Oct). CI exposed absent manifests for
  IF-tiny, IF-distance and the two conversation-v2 evaluation sets. Each dataset is byte-identical
  to its first Git commit; both IF sets also regenerate exactly and match the recorded result
  hash prefixes. These are newly recorded pins, not recovered historical manifests. Frozen
  data and tests are unchanged; full hashes and origins: `corpus/hashes/manifest_backfill_20261030.json`.

- **Smol CUDA runtime comparison recorded** (30 Oct, Ember 7b546d8). The existing
  SmolLM2-135M F32 model passes 96 full-row/greedy gates and 24 immutable timing trials
  on an L4 against eager HF; source, weights and raw trials independently replayed.
  Decode medians are 3.80-3.95x faster on three synthetic prompt shapes, including
  synchronous CPU-ID to CPU-row transfers and runtime wrappers. This is a scoped
  runtime result, not isolated-kernel, chat or model-quality evidence. Gemma CUDA
  remains unmeasured and unqueued. Receipt:
  `../ember.cpp/experiments/hf_cuda_2026-10-30/receipts/smol_v3_20261030_final.json`.
  ROADMAP now distinguishes the 49-CTest/65-integration publication checks from
  pending GitHub CI and the 17 pre-existing whole-file formatting failures.

- **HF export publication and durable numerical evidence** (30 Oct). Complete staging,
  explicit overwrite, caught-failure rollback and stale-loader refusal protect existing exports
  (fe9271589). New parity receipts atomically publish hashed float32 HF/native rows and support
  NumPy-only offline numerical/greedy replay (68ccaf90c). Source inputs are snapshotted before
  execution and rechecked afterwards. The existing tiny untied Llama-flex fixture reproduces
  all 24 IDs and its historical8.94e-8 maximum with retained rows; old receipts remain unchanged.
  Combined regression:208 passes,one optional skip. Records: `docs/receipts/hf_raw_evidence_20261030/`.

- **Registered five-seed size extension complete** (30 Oct). All 15 selected 20M/60M/150M
  summaries are home. Primary size slope is EQUIVALENT within the registered ±0.01 Y/doubling
  interval; the 60M→150M secondary remains unresolved. Language-model panel losses decrease with size,
  carry means do not. The fixed-token scope and unchanged original 150M pick are explicit in
  `experiments/tiny_chat_2026-10/size/RESULTS.md`; independent source/input/report and OLS audits passed.

- **BF16/flash per-shape profile complete** (30 Oct). All 40 cells/80 passes/2,400 finite-loss
  steps verified against raw events/specs and 4,718 immutable source files. Single descriptive
  BF16/flash/combined speedups span 1.23–2.01×/1.01–1.07×/1.34–2.09×; no universal or significance
  claim. Raw timing arithmetic and deterministic corpus hash independently checked. Records:
  `docs/receipts/bf16_profile_20261030/RESULTS.md`. Attention-product optimization remains open.

- **Future data-cache resume provenance** (30 Oct, 4050567c7). New builds atomically save and
  verify `human.cache.json`, restoring exact human/decontamination/fetch provenance on resume.
  Legacy caches without this proof refuse rather than infer provenance. Eleven tests pass;
  historical experiment ledgers and frozen jobs/results remain unchanged.

- **Studio sweep heatmap** (30 Oct). Opens the existing mtsweep `cells.jsonl` contract,
  with typed factor axes and explicit fixed levels for all other factors. Saved means,
  seed counts and population SD remain distinct from missing/ambiguous coordinates; no
  hidden seed pooling or significance claims. Actual seven-factor sweep compatibility,
  nine core tests and installed-Chrome desktop/mobile QA passed; all15 Research Mode
  contracts and its browser smoke still pass. Receipt: `docs/receipts/studio_sweep_20261030.json`.

- **Studio Research Mode** (30 Oct). Clone a full saved spec with one typed parameter change
  and a fresh name/output directory; unknown fields remain intact and the original stays
  unchanged. Compare two saved JSONL event logs with loss overlays, validation points and
  honest early-stop/resume status. Fifteen contract tests, installed-Chrome desktop/mobile
  checks and an actual six-step CPU run of the downloaded clone passed. File processing is
  local; the panel does not automatically launch training. Receipt:
  `docs/receipts/studio_research_20261030.json`.

- **Untied Llama-flavour flex HF export** (30 Oct). Preserves biased Q/K/V/O attention,
  per-head RoPE permutation of q/k weights and biases, RMSNorm eps1e-5, bias-free SwiGLU
  and an untied head. A clean committed-source CPU build (3eb746f) trained a tiny
  three-layer d_ff40/theta1234 fixture; all42 HF tensors load bitwise, and three actual
  source/HF roundtrips match24/24 IDs (worst raw-logit error8.941e-8). All165 HF tests
  pass, including explicit tied-head and other modifier refusals. Tied flex remains author
  WIP and is not supported. Receipt and exact source snapshots:
  `experiments/chat_general/receipts/hf_flex_llama_roundtrip_2026-10-30/`.

- **Carry-12 three-seed core complete** (30 Oct). The existing frozen finalizer published
  all seven cells with seeds 1–3, no missing reports and all validity gates passing.
  Independent source/input/report rehashes passed. HT-SFT is SUPPORTED (+9.16 points);
  the other primary tests are unresolved. HS1 qualifies for its registered B/S-gist
  seeds 4–5 extension review, while S-full does not trigger. Results and immutable
  completion receipts are in `experiments/tiny_chat_long_2026-10/`; no new paid launch.

- **Plain GPT2-flavour flex HF export** (30 Oct). Exact attention, pre LayerNorm, learned
  positions, tanh GELU and untied residual models map arbitrary positive depth/MLP width
  to GPT2Config. All 41 tensors of a three-layer, d_ff40 CPU fixture load bitwise; three
  actual source/HF roundtrips match 24/24 greedy IDs (worst raw-logit error1.193e-7).
  Complete resolved flavour records and inventories are required; SWA, expression slots,
  alternate placements, q/k norm, convolution, tied heads and other modifiers still refuse.
  All115 HF tests pass including independent C++ tokenizer checks. Receipt:
  `experiments/chat_general/receipts/hf_flex_gpt2_roundtrip_2026-10-30/`.

- **Attention Residuals CPU latency diagnosis** (30 Oct). Historical Stage L0 receipts show
  the plain model used the NumPy IR scorer while X5 used `mtstudio`; all 91 chunk key sequences
  match. Removed the unsupported same-backend regression from ROADMAP. Exact arguments and
  source/input inventories accompany the existing controlled same-backend timing receipt;
  no general performance claim, model rerun or frozen experiment change.

- **Cross-platform wheels reconciled** (30 Oct). GitHub run40207438451 atbc9b3fb5 already
  built and smoke-tested cp310–cp312 on Windows x64 and both macOS architectures. Rechecked
  all nine actual smoke successes and repaired wheel names; removed the stale Linux-only
  roadmap item. Workflow artifacts are available; no PyPI publication is claimed.

- **Faithful legacy tokenizer and plain GPT2 HF export** (30 Oct). Serialized legacy
  tokenizers preserve C-locale ASCII case/whitespace and UTF-8 byte punctuation, including
  EOS and original readable spacing. Three real vocabularies match independent C++, Python,
  serialized Tokenizer and AutoTokenizer on 9,045 encode/decode cases and 3,000 generated-ID
  decodes; invalid UTF-8 token bytes and unrepresentable vocabulary shapes still refuse.
  Plain exact two-block GPT2 now maps all 29 tensors, keeps Conv1D layout, transposes its
  untied head and records zero dropout/tanh GELU. A six-step tiny CPU fixture passes three
  actual HF/source roundtrips (24 greedy IDs, maximum raw-logit error 8.95e-8). All 73 HF
  tests pass, including unsupported-architecture/tensor-inventory rejection. Receipts:
  `experiments/chat_general/receipts/hf_gpt2_roundtrip_2026-10-30/` and the adjacent legacy
  tokenizer receipt. This is interoperability evidence, not model-quality evidence.

- **Historical device gate reconciliation** (30 Oct). Removed stale third-party CUDA and
  BF16 noise-floor work from ROADMAP after checking original hashed receipts:23 device legs,
  five trained models with40/40 greedy parity each, and recomputed primary/secondary loss
  arithmetic for all three BF16/flash configurations. These completed on26/25Oct respectively;
  no new GPU run. The full per-shape throughput profile remains open.

- **Standalone fetch0.1.1 release preparation** (30 Oct). HF skeleton emission now rejects
  incompatible explicit mechanisms and invalid geometry, preserves supported activation choices,
  and declares skeleton assumptions/constants. Independent review added absent-residual and
  bare-SiLU gating refusals;32 focused tests pass including HF AutoConfig reload. Distribution
  publication is not yet complete.

- **Bounded rule-based MoE fields** (30 Oct). Explicit adopted-model declarations and
  single-value architecture-table rows can supply quoted `moe`, `n_experts` and `top_k`.
  Baseline/related-work counts, human annotators and ordinary dense FFNs are excluded.
  Contradictions, preliminary/ambiguous adoption, own-model retractions and invalid expert/
  top-k relationships abstain; top-N accuracy/router metrics cannot become routing counts.
  All simple C++, mtsweep and packaged HF-config emitters refuse stated or unresolved MoE
  before creating an output, rather than silently producing a dense network. Metadata-only
  extraction remains available. This closes the bounded field-extraction item, not universal
  MoE parsing, gate-normalization inference or MoE reconstruction. The vendored package is
  synced; independent review also covers copular adoption and rejected dense alternatives.
  All 171 offline extractor/LLM-extractor tests pass, including 48 MoE cases; Ruff passes.

- **Extractor HTML field escaping** (30 Oct). Public-API identifiers, finding verdicts and
  runner-up scores are escaped before HTML rendering, matching values/evidence and the new
  HF comparison report. A regression covers all three formerly raw interpolation sites.

- **Offline HF-config cross-check** (30 Oct). Both extractor CLIs accept `--hf-config PATH`.
  Explicit numeric aliases are compared with paper-derived primary fields and every size
  variant independently. Differences, missing values and source quotes remain visible in
  JSON and escaped HTML; no checkpoint/variant is selected and no paper value is filled or
  overridden. Conflicting aliases, booleans and invalid numbers reject the supplied config.
  Local file hashes are recorded, and zero comparable fields is reported explicitly. The
  public package API exposes `crosscheck_hf_config`; its vendored extractor is synchronized.
  Offline regressions cover immutable paper evidence, aliases, variants, missing/nested values,
  malformed inputs, both CLIs and HTML escaping; 122 extractor/LLM-extractor tests pass.
  Unverified, quote-only, contested and unresolved paper findings retain their status/quote and
  cannot be promoted to a comparable assertion by matching HF values. The HF comparison is removed from ROADMAP;
  rule-based MoE field extraction remains open (the LLM extractor already has those fields).

- **HF export SFT roundtrip** (30 Oct). The exporter preserves the current llama-BPE tokenizer
  using its exact GGUF reconstruction, faithfully converts native paperkiln byte-BPE,
  and refuses legacy word-level exports until their exact source semantics are supported. Corrected the HF RMSNorm epsilon
  from 1e-6 to the training tape's 1e-5. The verifier resolves a real spec, checks every
  exported/loaded tensor against the source mapping, compares source/HF prompt IDs, and checks
  greedy IDs plus raw output logits instead of whitespace-split text. Thirty-five HF regressions pass (including independent C++ tokenizer parity).
  Non-default recorded RoPE theta is preserved; GQA, q/k norm, alternate/partial RoPE,
  untied heads, recursion, attention residuals, MoE, state tokens and other unmapped variants
  are explicitly refused.
  An existing 7.6M, four-layer SFT model passed three eight-token prompts (ordinary prose,
  punctuation/newline, non-ASCII): all 24 IDs match, all 38 mapped tensors match exactly,
  maximum raw-logit error 9.54e-6 under unchanged 2e-4 absolute/relative tolerances.
  The older CPU binary scored identical raw IDs with only the unused chat field omitted;
  receipts record that delta and original source hashes. This verifies this small SFT numeric
  roundtrip, not chat-template rendering, broad model families or language quality.
  Native byte-BPE uses a serializable byte-class split, canonical byte alphabet and ranked
  merges (no whole-token shortcut); only its three control strings are added tokens, so
  literal `<unk>` keeps native text semantics. Both existing 8,192-token vocabularies pass
  3,010 varied Unicode strings each after serialization/reload against the independent Python
  encoder, plus 5,600 newline-framed cases each against the existing C++ encoder, with exact
  IDs and lossless UTF-8 decoding. Malformed vocabulary layouts/merges are refused. This is
  tokenizer parity on valid UTF-8 text, not an arbitrary invalid-byte string API contract.
  Receipts: `experiments/chat_general/receipts/hf_sft_roundtrip_2026-10-30/`.

- **Fetcher lexical gaps closed** (30 Oct). Bare GeLU/Gelu spellings now normalize to GELU,
  while gated compounds remain GeGLU. A rejected runner-up stays in the reported mentions but
  cannot create a contested/family verdict; a further eligible alternative can still contest.
  Fifteen offline extraction tests pass, including the existing symmetric-use and top-veto controls;
  Ruff passes. Removed the completed lexical work from ROADMAP. No benchmark labels changed.
  Independent edge review also prevents a rejected top candidate from bypassing its veto via a
  same-GLU-family verdict; an eligible third candidate still creates a genuine contest. Seventeen
  extraction tests pass. Cached-corpus audit is 82 correct / 17 abstain / 0 wrong versus 79/20/0;
  both dated receipts preserve unchanged label and corpus identities.
- **Completed operational/release work removed from ROADMAP** (30 Oct). The VM relay snapshot
  was already implemented in 76f59c66: it copies and checks a checkpoint set before transfer,
  preserves a pending snapshot until home, and commits local state last. The browser F16 release
  is now published (Space 8e107b9, website fdbaa1d; root stack log has verification), so neither
  item remains open. No new experiment result is implied.

- **Preference-judge stopped-call accounting** (30 Oct). A CLI error with a reported nonzero cost
  is now recorded in the failure ledger before judging stops; a free account-limit response still
  creates no verdict or artificial failure. Subsequent per-call budget reservations include the
  largest historical failed call as well as successful calls. Offline post-training tests: 43 pass,
  3 optional skips. Full judging is blocked by the observed weekly reset (2 Nov, 9am Sydney);
  `experiments/chat_general/receipts/pref_full_blocked_20261030.json` records the zero-cost attempt.

- **Preserve judge rubric identity on Windows checkout** (30 Oct). Scoped `.gitattributes`
  rule forces LF for `tools/posttrain/rubrics/*.txt`; both working rubrics restored to their exact
  existing Git-blob bytes. Raw-byte rubric hashing and all cache keys stay unchanged. Before the
  fix the full-sample plan missed every cached ranking (6,040 calls); after it, 380 rankings are
  reused and 5,660 remain. Regression checks the shipped LF bytes and historical chat_v1 identity;
  17 post-training tests pass. Readiness receipt: `experiments/chat_general/receipts/pref_full_readiness_20261030.json`.
  No paid calls. The pilot-rate estimate including recorded cache spend and the 600-call gate is
  $198.47 against the $200 cap; actual future calls must still obey the existing budget guard.

- **Roadmap reconciled with completed stack work** (30 Oct). Removed stale running E5/E1/recursion
  status, the completed winner selection and preference pilots, the BF16 "building" entry, delegated
  MoR/MSC decisions (25 Oct), published Hub models and completed browser family implementation.
  Existing receipts establish real-L4 WebGPU gates as well as CPU/SwiftShader checks; newer op-specific
  device gates and browser speed limits remain open. Full preference execution follows the later
  00ecd0d4 decision (256 tokens, unchanged rubric, $200 total cap), not the superseded pilot hold.
  Full sampling is already home (3,020 candidate rows). Read-only judging plan: 5,660 calls remain
  with the original LF rubric/cache; restored CRLF bytes falsely require all 6,040 calls. No judge
  call was made; preserve the original rubric hash before executing the remaining work.
  The size extension has fired but lacks four matched runs; its initial-three-seed pick inputs are
  verified in `experiments/tiny_chat_2026-10/size/readiness_2026-10-30.json`. Removed completed research
  history from the open list, retained unverified follow-ups and parked work, and corrected the obsolete
  word-level-only claim without closing coalfire's separate tokenizer/parity contracts.

- **Resume archives preserve completed remote runs** (30 Oct; 6db8fcf6). The transfer runner
  skips archive members belonging to a destination run that already has `result.json` and accepts
  that completed result during verification. An adopted VM's final events and results cannot be
  replaced by an older local partial archive. Four filesystem regression cases execute the actual
  VM restore snippet; all pass. Removed the completed runner item from ROADMAP.

- **Batched `mtstudio score`, the fused head on the GPU, step_profile limits, runner_multi's home jobs** (28 Oct;
  LOCAL_LLM_STACK.md section 10). `mtstudio score --batch-tokens N [--no-window]` (opt-in) packs consecutive
  requests into one forward (`nn::Llama::forward_packed_hidden`, `Llama::head`; the head on scored rows only, one
  device step window): 7.5x and 8.1x on an L4 for a 60M model's Carry-5 and Carry-12 requests, within ROUNDING of
  the unbatched scores, forced choices identical; `--score-batch-tokens` in carry5_eval.py, carry12_eval.py and
  `tiny_chat_run.py post` (default 0: unchanged). `tests/test_score_batch.cpp` (in colab_cuda_validate.sh),
  `test_score_batch_cli.py`, `test_score_batch_post.py`, `tools/score_batch_bench.py`. `train.fused_head` passes
  its device gate (off byte-identical to c6365b88; 0.97-1.04x fp32; 11-34% less peak memory). step_profile.py:
  per-run time limit with a stall report, launch counts per step (`gpu_n`, `device::gpu_profile_counts`).
  runner_multi.py: a data job is home and FINISHED in the tick its last output lands; adopted state dirs mark
  such jobs FINISHED (`tests/test_runner_home.py`).

- **pk_base_v1 and pk_sft_v1 on the Hugging Face Hub** (28 Oct; LOCAL_LLM_STACK.md section 10).
  [johnnyteutonic/pk_base_v1](https://huggingface.co/johnnyteutonic/pk_base_v1) (MIT: F32 GGUF, the gated
  Q4_K_M and Q4_K_S, safetensors) and [johnnyteutonic/pk_sft_v1](https://huggingface.co/johnnyteutonic/pk_sft_v1)
  (CC-BY-NC-4.0, because its SFT set includes TinyChat: F32 GGUF, safetensors). `tools/publish_hf.py` gains a
  release mode: `--file` (chosen files), `--card` (a hand-written card, used as written when it has a metadata
  header), `--sha256` (expected hashes, checked before anything is staged), `--stage` (the repo as a folder of
  hard links, uploaded with the Hub's resumable large-folder upload), `--verify` / `--verify-only` (download
  every file back and check its sha256); `--env-file` now loads only the token lines; a run directory's README
  names the GGUF's own tokenizer instead of always "word-level". `tests/test_publish_hf.py` (9, no network).

- **New primitives drafted by Claude: Shaw-style relative positions, and Primer's MDHA wired** (27 Oct;
  LOCAL_LLM_STACK.md 5.4, D2; keramos v1.4.0 pinned). `ops::shaw_attention` (Shaw, Uszkoreit and Vaswani 2018,
  arXiv 1803.02155, eqs. 3-4): relative key and value vectors by the clipped query-key distance, shared by
  the heads or one set per head, keys only optional; forward in keramos's stated order, backward, host
  loops. flex `position: "shaw_relative"` (`rel_max_distance` = Shaw's k, `rel_per_head`, `rel_values`),
  exported as `attention.relative_kv` (IR_SPEC 5.15, rule 40). flex `qkv_conv: K` (Primer, So et al. 2021,
  arXiv 2109.08668): a causal depthwise convolution on the fused q, k, v projection before RoPE, no bias,
  exported as `conv1d_causal` over 3d channels with its recurrent slot; no new op. `ir_eval` runs
  `relative_kv` for the suite's op and gradient cases. The extractor's `positional` gains `shaw_relative`
  with `rel_per_head` and `rel_values` (boolean quotes must name a head or the values); the chain builds
  both mechanisms and takes `--assert family=decoder_only` to build an encoder-decoder paper's decoder.
  Gates: `tests/test_shaw_attention.cpp` 4/4 (binary64 formula 2.5e-7, FD 9.3e-5); `test_flex` 13/13;
  keramos v1.4.0 through mtstudio 158/0/307 (both new models within 1.4e-7, greedy 4/4; all 11 gradient
  cases); Shaw's paper live through the chain (gate $0.006, extraction $0.22).

- **keramos v1.3.5 pinned; llama_attnres under recursion gated for ember** (27 Oct; `spec/`, LOCAL_LLM_WORK.md
  D4). v1.3.5 states `"tensor_index": "application"` in full (IR_SPEC 7.2) with six cases on
  pk_llama_attnres_rec, makes an engine's own refusal of a case a skip (11.3) and adds Appendix C's ONNX Runtime
  GenAI row; no exporter change. `tests/test_llama_attnres_ir.py`: ar_rec (1 + [2] x 2 + 1, weight_index
  [0, 1, 2, 1, 2, 3], d 32, one pair of queries per application) is a gate model (ATTNRES_GATE_DIR writes its
  ember files, as for ar3 and ar_x5), and the test checks that the document without its tags moves the logits
  (2.1e-2 of max(1, |ref|) on 8 prompts); numpy IR vs mtstudio score 3.2e-7, greedy 40/40. Gate files
  `C:/ml_artifacts/attnres_gate_rec/ar_rec`.

- **The extractor bench under the 26 Oct prompt** (27 Oct; prompt e6c6c505, claude-opus-5-5 at effort high;
  tag bench3, $18.08 for 39 calls, Primer a cache hit from 26 Oct; Galactica fell back to claude-opus-4-8 as
  on 22 Oct; receipts `papers/llm_extract_receipts/2026-10-27_bench3.json`, `..._bench_scores.json`).
  - Flavours 81 correct / 11 abstain / 0 wrong (22 Oct prompt: 83 / 9 / 0): RoBERTa's GELU is inherited
    through a quote that names BERT-large by the paper's `\bertlarge{}` macro, which fails the ancestor
    check; GPT-NeoX's activation came back as "other:gelu (exact vs tanh form not stated)".
  - Sizes 569 / 2 / 71, 53 unlabelled (580 / 0 / 62, 44): Qwen2's MoE 57B-A14B left out of the configs (7
    abstain); RoBERTa's vocabulary not stated (2); DeepSeek-V2's context read as the extended 128K and the
    Lite's table entry 32K, where the labels have the 4K pre-training length (2 wrong). Block graph 91 / 2 / 2,
    unchanged.
  - The new fields, labelled blind off the LaTeX on 27 Oct (`new_fields` in `bench_truth_structure.json`;
    3 adjudications): per-sublayer placement 17 / 0 / 0 per sublayer (16 / 0 / 1: Primer's pre attention
    and branch_post FFN now stated); qk_norm and ssm 40 / 0 / 0 (no paper of the 40 has either, so these
    count false assertions only); positions outside the old lattice 5 / 0 / 2 (3 / 1 / 3: LaMDA's T5 bias,
    was relative); T5's 32 buckets and distance 128 right; new-op input roles 6 / 0 / 1 (squared ReLU,
    Primer's norm, Gemma 2's final soft cap, its attention soft cap as a primitive, the Transformer's and
    PaLM's scales as affine nodes); no learned input or gate a paper lacks (40 / 0 / 0).
  - `bench_score.py` scores these fields and prints them beside a baseline (`--baseline`); tests 10/10.

- **pk-bench2 at home** (27 Oct; `experiments/benchmarks_2026-10-26/RESULTS.md`; one RunPod L4 on 26 Oct at
  5e00cd40, $0.87).
  - HellaSwag for every model now scores each item by its own requests. The pk-bench runs are re-aggregated
    from their own chunks with fafc4750's keys (`bench2.py hellaswag`): acc_norm pk_base_v1 0.292 -> 0.299,
    pk_sft_v1 0.287 -> 0.295, live 0.264 -> 0.270, the first three picks 0.262-0.263 -> 0.266-0.267,
    Pythia-160m 0.298 -> 0.306, SmolLM2-135M 0.425 -> 0.436. ember's full-set pass gives the same values for
    pk_sft_v1 and live.
  - The benchmarks of pick_e1_L4, pick_e1_60m_L32 and pick_e5_s0 (hfgguf; CUDA mtstudio on every 400th item
    within 2.6e-5, argmax 3,827/3,827 each), and IF-tiny of pick_e1_L4 (macro 0.195, bare 0.077).
  - ember 580e108's POST /score against hfgguf and mtstudio, per item: every choice equal on all 28,548 items of
    five models, max |dlogprob| 1.0e-4; the 14 differing argmax positions are top-two ties within 1.5e-5 in
    the numpy IR.
  - pk-bench2's IF-tiny job stopped at prompt 375 of pick_e1_60m_L32: ember's /chat answered HTTP 500 for a reply
    cut inside a UTF-8 character. `tools/iftiny_eval.py` now raises with the server's message
    (`tests/test_iftiny_eval.py`); `bench_vm.sh` brings ember's stderr home; ember e57c0fb (branch
    `iftiny-utf8`) carries the fix, and `tools/colab_chat/bench_v3.json` reruns the two picks.
  - pk-bench3 (one RunPod L4, 27 Oct, ed614ee8, ember e57c0fb, $0.32): IF-tiny macro pick_e1_60m_L32 0.138
    (bare 0.085), pick_e5_s0 0.110 (bare 0.087); replay 1,474/1,474 each; parity 1,234/1,234 and 1,314/1,314.
    pick_e1_L4 at e57c0fb gives its pk-bench2 replies at fe01b35 bit for bit (1,474/1,474, both passes).

- **ember's draft table at export, `export.draft_table`** (27 Oct; LOCAL_LLM_STACK.md section 7, speculative
  decoding; off by default, and off is the export as before).
  - `tools/draft_table.py build --gguf M.gguf --out M.draft_table.bin SOURCES...`: the n-gram table ember's
    `EMBER_DRAFT=table` drafter loads (ember bace837, `draft_table.h`; chat speculation 1.91x / 2.23x plain
    decode on an L4), for each context of up to 4 tokens its most frequent next token, built from the chat text
    the model was trained on through the GGUF's own vocabulary. Sources: chat-format text, or JSONL rows
    (`{"messages"}` rendered by the GGUF's chat template, `{"segments"}`, `{"prompt", "completion"}`,
    `{"text"}`). The builder is ember's `experiments/draft_2026-10-27/dr_table.py`, vendored; the defaults are
    the settings ember measured. A receipt `<table>.json` records the sources' sha256, the token stream's
    sha256 and the table's. `check` applies ember's loader's checks; the vocabulary size is the model's as
    ember reads it, and disagreeing sizes refuse.
  - Byte identity (`experiments/draft_table_2026-10-27/`): ember's `h_c8_p10.bin` (10.3 MB, the 13 human
    dialogue sources, 40.1M tokens) and `s_c8_p10.bin` (8.4 MB, SODA + OASST, 36.2M tokens) are reproduced
    byte for byte, and h_c8_p10 again from the sources' `dialogues.jsonl` rows.
  - mtstudio: `"export": {"draft_table": true}` writes `<name>.draft_table.bin` beside the GGUF after `run`
    (from `data.sft_examples`) and after `export`; `{"sources": [...], "max_bytes": N}` names the text. It
    refuses at parse without a GGUF format or a source, with an unknown key, and before training when the tool
    is missing (`MTSTUDIO_DRAFT_TABLE`, `MTSTUDIO_PYTHON`); a failure after training leaves the weights, GGUFs
    and result.json and exits non-zero. `tests/test_draft_table.py`.
- **The arXiv chain builds T5's bias** (27 Oct; LOCAL_LLM_STACK.md section 11 items 4.2 and 8). ember runs
  `attention.position_bias` (keramos v1.3.3: 364 pass, 0 fail, 49 skip, pk_flex_t5 included), so
  `positional = t5_bias` builds flex with `position: "t5_relative"` from the paper's `rel_buckets` and
  `rel_max_distance` (the extractor's fields, quote-verified) or `--assert rel_buckets=N rel_max_distance=N`.
  A T5-bias paper that states neither refuses by name (never mtstudio's 32 / 128), as do sizes the bucketing
  cannot take and T5's bias on windowed attention.
- **keramos v1.3.0 pinned** (27 Oct; `spec/`, LOCAL_LLM_STACK.md section 11 item 8).
  - Exports follow rule 38 (IR_SPEC 3.1): a file names a llama.cpp architecture only when its IR is that
    architecture's graph. gpt2_swa (windowed gpt2 and windowed gpt2-flavour flex), kimi, recurrence and plain
    flex keep gpt2's tensors but are written with `general.architecture = "paperkiln"` and `paperkiln.*` keys
    (`Gpt2ExportConfig::architecture`), so llama.cpp refuses them instead of computing softmax gpt2; dense gpt2
    stays gpt2. The export event's `arch` says which. Earlier files: keramos's `tools/gguf_relabel.py` (the
    200 stored documents and GGUFs of `tools/ir_corpus.py` pass rule 38 as they are).
  - T5's relative bias exports: flex `position: "t5_relative"` writes keramos 1.3's `attention.position_bias`
    {weight `rel_pos_bias.weight`, buckets}; the `PAPERKILN_IR_UNRATIFIED` gate and the patched numpy lines are
    gone (`tests/test_flex_ir.py` checks the file with keramos's validator and reference). The arXiv chain
    still refuses `t5_bias` until ember runs `position_bias`. Recipe `pk_flex_t5` for the suite's model case.
- **The fused LM head + cross-entropy, `train.fused_head`** (26 Oct; LOCAL_LLM_STACK.md section 11 item 5.1;
  off by default, and off is the step as before).
  - `ops::linear_cross_entropy(h, W, layout, bias, targets, mask, upstream, chunk_rows)`: the mean NLL of
    softmax(h W + b) over the kept rows, never holding the [tokens, vocab] logits. The design is Liger-Kernel's
    FusedLinearCrossEntropy (arXiv 2410.10989): chunks of tokens, gradients computed in the forward, three GEMMs
    per chunk. As in Cut Cross-Entropy (arXiv 2411.09009) no logits are stored, and no gradient entry is
    filtered. It takes an untied head (`nn::Linear` W [d, vocab]) or a tied embedding ([vocab, d]), a bias, and
    the 0/1 row mask of `cross_entropy_masked` (SFT under loss policy all, final or each; state tokens; padding).
    The default chunk is Liger's rule (logits chunk about the size of the hidden states, at least 32 rows).
    `nn::Llama::forward_hidden` is `forward` up to its head, and `forward` is it plus the head, the same ops.
  - Host numerics against the unfused path (matmul -> cross_entropy[_masked] -> backward): the loss, dh and db
    are bit-identical wherever `ops::host_gemm_rows_free` holds (the vendored blocked GEMM's row split fixed by
    vocab and width alone). The head weight's gradient is summed in ascending row order, within the
    sequential-sum bound (n + 2) u sum |h dl| (worst 0.13 of it; the unfused 0.06) and 2.6e-7 to 4.3e-7 of the
    unfused one normwise. Every chunk size (1, 7, 32, default, all rows) and every OpenMP thread count from 1 to
    8 gives the same bits.
  - `tests/test_fused_head.cpp`: six cases (tied and untied, a bias, no mask, SFT, all-ones, scattered ignored
    rows, one kept row, upstream 1 and 1/2); a shape where the GEMM's rows are not free (dh 3.1e-7, dW 7.0e-7);
    finite differences (worst 2.9e-5); NoGrad; an unstated upstream (rescaled, 4e-7). On `nn::Llama` (tied and
    untied, with and without an SFT mask) the loss is bitwise, and so is every parameter's gradient but the
    head weight's (1.7e-7 to 2.3e-7).
  - mtstudio `train.fused_head` (llama family, lm and sft objectives; `train.fused_head_chunk` sets the chunk)
    is recorded in the model event and result.json when on. `tests/test_mtstudio_fused_head.py` (6/6): with
    the option off, an lm run (accum 2) and an SFT run train byte-identically to mtstudio at c6365b88 (every
    step's loss and grad norm, every eval, the safetensors and GGUF). With it on, the first step's loss is
    bitwise and the run tracks the unfused one; another chunk size gives the same run bit for bit.
  - `tools/fused_head_gate.py` on FineWeb-Edu text (d64 L2, T 64, batch 8, accum 2, 60 steps, the 16k llama3
    vocabulary, CPU): off vs c6365b88 bitwise, explicit off and a repeat bitwise; fused vs unfused first step
    bitwise, worst step 6.2e-8, final val 3.9e-9 relative, PASS
    (`docs/receipts/fused_head_gate_cpu_20261026.json`). The host path is the reference, not a speed path: 0.87x.
  - CUDA (`devops::linear_ce`; compiled with nvcc, not yet run on a GPU): the unfused CE kernels in place on
    each chunk, the GEMMs through the step-resident path. A new `gemm_tiled_acc` continues each element's FMA
    chain from the previous chunk, so with the tiled GEMM every output should be the unfused device path's bit
    for bit; under cuBLAS, float rounding. Device legs: `test_fused_head` leg 9 in
    `tools/colab_cuda_validate.sh` (tiled and cuBLAS), `tools/colab_chat/fused_head_cuda_v1.json` (the CUDA
    gate, `fused_head_gate.py` on 20m_L4 against c6365b88, `step_profile.py --configs fp32,fused_head` at 20m_L4,
    20m_L16, 60m_L32 and pk_base_v1; class head_fused).

- **Standard small benchmarks, IF-tiny and compliance by turn distance** (26 Oct;
  `experiments/benchmarks_2026-10-26/RESULTS.md`).
  - `tools/bench_build.py` builds HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, LAMBADA and BLiMP from
    pinned revisions; `sources.json` holds the licences and the sha256s, and the data are never committed.
  - `tools/bench_eval.py` scores log-prob forced choice in lm-evaluation-harness's zero-shot formats, with
    Wilson intervals and chance. Its engines are `mtstudio score`, HF transformers, `hfgguf` (a paperkiln GGUF
    through transformers' GGUF loader, batched; within 3.2e-5 of mtstudio, argmax 20,581/20,581), ember's
    POST /score and the numpy IR. `compare` holds two engines against each other on the same requests.
  - On the benchmarks, pk_base_v1 is level with Pythia-160m except on LAMBADA and BLiMP (HellaSwag 0.292, ARC-E
    0.437, PIQA 0.612).
  - IF-tiny v1 (`corpus/eval/iftiny_v1.jsonl`, 574 items, 15 families) and turn distance v1
    (`corpus/eval/ifdist_v1.jsonl`, 1,344 conversations) come from `tools/iftiny_build.py`. Their checkers are
    `tools/iftiny_checks.py`, with 64 unit tests in `tests/test_iftiny_checks.py`.
  - `tools/iftiny_eval.py` generates greedily through ember's /chat or HF. It replays a second pass, checks
    replies against mtstudio's argmax, and checks the server's prompt ids against schema v1.
  - On IF-tiny our models sit near their base rates (pk_sft_v1 macro 0.123, bare 0.062). Over distance their
    compliance rises with the compliant history, not the instruction.

- **The extractor's gaps, closed or refused by name** (26 Oct; LOCAL_LLM_STACK.md section 11 item 4.2, formerly 4.3).
  - **Flex options** (`tools/parity_model.hpp` FlexConfig, mtstudio `arch.custom`):
    - An expr activation or norm may read more inputs (`activation_inputs` / `norm_inputs`): a gate projection
      (`mlp.c_gate`), learned scalars and per-channel vectors with their inits. A norm's inits can differ by
      place (`init_attn`, `init_ffn`, `init_final`).
    - An expr on the embed output (`embed`, `embed_inputs`).
    - Norm placement per sublayer (`norm_placement`: pre, post, sandwich, branch_post, none, or
      `{"attn": .., "ffn": ..}`), with no output norm when both sublayers are post-norm unless `final_norm`.
    - `position: "t5_relative"`: T5's bucketed relative bias (`rel_buckets`, `rel_max_distance`).
  - **IR**: the learned inputs export as `scale` nodes over a row of ones (`expr` "1"), with tensors
    `blk.{i}.<slot>_p_<letter>.weight`, `embd_p_<letter>.weight` and `output_norm_p_<letter>.weight`. The
    sandwich norms are `attn_post_norm` and `ffn_post_norm`. None of this needs a keramos change.
    T5's bias has no keramos op: its GGUF export refuses by name. Only with
    `PAPERKILN_IR_UNRATIFIED=t5_relative` does the export write the proposed
    `attention.position_bias {weight, buckets}`, and only for the test of the proposal.
  - **Ops**: `ops::t5_position_buckets` (T5's causal table, equal to HF's f32 computation at five
    configurations) and `ops::relpos_attention` (forward and backward). At their defaults the options give the
    old model bit for bit (test_flex's equivalence pin).
  - **Checks**: `tests/test_flex.cpp` 9-11 cover gradient checks of every new input, every one of the 25
    placement pairs against the block composed by hand, and the bias op against causal attention and finite
    differences. `tests/test_flex_ir.py` runs ten models through ir_check, and the numpy IR matches
    `mtstudio score` within 2.7e-7 to 3.2e-7, greedy 40/40. The T5 model is checked against the proposed
    numpy lines.
  - **Extractor** (`papers/llm_extract.py`): new fields `norm_placement_attn` / `_ffn` (with `branch_post`),
    `qk_norm` (per_head, full, unspecified, none), positional `t5_bias` with `rel_buckets` and
    `rel_max_distance`, `ssm` (mamba1, s4) with `d_state`, `d_conv`, `expand` and `dt_rank`, and each new op
    input's role and init. The IR ops list gains conv1d_causal and selective_scan. Older replies read the new
    fields as not stated.
  - **Chain** (`tools/arxiv_to_chat.py`): it builds the above from the roles, including a norm's variants with
    other inits at other places. A per-head q/k norm builds on the llama family, and Mamba-1 builds from its
    sizes on the mamba preset. `--assert` takes `qk_norm`, `d_state`, `d_conv`, `expand` and `dt_rank`. A
    structural op in the block graph that no family wires there now refuses: Primer's conv1d_causal on q, k
    and v used to pass silently.
  - **Scoring** (`papers/bench_score.py`, `papers/bench_truth_structure.json`): sizes and the block graph by
    data flow, against labels read off the 40 papers' LaTeX. `flavor_bench.TRUTH` now gives Primer's norm as
    its modified layer norm (out of lattice), no longer the T5 baseline's RMSNorm.

- **The third-party ops train here** (26 Oct; LOCAL_LLM_STACK.md section 11 item 4.4, which this closes). The
  ops of the keramos thirdparty profile that paperkiln had only in numpy now have forward and backward passes in
  its autograd. A model that uses them trains, exports its GGUF with its IR, and runs in ember.
  - **Ops** (`include/microtorch/ops.hpp`):
    - `ops::apply_rope` takes `pairing` (interleaved, or NEOX's (x[j], x[j + n_rot/2])) and `n_rot` (partial
      rotary). At its defaults it is the original loop, statement for statement.
    - `ops::rmsnorm_group`: a per-head RMSNorm with a weight shared by every group.
    - `ops::repeat_heads`: grouped-query attention's kv sharing. The backward sums each group in ascending
      head order.
    - Mamba-1's `ops::conv1d_causal` and `ops::selective_scan`: the backward runs through time over the kept
      states, and every sequence of a stacked batch starts from a zero state.
  - **CUDA**: the rope kernel takes the pairing and the rotated width, still rounding as the host does.
    repeat_heads has kernels that are bit-exact with the host. rmsnorm_group runs the rmsnorm kernels over
    rows of the group's width. The two Mamba ops run on the host; paperkiln's CUDA path has no recurrent
    kernels.
  - **Checks** (`tests/test_thirdparty_ops.cpp`, ctest `thirdparty_ops`):
    - Every op against a float64 statement of its IR_SPEC formula: 1.0e-7 to 2.3e-7.
    - Every input's gradient against central finite differences: worst 3.5e-4.
    - RoPE equals a float re-statement of the host loop bit for bit, and rmsnorm_group over the whole row
      equals ops::rmsnorm bit for bit.
    - The llama options at their default values give the plain block bit for bit.
    - A grouped-query model equals the multi-head model whose k and v columns repeat, bit for bit.
    - KV-cached decode is within 6.0e-8 of the full forward, and a Mamba-1 batch equals its sequences bit for
      bit.
  - **Numpy agreement.** `tools/ir_eval` runs rope, rmsnorm, attention, conv1d_causal and selective_scan
    nodes through these training ops, and mtstudio's conformance adapter routes them there. Every keramos
    single-op case they express passes against numpy's expected values: rope interleaved, neox, neox_partial
    and grouped_query; rmsnorm row and group; attention causal and grouped_query; conv1d_causal; selective_scan.
    That is 10 pass, 0 fail. Three cases are skipped (freq_scale, weight_offset, a window), none of which
    paperkiln trains.
  - **mtstudio**:
    - The llama family takes `arch.custom.kv_heads`, `rope_pairing`, `rope_dims`, `qk_norm` and `rope_theta`.
      Each left out gives the plain model, unchanged, with an unchanged model event.
    - The mamba preset takes `arch.custom.ssm: "mamba1"` with `d_state`, `d_conv`, `expand` and `dt_rank`.
      The graph is keramos `tools/gguf_ir.py`'s mamba graph node for node, with llama.cpp's tensor names and
      A stored as the forward's -exp(A_log).
    - GQA alone exports architecture llama with head_count_kv. NEOX, partial rotary and q/k norms export as
      "paperkiln".
    - muon refuses grouped-query attention, because its per-head split assumes one kv head per query head.
      attn_res refuses the options, because llama_attnres runs its own attention.
  - **End to end** (`tests/test_thirdparty_ir.py`): five models train 40 steps on CPU and export.
    - The models: tp_gqa (4 q / 2 kv heads); tp_qwen (GQA + NEOX + q/k norms, theta 1e6); tp_mqa_partial (one
      kv head, rope_dims 4 of 8); tp_neox_partial; tp_mamba1.
    - keramos's checker accepts every file. Numpy IR against `mtstudio score` is within 2.5e-7 to 4.3e-7 of
      max(1, |ref|), greedy 40/40 on each.
    - The gate files for ember are in `C:/ml_artifacts/thirdparty_train_gate/`.

- **The host GELU gives the same bits at every OpenMP thread count** (26 Oct). `ops::gelu` on the CPU runs
  the vendored `Matrix::apply_gelu` on one thread (`apply_gelu_one_chunk`, src/ops.cpp). The core compiles that
  loop with -O3 -march=native -ffast-math, so GCC vectorises it with libmvec's tanhf: 8 lanes over each thread's
  chunk, then 4 lanes and scalar tanhf for the chunk's last chunk_len % 8 elements, and the three round
  differently. An element's bits therefore depended on where the chunks fell. At 3, 5, 6 or 7 threads a
  stacked batch differed from its sequences run alone, which is the deep_swa batch pin's failure with
  `OMP_NUM_THREADS=3`, present since the test was written.
  - Every thread count now gives the one-thread bits. Those are also the 2- and 4-thread bits wherever those
    chunks were multiples of 8 elements, which covers every gate's sizes. A probe hashed `ops::gelu` on seven
    shapes and FlexLM logits on three: at 1-5 threads every hash now equals the old one-thread hash.
  - mtstudio gpt2 and flex runs at 1-5 threads give weights byte-identical to the old one-thread run. Before
    the change the flex run differed at 5 threads.
  - `tests/test_deep_swa.cpp` section 4b runs the batch pin at 1-8 threads in process and checks each count's
    logits against one thread's. ctest gives 42/42 at 2 and at 3 threads.

- **Tiny chat, arm X5: Attention Residuals on the llama base** (26 Oct; `experiments/tiny_chat_2026-10/PREREGISTRATION.md`
  amendment 5). Jonathan's cross-layer idea on Carry-5. The arm is 20m_L16 with `llama_attnres` (full form at
  sublayer level: 33 depth sources, zero-initialised pseudo-queries, scale "sources"), against E1's three home L16
  runs paired by seed, at E1 L16's learning rate. HX5-main is an item-level paired test (3 x 888 items) with Stage
  L0's verdicts. The isoplot against the plain 20M runs is reported beside it.
  - The port's checks: `tests/test_llama_attnres_reference.py` (the fused op against
    `python/attn_res_reference.py` under torch autograd, worst 7.1e-6), and the `ar_x5` case of
    `tests/test_llama_attnres_ir.py` (the arm's exact shape: numpy IR vs mtstudio 2.7e-6, greedy 40/40, keramos
    v1.1.3's validator accepts it). The ember gate files are in `C:/ml_artifacts/attnres_gate_x5/`.
  - Jobs: `tools/colab_chat/tiny_chat_s0_x5.json` (CUDA gate, probes, a 500-step plain replay of E1's
    pt_e1_L16_r1 and an X5 smoke run), `tiny_chat_e1_x5_ctl.json` (E1's L16 post jobs re-run from their home
    weights for their per-item records) and `tiny_chat_e1_x5.json` (the gate needs the Stage 0 receipts; its post
    jobs bring home items, replies and the SFT model, so the pick runs at home when the seeds run on separate
    machines).
  - Tooling: `x5/bitid.py`, `analyze_x5.py` (in `analyze.py` as E1_x5_20m), `tools/carry5_attnres_trace.py` (the
    depth-attention weights at the answer token, through keramos's interpreter), and `tiny_chat_run.py gate
    --need-ok`.
  - Stage 0 (585206a8, one RunPod L4; A5.12):
    - CUDA gate 22/22;
    - X5 at 17,378 tokens/s against plain 20m_L16's 26,007 (0.668x, +15% peak memory);
    - a 500-step replay of E1's pt_e1_L16_r1 bit-identical to E1's log;
    - the smoke model numpy-vs-mtstudio 1.9e-6, greedy 40/40.
  - Ember e115fd3 is the post jobs' pin (`post --ember-commit`): ar_x5 40/40 on CPU, CUDA and WASM, and the
    control's model generated exactly as at cbdcb915.
  - Result (`experiments/tiny_chat_2026-10/x5/RESULTS.md`): **HX5-main EQUIVALENT**. D = Y(X5) - Y(L16) =
    -0.0101, 90% [-0.0299, +0.0096], over 2,664 item-level pairs; both cells pass the gate.
    - Resampled by item, frame or slot cluster, the 90% interval reaches below -0.03 through the knowledge
      family. The planted-fact families stay within 3 points, and every 95% upper bound is at most +0.021.
    - X5 lowers every loss by 0.03-0.05 nats and raises twin accuracy 2.8 points, with Y unchanged.
    - The isoplot through the 22 plain 20M runs is weak evidence (slope -0.019 Y per nat, unstable).
    - The control's re-runs reproduce E1's L16 summaries exactly, and the score audit leaves every Y unchanged.
    - The pick (seed 3) ran at home.

- **The general chat model's next step: pk_sft_v1** (25 Oct; paperkiln 0ad646d5..91f47fec, one RunPod L4,
  $3.19; `experiments/chat_general/RESULTS.md`). pk_base_v1 + lever 1's SFT set at T 512 as conversation rows
  (`tools/lever1_conversations.py`: one row per dialogue, the per-turn file's 463,695 replies in 313,606 rows)
  under `train.sft_loss` all, BF16 + flash, 18,620 steps, held-out loss 1.508: simple-QA 21/60 against big_sft
  11, ctrl_sft and moe_sft 7, the live model 2; ember aa50a61 greedy parity 40/40 on CUDA and CPU. The winner
  rule (PLAN.md, fixed before the results) names the live model as registered and pk_sft_v1 under its
  amendment A1. Tooling: `bpe.load_vocab` (chat_eval, qa_eval, chat_sample_ember and sample_candidates read the
  16k vocabulary), `tools/colab_chat/general_chat_v1.json`, `chat_eval_stage.py`, `ensure_ember_vm.py`,
  `chat_parity_vm.py`, `pref_sample_vm.py`, `experiments/chat_general/winner.py`; chat_eval deletes its
  reference-scoring runs (they filled a 40 GB volume).
- **The IR's specification is keramos** (25 Oct). The spec text, schema, validator, numpy reference, conformance
  suite (with engines' receipts), worked documents and prior-art survey are the keramos repository
  (github.com/JohnnyTeutonic/keramos, private; tags vR.S.P, IR_SPEC 2.6), vendored at the pinned tag in
  `spec/`, now v1.1.1 (6170ee05: revision 1.1 with the rulings on ember's review points):
  `spec/IR_SPEC.md`, `spec/conformance/`, the tools in `spec/tools/`, the worked documents in
  `spec/examples/`. `spec/KERAMOS_PIN.json` records the tag, its commit and the commit's manifest;
  `tests/test_keramos_pin.py` fails on any local edit and `tools/sync_keramos.sh TAG` refuses to overwrite
  one. paperkiln's side of the suite is `tools/ir_conformance_mtstudio.py` (mtstudio's adapter and the
  training of the pk_ models) and `tools/ir_corpus.py` (both repositories' stored documents).
  paperkiln's C++ follows v1.1.1: `ir::canonical` runs every digest check on the whole document before
  removing its `_` members, and `ops::recurrence` accepts a repeated index in a declaration (`W[c,c]`).
- **The CUDA mtstudio checked against the CPU build on trained models** (25 Oct; paperkiln ff00d60a,
  `tools/cpu_cuda_parity.py`, `tools/colab_chat/cpu_cuda_parity_vm.sh`, RunPod L4). Two builds of one
  commit, the CUDA one under the production environment, on the live chat model (dense, 14.7M), big_sft
  (dense, d768 L12, 91M) and fw_moe (MoE, E 8 top-2, 70M): `mtstudio chat` greedy on the 40 frozen
  prompts identical 40/40 on each (text, token count, stop reason); `mtstudio score --logits` on the
  CPU's replies: top-1 equal at every scored position (676, 677, 1,592), largest |dlogit| 7.4e-7,
  7.2e-7 and 1.15e-6 of the logit range, |dlogprob| under 1e-5; the reference-bits/byte SFT
  evaluation (on CUDA, the device evaluation) within 1.1e-5, 8.3e-6 and 3.0e-6 relative.
- **SFT evaluation on the device; deferred downloads for the gpt2 family (defect D1 fixed)** (25 Oct;
  paperkiln ff00d60a). When a run's own steps defer (CUDA, device op set, step residency, deferred
  downloads), mtstudio's SFT evaluation runs its forwards inside one step window: the parameters
  upload once per evaluation, and activations, logits and the softmax stay on the device. CUDA
  eval events carry `eval_device` and `eval_s`; `MICROTORCH_DEVICE_EVAL=0` keeps the windowless
  path; CPU builds and deferral-off runs are unchanged. D1 was LayerNorm's `xhat` forward cache
  dying stale inside the window (docs/CUDA_PHASE_B2.md, record D1); it is now a self-discarding
  cache, so gpt2-family and layernorm-flex runs can defer (`tools/arxiv_to_chat.py` no longer turns
  deferral off for them). Gates on an L4 (`tools/device_eval_gate.py`, `tests/test_gpt2_device.cpp`):
  val_loss identical to the windowless evaluation on llama (eval_batch 1 and 4), gpt2 and flex, and
  training losses bit-identical; 100 held-out examples of ~370 tokens evaluate in 0.55 s instead of
  8.37 s at d384 L6 and in 2.58 s instead of 40.0 s at d768 L12 (about 15x); gpt2 exact, swa, kimi,
  srd and 4-layer flex train 60 steps under deferral with glibc's heap checks on, every training and
  validation loss bit-identical to write-through. `mtstudio score --logits FILE` writes each scored
  position's logits row (float32), for the CUDA-vs-CPU gate.
- **Real k-quant blocks in the 4-bit exports** (25 Oct; paperkiln 255c9561). `gguf::quantize_q4_k`,
  `quantize_q5_k`, `quantize_q6_k` are llama.cpp b6000's (4762ad73) `quantize_row_q*_K_ref` with
  `make_qkx2_quants`, `make_qx_quants` and `nearest_int`, statement for statement, so `gguf_q4_k_m`,
  `gguf_q4_k_s` and `gguf_q4_0` store Q4_K / Q5_K / Q6_K wherever ne0 is a multiple of 256 instead of
  refusing (llama.cpp's fallbacks below 256 are unchanged). The 4-bit variants now write the IR keys
  before `general.quantization_version` and `general.file_type`, as llama.cpp's quantiser does (it
  copies the source keys, then appends those two): since the IR went into the GGUF the files had
  differed from llama.cpp's in that order alone. Gates: on pk_base_v1 (d 768, d_ff 2048, 12 layers)
  every one of the 111 tensors of q4_k_m (73 Q4_K, 13 Q6_K), q4_k_s (80 Q4_K, 5 Q5_K, 1 Q6_K) and q4_0
  is byte-identical to llama.cpp b6000's `llama-quantize` (built with GGML_NATIVE off) run on
  paperkiln's own F32 export, and so are the three whole files (sha256 78b3482f..., 7980aa54...,
  89d8581c...; `C:/ml_artifacts/next_base/base/pk_base_v1_q4_k_{m,s}.gguf`); `tests/test_gguf_q4.cpp` checks an 8-layer d 256 export's
  types and bytes and 9 patterns x 2 super-blocks per type against llama.cpp b6000's own reference
  rows (hashes of its bytes).
- **The build uses the vendored transformer_core by default** (25 Oct). `third_party/transformer_cpp`
  is canonical; a sibling `../coalfire.cpp` or `../transformer_cpp` is used only with
  `-DMICROTORCH_TCPP_SIBLING=ON` (or `-DMICROTORCH_TCPP_DIR=<path>`). A fresh configure from the
  workspace checkout had picked the lagging sibling and failed in `src/gguf.cpp`
  (`GGUFWriter::write_metadata_int32_array`); so had a clone placed in a home directory holding a
  stale `transformer_cpp`. `tools/sync_vendor.sh` refuses to overwrite the vendored copy while
  the two differ.
- **One conversation schema for instruction tuning and chat** (24 Oct; LOCAL_LLM_STACK.md section
  3, points 3-5). A conversation is `{"messages": [{role system | user | assistant, content,
  loss?}]}` (one leading system message, strict alternation, tool reserved); an instruction is a
  one-exchange conversation. The template is a versioned record (`paperkiln/1` = the old
  `<|user|>U<|assistant|>A<|endoftext|>` template byte for byte, system folded into the first
  user turn behind one constant; `paperkiln/legacy-word`; ember's ChatML Qwen2.5/Qwen3 and
  Gemma 3 records) in `tools/chat_template.py` and its header-only C++ twin
  `include/microtorch/chat_template.hpp` (render, Jinja, Ollama Modelfile, resolution). mtstudio:
  `{"messages"}` SFT rows under `train.sft_loss` {"turns": "all" | "final" | "each"} (default
  all), `chat.template` in the spec, `/chat` takes `system` (or a leading system history entry)
  and `thinking`, `/health` names the template, `score` and `sample-set` take `messages` rows,
  `sft-dump` writes the rows a spec trains on; every GGUF of a vocabulary with the v1 specials
  carries `paperkiln.chat_template` and the generated `tokenizer.chat_template`, with
  `<name>.Modelfile` beside the F32 file. The Python builders (chat_data, carry5_eval, chat_eval,
  carry12_build, `_common.render_dialogue`) render through it. Gates: v1 bit identity on 460
  frozen prompts, 9,288 Carry-5 and 24,684 Carry-12 prompt strings and 99,608 pinned dialogues
  (each corpus's pinned text.txt sha256 reproduced), ids per piece = whole under both
  vocabularies; mtstudio's conversation rows bit-identical to the segments and per-turn rows on
  600 pinned dialogues, T 96 and 512, state tokens k 0 and 4, both vocabularies; Jinja = record
  under jinja2 on 1,000 random conversations; the third-party records = their HF templates; 29
  edge cases in both languages (`tests/chat_template_cases.json`).
- **BF16 mixed precision and flash attention, opt-in** (24 Oct; `train.precision` "fp32" |
  "bf16", `train.attention_kernel` "tiled" | "flash", llama family; the defaults are the previous
  step bit for bit and write nothing new into events, results or GGUFs). BF16 rounds the inputs
  of every `ops::matmul` GEMM (forward and both backward products) to bfloat16, round to nearest
  even, and accumulates and returns FP32: on the device `cublasGemmEx` (CUDA_R_16BF in, CUDA_R_32F
  out, CUBLAS_COMPUTE_32F; sm80+), elsewhere a host emulation (rounded copies through the FP32
  path). Master weights, AdamW state, gradients, norms, softmax, RoPE, residual adds and the loss
  stay FP32; no loss scaling (bfloat16 has float32's exponent, and no gradient is stored in it).
  Recorded as `precision` in the model event and result.json and as `paperkiln.train.precision`
  in the F32/Q8_0 GGUFs. Flash attention (`ops::attention_heads_flash`, `src/cuda_flash.cu`):
  online-softmax forward that keeps the per-row log-sum-exp instead of the [H T, seq_len]
  weights, and a recompute backward (dK/dV per key tile, dQ per query tile, no atomics), FP32,
  on the full, causal and turn_state masks, head widths up to 64; host loops over the same tiles
  are the reference. CPU gates (`tests/test_mixed_precision.cpp`): rounding against a
  nearest-neighbour reference on 1e6 floats, BF16 GEMMs and `ops::matmul` equal to the rounded
  FP32 GEMMs bit for bit, flash vs tiled within 5.7e-7 (y) and 5.6e-7 (dqkv) relative on every
  mask, FD gradchecks of the flash op (worst 1.1e-3), a 60-step llama run: flash vs tiled 2.3e-7,
  BF16 vs FP32 4.5e-3 per-step loss. Through mtstudio on the CPU (`tools/precision_gate.py`, d 64
  L2, 150 steps on FineWeb-Edu text, 16k vocabulary): precision off vs the pre-change binary bit
  for bit (every step's loss and grad norm, the validation loss, the safetensors and GGUF sha256),
  as are the explicit-off spec and a rerun; 20-step moving-average loss vs FP32 tiled: flash
  1.9e-8, BF16 7.5e-5, both 1.0e-4 (receipts `docs/receipts/*_cpu_20261024.*`). Every CUDA
  target compiles (nvcc 12.6, sm_89; the flash kernels spill nothing). Device legs in `tools/colab_cuda_validate.sh`; the GPU
  gate and throughput run is `tools/colab_chat/bf16_gate_v1.json`. `tools/step_profile.py`
  (per-op-class step breakdown) and `tools/precision_gate.py` (off-bitwise and loss-curve legs
  through mtstudio) come with it.
- **Masked cross-entropy on the device** (24 Oct, the SFT loss): `ops::cross_entropy_masked`
  had no device path, so every SFT step downloaded the [rows, vocab] logits, ran the softmax on
  the host and uploaded the gradient (1 GiB each way at batch 32, T 512, 16k vocab; the tiny-chat
  pilot's SFT ran at 3,490 tok/s on an L4 against 25,113 for pretraining the same model). New
  `devops::ce_fwd_masked` / `ce_bwd_masked` run `k_ce_fwd` / `k_ce_bwd` instantiated with a mask
  (masked rows carry target -1: nll 0, gradient exactly 0); the unmasked instantiation compiles
  to the previous kernels, so `cross_entropy` is untouched, and an all-ones mask still returns
  `cross_entropy` itself. Kept rows' P and gradient are the unmasked kernels' bit for bit; the
  loss differs from the host op only by the sum's order. The host path is unchanged. Tests:
  `tests/test_cuda_ops.cpp` leg 9 (Colab, `tools/colab_cuda_validate.sh`).
- **Mixture-of-Recursions, causal token choice** (23 Oct, tiny-chat amendment 4):
  `arch.custom.recursion` schedule "mor" with {"mor": {route "token_choice", gate "score" |
  "none", balance 0.1, z_loss 0.001, kv_rows "active"}}: one linear router (state key
  `recursion.router.weight`, GGUF `rec.0.router.weight`) at step 0's input picks each token's
  depth 1..R by its largest logit; the token takes that many repeats, each x + g (S(x) - x) with
  g the chosen depth's probability, on its sequence's active rows only (gathered; attention causal
  among them at absolute positions, so prefill equals decode); the balancing loss and z-loss go
  through `MoEAux`. Step and eval events log the depth fractions, mean depth, entropy and router
  losses; `mtstudio score` writes each prompt's depth trace. IR: the `depth_route` node and the kv
  slot's `kv_rows: "active"` (normative in ir.hpp), family `llama_mor`; `tools/ir_numpy.py` and
  `ir_check.py` follow it. Tests: `tests/test_mor.cpp` (forced full depth with gate none = fixed
  recursion bit for bit, logits and gradients; prefix, decode and stacked-sequence causality with
  depths identical; per-application rows exactly the active tokens'; FD gradcheck worst 7.2e-3;
  the device-resident step for Colab) and `tests/test_mor_ir.py` (numpy IR vs mtstudio within
  4.0e-7, greedy 40/40, depth traces 40/40). Jobs: `tiny_chat_s0_mor.json`, `tiny_chat_e1_mor.json`.
- **Tied-block recursion** (23 Oct, the tiny-chat recursion arm): llama-family
  `arch.custom.recursion` {prelude P, block B, repeats R, coda C, schedule "fixed", inject
  "none" | "add"} stores P + B + C layers and applies P, then B R times, then C, each application
  with its own KV cache and full backprop through all of them (`nn::layer_schedule`). IR: the
  block key `weight_index` {"per_layer": [...]} (application i reads blk.{m_i}), family
  `llama_recursive`, GGUF architecture `paperkiln` with the shared layers stored once;
  `tools/ir_numpy.py` and `ir_check.py` follow it. Tests: `tests/test_recursion.cpp` (R = 1 bit
  for bit, tied vs untied bit for bit, FD gradcheck worst 2.7e-3, KV cache, AdamW once per
  tensor, the device-resident step for Colab) and `tests/test_recursion_ir.py` (numpy IR vs
  mtstudio within 3.8e-7, greedy 40/40).
- **Tiny-chat experiments, pre-registered and prepared** (23 Oct): `experiments/tiny_chat_2026-10/`
  (PREREGISTRATION.md, `make_jobs.py`, `analyze.py`, shape and SFT specs),
  `tools/colab_chat/tiny_chat_run.py` and `tiny_chat_data.py`, jobs files
  `tiny_chat_{s0,e5,e1,s0_60m,e1_60m}.json` (not launched; E5's dialogue levels follow a
  registered rule from the measured human corpus, `design.json`, provisional until the default set
  is home). mtstudio: SFT examples in a
  `segments` form (a whole multi-turn dialogue per row, every assistant turn trained; a one-turn
  record trains bit for bit like its `{prompt, completion}` twin), `train.eval_windows` (LM
  validation windows, default 8 as before), and a clear refusal of a corpus whose training part or
  held-out 5% is shorter than T + 2 tokens (the window draw used to wrap round and crash);
  `tools/chat_data.py sft --all-turns` writes the segments form. Tests:
  `tests/test_mtstudio_segments.py`, `test_tiny_chat_pipeline.py`, `test_tiny_chat_analyze.py`,
  `test_chat_data.py`.
- **Runtime-defined ops** (22 Oct, `1159d63e`): the `expr` op, a closed elementwise grammar
  with row reductions parsed at runtime into differentiable primitives (new tape ops exp, log,
  sqrt, tanh, erf, abs, div, pow, maximum, minimum, where, rowsum, rowmax, broadcast_cols);
  flex `activation: "expr:<e>"` with an IR `expr` node; `tools/ir_expr.py` numpy reference;
  `tools/expr_eval`; `tests/test_expr.cpp` and `tests/test_ir_expr.py`. C++ vs numpy within
  1.19e-7; ember.cpp vendors the parser.
- **Third-party architectures, reference side** (22 Oct, `daab3e1d`, `a5cddee1`, `09943ab4`,
  `93a9037e`): `tools/gguf_ir.py` builds the IR from llama.cpp metadata (qwen2, qwen3, gemma3,
  gpt2, gptneox, mamba, windowed llama/mistral); `tools/ir_numpy.py` gains grouped-query
  attention, NEOX and partial RoPE, per-head RMSNorm, per-layer attributes, weight_offset,
  freq_scale, conv1d_causal and selective_scan; `tools/thirdparty_parity.py`,
  `tools/thirdparty_synth.py`, `tools/thirdparty_hfcheck.py`. GPT-2 124M and Pythia-160m 40/40
  greedy vs HF; Gemma 3 and Mistral sliding window on random tiny models. `tools/ir_run.py`
  runs on `ir_numpy`'s kernels (`d2b6d402`).
- **LLM-based arXiv extractor** (21-22 Oct, `dd772abe` onwards): `papers/llm_extract.py`, one
  Claude call per paper, a deterministic verifier, receipts and a cost cap; 82 / 9 / 1 on the
  40-paper benchmark against the rule-based 72 / 20 / 0. The docs present it as the main path.
- **Qwen pre-tokeniser** (22 Oct, `a8510d93`): `llama3_tokenizer.hpp` `max_digits` for
  llama.cpp's qwen2 regex.
- **Chat-model data and mtstudio features** (20-21 Oct): lever 1's SFT set
  (`tools/lever1_sft.py`), lever 2's dialogue corpora (`corpus/fetchers/dialogue_pt.py`,
  `corpus/decontam.py`, `corpus/mix_pretrain.py`), the 16k Llama 3 scheme vocabulary and its
  exact C++ encoder, `.ids` corpora, `data.anneal`, the `/chat` repetition penalty, the KV cache
  (`chat --kv-cache`), batched SFT evaluation (`1f1f906b`); the next-base launch
  (`tools/colab_chat/make_next_base.py`); Colab data jobs, resumable uploads and weights-only
  resume (`runner_multi.py`, `jobs_vm.py`).
- **Atlas Stages 2-3 with `rope_heads=all`** (10 Oct): 84 runs; none of the nine affected
  findings overturned (`docs/ROPE_HEADS.md`).
- **sparse_s1_longbudget** (10 Oct): 20 runs, "not supported / not supported"; registry row
  S1f-longbudget-monotone.
- **The transfer paper** (submitted to Machine Learning 9 Oct; Study 3 done 11 Oct, reading
  mixed; registry row T3-rate-specific-disagreement) was retired on 20 Oct after the desk
  rejection.

- **Fixed: llama-family RoPE rotated only the first head** (6 Oct,
  `cb135a1`). `arch.rope_heads` selects `all` (the default for new runs) or
  `first` (every earlier llama run, which replays as recorded); exports and
  the browser engine follow the run's record. Nine Atlas Stage 2/3 findings
  are under review; see `docs/ROPE_HEADS.md`.
- **Chat quickstart** (6 Oct): `quickstart_chat.sh`; TinyChat corpus
  generator (v1 and v2) with an end-of-text token; `tinychat-quick` and
  `tinychat-better` specs; `mtstudio chat` and `serve` on cpp-httplib;
  `data.max_tokens`; `tools/model_card.py` (card with a fixed chat probe
  set); `tools/train_progress.py`; `tools/package_chat_model.sh`;
  `tools/publish_hf.py` (Hugging Face Hub); `web/chat/`, a zero-server
  browser chat engine in plain JavaScript with a parity test.
- **Extractor** (6 Oct): highway and sliding-window patterns, window and
  sink extraction (`AUX_PATTERNS`), Mistral as a named SWA ancestor.
- **`tools/synthesis/`** (16–17 Sep): architectural-synthesis generator,
  arXiv scoper with field-term backstops, hand check of the top OPEN ideas,
  a mathematics column, and the folk-claims scoping.
- **Fresh-clone path** (14 Sep): `tools/get_tinystories_data.py` builds a
  public corpus and vocabulary; `tools/hf_chat.py` chats with an exported
  model from Python.
- **transfer_s2** (11–16 Sep): pre-registered learning rate per width;
  stage 1 lr selection, stage 2 S, M and L arms and the exploratory Lx
  arm banked. The pre-registered falsifier fired (14 Sep): F1 0.627, not
  adopted; rho -0.60.
- **transfer_s1** (31 Aug–7 Sep): bridge gate passed, arms S (72), M (72)
  and L banked, licensed reading in `RESULTS_M.md` and `RESULTS_L.md`.
- **Checkpoint/resume with full optimiser state** (4 Sep, `884aa0f`),
  bit-identical on CUDA; events.jsonl trimmed to the checkpoint on resume.
- **`mtsweep --shard K/N`** (5 Sep) to split an arm across VMs.
- **Colab runner** (Sep): relays partial checkpoints across the
  60-minute session prune, verifies the resume restore before launch,
  chunked idempotent uploads, re-adopts orphaned VMs
  (`tools/colab_adopt.py`) and refreshes the session key.
- **Docs:** `ROADMAP.md` is the single list of open work; the in-flight,
  backlog and handoff pages are retired (9 Oct).

### Added
- **CUDA B2.2 (partial): masked attention softmax on-device** — fused
  (causal/block) and swa (window+sinks) forward kernels plus one shared
  backward (`ds = A .* (ds - rowdot(ds .* A)) * scale`; masked entries carry
  A == 0, so no mask bookkeeping), in place over the [T,T] scores through the
  B2.1b value-cache seam: under deferral the score/weight matrix never
  crosses the bus inside a step. Host loops retained as fallback and
  reference; `test_cuda_ops` leg 4 pins all three flavors x {off, on, defer}
  at the composed-tape tolerances.
- **CUDA B2.2 (complete): embedding + cross-entropy on-device** — embed
  gather (first activation born resident), CE forward as softmax + nll +
  on-device sum so the [R,vocab] logits never cross the bus and the host
  receives one float, CE backward (P - onehot) * g with the gradient first
  touching host at accumulate() (B2.3's choke). Leg 5 composes
  embedding -> CE and pins loss + scatter-add table grad across
  {off, on, defer}. **T4-VALIDATED 30 Aug** — 12/12 suites, 281 checks, 0
  fails (receipts docs/receipts/receipts_b22_t4_20260830.txt); the first run's leg-4
  heap-corruption crash (a deferred temporary dying stale, then step_end
  downloading into the freed pointer) is fixed by the new
  `device::discard()` primitive, confirmed both directions on T4. Next:
  B2.3 (optimizer + accumulate on-device).
- **CUDA Phase B COMPLETE — adoption gate passed (31 Aug)**: B2 beats CPU
  AVX **21x at d=256 and 30.5x at d=512** (T=512, L=4, Tesla T4) with
  identically-converging trajectories (final losses match at print
  precision). Rung C of the scale ladder runs on CUDA; a ~6-CPU-hour cell
  becomes ~12 minutes. Receipts docs/receipts/receipts_b2gate_t4_20260831.txt.
- **Fixed: deferred activations feeding a gemm uploaded stale host data.**
  `window_operand` consulted the B2.1b value cache only for slotless
  operands; every Variable's data carries a slot, so under deferral a gemm
  reading a deferred activation uploaded the untouched (zero) host buffer.
  Real models trained at exactly ln(vocab) loss while the 285-check suite
  passed — the bug needs "deferred activation -> slotted gemm with no
  intervening materialize", a shape no test had. The stale hit is now
  universal; regression leg 7 covers the shape. Found by the adoption
  benchmark, which was the first end-to-end run at realistic vocab/width.
- **CUDA B2.3 complete + T4-VALIDATED (30 Aug)** — full step residency:
  persistent device optimizer state (B2.3b), device-side gradient
  accumulation with the materialize choke moved to the step boundary
  (B2.3c, 27-site backward audit), 12/12 suites and 285 checks green with
  legs 4/5/6 identical between plain and fully-deferred configs. The gate
  caught a second dying-temporary heap corruption (gradient temps consumed
  by stale-hit axpy, then step_end writing their freed buffers); fixed by
  a self-discarding rvalue `accumulate(Matrix&&)` overload. The lifetime
  rule is now enforced three ways: ~Variable discards, backward()
  discards consumed non-leaf grads, rvalue accumulate discards temps.
  Remaining: the d=512 wall-clock adoption gate.
- **CUDA B2.3a: optimizer steps on device (write-through parity seam)** —
  `k_adamw_step`/`k_sgd_step` with host-computed bias corrections so the
  math matches nn.cpp verbatim; real SGD/AdamW try the devops path per
  param with host loops as reference; leg 6 drives 5-step trajectories.
  Persistent device state (B2.3b) and device-side accumulate (B2.3c) next.
- **Llama-family models** (`nn::Llama`): RMSNorm, RoPE, SwiGLU, GQA, tied
  embeddings; HF-native parameter names; module-level FD gradchecks.
- **mtstudio**: spec-driven run driver (JSON spec -> train -> eval -> export),
  JSONL event stream, gradient accumulation, early stopping, GGUF export with
  embedded vocabulary, `--serve` live dashboard mode.
- **Studio dashboard** (`studio/index.html`): loss/val curves, per-module
  gradient glow, SRD gate track, spec builder; renders a dropped
  `events.jsonl` or polls a live serve.
- **GGUF export** verified end-to-end: mtstudio-trained Llama generates
  coherent TinyStories text through tinyllama.cpp.
- **SRD (Surprise-Routed Density)** research tools: `srd_parity` (four-lane
  falsifier protocol), `srd_needle` (associative recall with batch, difficulty,
  and seed knobs); results ledger in docs/SPARSE_ATTENTION.md.
- **Mamba S4 through the tape**: `ops::ssm_scan` with real BPTT, FD-checked.
- **CUDA seam** (`MICROTORCH_CUDA`): `device::matmul` dispatch through the
  transformer_core kernel tree, validated on Colab T4.
- Vendored `transformer_cpp` build surface in `third_party/` so clones build
  without sibling checkouts (`tools/sync_vendor.sh` refreshes it).

### Fixed
- CI: artifact actions v3 -> v4, coverage package list, codecov v5, lcov 2.x
  strictness; whole-tree clang-format so the style gate reflects reality.

## 2026-08-29
- CUDA Phase B2.1b T4-VALIDATED: deferred downloads behind
  MICROTORCH_DEFER_DOWNLOADS=1 (value cache, materialize boundaries,
  accumulate choke point); defer-vs-write-through composed-tape diffs
  exactly 0.0. Receipts: docs/receipts/receipts_b21b_t4_20260829.txt.
- The seed lottery exhibit: atlas/SEED_LOTTERY.md + tools/seed_lottery.py
  render the banked boundary receipts as the case for multi-seed,
  pre-registered ablation (two one-seed labs contradict 48% of the time
  past 2000 steps). README gains the exhibit + the registry scoreboard.
- CITATION.cff added.
- sparse_s1_seeds pre-registered (B*(256): point or distribution; ten new
  seeds); sweep in flight.
- Rung B (sparse_s1_scale), the Highway registry pilot
  (experiments/registry_0001_highway + registry/0001_highway_networks),
  and their receipts brought under version control.

## 2026-08-28
- Repo root reorganised: results CSVs -> experiments/srd_needle_2026_07/,
  atlas docs -> atlas/, completed phase docs -> docs/history/, design and
  reference docs -> docs/; 38 files of cross-references rewritten.
- ROADMAP: uplift plan adopted (scale ladder as keystone, CUDA-B as its
  prerequisite, extractor decoupling, benchmark to D&B scale, mechanism
  freeze). README: extractor provenance framing corrected.

## [0.1.0] — Phase 1-3 foundation

- Reverse-mode autograd on a DAG tape; gradient checkpointing.
- SafeTensors loading; GPT-2 and Qwen parity verification.
- Novel mechanisms with unit + gradient tests: Kimi linear attention,
  cerebellum gating, Mamba state-space.
- GTest suite, benchmarks, Doxygen documentation workflow.

- Studio: offline, explicitly applied microbatch/accumulation VRAM planner for
  plain FP32 AdamW dense Llama/GPT2; transparent uncalibrated allowances and
  reserve, unsupported-path refusal, tied-head parameter counts checked against
  CPU constructors. No measured peak-memory or OOM guarantee.

## Parity-model export host-read boundaries (31 October 2026)

Debug guards now cover GPT2 parameter copies, family tensor copies and Flex
learned-expression expansion. The AttnRes query branch avoids copying a tensor
before its guard. Default builds and explicit materialization policy are unchanged.
Four CPU DEVCHECK/export tests pass; new device-negative cases require a separate
actual CUDA run. The broader host-loop audit remains open.
[Validation](docs/receipts/export_host_read_20261031/validation.json).

## Exact-GELU GPT2 Flex Hugging Face export (31 October 2026)

The explicitly recorded exact-erf activation now maps to HF GPT2's exact GELU,
for separate and tied Flex heads. Tanh remains the default; missing or conflicting
approximation records refuse. Actual trained, saved/reloaded CPU models pass
full-row logits and greedy checks, with independent raw replay retained.
[Validation](docs/receipts/hf_exact_gelu_20261031/validation.json).

## Configurable Kimi/SRD/recurrence depth (31 October 2026)

The shared parity model now honours positive requested depth in Studio training,
reload, exports and SRD mean-gate diagnostics. The default remains two blocks.
Actual old/new binary comparisons retain identical two-block trained tensors and
score bits; deeper models pass whole-model gradient and independent IR/Ember
prefill/decode checks. This is CPU validation of the existing architectures, not
GPU performance or model-quality evidence. The completed depth item leaves ROADMAP.
[Actual CLI and historical comparison](docs/receipts/parity_depth_20261031/cli_validation.json).

## ReLU GPT2 Flex Hugging Face export (31 October 2026)

Plain GPT2 Flex supports ReLU with tied or separate output heads, with actual
trained-model CPU reload parity and raw replay. This mapping is scoped to finite
execution: native ReLU masks NaNs and clears negative zero while HF preserves
those values. The exceptional differences are tested and documented; native
numerics and generic-expression refusals are unchanged.
[Validation](docs/receipts/hf_relu_flex_20261031/validation.json).

## 31 October 2026: standalone fetch 0.1.1 published

Published the previously prepared wheel and source archive to PyPI. Provider SHA-256 digests match the frozen release receipt. Publication evidence: `docs/receipts/fetch_release_011_publication_20261031.json`.

## 31 October 2026: conditional activation and RMSNorm fallback coherence

Switching device operations off while cached tensors remained device-fresh could
make activation and normalisation host fallbacks read stale data. Materialize the
required saved inputs, outputs and parameters before those host loops. Actual L4
checks reproduce 13 original numerical failures, preserve three coherent controls
and pass all 16 fixed forward/backward references. Independent landed-log replay
and all 58 combined CPU CTests pass. This is the bounded conditional fallback
subset, not an exhaustive host-loop audit. Evidence: `docs/receipts/host_fallback_cuda_20261031/`.

## 31 October 2026: API judging and explicit attention precision

- Validated a separately identified Anthropic API preference-judge transport, durable raw
  request/response provenance, conservative unknown-cost reservations, bound cache/ledger
  identity and a fail-stop campaign wrapper. The full campaign is running under its frozen
  cumulative cap; judging and downstream training are not complete.
- Integrated explicit C++ autograd attention precision. The actual L4 CUDA + DEVCHECK gate
  passes all 44 registered cases and independent raw replay. Existing FP32 dispatch stays
  unchanged; BF16 uses a declared straight-through surrogate with FP32 softmax/backward.
  This is not yet model adoption or a throughput/quality claim.

## 31 October 2026: ordered scan and Studio corrections

- Ordered selective-scan CUDA forward/backward passes 220 actual L4 cases and independent
  raw replay with unchanged bounds. Time remains sequential, and full-model residency,
  speed and training-quality evidence are separate gates.
- Studio VRAM estimates now count the actual untied CLI GPT2 output head; runtime constructor
  inventories and four actual CLI runs verify the correction. Tied Llama counts are unchanged.
- Zero or negative evaluation/checkpoint intervals now fail during spec parsing, avoiding
  a reproduced modulo-by-zero crash after training starts. Eleven focused CLI tests pass.

## 31 October 2026: composed Mamba1 CPU baseline

Two tiny one/two-layer Mamba1 CLI cases pass three CPU training steps, save/reload,
byte-identical FP32 export, and independent NumPy/Ember full-logit and argmax checks.
Parent replay verifies all raw artifact hashes and both inference modes. A source audit
identifies guarded host exp/scale in A() and CLI parameter-close transfers as residency
breaks. This is CPU composition evidence; full composed backward and CUDA validation,
parallel-time scan and whole-model speed remain open.
