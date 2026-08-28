# microtorch Studio — Master Plan

*2026-07-30. The competitive thesis, the build order, and tonight's first moves.*

## 1. Thesis

Feature-for-feature saturation against PyTorch/HF is a losing race. The winning
race is the one ComfyUI proved (~80K stars wrapping an engine that already
existed): make the **pipeline** visible, composable, and clickable. Every UI
player today (Ollama, LM Studio, ComfyUI) wraps **inference only**. Nobody has
built the ComfyUI of **training**.

microtorch Studio: compose an architecture → pick a corpus → **Train** → watch
the loss live → **Chat with the model you just made**. One box. We already own
every stage: autograd trainer, LoRA/QLoRA fine-tuning, int8 quantization
(Q4/Q8 enums already in the GGUF writer), byte-exact safetensors/GGUF export,
and tinyllama.cpp's HTTP server (server.cpp + www/) for the inference link.

Two hooks no incumbent can copy quickly:
1. **Research mechanisms as dropdown architectures** — Kimi linear, Mamba
   (BPTT-trainable as of 2026-07-30), SRD sparse attention. "Train with
   attention variants that exist nowhere else."
2. **Paste an arXiv ID into the architecture box** — papers/fetch.py populates
   the config from the paper's LaTeX. A launch-demo moment nobody has.
   **STATUS 2026-08-01: LIVE in the served studio.** Drop an arxiv.org link
   on the page (or type the id): `POST /fetch` forks the fetcher server-side,
   the extracted values land in the editable spec builder, the node graph
   shows the architecture preview, and the diff-to-paper evidence view is one
   click away — then ▶ train, all in one tab. Smoked end-to-end on
   1706.03762 (512/6/8 extracted over the wire).

Positioning line: **the glass-box ML studio** — every block in the UI links to
its readable source and the gradcheck that verifies it. Smallness becomes the
feature.

## 2. Inventory (what exists, with receipts)

| Stage | Status | Receipt |
|---|---|---|
| Train (pretrain) | ✅ | GPT-2 class, optimizers, schedulers, clip, dropout; suites 9/9 |
| Fine-tune | ✅ | LoRALinear/QLoRALinear, adapter-only checkpoints; test_lora_quant |
| Quantize | ✅ int8 (int4/NF4 roadmap) | quantize_int8, ~3.7x, half-step error bound |
| Export | ✅ | safetensors + GGUF; chat7b round-trip byte-identical |
| Serve | ✅ (via tinyllama.cpp) | generation verified from our re-export |
| HF import | ✅ | GPT-2 logit parity; Qwen load verified |
| Paper→config | ✅ | papers/fetch.py, 3 papers validated |
| Config precedent | ✅ | transformer_cpp config/*.json + config/architectures/ presets (llama/vanilla/custom) |

## 3. Gaps found in tonight's audit

1. **Early stopping is config-only in transformer_cpp.** `early_stopping_patience/
   threshold` exist in config.hpp:93-94 and transformer_config.json, but only
   hyperparameter_tuner.cpp reads them — the main training loop never honors
   patience. Fix: patience tracking around the validation-loss checkpoint in the
   trainer loop; microtorch's M1 driver gets early stopping natively from day one.
2. **The microtorch-TRAINED → serve leg is untested.** gguf_roundtrip proved
   trained-transformer_cpp weights survive our exporter byte-exactly and
   generate in tinyllama.cpp. Nobody has yet trained a model IN microtorch and
   chatted with it. This is M1's acceptance test, not a side item.
3. Studio-relevant misc: vocab/tokenizer story for from-scratch training (reuse
   the word-level GGUF vocab path as default; BPE later).

## 4. M1 — the run-spec CLI (the studio's engine; UI comes later)

One JSON file describes the whole lifecycle; one driver executes it.

```jsonc
// spec.json (draft schema v0)
{
  "name": "my-tinystories-gpt",
  "arch": {                        // exactly one of:
    "preset": "gpt2-small",        //  gpt2-{nano,small}, llama-tiny, kimi-tiny,
                                   //  mamba-tiny, srd-tiny ...
    "arxiv": "2302.13971",         //  or: populate from a paper
    "custom": {"d": 256, "layers": 4, "heads": 8, "vocab": 8192,
                "attention": "exact|kimi|srd", "norm": "layernorm|rmsnorm"}
  },
  "data": {"corpus": "tinystories|wikitext|/path/to.txt",
            "vocab": "chat7b|/path/to.gguf", "vocab_cap": 8192, "T": 256},
  "train": {"steps": 3000, "lr": 3e-4, "schedule": "cosine", "warmup": 100,
             "clip": 1.0, "early_stopping": {"patience": 5, "eval_every": 200},
             "checkpoint_every": 250},
  "finetune": {"base": "path/to/ckpt.safetensors", "lora_rank": 8,
                "quant_base_bits": 8},          // optional
  "export": {"formats": ["gguf", "safetensors"], "quant_bits": 32},
  "serve": {"on_finish": true, "engine": "tinyllama", "port": 8080}
}
```

Driver: `mtstudio run spec.json` (new tools/mtstudio.cpp) —
- validates the spec, prints the resolved plan, then executes stage by stage;
- training emits a JSONL event stream (`step`, `loss`, `val_loss`, `lr`,
  `gate_mean` for SRD) to stdout AND a file — this stream IS the UI's data feed
  later, so it's designed now;
- checkpoint/resume built in (the srd_parity pattern, already proven through
  VM reclaims);
- early stopping honored natively;
- on finish: export per spec, then (serve.on_finish) launch tinyllama.cpp on
  the exported GGUF and print the chat URL.

Acceptance test (closes gap 2): spec with gpt2-nano on TinyStories →
`mtstudio run` → generation from tinyllama.cpp is coherent. That transcript
goes in the README.

## 5. M2 — the UI veneer

A single-page local web app (no build system if avoidable; one HTML file the
driver can serve). It has NO logic of its own: dropdowns write a spec, the run
panel streams the JSONL events, Chat links to the tinyllama URL.

Pages/panels:
- **Build**: architecture picker (presets incl. research mechanisms | custom
  dims | arXiv ID box), corpus dropdown, hyperparams with sane defaults.
  Absorb transformer_cpp's config/architectures/*.json as importable presets.
- **Train**: live loss curve (canvas + the JSONL stream), gate histogram when
  SRD is selected, stop/early-stop status, checkpoint list.
- **Finetune**: base-checkpoint picker, LoRA rank, dataset upload (labelled
  text), quant toggle.
- **Export/Serve**: format checkboxes, quant bits, "Open chat" button.
- Every block links to source + its gradcheck (glass-box).

## 6. M3 — distribution

- `pip install microtorch` (pybind11 wheel; cibuildwheel in CI).
- README GIFs of the full loop; 90-second demo video (paste arXiv ID → train →
  chat is the money shot).
- Recipe sharing: specs are single files; a specs/ gallery in-repo.
- Launch surface is Jonathan's call (no conference constraint applies to
  software releases; HN Show-HN is the norm for this genre).

## 7. Sequencing

Tonight / tomorrow (light work day):
1. M1 skeleton: spec parser + resolved-plan printer + train stage for presets
   (gpt2-nano first), JSONL event stream, early stopping.
2. Acceptance test end-to-end (gap 2) with a short run.
3. transformer_cpp early-stopping fix (gap 1) — small, self-contained.

Next (more time arriving after the offer lands):
4. Fine-tune + quant + export stages in the driver; arXiv + custom arch paths.
5. M2 single-file UI over the event stream.
6. M3 wheels + demo assets.

Parallel research track (unchanged, docs/SPARSE_ATTENTION.md): needle amendment
rerun (T=128, longer budget), multi-seed, then the paper-spine conversation.

## 9. Performance roadmap (triaged external review, 2026-07-30)

An external spit-ball (Gemini) was triaged against the actual codebase.
Items marked ALREADY-TRUE validate existing design; the rest are ordered
by leverage. Deliberately rejected framings are recorded too.

**Already true (no action, worth advertising):**
- LoRA graph detachment: LoRALinear registers ONLY A/B as parameters; the
  frozen base is a no-grad data node, off the tape entirely, and forward is
  the parallel track h = Wx + B(Ax). Exactly the recommended design.
- mmap-able unified serialization: GGUF already is this; tinyllama.cpp
  mmaps it. AdamW: have it.

**Real gaps, high leverage (ordered):**
1. Mini-batching. DONE 2026-07-31: B sequences stack as [B*T, d] rows;
   `ops::attention_mask` gives a block-diagonal causal mask, positions
   restart per sequence (GPT2, Llama, ParityLM exact lanes; kimi/srd not
   yet block-aware). Receipts: tests/test_batching.cpp — stacked vs
   per-sequence logits/loss/grads equal to fp epsilon, isolation exactly
   0.0; 1.34x wall-clock on identical work at batch=4/T=128 (llama-tiny).
   Known cost: the stacked score matrix does B× redundant attention work;
   per-block row slicing is the follow-up if attention ever dominates.
2. Gradient checkpointing. DONE 2026-07-31: `microtorch::checkpoint(fn, x)`
   runs the segment under NoGrad on forward and rematerializes it on
   backward through a detached input copy; parameters captured by the
   segment accumulate grads in place. Wired as `checkpoint_blocks` on
   GPT2/Llama and `train.checkpoint_activations` in mtstudio (llama).
   Receipts (tests/test_checkpointing.cpp): loss and every param grad
   BIT-IDENTICAL with/without, alone and composed with batching; live
   tape nodes between forward and backward 211->11 (gpt2) / 229->9
   (llama) via the new live_variables() diagnostic — which is also the
   probe the Fit-to-VRAM budgeter needs. Constraint documented in
   autograd.hpp: segments must be recomputation-deterministic (Dropout's
   per-forward seed counter is excluded).
3. Fused operators. DONE 2026-07-31 for attention: ops::fused_attention
   keeps the two GEMMs on device::matmul and collapses scale+mask+softmax
   into one in-place pass — the mask matrix is never materialized, one
   [T,T] buffer (the weights, kept for backward) survives the forward,
   one tape node per head instead of ~5. Swapped into CausalSelfAttention
   and LlamaBlock. Receipts: 12 FD gradchecks (causal, causal+batch,
   bidir+batch on q/k/v + forward parity vs the composed path); measured
   68.0s -> 36.4s at batch=1 and 50.6s -> 31.0s at batch=4 on the
   identical-work benchmark (llama-tiny, T=128) — 2.19x total against the
   pre-batching baseline. Elementwise fusion (gelu/silu chains) remains
   open under gap 4.
4. Safe in-place elementwise ops. DONE 2026-08-01: relu_/sigmoid_/scale_
   mutate the producer's buffer and interpose on its backward_fn
   (grad *= f'(output), then the original chain) — chain tape nodes
   3->1, bit-identical grads, FD ~1e-6. Eligibility is a two-clause
   contract documented in ops.cpp: no other consumer of the input, and
   the producer's backward must not read its own output — which is why
   gelu/silu (input-dependent backwards) are excluded as mathematics,
   not as TODO.
5. Mixed precision: fp16/bf16 tape mode. DEFERRED 2026-08-01, on
   purpose: microtorch's CPU kernels have no fp16 SIMD path, so a
   half-precision tape on CPU would *cost* time (convert both ways
   around every fp32 op) and only save memory that checkpointing
   already reclaims 20x of. The honest value of fp16/bf16 arrives with
   the CUDA phase (transformer_cpp's half_precision_kernels.cu already
   exists to port — see PHASE0_KERNEL_AUDIT). Revisit as part of that
   phase, not before.

**Studio features unlocked by the above:**
- Fit-to-VRAM budgeter: profile the parsed architecture, auto-place
  checkpointing boundaries to fit a chosen memory cap (needs gap 2).
- Low-rank prototyping slider: load a checkpoint, strip projection ranks,
  preview memory/throughput vs quality trade-off (SVD or power-iteration
  needed; pairs with the existing LoRA/QLoRA stack).

## 10. M2 amendments (visual mechanics that earn the README GIF)

- **Live node-graph with gradient glow**: the JSONL event stream gains
  per-layer gradient-norm events (cheap: clip_grad_norm already computes
  the global norm; emit per-module norms too). UI renders the architecture
  as a node graph; nodes glow with grad magnitude, fade on vanishing,
  flash on explosion. This is the 5-second-GIF feature.
- **Diff-to-paper split view**: papers/fetch.py already emits an evidence
  snippet per extracted field; render paper text left, live graph right,
  with extraction highlights. The data for this exists TODAY.
- Launch framing (M3): not "another NN framework" but "the compiler that
  turns arXiv papers into trainable models and lightweight C++ inference."

## 11. Research lane R2 (survey-gated): backprop-free training

Status: PARKED behind a survey, per the sparse-attention discipline.
The literature is real (DFA: Nokland 2016, Launay 2020 scaling study;
Hinton's Forward-Forward 2022; GrAPE, OpenReview 2025; Split Forward
Gradients, arXiv 2607.16612; a dedicated LLM-focused survey exists), but
the honest state of the art is that these methods DEGRADE on transformers
at scale, and FF is classification-shaped. Treat exactly like sparse
attention: (1) arXiv survey with a bibliography and taxonomy, (2) pick the
strongest transformer-compatible variant (likely DFA-family), (3)
falsifier-first prototype -- our tape makes DFA nearly trivial (fixed
random feedback matrices, layer-local updates; no backward graph), and the
srd_parity harness pattern gives the 4-lane comparison for free.
Kill criterion up front: if the variant cannot match backprop within X%
on the TinyStories parity task, it is a negative result, documented, done.

## 12. transformer_cpp: the tunable-surface directive (2026-07-31)

Jonathan's directive: transformer_cpp should expose as many viable
hyperparameters and tunable components as a modern transformer needs.
State after the 07-31 cleanup:

- **Dead code removed**: main.cpp/main.hpp (the trainer that consumed the
  legacy config), the cuda_kernels ghost pair. `train` (ex train_wikitext)
  is THE central trainer; scripts updated.
- **Live tunables today** (CLI + --config JSON overlay): dims/heads/layers,
  seq len, epochs/steps, batch size, lr + warmup + cosine decay, family
  presets (llama|vanilla) incl. RMSNorm/RoPE/bias-freezing, fp16 (CUDA),
  LoRA (rank/alpha), quantization mode, doc-aligned + assistant-only-loss,
  export gguf/safetensors/HF, early stopping (patience/min-delta, wired
  07-30), **MoE (experts/top_k/aux coefficient, wired 07-31)**.
- **Receipts earned 2026-07-31** (wire-smoke-loss-falls, tiny/30-60 steps):
  MoE (4 experts top-2: loss 9.66->8.60; lags dense at toy scale as
  expected, ~3x step cost); Adam betas + weight decay (A/B: knob changes
  diverge trajectories, 7.87 vs 6.87 step-30); GQA (2 KV heads trains at
  parity with dense, 7.82 vs 7.87); early stopping (07-30).
- **Sliding window: receipt EARNED via a real bug find (2026-07-31).**
  The deterministic probe (swa_check.cpp) showed the window mask lived
  only in the single-sequence forward; forward_batched (the training
  path) was causal-only -- every SWA training run to date silently ran
  full attention, which is why the loss A/B was inconclusive. Fixed
  (window joins the pre-softmax mask, half-window semantics matching the
  single-seq path; CUDA batched path fails loudly until its kernel gains
  the term). Post-fix probe: far-outside 0.0, near-past 26.0,
  future-in-window 0.0, control 0.31. swa_check is a permanent gate.
- **Label smoothing: receipt earned 2026-07-31** (--label-smoothing;
  closed-form smoothed CE on the CPU path, loud CUDA guard; A/B: eps=0.2
  tracks ~0.10 above eps=0, the smoothed-CE floor signature).
- **Gradient accumulation: RECLASSIFIED, not wired.** The trainer's
  modules apply Adam during backward (fused design), so true accumulation
  is a cross-module refactor; batch_size already provides the
  effective-batch lever, and mtstudio carries true accumulation for the
  microtorch stack. Revisit only if a concrete need appears.
- **Wiring queue remaining**: per-component lr; dropout schedule.
- **Post-training track** (per 07-31 directive): MoE fine-tuning
  (dense-to-MoE upcycling), distillation (teacher logits from a big GGUF
  via tinyllama.cpp), RLHF-lite (the repo carries untested ppo_gae.cpp /
  distributed_rlhf infrastructure -- audit before promising anything),
  post-training quantization sweeps via the existing quant stack.
  Not enterprise -- but each item lands with the same discipline:
  wire, smoke, loss-falls-or-it-is-not-done.

## 13. Answering the parser question (asked externally)

papers/fetch.py is a DETERMINISTIC rule-based scraper: regex families
over detexed LaTeX plus a model-size table parser, every extraction
carrying a verbatim evidence snippet, unresolved fields reported rather
than guessed. No LLM in the loop -- by design: reproducible, offline,
CI-testable (papers/test_fetch.py runs without network). An optional
LLM-assist pass for gnarly papers is a possible v2 layer on top; the
deterministic core stays the default and the fallback.

### 13.1 Flavor-benchmark growth (sample size, noted 2026-08-01)

The contribution-vs-mention scorer's numbers (grouped AUROC 1.000,
pooled 0.783, 0 wrong assertions — papers/flavor_bench.py) rest on
**10 papers / 31 candidate-label pairs / 7 grouped comparisons**. That
is enough to kill the observed failure modes but NOT enough to call the
estimate stable — at n=31 the AUROC's own sampling error is wide, and a
perfect grouped score over 7 groups is exactly what a small sample can
gift. Jonathan's requirement: grow the sample until the law of large
numbers is doing the work, then write the result up properly.

**STATUS 2026-08-03: failure analysis + three refinements.** Where the
scorer breaks, decomposed on the 26-paper corpus:
1. **Recall, not ranking, is the dominant loss.** BERT, RoBERTa, ALBERT,
   GPT-3, OPT, Galactica produce ZERO candidates for their truth fields:
   they inherit an architecture by citation and never state norm /
   activation / position in extractable prose. Fixed by **inheritance
   resolution** (base + delta, `find_inheritance` + `ANCESTORS`):
   verdict `inherited`, the inheritance sentence as evidence, and the
   paper's own findings always override the base. Verdicts 40 → 47 of
   65 correct, still **0 wrong**, 6 inherited fields all correct.
   Guards earned by three real false positives found in tracing:
   method/abstract sections only (Falcon's "classifiers based on BERT
   models … so we" is a data-pipeline aside), self-reference must
   PRECEDE the claim, "setup" is not an architecture noun (Primer's "in
   a setup similar to GPT-3 XL"), and — the important one — the generic
   `transformer` ancestor NEVER fills fields, because "based on the
   Transformer" is said by every decoder LM while the deltas go
   unstated. **BERT is the counterexample that proves the rule**: it is
   "based on the Transformer" and silently switches to GELU + learned
   positions, so inheriting vanilla would assert two wrong values.
2. **The ablation-mention class is real but smaller** (Falcon, Primer):
   handled by the veto. Since the deployed system can never assert a
   vetoed candidate, the honest ranking metric is **post-veto pooled
   AUROC = 0.799** (vs 0.788 with vetoed candidates still in).
3. **`activation` is the weakest field (0.739)** because of GLU naming
   soup. When two members of one family contest each other the
   disagreement is about NAMING, not architecture, so the scorer now
   asserts the family (`gated-glu`) rather than flipping a coin.

**STATUS 2026-08-01 (same day): grown to 26 papers / 69 pairs / 17
groups, bootstrap CIs implemented (resampling papers).** The prediction
above came true on schedule: the perfect 1.000 deflated to grouped
**0.882 [95% CI 0.750–1.000]** — still above naive (0.863), pooled
0.788 [0.694–0.908] vs 0.754, and 0 wrong assertions held only after
four new cue classes earned it (sentence-scoped windows, negative
ablation results, passive replacement, declined-adoption VETO — the
Falcon "we choose not to adopt SwiGLU" case). Next milestone: 40+
papers, then the CI-band gate.

Protocol for the write-up:
- Grow TRUTH to **30–50 papers (~120+ pairs)**: encoder-only and
  seq2seq families (BERT, RoBERTa, T5 variants, DeBERTa's relative
  positions), more designed negatives (papers whose true flavor is
  outside the lattice, like Primer), and papers with adversarial
  related-work density. Every fetched-and-hand-verified paper becomes
  one TRUTH row — the table is built to accumulate.
- Report **bootstrap confidence intervals** (resample papers, not
  pairs — papers are the independent unit) for grouped and pooled
  AUROC, alongside verdict accuracy and the abstention rate.
- Keep the two hard gates (no wrong assertions; grouped >= naive) and
  add a CI-band gate once n supports it.
- Where it lands: a short methods section in whatever paper/README
  section presents the from-paper pipeline — the claim "we measured
  the disambiguator" is only as strong as its n.

## 8. Open decisions (deliberately deferred)

- BPE tokenizer training in-box vs word-level default (start word-level).
- UI framework: leaning single-file HTML+JS served by the driver; revisit only
  if the canvas plotting gets painful.
- Whether mtstudio absorbs transformer_cpp's trainer as a backend for big runs
  or stays pure-microtorch (start pure; the config-preset import keeps the
  bridge open).
