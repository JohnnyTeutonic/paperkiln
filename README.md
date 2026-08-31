# paperkiln

*Formerly `microtorch` — renamed 2026-08-01; GitHub redirects the old
URL. Its sibling engines renamed with it: **coalfire.cpp** (formerly
transformer/transformer_cpp, the original C++ trainer) and **ember.cpp**
(formerly tinyllama.cpp, the inference half). The C++ namespace and
header paths keep `microtorch` for now — code stability over branding.*

**Drop an arXiv link on a page. Train the paper's actual architecture —
not the nearest preset — watch its gradients glow layer by layer, export
a GGUF, and chat with it. One browser tab, one C++ stack you can read in
an afternoon, every claim carrying its receipt.**

**[▶ Watch the 5-minute walkthrough](docs/media/demo_walkthrough.mp4)** —
the whole loop, live and unedited: arXiv fetch on camera, evidence view,
training, export, chat.

```
Drop an arXiv link on the studio page
       ↓
Architecture extracted        papers/fetch.py — every value carries its
       ↓                      evidence snippet from the paper's LaTeX;
Review + edit hyperparameters used-vs-mentioned disambiguation is scored
       ↓                      and BENCHMARKED; contested calls ask a human
The paper's model, as written flex family: depth, d_ff, norm, activation,
       ↓                      position — all constructor-real
▶ TRAIN (a button, in-page)   loss curve + per-module gradient glow, live
       ↓
Export GGUF / safetensors     byte-exact, vocabulary embedded, download
       ↓                      links in the page
Chat with it — same page      ember.cpp, a separately written engine,
       ↓                      serves the file you just watched train
Share the run                 one spec file + one JSONL event stream
```

![the one-tab studio loop: live arXiv fetch → architecture diagram → ▶ train → loss curve → artifact downloads → chat panel](docs/media/studio_loop.gif)

*Cut from the real recording, no mockups — the live arXiv fetch, the
spec builder and architecture diagram, ▶ TRAIN, the loss curve filling
in, and the finish line: exported artifacts as download links with the
chat panel waiting below. The
[full 5-minute walkthrough](docs/media/demo_walkthrough.mp4) plays on
click.*

An educational yet research-capable LLM framework that exposes every major
component of the modern transformer training stack in readable C++ — while
remaining compatible with Hugging Face checkpoints, and with a measured
habit of publishing its negatives next to its wins.

**The core engine — tape, ops, layers, Llama family, quant, GGUF — is under
4,000 lines. The entire stack (studio, run driver, arXiv fetcher, 16 CI-gated
test suites) stays readable end to end. No CUDA required (T4-validated when
you want it). Builds in under two minutes.**

## Finding your way around

**[`docs/README.md`](docs/README.md) is the map** — every document in
this repo has a place in it. If you are picking the project up (or
resuming it), start there. The short version:

| you want | go to |
|---|---|
| what's running / what's next | [`docs/open/`](docs/open/) |
| why something was built this way | [`docs/decisions/`](docs/decisions/) |
| how a subsystem works | [`docs/`](docs/) |
| what the research has established | [`atlas/FINDINGS.md`](atlas/FINDINGS.md) |
| a specific experiment | [`experiments/`](experiments/) |

## What makes this different

Minimal autograd engines are a well-populated genre. Six things here are not:

1. **Papers in, running models out — with provenance.** `papers/fetch.py` pulls a
   paper's actual LaTeX source, extracts the architecture, and attaches an
   **evidence snippet to every extracted value**. Fields it cannot resolve are
   reported as unresolved rather than silently guessed. This is constrained
   config-delta extraction with citations, not free-form generation.
2. **The paper's architecture is what actually trains.** Extracted depth, d_ff,
   norm flavor, activation, and position encoding are **constructor-real** via
   the flex model family — *Attention Is All You Need* trains as
   512/6/8/2048 with LayerNorm, ReLU and sinusoidal positions; Primer's 110M
   decoder reconstructs at 114M. The generalization is pinned, not trusted:
   flex at its defaults reproduces the reference block **bit-for-bit**
   (`tests/test_flex.cpp`).
3. **The extraction is measured, not vibed.** A paper *mentions* many
   alternatives; it *uses* one. The contribution-vs-mention scorer separates
   them with explainable cues and is benchmarked on **40 real papers** with
   ground-truth architectures: grouped AUROC 0.905 [bootstrap 95% CI
   0.789–1.000] vs 0.841 for naive first-match, pooled 0.817 [0.735–0.898]
   vs 0.778, and **three documented wrong assertions in 92 verdicts** —
   explicit rejections ("we choose not to adopt SwiGLU") veto a candidate
   outright, and close calls abstain and ask the human. The zero-wrong
   record held for 29 papers and broke three times over the next eleven,
   in **two distinct ways**. Twice it was evidential — *what a paper uses
   is often stated indirectly, and indirect evidence loses to any
   direct-looking mention*: Megatron-LM states its activation only by
   attributing it to the models it copies, and Cerebras-GPT's true
   positional encoding is inherited from "a GPT-3-like architecture"
   while RoPE appears purely as *future work*. Once it was lexical:
   LaMDA's "gated-GELU" is GeGLU, and the scorer matched the substring
   "GELU". All three are registered with diagnoses and named fixes
   (`KNOWN_WRONG` — new failures still fail the build), because a
   benchmark that never fails is not measuring anything, and growing this
   one from 26 to 40 papers is what found them. The benchmark, its CIs and its growth protocol ship in
   the repo.
4. **Falsifiers ship inside the modules.** Novel mechanisms carry the experiment
   designed to kill them — `SurpriseRoutedAttention::shuffle_predictor` feeds the
   router a permuted input so the gate keeps its distribution but loses its
   information. When a result fails, [the negative gets
   published](docs/SPARSE_ATTENTION.md), not buried.
5. **The loop actually closes, across two engines.** A model trained on this tape
   exports to byte-exact GGUF and produces coherent English inside
   `ember.cpp` — a *separately written* inference engine. Both halves of the
   pipeline are in this stack, and the studio's chat panel talks to it from the
   same page that trained the model.
6. **It runs designed experiments on itself.** The
   [Architecture Atlas](atlas/ARCHITECTURE_ATLAS.md) treats architecture comparison as
   a science: Plackett–Burman screens, token-matched factorials, multi-seed
   cells, findings published with effect sizes and standard errors — including
   the finding that its own best-cell ranking was inside seed noise while the
   designed contrasts ran 6–10σ. The registry
   ([atlas/FINDINGS.md](atlas/FINDINGS.md)) currently holds **19 claims —
   including 3 published retractions and 3 supersessions** — every row with
   its receipts, reproducible via `python tools/reproduce.py <id>`. Labs
   that never retract anything aren't more careful; they're less honest
   about resolution.

## pip install paperkiln-fetch

The extractor now ships standalone — no C++ build, no repo checkout:

```bash
pip install paperkiln-fetch          # (PyPI upload pending; until then:
                                     #  pip install ./paperkiln_fetch)
paperfetch 1706.03762                # evidence-carrying summary
paperfetch 2302.13971 --emit-hf cfg.json
```

`--emit-hf` writes a config that `transformers.AutoConfig.from_pretrained`
opens directly — llama- or gpt2-family, chosen from the extracted flavors
— with every value's evidence snippet and every declared default riding
inside it under the `_paperkiln` key. A Hugging Face config with
citations built in. Package source: [paperkiln_fetch/](paperkiln_fetch/)
(vendors `papers/fetch.py` verbatim; drift is CI-checkable via
`python tools/sync_fetch_pkg.py --check`).

## The crossing theorem — and the zone where no experiment can answer

The registry's own results explain themselves. Sliding-window attention
is a *nested* subclass of exact attention — this repo pins that
**bitwise** (`tests/test_swa.cpp`: window ≥ T with sinks=0 reproduces
full causal attention exactly). Write the gap to each class's best
achievable loss and the comparison decomposes to

    Delta(b) = A - D(b),   A = L*_swa - L*_exact >= 0  (by nesting)

with `D` the difference of unclosed optimization gaps. Under nesting,
decay, and non-increasing `D`, **Delta is monotone, crosses at most
once (always sparse→dense, never back), and no basin is admissible** —
which is exactly the shape the 15-seed data shows. The corollary is the
part with teeth:

    zone width  ~=  2 t SD / ( sqrt(n) |dDelta/db| )

There is, for every finite seed budget, an interval of training budgets
in which the comparison **has no answer** — and it closes only as
√n, so halving it costs four times the seeds. At d=256 that zone opens
around 1600 steps and has not closed by 3600. The seed lottery below is
this corollary in anecdote form; [the theorem](atlas/THEOREM_CROSSING.md)
is it in closed form, with its own falsifier pre-registered
(`experiments/sparse_s1_longbudget/`): a *second* crossing at long
budget would prove the monotonicity assumption fails, and it should
appear if and only if the larger class overfits first.

## The seed lottery

Run five copies of an *identical* experiment — same code, same corpus, same
architecture, same budget — varying only the random seed, and ask: does
sliding-window attention beat exact attention? From the banked receipts of a
pre-registered experiment
([the full exhibit](atlas/SEED_LOTTERY.md), regenerated by
`python tools/seed_lottery.py`):

| budget (steps) | verdicts: exact / swa | P(two one-seed labs contradict) |
|---:|:---:|:---:|
| 400 | 0 / 5 | 0% |
| 1200 | 0 / 5 | 0% |
| 1600 | 1 / 4 | 32% |
| 2000–3600 | 3 / 2 | **48%** |

At 400 steps every seed agrees — on the conclusion that *reverses* by 1200.
From 2,000 steps on, **two labs publishing off one seed each contradict each
other on the sign of the effect almost every other pairing**. Single-seed,
single-budget ablation tables — the field's default evidence — are lottery
tickets, and this repo's methodology (pre-registered decision rules committed
to git before data exists, paired seeds, budget scopes on every claim, a
registry that records retractions instead of deleting them) is what it costs
to stop playing.

## Why it exists

PyTorch is 4M+ lines; understanding *why* your gradient is wrong means reading
dispatcher internals. paperkiln takes the opposite bet: every operation is a
readable forward + hand-derived backward pair, every backward is verified against
finite differences, and the whole tape fits in
[one header](include/microtorch/autograd.hpp).

That makes it three things at once:

1. **A working training stack** — load a HF GPT-2 or Qwen checkpoint, fine-tune
   it, save it back as safetensors.
2. **A research vehicle** — novel mechanisms (linear attention, selective
   computation, state-space models) land here in days, not framework-release
   cycles.
3. **A reference implementation** — if you want to know what RoPE or a
   cross-entropy backward *actually does*, the answer is one file away, not forty.

## Quick start

```bash
git clone git@github.com:JohnnyTeutonic/paperkiln.git
cd paperkiln
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure     # gradchecks + unit + integration tests
```

## The studio — spec in, chatting model out

![the terminal loop: paper → provenance-carrying extraction → train → Atlas row → chat](docs/media/demo.gif)

*The same loop driven from the terminal (`tools/demo.sh`): fetch,
evidence, train on the FD-checked tape, the run's Atlas row, and a chat
with the exported GGUF through ember.cpp.*

One JSON file describes the whole lifecycle; `mtstudio` executes it.

```bash
./mtstudio plan specs/tinystories-llama.json   # dry-run: resolve and print
./mtstudio run  specs/tinystories-llama.json   # train, eval, export, serve
./mtstudio serve /tmp/mtstudio_llama3k 8080    # live dashboard on a running job
```

That spec trains a Llama-family model (RMSNorm + RoPE + SwiGLU, tied embeddings)
on TinyStories with early stopping, checkpoint/resume and a JSONL event stream,
then exports safetensors **and** a GGUF with the vocabulary embedded. Feeding
that GGUF straight into ember.cpp (the serve command it prints):

> **prompt:** once upon a time
> **model (300 training steps, d=128):** she was a little big he had very time
> tree the girl and day a little big time they were so play to the man and said
>
> **same architecture, 3,000 steps (val loss 4.64 → 3.68):** there was a little
> girl named timmy he had a big hug and said goodbye to the park with her mom
> and she started to play outside in the sky but it was too late that they were
> playing together all day

Three hundred steps buys the word distribution; three thousand buys syntax —
clause chains, narrative connectives, attempted coreference (pronoun drift and
all). Both runs are CPU-only.

### The dashboard — `studio/index.html`

A single self-contained HTML file, no build step and no dependencies. Drop an
`events.jsonl` on it, or point it at a live run with `mtstudio serve` and it
polls every two seconds while training proceeds.

- **Loss chart** — train curve, validation points, and (for SRD runs) the mean
  gate on its own 0–1 axis as a dashed overlay.
- **Gradient glow** — per-module gradient L2 norms, taken pre-clip, drawn as
  bars. Colour is not absolute magnitude: it is `log10(norm / that module's own
  running median)`, so a module is red when *it* is hot relative to its own
  history and navy when it is fading. Vanishing and exploding gradients become
  visible per-layer while the run is still going.
- **Run stats** — step, train loss, best val, grad norm, mean gate, early-stop
  trigger.
- **Spec builder** — a form that writes the very same `spec.json` that
  `mtstudio run` consumes, previews the JSON live, downloads it, and prints the
  exact run and serve commands. Preset dropdown covers `llama-tiny`,
  `gpt2-nano`, `gpt2-small`, `kimi-tiny`, `srd-tiny`, `attnres-tiny`, plus
  editable d/layers/heads overrides (`arch.custom`).
- **Architecture diagram** — a labeled block diagram (pure SVG, still zero
  dependencies) of exactly what the current spec builds: tokens → embed →
  [norm → attention → ⊕ → norm → FFN → ⊕] × L → head, with family-aware
  labels, per-head dims, residual arcs (replaced by an attention-over-depth
  annotation for `attnres`), and a rough parameter estimate. Redraws on every
  keystroke.
- **From paper** (serve mode) — drag an arxiv.org link onto the page (or type
  the id): the server forks `papers/fetch.py` and **everything extracted
  becomes constructor-real** — d, layers, heads, d_ff, *and the flavors*
  (LayerNorm vs RMSNorm, GELU/ReLU/SwiGLU, learned/sinusoidal/RoPE
  positions) land in the editable builder via the flex family, so the model
  you train is the paper's architecture, not the nearest preset (verified:
  Attention Is All You Need trains as 512/6/8/2048 layernorm+relu+sinusoidal;
  Primer's 110M decoder reconstructs at 114M). Contested extractions and
  fields the fetcher can't build yet are reported, never silently applied.
  The diagram redraws, the evidence view is one click away, ▶ train runs in
  the same tab, and the exported `.safetensors`/`.gguf` end as download
  links.
- **Chat** — talks to `tinyllama_server`'s `/chat` API: the GGUF this page
  just exported, served by a separately written engine, answering in the
  same tab. The whole paper → config → train → export → **chat** loop is one
  page.

### Custom configuration — the spec format

Every field below is optional and falls back to a sane default. Use
`arch.preset` for a known configuration, or `arch.custom` to dimension a model
yourself and pick the attention mechanism.

```jsonc
{
  "name": "tinystories-llama-3k",
  "arch": {
    "preset": "llama-tiny",           // or omit and use "custom" below
    "custom": {                       // overrides the preset field-by-field
      "d": 256, "layers": 4, "heads": 8,
      "attention": "srd"              // exact | kimi | srd
    }
  },
  "data": {
    "corpus":    "data/train.txt",    // raw text
    "vocab":     "releases/chat7b.gguf",  // vocabulary lifted from a GGUF
    "vocab_cap": 4096,                // truncate to the top-N tokens
    "T":         256                  // context length
  },
  "train": {
    "steps": 3000, "lr": 3e-3, "clip": 1.0,
    "batch": 4,                       // sequences stacked into ONE forward
                                      // (positions + attention mask restart
                                      // per sequence; parity-tested)
    "accum": 2,                       // gradient accumulation over batches
    "checkpoint_activations": false,  // rematerialize block interiors on
                                      // backward (llama family; bit-identical
                                      // grads, ~95% fewer live tape nodes)
    "eval_every": 100,
    "checkpoint_every": 100,          // resume picks up from here
    "gradmap_every": 10,              // per-module grad-norm event cadence
    "early_stopping": { "patience": 8, "min_delta": 0.003 }
  },
  "export": { "formats": ["safetensors", "gguf"] },
  "serve":  { "on_finish": true },    // print the tinyllama serve command
  "out_dir": "/tmp/mtstudio_llama3k"
}
```

The event stream (`out_dir/events.jsonl`) is the contract between trainer and
UI — `start`, `step` (loss, grad_norm, optional per-module `grads` and SRD
`gate`), `eval`, `early_stop`, `export`, `done`. Anything that can read JSONL
can consume a run.

## Feature matrix

| Component | Status | Notes |
|---|---|---|
| Reverse-mode autograd | ✅ | DAG tape, topological sort, `NoGrad` scope |
| Gradient verification | ✅ | Every op finite-difference checked in CI |
| Layers | ✅ | Linear, LayerNorm, RMSNorm, Embedding, Attention, MLP, Dropout |
| Full GPT-2 | ✅ | Logit-parity verified against HF checkpoint |
| Llama-family ops | ✅ | RMSNorm + RoPE + SwiGLU; Qwen 1.5-1.8B load verified |
| Optimizers | ✅ | SGD (momentum), AdamW, **Muon** (K3 per-head Newton-Schulz, golden-pinned to the reference; `train.optimizer: "muon"` = deployment-faithful hybrid) |
| LR schedulers | ✅ | Cosine-with-warmup, StepLR |
| Grad clipping | ✅ | Global-norm (`clip_grad_norm`) |
| Mini-batching | ✅ | Stacked `[B*T, d]` batches, block-isolated attention; stacked-vs-per-seq logits/loss/grads equal to fp epsilon (`test_batching`), 1.34x measured at B=4/T=128 |
| Gradient checkpointing | ✅ | `checkpoint(fn, x)` rematerializes block interiors on backward: loss+grads **bit-identical**, live tape nodes after forward 211→11 (GPT-2) / 229→9 (Llama) (`test_checkpointing`) |
| Fused attention | ✅ | `fused_attention`: GEMMs + one in-place scale/mask/softmax pass, one tape node per head, mask never materialized; 12 FD receipts; 1.87x wall-clock (2.19x with batching) |
| Checkpoint IO | ✅ | safetensors load **and** save (HF round-trip) |
| **HF export** | ✅ | `tools/hf_export.py`: llama-family runs open in `transformers.from_pretrained` — weights re-laid out (incl. the interleaved→rotate-half RoPE permutation), config + WordLevel tokenizer emitted; **argmax parity verified** vs the tape (`hf_export_verify.py`) |
| GGUF export | ✅ | `export_gguf_llama`: state_dict → .gguf for ember.cpp |
| Cross-entropy loss | ✅ | Fused softmax backward |
| Python bindings | ✅ | pybind11, numpy interop (`-DMICROTORCH_BUILD_PYTHON=ON`) |
| **Run studio** | ✅ | Declarative spec → train/eval/export/serve; `plan` dry-run; resume |
| **Live dashboard** | ✅ | Loss + val + gate chart, per-module gradient glow, spec builder, SVG architecture diagram, in-page chat |
| **Flex family (paper-faithful)** | ✅ | Any depth; d_ff, LayerNorm/RMSNorm, GELU/ReLU/SwiGLU, learned/sinusoidal all spec-real; bitwise equivalence pin at defaults (`test_flex`, 6 receipts) |
| **From-paper flow** | ✅ | Drag an arXiv link → scored extraction (grouped AUROC 1.000, 0 wrong assertions on the 10-paper bench) → editable spec → ▶ train → artifact downloads → chat |
| **Atlas experiment engine** | ✅ | `mtsweep` (grid/PB12/fold-over PB12f, linked factors, aliasing advisories, resumable, OMP-aware) + `atlas_analyze` (main effects + two-way interactions, seed-based SEs); Stages 2–3 findings published |
| **Atlas viewer** | ✅ | `studio/atlas.html` (+ `/atlas` in serve mode): in-page effects, clickable interaction heatmap, seed spreads — client math pinned to the Python analyzer on real Stage 3 rows |
| LoRA | ✅ | `LoRALinear`: frozen base + rank-r adapters, `merged_weight()` |
| Quantization | ✅ | int8 blockwise (absmax/block), `QLinear` ~3.7x smaller weights |
| QLoRA | ✅ | `QLoRALinear`: quantized frozen base + trainable adapters |
| CUDA dispatch seam | ✅ | `device::matmul` → `cuda::matmul`; suites **pass on a T4** |
| **Kimi linear attention** | ✅ | O(n·d²) vs O(n²·d); drop-in `KimiLinearAttention` |
| **Cerebellum selective gating** | ✅ | Prediction-residual gating; skips compute on routine tokens |
| **Mamba / S4 state-space** | ✅ | Trainable through time: `ssm_scan` tape op, BPTT FD-gradchecked |
| **Surprise-routed density (SRD)** | 🧪 | Falsifier passed 5-6σ twice; gate concentrates on retrieval sites (5x replicated). Recall claim **failed replication and the negative is published** — [docs/SPARSE_ATTENTION.md](docs/SPARSE_ATTENTION.md) |
| GPU kernels | 🧪 | matmul dispatches to transformer_core CUDA; phase B = resident tensors |

## Train something (C++)

```cpp
#include "microtorch/nn.hpp"
using namespace microtorch;

nn::GPT2Config cfg;                 // 124M defaults
nn::GPT2 model(cfg, /*seed=*/42);

nn::AdamW opt(model.parameters(), /*lr=*/3e-4f);
nn::CosineWarmupLR<nn::AdamW> sched(opt, /*warmup=*/100, /*total=*/10000);

for (auto [ids, targets] : batches) {
    Var logits = model.forward(ids);
    Var loss = ops::cross_entropy(logits, targets);
    opt.zero_grad();
    backward(loss);
    ops::clip_grad_norm(model.parameters(), 1.0f);
    opt.step();
    sched.step();
}
save_safetensors("ckpt.safetensors", model.state_dict());
```

### Python

```bash
cmake .. -DMICROTORCH_BUILD_PYTHON=ON && make _microtorch
```

```python
import numpy as np, _microtorch as mt

x = mt.tensor(np.random.randn(8, 256).astype(np.float32), requires_grad=True)
attn = mt.nn.KimiLinearAttention(d=256, n_heads=4)
loss = mt.ops.mean(attn(x))
mt.backward(loss)
print(x.grad.shape)        # (8, 256)
```

### Load a HuggingFace checkpoint

```cpp
auto sd = load_safetensors("gpt2/model.safetensors");
nn::GPT2 model(cfg);
model.load_state_dict(sd, /*strict=*/true);   // fails loudly on any mismatch
```

## The novel mechanisms

### Kimi linear attention — [kimi_linear.hpp](include/microtorch/kimi_linear.hpp)

Standard attention pays O(n²·d) to build the full attention matrix. The
linear-attention family replaces softmax with a feature map φ(x) = elu(x)+1 and
reassociates the product, paying O(n·d²) instead:

```
out_i = φ(q_i)ᵀ · Σ_{j≤i} φ(k_j) v_jᵀ  /  φ(q_i)ᵀ · Σ_{j≤i} φ(k_j)
```

The cumulative sums preserve causality without a mask. `nn::KimiLinearAttention`
is interface-identical to `nn::CausalSelfAttention` — swap one line to try it.
Measured on CPU: ~8.9x faster forward+backward at seq=16, converging to ~1.1x as
n approaches d (the crossover regime the complexity analysis predicts).

### Cerebellum-inspired selective gating — [cerebellum.hpp](include/microtorch/cerebellum.hpp)

Motivated by cerebellar prediction-error filtering: a small predictor
(d → d/4 → d) learns what "routine" hidden states look like; the gate
`σ(‖x − predict(x)‖)` mixes the expensive layer's output with the identity path
per token. Surprising tokens get full compute, routine tokens skip it. Wraps any
layer via `std::function` — no changes to the wrapped code.

### Mamba / S4 state-space — [mamba.hpp](include/microtorch/mamba.hpp)

Discrete state-space recurrence `x[t+1] = A·x[t] + B·u[t]` as a drop-in sequence
backbone: O(1) memory per generated token instead of a growing KV cache.
Trainable through time — `ssm_scan` is a tape op with BPTT verified against
finite differences on all five inputs. The hardware-parallel scan is the next
milestone (see [docs/DESIGN.md](docs/DESIGN.md)).

### Surprise-routed density — [srd.hpp](include/microtorch/srd.hpp)

Research, and currently a partial negative. Per-query density routed by
prediction residual rather than attention-affinity scores: one shared qkv
projection feeds both an exact and a linear path, and the gate
`σ(scale·rms(x − predict(x)) + bias)` blends them per query. Training is soft and
fully differentiable; inference hardens the gate to `g > τ`, giving
O(ρn²d + nd²) for surprise rate ρ. `mean(gate())` is exposed so density can be
priced directly in the loss.

The falsifier (`shuffle_predictor`) permutes the predictor's view of the input,
preserving the gate's distribution while destroying its alignment. It passed at
5-6σ twice, and the gate demonstrably concentrates on retrieval-critical
positions across five runs. **The recall-performance claim did not replicate
across seeds and that failure is written up in full**, including what survived
and what the next pre-registered test is.

## The Architecture Atlas — a cumulative science of neural architectures

Leaderboards rank; they don't explain. The
[Atlas](atlas/ARCHITECTURE_ATLAS.md) treats architecture comparison the way
industrial statistics treats process optimization: designed experiments,
multi-seed cells, effect sizes with standard errors, and published
findings — negative ones included. Every mtstudio run already emits its
Atlas row (structural echo + behavioural metrics); `tools/mtsweep.py`
materializes whole designs (grid, Plackett–Burman, linked token-matched
factors) as spec files, and `tools/atlas_analyze.py` turns the rows into
main-effects tables.

**Stage 2 is complete** — a 7-factor Plackett–Burman screen, 36 runs in
one night on a laptop CPU ([full writeup](atlas/ATLAS_STAGE2_RESULTS.md), raw
rows in [experiments/atlas_stage2/](experiments/atlas_stage2/)):

| Finding | Evidence |
|---|---|
| **Muon is the strongest factor in the screen** — better final loss *and* a measured throughput price | loss-AUC t = −10.8, best_val −0.34 (t = −6.2), −148 tok/s (t = −2.4) |
| **Learning rate decouples speed from quality** — faster early descent, 3× the gradient spikes, zero final-loss gain | half-gap t = −8.8; spikes t = +3.5; best_val n.s. |
| **Head count doesn't matter at this scale** — a genuine null, recorded | \|t\| ≤ 0.5 on every metric |
| **Best-cell ranking was inside seed noise while the designed contrasts ran 6–10σ** — the case for designed experiments over leaderboard-style cell-hunting, demonstrated in our own data | top-2 cell gap 0.021 < seed noise 0.027 |

**Every claim is a row in the [findings registry](atlas/FINDINGS.md)**
(`atlas/findings.jsonl`): effect, SE, t, design, scope, status
(`supported/replicated/superseded/retracted/pending`) and receipt paths —
retractions are rows, never deletions, because a registry that visibly
corrects itself is the trustworthy kind. Three verbs operate on it:

```bash
python tools/atlas_findings.py render          # the registry as a table
python tools/atlas_findings.py advise "what lr with muon at tiny scale?"
python tools/reproduce.py S3-lrxopt            # plan + honest cost estimate
python tools/reproduce.py S3-lrxopt --run      # re-run the experiment, verdict:
                                               # REPLICATED / DID-NOT-REPLICATE
```

The **advisor answers from the registry with citations — or refuses**
("NO EVIDENCE … absence of evidence is the honest answer"), and surfaces
corrected records alongside live ones. **`reproduce` makes replication a
one-command verb**: cost quoted up front from the original run's own
wall-clock receipts, fresh out_root, machine-checked verdict against the
registered effect. All six checkable findings verify against their own
committed rows.

**Stage 3 is complete**: a full 2⁴ factorial on the survivors
({optimizer, context, lr, d} × 3 seeds, 48 runs) with **token-matched
context** — T=128×1200 steps vs T=256×600 steps, both exactly 614,400
tokens, de-aliasing "longer context" from "more data". Interactions are
the point: the full factorial leaves all six two-way effects
unconfounded.

## Verification philosophy

Nothing merges without:

1. **Finite-difference gradcheck** — every op's backward vs central differences,
   with the measured error printed, not merely asserted.
2. **Parity tests** — GPT-2 logits match the HF reference end-to-end; Qwen
   tensors load and map onto modules by name with zero translation tables.
3. **Round-trip tests** — save → load → bit-identical.
4. **Falsifiers for research claims** — every novel mechanism ships with the
   experiment that would kill it, and the result is published either way.

CI runs the full suite (plus cppcheck, clang-format, Valgrind) on every push —
see [.github/workflows](.github/workflows). API documentation is generated by
Doxygen (`doxygen docs/Doxyfile` → docs/html/) and published to GitHub Pages by
the docs workflow.

## Paper-to-architecture: the arXiv fetcher

```bash
pip install requests
python papers/fetch.py 1706.03762                     # Attention Is All You Need
python papers/fetch.py 2302.13971 --emit-cpp llama.cpp --json llama.json
```

`papers/fetch.py` downloads a paper's **LaTeX source** from arXiv, sweeps prose
and model-size tables for the architecture hyperparameters, and emits a
normalized config — every extracted value carries an **evidence snippet** from
the paper, and anything it cannot resolve is listed as unresolved rather than
guessed. `--emit-cpp` generates compilable microtorch code (GPT-2-family
`GPT2Config` or, when it detects RMSNorm/RoPE, a Llama-family
`LlamaExportConfig`).

Flavor fields (norm/activation/position) go through **contribution-vs-mention
scoring**: a paper *mentions* many alternatives (baselines, related work,
"such as" lists) but *uses* one, and first-match extraction confuses the two.
Every candidate occurrence is scored by explainable cues — usage verbs,
"replace X with Y" direction, config-table rows, section class (related-work
penalized), ablation/comparative phrasing, possessive attribution ("the
Transformer's ReLU"), plus-compound baseline names ("Transformer+GELU") — and
a value is asserted only when a clear winner exists; close calls are
**contested** (both candidates shown with evidence, nothing auto-applied).
Papers that never *state* their flavors — they inherit an architecture by
citation ("the same model and architecture as GPT-2") — are handled by
**inheritance resolution**: a small curated ancestor table fills what the
prose leaves open, marked `inherited` with the inheritance sentence as
evidence, and the paper's own statements always override the base. The
guard that matters: the generic `transformer` ancestor never fills
fields, because "based on the Transformer" is said by every decoder LM
while the deltas go unstated — BERT is "based on the Transformer" and
silently switches to GELU and learned positions.

Measured on `papers/flavor_bench.py`, **29 real papers** with ground-truth
architectures (Vaswani through OLMo, including designed negatives whose
true flavor is outside the lattice): **grouped AUROC 0.895** [bootstrap 95%
CI 0.761–1.000, resampling papers] vs 0.825 for naive first-match, pooled
0.819 post-veto, **53/71 field verdicts correct with zero wrong
assertions** — where first-match extraction claimed RoPE for the ALiBi
paper and SwiGLU for Primer *and* Falcon ("we choose not to adopt SwiGLU"
now vetoes the candidate outright). The first 10-paper cut scored a
grouped 1.000 — the larger sample deflated that honestly, which is exactly
what the benchmark is for; the growth protocol continues
(STUDIO_PLAN §13.1).

Validated live against: *Attention Is All You Need* (d_model=512, N=6, h=8,
d_ff=2048, sinusoidal), *LLaMA* (4096/32/32 from its model-size table +
RMSNorm/SwiGLU/RoPE), *TinyLlama* (22 layers, 32 heads, vocab 32000), plus
the ten-paper flavor benchmark above. Offline fixture tests:
`python papers/test_fetch.py` — no network needed in CI.

This is the constrained config-delta approach: most transformer papers are
deltas over a known skeleton, so the search space is tractable — and the
contribution is the part nobody else ships: **provenance**. Every extracted
field carries the evidence that produced it (the table cell, the sentence,
the equation it was read from), every unresolved field is surfaced loudly
instead of silently defaulted, and a wrong assertion is treated as a bug,
not a rounding error — the flavor benchmark's standing record is zero wrong
assertions, enforced by an abstention-first scorer. Free-form code
generation produces plausible architectures; evidence-linked extraction
produces *auditable* ones. That difference is the tool.

## The training → inference pipeline

paperkiln is the training half of a two-engine pipeline:

```
HF checkpoint ──load_safetensors──► paperkiln (fine-tune on the tape)
                                        │
                              export_gguf_llama(path, state_dict, cfg)
                                        │
                                        ▼
                              .gguf ──► ember.cpp (inference engine)
```

```cpp
#include "microtorch/gguf.hpp"

gguf::LlamaExportConfig cfg;
cfg.embedding_length = 2048; cfg.block_count = 24;
cfg.feed_forward_length = 5504; cfg.head_count = 16;
cfg.vocab_size = 151936; cfg.tokens = vocab;
gguf::export_gguf_llama("model.gguf", model.state_dict(), cfg);
```

The GGUF writer is the alignment-audited implementation from `transformer_core`
(the tensor-data section starts at `align_up(header_end, 32)` — the exact detail
that once turned a working model into word salad, now regression-tested in
`test_gguf_export`).

## Fine-tuning on a budget: LoRA / QLoRA

```cpp
#include "microtorch/quant.hpp"

// Frozen base + rank-8 adapters; only A and B train.
nn::LoRALinear lora(W_pretrained, /*rank=*/8, /*alpha=*/16.0f);
nn::AdamW opt(lora.parameters(), 1e-3f);      // 2 tensors, not the base
// ... train ...
Matrix W_final = lora.merged_weight();        // fold in for zero-cost inference

// QLoRA: the base lives as int8 blocks (~3.7x smaller), adapters in fp32.
nn::QLoRALinear qlora(W_pretrained, bias, /*rank=*/8, /*alpha=*/16.0f);
```

Properties enforced by `test_lora_quant`: adapter output is *exactly* the base at
init (B=0), gradients touch only A/B, `merged_weight()` matches the adapter
forward to 1e-4, int8 round-trip error is bounded by half a quantization step.

## CUDA

The dispatch seam is live: every matmul routes through `device::matmul`
([device.hpp](include/microtorch/device.hpp)). Default builds are CPU-only and
bit-identical to before. `-DMICROTORCH_CUDA=ON` compiles transformer_core's
kernel tree and `device::set(Device::CUDA)` (or `MICROTORCH_DEVICE=cuda`)
dispatches to `cuda::matmul`. Validation runs the same gradcheck suite on GPU —
[tools/colab_cuda_validate.sh](tools/colab_cuda_validate.sh). Phase B (resident
device memory instead of per-call round trips) is the next CUDA milestone.

## Repository layout

```
include/microtorch/   public headers (one concern per header)
  autograd.hpp        the tape: Variable, backward(), NoGrad
  ops.hpp             op set — every op forward + audited backward
  nn.hpp              Module, layers, optimizers, schedulers
  llama.hpp           Llama-family: RMSNorm + RoPE + SwiGLU, HF-native names
  kimi_linear.hpp     phase 3a: linear attention
  cerebellum.hpp      phase 3b: selective gating
  mamba.hpp           phase 3c: state-space models
  srd.hpp             research: surprise-routed density + its falsifier
  gguf.hpp            GGUF export (alignment-audited)
  quant.hpp           int8 blockwise, LoRA, QLoRA
  safetensors.hpp     HF checkpoint load/save
src/                  implementations
tools/                mtstudio (the run driver), parity checkers, benchmarks
studio/               the dashboard — one self-contained HTML file
specs/                example run specs
papers/               arXiv → architecture fetcher + offline fixture tests
tests/                gradchecks + unit tests (all in CI)
python/               pybind11 bindings
docs/                 Doxygen config (make docs)
```

## Roadmap

- arXiv fetcher v2: per-variant instantiation, GQA/MoE fields, HF-config
  cross-check
- CUDA phase B: resident device tensors (params uploaded once, activations
  on-device)
- int4/NF4 quantization (QLoRA paper's datatype; int8 is the current base)
- Parallel scan for Mamba (training-speed parity with attention)
- Technique transfer from open-weight frontier reports — attention residuals,
  KDA, Muon optimizer ([docs/TECH_TRANSFER.md](docs/TECH_TRANSFER.md))
- **The Architecture Atlas** ([atlas/ARCHITECTURE_ATLAS.md](atlas/ARCHITECTURE_ATLAS.md)):
  Stages 0–2 are **done** (structural echo in every run, taxonomy + constrained
  grammar, the PB12 screen with published findings) and Stage 3 (token-matched
  2⁴ factorial, interactions) is running; ahead lie the scale ladder,
  fingerprints/neighbours, and the Atlas surface — architectural fingerprints
  in the Studio
- **Sparse attention research phase**: survey the current literature and attempt
  original variants — the long-horizon flagship goal

## License

MIT — see [LICENSE](LICENSE).
