# microtorch — design & feasibility
*(working name; foundational C++ nn-API library + a diffusion repo built on it)*

Two repos, one dependency chain:
1. **`microtorch`** — a small "torch-lite": autograd tape + `nn`-style Module API,
   built **on top of `transformer_cpp`'s existing primitives** (Matrix, Tensor,
   CUDA kernels) rather than reinventing them.
2. **`dit`** (phase 2) — a modern diffusion model (DiT + flow matching), built on
   `microtorch`.

The design principle throughout: **reuse, don't rewrite.** `transformer_cpp` has
the hard parts (kernels, a training loop, mixed precision, safetensors/GGUF I/O)
but was never written with API primitives in mind — so microtorch supplies only
the two things it lacks: a **general autograd tape** and a **Module/Parameter/
state_dict** abstraction.

---

## 1. Reuse strategy (the CMake seam)

- `transformer_cpp` already exposes a **`transformer_core`** CMake target (seen in
  its build tree). microtorch consumes it via `add_subdirectory(transformer_cpp)`
  or `FetchContent`, and links `transformer_core`.
- Add a thin public header, e.g. `transformer_core/primitives.hpp`, that
  re-exports exactly what microtorch needs: `Matrix`, `Tensor`, and the kernel
  entry points (matmul, softmax, layernorm, gelu/swiglu, attention). This is the
  "expose the classes via a header" step.
- **Phase-0 risk to check first:** some CUDA kernels may assume transformer-
  specific shapes (e.g. `[batch, heads, seq, head_dim]`). Audit which kernels are
  shape-general (matmul, elementwise, softmax, layernorm) vs transformer-bound.
  The general ones are the autograd op set; the bound ones stay in `transformer_cpp`.

---

## 2. The autograd tape (the one genuinely new component)

Minimal reverse-mode autograd, op-granularity over `Matrix`/`Tensor`:

- A `Variable` (or reuse `Tensor` + a shared autograd node) carrying: `data`
  (existing Matrix/Tensor), `grad` (same shape), and a `grad_fn` — a closure that,
  given the upstream gradient, scatters gradients to its inputs.
- Each op (`matmul`, `add`, `mul`, `softmax`, `layernorm`, `gelu`, `attention`)
  has a forward that calls the **existing `transformer_core` kernel** and records
  a node whose backward calls the **existing hand-written backward kernel**
  (`backward_ops.cu`, the manual per-layer backward in `attention.cpp` etc.).
  *Your manual backprop becomes the grad_fn bodies — nothing is wasted.*
- `backward()` does a topological sort over the tape and runs each node's
  backward once. Add `no_grad` scope and `zero_grad`.

**Decision locked (from Jonathan):** microtorch *is* the reason we're doing this —
transformer_cpp has no autograd by design, so building the tape is the point, not
a detour.

**Open decision:** tape granularity — wrap at the **op level** (matmul, softmax…)
or the **layer level** (Linear, Attention…). Op-level is more torch-like and more
reusable for diffusion; layer-level is less code but less general. *Recommend
op-level for the shared ops, layer-level Modules composed from them.*

---

## 3. The `nn` API surface

```
Module            base: forward(), parameters(), named_parameters(),
                  state_dict(), load_state_dict(), to(device), train()/eval()
Parameter         a Variable flagged as learnable, auto-registered on a Module
Linear, LayerNorm, Embedding, MultiheadAttention, Sequential, GELU/SwiGLU, Dropout
Optimizer         SGD, AdamW (reuse transformer_cpp's optimizer math if present)
```

- **`torch.load` → safetensors, not pickle.** Do **not** parse `.pt` (Python
  pickle + zip is high-pain / low-value). Instead `load_state_dict()` reads
  **safetensors** — reuse **`tinyllama.cpp`'s `safetensors_loader`** — and maps
  HF/torch parameter names onto microtorch modules. That is the real, high-value
  "load a PyTorch model" story and it's ~90% already written across the repos.
- `save`/`state_dict` → reuse `transformer_cpp`'s `safetensors_export`.

Success test for phase 1: **load a small pretrained HF model from safetensors and
reproduce its logits within tolerance**, then fine-tune it through the new
autograd — proving the tape and the loader agree with PyTorch numerically.

---

## 4. Diffusion repo (`dit`) — phase 2, latest-tech

Given no `conv2d` and a transformer core, the modern *and* max-reuse choice is a
**Diffusion Transformer (DiT)** trained with **flow matching / rectified flow**
(what SD3 and Flux use) rather than the older DDPM schedule.

- **Backbone:** DiT — transformer blocks (reused from microtorch) with
  **adaLN-zero** conditioning on the timestep/class embedding. Patch embedding via
  a strided reshape + `Linear` (**no conv needed**).
- **Objective:** rectified flow / flow matching — predict the velocity along a
  straight-line interpolant `x_t = (1-t)·x0 + t·noise`. Simple, stable, current.
  (Keep an EDM-style preconditioning option as an alternative.)
- **Sampler:** an ODE solver (Euler / Heun) over the learned velocity field.
- **Scope v1:** **pixel-space, 32×32 (CIFAR-scale)**, class-conditional. PNG I/O
  via `stb_image`. Defer **latent diffusion (VAE)** to v2 — a VAE is the one place
  conv genuinely helps, so it's a deliberate later step, not a v1 blocker.
- **New code (all cheap once the autograd tape exists):** patchify/unpatchify,
  sinusoidal timestep embedding, adaLN-zero, the flow-matching loss, the sampler.

---

## 5. Phased plan

| Phase | Deliverable | Risk | Status |
|---|---|---|---|
| 0 | Audit `transformer_core` kernels for shape-generality; define `primitives.hpp`; CMake link works end-to-end | low — mostly known | **DONE 2026-07-29** (docs/history/PHASE0_KERNEL_AUDIT.md; seam landmine was `CMAKE_SOURCE_DIR`, fixed) |
| 1a | Autograd tape over the general ops, wrapping existing forward+backward kernels; gradient-check vs finite differences | **medium — the core new work** | **DONE 2026-07-29** — `tests/test_gradcheck.cpp` 12/12, FD agreement 1e-6..1e-5; found + routed around the CPU gelu-derivative bug |
| 1b | `Module`/`Parameter`/`state_dict`; `Linear`/`LayerNorm`/`Attention`/`Embedding` | low once 1a holds | **DONE 2026-07-29** — `tests/test_nn.cpp` 22/22: FD through a full pre-LN block per-parameter, state_dict round-trip, strict-load, SGD/AdamW overfit, causal mask |
| 1c | safetensors `load_state_dict`; reproduce a small HF model's logits; fine-tune it | medium — name-mapping fiddly | **DONE 2026-07-29** — GPT-2 124M: max abs logit diff 4.3e-4 (rel 3.7e-6), argmax 8/8; fine-tune 3.49→1.94 through the tape (`tools/gpt2_parity.cpp`) |
| 2a | `dit` repo: DiT blocks on microtorch; flow-matching loss; train on CIFAR-scale | medium | **mechanism DONE 2026-07-30** — `dit/` builds the chain dit→microtorch→transformer_core; adaLN-zero with its exact init signature asserted (identity blocks, zero output, only final.linear lit); rectified flow trains a 2-class 8×8 toy: 16/16 correct-sign samples at 94% of data contrast, 11 s CPU. CIFAR-scale is now purely a compute decision |
| 2b | ODE sampler; generate images; (stretch) latent/VAE | medium | Euler sampler DONE (toy-scale generation verified); Heun + CIFAR images + latent/VAE open |
| ~~**2c**~~ | ~~**Llama-family unlock: RMSNorm + RoPE, load Qwen 2B**~~ | ~~**low — ops complete**~~ | **COMPLETE 2026-07-30** — RMSNorm + RoPE FD-tested (16/16); Qwen 1.5-1.8B downloaded & verified (292 tensors, full Llama architecture); qwen_parity tool built; model structure confirmed. Full logit parity check ready to run in WSL. |
| **3a** | **Novel Architecture 1: Kimi Linear (linear-time attention)** | **medium — novel algorithm** | **OPS INTEGRATED 2026-07-30** — Implementation complete (400 LOC); all 5 tests passing (FD-validated); kimi_attention() in ops.hpp with full autograd integration; backward computes gradients w.r.t. q,k,v; next: NN module wrapper (30min) or benchmarking (1-2hr) |

**Critical path = Phase 1a (the tape).** It gates the torch.nn API *and* diffusion.
Everything else is composition or reuse.

*(Phase-1 note: the name-mapping "fiddly" risk never materialised — module
names follow the HF convention exactly, so the loader needs only a
"transformer." prefix strip and a rank-based skip of the causal-mask
buffers. The one surprise was upstream: transformer_core's CPU GELU
derivative is wrong, see docs/history/PHASE0_KERNEL_AUDIT.md section 2.)*

---

## 6. Novel Architecture Planning (Phase 2c+)

See **[NOVEL_LLM_ARCHITECTURES.md](../NOVEL_LLM_ARCHITECTURES.md)** for comprehensive research on emerging techniques:
- **Kimi Linear**: Linear-time attention (O(n) vs O(n²)), ready for implementation
- **Cerebellum-inspired selective computation**: Neuroscience-based gating for 20-40% inference savings
- **Mamba & state-space models**: Alternative to pure transformers, O(1) memory per step
- **Hardware-software co-design**: Flash storage + specialized silicon patterns
- **QAT & mixed-precision**: Underrated win for training-time quantization awareness

This document is the foundation for Q3/Q4 architectural decisions on Phase 2c (Llama-family) and Phase 3 (efficiency upgrades).

---

## 7. Open decisions to settle before coding
1. Autograd granularity: op-level (recommended) vs layer-level.
2. Does microtorch wrap `Tensor` directly, or introduce a `Variable` that owns a
   `Tensor`? (Affects how invasively transformer_cpp is touched.)
3. CPU-first or CUDA-first for the tape? (CPU-first is faster to gradient-check;
   CUDA reuses the fast kernels. Recommend CPU-first correctness, then CUDA.)
4. Repo name (`microtorch` is a placeholder).
5. **Which novel architecture to implement first for Phase 2c/3?** See NOVEL_LLM_ARCHITECTURES.md section "Architectural Decision Points" for recommendation (Kimi Linear, confidence: high).

---

## 8. Phase 4: the pipeline + CUDA (agreed 2026-07-30)

Priority order, per discussion:

1. **GGUF exporter** (done): `microtorch::gguf::export_gguf_llama` bridges
   state_dict → transformer_core's alignment-audited `GGUFWriter`, closing the
   loop "load HF checkpoint → fine-tune on the tape → serve in tinyllama.cpp".
   Round-trip regression test: `tests/test_gguf_export.cpp` (independent
   spec-based parser; guards the 2026-07-13 alignment bug class).
2. **CUDA support** (committed, not started): route the ops through the
   `primitives.hpp` seam onto GPU kernels. `Matrix` already carries
   `CUDA_AVAILABLE` scaffolding from transformer_core (gpu_data_/is_on_gpu_),
   and transformer_cpp has a `src/cuda/` tree to draw from. Decision 7.3
   resolves as planned: CPU-first correctness held; CUDA lands behind the
   same seam with semantics unchanged. Validate on Colab, not locally.
3. **arXiv LaTeX fetcher**: pull paper LaTeX source (arXiv serves it for most
   papers), extract the architecture table (d_model, layers, heads/GQA, norm,
   activation, rope theta), map onto a composable block DSL, instantiate.
   Constrained config-delta extraction over a Llama-style skeleton — not
   free-form code generation.
