# CUDA Phase B2 — training-step residency (design)

*Status: B2.0 VALIDATED on Colab T4, 13 Aug 2026 — full
gradcheck/nn/lora/resident-parity suites plus test_step_residency all
green: the CUDA training pin matched the CPU reference to max weight
diff 2.33e-07 over 8 AdamW steps (bound: 1e-4), and the staleness
probe confirmed the epoch-scope contract on hardware (a host poke
between windows is seen by the device). B1 was validated 12 Aug — see
CUDA_PHASE_B.md. B2.1-B2.3 below remain open; B2 is the actual Rung C
unlock: the training step's bytes stay on the device; the host sees
loss scalars and explicit materializations (checkpoints, eval samples)
only.*

## Why B1 cannot be stretched into B2

Three structural reasons, found in the seam, not assumed:

1. **Params mutate every step.** A B1 table refresh after each Adam
   step is a full H2D upload of every parameter — the traffic B2
   exists to remove.
2. **Backward creates untrackable operands.** The matmul grad closures
   call `Matrix::transpose()` on the host
   (`ops.cpp`: `device::matmul(self->grad, b->data.transpose())`).
   Each call constructs a *fresh host Matrix*, so a pointer-keyed
   table can never see it twice. No cache policy fixes this; the GEMM
   itself must accept transpose flags.
3. **Host-pointer keying is unsafe for activations.** Params live for
   the process; activations die with the tape, and freed host memory
   is recycled *within a step*. A pointer+dims table hit on a recycled
   address is silently wrong. B1's keying is correct for B1 precisely
   because residency there is explicit and parameter-only.

## Design

### 1. Device state is owned by the Variable, not a side table

`Variable` (our tape, `autograd.hpp` — the vendored Matrix stays
untouched) gains one opaque member:

```cpp
DevState* dev = nullptr;   // POD forward-declared; null in CPU builds
```

`DevState` = `{float* data; float* grad; bool data_dev_valid,
data_host_valid, grad_dev_valid, grad_host_valid;}`. The Variable
destructor releases the buffers through a function pointer registered
at CUDA init (no CUDA includes in the header; CPU builds carry a null
pointer and zero overhead). Ownership by the Variable kills the
aliasing class by construction: the buffer dies exactly when the
tensor dies, and no recycled host address can inherit stale state.

### 2. Validity flags maintained only at mutation sites

The contract that keeps the no-silent-staleness rule: during a
training step, the ONLY mutators of Var data/grad are tape ops,
`accumulate()`, `zero_grad()`, and the optimizer — each updates the
flags as it writes. Everything outside the step (checkpoint save,
studio event emission, sampling) calls `device::materialize(v)` /
`materialize_grad(v)` first. Debug builds (`MICROTORCH_DEVCHECK`)
assert `host_valid` on host reads, so a missed materialize is a loud
assert, not a wrong number.

Master switch: `device::set_step_residency(bool)`, env
`MICROTORCH_STEP_RESIDENCY=1`. Off = today's B1/Phase A behaviour,
bit-for-bit.

### 3. GEMM with transpose flags

`gemm(A, opA, B, opB, C)` — NN/NT/TN as index-math variants of the
audited 32x32 tiled kernel (no materialized transposes on either
side of the seam). This deletes the host `.transpose()` temporaries
from matmul backward and both attention nodes' backwards.

### 4. The on-device op set (self-contained, canonical formulas)

Same zero-dependency rule as B1: our own kernels in `src/cuda_*.cu`,
no cuBLAS, one canonical entry point per op (the Phase 0 audit's
duplicate-kernel landmine stays behind the seam). Needed by the
parity/flex training step:

- elementwise: add, sub, mul, scale, axpy (grad accumulate), fill
- sigmoid fwd/bwd (highway gates), GELU fwd/bwd — using the CORRECT
  derivative (the vendored CPU formula is the audited ~13%-error bug;
  our CPU path and the vendored CUDA kernel agree, and B2 matches
  them)
- softmax rowwise fwd/bwd (`dS = S .* (dP − rowsum(dP .* S))`)
- layernorm fwd/bwd, rmsnorm fwd/bwd
- embedding gather fwd / scatter-add bwd
- cross-entropy fwd (downloads ONE scalar) + fused grad
- AdamW update (m, v resident; bias correction + decoupled decay,
  matching `nn.cpp` exactly), SGD+momentum likewise

Attention stays COMPOSED (matmul + softmax + elementwise), per the
Phase 0 verdict — the fused vendored kernels are shape-bound and stay
behind. The per-head row slices in the fused tape nodes are offset
pointers into contiguous row-major buffers, so composition works on
device without copies; that audit is part of B2.1, stated here so it
is not discovered as a surprise.

### 5. What the host sees per step

Loss scalar (one float D2H). Nothing else. `zero_grad` = device
memset. Eval/sampling/checkpoint paths call materialize explicitly —
eval already runs under B1 semantics today and keeps doing so.

### Memory budget (stated, not hoped)

Rung C shape d=512, T=512, L=4: params ~13M floats (~50 MB), peak
activations + grads + Adam state comfortably under 1 GB. T4 = 16 GB.
Not a constraint until far above this ladder.

## Staging (each stage lands green or does not land)

- **B2.0** Variable DevState plumbing + gemm transpose variants,
  **write-through and epoch-scoped**: results always land in host
  storage (host data is never stale in B2.0 — no validity flags are
  live yet), and cached device operands are trusted only inside a
  `step_begin()`/`step_end()` window. Host mutations (optimizer, the
  gradcheck suite's FD pokes, checkpoint load) all fall between
  windows, so the staleness class is killed by SCOPE rather than by
  discipline — code that never opens a window gets today's behaviour
  unchanged. Result caching is deliberately deferred: in the real
  models every matmul→matmul chain passes through a host-side op
  (GELU, softmax, norm) until B2.1 moves those, so B2.0's win is
  operand dedup (params upload once per window instead of once per
  use) plus the in-kernel transposes. The full §2 flag contract with
  DEVCHECK asserts activates in B2.1 when downloads are deferred.
- **B2.1a — T4-VALIDATED 21 Aug 2026** (receipts:
  docs/receipts_b21a_t4_20260821.txt; Tesla T4, CC 7.5). Kernel parity:
  elementwise bitwise 0.00e+00, activations <= 2.4e-07, rowwise worst
  3.8e-06 (layernorm dgamma, column-sum order); composed-tape OFF vs ON:
  loss bitwise 0.00e+00, all leaf grads <= 2.3e-10. gradcheck (27 ok)
  and nn (22 ok, on the T4) reran green WITH the op set live — same FD
  oracle. The run also caught and fixed a latent repo defect: three
  CMake-referenced test files were never committed (paperkiln cbbed8b)
  and coalfire's remote CMakeLists referenced the pre-31-Jul
  train_wikitext (coalfire 9c0b023) — fresh clones of either repo had
  been failing at CMake generate for weeks while local builds passed.** the op set
  itself: elementwise add/sub/mul/scale/axpy/fill, sigmoid fwd/bwd, GELU
  fwd/bwd (the CORRECT derivative), softmax rowwise fwd/bwd, layernorm
  fwd/bwd, rmsnorm fwd/bwd — src/cuda_ops.cu, one canonical entry per
  op, gated by MICROTORCH_DEVICE_OPS=1 + device==CUDA, write-through
  (host never stale, B2.0's contract). Tape call sites in ops.cpp fall
  through to their own loops, so CPU numerics are untouched. BOTH
  attention nodes' matmuls now route through the transpose-flag gemm
  (the host .transpose() temporaries in their forwards and backwards
  are deleted — the design's section 3 debt, paid). New gate:
  tests/test_cuda_ops.cpp (kernel parity vs the CPU formulas at 1e-6
  elementwise / 1e-5 rowwise, plus a composed-tape OFF-vs-ON leg), and
  colab_cuda_validate.sh now reruns gradcheck/nn with the op set live.
- **B2.1b** deferred downloads: DevState grows grad buffers + the
  section-2 validity flags, downloads happen only at materialize()
  boundaries, DEVCHECK asserts on host reads. Slice-pointer audit for
  composed attention lands here (offset pointers into contiguous
  row-major buffers).

  **B2.1b working checklist (opened 28 Aug 2026; core landed
  29 Aug 2026 — design note below):**
  1. [x] Validity state — DESIGN AMENDED in implementation: instead of
     growing `DevState` with per-Variable `d_grad`+flags, B2.1b adds a
     VALUE CACHE (`g_vcache` in cuda_resident.cu): device-fresh op
     outputs AND grads keyed by host data pointer, epoch-stamped, with
     a `stale` flag = the validity contract. Param `DevState` slots
     stay as B2.0 built them (params never go host-stale in B2.1b —
     the optimizer host-mutates between windows). One mechanism covers
     activations, intermediates, and grads uniformly.
  2. [x] Deferral behind `MICROTORCH_DEFER_DOWNLOADS=1` (env read in
     device.cpp set_from_env; setter `set_defer_downloads`). Active
     only inside a step window (killed-by-scope, as B2.0). gemm and
     all 15 devops entry points route outputs through
     `detail::vc_output` (deferred: buffer retained, no D2H) and
     inputs through `detail::vc_operand` / `window_operand`
     (stale-value hit → no H2D re-upload). In-place axpy preloads the
     buffer unless the cache already holds a fresh copy. Aux per-row
     stat vectors stay write-through by design (small; host-consumed).
  3. [x] `materialize(m)` / `materialize_all()` are the download
     boundaries; `step_end()` materializes all when defer is on (the
     optimizer/eval/checkpoint still read host between windows), so
     cross-window staleness cannot exist. `host_stale(m)` exposed.
  4. [~] `DEVCHECK`: `devcheck_host_read()` + `MT_DEVCHECK_HOST_READ`
     macro land (compiled in under `-DMICROTORCH_DEVCHECK`); call-site
     wiring through ops.cpp host-read points is the NEXT increment.
  5. [ ] Slice-pointer audit for composed attention (offset pointers
     into contiguous row-major buffers vs the B2.1a transpose-flag
     gemm paths).
  6. [ ] Gate: extend tests/test_cuda_ops.cpp with a
     deferred-downloads leg (OFF vs ON: loss + leaf grads within
     B2.1a bounds). Local check DONE for the core: both .cu TUs
     compile clean under nvcc 12.6 + MSVC (29 Aug), CPU stubs
     g++ -fsyntax-only clean.
  7. [ ] T4 validation via colab_cuda_validate.sh per the contract
     below; receipts into docs/, phase doc + ROADMAP updated.
  Constraint for this pass: do not touch tools/mtstudio.cpp or
  tools/parity_model.hpp (uncommitted WIP present, 28 Aug).
- **B2.2** embedding + CE: the forward touches host only at the loss
  scalar.
- **B2.3** AdamW/SGD on device; optimizer state uploads once at
  construction; checkpoint = explicit materialize sweep. Full step
  resident.

## Validation contract

Local: compile-only (nvcc), as always — no local GPU execution.
On T4, per stage:
1. Full gradcheck/nn/lora suites under `MICROTORCH_DEVICE=cuda` with
   `MICROTORCH_STEP_RESIDENCY=1` — the same FD oracle that gates every
   other path.
2. `test_step_residency`: N=50 training steps, CPU vs B2, identical
   seeds/data. NOT bitwise — device reduction order differs; the pin
   is per-step loss within rtol 1e-4 early, plus final-weights max
   abs diff bound. Bitwise claims are reserved for same-backend pins
   (the SWA/highway precedent).
3. DEVCHECK build runs the studio smoke: any host read of a
   device-valid tensor asserts.

## Adoption criterion (the Rung C gate, stated before any numbers)

Benchmark = wall-clock per training step, d=256 and d=512 at T=512,
CPU AVX vs B2, on T4. B2 is adopted for Rung C iff it wins at d=512.
If AVX still wins there, that is a *finding* — Rung C runs on CPU,
the result goes in this file, and the GPU path waits for bigger dims
without shame.

## Non-goals (B2)

Streams/overlap, multi-GPU, fp16/mixed precision, fused attention
kernels, inference-path changes. Each is a separate decision after B2
measures.
