# CUDA Phase B — resident device memory

*Status: B1 VALIDATED on Colab T4, 12 Aug 2026: full gradcheck/nn/
lora suites under MICROTORCH_DEVICE=cuda plus test_resident_parity,
all green, max abs deviation ~1.4e-06 vs the CPU reference (log:
session cudaval). Phase A was validated previously — see CHANGELOG.*

## Why Phase A is not enough

Phase A ships every operand host->device and every result device->host
on every matmul. At tiny dims the PCIe round trip dwarfs the kernel;
GPU "acceleration" can lose to the AVX path. The win Rung C needs is
residency: bytes that stay on the device across calls.

## Staged plan, stated honestly

- **B1 (this commit) — explicit parameter residency.** An opt-in table
  maps host tensors to device copies. `device::matmul` uses device
  copies for any resident operand; non-resident operands are temp-
  uploaded; results still download. Pays where weights are frozen:
  generation, eval passes (every `eval_every` during training), the
  registry/studio demo path. It does NOT accelerate the training step:
  params mutate every step, and activations still round-trip.
- **B2 — training-step residency.** Forward activations, backward, and
  the Adam update live on-device between steps; host sees loss scalars
  and checkpoints only. The kernels exist in the vendored tree
  (layernorm bwd, CE grad, Adam — see PHASE0_KERNEL_AUDIT.md); the
  work is tape-level buffer management. This is the actual Rung C
  unlock and it is NOT part of B1. **Design: docs/CUDA_PHASE_B2.md
  (13 Aug 2026)** — Variable-owned device state (kills the
  pointer-recycling aliasing class), transpose-flag GEMM (kills the
  host `.transpose()` temporaries in backward), staged B2.0-B2.3,
  adoption gated on a d=512 wall-clock win vs AVX.

## B1 design

- **Explicit, never implicit.** `device::make_resident(m)` uploads
  now; `device::invalidate(m)` frees; `device::evict_all()` clears.
  Nothing becomes resident as a side effect, so there is no silent
  staleness class: if you mutate a resident matrix's host data you
  must `invalidate` (or re-`make_resident`) — the contract is stated
  at the API and enforced by the parity test's mutate/invalidate leg.
- **Keying.** The table keys on the host data pointer
  (`Matrix::get_data()`) with dims recorded. Contract: a resident
  matrix must not resize or reallocate while resident (params do
  neither). We deliberately do NOT use the vendored `Matrix`'s
  `gpu_data_`/`is_on_gpu_` fields: their semantics belong to
  transformer_cpp's own path and `sync_vendor.sh` would fight us.
- **Kernel.** A self-contained 32x32 tiled shared-memory GEMM in
  `src/cuda_resident.cu` (raw CUDA runtime API, no cuBLAS — the
  zero-dependency rule extends to the GPU path). Same tiling as the
  audited vendored kernel; edge tiles handled by bounds checks.
- **Fallback chain.** `device::matmul` on CUDA: residency path if
  enabled, else Phase A (`cuda::matmul` round trip), else CPU. The
  residency path returning `false` (disabled) costs one branch.

## Validation contract (unchanged in spirit from Phase A)

On Colab T4:
1. `test_resident_parity` — CPU reference vs Phase A vs B1-resident
   outputs agree (<=1e-4 max abs diff, non-tile-multiple dims), and
   the mutate->invalidate->re-resident leg matches a fresh CPU
   reference.
2. The full gradcheck suite under `MICROTORCH_DEVICE=cuda` with
   residency ON for parameters (same finite-difference oracle that
   gates every other path).
Local machine: compile-only (nvcc, no GPU execution) — the standing
no-local-GPU rule holds.

## Perf note for the record

B1's expected effect is measured, not assumed: the benchmark is
tokens/s on generation and eval-pass wall-clock at d=256/512, CPU vs
Phase A vs B1, on T4. Numbers go in this file when the Colab slot
frees. If B1-at-tiny-dims still loses to AVX, that is a finding, not
a failure — it sharpens the B2 case.
