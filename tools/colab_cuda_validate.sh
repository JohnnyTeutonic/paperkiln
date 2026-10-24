#!/usr/bin/env bash
# CUDA validation for microtorch's device seam — run on Colab (T4 is fine),
# never locally. Usage in a Colab cell:
#
#   !git clone https://github.com/JohnnyTeutonic/paperkiln.git
#   # transformer_cpp must sit next to microtorch (same layout as the repo):
#   !git clone <transformer_cpp remote> transformer_cpp
#   !bash microtorch/tools/colab_cuda_validate.sh
#
# Pass = the SAME gradcheck suite that gates the CPU path passes with
# device::matmul dispatching to CudaMatrix::matmul. That is the whole
# contract: numerics agree with the finite-difference oracle on GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/3 configure (MICROTORCH_CUDA=ON) =="
mkdir -p "$ROOT/build_cuda" && cd "$ROOT/build_cuda"
cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON

echo "== 2/3 build =="
make microtorch test_gradcheck test_nn test_lora_quant test_resident_parity \
     test_step_residency test_cuda_ops test_recursion test_mor test_state_tokens test_llama_attnres test_mixed_precision -j"$(nproc)"

echo "== 3/3 run gradchecks on CUDA =="
# device::set(CUDA) is process-wide; MICROTORCH_DEVICE=cuda is read by the
# test mains when present (CPU remains the default everywhere else).
MICROTORCH_DEVICE=cuda ./test_gradcheck
MICROTORCH_DEVICE=cuda ./test_nn
MICROTORCH_DEVICE=cuda ./test_lora_quant
# Phase B1 gate (docs/CUDA_PHASE_B.md): residency parity incl. the
# mutate->invalidate->re-resident contract leg.
MICROTORCH_DEVICE=cuda ./test_resident_parity
# Phase B2.0 gate (docs/CUDA_PHASE_B2.md): training pin + staleness probe
# (sets its own device internally; legs 2-3 activate on CUDA builds).
./test_step_residency
# Phase B2.1a gate (docs/CUDA_PHASE_B2.md section 4): the device op set --
# kernel parity vs the CPU formulas + composed-tape parity OFF vs ON.
# Leg 9 (24 Oct 2026) is the SFT loss: masked cross-entropy on the device
# vs the host op, kept-row gradient bitwise vs device cross_entropy, the
# all-ones mask bitwise vs cross_entropy, and matmul -> masked CE under
# deferral with the logits still device-only after the loss.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_cuda_ops
# The FULL suites again with the op set live: same FD oracle, ops on GPU.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_gradcheck
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_nn
# Phase B2.1b gate: deferred downloads. test_cuda_ops leg 3 covers the
# staleness contract; the full suites then rerun with deferral live --
# step-windowed paths defer, everything else must be untouched.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_STEP_RESIDENCY=1   MICROTORCH_DEFER_DOWNLOADS=1 ./test_cuda_ops
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_STEP_RESIDENCY=1   MICROTORCH_DEFER_DOWNLOADS=1 ./test_step_residency
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_STEP_RESIDENCY=1   MICROTORCH_DEFER_DOWNLOADS=1 ./test_gradcheck
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_STEP_RESIDENCY=1   MICROTORCH_DEFER_DOWNLOADS=1 ./test_nn
# Tied-block recursion: gradcheck, tied vs untied, and training steps with the
# op set, step residency and deferred downloads against the host path.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_recursion
# Mixture-of-Recursions (schedule mor): routing, causality, gradcheck, and the
# training-step comparison (first micro-batch depths identical on both paths).
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_mor
# Attention residuals on the llama block (llama_attnres): the depth_attend
# kernels against the host loop at d 320 / 33 sources, and training steps with
# the op set, step residency and deferred downloads against the host path.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_llama_attnres
# Per-turn state tokens (turn_state mask, IR_SPEC_DRAFT section 1): the device
# turn_state softmax kernel against the host loops (forward and backward), the
# gradchecks and prefill = decode with the op set live, the KV cache with and
# without eviction, and training steps with the op set, step residency and
# deferred downloads against the host path.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_state_tokens
# Mixed precision and flash attention (24 Oct 2026; train.precision bf16,
# train.attention_kernel flash): the device BF16 GEMM (cublasGemmEx) against
# the host emulation (1e-5), the device flash kernels against the host flash
# loops and the tiled path on every mask (y 2e-6, dqkv 1e-5), FD gradchecks of
# the flash op on the device, and device training runs (step residency +
# deferred downloads) against the host: FP32 rerun bitwise, flash vs tiled 1e-4,
# BF16 vs FP32 2e-2, device vs host 1e-4 (BF16 5e-3). Then without the op set
# (the Phase A emulation and the host flash loops on a CUDA build).
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_mixed_precision
MICROTORCH_DEVICE=cuda ./test_mixed_precision
echo "CUDA validation PASSED (Phase A + B1 + B2.0 + B2.1a op set + B2.1b deferred downloads + masked CE + bf16/flash)"
