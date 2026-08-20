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
     test_step_residency test_cuda_ops -j"$(nproc)"

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
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_cuda_ops
# The FULL suites again with the op set live: same FD oracle, ops on GPU.
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_gradcheck
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_nn
echo "CUDA validation PASSED (Phase A + B1 + B2.0 + B2.1a op set)"
