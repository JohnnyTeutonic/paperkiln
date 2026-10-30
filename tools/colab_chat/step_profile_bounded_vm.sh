#!/usr/bin/env bash
# One isolated command-only campaign: build committed sources, gate, then profile only.
set -euo pipefail
OUT=${1:?output}; VOCAB=${2:?vocabulary}
mkdir -p "$OUT"
cd /content/pk
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89 > "$OUT/cmake.log" 2>&1
cmake --build build_cuda --target mtstudio -j "$(nproc)" > "$OUT/build.log" 2>&1
python3 tools/colab_chat/tiny_chat_run.py cuda_gate --out "$OUT/cuda_gate" > "$OUT/cuda_gate_driver.log" 2>&1
python3 tools/step_profile_bounded.py --out "$OUT/profile" --vocab "$VOCAB" --binary /content/pk/build_cuda/mtstudio
