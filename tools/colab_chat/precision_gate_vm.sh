#!/usr/bin/env bash
# VM side of the precision gate (tools/precision_gate.py) on the GPU:
#   bash tools/colab_chat/precision_gate_vm.sh OUT UPLOAD PIN_COMMIT [SHAPE BATCH ACCUM STEPS]
# Builds mtstudio at PIN_COMMIT (the commit the registered tiny-chat arms run, in a git worktree
# beside /content/pk, idempotent), then runs every leg of the gate with the production device
# environment: PIN vs HEAD with precision off (bitwise), HEAD repeat and explicit-off (bitwise),
# flash, bf16 and both against FP32 tiled (loss tolerances, throughput). Writes OUT/gate.json.
set -euo pipefail
OUT=$1
UPLOAD=$2
PIN=$3
SHAPE=${4:-20m_L8}
BATCH=${5:-32}
ACCUM=${6:-1}
STEPS=${7:-300}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PIN_DIR=/content/pk_pin_$PIN
mkdir -p "$OUT"
if [ ! -x "$PIN_DIR/build_cuda/mtstudio" ]; then
  rm -rf "$PIN_DIR"
  git -C "$REPO" worktree prune
  git -C "$REPO" worktree add -f --detach "$PIN_DIR" "$PIN"
  mkdir -p "$PIN_DIR/build_cuda"
  cd "$PIN_DIR/build_cuda"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
    > "$OUT/pin_cmake.log" 2>&1
  make -j"$(nproc)" mtstudio > "$OUT/pin_make.log" 2>&1
fi
cd "$REPO"
python3 tools/precision_gate.py run --new /content/pk/build_cuda/mtstudio --old "$PIN_DIR/build_cuda/mtstudio" \
  --text corpus/data/fineweb_edu_sample_v1/pretrain_train.txt --vocab "$UPLOAD/vocab_llama3_bpe.gguf" \
  --out "$OUT" --shape "$SHAPE" --batch "$BATCH" --accum "$ACCUM" --steps "$STEPS"
