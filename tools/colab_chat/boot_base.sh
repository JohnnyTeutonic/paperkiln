#!/usr/bin/env bash
# VM side of the chat base pretraining (tools/colab_chat/runner.py).
# Idempotent: every relaunch runs it again and it resumes from whatever
# checkpoint the runner uploaded into /content/chat/upload/.
#   bash boot_base.sh COMMIT
set -euo pipefail
COMMIT=$1
cd /content
LOG=/content/chat/boot.log
mkdir -p /content/chat/base /content/chat/upload
exec >> "$LOG" 2>&1
echo "=== boot $(date -u) commit $COMMIT ==="
nvidia-smi -L || true

if [ ! -x /content/pk/build_cuda/mtstudio ]; then
  rm -rf /content/pk /content/transformer_cpp
  git clone -q https://github.com/JohnnyTeutonic/paperkiln.git /content/pk
  git -C /content/pk checkout -q "$COMMIT"
  git clone -q --depth 1 https://github.com/JohnnyTeutonic/coalfire.cpp.git /content/transformer_cpp
  echo "coalfire $(git -C /content/transformer_cpp rev-parse HEAD)"
  mkdir -p /content/pk/build_cuda && cd /content/pk/build_cuda
  cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON > cmake.log 2>&1
  make -j"$(nproc)" mtstudio > make.log 2>&1 || { tail -30 make.log; echo BOOT_FAILED > /content/chat/STATUS; exit 1; }
fi
cd /content/pk
pip -q install huggingface_hub pandas pyarrow
python3 corpus/fetchers/tinychat_hf.py sample_v1
python3 corpus/fetchers/vocab_tinychat_bpe.py
python3 corpus/fetchers/tinychat_hf.py full_v1 --pin   # pins on first fetch; the runner brings the hash home
cp corpus/hashes/tinychat_hf_full_v1.sha256 /content/chat/ 2>/dev/null || true

# Restore an uploaded checkpoint (reassembling 45 MB parts), once per upload.
cd /content/chat/upload
for f in model.safetensors optim.safetensors; do
  if ls "$f".part* > /dev/null 2>&1; then cat $(ls "$f".part* | sort) > "/content/chat/base/$f" && rm -f "$f".part*; fi
done
for f in state.txt events.jsonl; do [ -f "$f" ] && mv -f "$f" /content/chat/base/; done
cd /content/pk

cat > /content/chat/spec.json <<EOF
{"name": "chat_base", "arch": {"preset": "llama-tiny", "custom": {"d": 384, "layers": 6, "heads": 6}},
 "data": {"corpus": "/content/pk/corpus/data/tinychat_hf_full_v1/pretrain_train.txt",
          "vocab": "/content/pk/corpus/data/vocab_tinychat_bpe8k_v1/tinychat_bpe8k.gguf",
          "vocab_cap": 8192, "T": 256, "max_tokens": 60000000},
 "train": {"steps": 12000, "lr": 0.001, "batch": 16, "eval_every": 250, "checkpoint_every": 250,
           "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
 "export": {"formats": ["safetensors", "gguf"]}, "out_dir": "/content/chat/base"}
EOF
pkill -f "[b]uild_cuda/mtstudio run" || true
echo RUNNING > /content/chat/STATUS
# Residency + deferral + cuBLAS FP32: 2.47x at the chat shape on an L4, step
# losses unchanged; deferral passed 300 steps with evals, checkpoints, export
# and a resume (18 Oct 2026).
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_STEP_RESIDENCY=1 \
MICROTORCH_DEFER_DOWNLOADS=1 MICROTORCH_CUBLAS=1 OMP_NUM_THREADS=4 \
  nohup bash -c "/content/pk/build_cuda/mtstudio run /content/chat/spec.json > /content/chat/train.log 2>&1; echo FINISHED > /content/chat/STATUS" \
  > /dev/null 2>&1 &
echo "launched $(date -u)"
