#!/usr/bin/env bash
# VM side of the FineWeb-Edu base pretraining (experiments/chat_general/PLAN.md),
# driven by tools/colab_chat/runner.py --boot boot_fineweb.sh --model-name fw_base.
# Same layout as boot_base.sh (/content/chat/base, STATUS, upload/) on its own
# VM and session, so the runner needs no other change. Idempotent: every
# relaunch runs it again and it resumes from whatever checkpoint the runner
# uploaded into /content/chat/upload/.
#   bash boot_fineweb.sh COMMIT
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
# The sample and the vocabulary are pinned in the repo, so they reproduce or
# the boot stops; the full shard pins on its first fetch and the runner's
# relay brings nothing of it home (it is 3+ GB): its hash is in the boot log.
python3 corpus/fetchers/fineweb_edu.py sample_v1
python3 corpus/fetchers/tinychat_hf.py sample_v1
python3 corpus/fetchers/vocab_fineweb_tinychat_bpe.py
python3 corpus/fetchers/fineweb_edu.py shard0_v1 --pin
cat corpus/hashes/fineweb_edu_shard0_v1.sha256 || true

# Restore an uploaded checkpoint (reassembling 45 MB parts), once per upload.
cd /content/chat/upload
for f in model.safetensors optim.safetensors; do
  if ls "$f".part* > /dev/null 2>&1; then cat $(ls "$f".part* | sort) > "/content/chat/base/$f" && rm -f "$f".part*; fi
done
for f in state.txt events.jsonl; do [ -f "$f" ] && mv -f "$f" /content/chat/base/; done
cd /content/pk

# d384 L6 (option A): ~195M tokens at batch 16 x 256 = 47,500 steps, ~24
# L4-hours at the measured 2,250 tokens/s. The rate is constant, so the step
# count can rise on a resume if the run is faster than planned.
cat > /content/chat/spec.json <<EOF
{"name": "fw_base", "arch": {"preset": "llama-tiny", "custom": {"d": 384, "layers": 6, "heads": 6}},
 "data": {"corpus": "/content/pk/corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt",
          "vocab": "/content/pk/corpus/data/vocab_fineweb_tinychat_bpe8k_v1/fineweb_tinychat_bpe8k.gguf",
          "vocab_cap": 8192, "T": 256, "max_tokens": 200000000},
 "train": {"steps": 47500, "lr": 0.001, "batch": 16, "eval_every": 500, "checkpoint_every": 500,
           "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
 "export": {"formats": ["safetensors", "gguf"]}, "out_dir": "/content/chat/base"}
EOF
pkill -f "[b]uild_cuda/mtstudio run" || true
echo RUNNING > /content/chat/STATUS
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_STEP_RESIDENCY=1 \
MICROTORCH_DEFER_DOWNLOADS=1 MICROTORCH_CUBLAS=1 OMP_NUM_THREADS=4 \
  nohup bash -c "/content/pk/build_cuda/mtstudio run /content/chat/spec.json > /content/chat/train.log 2>&1; echo FINISHED > /content/chat/STATUS" \
  > /dev/null 2>&1 &
echo "launched $(date -u)"
