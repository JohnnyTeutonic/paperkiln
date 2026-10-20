#!/usr/bin/env bash
# VM side of a multi-job session (tools/colab_chat/runner_multi.py): build
# paperkiln at COMMIT for the L4 (sm_89) against its VENDORED transformer_core
# (the copy the local tests ran against, not coalfire's moving HEAD), fetch the
# pinned FineWeb-Edu shard, vocabulary and TinyChat sample, then hand the jobs
# file to jobs_vm.py. Idempotent: every relaunch runs it again and the jobs
# resume from the checkpoints the runner uploads.
#   bash boot_jobs.sh COMMIT JOBS_JSON_IN_REPO
set -euo pipefail
COMMIT=$1
JOBS=$2
cd /content
LOG=/content/chat/boot.log
mkdir -p /content/chat/upload
exec >> "$LOG" 2>&1
echo "=== boot $(date -u) commit $COMMIT jobs $JOBS ==="
nvidia-smi -L || true
nproc; free -g | head -2

if [ ! -x /content/pk/build_cuda/mtstudio ]; then
  rm -rf /content/pk
  git clone -q https://github.com/JohnnyTeutonic/paperkiln.git /content/pk
  git -C /content/pk checkout -q "$COMMIT"
  mkdir -p /content/pk/build_cuda && cd /content/pk/build_cuda
  cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 > cmake.log 2>&1
  grep -i "transformer_core from" cmake.log || true
  make -j"$(nproc)" mtstudio > make.log 2>&1 || { tail -30 make.log; echo BOOT_FAILED > /content/chat/STATUS; exit 1; }
fi
cd /content/pk
pip -q install huggingface_hub pandas pyarrow
python3 corpus/fetchers/fineweb_edu.py sample_v1
python3 corpus/fetchers/tinychat_hf.py sample_v1
python3 corpus/fetchers/vocab_fineweb_tinychat_bpe.py
python3 corpus/fetchers/fineweb_edu.py shard0_v1 --pin
cat corpus/hashes/fineweb_edu_shard0_v1.sha256 || true
# Text the FineWeb base never read: it consumed ~200M tokens (~0.9 GB) from
# the start of the shard, so continued-training jobs read from 1 GB on.
R=corpus/data/fineweb_edu_shard0_v1/rest.txt
[ -s $R ] || tail -c +1000000001 corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt > $R
ls -la $R
# The big model's corpus: the first 2.2 GB (~500M tokens). Tokenising the
# whole 3.46 GB shard took over 30 GB of host RAM and the OOM killer.
B=corpus/data/fineweb_edu_shard0_v1/big.txt
[ -s $B ] || head -c 2200000000 corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt > $B
ls -la $B
if pgrep -f "[j]obs_vm.py" > /dev/null; then echo "jobs_vm already running"; exit 0; fi
echo RUNNING > /content/chat/STATUS
nohup bash -c "python3 -u tools/colab_chat/jobs_vm.py $JOBS > /content/chat/jobs.log 2>&1; echo FINISHED > /content/chat/STATUS" \
  > /dev/null 2>&1 &
echo "jobs launched $(date -u)"
