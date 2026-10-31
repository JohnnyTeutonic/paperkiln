#!/usr/bin/env bash
# Named DPO-only boot. Dynamic runtime jobs are a separately hash-bound upload.
set -Eeuo pipefail
COMMIT=${1:?full commit}; LOCAL_JOBS=${2:?local runtime jobs marker}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$LOCAL_JOBS" =~ ^[A-Za-z0-9_./-]+[.]json$ ]]
mkdir -p /content/chat/upload
exec >> /content/chat/boot.log 2>&1
trap 'echo "DPO checkpoint boot failed at line $LINENO"; echo BOOT_FAILED > /content/chat/STATUS' ERR
if [ ! -d /content/pk/.git ]; then
 git init -q /content/pk
 git -C /content/pk remote add origin https://github.com/JohnnyTeutonic/paperkiln.git
fi
if [ "$(git -C /content/pk rev-parse --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
 git -C /content/pk fetch -q --depth=1 origin "$COMMIT"
 git -C /content/pk checkout -q --detach "$COMMIT"
fi
cd /content/pk
[ "$(git rev-parse HEAD)" = "$COMMIT" ]
git diff --exit-code HEAD --
JOBS=/content/chat/upload/DPO_RUNTIME_JOBS_V1.json
python3 - "$JOBS" <<'PY'
import sys
sys.path.insert(0,'tools/colab_chat')
from dpo_checkpoint_manager_v1 import validate_jobs
validate_jobs('/content/pk',sys.argv[1],'/content/chat/upload')
PY
# Validate the independently admitted source archive and uploaded inputs before build/training.
python3 tools/posttrain/dpo_vm_preflight_v7.py /content/chat/upload/dpo_binding.json /content/chat/upload/pk_dpo_v1.spec.json /content/chat/dpo_preflight
PIN=/content/pk/build_cuda/dpo_binary.sha256
if [ -f "$PIN" ] && [ "$(cat /content/pk/build_cuda/dpo_source_commit.txt 2>/dev/null || true)" = "$COMMIT" ]; then
 (cd /content/pk/build_cuda && sha256sum -c dpo_binary.sha256)
else
 cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89
 cmake --build build_cuda --target mtstudio --parallel 4
 (cd build_cuda && sha256sum mtstudio > dpo_binary.sha256)
 echo "$COMMIT" > build_cuda/dpo_source_commit.txt
fi
python3 tools/colab_chat/provenance.py /content/chat/provenance.json
if pgrep -f '[j]obs_vm.py' >/dev/null; then echo 'refuse competing manager'; exit 1; fi
[ ! -e /content/chat/DPO_CHECKPOINT_V1_HALTED.json ]
echo RUNNING > /content/chat/STATUS
nohup python3 -u tools/colab_chat/dpo_checkpoint_manager_v1.py --jobs "$JOBS" > /content/chat/jobs.log 2>&1 &
echo "DPO jobs launched $(date -u)"
