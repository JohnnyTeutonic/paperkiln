#!/usr/bin/env bash
# Explicit control/numerical split: modern operational overlay, original arithmetic.
set -Eeuo pipefail
CONTROL_COMMIT=${1:?control commit}; JOBS=${2:?relative V2 jobs}
[[ "$CONTROL_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$JOBS" =~ ^tools/colab_chat/carry12_sfull_v2_r[123]\.json$ ]]
NUMERICAL_COMMIT=1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1
mkdir -p /content/chat/upload
exec >> /content/chat/boot.log 2>&1
trap 'echo "S-full V2 boot failed at line $LINENO"; echo BOOT_FAILED > /content/chat/STATUS' ERR
checkout() {
 local dir=$1 rev=$2
 if [ ! -d "$dir/.git" ]; then
   mkdir -p "$dir"
   git init -q "$dir"
   git -C "$dir" remote add origin https://github.com/JohnnyTeutonic/paperkiln.git
 fi
 if [ "$(git -C "$dir" rev-parse --verify HEAD 2>/dev/null || true)" != "$rev" ]; then
   git -C "$dir" fetch -q --depth=1 --filter=blob:none origin "$rev"
   git -C "$dir" checkout -q --detach "$rev"
 fi
 [ "$(git -C "$dir" rev-parse HEAD)" = "$rev" ]
 git -C "$dir" diff --quiet HEAD --
}
checkout /content/sfull_control "$CONTROL_COMMIT"
checkout /content/pk "$NUMERICAL_COMMIT"
CONTROL=/content/sfull_control
python3 "$CONTROL/tools/colab_chat/sfull_operational_v2.py" validate --jobs "$CONTROL/$JOBS"
cd /content/pk
# A resumed binary is accepted only with this source commit and its exact bytes.
STAMP=/content/pk/build_cuda/sfull_v2_binary.sha256
if [ -x build_cuda/mtstudio ] && [ -f "$STAMP" ]; then
 (cd build_cuda && sha256sum -c sfull_v2_binary.sha256)
else
 cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89 > /content/chat/cmake.log 2>&1
 cmake --build build_cuda --target mtstudio --parallel 4 > /content/chat/make.log 2>&1
 (cd build_cuda && sha256sum mtstudio > sfull_v2_binary.sha256)
fi
python3 tools/colab_chat/provenance.py /content/chat/provenance.json
pip -q install huggingface_hub pandas pyarrow tokenizers==0.23.3
# No --pin: a source mismatch is an error, never a new expectation.
python3 corpus/fetchers/fineweb_edu.py shard0_v1
(cd corpus/data/fineweb_edu_shard0_v1 && sha256sum -c ../../hashes/fineweb_edu_shard0_v1.sha256)
if pgrep -f '[j]obs_vm.py' >/dev/null; then
 echo 'existing manager; refuse duplicate'
 exit 1
fi
export MICROTORCH_CHECKPOINT_PUBLISH_V2="$CONTROL/tools/colab_chat/sfull_checkpoint_v2.py"
export S_FULL_NUMERICAL_COMMIT="$NUMERICAL_COMMIT"
test ! -e /content/chat/CHECKPOINT_V2_HALTED.json
echo RUNNING > /content/chat/STATUS
nohup python3 -u "$CONTROL/tools/colab_chat/sfull_operational_v2.py" manage --jobs "$CONTROL/$JOBS" > /content/chat/jobs.log 2>&1 < /dev/null &
