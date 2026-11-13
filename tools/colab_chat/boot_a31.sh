#!/usr/bin/env bash
# VM side of A31 (lane PROTOCOL.md section 50; tools/colab_chat/runner_multi.py --boot-script). Claude Opus 5.5, 13 Nov 2026.
# Builds paperkiln at COMMIT for the pod's GPU (its compute capability, read from nvidia-smi), checks every upload against
# a31_inputs.json (the runner uploads it with the inputs), records provenance, then hands the jobs file to jobs_vm.py.
# Idempotent: a relaunch on the campaign's volume reuses the build and resumes the jobs. No corpus fetch: A31 needs none.
#   bash boot_a31.sh COMMIT JOBS_JSON_IN_REPO
set -euo pipefail
COMMIT=$1
JOBS=$2
cd /content
LOG=/content/chat/boot.log
mkdir -p /content/chat/upload
exec >> "$LOG" 2>&1
trap 'echo "boot failed at line $LINENO"; echo BOOT_FAILED > /content/chat/STATUS' ERR
echo "=== boot $(date -u) commit $COMMIT jobs $JOBS ==="
nvidia-smi -L || true
nproc; free -g | head -2
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
echo "compute capability $CAP"
if [ ! -x /content/pk/build_cuda/mtstudio ] || [ "$(git -C /content/pk rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]; then
  rm -rf /content/pk
  git clone -q https://github.com/JohnnyTeutonic/paperkiln.git /content/pk
  git -C /content/pk checkout -q "$COMMIT"
  mkdir -p /content/pk/build_cuda && cd /content/pk/build_cuda
  cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="$CAP" > cmake.log 2>&1
  make -j"$(nproc)" mtstudio > make.log 2>&1 || { tail -30 make.log; echo BOOT_FAILED > /content/chat/STATUS; exit 1; }
fi
cd /content/pk
[ "$(git rev-parse HEAD)" = "$COMMIT" ] && [ -z "$(git status --porcelain --untracked-files=no)" ]
sha256sum build_cuda/mtstudio > /content/chat/mtstudio.sha256
python3 tools/colab_chat/provenance.py /content/chat/provenance.json || true
# every input as pinned: a31_inputs.json maps upload name -> sha256
python3 - <<'PY'
import hashlib, json, sys
pins = json.load(open("/content/chat/upload/a31_inputs.json"))
bad = []
for name, want in pins.items():
    h = hashlib.sha256()
    with open(f"/content/chat/upload/{name}", "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != want:
        bad.append(name)
if bad:
    sys.exit(f"inputs differ from a31_inputs.json: {bad}")
print(f"inputs verified: {len(pins)}")
PY
if pgrep -f "[j]obs_vm.py" > /dev/null; then echo "jobs_vm already running"; exit 0; fi
echo RUNNING > /content/chat/STATUS
nohup bash -c "python3 -u tools/colab_chat/jobs_vm.py $JOBS > /content/chat/jobs.log 2>&1; echo FINISHED > /content/chat/STATUS" \
  > /dev/null 2>&1 &
echo "jobs launched $(date -u)"
