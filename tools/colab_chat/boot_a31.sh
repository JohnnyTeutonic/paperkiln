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
if [ ! -x /content/pk/build_cuda/mtstudio ] || [ ! -x /content/pk/build_cuda/test_twin_device ] || [ "$(git -C /content/pk rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]; then
  rm -rf /content/pk
  git clone -q https://github.com/JohnnyTeutonic/paperkiln.git /content/pk
  git -C /content/pk checkout -q "$COMMIT"
  mkdir -p /content/pk/build_cuda && cd /content/pk/build_cuda
  cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="$CAP" > cmake.log 2>&1
  timeout 3600 make -j"$(nproc)" mtstudio test_twin_device > make.log 2>&1 || { tail -30 make.log; echo BOOT_FAILED > /content/chat/STATUS; exit 1; }
fi
# A dead-man on the pod (A31 review, 13 Nov 2026): at the campaign's deadline plus 30 minutes the pod removes itself through
# runpodctl when RunPod provides it and the pod's own credentials; otherwise this is logged, and the home runner, the watchdog's
# sweep and the account's spend limit remain the bounds. The deadline arrives as an upload (a31_deadline.txt, epoch seconds).
if [ -s /content/chat/upload/a31_deadline.txt ] && ! pgrep -f "[a]31_deadman" > /dev/null; then
  DL=$(( $(cat /content/chat/upload/a31_deadline.txt) + 1800 ))
  if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    nohup bash -c "exec -a a31_deadman bash -c 'while [ \$(date +%s) -lt $DL ]; do sleep 60; done; runpodctl remove pod $RUNPOD_POD_ID'" \
      > /content/chat/deadman.log 2>&1 &
    echo "dead-man set for $(date -u -d @$DL)"
  else
    echo "dead-man not available here (runpodctl or RUNPOD_POD_ID missing)"
  fi
fi
cd /content/pk
if [ "$(git rev-parse HEAD)" != "$COMMIT" ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "the checkout is not $COMMIT, clean"; echo BOOT_FAILED > /content/chat/STATUS; exit 1
fi
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
# FINISHED only when jobs_vm.py exits 0; otherwise FAILED, which the runner treats as the session's end (A31 review, 13 Nov 2026)
nohup bash -c "if python3 -u tools/colab_chat/jobs_vm.py $JOBS > /content/chat/jobs.log 2>&1; then echo FINISHED; else echo FAILED; fi > /content/chat/STATUS" \
  > /dev/null 2>&1 &
echo "jobs launched $(date -u)"
