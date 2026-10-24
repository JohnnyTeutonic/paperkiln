#!/usr/bin/env bash
# RunPod boot prelude (runner_multi.py --backend runpod, runpod_transport.py): run over ssh on every
# new pod before prepare() and boot_jobs.sh, so the pod looks like the Colab VM boot_jobs.sh expects.
# The campaign's network volume is mounted at /content (so /content/chat and /content/pk are the
# same paths as on Colab and survive the pod). Idempotent.
#   bash runpod_prelude.sh COMMIT
set -euo pipefail
COMMIT=$1
mkdir -p /content/chat/upload
echo "=== runpod prelude $(date -u) commit $COMMIT ==="

# A previous pod's process ids mean nothing on this one (pids restart from 1 in a container):
# jobs_vm.py's lock and the jobs' pid files would make it refuse to start, or adopt a stranger.
rm -f /content/chat/jobs_vm.pid /content/chat/*.pid /content/chat/STATUS /content/chat/provenance.json
# Resume uploads begun for the old pod are void: prepare() decides afresh what goes up, and a stale
# checkpoint part here would be restored over the newer checkpoint the volume holds.
find /content/chat/upload -mindepth 2 -maxdepth 2 -type f -delete 2>/dev/null || true
# A build (and its clone) from another commit on this volume is never reused.
if [ -d /content/pk/.git ]; then
  head=$(git -C /content/pk rev-parse HEAD 2>/dev/null || echo none)
  case "$head" in
    "$COMMIT"*) echo "paperkiln build on the volume is at $head" ;;
    *) echo "paperkiln on the volume is at $head, not $COMMIT: removed"; rm -rf /content/pk ;;
  esac
fi

# What boot_jobs.sh and the jobs use that the image may lack (Colab has all of it).
export DEBIAN_FRONTEND=noninteractive
need=""
for c in rsync pgrep cmake git make g++; do command -v "$c" > /dev/null || need="$need $c"; done
dpkg -s libboost-regex-dev > /dev/null 2>&1 || need="$need libboost-regex-dev"
if [ -n "$need" ]; then
  echo "installing:$need"
  apt-get update -qq
  apt-get install -y -q rsync procps cmake git build-essential libboost-regex-dev > /dev/null
fi
python3 -m pip -q install --root-user-action=ignore huggingface_hub pandas pyarrow tokenizers numpy 2>&1 | tail -3 || true
nvidia-smi -L
nvcc --version | tail -2 || true
# The host driver's CUDA (the create carries no allowedCudaVersions filter): the image's 12.4 runtime and
# sm_89 binaries run on any 12.x driver (minor-version compatibility); older is refused.
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\..*/\1/p' | head -1)
echo "host driver CUDA major: ${cuda:-unknown}"
if [ -z "$cuda" ] || [ "$cuda" -lt 12 ]; then echo "host CUDA driver older than 12: refused"; exit 3; fi
nproc; free -g | head -2; df -h /content | tail -1
echo "prelude done $(date -u)"
