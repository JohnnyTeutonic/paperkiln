#!/usr/bin/env bash
# VM side of a store-backed session (runner_multi.py --ckpt-store hf; AI_ML/chat_screens_claude/base2/BASE2_PLAN.md
# section 1.5): build paperkiln at COMMIT as boot_jobs.sh does, then take everything else from the checkpoint store
# (tools/colab_chat/hf_store.py): the jobs file's "store_inputs" (token ids, panels, vocabulary, initial weights),
# each file checked by sha256, and each unfinished job's newest whole generation, placed in chat/upload/<job>/ with
# READY for jobs_vm.py's restore. Then the courier (ckpt_courier.py) and jobs_vm.py start. Nothing is fetched from
# FineWeb here. Idempotent: every relaunch runs it again.
#   bash boot_base2.sh COMMIT JOBS_JSON_IN_REPO
# The runner has put chat/store.json (repo, branches) and chat/upload/store_token before this runs. The token is
# moved to a tmpfs, so it never sits on a volume, and is never echoed: no `set -x` in this file.
# PK_BOOT_ROOT (default /content) moves the whole tree for a local rehearsal with a dir: store.
set -euo pipefail
COMMIT=$1
JOBS=$2
ROOT=${PK_BOOT_ROOT:-/content}
CHAT=$ROOT/chat
PK=$ROOT/pk
CFG=$CHAT/store.json
mkdir -p "$CHAT/upload"
cd "$ROOT"
LOG=$CHAT/boot.log
exec >> "$LOG" 2>&1
# Any failed step ends the boot as BOOT_FAILED (boot_jobs.sh, 24 Oct 2026). A store that cannot be read is such a
# step: a job must never start from step 0 because its checkpoint could not be pulled.
trap 'echo "boot failed at line $LINENO"; echo BOOT_FAILED > "$CHAT/STATUS"' ERR
echo "=== boot_base2 $(date -u) commit $COMMIT jobs $JOBS ==="
if [ "$ROOT" = /content ]; then
  nvidia-smi -L || true
  nproc; free -g | head -2; df -h /content | tail -1
fi

T0=$(date +%s)
if [ ! -x "$PK/build_cuda/mtstudio" ]; then
  for try in 1 2 3; do
    rm -rf "$PK"
    git clone -q https://github.com/JohnnyTeutonic/paperkiln.git "$PK" && break
    sleep 30
  done
  git -C "$PK" checkout -q "$COMMIT"
  mkdir -p "$PK/build_cuda" && cd "$PK/build_cuda"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 > cmake.log 2>&1
  grep -i "transformer_core from" cmake.log || true
  make -j"$(nproc)" mtstudio > make.log 2>&1 || { tail -30 make.log; echo BOOT_FAILED > "$CHAT/STATUS"; exit 1; }
fi
echo "build: $(( $(date +%s) - T0 )) s"
cd "$PK"
if [ "$ROOT" = /content ]; then  # a local rehearsal never queries the laptop's GPU
  python3 tools/colab_chat/provenance.py "$CHAT/provenance.json" || true
fi

[ -s "$CFG" ] || { echo "no store config at $CFG"; false; }
if ! grep -q '"repo": "dir:' "$CFG"; then
  python3 -c "import huggingface_hub" 2> /dev/null || {
    for try in 1 2 3; do pip -q install huggingface_hub && break; sleep 20; done
    python3 -c "import huggingface_hub"
  }
fi
python3 tools/colab_chat/hf_store.py adopt-token --config "$CFG" --from "$CHAT/upload/store_token" \
  --dir "${PK_STORE_TOKEN_DIR:-/dev/shm/pk_store}"

T0=$(date +%s)
python3 tools/colab_chat/hf_store.py inputs --config "$CFG" --jobs "$JOBS" --root "$ROOT"
echo "inputs: $(( $(date +%s) - T0 )) s"

alive() {  # a pid file whose process is this script's kind
  [ -s "$1" ] && [ -r "/proc/$(cat "$1")/cmdline" ] && tr '\0' ' ' < "/proc/$(cat "$1")/cmdline" | grep -q "$2"
}
start_courier() {
  alive "$CHAT/courier.pid" ckpt_courier.py && return 0
  nohup python3 -u tools/colab_chat/ckpt_courier.py run --config "$CFG" --jobs "$JOBS" --chat "$CHAT" \
    ${PK_COURIER_ARGS:-} > /dev/null 2>&1 &
}
if alive "$CHAT/jobs_vm.pid" jobs_vm.py; then
  echo "jobs_vm already running"
  start_courier
  exit 0
fi
T0=$(date +%s)
python3 tools/colab_chat/ckpt_courier.py restore --config "$CFG" --jobs "$JOBS" --chat "$CHAT" ${PK_RESTORE_ARGS:-}
echo "restore: $(( $(date +%s) - T0 )) s"
start_courier
if [ "$ROOT" != /content ]; then
  export JOBS_CHAT_DIR=$CHAT JOBS_REPO=$PK JOBS_MTSTUDIO=${JOBS_MTSTUDIO:-$PK/build_cuda/mtstudio}
fi
echo RUNNING > "$CHAT/STATUS"
nohup bash -c "python3 -u tools/colab_chat/jobs_vm.py $JOBS > $CHAT/jobs.log 2>&1; echo FINISHED > $CHAT/STATUS" \
  > /dev/null 2>&1 &
echo "jobs launched $(date -u)"
