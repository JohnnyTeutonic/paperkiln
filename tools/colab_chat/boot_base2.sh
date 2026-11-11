#!/usr/bin/env bash
# VM side of a store-backed session (runner_multi.py --ckpt-store hf; AI_ML/chat_screens_claude/base2/BASE2_PLAN.md
# section 1.5): build paperkiln at COMMIT as boot_jobs.sh does, then take everything else from the checkpoint store
# (tools/colab_chat/hf_store.py): the jobs file's "store_inputs" (token ids, panels, vocabulary, initial weights),
# each file checked by sha256, and each unfinished job's newest whole generation, installed as the job's live
# trainer generation (ckpt_courier.py restore). Then the courier and jobs_vm.py start. Nothing is fetched from
# FineWeb here. Idempotent: every relaunch runs it again.
#   bash boot_base2.sh COMMIT JOBS_JSON_IN_REPO
# The runner has put chat/store.json (repo, branches, campaign, epoch, min_step) and has moved the token to a tmpfs
# (its path is "token_file" in store.json) before this runs. The token is never echoed: no `set -x` in this file.
# A failed step, or a boot longer than PK_BOOT_LIMIT_S, ends as BOOT_FAILED with the courier stopped and every copy
# of the token removed. When every job has ended and the courier has pushed the last of it, the token is removed too.
# PK_BOOT_ROOT (default /content) moves the whole tree for a local rehearsal with a dir: store.
set -euo pipefail
COMMIT=$1
JOBS=$2
ROOT=${PK_BOOT_ROOT:-/content}
CHAT=$ROOT/chat
PK=$ROOT/pk
CFG=$CHAT/store.json
HUB=0.35.3   # hf_store.HUB_VERSION: the version the store's calls were written against
LIMIT=${PK_BOOT_LIMIT_S:-3600}
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # no fetch on this VM sends a token it was not handed
mkdir -p "$CHAT/upload"
cd "$ROOT"
LOG=$CHAT/boot.log
exec >> "$LOG" 2>&1
forget() {  # the courier and its children first, then every copy of the token
  python3 "$PK/tools/colab_chat/hf_store.py" forget-token --config "$CFG" --chat "$CHAT" 2> /dev/null \
    || rm -f "$CHAT/upload/store_token" "${PK_STORE_TOKEN_DIR:-/dev/shm/pk_store}"/token*
}
failed() {
  echo "boot failed: $1"
  echo BOOT_FAILED > "$CHAT/STATUS"
  forget || true
}
# Any failed step ends the boot as BOOT_FAILED (boot_jobs.sh, 24 Oct 2026). A store that cannot be read is such a
# step: a job must never start from step 0 because its checkpoint could not be pulled.
trap 'failed "line $LINENO"' ERR
echo "=== boot_base2 $(date -u) commit $COMMIT jobs $JOBS ==="
rm -f "$CHAT/BOOT_DONE"
BOOT_PID=$$
( sleep "$LIMIT"
  if [ ! -e "$CHAT/BOOT_DONE" ] && kill -0 "$BOOT_PID" 2> /dev/null; then
    echo "boot over $LIMIT s: ended" >> "$LOG"
    kids=$(pgrep -P "$BOOT_PID" || true)
    kill -TERM "$BOOT_PID" 2> /dev/null || true
    for k in $kids; do [ "$k" = "$BASHPID" ] || kill -TERM "$k" 2> /dev/null || true; done
    echo BOOT_FAILED > "$CHAT/STATUS"
    forget >> "$LOG" 2>&1 || true
  fi ) > /dev/null 2>&1 &
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
  make -j"$(nproc)" mtstudio > make.log 2>&1 || { tail -30 make.log; failed "make"; exit 1; }
fi
echo "build: $(( $(date +%s) - T0 )) s"
cd "$PK"
if [ "$ROOT" = /content ]; then  # a local rehearsal never queries the laptop's GPU
  python3 tools/colab_chat/provenance.py "$CHAT/provenance.json" || true
fi

[ -s "$CFG" ] || { echo "no store config at $CFG"; false; }
if ! grep -q '"repo": "dir:' "$CFG"; then
  have=$(python3 -c "import huggingface_hub as h; print(h.__version__)" 2> /dev/null || true)
  if [ "$have" != "$HUB" ]; then
    for try in 1 2 3; do timeout 600 pip -q install "huggingface_hub==$HUB" && break; sleep 20; done
  fi
  [ "$(python3 -c "import huggingface_hub as h; print(h.__version__)")" = "$HUB" ]
  echo "huggingface_hub $HUB"
fi
# The runner adopts the token itself, seconds after it lands; this covers a token still in upload/.
python3 tools/colab_chat/hf_store.py adopt-token --config "$CFG" --from "$CHAT/upload/store_token" \
  --dir "${PK_STORE_TOKEN_DIR:-/dev/shm/pk_store}"
python3 -c "import uuid; print(uuid.uuid4().hex)" > "$CHAT/boot_id"

T0=$(date +%s)
python3 tools/colab_chat/hf_store.py inputs --config "$CFG" --jobs "$JOBS" --root "$ROOT" \
  --record "$CHAT/store_inputs.json" --limit "${PK_INPUTS_LIMIT_S:-2700}"
echo "inputs: $(( $(date +%s) - T0 )) s"

alive() {  # a pid file whose process is this script's kind
  [ -s "$1" ] && [ -r "/proc/$(cat "$1")/cmdline" ] && tr '\0' ' ' < "/proc/$(cat "$1")/cmdline" | grep -q "$2"
}
start_courier() {
  alive "$CHAT/courier.pid" ckpt_courier.py && return 0
  nohup setsid python3 -u tools/colab_chat/ckpt_courier.py run --config "$CFG" --jobs "$JOBS" --chat "$CHAT" \
    ${PK_COURIER_ARGS:-} > /dev/null 2>&1 &
}
if alive "$CHAT/jobs_vm.pid" jobs_vm.py; then
  echo "jobs_vm already running"
  start_courier
  touch "$CHAT/BOOT_DONE"
  exit 0
fi
T0=$(date +%s)
python3 tools/colab_chat/ckpt_courier.py restore --config "$CFG" --jobs "$JOBS" --chat "$CHAT" \
  --limit "${PK_RESTORE_LIMIT_S:-1500}" ${PK_RESTORE_ARGS:-}
echo "restore: $(( $(date +%s) - T0 )) s"
start_courier
if [ "$ROOT" != /content ]; then
  export JOBS_CHAT_DIR=$CHAT JOBS_REPO=$PK JOBS_MTSTUDIO=${JOBS_MTSTUDIO:-$PK/build_cuda/mtstudio}
fi
echo RUNNING > "$CHAT/STATUS"
touch "$CHAT/BOOT_DONE"
# When the manager has ended: wait for the courier to push the last of it and exit, then take the token off the VM.
nohup bash -c "python3 -u tools/colab_chat/jobs_vm.py $JOBS > $CHAT/jobs.log 2>&1; echo FINISHED > $CHAT/STATUS
  while [ -s $CHAT/courier.pid ] && kill -0 \$(cat $CHAT/courier.pid) 2> /dev/null; do sleep 5; done
  python3 tools/colab_chat/hf_store.py forget-token --config $CFG --chat $CHAT >> $LOG 2>&1" \
  > /dev/null 2>&1 &
echo "jobs launched $(date -u)"
