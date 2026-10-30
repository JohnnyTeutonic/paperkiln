#!/usr/bin/env bash
# Uploaded-script command jobs need the pinned manager, not training/vendor history.
set -euo pipefail
COMMIT=${1:?commit}; JOBS=${2:?jobs JSON}
ROOT=${COMMAND_BOOT_ROOT:-/content}
URL=${COMMAND_BOOT_REPO:-https://github.com/JohnnyTeutonic/paperkiln.git}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$JOBS" == "tools/colab_chat/${JOBS##*/}" && "$JOBS" == *.json ]]
mkdir -p "$ROOT/chat/upload" "$ROOT/pk"
exec >> "$ROOT/chat/boot.log" 2>&1
trap 'echo "sparse command boot failed at line $LINENO"; echo BOOT_FAILED > "$ROOT/chat/STATUS"' ERR
if [ ! -d "$ROOT/pk/.git" ]; then
 git init -q "$ROOT/pk"
 git -C "$ROOT/pk" remote add origin "$URL"
 git -C "$ROOT/pk" config remote.origin.promisor true
 git -C "$ROOT/pk" config remote.origin.partialclonefilter blob:none
 git -C "$ROOT/pk" sparse-checkout init --no-cone
 git -C "$ROOT/pk" sparse-checkout set '/tools/colab_chat/'
fi
if [ "$(git -C "$ROOT/pk" rev-parse --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
 git -C "$ROOT/pk" fetch -q --depth=1 --filter=blob:none origin "$COMMIT"
fi
git -C "$ROOT/pk" checkout -q --detach "$COMMIT"
cd "$ROOT/pk"
[ "$(git rev-parse HEAD)" = "$COMMIT" ]
python3 - "$JOBS" <<'PY'
import json, sys
jobs = json.load(open(sys.argv[1]))['jobs']
if not jobs or any('cmd' not in job for job in jobs):
    raise SystemExit('sparse command boot refuses training jobs')
PY
export JOBS_CHAT_DIR="$ROOT/chat" JOBS_REPO="$ROOT/pk"
python3 tools/colab_chat/provenance.py "$ROOT/chat/provenance.json" || true
if pgrep -f '[j]obs_vm.py' > /dev/null; then echo 'jobs_vm already running'; exit 0; fi
echo RUNNING > "$ROOT/chat/STATUS"
nohup bash -c 'python3 -u tools/colab_chat/jobs_vm.py "$1" > "$2/chat/jobs.log" 2>&1; echo FINISHED > "$2/chat/STATUS"' \
 _ "$JOBS" "$ROOT" > /dev/null 2>&1 &
echo "sparse command jobs launched $(date -u)"
