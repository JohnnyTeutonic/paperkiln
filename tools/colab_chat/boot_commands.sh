#!/usr/bin/env bash
# Command-only jobs on the normal runner/watchdog; no unused trainer or corpus setup.
set -euo pipefail
COMMIT=${1:?commit}; JOBS=${2:?jobs JSON}
cd /content
mkdir -p /content/chat/upload
exec >> /content/chat/boot.log 2>&1
trap 'echo "command boot failed at line $LINENO"; echo BOOT_FAILED > /content/chat/STATUS' ERR
if [ ! -d /content/pk/.git ]; then
 git clone -q --no-checkout https://github.com/JohnnyTeutonic/paperkiln.git /content/pk
 git -C /content/pk checkout -q "$COMMIT"
fi
cd /content/pk
[ "$(git rev-parse HEAD)" = "$COMMIT" ]
python3 - "$JOBS" <<'PY'
import json, sys
jobs = json.load(open(sys.argv[1]))['jobs']
if not jobs or any('cmd' not in job for job in jobs):
    raise SystemExit('command-only boot refuses training jobs')
PY
python3 tools/colab_chat/provenance.py /content/chat/provenance.json || true
if pgrep -f '[j]obs_vm.py' > /dev/null; then echo 'jobs_vm already running'; exit 0; fi
echo RUNNING > /content/chat/STATUS
nohup bash -c 'python3 -u tools/colab_chat/jobs_vm.py "$1" > /content/chat/jobs.log 2>&1; echo FINISHED > /content/chat/STATUS' \
 _ "$JOBS" > /dev/null 2>&1 &
echo "command jobs launched $(date -u)"
