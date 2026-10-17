#!/usr/bin/env bash
# Client side of tools/colab_throughput_probe.py. AGENTS.md path C: one
# session, stopped on every exit path (trap). Run in WSL from the repo root:
#   bash tools/colab_throughput_probe.sh OUT_DIR [blocked]
# Every colab call has a hard timeout (a dead connection once hung one poll
# for 45 min); the session key is refreshed each poll (keys expire hourly);
# and if the CLI drops the session, the VM's own recorded endpoint is
# re-adopted, so a session that is not ours is never touched.
set -u
COL=~/.local/bin/colab
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
S=pk-probe
OUT=${1:-experiments/chat_sizing/receipts}
MODE=${2:-stacked}
RES=$([ "$MODE" = blocked ] && echo results_blocked.json || echo results.json)
mkdir -p "$OUT"
trap 'timeout 120 $COL stop -s $S > /dev/null 2>&1; echo "session $S stopped"' EXIT
timeout 600 $COL new -s $S --gpu L4 || { echo "could not create session"; exit 1; }
EP=$(timeout 120 $COL sessions 2>/dev/null | grep "^\[$S\]" | awk '{print $2}')
echo "endpoint $EP"
cat > /tmp/pk_probe_launch.py <<PY
import subprocess, urllib.request
src = urllib.request.urlopen("https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/master/tools/colab_throughput_probe.py").read()
open("/content/colab_throughput_probe.py", "wb").write(src)
subprocess.Popen("nohup python3 /content/colab_throughput_probe.py $MODE > /content/probe_driver.log 2>&1 &", shell=True)
print("LAUNCHED")
PY
timeout 200 $COL exec -s $S -f /tmp/pk_probe_launch.py --timeout 120 || { echo "launch failed"; exit 1; }
cat > /tmp/pk_probe_poll.py <<'PY'
import os
print("DONE" if os.path.exists("/content/probe/PROBE_DONE") else "RUNNING")
PY
deadline=$(( $(date +%s) + 110*60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 180
  timeout 180 $PY tools/colab_adopt.py --refresh > /dev/null 2>&1
  if ! timeout 120 $COL sessions 2>/dev/null | grep -q "^\[$S\]"; then
    [ -n "$EP" ] && timeout 180 $PY tools/colab_adopt.py --adopt "$EP=$S" | tail -1
  fi
  r=$(timeout 200 $COL exec -s $S -f /tmp/pk_probe_poll.py --timeout 120 2>&1 | grep -E '^(DONE|RUNNING)' | head -1)
  echo "$(date +%H:%M) ${r:-no-answer}"
  [ "$r" = DONE ] && break
done
timeout 300 $COL download -s $S /content/probe/$RES "$OUT/throughput_probe_$RES" && echo "results -> $OUT"
pkill -f "[c]olab_adopt.py --adopt $EP" 2>/dev/null
