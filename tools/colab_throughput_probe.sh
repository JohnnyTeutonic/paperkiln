#!/usr/bin/env bash
# Client side of tools/colab_throughput_probe.py. AGENTS.md path C: one
# script, one session, stopped on every exit path (trap). Run in WSL from
# the repository root:  bash tools/colab_throughput_probe.sh [OUT_DIR]
# The VM job runs detached; this script polls a done marker with short
# execs (long execs are where sessions die) and gives up after 110 minutes,
# inside the ~2 h headless-session lifetime.
set -u
COL=~/.local/bin/colab
S=pk-probe
OUT=${1:-experiments/chat_sizing/receipts}
mkdir -p "$OUT"
trap '$COL stop -s $S > /dev/null 2>&1; echo "session $S stopped"' EXIT
$COL new -s $S --gpu L4 || { echo "could not create session"; exit 1; }
cat > /tmp/pk_probe_launch.py <<'PY'
import subprocess, urllib.request
src = urllib.request.urlopen("https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/master/tools/colab_throughput_probe.py").read()
open("/content/colab_throughput_probe.py", "wb").write(src)
subprocess.Popen("nohup python3 /content/colab_throughput_probe.py > /content/probe_driver.log 2>&1 &", shell=True)
print("LAUNCHED")
PY
$COL exec -s $S -f /tmp/pk_probe_launch.py --timeout 120 || { echo "launch failed"; exit 1; }
cat > /tmp/pk_probe_poll.py <<'PY'
import os
print("DONE" if os.path.exists("/content/probe/PROBE_DONE") else "RUNNING")
print(open("/content/probe/probe.log").read()[-300:] if os.path.exists("/content/probe/probe.log") else "")
PY
deadline=$(( $(date +%s) + 110*60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 180
  r=$($COL exec -s $S -f /tmp/pk_probe_poll.py --timeout 120 2>&1)
  echo "$(date +%H:%M) $(echo "$r" | grep -E '^(DONE|RUNNING)' | head -1)"
  echo "$r" | grep -q '^DONE' && break
done
$COL download -s $S /content/probe/results.json "$OUT/throughput_probe_results.json" && echo "results -> $OUT"
