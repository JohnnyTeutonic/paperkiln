#!/usr/bin/env bash
# Local side, AGENTS.md path C: one L4 session, stopped on every exit path. Run in WSL from the paperkiln
# checkout's root: bash experiments/mamba_parallel_scan_2026-11-03/drive.sh COMMIT OUT_DIR
# The VM fetches vm_run.sh at COMMIT from GitHub and runs it; this script polls for its DONE marker, then
# downloads one archive of /content/scan_out. Every colab call has a hard timeout; the key is refreshed each poll.
set -u
COMMIT=$1
OUT=$2
COL=~/.local/bin/colab
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
S=claude-scan
mkdir -p "$OUT"
trap 'timeout 120 $COL stop -s $S > /dev/null 2>&1; echo "session $S stopped"' EXIT
timeout 600 $COL new -s $S --gpu L4 || { echo "could not create session"; exit 1; }
EP=$(timeout 120 $COL sessions 2>/dev/null | grep "^\[$S\]" | awk '{print $2}')
echo "endpoint $EP $(date -u +%FT%TZ)"
cat > /tmp/claude_scan_launch.py <<PY
import subprocess, urllib.request
src = urllib.request.urlopen("https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/$COMMIT/experiments/mamba_parallel_scan_2026-11-03/vm_run.sh").read()
open("/content/vm_run.sh", "wb").write(src.replace(b"\r\n", b"\n"))
subprocess.Popen("nohup bash /content/vm_run.sh $COMMIT > /content/vm_run.log 2>&1 &", shell=True)
print("LAUNCHED")
PY
timeout 200 $COL exec -s $S -f /tmp/claude_scan_launch.py --timeout 120 || { echo "launch failed"; exit 1; }
cat > /tmp/claude_scan_poll.py <<'PY'
import os
print("DONE" if os.path.exists("/content/scan_out/DONE") else "RUNNING")
PY
cat > /tmp/claude_scan_pack.py <<'PY'
import subprocess
subprocess.run("cp /content/vm_run.log /content/scan_out/ ; tar czf /content/scan_out.tgz -C /content scan_out", shell=True, check=True)
print("PACKED")
PY
deadline=$(( $(date +%s) + 85*60 ))
state=TIMEOUT
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 120
  timeout 180 $PY tools/colab_adopt.py --refresh > /dev/null 2>&1
  if ! timeout 120 $COL sessions 2>/dev/null | grep -q "^\[$S\]"; then
    [ -n "$EP" ] && timeout 180 $PY tools/colab_adopt.py --adopt "$EP=$S" | tail -1
  fi
  r=$(timeout 200 $COL exec -s $S -f /tmp/claude_scan_poll.py --timeout 120 2>&1 | grep -E '^(DONE|RUNNING)' | head -1)
  echo "$(date +%H:%M) ${r:-no-answer}"
  [ "$r" = DONE ] && { state=DONE; break; }
done
timeout 300 $COL exec -s $S -f /tmp/claude_scan_pack.py --timeout 240 | grep PACKED
timeout 600 $COL download -s $S /content/scan_out.tgz "$OUT/scan_out.tgz" && tar xzf "$OUT/scan_out.tgz" -C "$OUT"
echo "state $state; results in $OUT/scan_out"
pkill -f "[c]olab_adopt.py --adopt $EP" 2>/dev/null
