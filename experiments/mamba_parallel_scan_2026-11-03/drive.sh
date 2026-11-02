#!/usr/bin/env bash
# Local side, microtorch/AGENTS.md path C: one L4 session for one run. Run in a foreground WSL process from the
# paperkiln checkout's root:
#   bash experiments/mamba_parallel_scan_2026-11-03/drive.sh FULL_COMMIT_SHA RUN_ID
# Outputs: /mnt/c/ml_artifacts/mamba_parallel_scan_20261103/RUN_ID/ (driver.log, scan_out.tgz, run_RUN_ID/out/),
# with the receipts copied into this experiment's receipts/RUN_ID/ before any analysis.
# Absolute deadline: 90 minutes from start, 15 of them reserved for retrieval and shutdown; every colab call has
# a hard timeout capped by the time left. Cleanup verifies the session is gone and says so explicitly.
set -u
COMMIT=${1:-}
RUN=${2:-}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "usage: drive.sh FULL_COMMIT_SHA RUN_ID" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "RUN_ID: letters, digits, - and _ only" >&2; exit 2; }
COL=~/.local/bin/colab
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
S=claude-scan
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/mnt/c/ml_artifacts/mamba_parallel_scan_20261103/$RUN
RECEIPTS=$HERE/receipts/$RUN
[ -e "$OUT" ] && { echo "$OUT exists: one directory per run" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
START=$(date +%s)
DEADLINE=$(( START + 90 * 60 ))
POLL_END=$(( DEADLINE - 15 * 60 ))
left() { local l=$(( DEADLINE - $(date +%s) )); [ $l -gt 0 ] && echo $l || echo 1; }
cap() { local want=$1 l; l=$(left); [ $want -lt $l ] && echo $want || echo $l; }
EP=""
CREATED=0

inventory() {
    echo "== inventory $(date -u +%FT%TZ)"
    timeout 120 $COL sessions 2>&1 | sed 's/^/  sessions: /'
    timeout 300 python3 tools/colab_reap_orphans.py 2>&1 | sed 's/^/  orphans: /'
}
cleanup() {
    trap - EXIT INT TERM HUP
    if [ $CREATED = 1 ]; then
        local gone=0
        for attempt in 1 2 3; do
            timeout 120 $COL stop -s $S > /dev/null 2>&1
            sleep 10
            if ! timeout 120 $COL sessions 2>/dev/null | grep -q -e "^\[$S\]" ${EP:+-e "$EP"}; then gone=1; break; fi
            # the CLI may have lost the name: adopt the known endpoint and stop it again
            [ -n "$EP" ] && timeout 180 $PY tools/colab_adopt.py --adopt "$EP=$S" > /dev/null 2>&1
        done
        if [ $gone = 1 ]; then echo "CLEANUP OK: session $S ($EP) no longer listed"
        else echo "CLEANUP UNRESOLVED: session $S ($EP) still listed; stop it by hand: $COL stop -s $S"; fi
    fi
    inventory
    pkill -f "[c]olab_adopt.py --adopt $EP" 2>/dev/null
}
trap cleanup EXIT
trap 'echo "signal received"; exit 130' INT TERM HUP

echo "run $RUN commit $COMMIT start $(date -u +%FT%TZ) deadline $(date -u -d @$DEADLINE +%FT%TZ)"
# preflight (AGENTS.md section 2): what is running, orphans, and no session already holding our name
inventory
if timeout 120 $COL sessions 2>/dev/null | grep -q "^\[$S\]"; then
    echo "a session named $S already exists: not ours to touch; refusing"; exit 1
fi
timeout "$(cap 600)" $COL new -s $S --gpu L4 || { echo "could not create session"; exit 1; }
CREATED=1
EP=$(timeout 120 $COL sessions 2>/dev/null | grep "^\[$S\]" | awk '{print $2}')
echo "endpoint ${EP:-unknown} $(date -u +%FT%TZ)"
cat > /tmp/claude_scan_launch_$RUN.py <<PY
import os, subprocess, urllib.request
src = urllib.request.urlopen("https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/$COMMIT/experiments/mamba_parallel_scan_2026-11-03/vm_run.sh").read()
open("/content/vm_run_$RUN.sh", "wb").write(src.replace(b"\r\n", b"\n"))
subprocess.Popen("nohup bash /content/vm_run_$RUN.sh $COMMIT $RUN > /content/vm_run_$RUN.log 2>&1 &", shell=True)
print("LAUNCHED")
PY
timeout "$(cap 200)" $COL exec -s $S -f /tmp/claude_scan_launch_$RUN.py --timeout 120 | grep -q LAUNCHED \
    || { echo "launch failed"; exit 1; }
cat > /tmp/claude_scan_poll_$RUN.py <<PY
import os
p = "/content/run_$RUN/out/DONE"
print("DONE" if os.path.exists(p) and open(p).read().strip() == "$RUN" else "RUNNING")
PY
cat > /tmp/claude_scan_pack_$RUN.py <<PY
import subprocess
subprocess.run("cp /content/vm_run_$RUN.log /content/run_$RUN/out/ 2>/dev/null; tar czf /content/scan_$RUN.tgz -C /content run_$RUN/out", shell=True, check=True)
print("PACKED")
PY
state=DEADLINE
while [ "$(date +%s)" -lt "$POLL_END" ]; do
    sleep "$(cap 120)"
    timeout "$(cap 180)" $PY tools/colab_adopt.py --refresh > /dev/null 2>&1
    if ! timeout "$(cap 120)" $COL sessions 2>/dev/null | grep -q "^\[$S\]"; then
        [ -n "$EP" ] && timeout "$(cap 180)" $PY tools/colab_adopt.py --adopt "$EP=$S" | tail -1
    fi
    r=$(timeout "$(cap 200)" $COL exec -s $S -f /tmp/claude_scan_poll_$RUN.py --timeout 120 2>&1 | grep -E '^(DONE|RUNNING)' | head -1)
    echo "$(date +%H:%M) ${r:-no-answer}"
    [ "$r" = DONE ] && { state=DONE; break; }
done
echo "polling ended: $state"
# retrieval, also after the deadline (partial outputs are evidence)
ok=1
timeout "$(cap 300)" $COL exec -s $S -f /tmp/claude_scan_pack_$RUN.py --timeout 240 | grep -q PACKED || { echo "PACK FAILED"; ok=0; }
if [ $ok = 1 ]; then
    timeout "$(cap 600)" $COL download -s $S /content/scan_$RUN.tgz "$OUT/scan_out.tgz" || { echo "DOWNLOAD FAILED"; ok=0; }
fi
if [ $ok = 1 ]; then
    tar xzf "$OUT/scan_out.tgz" -C "$OUT" || { echo "EXTRACT FAILED"; ok=0; }
fi
if [ $ok = 1 ]; then
    mkdir -p "$RECEIPTS" && cp -r "$OUT/run_$RUN/out/." "$RECEIPTS/" && cp "$OUT/driver.log" "$RECEIPTS/" \
        && echo "receipts copied to $RECEIPTS"
    python3 - "$OUT/run_$RUN/out/status.json" "$RUN" "$COMMIT" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print("STATUS MISSING OR UNREADABLE:", e); sys.exit(0)
match = s.get("run_id") == sys.argv[2] and s.get("commit") == sys.argv[3] and s.get("head") == sys.argv[3]
print("status: run/commit match" if match else "STATUS MISMATCH", "| pass:", s.get("pass"), "| gates:", s.get("gates"))
PY
fi
echo "state $state; retrieval $([ $ok = 1 ] && echo ok || echo FAILED); outputs in $OUT"
