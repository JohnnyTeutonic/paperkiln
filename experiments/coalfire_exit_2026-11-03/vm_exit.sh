#!/usr/bin/env bash
# Runs ON the Colab VM. Usage: bash vm_exit.sh FULL_COMMIT_SHA RUN_ID
# PROTOCOL.md (this folder): does coalfire's CUDA build exit cleanly after a forward, through its normal destructors,
# before and after the LanguageModelHead member fix? The C2 checkpoint and ids are the workload; the inputs were
# uploaded by drive.py to /content/upload_RUN_ID/ and are checked against inputs.sha256 at the commit. Nothing is
# fetched except the two repositories at full SHAs.
# The run directory, status.json/DONE publication and stop handling are vm_c2.sh's (the C2 CUDA leg), unchanged:
# /content/run_RUN_ID is claimed with an atomic mkdir; status.json (each gate's exit status, the incoming shell
# status, the final exit status, protocol, run id, commit, HEAD and both coalfire HEADs) is written atomically on
# every exit path, then DONE; pass is true only if exactly the required gates ran, all 0, and the script exits 0.
# Stop: SIGTERM or SIGINT to the PID in /content/vm_RUN_ID.pid; every gate command runs in its own process group.
set -u
COMMIT=${1:-}
RUN=${2:-}
PROTO=cfexit
# coalfire main after #39 (the config initialisers), and the LanguageModelHead member fix on top of it
BASE=642de84c85669938ae66b067fa3823629016f1fb
FIX=72b1f7509f731f7c06346e73367bbd1e94300cbb
# the C2 CUDA leg's logits (runs c1 and c2, sha256 of 48 x 2,275 float32); this check's forwards must reproduce them
WANT=de52befb3ce4241c3493096768111cec1ccdafeaf712d993a15ea24519c92e24
REPS=10
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
[ $# -le 2 ] || { echo "usage: vm_exit.sh FULL_COMMIT_SHA RUN_ID" >&2; exit 2; }
case $- in *m*) echo "job control must be off (setsid would fork)" >&2; exit 2;; esac
ROOT=/content
[ "${VM_TEST_HARNESS:-}" = 1 ] && ROOT=${VM_ROOT:-/content}   # test_vm_exit.py only
TIMEOUT=120
if [ "${VM_TEST_HARNESS:-}" = 1 ]; then   # test_vm_exit.py only: stub logits and short runs
    [ -n "${VM_TEST_REPS:-}" ] && REPS=$VM_TEST_REPS
    [ -n "${VM_TEST_WANT:-}" ] && WANT=$VM_TEST_WANT
    [ -n "${VM_TEST_TIMEOUT:-}" ] && TIMEOUT=$VM_TEST_TIMEOUT
fi
signal_point() { [ "${VM_TEST_HARNESS:-}" = 1 ] && [ "${VM_TEST_SIGNAL_AT:-}" = "$1" ] && kill -TERM $$; }  # tests
W=$ROOT/run_$RUN
O=$W/out
U=$ROOT/upload_$RUN
mkdir "$W" 2>/dev/null || { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O" || exit 2
declare -A G=()
REQUIRED="build checkout configure corpus device_required dump_build exit_base exit_fix inputs provenance"
finish() {
    local rc=$? pass=true keys final
    trap '' TERM INT   # publication is never interrupted; a stop arriving now finds DONE shortly
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1
    {
        printf '{"run_id": "%s", "protocol": "%s", "commit": "%s", "head": "%s", "coalfire_base_head": "%s", "coalfire_fix_head": "%s", "shell_status": %d, "exit_status": %d, "pass": %s, "gates": {' \
            "$RUN" "$PROTO" "$COMMIT" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" \
            "$(git -C "$W/cf_base" rev-parse HEAD 2>/dev/null)" "$(git -C "$W/cf_fix" rev-parse HEAD 2>/dev/null)" \
            "$rc" "$final" "$pass"
        local first=1
        for k in $(printf '%s\n' "${!G[@]}" | LC_ALL=C sort); do
            [ $first = 1 ] || printf ', '
            first=0
            printf '"%s": %d' "$k" "${G[$k]}"
        done
        printf '}}\n'
    } > "$O/status.json.tmp" && mv "$O/status.json.tmp" "$O/status.json" || exit 1   # no DONE without a status
    echo "$RUN" > "$O/DONE.tmp" && mv "$O/DONE.tmp" "$O/DONE" || exit 1
    exit $final
}
trap finish EXIT
fail() { G[$1]=$2; exit 1; }   # fail GATE STATUS, with the failed command's own status

# Stop handling, copied from vm_c2.sh unchanged.
CPID=""
STEP=""
STOP_PENDING=""
live() {
    [ -n "$CPID" ] || return 1
    local t
    t=$(ps -eo pid=,pgid=,stat=) && [ -n "$t" ] || return 0
    awk -v c="$CPID" 'NF != 3 || $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ {bad = 1}
        ($1 == c || $2 == c) && $3 !~ /^Z/ {f = 1} END {exit (f || bad) ? 0 : 1}' <<< "$t"
    [ $? -ne 1 ]
}
on_stop() {
    trap '' TERM INT
    local gone=true started=false
    if [ -n "$CPID" ]; then
        started=true
        for _ in $(seq 60); do
            live || break
            kill -TERM -- -"$CPID" "$CPID" 2>/dev/null
            sleep 1
        done
        if live; then kill -KILL -- -"$CPID" "$CPID" 2>/dev/null; sleep 2; fi
        live && { gone=false; : > "$O/QUIESCENCE_UNCONFIRMED"; }
        [ -n "$STEP" ] && G[$STEP]=143
    fi
    pkill -TERM -P $$ 2>/dev/null
    printf '{"stopped_at": "%s", "step": "%s", "step_started": %s, "step_group_gone": %s}\n' \
        "$(date -u +%FT%TZ)" "$STEP" "$started" "$gone" > "$O/stop.json"
    exit 1
}
trap on_stop TERM INT
run() {
    local rc
    STEP=$1
    echo "$(date -u +%FT%TZ) start $1" >> "$O/steps.log"
    trap 'STOP_PENDING=1' TERM INT
    setsid "${@:3}" >> "$2" 2>> "${RUN_ERR:-$2}" < /dev/null &
    CPID=$!
    trap on_stop TERM INT
    [ -n "$STOP_PENDING" ] && on_stop
    signal_point "run_$STEP"
    wait "$CPID"
    rc=$?
    if live; then
        kill -KILL -- -"$CPID" 2>/dev/null; sleep 2
        live && : > "$O/QUIESCENCE_UNCONFIRMED"
        rc=97
    fi
    echo "$(date -u +%FT%TZ) end $1 $rc" >> "$O/steps.log"
    CPID=""
    STEP=""
    return $rc
}
{ echo $$ > "$ROOT/vm_$RUN.pid.tmp" && mv "$ROOT/vm_$RUN.pid.tmp" "$ROOT/vm_$RUN.pid"; } || exit 2
cd "$W"
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; cmake --version | head -1
  g++ --version | head -1; nproc; python3 --version; free -m; } > "$O/hardware.log" 2>&1
{ [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1 ] \
  && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L4 \
  && nvcc --version > "$O/nvcc.txt" && ls -d /usr/local/cuda* > "$O/cuda_dirs.txt" 2>&1; } \
  > "$O/provenance.log" 2>&1 || fail provenance 1
G[provenance]=0
# checkout: this packet at COMMIT; coalfire at BASE and at FIX, each verified at HEAD with a clean tree
run checkout "$O/clone.log" git clone -q https://github.com/JohnnyTeutonic/paperkiln.git src || fail checkout $?
run checkout "$O/clone.log" git -C src -c advice.detachedHead=false checkout -q --detach "$COMMIT" || fail checkout $?
run checkout "$O/clone.log" git clone -q https://github.com/JohnnyTeutonic/coalfire.cpp.git cf_base || fail checkout $?
run checkout "$O/clone.log" git -C cf_base -c advice.detachedHead=false checkout -q --detach "$BASE" || fail checkout $?
run checkout "$O/clone.log" git clone -q https://github.com/JohnnyTeutonic/coalfire.cpp.git cf_fix || fail checkout $?
run checkout "$O/clone.log" git -C cf_fix -c advice.detachedHead=false checkout -q --detach "$FIX" || fail checkout $?
{ [ "$(git -C src rev-parse HEAD)" = "$COMMIT" ] && [ -z "$(git -C src status --porcelain)" ] \
  && [ "$(git -C cf_base rev-parse HEAD)" = "$BASE" ] && [ -z "$(git -C cf_base status --porcelain)" ] \
  && [ "$(git -C cf_fix rev-parse HEAD)" = "$FIX" ] && [ -z "$(git -C cf_fix status --porcelain)" ]; } \
  || fail checkout 99
G[checkout]=0
E=src/experiments/coalfire_exit_2026-11-03
mkdir -p in/hf
{ (cd "$U" && sha256sum -c --strict "$W/$E/inputs.sha256") \
  && [ "$(ls "$U" | LC_ALL=C sort | tr '\n' ' ')" = "$(awk '{print $2}' "$E/inputs.sha256" | LC_ALL=C sort | tr '\n' ' ')" ] \
  && cp "$U/pol_checkpoint_0.ckpt" "$U/ids.txt" in/; } > "$O/inputs.log" 2>&1 || fail inputs 1
G[inputs]=0
n=$(wc -l < cf_base/data/training_pairs.txt)
{ head -n $(( n / 2 )) cf_base/data/training_pairs.txt | tr '|' ' ' > in/train-0-small.txt \
  && tail -n +$(( n / 2 + 1 )) cf_base/data/training_pairs.txt | tr '|' ' ' > in/train-1-small.txt \
  && (cd in && sha256sum -c --strict "$W/$E/corpus.sha256"); } > "$O/corpus.log" 2>&1 || fail corpus 1
G[corpus]=0
for t in base fix; do
    run configure "$O/configure_$t.log" cmake -S cf_$t -B build_$t -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON || fail configure $?
done
G[configure]=0
for t in base fix; do
    run build "$O/build_$t.log" cmake --build build_$t --target transformer_core --parallel "$(nproc)" || fail build $?
done
{ [ -z "$(git -C cf_base status --porcelain)" ] && [ -z "$(git -C cf_fix status --porcelain)" ] \
  && [ -z "$(git -C src status --porcelain)" ]; } || fail build 98
G[build]=0
CUDA_LIB=$(dirname "$(dirname "$(command -v nvcc)")")/lib64
{ echo "nvcc $(command -v nvcc) -> $(readlink -f "$(command -v nvcc)")"
  echo "g++ $(command -v g++) -> $(readlink -f "$(command -v g++)")"
  echo "cuda_lib $CUDA_LIB -> $(readlink -f "$CUDA_LIB")"; } > "$O/toolchain.txt"
for t in base fix; do
    python3 "$E/dump_flags.py" build_$t/compile_commands.json src/transformer.cpp > "$O/dump_flags_$t.txt" 2>&1 \
        || fail dump_build 1
    grep -qx -- '-DUSE_CUDA' "$O/dump_flags_$t.txt" || fail dump_build 95
    cp build_$t/compile_commands.json "$O/compile_commands_$t.json"
    mapfile -t FL < "$O/dump_flags_$t.txt"
    run dump_build "$O/dump_build_$t.log" g++ "${FL[@]}" "$E/coalfire_dump.cpp" build_$t/libtransformer_core.a \
        -L"$CUDA_LIB" -Wl,-rpath,"$CUDA_LIB" -lcudart -lcublas -fopenmp -o dump_$t || fail dump_build $?
    ldd dump_$t > "$O/ldd_$t.txt" 2>&1
done
G[dump_build]=0
ARGS=(in/pol_checkpoint_0.ckpt in/train-0-small.txt in/train-1-small.txt 64 in/ids.txt)
run device_required "$O/dump_nodevice.log" env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES= ./dump_fix \
    "${ARGS[@]}" "$O/nodevice.f32"
rc=$?
{ [ "$rc" = 3 ] && grep -q '"use_cuda": true, "devices": 0' "$O/dump_nodevice.log" && [ ! -s "$O/nodevice.f32" ]; } \
    || fail device_required 1
G[device_required]=0
# REPS forwards per build with the normal exit (every destructor and atexit handler runs), each in its own process
# with a TIMEOUT (120 s) bound. Each attempt appends one JSON line to exit_TAG.jsonl: its exit status, whether the device line,
# the written marker and the normal-exit marker appeared, and the logits' sha256.
attempt() {   # attempt TAG I
    local t=$1 i=$2 log=$O/exit_$1_$2.log f=$W/logits_$1_$2.f32 rc dev wr nx to sha
    # --verbose makes timeout say when it sends the signal, so a timeout is told apart from a crash
    run exit_$t "$log" timeout --verbose -s KILL "$TIMEOUT" env NVIDIA_TF32_OVERRIDE=0 COALFIRE_DUMP_NORMAL_EXIT=1 \
        ./dump_$t "${ARGS[@]}" "$f"
    rc=$?
    grep -q '"use_cuda": true, "devices": 1, "device": "NVIDIA L4", "cc": "8.9"' "$log" && dev=true || dev=false
    grep -qx '{"logits_written": true}' "$log" && wr=true || wr=false
    grep -qx '{"exit": "normal"}' "$log" && nx=true || nx=false
    grep -q '^timeout: sending signal' "$log" && to=true || to=false
    sha=$( [ -s "$f" ] && sha256sum "$f" | cut -d' ' -f1 )
    # a mismatching output is kept for diagnosis (a matching one is the C2 file, by its hash)
    [ -s "$f" ] && [ "$sha" != "$WANT" ] && cp "$f" "$O/logits_${t}_$i.f32"
    printf '{"build": "%s", "rep": %d, "exit": %d, "timed_out": %s, "device": %s, "written": %s, "normal_exit_reached": %s, "sha256": "%s", "matches_c2": %s}\n' \
        "$t" "$i" "$rc" "$to" "$dev" "$wr" "$nx" "$sha" "$( [ "$sha" = "$WANT" ] && echo true || echo false )" \
        >> "$O/exit_$t.jsonl"
}
# base first (recorded: its crashes are the defect being measured, not a failure of this run), then fix (gated)
for i in $(seq "$REPS"); do attempt base "$i"; done
[ "$(wc -l < "$O/exit_base.jsonl")" = "$REPS" ] || fail exit_base 1
G[exit_base]=0
for i in $(seq "$REPS"); do attempt fix "$i"; done
ok=$(grep -c '"exit": 0, "timed_out": false, "device": true, "written": true, "normal_exit_reached": true, "sha256": "[0-9a-f]*", "matches_c2": true' \
     "$O/exit_fix.jsonl")
# summary.json, per build: attempts; exit 0; nonzero; of the nonzero, timeouts, failures after the normal-exit marker
# that were not timeouts (teardown failures) and failures before it (the attempt never reached teardown); and how
# many attempts showed the device, the written and normal-exit markers and the C2 logits
python3 - "$O" > "$O/summary.json" <<'PY' || fail exit_fix 92
import json, sys
o = sys.argv[1]
s = {}
for t in ("base", "fix"):
    rows = [json.loads(l) for l in open(f"{o}/exit_{t}.jsonl")]
    bad = [r for r in rows if r["exit"] != 0]
    s[t] = {"attempts": len(rows), "exit_0": len(rows) - len(bad), "nonzero": len(bad),
            "timeouts": sum(r["timed_out"] for r in bad),
            "teardown_failures": sum(r["normal_exit_reached"] and not r["timed_out"] for r in bad),
            "failures_before_teardown": sum(not r["normal_exit_reached"] and not r["timed_out"] for r in bad),
            "exit_codes": sorted({r["exit"] for r in rows}),
            "device": sum(r["device"] for r in rows), "written": sum(r["written"] for r in rows),
            "normal_exit_reached": sum(r["normal_exit_reached"] for r in rows),
            "logits_match_c2": sum(r["matches_c2"] for r in rows)}
print(json.dumps(s))
PY
sha256sum dump_base dump_fix build_base/libtransformer_core.a build_fix/libtransformer_core.a > "$O/binaries.sha256" \
    || fail exit_fix 91
[ "$ok" = "$REPS" ] || fail exit_fix 1
G[exit_fix]=0
