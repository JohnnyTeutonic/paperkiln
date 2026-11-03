#!/usr/bin/env bash
# Runs ON the Colab VM. Usage: bash vm_c2.sh FULL_COMMIT_SHA RUN_ID
# PROTOCOL.md (this folder): coalfire's CUDA build on the C2 checkpoint, against the frozen C2 float64 referee.
# The inputs were uploaded by drive.py to /content/upload_RUN_ID/ and are checked here against inputs.sha256 at the
# commit; nothing is fetched except the two repositories at full SHAs.
# The run directory /content/run_RUN_ID is claimed with an atomic mkdir, so a second launch of the same run exits
# before traps, PID or any work. Once it exists, status.json (each gate's exit status, the incoming shell status, the
# script's final exit status, protocol, run id, commit, HEAD and coalfire's HEAD) is written atomically on every exit
# path, then, only if that succeeded, the DONE marker holding the run id. pass is true only if exactly the required
# gates ran, all 0, and the script exits 0. A failed gate stops the run. No source tree is modified.
# Stop: SIGTERM or SIGINT to the PID published in /content/vm_RUN_ID.pid. Every gate command runs in its own process
# group (setsid) and this shell waits on it; the handler stops that group (TERM every second for up to 60 s, then
# KILL), writes stop.json and exits through the trap that writes status.json and DONE. (As the Mamba packets'
# vm_run.sh, from which the stop handling is copied unchanged.)
set -u
COMMIT=${1:-}
RUN=${2:-}
PROTO=c2cuda
# coalfire main after #38 (the FFN-bias fix #32, the shared GGUF writer #33, the tokenizer #37 and the export guards)
COALFIRE=605685e8e212eecdf86d01630f76b9ac47b5d5a8
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
[ $# -le 2 ] || { echo "usage: vm_c2.sh FULL_COMMIT_SHA RUN_ID" >&2; exit 2; }
case $- in *m*) echo "job control must be off (setsid would fork)" >&2; exit 2;; esac
ROOT=/content
[ "${VM_TEST_HARNESS:-}" = 1 ] && ROOT=${VM_ROOT:-/content}   # test_vm_c2.py only
signal_point() { [ "${VM_TEST_HARNESS:-}" = 1 ] && [ "${VM_TEST_SIGNAL_AT:-}" = "$1" ] && kill -TERM $$; }  # tests
W=$ROOT/run_$RUN
O=$W/out
U=$ROOT/upload_$RUN
mkdir "$W" 2>/dev/null || { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O" || exit 2
declare -A G=()
REQUIRED="build checkout compare configure corpus device_required dump_build dump_cuda dump_cuda_repeat inputs numpy provenance"
finish() {
    local rc=$? pass=true keys final
    trap '' TERM INT   # publication is never interrupted; a stop arriving now finds DONE shortly
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1   # the script's actual exit status, recorded as exit_status
    {
        printf '{"run_id": "%s", "protocol": "%s", "commit": "%s", "head": "%s", "coalfire_head": "%s", "shell_status": %d, "exit_status": %d, "pass": %s, "gates": {' \
            "$RUN" "$PROTO" "$COMMIT" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" \
            "$(git -C "$W/coalfire" rev-parse HEAD 2>/dev/null)" "$rc" "$final" "$pass"
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

# Stop handling, copied from the Mamba packets' vm_run.sh: CPID is the running gate command's PID and process group.
# A stop arriving while a command is being forked and registered is held in STOP_PENDING and delivered right after
# registration. live() sees any non-zombie process whose PID or process group is CPID; a failed or malformed process
# listing counts as live (absence is never assumed).
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
# run GATE LOG CMD...: CMD in its own process group, stdin from /dev/null, output appended to LOG (stderr to
# RUN_ERR instead, when that is set for the call); its exit status is returned (97 if a member of its group outlived
# it and had to be killed). Each call's start and end are appended to steps.log.
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
# the driver stops this run through this PID (SIGTERM; see on_stop); published only once the traps are in place
{ echo $$ > "$ROOT/vm_$RUN.pid.tmp" && mv "$ROOT/vm_$RUN.pid.tmp" "$ROOT/vm_$RUN.pid"; } || exit 2
cd "$W"
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; cmake --version | head -1
  g++ --version | head -1; nproc; python3 --version; free -m; } > "$O/hardware.log" 2>&1
# provenance: exactly one NVIDIA L4, and the image's own toolkit (recorded, not installed)
{ [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1 ] \
  && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L4 \
  && nvcc --version > "$O/nvcc.txt" && ls -d /usr/local/cuda* > "$O/cuda_dirs.txt" 2>&1; } \
  > "$O/provenance.log" 2>&1 || fail provenance 1
G[provenance]=0
# checkout: this packet at COMMIT, and coalfire at COALFIRE, each at HEAD with a clean tree
run checkout "$O/clone.log" git clone -q https://github.com/JohnnyTeutonic/paperkiln.git src || fail checkout $?
run checkout "$O/clone.log" git -C src -c advice.detachedHead=false checkout -q --detach "$COMMIT" || fail checkout $?
run checkout "$O/clone.log" git clone -q https://github.com/JohnnyTeutonic/coalfire.cpp.git coalfire || fail checkout $?
run checkout "$O/clone.log" git -C coalfire -c advice.detachedHead=false checkout -q --detach "$COALFIRE" \
    || fail checkout $?
{ [ "$(git -C src rev-parse HEAD)" = "$COMMIT" ] && [ -z "$(git -C src status --porcelain)" ] \
  && [ "$(git -C coalfire rev-parse HEAD)" = "$COALFIRE" ] && [ -z "$(git -C coalfire status --porcelain)" ]; } \
  || fail checkout 99
G[checkout]=0
E=src/experiments/c2_coalfire_cuda_2026-11-03
# inputs: the uploaded files are exactly inputs.sha256's (the C2 receipt's after_fix/sha256.txt hashes), and land in
# an HF directory (config.json, model.safetensors) beside the checkpoint, ids and the two CPU logit files
mkdir -p in/hf
{ (cd "$U" && sha256sum -c --strict "$W/$E/inputs.sha256") \
  && [ "$(ls "$U" | LC_ALL=C sort | tr '\n' ' ')" = "$(awk '{print $2}' "$E/inputs.sha256" | LC_ALL=C sort | tr '\n' ' ')" ] \
  && cp "$U/config.json" "$U/model.safetensors" in/hf/ \
  && cp "$U/pol_checkpoint_0.ckpt" "$U/ids.txt" "$U/coalfire.f32" "$U/paperkiln.f32" in/; } \
  > "$O/inputs.log" 2>&1 || fail inputs 1
G[inputs]=0
# corpus: the tokenizer's two vocabulary files, made from coalfire's data/training_pairs.txt as C2's run.sh made them
n=$(wc -l < coalfire/data/training_pairs.txt)
{ head -n $(( n / 2 )) coalfire/data/training_pairs.txt | tr '|' ' ' > in/train-0-small.txt \
  && tail -n +$(( n / 2 + 1 )) coalfire/data/training_pairs.txt | tr '|' ' ' > in/train-1-small.txt \
  && (cd in && sha256sum -c --strict "$W/$E/corpus.sha256"); } > "$O/corpus.log" 2>&1 || fail corpus 1
G[corpus]=0
run configure "$O/configure.log" cmake -S coalfire -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON || fail configure $?
G[configure]=0
run build "$O/build.log" cmake --build build --target transformer_core --parallel "$(nproc)" || fail build $?
{ [ -z "$(git -C coalfire status --porcelain)" ] && [ -z "$(git -C src status --porcelain)" ]; } || fail build 98
G[build]=0
# the dump tool, with exactly the flags the library's src/transformer.cpp was compiled with (a CUDA build: USE_CUDA)
python3 "$E/dump_flags.py" build/compile_commands.json src/transformer.cpp > "$O/dump_flags.txt" 2>&1 \
    || fail dump_build 1
grep -q -- '-DUSE_CUDA' "$O/dump_flags.txt" || fail dump_build 95
read -r -a FL < "$O/dump_flags.txt"   # the VM's paths have no spaces or shell characters, so words split cleanly
CUDA_LIB=$(dirname "$(dirname "$(command -v nvcc)")")/lib64
run dump_build "$O/dump_build.log" g++ "${FL[@]}" "$E/coalfire_dump.cpp" build/libtransformer_core.a \
    -L"$CUDA_LIB" -Wl,-rpath,"$CUDA_LIB" -lcudart -lcublas -fopenmp -o coalfire_dump || fail dump_build $?
G[dump_build]=0
ARGS=(in/pol_checkpoint_0.ckpt in/train-0-small.txt in/train-1-small.txt 64 in/ids.txt)
# every dump runs with TF32 forced off (coalfire's forward calls cublasSgemm in the default math mode, which is FP32
# already; the override makes that independent of the library's defaults)
# negative control: with no device visible, the CUDA build refuses (exit 3) instead of running
run device_required "$O/dump_nodevice.log" env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES= ./coalfire_dump \
    "${ARGS[@]}" "$O/nodevice.f32"
rc=$?
{ [ "$rc" = 3 ] && grep -q '"use_cuda": true, "devices": 0' "$O/dump_nodevice.log" && [ ! -s "$O/nodevice.f32" ]; } \
    || fail device_required 1
G[device_required]=0
# the CUDA leg, twice; each reports the device it ran on
for k in dump_cuda dump_cuda_repeat; do
    f=$O/coalfire_cuda.f32
    [ $k = dump_cuda_repeat ] && f=$O/coalfire_cuda_repeat.f32
    run $k "$O/$k.log" env NVIDIA_TF32_OVERRIDE=0 ./coalfire_dump "${ARGS[@]}" "$f" || fail $k $?
    grep -q '"use_cuda": true, "devices": 1, "device": "NVIDIA L4", "cc": "8.9"' "$O/$k.log" || fail $k 94
    G[$k]=0
done
# the referee needs NumPy; the image's own, recorded (nothing is installed)
run numpy "$O/numpy.txt" python3 -c "import numpy; print(numpy.__version__)" || fail numpy $?
G[numpy]=0
RUN_ERR="$O/compare.err" run compare "$O/compare.log" python3 "$E/compare_cuda.py" in/hf in/ids.txt \
    "$O/coalfire_cuda.f32" "$O/coalfire_cuda_repeat.f32" in/coalfire.f32 in/paperkiln.f32 "$O/result.json"
G[compare]=$?
sha256sum coalfire_dump build/libtransformer_core.a "$O/coalfire_cuda.f32" "$O/coalfire_cuda_repeat.f32" \
    > "$O/binaries.sha256" 2>&1
[ "${G[compare]}" = 0 ]   # the script's own exit status follows the last gate
