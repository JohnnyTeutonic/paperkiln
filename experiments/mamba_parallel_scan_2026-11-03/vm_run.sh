#!/usr/bin/env bash
# Runs ON the Colab VM (PROTOCOL.md gates 1-3). Usage: bash vm_run.sh FULL_COMMIT_SHA RUN_ID
# Everything goes under a fresh /content/run_RUN_ID. On every exit path once that directory exists,
# status.json (each gate's exit status, the incoming shell status, the script's final exit status, run id, commit
# and HEAD) is written atomically, then, only if that succeeded, the DONE marker holding the run id. pass is true only if exactly the 13 required gates ran, all 0, and the
# script exits 0. A prerequisite failure (checkout, build, host test, ordered control) stops further paid work.
# The source tree is never modified.
set -u
COMMIT=${1:-}
RUN=${2:-}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
W=${VM_ROOT:-/content}/run_$RUN   # VM_ROOT only for test_vm_run.py
O=$W/out
[ -e "$W" ] && { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O" || exit 2
declare -A G=()
REQUIRED="build checkout configure host_test numpy replay_2 replay_3 replay_unset scan_bench strict_2 strict_3 strict_unset switch_refuses_malformed"
finish() {
    local rc=$? pass=true keys final
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1   # the script's actual exit status, recorded as exit_status
    {
        printf '{"run_id": "%s", "commit": "%s", "head": "%s", "shell_status": %d, "exit_status": %d, "pass": %s, "gates": {' \
            "$RUN" "$COMMIT" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" "$rc" "$final" "$pass"
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
cd "$W"
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; nproc; } > "$O/hardware.log" 2>&1
git clone -q https://github.com/JohnnyTeutonic/paperkiln.git src > "$O/clone.log" 2>&1 || fail checkout $?
git -C src -c advice.detachedHead=false checkout -q --detach "$COMMIT" >> "$O/clone.log" 2>&1 || fail checkout $?
{ [ "$(git -C src rev-parse HEAD)" = "$COMMIT" ] && [ -z "$(git -C src status --porcelain)" ]; } || fail checkout 99
G[checkout]=0
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DMICROTORCH_DEVCHECK=ON \
      -DMICROTORCH_BUILD_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES=89 > "$O/configure.log" 2>&1 || fail configure $?
G[configure]=0
cmake --build build --target test_selective_scan_cuda test_selective_scan_chunked --parallel "$(nproc)" \
      > "$O/build.log" 2>&1 || fail build $?
[ -z "$(git -C src status --porcelain)" ] || fail build 98   # the build wrote nothing into the source tree
G[build]=0
# gate 1: the host test, again here, on the same commit
build/test_selective_scan_chunked > "$O/host_test.json" 2>&1 || fail host_test $?
G[host_test]=0
python3 -m pip install -q numpy==1.26.4 > "$O/numpy_install.log" 2>&1 || fail numpy $?
G[numpy]=0
# gate 2: the frozen 220-case test and its independent replay; the ordered control first (a prerequisite)
for L in unset 2 3; do
    if [ $L = unset ]; then
        env -u MICROTORCH_SCAN_CHUNK MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > "$O/strict_$L.log" 2>&1
    else
        MICROTORCH_SCAN_CHUNK=$L MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > "$O/strict_$L.log" 2>&1
    fi
    G[strict_$L]=$?
    python3 -O src/tools/replay_selective_scan.py "$O/strict_$L.log" --require-cuda > "$O/replay_$L.log" 2>&1
    G[replay_$L]=$?
    if [ $L = unset ] && { [ "${G[strict_unset]}" != 0 ] || [ "${G[replay_unset]}" != 0 ]; }; then exit 1; fi
done
# path selection: the device path parses the switch, so a malformed value must be refused with the parser's message
MICROTORCH_SCAN_CHUNK=" 2" MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > "$O/strict_malformed.log" 2>&1
malformed_rc=$?
if [ $malformed_rc != 0 ] && grep -q "MICROTORCH_SCAN_CHUNK must be a decimal count" "$O/strict_malformed.log"; then
    G[switch_refuses_malformed]=0
else
    G[switch_refuses_malformed]=1
fi
# gate 3 and the timings
nvcc -O3 -std=c++17 -arch=sm_89 -I src/src src/tools/scan_bench.cu -o scan_bench > "$O/bench_build.log" 2>&1 || fail scan_bench $?
./scan_bench 20 > "$O/scan_bench.jsonl" 2> "$O/scan_bench.err"
G[scan_bench]=$?
sha256sum build/test_selective_scan_cuda build/test_selective_scan_chunked scan_bench > "$O/binaries.sha256" 2>&1
[ "${G[scan_bench]}" = 0 ]   # the script's own exit status follows the last gate
