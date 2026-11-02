#!/usr/bin/env bash
# Runs ON the Colab VM (PROTOCOL.md gates 1-3). Usage: bash vm_run.sh FULL_COMMIT_SHA RUN_ID
# Everything goes under a fresh /content/run_RUN_ID. status.json is written before the DONE marker on every exit
# path; DONE holds the run id. A prerequisite failure (checkout, build, CPU gate, ordered control) stops further
# paid work.
set -u
COMMIT=$1
RUN=$2
W=/content/run_$RUN
O=$W/out
declare -A G   # gate -> exit status
finish() {
    local pass=true
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ ${#G[@]} -eq 13 ] || pass=false   # all 13 gates below must have run
    {
        printf '{"run_id": "%s", "commit": "%s", "head": "%s", "pass": %s, "gates": {' "$RUN" "$COMMIT" \
            "$(git -C $W/src rev-parse HEAD 2>/dev/null)" "$pass"
        local first=1
        for k in $(printf '%s\n' "${!G[@]}" | sort); do
            [ $first = 1 ] || printf ', '
            first=0
            printf '"%s": %s' "$k" "${G[$k]}"
        done
        printf '}}\n'
    } > $O/status.json
    echo "$RUN" > $O/DONE
}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[ -e "$W" ] && { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p $O
trap finish EXIT
fail() { G[$1]=${2:-1}; exit 1; }
cd $W
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; nproc; } > $O/hardware.log 2>&1
git clone -q https://github.com/JohnnyTeutonic/paperkiln.git src > $O/clone.log 2>&1 || fail clone
git -C src -c advice.detachedHead=false checkout -q --detach "$COMMIT" >> $O/clone.log 2>&1 || fail checkout
[ "$(git -C src rev-parse HEAD)" = "$COMMIT" ] && [ -z "$(git -C src status --porcelain)" ] || fail checkout
G[checkout]=0
find src -name "*.sh" -exec sed -i "s/\r$//" {} +
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DMICROTORCH_DEVCHECK=ON \
      -DMICROTORCH_BUILD_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES=89 > $O/configure.log 2>&1 || fail configure
G[configure]=0
cmake --build build --target test_selective_scan_cuda test_selective_scan_chunked --parallel "$(nproc)" \
      > $O/build.log 2>&1 || fail build
G[build]=0
# gate 1: the host test, again here, on the same commit
build/test_selective_scan_chunked > $O/host_test.json 2>&1 || fail host_test
G[host_test]=0
python3 -m pip install -q numpy==1.26.4 > $O/numpy_install.log 2>&1 || fail numpy
G[numpy]=0
# gate 2: the frozen 220-case test and its independent replay; the ordered control first (a prerequisite)
for L in unset 2 3; do
    if [ $L = unset ]; then env -u MICROTORCH_SCAN_CHUNK MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > $O/strict_$L.log 2>&1
    else MICROTORCH_SCAN_CHUNK=$L MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > $O/strict_$L.log 2>&1; fi
    G[strict_$L]=$?
    python3 -O src/tools/replay_selective_scan.py $O/strict_$L.log --require-cuda > $O/replay_$L.log 2>&1
    G[replay_$L]=$?
    if [ $L = unset ] && { [ ${G[strict_unset]} != 0 ] || [ ${G[replay_unset]} != 0 ]; }; then exit 1; fi
done
# path selection: the device path reads the switch, so a malformed value must be refused (nonzero exit)
MICROTORCH_SCAN_CHUNK=" 2" MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > $O/strict_malformed.log 2>&1
[ $? != 0 ] && G[switch_refuses_malformed]=0 || G[switch_refuses_malformed]=1
# gate 3 and the timings
nvcc -O3 -std=c++17 -arch=sm_89 -I src/src src/tools/scan_bench.cu -o scan_bench > $O/bench_build.log 2>&1 || fail bench_build
./scan_bench 20 > $O/scan_bench.jsonl 2> $O/scan_bench.err
G[scan_bench]=$?
sha256sum build/test_selective_scan_cuda build/test_selective_scan_chunked scan_bench > $O/binaries.sha256
