#!/usr/bin/env bash
# Runs ON the Colab VM. Usage: bash vm_run.sh FULL_COMMIT_SHA RUN_ID [forward|backward|prefill]
#   forward  (the default): PROTOCOL.md gates 1-3, the gate set of run r1-20261103;
#   backward: PROTOCOL_BACKWARD.md gates 1-6;
#   prefill:  PROTOCOL_PREFILL.md, the end-to-end prefill gates and timings.
# The run directory /content/run_RUN_ID is claimed with an atomic mkdir, so a second launch of the same run (the
# driver retries a launch only after its probe found no script; a delayed first call may still arrive) exits before
# traps, PID or any work. Once it exists, status.json (each gate's exit status, the incoming shell status, the
# script's final exit status, protocol, run id, commit and HEAD) is written atomically on every exit path, then, only
# if that succeeded, the DONE marker holding the run id. pass is true only if exactly the protocol's required gates
# ran, all 0, and the script exits 0. A prerequisite failure stops further paid work. The source tree is never
# modified.
# Stop: SIGTERM or SIGINT to the PID published in /content/vm_RUN_ID.pid. Every gate command runs in its own process
# group (setsid) and this shell waits on it, so a stop is handled at once; the handler stops that group (TERM every
# second for up to 60 s, then KILL), writes stop.json and exits through the trap that writes status.json and DONE.
set -u
COMMIT=${1:-}
RUN=${2:-}
PROTO=${3:-forward}
# the backward protocol's base: paperkiln master when the branch was cut; its default path is the reference for gate 5
BASE=d0390e392006a383f8be4565b1b67fcfef513ff6
# the prefill protocol's base: master with both chunked kernels merged (PR #15); the engine and the model are its own
PREFILL_BASE=da5febf2f0a4578bbf17384123c8962e8c623bf5
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
case "$PROTO" in forward|backward|prefill) ;; *) echo "protocol must be forward, backward or prefill" >&2; exit 2;; esac
case $- in *m*) echo "job control must be off (setsid would fork)" >&2; exit 2;; esac
ROOT=/content
[ "${VM_TEST_HARNESS:-}" = 1 ] && ROOT=${VM_ROOT:-/content}   # test_vm_run.py only
signal_point() { [ "${VM_TEST_HARNESS:-}" = 1 ] && [ "${VM_TEST_SIGNAL_AT:-}" = "$1" ] && kill -TERM $$; }  # tests
W=$ROOT/run_$RUN
O=$W/out
mkdir "$W" 2>/dev/null || { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O" || exit 2
declare -A G=()
if [ "$PROTO" = forward ]; then
    REQUIRED="build checkout configure host_test numpy replay_2 replay_3 replay_unset scan_bench strict_2 strict_3 strict_unset switch_refuses_malformed"
elif [ "$PROTO" = backward ]; then
    REQUIRED="build build_base checkout configure default_identical host_test host_test_bwd numpy provenance replay_2 replay_3 replay_unset scan_bwd_bench strict_2 strict_3 strict_unset switch_refuses_malformed"
else
    REQUIRED="build checkout configure engine_unchanged host_test prefill_gate prefill_gate_cublas prefill_profile prefill_profile_cublas prefill_time prefill_time_cublas provenance resources"
fi
finish() {
    local rc=$? pass=true keys final
    trap '' TERM INT   # publication is never interrupted; a stop arriving now finds DONE shortly
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1   # the script's actual exit status, recorded as exit_status
    {
        printf '{"run_id": "%s", "protocol": "%s", "commit": "%s", "head": "%s", "shell_status": %d, "exit_status": %d, "pass": %s, "gates": {' \
            "$RUN" "$PROTO" "$COMMIT" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" "$rc" "$final" "$pass"
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

# Stop handling (as the Gemma packet's vm_gemma.sh, for a sequence of gate commands): CPID is the running gate
# command's PID and process group. A stop arriving while a command is being forked and registered is held in
# STOP_PENDING and delivered right after registration. live() sees any non-zombie process whose PID or process
# group is CPID; a failed or malformed process listing counts as live (absence is never assumed).
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
# it and had to be killed)
run() {
    local rc
    STEP=$1
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
    CPID=""
    STEP=""
    return $rc
}
# the driver stops this run through this PID (SIGTERM; see on_stop); published only once the traps are in place
{ echo $$ > "$ROOT/vm_$RUN.pid.tmp" && mv "$ROOT/vm_$RUN.pid.tmp" "$ROOT/vm_$RUN.pid"; } || exit 2
cd "$W"
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; nproc; python3 --version; } > "$O/hardware.log" 2>&1
if [ "$PROTO" != forward ]; then
    # exactly one NVIDIA L4, and the image's toolkit (CUDA 13.0 on 3 Nov; r1-20261103 built this repository with it)
    { [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1 ] \
      && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L4 \
      && nvcc --version > "$O/nvcc.txt" && ls -d /usr/local/cuda* > "$O/cuda_dirs.txt" 2>&1; } \
      > "$O/provenance.log" 2>&1 || fail provenance 1
    G[provenance]=0
fi
JOBS=$(nproc)
if [ "$PROTO" = prefill ]; then
    # Resources (PROTOCOL_PREFILL.md gate 2b): host memory available to this run and device memory free, recorded at
    # the start and again just before the correctness gate. Below NEED_KIB on the host or NEED_DEV_MIB on the device,
    # or host headroom that cannot be established, is an operational stop before any heavyweight step. The gate's
    # batched d768x24 case holds ~7.4 GB of live result objects (two recorded runs and a live logits matrix) plus the
    # models, parameters and allocator retention: ~9 GB estimated, unmeasured; 16 GiB leaves headroom. The device
    # estimate is ~5 GB of the L4's 23 GB. Build jobs are bounded by host memory too (3 GiB per job) as well as by
    # nproc.
    #
    # Host headroom is the minimum of MemAvailable and, at every level of this process's own memory cgroup from its
    # leaf up to the mount's root that has a finite limit, limit - usage. The cgroup is resolved from
    # /proc/self/cgroup and /proc/self/mountinfo: on v1, the memory controller's line under a cgroup mount with the
    # memory option (memory.limit_in_bytes, memory.usage_in_bytes; ~2^63 means no limit); otherwise v2, the 0::
    # line under the cgroup2 mount (memory.max, memory.current; "max" means no limit). A level with a finite limit
    # must have a valid usage. Membership that cannot be mapped to a readable directory is an error, never "no
    # limit". With no memory cgroup membership at all, MemAvailable alone applies.
    MEMINFO=/proc/meminfo
    PROC_CGROUP=/proc/self/cgroup
    MOUNTINFO=/proc/self/mountinfo
    if [ "${VM_TEST_HARNESS:-}" = 1 ]; then
        MEMINFO=${VM_MEMINFO:-$MEMINFO}
        PROC_CGROUP=${VM_PROC_CGROUP:-$PROC_CGROUP}
        MOUNTINFO=${VM_MOUNTINFO:-$MOUNTINFO}
    fi
    NEED_KIB=16777216
    NEED_DEV_MIB=12288
    JOB_KIB=3145728
    # cgroup_scan: sets CG_VER (v1, v2 or none), CG_PATH, CG_HEAD (KiB, empty if no level has a finite limit),
    # CG_LEVELS (dir=limit/usage per level, KiB) and CG_ERR (why headroom could not be established; empty if it was)
    cgroup_scan() {
        local line rel mnt root leaf d lim use limf usef v
        CG_VER=none CG_PATH="" CG_HEAD="" CG_LEVELS="" CG_ERR=""
        [ -r "$PROC_CGROUP" ] || return 0   # no cgroup membership to account for
        line=$(awk -F: '{n = split($2, c, ","); for (i = 1; i <= n; i++) if (c[i] == "memory") {print $3; exit}}' \
            "$PROC_CGROUP")
        if [ -n "$line" ]; then
            CG_VER=v1
            CG_PATH=$line
            v=$(awk '{for (i = 7; i <= NF; i++) if ($i == "-") break;
                      n = split($(i + 3), o, ","); for (j = 1; j <= n; j++) if ($(i + 1) == "cgroup" && o[j] == "memory")
                      {print $4, $5; exit}}' "$MOUNTINFO" 2>/dev/null)
            limf=memory.limit_in_bytes usef=memory.usage_in_bytes
        else
            line=$(awk -F: '$1 == "0" && $2 == "" {print $3; exit}' "$PROC_CGROUP")
            [ -n "$line" ] || return 0   # no memory controller and no unified line
            CG_VER=v2
            CG_PATH=$line
            v=$(awk '{for (i = 7; i <= NF; i++) if ($i == "-") break; if ($(i + 1) == "cgroup2") {print $4, $5; exit}}' \
                "$MOUNTINFO" 2>/dev/null)
            limf=memory.max usef=memory.current
        fi
        [ -n "$v" ] || { CG_ERR="$CG_VER membership $CG_PATH but no matching mount in mountinfo"; return 1; }
        root=${v%% *}
        mnt=${v#* }
        rel=$CG_PATH
        if [ "$root" != / ]; then
            case "$rel" in
                "$root") rel=/ ;;
                "$root"/*) rel=${rel#"$root"} ;;
                *) CG_ERR="cgroup path $CG_PATH is outside the mount root $root"; return 1 ;;
            esac
        fi
        rel=${rel%/}
        leaf=$mnt$rel
        [ -d "$leaf" ] || { CG_ERR="cgroup directory $leaf missing"; return 1; }
        d=$leaf
        while :; do
            if [ -e "$d/$limf" ]; then
                lim=$(cat "$d/$limf" 2>/dev/null)
                if [[ "$lim" =~ ^[0-9]+$ ]] && { [ "$CG_VER" = v2 ] || [ "${#lim}" -lt 19 ]; }; then
                    use=$(cat "$d/$usef" 2>/dev/null)
                    [[ "$use" =~ ^[0-9]+$ ]] || { CG_ERR="finite limit in $d but usage missing or malformed"; return 1; }
                    v=$(((lim - use) / 1024))
                    CG_LEVELS="$CG_LEVELS$d=$((lim / 1024))/$((use / 1024));"
                    if [ -z "$CG_HEAD" ] || [ "$v" -lt "$CG_HEAD" ]; then CG_HEAD=$v; fi
                elif [ "$lim" != max ] && ! [[ "$lim" =~ ^[0-9]+$ ]]; then
                    CG_ERR="unreadable limit in $d"
                    return 1
                fi
            fi
            [ "$d" = "$mnt" ] && break
            d=$(dirname "$d")
            case "$d" in "$mnt"|"$mnt"/*) ;; *) break ;; esac
        done
        return 0
    }
    resources_check() {   # resources_check LABEL: writes resources_LABEL.json and the raw readings; 0 if enough
        local avail head dev ok=true
        cp "$MEMINFO" "$O/meminfo_$1.txt" 2>/dev/null
        cp "$PROC_CGROUP" "$O/proc_cgroup_$1.txt" 2>/dev/null
        free -m > "$O/free_$1.txt" 2>&1
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv > "$O/gpu_memory_$1.txt" 2>&1
        avail=$(awk '$1 == "MemAvailable:" {print $2}' "$MEMINFO" 2>/dev/null)
        [[ "$avail" =~ ^[0-9]+$ ]] || { avail=0; ok=false; }
        cgroup_scan || ok=false
        head=$avail
        if [ -n "$CG_HEAD" ] && [ "$CG_HEAD" -lt "$head" ]; then head=$CG_HEAD; fi
        [ "$head" -lt 0 ] && head=0
        dev=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [[ "$dev" =~ ^[0-9]+$ ]] || { dev=0; ok=false; }
        [ "$head" -ge "$NEED_KIB" ] || ok=false
        [ "$dev" -ge "$NEED_DEV_MIB" ] || ok=false
        printf '{"label": "%s", "mem_available_kib": %d, "cgroup_version": "%s", "cgroup_path": "%s", "cgroup_headroom_kib": %s, "cgroup_levels": "%s", "cgroup_error": "%s", "headroom_kib": %d, "need_kib": %d, "device_free_mib": %d, "need_device_mib": %d, "nproc": %d, "ok": %s}\n' \
            "$1" "$avail" "$CG_VER" "$CG_PATH" "${CG_HEAD:-null}" "$CG_LEVELS" "$CG_ERR" "$head" "$NEED_KIB" "$dev" \
            "$NEED_DEV_MIB" "$(nproc)" "$ok" > "$O/resources_$1.json"
        [ "$ok" = true ] || return 1
        JOBS=$((head / JOB_KIB))
        [ "$JOBS" -lt 1 ] && JOBS=1
        [ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
        return 0
    }
    resources_check start || fail resources 1
    G[resources]=0
fi
run checkout "$O/clone.log" git clone -q https://github.com/JohnnyTeutonic/paperkiln.git src || fail checkout $?
run checkout "$O/clone.log" git -C src -c advice.detachedHead=false checkout -q --detach "$COMMIT" || fail checkout $?
{ [ "$(git -C src rev-parse HEAD)" = "$COMMIT" ] && [ -z "$(git -C src status --porcelain)" ]; } || fail checkout 99
if [ "$PROTO" = backward ]; then
    run checkout "$O/clone.log" git -C src worktree add -q --detach "$W/base" "$BASE" || fail checkout $?
    { [ "$(git -C base rev-parse HEAD)" = "$BASE" ] && [ -z "$(git -C base status --porcelain)" ]; } || fail checkout 99
fi
G[checkout]=0
if [ "$PROTO" = prefill ]; then
    # the engine (src/, include/) and the model (tools/parity_model.hpp) are master's at PREFILL_BASE, the code the
    # kernel runs r1 and b1 gated, and the commit descends from it
    { git -C src merge-base --is-ancestor "$PREFILL_BASE" "$COMMIT" \
      && git -C src diff --quiet "$PREFILL_BASE" "$COMMIT" -- src include tools/parity_model.hpp; } \
      > "$O/engine_unchanged.log" 2>&1 || fail engine_unchanged 1
    G[engine_unchanged]=0
fi
run configure "$O/configure.log" cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON \
    -DMICROTORCH_DEVCHECK=ON -DMICROTORCH_BUILD_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES=89 || fail configure $?
G[configure]=0
TARGETS="test_selective_scan_cuda test_selective_scan_chunked"
[ "$PROTO" = backward ] && TARGETS="$TARGETS test_selective_scan_chunked_backward"
[ "$PROTO" = prefill ] && TARGETS="mamba_prefill_bench test_mamba_prefill_bench"
# prefill: every step from the build on runs under measure.py (the checked-out commit's), which records its elapsed
# time and peak RSS in STEP.time.json. A step that exited 0 without its receipt fails with 96 (measure.py keeps the
# command's status and reports a receipt it could not write; the missing receipt is what fails the step).
M=""
[ "$PROTO" = prefill ] && M="python3 src/experiments/mamba_parallel_scan_2026-11-03/measure.py"
receipt() { [ -z "$M" ] || [ -s "$O/$1.time.json" ]; }
run build "$O/build.log" ${M:+$M "$O/build.time.json" --} cmake --build build --target $TARGETS --parallel "$JOBS" \
    || fail build $?
receipt build || fail build 96
[ -z "$(git -C src status --porcelain)" ] || fail build 98   # the build wrote nothing into the source tree
G[build]=0
if [ "$PROTO" = prefill ]; then
    # gate 1: the host test, again here, on the same commit
    run host_test "$O/host_test.json" $M "$O/host_test.time.json" -- build/test_mamba_prefill_bench \
        || fail host_test $?
    receipt host_test || fail host_test 96
    G[host_test]=0
    # Every bench process runs with the scan switch, TF32 and the profile unset unless its step sets them; the bench
    # refuses MICROTORCH_SCAN_CHUNK set from outside, and sets it per arm itself.
    B="env -u MICROTORCH_SCAN_CHUNK -u MICROTORCH_TF32 -u MICROTORCH_GPU_PROFILE -u MICROTORCH_CUBLAS"
    # the resources again, just before the heavyweight gate; short is an operational stop before it
    resources_check before_gate || fail resources 2
    # gates 2-3: correctness before any timing, for each GEMM backend that is timed; a failure stops paid work
    RUN_ERR="$O/prefill_gate.err" run prefill_gate "$O/prefill_gate.jsonl" $M "$O/prefill_gate.time.json" -- \
        $B build/mamba_prefill_bench --mode gate --require-cuda || fail prefill_gate $?
    receipt prefill_gate || fail prefill_gate 96
    G[prefill_gate]=0
    RUN_ERR="$O/prefill_gate_cublas.err" run prefill_gate_cublas "$O/prefill_gate_cublas.jsonl" \
        $M "$O/prefill_gate_cublas.time.json" -- \
        $B MICROTORCH_CUBLAS=1 build/mamba_prefill_bench --mode gate --require-cuda || fail prefill_gate_cublas $?
    receipt prefill_gate_cublas || fail prefill_gate_cublas 96
    G[prefill_gate_cublas]=0
    # the timings (profile off), then the per-op profile (its own processes: the profile switch is read once)
    RUN_ERR="$O/prefill_time.err" run prefill_time "$O/prefill_time.jsonl" $M "$O/prefill_time.time.json" -- \
        $B build/mamba_prefill_bench --mode time --require-cuda
    G[prefill_time]=$?
    [ "${G[prefill_time]}" = 0 ] && ! receipt prefill_time && G[prefill_time]=96
    RUN_ERR="$O/prefill_time_cublas.err" run prefill_time_cublas "$O/prefill_time_cublas.jsonl" \
        $M "$O/prefill_time_cublas.time.json" -- \
        $B MICROTORCH_CUBLAS=1 build/mamba_prefill_bench --mode time --require-cuda
    G[prefill_time_cublas]=$?
    [ "${G[prefill_time_cublas]}" = 0 ] && ! receipt prefill_time_cublas && G[prefill_time_cublas]=96
    RUN_ERR="$O/prefill_profile.err" run prefill_profile "$O/prefill_profile.jsonl" \
        $M "$O/prefill_profile.time.json" -- \
        $B MICROTORCH_GPU_PROFILE=1 build/mamba_prefill_bench --mode profile --require-cuda
    G[prefill_profile]=$?
    [ "${G[prefill_profile]}" = 0 ] && ! receipt prefill_profile && G[prefill_profile]=96
    RUN_ERR="$O/prefill_profile_cublas.err" run prefill_profile_cublas "$O/prefill_profile_cublas.jsonl" \
        $M "$O/prefill_profile_cublas.time.json" -- \
        $B MICROTORCH_CUBLAS=1 MICROTORCH_GPU_PROFILE=1 build/mamba_prefill_bench --mode profile --require-cuda
    G[prefill_profile_cublas]=$?
    [ "${G[prefill_profile_cublas]}" = 0 ] && ! receipt prefill_profile_cublas && G[prefill_profile_cublas]=96
    resources_check end > /dev/null 2>&1   # a record only, made when the run gets this far
    sha256sum build/mamba_prefill_bench build/test_mamba_prefill_bench > "$O/binaries.sha256" 2>&1
    for k in prefill_time prefill_time_cublas prefill_profile prefill_profile_cublas; do
        [ "${G[$k]}" = 0 ] || exit 1
    done
    exit 0
fi
# gate 1: the host test(s), again here, on the same commit
run host_test "$O/host_test.json" build/test_selective_scan_chunked || fail host_test $?
G[host_test]=0
if [ "$PROTO" = backward ]; then
    run host_test_bwd "$O/host_test_bwd.json" build/test_selective_scan_chunked_backward || fail host_test_bwd $?
    G[host_test_bwd]=0
    # a binary wheel only (the image's Python was 3.13 on 3 Nov; numpy 2.2.6 has wheels for 3.10-3.13)
    run numpy "$O/numpy_install.log" python3 -m pip install -q --only-binary=:all: numpy==2.2.6 || fail numpy $?
else
    run numpy "$O/numpy_install.log" python3 -m pip install -q numpy==1.26.4 || fail numpy $?
fi
G[numpy]=0
# the frozen 220-case test and its independent replay; the ordered control first (a prerequisite). With L set, the
# op runs the chunked forward and, on a backward commit, the chunked backward.
for L in unset 2 3; do
    if [ $L = unset ]; then
        run strict_$L "$O/strict_$L.log" env -u MICROTORCH_SCAN_CHUNK MICROTORCH_GPU_PROFILE=1 \
            build/test_selective_scan_cuda --require-cuda
    else
        run strict_$L "$O/strict_$L.log" env MICROTORCH_SCAN_CHUNK=$L MICROTORCH_GPU_PROFILE=1 \
            build/test_selective_scan_cuda --require-cuda
    fi
    G[strict_$L]=$?
    run replay_$L "$O/replay_$L.log" python3 -O src/tools/replay_selective_scan.py "$O/strict_$L.log" --require-cuda
    G[replay_$L]=$?
    if [ $L = unset ] && { [ "${G[strict_unset]}" != 0 ] || [ "${G[replay_unset]}" != 0 ]; }; then exit 1; fi
done
# path selection: the device path parses the switch, so a malformed value must be refused with the parser's message
run switch_refuses_malformed "$O/strict_malformed.log" env "MICROTORCH_SCAN_CHUNK= 2" MICROTORCH_GPU_PROFILE=1 \
    build/test_selective_scan_cuda --require-cuda
malformed_rc=$?
if [ $malformed_rc != 0 ] && grep -q "MICROTORCH_SCAN_CHUNK must be a decimal count" "$O/strict_malformed.log"; then
    G[switch_refuses_malformed]=0
else
    G[switch_refuses_malformed]=1
fi
if [ "$PROTO" = forward ]; then
    # gate 3 and the timings
    run scan_bench "$O/bench_build.log" nvcc -O3 -std=c++17 -arch=sm_89 -I src/src src/tools/scan_bench.cu -o scan_bench \
        || fail scan_bench $?
    RUN_ERR="$O/scan_bench.err" run scan_bench "$O/scan_bench.jsonl" ./scan_bench 20
    G[scan_bench]=$?
    sha256sum build/test_selective_scan_cuda build/test_selective_scan_chunked scan_bench > "$O/binaries.sha256" 2>&1
    [ "${G[scan_bench]}" = 0 ]   # the script's own exit status follows the last gate
    exit $?
fi
# gate 5: the default path's device outputs and gradients are master's: the base commit's frozen test, built the same
# way, with MICROTORCH_SCAN_CHUNK unset; every case line (inputs, gradients and y at %.9g) must match the branch's
run build_base "$O/configure_base.log" cmake -S base -B base_build -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON \
    -DMICROTORCH_DEVCHECK=ON -DMICROTORCH_BUILD_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES=89 || fail build_base $?
run build_base "$O/build_base.log" cmake --build base_build --target test_selective_scan_cuda --parallel "$(nproc)" \
    || fail build_base $?
[ -z "$(git -C base status --porcelain)" ] || fail build_base 98
G[build_base]=0
run default_identical "$O/base_strict_unset.log" env -u MICROTORCH_SCAN_CHUNK MICROTORCH_GPU_PROFILE=1 \
    base_build/test_selective_scan_cuda --require-cuda
base_rc=$?
grep '"raw_case":true' "$O/strict_unset.log" > "$O/cases_branch.txt"
grep '"raw_case":true' "$O/base_strict_unset.log" > "$O/cases_base.txt"
if [ $base_rc = 0 ] && [ "$(wc -l < "$O/cases_branch.txt")" = 220 ] && cmp -s "$O/cases_branch.txt" "$O/cases_base.txt"; then
    G[default_identical]=0
else
    G[default_identical]=1
fi
# gate 4 and the timings
run scan_bwd_bench "$O/bench_build.log" nvcc -O3 -std=c++17 -arch=sm_89 -I src/src -I src/tests \
    src/tools/scan_bwd_bench.cu -o scan_bwd_bench || fail scan_bwd_bench $?
RUN_ERR="$O/scan_bwd_bench.err" run scan_bwd_bench "$O/scan_bwd_bench.jsonl" ./scan_bwd_bench 20
G[scan_bwd_bench]=$?
sha256sum build/test_selective_scan_cuda build/test_selective_scan_chunked build/test_selective_scan_chunked_backward \
    base_build/test_selective_scan_cuda scan_bwd_bench > "$O/binaries.sha256" 2>&1
[ "${G[scan_bwd_bench]}" = 0 ]   # the script's own exit status follows the last gate
