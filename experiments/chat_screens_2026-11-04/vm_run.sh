#!/usr/bin/env bash
# Runs ON the Colab VM. Usage: bash vm_run.sh FULL_COMMIT_SHA RUN_ID MANIFEST_SHA256
#   FULL_COMMIT_SHA: this experiment's commit (the commit vm_run.sh was fetched at), recorded in status.json;
#   MANIFEST_SHA256: the hash of $ROOT/screens_in/manifest.json, which pins every uploaded input.
# The first-wave screening session of Claude's chat screening lane (the root workspace's
# AI_ML/chat_screens_claude/PROTOCOL.md v1.2 and SESSION.md): S0, S1, S2 and S3 trained in turn with arm A's recipe,
# the trainer built from paperkiln's numerical source SOURCE below, each screen's GGUF packaged in parts the driver
# collects while the session continues.
# The run directory /content/run_RUN_ID is claimed with an atomic mkdir, so a second launch of the same run exits
# before traps, PID or any work. Once it exists, status.json (each gate's exit status, the incoming shell status, the
# script's final exit status, run id, commit, the source HEAD and the screens packaged) is written atomically on every
# exit path, then, only if that succeeded, the DONE marker holding the run id. pass is true only if exactly the
# required gates ran, all 0, and the script exits 0. A failed prerequisite stops further paid work; a failed screen
# stops the session (no later screen runs, PROTOCOL.md section 7: no automatic rerun).
# Stop: SIGTERM or SIGINT to the PID published in /content/vm_RUN_ID.pid. Every gate command runs in its own process
# group (setsid) and this shell waits on it, so a stop is handled at once; the handler stops that group (TERM every
# second for up to 60 s, then KILL), writes stop.json and exits through the trap that writes status.json and DONE.
set -u
COMMIT=${1:-}
RUN=${2:-}
MANIFEST_SHA=${3:-}
# paperkiln's numerical source: arm A's admitted trainer source (PROTOCOL.md section 2)
SOURCE=1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1
SCREENS="S0 S1 S2 S3"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "COMMIT must be a full 40-character SHA" >&2; exit 2; }
[[ "$RUN" =~ ^[A-Za-z0-9_-]{1,32}$ ]] || { echo "RUN_ID: 1-32 letters, digits, - or _" >&2; exit 2; }
[[ "$MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "MANIFEST_SHA256 must be 64 hex characters" >&2; exit 2; }
case $- in *m*) echo "job control must be off (setsid would fork)" >&2; exit 2;; esac
ROOT=/content
[ "${VM_TEST_HARNESS:-}" = 1 ] && ROOT=${VM_ROOT:-/content}   # test_vm_run.py only
signal_point() { [ "${VM_TEST_HARNESS:-}" = 1 ] && [ "${VM_TEST_SIGNAL_AT:-}" = "$1" ] && kill -TERM $$; }  # tests
IN=$ROOT/screens_in
W=$ROOT/run_$RUN
O=$W/out
mkdir "$W" 2>/dev/null || { echo "$W exists: refusing to reuse a run directory" >&2; exit 2; }
mkdir -p "$O/screens" || exit 2
declare -A G=()
REQUIRED=$(printf '%s\n' build checkout configure inputs provenance specs \
    $(for s in $SCREENS; do echo "train_$s"; echo "package_$s"; done) | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
finish() {
    local rc=$? pass=true keys final packaged
    trap '' TERM INT   # publication is never interrupted; a stop arriving now finds DONE shortly
    keys=$(printf '%s\n' "${!G[@]}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')
    [ "$keys" = "$REQUIRED" ] || pass=false
    for k in "${!G[@]}"; do [ "${G[$k]}" = 0 ] || pass=false; done
    [ "$rc" = 0 ] || pass=false
    [ "$pass" = true ] && final=0 || final=1   # the script's actual exit status, recorded as exit_status
    packaged=$(for s in $SCREENS; do [ -f "$O/screens/$s/READY" ] && printf '"%s",' "$s"; done | sed 's/,$//')
    {
        printf '{"run_id": "%s", "commit": "%s", "source": "%s", "head": "%s", "manifest_sha256": "%s", "shell_status": %d, "exit_status": %d, "pass": %s, "packaged": [%s], "gates": {' \
            "$RUN" "$COMMIT" "$SOURCE" "$(git -C "$W/src" rev-parse HEAD 2>/dev/null)" "$MANIFEST_SHA" "$rc" "$final" \
            "$pass" "$packaged"
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

# Stop handling, unchanged from the Mamba scan packet: CPID is the running gate command's PID and process group. A
# stop arriving while a command is being forked and registered is held in STOP_PENDING and delivered right after
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
# run GATE LOG CMD...: CMD in its own process group, stdin from /dev/null, output appended to LOG; its exit status is
# returned (97 if a member of its group outlived it and had to be killed)
run() {
    local rc
    STEP=$1
    trap 'STOP_PENDING=1' TERM INT
    setsid "${@:3}" >> "$2" 2>&1 < /dev/null &
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
HERE=$(dirname "$(readlink -f "$0")")
PY=${VM_TEST_PYTHON:-python3}
CMAKE=${VM_TEST_CMAKE:-cmake}
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1; nproc; python3 --version; } > "$O/hardware.log" 2>&1

# provenance: exactly one NVIDIA L4 (PROTOCOL.md section 5)
if [ "${VM_TEST_HARNESS:-}" = 1 ]; then
    [ "${VM_TEST_PROVENANCE:-0}" = 0 ] || fail provenance 1
else
    { [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1 ] \
      && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L4; } > "$O/provenance.log" 2>&1 || fail provenance 1
fi
G[provenance]=0

# checkout: paperkiln at the numerical source, clean
CLONE_URL=${VM_TEST_CLONE_URL:-https://github.com/JohnnyTeutonic/paperkiln.git}
[ "${VM_TEST_HARNESS:-}" = 1 ] || CLONE_URL=https://github.com/JohnnyTeutonic/paperkiln.git
run checkout "$O/clone.log" git clone -q "$CLONE_URL" src || fail checkout $?
run checkout "$O/clone.log" git -C src -c advice.detachedHead=false checkout -q --detach "$SOURCE" || fail checkout $?
{ [ "$(git -C src rev-parse HEAD)" = "$SOURCE" ] && [ -z "$(git -C src status --porcelain)" ]; } || fail checkout 99
G[checkout]=0

# inputs: the manifest is the one the driver pinned; every listed file hashes as listed; the base weights are joined
# from their parts and hash as listed. Files land in $W/upload under the names arm A's spec uses.
mkdir -p "$W/upload"
run inputs "$O/inputs.log" "$PY" - "$IN" "$W/upload" "$MANIFEST_SHA" <<'PY' || fail inputs $?
import hashlib, json, os, sys
src, dst, want = sys.argv[1], sys.argv[2], sys.argv[3]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
man_path = os.path.join(src, "manifest.json")
if sha(man_path) != want:
    sys.exit("manifest hash differs from the driver's pin")
man = json.load(open(man_path))
parts = man["base_parts"]
with open(os.path.join(dst, "sfull_sft1500.safetensors.tmp"), "wb") as out:
    for p in parts:
        name, digest = p["name"], p["sha256"]
        if os.path.basename(name) != name:
            sys.exit("part name is not a plain file name: " + name)
        path = os.path.join(src, name)
        if sha(path) != digest:
            sys.exit("part hash differs: " + name)
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                out.write(b)
os.replace(os.path.join(dst, "sfull_sft1500.safetensors.tmp"), os.path.join(dst, "sfull_sft1500.safetensors"))
for name, digest in man["files"].items():
    if os.path.basename(name) != name:
        sys.exit("file name is not a plain file name: " + name)
    path = os.path.join(dst, name) if name == "sfull_sft1500.safetensors" else os.path.join(src, name)
    if sha(path) != digest:
        sys.exit("file hash differs: " + name)
    if name != "sfull_sft1500.safetensors":
        with open(path, "rb") as f, open(os.path.join(dst, name), "wb") as g:
            g.write(f.read())
        if sha(os.path.join(dst, name)) != digest:
            sys.exit("copied file hash differs: " + name)
print(json.dumps({"verified": sorted(man["files"]), "base_parts": len(parts)}))
PY
G[inputs]=0

# configure and build: arm A's boot, target mtstudio, sm_89
run configure "$O/configure.log" "$CMAKE" -S src -B build -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON \
    -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89 || fail configure $?
G[configure]=0
run build "$O/build.log" "$CMAKE" --build build --target mtstudio --parallel "$(nproc)" || fail build $?
[ -x build/mtstudio ] || fail build 98
sha256sum build/mtstudio > "$O/mtstudio.sha256"
G[build]=0

# specs: arm A's spec (arm_a_lr3e5.json beside this script, its hash pinned in the manifest) with only the name,
# out_dir and the three input paths changed; any other difference fails the gate
run specs "$O/specs.log" "$PY" - "$HERE/arm_a_lr3e5.json" "$W" "$IN/manifest.json" $SCREENS <<'PY' || fail specs $?
import copy, hashlib, json, sys
arm_a_path, work, man_path, screens = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
arm_a_bytes = open(arm_a_path, "rb").read()
man = json.load(open(man_path))
if hashlib.sha256(arm_a_bytes).hexdigest() != man["arm_a_spec_sha256"]:
    sys.exit("arm A spec differs from the manifest's pin")
arm_a = json.loads(arm_a_bytes)
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out
allowed = {"name", "out_dir", "data.sft_examples", "data.vocab", "train.init_from"}
for s in screens:
    spec = copy.deepcopy(arm_a)
    spec["name"] = "screen_" + s
    spec["out_dir"] = f"{work}/train_{s}"
    spec["data"]["vocab"] = f"{work}/upload/vocab.gguf"
    spec["data"]["sft_examples"] = f"{work}/upload/{s}.jsonl"
    spec["train"]["init_from"] = f"{work}/upload/sfull_sft1500.safetensors"
    a, b = flat(arm_a), flat(spec)
    changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if not changed <= allowed or set(a) != set(b):
        sys.exit(f"{s}: spec differs from arm A outside the allowed keys: {sorted(changed - allowed)}")
    with open(f"{work}/spec_{s}.json", "w") as f:
        json.dump(spec, f, indent=2)
print(json.dumps({"specs": screens, "allowed_changes": sorted(allowed)}))
PY
cp "$W"/spec_S*.json "$O/" 2>/dev/null
G[specs]=0

# train and package, one screen at a time. Arm A's backend environment (Codex's uplift_contract.py BACKEND_ENV).
BACKEND=(MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_STEP_RESIDENCY=1
         MICROTORCH_DEFER_DOWNLOADS=1 MICROTORCH_CUBLAS=1 OMP_NUM_THREADS=4)
MTSTUDIO=${VM_TEST_MTSTUDIO:-$W/build/mtstudio}
[ "${VM_TEST_HARNESS:-}" = 1 ] || MTSTUDIO=$W/build/mtstudio
for s in $SCREENS; do
    signal_point "screen_$s"
    run "train_$s" "$O/train_$s.log" env "${BACKEND[@]}" "$MTSTUDIO" run "$W/spec_$s.json" || fail "train_$s" $?
    # the run reached update 256 with finite losses, and exported exactly one GGUF
    "$PY" - "$O/train_$s.log" "$W/train_$s" > "$O/train_${s}_check.json" 2>&1 <<'PY' || fail "train_$s" 96
import glob, json, math, sys
steps = []
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if line.startswith("{"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("event") == "step":
            steps.append(d)
ggufs = glob.glob(sys.argv[2] + "/*.gguf")
ok = bool(steps) and steps[-1]["step"] == 256 and all(math.isfinite(d["loss"]) for d in steps) and len(ggufs) == 1
print(json.dumps({"last_step": steps[-1]["step"] if steps else None, "steps_logged": len(steps),
                  "final_loss": steps[-1]["loss"] if steps else None, "ggufs": ggufs, "ok": ok}))
sys.exit(0 if ok else 1)
PY
    G["train_$s"]=0
    # package: the GGUF in parts under 90 MB, a manifest of the whole and the parts, then READY (atomic)
    P=$O/screens/$s
    mkdir -p "$P"
    GGUF=$(ls "$W/train_$s"/*.gguf)
    run "package_$s" "$O/package_$s.log" split -b 90M -d -a 2 "$GGUF" "$P/model.gguf.part" || fail "package_$s" $?
    "$PY" - "$GGUF" "$P" "$s" "$RUN" <<'PY' >> "$O/package_$s.log" 2>&1 || fail "package_$s" 96
import glob, hashlib, json, os, sys
gguf, folder, screen, run = sys.argv[1:5]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
parts = sorted(glob.glob(folder + "/model.gguf.part*"))
man = {"screen": screen, "run_id": run, "gguf_sha256": sha(gguf), "gguf_bytes": os.path.getsize(gguf),
       "parts": [{"name": os.path.basename(p), "sha256": sha(p), "bytes": os.path.getsize(p)} for p in parts]}
if sum(p["bytes"] for p in man["parts"]) != man["gguf_bytes"]:
    sys.exit("parts do not add up to the GGUF")
with open(folder + "/manifest.json.tmp", "w") as f:
    json.dump(man, f, indent=2)
os.replace(folder + "/manifest.json.tmp", folder + "/manifest.json")
PY
    cp "$O/train_$s.log" "$P/train.log"
    echo "$RUN" > "$P/READY.tmp" && mv "$P/READY.tmp" "$P/READY" || fail "package_$s" 95
    G["package_$s"]=0
done
true
