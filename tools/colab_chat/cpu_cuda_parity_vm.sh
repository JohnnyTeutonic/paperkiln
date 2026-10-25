#!/usr/bin/env bash
# CUDA mtstudio against the CPU build of the same commit on trained models (a runner_multi "cmd"
# job; LOCAL_LLM_STACK.md section 11, item 4.1). Builds a CPU mtstudio beside boot_jobs.sh's CUDA
# one, stages each uploaded model as chat_eval_vm.sh does, and runs tools/cpu_cuda_parity.py.
#   bash cpu_cuda_parity_vm.sh OUT TAG...
# Uploads per TAG: TAG.safetensors and TAG.spec.json (an eval spec: its vocabulary is an upload of
# the same file name when present, else the VM's checkout, as its corpus; its out_dir becomes
# /content/eval/TAG).
set -euo pipefail
OUT=$1
shift
U=/content/chat/upload
mkdir -p "$OUT"
cd /content/pk
if [ ! -x build_cpu/mtstudio ]; then
  cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release > "$OUT/cpu_cmake.log" 2>&1
  grep -i "transformer_core from" "$OUT/cpu_cmake.log" || true
  make -C build_cpu -j"$(nproc)" mtstudio > "$OUT/cpu_make.log" 2>&1 || { tail -30 "$OUT/cpu_make.log"; exit 1; }
fi
MODELS=""
for t in "$@"; do
  D=/content/eval/$t
  mkdir -p "$D"
  python3 - "$t" "$D" <<'P'
import json, os, sys
t, d = sys.argv[1], sys.argv[2]
u = "/content/chat/upload"
s = json.load(open(f"{u}/{t}.spec.json"))
for k in ("vocab", "corpus"):
    p = s["data"][k]
    up = f"{u}/{os.path.basename(p)}"  # a vocabulary the boot does not fetch arrives as an upload
    s["data"][k] = up if k == "vocab" and os.path.exists(up) else "/content/pk/" + p.split("/microtorch/", 1)[1]
    if not os.path.exists(s["data"][k]):
        raise SystemExit(f"{t}: data.{k} {p} is neither uploaded nor on the VM")
s["out_dir"] = d
json.dump(s, open(f"{d}/spec.json", "w"), indent=1)
src, dst = f"{u}/{t}.safetensors", f"{d}/{s['name']}.safetensors"
if os.path.exists(src):
    os.replace(src, dst)
if not os.path.exists(dst):
    raise SystemExit(f"{t}: {dst} missing")
print(t, "ready:", s["name"], s["data"]["vocab"])
P
  MODELS="$MODELS --model $t=$D/spec.json"
done
OMP_NUM_THREADS=$(nproc) python3 tools/cpu_cuda_parity.py --cpu-bin build_cpu/mtstudio --cuda-bin build_cuda/mtstudio \
  $MODELS --out "$OUT" 2>&1 | tee "$OUT/cpu_cuda_parity.log"
exit "${PIPESTATUS[0]}"
