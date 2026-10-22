#!/usr/bin/env bash
# Chat + QA evals of chat models on a Colab VM (a runner_multi "cmd" job): the
# models' GGUFs, safetensors and specs, their vocabularies and an ember.cpp
# source tarball arrive as runner uploads; ember is built here for the L4, and
# tools/chat_eval.py (frozen chat prompts, then chat_prompts_v1 per source) and
# tools/qa_eval.py run each model through ember's CUDA server and the CUDA
# mtstudio (reference bits/byte). Local GPU work moved to Colab on 22 Oct 2026.
#   bash chat_eval_vm.sh OUT TAG...
# Uploads per TAG: TAG.gguf, TAG.safetensors, TAG.spec.json (a local eval spec;
# its vocabulary is taken from an upload of the same file name when present,
# its corpus from the VM's checkout). Also ember_src.tar.gz.
set -euo pipefail
OUT=$1
shift
U=/content/chat/upload
E=/content/ember
mkdir -p "$OUT"
# A failure still brings the logs home (the runner fetches a FAILED job's outputs best-effort).
trap 'tar -czf /tmp/chat_eval_results.tar.gz -C "$OUT" --exclude="*refeval*" --exclude=www \
  --exclude="*_ref.jsonl" --exclude="*.tar.gz" . && mv /tmp/chat_eval_results.tar.gz "$OUT/" || true' ERR
if [ ! -x $E/build/tinyllama_server ]; then
  # ember's CMake needs Boost.Regex, which the Colab image lacks (as in ember's tools/colab_device_gate_vm.sh).
  apt-get update -qq > /dev/null 2>&1
  apt-get install -y -q libboost-regex-dev > "$OUT/apt.log" 2>&1
  rm -rf $E && mkdir -p $E && tar -xzf $U/ember_src.tar.gz -C $E
  cmake -S $E -B $E/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > "$OUT/ember_cmake.log" 2>&1
  make -C $E/build -j"$(nproc)" tinyllama_server > "$OUT/ember_make.log" 2>&1
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
    up = f"{u}/{os.path.basename(p)}"
    vm = "/content/pk/" + p.split("/microtorch/", 1)[1]
    s["data"][k] = up if k == "vocab" and os.path.exists(up) else vm
    if not os.path.exists(s["data"][k]):
        raise SystemExit(f"{t}: data.{k} {p} is neither uploaded nor on the VM")
s["out_dir"] = d
json.dump(s, open(f"{d}/spec.json", "w"), indent=1)
for ext in ("gguf", "safetensors"):
    src, dst = f"{u}/{t}.{ext}", f"{d}/{s['name']}.{ext}"
    if os.path.exists(src):
        os.replace(src, dst)
    if not os.path.exists(dst):
        raise SystemExit(f"{t}: {dst} missing")
print(t, "ready:", s["name"], s["data"]["vocab"])
P
  MODELS="$MODELS --model $t=$D/spec.json"
done
export EMBER_GPU_LAYERS=-1 MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 \
  MICROTORCH_CUBLAS=1 MICROTORCH_STEP_RESIDENCY=1 MICROTORCH_DEFER_DOWNLOADS=1
MT=/content/pk/build_cuda/mtstudio
BIN=$E/build/tinyllama_server
cd /content/pk
# shellcheck disable=SC2086
python3 tools/chat_eval.py --mtstudio $MT --ember-bin $BIN --port 8091 $MODELS \
  --prompts corpus/eval/tinychat_hf_prompts_v1.jsonl --out "$OUT/chat" --n 1 --temp 0.7 --max-new 64 --limit 100 \
  > "$OUT/chat.log" 2>&1
# shellcheck disable=SC2086
python3 tools/chat_eval.py --mtstudio $MT --ember-bin $BIN --port 8091 $MODELS \
  --prompts corpus/eval/chat_prompts_v1.jsonl --out "$OUT/chat2" --n 1 --temp 0.7 --max-new 64 --limit 200 \
  > "$OUT/chat2.log" 2>&1
# shellcheck disable=SC2086
python3 tools/qa_eval.py --ember-bin $BIN $MODELS --qa corpus/eval/simple_qa_v1.jsonl --out "$OUT/qa" --port 8091 \
  > "$OUT/qa.log" 2>&1
tar -czf "$OUT/chat_eval_results.tar.gz" -C "$OUT" --exclude='*refeval*' --exclude=www --exclude='*_ref.jsonl' \
  chat chat2 qa chat.log chat2.log qa.log
echo "results: $OUT/chat_eval_results.tar.gz"
