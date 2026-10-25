#!/usr/bin/env bash
# Chat + QA evals of chat models on a cloud VM (a runner_multi "cmd" job): ember.cpp's server is
# built here for the L4, and tools/chat_eval.py (frozen chat prompts, then chat_prompts_v1 per
# source) and tools/qa_eval.py run each model through ember's CUDA server and the CUDA mtstudio
# (reference bits/byte). Local GPU work moved off the laptop on 22 Oct 2026.
#   bash chat_eval_vm.sh OUT TAG...
# A TAG is either
#   NAME      an uploaded model: NAME.gguf and NAME.safetensors in /content/chat/upload, and its eval
#             spec NAME.spec.json uploaded or in experiments/chat_general/eval_specs/ (a local spec;
#             its vocabulary is taken from an upload of the same file name when present, its corpus
#             from the VM's checkout); the files are linked, not moved, so a later job (or a
#             relaunch on the same RunPod volume) finds them again;
#   NAME@JOB  a model this session trained: job JOB's spec (/content/chat/JOB_spec.json) and its
#             out_dir /content/chat/JOB.
# ember: an uploaded ember_src.tar.gz when present (the 23 Oct 2026 run), else ember.cpp at
# tiny_chat_run.EMBER_COMMIT (ember main, 25 Oct 2026), built once per VM by
# tiny_chat_run.ensure_ember; TF32 GEMMs off (EMBER_GEMM_TF32=0), as in every parity gate.
# OUT/manifest.json records both commits, the GPU, and the sha256 of every served file.
set -euo pipefail
OUT=$1
shift
U=/content/chat/upload
mkdir -p "$OUT"
# A failure still brings the logs home (the runner fetches a FAILED job's outputs best-effort).
trap 'tar -czf /tmp/chat_eval_results.tar.gz -C "$OUT" --exclude="*refeval*" --exclude=www \
  --exclude="*_ref.jsonl" --exclude="*.tar.gz" . && mv /tmp/chat_eval_results.tar.gz "$OUT/" || true' ERR
cd /content/pk
if [ -f $U/ember_src.tar.gz ]; then
  E=/content/ember_tarball
  if [ ! -x $E/build/tinyllama_server ]; then
    # ember's CMake needs Boost.Regex, which the Colab image lacks (as in ember's tools/colab_device_gate_vm.sh).
    apt-get update -qq > /dev/null 2>&1
    apt-get install -y -q libboost-regex-dev > "$OUT/apt.log" 2>&1
    rm -rf $E && mkdir -p $E && tar -xzf $U/ember_src.tar.gz -C $E
    cmake -S $E -B $E/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > "$OUT/ember_cmake.log" 2>&1
    make -C $E/build -j"$(nproc)" tinyllama_server > "$OUT/ember_make.log" 2>&1
  fi
  BIN=$E/build/tinyllama_server
  echo '{"source": "ember_src.tar.gz"}' > "$OUT/ember_build.json"
else
  python3 tools/colab_chat/ensure_ember_vm.py "$OUT/ember_build.json" "$OUT/ember_build.log"
  BIN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bin"])' "$OUT/ember_build.json")
fi
MODELS=""
for t in "$@"; do
  SPEC=$(python3 tools/colab_chat/chat_eval_stage.py "$t")
  MODELS="$MODELS --model ${t%@*}=$SPEC"
done
python3 tools/colab_chat/chat_eval_stage.py --manifest "$OUT/manifest.json" "$OUT/ember_build.json" "$@"
export EMBER_GPU_LAYERS=-1 EMBER_GEMM_TF32=0 MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 \
  MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_CUBLAS=1 MICROTORCH_STEP_RESIDENCY=1 MICROTORCH_DEFER_DOWNLOADS=1
MT=/content/pk/build_cuda/mtstudio
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
  chat chat2 qa chat.log chat2.log qa.log manifest.json ember_build.json
echo "results: $OUT/chat_eval_results.tar.gz"
