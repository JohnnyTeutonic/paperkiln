#!/usr/bin/env bash
# The benchmarks campaign on a cloud VM (runner_multi "cmd" jobs; tools/colab_chat/bench_v1.json;
# experiments/benchmarks_2026-10-26/). One subcommand per job:
#   bash bench_vm.sh items OUT                 fetch the pinned benchmark files, build and check the item file
#   bash bench_vm.sh score OUT BLIMP_PER TAG... tools/bench_eval.py with the CUDA mtstudio score on uploaded models
#   bash bench_vm.sh hf OUT BLIMP_PER TAG=REPO@REV...   tools/bench_eval.py with HF transformers (float32)
#   bash bench_vm.sh iftiny OUT TAG...         tools/iftiny_eval.py: ember (CUDA, BENCH_EMBER_COMMIT; EMBER_DETERMINISTIC from the job) greedy, two
#                                              passes, the mtstudio parity check
#   bash bench_vm.sh iftiny_hf OUT TAG=REPO@REV...   tools/iftiny_eval.py with HF generate (one pass)
#   bash bench_vm.sh smolgguf OUT REPO@REV     SmolLM2-135M as an F32 GGUF (llama.cpp b6000's converter)
# Every subcommand resumes (the tools write per chunk / per reply on the VM's volume) and ends with
# OUT/results.tar.gz, which the runner brings home; a failure still packs what there is.
set -euo pipefail
CMD=$1
OUT=$2
shift 2
mkdir -p "$OUT"
cd /content/pk
ITEMS=/content/bench/bench_v1.jsonl
# ember's server runs in an ember_cwd folder whose debugging.log points at /dev/null; its stderr
# (ember_stderr.log there: warnings and errors) comes home, as the cause of a failed request is logged there.
pack() { cp /content/chat/provenance.json "$OUT/" 2>/dev/null || true; nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || true
  tar -czf /tmp/bench_results.tar.gz -C "$OUT" --exclude='*.tar.gz' --exclude='*.gguf' --exclude=www --exclude='debugging.log' . \
  && mv /tmp/bench_results.tar.gz "$OUT/results.tar.gz"; }
trap 'pack || true' ERR
export MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_CUBLAS=1 \
  MICROTORCH_STEP_RESIDENCY=1 MICROTORCH_DEFER_DOWNLOADS=1
MT=/content/pk/build_cuda/mtstudio
hf_deps() {
  # One pip at a time on the VM: two jobs installing at once broke huggingface_hub on 26 Oct 2026
  # ("invalid distribution ~uggingface-hub"); every pip in this script runs under /content/bench_pip.lock.
  exec 9> /content/bench_pip.lock
  flock 9
  python3 -c "import huggingface_hub" 2>/dev/null || \
    python3 -m pip -q install --root-user-action=ignore --force-reinstall "huggingface_hub==0.26.2" \
      > "$OUT/pip_hub.log" 2>&1 || true
  python3 -c "import transformers" 2>/dev/null || \
    python3 -m pip -q install --root-user-action=ignore "transformers==4.46.3" > "$OUT/pip.log" 2>&1
  python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())" \
    > "$OUT/hf_versions.txt"
  flock -u 9
}
case "$CMD" in
  items)
    mkdir -p /content/bench/raw
    python3 -I tools/bench_build.py fetch --raw /content/bench/raw > "$OUT/fetch.log" 2>&1
    python3 -I tools/bench_build.py build --raw /content/bench/raw --out $ITEMS > "$OUT/items_build.json"
    cp "$OUT/items_build.json" /content/bench/items_build.json
    ;;
  score)
    BP=$1
    shift
    for t in "$@"; do
      SPEC=$(python3 tools/colab_chat/bench_stage.py "$t")
      python3 tools/bench_eval.py run --items $ITEMS --out "$OUT/bench" --scorer mtstudio --mtstudio $MT \
        --model "$t=$SPEC" --blimp-per "$BP" >> "$OUT/score_$t.log" 2>&1
    done
    cp /content/eval/staged.jsonl "$OUT/" 2>/dev/null || true
    ;;
  hfgguf)
    # paperkiln models through transformers' GGUF loader, batched (tools/bench_eval.py hfgguf), and `mtstudio
    # score` on every EVERY-th item of the same requests as the cross-check (mtstudio scores one request at a
    # time: ~75 ms each for the 14.6M live model on an L4, so it cannot carry the full sets for every model).
    BP=$1
    EVERY=$2
    shift 2
    hf_deps
    flock 9
    [ -d /content/llamacpp_b6000 ] || git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp \
      /content/llamacpp_b6000
    python3 -c "import gguf, accelerate" 2>/dev/null || python3 -m pip -q install --root-user-action=ignore \
      /content/llamacpp_b6000/gguf-py "accelerate==1.1.1" > "$OUT/pip_gguf.log" 2>&1
    flock -u 9
    for t in "$@"; do
      SPEC=$(python3 tools/colab_chat/bench_stage.py "$t" --gguf)
      python3 tools/bench_eval.py run --items $ITEMS --out "$OUT/bench" --scorer hfgguf --model "$t=$SPEC" \
        --blimp-per "$BP" >> "$OUT/hfgguf_$t.log" 2>&1
      python3 tools/bench_eval.py run --items $ITEMS --out "$OUT/xcheck_mtstudio" --scorer mtstudio --mtstudio $MT \
        --model "$t=$SPEC" --blimp-per "$BP" --every "$EVERY" --chunk 128 >> "$OUT/xcheck_$t.log" 2>&1
      python3 tools/bench_eval.py compare "$OUT/xcheck_mtstudio/$t" "$OUT/bench/$t" \
        --json "$OUT/compare_${t}_mtstudio_vs_hfgguf.json" >> "$OUT/xcheck_$t.log" 2>&1
    done
    cp /content/eval/staged.jsonl "$OUT/" 2>/dev/null || true
    ;;
  ember)
    # ember's POST /score (BENCH_SCORE_COMMIT, gated by ember on 26 Oct 2026; CUDA scoring runs FP32 GEMMs;
    # EMBER_DETERMINISTIC from the job) on each model's GGUF, every EVERY-th item; its own build directory, so
    # the iftiny job's ember build at another commit is left alone.
    BP=$1
    EVERY=$2
    shift 2
    TINY_CHAT_EMBER_DIR=/content/ember_score python3 tools/colab_chat/ensure_ember_vm.py "$OUT/ember_build.json" \
      "$OUT/ember_build.log" "${BENCH_SCORE_COMMIT:-}"
    BIN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bin"])' "$OUT/ember_build.json")
    export EMBER_GPU_LAYERS=-1
    mkdir -p "$OUT/ember_cwd" "$OUT/www"
    ln -sf /dev/null "$OUT/ember_cwd/debugging.log"
    for t in "$@"; do
      SPEC=$(python3 tools/colab_chat/bench_stage.py "$t" --gguf)
      G=$(python3 -c 'import json, os, sys; s = json.load(open(sys.argv[1])); print(os.path.join(s["out_dir"], s["name"] + ".gguf"))' "$SPEC")
      (cd "$OUT/ember_cwd" && exec "$BIN" "$G" 8097 127.0.0.1 "$OUT/www" -1) 2>> "$OUT/ember_server_$t.log" &
      SRV=$!
      for _ in $(seq 1 300); do
        python3 -c 'import sys, urllib.request; sys.stdout.write(urllib.request.urlopen("http://127.0.0.1:8097/health", timeout=2).read().decode())' \
          > "$OUT/health_$t.json" 2>/dev/null && break
        sleep 1
      done
      python3 tools/bench_eval.py run --items $ITEMS --out "$OUT/bench" --scorer ember --ember-url http://127.0.0.1:8097 \
        --model "$t=$SPEC" --blimp-per "$BP" --every "$EVERY" >> "$OUT/ember_$t.log" 2>&1 || { kill $SRV; exit 1; }
      kill $SRV
      wait $SRV 2>/dev/null || true
    done
    cp /content/eval/staged.jsonl "$OUT/" 2>/dev/null || true
    ;;
  hf)
    BP=$1
    shift
    hf_deps
    for m in "$@"; do
      python3 tools/bench_eval.py run --items $ITEMS --out "$OUT/bench" --scorer hf --hf "$m" --blimp-per "$BP" \
        >> "$OUT/hf_${m%%=*}.log" 2>&1
    done
    ;;
  iftiny)
    python3 tools/colab_chat/ensure_ember_vm.py "$OUT/ember_build.json" "$OUT/ember_build.log" "${BENCH_EMBER_COMMIT:-}"
    BIN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bin"])' "$OUT/ember_build.json")
    export EMBER_GPU_LAYERS=-1 EMBER_GEMM_TF32=0
    for t in "$@"; do
      SPEC=$(python3 tools/colab_chat/bench_stage.py "$t" --gguf)
      python3 tools/iftiny_eval.py run --gen ember --ember-bin "$BIN" --model "$t=$SPEC" --out "$OUT/iftiny" \
        --passes 2 --mtstudio $MT >> "$OUT/iftiny_$t.log" 2>&1
    done
    cp /content/eval/staged.jsonl "$OUT/" 2>/dev/null || true
    ;;
  iftiny_hf)
    hf_deps
    for m in "$@"; do
      python3 tools/iftiny_eval.py run --gen hf --hf "$m" --out "$OUT/iftiny" >> "$OUT/iftiny_hf_${m%%=*}.log" 2>&1
    done
    ;;
  smolgguf)
    # SmolLM2-135M as an F32 GGUF (llama.cpp b6000's converter and gguf-py), for ember's /score gate.
    REPO=${1%@*}
    REV=${1#*@}
    hf_deps
    [ -d /content/llamacpp_b6000 ] || git clone -q --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp       /content/llamacpp_b6000
    flock 9
    python3 -m pip -q install --root-user-action=ignore /content/llamacpp_b6000/gguf-py sentencepiece > "$OUT/pip_gguf.log" 2>&1
    flock -u 9
    python3 -c "from huggingface_hub import snapshot_download as s; import sys; print(s(sys.argv[1], revision=sys.argv[2], local_dir='/content/hf_smollm2'))"       "$REPO" "$REV" > "$OUT/download.log" 2>&1
    python3 /content/llamacpp_b6000/convert_hf_to_gguf.py /content/hf_smollm2 --outtype f32       --outfile "$OUT/SmolLM2-135M-f32.gguf" > "$OUT/convert.log" 2>&1
    python3 -c "import hashlib, json, sys; print(json.dumps({'repo': sys.argv[1], 'revision': sys.argv[2], 'converter': 'llama.cpp b6000 convert_hf_to_gguf.py --outtype f32', 'sha256': hashlib.sha256(open(sys.argv[3], 'rb').read()).hexdigest()}))"       "$REPO" "$REV" "$OUT/SmolLM2-135M-f32.gguf" > "$OUT/smolgguf.json"
    ;;
  *)
    echo "unknown subcommand $CMD" >&2
    exit 2
    ;;
esac
pack
echo "results: $OUT/results.tar.gz"
