#!/usr/bin/env bash
# The suite against ONNX Runtime GenAI in the configurations FINDINGS.txt reports, from the keramos root, in WSL.
#
#   PY=~/ort_keramos/venv/bin/python OUT=~/ort_keramos/runs [TF4_PY=~/ort_keramos/venv_tf4/bin/python] \
#       bash adapters/ortgenai/run_suite.sh
#
# receipt   the builder as shipped (-p fp32 -e cpu), export input (conformance/receipts/ortgenai_cpu.json)
# unfused   GroupQueryAttention without fused rotary (a separate RotaryEmbedding node; fuse_qkv=false) and the
#           transformers 4 view of Gemma 3's config: the configuration in which head_size 8 runs on CPU
# fixed     unfused, plus Gemma 3's local RoPE caches without the global linear factor (the fix of FINDINGS 3)
# mha       MultiHeadAttention in place of GroupQueryAttention: the configuration in which head_size 4 runs
# gguf      unfused, the builder reading the GGUF itself (its loaders/gguf.py) with the export's config
# Builds are cached under ~/ort_keramos/onnx; a first run builds every model (a few seconds each), then
# each run takes well under a minute on 2 threads. Then probe.py (with TF4_PY, its gemma3_tf4 section too).
set -uo pipefail
PY="${PY:-$HOME/ort_keramos/venv/bin/python}"
OUT="${OUT:-$HOME/ort_keramos/runs}"
TF4_PY="${TF4_PY:-}"
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=
run() {
    local name="$1"; shift
    nice -n 10 "$PY" adapters/ortgenai/ortgenai_adapter.py --adapter ortgenai "$@" --json "$OUT/run_$name.json" \
        > "$OUT/run_$name.out" 2> "$OUT/run_$name.err"
    echo "$name: $(tail -1 "$OUT/run_$name.out")"
}
run receipt
run unfused --ort-patch unfused_rope,gemma3_config
run fixed --ort-patch unfused_rope,gemma3_config,local_rope_unscaled
run mha --ort-patch no_gqa,gemma3_config
run gguf --ort-loader gguf --ort-patch unfused_rope,gemma3_config
nice -n 10 "$PY" adapters/ortgenai/probe.py ${TF4_PY:+--tf4-py "$TF4_PY"} --json "$OUT/probe.json" \
    > "$OUT/probe.out" 2> "$OUT/probe.err"
echo "probe: $(wc -l < "$OUT/probe.out") lines in $OUT/probe.out"
