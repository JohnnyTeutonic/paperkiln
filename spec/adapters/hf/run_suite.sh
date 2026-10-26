#!/usr/bin/env bash
# The suite against transformers in the configurations FINDINGS.txt reports, from the keramos root, in WSL.
#
#   PY=~/hf_keramos/venv/bin/python OUT=~/hf_keramos/runs bash adapters/hf/run_suite.sh
#
# receipt    sdpa (transformers' default), positions from the cache (conformance/receipts/hf_cpu.json)
# eager      eager attention
# explicit   sdpa, position_ids passed (len .. len + n - 1)
# gguf       transformers' own GGUF reader (from_pretrained(gguf_file=...)) in place of the keramos exporter;
#            tp_mistral as llama.cpp's convert writes Mistral (architecture llama, general.name Mistral-7B-v0.1)
# Each run takes well under a minute on 2 threads; then probe.py.
set -uo pipefail
PY="${PY:-$HOME/hf_keramos/venv/bin/python}"
OUT="${OUT:-$HOME/hf_keramos/runs}"
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TQDM_DISABLE=1
run() {
    local name="$1"; shift
    nice -n 10 "$PY" adapters/hf/hf_adapter.py --adapter hf "$@" --json "$OUT/run_$name.json" \
        > "$OUT/run_$name.out" 2> "$OUT/run_$name.err"
    echo "$name: $(tail -1 "$OUT/run_$name.out")"
}
run receipt
run eager --hf-attn eager
run explicit --hf-positions explicit
run gguf --hf-loader gguf --hf-gguf-name Mistral-7B-v0.1
nice -n 10 "$PY" adapters/hf/probe.py --json "$OUT/probe.json" > "$OUT/probe.out" 2> "$OUT/probe.err"
echo "probe: $(wc -l < "$OUT/probe.out") lines in $OUT/probe.out"
