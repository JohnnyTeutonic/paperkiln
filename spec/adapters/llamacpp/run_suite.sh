#!/usr/bin/env bash
# The suite against llama.cpp in the configurations FINDINGS.txt reports, from the keramos root, in WSL.
#
#   BIN=~/lc_keramos/llamacpp_adapter OUT=~/lc_keramos bash adapters/llamacpp/run_suite.sh
#
# receipt   --alias mistral=llama, f32 kv cache, no flash attention (conformance/receipts/llamacpp_cpu.json)
# noalias   the suite's files as they are (tp_mistral's architecture "mistral" is not a llama.cpp one)
# kvf16     llama.cpp's default kv cache type, f16
# fa        flash attention on (f32 kv)
# fa_kvf16  flash attention on, f16 kv
# Each run takes about 12 s on 2 threads; the probe (probe.py) about 10 s.
set -uo pipefail
BIN="${BIN:-$HOME/lc_keramos/llamacpp_adapter}"
OUT="${OUT:-$HOME/lc_keramos}"
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
run() {
    local name="$1"; shift
    python3 tools/ir_conformance.py --adapter "cmd:$BIN --threads 2 $*" --json "$OUT/run_$name.json" \
        > "$OUT/run_$name.out" 2> "$OUT/run_$name.err"
    echo "$name: $(tail -1 "$OUT/run_$name.out")"
}
run receipt --alias mistral=llama
run noalias
run kvf16 --alias mistral=llama --kv-type f16
run fa --alias mistral=llama --flash-attn 1
run fa_kvf16 --alias mistral=llama --flash-attn 1 --kv-type f16
python3 adapters/llamacpp/probe.py --bin "$BIN" --json "$OUT/probe.json" > "$OUT/probe.out" 2> "$OUT/probe.err"
echo "probe: $(wc -l < "$OUT/probe.out") lines in $OUT/probe.out"
