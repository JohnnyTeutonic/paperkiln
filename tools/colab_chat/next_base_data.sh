#!/usr/bin/env bash
# Colab data job for the next base (LOCAL_LLM_STACK.md section 7, levers 2-3):
# mix FineWeb-Edu with lever 2's dialogue text (corpus/mix_pretrain.py: a
# main part and an annealing tail, documents interleaved by seed) and
# pre-tokenise both parts with the llama3 vocabulary (tools/corpus_ids.py),
# so mtstudio reads .ids instead of tokenising gigabytes at every resume.
#   bash tools/colab_chat/next_base_data.sh OUT VOCAB_GGUF LEVER2_DIR FINEWEB_TXT [FINEWEB_TXT...]
# Env (set by the jobs file, from tools/colab_chat/make_next_base.py):
# MIX_MAIN, MIX_TAIL (ratio lists), MIX_BYTES (total), TAIL_FRACTION, MIX_SEED.
# Outputs in OUT: mix.main.ids, mix.tail.ids, mix.ledger.json, ids.json.
set -euo pipefail
OUT=${1:?out dir}
VOCAB=${2:?vocab gguf}
L2=${3:?lever 2 output dir}
shift 3
mkdir -p "$OUT"
FW="$OUT/fineweb.txt"
# Every shard file ends with "<|endoftext|>\n", so concatenation keeps documents whole.
[ -s "$FW" ] || cat "$@" > "$FW"
python3 corpus/mix_pretrain.py "$OUT/mix" --source fineweb="$FW" \
  --source soda="$L2/dialogue_soda_v1/text.txt" --source ultrachat="$L2/dialogue_ultrachat_v1/text.txt" \
  --source oasst="$L2/dialogue_oasst_v1/text.txt" --source everyday="$L2/dialogue_everyday_v1/text.txt" \
  --main "${MIX_MAIN:?MIX_MAIN}" --tail "${MIX_TAIL:?MIX_TAIL}" --tail-fraction "${TAIL_FRACTION:-0.15}" \
  --total-bytes "${MIX_BYTES:?MIX_BYTES}" --seed "${MIX_SEED:-1}"
: > "$OUT/ids.json"
for part in main tail; do
  python3 tools/corpus_ids.py "$OUT/mix.$part.txt" "$VOCAB" "$OUT/mix.$part.ids" --threads 6 >> "$OUT/ids.json"
done
rm -f "$FW" "$OUT/mix.main.txt" "$OUT/mix.tail.txt"  # the ledger records their hashes; the ids are what trains
