#!/usr/bin/env bash
# Colab data job: train the next base's llama3-scheme vocabulary on FineWeb-Edu + lever 2's dialogue
# text, then prove llama.cpp tokenises it identically (vocab_llama3_bpe.py check with llama.cpp's own
# llama-tokenize, built here at a pinned release).
#   bash corpus/fetchers/vocab_llama3_job.sh OUT FINEWEB_TXT LEVER2_DIR
# LEVER2_DIR holds dialogue_{soda,oasst,ultrachat,everyday}_v1/text.txt (dialogue_pt.py's outputs).
# Training sample: 140 MB of FineWeb-Edu and 60 MB of dialogue (SODA 20, UltraChat 25, oasst 10,
# everyday up to 5), each cut at a document boundary; held out: the 200 kB after each cut.
# Outputs in OUT: tokenizer.json, vocab_llama3_bpe.gguf, train_ledger.json, check.json.
# With MIX_MAIN and MIX_TAIL set (mix_pretrain.py ratio lists, e.g. MIX_MAIN=fineweb=0.7,soda=0.08,...),
# and the check passed, it also writes the next base's corpus: mix.main.txt / mix.tail.txt /
# mix.ledger.json (corpus/mix_pretrain.py, TAIL_FRACTION default 0.15, MIX_SEED default 1, MIX_BYTES
# default max) and mix.main.ids / mix.tail.ids (tools/corpus_ids.py with the trained vocabulary).
set -euo pipefail
OUT=${1:?out dir}
FW=${2:?FineWeb-Edu text}
L2=${3:?lever 2 output dir}
LLAMA_TAG=b11460   # ggml-org/llama.cpp release of 2026-10-07
mkdir -p "$OUT"
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-6}
SRC=(--source "$FW=146800640"
     --source "$L2/dialogue_soda_v1/text.txt=20971520"
     --source "$L2/dialogue_ultrachat_v1/text.txt=26214400"
     --source "$L2/dialogue_oasst_v1/text.txt=10485760"
     --source "$L2/dialogue_everyday_v1/text.txt=5242880")
python3 corpus/fetchers/vocab_llama3_bpe.py train "${SRC[@]}" --size 16384 --out "$OUT"
LC="$OUT/llama.cpp"
if [ ! -x "$LC/build/bin/llama-tokenize" ]; then
  rm -rf "$LC"
  git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LC"
  cmake -S "$LC" -B "$LC/build" -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release > "$OUT/llama_cmake.log"
  cmake --build "$LC/build" -j 6 --target llama-tokenize > "$OUT/llama_build.log"
fi
HELD=()
for s in "${SRC[@]}"; do [ "$s" = --source ] || HELD+=(--heldout "$s"); done
python3 corpus/fetchers/vocab_llama3_bpe.py check --dir "$OUT" "${HELD[@]}" --llama-tokenize "$LC/build/bin/llama-tokenize"
if [ -n "${MIX_MAIN:-}" ]; then
  python3 corpus/mix_pretrain.py "$OUT/mix" --source fineweb="$FW" \
    --source soda="$L2/dialogue_soda_v1/text.txt" --source ultrachat="$L2/dialogue_ultrachat_v1/text.txt" \
    --source oasst="$L2/dialogue_oasst_v1/text.txt" --source everyday="$L2/dialogue_everyday_v1/text.txt" \
    --main "$MIX_MAIN" --tail "${MIX_TAIL:?MIX_TAIL}" --tail-fraction "${TAIL_FRACTION:-0.15}" \
    --total-bytes "${MIX_BYTES:-max}" --seed "${MIX_SEED:-1}"
  for part in main tail; do
    python3 tools/corpus_ids.py "$OUT/mix.$part.txt" "$OUT/vocab_llama3_bpe.gguf" "$OUT/mix.$part.ids" --threads 6
  done
fi
