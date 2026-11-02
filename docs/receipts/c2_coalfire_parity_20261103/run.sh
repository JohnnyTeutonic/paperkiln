#!/usr/bin/env bash
# C2 end to end, CPU only. Usage: bash run.sh COALFIRE_CHECKOUT COALFIRE_BUILD PAPERKILN_BUILD OUT
#  1. a small word-level corpus from coalfire's own data/training_pairs.txt (split in two, '|' to a space);
#  2. coalfire trains its --tiny llama-family model for 40 steps, saving the final state as a checkpoint
#     (--pol-save) and as an HF directory (--export-hf) from the same in-memory model;
#  3. coalfire's forward_batch on the checkpoint and paperkiln's Llama on the HF export, over 48 fixed ids;
#  4. compare.py: both against an independent float64 forward of the HF export.
set -euo pipefail
C=$1; CB=$2; PB=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/data"
n=$(wc -l < "$C/data/training_pairs.txt")
head -n $(( n / 2 )) "$C/data/training_pairs.txt" | tr '|' ' ' > "$OUT/data/train-0-small.txt"
tail -n +$(( n / 2 + 1 )) "$C/data/training_pairs.txt" | tr '|' ' ' > "$OUT/data/train-1-small.txt"
cp "$OUT/data/train-0-small.txt" "$OUT/data/train-0.txt"   # train.cpp reads the -small names in this mode
cp "$OUT/data/train-1-small.txt" "$OUT/data/train-1.txt"
tr '|' ' ' < "$C/data/validation_pairs.txt" > "$OUT/data/validation.txt"
(cd "$C" && "$CB/train" "$OUT/data" --tiny --max-steps 40 --seq-len 64 --output-dir "$OUT/ckpts" \
    --pol-save "$OUT/pol" --export-hf "$OUT/hf") > "$OUT/train.log" 2>&1
ls "$OUT/pol/pol_checkpoint_0.ckpt" "$OUT/hf/config.json" "$OUT/hf/model.safetensors" > /dev/null
python3 - "$OUT/hf/config.json" "$OUT/ids.txt" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))["vocab_size"]
open(sys.argv[2], "w").write(" ".join(str(2 + (i * 7919 + 13) % (v - 2)) for i in range(48)) + "\n")
PY
J=$(find "$CB" "$PB" -path "*json-src/include" -type d 2>/dev/null | head -1)   # nlohmann/json, fetched by the builds
g++ -O2 -std=c++17 -fopenmp -I "$C/include" -I "$C/third_party" ${J:+-I "$J"} "$HERE/coalfire_dump.cpp" \
    "$CB/libtransformer_core.a" -o "$OUT/coalfire_dump"
"$OUT/coalfire_dump" "$OUT/pol/pol_checkpoint_0.ckpt" "$OUT/data/train-0-small.txt" "$OUT/data/train-1-small.txt" 64 \
    "$OUT/ids.txt" "$OUT/coalfire.f32" | tee "$OUT/coalfire_dump.json"
"$PB/c2_parity" "$OUT/hf" "$OUT/ids.txt" "$OUT/paperkiln.f32" | tee "$OUT/paperkiln.json"
python3 "$HERE/compare.py" "$OUT/hf" "$OUT/ids.txt" "$OUT/coalfire.f32" "$OUT/paperkiln.f32" "$OUT/result.json"
sha256sum "$OUT/pol/pol_checkpoint_0.ckpt" "$OUT/hf/model.safetensors" "$OUT/hf/config.json" "$OUT/ids.txt" \
    "$OUT/coalfire.f32" "$OUT/paperkiln.f32" > "$OUT/sha256.txt"
