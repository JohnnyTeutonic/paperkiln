#!/usr/bin/env bash
# The stronger base's data job (AI_ML/tiny_chat_benchmarks/stronger_base_spec.md, section 10). Claude Opus 5.5, 7 Nov 2026.
#   bash tools/colab_chat/base_b1_data.sh OUT VOCAB_GGUF stage0|b1
# S-full's data job first (the same panels, SFT rows and mix_s35.ids, so one job proves the split is intact), then the long
# sets, lever 2's dialogue sets and FineWeb-Edu shard 1, every source pinned; each dialogue source filtered against the
# frozen eval sets, Carry-5, Carry-12 and the held-out split (tc_b1_filter.py); then the 6.6 GB mix and its .ids. stage0 records
# the mix's hashes; b1 refuses unless they equal Stage 0's (tc_b1_pins.py). Idempotent: a rerun skips what is finished.
set -euo pipefail
OUT=${1:?out dir}
V=${2:?vocab gguf}
MODE=${3:?stage0 or b1}
S13=oasst,convai2,wow,bst,taskmaster,multiwoz,self_dialogue,topical_chat,dolly,quac,empathetic,dailydialog,no_robots
LONG=taskmaster3,abcd,ccpe,p4g,aya_en,msc
# Byte ratios, from Stage 1's measured filtered sizes (spec section 10): main 73/27, tail 53/47.
MAIN="fineweb=0.730000,human=0.051332,soda=0.090613,ultrachat=0.127815,everyday=0.000240"
TAIL="fineweb=0.530000,human=0.089356,soda=0.157733,ultrachat=0.222493,everyday=0.000418"
mkdir -p "$OUT"

# A pod lost mid-way must not leave half a step: the fetch runs here, whole, behind a marker, and tiny_chat_data.py trusts its
# three split files by their presence, so they go until its own run has finished.
if [ ! -e "$OUT/human/.fetched" ]; then
  rm -rf "$OUT/human"
  python3 corpus/fetchers/human_dialogue.py "$S13" --allow-sa --allow-nc --out-dir "$OUT/human"
  touch "$OUT/human/.fetched"
fi
[ -e "$OUT/.split_done" ] || rm -f "$OUT/human_pt.txt" "$OUT/human_val.txt" "$OUT/human_sft.jsonl"
python3 tools/colab_chat/tiny_chat_data.py --out "$OUT" --vocab "$V" --fineweb corpus/data/fineweb_edu_shard0_v1 \
  --shares 35 --mix-bytes 800000000 --max-epochs 4 --T 1024 --seed 1 --require-pinned --carry12 --long-sft 10:4 \
  --sources "$S13" \
  --human-fetch "python3 corpus/fetchers/human_dialogue.py $S13 --allow-sa --allow-nc --out-dir {human}"
touch "$OUT/.split_done"

# msc has no stated licence and aya_en carries a documented LLM risk: both in by Jonathan's ruling of 7 Nov 2026, recorded
# per source by the fetchers' ledgers and filter.ledger.json.
# Each ledger is written last, so it marks a whole fetch.
[ -s "$OUT/long/human_dialogue_v1.ledger.json" ] || \
  python3 corpus/fetchers/human_dialogue.py "$LONG" --allow-llm-risk --allow-unstated-licence --out-dir "$OUT/long"
[ -s "$OUT/lever2/dialogue_pt_v1.ledger.json" ] || python3 corpus/fetchers/dialogue_pt.py all --out-dir "$OUT/lever2"
python3 corpus/fetchers/fineweb_edu.py shard1_v1  # no --pin: the committed pin checks these bytes

F="$OUT/filtered"
if [ ! -s "$F/filter.ledger.json" ]; then
  python3 tools/colab_chat/tc_b1_filter.py --repo . --heldout "$OUT/human_val.txt" "$OUT/human_sft.jsonl" --out "$F" \
    --source "soda=$OUT/lever2/dialogue_soda_v1/dialogues.jsonl" \
    --source "ultrachat=$OUT/lever2/dialogue_ultrachat_v1/dialogues.jsonl" \
    --source "everyday=$OUT/lever2/dialogue_everyday_v1/dialogues.jsonl" \
    --source "taskmaster3=$OUT/long/human_taskmaster3_v1/dialogues.jsonl" \
    --source "abcd=$OUT/long/human_abcd_v1/dialogues.jsonl" \
    --source "ccpe=$OUT/long/human_ccpe_v1/dialogues.jsonl" \
    --source "p4g=$OUT/long/human_p4g_v1/dialogues.jsonl" \
    --source "aya_en=$OUT/long/human_aya_en_v1/dialogues.jsonl" \
    --source "msc=$OUT/long/human_msc_v1/dialogues.jsonl"
fi
cp "$F/filter.ledger.json" "$OUT/filter.ledger.json"
python3 tools/colab_chat/tc_b1_pins.py "$OUT" sources  # a wrong source stops here, not after half an hour of mixing

if [ ! -s "$OUT/mix_b1.main.ids" ] || [ ! -s "$OUT/mix_b1.tail.ids" ]; then
  # Two copies of the human text; the long sets' documents follow S-full's pretraining split.
  HUMAN="$OUT/human2.txt"
  LONGF="$F/taskmaster3.txt $F/abcd.txt $F/ccpe.txt $F/p4g.txt $F/aya_en.txt $F/msc.txt"
  # shellcheck disable=SC2086
  [ -s "$HUMAN" ] || { cat "$OUT/human_pt.txt" $LONGF "$OUT/human_pt.txt" $LONGF > "$HUMAN.tmp" && mv "$HUMAN.tmp" "$HUMAN"; }
  FW="$OUT/fineweb01.txt"
  [ -s "$FW" ] || python3 tools/colab_chat/tc_b1_fineweb.py "$FW" --val corpus/data/fineweb_edu_shard0_v1/val.txt \
    corpus/data/fineweb_edu_shard0_v1/pretrain_train.txt corpus/data/fineweb_edu_shard1_v1/pretrain_train.txt
  cp "$FW.ledger.json" "$OUT/fineweb_b1.ledger.json"
  python3 corpus/mix_pretrain.py "$OUT/mix_b1" --seed 1 --tail-fraction 0.15 --total-bytes 6600000000 \
    --source "fineweb=$FW" --source "human=$HUMAN" --source "soda=$F/soda.txt" --source "ultrachat=$F/ultrachat.txt" \
    --source "everyday=$F/everyday.txt" --main "$MAIN" --tail "$TAIL"
  : > "$OUT/ids_b1.json.tmp"
  for part in main tail; do
    python3 tools/corpus_ids.py "$OUT/mix_b1.$part.txt" "$V" "$OUT/mix_b1.$part.ids.tmp" --threads 6 >> "$OUT/ids_b1.json.tmp"
    mv "$OUT/mix_b1.$part.ids.tmp" "$OUT/mix_b1.$part.ids"
  done
  mv "$OUT/ids_b1.json.tmp" "$OUT/ids_b1.json"
  rm -f "$FW" "$HUMAN"
  # The ledger holds the text's hashes and the ids are what trains. TC_B1_KEEP_MIX_TEXT keeps it for a scan of the final mix.
  [ "${TC_B1_KEEP_MIX_TEXT:-0}" = 1 ] || rm -f "$OUT/mix_b1.main.txt" "$OUT/mix_b1.tail.txt"
fi
python3 tools/colab_chat/tc_b1_pins.py "$OUT" "$MODE"
