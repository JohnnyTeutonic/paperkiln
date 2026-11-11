#!/usr/bin/env bash
# Base 2's corpus build (BASE2_PLAN.md sections 2.3 and 2.4, in the author's research tree). Claude Fable 5.2, 11 Nov 2026.
#   bash tools/colab_chat/base2_data.sh OUT VOCAB_GGUF stage1|stage2|stage3|all|rehearsal
# Three stages, each pushed to the store when it is whole, so a lost session repeats one stage at most:
#   stage1  S-full's data job first, unchanged (the same panels and split as base A's), the long sets, lever 2's dialogue
#           sets, every dialogue source filtered against the frozen eval sets, Carry-5, Carry-12 and the held-out split
#           (tc_b1_filter.py), FineWeb-Edu shards 2 to 5 joined without a document of the FineWeb panel, Simple English
#           Wikipedia through the same two rules. Store: base2-build:build/src (the text), main:corpus/panels.
#   stage2  the natural mix (arm N), its ratios set from stage 1's measured sizes, and its ids. Store: main:corpus/N.
#   stage3  the scenario stream from the pinned engine, the held-out scenario panel, the scenario mix (arms S and C) with
#           the stream at 15% of tokens by the measured bytes a token, and its ids. Store: main:corpus/S,
#           main:corpus/panel_scenario.
# A stage is finished when OUT/base2_stageN.json exists or the store holds base2-build:build/stageN; a rerun skips it, and a
# stage that finds its inputs missing pulls them from the store. Inside a stage every step is behind its output, written
# under a temporary name first. rehearsal builds the ids of the 50 MB FineWeb-Edu sample (main:corpus/rehearsal).
# Environment (defaults are the campaign's): BASE2_STORE_CONFIG, BASE2_PINS, BASE2_ENGINE_DIR,
# BASE2_THREADS, BASE2_JOBS, BASE2_SCN_SEED, BASE2_SCN_SHARD_DOCS, BASE2_SCN_HELDOUT_DOCS, BASE2_SCN_MAX_DROP,
# BASE2_SCN_VALIDATE, BASE2_MIN_FREE_GB, BASE2_TIMING_BYTES, BASE2_KEEP_MIX_TEXT. BASE2_FIXTURE=DIR stands in for the
# network fetches of stage 1 with a folder of the same files (tests).
set -euo pipefail
OUT=${1:?out dir}
V=${2:?vocab gguf}
MODE=${3:?stage1, stage2, stage3, all or rehearsal}
CC=tools/colab_chat
STORE=${BASE2_STORE_CONFIG:-/content/chat/store.json}
BB=base2-build
ENGINE=${BASE2_ENGINE_DIR:-/content/chat/upload/scenario_engine}
THREADS=${BASE2_THREADS:-10}
JOBS=${BASE2_JOBS:-8}
SEED=${BASE2_SCN_SEED:-20261111}
SHARD_DOCS=${BASE2_SCN_SHARD_DOCS:-200000}
HELD_DOCS=${BASE2_SCN_HELDOUT_DOCS:-20000}
MAX_DROP=${BASE2_SCN_MAX_DROP:-0.02}
MIN_FREE_GB=${BASE2_MIN_FREE_GB:-80}
TIMING_BYTES=${BASE2_TIMING_BYTES:-200000000}
S13=oasst,convai2,wow,bst,taskmaster,multiwoz,self_dialogue,topical_chat,dolly,quac,empathetic,dailydialog,no_robots
LONG=taskmaster3,abcd,ccpe,p4g,aya_en,msc
FW_SHARDS="2 3 4 5"   # unseen by base A, whose mix read shards 0 and 1
SRC="$OUT/src"
WORK="$OUT/work"
mkdir -p "$OUT" "$SRC"
B2() { python3 $CC/base2_corpus.py "$@"; }
PINS() { python3 $CC/base2_pins.py "$@"; }
say() { echo "base2_data $(date -u +%H:%M:%S) $*"; }

push_set() {  # BRANCH PREFIX FILE...: the files as one set, one commit
  local branch=$1 prefix=$2 d="$OUT/.push"
  shift 2
  rm -rf "$d" && mkdir -p "$d"
  for f in "$@"; do ln -f "$f" "$d/$(basename "$f")" 2>/dev/null || cp "$f" "$d/$(basename "$f")"; done
  python3 $CC/hf_store.py push --config "$STORE" --branch "$branch" --prefix "$prefix" --dir "$d" > "$OUT/push_$(basename "$prefix").json"
  rm -rf "$d"
  say "pushed $branch:$prefix"
}
pull_set() {  # BRANCH PREFIX DIR
  python3 $CC/hf_store.py pull --config "$STORE" --branch "$1" --prefix "$2" --to "$3" > /dev/null
  say "pulled $1:$2"
}
stage_done() {  # N: 0 when the stage is finished here or in the store (its record is then pulled)
  [ -s "$OUT/base2_stage$1.json" ] && return 0
  local rc=0
  B2 store-has "$STORE" "$BB" "build/stage$1" || rc=$?
  [ "$rc" = 1 ] && return 1
  [ "$rc" = 0 ] || { echo "the store could not be read; stopping"; exit 1; }
  pull_set "$BB" "build/stage$1" "$OUT"
}
need_src() {
  for f in fineweb2345.txt simplewiki.txt human1.txt soda.txt ultrachat.txt everyday.txt human_val.txt human_sft.jsonl; do
    [ -s "$SRC/$f" ] || { pull_set "$BB" build/src "$SRC"; break; }
  done
}
copies() {  # N OUT FILE: FILE N times
  local n=$1 out=$2 f=$3 i
  [ -s "$out" ] && return 0
  : > "$out.tmp"
  for i in $(seq "$n"); do cat "$f" >> "$out.tmp"; done
  mv "$out.tmp" "$out"
}
dialogue_copies() {  # section 2.3: human three copies, the other dialogue sources two
  mkdir -p "$WORK"
  copies 3 "$WORK/human3.txt" "$SRC/human1.txt"
  copies 2 "$WORK/soda2.txt" "$SRC/soda.txt"
  copies 2 "$WORK/ultrachat2.txt" "$SRC/ultrachat.txt"
  copies 2 "$WORK/everyday2.txt" "$SRC/everyday.txt"
}
NATURAL_SOURCES() {
  echo --source "fineweb=$SRC/fineweb2345.txt" --source "simplewiki=$SRC/simplewiki.txt" --source "human=$WORK/human3.txt" \
    --source "soda=$WORK/soda2.txt" --source "ultrachat=$WORK/ultrachat2.txt" --source "everyday=$WORK/everyday2.txt"
}
mix_and_ids() {  # ARM RECIPE EXTRA_SOURCE...: OUT/ARM.{main,tail}.ids, ARM.ledger.json, ids_ARM.json; the text stays
  local arm=$1 recipe=$2 part
  shift 2
  if [ ! -s "$OUT/$arm.ledger.json" ] || [ ! -s "$OUT/$arm.main.txt" ] || [ ! -s "$OUT/$arm.tail.txt" ]; then
    rm -f "$OUT/$arm.ledger.json"
    # shellcheck disable=SC2046
    python3 corpus/mix_pretrain.py "$OUT/$arm" --seed 1 --tail-fraction "$(B2 get "$recipe" tail_fraction)" \
      --total-bytes "$(B2 get "$recipe" total_bytes)" $(NATURAL_SOURCES) "$@" \
      --main "$(B2 get "$recipe" main_arg)" --tail "$(B2 get "$recipe" tail_arg)"
  fi
  : > "$OUT/ids_$arm.json.tmp"
  for part in main tail; do
    python3 tools/corpus_ids.py "$OUT/$arm.$part.txt" "$V" "$OUT/$arm.$part.ids.tmp" --threads "$THREADS" >> "$OUT/ids_$arm.json.tmp"
    mv "$OUT/$arm.$part.ids.tmp" "$OUT/$arm.$part.ids"
  done
  sed -i "s/\.ids\.tmp\"/.ids\"/" "$OUT/ids_$arm.json.tmp"
  mv "$OUT/ids_$arm.json.tmp" "$OUT/ids_$arm.json"
}
drop_mix_text() { [ "${BASE2_KEEP_MIX_TEXT:-0}" = 1 ] || rm -f "$OUT/$1.main.txt" "$OUT/$1.tail.txt"; }

fetch_sources() {  # the network half of stage 1, as base_b1_data.sh fetches for base A
  python3 -c "import pyarrow, pandas, huggingface_hub, tokenizers" 2>/dev/null || pip -q install huggingface_hub pandas pyarrow tokenizers
  python3 corpus/fetchers/fineweb_edu.py shard0_v1   # the committed pin checks these bytes; its val.txt is the FineWeb panel
  if [ ! -e "$OUT/human/.fetched" ]; then
    rm -rf "$OUT/human"
    python3 corpus/fetchers/human_dialogue.py "$S13" --allow-sa --allow-nc --out-dir "$OUT/human"
    touch "$OUT/human/.fetched"
  fi
  [ -e "$OUT/.split_done" ] || rm -f "$OUT/human_pt.txt" "$OUT/human_val.txt" "$OUT/human_sft.jsonl"
  python3 $CC/tiny_chat_data.py --out "$OUT" --vocab "$V" --fineweb corpus/data/fineweb_edu_shard0_v1 \
    --shares 35 --mix-bytes 800000000 --max-epochs 4 --T 1024 --seed 1 --require-pinned --carry12 --long-sft 10:4 \
    --sources "$S13" \
    --human-fetch "python3 corpus/fetchers/human_dialogue.py $S13 --allow-sa --allow-nc --out-dir {human}"
  touch "$OUT/.split_done"
  # msc and aya_en: in by Jonathan's ruling of 7 Nov 2026, as for base A. Each ledger is written last, so it marks a whole fetch.
  [ -s "$OUT/long/human_dialogue_v1.ledger.json" ] || \
    python3 corpus/fetchers/human_dialogue.py "$LONG" --allow-llm-risk --allow-unstated-licence --out-dir "$OUT/long"
  [ -s "$OUT/lever2/dialogue_pt_v1.ledger.json" ] || python3 corpus/fetchers/dialogue_pt.py all --out-dir "$OUT/lever2"
  local F="$OUT/filtered"
  if [ ! -s "$F/filter.ledger.json" ]; then
    python3 $CC/tc_b1_filter.py --repo . --heldout "$OUT/human_val.txt" "$OUT/human_sft.jsonl" --out "$F" \
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
  PINS "$OUT" sources   # the bytes base A's data job met: a wrong source stops here
  # Shards 2 to 5 are pinned on this first fetch; their hashes files go home from OUT and are committed. A shard's
  # parquet leaves the cache once its text is written, so the disk holds one at a time.
  local n
  if [ ! -s "$SRC/fineweb2345.txt" ]; then
    for n in $FW_SHARDS; do
      python3 corpus/fetchers/fineweb_edu.py "shard${n}_v1" --pin
      rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--HuggingFaceFW--fineweb-edu"
    done
  fi
  for n in $FW_SHARDS; do
    cp "corpus/hashes/fineweb_edu_shard${n}_v1.sha256" "$OUT/"
  done
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--HuggingFaceFW--fineweb-edu"
  python3 corpus/fetchers/human_dialogue.py simplewiki --allow-sa   # checked against corpus/hashes/human_simplewiki_v1.sha256
  FW=corpus/data
  SW=corpus/data/human_simplewiki_v1/text.txt
}

stage1() {
  local free_gb
  free_gb=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
  say "stage 1: $free_gb GB free, $(nproc) processors"
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "stage 1 wants $MIN_FREE_GB GB free and has $free_gb"; exit 1; }
  if [ -n "${BASE2_FIXTURE:-}" ]; then
    [ -e "$OUT/.fixture_copied" ] || { cp -r "$BASE2_FIXTURE"/. "$OUT"/ && touch "$OUT/.fixture_copied"; }
    FW="$OUT/fw"
    SW="$OUT/fw/simplewiki.txt"
  else
    fetch_sources
  fi
  local F="$OUT/filtered" n trains=""
  if [ ! -s "$SRC/fineweb2345.txt" ]; then
    for n in $FW_SHARDS; do trains="$trains $FW/fineweb_edu_shard${n}_v1/pretrain_train.txt"; done
    # shellcheck disable=SC2086
    python3 $CC/tc_b1_fineweb.py "$SRC/fineweb2345.txt" --val "$FW/fineweb_edu_shard0_v1/val.txt" $trains
    # The joined text is the source from here on; the shards' own text goes.
    [ -n "${BASE2_FIXTURE:-}" ] || for n in $FW_SHARDS; do rm -f "$FW/fineweb_edu_shard${n}_v1/pretrain_train.txt"; done
  fi
  cp "$SRC/fineweb2345.txt.ledger.json" "$OUT/fineweb_base2.ledger.json"
  [ -s "$SRC/simplewiki.txt" ] || B2 docfilter --repo . --heldout "$OUT/human_val.txt" "$OUT/human_sft.jsonl" \
    --in "$SW" --out "$SRC/simplewiki.txt" --ledger "$OUT/simplewiki.ledger.json"
  # One copy of the human text: S-full's pretraining split and the six long sets, as base A's mix read them.
  [ -s "$SRC/human1.txt" ] || { cat "$OUT/human_pt.txt" "$F/taskmaster3.txt" "$F/abcd.txt" "$F/ccpe.txt" "$F/p4g.txt" \
    "$F/aya_en.txt" "$F/msc.txt" > "$SRC/human1.txt.tmp" && mv "$SRC/human1.txt.tmp" "$SRC/human1.txt"; }
  for n in soda ultrachat everyday; do ln -f "$F/$n.txt" "$SRC/$n.txt" 2>/dev/null || cp "$F/$n.txt" "$SRC/$n.txt"; done
  for n in human_val.txt human_sft.jsonl; do ln -f "$OUT/$n" "$SRC/$n" 2>/dev/null || cp "$OUT/$n" "$SRC/$n"; done
  # The tokeniser's rate on this machine, before two mixes depend on it (the plan's risk 7).
  if [ ! -s "$OUT/tokeniser_timing.json" ]; then
    mkdir -p "$WORK"
    B2 head-docs "$SRC/fineweb2345.txt" "$WORK/timing.txt" "$TIMING_BYTES" > /dev/null
    local t0 t1
    t0=$(date +%s)
    python3 tools/corpus_ids.py "$WORK/timing.txt" "$V" "$WORK/timing.ids" --threads "$THREADS" > "$WORK/timing.json"
    t1=$(date +%s)
    B2 stage-ledger --out "$OUT/tokeniser_timing.json" --file "text=$WORK/timing.txt" --json "ids=$WORK/timing.json" \
      --kv "seconds=$((t1 - t0))" --kv "threads=$THREADS"
    rm -f "$WORK/timing.txt" "$WORK/timing.ids" "$WORK/timing.json"
  fi
  PINS "$OUT" stage1
  B2 stage-ledger --out "$OUT/base2_stage1.json" --kv stage=1 --kv "free_gb_at_start=$free_gb" \
    --file "fineweb=$SRC/fineweb2345.txt" --file "simplewiki=$SRC/simplewiki.txt" --file "human=$SRC/human1.txt" \
    --file "soda=$SRC/soda.txt" --file "ultrachat=$SRC/ultrachat.txt" --file "everyday=$SRC/everyday.txt" \
    --file "panel_fineweb=$OUT/panel_fineweb.ids" --file "panel_dialogue=$OUT/panel_dialogue.ids" \
    --json "fineweb_join=$OUT/fineweb_base2.ledger.json" --json "simplewiki_filter=$OUT/simplewiki.ledger.json" \
    --json "tokeniser_timing=$OUT/tokeniser_timing.json" --json "pins=$OUT/base2_pins_observed.json"
  push_set "$BB" build/src "$SRC"/fineweb2345.txt "$SRC"/simplewiki.txt "$SRC"/human1.txt "$SRC"/soda.txt \
    "$SRC"/ultrachat.txt "$SRC"/everyday.txt "$SRC"/human_val.txt "$SRC"/human_sft.jsonl
  push_set main corpus/panels "$OUT/panel_fineweb.ids" "$OUT/panel_dialogue.ids"
  local extra=""
  for n in data.ledger.json filter.ledger.json; do [ ! -s "$OUT/$n" ] || extra="$extra $OUT/$n"; done
  for n in "$OUT"/fineweb_edu_shard*_v1.sha256; do [ ! -s "$n" ] || extra="$extra $n"; done
  # shellcheck disable=SC2086
  push_set "$BB" build/stage1 "$OUT/base2_stage1.json" "$OUT/fineweb_base2.ledger.json" "$OUT/simplewiki.ledger.json" \
    "$OUT/tokeniser_timing.json" $extra
  say "stage 1 done"
}

stage2() {
  need_src
  [ -s "$OUT/base2_stage1.json" ] || pull_set "$BB" build/stage1 "$OUT"
  local avail="" s
  for s in fineweb:fineweb2345 simplewiki:simplewiki human:human1 soda:soda ultrachat:ultrachat everyday:everyday; do
    avail="$avail${avail:+,}${s%%:*}=$(stat -c %s "$SRC/${s##*:}.txt")"
  done
  [ -s "$OUT/recipe_N.json" ] || B2 recipe-n --avail "$avail" --out "$OUT/recipe_N.json"
  if [ ! -s "$OUT/N.main.ids" ] || [ ! -s "$OUT/N.tail.ids" ] || [ ! -s "$OUT/ids_N.json" ]; then
    dialogue_copies
    mix_and_ids N "$OUT/recipe_N.json"
  fi
  PINS "$OUT" build-N
  drop_mix_text N
  B2 stage-ledger --out "$OUT/base2_stage2.json" --kv stage=2 --file "N.main.ids=$OUT/N.main.ids" \
    --file "N.tail.ids=$OUT/N.tail.ids" --json "recipe=$OUT/recipe_N.json" --json "mix=$OUT/N.ledger.json" \
    --jsonl "ids=$OUT/ids_N.json" --json "pins=$OUT/base2_pins_observed.json"
  push_set main corpus/N "$OUT/N.main.ids" "$OUT/N.tail.ids" "$OUT/N.ledger.json" "$OUT/ids_N.json" "$OUT/recipe_N.json"
  push_set "$BB" build/stage2 "$OUT/base2_stage2.json" "$OUT/N.ledger.json" "$OUT/ids_N.json" "$OUT/recipe_N.json"
  say "stage 2 done"
}

scn_shard() {  # K: shard K of the training stream, generated and filtered. Shard 0 keeps its metadata for the validator.
  local k=$1 d="$OUT/scn" meta=/dev/null
  [ -s "$d/shard_$k.ledger.json" ] && [ -e "$d/shard_$k.txt" ] && return 0
  [ "$k" = 0 ] && meta="$d/raw_0.meta.jsonl"
  python3 "$ENGINE/generate.py" --seed "$SEED" --docs "$SHARD_DOCS" --start $((k * SHARD_DOCS / 2)) \
    --out "$d/raw_$k.txt" --meta "$meta" > /dev/null
  if [ "$k" = 0 ] && [ "${BASE2_SCN_VALIDATE:-1}" = 1 ]; then
    python3 "$ENGINE/validate.py" "$d/raw_0.txt" "$d/raw_0.meta.jsonl" --report "$OUT/scenario_validate_train.json" > /dev/null
  fi
  B2 docfilter --repo . --heldout "$SRC/human_val.txt" "$SRC/human_sft.jsonl" --exclude "$d/panel_scenario.txt" \
    --in "$d/raw_$k.txt" --out "$d/shard_$k.txt" --ledger "$d/shard_$k.ledger.json.tmp" > /dev/null
  rm -f "$d/raw_$k.txt" "$d/raw_$k.meta.jsonl"
  mv "$d/shard_$k.ledger.json.tmp" "$d/shard_$k.ledger.json"
}

stage3() {
  need_src
  [ -s "$OUT/recipe_N.json" ] && [ -s "$OUT/N.ledger.json" ] && [ -s "$OUT/ids_N.json" ] || pull_set "$BB" build/stage2 "$OUT"
  # The two panels of stage 1 are pinned beside the scenario panel; a VM that never ran stage 1 takes them from the store.
  [ -s "$OUT/panel_fineweb.ids" ] && [ -s "$OUT/panel_dialogue.ids" ] || pull_set main corpus/panels "$OUT"
  PINS "$ENGINE" engine   # the generator's files are the pinned ones, or nothing is generated
  cp "$ENGINE/base2_pins_observed.json" "$OUT/engine_pins_observed.json" 2>/dev/null || true
  local d="$OUT/scn" k shards
  mkdir -p "$d" "$WORK"
  # The held-out panel first: no training document may equal one of its documents.
  if [ ! -s "$OUT/panel_scenario.ids" ] || [ ! -s "$OUT/panel_scenario.json" ] || [ ! -s "$d/panel_scenario.txt" ]; then
    python3 "$ENGINE/generate.py" --seed "$SEED" --docs "$HELD_DOCS" --heldout --out "$d/panel_scenario.txt.tmp" \
      --meta "$d/panel_scenario.meta.jsonl" > /dev/null
    [ "${BASE2_SCN_VALIDATE:-1}" != 1 ] || python3 "$ENGINE/validate.py" "$d/panel_scenario.txt.tmp" \
      "$d/panel_scenario.meta.jsonl" --heldout --report "$OUT/scenario_validate_heldout.json" > /dev/null
    rm -f "$d/panel_scenario.meta.jsonl"
    mv "$d/panel_scenario.txt.tmp" "$d/panel_scenario.txt"
    python3 tools/corpus_ids.py "$d/panel_scenario.txt" "$V" "$OUT/panel_scenario.ids.tmp" --threads "$THREADS" > "$OUT/panel_scenario.json.tmp"
    mv "$OUT/panel_scenario.ids.tmp" "$OUT/panel_scenario.ids"
    sed -i "s/\.ids\.tmp\"/.ids\"/" "$OUT/panel_scenario.json.tmp"
    mv "$OUT/panel_scenario.json.tmp" "$OUT/panel_scenario.json"
  fi
  # Shard 0 is the sample: its bytes a token under the real tokeniser set the scenario ratio (the plan's risk 8).
  scn_shard 0
  if [ ! -s "$OUT/scenario_sample.json" ]; then
    python3 tools/corpus_ids.py "$d/shard_0.txt" "$V" "$WORK/scn0.ids" --threads "$THREADS" > "$OUT/scenario_sample.json.tmp"
    rm -f "$WORK/scn0.ids"
    mv "$OUT/scenario_sample.json.tmp" "$OUT/scenario_sample.json"
  fi
  [ -s "$OUT/recipe_S.json" ] || B2 recipe-s --recipe-n "$OUT/recipe_N.json" --n-ledger "$OUT/N.ledger.json" \
    --n-ids "$OUT/ids_N.json" --scn-bytes "$(stat -c %s "$d/shard_0.txt")" --scn-tokens "$(B2 get "$OUT/scenario_sample.json" tokens)" \
    --scn-docs "$(B2 get "$d/shard_0.ledger.json" documents_kept)" --shard-docs "$SHARD_DOCS" --max-drop "$MAX_DROP" \
    --out "$OUT/recipe_S.json"
  shards=$(B2 get "$OUT/recipe_S.json" scenario_shards)
  if [ ! -s "$OUT/S.main.ids" ] || [ ! -s "$OUT/S.tail.ids" ] || [ ! -s "$OUT/ids_S.json" ] || [ ! -s "$OUT/scenario.ledger.json" ]; then
    say "scenario stream: $shards shards of $SHARD_DOCS documents, $JOBS at a time"
    seq 1 $((shards - 1)) | xargs -r -P "$JOBS" -I{} bash "$0" "$OUT" "$V" _shard {}
    if [ ! -s "$WORK/scenario.txt" ]; then
      : > "$WORK/scenario.txt.tmp"
      for k in $(seq 0 $((shards - 1))); do cat "$d/shard_$k.txt" >> "$WORK/scenario.txt.tmp"; done
      mv "$WORK/scenario.txt.tmp" "$WORK/scenario.txt"
    fi
    B2 scn-ledger --dir "$d" --shards "$shards" --text "$WORK/scenario.txt" --recipe "$OUT/recipe_S.json" \
      --seed "$SEED" --max-drop "$MAX_DROP" --out "$OUT/scenario.ledger.json"
    dialogue_copies
    mix_and_ids S "$OUT/recipe_S.json" --source "scenario=$WORK/scenario.txt"
  fi
  B2 subset "$OUT/N.ledger.json" "$OUT/S.ledger.json" > "$OUT/subset_S_in_N.json"
  # The registered share, measured on the mixed text by the real tokeniser: outside 14% to 16% the stage fails.
  local part
  for part in main tail; do
    [ -s "$OUT/share_S.$part.json" ] || B2 share --mix "$OUT/S.$part.txt" --scenario "$WORK/scenario.txt" --vocab "$V" \
      --ids-record "$OUT/ids_S.json" --out "$OUT/share_S.$part.json" > /dev/null
    [ "$(B2 get "$OUT/share_S.$part.json" ok)" = True ] || { echo "scenario share out of bounds in the $part part"; exit 1; }
  done
  PINS "$OUT" build-S
  drop_mix_text S
  B2 stage-ledger --out "$OUT/base2_stage3.json" --kv stage=3 --kv "scenario_seed=$SEED" \
    --file "S.main.ids=$OUT/S.main.ids" --file "S.tail.ids=$OUT/S.tail.ids" --file "panel_scenario.ids=$OUT/panel_scenario.ids" \
    --json "recipe=$OUT/recipe_S.json" --json "mix=$OUT/S.ledger.json" --jsonl "ids=$OUT/ids_S.json" \
    --json "scenario=$OUT/scenario.ledger.json" --json "scenario_sample=$OUT/scenario_sample.json" \
    --json "panel_scenario=$OUT/panel_scenario.json" --json "share_main=$OUT/share_S.main.json" \
    --json "share_tail=$OUT/share_S.tail.json" --json "subset=$OUT/subset_S_in_N.json" \
    --json "engine_pins=$OUT/engine_pins_observed.json" --json "pins=$OUT/base2_pins_observed.json"
  push_set main corpus/panel_scenario "$OUT/panel_scenario.ids" "$OUT/panel_scenario.json"
  push_set main corpus/S "$OUT/S.main.ids" "$OUT/S.tail.ids" "$OUT/S.ledger.json" "$OUT/ids_S.json" "$OUT/recipe_S.json" \
    "$OUT/scenario.ledger.json" "$OUT/share_S.main.json" "$OUT/share_S.tail.json"
  push_set "$BB" build/stage3 "$OUT/base2_stage3.json" "$OUT/S.ledger.json" "$OUT/ids_S.json" "$OUT/recipe_S.json" \
    "$OUT/scenario.ledger.json"
  rm -rf "$d" "$WORK"
  say "stage 3 done"
}

rehearsal() {  # the ids of the pinned 50 MB FineWeb-Edu sample, for base2_rehearsal.json
  if [ ! -s "$OUT/rehearsal.ids" ]; then
    local rc=0
    B2 store-has "$STORE" main corpus/rehearsal || rc=$?
    if [ "$rc" = 0 ]; then
      pull_set main corpus/rehearsal "$OUT"
    elif [ "$rc" = 1 ]; then
      local text=corpus/data/fineweb_edu_sample_v1/pretrain_train.txt
      if [ -n "${BASE2_FIXTURE:-}" ]; then text="$BASE2_FIXTURE/fw/sample.txt"; else python3 corpus/fetchers/fineweb_edu.py sample_v1; fi
      python3 tools/corpus_ids.py "$text" "$V" "$OUT/rehearsal.ids.tmp" --threads "$THREADS" > "$OUT/rehearsal.json.tmp"
      mv "$OUT/rehearsal.ids.tmp" "$OUT/rehearsal.ids"
      sed -i "s/\.ids\.tmp\"/.ids\"/" "$OUT/rehearsal.json.tmp"
      mv "$OUT/rehearsal.json.tmp" "$OUT/rehearsal.json"
      push_set main corpus/rehearsal "$OUT/rehearsal.ids" "$OUT/rehearsal.json"
    else
      echo "the store could not be read; stopping"; exit 1
    fi
  fi
  PINS "$OUT" rehearsal --upload "$(dirname "$V")"
}

# The vocabulary is the pinned one before anything is tokenised with it.
[ "$MODE" = _shard ] || PINS "$OUT" vocab --upload "$(dirname "$V")"
run_stage() { if stage_done "$1"; then say "stage $1: finished already"; else "stage$1"; fi; }
case "$MODE" in
  stage1) run_stage 1 ;;
  stage2) run_stage 2 ;;
  stage3) run_stage 3 ;;
  all) run_stage 1; run_stage 2; run_stage 3 ;;
  rehearsal) rehearsal ;;
  _shard) scn_shard "${4:?shard}" ;;
  *) echo "mode is stage1, stage2, stage3, all or rehearsal"; exit 1 ;;
esac
