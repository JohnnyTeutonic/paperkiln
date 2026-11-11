#!/usr/bin/env bash
# Base 2's corpus build (BASE2_PLAN.md sections 2.3 and 2.4, in the author's research tree). Claude Fable 5.2, 11 Nov 2026.
#   bash tools/colab_chat/base2_data.sh OUT VOCAB_GGUF stage1|stage2|stage3|all|rehearsal
#   bash tools/colab_chat/base2_data.sh OUT VOCAB_GGUF loadprobe PROBE_OUT
# Three stages, so a lost session repeats one stage at most:
#   stage1  S-full's data job first, unchanged (the same panels and split as base A's), the long sets, lever 2's dialogue
#           sets, FineWeb-Edu shards 2 to 5 joined, Simple English Wikipedia. Store: base2-build:build/src (the text),
#           main:corpus/panels.
#   stage2  the natural mix (arm N), its ratios set from stage 1's measured sizes, and its ids. Store: main:corpus/N.
#   stage3  the scenario stream from the pinned engine, the held-out scenario panel, the scenario mix (arms S and C) with
#           the stream at 15% of tokens by the measured bytes a token, and its ids. Store: main:corpus/S,
#           main:corpus/panel_scenario.
#
# What the decontamination covers, source by source, and what it does not:
#   dialogue sources (the 13 human sets through tiny_chat_data.py; the six long sets, SODA, UltraChat and everyday
#       conversations through tc_b1_filter.py): corpus/decontam.py's rules against the frozen eval sets, Carry-5 and
#       Carry-12, and no word 13-gram shared with a held-out dialogue (the dialogue panel and the SFT rows).
#   Simple English Wikipedia and the scenario stream (base2_corpus.py docfilter): the same two rules. The scenario stream
#       also drops any document equal, byte for byte, to a document of the held-out scenario panel; nothing looser than
#       equality is checked against that panel (the engine keeps the two streams apart by family and bank item).
#   FineWeb-Edu: exact copies of the FineWeb panel's documents are dropped, and repeats inside the shards. Nothing else:
#       FineWeb-Edu is NOT put through the eval rules, Carry-5, Carry-12 or the held-out 13-gram check, here or for base A.
#   N against S: S's natural documents are checked, document by document, to be among N's.
#
# Built is not published. A stage is finished only when the store holds its receipt, base2-build:build/stageN, and every
# set that stage publishes; the receipt is pushed last, after each output set has been pushed and read back from the store
# against the local bytes. A local file never marks a stage finished. A rerun of an unfinished stage builds only what is
# missing or stale and then publishes: every cached file (recipe, ledger, ids record, share) carries the sha256 of the
# inputs it was built from (FILE.inputs.json) and is rebuilt when any of them differs. A mix's text stays on disk until
# its stage's receipt is in the store. A stage that finds its inputs missing pulls them from the store.
#   rehearsal  two small corpora of different sizes from the pinned 50 MB FineWeb-Edu sample (main:corpus/rehearsal).
#   loadprobe  S's ids pulled and verified if this VM lacks them, then the 20-step load measurement.
# Environment (defaults are the campaign's): BASE2_STORE_CONFIG, BASE2_PINS, BASE2_ENGINE_DIR, BASE2_HF_HUB_VERSION,
# BASE2_THREADS, BASE2_JOBS, BASE2_SCN_SEED, BASE2_SCN_SHARD_DOCS, BASE2_SCN_HELDOUT_DOCS, BASE2_SCN_MAX_DROP,
# BASE2_SCN_VALIDATE, BASE2_TOKEN_REL_MAX, BASE2_MIN_FREE_GB, BASE2_TIMING_BYTES, BASE2_REH_MAIN_BYTES,
# BASE2_STORE_RETRY_S, BASE2_STORE_LIMIT_S, BASE2_STORE_XFER_LIMIT_S, BASE2_KEEP_MIX_TEXT. BASE2_FIXTURE=DIR stands in for the network fetches of stage 1 with a folder
# of the same files (tests).
set -euo pipefail
OUT=${1:?out dir}
V=${2:?vocab gguf}
MODE=${3:?stage1, stage2, stage3, all, rehearsal or loadprobe}
CC=tools/colab_chat
STORE=${BASE2_STORE_CONFIG:-/content/chat/store.json}
BB=base2-build
ENGINE=${BASE2_ENGINE_DIR:-/content/chat/upload/scenario_engine}
# One version of the store's client library on every VM: the store agent's boot installs the same one.
HF_HUB_VERSION=${BASE2_HF_HUB_VERSION:-0.35.3}
THREADS=${BASE2_THREADS:-10}
JOBS=${BASE2_JOBS:-8}
SEED=${BASE2_SCN_SEED:-20261111}
SHARD_DOCS=${BASE2_SCN_SHARD_DOCS:-200000}
HELD_DOCS=${BASE2_SCN_HELDOUT_DOCS:-20000}
MAX_DROP=${BASE2_SCN_MAX_DROP:-0.02}
VALIDATE=${BASE2_SCN_VALIDATE:-1}
TOKEN_REL_MAX=${BASE2_TOKEN_REL_MAX:-0.01}
MIN_FREE_GB=${BASE2_MIN_FREE_GB:-80}
TIMING_BYTES=${BASE2_TIMING_BYTES:-200000000}
REH_MAIN_BYTES=${BASE2_REH_MAIN_BYTES:-36000000}
RETRY_S=${BASE2_STORE_RETRY_S:-20}
# Every store call of this script has a wall-time limit and runs in a child that is ended at it: a push or a pull of a
# set (hf_store.py --limit), and each read of a manifest by store-has and store-verify (BASE2_STORE_LIMIT_S, default 300,
# read by base2_corpus.py). A call that hangs is a failed try and the script stops; it never waits without end.
XFER_LIMIT_S=${BASE2_STORE_XFER_LIMIT_S:-5400}
# The hub limits commit creation per repository (128 an hour for every writer together, 12 Nov 2026). The whole build
# is eight commits (one a set; the rehearsal's corpus is one), each tried at most four times, so 32 at the very worst;
# they pass a budget all the same, counted beside the
# store's config with every other process of this VM, and a push that finds the budget spent waits for its turn.
COMMIT_BUDGET=${BASE2_STORE_COMMIT_BUDGET:-60}
COMMIT_WAIT_S=${BASE2_STORE_COMMIT_WAIT_S:-900}
export BASE2_STORE_LIMIT_S=${BASE2_STORE_LIMIT_S:-300}
S13=oasst,convai2,wow,bst,taskmaster,multiwoz,self_dialogue,topical_chat,dolly,quac,empathetic,dailydialog,no_robots
LONG=taskmaster3,abcd,ccpe,p4g,aya_en,msc
FW_SHARDS="2 3 4 5"   # unseen by base A, whose mix read shards 0 and 1
SRC="$OUT/src"
WORK="$OUT/work"
mkdir -p "$OUT" "$SRC"
export BASE2_HASH_CACHE="$OUT/.hash_cache.json"
# The public datasets need no credential, and the store's token is handed to the store's own calls alone.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
B2() { python3 $CC/base2_corpus.py "$@"; }
PINS() { python3 $CC/base2_pins.py "$@"; }
say() { echo "base2_data $(date -u +%H:%M:%S) $*"; }
cached() { B2 bound "$@"; }   # OUT --input F...: 0 when OUT was built from exactly these bytes; a stale OUT is removed
stamp() { B2 bind "$@"; }     # OUT --input F...: record them

push_set() {  # BRANCH PREFIX FILE...: the files as one set, one commit, read back from the store against the local bytes
  local branch=$1 prefix=$2 d="$OUT/.push" try name rec
  shift 2
  name=$(echo "$prefix" | tr / _)
  rec="$OUT/published_$name.json"
  rm -f "$rec"
  if B2 store-verify "$STORE" "$branch" "$prefix" "$@" --out "$rec" 2> /dev/null; then
    say "published $branch:$prefix (the store held these bytes already)"
    return 0
  fi
  for try in 1 2 3 4; do
    rm -rf "$d" && mkdir -p "$d"
    for f in "$@"; do ln -f "$f" "$d/$(basename "$f")" 2> /dev/null || cp "$f" "$d/$(basename "$f")"; done
    if python3 $CC/hf_store.py push --config "$STORE" --branch "$branch" --prefix "$prefix" --dir "$d" --limit "$XFER_LIMIT_S" \
        --commit-budget "$COMMIT_BUDGET" --commit-wait "$COMMIT_WAIT_S" > "$OUT/push_$name.log" 2>&1 \
        && B2 store-verify "$STORE" "$branch" "$prefix" "$@" --out "$rec"; then
      rm -rf "$d"
      say "published $branch:$prefix (try $try, verified)"
      return 0
    fi
    say "push of $branch:$prefix failed or did not verify (try $try of 4)"
    sleep $((RETRY_S * try))
  done
  rm -rf "$d"
  echo "base2_data: $branch:$prefix is not published; stopping"
  exit 1
}
pull_set() {  # BRANCH PREFIX DIR: every file checked by sha256 against the set's manifest (hf_store.py pull)
  python3 $CC/hf_store.py pull --config "$STORE" --branch "$1" --prefix "$2" --to "$3" --limit "$XFER_LIMIT_S" > /dev/null
  say "pulled $1:$2"
}
in_store() {  # BRANCH PREFIX: 0 there, 1 not there; an unreadable store stops the build
  local rc=0
  B2 store-has "$STORE" "$1" "$2" || rc=$?
  [ "$rc" = 0 ] && return 0
  [ "$rc" = 1 ] && return 1
  echo "base2_data: the store could not be read ($1:$2); stopping"
  exit 1
}
sets_of() {  # the output sets a stage publishes before its receipt
  case "$1" in
    1) echo "$BB:build/src main:corpus/panels" ;;
    2) echo "main:corpus/N" ;;
    3) echo "main:corpus/S main:corpus/panel_scenario" ;;
  esac
}
need_python_packages() {
  python3 -c "import sys, pyarrow, pandas, tokenizers, huggingface_hub as h; sys.exit(h.__version__ != '$HF_HUB_VERSION')" 2> /dev/null \
    || pip -q install "huggingface_hub==$HF_HUB_VERSION" pandas pyarrow tokenizers
}
need_src() {
  for f in fineweb2345.txt simplewiki.txt human1.txt soda.txt ultrachat.txt everyday.txt human_val.txt human_sft.jsonl; do
    [ -s "$SRC/$f" ] || { pull_set "$BB" build/src "$SRC"; break; }
  done
}
copies() {  # N OUT FILE: FILE N times
  local n=$1 out=$2 f=$3 i
  if cached "$out" --input "$f" --key "copies=$n"; then return 0; fi
  : > "$out.tmp"
  for i in $(seq "$n"); do cat "$f" >> "$out.tmp"; done
  mv "$out.tmp" "$out"
  stamp "$out" --input "$f" --key "copies=$n"
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
ENGINE_INPUTS() {  # the engine's Python files, tests left out, as --input arguments
  local f
  for f in "$ENGINE"/*.py; do
    case "$(basename "$f")" in test_*) ;; *) printf -- '--input %s ' "$f" ;; esac
  done
}
MIX_DEPS=""   # further --input arguments the next mix is bound to
mix_and_ids() {  # ARM RECIPE EXTRA_SOURCE...: OUT/ARM.{main,tail}.ids, ARM.ledger.json, ids_ARM.json; the text stays
  local arm=$1 recipe=$2 part
  shift 2
  # shellcheck disable=SC2086
  if ! cached "$OUT/$arm.ledger.json" --input "$recipe" $MIX_DEPS || [ ! -s "$OUT/$arm.main.txt" ] || [ ! -s "$OUT/$arm.tail.txt" ]; then
    rm -f "$OUT/$arm.ledger.json" "$OUT/$arm.ledger.json.inputs.json"
    # shellcheck disable=SC2046
    python3 corpus/mix_pretrain.py "$OUT/$arm" --seed 1 --tail-fraction "$(B2 get "$recipe" tail_fraction)" \
      --total-bytes "$(B2 get "$recipe" total_bytes)" $(NATURAL_SOURCES) "$@" \
      --main "$(B2 get "$recipe" main_arg)" --tail "$(B2 get "$recipe" tail_arg)"
    # shellcheck disable=SC2086
    stamp "$OUT/$arm.ledger.json" --input "$recipe" $MIX_DEPS
  fi
  # The ledger holds the sha256 of both parts' text, so ids bound to it are bound to the text they encode.
  if ! cached "$OUT/ids_$arm.json" --input "$OUT/$arm.ledger.json" --input "$V" || [ ! -s "$OUT/$arm.main.ids" ] || [ ! -s "$OUT/$arm.tail.ids" ]; then
    : > "$OUT/ids_$arm.json.tmp"
    for part in main tail; do
      python3 tools/corpus_ids.py "$OUT/$arm.$part.txt" "$V" "$OUT/$arm.$part.ids.tmp" --threads "$THREADS" >> "$OUT/ids_$arm.json.tmp"
      mv "$OUT/$arm.$part.ids.tmp" "$OUT/$arm.$part.ids"
    done
    sed -i "s/\.ids\.tmp\"/.ids\"/" "$OUT/ids_$arm.json.tmp"
    mv "$OUT/ids_$arm.json.tmp" "$OUT/ids_$arm.json"
    stamp "$OUT/ids_$arm.json" --input "$OUT/$arm.ledger.json" --input "$V"
  fi
}
drop_mix_text() { [ "${BASE2_KEEP_MIX_TEXT:-0}" = 1 ] || rm -f "$OUT/$1.main.txt" "$OUT/$1.tail.txt"; }

fetch_sources() {  # the network half of stage 1, as base_b1_data.sh fetches for base A
  need_python_packages
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

build1() {
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
  local F="$OUT/filtered" n trains="" join="" held
  held="--input $OUT/human_val.txt --input $OUT/human_sft.jsonl"
  # The join is bound to the shards' pinned hashes (their text goes once it is joined) and to the panel's text.
  for n in $FW_SHARDS; do
    trains="$trains $FW/fineweb_edu_shard${n}_v1/pretrain_train.txt"
    if [ -n "${BASE2_FIXTURE:-}" ]; then join="$join --input $FW/fineweb_edu_shard${n}_v1/pretrain_train.txt"
    else join="$join --input corpus/hashes/fineweb_edu_shard${n}_v1.sha256"; fi
  done
  join="$join --input $FW/fineweb_edu_shard0_v1/val.txt"
  # shellcheck disable=SC2086
  if ! cached "$SRC/fineweb2345.txt" $join; then
    # FineWeb-Edu's whole decontamination: no document equal to one of the FineWeb panel, and each document once.
    # shellcheck disable=SC2086
    python3 $CC/tc_b1_fineweb.py "$SRC/fineweb2345.txt" --val "$FW/fineweb_edu_shard0_v1/val.txt" $trains
    # shellcheck disable=SC2086
    stamp "$SRC/fineweb2345.txt" $join
    # The joined text is the source from here on; the shards' own text goes.
    [ -n "${BASE2_FIXTURE:-}" ] || for n in $FW_SHARDS; do rm -f "$FW/fineweb_edu_shard${n}_v1/pretrain_train.txt"; done
  fi
  cp "$SRC/fineweb2345.txt.ledger.json" "$OUT/fineweb_base2.ledger.json"
  # shellcheck disable=SC2086
  if ! cached "$SRC/simplewiki.txt" --input "$SW" $held; then
    B2 docfilter --repo . --heldout "$OUT/human_val.txt" "$OUT/human_sft.jsonl" \
      --in "$SW" --out "$SRC/simplewiki.txt" --ledger "$OUT/simplewiki.ledger.json"
    # shellcheck disable=SC2086
    stamp "$SRC/simplewiki.txt" --input "$SW" $held
  fi
  # One copy of the human text: S-full's pretraining split and the six long sets, as base A's mix read them.
  local parts="$OUT/human_pt.txt $F/taskmaster3.txt $F/abcd.txt $F/ccpe.txt $F/p4g.txt $F/aya_en.txt $F/msc.txt" hin="" p
  for p in $parts; do hin="$hin --input $p"; done
  # shellcheck disable=SC2086
  if ! cached "$SRC/human1.txt" $hin; then
    # shellcheck disable=SC2086
    cat $parts > "$SRC/human1.txt.tmp" && mv "$SRC/human1.txt.tmp" "$SRC/human1.txt"
    # shellcheck disable=SC2086
    stamp "$SRC/human1.txt" $hin
  fi
  for n in soda ultrachat everyday; do ln -f "$F/$n.txt" "$SRC/$n.txt" 2> /dev/null || cp "$F/$n.txt" "$SRC/$n.txt"; done
  for n in human_val.txt human_sft.jsonl; do ln -f "$OUT/$n" "$SRC/$n" 2> /dev/null || cp "$OUT/$n" "$SRC/$n"; done
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
  B2 stage-ledger --out "$OUT/base2_stage1.built.json" --kv stage=1 --kv "free_gb_at_start=$free_gb" \
    --file "fineweb=$SRC/fineweb2345.txt" --file "simplewiki=$SRC/simplewiki.txt" --file "human=$SRC/human1.txt" \
    --file "soda=$SRC/soda.txt" --file "ultrachat=$SRC/ultrachat.txt" --file "everyday=$SRC/everyday.txt" \
    --file "panel_fineweb=$OUT/panel_fineweb.ids" --file "panel_dialogue=$OUT/panel_dialogue.ids" \
    --json "fineweb_join=$OUT/fineweb_base2.ledger.json" --json "simplewiki_filter=$OUT/simplewiki.ledger.json" \
    --json "tokeniser_timing=$OUT/tokeniser_timing.json" --json "pins=$OUT/base2_pins_observed.json"
  say "stage 1 built"
}
publish1() {
  push_set "$BB" build/src "$SRC"/fineweb2345.txt "$SRC"/simplewiki.txt "$SRC"/human1.txt "$SRC"/soda.txt \
    "$SRC"/ultrachat.txt "$SRC"/everyday.txt "$SRC"/human_val.txt "$SRC"/human_sft.jsonl
  push_set main corpus/panels "$OUT/panel_fineweb.ids" "$OUT/panel_dialogue.ids"
  B2 stage-ledger --out "$OUT/base2_stage1.json" --kv stage=1 --json "built=$OUT/base2_stage1.built.json" \
    --json "published_src=$OUT/published_build_src.json" --json "published_panels=$OUT/published_corpus_panels.json"
  local extra="" n
  for n in data.ledger.json filter.ledger.json; do [ ! -s "$OUT/$n" ] || extra="$extra $OUT/$n"; done
  for n in "$OUT"/fineweb_edu_shard*_v1.sha256; do [ ! -s "$n" ] || extra="$extra $n"; done
  # The receipt, last.
  # shellcheck disable=SC2086
  push_set "$BB" build/stage1 "$OUT/base2_stage1.json" "$OUT/fineweb_base2.ledger.json" "$OUT/simplewiki.ledger.json" \
    "$OUT/tokeniser_timing.json" $extra
  say "stage 1 done"
}

build2() {
  need_src
  [ -s "$OUT/base2_stage1.json" ] || pull_set "$BB" build/stage1 "$OUT"
  local avail="" s srcin=""
  for s in fineweb:fineweb2345 simplewiki:simplewiki human:human1 soda:soda ultrachat:ultrachat everyday:everyday; do
    avail="$avail${avail:+,}${s%%:*}=$(stat -c %s "$SRC/${s##*:}.txt")"
    srcin="$srcin --input $SRC/${s##*:}.txt"
  done
  # shellcheck disable=SC2086
  if ! cached "$OUT/recipe_N.json" $srcin; then
    B2 recipe-n --avail "$avail" --out "$OUT/recipe_N.json"
    # shellcheck disable=SC2086
    stamp "$OUT/recipe_N.json" $srcin
  fi
  dialogue_copies
  MIX_DEPS=""
  mix_and_ids N "$OUT/recipe_N.json"
  # N's documents by digest, kept so stage 3 can compare S's with them after N's text has gone.
  if ! cached "$OUT/N.keys" --input "$OUT/N.ledger.json"; then
    B2 doc-keys "$OUT/N.keys" "$OUT/N.main.txt" "$OUT/N.tail.txt" > "$OUT/N.keys.json"
    stamp "$OUT/N.keys" --input "$OUT/N.ledger.json"
  fi
  PINS "$OUT" build-N
  B2 stage-ledger --out "$OUT/base2_stage2.built.json" --kv stage=2 --file "N.main.ids=$OUT/N.main.ids" \
    --file "N.tail.ids=$OUT/N.tail.ids" --json "recipe=$OUT/recipe_N.json" --json "mix=$OUT/N.ledger.json" \
    --jsonl "ids=$OUT/ids_N.json" --json "documents=$OUT/N.keys.json" --json "pins=$OUT/base2_pins_observed.json"
  say "stage 2 built"
}
publish2() {
  push_set main corpus/N "$OUT/N.main.ids" "$OUT/N.tail.ids" "$OUT/N.ledger.json" "$OUT/ids_N.json" "$OUT/recipe_N.json"
  B2 stage-ledger --out "$OUT/base2_stage2.json" --kv stage=2 --json "built=$OUT/base2_stage2.built.json" \
    --json "published_N=$OUT/published_corpus_N.json"
  push_set "$BB" build/stage2 "$OUT/base2_stage2.json" "$OUT/N.ledger.json" "$OUT/ids_N.json" "$OUT/recipe_N.json" "$OUT/N.keys"
  drop_mix_text N   # only now: the receipt is in the store
  say "stage 2 done"
}

scn_shard() {  # K: shard K of the training stream: generated, put through the engine's validator, then filtered
  local k=$1 d="$OUT/scn" led bind
  led="$d/shard_$k.ledger.json"
  bind="$(ENGINE_INPUTS) --input $d/panel_scenario.txt --input $SRC/human_val.txt --input $SRC/human_sft.jsonl \
    --key seed=$SEED --key docs=$SHARD_DOCS --key shard=$k --key validate=$VALIDATE"
  # shellcheck disable=SC2086
  if cached "$led" $bind && [ -e "$d/shard_$k.txt" ]; then return 0; fi
  python3 "$ENGINE/generate.py" --seed "$SEED" --docs "$SHARD_DOCS" --start $((k * SHARD_DOCS / 2)) \
    --out "$d/raw_$k.txt" --meta "$d/raw_$k.meta.jsonl" > /dev/null
  rm -f "$d/validate_$k.json"
  if [ "$VALIDATE" = 1 ]; then
    python3 "$ENGINE/validate.py" "$d/raw_$k.txt" "$d/raw_$k.meta.jsonl" --report "$d/validate_$k.json" > "$d/validate_$k.log" 2>&1 \
      || { echo "base2_data: scenario shard $k fails the engine's validator ($d/validate_$k.json)"; exit 1; }
  fi
  B2 docfilter --repo . --heldout "$SRC/human_val.txt" "$SRC/human_sft.jsonl" --exclude "$d/panel_scenario.txt" \
    --in "$d/raw_$k.txt" --out "$d/shard_$k.txt" --ledger "$led" > /dev/null
  rm -f "$d/raw_$k.txt" "$d/raw_$k.meta.jsonl"
  # shellcheck disable=SC2086
  stamp "$led" $bind
}

build3() {
  need_src
  [ -s "$OUT/recipe_N.json" ] && [ -s "$OUT/N.ledger.json" ] && [ -s "$OUT/ids_N.json" ] && [ -s "$OUT/N.keys" ] \
    || pull_set "$BB" build/stage2 "$OUT"
  # The two panels of stage 1 are pinned beside the scenario panel; a VM that never ran stage 1 takes them from the store.
  [ -s "$OUT/panel_fineweb.ids" ] && [ -s "$OUT/panel_dialogue.ids" ] || pull_set main corpus/panels "$OUT"
  PINS "$ENGINE" engine   # the generator's files are the pinned ones, or nothing is generated
  cp "$ENGINE/base2_pins_observed.json" "$OUT/engine_pins_observed.json" 2> /dev/null || true
  local d="$OUT/scn" k shards part bind
  mkdir -p "$d" "$WORK"
  # The held-out panel first: no training document may equal one of its documents.
  bind="$(ENGINE_INPUTS) --input $V --key seed=$SEED --key docs=$HELD_DOCS --key validate=$VALIDATE"
  # shellcheck disable=SC2086
  if ! cached "$OUT/panel_scenario.json" $bind || [ ! -s "$OUT/panel_scenario.ids" ] || [ ! -s "$d/panel_scenario.txt" ]; then
    python3 "$ENGINE/generate.py" --seed "$SEED" --docs "$HELD_DOCS" --heldout --out "$d/panel_scenario.txt.tmp" \
      --meta "$d/panel_scenario.meta.jsonl" > /dev/null
    if [ "$VALIDATE" = 1 ]; then
      python3 "$ENGINE/validate.py" "$d/panel_scenario.txt.tmp" "$d/panel_scenario.meta.jsonl" --heldout \
        --report "$OUT/scenario_validate_heldout.json" > "$d/validate_heldout.log" 2>&1 \
        || { echo "base2_data: the held-out scenario panel fails the engine's validator"; exit 1; }
    fi
    rm -f "$d/panel_scenario.meta.jsonl"
    mv "$d/panel_scenario.txt.tmp" "$d/panel_scenario.txt"
    python3 tools/corpus_ids.py "$d/panel_scenario.txt" "$V" "$OUT/panel_scenario.ids.tmp" --threads "$THREADS" > "$OUT/panel_scenario.json.tmp"
    mv "$OUT/panel_scenario.ids.tmp" "$OUT/panel_scenario.ids"
    sed -i "s/\.ids\.tmp\"/.ids\"/" "$OUT/panel_scenario.json.tmp"
    mv "$OUT/panel_scenario.json.tmp" "$OUT/panel_scenario.json"
    # shellcheck disable=SC2086
    stamp "$OUT/panel_scenario.json" $bind
  fi
  # Shard 0 is the sample: its bytes a token under the real tokeniser set the scenario ratio (the plan's risk 8).
  scn_shard 0
  if ! cached "$OUT/scenario_sample.json" --input "$d/shard_0.ledger.json" --input "$V"; then
    python3 tools/corpus_ids.py "$d/shard_0.txt" "$V" "$WORK/scn0.ids" --threads "$THREADS" > "$OUT/scenario_sample.json.tmp"
    rm -f "$WORK/scn0.ids"
    mv "$OUT/scenario_sample.json.tmp" "$OUT/scenario_sample.json"
    stamp "$OUT/scenario_sample.json" --input "$d/shard_0.ledger.json" --input "$V"
  fi
  bind="--input $OUT/recipe_N.json --input $OUT/N.ledger.json --input $OUT/ids_N.json --input $OUT/scenario_sample.json \
    --input $d/shard_0.ledger.json --key shard_docs=$SHARD_DOCS --key max_drop=$MAX_DROP"
  # shellcheck disable=SC2086
  if ! cached "$OUT/recipe_S.json" $bind; then
    B2 recipe-s --recipe-n "$OUT/recipe_N.json" --n-ledger "$OUT/N.ledger.json" \
      --n-ids "$OUT/ids_N.json" --scn-bytes "$(stat -c %s "$d/shard_0.txt")" --scn-tokens "$(B2 get "$OUT/scenario_sample.json" tokens)" \
      --scn-docs "$(B2 get "$d/shard_0.ledger.json" documents_kept)" --shard-docs "$SHARD_DOCS" --max-drop "$MAX_DROP" \
      --out "$OUT/recipe_S.json"
    # shellcheck disable=SC2086
    stamp "$OUT/recipe_S.json" $bind
  fi
  shards=$(B2 get "$OUT/recipe_S.json" scenario_shards)
  say "scenario stream: $shards shards of $SHARD_DOCS documents, $JOBS at a time, validator $([ "$VALIDATE" = 1 ] && echo on every shard || echo off)"
  seq 1 $((shards - 1)) | xargs -r -P "$JOBS" -I{} bash "$0" "$OUT" "$V" _shard {} \
    || { echo "base2_data: a scenario shard was not built; stopping"; exit 1; }
  bind=""
  for k in $(seq 0 $((shards - 1))); do bind="$bind --input $d/shard_$k.ledger.json"; done
  # shellcheck disable=SC2086
  if ! cached "$WORK/scenario.txt" $bind; then
    : > "$WORK/scenario.txt.tmp"
    for k in $(seq 0 $((shards - 1))); do cat "$d/shard_$k.txt" >> "$WORK/scenario.txt.tmp"; done
    mv "$WORK/scenario.txt.tmp" "$WORK/scenario.txt"
    # shellcheck disable=SC2086
    stamp "$WORK/scenario.txt" $bind
  fi
  # shellcheck disable=SC2046
  B2 scn-ledger --dir "$d" --shards "$shards" --text "$WORK/scenario.txt" --recipe "$OUT/recipe_S.json" \
    --seed "$SEED" --max-drop "$MAX_DROP" $([ "$VALIDATE" = 1 ] && echo --validated) --out "$OUT/scenario.ledger.json"
  dialogue_copies
  MIX_DEPS="--input $OUT/scenario.ledger.json"
  mix_and_ids S "$OUT/recipe_S.json" --source "scenario=$WORK/scenario.txt"
  MIX_DEPS=""
  # S's natural documents are among N's: by the mixer's ledgers, and document by document against N's digests.
  B2 subset "$OUT/N.ledger.json" "$OUT/S.ledger.json" > "$OUT/subset_S_in_N.json"
  if ! cached "$OUT/subset_docs_S_in_N.json" --input "$OUT/S.ledger.json" --input "$OUT/N.keys" --input "$OUT/scenario.ledger.json"; then
    B2 subset-docs --n-keys "$OUT/N.keys" --scenario "$WORK/scenario.txt" --out "$OUT/subset_docs_S_in_N.json" \
      "$OUT/S.main.txt" "$OUT/S.tail.txt" > /dev/null
    stamp "$OUT/subset_docs_S_in_N.json" --input "$OUT/S.ledger.json" --input "$OUT/N.keys" --input "$OUT/scenario.ledger.json"
  fi
  # The registered share, measured on the mixed text by the real tokeniser: outside 14% to 16% the stage fails.
  for part in main tail; do
    bind="--input $OUT/S.ledger.json --input $OUT/ids_S.json --input $OUT/scenario.ledger.json --input $V"
    # shellcheck disable=SC2086
    if ! cached "$OUT/share_S.$part.json" $bind; then
      B2 share --mix "$OUT/S.$part.txt" --scenario "$WORK/scenario.txt" --vocab "$V" \
        --ids-record "$OUT/ids_S.json" --out "$OUT/share_S.$part.json" > /dev/null \
        || { echo "base2_data: scenario share out of bounds in the $part part"; exit 1; }
      # shellcheck disable=SC2086
      stamp "$OUT/share_S.$part.json" $bind
    fi
    [ "$(B2 get "$OUT/share_S.$part.json" ok)" = True ] || { echo "base2_data: scenario share out of bounds in the $part part"; exit 1; }
  done
  # N's and S's actual token counts, part by part; S further than BASE2_TOKEN_REL_MAX from N's fails the stage.
  B2 token-report --n-ids "$OUT/ids_N.json" --s-ids "$OUT/ids_S.json" --max-rel "$TOKEN_REL_MAX" --out "$OUT/token_counts.json" > /dev/null \
    || { echo "base2_data: S's token count is not N's (token_counts.json)"; exit 1; }
  PINS "$OUT" build-S
  B2 stage-ledger --out "$OUT/base2_stage3.built.json" --kv stage=3 --kv "scenario_seed=$SEED" \
    --file "S.main.ids=$OUT/S.main.ids" --file "S.tail.ids=$OUT/S.tail.ids" --file "panel_scenario.ids=$OUT/panel_scenario.ids" \
    --json "recipe=$OUT/recipe_S.json" --json "mix=$OUT/S.ledger.json" --jsonl "ids=$OUT/ids_S.json" \
    --json "scenario=$OUT/scenario.ledger.json" --json "scenario_sample=$OUT/scenario_sample.json" \
    --json "panel_scenario=$OUT/panel_scenario.json" --json "share_main=$OUT/share_S.main.json" \
    --json "share_tail=$OUT/share_S.tail.json" --json "token_counts=$OUT/token_counts.json" \
    --json "subset=$OUT/subset_S_in_N.json" --json "subset_documents=$OUT/subset_docs_S_in_N.json" \
    --json "engine_pins=$OUT/engine_pins_observed.json" --json "pins=$OUT/base2_pins_observed.json"
  say "stage 3 built"
}
publish3() {
  push_set main corpus/panel_scenario "$OUT/panel_scenario.ids" "$OUT/panel_scenario.json"
  push_set main corpus/S "$OUT/S.main.ids" "$OUT/S.tail.ids" "$OUT/S.ledger.json" "$OUT/ids_S.json" "$OUT/recipe_S.json" \
    "$OUT/scenario.ledger.json" "$OUT/share_S.main.json" "$OUT/share_S.tail.json" "$OUT/token_counts.json"
  B2 stage-ledger --out "$OUT/base2_stage3.json" --kv stage=3 --json "built=$OUT/base2_stage3.built.json" \
    --json "published_S=$OUT/published_corpus_S.json" --json "published_panel_scenario=$OUT/published_corpus_panel_scenario.json"
  push_set "$BB" build/stage3 "$OUT/base2_stage3.json" "$OUT/S.ledger.json" "$OUT/ids_S.json" "$OUT/recipe_S.json" \
    "$OUT/scenario.ledger.json"
  drop_mix_text S   # only now: the receipt is in the store
  rm -rf "$OUT/scn" "$WORK"
  say "stage 3 done"
}

rehearsal() {  # a main and a tail corpus of different sizes from the pinned 50 MB FineWeb-Edu sample
  if [ ! -s "$OUT/rehearsal.main.ids" ] || [ ! -s "$OUT/rehearsal.tail.ids" ] || [ ! -s "$OUT/rehearsal.json" ]; then
    if in_store main corpus/rehearsal; then
      pull_set main corpus/rehearsal "$OUT"
    else
      local text=corpus/data/fineweb_edu_sample_v1/pretrain_train.txt part
      if [ -n "${BASE2_FIXTURE:-}" ]; then text="$BASE2_FIXTURE/fw/sample.txt"; else need_python_packages; python3 corpus/fetchers/fineweb_edu.py sample_v1; fi
      mkdir -p "$WORK"
      B2 split-docs "$text" "$WORK/rehearsal.main.txt" "$WORK/rehearsal.tail.txt" "$REH_MAIN_BYTES" > /dev/null
      : > "$OUT/rehearsal.json.tmp"
      for part in main tail; do
        python3 tools/corpus_ids.py "$WORK/rehearsal.$part.txt" "$V" "$OUT/rehearsal.$part.ids.tmp" --threads "$THREADS" >> "$OUT/rehearsal.json.tmp"
        mv "$OUT/rehearsal.$part.ids.tmp" "$OUT/rehearsal.$part.ids"
      done
      sed -i "s/\.ids\.tmp\"/.ids\"/" "$OUT/rehearsal.json.tmp"
      mv "$OUT/rehearsal.json.tmp" "$OUT/rehearsal.json"
      rm -f "$WORK/rehearsal.main.txt" "$WORK/rehearsal.tail.txt"
      push_set main corpus/rehearsal "$OUT/rehearsal.main.ids" "$OUT/rehearsal.tail.ids" "$OUT/rehearsal.json"
    fi
  fi
  PINS "$OUT" rehearsal --upload "$(dirname "$V")"
}

loadprobe() {  # PROBE_OUT: S's ids and the panels are on this VM, checked against the store's manifests, before the load
  # A VM on which every stage was skipped holds the stages' receipts and none of their ids. hf_store.py pull keeps a
  # local file whose sha256 is the manifest's and fetches the rest, so this is the check on an unbroken build too.
  in_store main corpus/S || { echo "base2_data: main:corpus/S is not in the store; there is nothing to load"; exit 1; }
  pull_set main corpus/S "$OUT"
  pull_set main corpus/panels "$OUT"
  pull_set main corpus/panel_scenario "$OUT"
  PINS "$OUT" build-S
  B2 loadprobe --spec $CC/base2_load_probe.json --out "${1:?probe out dir}"
}

run_stage() {  # N: skip only on the store's word; otherwise build what is missing, then publish, receipt last
  local n=$1 s
  if in_store "$BB" "build/stage$n"; then
    for s in $(sets_of "$n"); do
      in_store "${s%%:*}" "${s##*:}" || { echo "base2_data: the store holds stage $n's receipt and not $s; stopping"; exit 1; }
    done
    [ -s "$OUT/base2_stage$n.json" ] || pull_set "$BB" "build/stage$n" "$OUT"
    say "stage $n: finished already"
    return 0
  fi
  rm -f "$OUT/base2_stage$n.json"   # a receipt the store does not hold is not a receipt
  "build$n"
  "publish$n"
}
# The vocabulary is the pinned one before anything is tokenised with it.
[ "$MODE" = _shard ] || PINS "$OUT" vocab --upload "$(dirname "$V")"
case "$MODE" in
  stage1) run_stage 1 ;;
  stage2) run_stage 2 ;;
  stage3) run_stage 3 ;;
  all) run_stage 1; run_stage 2; run_stage 3 ;;
  rehearsal) rehearsal ;;
  loadprobe) loadprobe "${4:-}" ;;
  _shard) scn_shard "${4:?shard}" ;;
  *) echo "mode is stage1, stage2, stage3, all, rehearsal or loadprobe"; exit 1 ;;
esac
