#!/usr/bin/env bash
# The 90-second demo, scripted (STUDIO_PLAN M3): paste an arXiv ID ->
# provenance-carrying extraction -> train a model -> chat with it in a
# separately-built inference engine. This is the take a screen recording
# runs through; every step prints what it just proved.
#
#   tools/demo.sh CORPUS.txt VOCAB.gguf [TINYLLAMA_BIN]
#
# Requires: built mtstudio in cwd or ./build; python3 + requests for the
# arXiv step (skipped gracefully offline); tinyllama binary optional for
# the final chat step.
set -e
cd "$(dirname "$0")/.."   # repo root, wherever the script is launched from
_SIB=$(cd .. 2>/dev/null && pwd)/coalfire.cpp
CORPUS=${1:-${DEMO_CORPUS:-$_SIB/data/tinystories-txt/train-0-small.txt}}
VOCAB=${2:-${DEMO_VOCAB:-$_SIB/releases/chat7b.gguf}}
TL=${3:-${DEMO_TL:-$HOME/tlbuild/tinyllama}}
[ -f "$CORPUS" ] || { echo "corpus not found: $CORPUS"; \
  echo "usage: demo.sh [CORPUS.txt] [VOCAB.gguf] [TINYLLAMA_BIN]"; exit 1; }
[ -f "$VOCAB" ] || { echo "vocab gguf not found: $VOCAB"; \
  echo "usage: demo.sh [CORPUS.txt] [VOCAB.gguf] [TINYLLAMA_BIN]"; exit 1; }
MT=$(command -v ./mtstudio || command -v build/mtstudio || echo "$HOME/mtrel/mtstudio")
OUT=/tmp/mtdemo
step() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }

step "1/5  paper -> architecture, every value with its evidence"
python3 papers/fetch.py 1706.03762 --json /tmp/mtdemo_arch.json \
    --emit-html /tmp/mtdemo_paper.html 2>/dev/null \
  && echo "     (open /tmp/mtdemo_paper.html for the diff-to-paper view)" \
  || echo "     [offline: skipping the live arXiv fetch — fixture-tested in CI]"

step "2/5  one spec file describes the whole lifecycle"
rm -rf "$OUT"
cat > /tmp/mtdemo_spec.json <<EOF
{
  "name": "demo",
  "arch": {"preset": "llama-tiny"},
  "data": {"corpus": "$CORPUS", "vocab": "$VOCAB", "vocab_cap": 4096, "T": 128},
  "train": {"steps": ${DEMO_STEPS:-300}, "batch": 4, "eval_every": 50,
             "checkpoint_every": 100,
             "early_stopping": {"patience": 6, "min_delta": 0.003}},
  "export": {"formats": ["safetensors", "gguf"]},
  "serve": {"on_finish": true},
  "out_dir": "$OUT"
}
EOF
"$MT" plan /tmp/mtdemo_spec.json

step "3/5  train (RMSNorm + RoPE + SwiGLU on a hand-derived, FD-checked tape)"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=PASSIVE
# --line-buffered so eval events stream as they happen even when piped
# (mtstudio already fflushes per event; the pipe must not re-buffer).
# DEMO_VERBOSE=1 keeps the per-step lines — the live heartbeat a
# recording wants; default keeps just the milestones.
FILTER='"eval"|"done"|"export"'
[ -n "${DEMO_VERBOSE:-}" ] && FILTER='"step"|"eval"|"done"|"export"'
"$MT" run /tmp/mtdemo_spec.json | grep --line-buffered -E "$FILTER"
echo "     (mtstudio serve $OUT 8080 tails this live: loss curve + node-graph glow)"

step "4/5  the run is now an Atlas data point"
python3 tools/atlas_extract.py "$OUT" | head -24

step "5/5  chat with it — a separately written inference engine reads the GGUF"
if [ -n "$TL" ] && [ -x "$TL" ]; then
  "$TL" "$OUT/demo.gguf" "$OUT/demo.gguf" 4 prompt "once upon a time" \
      --max-tokens 40 -ngl 0 --top-k 1 --raw-prompt 2>/dev/null | tail -2
else
  echo "     tinyllama <out>/demo.gguf <out>/demo.gguf 4 prompt \"once upon a time\" ..."
  echo "     [pass the tinyllama binary as arg 3 to run this step live]"
fi

step "done — paper to chatting model, one stack, every claim receipted"
