#!/usr/bin/env bash
# VM side of the PREVIEW chat SFT (18 Oct 2026). Runs on the base run's own
# VM (one Colab session at a time), beside the base training, from an early
# base checkpoint, so a chat model exists to talk to hours before the base
# finishes. The full chain is repeated from the final base checkpoint.
#   bash sft_preview.sh FROM_STEP [TAG] [MTSTUDIO]
# TAG (default sft) names the output directory /content/chat/TAG and the
# status file (STATUS_SFT for the default, STATUS_TAG otherwise), so a later
# preview never overwrites an earlier one. MTSTUDIO defaults to the boot
# build; a build with paperkiln c939965 or later also writes the BPE merges
# into the GGUF.
# Steps:
#   1. wait until the base checkpoint reaches FROM_STEP, then copy
#      model.safetensors, keeping it only if state.txt named the same step
#      before and after the copy (a torn copy would load mixed weights);
#   2. build SFT examples in the served chat template (tools/chat_data.py),
#      once: later previews reuse the same examples;
#   3. freeze the held-out eval prompts (tinychat_hf.py eval_v1 --pin), once;
#   4. fine-tune; the status file says where it is.
set -euo pipefail
FROM=$1
TAG=${2:-sft}
BIN=${3:-/content/pk/build_cuda/mtstudio}
C=/content/chat
ST=$C/STATUS_SFT
[ "$TAG" = sft ] || ST=$C/STATUS_$TAG
INITD=$C/${TAG}_init
LOG=$C/$TAG.log
mkdir -p $INITD $C/$TAG
exec >> "$LOG" 2>&1
echo "=== sft preview $(date -u) from step $FROM, tag $TAG, binary $BIN ==="
echo WAITING > $ST
cd /content/pk
while true; do
  s1=$(head -1 $C/base/state.txt 2>/dev/null || echo 0)
  if [ "${s1:-0}" -ge "$FROM" ]; then
    cp $C/base/model.safetensors $INITD/model.safetensors.tmp
    s2=$(head -1 $C/base/state.txt 2>/dev/null || echo 0)
    if [ "$s1" = "$s2" ]; then
      mv -f $INITD/model.safetensors.tmp $INITD/base_step$s1.safetensors
      echo "init from base step $s1"
      break
    fi
  fi
  sleep 60
done
INIT=$(ls $INITD/base_step*.safetensors | head -1)

D=corpus/data/tinychat_hf_full_v1
V=corpus/data/vocab_tinychat_bpe8k_v1/tinychat_bpe8k.gguf
[ -s $C/sft_examples.jsonl ] ||
  python3 tools/chat_data.py sft $D/dialogues_train.jsonl --vocab $V --T 256 --max 24000 --out $C/sft_examples.jsonl
if [ ! -s $C/tinychat_hf_prompts_v1.jsonl ]; then
  python3 corpus/fetchers/tinychat_hf.py eval_v1 --pin
  cp corpus/eval/tinychat_hf_prompts_v1.jsonl corpus/hashes/tinychat_hf_prompts_v1.sha256 $C/ 2>/dev/null || true
fi

cat > $C/${TAG}_spec.json <<EOF
{"name": "chat_$TAG", "arch": {"preset": "llama-tiny", "custom": {"d": 384, "layers": 6, "heads": 6}},
 "data": {"vocab": "/content/pk/$V", "vocab_cap": 8192, "T": 256, "sft_examples": "$C/sft_examples.jsonl"},
 "train": {"steps": 1500, "lr": 0.0003, "batch": 16, "eval_every": 250, "checkpoint_every": 250,
           "gradmap_every": 0, "seed": 1, "optimizer": "adamw", "objective": "sft", "init_from": "$INIT"},
 "export": {"formats": ["safetensors", "gguf"]}, "out_dir": "$C/$TAG"}
EOF
echo RUNNING > $ST
MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_STEP_RESIDENCY=1 \
MICROTORCH_DEFER_DOWNLOADS=1 MICROTORCH_CUBLAS=1 OMP_NUM_THREADS=4 \
  $BIN run $C/${TAG}_spec.json > $C/${TAG}_train.log 2>&1 \
  && echo FINISHED > $ST || echo FAILED > $ST
echo "done $(date -u)"
