#!/usr/bin/env bash
# Conversation schema v2's measurement on the VM (experiments/chat_general/RESULTS.md, "Conversation schema v2"):
# IF-tiny with the placement set (tools/iftiny_eval.py --placement) and, with --tools, the tool-call set
# (tools/chat_v2_eval.py tools), generated greedily by `mtstudio chat --kv-cache` on the CPU (paperkiln's own
# engine: it serves paperkiln/2 models before ember does).
#
#   bash tools/colab_chat/chat_v2_eval_vm.sh OUT TAG SPEC [--tools]
#   bash tools/colab_chat/chat_v2_eval_vm.sh OUT TAG --stage UPLOAD_NAME VOCAB_UPLOAD [--tools]
#
# --stage links /content/chat/upload/UPLOAD_NAME.safetensors as /content/eval/TAG/<name>.safetensors under
# experiments/chat_general/specs/TAG.spec.json with data.vocab the upload VOCAB_UPLOAD (pk_sft_v1, uploaded).
# Writes OUT/eval_TAG.tar.gz (iftiny/TAG, tools/TAG_*); resumable (both harnesses append and skip done items).
set -euo pipefail
OUT=$1
TAG=$2
shift 2
REPO=/content/pk
MT=$REPO/build_cuda/mtstudio
cd "$REPO"
mkdir -p "$OUT"
if [ "${1:-}" = "--stage" ]; then
  UP=$2
  VOC=$3
  shift 3
  SPEC=$(python3 - "$TAG" "$UP" "$VOC" <<'PY'
import json, os, sys
tag, up, voc = sys.argv[1:4]
s = json.load(open(f"experiments/chat_general/specs/{tag}.spec.json"))
s["data"]["vocab"] = f"/content/chat/upload/{voc}"
d = s["out_dir"] = f"/content/eval/{tag}"
os.makedirs(d, exist_ok=True)
link = f"{d}/{s['name']}.safetensors"
if not os.path.islink(link):
    os.symlink(f"/content/chat/upload/{up}.safetensors", link)
json.dump(s, open(f"{d}/spec.json", "w"), indent=1)
print(f"{d}/spec.json")
PY
)
else
  SPEC=$1
  shift
fi
export OMP_NUM_THREADS=${EVAL_THREADS:-4}
python3 tools/iftiny_eval.py run --gen mtstudio --mtstudio "$MT" --model "$TAG=$SPEC" --placement \
  --out "$OUT/iftiny" --port 8095 > "$OUT/iftiny_$TAG.log" 2>&1
PARTS="iftiny/$TAG"
if [ "${1:-}" = "--tools" ]; then
  python3 tools/chat_v2_eval.py tools --mtstudio "$MT" --model "$TAG=$SPEC" --out "$OUT/tools" --port 8096 \
    > "$OUT/tools_$TAG.log" 2>&1
  PARTS="$PARTS tools"
fi
tar czf "$OUT/eval_$TAG.tar.gz.tmp" -C "$OUT" $PARTS "iftiny_$TAG.log"
mv "$OUT/eval_$TAG.tar.gz.tmp" "$OUT/eval_$TAG.tar.gz"
