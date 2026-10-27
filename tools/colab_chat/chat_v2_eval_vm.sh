#!/usr/bin/env bash
# Conversation schema v2's measurement on the VM (experiments/chat_general/RESULTS.md, "Conversation schema v2"):
# IF-tiny with the placement set (tools/iftiny_eval.py --placement) and, with --tools, the tool-call set
# (tools/chat_v2_eval.py tools), generated greedily by ember's CUDA server (ember b49eb5c, the first ember with
# paperkiln/2; EMBER_GEMM_TF32=0, EMBER_DETERMINISTIC=1), with `mtstudio score` re-scoring every reply that
# ended on a stop token (--parity). `mtstudio chat` generated at about 17 s a prompt on the pod (27 Oct 2026,
# pk-chat-v2's first launch), ten hours a model, so ember generates and mtstudio checks.
#
#   bash tools/colab_chat/chat_v2_eval_vm.sh OUT TAG SPEC [--tools] [--parity]
#   bash tools/colab_chat/chat_v2_eval_vm.sh OUT TAG --stage UPLOAD_NAME VOCAB_UPLOAD [--tools] [--parity]
#
# --stage links /content/chat/upload/UPLOAD_NAME.safetensors as /content/eval/TAG/<name>.safetensors under
# experiments/chat_general/specs/TAG.spec.json with data.vocab the upload VOCAB_UPLOAD, and exports its F32
# GGUF there with mtstudio (pk_sft_v1, uploaded). Writes OUT/eval_TAG.tar.gz; resumable.
set -euo pipefail
OUT=$1
TAG=$2
shift 2
REPO=/content/pk
MT=$REPO/build_cuda/mtstudio
EMBER_COMMIT=b49eb5c16852e1120d10db82b9616f05c754b8c2
cd "$REPO"
W="$OUT/ember"
mkdir -p "$W"
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
  G=$(python3 -c 'import json, os, sys; s = json.load(open(sys.argv[1])); print(os.path.join(s["out_dir"], s["name"] + ".gguf"))' "$SPEC")
  if [ ! -f "$G" ]; then
    "$MT" export "$SPEC" --from "$(dirname "$G")/$(basename "$G" .gguf).safetensors" --formats gguf > "$W/export_$TAG.log" 2>&1
  fi
else
  SPEC=$1
  shift
fi
TOOLS=0
PARITY=""
for a in "$@"; do
  case "$a" in
    --tools) TOOLS=1 ;;
    --parity) PARITY="--mtstudio $MT" ;;
  esac
done
python3 tools/colab_chat/ensure_ember_vm.py "$W/ember_build.json" "$W/ember_build.log" "$EMBER_COMMIT"
BIN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bin"])' "$W/ember_build.json")
export EMBER_GPU_LAYERS=-1 EMBER_GEMM_TF32=0 EMBER_DETERMINISTIC=1
python3 tools/iftiny_eval.py run --gen ember --ember-bin "$BIN" --model "$TAG=$SPEC" --placement \
  --out "$W/iftiny" --port 8095 $PARITY > "$W/iftiny_$TAG.log" 2>&1
PARTS="ember/iftiny/$TAG ember/iftiny_$TAG.log ember/ember_build.json"
if [ "$TOOLS" = 1 ]; then
  python3 tools/chat_v2_eval.py tools --gen ember --ember-bin "$BIN" --model "$TAG=$SPEC" --out "$W/tools" \
    --port 8096 > "$W/tools_$TAG.log" 2>&1
  PARTS="$PARTS ember/tools ember/tools_$TAG.log"
fi
tar czf "$OUT/eval_$TAG.tar.gz.tmp" -C "$OUT" $PARTS
mv "$OUT/eval_$TAG.tar.gz.tmp" "$OUT/eval_$TAG.tar.gz"
