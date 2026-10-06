#!/usr/bin/env bash
# Package a trained chat run as the download quickstart_chat.sh's "ready-made model" fetches.
#
#   bash tools/package_chat_model.sh runs/tinychat-better [out.tar.gz]
#
# The archive holds one top-level folder with only what `mtstudio chat` and a reader need:
# spec.json, the exported weights (safetensors), the exported GGUF (which also carries the vocabulary, so the
# package needs no data/ or releases/ files), result.json and the model card. Optimiser
# state, events and logs stay behind.
set -euo pipefail
RUN="${1:?usage: package_chat_model.sh <run_dir> [out.tar.gz]}"
RUN="${RUN%/}"
NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$RUN/spec.json")"
OUT="${2:-$NAME.tar.gz}"

for f in spec.json "$NAME.safetensors" result.json; do
  [ -f "$RUN/$f" ] || { echo "missing $RUN/$f (train and export the run first)"; exit 1; }
done
[ -f "$RUN/card.md" ] || echo "note: no card.md; run python3 tools/model_card.py $RUN --probe first for a complete package"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/$NAME"
cp "$RUN/spec.json" "$RUN/$NAME.safetensors" "$RUN/result.json" "$STAGE/$NAME/"
# The GGUF carries the vocabulary; families that export none (gpt2) take it from the spec's vocab file.
if [ -f "$RUN/$NAME.gguf" ]; then cp "$RUN/$NAME.gguf" "$STAGE/$NAME/"
else
  VOCAB="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["vocab"])' "$RUN/spec.json")"
  [ -f "$VOCAB" ] || { echo "no GGUF and no vocab file $VOCAB (run from the training directory)"; exit 1; }
  cp "$VOCAB" "$STAGE/$NAME/$NAME.gguf"
fi
for f in card.md card.json; do [ -f "$RUN/$f" ] && cp "$RUN/$f" "$STAGE/$NAME/"; done
tar -czf "$OUT" -C "$STAGE" "$NAME"
echo "packaged $OUT ($(du -h "$OUT" | cut -f1)): $(tar -tzf "$OUT" | tr '\n' ' ')"
command -v sha256sum >/dev/null && sha256sum "$OUT"
