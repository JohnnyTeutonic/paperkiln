#!/usr/bin/env bash
# Sync the vendored transformer_cpp sources from the sibling development
# tree. Run before pushing when transformer_cpp changed. (The sibling wins
# locally; clones build from the vendored copy.)
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/../coalfire.cpp"
DST="$HERE/third_party/transformer_cpp"
[ -f "$SRC/CMakeLists.txt" ] || { echo "no sibling transformer_cpp"; exit 1; }
rm -rf "$DST" && mkdir -p "$DST"
cp "$SRC/CMakeLists.txt" "$DST/"
rsync -a --exclude='__pycache__' "$SRC/include/" "$DST/include/"
rsync -a --exclude='__pycache__' "$SRC/src/" "$DST/src/"
rsync -a --exclude='__pycache__' "$SRC/third_party/" "$DST/third_party/"
echo "vendored: $(du -sh "$DST" | cut -f1)"
