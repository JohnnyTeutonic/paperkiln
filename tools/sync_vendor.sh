#!/usr/bin/env bash
# Copy a sibling coalfire.cpp tree over the vendored third_party/transformer_cpp.
#
# The vendored copy is canonical (CMakeLists.txt builds from it unless
# MICROTORCH_TCPP_DIR or MICROTORCH_TCPP_SIBLING says otherwise), and paperkiln's
# own transformer_core changes land there first. So this script refuses to run
# while the two trees differ, listing the differing files, unless
# --overwrite-vendored is given: bring the sibling up to date first.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/../coalfire.cpp"
DST="$HERE/third_party/transformer_cpp"
[ -f "$SRC/CMakeLists.txt" ] || { echo "no sibling coalfire.cpp"; exit 1; }
if [ "$1" != "--overwrite-vendored" ]; then
    differ="$(diff -rq -x __pycache__ "$SRC/include" "$DST/include"; \
              diff -rq -x __pycache__ "$SRC/src" "$DST/src"; \
              diff -q "$SRC/CMakeLists.txt" "$DST/CMakeLists.txt")" || true
    if [ -n "$differ" ]; then
        echo "the sibling and the vendored copy differ; the vendored copy is canonical:"
        echo "$differ"
        echo "port the vendored changes to the sibling, or pass --overwrite-vendored"
        exit 1
    fi
fi
rm -rf "$DST" && mkdir -p "$DST"
cp "$SRC/CMakeLists.txt" "$DST/"
rsync -a --exclude='__pycache__' "$SRC/include/" "$DST/include/"
rsync -a --exclude='__pycache__' "$SRC/src/" "$DST/src/"
rsync -a --exclude='__pycache__' "$SRC/third_party/" "$DST/third_party/"
echo "vendored: $(du -sh "$DST" | cut -f1)"
