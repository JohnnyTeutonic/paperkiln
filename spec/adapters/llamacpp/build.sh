#!/usr/bin/env bash
# Build the llama.cpp conformance adapter against a built llama.cpp tree (shared libraries).
#
#   LLAMACPP=~/llamacpp_b6000_build OUT=~/lc_keramos bash build.sh
#
# LLAMACPP is a llama.cpp checkout built with -DBUILD_SHARED_LIBS=ON (build/bin/libllama.so); OUT is where
# the binary goes (WSL ext4, not OneDrive). The llama.cpp commit is compiled into hello's "version".
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
L="${LLAMACPP:-$HOME/llamacpp_b6000_build}"
OUT="${OUT:-$HOME/lc_keramos}"
mkdir -p "$OUT"
commit="$(git -C "$L" rev-parse HEAD 2>/dev/null || echo unknown)"
tag="$(git -C "$L" describe --tags --exact-match 2>/dev/null || true)"
label="${tag:+$tag }$commit"
g++ -O2 -std=c++17 -Wall -Wno-unused-result \
    -DLC_LLAMA_COMMIT="\"$label\"" \
    -I"$L/include" -I"$L/ggml/include" -I"$L/vendor" \
    "$here/llamacpp_adapter.cpp" \
    -L"$L/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin" \
    -o "$OUT/llamacpp_adapter"
echo "built $OUT/llamacpp_adapter (llama.cpp $label)"
