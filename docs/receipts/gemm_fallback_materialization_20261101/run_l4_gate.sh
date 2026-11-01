#!/usr/bin/env bash
# Run only inside a separately root-admitted serial L4 campaign.
set -Eeuo pipefail
: "${BUILD:?existing admitted CUDA build directory}"
: "${OUT:?new output directory}"
[ ! -e "$OUT" ]; mkdir "$OUT"
cmake --build "$BUILD" --target gemm_fallback_fixed gemm_fallback_old --parallel 2
for variant in old fixed; do
 for transition in active cpu stepoff; do
  set +e
  "$BUILD/gemm_fallback_$variant" --require-cuda --transition "$transition" > "$OUT/${variant}_${transition}.txt" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$OUT/${variant}_${transition}.rc"
 done
done
