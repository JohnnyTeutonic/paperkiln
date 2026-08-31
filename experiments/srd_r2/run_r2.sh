#!/usr/bin/env bash
# SRD rung 2 (srd_r2/PREREGISTRATION_R2.md): the 2x2 x 3 seeds, polite profile.
#   experiments/srd_r2/run_r2.sh [STEPS] [OUT_DIR]
# Resumable at cell granularity: a cell whose probe CSV already has its
# final step is skipped, so this can be interrupted and relaunched.
set -e
STEPS=${1:-600}
OUT=${2:-/tmp/srd_r2}
BIN=${SRD_BIN:-$HOME/mtrel/srd_needle}
T=256; D=128; B=4; NPAIRS=8; NKEYS=64
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=PASSIVE
mkdir -p "$OUT"

for needle in distinct indist; do
  for decoys in 0 2; do
    for seed in 1 2 3; do
      cell="${needle}_d${decoys}_s${seed}"
      prefix="$OUT/$cell"
      if [ -f "${prefix}_probe.csv" ] && \
         grep -q "^${STEPS}," "${prefix}_probe.csv" 2>/dev/null; then
        echo "== $cell  (done, skipped)"
        continue
      fi
      echo "== $cell"
      SRD_CKPT_DIR="$OUT/ckpt_$cell" nice -n 19 "$BIN" \
        "$STEPS" "$T" "$D" "$prefix" "$B" "$NPAIRS" "$NKEYS" "$seed" \
        "$needle" "$decoys" 2>&1 | tail -3
    done
  done
done
echo "R2-SWEEP-DONE -> $OUT (analyse: python3 tools/srd_r2_analyze.py $OUT)"
