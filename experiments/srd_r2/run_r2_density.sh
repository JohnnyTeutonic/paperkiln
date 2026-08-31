#!/usr/bin/env bash
# SRD rung 2, P5 control lane (srd_r2/PREREGISTRATION_R2.md): matched-density
# evaluation on the checkpoints the 2x2 sweep trained. Run AFTER
# run_r2.sh completes; training no-ops on finished checkpoints and only
# the density evaluation executes.
#   experiments/srd_r2/run_r2_density.sh [STEPS] [OUT_DIR]
set -e
STEPS=${1:-600}
OUT=${2:-/tmp/srd_r2}
BIN=${SRD_BIN:-$HOME/mtrel/srd_needle}
T=256; D=128; B=4; NPAIRS=8; NKEYS=64
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=PASSIVE

for needle in distinct indist; do
  for decoys in 0 2; do
    for seed in 1 2 3; do
      cell="${needle}_d${decoys}_s${seed}"
      prefix="$OUT/$cell"
      if [ -f "${prefix}_density.csv" ]; then
        echo "== $cell (density done, skipped)"
        continue
      fi
      echo "== $cell density"
      SRD_CKPT_DIR="$OUT/ckpt_$cell" SRD_DENSITY_EVAL=1 nice -n 19 "$BIN" \
        "$STEPS" "$T" "$D" "$prefix" "$B" "$NPAIRS" "$NKEYS" "$seed" \
        "$needle" "$decoys" 2>&1 | grep -E "density|resum" | tail -9
    done
  done
done
echo "R2-DENSITY-DONE -> $OUT"
