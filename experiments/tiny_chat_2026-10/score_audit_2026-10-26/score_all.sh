# CPU `mtstudio score` (WSL build at d53c6939) over the rebuilt Carry-5 chunks; resumable per chunk.
set -e
SAW=/mnt/c/ml_artifacts/score_audit/work
BIN=~/sa_build_d53c/mtstudio
export OMP_NUM_THREADS=2
unset MICROTORCH_DEVICE MICROTORCH_DEVICE_OPS
mkdir -p ~/sa_models/ctl_r1
ln -sf /mnt/c/ml_artifacts/benchmarks/weights/pick_e1_L16.safetensors ~/sa_models/ctl_r1/tc_sft_L16_r1.safetensors
ln -sf /mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf ~/sa_models/ctl_r1/tc_sft_L16_r1.gguf
for run in ${RUNS:-ctl_r1 x5_r1 x5_r2 x5_r3}; do
  mkdir -p ~/sa_out/$run
  for inp in $SAW/sa_$run/chunk*.in.jsonl; do
    b=$(basename $inp .in.jsonl)
    o=~/sa_out/$run/$b.out.jsonl
    [ -f $o ] && continue
    t0=$(date +%s)
    $BIN score $SAW/sa_spec_$run.json --in $inp --out $o.tmp 2>> ~/sa_out/$run/stderr.txt
    mv $o.tmp $o
    echo "$run $b $(( $(date +%s) - t0 )) s"
  done
  mkdir -p $SAW/sa_$run/cpu
  cp ~/sa_out/$run/*.out.jsonl $SAW/sa_$run/cpu/
done
echo done
