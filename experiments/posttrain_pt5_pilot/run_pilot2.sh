#!/usr/bin/env bash
# posttrain_pt5_pilot round 2 (PREREGISTRATION.md, Amendment 1): beta 0.1, lr 1e-3,
# NLL coefficient in {0.2, 1.0, 5.0}. Reuses round 1's base and samples (WORK/beta_0.1).
#   bash experiments/posttrain_pt5_pilot/run_pilot2.sh WORK MTSTUDIO VERIFY
set -euo pipefail
WORK=$1; MT=$2; VF=$3
D=experiments/posttrain_pt5/run_pt5.py
for A in 0.2 1.0 5.0; do
  mkdir -p "$WORK/nll_$A/seed_391"
  for f in base base_spec.json sampling_prompts.jsonl samples.jsonl refs_test.jsonl refs_hackprobe.jsonl \
           eval_base_test.jsonl eval_base_hackprobe.jsonl; do
    cp -r "$WORK/beta_0.1/seed_391/$f" "$WORK/nll_$A/seed_391/"
  done
done
python3 $D --work "$WORK/nll_0.2" --mtstudio "$MT" --verify "$VF" --jobs 1 --seeds 391 --beta 0.1 --nll 0.2 --arms E J R Mu &
python3 $D --work "$WORK/nll_1.0" --mtstudio "$MT" --verify "$VF" --jobs 1 --seeds 391 --beta 0.1 --nll 1.0 --arms E J R Mu &
python3 $D --work "$WORK/nll_5.0" --mtstudio "$MT" --verify "$VF" --jobs 1 --seeds 391 --beta 0.1 --nll 5.0 --arms E J R Mu
wait
python3 experiments/posttrain_pt5_pilot/analyze_pilot.py "$WORK" --round2
