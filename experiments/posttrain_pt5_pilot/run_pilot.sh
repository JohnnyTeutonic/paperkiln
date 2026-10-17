#!/usr/bin/env bash
# posttrain_pt5_pilot (PREREGISTRATION.md). Run from the repository root of a
# clean checkout of the licence commit:  bash experiments/posttrain_pt5_pilot/run_pilot.sh WORK MTSTUDIO VERIFY
# The base and its samples are produced once (beta 0.1) and copied, so the
# three betas differ only in the DPO runs.
set -euo pipefail
WORK=$1; MT=$2; VF=$3
D=experiments/posttrain_pt5/run_pt5.py
python3 $D --work "$WORK/beta_0.1" --mtstudio "$MT" --verify "$VF" --jobs 1 --seeds 391 --beta 0.1 --arms E J R Mu
for B in 0.5 2.0; do
  mkdir -p "$WORK/beta_$B/seed_391"
  for f in base base_spec.json sampling_prompts.jsonl samples.jsonl refs_test.jsonl refs_hackprobe.jsonl \
           eval_base_test.jsonl eval_base_hackprobe.jsonl; do
    cp -r "$WORK/beta_0.1/seed_391/$f" "$WORK/beta_$B/seed_391/"
  done
done
python3 $D --work "$WORK/beta_0.5" --mtstudio "$MT" --verify "$VF" --jobs 1 --seeds 391 --beta 0.5 --arms E J R Mu &
python3 $D --work "$WORK/beta_2.0" --mtstudio "$MT" --verify "$VF" --jobs 1 --seeds 391 --beta 2.0 --arms E J R Mu
wait
python3 experiments/posttrain_pt5_pilot/analyze_pilot.py "$WORK"
