#!/usr/bin/env bash
# The CUDA leg of the 26 Oct 2026 score audit (experiments/tiny_chat_2026-10/SCORE_AUDIT_2026-10-26.md;
# tools/colab_chat/score_audit_v1.json): the pod's CUDA `mtstudio score`, in the jobs' environment, on
#   - the benchmarks cross-check's chunk 0 (its first 128 requests, HellaSwag 17234's first item at rows 12-15)
#     followed by the four requests of the second item that bench_v1.jsonl also calls 17234, on pick_e1_L16;
#   - Carry-5 chunk 0 of arm X's control seed 1 (pick_e1_L16) and of X5 seed 1, the requests the post jobs
#     scored (rebuilt from the runs' gen.jsonl at home).
# Each through tools/score_sequence_check.py: in order, reversed, one request per process (the first 64 of a
# Carry-5 chunk), and against the home CPU build's output for the same file (cpu_*.out.jsonl).
# Uploads (/content/chat/upload): pick_e1_L16.safetensors, tc_sft_X5_r1.safetensors, vocab_llama3_bpe.gguf,
# spec_pick_e1_L16.json, spec_x5_r1.json, the three *.in.jsonl and their cpu_*.out.jsonl
# (C:/ml_artifacts/score_audit/inputs). Ends with OUT/results.tar.gz; exits 1 when a gate fails.
set -uo pipefail
OUT=$1
mkdir -p "$OUT"
cd /content/pk
U=/content/chat/upload
export MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 MICROTORCH_ATTN_BLOCKED=1 MICROTORCH_CUBLAS=1 \
  MICROTORCH_STEP_RESIDENCY=1 MICROTORCH_DEFER_DOWNLOADS=1
MT=/content/pk/build_cuda/mtstudio
mkdir -p /content/sa/pick_e1_L16 /content/sa/x5_r1
cp $U/pick_e1_L16.safetensors /content/sa/pick_e1_L16/tc_sft_L16_r1.safetensors
cp $U/tc_sft_X5_r1.safetensors /content/sa/x5_r1/tc_sft_X5_r1.safetensors
sha256sum /content/sa/*/*.safetensors $U/*.in.jsonl $U/cpu_*.out.jsonl $U/spec_*.json > "$OUT/staged.sha256"
git -C /content/pk rev-parse HEAD > "$OUT/commit.txt"
rc=0
check() {  # NAME SPEC IN REF ALONE
  python3 tools/score_sequence_check.py --mtstudio $MT --spec "$2" --in "$3" --out "$OUT/$1" --alone "$5" \
    --tol 1e-4 --ref "$4" --ref-tol 1e-3 > "$OUT/$1.log" 2>&1 || rc=1
}
check bench_l16 $U/spec_pick_e1_L16.json $U/bench_xcheck_chunk0_plus_17234b.in.jsonl \
  $U/cpu_bench_xcheck_chunk0_plus_17234b.out.jsonl 0
check carry5_ctl_r1 $U/spec_pick_e1_L16.json $U/carry5_ctl_r1_chunk0000.in.jsonl $U/cpu_carry5_ctl_r1_chunk0000.out.jsonl 64
check carry5_x5_r1 $U/spec_x5_r1.json $U/carry5_x5_r1_chunk0000.in.jsonl $U/cpu_carry5_x5_r1_chunk0000.out.jsonl 64
cp /content/chat/provenance.json "$OUT/" 2>/dev/null || true
nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || true
tar -czf /tmp/score_audit.tar.gz -C "$OUT" --exclude='*.tar.gz' . && mv /tmp/score_audit.tar.gz "$OUT/results.tar.gz"
exit $rc
