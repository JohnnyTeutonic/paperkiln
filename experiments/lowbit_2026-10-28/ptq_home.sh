#!/bin/bash
# Home half of the PTQ measurement (LOCAL_LLM_WORK R3; experiments/lowbit_2026-10-28): the F32, Q8_0, Q4_K_M,
# Q4_K_S and Q4_0 exports of the four models with the CPU mtstudio (sha256s and sizes, held against the VM's
# exports of the same uploads), then greedy.py's ember runs on the CPU. Run under the CPU budget at 2 cores:
#   python scripts/cpu_budget.py run --session paperkiln --cores 2 ... -- wsl.exe --cd ~ bash ptq_home.sh
# Resumes: finished exports and finished greedy files are skipped.
set -euo pipefail
X=$(dirname "$(readlink -f "$0")")
MT=${MT:-$HOME/r3_build/mtstudio}
W=${R3_WORK:-$HOME/r3_ptq}
VOCAB=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf
export OMP_NUM_THREADS=2 MICROTORCH_DEVICE=cpu CUDA_VISIBLE_DEVICES=
declare -A SRC=(
  [pick_e1_L16]=/mnt/c/ml_artifacts/benchmarks/weights/pick_e1_L16.safetensors
  [pick_e1_60m_L32]=/mnt/c/ml_artifacts/benchmarks/weights/pick_e1_60m_L32.safetensors
  [pkbase]=/mnt/c/ml_artifacts/next_base/base/pk_base_v1.safetensors
  [pksft]=/mnt/c/ml_artifacts/general_chat/sft/pk_sft_v1.safetensors
)
for tag in pick_e1_L16 pick_e1_60m_L32 pkbase pksft; do
  d=$W/$tag
  mkdir -p "$d"
  name=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$X/../benchmarks_2026-10-26/specs/$tag.spec.json")
  python3 - "$X/../benchmarks_2026-10-26/specs/$tag.spec.json" "$d" "$VOCAB" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
s["data"]["vocab"] = sys.argv[3]
s["out_dir"] = sys.argv[2]
json.dump(s, open(sys.argv[2] + "/spec.json", "w"), indent=1)
EOF
  ln -sf "${SRC[$tag]}" "$d/$name.safetensors"
  if [ ! -f "$d/${name}_q4_0.gguf" ]; then
    "$MT" export "$d/spec.json" --from "${SRC[$tag]}" --formats gguf,gguf_q8_0_all,gguf_q4_k_m,gguf_q4_k_s,gguf_q4_0 > "$d/export.log" 2>&1
  fi
  ( cd "$d" && sha256sum "$name".gguf "$name"_*.gguf && ls -l "$name".gguf "$name"_*.gguf ) > "$d/exports.txt"
  for v in f32 q8_0_all q4_k_m q4_k_s q4_0; do
    g="$d/${name}_$v.gguf"; [ "$v" = f32 ] && g="$d/$name.gguf"
    python3 "$X/greedy.py" run "$tag" "$v" "$g" "$W/greedy"
  done
  python3 "$X/greedy.py" compare "$W/greedy" "$tag" "$d/$name.gguf" > "$W/greedy/compare_$tag.json"
  echo "done $tag"
done
echo PTQ_HOME_DONE
