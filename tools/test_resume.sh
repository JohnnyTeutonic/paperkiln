#!/usr/bin/env bash
# Checkpoint/resume acceptance test (WSL, CPU build in ~/mtrel, outputs on
# ext4 under /tmp).
#
# The claim under test: a run that is stopped at step N and resumed from
# its checkpoint follows the SAME trajectory as an uninterrupted run,
# to the bit, on the CPU path. Same losses at every later step, and a
# byte-identical final weights file (save_safetensors fixes tensor order,
# so identical dicts give identical bytes).
#
# Without optimizer state in the checkpoint this test fails at step N+1:
# AdamW restarts with zero moments and t=0, and the update is different.
# That is the gap that blocked transfer_s1 arms M and L.
set -euo pipefail
REPO=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
MT=$HOME/mtrel/mtstudio
CORPUS=$REPO/transformer_cpp/data/tinystories-txt/train-0-small.txt
VOCAB=$REPO/transformer_cpp/releases/chat7b.gguf
W=/tmp/resume_test
rm -rf "$W"; mkdir -p "$W"

echo "== build =="
cmake --build "$HOME/mtrel" --target mtstudio -j8 2>&1 | tail -3

spec() {  # $1 out_dir  $2 steps  $3 optimizer
  cat <<EOF
{
  "name": "resume_test",
  "arch": {"preset": "gpt2-nano", "custom": {"d": 64, "heads": 4, "attention": "swa", "window": 16, "sinks": 1}},
  "data": {"corpus": "$CORPUS", "vocab": "$VOCAB", "vocab_cap": 1024, "T": 32},
  "train": {"batch": 2, "lr": 0.001, "steps": $2, "eval_every": 20, "checkpoint_every": 20,
            "gradmap_every": 0, "seed": 3, "optimizer": "$3"},
  "export": {"formats": ["safetensors"]},
  "out_dir": "$1"
}
EOF
}

fail=0
for OPT in adamw muon; do
  echo "== optimizer=$OPT =="
  A="$W/A_$OPT"; B="$W/B_$OPT"; mkdir -p "$A" "$B"
  spec "$A" 40 "$OPT" > "$W/A_$OPT.json"
  spec "$B" 20 "$OPT" > "$W/B1_$OPT.json"
  spec "$B" 40 "$OPT" > "$W/B2_$OPT.json"
  "$MT" run "$W/A_$OPT.json"  > "$W/A_$OPT.log"  2>&1
  "$MT" run "$W/B1_$OPT.json" > "$W/B1_$OPT.log" 2>&1
  test -f "$B/optim.safetensors" || { echo "FAIL: no optim.safetensors written"; fail=1; }
  "$MT" run "$W/B2_$OPT.json" > "$W/B2_$OPT.log" 2>&1
  grep -q '"event":"resume"' "$B/events.jsonl" || { echo "FAIL: no resume event"; fail=1; }
  grep '"event":"resume"' "$B/events.jsonl" | tail -1
  python3 - "$A/events.jsonl" "$B/events.jsonl" <<'PY' || fail=1
import json, sys
def losses(p):
    out = {}
    for line in open(p):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") == "step": out[e["step"]] = e["loss"]
    return out
a, b = losses(sys.argv[1]), losses(sys.argv[2])
steps = [s for s in range(21, 41)]
bad = [(s, a.get(s), b.get(s)) for s in steps if a.get(s) != b.get(s)]
if bad:
    print("FAIL: trajectories diverge after resume; first mismatches:", bad[:3])
    sys.exit(1)
print(f"OK: steps 21..40 losses identical ({len(steps)} steps)")
PY
  if cmp -s "$A/resume_test.safetensors" "$B/resume_test.safetensors"; then
    echo "OK: final weights byte-identical"
  else
    echo "FAIL: final weights differ"; fail=1
  fi
done
echo "== negative control: resume with optimizer state removed must diverge =="
C="$W/C_adamw"; mkdir -p "$C"
spec "$C" 20 adamw > "$W/C1.json"; spec "$C" 40 adamw > "$W/C2.json"
"$MT" run "$W/C1.json" > "$W/C1.log" 2>&1
rm -f "$C/optim.safetensors"
"$MT" run "$W/C2.json" > "$W/C2.log" 2>&1
grep '"event":"resume"' "$C/events.jsonl" | tail -1
if cmp -s "$W/A_adamw/resume_test.safetensors" "$C/resume_test.safetensors"; then
  echo "FAIL: cold-optimizer resume matched the uninterrupted run (test has no power)"; fail=1
else
  echo "OK: cold-optimizer resume diverges, as it must"
fi
[ $fail -eq 0 ] && echo "ALL PASS" || { echo "FAILURES"; exit 1; }
