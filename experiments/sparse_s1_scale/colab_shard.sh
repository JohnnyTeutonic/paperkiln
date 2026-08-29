#!/usr/bin/env bash
# VM-side sharded runner for sparse_s1_scale Rung B (see PREREGISTRATION.md).
# Contract (colab-sweep): accepts --shard I N, resumable, prints a done
# marker, zips results continuously.
#
# Layout on the VM:
#   /content/src/            microtorch + transformer_cpp sources (unzipped)
#   /content/chat7b.gguf     vocab source
#   /content/corpus.txt      TinyStories slice
#   /content/out/            run outputs (mtsweep out_root)
#   /content/results.zip     relayed snapshot (atomic temp+mv)
#   /content/SPARSE_S1_SCALE_DONE   completion marker for this shard
set -uo pipefail
cd /content

SHARD=0; NSHARDS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --shard) SHARD="$2"; NSHARDS="$3"; shift 3 ;;
    *) shift ;;
  esac
done
echo "=== sparse_s1_scale shard ${SHARD}/${NSHARDS} boot $(date) ==="
[ -f SPARSE_S1_SCALE_DONE ] && { echo "already done"; exit 0; }

[ -d /content/src ] || unzip -oq src.zip -d /content/src

# Build (CPU; cached across boot retries on the same VM).
if [ ! -x /content/build/mtstudio ]; then
    mkdir -p /content/build && cd /content/build
    cmake /content/src/microtorch -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
    make mtstudio -j"$(nproc)" 2>&1 | tail -2
    cd /content
fi
[ -x /content/build/mtstudio ] || { echo "BUILD FAILED"; exit 1; }

# Generate one single-run spec per assigned (lane, seed), round-robin
# by global run index. Deterministic: index = lane_i * 5 + (seed - 1).
python3 - "$SHARD" "$NSHARDS" <<'EOF'
import json, sys
shard, n = int(sys.argv[1]), int(sys.argv[2])
spec = json.load(open("/content/src/microtorch/experiments/sparse_s1_scale/sweep_rungB.json"))
lanes = spec["factors"]["arch.custom.attention"]
seeds = spec["seeds"]
runs = [(li, s) for li, _ in enumerate(lanes) for s in seeds]
mine = [(li, s) for i, (li, s) in enumerate(runs) if i % n == shard]
for li, s in mine:
    one = json.loads(json.dumps(spec))
    one["factors"]["arch.custom.attention"] = [lanes[li]]
    one["seeds"] = [s]
    one["out_root"] = f"/content/out/{lanes[li]}_s{s}"
    path = f"/content/spec_{lanes[li]}_s{s}.json"
    json.dump(one, open(path, "w"), indent=1)
    print(path)
EOF
python3 - "$SHARD" "$NSHARDS" > /content/myspecs.txt <<'EOF'
import json, sys
shard, n = int(sys.argv[1]), int(sys.argv[2])
spec = json.load(open("/content/src/microtorch/experiments/sparse_s1_scale/sweep_rungB.json"))
lanes = spec["factors"]["arch.custom.attention"]
runs = [(li, s) for li, _ in enumerate(lanes) for s in spec["seeds"]]
for i, (li, s) in enumerate(runs):
    if i % n == shard:
        print(f"/content/spec_{lanes[li]}_s{s}.json")
EOF

relay() {  # atomic results snapshot
    ( cd /content && zip -qr results_new.zip out ) && \
    mv -f /content/results_new.zip /content/results.zip
}

while read -r specfile; do
    name=$(basename "$specfile" .json)
    out_root=$(python3 -c "import json,sys;print(json.load(open('$specfile'))['out_root'])")
    if ls "$out_root"/*/result.json >/dev/null 2>&1; then
        echo "skip finished: $name"; continue
    fi
    echo "--- running $name $(date +%H:%M:%S) ---"
    python3 /content/src/microtorch/tools/mtsweep.py "$specfile" \
        --jobs 1 --omp "$(nproc)" \
        --mtstudio /content/build/mtstudio \
        || { echo "RUN FAILED: $name"; relay; exit 1; }
    relay
    echo "--- done $name $(date +%H:%M:%S) ---"
done < /content/myspecs.txt

relay
touch /content/SPARSE_S1_SCALE_DONE
echo "=== SHARD ${SHARD}/${NSHARDS} COMPLETE $(date) ==="
