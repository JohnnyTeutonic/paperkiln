#!/usr/bin/env bash
# Runs ON the Colab VM (PROTOCOL.md gates 2 and 3). Usage: bash vm_run.sh COMMIT
# Writes everything under /content/scan_out and touches /content/scan_out/DONE at the end (also on failure).
set -u
COMMIT=$1
O=/content/scan_out
mkdir -p $O
trap 'touch $O/DONE' EXIT
cd /content
{
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  nvcc --version | tail -1
} > $O/hardware.log 2>&1
git clone -q https://github.com/JohnnyTeutonic/paperkiln.git src > $O/clone.log 2>&1
git -C src checkout -q "$COMMIT" >> $O/clone.log 2>&1 || { echo "checkout failed" > $O/FAILED; exit 1; }
git -C src rev-parse HEAD > $O/commit.txt
find src -name "*.sh" -exec sed -i "s/\r$//" {} +
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON -DMICROTORCH_DEVCHECK=ON \
      -DMICROTORCH_BUILD_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES=89 > $O/configure.log 2>&1 || { echo configure > $O/FAILED; exit 1; }
cmake --build build --target test_selective_scan_cuda --parallel "$(nproc)" > $O/build.log 2>&1 || { echo build > $O/FAILED; exit 1; }
python3 -m pip install -q numpy==1.26.4 > $O/numpy_install.log 2>&1
# gate 2: the frozen 220-case test and its independent replay, ordered (control) then chunked with L = 2 and 3
for L in unset 2 3; do
  if [ $L = unset ]; then env -u MICROTORCH_SCAN_CHUNK MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > $O/strict_$L.log 2>&1
  else MICROTORCH_SCAN_CHUNK=$L MICROTORCH_GPU_PROFILE=1 build/test_selective_scan_cuda --require-cuda > $O/strict_$L.log 2>&1; fi
  echo "strict $L exit $?" >> $O/status.log
  python3 -O src/tools/replay_selective_scan.py $O/strict_$L.log --require-cuda > $O/replay_$L.log 2>&1
  echo "replay $L exit $?" >> $O/status.log
done
# gate 3 and the timings
nvcc -O3 -std=c++17 -arch=sm_89 -I src/src src/tools/scan_bench.cu -o scan_bench > $O/bench_build.log 2>&1 || { echo bench_build > $O/FAILED; exit 1; }
./scan_bench 20 > $O/scan_bench.jsonl 2> $O/scan_bench.err
echo "scan_bench exit $?" >> $O/status.log
sha256sum build/test_selective_scan_cuda scan_bench > $O/binaries.sha256
