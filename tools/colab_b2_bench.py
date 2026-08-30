#!/usr/bin/env python3
"""B2 adoption-gate wall-clock benchmark on T4 (docs/CUDA_PHASE_B2.md,
B2.3 item 5). Builds bench_b2 and times CPU-AVX vs the full B2 stack at
d=256 and d=512 (T=512, L=4 — the Rung C shape). The verdict line is
mechanical: B2 is adopted for Rung C iff it wins at d=512.

argv-compatible with the colab-sweep skill. Done marker: B2_BENCH_DONE.
"""
import argparse
import re
import subprocess
import sys

BRANCH = "master"
PAPERKILN = "https://github.com/JohnnyTeutonic/paperkiln.git"
COALFIRE = "https://github.com/JohnnyTeutonic/coalfire.cpp.git"
DONE = "B2_BENCH_DONE"


def run(cmd, **kw):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw).returncode


def sh_out(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=[0, 1])
    ap.parse_args()

    run("rm -rf microtorch transformer_cpp")
    rc = run(f"git clone --depth 1 -b {BRANCH} {PAPERKILN} microtorch")
    rc |= run(f"git clone --depth 1 {COALFIRE} transformer_cpp")
    if rc:
        sys.exit("clone failed")
    print(f"VALIDATING_PAPERKILN_COMMIT: "
          f"{sh_out('git -C microtorch rev-parse HEAD').strip()}", flush=True)

    rc = run("mkdir -p microtorch/build_cuda && cd microtorch/build_cuda && "
             "cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON "
             "> cmake.log 2>&1 && make microtorch bench_b2 -j$(nproc) "
             "> build.log 2>&1")
    if rc:
        run("tail -40 microtorch/build_cuda/build.log")
        print(f"{DONE}: BUILD_FAILED")
        sys.exit(1)

    results = {}
    for d in (256, 512):
        for eng in ("cpu", "b2"):
            steps = 10
            env = ("MICROTORCH_DEVICE=cuda " if eng == "b2" else
                   "MICROTORCH_DEVICE=cpu ")
            out = sh_out(f"cd microtorch/build_cuda && {env}"
                         f"./bench_b2 {d} 512 4 {steps} {eng}")
            print(out, end="", flush=True)
            m = re.search(r"ms_per_step=([0-9.]+)", out)
            if m:
                results[(d, eng)] = float(m.group(1))

    print("\n== ADOPTION GATE ==", flush=True)
    for d in (256, 512):
        c, b = results.get((d, "cpu")), results.get((d, "b2"))
        if c and b:
            print(f"d={d}: cpu {c:.1f} ms/step vs b2 {b:.1f} ms/step -> "
                  f"{'B2 WINS' if b < c else 'CPU WINS'} "
                  f"({c / b:.2f}x)", flush=True)
    c512, b512 = results.get((512, "cpu")), results.get((512, "b2"))
    if c512 and b512:
        verdict = "ADOPT_B2_FOR_RUNG_C" if b512 < c512 else "RUNG_C_STAYS_CPU"
    else:
        verdict = "INCOMPLETE"
    print(f"{DONE}: {verdict}")


if __name__ == "__main__":
    main()
