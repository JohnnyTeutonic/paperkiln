#!/usr/bin/env python3
"""B2.x heap-corruption debug runner (T4). Three probes, in order of
information-per-minute:

  1. NORMAL build, `stdbuf -oL -eL` + MALLOC_CHECK_=3 MALLOC_PERTURB_=85:
     line-buffered stdout survives the abort, so the log shows the LAST
     leg/check reached (plain runs lose the buffered tail — the
     2026-08-30 'crashed before any leg output' misread).
  2. ASan build of the host code (CXX flags only; nvcc TUs stay plain):
     the definitive bad-write stack.
  3. (fallback) valgrind with -mno-avx512f so VEX can decode coalfire's
     apply_gelu — only if ASan fails to build/run.

argv-compatible with the colab-sweep skill. Done marker: B22_DEBUG_DONE.
"""
import argparse
import subprocess
import sys

BRANCH = "master"
PAPERKILN = "https://github.com/JohnnyTeutonic/paperkiln.git"
COALFIRE = "https://github.com/JohnnyTeutonic/coalfire.cpp.git"
DONE = "B22_DEBUG_DONE"


def run(cmd, **kw):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw).returncode


def sh_out(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout.strip()


def stream(cmd, logname):
    with open(logname, "w") as log:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p.wait()
    print(f"[{logname}] exit: {p.returncode}", flush=True)
    return p.returncode


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
          f"{sh_out('git -C microtorch rev-parse HEAD')}", flush=True)
    print(f"VALIDATING_COALFIRE_COMMIT: "
          f"{sh_out('git -C transformer_cpp rev-parse HEAD')}", flush=True)

    rc = run("mkdir -p microtorch/build_cuda && cd microtorch/build_cuda && "
             "cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMICROTORCH_CUDA=ON "
             "> cmake.log 2>&1 && make microtorch test_cuda_ops -j$(nproc) "
             "> build.log 2>&1")
    if rc:
        run("tail -40 microtorch/build_cuda/build.log")
        print(f"{DONE}: BUILD_FAILED")
        sys.exit(1)

    # Probe 1: line-buffered + strict heap checks — WHERE does it die?
    stream("cd microtorch/build_cuda && MICROTORCH_DEVICE=cuda "
           "MICROTORCH_DEVICE_OPS=1 MALLOC_CHECK_=3 MALLOC_PERTURB_=85 "
           "stdbuf -oL -eL ./test_cuda_ops", "mcheck.log")
    # Probe 1b: same, fully deferred config.
    stream("cd microtorch/build_cuda && MICROTORCH_DEVICE=cuda "
           "MICROTORCH_DEVICE_OPS=1 MICROTORCH_STEP_RESIDENCY=1 "
           "MICROTORCH_DEFER_DOWNLOADS=1 MALLOC_CHECK_=3 MALLOC_PERTURB_=85 "
           "stdbuf -oL -eL ./test_cuda_ops", "mcheck_defer.log")

    # Probe 2: ASan on the host code — WHAT wrote out of bounds?
    rc = run("mkdir -p microtorch/build_asan && cd microtorch/build_asan && "
             "cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMICROTORCH_CUDA=ON "
             "-DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' "
             "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address "
             "> cmake.log 2>&1 && make microtorch test_cuda_ops -j$(nproc) "
             "> build.log 2>&1")
    if rc == 0:
        stream("cd microtorch/build_asan && MICROTORCH_DEVICE=cuda "
               "MICROTORCH_DEVICE_OPS=1 "
               "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0 "
               "stdbuf -oL -eL ./test_cuda_ops", "asan.log")
    else:
        run("tail -30 microtorch/build_asan/build.log")
        print("ASAN BUILD FAILED — falling back to valgrind", flush=True)
        run("apt-get -qq install -y valgrind > /dev/null 2>&1 || true")
        stream("cd microtorch/build_cuda && MICROTORCH_DEVICE=cuda "
               "MICROTORCH_DEVICE_OPS=1 valgrind --error-limit=no "
               "--num-callers=25 ./test_cuda_ops", "valgrind.log")

    run("zip -q b22_debug.zip mcheck.log mcheck_defer.log asan.log "
        "valgrind.log 2>/dev/null; true")
    print(f"{DONE}: COMPLETE")


if __name__ == "__main__":
    main()
