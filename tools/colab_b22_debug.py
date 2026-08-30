#!/usr/bin/env python3
"""B2.2 leg-4 heap-corruption debug runner (T4). One purpose: get the
stack trace of the host-side bad write that SIGABRTs test_cuda_ops at
leg 4 (2026-08-30 validate run, rc=134, "malloc(): unsorted double
linked list corrupted").

Builds ONLY test_cuda_ops and runs it twice:
  1. under valgrind (definitive invalid read/write stacks; CUDA emits
     harmless noise — we want the FIRST "Invalid write"/"Invalid read"
     block against a host address);
  2. natively with MALLOC_CHECK_=3 MALLOC_PERTURB_=85 as a cross-check.

argv-compatible with the colab-sweep skill (--shard I N accepted and
ignored). Done marker: B22_DEBUG_DONE.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=[0, 1])
    ap.parse_args()

    run("rm -rf microtorch transformer_cpp")
    rc = run(f"git clone --depth 1 -b {BRANCH} {PAPERKILN} microtorch")
    rc |= run(f"git clone --depth 1 {COALFIRE} transformer_cpp")
    if rc:
        sys.exit("clone failed")
    # Captured in PYTHON so the line reaches the log even when shell
    # stdout interleaving eats an echo (the 2026-08-30 freshness-gate
    # lesson).
    print(f"VALIDATING_PAPERKILN_COMMIT: "
          f"{sh_out('git -C microtorch rev-parse HEAD')}", flush=True)
    print(f"VALIDATING_COALFIRE_COMMIT: "
          f"{sh_out('git -C transformer_cpp rev-parse HEAD')}", flush=True)

    run("apt-get -qq install -y valgrind > /dev/null 2>&1 || "
        "apt-get install -y valgrind")

    rc = run("mkdir -p microtorch/build_cuda && cd microtorch/build_cuda && "
             "cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMICROTORCH_CUDA=ON "
             "> cmake.log 2>&1 && make microtorch test_cuda_ops -j$(nproc) "
             "> build.log 2>&1")
    if rc:
        run("tail -40 microtorch/build_cuda/build.log")
        print(f"{DONE}: BUILD_FAILED")
        sys.exit(1)

    with open("valgrind.log", "w") as log:
        p = subprocess.Popen(
            "cd microtorch/build_cuda && MICROTORCH_DEVICE=cuda "
            "MICROTORCH_DEVICE_OPS=1 valgrind --error-limit=no "
            "--num-callers=25 ./test_cuda_ops",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p.wait()
    print(f"valgrind exit: {p.returncode}", flush=True)

    with open("mcheck.log", "w") as log:
        p2 = subprocess.Popen(
            "cd microtorch/build_cuda && MICROTORCH_DEVICE=cuda "
            "MICROTORCH_DEVICE_OPS=1 MALLOC_CHECK_=3 MALLOC_PERTURB_=85 "
            "./test_cuda_ops",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        for line in p2.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p2.wait()
    print(f"mcheck exit: {p2.returncode}", flush=True)

    run("zip -q b22_debug.zip valgrind.log mcheck.log")
    print(f"{DONE}: COMPLETE (valgrind={p.returncode}, "
          f"mcheck={p2.returncode})")


if __name__ == "__main__":
    main()
