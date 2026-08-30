#!/usr/bin/env python3
"""B2.1a T4 validation runner — paste into ONE Colab cell (T4 runtime), or
fan out via the colab-sweep skill (it is argv-compatible: --shard I N is
accepted and ignored; there is exactly one shard of work).

Clones both repos at the required layout, runs tools/colab_cuda_validate.sh
(which now carries the B2.1a gate: test_cuda_ops + gradcheck/nn rerun with
MICROTORCH_DEVICE_OPS=1), and zips the logs.

Colab cell:
    !wget -q https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/master/tools/colab_b21_validate.py
    !python colab_b21_validate.py

PREREQUISITE: the working tree (branch below) must be pushed first —
pushes need Jonathan's terminal (SSH key), never the assistant's shell.
"""
import argparse
import subprocess
import sys

BRANCH = "master"  # update if validating from a feature branch
PAPERKILN = "https://github.com/JohnnyTeutonic/paperkiln.git"
COALFIRE = "https://github.com/JohnnyTeutonic/coalfire.cpp.git"
DONE_MARKER = "B21_VALIDATION_DONE"


def run(cmd, **kw):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=[0, 1])
    ap.parse_args()

    # Idempotent on a reused VM (and on any stale /content): a leftover
    # checkout must never masquerade as the commit under test.
    run("rm -rf microtorch transformer_cpp")
    rc = run(f"git clone --depth 1 -b {BRANCH} {PAPERKILN} microtorch")
    rc |= run(f"git clone --depth 1 {COALFIRE} transformer_cpp")
    if rc:
        sys.exit("clone failed — was the branch pushed?")
    # Print the EXACT commits under test — the aggregation step must
    # refuse a log that lacks these lines (the 2026-08-30 lesson: a
    # stale results zip validated yesterday's binary and read as green).
    # Captured via Python, not a shell echo: the echoed stdout of a
    # subprocess failed to land in either log on the first rerun.
    def _sha(repo):
        return subprocess.run(f"git -C {repo} rev-parse HEAD", shell=True,
                              capture_output=True, text=True).stdout.strip()
    commit_lines = (f"VALIDATING_PAPERKILN_COMMIT: {_sha('microtorch')}\n"
                    f"VALIDATING_COALFIRE_COMMIT: {_sha('transformer_cpp')}\n")
    print(commit_lines, end="", flush=True)

    with open("validate.log", "w") as log:
        log.write(commit_lines)  # the freshness gate reads validate.log
        p = subprocess.Popen(
            "bash microtorch/tools/colab_cuda_validate.sh",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p.wait()

    run("zip -q b21_results.zip validate.log")
    verdict = "PASSED" if p.returncode == 0 else f"FAILED rc={p.returncode}"
    print(f"{DONE_MARKER}: {verdict}")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
