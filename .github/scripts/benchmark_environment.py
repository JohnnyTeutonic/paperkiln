"""Partition benchmark histories by measured execution environment, not source revision."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

SCHEMA = "paperkiln.benchmark-environment.v1"
THREAD_KEYS = ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS", "GOMP_CPU_AFFINITY", "OMP_WAIT_POLICY")


def command(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def collect(build):
    build = Path(build)
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    compiler = cache["CMAKE_CXX_COMPILER"]
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    cpu = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            cpu.setdefault(key.strip(), value.strip())
    flags = {}
    for target in ("benchmark_attention", "benchmark_mamba"):
        lines = (build / "CMakeFiles" / (target + ".dir") / "flags.make").read_text().splitlines()
        flags[target] = {key: next(s.split("=", 1)[1].strip() for s in lines if s.startswith(key + " ="))
                         for key in ("CXX_DEFINES", "CXX_FLAGS")}
    return {"os": {"id": release["ID"], "version": release["VERSION_ID"]},
            "compiler": {"version": command([compiler, "--version"]), "target": command([compiler, "-dumpmachine"])},
            "cpu": {"architecture": platform.machine(), "model": cpu["model name"],
                    "flags": sorted(cpu["flags"].split()), "logical_cores": os.cpu_count(),
                    "usable_cores": len(os.sched_getaffinity(0))},
            "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
            "build_type": cache["CMAKE_BUILD_TYPE"], "effective_flags": flags,
            "libc": list(platform.libc_ver())}


def report(identity):
    if not identity or not identity["compiler"]["version"] or not identity["cpu"]["model"]:
        raise ValueError("incomplete benchmark environment")
    encoded = json.dumps({"schema": SCHEMA, "identity": identity}, sort_keys=True, separators=(",", ":"))
    return {"schema": SCHEMA, "identity": identity, "cohort": hashlib.sha256(encoded.encode()).hexdigest()}


def validate_cache(folder, current):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "environment.json"
    if metadata.exists():
        if json.loads(metadata.read_text()) != current:
            raise ValueError("cached benchmark execution environment differs")
    elif (folder / "benchmark-data.json").exists():
        raise ValueError("cached benchmark history has no environment evidence")
    else:
        with metadata.open("x") as f:
            json.dump(current, f, indent=2)
            f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, default=Path("build"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--github-output", type=Path)
    p.add_argument("--validate-cache", type=Path)
    args = p.parse_args()
    if args.validate_cache:
        current = json.loads(args.out.read_text())
        if current != report(current["identity"]):
            raise ValueError("environment receipt fingerprint differs")
        validate_cache(args.validate_cache, current)
        print("Matched baseline environment verified; missing history starts a new cohort.")
        return
    current = report(collect(args.build))
    with args.out.open("x") as f:
        json.dump(current, f, indent=2)
        f.write("\n")
    if args.github_output:
        with args.github_output.open("a") as f:
            f.write("cohort=" + current["cohort"] + "\n")
    print(json.dumps(current, sort_keys=True))


if __name__ == "__main__":
    main()
