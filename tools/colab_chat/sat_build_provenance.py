#!/usr/bin/env python3
"""CPU builds of the trainer with provenance, for SAT's tests and launch check (GPT's delta review 2, point 13).
Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/sat_build_provenance.py build --src CHECKOUT --build-dir DIR [--jobs 2]
    python3 tools/colab_chat/sat_build_provenance.py verify --exe DIR/mtstudio --checkout REPO --commit SHA [--clean]

build configures and builds mtstudio with CUDA off (CUDA_VISIBLE_DEVICES empty), from a checkout whose trainer sources
(src, include, third_party, tools/mtstudio.cpp, CMakeLists.txt) have no uncommitted change, then writes
DIR/mtstudio.provenance.json binding the executable's sha256 to the checkout's commit, the git object ids of those
trainer sources at that commit, and the build configuration (CMakeCache: build type, CUDA switches, compiler, flags,
source directory) and compiler version. Run it through scripts/cpu_budget.py.

verify checks the executable actually given: its sha256 equals the provenance's, the trainer sources' object ids at
--commit (in --checkout) equal the ones it was built from, the build is Release with CUDA off, and, with --clean,
that --checkout has no uncommitted change to the trainer sources (so --commit HEAD means the files on disk). Prints a
JSON verdict; exit 0 only when every check holds.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

TRAINER_PATHS = ("src", "include", "third_party", "tools/mtstudio.cpp", "CMakeLists.txt")
CACHE_KEYS = ("CMAKE_BUILD_TYPE", "MICROTORCH_CUDA", "CUDA_ENABLED", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
              "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_HOME_DIRECTORY")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def trainer_objects(repo, commit):
    return {p: git(repo, "rev-parse", f"{commit}:{p}") for p in TRAINER_PATHS}


def dirty(repo):
    return git(repo, "status", "--porcelain", "--untracked-files=normal", "--", *TRAINER_PATHS)


def cache(build_dir):
    out = {}
    with open(os.path.join(build_dir, "CMakeCache.txt"), encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("//", "#")) or "=" not in line or ":" not in line.split("=", 1)[0]:
                continue
            k, v = line.rstrip("\n").split("=", 1)
            name = k.split(":", 1)[0]
            if name in CACHE_KEYS:
                out[name] = v
    return out


def prov_path(exe):
    return os.path.join(os.path.dirname(os.path.abspath(exe)), "mtstudio.provenance.json")


def build(a):
    src, bd = os.path.realpath(a.src), os.path.realpath(a.build_dir)
    d = dirty(src)
    if d:
        sys.exit(f"the trainer sources in {src} have uncommitted changes; commit them first:\n{d}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    os.makedirs(bd, exist_ok=True)
    exe = os.path.join(bd, "mtstudio")
    if os.path.exists(exe):
        os.remove(exe)  # never leave a stale executable that a failed build would not replace
    subprocess.run(["cmake", "-S", src, "-B", bd, "-DCMAKE_BUILD_TYPE=Release", "-DMICROTORCH_CUDA=OFF"], env=env,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", bd, "--target", "mtstudio", f"-j{a.jobs}"], env=env, check=True)
    commit = git(src, "rev-parse", "HEAD")
    if dirty(src) or git(src, "rev-parse", "HEAD") != commit:
        sys.exit("the trainer sources changed during the build: no provenance written")
    cfg = cache(bd)
    if cfg.get("CMAKE_HOME_DIRECTORY") != src:
        sys.exit(f"the build directory was configured from {cfg.get('CMAKE_HOME_DIRECTORY')}, not {src}")
    cxx = subprocess.run([cfg.get("CMAKE_CXX_COMPILER", "c++"), "--version"], capture_output=True, text=True).stdout
    rec = {"format": "sat.build_provenance/1", "executable": exe, "executable_sha256": sha256(exe),
           "source_dir": src, "source_commit": commit, "trainer_objects": trainer_objects(src, commit),
           "mtstudio_cpp_sha256": sha256(os.path.join(src, "tools", "mtstudio.cpp")), "cmake_cache": cfg,
           "compiler_version": cxx.splitlines()[0] if cxx else "", "cuda_visible_devices": "",
           "built_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    tmp = prov_path(exe) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, prov_path(exe))
    print(json.dumps({"built": exe, "source_commit": commit, "executable_sha256": rec["executable_sha256"]}))
    return 0


def check(exe, checkout, commit, clean=False):
    """[] when the executable is a provenance-recorded CPU Release build of commit's trainer sources, else problems."""
    p = prov_path(exe)
    if not os.path.isfile(exe):
        return [f"no executable at {exe}"]
    if not os.path.isfile(p):
        return [f"no provenance beside {exe} (build it with sat_build_provenance.py build)"]
    rec = json.load(open(p, encoding="utf-8"))
    problems = []
    if rec.get("format") != "sat.build_provenance/1":
        problems.append("unknown provenance format")
    if sha256(exe) != rec.get("executable_sha256"):
        problems.append("the executable is not the one the provenance records (sha256 differs)")
    try:
        want = trainer_objects(checkout, commit)
    except subprocess.CalledProcessError:
        return problems + [f"{commit} is not a commit of {checkout}"]
    if rec.get("trainer_objects") != want:
        diff = sorted(k for k in want if rec.get("trainer_objects", {}).get(k) != want[k])
        problems.append(f"built from other trainer sources than {commit}'s: {diff}")
    cfg = rec.get("cmake_cache", {})
    if cfg.get("CMAKE_BUILD_TYPE") != "Release" or cfg.get("MICROTORCH_CUDA") != "OFF" or cfg.get("CUDA_ENABLED") not in (None, "OFF"):
        problems.append(f"not a Release CPU build: {cfg}")
    if clean and dirty(checkout):
        problems.append(f"{checkout} has uncommitted changes to the trainer sources")
    return problems


def verify(a):
    problems = check(a.exe, a.checkout, a.commit, a.clean)
    print(json.dumps({"exe": a.exe, "commit": a.commit, "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--build-dir", required=True)
    b.add_argument("--jobs", type=int, default=2)
    v = sub.add_parser("verify")
    v.add_argument("--exe", required=True)
    v.add_argument("--checkout", required=True)
    v.add_argument("--commit", required=True)
    v.add_argument("--clean", action="store_true")
    a = ap.parse_args(argv)
    return build(a) if a.cmd == "build" else verify(a)


if __name__ == "__main__":
    sys.exit(main())
