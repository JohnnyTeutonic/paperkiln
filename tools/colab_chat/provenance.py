#!/usr/bin/env python3
"""Where a run ran (experiments/tiny_chat_2026-10/PREREGISTRATION.md, implementation note of 24 Oct 2026):
written at boot by boot_jobs.sh on every VM, Colab or RunPod, and copied by tiny_chat_run.py and
jobs_vm.py into every summary and result they write (the `provenance` key).

    python3 tools/colab_chat/provenance.py OUT.json

Provider, GPU model, driver and CUDA versions (nvidia-smi; nvcc's release), image and its digest
(RunPod: from /content/chat/provider.json, which runner_multi's RunPod backend writes before the
boot; Colab exposes no image digest, so its runtime release tag stands in), host CPU and memory,
and the paperkiln commit. Every field is best-effort: a missing tool leaves its field None.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time

CHAT = os.environ.get("JOBS_CHAT_DIR", "/content/chat")


def run(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpus():
    out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,pci.bus_id",
               "--format=csv,noheader"])
    rows = []
    for ln in (out or "").strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 5:
            rows.append({"name": f[0], "driver": f[1], "memory": f[2], "power_limit": f[3], "pci": f[4]})
    return rows


def collect():
    prov = {}
    pj = os.path.join(CHAT, "provider.json")
    if os.path.exists(pj):
        try:
            prov = json.load(open(pj))
        except ValueError:
            prov = {"provider_json": "unreadable"}
    else:
        try:
            import google.colab  # noqa: F401  (Colab's runtime only)
            colab = True
        except ImportError:
            colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
        prov = {"provider": "colab" if colab else "unknown", "image": "colab runtime" if colab else None,
                "image_digest": None, "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG")}
    g = gpus()
    smi = run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
    nvcc = run(["nvcc", "--version"]) or ""
    n = re.search(r"release ([0-9.]+)", nvcc)
    cpu = None
    try:
        cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), None)
    except OSError:
        pass
    mem = None
    try:
        kb = next(int(ln.split()[1]) for ln in open("/proc/meminfo") if ln.startswith("MemTotal"))
        mem = round(kb / 1024 / 1024, 1)
    except (OSError, StopIteration, ValueError):
        pass
    osr = None
    try:
        osr = next((ln.split("=", 1)[1].strip().strip('"') for ln in open("/etc/os-release")
                    if ln.startswith("PRETTY_NAME")), None)
    except OSError:
        pass
    head = run(["git", "-C", os.environ.get("JOBS_REPO", "/content/pk"), "rev-parse", "HEAD"])
    return {**prov, "gpu": g[0]["name"] if g else None, "gpus": g, "driver_version": g[0]["driver"] if g else None,
            "cuda_driver_api": m.group(1) if m else None, "nvcc": n.group(1) if n else None,
            "cpu": cpu, "cpus": os.cpu_count(), "mem_gb": mem, "os": osr, "kernel": platform.release(),
            "python": platform.python_version(), "paperkiln_commit": head.strip() if head else None,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def read(chat=None):
    """The VM's provenance.json, or None (a VM booted before provenance existed)."""
    p = os.path.join(chat or CHAT, "provenance.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CHAT, "provenance.json")
    d = collect()
    with open(out + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(out + ".tmp", out)
    print(json.dumps({k: d.get(k) for k in ("provider", "gpu", "driver_version", "cuda_driver_api", "nvcc",
                                            "image", "image_digest")}))
