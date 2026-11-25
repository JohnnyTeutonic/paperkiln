#!/usr/bin/env python3
"""The W169 rehearsal's watcher (the author's research tree, AI_ML/chat_screens_claude/base2/w169/REGISTRATION.md
section 7). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/w169_watch.py --chat /content/chat --job pt_w169_reh --out OUT [--interval 30] [--limit-s 14400]

Every `interval` seconds, one line to OUT/watch.jsonl: base 2's drill sample (base2_corpus.watch_sample: the trainer's
training step, its saved step, the courier's pushed step) and the machine's memory: host RAM used and available (from
/proc/meminfo), the largest resident set among processes, GPU memory used (nvidia-smi), and the free space under /content.
Writes on the VM only; it never opens the store. Ends when the job's STATUS file reads FINISHED or FAILED, or after
`limit_s`, and writes OUT/watch_summary.json with the peaks.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base2_corpus as B  # noqa: E402  (base 2's drill sample, unchanged)


def meminfo():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                out[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return out


def max_rss_mb():
    best = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    best = max(best, int(line.split()[1]) // 1024)
                    break
        except OSError:
            continue
    return best


def gpu_used_mb():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(p.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def disk_free_gb(path="/content"):
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--limit-s", type=float, default=14400)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0, n, peak = time.time(), 0, {"host_used_mb": 0, "max_rss_mb": 0, "gpu_used_mb": 0, "disk_free_min_gb": None}
    rec = {}
    while True:
        try:
            status = open(os.path.join(a.chat, f"STATUS_{a.job}")).read().strip()
        except OSError:
            status = "NONE"
        m = meminfo()
        used = m.get("MemTotal", 0) - m.get("MemAvailable", 0)
        rss, gpu, disk = max_rss_mb(), gpu_used_mb(), disk_free_gb()
        rec = {**B.watch_sample(a.chat, a.job), "status": status, "sample": n, "host_total_mb": m.get("MemTotal"),
               "host_used_mb": used, "max_rss_mb": rss, "gpu_used_mb": gpu, "disk_free_gb": disk}
        peak["host_used_mb"] = max(peak["host_used_mb"], used)
        peak["max_rss_mb"] = max(peak["max_rss_mb"], rss)
        peak["gpu_used_mb"] = max(peak["gpu_used_mb"], gpu or 0)
        if disk is not None:
            peak["disk_free_min_gb"] = disk if peak["disk_free_min_gb"] is None else min(peak["disk_free_min_gb"], disk)
        with open(os.path.join(a.out, "watch.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if status in ("FINISHED", "FAILED") or time.time() - t0 > a.limit_s:
            break
        time.sleep(a.interval)
    with open(os.path.join(a.out, "watch_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"job": a.job, "samples": n, "peak": peak, "host_total_mb": rec.get("host_total_mb"), "last": rec}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
