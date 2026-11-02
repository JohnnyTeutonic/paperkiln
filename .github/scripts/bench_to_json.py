#!/usr/bin/env python3
"""Convert paperkiln's benchmark text output to github-action-benchmark JSON.

Usage: bench_to_json.py attention_benchmark.txt mamba_benchmark.txt > benchmark.json
Emits the customSmallerIsBetter format: a list of {"name", "unit", "value"}.
Fails (exit 1) if a file yields no measurements, so a format change cannot pass silently.
"""
import json
import re
import sys


def attention(text):
    out = []
    config = None
    for line in text.splitlines():
        m = re.match(r"\s*Config \[\s*(\d+),\s*(\d+),\s*(\d+)\]:", line)
        if m:
            config = "seq%s d%s h%s" % m.groups()
            continue
        m = re.match(r"\s*(Standard Attention|Kimi Linear):.*Total\s+([0-9.]+)ms", line)
        if m and config:
            kind = "standard" if m.group(1).startswith("Standard") else "kimi"
            out.append({"name": "attention %s %s total" % (kind, config),
                        "unit": "ms", "value": float(m.group(2))})
    return out


def mamba(text):
    out = []
    seq = None
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*Fwd:", line)
        if m:
            seq = m.group(1)
            continue
        m = re.match(r"\s*\|\s*Total:\s*(-?[0-9.]+) ms\s*\|\s*Total:\s*(-?[0-9.]+) ms\s*\|\s*Total:\s*(-?[0-9.]+) ms", line)
        if m and seq:
            for kind, value in zip(("mamba", "kimi", "standard"), m.groups()):
                out.append({"name": "mamba-table %s seq%s total" % (kind, seq),
                            "unit": "ms", "value": float(value)})
    return out


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    results = []
    for path, parse in ((argv[1], attention), (argv[2], mamba)):
        with open(path, encoding="utf-8", errors="replace") as f:
            found = parse(f.read())
        if not found:
            sys.exit("no measurements parsed from " + path)
        negative = [r["name"] for r in found if r["value"] < 0]
        if negative:
            sys.exit("negative timings in %s: %s" % (path, ", ".join(negative)))
        results += found
    json.dump(results, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv)
