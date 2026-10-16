#!/usr/bin/env python3
"""posttrain_calib measurement (PREREGISTRATION.md, "Measurement").

    python experiments/posttrain_calib/measure.py OUT_ROOT \\
        --mtstudio ~/mtrel/mtstudio --verify ~/mtrel/posttrain_verify

For every run directory under OUT_ROOT/runs it samples the level's frozen
dev set once (8 per prompt, temperature 1.0, seed 17, at most 8 new
tokens). It scores those same samples with exact_match, the default
synthetic_judge and format, and writes receipts/measurements.json beside
this file plus a per-run scored file. The level is read from the run's own
spec.json (data.corpus), never from the directory name.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N, TEMP, SEED, MAX_NEW = 8, 1.0, 17, 8


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(binary, kind, samples_path, refs=None):
    args = [binary, kind] + (["--refs", refs] if refs else [])
    with open(samples_path, "rb") as fin:
        out = subprocess.run(args, stdin=fin, capture_output=True, check=True).stdout.decode("utf-8")
    return [json.loads(x) for x in out.splitlines() if x.strip()]


def measure_run(run_dir, mtstudio, verifier, work):
    spec_path = os.path.join(run_dir, "spec.json")
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    m = re.search(r"tinyarith_r(\d+)_v1", spec["data"]["corpus"])
    if not m:
        raise SystemExit(f"{run_dir}: spec does not name a tinyarith corpus")
    R = int(m.group(1))
    dev = os.path.join(ROOT, "corpus", "eval", f"tinyarith_r{R}_dev_v1.jsonl")
    ckpt = os.path.join(spec["out_dir"], spec["name"] + ".safetensors")
    os.makedirs(work, exist_ok=True)
    refs = os.path.join(work, "refs.jsonl")
    with open(dev, encoding="utf-8") as f, open(refs, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if line.strip():
                r = json.loads(line)
                g.write(json.dumps({"prompt": r["prompt"], "answer": r["answer"], "uri": r["uri"]}) + "\n")
    samples = os.path.join(work, "samples.jsonl")
    subprocess.run([mtstudio, "sample-set", spec_path, "--prompts", dev, "--out", samples, "--n", str(N),
                    "--temp", repr(TEMP), "--seed", str(SEED), "--max-new", str(MAX_NEW)],
                   check=True, capture_output=True)
    em = verify(verifier, "exact_match", samples, refs)
    sj = verify(verifier, "synthetic_judge", samples, refs)
    fm = verify(verifier, "format", samples)
    assert [a["completion"] for a in em] == [b["completion"] for b in sj] == [c["completion"] for c in fm]
    n = len(em)
    with open(os.path.join(work, "scored.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for a, b, c in zip(em, sj, fm):
            f.write(json.dumps({"prompt": a["prompt"], "completion": a["completion"], "exact_match": a["reward"],
                                "synthetic_judge": b["reward"], "format": c["reward"],
                                "prompt_uri": a.get("prompt_uri"), "logp": a["logp"]}) + "\n")
    return {"run": os.path.basename(run_dir), "level": R, "seed": spec["train"]["seed"],
            "checkpoint_sha256": sha256_file(ckpt), "dev_sha256": sha256_file(dev), "samples": n,
            "truth_acc": sum(r["reward"] for r in em) / n,
            "judge_reward": sum(r["reward"] for r in sj) / n,
            "format_pass_truth_fail": sum(1 for a, c in zip(em, fm) if c["reward"] == 1 and a["reward"] == 0) / n,
            "stop_eos": sum(r["stop"] == "eos" for r in em) / n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_root")
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--verify", required=True)
    a = ap.parse_args(argv)
    runs = sorted(d for d in glob.glob(os.path.join(a.out_root, "runs", "*")) if os.path.isdir(d))
    if not runs:
        raise SystemExit(f"no runs under {a.out_root}/runs")
    rec = os.path.join(HERE, "receipts")
    os.makedirs(rec, exist_ok=True)
    rows = []
    for d in runs:
        row = measure_run(d, a.mtstudio, a.verify, os.path.join(rec, "scored", os.path.basename(d)))
        rows.append(row)
        print(json.dumps(row))
    with open(os.path.join(rec, "measurements.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"protocol": {"n": N, "temperature": TEMP, "seed": SEED, "max_new": MAX_NEW},
                   "runs": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
