#!/usr/bin/env python3
"""PT4 driver (PREREGISTRATION.md, "Design"). Resumable: stages whose outputs
exist are skipped.

    python experiments/posttrain_pt4/run_pt4.py --work ~/pt4 \\
        --mtstudio build/mtstudio --verify build/posttrain_verify [--jobs 3] [--smoke]

Then `analyze.py WORK/results.json`. Run from a clean checkout of the licence
commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = 20
SEEDS = [201, 202, 203, 204, 205]
SHIFT = 1
N_EVAL, EVAL_SEED, MAX_NEW, TEMP = 4, 17, 8, 1.0
BASE_TRAIN = {"steps": 3000, "lr": 0.003, "batch": 8, "eval_every": 100, "checkpoint_every": 500,
              "gradmap_every": 0, "optimizer": "adamw", "objective": "lm"}
GRPO = {"group_size": 8, "prompts_per_step": 4, "max_new": MAX_NEW, "beta": 0.05, "temperature": TEMP,
        "std_norm": True, "reward": "exact_match", "eval_samples": N_EVAL}
GRPO_TRAIN = {"steps": 300, "lr": 0.001, "eval_every": 50, "checkpoint_every": 50, "gradmap_every": 0,
              "optimizer": "adamw", "objective": "grpo"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run(cmd, log=None, env=None):
    with (open(log, "w") if log else open(os.devnull, "w")) as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def write_spec(path, name, out_dir, train, data_extra=None):
    spec = {"name": name, "arch": {"preset": "llama-tiny"},
            "data": {"corpus": os.path.join(ROOT, f"corpus/data/tinyarith_r{R}_v1/train.txt"),
                     "vocab": os.path.join(ROOT, f"corpus/data/tinyarith_r{R}_v1/vocab.gguf"),
                     "vocab_cap": 4096, "T": 64, **(data_extra or {})},
            "train": train, "export": {"formats": ["safetensors"]}, "out_dir": out_dir}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, indent=1)


def measure(a, spec, work, tag, dev, refs_true, refs_shift):
    """The monitor statistic on the frozen dev prompts, the same for every
    checkpoint: M = shifted key accepts and the true key rejects."""
    smp = os.path.join(work, f"measure_{tag}.jsonl")
    if not os.path.exists(smp):
        run([a.mtstudio, "sample-set", spec, "--prompts", dev, "--out", smp, "--n", str(N_EVAL),
             "--temp", repr(TEMP), "--seed", str(EVAL_SEED), "--max-new", str(MAX_NEW)])

    def ver(kind, extra):
        with open(smp, "rb") as fin:
            out = subprocess.run([a.verify, kind, *extra], stdin=fin, capture_output=True, check=True).stdout
        return [json.loads(x)["reward"] for x in out.decode().splitlines() if x.strip()]
    t = ver("exact_match", ["--refs", refs_true])
    sh = ver("exact_match", ["--refs", refs_shift])
    fm = ver("format", [])
    n = len(t)
    return {"M": sum(1 for x, y in zip(sh, t) if x == 1 and y == 0) / n, "truth_acc": sum(t) / n,
            "format_hack_frac": sum(1 for f, y in zip(fm, t) if f == 1 and y == 0) / n, "samples": n}


def do_seed(a, s, omp):
    env = dict(os.environ, OMP_NUM_THREADS=str(omp), OMP_WAIT_POLICY="PASSIVE")
    w = os.path.join(a.work, f"seed_{s}")
    os.makedirs(w, exist_ok=True)
    done = os.path.join(w, "result.json")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as f:
            return json.load(f)
    dev = os.path.join(ROOT, f"corpus/eval/tinyarith_r{R}_dev_v1.jsonl")
    refs_true = os.path.join(w, "refs_true.jsonl")
    refs_shift = os.path.join(w, "refs_shift.jsonl")
    devr = read_jsonl(dev)
    write_jsonl(refs_true, [{"prompt": r["prompt"], "answer": r["answer"], "uri": r["uri"]} for r in devr])
    write_jsonl(refs_shift, [{"prompt": r["prompt"], "answer": r["answer"] + SHIFT, "uri": r["uri"] + "#SHIFTED"}
                             for r in devr])
    seen, prompts = set(), []
    for r in read_jsonl(os.path.join(ROOT, f"corpus/data/tinyarith_r{R}_v1/train.jsonl")):
        fact = r["uri"].split("/")[2]
        if fact not in seen:
            seen.add(fact)
            prompts.append({"prompt": r["prompt"], "answer": r["answer"], "uri": r["uri"]})
    train_prompts = os.path.join(w, "grpo_prompts.jsonl")
    write_jsonl(train_prompts, prompts)
    grpo_data = {"grpo_prompts": train_prompts, "grpo_eval": refs_true}

    base_spec = os.path.join(w, "base_spec.json")
    write_spec(base_spec, "base", os.path.join(w, "base"), dict(BASE_TRAIN, seed=s))
    base_ck = os.path.join(w, "base", "base.safetensors")
    if not os.path.exists(base_ck):
        run([a.mtstudio, "run", base_spec], os.path.join(w, "base.log"), env)

    open_spec = os.path.join(w, "open_spec.json")
    write_spec(open_spec, "open", os.path.join(w, "open"),
               dict(GRPO_TRAIN, seed=s, init_from=base_ck, grpo=dict(GRPO, reward_shift=SHIFT)), grpo_data)
    open_ck = os.path.join(w, "open", "open.safetensors")
    if not os.path.exists(open_ck):
        run([a.mtstudio, "run", open_spec], os.path.join(w, "open.log"), env)

    closed_spec = os.path.join(w, "closed_spec.json")
    write_spec(closed_spec, "closed", os.path.join(w, "closed"),
               dict(GRPO_TRAIN, seed=s, init_from=open_ck, grpo=dict(GRPO, reward_shift=0)), grpo_data)
    closed_ck = os.path.join(w, "closed", "closed.safetensors")
    if not os.path.exists(closed_ck):
        run([a.mtstudio, "run", closed_spec], os.path.join(w, "closed.log"), env)

    res = {"seed": s, "points": {}}
    for tag, spec, ck in (("base", base_spec, base_ck), ("open", open_spec, open_ck),
                          ("closed", closed_spec, closed_ck)):
        res["points"][tag] = dict(measure(a, spec, w, tag, dev, refs_true, refs_shift),
                                  checkpoint_sha256=sha256_file(ck))
    with open(done, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(f"seed {s}: done", flush=True)
    return res


def main(argv=None):
    global SHIFT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="plumbing check: 60 base and 10 GRPO steps")
    ap.add_argument("--shift", type=int, default=SHIFT, help="the reward-key shift (PT4: 1; PT4b sets its own)")
    a = ap.parse_args(argv)
    SHIFT = a.shift
    if a.smoke:
        BASE_TRAIN.update(steps=60, eval_every=20)
        GRPO_TRAIN.update(steps=10, eval_every=5, checkpoint_every=5)
    for p in (a.mtstudio, a.verify):
        if not (os.path.isfile(p) and os.access(p, os.X_OK)):
            raise SystemExit(f"{p}: not an executable file")
    os.makedirs(a.work, exist_ok=True)
    omp = max(1, (os.cpu_count() or 4) // a.jobs)
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda s: do_seed(a, s, omp), a.seeds))
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--short", "-uno"], capture_output=True, text=True).stdout.strip()
    out = {"seeds": a.seeds, "smoke": a.smoke, "shift": SHIFT, "repo_commit": commit, "repo_dirty": bool(dirty),
           "mtstudio_sha256": sha256_file(a.mtstudio), "verify_sha256": sha256_file(a.verify), "results": results}
    with open(os.path.join(a.work, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"results -> {os.path.join(a.work, 'results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
