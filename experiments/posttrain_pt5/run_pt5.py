#!/usr/bin/env python3
"""PT5 driver (PREREGISTRATION.md). Resumable. Reuses PT2's base recipe,
sampling, scoring and evaluation (experiments/posttrain_pt2/run_pt2.py) so
the two studies measure with the same code.

    python experiments/posttrain_pt5/run_pt5.py --work ~/pt5 \\
        --mtstudio build/mtstudio --verify build/posttrain_verify [--jobs 4] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "posttrain_pt2"))
import run_pt2 as P2  # noqa: E402

R = 20
SEEDS = list(range(301, 311))
ARMS = ["E", "J", "R", "Mu", "Mw", "Md"]
PER_PROMPT = 4
PO_TRAIN = {"steps": 400, "lr": 0.001, "batch": 8, "eval_every": 100, "checkpoint_every": 1000,
            "gradmap_every": 0, "optimizer": "adamw", "objective": "provenance_po"}
BETA = 0.1


# ---- pure parts (tested by smoke_analyze.py) -------------------------------

def pools(prompt_of, text, exact, judge, fmt, seed):
    """Candidate pairs per pool, as (chosen index, rejected index). At most
    PER_PROMPT per prompt per pool; identical completions never pair; both
    completions must pass the format verifier, so the arms differ in WHICH
    well-formed answer is preferred, never in whether the output is
    well-formed (the smoke run showed format-mixed tie pairs teaching the R
    arm to stop writing answers at all)."""
    by = {}
    for i, p in enumerate(prompt_of):
        by.setdefault(p, []).append(i)
    out = {"E": [], "J": [], "tie": []}
    for p in sorted(by):
        idx = by[p]
        cand = {"E": [], "J": [], "tie": []}
        for a in idx:
            for b in idx:
                if a == b or text[a] == text[b] or not (fmt[a] and fmt[b]):
                    continue
                if exact[a] == 1 and exact[b] == 0:
                    cand["E"].append((a, b))
                if judge[a] == 1 and judge[b] == 0:
                    cand["J"].append((a, b))
                if a < b and exact[a] == exact[b]:
                    cand["tie"].append((a, b))
        for k in cand:
            rng = random.Random(f"pt5/pool/{seed}/{p}/{k}")
            rng.shuffle(cand[k])
            out[k] += cand[k][:PER_PROMPT]
    return out


def build_arms(pl, exact, seed):
    """Equal-N arms. Returns {arm: [(chosen, rejected, source)]}, N and the
    J error composition. source in {E, J, Rflip, Rkeep, Rtie}."""
    J = pl["J"]
    if not J or not pl["E"]:
        raise SystemExit(f"seed {seed}: an empty pool (E {len(pl['E'])}, J {len(J)})")
    inv = sum(1 for a, b in J if exact[a] == 0 and exact[b] == 1) / len(J)
    tie = sum(1 for a, b in J if exact[a] == exact[b]) / len(J)
    caps = [len(pl["E"]), len(J)]
    if tie > 0:
        caps.append(int(len(pl["tie"]) / tie))
    N = min(caps)
    N -= N % 2
    if N < 2:
        raise SystemExit(f"seed {seed}: N = {N}")
    rng = random.Random(f"pt5/arms/{seed}")
    E = [(a, b, "E") for a, b in rng.sample(pl["E"], N)]
    Jp = [(a, b, "J") for a, b in rng.sample(J, N)]
    n_tie = round(tie * N)
    flip = inv / (1 - tie) if tie < 1 else 0.0
    Rp = [(a, b, "Rtie") for a, b in rng.sample(pl["tie"], n_tie)]
    for a, b in rng.sample(pl["E"], N - n_tie):
        Rp.append((b, a, "Rflip") if rng.random() < flip else (a, b, "Rkeep"))
    half_e = [(a, b, "E") for a, b in rng.sample(pl["E"], N // 2)]
    half_j = [(a, b, "J") for a, b in rng.sample(J, N - N // 2)]
    mix = half_e + half_j
    rng.shuffle(mix)
    arms = {"E": E, "J": Jp, "R": Rp, "Mu": mix, "Mw": mix, "Md": mix}
    return arms, N, {"f_inv": inv, "f_tie": tie, "pool_E": len(pl["E"]), "pool_J": len(J),
                     "pool_tie": len(pl["tie"])}


# ---- stages -----------------------------------------------------------------

def drop_if_partial(path, expected_lines):
    """A killed sampler leaves a short or empty file, and resuming must not
    take it for a finished one (found 17 Oct 2026: a smoke run killed by its
    wrapper's timeout left an empty evaluation file that the resume
    skipped). Delete anything short so the stage reruns."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    if n != expected_lines:
        os.remove(path)


def clean_partials(w, tags, n_sample):
    for split in ("test", "hackprobe"):
        n_prompts = len(P2.read_jsonl(os.path.join(ROOT, f"corpus/eval/tinyarith_r{R}_{split}_v1.jsonl")))
        for tag in tags:
            drop_if_partial(os.path.join(w, f"eval_{tag}_{split}.jsonl"), n_prompts * n_sample)


def receipt(rec, kind_reward, kind, note=None):
    r = {"evidence_kind": kind, "evidence_uri": rec.get("evidence_uri", "pt5"), "reward": kind_reward,
         "audited": False, "rationale": note or rec.get("rationale", "")}
    return r


def pair_rows(arm, pairs, recs, exact, judge):
    rows = []
    for a, b, src in pairs:
        kind = "judge" if src == "J" else "exact_match"
        rv = judge if kind == "judge" else exact
        note = None
        if src == "Rflip":
            note = "PT5 noise-matched arm: exact-ordered pair flipped by the pre-registered random rule"
        elif src == "Rtie":
            note = "PT5 noise-matched arm: a truth-tied pair drawn by the pre-registered rule"
        rows.append({"prompt": recs[a]["prompt"], "chosen": recs[a]["completion"],
                     "rejected": recs[b]["completion"], "source": src, "arm": arm,
                     "chosen_evidence": receipt(recs[a], rv[a], kind, note),
                     "rejected_evidence": receipt(recs[b], rv[b], kind, note)})
    return rows


def do_seed(a, s, omp):
    env = dict(os.environ, OMP_NUM_THREADS=str(omp), OMP_WAIT_POLICY="PASSIVE")
    w = os.path.join(a.work, f"seed_{s}")
    os.makedirs(w, exist_ok=True)
    done = os.path.join(w, "result.json")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as f:
            return json.load(f)
    base_spec = os.path.join(w, "base_spec.json")
    P2.write_spec(base_spec, "base", os.path.join(w, "base"), R, dict(P2.BASE_TRAIN, seed=s))
    base_ck = os.path.join(w, "base", "base.safetensors")
    if not os.path.exists(base_ck):
        P2.run([a.mtstudio, "run", base_spec], os.path.join(w, "base.log"), env)
    prompts = P2.sampling_prompts(R, os.path.join(w, "sampling_prompts.jsonl"))
    smp = os.path.join(w, "samples.jsonl")
    drop_if_partial(smp, len(P2.read_jsonl(prompts)) * P2.N_SAMPLE)
    clean_partials(w, ["base"] + ARMS, P2.N_SAMPLE)
    if not os.path.exists(smp):
        P2.run([a.mtstudio, "sample-set", base_spec, "--prompts", prompts, "--out", smp, "--n",
                str(P2.N_SAMPLE), "--temp", repr(P2.TEMP), "--seed", str(s), "--max-new", str(P2.MAX_NEW)], env=env)
    sc = P2.score_file(a, smp, prompts)
    recs = sc["records"]
    exact = [int(x) for x in sc["exact"]]
    judge = [int(x) for x in sc["J25"]]
    fmt = [int(x) for x in sc["format"]]
    prompt_of = [r["prompt"] for r in recs]
    text = [r["completion"] for r in recs]
    arms, N, comp = build_arms(pools(prompt_of, text, exact, judge, fmt, s), exact, s)
    result = {"seed": s, "N": N, "composition": comp, "models": {}}
    result["models"]["base"] = P2.evaluate(a, base_spec, w, R, "base")
    for arm in ARMS:
        pf = os.path.join(w, f"pairs_{arm}.jsonl")
        if not os.path.exists(pf):
            P2.write_jsonl(pf, pair_rows(arm, arms[arm], recs, exact, judge))
        po = {"beta": BETA, "weight_mode": "uniform" if arm in ("E", "J", "R", "Mu") else "provenance"}
        if arm == "Md":
            po["weights"] = {"judge": 0.0}
        spec = os.path.join(w, f"po_{arm}_spec.json")
        P2.write_spec(spec, f"po_{arm}", os.path.join(w, f"po_{arm}"), R,
                      dict(PO_TRAIN, seed=s, init_from=base_ck, po=po), {"pref_pairs": pf})
        ck = os.path.join(w, f"po_{arm}", f"po_{arm}.safetensors")
        if not os.path.exists(ck):
            P2.run([a.mtstudio, "run", spec], os.path.join(w, f"po_{arm}.log"), env)
        result["models"][arm] = P2.evaluate(a, spec, w, R, arm)
        result["models"][arm]["checkpoint_sha256"] = P2.sha256_file(ck)
        ev = [json.loads(x) for x in open(os.path.join(w, f"po_{arm}", "events.jsonl"), encoding="utf-8")]
        d = next(e for e in ev if e["event"] == "data")
        result["models"][arm]["dataset_ess"] = d["ess_kish"]
        result["models"][arm]["weight_sum"] = d["weight_sum"]
    with open(done, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    print(f"seed {s}: done (N={N})", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="plumbing check: 60 base and 20 PO steps")
    a = ap.parse_args(argv)
    if a.smoke:
        P2.BASE_TRAIN.update(steps=60, eval_every=20)
        PO_TRAIN.update(steps=20, eval_every=10)
    for p in (a.mtstudio, a.verify):
        if not (os.path.isfile(p) and os.access(p, os.X_OK)):
            raise SystemExit(f"{p}: not an executable file")
    os.makedirs(a.work, exist_ok=True)
    omp = max(1, (os.cpu_count() or 4) // a.jobs)
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda s: do_seed(a, s, omp), a.seeds))
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--short", "-uno"], capture_output=True, text=True).stdout.strip()
    out = {"seeds": a.seeds, "smoke": a.smoke, "repo_commit": commit, "repo_dirty": bool(dirty),
           "mtstudio_sha256": P2.sha256_file(a.mtstudio), "verify_sha256": P2.sha256_file(a.verify),
           "results": results}
    with open(os.path.join(a.work, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"results -> {os.path.join(a.work, 'results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
