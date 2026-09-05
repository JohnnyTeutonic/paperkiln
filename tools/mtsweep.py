#!/usr/bin/env python3
"""Atlas stage 1: the sweep runner (atlas/ARCHITECTURE_ATLAS.md).

A design matrix row IS a spec file — this tool makes that literal. It
expands a sweep description into mtstudio specs, executes them (resumably,
in parallel), and aggregates the Atlas rows, including the per-cell
seed statistics that single runs cannot have.

    python tools/mtsweep.py sweep.json [--jobs 2] [--dry-run]
                            [--mtstudio PATH]
    python tools/mtsweep.py --selftest

Sweep description:
{
  "base":    { ... a full mtstudio spec ... },      # or "base_path": "spec.json"
  "factors": { "train.lr": [1e-3, 3e-3],            # dotted spec paths
               "arch.custom.attention": ["exact", "kimi"] },
  "design":  "grid",                                 # or "pb12"
  "seeds":   [1, 2, 3],
  "out_root": "/tmp/sweep_demo"
}

Designs:
  grid  full factorial over the factor levels (any number of levels each)
  pb12  Plackett-Burman 12-run screen: up to 11 factors, each with
        exactly TWO levels — all main effects in 12 runs (x seeds).
        The screening pass of atlas/ARCHITECTURE_ATLAS.md section 5.3.

Seeds multiply the design; every run gets train.seed set. Aggregation
groups runs into CELLS (identical factor values, seeds pooled) and
reports mean/std/min of best_val per cell — seed variance is a cell
property, per the Atlas doc, and cells with std comparable to their
between-cell differences are flagged.

Parallelism: --jobs N runs N mtstudio processes with OMP_NUM_THREADS
split evenly and OMP_WAIT_POLICY=PASSIVE — the 2026-07-31 lesson that 2
processes x default threads on one box spin-locked to 1 step/min is
baked in here so it cannot recur.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_extract import extract_row  # noqa: E402
from atlas_taxonomy import spec_assignment, violations  # noqa: E402

# Classic PB12 first row; rows 2-11 are cyclic shifts, row 12 is all-minus.
_PB12_FIRST = [+1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1]


def pb12_matrix(n_factors):
    if not 1 <= n_factors <= 11:
        raise SystemExit("pb12: 1..11 factors")
    rows = []
    row = list(_PB12_FIRST)
    for _ in range(11):
        rows.append(row[:n_factors])
        row = [row[-1]] + row[:-1]
    rows.append([-1] * n_factors)
    return rows


def set_dotted(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand(sweep):
    factors = sweep.get("factors", {})
    names = list(factors.keys())
    design = sweep.get("design", "grid")
    if design == "grid":
        combos = list(itertools.product(*[factors[n] for n in names]))
    elif design in ("pb12", "pb12f"):
        for n in names:
            if len(factors[n]) != 2:
                raise SystemExit(f"{design} factor {n} needs exactly 2 levels")
        rows_ = pb12_matrix(len(names))
        if design == "pb12f":
            # FOLD-OVER (the Stage-2 lesson institutionalised): append the
            # mirror of every row. The combined 24-run design is
            # resolution IV — main effects come out CLEAR of two-way
            # interactions, which plain PB12 aliases. Costs one more
            # night; buys immunity to the exact failure Stage 3 exposed
            # (the lr main effect that was two conditional effects
            # cancelling).
            rows_ = rows_ + [[-s for s in r] for r in rows_]
        combos = [tuple(factors[n][0 if s < 0 else 1] for n, s in zip(names, r))
                  for r in rows_]
    else:
        raise SystemExit(f"unknown design {design}")

    # Aliasing advisory (the token-matching lesson, automated): a factor
    # on data.T or train.batch at fixed train.steps varies TOKENS SEEN
    # alongside the named quantity — Stage 2's "T=256 is better" was
    # "more data" wearing "longer context" clothes until Stage 3
    # token-matched it with a linked factor. Warn at design time.
    budget_paths = {"data.T": "context length", "train.batch": "batch size"}
    linked_paths = set()
    for levels in factors.values():
        for lv in levels:
            if isinstance(lv, dict):
                linked_paths.update(lv.keys())
    for name in names:
        if name in budget_paths and "train.steps" not in linked_paths:
            print(f"WARNING: factor {name} varies {budget_paths[name]} AND "
                  f"tokens-seen at fixed steps — its effect will alias "
                  f"data budget. Consider a LINKED factor pairing it with "
                  f"train.steps (see experiments/atlas_stage3/sweep.json).",
                  file=sys.stderr)

    seeds = sweep.get("seeds", [7])
    runs = []
    for ci, combo in enumerate(combos):
        for seed in seeds:
            runs.append({"cell": ci,
                         "factors": dict(zip(names, combo)),
                         "seed": seed})
    return names, combos, runs


def materialise(sweep, runs, out_root):
    base = sweep.get("base")
    if base is None:
        with open(sweep["base_path"], "r", encoding="utf-8") as f:
            base = json.load(f)
    os.makedirs(os.path.join(out_root, "specs"), exist_ok=True)
    spec_paths = []
    for i, r in enumerate(runs):
        spec = copy.deepcopy(base)
        for path, value in r["factors"].items():
            # LINKED factor: a dict level sets several spec paths at
            # once (the factor name is then a label, not a path). This
            # is how token-matched context works: {"data.T": 256,
            # "train.steps": 600} vs {"data.T": 128, "train.steps":
            # 1200} varies context length at constant tokens seen.
            if isinstance(value, dict):
                for p2, v2 in value.items():
                    set_dotted(spec, p2, v2)
            else:
                set_dotted(spec, path, value)
        set_dotted(spec, "train.seed", r["seed"])
        name = f"run_{i:03d}_c{r['cell']:02d}_s{r['seed']}"
        spec["name"] = name
        spec["out_dir"] = os.path.join(out_root, "runs", name)
        spec.setdefault("serve", {})["on_finish"] = False
        # The grammar gate: refuse to spend compute on an illegal config
        # (ARCHITECTURE_ATLAS stage 1 — validation is one of the
        # taxonomy's three consumers).
        v = violations(spec_assignment(spec))
        if v:
            raise SystemExit(f"{name}: invalid config: {'; '.join(v)}")
        p = os.path.join(out_root, "specs", name + ".json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        spec_paths.append((p, spec["out_dir"]))
    return spec_paths


def _git(args, cwd):
    try:
        p = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                           text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def build_provenance(binary, sweep_path):
    """WHICH CODE PRODUCED THIS RECEIPT.

    Found the hard way (31 Aug 2026): a whole night of Colab sweeps ran a
    CPU-only mtstudio because the CUDA wiring lived in an UNCOMMITTED
    working copy, and nothing in the receipt could have told us. The
    events stream records what the run did; it never recorded what built
    it. A registry whose receipts cannot name their own binary is one
    silent rebuild away from unreproducible.

    Written beside result.json in every run directory.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = ""
    try:
        h = hashlib.sha256()
        with open(binary, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    except OSError:
        pass
    dirty = _git(["status", "--porcelain"], repo)
    return {
        "repo_commit": _git(["rev-parse", "HEAD"], repo),
        "repo_dirty": bool(dirty),
        "repo_dirty_files": [ln[3:] for ln in dirty.splitlines()][:40],
        "binary_path": binary,
        "binary_sha256": sha,
        "binary_bytes": (os.path.getsize(binary)
                         if os.path.exists(binary) else 0),
        "sweep": os.path.abspath(sweep_path),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def find_mtstudio(cli):
    for cand in ([cli] if cli else []) + [
            "./mtstudio", "build/mtstudio",
            os.path.expanduser("~/mtrel/mtstudio")]:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    raise SystemExit("mtstudio binary not found; pass --mtstudio PATH")


def run_all(spec_paths, binary, jobs, omp=None, provenance=None):
    # omp overrides the even split — the polite profile for a machine
    # someone is USING (e.g. streaming): --jobs 1 --omp 4 leaves half
    # the cores untouched instead of soaking all of them.
    threads = omp or max(1, (os.cpu_count() or 4) // max(1, jobs))
    env = dict(os.environ,
               OMP_NUM_THREADS=str(threads),
               OMP_WAIT_POLICY="PASSIVE")

    def one(args):
        spec_path, out_dir = args
        if os.path.exists(os.path.join(out_dir, "result.json")):
            return (spec_path, "resumed-skip")
        r = subprocess.run([binary, "run", spec_path], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # Provenance rides WITH the receipt, not only in the sweep root:
        # run dirs are what get copied into experiments/*/receipts/.
        if provenance is not None and os.path.isdir(out_dir):
            try:
                with open(os.path.join(out_dir, "provenance.json"), "w",
                          encoding="utf-8") as pf:
                    json.dump(provenance, pf, indent=2)
            except OSError:
                pass
        return (spec_path, "ok" if r.returncode == 0 else f"EXIT {r.returncode}")

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for spec_path, status in ex.map(one, spec_paths):
            print(f"  {os.path.basename(spec_path):<28} {status}", flush=True)
            results.append(status)
    return results


def aggregate(runs, spec_paths, names, out_root):
    rows = []
    for r, (_, out_dir) in zip(runs, spec_paths):
        row = extract_row(out_dir)
        row["cell"] = r["cell"]
        # Dict (linked) levels serialize to a stable string so the
        # analyzer can sort/group them like any two-level factor.
        row.update({f"factor.{k}":
                    (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
                    for k, v in r["factors"].items()})
        rows.append(row)
    rows_path = os.path.join(out_root, "atlas_rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    cells = []
    for ci in sorted({r["cell"] for r in runs}):
        vals = [row.get("best_val") for row in rows
                if row["cell"] == ci and row.get("best_val") is not None]
        cell = {"cell": ci,
                "factors": next(r["factors"] for r in runs if r["cell"] == ci),
                "n_seeds": len(vals)}
        if vals:
            cell["best_val_mean"] = statistics.fmean(vals)
            cell["best_val_std"] = statistics.pstdev(vals) if len(vals) > 1 else None
            cell["best_val_min"] = min(vals)
        cells.append(cell)
    cells_path = os.path.join(out_root, "cells.jsonl")
    with open(cells_path, "w", encoding="utf-8") as f:
        for c in cells:
            f.write(json.dumps(c, sort_keys=True) + "\n")

    print(f"\ncells ({len(cells)}), by best_val_mean:")
    ranked = sorted([c for c in cells if "best_val_mean" in c],
                    key=lambda c: c["best_val_mean"])
    stds = [c["best_val_std"] for c in ranked if c.get("best_val_std")]
    seed_noise = statistics.fmean(stds) if stds else None
    for c in ranked:
        fac = " ".join(f"{k.split('.')[-1]}={v}" for k, v in c["factors"].items())
        std = f" ±{c['best_val_std']:.4f}" if c.get("best_val_std") else ""
        print(f"  {c['best_val_mean']:.4f}{std}  [{fac}] (n={c['n_seeds']})")
    if seed_noise is not None and len(ranked) >= 2:
        gap = ranked[1]["best_val_mean"] - ranked[0]["best_val_mean"]
        verdict = "SEPARATED from" if gap > 2 * seed_noise else "WITHIN"
        print(f"\n  top-2 gap {gap:.4f} is {verdict} mean seed noise "
              f"{seed_noise:.4f} — {'ordering is signal' if gap > 2 * seed_noise else 'ordering is NOT yet signal; more seeds or longer runs'}")
    print(f"\nrows -> {rows_path}\ncells -> {cells_path}")


def selftest():
    m = pb12_matrix(11)
    assert len(m) == 12 and all(len(r) == 11 for r in m)
    # Balance: every column has six + and six -.
    for j in range(11):
        col = [r[j] for r in m]
        assert col.count(+1) == 6 and col.count(-1) == 6, f"col {j} unbalanced"
    # Orthogonality: every column pair agrees on exactly 6 rows.
    for a in range(11):
        for b in range(a + 1, 11):
            agree = sum(1 for r in m if r[a] == r[b])
            assert agree == 6, f"cols {a},{b} not orthogonal"
    # Dotted set + grid expansion.
    d = {}
    set_dotted(d, "train.lr", 0.001)
    assert d == {"train": {"lr": 0.001}}
    names, combos, runs = expand({"factors": {"a": [1, 2], "b": [3, 4]},
                                  "seeds": [1, 2]})
    assert len(combos) == 4 and len(runs) == 8
    _, combos_pb, _ = expand({"factors": {f"f{i}": [0, 1] for i in range(11)},
                              "design": "pb12", "seeds": [1]})
    assert len(combos_pb) == 12
    # Fold-over: 24 rows, and RESOLUTION IV — every main-effect column is
    # orthogonal to every two-way interaction column in the combined
    # design (sum over rows of col_i * col_j*col_k == 0), which is
    # exactly the property plain PB12 lacks.
    m = pb12_matrix(7)
    mf = m + [[-s for s in r] for r in m]
    for i in range(7):
        for j in range(7):
            for k in range(j + 1, 7):
                dot = sum(r[i] * r[j] * r[k] for r in mf)
                assert dot == 0, (i, j, k, dot)
    _, combos_f, _ = expand({"factors": {f"f{i}": [0, 1] for i in range(7)},
                             "design": "pb12f", "seeds": [1]})
    assert len(combos_f) == 24
    # Aliasing advisory fires for an unlinked data.T factor.
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        expand({"factors": {"data.T": [128, 256], "train.lr": [1e-3, 3e-3]},
                "seeds": [1]})
    assert "tokens-seen" in err.getvalue(), err.getvalue()
    err2 = io.StringIO()
    with contextlib.redirect_stderr(err2):
        expand({"factors": {"ctx": [{"data.T": 128, "train.steps": 1200},
                                    {"data.T": 256, "train.steps": 600}]},
                "seeds": [1]})
    assert "tokens-seen" not in err2.getvalue(), err2.getvalue()
    # Linked factor: a dict level lands on ALL its paths (token-matched
    # context is the motivating case).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sw = {"base": {"data": {}, "train": {}},
              "factors": {"ctx": [{"data.T": 128, "train.steps": 1200},
                                  {"data.T": 256, "train.steps": 600}]},
              "seeds": [1]}
        _, _, lruns = expand(sw)
        paths = materialise(sw, lruns, td)
        specs = [json.load(open(p)) for p, _ in paths]
        got = {(s["data"]["T"], s["train"]["steps"]) for s in specs}
        assert got == {(128, 1200), (256, 600)}, got
    print("SELFTEST OK: pb12 balanced+orthogonal, grid/pb expansion, "
          "dotted set, linked factors, pb12f fold-over resolution-IV, "
          "aliasing advisory (fires unlinked, silent linked)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--omp", type=int, help="OMP threads per worker "
                    "(default: cores // jobs); lower it to stay polite "
                    "on a machine in active use")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mtstudio")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shard", default="",
                    help="K/N: run only cells whose global index i has "
                         "i %% N == K. Names keep the global index, so shards "
                         "run on different machines merge by copying run dirs. "
                         "Added 5 Sep 2026 to split transfer_s1 arm M across "
                         "two Colab vms.")
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to run from a dirty working tree — the "
                         "setting for any pre-registered experiment, whose "
                         "receipts must be reproducible from a commit")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.sweep:
        ap.error("give a sweep.json (or --selftest)")

    with open(args.sweep, "r", encoding="utf-8") as f:
        sweep = json.load(f)
    out_root = sweep["out_root"]
    names, combos, runs = expand(sweep)
    spec_paths = materialise(sweep, runs, out_root)
    print(f"{len(combos)} cells x {len(sweep.get('seeds', [7]))} seeds = "
          f"{len(runs)} runs -> {out_root}")
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        keep = [i for i in range(len(runs)) if i % n == k]
        spec_paths = [spec_paths[i] for i in keep]
        runs = [runs[i] for i in keep]
        print(f"shard {k}/{n}: {len(runs)} runs on this machine")
    if args.dry_run:
        for p, _ in spec_paths:
            print(f"  {p}")
        return
    binary = find_mtstudio(args.mtstudio)
    prov = build_provenance(binary, args.sweep)
    if prov["repo_dirty"]:
        msg = (f"WORKING TREE IS DIRTY ({len(prov['repo_dirty_files'])} "
               f"files) — these runs will not be reproducible from any "
               f"commit. Receipts will record repo_dirty=true.")
        if args.require_clean:
            raise SystemExit("REFUSING TO RUN: " + msg +
                             " (--require-clean)")
        print("WARNING: " + msg, file=sys.stderr)
    with open(os.path.join(out_root, "provenance.json"), "w",
              encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    run_all(spec_paths, binary, args.jobs, args.omp, prov)
    aggregate(runs, spec_paths, names, out_root)


if __name__ == "__main__":
    main()
