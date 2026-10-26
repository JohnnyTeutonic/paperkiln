#!/usr/bin/env python3
"""Amendment 6 (PROPOSED, not yet registered): the size arm on Carry-5 at 150M non-embedding parameters.
Writes the arm's specs and its Stage 0 jobs files, prints the shape and budget tables, and applies the
depth rule once E1 at 60M is home. Deterministic: rerunning it rewrites the same files. It imports
make_jobs.py (this folder's parent) as a library and writes none of make_jobs.py's files.

    python experiments/tiny_chat_2026-10/size/make_size_jobs.py            # write specs + Stage 0 files, print tables
    python experiments/tiny_chat_2026-10/size/make_size_jobs.py --tables   # print the tables only
    python experiments/tiny_chat_2026-10/size/make_size_jobs.py depth --art C:/ml_artifacts/tiny_chat \
        --out experiments/tiny_chat_2026-10/size/depth_choice.json          # A6.1's depth rule, at home

Writes
  experiments/tiny_chat_2026-10/size/specs/shape_150m_L<4|8|16|32>.json   one pretraining spec per candidate depth
  tools/colab_chat/tiny_chat_s0_150m_L<4|8|16|32>.json                   Stage 0, one file (one L4) per depth

Stage 0 per depth, one GPU job at a time: the CUDA gate (every leg of tools/colab_cuda_validate.sh, on that
machine's GPU), the gate (design final, the floor pilot's PASS), the 35% data job, a 300-step probe alone on
the GPU (tokens/s, peak memory at the registered micro-batching) and lr screens over {2.5e-4, 5e-4, 1e-3,
2e-3}, 1,500 steps, seed 101 (`tiny_chat_run.py choose_rec`, the four-rate grid of A3.3/A4.4). The L4 file
also holds the plain replay: E1 at 60M's pt_e1_60m_L32_r2 spec for its first 500 steps, compared at home with
E1 at 60M's log by x5/bitid.py.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
REL = os.path.relpath(HERE, ROOT).replace(os.sep, "/")
EXP_REL = os.path.relpath(EXP, ROOT).replace(os.sep, "/")
COLAB = os.path.join(ROOT, "tools", "colab_chat")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mj = _load("tiny_chat_make_jobs", os.path.join(EXP, "make_jobs.py"))
an = _load("tiny_chat_analyze", os.path.join(EXP, "analyze.py"))
tcr = mj.tcr

SIZE = "150m"
TARGET = 150_000_000
DEPTHS = [4, 8, 16, 32]                  # the candidate depths: E1 at 60M's four (A6.1)
SCREEN_LRS = list(tcr.REC_SCREEN_LRS)    # {2.5e-4, 5e-4, 1e-3, 2e-3}, the four-rate grid choose_rec reads
CHOICE = f"{EXP_REL}/stage0/lr_choice_150m.json"
DEPTH_CHOICE = f"{REL}/depth_choice.json"
SPECS = os.path.join(HERE, "specs")
STAGE0_HOME = os.path.join(HERE, "stage0")   # the probes' summary.json once home (stage0/probe_150m_L<L>/)
REPLAY = {"job": "replay_e1_60m_L32_r2", "sid": "60m_L32", "seed": 2, "steps": 500,
          "home": "C:/ml_artifacts/tiny_chat/e1_60m/pt_e1_60m_L32_r2",
          "commit": "a90ec0e2b2aaf6052a5ff953acd1e0a40a9a222c"}   # E1 at 60M's pinned commit
IN_RUN = 1.08            # in-run time over the probe's rate: E1 at 60M measured 1.06-1.11 (evals, checkpoints, rungs)
POST_EVAL_150M_H = 2.0   # per post job beyond its SFT at 150M; E1 at 60M measured 0.78-2.88 h (median 1.36)
PEAK_LIMIT_MIB = 19_500  # A6.2: a probe above this leaves too little of the L4 for the light rung job
USD_PER_L4_H = 0.59
TIE_Y = 0.01             # A6.1: depths within 1 point of the best mean Y are tied
CAP_MARGIN = 1.3         # a runner's --max-usd over its estimate


def batching(L):
    """A6.2: the micro-batching at 150M (the same 32 x 512 tokens per step and the same update; section 1
    accumulates for activation memory). 16 x 2 up to 16 layers, 8 x 4 at 32."""
    return (8, 4) if L >= 32 else (16, 2)


def shapes():
    out = {}
    for L in DEPTHS:
        s = mj.shape_for(TARGET, L)
        n = mj.nonemb(L, s["d"], s["d_ff"])
        e = mj.V * s["d"]
        out[f"{SIZE}_L{L}"] = {**s, "size": SIZE, "nonemb": n, "emb": e, "total": n + e}
    return out


def _fit():
    """us per token = a GFLOP + b layers + c, least squares through Stage 0's nine probes (A2.1)."""
    sh = mj.shapes()
    rows = [(mj.flops_per_token(sh[f"{z}_L{L}"]) / 1e9, L, 1e6 / v) for (z, L), v in sorted(mj.PROBES.items())]
    # the normal equations for three unknowns, solved by Gauss-Jordan elimination (no numpy on the VM)
    X = [[f, L, 1.0] for f, L, _ in rows]
    y = [t for *_, t in rows]
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(3)] for i in range(3)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(3)]
    n = 3
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


FIT = _fit() if mj.PROBES else None


def home_probes():
    out = {}
    for p in sorted(glob.glob(os.path.join(STAGE0_HOME, f"probe_{SIZE}_L*", "summary.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("rc") == 0 and d.get("tok_per_s_steady"):
            out[os.path.basename(os.path.dirname(p))[len("probe_"):]] = d
    return out


PROBES = home_probes()


def tok_per_s(sid, s):
    """The probe's rate once home; until then the planning rate, the lower of two extrapolations from
    Stage 0's probes: the fit through all nine (time per token in FLOPs and layers) and 60M's probe at the
    same depth scaled by training FLOPs per token."""
    if sid in PROBES:
        return PROBES[sid]["tok_per_s_steady"]
    f = mj.flops_per_token(s)
    a, b, c = FIT
    fit = 1e6 / (a * f / 1e9 + b * s["layers"] + c)
    s60 = mj.shapes()[f"60m_L{s['layers']}"]
    ratio = mj.PROBES[("60m", s["layers"])] * mj.flops_per_token(s60) / f
    return min(fit, ratio)


def hours(sid, s, steps):
    return steps * mj.TOKENS_PER_STEP / tok_per_s(sid, s) / 3600


def run_hours(sid, s):
    """One 150M run: pretraining and SFT at the planning rate times the in-run factor, plus the post job's
    evaluation (rung jobs run in the light slot beside the pretraining)."""
    return IN_RUN * (hours(sid, s, mj.STEPS) + hours(sid, s, mj.SFT_STEPS)) + POST_EVAL_150M_H


def spec_path(sid):
    return f"{REL}/specs/shape_{sid}.json"


def pt_spec(sid, s, lr, seed, steps=mj.STEPS):
    spec = mj.pt_spec(sid, s, mj.corpus_of(mj.E1_SHARE), lr, seed, steps=steps)
    b, a = batching(s["layers"])
    spec["train"].update({"batch": b, "accum": a})
    assert spec["train"]["batch"] * spec["train"]["accum"] * spec["data"]["T"] == mj.TOKENS_PER_STEP
    return spec


def shape_doc(sid, s):
    spec = pt_spec(sid, s, mj.PROVISIONAL_LR, mj.SEEDS[0])
    spec["_shape"] = {k: s[k] for k in ("size", "nonemb", "emb", "total")}
    return spec


def replay_spec():
    sh = mj.shapes()
    lrs, _ = mj.lr_table()
    return mj.pt_spec(REPLAY["sid"], sh[REPLAY["sid"]], mj.corpus_of(mj.E1_SHARE), lrs[REPLAY["sid"]],
                      REPLAY["seed"], steps=REPLAY["steps"])


PRE_GATE = [*mj.PILOT_GATE, "--need-pilot", "PASS"]


def stage0_name(L):
    return f"tiny_chat_s0_{SIZE}_L{L}.json"


def stage0_doc(sid, s, replay=False):
    L = s["layers"]
    name = stage0_name(L)
    gate = {"name": "gate", "cmd": ["python3", mj.RUN, "gate", "--out", "{out}", "--stage0", "--jobs",
                                    f"tools/colab_chat/{name}", "--design", f"{EXP_REL}/design.json", *PRE_GATE],
            "outputs": ["gate.json"]}
    cuda = {"name": "cuda_gate", "cmd": ["python3", mj.RUN, "cuda_gate", "--out", "{out}"],
            "outputs": ["cuda_gate.json", "cuda_gate.log"]}
    jobs = [cuda, gate, {**mj.data_job([mj.E1_SHARE]), "after": "gate"}]
    after = ["data", "cuda_gate"]
    pre = ["--cuda-gate", f"{mj.C}/cuda_gate", "--data-job", f"tools/colab_chat/{name}"]
    corpus = mj.corpus_of(mj.E1_SHARE)
    jobs.append({"name": f"probe_{sid}", "after": after,
                 "cmd": ["python3", mj.RUN, "probe", "--out", "{out}", "--spec", spec_path(sid), "--corpus", corpus,
                         "--steps", str(mj.PROBE_STEPS), *pre],
                 "outputs": ["summary.json"]})
    prev = f"probe_{sid}"   # a probe that fails (memory) holds the screens back: they would fail the same way
    est = mj.CUDA_GATE_H + mj.DATA_H + hours(sid, s, mj.PROBE_STEPS)
    if replay:   # independent of this depth's probe and screens, so neither outcome holds the other back
        jobs.append({"name": REPLAY["job"], "model_name": f"tc_e1_60m_L32_r{REPLAY['seed']}", "after": after,
                     "spec": replay_spec()})
        est += mj.hours(mj.shapes()[REPLAY["sid"]], REPLAY["steps"])
    for k, lr in enumerate(SCREEN_LRS):
        jobs.append({"name": f"screen_{sid}_lr{k}", "after": prev,
                     "cmd": ["python3", mj.RUN, "screen", "--out", "{out}", "--spec", spec_path(sid), "--corpus", corpus,
                             "--lr", repr(lr), "--steps", str(mj.SCREEN_STEPS), "--seed", str(mj.SCREEN_SEED), *pre],
                     "outputs": ["summary.json"]})
        est += IN_RUN * hours(sid, s, mj.SCREEN_STEPS)
    rep = (f" Also the plain replay {REPLAY['job']}: E1 at 60M's pt_e1_60m_L32_r{REPLAY['seed']} spec at this commit "
           f"for its first {REPLAY['steps']} steps (the warmup), compared at home step by step with its log at "
           f"{REPLAY['commit'][:8]} by {EXP_REL}/x5/bitid.py into {REL}/plain_bitid.json.") if replay else ""
    doc = {"_comment": f"tiny-chat size arm, Stage 0 for {sid} ({EXP_REL}/PREREGISTRATION.md amendment 6, PROPOSED): "
                       f"the CUDA gate on this machine's GPU, the gate (design final, the floor pilot's PASS), the "
                       f"{mj.E1_SHARE}% data job, a {mj.PROBE_STEPS}-step probe alone on the GPU (tokens/s and peak "
                       f"memory at batch {batching(L)[0]} x accum {batching(L)[1]} x {mj.T}) and lr screens "
                       f"{SCREEN_LRS} x {mj.SCREEN_STEPS} steps (seed {mj.SCREEN_SEED}), one GPU job at a time."
                       + rep + f" Estimated {est:.1f} L4 VM-hours. Afterwards, at home: copy ART/probe_{sid} to "
                       f"{REL}/stage0/ and ART/cuda_gate/cuda_gate.json to {REL}/stage0/cuda_gate_{sid}.json; once "
                       f"the depth is fixed (make_size_jobs.py depth), python {mj.RUN} choose_rec --art ART_of_that_depth "
                       f"--out {CHOICE}; commit. Upload vocab_llama3_bpe.gguf with the runner's --upload." + mj.DESIGN_NOTE,
           "max_parallel": mj.GPU_JOBS_PER_VM, "jobs": jobs}
    return name, doc, est


def use_design():
    with open(mj.DESIGN, encoding="utf-8") as f:
        design = json.load(f)
    mj.SHARES, mj.E1_SHARE = design["levels"], design["e1_share"]
    mj.DESIGN_NOTE = (f" Dialogue levels {mj.SHARES}% ({design['mode']}), E1 at {mj.E1_SHARE}%"
                      + (" (design.json final)." if design.get("final") else " PROVISIONAL."))
    return design


def x_of(nonemb):
    return math.log2(nonemb)


def slope_weights_for(L):
    """A6.4: the per-seed OLS weights of Y on log2(non-embedding parameters) over 20M, 60M and 150M at depth L."""
    n = [mj.shapes()[f"20m_L{L}"]["nonemb"], mj.shapes()[f"60m_L{L}"]["nonemb"], shapes()[f"{SIZE}_L{L}"]["nonemb"]]
    return an.slope_weights([x_of(v) for v in n]), n


def tables():
    sh = shapes()
    rows = ["| shape | layers | d | heads | d_ff | d_ff/d | non-emb | emb | total | emb share | GFLOP/token | "
            "micro-batch | tok/s (planning) | PT h | SFT h | run h |", "|" + "---|" * 16]
    for sid, s in sh.items():
        b, a = batching(s["layers"])
        rows.append(f"| {sid} | {s['layers']} | {s['d']} | {s['heads']} | {s['d_ff']} | {s['d_ff'] / s['d']:.2f} | "
                    f"{s['nonemb'] / 1e6:.2f}M | {s['emb'] / 1e6:.2f}M | {s['total'] / 1e6:.2f}M | "
                    f"{s['emb'] / s['total']:.0%} | {mj.flops_per_token(s) / 1e9:.2f} | {b} x {a} | "
                    f"{tok_per_s(sid, s) / 1e3:.2f}k{' (probe)' if sid in PROBES else ''} | "
                    f"{IN_RUN * hours(sid, s, mj.STEPS):.1f} | {IN_RUN * hours(sid, s, mj.SFT_STEPS):.2f} | "
                    f"{run_hours(sid, s):.1f} |")
    out = ["\n".join(rows), ""]
    b = ["| block | L4 VM-hours | USD at $0.59/h | runner cap |", "|---|---|---|---|"]
    tot0 = 0.0
    for sid, s in sh.items():
        _, _, est = stage0_doc(sid, s, replay=(s["layers"] == 4))
        tot0 += est
        b.append(f"| Stage 0 {sid} (CUDA gate, data, probe, 4 screens{', replay' if s['layers'] == 4 else ''}) | "
                 f"{est:.1f} | {est * USD_PER_L4_H:.2f} | ${cap(est)} |")
    b.append(f"| Stage 0, all four depths | {tot0:.1f} | {tot0 * USD_PER_L4_H:.2f} | |")
    for sid, s in sh.items():
        r = mj.CUDA_GATE_H + mj.DATA_H + run_hours(sid, s)
        b.append(f"| runs at {sid}: 3 seeds, one L4 per seed (CUDA gate + data each) | {3 * r:.1f} | "
                 f"{3 * r * USD_PER_L4_H:.2f} | ${cap(r)} per seed |")
    out.append("\n".join(b))
    w, n = slope_weights_for(32)
    out.append("\nA6.4 slope weights at L32 over log2 non-embedding (" + ", ".join(f"{v / 1e6:.2f}M" for v in n)
               + "): " + ", ".join(f"{x:+.3f}" for x in w))
    return "\n".join(out)


def cap(h):
    return int(math.ceil(h * USD_PER_L4_H * CAP_MARGIN))


# ---------------------------------------------------------------- A6.1: the depth rule, at home
def load_60m(art):
    """E1 at 60M's post summaries from ART/e1_60m and its per-cell split folders (implementation note of 25 Oct)."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(art, "e1_60m*"))):
        runs.update(an.load(art, os.path.basename(d)))
    return runs


def depth_rule(runs):
    """A6.1, fixed before E1 at 60M is home: once all twelve of its runs (L4, L8, L16, L32 x seeds 1-3) have their
    post summaries, the 150M depth is the 60M depth with the highest mean Y (A2.5's primary) over seeds 1-3 among
    the cells that pass the validity gate; depths within TIE_Y of that mean are tied and the one with the higher
    60M probe rate (A2.1) wins."""
    cells = [f"L{L}" for L in DEPTHS]
    seeds = an.SEEDS_FIRST
    miss = an.cells_complete(runs, cells, seeds)
    if miss:
        raise SystemExit(f"E1 at 60M incomplete: {miss}; the depth rule waits")
    tab = an.cell_table(runs, cells, seeds, ["Y", "twin_self", "twin_minus_qalone_ctx"])
    ok = {c: an.floor_ok(runs, c, seeds) for c in cells}
    good = [c for c in cells if ok[c]]
    if not good:
        raise SystemExit("every 60M cell is FLOOR: no depth is chosen and Jonathan decides")
    best = max(tab[c]["Y"]["mean"] for c in good)
    tied = [c for c in good if tab[c]["Y"]["mean"] >= best - TIE_Y]
    rate = {c: mj.PROBES[("60m", int(c[1:]))] for c in cells}
    pick = max(tied, key=lambda c: (rate[c], -int(c[1:])))
    return {"rule": " ".join(depth_rule.__doc__.split()), "seeds": seeds,
            "mean_Y": {c: tab[c]["Y"]["mean"] for c in cells}, "floor_ok": ok, "tied": tied,
            "probe_tok_per_s_60m": rate, "depth": int(pick[1:]), "shape": f"{SIZE}_{pick}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", nargs="?", default="write", choices=["write", "depth"])
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--art", default="C:/ml_artifacts/tiny_chat")
    ap.add_argument("--out", default=os.path.join(ROOT, DEPTH_CHOICE))
    a = ap.parse_args(argv)
    use_design()
    if a.mode == "depth":
        r = depth_rule(load_60m(a.art))
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: the depth is chosen once")
        mj.write(a.out, r)
        print(json.dumps(r, indent=1))
        return 0
    print(tables())
    if a.tables:
        return 0
    os.makedirs(SPECS, exist_ok=True)
    for sid, s in shapes().items():
        mj.write(os.path.join(SPECS, f"shape_{sid}.json"), shape_doc(sid, s))
        name, doc, _ = stage0_doc(sid, s, replay=(s["layers"] == 4))
        mj.write(os.path.join(COLAB, name), doc)
    print("\nwrote size/specs/shape_150m_L*.json and tools/colab_chat/tiny_chat_s0_150m_L*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
