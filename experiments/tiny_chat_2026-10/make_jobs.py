#!/usr/bin/env python3
"""Specs and Colab jobs files for the tiny-chat experiments (PREREGISTRATION.md
in this folder). Deterministic: rerunning it rewrites the same files.

    python experiments/tiny_chat_2026-10/make_jobs.py            # write everything, print the tables
    python experiments/tiny_chat_2026-10/make_jobs.py --tables   # print the shape and budget tables only

Writes
  experiments/tiny_chat_2026-10/specs/shape_<id>.json   one pretraining spec per shape (mtstudio)
  experiments/tiny_chat_2026-10/specs/sft.json          the SFT stage (its arch comes from the shape)
  tools/colab_chat/tiny_chat_s0.json                    Stage 0 at 20M: data, throughput probes, lr screens
  tools/colab_chat/tiny_chat_e5.json                    E5: dialogue share 0/25/50/100 at 20m_L8, 3 seeds
  tools/colab_chat/tiny_chat_e1.json                    E1 at 20M: L2, L4, L16, L32 x 3 seeds (L8 is E5's 50% cell)
  tools/colab_chat/tiny_chat_s0_60m.json, tiny_chat_e1_60m.json   the 60M stage (on Jonathan's go)

Learning rates come from experiments/tiny_chat_2026-10/stage0/lr_choice.json (written at home by
`tiny_chat_run.py choose` from Stage 0's screens, by the registered rule). Until that file exists the
E files carry the PROVISIONAL rate 1e-3 and their gate job fails on the VM, so no E job can train on
a rate the screen did not choose.
"""
from __future__ import annotations

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REL = os.path.relpath(HERE, ROOT).replace(os.sep, "/")
COLAB = os.path.join(ROOT, "tools", "colab_chat")
C = "/content/chat"
VOCAB = f"{C}/upload/vocab_llama3_bpe.gguf"
V = 16384
T = 512
TOKENS_PER_STEP = 32 * T          # batch 32 x T, or batch 16 x accum 2 for L >= 16 (activation memory)
STEPS = 10_000                    # 163,840,000 tokens per pretraining run
SFT_STEPS = 1500
SEEDS = [1, 2, 3]
SCREEN_SEED = 101
SCREEN_STEPS = 1500
SCREEN_LRS = [5e-4, 1e-3, 2e-3]
PROBE_STEPS = 300
NOMINAL_SHARES = [0, 25, 50, 100]  # dialogue share of the pretraining mix, % of bytes, when the corpus allows
MAX_COPIES = 4                     # passes over the pretraining dialogues (Muennighoff et al. 2023)
MIX_BYTES = 800_000_000            # ~182M tokens at ~4.4 bytes/token: the 95% train part covers one budget
PT_FRACTION = 0.85                 # tiny_chat_data.py's pretraining split of the human dialogues
DEFAULT_SOURCES = ["oasst", "convai2", "wow", "bst", "taskmaster", "multiwoz", "self_dialogue"]
DESIGN = os.path.join(HERE, "design.json")
SHARES, E1_SHARE = NOMINAL_SHARES, 50  # replaced from design.json in main()
DESIGN_NOTE = ""


def dialogue_levels(pt_bytes):
    """The registered level rule (PREREGISTRATION.md section 5). P = rendered bytes of the
    pretraining split, of which 95% is usable per copy (the mixer's document overshoot margin).
    If four copies cover a whole mix, the levels are the nominal shares 0/25/50/100%. Otherwise
    they are 0 and one, two and four copies: share_k = floor(100 k 0.95 P / MIX_BYTES)%. E1's
    level is the non-zero level nearest 50% (ties: the smaller)."""
    per_copy = 0.95 * pt_bytes / MIX_BYTES
    if MAX_COPIES * per_copy >= 1.0:
        levels, mode = list(NOMINAL_SHARES), "shares"
    else:
        levels, mode = [0] + [int(100 * k * per_copy) for k in (1, 2, MAX_COPIES)], "copies"
    if len(set(levels)) != len(levels) or levels[1] < 1:
        raise SystemExit(f"dialogue levels {levels} are not distinct: the corpus is too small for E5")
    e1 = min(levels[1:], key=lambda s: (abs(s - 50), s))
    return {"pt_bytes": pt_bytes, "mode": mode, "levels": levels, "e1_share": e1}


def estimate_pt_bytes(dirs):
    """P from the built default-set corpora: PT_FRACTION of their rendered text.txt bytes."""
    tot, seen = 0, []
    for d in dirs:
        for s in DEFAULT_SOURCES:
            p = os.path.join(d, f"human_{s}_v1", "text.txt")
            if os.path.exists(p) and s not in seen:
                tot += os.path.getsize(p)
                seen.append(s)
    return int(PT_FRACTION * tot), seen
PROVISIONAL_LR = 1e-3
SIZES = {"20m": (20_000_000, [2, 4, 8, 16, 32]), "60m": (60_000_000, [4, 8, 16, 32])}
E5_SHAPE = "20m_L8"
RUN = "tools/colab_chat/tiny_chat_run.py"

# Throughput model (PREREGISTRATION.md, compute budget): per job on an L4 shared with 1-3 others,
# min(per-layer rate, FLOP ceiling). Anchors from the receipts: arxiv_2302_13971_base (d384 L6, T 512,
# batch 16, 16k vocab) 19,388 tok/s beside pk_base_v1; pk_base_v1 (d768 L12, same T and batch)
# 0.84-1.01 s/step = 8.1-9.8k tok/s. Both give ~0.07 s per layer per 8,192 tokens.
LAYER_TOKENS_PER_S = 8192 / 0.07      # tokens/s for a 1-layer model
FLOP_CEILING = 6.0e12                 # sustained FLOP/s seen on the L4 (pk_base_v1)
VM_SPEEDUP = 2.5                      # a VM running 3-4 jobs delivers ~2.5 jobs' worth (assumption; Stage 0 measures)


def nonemb(L, d, dff):
    """Llama block without biases: q,k,v,o (4 d^2) + SwiGLU (3 d dff) + 2 RMSNorm gains; + final norm."""
    return L * (4 * d * d + 3 * d * dff + 2 * d) + d


def shape_for(target, L):
    """d a multiple of 64 (head dim 64) whose matching d_ff/d is closest to 3 in log terms; d_ff a
    multiple of 32 bringing the non-embedding count nearest the target."""
    best = None
    for d in range(64, 2049, 64):
        per = target / L - 4 * d * d - 2 * d
        if per <= 0:
            continue
        dff = max(32, int(round(per / (3 * d) / 32)) * 32)
        score = abs(math.log(dff / d / 3))
        if best is None or score < best[0]:
            best = (score, d, dff)
    _, d, dff = best
    return {"layers": L, "d": d, "heads": d // 64, "d_ff": dff}


def shapes():
    out = {}
    for size, (target, layers) in SIZES.items():
        for L in layers:
            s = shape_for(target, L)
            n = nonemb(L, s["d"], s["d_ff"])
            e = V * s["d"]
            out[f"{size}_L{L}"] = {**s, "size": size, "nonemb": n, "emb": e, "total": n + e}
    return out


def batching(L):
    return (16, 2) if L >= 16 else (32, 1)


def flops_per_token(s):
    # 6 x (non-embedding + tied head matmul) + attention (Kaplan's 6 L T d for training)
    return 6 * (s["nonemb"] + V * s["d"]) + 6 * s["layers"] * T * s["d"]


def tok_per_s(s):
    return min(LAYER_TOKENS_PER_S / s["layers"], FLOP_CEILING / flops_per_token(s))


def hours(s, steps):
    return steps * TOKENS_PER_STEP / tok_per_s(s) / 3600


def lr_table():
    p = os.path.join(HERE, "stage0", "lr_choice.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["lr"], False
    return {}, True


def pt_spec(sid, s, corpus, lr, seed, steps=STEPS):
    b, a = batching(s["layers"])
    return {"name": f"tc_{sid}",
            "arch": {"preset": "llama-tiny", "rope_heads": "all",
                     "custom": {"d": s["d"], "layers": s["layers"], "heads": s["heads"], "d_ff": s["d_ff"]}},
            "data": {"corpus": corpus, "vocab": VOCAB, "vocab_cap": V, "T": T, "max_tokens": 0},
            "train": {"steps": steps, "lr": lr, "batch": b, "accum": a, "clip": 1.0, "eval_every": 1000,
                      "eval_windows": 32, "checkpoint_every": 500, "gradmap_every": 0, "seed": seed,
                      "optimizer": "adamw",
                      "lr_schedule": {"warmup_steps": 500, "decay": "cosine", "min_lr_frac": 0.1}},
            "export": {"formats": ["safetensors", "gguf"]}}


SFT = {"data": {"vocab": VOCAB, "vocab_cap": V, "T": T, "sft_examples": f"{C}/data/sft_human.jsonl"},
       "train": {"objective": "sft", "steps": SFT_STEPS, "lr": 3e-4, "batch": 32, "accum": 1, "clip": 1.0,
                 "eval_every": 500, "eval_batch": 8, "checkpoint_every": 500, "gradmap_every": 0, "seed": 1,
                 "optimizer": "adamw", "lr_schedule": {"warmup_steps": 100, "decay": "cosine", "min_lr_frac": 0.1}},
       "export": {"formats": ["safetensors", "gguf"]},
       "_comment": "SFT stage of every run: arch from the run's shape spec, batch/accum by the shape rule "
                   "(16 x 2 for L >= 16), train.seed = the run's seed, train.init_from = its pretrained weights. "
                   "sft_human.jsonl: tiny_chat_data.py, segments form (every assistant turn trained, "
                   "tools/chat_data.py sft --all-turns)."}


def data_job(shares):
    return {"name": "data",
            "cmd": ["python3", "tools/colab_chat/tiny_chat_data.py", "--out", "{out}", "--vocab", VOCAB,
                    "--fineweb", "corpus/data/fineweb_edu_shard0_v1", "--shares", ",".join(map(str, shares)),
                    "--mix-bytes", str(MIX_BYTES), "--max-epochs", "4", "--T", str(T), "--seed", "1"],
            "outputs": ["data.ledger.json"]}


def corpus_of(share):
    return f"{C}/data/mix_s{share}.ids"


def spec_path(sid):
    return f"{REL}/specs/shape_{sid}.json"


def stage0(sh, size):
    ids = [k for k in sh if sh[k]["size"] == size]
    gate = {"name": "gate", "cmd": ["python3", RUN, "gate", "--out", "{out}", "--stage0", "--jobs",
                                    f"tools/colab_chat/tiny_chat_s0{'' if size == '20m' else '_' + size}.json",
                                    "--design", f"{REL}/design.json"], "outputs": ["gate.json"]}
    jobs = [gate, {**data_job([E1_SHARE]), "after": "gate"}]
    prev = "data"
    for sid in ids:  # alone on the VM, one after another: clean per-shape throughput and peak memory
        jobs.append({"name": f"probe_{sid}", "after": prev,
                     "cmd": ["python3", RUN, "probe", "--out", "{out}", "--spec", spec_path(sid),
                             "--corpus", corpus_of(E1_SHARE), "--steps", str(PROBE_STEPS)],
                     "outputs": ["summary.json"]})
        prev = f"probe_{sid}"
    for sid in sorted(ids, key=lambda k: -sh[k]["layers"]):  # longest first
        for k, lr in enumerate(SCREEN_LRS):
            jobs.append({"name": f"screen_{sid}_lr{k}", "after": prev,
                         "cmd": ["python3", RUN, "screen", "--out", "{out}", "--spec", spec_path(sid),
                                 "--corpus", corpus_of(E1_SHARE), "--lr", repr(lr), "--steps", str(SCREEN_STEPS),
                                 "--seed", str(SCREEN_SEED)],
                         "outputs": ["summary.json"]})
    est = sum(hours(sh[s], PROBE_STEPS) for s in ids) + sum(hours(sh[s], SCREEN_STEPS) * 3 for s in ids)
    doc = {"_comment": f"tiny-chat Stage 0 ({size}; {REL}/PREREGISTRATION.md): the {E1_SHARE}% dialogue mix (E1 level; design.json) built on the VM, "
                       f"a {PROBE_STEPS}-step throughput probe per shape alone on the GPU, then lr screens "
                       f"{SCREEN_LRS} x {SCREEN_STEPS} steps per shape (seed {SCREEN_SEED}), three at a time. "
                       f"Estimated {est:.1f} job-hours. Brings home one summary.json per job. Afterwards, at home: "
                       f"python {RUN} choose --art ART --out {REL}/stage0/lr_choice.json, commit, rerun make_jobs.py. "
                       f"Upload vocab_llama3_bpe.gguf with the runner's --upload." + DESIGN_NOTE,
           "max_parallel": 3, "jobs": jobs}
    return doc


def run_jobs(tag, cells, sh, lrs, provisional):
    """cells: [(cell id, shape id, share)]. PT = training job (relayed, resumable across VMs); post = SFT,
    panels and Carry-5 as a data job (init_job brings the PT weights to a fresh VM); pick = the cell's
    best SFT model by the primary metric."""
    pts, posts, picks = [], [], []
    for seed in SEEDS:
        for cell, sid, share in cells:
            s = sh[sid]
            name = f"pt_{tag}_{cell}_r{seed}"
            lr = lrs.get(sid, PROVISIONAL_LR)
            pts.append({"name": name, "model_name": f"tc_{tag}_{cell}_r{seed}", "after": ["data", "gate"],
                        "relay_every_steps": 2500,
                        "spec": pt_spec(sid, s, corpus_of(share), lr, seed)})
            posts.append({"name": f"post_{tag}_{cell}_r{seed}", "after": name, "init_job": name,
                          "cmd": ["python3", RUN, "post", "--out", "{out}", "--spec", spec_path(sid),
                                  "--pt-weights", f"{C}/{name}/tc_{tag}_{cell}_r{seed}.safetensors",
                                  "--pt-dir", f"{C}/{name}", "--sft-template", f"{REL}/specs/sft.json",
                                  "--seed", str(seed), "--cell", cell, "--share", str(share),
                                  "--panel", f"fineweb={C}/data/panel_fineweb.ids",
                                  "--panel", f"dialogue={C}/data/panel_dialogue.ids",
                                  "--carry5-items", "corpus/eval/carry5_v1.jsonl"],
                          "outputs": ["summary.json"]})
    for cell, sid, share in cells:
        picks.append({"name": f"pick_{tag}_{cell}", "after": [f"post_{tag}_{cell}_r{r}" for r in SEEDS],
                      "cmd": ["python3", RUN, "pick", "--out", "{out}",
                              *[a for r in SEEDS for a in ("--run", f"{C}/post_{tag}_{cell}_r{r}")]],
                      "outputs": ["best.json", "model.safetensors", "model.gguf", "spec.json"]})
    # Longest pretraining first, so the critical path starts at once.
    pts.sort(key=lambda j: -j["spec"]["arch"]["custom"]["layers"])
    gate = {"name": "gate",
            "cmd": ["python3", RUN, "gate", "--out", "{out}", "--jobs", f"tools/colab_chat/tiny_chat_{tag}.json",
                    "--choice", f"{REL}/stage0/lr_choice.json", "--design", f"{REL}/design.json"],
            "outputs": ["gate.json"]}
    est = sum(hours(sh[sid], STEPS) + hours(sh[sid], SFT_STEPS) for _, sid, _ in cells) * len(SEEDS)
    return pts, posts, picks, gate, est


def e_doc(tag, cells, sh, lrs, provisional, shares, title):
    pts, posts, picks, gate, est = run_jobs(tag, cells, sh, lrs, provisional)
    lr_note = ("PROVISIONAL learning rates (1e-3): the gate job fails until stage0/lr_choice.json exists and "
               "matches; rerun make_jobs.py after Stage 0" if provisional else
               "learning rates from stage0/lr_choice.json: " + ", ".join(f"{k} {lrs[k]:g}" for k in sorted(lrs)
                                                                         if k in {c[1] for c in cells}))
    doc = {"_comment": f"tiny-chat {title} ({REL}/PREREGISTRATION.md). {len(pts)} pretraining runs of {STEPS} steps x "
                       f"{TOKENS_PER_STEP} tokens ({STEPS * TOKENS_PER_STEP / 1e6:.2f}M), each then SFT ({SFT_STEPS} "
                       f"steps on human dialogues), two fixed text panels and Carry-5 (post job); one pick per cell. "
                       f"{lr_note}. Estimated {est:.0f} job-hours ({est / VM_SPEEDUP:.0f} L4 VM-hours at "
                       f"{VM_SPEEDUP}x per VM). Home: each PT job's weights (relays + final), each post's summary.json, "
                       f"each pick's best model. Upload vocab_llama3_bpe.gguf with the runner's --upload." + DESIGN_NOTE,
           "max_parallel": 3 if tag == "e5" else 4,
           "jobs": [data_job(shares), gate, *posts, *picks, *pts]}
    return doc


def tables(sh):
    rows = ["| shape | layers | d | heads | d_ff | d_ff/d | non-emb | emb (V x d) | total | emb share | "
            "est. tok/s | PT h | SFT h |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid, s in sh.items():
        rows.append(f"| {sid} | {s['layers']} | {s['d']} | {s['heads']} | {s['d_ff']} | {s['d_ff'] / s['d']:.2f} | "
                    f"{s['nonemb'] / 1e6:.2f}M | {s['emb'] / 1e6:.2f}M | {s['total'] / 1e6:.2f}M | "
                    f"{s['emb'] / s['total']:.0%} | {tok_per_s(s) / 1e3:.1f}k | {hours(s, STEPS):.1f} | "
                    f"{hours(s, SFT_STEPS):.2f} |")
    return "\n".join(rows)


def budget(sh):
    s20 = [k for k in sh if sh[k]["size"] == "20m"]
    s60 = [k for k in sh if sh[k]["size"] == "60m"]
    st0 = lambda ids: sum(hours(sh[s], PROBE_STEPS) + 3 * hours(sh[s], SCREEN_STEPS) for s in ids)
    run = lambda sid: hours(sh[sid], STEPS) + hours(sh[sid], SFT_STEPS)
    lines = [("Stage 0, 20M (probes + screens)", 5 + 15, st0(s20)),
             ("E5 (4 dialogue levels x 3 seeds at 20m_L8)", 12, 4 * len(SEEDS) * run(E5_SHAPE)),
             ("E1 20M (L2, L4, L16, L32 x 3 seeds; L8 from E5)", 12,
              len(SEEDS) * sum(run(s) for s in s20 if s != E5_SHAPE)),
             ("Stage 0, 60M (on go)", 4 + 12, st0(s60)),
             ("E1 60M (4 shapes x 3 seeds, on go)", 12, len(SEEDS) * sum(run(s) for s in s60))]
    out = ["| block | jobs | job-hours | L4 VM-hours |", "|---|---|---|---|"]
    for name, n, h in lines:
        out.append(f"| {name} | {n} | {h:.0f} | {h / VM_SPEEDUP:.0f} |")
    core = sum(h for _, _, h in lines[:3])
    out.append(f"| core (Stage 0 + E5 + E1 20M) | | {core:.0f} | {core / VM_SPEEDUP:.0f} |")
    out.append(f"| with the 60M stage | | {sum(h for _, _, h in lines):.0f} | "
               f"{sum(h for _, _, h in lines) / VM_SPEEDUP:.0f} |")
    return "\n".join(out)


def write(path, doc):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--dialogue-pt-bytes", type=int, default=None,
                    help="P, rendered bytes of the pretraining split (data.ledger.json human_pt_bytes, or 0.85 x the "
                         "default set's text.txt bytes); fixes design.json")
    ap.add_argument("--human-dir", action="append", default=None,
                    help="where built human_<source>_v1/ corpora are (default corpus/data and "
                         "C:/ml_artifacts/human_dialogue) for the estimate when design.json is absent")
    a = ap.parse_args(argv)
    sh = shapes()
    print(tables(sh))
    print()
    print(budget(sh))
    if a.tables:
        return 0
    global SHARES, E1_SHARE, DESIGN_NOTE
    if a.dialogue_pt_bytes is not None:
        design = {**dialogue_levels(a.dialogue_pt_bytes), "source": "given (--dialogue-pt-bytes)", "final": True}
        write(DESIGN, design)
    elif os.path.exists(DESIGN):
        with open(DESIGN, encoding="utf-8") as f:
            design = json.load(f)
    else:
        dirs = a.human_dir or [os.path.join(ROOT, "corpus", "data"), "C:/ml_artifacts/human_dialogue"]
        p, seen = estimate_pt_bytes(dirs)
        missing = [s for s in DEFAULT_SOURCES if s not in seen]
        design = {**dialogue_levels(p), "source": f"estimated from {seen}", "missing_sources": missing,
                  "final": not missing}
        write(DESIGN, design)
    SHARES, E1_SHARE = design["levels"], design["e1_share"]
    DESIGN_NOTE = (f" Dialogue levels {SHARES}% ({design['mode']}), E1 at {E1_SHARE}%"
                   + (" (design.json final)." if design.get("final") else
                      " PROVISIONAL (design.json not final: the gate job refuses; rerun make_jobs.py "
                      "--dialogue-pt-bytes once the whole default human set is measured)."))
    print(f"\ndialogue levels {SHARES} ({design['mode']}), E1 at {E1_SHARE}%"
          + ("" if design.get("final") else f"; PROVISIONAL: missing {design.get('missing_sources')}"))
    lrs, provisional = lr_table()
    os.makedirs(os.path.join(HERE, "specs"), exist_ok=True)
    for sid, s in sh.items():
        spec = pt_spec(sid, s, corpus_of(E1_SHARE), lrs.get(sid, PROVISIONAL_LR), SEEDS[0])
        spec["_shape"] = {k: s[k] for k in ("size", "nonemb", "emb", "total")}
        write(os.path.join(HERE, "specs", f"shape_{sid}.json"), spec)
    write(os.path.join(HERE, "specs", "sft.json"), SFT)
    write(os.path.join(COLAB, "tiny_chat_s0.json"), stage0(sh, "20m"))
    write(os.path.join(COLAB, "tiny_chat_s0_60m.json"), stage0(sh, "60m"))
    e5 = [(f"s{p}", E5_SHAPE, p) for p in SHARES]
    write(os.path.join(COLAB, "tiny_chat_e5.json"), e_doc("e5", e5, sh, lrs, provisional, SHARES, "E5, the data bar"))
    e1 = [(sid.split("_")[1], sid, E1_SHARE) for sid in sh if sh[sid]["size"] == "20m" and sid != E5_SHAPE]
    write(os.path.join(COLAB, "tiny_chat_e1.json"), e_doc("e1", e1, sh, lrs, provisional, [E1_SHARE],
                                                          "E1 at 20M, the shape sweep"))
    e1b = [(sid.split("_")[1], sid, E1_SHARE) for sid in sh if sh[sid]["size"] == "60m"]
    write(os.path.join(COLAB, "tiny_chat_e1_60m.json"), e_doc("e1_60m", e1b, sh, lrs, provisional, [E1_SHARE],
                                                              "E1 at 60M, the shape sweep (on Jonathan's go)"))
    print("\nwrote specs/, tiny_chat_s0.json, tiny_chat_s0_60m.json, tiny_chat_e5.json, tiny_chat_e1.json, "
          "tiny_chat_e1_60m.json" + (" (PROVISIONAL learning rates)" if provisional else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
