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
  tools/colab_chat/tiny_chat_s0_60m.json, tiny_chat_e1_60m.json   the 60M stage (authorised 23 Oct 2026)
  tools/colab_chat/tiny_chat_asha.json                  the exploratory successive-halving arm (amendment 1)
  tools/colab_chat/tiny_chat_pilot.json                 the floor pilot (amendment 3): one 20m_L8 run, seed 99
  tools/colab_chat/tiny_chat_e5_60m.json                E5 at 60M, run only if the floor pilot FAILS (amendment 3)
  tools/colab_chat/tiny_chat_s0_rec.json                the recursion arm's Stage 0 (A3.3): CUDA gate, probes, screens
  tools/colab_chat/tiny_chat_s0_rec_r8.json             R8's screens, on A3.3's R8 rule (recursion/rec_decision.json)
  tools/colab_chat/tiny_chat_e1_rec.json                E1-R: R2, R4, MC7 x 3 seeds at E1's level
  tools/colab_chat/tiny_chat_e1_rec_r8.json             E1-R's R8 x 3 seeds, on the R8 rule
  experiments/tiny_chat_2026-10/recursion/specs/shape_20m_<R2|R4|R8|MC7>.json   the recursion cells' specs
  experiments/tiny_chat_2026-10/asha_plan.json          its sampled configurations, rounds and cost

Every E pretraining job snapshots its weights at the rungs (5/10/25/50% of its steps; jobs_vm.py
snapshot_steps) and a light rung job per snapshot scores them (tiny_chat_run.py rung): the
early-ranking study of amendment 1. The runs themselves always go to full length.

Learning rates come from experiments/tiny_chat_2026-10/stage0/lr_choice.json (written at home by
`tiny_chat_run.py choose` from Stage 0's screens, by the registered rule). Until that file exists the
E files carry the PROVISIONAL rate 1e-3 and their gate job fails on the VM, so no E job can train on
a rate the screen did not choose.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REL = os.path.relpath(HERE, ROOT).replace(os.sep, "/")
COLAB = os.path.join(ROOT, "tools", "colab_chat")
_spec = importlib.util.spec_from_file_location("tiny_chat_run", os.path.join(COLAB, "tiny_chat_run.py"))
tcr = importlib.util.module_from_spec(_spec)  # screen_status: the one definition of FAILED vs DIVERGED
_spec.loader.exec_module(tcr)
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
# Amendment 1 (Jonathan, 23 Oct 2026): share-alike and non-commercial dialogue sources in; aya_en
# (the one source with evidence of ChatGPT text) out; Simple English Wikipedia is not dialogue.
OPTIN_SOURCES = ["topical_chat", "dolly", "quac", "empathetic", "dailydialog", "no_robots"]
DIALOGUE_SOURCES = DEFAULT_SOURCES + OPTIN_SOURCES
RUNG_FRACTIONS = [0.05, 0.10, 0.25, 0.50]   # amendment 1: the early-ranking rungs (100% is the post job)
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
    """P from the built corpora: PT_FRACTION of their rendered text.txt bytes (a provisional
    estimate; the final P is measured by tiny_chat_data.py --measure-only)."""
    tot, seen = 0, []
    for d in dirs:
        for s in DIALOGUE_SOURCES:
            p = os.path.join(d, f"human_{s}_v1", "text.txt")
            if os.path.exists(p) and s not in seen:
                tot += os.path.getsize(p)
                seen.append(s)
    return int(PT_FRACTION * tot), seen
PROVISIONAL_LR = 1e-3
SIZES = {"20m": (20_000_000, [2, 4, 8, 16, 32]), "60m": (60_000_000, [4, 8, 16, 32])}
E5_SHAPE = "20m_L8"
# Amendment 3: the floor pilot, one complete 20m_L8 run at E1's level on a seed outside 1-3, excluded
# from all analysis; its verdict (the A2.5 validity gate) decides whether the primary size stays at 20M.
PILOT_SEED = 99
PILOT_LR = os.path.join(HERE, "stage0", "pilot_lr.json")   # tiny_chat_run.py pilot_lr, when lr_choice lacks 20m_L8
E5_60M_SHAPE = "60m_L8"                                     # E5 at 60M if the pilot FAILS
RUN = "tools/colab_chat/tiny_chat_run.py"

# Throughput model (PREREGISTRATION.md, compute budget): per job on an L4 shared with 1-3 others,
# min(per-layer rate, FLOP ceiling). Anchors from the receipts: arxiv_2302_13971_base (d384 L6, T 512,
# batch 16, 16k vocab) 19,388 tok/s beside pk_base_v1; pk_base_v1 (d768 L12, same T and batch)
# 0.84-1.01 s/step = 8.1-9.8k tok/s. Both give ~0.07 s per layer per 8,192 tokens.
# Used only for the Stage 0 files as launched (attempt 1); amendment 2 replaces it with the probes.
LAYER_TOKENS_PER_S = 8192 / 0.07      # tokens/s for a 1-layer model
FLOP_CEILING = 6.0e12                 # sustained FLOP/s seen on the L4 (pk_base_v1)
MODEL_VM_SPEEDUP = 2.5                # the model's assumption: a VM running 3-4 jobs delivers ~2.5 jobs' worth
# Amendment 2: Stage 0 measured it. Three screens on one L4 together ran at 0.93-1.05 of one job alone
# (the GPU is saturated by one job), and the sharing crashed 22 of 27 screens (each mtstudio process
# caches up to 3/4 of the device: src/device_pool.cuh). GPU jobs now run one at a time per VM.
VM_SPEEDUP = 1.0
GPU_JOBS_PER_VM = 1
LIGHT_POOL_MB = 2048                  # a light (scoring) job beside the training job caches at most this
POST_EVAL_H = 0.30                    # per post job beyond its SFT: panels, Carry-5 with generation, base Carry-5
RUNG_H = 0.08                         # per rung job (panels + base Carry-5); both unmeasured until the first post
STAGE0_DIR = os.path.join(HERE, "stage0", "attempt1")   # Stage 0's first attempt: the probes and the screens


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
    # 6 x (non-embedding + tied head matmul) + attention (Kaplan's 6 L T d for training). A recursed
    # shape (A3.3) applies its stored layers eff_layers times: its applied non-embedding count.
    L = s.get("eff_layers", s["layers"])
    return 6 * (s.get("nonemb_applied", s["nonemb"]) + V * s["d"]) + 6 * L * T * s["d"]


def model_tok_per_s(s):
    return min(LAYER_TOKENS_PER_S / s["layers"], FLOP_CEILING / flops_per_token(s))


def probes():
    """{(size, layers): steady tokens/s alone on the L4} from Stage 0's probe receipts."""
    out = {}
    for d in ("s0", "s0_60m"):
        root = os.path.join(STAGE0_DIR, d)
        if not os.path.isdir(root):
            continue
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n, "summary.json")
            if n.startswith("probe_") and os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("rc") == 0 and s.get("tok_per_s_steady"):
                    size, L = n[len("probe_"):].split("_L")
                    out[(size, int(L))] = s["tok_per_s_steady"]
    return out


PROBES = probes()


def tok_per_s(s):
    """Amendment 2: the probe's rate for a probed shape; between probed depths of the same size, the
    time per token interpolated linearly in log2(layers) (the ASHA-only depths 3, 6, 12); the old
    model only where no probe exists. A recursed shape (A3.3): its own probe once home (stage0/rec/), else its base shape's
    probe scaled by training FLOPs per token (the literature check's extrapolation)."""
    if "recursion" in s:
        if s["sid"] in REC_PROBES:
            return REC_PROBES[s["sid"]]
        b = s["base"]
        return tok_per_s(b) * flops_per_token(b) / flops_per_token(s)
    k = (s["size"], s["layers"])
    if k in PROBES:
        return PROBES[k]
    same = sorted(L for (z, L) in PROBES if z == s["size"])
    lo = [L for L in same if L < s["layers"]]
    hi = [L for L in same if L > s["layers"]]
    if lo and hi:
        a, b = lo[-1], hi[0]
        w = (math.log2(s["layers"]) - math.log2(a)) / (math.log2(b) - math.log2(a))
        t = (1 - w) / PROBES[(s["size"], a)] + w / PROBES[(s["size"], b)]
        return 1 / t
    return model_tok_per_s(s)


def hours(s, steps):
    return steps * TOKENS_PER_STEP / tok_per_s(s) / 3600


def model_hours(s, steps):
    return steps * TOKENS_PER_STEP / model_tok_per_s(s) / 3600


def lr_table():
    p = os.path.join(HERE, "stage0", "lr_choice.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["lr"], False
    return {}, True


def pt_spec(sid, s, corpus, lr, seed, steps=STEPS, warmup=500):
    # eval_every 500 (amendment 1): a validation loss at every rung. mtstudio's evaluation draws its
    # windows from its own fixed generator, so the training data order is unchanged. A recursed shape
    # (A3.3) stores `layers` and batches by its effective depth.
    b, a = batching(s.get("eff_layers", s["layers"]))
    custom = {"d": s["d"], "layers": s["layers"], "heads": s["heads"], "d_ff": s["d_ff"]}
    if "recursion" in s:
        custom["recursion"] = dict(s["recursion"], schedule="fixed", inject="none")
    return {"name": f"tc_{sid}",
            "arch": {"preset": "llama-tiny", "rope_heads": "all", "custom": custom},
            "data": {"corpus": corpus, "vocab": VOCAB, "vocab_cap": V, "T": T, "max_tokens": 0},
            "train": {"steps": steps, "lr": lr, "batch": b, "accum": a, "clip": 1.0, "eval_every": 500,
                      "eval_windows": 32, "checkpoint_every": 500, "gradmap_every": 0, "seed": seed,
                      "optimizer": "adamw",
                      "lr_schedule": {"warmup_steps": warmup, "decay": "cosine", "min_lr_frac": 0.1}},
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
    """The VM builds the 13 human dialogue sources (pinned; --require-pinned refuses any other
    bytes), decontaminates them against the frozen sets and Carry-5, splits and mixes them."""
    src = ",".join(DIALOGUE_SOURCES)
    fetch = f"python3 corpus/fetchers/human_dialogue.py {src} --allow-sa --allow-nc --out-dir {{human}}"
    return {"name": "data",
            "cmd": ["python3", "tools/colab_chat/tiny_chat_data.py", "--out", "{out}", "--vocab", VOCAB,
                    "--fineweb", "corpus/data/fineweb_edu_shard0_v1", "--shares", ",".join(map(str, shares)),
                    "--mix-bytes", str(MIX_BYTES), "--max-epochs", "4", "--T", str(T), "--seed", "1",
                    "--human-fetch", fetch, "--sources", src, "--require-pinned"],
            "outputs": ["data.ledger.json"]}


def rung_steps(steps=STEPS):
    return [int(f * steps) for f in RUNG_FRACTIONS]


PANELS = ["--panel", f"fineweb={C}/data/panel_fineweb.ids", "--panel", f"dialogue={C}/data/panel_dialogue.ids"]
ITEMS = ["--carry5-items", "corpus/eval/carry5_v1.jsonl"]


def corpus_of(share):
    return f"{C}/data/mix_s{share}.ids"


def spec_path(sid):
    if sid.split("_", 1)[-1] in REC_CELLS:
        return f"{REL}/recursion/specs/shape_{sid}.json"
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
    # The file as launched for attempt 1 (amendment 1's commit): the model's estimate, three at a time.
    est = sum(model_hours(sh[s], PROBE_STEPS) for s in ids) + sum(model_hours(sh[s], SCREEN_STEPS) * 3 for s in ids)
    doc = {"_comment": f"tiny-chat Stage 0 ({size}; {REL}/PREREGISTRATION.md): the {E1_SHARE}% dialogue mix (E1 level; design.json) built on the VM, "
                       f"a {PROBE_STEPS}-step throughput probe per shape alone on the GPU, then lr screens "
                       f"{SCREEN_LRS} x {SCREEN_STEPS} steps per shape (seed {SCREEN_SEED}), three at a time. "
                       f"Estimated {est:.1f} job-hours. Brings home one summary.json per job. Afterwards, at home: "
                       f"python {RUN} choose --art ART --out {REL}/stage0/lr_choice.json, commit, rerun make_jobs.py. "
                       f"Upload vocab_llama3_bpe.gguf with the runner's --upload." + DESIGN_NOTE,
           "max_parallel": 3, "jobs": jobs}
    return doc


def failed_screens(size):
    """Stage 0 attempt 1's screens that FAILED (a crash with finite losses; tiny_chat_run.screen_status),
    from the receipts in stage0/attempt1/. Amendment 2: each is rerun unchanged, alone on the GPU."""
    root = os.path.join(STAGE0_DIR, "s0" if size == "20m" else f"s0_{size}")
    out = []
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, n, "summary.json")
        if n.startswith("screen_") and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                if tcr.screen_status(json.load(f)) == "FAILED":
                    out.append(n)
    return out


def stage0_rerun(sh, size):
    """Amendment 2: the failed screens of attempt 1, the same commands (spec, lr, seed 101, 1,500 steps,
    the 35% mix rebuilt from the pinned sources), one GPU job at a time."""
    ids = [k for k in sh if sh[k]["size"] == size]
    name = f"tiny_chat_s0{'' if size == '20m' else '_' + size}_rerun.json"
    again = set(failed_screens(size))
    gate = {"name": "gate", "cmd": ["python3", RUN, "gate", "--out", "{out}", "--stage0", "--jobs",
                                    f"tools/colab_chat/{name}", "--design", f"{REL}/design.json"],
            "outputs": ["gate.json"]}
    jobs = [gate, {**data_job([E1_SHARE]), "after": "gate"}]
    est = 0.0
    for sid in sorted(ids, key=lambda k: -sh[k]["layers"]):
        for k, lr in enumerate(SCREEN_LRS):
            n = f"screen_{sid}_lr{k}"
            if n not in again:
                continue
            est += hours(sh[sid], SCREEN_STEPS)
            jobs.append({"name": n, "after": "data",
                         "cmd": ["python3", RUN, "screen", "--out", "{out}", "--spec", spec_path(sid),
                                 "--corpus", corpus_of(E1_SHARE), "--lr", repr(lr), "--steps", str(SCREEN_STEPS),
                                 "--seed", str(SCREEN_SEED)],
                         "outputs": ["summary.json"]})
    assert len(jobs) - 2 == len(again), (sorted(again), [j["name"] for j in jobs])
    return name, {"_comment": f"tiny-chat Stage 0 RERUN ({size}; {REL}/PREREGISTRATION.md amendment 2): the {len(again)} "
                              f"screens of attempt 1 that FAILED (crashes with finite losses, not divergence; receipts in "
                              f"{REL}/stage0/attempt1/), rerun unchanged and one at a time (max_parallel 1). Measured "
                              f"estimate {est:.1f} job-hours plus the data job. Afterwards, at home: python {RUN} choose "
                              f"--art ATTEMPT1_ART --art THIS_ART --out {REL}/stage0/lr_choice.json, commit, rerun "
                              f"make_jobs.py. Upload vocab_llama3_bpe.gguf with the runner's --upload." + DESIGN_NOTE,
                  "max_parallel": GPU_JOBS_PER_VM, "jobs": jobs}


def run_jobs(tag, cells, sh, lrs, provisional, seeds=SEEDS, gate_extra=(), choice=f"{REL}/stage0/lr_choice.json"):
    """cells: [(cell id, shape id, share)]. PT = training job (relayed, resumable across VMs); post = SFT,
    panels and Carry-5 as a data job (init_job brings the PT weights to a fresh VM); pick = the cell's
    best SFT model by the primary metric; rung = one light job per rung snapshot (amendment 1)."""
    pts, posts, picks, rungs = [], [], [], []
    for seed in seeds:
        for cell, sid, share in cells:
            s = sh[sid]
            name = f"pt_{tag}_{cell}_r{seed}"
            lr = lrs.get(sid, PROVISIONAL_LR)
            pts.append({"name": name, "model_name": f"tc_{tag}_{cell}_r{seed}", "after": ["data", "gate"],
                        "relay_every_steps": 2500, "snapshot_steps": rung_steps(),
                        "spec": pt_spec(sid, s, corpus_of(share), lr, seed)})
            posts.append({"name": f"post_{tag}_{cell}_r{seed}", "after": name, "init_job": name,
                          "cmd": ["python3", RUN, "post", "--out", "{out}", "--spec", spec_path(sid),
                                  "--pt-weights", f"{C}/{name}/tc_{tag}_{cell}_r{seed}.safetensors",
                                  "--pt-dir", f"{C}/{name}", "--sft-template", f"{REL}/specs/sft.json",
                                  "--seed", str(seed), "--cell", cell, "--share", str(share), *PANELS, *ITEMS],
                          "outputs": ["summary.json"]})
            for st in rung_steps():
                rungs.append({"name": f"rung_{tag}_{cell}_r{seed}_s{st}", "after": "data", "light": True,
                              "rung_of": {"job": name, "step": st}, "env": {"MICROTORCH_POOL_MB": LIGHT_POOL_MB},
                              "cmd": ["python3", RUN, "rung", "--out", "{out}", "--spec", spec_path(sid),
                                      "--weights", f"{C}/{name}/rung_{st}/model.safetensors", "--pt-dir",
                                      f"{C}/{name}", "--step", str(st), "--drop-snapshot", "--job", name,
                                      "--cell", cell,
                                      "--seed", str(seed), "--share", str(share), *PANELS, *ITEMS],
                              "outputs": ["rung.json"]})
    for cell, sid, share in cells:
        picks.append({"name": f"pick_{tag}_{cell}", "after": [f"post_{tag}_{cell}_r{r}" for r in seeds],
                      "cmd": ["python3", RUN, "pick", "--out", "{out}",
                              *[a for r in seeds for a in ("--run", f"{C}/post_{tag}_{cell}_r{r}")]],
                      "outputs": ["best.json", "model.safetensors", "model.gguf", "spec.json"]})
    # Longest pretraining first, so the critical path starts at once (a recursed shape by its effective depth).
    pts.sort(key=lambda j: -tcr.effective_layers(j["spec"]["arch"]["custom"]))
    gate = {"name": "gate",
            "cmd": ["python3", RUN, "gate", "--out", "{out}", "--jobs", f"tools/colab_chat/tiny_chat_{tag}.json",
                    "--choice", choice, "--design", f"{REL}/design.json", *gate_extra],
            "outputs": ["gate.json"]}
    est = sum(run_hours(sh[sid]) for _, sid, _ in cells) * len(seeds)
    return pts, posts, picks, rungs, gate, est


def run_hours(s):
    """One E run on the VM (amendment 2): pretraining and SFT at the probe's rate, the post job's
    evaluation and four rung jobs (allowances, unmeasured)."""
    return hours(s, STEPS) + hours(s, SFT_STEPS) + POST_EVAL_H + len(RUNG_FRACTIONS) * RUNG_H


def e_doc(tag, cells, sh, lrs, provisional, shares, title, seeds=SEEDS, gate_extra=(), note="", lr_note=None,
          choice=f"{REL}/stage0/lr_choice.json"):
    pts, posts, picks, rungs, gate, est = run_jobs(tag, cells, sh, lrs, provisional, seeds, gate_extra, choice)
    base_note = ("PROVISIONAL learning rates (1e-3): the gate job fails until stage0/lr_choice.json exists and "
               "matches; rerun make_jobs.py after Stage 0" if provisional else
               "learning rates from stage0/lr_choice.json: " + ", ".join(f"{k} {lrs[k]:g}" for k in sorted(lrs)
                                                                         if k in {c[1] for c in cells}))
    lr_note = lr_note or base_note
    doc = {"_comment": f"tiny-chat {title} ({REL}/PREREGISTRATION.md). {len(pts)} pretraining run{'s' if len(pts) != 1 else ''} of {STEPS} steps x "
                       f"{TOKENS_PER_STEP} tokens ({STEPS * TOKENS_PER_STEP / 1e6:.2f}M), each then SFT ({SFT_STEPS} "
                       f"steps on human dialogues), two fixed text panels and Carry-5 (post job); one pick per cell. "
                       f"{lr_note}. One GPU job at a time (amendment 2: Stage 0 measured no throughput gain from "
                       f"sharing the L4, and sharing crashed its screens); rung jobs beside it in the light slot, their "
                       f"cache capped at {LIGHT_POOL_MB} MB. Estimated {est:.0f} L4 VM-hours (the Stage 0 probes, plus "
                       f"{POST_EVAL_H} h per post job and {RUNG_H} h per rung, unmeasured). Rungs (amendment 1): every "
                       f"PT job snapshots its weights at steps "
                       f"{rung_steps()} and a light rung job scores each snapshot (panels, base-model Carry-5 "
                       f"teacher-forced; rung.json home); runs are never stopped early. Home: each PT job's weights "
                       f"(relays + final), each post's summary.json, each rung's rung.json, each pick's best model. "
                       f"Upload vocab_llama3_bpe.gguf with the runner's --upload." + note + DESIGN_NOTE,
           "max_parallel": GPU_JOBS_PER_VM, "max_parallel_light": 1,
           "jobs": [data_job(shares), gate, *rungs, *posts, *picks, *pts]}
    return doc


# ---------------------------------------------------------------- the exploratory ASHA arm (amendment 1)
ASHA_SEED = 20261023
ASHA_N = 27
ASHA_DEPTHS = [2, 3, 4, 6, 8, 12, 16]   # 20M non-embedding; E1 covers L32, at 3-4x the cost per step
ASHA_LR = (2.5e-4, 4e-3)                # peak lr, log-uniform
ASHA_WARMUP = [0.02, 0.05, 0.10]         # warmup as a fraction of the budget
ASHA_ROUNDS = [("r0", 1000, ASHA_N), ("r1", 3000, 9), ("r2", STEPS, 3)]  # (id, budget in steps, candidates)
ASHA_CAP_VM_HOURS = 20                   # the session's --max-hours: one VM at a time, so a hard cap
ASHA_SCORE_H = 0.08                      # assumed hours per scoring (panels + base Carry-5); measured in the run
PLAN = os.path.join(HERE, "asha_plan.json")


def asha_shapes():
    out = {}
    for L in ASHA_DEPTHS:
        s = shape_for(SIZES["20m"][0], L)
        n = nonemb(L, s["d"], s["d_ff"])
        out[f"20m_L{L}"] = {**s, "size": "20m", "nonemb": n, "emb": V * s["d"], "total": n + V * s["d"]}
    return out


def asha_plan(levels):
    """27 configurations drawn with a fixed seed; the rounds, the promotion rule and the cost."""
    rng = random.Random(ASHA_SEED)
    sh = asha_shapes()
    cfgs = []
    for i in range(ASHA_N):
        L = rng.choice(ASHA_DEPTHS)
        lr = float(f"{math.exp(rng.uniform(math.log(ASHA_LR[0]), math.log(ASHA_LR[1]))):.3g}")
        cfgs.append({"id": f"c{i:02d}", "shape": f"20m_L{L}", "lr": lr, "warmup_frac": rng.choice(ASHA_WARMUP),
                     "share": rng.choice(levels), "seed": 1000 + i})
    per_step = sorted((hours(sh[c["shape"]], 1) for c in cfgs), reverse=True)
    (_, b0, _), (_, b1, k1), (_, b2, k2) = ASHA_ROUNDS
    r0 = sum(p * b0 + ASHA_SCORE_H for p in per_step)
    r1 = sum(p * b1 + ASHA_SCORE_H for p in per_step[:k1])                      # worst case: the dearest 9
    r2 = sum(p * (b2 + SFT_STEPS) + POST_EVAL_H for p in per_step[:k2])         # and the dearest 3, with the post job
    worst = (r0 + r1 + r2) / VM_SPEEDUP
    if worst > ASHA_CAP_VM_HOURS:
        raise SystemExit(f"ASHA worst case {worst:.1f} L4 VM-hours exceeds the cap {ASHA_CAP_VM_HOURS}")
    return {"_comment": "Exploratory successive-halving arm of the tiny-chat programme (PREREGISTRATION.md amendment "
                        "1). Not confirmatory. Written by make_jobs.py; deterministic from ASHA_SEED and the design's "
                        "dialogue levels.",
            "seed": ASHA_SEED,
            "space": {"shape": [f"20m_L{L}" for L in ASHA_DEPTHS], "lr_log_uniform": list(ASHA_LR),
                      "warmup_frac": ASHA_WARMUP, "share": list(levels), "seed": "1000 + index"},
            "rounds": [{"id": r, "budget": b, "keep": k} for r, b, k in ASHA_ROUNDS],
            "schedule": "each round trains from scratch: warmup = warmup_frac x budget, cosine to 0.1x at the budget",
            "promotion": "into each round, the top `keep` of the previous round's candidates by base-model Carry-5 "
                         "forced-choice accuracy with gold history (acc_gold) over the primary metric's families (amendment 2: "
                         "the instruction family out), ties by lower dialogue panel loss, "
                         "then config id; diverged or failed runs rank last and are never promoted",
            "cap_vm_hours": ASHA_CAP_VM_HOURS,
            "cost_job_hours": {"r0": round(r0, 1), "r1_worst": round(r1, 1), "r2_worst": round(r2, 1),
                               "worst_vm_hours": round(worst, 1)},
            "configs": cfgs}


def asha_doc(plan, sh):
    def chk(cid, rid):  # jobs_vm.py promoted_by: the VM's own decision, else the uploaded state file
        return {"files": [f"{C}/asha_promote_{rid}/decisions.json", f"{C}/upload/asha_state.json"],
                "round": rid, "key": cid}

    def cost(c):  # dearest first, so the critical path starts at once
        return -hours(sh[c["shape"]], 1)
    cfg = {c["id"]: c for c in plan["configs"]}
    gate = {"name": "gate", "cmd": ["python3", RUN, "gate", "--out", "{out}", "--stage0", "--need-carry5", "--jobs",
                                    "tools/colab_chat/tiny_chat_asha.json", "--design", f"{REL}/design.json"],
            "outputs": ["gate.json"]}
    jobs = [gate, {**data_job(plan["space"]["share"]), "after": "gate"}]
    names = {}
    for k, rnd in enumerate(plan["rounds"]):
        rid, budget = rnd["id"], rnd["budget"]
        if k > 0:
            jobs.append({"name": f"asha_promote_{rid}", "light": True, "after_terminal": True, "after": names[k - 1],
                         "cmd": ["python3", RUN, "asha_promote", "--out", "{out}", "--plan", f"{REL}/asha_plan.json",
                                 "--round", rid, "--state", "{upload}/asha_state.json"],
                         "outputs": ["decisions.json"]})
        names[k] = []
        last = k == len(plan["rounds"]) - 1
        for c in sorted(plan["configs"], key=cost):
            cid, s = c["id"], sh[c["shape"]]
            name = f"asha_{rid}_{cid}"
            names[k].append(name)
            gating = {"after": "data"} if k == 0 else {"after": f"asha_promote_{rid}", "promoted_by": chk(cid, rid)}
            if not last:
                jobs.append({"name": name, **gating,
                             "cmd": ["python3", RUN, "asha_train", "--out", "{out}", "--spec", spec_path(c["shape"]),
                                     "--corpus", corpus_of(c["share"]), "--budget", str(budget), "--lr", repr(c["lr"]),
                                     "--warmup-frac", repr(c["warmup_frac"]), "--seed", str(c["seed"]), "--share",
                                     str(c["share"]), "--config", cid, "--round", rid, *PANELS, *ITEMS],
                             "outputs": ["rung.json", "run/events.jsonl"]})
            else:  # the finalists: full pretraining runs (relayed, resumable), then the E cells' post job
                jobs.append({"name": name, "model_name": f"tc_asha_{cid}", **gating, "relay_every_steps": 2500,
                             "spec": pt_spec(c["shape"], s, corpus_of(c["share"]), c["lr"], c["seed"], steps=budget,
                                             warmup=max(1, round(c["warmup_frac"] * budget)))})
                jobs.append({"name": f"asha_post_{cid}", "after": name, "init_job": name,
                             "cmd": ["python3", RUN, "post", "--out", "{out}", "--spec", spec_path(c["shape"]),
                                     "--pt-weights", f"{C}/{name}/tc_asha_{cid}.safetensors", "--pt-dir", f"{C}/{name}",
                                     "--sft-template", f"{REL}/specs/sft.json", "--seed", str(c["seed"]), "--cell",
                                     f"asha_{cid}", "--share", str(c["share"]), "--pt-lr", repr(c["lr"]),
                                     *PANELS, *ITEMS],
                             "outputs": ["summary.json"]})
    assert set(cfg) == {c["id"] for c in plan["configs"]}
    cj = plan["cost_job_hours"]
    return {"_comment": f"tiny-chat EXPLORATORY successive-halving arm ({REL}/PREREGISTRATION.md amendment 1, one GPU job at a time by amendment 2; plan "
                        f"{REL}/asha_plan.json). {ASHA_N} random configurations (shape, peak lr, warmup, dialogue share) "
                        f"trained from scratch for {ASHA_ROUNDS[0][1]} steps; the top {ASHA_ROUNDS[1][2]} by base-model "
                        f"Carry-5 acc_gold retrain for {ASHA_ROUNDS[1][1]}; the top {ASHA_ROUNDS[2][2]} for the full "
                        f"{STEPS} steps, then SFT and Carry-5 as the E cells. Promotion runs on the VM "
                        f"(asha_promote_*); candidates not promoted end FAILED with 'not promoted' in their log. Worst "
                        f"case {cj['worst_vm_hours']} L4 VM-hours; cap {ASHA_CAP_VM_HOURS} (the runner's --max-hours). "
                        f"Launch through tools/colab_chat/tiny_chat_asha.py drive, which keeps ART/asha_state.json (every "
                        f"result and decision home so far) current for the runner's --upload asha_state.json=..., so a "
                        f"replaced VM can still promote. Upload vocab_llama3_bpe.gguf too." + DESIGN_NOTE,
            "max_parallel": GPU_JOBS_PER_VM, "max_parallel_light": 1, "jobs": jobs}


# ---------------------------------------------------------------- amendment 3: the floor pilot
def pilot_rate(lrs):
    """(lr, where from, provisional): Stage 0's choice for 20m_L8 when lr_choice.json holds it, else
    the pilot's own rate (stage0/pilot_lr.json: section 4's rule on 20m_L8 alone, or the registered
    fallback), else PROVISIONAL (the pilot's gate refuses on the VM)."""
    if E5_SHAPE in lrs:
        return lrs[E5_SHAPE], "stage0/lr_choice.json", False
    if os.path.exists(PILOT_LR):
        with open(PILOT_LR, encoding="utf-8") as f:
            pl = json.load(f)
        assert pl["shape"] == E5_SHAPE, pl
        return pl["lr"], f"stage0/pilot_lr.json ({pl['source']})", False
    return PROVISIONAL_LR, "PROVISIONAL", True


PILOT_GATE = ["--pilot-verdict", f"{REL}/pilot/verdict.json"]


def pilot_doc(sh, lrs):
    lr, src, prov = pilot_rate(lrs)
    assert PILOT_SEED not in SEEDS and PILOT_SEED != SCREEN_SEED
    note = (f" FLOOR PILOT (amendment 3): seed {PILOT_SEED}, declared before it runs and excluded from every analysis "
            f"(analyze.py reads only the verdict's gate quantities). Afterwards, at home: python "
            f"{REL}/analyze.py --art ART --pilot-verdict {REL}/pilot/verdict.json, commit. PASS: E5 and E1 at 20M "
            f"launch as registered; FAIL: the primary size moves to 60M (tiny_chat_e5_60m.json, beside E1 at 60M).")
    lr_note = ("PROVISIONAL learning rate (1e-3): the gate job fails until stage0/lr_choice.json holds 20m_L8 or "
               "stage0/pilot_lr.json exists; rerun make_jobs.py then" if prov else f"learning rate {lr:g} from {src}")
    return e_doc("pilot", [(f"s{E1_SHARE}", E5_SHAPE, E1_SHARE)], sh, {E5_SHAPE: lr}, prov, [E1_SHARE],
                 "floor pilot", seeds=[PILOT_SEED], gate_extra=["--pilot-lr", f"{REL}/stage0/pilot_lr.json"],
                 note=note, lr_note=lr_note)


def e5_60m_cells():
    """E5 at 60M (amendment 3, only if the pilot FAILS): the levels other than E1's at 60m_L8; its
    E1-level cell is E1 at 60M's L8 cell (the same runs), as E1's L8 cell at 20M is E5's."""
    return [(f"s{p}", E5_60M_SHAPE, p) for p in SHARES if p != E1_SHARE]


def budget_a3_rows(sh):
    """Amendment 3: the pilot, E5 at 60M, and the two branches after the pilot (VM-hours)."""
    rows = budget_rows(sh)
    tot = sum(h for _, _, h in rows)
    e5_20, e1_20, asha = rows[2][2], rows[3][2], rows[-1][2]
    pilot = DATA_H + run_hours(sh[E5_SHAPE])
    e5_60 = DATA_H + len(e5_60m_cells()) * len(SEEDS) * run_hours(sh[E5_60M_SHAPE])
    return {"pilot": pilot, "e5_60m": e5_60, "pass": tot + pilot, "fail": tot + pilot - e5_20 - e1_20 - asha + e5_60}


def budget_a3(sh):
    b, u = budget_a3_rows(sh), UNITS_PER_L4_HOUR
    rows = [("floor pilot (one 20m_L8 run at E1's level, seed 99)", b["pilot"]),
            (f"E5 at 60M ({len(e5_60m_cells())} levels x 3 seeds at {E5_60M_SHAPE}; FAIL branch only)", b["e5_60m"]),
            ("PASS branch: amendment 2's everything + the pilot", b["pass"]),
            ("FAIL branch: + the pilot + E5 at 60M - E5 and E1 at 20M - the held arm", b["fail"])]
    return chr(10).join(["| amendment 3 | L4 VM-hours | compute units |", "|---|---|---|"]
                   + [f"| {n} | {h:.1f} | {h * u:.0f} |" for n, h in rows])


# ---------------------------------------------------------------- amendment 3: the recursion arm (A3.3)
REC_BASE = "20m_L4"                   # every cell recurses E1's 20m_L4 shape (d 640, 10 heads, d_ff 1,760)
REC_CELLS = {                         # cell: ((prelude, block, repeats, coda), plain twin in E1)
    "R2": ((0, 4, 2, 0), "L8"),
    "R4": ((0, 4, 4, 0), "L16"),
    "R8": ((0, 4, 8, 0), "L32"),       # conditional: A3.3's R8 rule
    "MC7": ((1, 2, 7, 1), "L16"),      # Middle-Cycle topology; also compared with R4
}
REC_R8 = "R8"
REC_SCREEN_LRS = list(tcr.REC_SCREEN_LRS)   # {2.5e-4, 5e-4, 1e-3, 2e-3}: shared weights sum r gradients
REC_SPECS = os.path.join(HERE, "recursion", "specs")
REC_CHOICE = f"{REL}/stage0/lr_choice_rec.json"        # tiny_chat_run.py choose_rec, at home
REC_DECISION = f"{REL}/recursion/rec_decision.json"    # analyze.py --rec-decision: A3.3's R8 rule
REC_CUDA_GATE = f"{REL}/recursion/cuda_gate.json"      # Stage 0's cuda_gate.json, copied home and committed
REC_STAGE0_DIR = os.path.join(HERE, "stage0", "rec")   # the arm's probe receipts once home (probe_20m_<cell>/)
CUDA_GATE_H = 0.30                    # build the validation tests and run every device leg (unmeasured)


def rec_probes():
    out = {}
    for n in sorted(os.listdir(REC_STAGE0_DIR)) if os.path.isdir(REC_STAGE0_DIR) else []:
        p = os.path.join(REC_STAGE0_DIR, n, "summary.json")
        if n.startswith("probe_") and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("rc") == 0 and s.get("tok_per_s_steady"):
                out[n[len("probe_"):]] = s["tok_per_s_steady"]
    return out


REC_PROBES = rec_probes()


def rec_shapes(sh):
    """{sid: shape} for the recursion cells: the base shape's stored layers, applied P + B x r + C
    times; nonemb counts stored parameters (the matching), nonemb_applied the applied ones (FLOPs)."""
    b = sh[REC_BASE]
    per_layer = 4 * b["d"] ** 2 + 3 * b["d"] * b["d_ff"] + 2 * b["d"]
    out = {}
    for cell, ((P, B, R, Cd), twin) in REC_CELLS.items():
        assert P + B + Cd == b["layers"], cell
        eff = P + B * R + Cd
        out[f"20m_{cell}"] = {**b, "sid": f"20m_{cell}", "cell": cell, "twin": twin, "base": b,
                              "recursion": {"prelude": P, "block": B, "repeats": R, "coda": Cd},
                              "eff_layers": eff, "nonemb_applied": eff * per_layer + b["d"]}
    return out


def rec_lr_table():
    """(lr by sid, dropped sids, provisional): stage0/lr_choice_rec.json once choose_rec has run."""
    p = os.path.join(ROOT, REC_CHOICE)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d["lr"], sorted(d.get("dropped", {})), False
    return {}, [], True


def rec_spec(sid, s, lr):
    spec = pt_spec(sid, s, corpus_of(E1_SHARE), lr, SEEDS[0])
    spec["_shape"] = {**{k: s[k] for k in ("size", "nonemb", "emb", "total")},
                      "recursion": dict(s["recursion"]), "effective_layers": s["eff_layers"],
                      "stored_layers": s["layers"]}
    return spec


def s0_rec_doc(rs, r8=False):
    """The arm's Stage 0 (A3.3), one GPU job at a time (amendment 2). cuda_gate builds and runs every
    device leg of tools/colab_cuda_validate.sh on the VM's L4 (test_recursion's device leg among them,
    never run on a GPU before) and test_recursion again with the kernels mtstudio trains with; it runs
    first and alone. The gate refuses unless the floor pilot PASSED (A3.2: the arm runs at 20M only on
    PASS; R8's file also needs the R8 rule). Probes (300 steps, every cell) then screens (1,500 steps,
    seed 101, four rates, dearest first) wait for the data job and the CUDA gate, and each refuses on
    its own unless the gate passed on its VM (--cuda-gate; a fresh VM reruns the gate first) and
    rebuilds a corpus a fresh VM lacks (--data-job). R8's screens are the separate file."""
    name = "tiny_chat_s0_rec_r8.json" if r8 else "tiny_chat_s0_rec.json"
    gate = {"name": "gate", "cmd": ["python3", RUN, "gate", "--out", "{out}", "--stage0", "--jobs",
                                    f"tools/colab_chat/{name}", "--design", f"{REL}/design.json", *PILOT_GATE,
                                    "--need-pilot", "PASS", *(["--need-rec-r8", REC_DECISION] if r8 else [])],
            "outputs": ["gate.json"]}
    cuda = {"name": "cuda_gate", "cmd": ["python3", RUN, "cuda_gate", "--out", "{out}"],
            "outputs": ["cuda_gate.json", "cuda_gate.log"]}
    jobs = [cuda, gate, {**data_job([E1_SHARE]), "after": "gate"}]
    after = ["data", "cuda_gate"]
    pre = ["--cuda-gate", f"{C}/cuda_gate", "--data-job", f"tools/colab_chat/{name}"]
    probed = [] if r8 else list(rs)
    screened = [f"20m_{REC_R8}"] if r8 else [k for k in rs if rs[k]["cell"] != REC_R8]
    for sid in probed:
        jobs.append({"name": f"probe_{sid}", "after": after,
                     "cmd": ["python3", RUN, "probe", "--out", "{out}", "--spec", spec_path(sid),
                             "--corpus", corpus_of(E1_SHARE), "--steps", str(PROBE_STEPS), *pre],
                     "outputs": ["summary.json"]})
    for sid in sorted(screened, key=lambda k: -flops_per_token(rs[k])):  # dearest first
        for k, lr in enumerate(REC_SCREEN_LRS):
            jobs.append({"name": f"screen_{sid}_lr{k}", "after": after,
                         "cmd": ["python3", RUN, "screen", "--out", "{out}", "--spec", spec_path(sid),
                                 "--corpus", corpus_of(E1_SHARE), "--lr", repr(lr), "--steps", str(SCREEN_STEPS),
                                 "--seed", str(SCREEN_SEED), *pre],
                         "outputs": ["summary.json"]})
    est = s0_rec_hours(rs, r8)
    what = ("R8's screens (A3.3's R8 rule: recursion/rec_decision.json run_r8 true)" if r8 else
            f"the CUDA gate, a {PROBE_STEPS}-step probe per cell ({', '.join(rs)}), then the screens of "
            f"{', '.join(k for k in rs if rs[k]['cell'] != REC_R8)} (R8's wait for the R8 rule)")
    doc = {"_comment": f"tiny-chat recursion arm, Stage 0 ({REL}/PREREGISTRATION.md amendment 3, A3.3): {what}; screens at "
                       f"{REC_SCREEN_LRS} x {SCREEN_STEPS} steps, seed {SCREEN_SEED}, on the {E1_SHARE}% mix. One GPU job "
                       f"at a time. The gate needs the floor pilot's PASS; probes and screens need the CUDA gate passed "
                       f"on their VM. Estimated {est:.1f} L4 VM-hours (probes scaled from 20m_L4's by FLOPs per token "
                       f"until the arm's own come home). Afterwards, at home: copy ART/cuda_gate/cuda_gate.json to "
                       f"{REC_CUDA_GATE} and the probe folders to {REL}/stage0/rec/, then python {RUN} choose_rec "
                       f"--art ART --out {REC_CHOICE}, commit, rerun make_jobs.py. Upload vocab_llama3_bpe.gguf with "
                       f"the runner's --upload." + DESIGN_NOTE,
           "max_parallel": GPU_JOBS_PER_VM, "jobs": jobs}
    return name, doc


def s0_rec_hours(rs, r8=False):
    if r8:
        return CUDA_GATE_H + DATA_H + len(REC_SCREEN_LRS) * hours(rs[f"20m_{REC_R8}"], SCREEN_STEPS)
    return (CUDA_GATE_H + DATA_H + sum(hours(s, PROBE_STEPS) for s in rs.values())
            + sum(len(REC_SCREEN_LRS) * hours(s, SCREEN_STEPS) for s in rs.values() if s["cell"] != REC_R8))


REC_NOTE = (" Recursion arm (A3.3): the gate refuses unless the floor pilot PASSED, the arm's CUDA gate passed "
            f"({REC_CUDA_GATE}) and every pretraining job's lr is the cell's choice in {REC_CHOICE} (choose_rec: "
            "section 4's rule over the four-rate grid). Shared-weight cells are matched on non-embedding parameters "
            "only; they spend more FLOPs per token than their twins.")


def e1_rec_docs(sh, rs):
    """tiny_chat_e1_rec.json (R2, R4, MC7 x 3 seeds; a dropped cell is left out and reported) and
    tiny_chat_e1_rec_r8.json (R8 x 3 seeds, gated on A3.3's R8 rule as well)."""
    lrs, dropped, prov = rec_lr_table()
    allsh = {**sh, **rs}
    base_gate = [*PILOT_GATE, "--need-pilot", "PASS", "--need-cuda-gate", REC_CUDA_GATE]
    cells = [(rs[k]["cell"], k, E1_SHARE) for k in rs if rs[k]["cell"] != REC_R8 and k not in dropped]

    def lr_note(cs):
        if prov:
            return (f"PROVISIONAL learning rates (1e-3): the gate job fails until {REC_CHOICE} exists and matches; "
                    "rerun make_jobs.py after the arm's Stage 0")
        return f"learning rates from {REC_CHOICE}: " + ", ".join(f"{k} {lrs[k]:g}" for _, k, _ in cs if k in lrs)
    drop_note = f" Dropped (all screens diverged): {dropped}." if dropped else ""
    main = e_doc("e1_rec", cells, allsh, lrs, prov, [E1_SHARE], "E1-R, the recursion arm at 20M (amendment 3, A3.3)",
                 gate_extra=base_gate, note=REC_NOTE + drop_note, lr_note=lr_note(cells), choice=REC_CHOICE)
    r8 = [(REC_R8, f"20m_{REC_R8}", E1_SHARE)] if f"20m_{REC_R8}" not in dropped else []
    r8doc = e_doc("e1_rec_r8", r8, allsh, lrs, prov, [E1_SHARE], "E1-R's R8 cell, on A3.3's R8 rule",
                  gate_extra=[*base_gate, "--need-rec-r8", REC_DECISION],
                  note=REC_NOTE + f" R8 runs only if R2 and R4 pass the validity gate on their three seeds and the "
                                  f"mean over seeds of Y(R4) - Y(R2) is at least 1 point ({REC_DECISION}, written by "
                                  f"analyze.py --rec-decision and committed before any R8 job).",
                  lr_note=lr_note(r8), choice=REC_CHOICE)
    return main, r8doc


def budget_rec_rows(sh):
    """A3.3's arm in L4 VM-hours: (row, hours). Runs at the arm's probe rates once home, else 20m_L4's
    probe scaled by FLOPs per token."""
    rs = rec_shapes(sh)
    run = lambda c: len(SEEDS) * run_hours(rs[f"20m_{c}"])  # noqa: E731
    rows = [("Stage 0-R: CUDA gate, data, 4 probes, 12 screens (R2, R4, MC7)", s0_rec_hours(rs)),
            ("E1-R: R2, R4, MC7 x 3 seeds", DATA_H + run("R2") + run("R4") + run("MC7")),
            ("R8 (rule): CUDA gate, data, 4 screens", s0_rec_hours(rs, r8=True)),
            ("R8 (rule): 3 seeds", DATA_H + run("R8"))]
    return rows


def budget_rec(sh):
    rows, u = budget_rec_rows(sh), UNITS_PER_L4_HOUR
    rs = rec_shapes(sh)
    src = "the arm's probes" if len(REC_PROBES) == len(rs) else "20m_L4's probe scaled by FLOPs per token"
    runs_all = rows[1][1] + rows[3][1]
    out = ["| recursion arm (A3.3) | L4 VM-hours | compute units |", "|---|---|---|"]
    out += [f"| {n} | {h:.1f} | {h * u:.0f} |" for n, h in rows]
    out.append(f"| without R8 | {rows[0][1] + rows[1][1]:.1f} | {(rows[0][1] + rows[1][1]) * u:.0f} |")
    out.append(f"| with R8 | {sum(h for _, h in rows):.1f} | {sum(h for _, h in rows) * u:.0f} |")
    out.append(f"\nrates from {src}: " + ", ".join(
        f"{rs[k]['cell']} {tok_per_s(rs[k]) / 1e3:.1f}k tok/s ({flops_per_token(rs[k]) / 1e9:.2f} GFLOP/token)"
        for k in rs) + f". A3.3: the runs, R8 included, {runs_all:.1f} VM-hours, "
        + ("ABOVE" if runs_all > 80 else "within") + " the 80 at which Jonathan decides before they launch")
    return chr(10).join(out)


def tables(sh):
    rows = ["| shape | layers | d | heads | d_ff | d_ff/d | non-emb | emb (V x d) | total | emb share | "
            "tok/s (probe) | PT h | SFT h |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid, s in sh.items():
        rows.append(f"| {sid} | {s['layers']} | {s['d']} | {s['heads']} | {s['d_ff']} | {s['d_ff'] / s['d']:.2f} | "
                    f"{s['nonemb'] / 1e6:.2f}M | {s['emb'] / 1e6:.2f}M | {s['total'] / 1e6:.2f}M | "
                    f"{s['emb'] / s['total']:.0%} | {tok_per_s(s) / 1e3:.1f}k | {hours(s, STEPS):.1f} | "
                    f"{hours(s, SFT_STEPS):.2f} |")
    return "\n".join(rows)


# Stage 0 attempt 1 as it ran (runner.log in stage0/attempt1/): session up to session stopped.
STAGE0_ATTEMPT1_VM_H = {"20m": (9 * 60 + 56 + 13 / 60 - (7 * 60 + 10 + 53 / 60)) / 60,   # 07:10:53-09:56:13
                        "60m": (8 * 60 + 6 + 8 / 60 - (7 * 60 + 10 + 55 / 60)) / 60}     # 07:10:55-08:06:08
DATA_H = 0.25                         # gate + data job on a fresh VM (attempt 1: 13-19 minutes)
UNITS_PER_L4_HOUR = 4.8               # Colab compute units per L4 hour (section 10's figure)
CORE_RULE_VM_H = 80                   # section 8 rule 5


def budget_rows(sh):
    """Amendment 2's budget from the probes: (block, runs, VM-hours), one GPU job at a time."""
    s20 = [k for k in sh if sh[k]["size"] == "20m"]
    s60 = [k for k in sh if sh[k]["size"] == "60m"]
    rerun = lambda size: DATA_H + sum(hours(sh[n[len("screen_"):].rsplit("_lr", 1)[0]], SCREEN_STEPS)
                                      for n in failed_screens(size))
    plan_p = PLAN if os.path.exists(PLAN) else None
    asha = None
    if plan_p:
        with open(plan_p, encoding="utf-8") as f:
            asha = json.load(f)["cost_job_hours"]
    return [("Stage 0, 20M: attempt 1 (as run)", 20, STAGE0_ATTEMPT1_VM_H["20m"]),
            (f"Stage 0, 20M: rerun of the {len(failed_screens('20m'))} failed screens", len(failed_screens("20m")),
             rerun("20m")),
            ("E5 (4 dialogue levels x 3 seeds at 20m_L8)", 12, DATA_H + 4 * len(SEEDS) * run_hours(sh[E5_SHAPE])),
            ("E1 20M (L2, L4, L16, L32 x 3 seeds; L8 from E5)", 12,
             DATA_H + len(SEEDS) * sum(run_hours(sh[s]) for s in s20 if s != E5_SHAPE)),
            ("Stage 0, 60M: attempt 1 (as run)", 16, STAGE0_ATTEMPT1_VM_H["60m"]),
            (f"Stage 0, 60M: rerun of the {len(failed_screens('60m'))} failed screens", len(failed_screens("60m")),
             rerun("60m")),
            ("E1 60M (4 shapes x 3 seeds)", 12, DATA_H + len(SEEDS) * sum(run_hours(sh[s]) for s in s60)),
            ("successive halving (exploratory; worst case, cap 20)", 27,
             (asha["worst_vm_hours"] + DATA_H) if asha else float("nan"))]


def budget(sh):
    rows = budget_rows(sh)
    out = ["| block | runs | L4 VM-hours | compute units |", "|---|---|---|---|"]
    for name, n, h in rows:
        out.append(f"| {name} | {n} | {h:.1f} | {h * UNITS_PER_L4_HOUR:.0f} |")
    core = sum(h for _, _, h in rows[:4])
    tot = sum(h for _, _, h in rows)
    out.append(f"| **core (Stage 0 20M + E5 + E1 20M; rule 5)** | | **{core:.1f}** | **{core * UNITS_PER_L4_HOUR:.0f}** |")
    out.append(f"| everything (with the 60M stage and the arm) | | {tot:.1f} | {tot * UNITS_PER_L4_HOUR:.0f} |")
    out.append(f"\nsection 8 rule 5: the core is {core:.1f} L4 VM-hours, "
               + ("ABOVE" if core > CORE_RULE_VM_H else "below") + f" {CORE_RULE_VM_H}; per-token time L32/L2 at 20M "
               f"{tok_per_s(sh['20m_L2']) / tok_per_s(sh['20m_L32']):.2f}x (section 6 predicted at least 3x)")
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
    ap.add_argument("--design-source", default="given (--dialogue-pt-bytes)",
                    help="how P was measured, recorded in design.json")
    ap.add_argument("--human-dir", action="append", default=None,
                    help="where built human_<source>_v1/ corpora are (default corpus/data and "
                         "C:/ml_artifacts/human_dialogue) for the estimate when design.json is absent")
    a = ap.parse_args(argv)
    sh = shapes()
    print(tables(sh))
    print()
    if a.tables:
        print(budget(sh))
        print()
        print(budget_a3(sh))
        print()
        print(budget_rec(sh))
        return 0
    global SHARES, E1_SHARE, DESIGN_NOTE
    if a.dialogue_pt_bytes is not None:
        design = {**dialogue_levels(a.dialogue_pt_bytes), "source": a.design_source, "sources": DIALOGUE_SOURCES,
                  "final": True}
        write(DESIGN, design)
    elif os.path.exists(DESIGN):
        with open(DESIGN, encoding="utf-8") as f:
            design = json.load(f)
    else:
        dirs = a.human_dir or [os.path.join(ROOT, "corpus", "data"), "C:/ml_artifacts/human_dialogue"]
        p, seen = estimate_pt_bytes(dirs)
        missing = [s for s in DIALOGUE_SOURCES if s not in seen]
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
    for size in ("20m", "60m"):  # amendment 2: the failed screens of attempt 1, one at a time
        if failed_screens(size):
            n, doc = stage0_rerun(sh, size)
            write(os.path.join(COLAB, n), doc)
    need_pass = [*PILOT_GATE, "--need-pilot", "PASS"]  # amendment 3: the floor pilot first
    pass_note = " Amendment 3: the gate refuses unless the floor pilot PASSED (pilot/verdict.json)."
    e5 = [(f"s{p}", E5_SHAPE, p) for p in SHARES]
    write(os.path.join(COLAB, "tiny_chat_e5.json"), e_doc("e5", e5, sh, lrs, provisional, SHARES, "E5, the data bar",
                                                          gate_extra=need_pass, note=pass_note))
    e1 = [(sid.split("_")[1], sid, E1_SHARE) for sid in sh if sh[sid]["size"] == "20m" and sid != E5_SHAPE]
    write(os.path.join(COLAB, "tiny_chat_e1.json"), e_doc("e1", e1, sh, lrs, provisional, [E1_SHARE],
                                                          "E1 at 20M, the shape sweep", gate_extra=need_pass,
                                                          note=pass_note))
    write(os.path.join(COLAB, "tiny_chat_pilot.json"), pilot_doc(sh, lrs))
    write(os.path.join(COLAB, "tiny_chat_e5_60m.json"),
          e_doc("e5_60m", e5_60m_cells(), sh, lrs, provisional, [p for p in SHARES if p != E1_SHARE],
                "E5 at 60M, the data bar if the floor pilot FAILS (amendment 3)",
                gate_extra=[*PILOT_GATE, "--need-pilot", "FAIL"],
                note=f" Amendment 3: the gate refuses unless the floor pilot FAILED (pilot/verdict.json). The "
                     f"{E1_SHARE}% cell is E1 at 60M's L8 cell (tiny_chat_e1_60m.json, the same runs)."))
    e1b = [(sid.split("_")[1], sid, E1_SHARE) for sid in sh if sh[sid]["size"] == "60m"]
    write(os.path.join(COLAB, "tiny_chat_e1_60m.json"), e_doc("e1_60m", e1b, sh, lrs, provisional, [E1_SHARE],
                                                              "E1 at 60M, the shape sweep (authorised 23 Oct 2026)"))
    ash = asha_shapes()
    for sid, s in ash.items():
        if sid not in sh:  # the ASHA-only depths; lr and seed are set per configuration
            spec = pt_spec(sid, s, corpus_of(E1_SHARE), PROVISIONAL_LR, SEEDS[0])
            spec["_shape"] = {k: s[k] for k in ("size", "nonemb", "emb", "total")}
            write(os.path.join(HERE, "specs", f"shape_{sid}.json"), spec)
    plan = asha_plan(SHARES)
    write(PLAN, plan)
    write(os.path.join(COLAB, "tiny_chat_asha.json"), asha_doc(plan, ash))
    print(f"ASHA arm: worst case {plan['cost_job_hours']['worst_vm_hours']} L4 VM-hours (cap {ASHA_CAP_VM_HOURS})")
    # Amendment 3, A3.3: the recursion arm (paperkiln 567a441d holds arch.custom.recursion).
    rs = rec_shapes(sh)
    rlrs, _, _ = rec_lr_table()
    os.makedirs(REC_SPECS, exist_ok=True)
    for sid, s in rs.items():
        write(os.path.join(REC_SPECS, f"shape_{sid}.json"), rec_spec(sid, s, rlrs.get(sid, PROVISIONAL_LR)))
    for r8 in (False, True):
        n, doc = s0_rec_doc(rs, r8)
        write(os.path.join(COLAB, n), doc)
    e1r, e1r8 = e1_rec_docs(sh, rs)
    write(os.path.join(COLAB, "tiny_chat_e1_rec.json"), e1r)
    write(os.path.join(COLAB, "tiny_chat_e1_rec_r8.json"), e1r8)
    print("\nwrote specs/, asha_plan.json, tiny_chat_s0.json, tiny_chat_s0_60m.json, tiny_chat_e5.json, "
          "tiny_chat_e1.json, tiny_chat_e1_60m.json, tiny_chat_asha.json, the Stage 0 rerun files, "
          "tiny_chat_pilot.json, tiny_chat_e5_60m.json, recursion/specs/, tiny_chat_s0_rec.json, "
          "tiny_chat_s0_rec_r8.json, tiny_chat_e1_rec.json, tiny_chat_e1_rec_r8.json"
          + (" (PROVISIONAL learning rates)" if provisional else ""))
    print()
    print(budget(sh))
    print()
    print(budget_a3(sh))
    print()
    print(budget_rec(sh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
