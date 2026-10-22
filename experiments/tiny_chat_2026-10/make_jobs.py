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
import json
import math
import os
import random

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


def pt_spec(sid, s, corpus, lr, seed, steps=STEPS, warmup=500):
    # eval_every 500 (amendment 1): a validation loss at every rung. mtstudio's evaluation draws its
    # windows from its own fixed generator, so the training data order is unchanged.
    b, a = batching(s["layers"])
    return {"name": f"tc_{sid}",
            "arch": {"preset": "llama-tiny", "rope_heads": "all",
                     "custom": {"d": s["d"], "layers": s["layers"], "heads": s["heads"], "d_ff": s["d_ff"]}},
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
    best SFT model by the primary metric; rung = one light job per rung snapshot (amendment 1)."""
    pts, posts, picks, rungs = [], [], [], []
    for seed in SEEDS:
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
                              "rung_of": {"job": name, "step": st},
                              "cmd": ["python3", RUN, "rung", "--out", "{out}", "--spec", spec_path(sid),
                                      "--weights", f"{C}/{name}/rung_{st}/model.safetensors", "--pt-dir",
                                      f"{C}/{name}", "--step", str(st), "--drop-snapshot", "--job", name,
                                      "--cell", cell,
                                      "--seed", str(seed), "--share", str(share), *PANELS, *ITEMS],
                              "outputs": ["rung.json"]})
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
    return pts, posts, picks, rungs, gate, est


def e_doc(tag, cells, sh, lrs, provisional, shares, title):
    pts, posts, picks, rungs, gate, est = run_jobs(tag, cells, sh, lrs, provisional)
    lr_note = ("PROVISIONAL learning rates (1e-3): the gate job fails until stage0/lr_choice.json exists and "
               "matches; rerun make_jobs.py after Stage 0" if provisional else
               "learning rates from stage0/lr_choice.json: " + ", ".join(f"{k} {lrs[k]:g}" for k in sorted(lrs)
                                                                         if k in {c[1] for c in cells}))
    doc = {"_comment": f"tiny-chat {title} ({REL}/PREREGISTRATION.md). {len(pts)} pretraining runs of {STEPS} steps x "
                       f"{TOKENS_PER_STEP} tokens ({STEPS * TOKENS_PER_STEP / 1e6:.2f}M), each then SFT ({SFT_STEPS} "
                       f"steps on human dialogues), two fixed text panels and Carry-5 (post job); one pick per cell. "
                       f"{lr_note}. Estimated {est:.0f} job-hours ({est / VM_SPEEDUP:.0f} L4 VM-hours at "
                       f"{VM_SPEEDUP}x per VM). Rungs (amendment 1): every PT job snapshots its weights at steps "
                       f"{rung_steps()} and a light rung job scores each snapshot (panels, base-model Carry-5 "
                       f"teacher-forced; rung.json home); runs are never stopped early. Home: each PT job's weights "
                       f"(relays + final), each post's summary.json, each rung's rung.json, each pick's best model. "
                       f"Upload vocab_llama3_bpe.gguf with the runner's --upload." + DESIGN_NOTE,
           "max_parallel": 3 if tag == "e5" else 4, "max_parallel_light": 1,
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
    r2 = sum(p * (b2 + SFT_STEPS) + 2 * ASHA_SCORE_H for p in per_step[:k2])    # and the dearest 3
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
                         "forced-choice accuracy with gold history (acc_gold), ties by lower dialogue panel loss, "
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
    return {"_comment": f"tiny-chat EXPLORATORY successive-halving arm ({REL}/PREREGISTRATION.md amendment 1; plan "
                        f"{REL}/asha_plan.json). {ASHA_N} random configurations (shape, peak lr, warmup, dialogue share) "
                        f"trained from scratch for {ASHA_ROUNDS[0][1]} steps; the top {ASHA_ROUNDS[1][2]} by base-model "
                        f"Carry-5 acc_gold retrain for {ASHA_ROUNDS[1][1]}; the top {ASHA_ROUNDS[2][2]} for the full "
                        f"{STEPS} steps, then SFT and Carry-5 as the E cells. Promotion runs on the VM "
                        f"(asha_promote_*); candidates not promoted end FAILED with 'not promoted' in their log. Worst "
                        f"case {cj['worst_vm_hours']} L4 VM-hours; cap {ASHA_CAP_VM_HOURS} (the runner's --max-hours). "
                        f"Launch through tools/colab_chat/tiny_chat_asha.py drive, which keeps ART/asha_state.json (every "
                        f"result and decision home so far) current for the runner's --upload asha_state.json=..., so a "
                        f"replaced VM can still promote. Upload vocab_llama3_bpe.gguf too." + DESIGN_NOTE,
            "max_parallel": 4, "max_parallel_light": 1, "jobs": jobs}


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
    ap.add_argument("--design-source", default="given (--dialogue-pt-bytes)",
                    help="how P was measured, recorded in design.json")
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
    e5 = [(f"s{p}", E5_SHAPE, p) for p in SHARES]
    write(os.path.join(COLAB, "tiny_chat_e5.json"), e_doc("e5", e5, sh, lrs, provisional, SHARES, "E5, the data bar"))
    e1 = [(sid.split("_")[1], sid, E1_SHARE) for sid in sh if sh[sid]["size"] == "20m" and sid != E5_SHAPE]
    write(os.path.join(COLAB, "tiny_chat_e1.json"), e_doc("e1", e1, sh, lrs, provisional, [E1_SHARE],
                                                          "E1 at 20M, the shape sweep"))
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
    print("\nwrote specs/, asha_plan.json, tiny_chat_s0.json, tiny_chat_s0_60m.json, tiny_chat_e5.json, "
          "tiny_chat_e1.json, tiny_chat_e1_60m.json, tiny_chat_asha.json"
          + (" (PROVISIONAL learning rates)" if provisional else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
