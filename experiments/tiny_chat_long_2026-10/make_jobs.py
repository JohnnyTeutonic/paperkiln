#!/usr/bin/env python3
"""Carry-12's cells (PREREGISTRATION_DRAFT.md sections 6, 9 and 10): their pretraining specs, SFT templates and
jobs files, one RunPod L4 per (cell, seed). Deterministic: rerunning it rewrites the same files. It imports
../tiny_chat_2026-10/make_jobs.py (E1's spec rule, panels, data job) and make_stage0_jobs.py (the T 1,024 data
job) as libraries and writes none of their files.

    python experiments/tiny_chat_long_2026-10/make_jobs.py            # write the specs and jobs files, print the table
    python experiments/tiny_chat_long_2026-10/make_jobs.py --table    # print the table only

Writes
  experiments/tiny_chat_long_2026-10/specs/cell_<B|Sgist|P16|X>.json   pretraining specs at T 1,024
  experiments/tiny_chat_long_2026-10/specs/sft_T1024.json              the SFT template (16 x 1,024 per step)
  experiments/tiny_chat_long_2026-10/specs/sft_T1024_long4.json        cell D's (sft_human_long4.jsonl)
  experiments/tiny_chat_long_2026-10/specs/sft_T1024_tsft.json         cell TS's (sft_human_tsft.jsonl)
  experiments/tiny_chat_long_2026-10/specs/sft_w512.json               cell W512's re-run (E1's template, T 512)
  tools/colab_chat/carry12_<B|Sgist|P16|X>_r<1|2|3>.json               one machine each; D, TS and W512 on B's

The cells (section 6): B is 60m_L* at T 1,024 (L* = experiments/tiny_chat_2026-10/size/depth_choice.json); S-gist
is B plus k = 4 state slots under the turn_state mask (arch.state_tokens); P16 is plain 60m_L16 (dropped when
L* = 16: P16 is then B); X is 60m_L16 plus full attention residuals (arch.custom.attn_res, arm X5's). Every cell
trains E1's recipe at T 1,024: 10,000 steps of 16,384 tokens (16 x 1,024 at L <= 8, 8 x 2 from 16 layers), the
35% mix, AdamW and cosine, at E1 at 60M's T 512 rate for its plain shape (stage0/lr_choice.json: D and S-gist at
B's, X at P16's), seeds 1-3. The pretraining specs are named after the shape whose rate they take (tc_60m_L8,
tc_60m_L16), so tiny_chat_run.py gate checks each rate against lr_choice.json. No rungs (not registered).

Each machine: cuda_gate (every CUDA leg on this GPU); data (light; the T 1,024 data job of the pre-freeze campaign:
E1's sources, split and 35% mix, Carry-12 in the decontamination beside Carry-5, SFT rows at T 1,024 and cell D's
set, --long-sft 10:4); gate (refuses unless experiments/tiny_chat_long_2026-10/PREREGISTRATION.md, the frozen file,
is in the commit, the pre-freeze campaign's CUDA gate passed, design.json is final and every pretraining rate is
lr_choice.json's); the pretraining run; the post job: SFT at the same 16,384 tokens per step (post --sft-batching
spec), the zero-rate panels (512 windows of 1,024 tokens), Carry-5 (ember generation after the 40-item
spot-check, base-model Carry-5) and then Carry-12 (tools/carry12_eval.py: forced choice in gold, self, near, twin and
noctx by mtstudio score; self history to the target at --max-new 48 by ember after its 20-item spot-check; no
near_self; every reply through the parity gate), all generation at ember ec3201b with EMBER_GEMM_TF32=0 and
EMBER_DETERMINISTIC=1. B's machine also runs D's post job from B's pretrained weights (sft_human_long4.jsonl, B's
1,500 steps), TS's (the template-SFT cell: sft_human_tsft.jsonl, the same human rows plus the 4,224 pinned
template dialogues of tools/carry12_tsft_build.py, 20% of its rows; B's 1,500 steps, so the SFT token budget is
B's; its data job adds --template-sft) and W512's: E1 at 60M's L* runs (T 512) have no per-seed SFT model at home
(their post jobs kept them on the VM; Stage L0 scored the pick only), so W512's seed s is E1's pretrained weights
(uploaded) through E1's own SFT (data512: E1's T 512 data job; E1's template; its sft_weights_sha256 is compared at
home with E1's post summary) and Carry-12 forced choice (gold, near, twin, noctx; --carry12-gen none), all HT
needs. B, D and TS read the panels again at their SFT weights (post --sft-panels: their pretrained weights are one,
so only the SFT weights show what the SFT data did to general text and natural dialogue), and a last job
(chat_c12_r<seed>, tools/colab_chat/chat_eval_vm.sh) runs the chat-prompt checks (tools/chat_eval.py on
tinychat_hf_prompts_v1 and chat_prompts_v1: clean stops, repetition, reference bits per byte; tools/qa_eval.py on
simple_qa_v1) on the three SFT models side by side, so a Carry-12 gain bought by degrading general chat shows.

Nothing here launches: every file's gate refuses until the frozen PREREGISTRATION.md is committed, and the launch
needs Jonathan's go on the measured total (section 11 item 6). At launch, each file runs under its own
runner_multi.py entry (printed by this script) with its --max-usd cap: ceil(1.3 x estimate x $0.59), and its
--max-hours: 23.5, or ceil(1.3 x estimate) when that is longer (B's machines, about 24 h of jobs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REL = os.path.relpath(HERE, ROOT).replace(os.sep, "/")
EXP = os.path.join(os.path.dirname(HERE), "tiny_chat_2026-10")
EXP_REL = os.path.relpath(EXP, ROOT).replace(os.sep, "/")
COLAB = os.path.join(ROOT, "tools", "colab_chat")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mj = _load("tiny_chat_make_jobs", os.path.join(EXP, "make_jobs.py"))
s0 = _load("carry12_make_stage0_jobs", os.path.join(HERE, "make_stage0_jobs.py"))

C = mj.C
RUN = mj.RUN
T = 1024
SHARE = 35                       # the 35% mix (section 6)
SEEDS = [1, 2, 3]
TOKENS_PER_STEP = 16_384         # 16 x 1,024 (or 8 x 2 x 1,024): E1's tokens per step
K_STATE = 4
EMBER = "ec3201b04e6f2fcb6dffd3496b0c72b3bec0c54a"   # section 2's pin (carry12_l0.EMBER_COMMIT)
CARRY12 = "corpus/eval/carry12_v1.jsonl"
CARRY12_SPOT = 20
FROZEN = f"{REL}/PREREGISTRATION.md"
CUDA_GATE_HOME = f"{REL}/stage0/cuda_gate_c12_s0.json"
CHOICE = f"{EXP_REL}/stage0/lr_choice.json"
DESIGN = f"{EXP_REL}/design.json"
DEPTH_CHOICE = os.path.join(EXP, "size", "depth_choice.json")
TOTAL = os.path.join(HERE, "stage0", "total.json")
E1_WALLS = os.path.join(HERE, "stage0", "e1_60m_walls_t512.json")
COST_L8 = os.path.join(HERE, "stage0", "cost_pick_e1_60m_L8.l0.json")
BENCH = os.path.join(ROOT, "docs", "receipts", "score_batch_bench_20261028.json")
SCORE_BATCH_TOKENS = 16384       # every Carry-5 and Carry-12 score batched (mtstudio score --batch-tokens; E5.2)
USD = 0.59
CAP_MARGIN = 1.3
MAX_HOURS = 23.5                 # runner_multi's default deadline; B's machines get ceil(1.3 x estimate)
SFT_PANELS_H = 0.05              # per post job: the two panels again at the SFT weights (W512's allowance for its panels)
CHAT_H = 0.5                     # per B machine: chat_eval_vm.sh on B, D and TS (an allowance; not measured)
TSFT_SHARE = 0.20                # TS's template rows: 4,224 beside the T 1,024 data job's 16,932 human SFT rows
VOCAB_UPLOAD = "vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
CELL_NAMES = {"B": "B", "Sgist": "S-gist", "P16": "P16", "X": "X"}


def lstar():
    with open(DEPTH_CHOICE, encoding="utf-8") as f:
        return int(json.load(f)["depth"])


def cells():
    """The core's machines' cells: B, S-gist, P16 (not when L* = 16) and X."""
    return ["B", "Sgist"] + (["P16"] if lstar() != 16 else []) + ["X"]


def batching(L):
    """Section 3: 16 x 1,024 at L <= 8, 8 x 1,024 with 2-step accumulation from 16 layers."""
    return (8, 2) if L >= 16 else (16, 1)


def shape_of(cell):
    return f"60m_L{lstar()}" if cell in ("B", "Sgist") else "60m_L16"


def cell_spec(cell, seed=SEEDS[0]):
    sid = shape_of(cell)
    s = dict(mj.shapes()[sid])
    if cell == "X":
        s["attn_res"] = dict(mj.X5_ATTN_RES)
    lrs, provisional = mj.lr_table()
    if provisional or sid not in lrs:
        raise SystemExit(f"{CHOICE} holds no rate for {sid}")
    spec = mj.pt_spec(sid, s, mj.corpus_of(SHARE), lrs[sid], seed)
    b, a = batching(s["layers"])
    spec["data"]["T"] = T
    spec["train"].update({"batch": b, "accum": a})
    assert b * a * T == TOKENS_PER_STEP
    if cell == "Sgist":
        spec["arch"]["state_tokens"] = {"k": K_STATE, "mask": "turn_state"}
    shape = {k: s[k] for k in ("size", "nonemb", "emb", "total")}
    if cell == "Sgist":
        shape["state_params"] = K_STATE * s["d"]
    if cell == "X":
        shape["attn_res_query_params"] = 2 * s["layers"] * s["d"]
        shape["depth_sources"] = 2 * s["layers"] + 1
    spec["_shape"] = shape
    return spec


def cell_spec_path(cell):
    return f"{REL}/specs/cell_{cell}.json"


SFT_T1024 = f"{REL}/specs/sft_T1024.json"
SFT_LONG4 = f"{REL}/specs/sft_T1024_long4.json"
SFT_TSFT = f"{REL}/specs/sft_T1024_tsft.json"
SFT_W512 = f"{REL}/specs/sft_w512.json"


def sft_templates():
    base = json.loads(json.dumps(mj.SFT))
    base.pop("_comment", None)
    t1024 = json.loads(json.dumps(base))
    t1024["data"]["T"] = T
    t1024["train"].update({"batch": 16, "accum": 1})
    t1024["_comment"] = ("Carry-12 cells' SFT (PREREGISTRATION_DRAFT.md section 6): E1's recipe at T 1,024, 16 rows of "
                         "1,024 per step (post --sft-batching spec takes the shape spec's batch and accumulation: "
                         "16 x 1 at 60m_L8, 8 x 2 at 60m_L16), the T 1,024 data job's sft_human.jsonl.")
    long4 = json.loads(json.dumps(t1024))
    long4["data"]["sft_examples"] = f"{C}/data/sft_human_long4.jsonl"
    long4["_comment"] = ("Cell D's SFT: as sft_T1024.json on sft_human_long4.jsonl (the T 1,024 data job's SFT rows, "
                         "each row of 10+ trained turns four times; tiny_chat_data.py --long-sft 10:4), B's 1,500 steps.")
    tsft = json.loads(json.dumps(t1024))
    tsft["data"]["sft_examples"] = f"{C}/data/sft_human_tsft.jsonl"
    tsft["_comment"] = ("Cell TS's SFT: as sft_T1024.json on sft_human_tsft.jsonl (the T 1,024 data job's SFT rows once "
                        "each plus the 4,224 pinned template dialogues of tools/carry12_tsft_build.py, 20% of the rows; "
                        "tiny_chat_data.py --template-sft), B's 1,500 steps: the SFT token budget is B's.")
    w512 = json.loads(json.dumps(mj.SFT))
    w512["data"]["sft_examples"] = f"{C}/data512/sft_human.jsonl"
    w512["_comment"] = ("Cell W512's re-run: E1's SFT template (experiments/tiny_chat_2026-10/specs/sft.json) byte for "
                        "byte but the path: E1's T 512 data job runs as data512 beside the T 1,024 one.")
    return {SFT_T1024: t1024, SFT_LONG4: long4, SFT_TSFT: tsft, SFT_W512: w512}


# ---------------------------------------------------------------- cost per machine (stage0/total.json)
def batch_bench():
    """The score batching bench (docs/receipts/score_batch_bench_20261028.json, an L4, pick_e1_60m_L8, --batch-tokens
    16,384): per eval, the batched speedup and the unbatched scoring wall scaled to the whole set (it scored every
    Nth item)."""
    with open(BENCH, encoding="utf-8") as f:
        ev = json.load(f)["evals"]
    return {k: {"speedup": ev[k]["variants"]["batched"]["speedup"],
                "unbatched_full_s": ev[k]["variants"]["unbatched"]["wall_s"] * ev[k]["every"]} for k in ("carry5", "carry12")}


def per_run():
    """{total.json cell: hours per run} with every Carry-5 and Carry-12 score batched (--score-batch-tokens 16,384):
    the measured Carry-12 part with its scoring divided by the bench's Carry-12 speedup, and the rest of the post job
    less the scoring the bench saves on its two Carry-5 passes (the SFT model's and the base model's; the bench's
    three-mode set, so the saving is a lower bound)."""
    with open(TOTAL, encoding="utf-8") as f:
        tot = json.load(f)
    bb = batch_bench()
    c5_save_h = 2 * bb["carry5"]["unbatched_full_s"] * (1 - 1 / bb["carry5"]["speedup"]) / 3600
    c12 = {sid: (r["eval_s"] - r["timing"]["score_s"] + r["timing"]["score_s"] / bb["carry12"]["speedup"]
                 + r["ember_build_s"]) / 3600 for sid, r in tot["carry12_post"].items()}
    out = {}
    for name, c in tot["cells"].items():
        out[name] = c["pt_h"] + c["sft_h"] + c["post_rest_h"] - c5_save_h + c12[c["shape"]]
    return out, c5_save_h, c12


def additions():
    """Hours per B machine beyond B and D: W512's re-run; TS at D's rates (SFT, the rest of the post job and Carry-12,
    from B's weights), plus its SFT panels; the SFT panels of B and D; and the chat-prompt checks (an allowance)."""
    with open(TOTAL, encoding="utf-8") as f:
        tot = json.load(f)
    with open(COST_L8, encoding="utf-8") as f:
        cost = json.load(f)["cost_run"]["timing"]
    sid = f"60m_L{lstar()}"
    sft512 = tot["base_t512"][sid]["sft_s_512"] / 3600
    # W512's Carry-12 is forced choice in four modes, no generation and no gate rows: the cost run's scoring time
    # scaled by its rows (gold, near, twin, noctx x 4 candidates against five modes x 4 plus one gate row per reply),
    # batched as every cell's.
    fc4, all_rows = 1452 * 4 * 4, 1452 * 5 * 4 + 17424
    w512 = sft512 + cost["score_s"] * fc4 / all_rows / batch_bench()["carry12"]["speedup"] / 3600 + 0.05
    pr, _, _ = per_run()
    return {"W512": w512, "TS": pr["D"] + SFT_PANELS_H, "sft_panels_B_D": 2 * SFT_PANELS_H, "chat": CHAT_H}


def hours():
    """{cell: hours per machine}, every score batched (per_run); B's machine adds D and additions(): W512, TS, the SFT
    panels and the chat-prompt checks."""
    with open(TOTAL, encoding="utf-8") as f:
        mh = json.load(f)["machine_h"]
    with open(E1_WALLS, encoding="utf-8") as f:
        walls = json.load(f)
    add = additions()
    pr, _, _ = per_run()
    out = {"B": mh + pr["B"] + pr["D"] + sum(add.values()), "Sgist": mh + pr["S-gist"], "P16": mh + pr["P16"],
           "X": mh + pr["X"]}
    return {k: v for k, v in out.items() if k in cells()}, add["W512"], walls


def cap(h):
    return int(math.ceil(h * USD * CAP_MARGIN))


def max_hours(h):
    """The runner's deadline: 23.5 h, or ceil(1.3 x estimate) for a machine whose jobs run longer."""
    return max(MAX_HOURS, float(math.ceil(h * CAP_MARGIN)))


# ---------------------------------------------------------------- jobs
PANELS_T512 = ["--panel", f"fineweb={C}/data512/panel_fineweb.ids", "--panel", f"dialogue={C}/data512/panel_dialogue.ids"]
POST_C12 = ["--gen", "ember", "--ember-commit", EMBER, "--ember-deterministic", "--sft-batching", "spec",
            "--carry12-items", CARRY12, "--carry12-spot", str(CARRY12_SPOT),
            "--score-batch-tokens", str(SCORE_BATCH_TOKENS)]


def post_outputs(cell, seed):
    return ["summary.json", "carry12/summary.json", "carry12/m/items.jsonl", "carry12/m/gen.jsonl",
            "carry12/m/gen_engine.json", "carry5/m/items.jsonl", "carry5/m/gen.jsonl", "gen_engine.json",
            "sft_spec.json", f"sft/tc_sft_{cell}_r{seed}.safetensors", f"sft/tc_sft_{cell}_r{seed}.gguf"]


CHAT_CELLS = ["B", "D", "TS"]


def chat_job(seed):
    """The chat-prompt checks on B's, D's and TS's SFT models side by side (chat_eval_vm.sh: tools/chat_eval.py on
    tinychat_hf_prompts_v1 and chat_prompts_v1, tools/qa_eval.py on simple_qa_v1, through ember's CUDA server and
    the CUDA mtstudio); each model is NAME@JOB, its post job's SFT spec (/content/chat/<post>/sft_spec.json)."""
    return {"name": f"chat_c12_r{seed}", "after": [f"post_c12_{c}_r{seed}" for c in CHAT_CELLS],
            "cmd": ["bash", "tools/colab_chat/chat_eval_vm.sh", "{out}",
                    *(f"{c}@post_c12_{c}_r{seed}/sft" for c in CHAT_CELLS)],
            "outputs": ["chat_eval_results.tar.gz"]}


def jobs_name(cell, seed):
    return f"carry12_{cell}_r{seed}.json"


def w512_weights(seed):
    L = lstar()
    return (f"tc_e1_60m_L{L}_r{seed}.safetensors",
            f"/mnt/c/ml_artifacts/tiny_chat/e1_60m_L{L}/pt_e1_60m_L{L}_r{seed}/tc_e1_60m_L{L}_r{seed}.safetensors")


def machine_doc(cell, seed, est):
    name = jobs_name(cell, seed)
    pt = f"pt_c12_{cell}_r{seed}"
    model = f"tc_c12_{cell}_r{seed}"
    cuda = {"name": "cuda_gate", "cmd": ["python3", RUN, "cuda_gate", "--out", "{out}"],
            "outputs": ["cuda_gate.json", "cuda_gate.log"]}
    data = {"name": "data", "light": True, "cmd": s0.data_cmd(), "outputs": ["data.ledger.json", "human_decontam.json"]}
    if cell == "B":  # TS's SFT set: the same data job plus the pinned template rows (its other outputs unchanged)
        data = {**data, "cmd": data["cmd"] + ["--template-sft"], "outputs": data["outputs"] + ["tsft_decontam.json"]}
    gate = {"name": "gate", "cmd": ["python3", RUN, "gate", "--out", "{out}", "--jobs", f"tools/colab_chat/{name}",
                                    "--choice", CHOICE, "--design", DESIGN, "--need-cuda-gate", CUDA_GATE_HOME,
                                    "--need-file", FROZEN],
            "outputs": ["gate.json"]}
    spec = cell_spec(cell, seed)
    spec.pop("_shape")
    jobs = [cuda, data, gate,
            {"name": pt, "model_name": model, "after": ["data", "gate", "cuda_gate"], "relay_every_steps": 2500,
             "spec": spec}]

    sftp = ["--sft-panels"] if cell == "B" else []   # B, D and TS: the panels at the SFT weights too

    def post(c, sft, sp=cell_spec_path(cell)):
        return {"name": f"post_c12_{c}_r{seed}", "after": pt, "init_job": pt,
                "cmd": ["python3", RUN, "post", "--out", "{out}", "--spec", sp, "--pt-weights",
                        f"{C}/{pt}/{model}.safetensors", "--pt-dir", f"{C}/{pt}", "--sft-template", sft,
                        "--seed", str(seed), "--cell", c, "--share", str(SHARE), *mj.PANELS, *mj.ITEMS, *POST_C12,
                        *sftp],
                "outputs": post_outputs(c, seed)}
    jobs.append(post(cell, SFT_T1024))
    uploads = [VOCAB_UPLOAD]
    if cell == "B":
        jobs.append(post("D", SFT_LONG4))
        jobs.append(post("TS", SFT_TSFT))
        jobs.append({**mj.data_job([SHARE]), "name": "data512", "light": True})
        w, src = w512_weights(seed)
        uploads.append(f"{w}={src}")
        jobs.append({"name": f"post_c12_W512_r{seed}", "after": ["data512", "gate", pt],
                     "cmd": ["python3", RUN, "post", "--out", "{out}", "--spec",
                             f"{EXP_REL}/specs/shape_60m_L{lstar()}.json", "--pt-weights", f"{C}/upload/{w}",
                             "--pt-dir", f"{C}/upload/no_pt_dir_r{seed}", "--sft-template", SFT_W512,
                             "--seed", str(seed), "--cell", "W512", "--share", str(SHARE), *PANELS_T512, *mj.ITEMS,
                             "--no-carry5", "--carry12-items", CARRY12, "--carry12-gen", "none",
                             "--score-batch-tokens", str(SCORE_BATCH_TOKENS)],
                     "outputs": ["summary.json", "carry12/summary.json", "carry12/m/items.jsonl", "sft_spec.json"]})
        jobs.append(chat_job(seed))
    title = {"B": f"B (60m_L{lstar()} at T 1,024) and, from its pretrained weights, D and TS; then W512's re-run and "
                  f"the chat-prompt checks of B, D and TS",
             "Sgist": f"S-gist (60m_L{lstar()} + k {K_STATE} state slots, turn_state, at T 1,024)",
             "P16": "P16 (plain 60m_L16 at T 1,024)", "X": "X (60m_L16 + full attention residuals at T 1,024)"}[cell]
    runner = runner_command(cell, seed, est, uploads)
    doc = {"_comment": f"Carry-12 cell {title}, seed {seed} ({REL}/PREREGISTRATION_DRAFT.md sections 6 and 10; written "
                       f"by {REL}/make_jobs.py, whose docstring says what each job does). One RunPod L4, one GPU job "
                       f"at a time; estimated {est:.1f} L4 VM-hours (stage0/total.json), cap ${cap(est)}. NOT "
                       f"LAUNCHED: the gate refuses until {FROZEN} is committed, and the launch needs Jonathan's go. "
                       f"Runner (disabled until then): {runner}",
           "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}
    return name, doc, uploads


def runner_command(cell, seed, est, uploads):
    tag = f"pk-c12-{cell.lower()}-r{seed}"
    return (f"python3 -u tools/colab_chat/runner_multi.py --backend runpod --gpu L4 --commit FREEZE_COMMIT --jobs "
            f"tools/colab_chat/{jobs_name(cell, seed)} --name {tag} --art /mnt/c/ml_artifacts/tiny_chat/c12_{cell}_r{seed} "
            f"--state-dir /home/jonat/{tag.replace('-', '_')}_runner --max-hours {max_hours(est):g} --volume-gb 80 "
            f"--max-usd {cap(est)} " + " ".join(f"--upload {u}" for u in uploads))


def table():
    hs, w512, _ = hours()
    add = additions()
    rows = ["| machine (x 3 seeds) | cells | L4 h per machine | cap per machine | VM-hours | USD |", "|---|---|---:|---:|---:|---:|"]
    tot = 0.0
    for cell, h in hs.items():
        tot += 3 * h
        what = "B, D, TS, W512, chat checks" if cell == "B" else CELL_NAMES[cell]
        rows.append(f"| {CELL_NAMES[cell]} | {what} | {h:.2f} | ${cap(h)} | {3 * h:.1f} | {3 * h * USD:.2f} |")
    rows.append(f"| **all {3 * len(hs)} machines** | | | ${3 * sum(cap(h) for h in hs.values())} | **{tot:.1f}** | "
                f"**{tot * USD:.2f}** |")
    pr, c5_save, c12 = per_run()
    t = add["TS"]
    rest = add["sft_panels_B_D"] + add["chat"]
    rows.append(f"\nPer run from stage0/total.json's measured parts with every score batched (--score-batch-tokens "
                f"{SCORE_BATCH_TOKENS}; docs/receipts/score_batch_bench_20261028.json): the Carry-12 part "
                + ", ".join(f"{h:.2f} h at {sid}" for sid, h in c12.items())
                + f", the rest of the post job {c5_save:.2f} h shorter (two Carry-5 scoring passes); per run "
                + ", ".join(f"{k} {v:.2f} h" for k, v in pr.items())
                + f". W512's re-run {w512:.2f} h per seed; TS at D's rates plus its SFT panels, {t:.2f} h per seed, "
                f"{3 * t:.1f} VM-hours (${3 * t * USD:.2f}); B's and D's SFT panels and the chat-prompt checks, {rest:.2f} h "
                f"per machine. In all: {tot:.1f} VM-hours, ${tot * USD:.2f}. B's machines: {hs['B']:.1f} h each, "
                f"--max-hours {max_hours(hs['B']):g}.")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    if not a.table:
        os.makedirs(os.path.join(HERE, "specs"), exist_ok=True)
        for cell in cells():
            mj.write(os.path.join(ROOT, cell_spec_path(cell)), cell_spec(cell))
        for path, doc in sft_templates().items():
            mj.write(os.path.join(ROOT, path), doc)
        hs, _, _ = hours()
        for cell in cells():
            for seed in SEEDS:
                name, doc, _ = machine_doc(cell, seed, hs[cell])
                mj.write(os.path.join(COLAB, name), doc)
        print(f"wrote {len(cells())} cell specs, {len(sft_templates())} SFT templates and {3 * len(cells())} jobs files "
              f"(tools/colab_chat/carry12_*_r*.json) at L* = {lstar()}")
    print(table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
