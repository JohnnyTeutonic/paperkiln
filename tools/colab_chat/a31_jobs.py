#!/usr/bin/env python3
"""A31's jobs file (the root workspace's AI_ML/chat_screens_claude/PROTOCOL.md section 50): the cue-selective SFT screen on base A.
Claude Opus 5.5, 13 Nov 2026; revised after the launch reviews of 13 Nov 2026. Writes tools/colab_chat/a31_cue.json; `--check`
compares the file on disk with a fresh rendering.

Every training job is S71's spec (lane screen S71: arm A's spec with S64's recipe line, from A's SFT1500) with only these changes:
the packet (S73's: S71's packet with the training twins as ordinary rows), the seed, checkpoint_every 128 and eval_examples 256
(I/O and evaluation only; S71: 25 and 64), the sweep's export (safetensors only), and, for the margin arms, data.twin_examples and
train.twin_loss.

  gate             a31_gate.py: two steps of a twin run on CUDA and on the CPU, and of the plain spec on both, compared
  D0, Dm_L, Dn_L   the dev sweep at seed 99: plain; margin at lambda L; margin and nuisance at lambda = mu = L. Scored on the
                   early requests only (dev and heldout_dev): a sweep model never reads a final test set
  select           a31_select.py, after every sweep job has ended (FINISHED or FAILED): the registered selection rule
  P_S              arm P, plain SFT on S73's packet, seed S
  M_S_L, N_S_L     arms M and N at L, run only if select promotes M_L or N_L
  early_J_S        a31_early.py at steps 128, 256, 512 of every training job: the health rule, then the early requests only
  score_J          the final readout requests (a31_readout_req.jsonl), for S71 and the six arm models only, all after select

    python3 tools/colab_chat/a31_jobs.py [--check] [--gate-spec F] [--s71-spec F]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "a31_cue.json")
U = "/content/chat/upload"
C = "/content/chat"
BIN = "/content/pk/build_cuda/mtstudio"

SEEDS = (1, 50)        # seed 50 shares no initialisation draw with seed 1 (adjacent seeds overlap); SFT order streams are distinct
DEV_SEED = 99
GRID = (0.1, 0.3, 1.0)  # lambda for arm M; lambda = mu for arm N
MARGIN = 1.0           # nats per reply token
GROUPS = 4             # twin groups per step
EARLY = (128, 256, 512)   # 12.5%, 25% and 50% of 1,024 steps: multiples of checkpoint_every, so jobs_vm.py can snapshot them
FINAL_REQ, EARLY_REQ = f"{U}/a31_readout_req.jsonl", f"{U}/a31_early_req.jsonl"

S71 = {  # lane runs/a26-s71-20261111/S71/spec_S71.json, paths aside
    "arch": {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
             "state_tokens": {"k": 4, "mask": "causal"}},
    "data": {"vocab": f"{U}/vocab.gguf", "vocab_cap": 16384, "T": 1024, "sft_examples": f"{U}/S73.jsonl"},
    "train": {"objective": "sft", "steps": 1024, "lr": 0.0001, "batch": 16, "accum": 1, "clip": 1.0, "eval_every": 32, "eval_batch": 8,
              "checkpoint_every": 128, "gradmap_every": 0, "seed": 1, "optimizer": "adamw", "init_from": f"{U}/tc_sft_b1_r1.safetensors",
              "lr_schedule": {"warmup_steps": 64, "decay": "cosine", "min_lr_frac": 0.1}, "eval_examples": 256},
    "export": {"formats": ["safetensors", "gguf"]},
}


def spec(seed, lam=None, mu=None, dev=False):
    s = json.loads(json.dumps(S71))
    s["train"]["seed"] = seed
    if lam is not None:
        s["data"]["twin_examples"] = f"{U}/train_twins.jsonl"
        s["train"]["twin_loss"] = {"groups": GROUPS, "margin": MARGIN, "lambda": lam, "mu": mu or 0.0}
    if dev:
        s["export"] = {"formats": ["safetensors"]}
    return s


def lname(lam):
    return str(lam).replace(".", "p")


def early_jobs(name, ref):
    """The early readouts of one training job (a31_early.py), each a rung job on its snapshot."""
    return [{"name": f"early_{name}_{s}", "light": True, "rung_of": {"job": name, "step": s},
             "cmd": ["python3", "tools/colab_chat/a31_early.py", "--job", name, "--step", str(s), "--ref", ref, "--out", "{out}"],
             "outputs": ["early.json", "scores.jsonl"]} for s in EARLY]


def train_job(name, s, promoted=None):
    j = {"name": name, "spec": s, "after": "gate", "relay_every_steps": 100000, "snapshot_steps": list(EARLY)}
    if promoted:
        j["promoted_by"] = {"files": [f"{C}/select/decisions.json"], "round": "a31", "key": promoted}
        j["after"] = ["gate", "select"]
    return j


def score_job(name, req, after):
    return {"name": f"score_{name}", "light": True, "after": after,
            "cmd": ["timeout", "900", BIN, "score", f"{C}/{name}", "--in", req, "--out", "{out}/scores.jsonl", "--batch-tokens", "8192"],
            "outputs": ["scores.jsonl"]}


def render():
    jobs = [{"name": "gate", "cmd": ["python3", "tools/colab_chat/a31_gate.py", "--out", "{out}", "--spec", f"{U}/a31_gate_spec.json"],
             "outputs": ["gate.json"]}]
    dev = [("D0", spec(DEV_SEED, dev=True))]
    dev += [(f"Dm_{lname(x)}", spec(DEV_SEED, x, 0.0, dev=True)) for x in GRID]
    dev += [(f"Dn_{lname(x)}", spec(DEV_SEED, x, x, dev=True)) for x in GRID]
    for n, s in dev:   # early readouts are listed before the scoring, so they take the light slots first
        jobs += [train_job(n, s)] + early_jobs(n, "D0") + [score_job(n, EARLY_REQ, n)]
    jobs.append({"name": "select", "after_terminal": True,
                 "after": [n for n, _ in dev] + [f"score_{n}" for n, _ in dev] + [f"early_{n}_{s}" for n, _ in dev for s in EARLY],
                 "cmd": ["python3", "tools/colab_chat/a31_select.py", "--chat", C, "--out", "{out}"],
                 "outputs": ["decisions.json"]})
    # S71 itself on every final readout request: the reference level (reported, not judged), after the selection is sealed
    jobs.append({"name": "score_S71", "light": True, "after": "select",
                 "cmd": ["bash", "-c", "mkdir -p {out}/m && ln -sf {upload}/s71.safetensors {out}/m/screen_S71.safetensors && "
                         "cp {upload}/s71_spec.json {out}/m/spec.json && timeout 900 " + BIN + " score {out}/m --in " + FINAL_REQ +
                         " --out {out}/scores.jsonl --batch-tokens 8192"],
                 "outputs": ["scores.jsonl"]})
    for sd in SEEDS:
        n = f"P_{sd}"
        jobs += [train_job(n, spec(sd))] + early_jobs(n, n) + [score_job(n, FINAL_REQ, [n, "select"])]
    for sd in SEEDS:
        for x in GRID:
            for arm, mu in (("M", 0.0), ("N", x)):
                n = f"{arm}_{sd}_{lname(x)}"
                jobs += [train_job(n, spec(sd, x, mu), f"{arm}_{lname(x)}")] + early_jobs(n, f"P_{sd}") + [score_job(n, FINAL_REQ, n)]
    return {"_comment": "A31 (lane PROTOCOL.md section 50), rendered by tools/colab_chat/a31_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}


def gate_spec():
    """The gate's spec: S71's, at batch 2, two steps at the full learning rate from step 1, a twin loss on two groups at weight 1."""
    s = spec(1, 1.0, 1.0)
    s["train"].update({"batch": 2, "steps": 2, "eval_every": 1000, "checkpoint_every": 1000})
    s["train"]["lr_schedule"] = {"warmup_steps": 0, "decay": "none", "min_lr_frac": 0.1}
    s["train"]["twin_loss"]["groups"] = 2
    s["export"] = {"formats": ["safetensors"]}
    return s


def s71_spec():
    """S71's spec for scoring S71's weights on the pod (its name, so the folder's screen_S71.safetensors loads)."""
    s = json.loads(json.dumps(S71))
    s["name"] = "screen_S71"
    s["data"]["sft_examples"] = f"{U}/S73.jsonl"   # unused by score; a path that exists
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate-spec", help="also write the gate's spec here (uploaded as a31_gate_spec.json)")
    ap.add_argument("--s71-spec", help="also write S71's scoring spec here (uploaded as s71_spec.json)")
    a = ap.parse_args()
    text = json.dumps(render(), indent=1) + "\n"
    if a.gate_spec:
        with open(a.gate_spec, "w") as f:
            f.write(json.dumps(gate_spec(), indent=1) + "\n")
    if a.s71_spec:
        with open(a.s71_spec, "w") as f:
            f.write(json.dumps(s71_spec(), indent=1) + "\n")
    if a.check:
        if open(OUT).read() != text:
            sys.exit("a31_cue.json differs from its rendering")
        print("a31_cue.json matches")
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"{len(render()['jobs'])} jobs -> {OUT}")


if __name__ == "__main__":
    main()
