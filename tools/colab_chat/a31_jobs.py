#!/usr/bin/env python3
"""A31's jobs file (the root workspace's AI_ML/chat_screens_claude/PROTOCOL.md section 50): the cue-selective SFT screen on base A.
Claude Opus 5.5, 13 Nov 2026. Writes tools/colab_chat/a31_cue.json; `--check` compares the file on disk with a fresh rendering.

Every training job is S71's spec (lane screen S71: arm A's spec with S64's recipe line, from A's SFT1500) with only these changes:
the packet (S73's: S71's packet with the training twins as ordinary rows), the seed, checkpoint_every 128 (I/O only; S71 saved
every 25), the dev jobs' export (safetensors only), and, for the margin arms, data.twin_examples and train.twin_loss.

  gate        a31_gate.py: one step of a twin run on CUDA and on the CPU, compared; and a plain step, whose SFT loss must equal
              the twin run's bit for bit (the SFT batches are the plain spec's)
  D0, Dm_L, Dn_L   the dev sweep at seed 99: plain; margin at lambda L; margin and nuisance at lambda = mu = L
  score_*     `mtstudio score` of the job's model on every readout request (a31_readout_req.jsonl), on the GPU
  select      a31_select.py: the registered selection rule on the dev scores and the dev runs' held-out SFT loss
  P_S         arm 1, plain SFT on S73's packet, seed S
  M_S_L       arm 2, margin at L, run only if select promotes M_L
  N_S_L       arm 3, margin and nuisance at L, run only if select promotes N_L

    python3 tools/colab_chat/a31_jobs.py [--check]
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
GRID = (0.1, 0.3, 1.0)  # lambda for arm 2; lambda = mu for arm 3
MARGIN = 1.0           # nats per reply token
GROUPS = 4             # twin groups per step

S71 = {  # lane runs/a26-s71-20261111/S71/spec_S71.json, paths aside
    "arch": {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
             "state_tokens": {"k": 4, "mask": "causal"}},
    "data": {"vocab": f"{U}/vocab.gguf", "vocab_cap": 16384, "T": 1024, "sft_examples": f"{U}/S73.jsonl"},
    "train": {"objective": "sft", "steps": 1024, "lr": 0.0001, "batch": 16, "accum": 1, "clip": 1.0, "eval_every": 32, "eval_batch": 8,
              "checkpoint_every": 128, "gradmap_every": 0, "seed": 1, "optimizer": "adamw", "init_from": f"{U}/tc_sft_b1_r1.safetensors",
              "lr_schedule": {"warmup_steps": 64, "decay": "cosine", "min_lr_frac": 0.1}, "eval_examples": 64},
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


EARLY = (128, 256, 512)   # 12.5%, 25% and 50% of 1,024 steps: multiples of checkpoint_every, so jobs_vm.py can snapshot them


def early_jobs(name, ref):
    """The early readouts of one training job (a31_early.py), each a rung job on its snapshot."""
    return [{"name": f"early_{name}_{s}", "light": True, "rung_of": {"job": name, "step": s},
             "cmd": ["python3", "tools/colab_chat/a31_early.py", "--job", name, "--step", str(s), "--ref", ref, "--out", "{out}"],
             "outputs": ["early.json", "scores.jsonl"]} for s in EARLY]


def train_job(name, s, promoted=None):
    j = {"name": name, "spec": s, "after": "gate", "relay_every_steps": 100000, "snapshot_steps": list(EARLY)}
    if promoted:
        j["promoted_by"] = {"files": [f"{C}/select/decisions.json"], "round": "a31", "key": promoted}
    return j


def score_job(name):
    return {"name": f"score_{name}", "light": True, "after": name,
            "cmd": [BIN, "score", f"{C}/{name}", "--in", f"{U}/a31_readout_req.jsonl", "--out", "{out}/scores.jsonl", "--batch-tokens", "8192"],
            "outputs": ["scores.jsonl"]}


def render():
    jobs = [{"name": "gate", "cmd": ["python3", "tools/colab_chat/a31_gate.py", "--out", "{out}", "--spec", f"{U}/a31_gate_spec.json"],
             "outputs": ["gate.json"]},
            # S71 itself on every readout request: the reference level (reported, not judged)
            {"name": "score_S71", "light": True, "after": "gate",
             "cmd": ["bash", "-c", "mkdir -p {out}/m && ln -sf {upload}/s71.safetensors {out}/m/screen_S71.safetensors && "
                     "cp {upload}/s71_spec.json {out}/m/spec.json && " + BIN + " score {out}/m --in {upload}/a31_readout_req.jsonl "
                     "--out {out}/scores.jsonl --batch-tokens 8192"],
             "outputs": ["scores.jsonl"]}]
    dev = [("D0", spec(DEV_SEED, dev=True))]
    dev += [(f"Dm_{lname(x)}", spec(DEV_SEED, x, 0.0, dev=True)) for x in GRID]
    dev += [(f"Dn_{lname(x)}", spec(DEV_SEED, x, x, dev=True)) for x in GRID]
    for n, s in dev:
        jobs += [train_job(n, s), score_job(n)] + early_jobs(n, "D0")
    jobs.append({"name": "select", "after": [f"score_{n}" for n, _ in dev] + [n for n, _ in dev],
                 "cmd": ["python3", "tools/colab_chat/a31_select.py", "--chat", C, "--out", "{out}", "--manifest", f"{U}/a31_readout_req.manifest.json"],
                 "outputs": ["decisions.json"]})
    for sd in SEEDS:
        jobs += [train_job(f"P_{sd}", spec(sd)), score_job(f"P_{sd}")] + early_jobs(f"P_{sd}", f"P_{sd}")
    for sd in SEEDS:
        for x in GRID:
            for arm, mu in (("M", 0.0), ("N", x)):
                n = f"{arm}_{sd}_{lname(x)}"
                jobs += [train_job(n, spec(sd, x, mu), f"{arm}_{lname(x)}"), score_job(n)] + early_jobs(n, f"P_{sd}")
    for j in jobs:
        if j.get("promoted_by"):
            j["after"] = ["gate", "select"]
    return {"_comment": "A31 (lane PROTOCOL.md section 50), rendered by tools/colab_chat/a31_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 1, "jobs": jobs}


def gate_spec():
    """The gate's spec: S71's, at batch 2, one step, a twin loss on two groups."""
    s = spec(1, 0.3, 0.3)
    s["train"].update({"batch": 2, "steps": 1, "eval_every": 1000, "checkpoint_every": 1000})
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
