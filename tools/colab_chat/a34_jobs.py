#!/usr/bin/env python3
"""A34's jobs file (the root workspace's AI_ML/chat_screens_claude/cue_selective/a34/REGISTRATION.md): the own-context floor on
A31's setup. Claude Opus 5.5, 15 Nov 2026. Writes tools/colab_chat/a34_floor.json; `--check` compares the file with a rendering.

Every training job is A31's spec (a31_jobs.S71: S71's spec, S73's packet, A's SFT1500, 1,024 steps) with A31's twin keys where it
has a twin term; only these are new: the floor (train.twin_loss.floor, data.twin_floor_ref), more snapshot steps on the reference
runs (I/O only), and the job names.

  gate              a34_gate.py: A31's gate checks on a floor spec, the device test's floor legs, and the scorer/trainer parity
  D0r, P_1r, P_50r  plain SFT at seeds 99, 1 and 50 (A31's D0, P_1, P_50 again on this build), snapshotted every 128 steps:
                    the floor's reference trajectories, and the reproduction check for P
  ref_start         a34_ref.py start: the start weights (A's SFT1500) scored on the training twins' own-context rows (knot 0)
  ref_<R>           a34_ref.py build: R's snapshots and final weights scored on the same rows; floor_ref.jsonl (knots 0, 128,
                    ..., 896, 1024), its receipt
  Fd_<f>            the floor sweep at seed 99: margin (lambda 0.3, mu 0) and floor f, references from D0r; scored on the early
                    requests only (dev and heldout_dev), never on a final set
  select            a34_select.py, after every sweep job has ended: the registered rule for the floor weight
  M_1r              A31's M_1_0p3 again on this build (the reproduction check for M)
  F_<s>_<f>         margin and floor, seeds 1 and 50, references from P_<s>r; run only if select promotes F_<f>
  NF_<s>_<f>        margin, nuisance invariance (lambda = mu = 0.3) and floor; run only if select promotes NF_<f>
  early_<J>_<s>     A31's a31_early.py at steps 128, 256, 512 of every training job (health first; early requests only)
  score_<J>         the sweep on the early requests; every other model on A31's final requests (after select)

    python3 tools/colab_chat/a34_jobs.py [--check] [--gate-spec F]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a31_jobs as A31  # noqa: E402  (A31's spec, paths and job shapes, imported unchanged)

OUT = os.path.join(HERE, "a34_floor.json")
U, C, BIN, DEVICE_TEST = A31.U, A31.C, A31.BIN, A31.DEVICE_TEST
SEEDS, DEV_SEED, MARGIN, GROUPS, EARLY = A31.SEEDS, A31.DEV_SEED, A31.MARGIN, A31.GROUPS, A31.EARLY
LAM = 0.3                       # A31's selected lambda for M and N (lambda = mu for N)
FLOOR_GRID = (0.3, 1.0, 3.0)    # the floor weight mu_f; the smallest that passes the rule (a34_select.py)
REF_STEPS = (128, 256, 384, 512, 640, 768, 896)   # snapshot knots of the reference runs (multiples of checkpoint_every 128)
FINAL_REQ, EARLY_REQ = A31.FINAL_REQ, A31.EARLY_REQ
OWN_REQ = f"{U}/a34_own_req.jsonl"
REF_TIMEOUT = "5400"            # a reference build: nine scorings of 3,000 rows


def lname(x):
    return A31.lname(x)


def spec(seed, lam=None, mu=None, floor=None, ref_job=None, dev=False):
    s = A31.spec(seed, lam, mu, dev=dev)
    if floor is not None:
        s["train"]["twin_loss"]["floor"] = floor
        s["data"]["twin_floor_ref"] = f"{C}/ref_{ref_job}/floor_ref.jsonl"
    return s


def train_job(name, s, after=("gate",), promoted=None, snaps=EARLY):
    j = {"name": name, "spec": s, "after": list(after), "relay_every_steps": 100000, "snapshot_steps": sorted(set(snaps))}
    if promoted:
        j["promoted_by"] = {"files": [f"{C}/select/decisions.json"], "round": "a34", "key": promoted}
    return j


def early_jobs(name, ref):
    return A31.early_jobs(name, ref)


def score_job(name, req, after):
    j = A31.score_job(name, req, after)
    j["after"] = after if isinstance(after, list) else [after]
    return j


def ref_job(run):
    """The floor references from a reference run's snapshots and final weights (a34_ref.py build)."""
    return {"name": f"ref_{run}", "light": True, "after": [run, "ref_start"],
            "cmd": ["bash", "-c", f"timeout {REF_TIMEOUT} python3 tools/colab_chat/a34_ref.py build --job {run} --chat {C} "
                    f"--bin {BIN} --out {{out}}"],
            "outputs": ["floor_ref.jsonl", "receipt.json"] + [f"scores_{k}{x}" for k in REF_STEPS + (1024,)
                                                                for x in (".jsonl", ".binding.json")]}


def render():
    jobs = [{"name": "gate", "cmd": ["python3", "tools/colab_chat/a34_gate.py", "--out", "{out}", "--spec", f"{U}/a34_gate_spec.json",
                                     "--bin", BIN, "--device-test", DEVICE_TEST], "outputs": ["gate.json"]},
            {"name": "ref_start", "light": True, "after": ["gate"],
             "cmd": ["bash", "-c", f"timeout 1800 python3 tools/colab_chat/a34_ref.py start --chat {C} --bin {BIN} --out {{out}}"],
             "outputs": ["start_own.jsonl", "receipt.json", "scores_0.jsonl", "scores_0.binding.json"]}]
    scores, refs = [], []
    # training jobs in the order the single training slot should take them when several are ready
    D0r = ("D0r", spec(DEV_SEED, dev=True))
    jobs += [train_job("D0r", D0r[1], snaps=REF_STEPS + EARLY)] + early_jobs("D0r", "D0r")
    refs.append(ref_job("D0r"))
    scores.append(score_job("D0r", EARLY_REQ, ["D0r"]))
    jobs += [train_job("P_1r", spec(1), snaps=REF_STEPS + EARLY)] + early_jobs("P_1r", "P_1r")
    refs.append(ref_job("P_1r"))
    sweep = []
    for f in FLOOR_GRID:
        n = f"Fd_{lname(f)}"
        sweep.append(n)
        jobs += [train_job(n, spec(DEV_SEED, LAM, 0.0, f, "D0r", dev=True), after=("gate", "ref_D0r"))] + early_jobs(n, "D0r")
        scores.append(score_job(n, EARLY_REQ, [n]))
    jobs += [train_job("P_50r", spec(50), snaps=REF_STEPS + EARLY)] + early_jobs("P_50r", "P_50r")
    refs.append(ref_job("P_50r"))
    jobs += [train_job("M_1r", spec(1, LAM, 0.0))] + early_jobs("M_1r", "P_1r")
    sel = {"name": "select", "light": True, "after_terminal": True,
           "after": ["D0r", "score_D0r", "ref_D0r"] + sweep + [f"score_{n}" for n in sweep] +
                    [f"early_{n}_{s}" for n in ["D0r"] + sweep for s in EARLY],
           "cmd": ["python3", "tools/colab_chat/a34_select.py", "--chat", C, "--out", "{out}"], "outputs": ["decisions.json"]}
    finals = ["P_1r", "P_50r", "M_1r"]
    for sd in SEEDS:
        for f in FLOOR_GRID:
            for arm, mu in (("F", 0.0), ("NF", LAM)):
                n = f"{arm}_{sd}_{lname(f)}"
                jobs += [train_job(n, spec(sd, LAM, mu, f, f"P_{sd}r"), after=("gate", "select", f"ref_P_{sd}r"),
                                   promoted=f"{arm}_{lname(f)}")] + early_jobs(n, f"P_{sd}r")
                finals.append(n)
    scores += [score_job(n, FINAL_REQ, [n, "select"]) for n in finals]
    jobs += refs + [sel] + scores
    return {"_comment": "A34 (lane cue_selective/a34/REGISTRATION.md), rendered by tools/colab_chat/a34_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "jobs": jobs}


def gate_spec():
    """A31's gate spec with the floor: two groups a step of the 16 gate twins, lambda = mu = 1, floor 1, references from
    a34_gate_ref.jsonl (start: the start weights' own terms; knots: those less 0.5 at 0 and plus 0.5 at 1)."""
    s = A31.gate_spec()
    s["train"]["twin_loss"]["floor"] = 1.0
    s["data"]["twin_floor_ref"] = f"{U}/a34_gate_ref.jsonl"
    s["name"] = "a34_gate"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate-spec", help="also write the gate's spec here (uploaded as a34_gate_spec.json)")
    a = ap.parse_args()
    text = json.dumps(render(), indent=1) + "\n"
    if a.gate_spec:
        with open(a.gate_spec, "w") as f:
            f.write(json.dumps(gate_spec(), indent=1) + "\n")
    if a.check:
        if open(OUT).read() != text:
            sys.exit("a34_floor.json differs from its rendering")
        print("a34_floor.json matches")
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"{len(render()['jobs'])} jobs -> {OUT}")


if __name__ == "__main__":
    main()
