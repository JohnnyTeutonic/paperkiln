#!/usr/bin/env python3
"""The crossed curriculum's jobs files (the research tree's AI_ML/chat_screens_claude/cue_selective/crossed_curriculum/REGISTRATION.md,
sections 2 and 8). Claude Opus 5.5 (a fork of the coordinating session), 26 Nov 2026. A32's renderer (b2cue_jobs.py) imported
unchanged: its arm spec (S71's spec at D1), training, scoring and gate jobs, with this campaign's names, packets and starts.

    python3 tools/colab_chat/cc_jobs.py --campaign dev [--check]
    python3 tools/colab_chat/cc_jobs.py --campaign main1|main2 --lam 0.3|1.0|3.0 [--check]
    python3 tools/colab_chat/cc_jobs.py --all [--check]                  every jobs file the registration can launch
    python3 tools/colab_chat/cc_jobs.py --stage-specs DIR                the gate's spec (cc_gate_D1.json) beside the uploads

Every model starts from N250's stage-1 SFT weights (sft_tlN250.safetensors, uploaded and bound by hash at launch) or, in stage 2,
from its own stage-1 model trained on the same pod (init_job), and is S71's spec at D1 (A31's, through A32's arm_spec) with only
the seed, the packet and, for the margin arms, data.twin_examples and train.twin_loss (groups 4, margin 1 nat, lambda = mu) changed.

  dev     section 2.5's margin-weight sweep at chat seed 99: the gate; SYM1_N250_99 (S71SYM1 from sft_tlN250); from it the bridge
          plain (D0_N250_99, S71BR) and with the margin at lambda = mu in {0.3, 1.0, 3.0} (Dn_<w>_N250_99, S71BR and
          train_twins_s99.jsonl). Every model is scored on A31's early-readout requests (a31_bind's score.json, which the guards
          bind) and every stage-2 model on the development slice's margin requests (cc_dev_req.jsonl, built from dev_twins.jsonl at
          staging, five requests a group in readout.py's order), read at home.
  main1   section 8's registered models at chat seed 1, one pod: the stage-1 models L, W, S (NAT1), SYM1, FIL1 from sft_tlN250; then
  main2   from SYM1 the bridges SB and SBC, from S NB and NBC, from FIL1 FB and FBC; then WS and SCR last (section 8 cuts WS first,
          then SCR). The same at chat seed 2. The margin arms carry train_twins_s<seed>.jsonl and the chosen lambda (--lam), which
          REGISTRATION.md records as 'CHOSEN LAMBDA: <x>' before either launches (launch_cc.sh checks it).
Stage-2 jobs wait on the gate and their stage-1 model (jobs_vm skips a job whose prerequisite failed). Names are <ARM>_N250_<seed>,
which no other job, folder or model uses. No early readouts and no rung snapshots (the coverage campaign's rule).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2cue_jobs as B  # noqa: E402  (A32's renderer, unchanged)

U, C = B.U, B.C
START = f"{U}/sft_tlN250.safetensors"
EARLY_REQ = f"{U}/a31_early_req.jsonl"
DEV_REQ = f"{U}/cc_dev_req.jsonl"
LAMBDAS = (0.3, 1.0, 3.0)
DEV_SEED = 99
STAGE1 = ("L", "W", "S", "SYM1", "FIL1")          # S is NAT1 (the lead's ruling 1, 26 Nov 2026)
STAGE2 = (("SB", "SYM1", False), ("SBC", "SYM1", True), ("NB", "S", False), ("NBC", "S", True),
          ("FB", "FIL1", False), ("FBC", "FIL1", True))
LAST = ("WS", "SCR")                                # section 8's cut order: WS first, then SCR
CAMPAIGNS = {"dev": DEV_SEED, "main1": 1, "main2": 2}


def name(arm, seed):
    return f"{arm}_N250_{seed}"


def lname(lam):
    return B.lname(lam)


def spec(init_from, seed, packet, lam=None, twins=None):
    """S71's spec at D1 (b2cue_jobs.arm_spec, unchanged) on this campaign's packet; the margin arms with their seed's twin file."""
    s = B.arm_spec("D1", init_from, seed, "P" if lam is None else "N", lam, packet=packet)
    if lam is not None:
        s["data"]["twin_examples"] = f"{U}/{twins}"
    return s


def job(n, sp, after, init_job=None):
    j = B.train_job(n, sp, after=after, init_job=init_job)
    j.pop("snapshot_steps")        # no rung snapshots (the coverage campaign's rule)
    return j


def stage2(arm, seed, parent, lam, twins):
    init = f"{C}/{parent}/{parent}.safetensors"       # as jobs_vm.spec_for resolves init_job
    return job(name(arm, seed), spec(init, seed, "S71BR.jsonl", lam, twins), after=["gate", parent], init_job=parent)


def render(campaign, lam=None):
    seed = CAMPAIGNS[campaign]
    jobs = [B.gate_job(["D1"])]
    jobs[0]["cmd"] = [x.replace("b2cue_gate_D1.json", "cc_gate_D1.json") for x in jobs[0]["cmd"]]
    scores = []
    if campaign == "dev":
        twins = f"train_twins_s{seed}.jsonl"
        s1 = name("SYM1", seed)
        jobs.append(job(s1, spec(START, seed, "S71SYM1.jsonl"), after="gate"))
        scores.append(B.score_job(s1, EARLY_REQ, s1))
        for arm, w in [("D0", None)] + [(f"Dn_{lname(x)}", x) for x in LAMBDAS]:
            n = name(arm, seed)
            jobs.append(stage2(arm, seed, s1, w, twins))
            scores.append(B.score_job(n, EARLY_REQ, n))
            m = B.score_job(n, DEV_REQ, n)
            m["name"] = f"devmargin_{n}"
            scores.append(m)
    else:
        if lam not in LAMBDAS:
            raise SystemExit(f"--lam must be the sweep's chosen weight, one of {LAMBDAS}")
        twins = f"train_twins_s{seed}.jsonl"
        packets = {a: f"S71{a}.jsonl" for a in STAGE1 + LAST}
        for arm in STAGE1:
            n = name(arm, seed)
            jobs.append(job(n, spec(START, seed, packets[arm]), after="gate"))
        for arm, parent, margin in STAGE2:
            jobs.append(stage2(arm, seed, name(parent, seed), lam if margin else None, twins))
        for arm in LAST:
            n = name(arm, seed)
            jobs.append(job(n, spec(START, seed, packets[arm]), after="gate"))
        scores = [B.score_job(j["name"], EARLY_REQ, j["name"]) for j in jobs if "spec" in j]
    jobs += scores
    return {"_comment": f"The crossed curriculum, campaign {campaign}, rendered by tools/colab_chat/cc_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def gate_spec():
    """The gate's spec (b2cue_jobs.gate_spec's rule, A32's C2): the arm spec at batch 1, two steps at the full learning rate, a twin
    loss on two groups of 16 gate twins at weight 1 (mu = 1), from sft_tlN250 on S71BR."""
    s = spec(START, 1, "S71BR.jsonl", 1.0, "gate_twins.jsonl")
    s["train"].update({"batch": 1, "steps": 2, "eval_every": 1000, "checkpoint_every": 1000, "eval_examples": 1})
    s["train"]["lr_schedule"] = {"warmup_steps": 0, "decay": "none", "min_lr_frac": 0.1}
    s["train"]["twin_loss"]["groups"] = 2
    s["name"] = "cc_gate_D1"
    s["export"] = {"formats": ["safetensors"]}
    return s


def out_name(campaign, lam=None):
    tag = campaign if campaign == "dev" else f"{campaign}_{lname(lam)}"
    return os.path.join(HERE, f"cc_{tag}.json")


def variants():
    return [("dev", None)] + [(c, x) for c in ("main1", "main2") for x in LAMBDAS]


def text(campaign, lam=None):
    return json.dumps(render(campaign, lam), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=tuple(CAMPAIGNS))
    ap.add_argument("--lam", type=float)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage-specs")
    a = ap.parse_args(argv)
    if a.stage_specs:
        with open(os.path.join(a.stage_specs, "cc_gate_D1.json"), "w", newline="\n") as f:
            f.write(json.dumps(gate_spec(), indent=1) + "\n")
        print("cc_gate_D1.json written")
        return
    todo = variants() if a.all else [(a.campaign, None if a.campaign == "dev" else a.lam)]
    if not a.all and not a.campaign:
        sys.exit("--campaign or --all")
    bad = []
    for c, lam in todo:
        p, t = out_name(c, lam), text(c, lam)
        if a.check:
            if not os.path.exists(p) or open(p).read() != t:
                bad.append(os.path.basename(p))
        else:
            with open(p, "w", newline="\n") as f:
                f.write(t)
    if bad:
        sys.exit(f"differ from their rendering: {bad}")
    print(f"{len(todo)} jobs files {'match' if a.check else 'written'}")


if __name__ == "__main__":
    main()
