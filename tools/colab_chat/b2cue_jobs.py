#!/usr/bin/env python3
"""A32's jobs files (b2cue; the root workspace's AI_ML/chat_screens_claude/cue_selective/b2cue/REGISTRATION_DRAFT.md, frozen at
eef513ad). Claude Opus 5.5, 15 Nov 2026. Four campaigns, each its own RunPod runner, cap and deadline:

  C0   gate; N250's SFT1500 (and its watch); the lambda sweep at seed 37 (D0, Dn at 0.1, 0.3, 1.0) with twin-scored early readouts
       on the early-readout set; the sweep scored on the selection request file; select (b2cue_select.py)
  C1a  gate; N250's start scored; N250: P, N, NA, M at seeds 1, 50, 125, P0 at 1 and 50, P's and P0's scoring of the selection file
       (the speaker cells); then S: SFT1500 (and its watch), start scored, P and N at 1, 50, 125; last, P and N at 175 and 213 on
       N250 (Decision 3, 15 Nov 2026)
  C1b  gate; A-ref's start (A's SFT1500) scored; N60: SFT1500, start scored, P, N, NA at 1, 50, 125; A-ref: P, N, NA at 1, 50, 125;
       A31's six deciding models re-scored on this run's confirmation file
  C2   gate; N2: SFT1500, start scored, P, N, NA, M at 1, 50, 125; S2: SFT1500, start scored, P and N at 1, 50, 125; last, P and N at
       175 and 213 on N2. Variants: N2 alone (`--c2 n2`)

Every arm is S71's spec (A31's) at its base's shape, from its base's SFT1500, on S74's packet (P0: P0's packet), with only these
changes: the seed, data.twin_examples and train.twin_loss for the margin arms (N: lambda = mu = L; NA: the same with
answer_mask; M: mu = 0), and the shape. Every SFT1500 is A's sft_b1_r1 recipe at the base's shape, from the base, with
train.eval_at [188, 375, 750] added (the frozen design's early readout for these jobs). Light jobs start in list order.

    python3 tools/colab_chat/b2cue_jobs.py --campaign c0 [--check]
    python3 tools/colab_chat/b2cue_jobs.py --campaign c1a|c1b|c2 --lam L [--c2 both|n2] [--no-s] [--check]
    python3 tools/colab_chat/b2cue_jobs.py --campaign ... --no-arm      (no eligible lambda: P and P0 only)
    python3 tools/colab_chat/b2cue_jobs.py --gate-specs DIR             (writes the gate's specs for every campaign)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
U = "/content/chat/upload"
C = "/content/chat"
BIN = "/content/pk/build_cuda/mtstudio"
DEVICE_TEST = "/content/pk/build_cuda/test_twin_device"

SEEDS = (1, 50, 125)
EXTRA_SEEDS = (175, 213)       # Decision 3 (Jonathan, 15 Nov 2026): five seeds for P and N on N250 and N2, run last
SWEEP_SEED = 37
GRID = (0.1, 0.3, 1.0)
MARGIN, GROUPS = 1.0, 4
EARLY = (128, 256, 512)
SFT_EVAL_AT = [188, 375, 750]
SCORE_TIMEOUT = "6600"         # A31's 2,400 s times D1's factor (1.344) times 2, rounded up
EARLY_SCORE_TIMEOUT = "2400"   # A31's 900 s, the same scaling

A_ARCH = {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
          "state_tokens": {"k": 4, "mask": "causal"}}
D1_ARCH = {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 512, "layers": 18, "heads": 8, "d_ff": 1536},
           "state_tokens": {"k": 4, "mask": "causal"}}
ARCH = {"A": A_ARCH, "D1": D1_ARCH}

S71 = {  # A31's arm spec (tools/colab_chat/a31_jobs.py, S71's spec), paths aside
    "arch": A_ARCH,
    "data": {"vocab": f"{U}/vocab.gguf", "vocab_cap": 16384, "T": 1024, "sft_examples": f"{U}/S74.jsonl"},
    "train": {"objective": "sft", "steps": 1024, "lr": 0.0001, "batch": 16, "accum": 1, "clip": 1.0, "eval_every": 32, "eval_batch": 8,
              "checkpoint_every": 128, "gradmap_every": 0, "seed": 1, "optimizer": "adamw", "init_from": None,
              "lr_schedule": {"warmup_steps": 64, "decay": "cosine", "min_lr_frac": 0.1}, "eval_examples": 256},
    "export": {"formats": ["safetensors", "gguf"]},
}
SFT1500 = {  # A's sft_b1_r1 (paperkiln ab15db97, tools/colab_chat/tc_b1_winner_sft_20261109.json), paths aside, eval_at added
    "data": {"vocab": f"{U}/vocab.gguf", "vocab_cap": 16384, "T": 1024, "sft_examples": f"{U}/sft_human.jsonl"},
    "train": {"objective": "sft", "steps": 1500, "lr": 0.0003, "batch": 16, "accum": 1, "clip": 1.0, "eval_every": 500, "eval_batch": 8,
              "checkpoint_every": 25, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
              "lr_schedule": {"warmup_steps": 100, "decay": "cosine", "min_lr_frac": 0.1}, "eval_at": SFT_EVAL_AT},
    "export": {"formats": ["safetensors", "gguf"]},
}
SHAPE = {"N250": "D1", "N60": "D1", "S": "D1", "N2": "D1", "S2": "D1", "Aref": "A"}


def lname(lam):
    return str(lam).replace(".", "p")


def arm_spec(shape, init_from, seed, arm, lam=None, packet="S74.jsonl", dev=False):
    s = json.loads(json.dumps(S71))
    s["arch"] = json.loads(json.dumps(ARCH[shape]))
    s["train"]["seed"] = seed
    s["train"]["init_from"] = init_from
    s["data"]["sft_examples"] = f"{U}/{packet}"
    if arm in ("N", "NA", "M", "Dn"):
        s["data"]["twin_examples"] = f"{U}/train_twins.jsonl"
        mu = 0.0 if arm == "M" else lam
        s["train"]["twin_loss"] = {"groups": GROUPS, "margin": MARGIN, "lambda": lam, "mu": mu}
        if arm == "NA":
            s["train"]["twin_loss"]["answer_mask"] = True
    if dev:
        s["export"] = {"formats": ["safetensors"]}
    return s


def sft1500_spec(shape, base_file):
    s = json.loads(json.dumps(SFT1500))
    s["arch"] = json.loads(json.dumps(ARCH[shape]))
    s["train"]["init_from"] = f"{U}/{base_file}"
    return s


def early_jobs(name, ref, scored):
    """The early readouts of one 1,024-step job (b2cue_early.py) at 12.5%, 25% and 50%: the health rule, then (C0's sweep only) the
    twin-scored readout on the early-readout set."""
    cmd = ["python3", "tools/colab_chat/b2cue_early.py", "--job", name, "--ref", ref, "--out", "{out}", "--chat", C, "--bin", BIN]
    if scored:
        cmd += ["--score-timeout", EARLY_SCORE_TIMEOUT]
    else:
        cmd += ["--no-score"]
    return [{"name": f"early_{name}_{s}", "light": True, "rung_of": {"job": name, "step": s}, "cmd": cmd[:4] + ["--step", str(s)] + cmd[4:],
             "outputs": ["early.json"]} for s in EARLY]


def train_job(name, spec, after="gate", init_job=None):
    j = {"name": name, "spec": spec, "after": after, "relay_every_steps": 100000, "snapshot_steps": list(EARLY)}
    if init_job:
        j["init_job"] = init_job
    return j


def last_heavy(jobs):
    """The last training job listed so far: the heavy pool runs in list order, so a watch that waits on it starts with its SFT1500."""
    heavy = [j["name"] for j in jobs if "spec" in j]
    return heavy[-1] if heavy else "gate"


def sft_jobs(base, shape, base_file, watch_after="gate"):
    """A base's SFT1500 and its watch (b2cue_early.py --watch: the non-finite stop at once, and the relayed readouts at 188, 375, 750).
    The watch starts when the training job listed before the SFT1500 has ended (after_terminal), so it holds a light slot only while
    its SFT1500 runs."""
    name = f"sft_{base}"
    j = {"name": name, "spec": sft1500_spec(shape, base_file), "after": "gate", "relay_every_steps": 100000,
         "env": {"MICROTORCH_CHECKPOINT_TXN": "1"}}
    w = {"name": f"watch_{name}", "light": True, "after": ["gate"] + ([watch_after] if watch_after != "gate" else []), "after_terminal": True,
         "cmd": ["python3", "tools/colab_chat/b2cue_early.py", "--watch", "--job", name, "--out", "{out}", "--chat", C,
                 "--at", ",".join(map(str, SFT_EVAL_AT)), "--steps", "1500"],
         "outputs": ["watch.json"]}
    return [j, w]


def score_job(name, req, after, weights=None, spec_path=None, gguf=True):
    """`mtstudio score` of a job's final weights (or an uploaded model), then a31_bind.py's score.json (A31's binding, unchanged)."""
    if weights is None:
        model_dir, w, sp = f"{C}/{name}", f"{C}/{name}/{name}.safetensors", f"{C}/{name}/spec.json"
        pre = ""
    else:   # an uploaded model: linked into the job's own folder with its spec
        model_dir, w, sp = "{out}/m", weights, "{out}/m/spec.json"
        mn = json.load(open(spec_path))["name"] if spec_path and os.path.exists(spec_path) else name
        pre = f"mkdir -p {{out}}/m && ln -sf {weights} {{out}}/m/{mn}.safetensors && cp {spec_path} {{out}}/m/spec.json && "
    g = f" --gguf {C}/{name}/{name}.gguf" if (gguf and weights is None) else ""
    bind = (f"python3 tools/colab_chat/a31_bind.py --scores {{out}}/scores.jsonl --weights {w}{g} --requests {req} --bin {BIN} "
            f"--spec {sp} --out {{out}}/score.json")
    return {"name": f"score_{name}" if weights is None else name, "light": True, "after": after,
            "cmd": ["bash", "-c", f"{pre}timeout {SCORE_TIMEOUT} {BIN} score {model_dir} --in {req} --out {{out}}/scores.jsonl "
                    f"--batch-tokens 8192 && {bind}"],
            "outputs": ["scores.jsonl", "score.json"]}


def start_score(base, after, req):
    """A base's SFT1500 start scored on the confirmation file: from the job's own output when it trained in this campaign, else the
    uploaded weights (named start_<base>.safetensors, with its spec start_<base>_spec.json)."""
    if after == f"sft_{base}":
        j = score_job(f"sft_{base}", req, after)
        j["name"] = f"score_start_{base}"
        return j
    return score_job(f"score_start_{base}", req, after, weights=f"{U}/start_{base}.safetensors", spec_path=f"{U}/start_{base}_spec.json")


def gate_job(shapes):
    return {"name": "gate", "cmd": ["python3", "tools/colab_chat/b2cue_gate.py", "--out", "{out}", "--bin", BIN, "--device-test", DEVICE_TEST]
            + [x for sh in shapes for x in ("--spec", f"{U}/b2cue_gate_{sh}.json")], "outputs": ["gate.json"]}


def arms(base, lam, seeds, which, init_job=None, init_from=None, req=None, packet_p0=False):
    """Training, early and scoring jobs of one base's arms at the given seeds, P first at every seed (P is the health reference)."""
    shape = SHAPE[base]
    init = init_from or f"{C}/sft_{base}/sft_{base}.safetensors"
    jobs, scores = [], []
    order = [a for a in ("P", "N", "NA", "M") if a in which]
    for arm in order:
        for sd in seeds:
            name = f"{arm}_{base}_{sd}"
            spec = arm_spec(shape, init, sd, arm, None if arm == "P" else lam)
            j = train_job(name, spec, after=["gate"] + ([init_job] if init_job else []))
            jobs += [j] + early_jobs(name, name if arm == "P" else f"P_{base}_{sd}", scored=False)
            scores.append(score_job(name, req, name))
    if packet_p0:
        for sd in (1, 50):
            name = f"P0_{base}_{sd}"
            spec = arm_spec(shape, init, sd, "P", packet="P0.jsonl")
            jobs += [train_job(name, spec, after=["gate"] + ([init_job] if init_job else []))] + early_jobs(name, name, scored=False)
            scores.append(score_job(name, req, name))
    return jobs, scores


def extra_seed_jobs(base, lam, req, init_job=None):
    """Decision 3's seeds 175 and 213 for P and N, last in the campaign; they start only after every other job has ended and its
    score files are home (b2cue_early.py --barrier runs first, a light job the extra seeds' training waits on)."""
    jobs, scores = [], []
    for arm in ("P", "N"):
        for sd in EXTRA_SEEDS:
            name = f"{arm}_{base}_{sd}"
            spec = arm_spec(SHAPE[base], f"{C}/sft_{base}/sft_{base}.safetensors", sd, arm, None if arm == "P" else lam)
            jobs += [train_job(name, spec, after=["gate", "barrier"] + ([init_job] if init_job else []))] + \
                early_jobs(name, name if arm == "P" else f"P_{base}_{sd}", scored=False)
            scores.append(score_job(name, req, name))
    return jobs, scores


def barrier_job(others):
    return {"name": "barrier", "light": True, "after_terminal": True, "after": others,
            "cmd": ["python3", "tools/colab_chat/b2cue_early.py", "--barrier", "--out", "{out}", "--chat", C, "--jobs", ",".join(others)],
            "outputs": ["barrier.json"]}


def render(campaign, lam=None, c2="both", no_s=False, no_arm=False):
    CONF = f"{U}/b2cue_confirmation_req.jsonl"
    SEL = f"{U}/b2cue_selection_req.jsonl"
    jobs, scores = [], []
    which = ("P",) if no_arm else ("P", "N", "NA", "M")
    if campaign == "c0":
        jobs.append(gate_job(["D1"]))
        jobs += sft_jobs("N250", "D1", "base_N250.safetensors")
        sweep = [("D0", None)] + [(f"Dn_{lname(x)}", x) for x in GRID]
        for n, x in sweep:
            spec = arm_spec("D1", None, SWEEP_SEED, "P" if x is None else "Dn", x, dev=True)
            jobs += [train_job(n, spec, after=["gate", "sft_N250"], init_job="sft_N250")] + early_jobs(n, "D0", scored=True)
            scores.append(score_job(n, SEL, n, gguf=False))
        sel = {"name": "select", "light": True, "after_terminal": True,
               "after": [n for n, _ in sweep] + [f"score_{n}" for n, _ in sweep] + [f"early_{n}_{s}" for n, _ in sweep for s in EARLY],
               "cmd": ["python3", "tools/colab_chat/b2cue_select.py", "--chat", C, "--out", "{out}"], "outputs": ["decisions.json"]}
        jobs += scores + [sel]
    elif campaign == "c1a":
        jobs.append(gate_job(["D1"]))
        jobs.append(start_score("N250", "gate", CONF))
        a, s = arms("N250", lam, SEEDS, which, req=CONF, init_from=f"{U}/start_N250.safetensors", packet_p0=True)
        jobs += a
        scores += s
        for sd in (1, 50):   # the speaker cells: P and P0 on the selection file, after the selection has used it
            for arm in ("P", "P0"):
                n = f"{arm}_N250_{sd}"
                j = score_job(n, SEL, n)
                j["name"] = f"score_spk_{n}"
                scores.append(j)
        if not no_s:
            jobs += sft_jobs("S", "D1", "base_S.safetensors", last_heavy(jobs))
            jobs.append(start_score("S", "sft_S", CONF))
            a, s = arms("S", lam, SEEDS, ("P",) if no_arm else ("P", "N"), init_job="sft_S", req=CONF)
            jobs += a
            scores += s
        jobs += scores
        if not no_arm:
            others = [j["name"] for j in jobs]
            jobs.append(barrier_job(others))
            a, s = extra_seed_jobs("N250", lam, CONF)
            for j in a:
                if j.get("spec"):
                    j["spec"]["train"]["init_from"] = f"{U}/start_N250.safetensors"
            jobs += a + s
    elif campaign == "c1b":
        jobs.append(gate_job(["D1", "A"]))
        jobs.append(start_score("Aref", "gate", CONF))
        jobs += sft_jobs("N60", "D1", "base_N60.safetensors")
        jobs.append(start_score("N60", "sft_N60", CONF))
        w = ("P",) if no_arm else ("P", "N", "NA")
        a, s = arms("N60", lam, SEEDS, w, init_job="sft_N60", req=CONF)
        jobs += a
        scores += s
        a, s = arms("Aref", lam, SEEDS, w, init_from=f"{U}/start_Aref.safetensors", req=CONF)
        jobs += a
        scores += s
        for j in ("P_1", "P_50", "M_1_0p3", "M_50_0p3", "N_1_0p3", "N_50_0p3"):   # A31's deciding models, trained on 2.1 data
            scores.append(score_job(f"score_a31_{j}", CONF, "gate", weights=f"{U}/a31_{j}.safetensors", spec_path=f"{U}/a31_spec_{j}.json"))
        jobs += scores
    elif campaign == "c2":
        jobs.append(gate_job(["D1"]))
        jobs += sft_jobs("N2", "D1", "base_N2.safetensors")
        jobs.append(start_score("N2", "sft_N2", CONF))
        a, s = arms("N2", lam, SEEDS, which, init_job="sft_N2", req=CONF)
        jobs += a
        scores += s
        if c2 == "both":
            jobs += sft_jobs("S2", "D1", "base_S2.safetensors", last_heavy(jobs))
            jobs.append(start_score("S2", "sft_S2", CONF))
            a, s = arms("S2", lam, SEEDS, ("P",) if no_arm else ("P", "N"), init_job="sft_S2", req=CONF)
            jobs += a
            scores += s
        jobs += scores
        if not no_arm:
            others = [j["name"] for j in jobs]
            jobs.append(barrier_job(others))
            a, s = extra_seed_jobs("N2", lam, CONF, init_job="sft_N2")
            jobs += a + s
    else:
        raise SystemExit(f"unknown campaign {campaign}")
    return {"_comment": f"A32 (b2cue) campaign {campaign}, rendered by tools/colab_chat/b2cue_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def gate_spec(shape):
    """The gate's spec at one shape: the arm spec at batch 1, two steps at the full learning rate, a twin loss on two groups of the 16
    gate twins at weight 1 (mu = 1), from the campaign's first uploaded model of that shape (b2cue_gate.py swaps answer_mask in and
    the twin term out for its legs, and runs one full-batch step on CUDA)."""
    init = {"D1": f"{U}/b2cue_gate_init_D1.safetensors", "A": f"{U}/start_Aref.safetensors"}[shape]
    s = arm_spec(shape, init, 1, "N", 1.0)
    s["train"].update({"batch": 1, "steps": 2, "eval_every": 1000, "checkpoint_every": 1000, "eval_examples": 1})
    s["data"]["twin_examples"] = f"{U}/gate_twins.jsonl"
    s["train"]["lr_schedule"] = {"warmup_steps": 0, "decay": "none", "min_lr_frac": 0.1}
    s["train"]["twin_loss"]["groups"] = 2
    s["name"] = f"b2cue_gate_{shape}"
    s["export"] = {"formats": ["safetensors"]}
    return s


def out_name(campaign, lam, c2, no_s, no_arm):
    tag = campaign if campaign == "c0" else f"{campaign}_{'noarm' if no_arm else lname(lam)}"
    if campaign == "c2" and c2 == "n2":
        tag += "_n2only"
    if campaign == "c1a" and no_s:
        tag += "_nos"
    return os.path.join(HERE, f"b2cue_{tag}.json")


def variants():
    """Every jobs file the registration can launch: C0; for each lambda in the grid and for no_arm, C1a (with and without S), C1b,
    C2 (both bases, and N2 alone)."""
    out = [("c0", None, "both", False, False)]
    for lam, no_arm in [(x, False) for x in GRID] + [(None, True)]:
        out += [("c1a", lam, "both", False, no_arm), ("c1a", lam, "both", True, no_arm), ("c1b", lam, "both", False, no_arm),
                ("c2", lam, "both", False, no_arm), ("c2", lam, "n2", False, no_arm)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="render (or with --check, check) every variant")
    ap.add_argument("--campaign", choices=("c0", "c1a", "c1b", "c2"))
    ap.add_argument("--lam", type=float)
    ap.add_argument("--c2", choices=("both", "n2"), default="both")
    ap.add_argument("--no-s", action="store_true")
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate-specs", help="write b2cue_gate_D1.json and b2cue_gate_A.json here")
    a = ap.parse_args()
    if a.gate_specs:
        for sh in ("D1", "A"):
            with open(os.path.join(a.gate_specs, f"b2cue_gate_{sh}.json"), "w") as f:
                f.write(json.dumps(gate_spec(sh), indent=1) + "\n")
        print("gate specs written")
        if not a.campaign:
            return
    if a.all:
        bad = []
        for v in variants():
            text = json.dumps(render(*v), indent=1) + "\n"
            p = out_name(v[0], v[1], v[2], v[3], v[4])
            if a.check:
                if not os.path.exists(p) or open(p).read() != text:
                    bad.append(os.path.basename(p))
            else:
                with open(p, "w") as f:
                    f.write(text)
        if bad:
            sys.exit(f"differ from their rendering: {bad}")
        print(f"{len(variants())} jobs files {'match' if a.check else 'written'}")
        return
    if a.campaign != "c0" and not a.no_arm and a.lam not in GRID:
        sys.exit("--lam must be the selection's lambda, one of " + str(GRID))
    text = json.dumps(render(a.campaign, a.lam, a.c2, a.no_s, a.no_arm), indent=1) + "\n"
    p = out_name(a.campaign, a.lam, a.c2, a.no_s, a.no_arm)
    if a.check:
        if open(p).read() != text:
            sys.exit(f"{os.path.basename(p)} differs from its rendering")
        print(f"{os.path.basename(p)} matches")
        return
    with open(p, "w") as f:
        f.write(text)
    print(f"{len(json.loads(text)['jobs'])} jobs -> {p}")


if __name__ == "__main__":
    main()
