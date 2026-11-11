#!/usr/bin/env python3
"""The jobs files of base 2 (AI_ML/chat_screens_claude/base2/BASE2_PLAN.md sections 2 to 5, in the author's research tree).
Claude Fable 5.2, 11 Nov 2026.

    python3 tools/colab_chat/make_base2.py            # writes the files below
    python3 tools/colab_chat/make_base2.py --check    # compares them with what is on disk; non-zero on any difference

    base2_data.json        the corpus build (base2_data.sh, three stages) and a 20-step load of the real ids
    base2_N.json           arm N: D1 from scratch on the natural corpus
    base2_S.json           arm S: D1 from scratch, the same recipe with 15% scenario tokens
    base2_C.json           arm C: base A continued on S's corpus, for equal compute
    base2_rehearsal.json   the two-session rehearsal of the store path, and its uninterrupted control
    base2_rung_D1.json     the D1 shape for the rung job (arch and T only reach it)
    base2_load_probe.json  the load measurement's spec

Every file runs under boot_base2.sh with a checkpoint store (runner_multi.py --ckpt-store hf): "store_inputs" names the
sets the boot pulls and checks, a training job's "store_branch" is where the courier keeps its generations, and its
"snapshot_steps" are the marks the courier pushes to the store's main branch. Each arm's first job hashes its ids, panels,
vocabulary (and, for C, base A's weights) against base2_pins.json and training waits on it, on every VM.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UP = "/content/chat/upload"
DATA = "/content/chat/data"
V = f"{UP}/vocab_llama3_bpe.gguf"
TXN = {"MICROTORCH_CHECKPOINT_TXN": "1"}
SHAPE_A = "experiments/tiny_chat_long_2026-10/specs/cell_Sfull.json"   # base A's shape, as base A's rungs read it
SHAPE_D1 = "tools/colab_chat/base2_rung_D1.json"
A_WEIGHTS = "tc_b1_r1.safetensors"
T, ROWS = 1024, 16                  # positions a step: 16 rows of 1,024, as 8 x 2 or 16 x 1
STEPS = {"N": 250000, "S": 250000, "C": 330000}
# Section 3.2. The last mark of each arm is its end.
MARKS = {"N": [10000, 60000, 122000, 244000, 250000], "S": [10000, 60000, 122000, 244000, 250000],
         "C": [10000, 60000, 122000, 244000, 305000, 330000]}
TAIL_FROM = 0.85
CHECKPOINT_EVERY = 500
RUNG_TIMEOUT_S = 3600
# Base A's rung settings (make_tc_b1.py): Carry-5 in forwards of at most 4,096 ids, a 1,024 MiB pool beside the held trainer.
RUNG_SCORE_BATCH = 4096
RUNG_SCORE_CHUNK = 4096
RUNG_POOL_MB = 1024
PANELS = ["fineweb", "dialogue", "scenario"]
SEED_VOCAB = {"branch": "main", "prefix": "seed/vocab", "to": "chat/upload"}
SEED_ENGINE = {"branch": "main", "prefix": "seed/scenario_engine", "to": "chat/upload/scenario_engine"}
SEED_WEIGHTS = {"branch": "main", "prefix": "seed/weights", "to": "chat/upload"}
PANEL_SETS = [{"branch": "main", "prefix": "corpus/panels", "to": "chat/data"},
              {"branch": "main", "prefix": "corpus/panel_scenario", "to": "chat/data", "files": ["panel_scenario.ids"]}]


def corpus_set(arm):
    return {"branch": "main", "prefix": f"corpus/{arm}", "to": "chat/data", "files": [f"{arm}.main.ids", f"{arm}.tail.ids"]}


def params(d, layers, d_ff, vocab=16384, k=4):
    """Token embedding (shared with the head), state embedding, per layer four attention projections, three SwiGLU matrices
    and two RMSNorm gains, and the final gain."""
    return vocab * d + k * d + layers * (4 * d * d + 3 * d * d_ff + 2 * d) + d


def arch_d1():
    return {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 512, "layers": 18, "heads": 8, "d_ff": 1536},
            "state_tokens": {"k": 4, "mask": "causal"}}


def arch_a():
    return {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
            "state_tokens": {"k": 4, "mask": "causal"}}


PARAMS = {"D1": params(512, 18, 1536), "A": params(768, 8, 2240)}
assert PARAMS == {"D1": 69751296, "A": 72761088}


def spec(arm):
    """Section 3.1. N and S differ in the corpus alone; C is base A's shape on S's corpus at 16 x 1."""
    corpus = "N" if arm == "N" else "S"
    d1 = arm != "C"
    train = {"steps": STEPS[arm], "lr": 0.0005 if d1 else 0.00025, "batch": 8 if d1 else 16, "accum": 2 if d1 else 1,
             "clip": 1.0, "eval_every": 1000, "eval_windows": 32, "checkpoint_every": CHECKPOINT_EVERY, "gradmap_every": 0,
             "seed": 1, "optimizer": "adamw", "precision": "bf16", "attention_kernel": "flash",
             "lr_schedule": {"warmup_steps": 4000 if d1 else 2000, "decay": "cosine", "min_lr_frac": 0.1}}
    return {"name": f"base2_{arm}", "arch": arch_d1() if d1 else arch_a(),
            "data": {"corpus": f"{DATA}/{corpus}.main.ids", "vocab": V, "vocab_cap": 16384, "T": T, "max_tokens": 0,
                     "anneal": {"corpus": f"{DATA}/{corpus}.tail.ids", "from_fraction": TAIL_FROM}},
            "train": train, "export": {"formats": ["safetensors", "gguf"]}, "out_dir": f"/content/chat/pt_base2_{arm}"}


def rung_d1():
    s = spec("N")
    s.pop("out_dir")
    s["name"] = "base2_rung_D1"
    s["data"].pop("anneal")
    s["_shape"] = {"note": "the rung job's shape: arch and T only reach it", "params": PARAMS["D1"]}
    return s


def load_probe():
    """Twenty steps of D1 on S's real ids, no eval and no save: the trainer's start-up time and peak memory with 3.8
    billion ids held (the plan's risk 6)."""
    s = spec("S")
    s.pop("out_dir")
    s["name"] = "base2_load_probe"
    s["train"].update(steps=20, eval_every=1000, checkpoint_every=1000)
    s["train"]["lr_schedule"]["warmup_steps"] = 10
    s["export"] = {"formats": []}
    return s


def rung(job, step, tag, shape, panels, of=True, after=None):
    """The zero-rate panels and Carry-5 on the weights at a mark, with training held. of: the mark is a snapshot
    (rung_<step>/); otherwise the run's own last checkpoint, once it has ended."""
    weights = f"/content/chat/{job}/rung_{step}/model.safetensors" if of else f"/content/chat/{job}/model.safetensors"
    j = {"name": f"rung_{tag}_s{step}", "after": after or "pins", "light": True,
         "env": {"MICROTORCH_POOL_MB": RUNG_POOL_MB},
         "cmd": ["python3", "tools/colab_chat/tiny_chat_run.py", "rung", "--out", "{out}", "--spec", shape, "--export-gguf",
                 "--weights", weights, "--pt-dir", f"/content/chat/{job}", "--step", str(step), "--job", job,
                 "--cell", f"BASE2_{tag}", "--seed", "1", "--share", "28",
                 *[x for name, path in panels for x in ("--panel", f"{name}={path}")],
                 "--carry5-items", "corpus/eval/carry5_v1.jsonl",
                 "--score-batch-tokens", str(RUNG_SCORE_BATCH), "--score-chunk", str(RUNG_SCORE_CHUNK)],
         "outputs": ["rung.json"]}
    if of:
        j["rung_of"] = {"job": job, "step": step}
        j["exclusive"], j["timeout_s"] = True, RUNG_TIMEOUT_S
    return j


def pins_job(mode):
    return {"name": "pins", "light": True, "halt_on_failure": True,
            "cmd": ["python3", "tools/colab_chat/base2_pins.py", DATA, mode, "--upload", UP, "--out", "{out}/pins.json"],
            "outputs": ["pins.json"]}


def arm(a):
    job = f"pt_base2_{a}"
    corpus = "N" if a == "N" else "S"
    s = spec(a)
    marks = MARKS[a]
    panels = [(p, f"{DATA}/panel_{p}.ids") for p in PANELS]
    shape = SHAPE_A if a == "C" else SHAPE_D1
    train = {"name": job, "model_name": f"base2_{a}", "store_branch": f"ckpt-{a}", "after": ["pins"],
             "snapshot_steps": marks, "halt_on_failure": True, "env": TXN, "spec": s}
    if a == "C":
        train["init_file"] = A_WEIGHTS
    jobs = [pins_job(f"arm-{a}"), train]
    jobs += [rung(job, m, a, shape, panels) for m in marks[:-1]]
    jobs += [rung(job, marks[-1], a, shape, panels, of=False, after=[job])]
    steps = s["train"]["steps"]
    what = {"N": "D1 (18 x 512) from scratch on the natural corpus",
            "S": "D1 (18 x 512) from scratch on the scenario corpus: N's recipe with 15% scenario tokens",
            "C": "base A (8 x 768) continued from its final weights on the scenario corpus, for equal compute"}[a]
    return {"_comment": f"Base 2, arm {a}: {what}. Not admitted: the launch gate is two reviews and Jonathan's dated line.",
            "_base2": {"arm": a, "shape": "A" if a == "C" else "D1", "params": PARAMS["A" if a == "C" else "D1"],
                       "positions_per_step": ROWS * T, "steps": steps, "positions": steps * ROWS * T,
                       "tail_from_step": round(TAIL_FROM * steps), "marks": marks,
                       "mark_positions": {str(m): m * ROWS * T for m in marks}, "corpus": corpus},
            "store_inputs": [SEED_VOCAB, *([SEED_WEIGHTS] if a == "C" else []), corpus_set(corpus), *PANEL_SETS],
            "max_parallel": 1, "max_parallel_light": 1, "jobs": jobs}


def data():
    return {"_comment": "Base 2's corpus build on one L4 session taken for its processors and memory: three stages, each "
                        "pushed to the store, so a lost session repeats one at most; then a 20-step load of S's ids.",
            "store_inputs": [SEED_VOCAB, SEED_ENGINE],
            "max_parallel": 1, "max_parallel_light": 1,
            "jobs": [
                {"name": "data", "light": True, "halt_on_failure": True,
                 "cmd": ["bash", "tools/colab_chat/base2_data.sh", "{out}", V, "all"],
                 "outputs": ["base2_stage1.json", "base2_stage2.json", "base2_stage3.json", "data.ledger.json",
                             "filter.ledger.json", *[f"fineweb_edu_shard{n}_v1.sha256" for n in (2, 3, 4, 5)]]},
                {"name": "load_probe", "after": "data", "halt_on_failure": True,
                 "cmd": ["python3", "tools/colab_chat/base2_corpus.py", "loadprobe", "--spec",
                         "tools/colab_chat/base2_load_probe.json", "--out", "{out}"],
                 "outputs": ["loadprobe.json", "run.log"]},
            ]}


def rehearsal():
    """Section 5: D1, mixed precision, 8 x 2, T 1,024, 400 steps, a save every 50, a mark at step 100, on the ids of the
    50 MB FineWeb-Edu sample; then the same 400 steps without interruption. The driver stops session one between two saves."""
    s = spec("N")
    s["name"] = "base2_rehearsal"
    s["out_dir"] = "/content/chat/pt_reh"
    s["data"] = {"corpus": f"{DATA}/rehearsal.ids", "vocab": V, "vocab_cap": 16384, "T": T, "max_tokens": 0}
    s["train"].update(steps=400, checkpoint_every=50, eval_every=100)
    s["train"]["lr_schedule"]["warmup_steps"] = 40
    s["export"] = {"formats": ["safetensors"]}
    c = json.loads(json.dumps(s))
    c["name"], c["out_dir"] = "base2_rehearsal_control", "/content/chat/pt_reh_control"
    r = rung("pt_reh", 100, "R", SHAPE_D1, [("sample", f"{DATA}/rehearsal.ids")])
    r["after"] = "data"
    return {"_comment": "Base 2's rehearsal of the store path on a small corpus (BASE2_PLAN.md section 5). The control runs "
                        "after the interrupted run, in session two.",
            "store_inputs": [SEED_VOCAB],
            "max_parallel": 1, "max_parallel_light": 1,
            "jobs": [
                {"name": "data", "light": True, "halt_on_failure": True,
                 "cmd": ["bash", "tools/colab_chat/base2_data.sh", "{out}", V, "rehearsal"],
                 "outputs": ["rehearsal.json", "base2_pins_observed.json"]},
                {"name": "pt_reh", "model_name": "base2_rehearsal", "store_branch": "ckpt-R", "after": ["data"],
                 "snapshot_steps": [100], "halt_on_failure": True, "env": TXN, "spec": s},
                r,
                {"name": "pt_reh_control", "model_name": "base2_rehearsal_control", "store_branch": "ckpt-Rc",
                 "after": ["pt_reh"], "halt_on_failure": True, "env": TXN, "spec": c},
            ]}


def files():
    return {"base2_data.json": data(), "base2_N.json": arm("N"), "base2_S.json": arm("S"), "base2_C.json": arm("C"),
            "base2_rehearsal.json": rehearsal(), "base2_rung_D1.json": rung_d1(), "base2_load_probe.json": load_probe()}


def text(obj):
    return json.dumps(obj, indent=1) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    bad = []
    for name, obj in files().items():
        path = os.path.join(HERE, name)
        if check:
            try:
                same = open(path, encoding="utf-8").read() == text(obj)
            except OSError:
                same = False
            if not same:
                bad.append(name)
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text(obj))
    if check:
        print(json.dumps({"check": "equal" if not bad else "different", "files": sorted(files()), "different": bad}))
        return 1 if bad else 0
    print(json.dumps({"written": sorted(files())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
