#!/usr/bin/env python3
"""VM side of the tiny-chat experiments (experiments/tiny_chat_2026-10/PREREGISTRATION.md): the
data-job commands that tiny_chat_*.json run through jobs_vm.py, plus `choose` at home.

    tiny_chat_run.py probe  --out D --spec SHAPE.json --corpus C.ids --steps 300
    tiny_chat_run.py screen --out D --spec SHAPE.json --corpus C.ids --lr X --steps 1500 --seed 101
    tiny_chat_run.py post   --out D --spec SHAPE.json --pt-weights W --pt-dir P --sft-template SFT.json
                            --seed S --cell C --share N --panel NAME=IDS ... --carry5-items ITEMS
    tiny_chat_run.py pick   --out D --run POST_DIR [--run ...]
    tiny_chat_run.py gate   --out D --jobs JOBS.json --choice LR_CHOICE.json
    tiny_chat_run.py choose --art ART_DIR --out LR_CHOICE.json          (at home, after Stage 0)
    tiny_chat_run.py rung   --out D --spec SHAPE.json --weights SNAP/model.safetensors --pt-dir P --step N
                            --job J --cell C --seed S --share N --panel NAME=IDS ... --carry5-items ITEMS
    tiny_chat_run.py asha_train   --out D --spec SHAPE.json --corpus C.ids --budget N --lr X --warmup-frac F
                                  --seed S --share N --config ID --round R --panel ... --carry5-items ITEMS
    tiny_chat_run.py asha_promote --out D --plan PLAN.json --round R [--state STATE.json]

rung, asha_train and asha_promote serve the early-stopping study (PREREGISTRATION.md amendment 1):
rung scores a pretraining run's weights at a rung (5/10/25/50% of its steps; jobs_vm.py keeps the
snapshot) with the zero-rate panels and base-model Carry-5 teacher-forced; asha_train trains one
successive-halving candidate at one budget and scores it the same way; asha_promote applies the
registered promotion rule (promote_rank) between rounds.

probe: the shape alone on the GPU for a few hundred steps; steady tokens/s (timed between step 50
and the last step from mtstudio's own step events) and peak GPU memory. screen: one lr of the
registered screen; final validation loss. post: SFT from the pretrained weights, the zero-rate
panel passes (1 step at lr 0 from the weights: the panel's validation loss, weights untouched),
Carry-5 on the SFT model (and, tolerated if it fails, on the base). pick: the cell's best seed by
the primary metric, its SFT model copied out. gate: refuses (exit 1) unless every pretraining job
in JOBS uses the learning rate Stage 0 chose for its shape and the Carry-5 benchmark is present, so
an E launch cannot train on a provisional rate or without its primary metric.
Every mode writes summary.json (gate: gate.json; pick: best.json) in --out; reruns are idempotent
(finished stages are skipped, an interrupted mtstudio run resumes from its own checkpoint).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")
CARRY5_TOOL = os.environ.get("TINY_CHAT_CARRY5", "tools/carry5_eval.py")  # override: tests only
SCREEN_TIE = 0.01  # nats: screens closer than this to the best count as tied; the smaller lr wins


# ---------------------------------------------------------------- the Carry-5 interface
CARRY5_MODES = ("gold", "self", "twin", "noctx")
CARRY5_CHANCE = 0.25  # four candidates per item; carry5_eval's uniform scorer is exactly this


def carry5_argv(spec, items, out_dir, gen, port):
    """tools/carry5_eval.py (benchmark agent, 23 Oct 2026): `run` scores the model named by SPEC
    (out_dir/name.safetensors) with `mtstudio score` and, with --gen mtstudio, plays the self
    history through `mtstudio chat` on PORT (one port per job: post jobs share a VM). Writes
    OUT_DIR/summary.json = {"meta", "models": {"m": {"fc": {"acc"|"acc_norm": {mode: {"all":
    {"mean", "n", "ci95"}, "family", "distance", "interference", ...}}}, "gen", "decay", ...}}}.
    Reconcile HERE if its flags change; only this function and carry5_read depend on them."""
    return [sys.executable, CARRY5_TOOL, "run", "--model", f"m={spec}", "--out", out_dir, "--scorer", "mtstudio",
            "--mtstudio", BIN, "--gen", gen, "--set", items, "--port", str(port)]


def carry5_read(out_dir):
    """The model's summary plus the headline means: acc and acc_norm per history mode."""
    s = jload(os.path.join(out_dir, "summary.json"))
    m = s["models"]["m"]
    return {"meta": s["meta"], "chance": CARRY5_CHANCE,
            "acc": {k: m["fc"]["acc"][k]["all"]["mean"] for k in CARRY5_MODES},
            "acc_norm": {k: m["fc"]["acc_norm"][k]["all"]["mean"] for k in CARRY5_MODES},
            "summary": m}


def primary(c5):
    """The registered primary metric: forced-choice accuracy (acc), SELF history, multi-turn minus twin."""
    try:
        return float(c5["acc"]["self"]) - float(c5["acc"]["twin"])
    except (KeyError, TypeError, ValueError):
        return None


def port_for(path):
    return 8100 + int(hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:6], 16) % 800


# ---------------------------------------------------------------- helpers
def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsave(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def events(out_dir):
    p = os.path.join(out_dir, "events.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def shape_spec(path):
    s = jload(path)
    s.pop("_shape", None)
    return s


def batching(layers):
    return (16, 2) if layers >= 16 else (32, 1)


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def concurrent_mtstudio():
    """Other mtstudio processes on the VM (jobs share the GPU; throughput is read with this beside it)."""
    n = 0
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if d.isdigit():
            try:
                if os.path.basename(open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")[0]) == b"mtstudio":
                    n += 1
            except (OSError, IndexError):
                pass
    return n


class GpuPeak(threading.Thread):
    """Peak GPU memory in use (MiB, whole GPU) while a probe runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=20).stdout
                self.peak = max([self.peak] + [int(x) for x in out.split() if x.strip().isdigit()])
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            self.stop.wait(2.0)


def run_mtstudio(spec, out_dir, log_path, t_from=None):
    """mtstudio run SPEC with out_dir; stdout to log_path. Returns (rc, steady tokens/s or None): the
    rate between step t_from and the last step, timed here on mtstudio's own step lines."""
    spec = dict(spec, out_dir=out_dir)
    os.makedirs(os.path.dirname(out_dir.rstrip("/")) or ".", exist_ok=True)
    sp = out_dir.rstrip("/") + "_spec.json"
    jsave(sp, spec)
    tr = spec["train"]
    per_step = tr.get("batch", 1) * tr.get("accum", 1) * spec["data"]["T"]
    marks = {}
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen([BIN, "run", sp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            if t_from is not None and line.startswith('{"event":"step"'):
                try:
                    st = json.loads(line)["step"]
                except (ValueError, KeyError):
                    continue
                now = time.monotonic()  # mtstudio flushes every event line
                if st >= t_from:
                    marks.setdefault("first", (st, now))
                marks["last"] = (st, now)
        rc = p.wait()
    rate = None
    if "first" in marks and marks["last"][0] > marks["first"][0]:
        (s0, t0), (s1, t1) = marks["first"], marks["last"]
        rate = (s1 - s0) * per_step / max(t1 - t0, 1e-9)
    return rc, rate


def final_eval(out_dir):
    ev = [e for e in events(out_dir) if e.get("event") == "eval"]
    return ev[-1]["val_loss"] if ev else None


# ---------------------------------------------------------------- modes
def cmd_probe(a):
    spec = shape_spec(a.spec)
    spec["data"]["corpus"] = a.corpus
    spec["train"].update({"steps": a.steps, "eval_every": a.steps, "eval_windows": 1, "checkpoint_every": 10 ** 7})
    spec["export"] = {"formats": []}
    run_dir = os.path.join(a.out, "run")
    shutil.rmtree(run_dir, ignore_errors=True)  # a probe always measures a fresh run
    others = concurrent_mtstudio()
    mon = GpuPeak()
    mon.start()
    rc, rate = run_mtstudio(spec, run_dir, os.path.join(a.out, "mtstudio.log"), t_from=50)
    mon.stop.set()
    mon.join(5)
    res = jload(os.path.join(run_dir, "result.json")) if os.path.exists(os.path.join(run_dir, "result.json")) else {}
    s = {"mode": "probe", "spec": a.spec, "rc": rc, "steps": a.steps, "arch": spec["arch"]["custom"],
         "tokens_per_step": spec["train"]["batch"] * spec["train"].get("accum", 1) * spec["data"]["T"],
         "tok_per_s_steady": rate, "tok_per_s_result": res.get("tokens_per_second"),
         "wall_seconds": res.get("wall_seconds"), "peak_gpu_mem_mib": mon.peak, "gpu": gpu_name(),
         "other_mtstudio_at_start": others, "params": res.get("params")}
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps(s))
    return 0 if rc == 0 else 1


def cmd_screen(a):
    spec = shape_spec(a.spec)
    spec["data"]["corpus"] = a.corpus
    spec["train"].update({"steps": a.steps, "lr": a.lr, "seed": a.seed, "eval_every": a.steps, "eval_windows": 128,
                          "checkpoint_every": 500,
                          "lr_schedule": {"warmup_steps": max(1, a.steps // 10), "decay": "cosine",
                                          "min_lr_frac": 0.1}})
    spec["export"] = {"formats": []}
    run_dir = os.path.join(a.out, "run")
    rc, rate = run_mtstudio(spec, run_dir, os.path.join(a.out, "mtstudio.log"), t_from=50)
    ev = events(run_dir)
    losses = [e["loss"] for e in ev if e.get("event") == "step"]
    val = final_eval(run_dir)
    bad = val is None or not math.isfinite(val) or (losses and val > losses[0])
    s = {"mode": "screen", "spec": a.spec, "rc": rc, "lr": a.lr, "seed": a.seed, "steps": a.steps,
         "final_val": val, "diverged": bool(bad), "first_loss": losses[0] if losses else None,
         "last_loss": losses[-1] if losses else None, "tok_per_s_steady": rate,
         "other_mtstudio_at_end": concurrent_mtstudio(), "gpu": gpu_name()}
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps(s))
    return 0 if rc == 0 else 1


def zero_rate_panels(shape, weights, out, panel_kvs):
    """{name: nats/token} for each NAME=IDS panel: 1 step at lr 0 from the weights, 512 held-out
    windows (the weights stay byte-identical). Finished panels are not rerun."""
    res = {}
    for kv in panel_kvs:
        name, path = kv.split("=", 1)
        d = os.path.join(out, f"panel_{name}")
        if not os.path.exists(os.path.join(d, "result.json")):
            shutil.rmtree(d, ignore_errors=True)  # init_from applies to fresh runs only
            spec = {"name": f"panel_{name}", "arch": shape["arch"],
                    "data": {**shape["data"], "corpus": path, "max_tokens": 0},
                    "train": {"steps": 1, "lr": 0.0, "batch": 1, "accum": 1, "eval_every": 1, "eval_windows": 512,
                              "checkpoint_every": 10 ** 7, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
                              "init_from": weights},
                    "export": {"formats": []}}
            rc, _ = run_mtstudio(spec, d, os.path.join(out, "mtstudio.log"))
            if rc != 0:
                raise SystemExit(f"panel {name} failed (rc {rc})")
        res[name] = final_eval(d)
    return res


def carry5_base(shape, weights, items, out, required=False):
    """Carry-5 on a base (pretrained, not SFT) model, teacher-forced: gold, twin and noctx forced
    choice, no generation (--gen none). A failure is recorded, or raised when required."""
    base_spec = os.path.join(out, "base_spec.json")
    jsave(base_spec, {**shape, "name": os.path.splitext(os.path.basename(weights))[0],
                      "out_dir": os.path.dirname(os.path.abspath(weights))})
    c5b = os.path.join(out, "carry5_base")
    pb = subprocess.run(carry5_argv(base_spec, items, c5b, "none", port_for(c5b)), capture_output=True, text=True)
    open(os.path.join(out, "carry5_base.log"), "w", encoding="utf-8").write(pb.stdout[-20000:] + pb.stderr[-20000:])
    if pb.returncode == 0 and os.path.exists(os.path.join(c5b, "summary.json")):
        return carry5_read(c5b)
    if required:
        raise SystemExit(f"carry5_eval (base) failed (rc {pb.returncode}): {pb.stderr[-500:]}")
    return {"error": pb.stderr[-500:]}


def rung_metrics(c5b):
    """The rung predictors (amendment 1): base-model forced-choice accuracy with gold history
    (acc_gold) and Y_base = acc_gold - acc_twin, both teacher-forced."""
    try:
        g, t = float(c5b["acc"]["gold"]), float(c5b["acc"]["twin"])
    except (KeyError, TypeError, ValueError):
        return {"acc_gold": None, "Y_base": None}
    return {"acc_gold": g, "Y_base": g - t}


def preflight(items):
    missing = [p for p in (CARRY5_TOOL, items) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Carry-5 is not in this commit: missing {missing}")


def cmd_post(a):
    done = os.path.join(a.out, "summary.json")
    if os.path.exists(done) and jload(done).get("primary") is not None:
        print(f"{done}: present")
        return 0
    preflight(a.carry5_items)
    shape = shape_spec(a.spec)
    custom = shape["arch"]["custom"]
    b, acc = batching(custom["layers"])
    if not os.path.exists(a.pt_weights):
        raise SystemExit(f"pretrained weights not found: {a.pt_weights}")
    s = {"mode": "post", "cell": a.cell, "share": a.share, "seed": a.seed, "spec": a.spec, "arch": custom,
         "shape": jload(a.spec).get("_shape"), "pt_lr": a.pt_lr if a.pt_lr is not None else shape["train"]["lr"],
         "pt_weights_sha256": sha256(a.pt_weights),
         "gpu": gpu_name()}
    # Pretraining receipts, when this VM ran the job (on a fresh VM they are home already).
    if os.path.exists(os.path.join(a.pt_dir, "result.json")):
        s["pt_result"] = jload(os.path.join(a.pt_dir, "result.json"))
    s["pt_evals"] = [(e["step"], e["val_loss"]) for e in events(a.pt_dir) if e.get("event") == "eval"]
    # 1. Panels: the zero-rate pass (1 step at lr 0 from the weights) reads the panel's held-out windows.
    s["panels"] = zero_rate_panels(shape, a.pt_weights, a.out, a.panel)
    # 2. SFT from the pretrained weights (resumes from its own checkpoint on this VM).
    sft = jload(a.sft_template)
    sft.pop("_comment", None)
    model = f"tc_sft_{a.cell}_r{a.seed}"
    sd = os.path.join(a.out, "sft")
    spec = {"name": model, "arch": shape["arch"], "data": sft["data"],
            "train": {**sft["train"], "batch": b, "accum": acc, "seed": a.seed, "init_from": a.pt_weights},
            "export": sft["export"]}
    t0 = time.time()
    if not os.path.exists(os.path.join(sd, "result.json")):
        rc, rate = run_mtstudio(spec, sd, os.path.join(a.out, "mtstudio.log"), t_from=20)
        if rc != 0:
            raise SystemExit(f"SFT failed (rc {rc})")
        s["sft_tok_per_s_steady"] = rate
    s["sft_seconds"] = round(time.time() - t0, 1)
    s["sft_result"] = jload(os.path.join(sd, "result.json"))
    s["sft_evals"] = [(e["step"], e["val_loss"]) for e in events(sd) if e.get("event") == "eval"]
    sw = os.path.join(sd, model + ".safetensors")
    s["sft_weights_sha256"] = sha256(sw)
    s["sft_data_sha256"] = sha256(sft["data"]["sft_examples"])
    # 3. Carry-5: the SFT model with generated (self) history (required), then the base model
    #    teacher-forced only (descriptive; a failure is recorded, not fatal).
    c5 = os.path.join(a.out, "carry5")
    p = subprocess.run(carry5_argv(sd + "_spec.json", a.carry5_items, c5, "mtstudio", port_for(a.out)),
                       capture_output=True, text=True)
    open(os.path.join(a.out, "carry5.log"), "w", encoding="utf-8").write(p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0 or not os.path.exists(os.path.join(c5, "summary.json")):
        raise SystemExit(f"carry5_eval failed (rc {p.returncode}): {p.stderr[-500:]}")
    s["carry5"] = carry5_read(c5)
    s["primary"] = primary(s["carry5"])
    s["carry5_base"] = carry5_base(shape, a.pt_weights, a.carry5_items, a.out)
    s["final_rung"] = rung_metrics(s["carry5_base"])  # the 100% rung of the early-ranking analysis
    s["other_mtstudio_at_end"] = concurrent_mtstudio()
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps({k: s[k] for k in ("cell", "seed", "primary", "panels")}))
    return 0


def cmd_pick(a):
    rows = []
    for d in a.run:
        p = os.path.join(d, "summary.json")
        sm = jload(p) if os.path.exists(p) else {}
        rows.append({"run": d, "seed": sm.get("seed"), "primary": sm.get("primary"), "summary": os.path.exists(p)})
    ok = [r for r in rows if r["primary"] is not None]
    best = max(ok, key=lambda r: (r["primary"], -r["seed"])) if ok else None
    out = {"rule": "highest primary metric (forced choice, self history, multi minus twin); tie -> lower seed",
           "runs": rows, "best": best}
    rc = 1
    if best:
        sd = os.path.join(best["run"], "sft")
        w = glob.glob(os.path.join(sd, "*.safetensors"))
        w = [x for x in w if os.path.basename(x) != "model.safetensors"]
        g = [x for x in glob.glob(os.path.join(sd, "*.gguf"))]
        if w and g and os.path.exists(sd + "_spec.json"):
            shutil.copyfile(w[0], os.path.join(a.out, "model.safetensors"))
            shutil.copyfile(g[0], os.path.join(a.out, "model.gguf"))
            shutil.copyfile(sd + "_spec.json", os.path.join(a.out, "spec.json"))
            out["weights_sha256"] = sha256(os.path.join(a.out, "model.safetensors"))
            rc = 0
        else:
            out["weights"] = "lost (VM replaced after the post job); rerun that post job for the model"
    jsave(os.path.join(a.out, "best.json"), out)
    print(json.dumps(out))
    return rc


def ckpt_step(d):
    try:
        return int(open(os.path.join(d, "state.txt")).readline().strip() or 0)
    except (OSError, ValueError):
        return None


def cmd_rung(a):
    """A pretraining run's weights at a rung (jobs_vm.py's snapshot at that step): the zero-rate
    panels and base-model Carry-5 (teacher-forced), plus the run's validation loss at that step."""
    done = os.path.join(a.out, "rung.json")
    if os.path.exists(done) and jload(done).get("acc_gold") is not None:
        print(f"{done}: present")
        return 0
    preflight(a.carry5_items)
    if not os.path.exists(a.weights):
        raise SystemExit(f"rung snapshot not found: {a.weights}")
    snap = ckpt_step(os.path.dirname(a.weights))
    if snap != a.step:
        raise SystemExit(f"{a.weights} holds step {snap}, not {a.step}")
    shape = shape_spec(a.spec)
    ev = events(a.pt_dir)
    tl = [e["loss"] for e in ev if e.get("event") == "step" and a.step - 50 < e.get("step", -1) <= a.step]
    s = {"mode": "rung", "job": a.job, "cell": a.cell, "seed": a.seed, "share": a.share, "step": a.step,
         "spec": a.spec, "weights_sha256": sha256(a.weights),
         "val_loss": next((e["val_loss"] for e in ev if e.get("event") == "eval" and e.get("step") == a.step), None),
         "train_loss_last50": sum(tl) / len(tl) if tl else None}
    s["panels"] = zero_rate_panels(shape, a.weights, a.out, a.panel)
    s["carry5_base"] = carry5_base(shape, a.weights, a.carry5_items, a.out, required=True)
    s.update(rung_metrics(s["carry5_base"]))
    s["gpu"] = gpu_name()
    jsave(done, s)
    if a.drop_snapshot:  # scored and recorded: the VM's disk need not hold 4 copies of every run
        os.remove(a.weights)
    print(json.dumps({k: s[k] for k in ("job", "step", "acc_gold", "Y_base", "panels", "val_loss")}))
    return 0


def cmd_asha_train(a):
    """One successive-halving candidate at one budget (amendment 1, exploratory arm): pretrain from
    scratch for --budget steps (warmup a fraction of the budget, cosine to 0.1x at the budget), then
    the rung metrics on the final weights. A diverged run is recorded (no metrics) and exits 0, so
    the promotion step ranks it last; an mtstudio failure exits 1."""
    done = os.path.join(a.out, "rung.json")
    if os.path.exists(done):
        d = jload(done)
        if d.get("acc_gold") is not None or d.get("diverged"):
            print(f"{done}: present")
            return 0
    preflight(a.carry5_items)
    shape = shape_spec(a.spec)
    spec = json.loads(json.dumps(shape))
    spec["name"] = f"asha_{a.config}_{a.round}"
    spec["data"]["corpus"] = a.corpus
    every = max(1, a.budget // 10)
    spec["train"].update({"steps": a.budget, "lr": a.lr, "seed": a.seed, "eval_every": every, "eval_windows": 32,
                          "checkpoint_every": every,
                          "lr_schedule": {"warmup_steps": max(1, round(a.warmup_frac * a.budget)), "decay": "cosine",
                                          "min_lr_frac": 0.1}})
    spec["export"] = {"formats": ["safetensors"]}
    run_dir = os.path.join(a.out, "run")
    rate = None
    if not os.path.exists(os.path.join(run_dir, "result.json")):  # resumes from its own checkpoint on this VM
        rc, rate = run_mtstudio(spec, run_dir, os.path.join(a.out, "mtstudio.log"), t_from=min(50, a.budget // 2))
        if rc != 0:
            raise SystemExit(f"mtstudio failed (rc {rc})")
    ev = events(run_dir)
    losses = [e["loss"] for e in ev if e.get("event") == "step"]
    val = final_eval(run_dir)
    diverged = val is None or not math.isfinite(val) or bool(losses and val > losses[0])
    s = {"mode": "asha_train", "config": a.config, "round": a.round, "budget": a.budget, "spec": a.spec,
         "arch": shape["arch"]["custom"], "lr": a.lr, "warmup_frac": a.warmup_frac, "share": a.share, "seed": a.seed,
         "final_val": val, "diverged": diverged, "first_loss": losses[0] if losses else None,
         "last_loss": losses[-1] if losses else None, "tok_per_s_steady": rate,
         "evals": [(e["step"], e["val_loss"]) for e in ev if e.get("event") == "eval"], "gpu": gpu_name(),
         "acc_gold": None, "Y_base": None}
    if not diverged:
        w = os.path.join(run_dir, spec["name"] + ".safetensors")
        s["panels"] = zero_rate_panels(shape, w, a.out, a.panel)
        s["carry5_base"] = carry5_base(shape, w, a.carry5_items, a.out, required=True)
        s.update(rung_metrics(s["carry5_base"]))
    jsave(done, s)
    print(json.dumps({k: s[k] for k in ("config", "round", "budget", "final_val", "diverged", "acc_gold")}))
    return 0


def decisions_from(files, rnd):
    """{config: bool} for round rnd from the first file that holds it (jobs_vm.promotion's format)."""
    for f in files:
        try:
            d = jload(f)["decisions"][rnd]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return d
    return None


def promote_rank(rows, keep):
    """The registered promotion rule: candidates with metrics first, by base-model Carry-5 acc_gold
    (higher first), then dialogue panel loss (lower first), then config id; a diverged or failed run
    ranks last and is never promoted. The first `keep` with metrics go on."""
    def key(r):
        ok = r.get("acc_gold") is not None
        dl = (r.get("panels") or {}).get("dialogue")
        return (0 if ok else 1, -(r.get("acc_gold") or 0.0), dl if dl is not None else math.inf, r["config"])
    ranked = sorted(rows, key=key)
    return ranked, [r["config"] for r in ranked if r.get("acc_gold") is not None][:keep]


def cmd_asha_promote(a):
    """The promotion step into round --round: reads the previous round's candidates' rung.json
    (on this VM, or from the uploaded state file the home driver keeps: tiny_chat_asha.py sync),
    ranks them by promote_rank and writes decisions.json {"decisions": {round: {config: bool}}}.
    Refuses (exit 1) while a candidate's result is neither here nor in the state file."""
    plan = jload(a.plan)
    ids = [c["id"] for c in plan["configs"]]
    rounds = [r["id"] for r in plan["rounds"]]
    k = rounds.index(a.round)
    prev, keep = rounds[k - 1], plan["rounds"][k]["keep"]
    state = jload(a.state) if a.state and os.path.exists(a.state) else {}
    chat = os.environ.get("JOBS_CHAT_DIR", "/content/chat")
    if k - 1 == 0:
        cands = ids
    else:
        dec = decisions_from([os.path.join(chat, f"asha_promote_{prev}", "decisions.json"), a.state], prev)
        if dec is None:
            raise SystemExit(f"the {prev} decisions are neither on this VM nor in {a.state}")
        cands = [c for c in ids if dec.get(c)]
    rows, missing, where = [], [], {}
    for c in cands:
        job = f"asha_{prev}_{c}"
        local = os.path.join(chat, job, "rung.json")
        st_path = os.path.join(chat, f"STATUS_{job}")
        status = open(st_path).read().strip() if os.path.exists(st_path) else None
        if os.path.exists(local):
            rows.append(jload(local))
            where[c] = "vm"
        elif c in state.get("rungs", {}).get(prev, {}):
            rows.append(state["rungs"][prev][c])
            where[c] = "state"
        elif status == "FAILED" or c in state.get("failed", {}).get(prev, []):
            rows.append({"config": c, "acc_gold": None, "failed": True})
            where[c] = "failed"
        else:
            missing.append(c)
    if missing:
        raise SystemExit(f"{a.round}: no {prev} result for {missing} here or in {a.state}; run "
                         "tools/colab_chat/tiny_chat_asha.py sync at home and relaunch")
    ranked, promoted = promote_rank(rows, keep)
    out = {"round": a.round, "from": prev, "keep": keep, "rule": " ".join(promote_rank.__doc__.split()),
           "decisions": {a.round: {c: c in promoted for c in ids}},
           "ranking": [{"config": r["config"], "acc_gold": r.get("acc_gold"), "Y_base": r.get("Y_base"),
                        "dialogue_panel": (r.get("panels") or {}).get("dialogue"),
                        "diverged": r.get("diverged"), "failed": r.get("failed", False), "from": where[r["config"]]}
                       for r in ranked]}
    jsave(os.path.join(a.out, "decisions.json"), out)
    print(json.dumps({"round": a.round, "promoted": promoted}))
    return 0


def cmd_gate(a):
    problems = []
    design = jload(a.design) if os.path.exists(a.design) else {}
    if not design.get("final"):
        problems.append(f"{a.design} is missing or provisional: the dialogue levels need the whole default "
                        "human set measured (make_jobs.py --dialogue-pt-bytes)")
    if a.stage0:
        if a.need_carry5:
            problems += [f"{p} missing: the rung metrics cannot be computed" for p in (CARRY5_TOOL, a.carry5_items)
                         if not os.path.exists(p)]
        os.makedirs(a.out, exist_ok=True)
        jsave(os.path.join(a.out, "gate.json"), {"ok": not problems, "problems": problems})
        print(json.dumps({"ok": not problems, "problems": problems}))
        return 1 if problems else 0
    choice = jload(a.choice)["lr"] if os.path.exists(a.choice) else None
    if choice is None:
        problems.append(f"{a.choice} missing: Stage 0 has not chosen the learning rates")
    for j in jload(a.jobs)["jobs"]:
        if "spec" not in j:
            continue
        sid = j["spec"]["name"][len("tc_"):]
        lr = j["spec"]["train"]["lr"]
        if choice is not None and choice.get(sid) != lr:
            problems.append(f"{j['name']}: lr {lr} but Stage 0 chose {choice.get(sid)} for {sid}")
    for p in (CARRY5_TOOL, a.carry5_items):
        if not os.path.exists(p):
            problems.append(f"{p} missing: the primary metric cannot be computed")
    os.makedirs(a.out, exist_ok=True)
    jsave(os.path.join(a.out, "gate.json"), {"ok": not problems, "problems": problems})
    print(json.dumps({"ok": not problems, "problems": problems}))
    return 1 if problems else 0


def cmd_choose(a):
    """The registered screen rule, per shape: among screens that did not diverge, the lowest final
    validation loss; any screen within SCREEN_TIE nats of it counts as tied and the smallest tied lr
    wins. A shape whose screens all diverged stops the programme (no rate is chosen)."""
    by = {}
    for p in sorted(glob.glob(os.path.join(a.art, "screen_*", "summary.json"))):
        s = jload(p)
        sid = os.path.basename(os.path.dirname(p))[len("screen_"):].rsplit("_lr", 1)[0]
        by.setdefault(sid, []).append(s)
    lr, detail = {}, {}
    for sid, ss in sorted(by.items()):
        good = [s for s in ss if not s["diverged"] and s["rc"] == 0]
        if not good:
            raise SystemExit(f"{sid}: every screen diverged or failed; no rate is chosen")
        best = min(s["final_val"] for s in good)
        tied = [s for s in good if s["final_val"] <= best + SCREEN_TIE]
        lr[sid] = min(s["lr"] for s in tied)
        detail[sid] = [{k: s[k] for k in ("lr", "final_val", "diverged", "tok_per_s_steady")} for s in ss]
    out = {"rule": " ".join(cmd_choose.__doc__.split()), "tie_nats": SCREEN_TIE, "lr": {}, "screens": {}}
    if os.path.exists(a.out):  # the 60M stage adds its shapes to the 20M choice; a shape is chosen once
        out = jload(a.out)
        again = sorted(set(lr) & set(out["lr"]))
        if again:
            raise SystemExit(f"{a.out} already holds a choice for {again}")
    out["lr"].update(lr)
    out["screens"].update(detail)
    lr = out["lr"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jsave(a.out, out)
    print(json.dumps({"lr": lr}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--steps", type=int, default=300)
    p = sub.add_parser("screen")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("post")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--pt-weights", required=True)
    p.add_argument("--pt-dir", required=True)
    p.add_argument("--sft-template", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--share", type=int, required=True)
    p.add_argument("--panel", action="append", default=[])
    p.add_argument("--carry5-items", required=True)
    p.add_argument("--pt-lr", type=float, default=None, help="the pretraining lr to record (default: the spec's)")
    p = sub.add_parser("pick")
    p.add_argument("--out", required=True)
    p.add_argument("--run", action="append", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--choice", default="experiments/tiny_chat_2026-10/stage0/lr_choice.json")
    p.add_argument("--design", default="experiments/tiny_chat_2026-10/design.json")
    p.add_argument("--stage0", action="store_true", help="check the design only (before the screens)")
    p.add_argument("--carry5-items", default="corpus/eval/carry5_v1.jsonl")
    p.add_argument("--need-carry5", action="store_true", help="with --stage0: also require Carry-5 (the ASHA arm)")
    p = sub.add_parser("choose")
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("rung")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--weights", required=True, help="the snapshot's model.safetensors (state.txt beside it)")
    p.add_argument("--pt-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--drop-snapshot", action="store_true", help="delete the snapshot's weights once rung.json is saved")
    p.add_argument("--job", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--share", type=int, required=True)
    p.add_argument("--panel", action="append", default=[])
    p.add_argument("--carry5-items", required=True)
    p = sub.add_parser("asha_train")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--warmup-frac", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--share", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--round", required=True)
    p.add_argument("--panel", action="append", default=[])
    p.add_argument("--carry5-items", required=True)
    p = sub.add_parser("asha_promote")
    p.add_argument("--out", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--round", required=True)
    p.add_argument("--state", default="", help="the uploaded state file (tiny_chat_asha.py sync)")
    a = ap.parse_args(argv)
    if a.mode != "choose":
        os.makedirs(a.out, exist_ok=True)
    return {"probe": cmd_probe, "screen": cmd_screen, "post": cmd_post, "pick": cmd_pick, "gate": cmd_gate,
            "choose": cmd_choose, "rung": cmd_rung, "asha_train": cmd_asha_train,
            "asha_promote": cmd_asha_promote}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
