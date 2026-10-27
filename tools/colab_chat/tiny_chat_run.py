#!/usr/bin/env python3
"""VM side of the tiny-chat experiments (experiments/tiny_chat_2026-10/PREREGISTRATION.md): the
data-job commands that tiny_chat_*.json run through jobs_vm.py, plus `choose` at home.

    tiny_chat_run.py probe  --out D --spec SHAPE.json --corpus C.ids --steps 300
    tiny_chat_run.py screen --out D --spec SHAPE.json --corpus C.ids --lr X --steps 1500 --seed 101
    tiny_chat_run.py post   --out D --spec SHAPE.json --pt-weights W --pt-dir P --sft-template SFT.json
                            --seed S --cell C --share N --panel NAME=IDS ... --carry5-items ITEMS
                            [--gen ember|mtstudio] [--ember-commit C] [--ember-deterministic]
                            [--sft-batching depth|spec] [--carry12-items C12 [--carry12-gen ember|mtstudio|none]
                            [--carry12-spot 20] [--no-carry5]]
    tiny_chat_run.py pick  --out D --run POST_DIR [--run ...]
    tiny_chat_run.py gate   --out D --jobs JOBS.json --choice LR_CHOICE.json
                            [--pilot-lr PILOT_LR.json] [--pilot-verdict VERDICT.json --need-pilot PASS|FAIL]
    tiny_chat_run.py choose --art ART_DIR [--art RERUN_DIR] --out LR_CHOICE.json   (at home, after Stage 0)
    tiny_chat_run.py pilot_lr --art ART_DIR --art RERUN_DIR [--session-ended] --out PILOT_LR.json
                            (at home: the floor pilot's rate, amendment 3)
    tiny_chat_run.py rung   --out D --spec SHAPE.json --weights SNAP/model.safetensors --pt-dir P --step N
                            --job J --cell C --seed S --share N --panel NAME=IDS ... --carry5-items ITEMS
    tiny_chat_run.py asha_train   --out D --spec SHAPE.json --corpus C.ids --budget N --lr X --warmup-frac F
                                  --seed S --share N --config ID --round R --panel ... --carry5-items ITEMS
    tiny_chat_run.py asha_promote --out D --plan PLAN.json --round R [--state STATE.json]
    tiny_chat_run.py cuda_gate --out D [--script tools/colab_cuda_validate.sh] [--build-dir DIR]
    tiny_chat_run.py choose_rec --art ART_DIR [--art ...] --out LR_CHOICE_REC.json   (at home: the recursion arm)

The recursion arm (amendment 3, A3.3): cuda_gate builds and runs, on the VM's GPU, every device leg of
tools/colab_cuda_validate.sh (test_recursion's device leg among them) and test_recursion once more with
the kernels jobs_vm.py trains with; cuda_gate.json records each leg's output. probe and screen take
--cuda-gate DIR (refuse unless DIR/cuda_gate.json passed; on a fresh VM, where the gate job is home and
not rerun, they run the gate there first) and --data-job JOBS.json (rebuild the corpus with that file's
data job when a fresh VM lacks it). choose_rec is choose over A3.3's four-rate grid; a cell whose four
screens all diverged is dropped and reported, not chosen. post batches by effective depth (prelude +
block x repeats + coda).

The MoR block (amendment 4, A4.5): for a spec whose arch.custom.recursion schedule is "mor", post also
records the pretraining run's realised depth (pt_mor: m_s, the mean over its logged steps of each step's
mean depth, from the step events' depth fractions) and runs H-MoR2's depth trace on the SFT model
(tools/carry5_depth.py over Carry-5's context families, gold history teacher-forced; mor_depth: Delta, the
slope on fact distance, the depth fractions and their normalised entropy, the histograms; written to
OUT/mor_depth/ and copied into summary.json). A failed trace is recorded and the job exits 1 after writing
the summary; a rerun repeats only what is missing.

Generation (24 Oct 2026; PREREGISTRATION.md, implementation note of that date): post --gen ember
generates Carry-5's replies with ember.cpp's server (EMBER_COMMIT, built once per VM in EMBER_DIR under a
lock, EMBER_GEMM_TF32=0, all layers on the GPU) on the SFT model's F32 GGUF. A spot-check first generates
SPOT_N fixed items (every SPOT_EVERY-th) with ember and with `mtstudio chat --kv-cache` and compares every
reply (text, stop reason, history used); only N/N lets ember generate the run, otherwise mtstudio
generates all of it and summary.json carries gen_flag. For llama_recursive a failed spot-check on ember's
CUDA interpreter is repeated with EMBER_IR_DEVICE=cpu (recorded and flagged when that one passes).
OUT/gen_engine.json holds the decision, the build's commit and both spot-checks; forced-choice scoring
stays with `mtstudio score`. post --gen mtstudio (the default) generates with mtstudio as before.

Carry-12 (the long programme, experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md; its cells' jobs files
tools/colab_chat/carry12_<cell>_r<seed>.json): post --carry12-items scores the SFT model on Carry-12 after Carry-5
(run_carry12: tools/carry12_eval.py, forced choice in gold, self, near, twin and noctx; self history to the target
at --max-new 48 by ember at the same --ember-commit after its spot-check; no near_self; the parity gate) and
records summary["carry12"], Y12 and Y12_gold; --ember-deterministic adds EMBER_DETERMINISTIC=1 to Carry-5's ember
servers (carry12_eval sets it for its own); --sft-batching spec keeps 16,384 SFT tokens per step at the template's T
(16 rows of 1,024); --no-carry5 with --carry12-gen none is cell W512's re-run (SFT and Carry-12 forced choice).
gate --need-file refuses until a file exists (the frozen PREREGISTRATION.md).

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
MOR_DEPTH_TOOL = os.environ.get("TINY_CHAT_MOR_DEPTH", "tools/carry5_depth.py")  # amendment 4: H-MoR2's trace
SCREEN_TIE = 0.01  # nats: screens closer than this to the best count as tied; the smaller lr wins
SCREEN_LRS_EXPECTED = (5e-4, 1e-3, 2e-3)  # section 4's grid: a shape is chosen only with all three screens
REC_SCREEN_LRS = (2.5e-4, 5e-4, 1e-3, 2e-3)  # A3.3: the recursion cells' grid (shared weights sum r gradients)
CUDA_SCRIPT = "tools/colab_cuda_validate.sh"


# ---------------------------------------------------------------- the Carry-5 interface
CARRY5_MODES = ("gold", "self", "twin", "noctx")
CARRY5_CHANCE = 0.25  # four candidates per item; carry5_eval's uniform scorer is exactly this


def carry5_argv(spec, items, out_dir, gen, port, ember=None, scorer="mtstudio", extra=()):
    """tools/carry5_eval.py (benchmark agent, 23 Oct 2026): `run` scores the model named by SPEC
    (out_dir/name.safetensors) with `mtstudio score` and plays the self history through a /chat
    server on PORT (one port per job: post jobs share a VM): `mtstudio chat` with --gen mtstudio,
    ember.cpp's server on the GGUF beside the safetensors with --gen ember (EMBER = {"bin",
    "gpu_layers"}). Writes OUT_DIR/summary.json = {"meta", "models": {"m": {"fc": {"acc"|"acc_norm":
    {mode: {"all": {"mean", "n", "ci95"}, "family", "distance", "interference", ...}}}, "gen",
    "decay", ...}}} and OUT_DIR/m/gen.jsonl (the replies). Reconcile HERE if its flags change; only
    this function and carry5_read depend on them."""
    argv = [sys.executable, CARRY5_TOOL, "run", "--model", f"m={spec}", "--out", out_dir, "--scorer", scorer,
            "--mtstudio", BIN, "--gen", gen, "--set", items, "--port", str(port)]
    if gen == "ember":
        argv += ["--ember-bin", ember["bin"], "--gpu-layers", str(ember["gpu_layers"])]
    return argv + list(extra)


# Amendment 2 (PREREGISTRATION.md): the instruction family's forced choice is confounded by answer
# length (the compliant candidate is always the shortest; the question alone scores ~0.83 on the live
# model), so the primary metric pools the other four families. The question-alone floor is read on
# the families whose twin carries a statement the question alone lacks (knowledge's twin IS its
# question alone, verbatim).
PRIMARY_FAMILIES = ("recall", "correction", "topic", "knowledge")
CONTEXT_FAMILIES = ("recall", "correction", "topic")


def carry5_pools(items_path):
    """Exact pooled means from carry5_eval's per-item records (items.jsonl): forced-choice acc and
    acc_norm per history mode over the PRIMARY_FAMILIES ("primary") and the CONTEXT_FAMILIES
    ("context"), per family, and the instruction family's generation-scored compliance per mode
    (its machine check on the greedy reply). Modes a run did not score (self without generation)
    are absent."""
    with open(items_path, encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    out = {"primary_families": list(PRIMARY_FAMILIES), "context_families": list(CONTEXT_FAMILIES)}
    for kind in ("acc", "acc_norm"):
        pools = {"primary": {}, "context": {}, "family": {}}
        for mode in CARRY5_MODES:
            vals = {}
            for r in recs:
                if mode in r.get("fc", {}):
                    vals.setdefault(r["family"], []).append(float(r["fc"][mode][kind]))
            if not vals:
                continue
            for pool, fams in (("primary", PRIMARY_FAMILIES), ("context", CONTEXT_FAMILIES)):
                xs = [v for fam in fams for v in vals.get(fam, [])]
                pools[pool][mode] = {"mean": sum(xs) / len(xs) if xs else None, "n": len(xs)}
            pools["family"][mode] = {fam: {"mean": sum(v) / len(v), "n": len(v)} for fam, v in sorted(vals.items())}
        out[kind] = pools
    comp = {}
    for mode in CARRY5_MODES:
        xs = [bool(r["gen"][mode]["complies"]) for r in recs if r["family"] == "instruction"
              and mode in r.get("gen", {}) and r["gen"][mode].get("complies") is not None]
        if xs:
            comp[mode] = {"mean": sum(xs) / len(xs), "n": len(xs)}
    out["instruction_complies_gen"] = comp
    return out


def carry5_read(out_dir):
    """The model's summary plus the headline means: acc and acc_norm per history mode over all
    items (the frozen definition), and the amended pools (carry5_pools) from the per-item records,
    which the amended primary metric and floor need."""
    s = jload(os.path.join(out_dir, "summary.json"))
    m = s["models"]["m"]
    items = os.path.join(out_dir, "m", "items.jsonl")
    if not os.path.exists(items):
        raise SystemExit(f"{items} missing: the amended primary metric (amendment 2) needs carry5_eval's item records")
    pools = carry5_pools(items)
    return {"meta": s["meta"], "chance": CARRY5_CHANCE,
            "acc": {k: m["fc"]["acc"][k]["all"]["mean"] for k in CARRY5_MODES},
            "acc_norm": {k: m["fc"]["acc_norm"][k]["all"]["mean"] for k in CARRY5_MODES},
            "pool": {k: v["mean"] for k, v in pools["acc"]["primary"].items()},
            "ctx": {k: v["mean"] for k, v in pools["acc"]["context"].items()},
            "pools": pools, "summary": m}


def primary(c5):
    """The primary metric as amended (amendment 2): forced-choice accuracy (acc), SELF history,
    multi-turn minus twin, over the PRIMARY_FAMILIES (the instruction family excluded)."""
    try:
        return float(c5["pool"]["self"]) - float(c5["pool"]["twin"])
    except (KeyError, TypeError, ValueError):
        return None


def primary_all_items(c5):
    """Section 3's frozen definition over all items, instruction included (reported beside it)."""
    try:
        return float(c5["acc"]["self"]) - float(c5["acc"]["twin"])
    except (KeyError, TypeError, ValueError):
        return None


def port_for(path):
    return 8100 + int(hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:6], 16) % 800


# ---------------------------------------------------------------- generation through ember.cpp (24 Oct 2026)
# The post job's greedy generation (Carry-5's self history) runs on ember.cpp's CUDA server: `mtstudio
# chat` decoded at ~0.1-0.2 s/token and the floor pilot's post job spent over 2 h there. Greedy generation
# is engine-independent by the stack's parity contract (LOCAL_LLM_STACK.md: ember vs mtstudio greedy
# 40/40 on every family); the spot-check below enforces it on every run, and a run whose spot-check is
# not N/N generates with mtstudio throughout (the engines are never mixed within a run).
EMBER_REPO = os.environ.get("TINY_CHAT_EMBER_REPO", "https://github.com/JohnnyTeutonic/ember.cpp.git")
# ember main, 25 Oct 2026: conformant with spec/IR_SPEC.md v1 (reads ir_version 0 and 1; its parity receipts
# byte-identical to 69d7ca26d5cd's, the CUDA prefill rmsnorm fix it carries).
EMBER_COMMIT = "aa50a6112dfd706078e5c7849070f8ca5af620b4"
EMBER_DIR = os.environ.get("TINY_CHAT_EMBER_DIR", "/content/ember")  # one build per VM, shared by post jobs
EMBER_CMAKE = os.environ.get("TINY_CHAT_EMBER_CMAKE", "-DCMAKE_CUDA_ARCHITECTURES=89").split()
EMBER_GPU_LAYERS = os.environ.get("TINY_CHAT_EMBER_GPU_LAYERS", "-1")  # all layers on the GPU
# The server's environment: TF32 GEMMs (ember's CUDA default) can flip greedy tokens against mtstudio's
# f32, so they are off; flash-attention-2 stays on (parity-gated in ember).
EMBER_ENV = {"EMBER_GEMM_TF32": "0"}
SPOT_EVERY, SPOT_N = 26, 40  # the spot-check's fixed items: Carry-5 items 0, 26, ..., 1014 (all five families)
SPOT_FIELDS = ("reply", "stop", "history_used")


def ember_family(custom):
    """ember's name for the model's family: llama_mor (ember serves MoR on its CPU path, also in CUDA
    builds), llama_recursive (the IR interpreter; on CUDA in CUDA builds), else llama."""
    if is_mor(custom):
        return "llama_mor"
    return "llama_recursive" if custom.get("recursion") else "llama"


def ensure_ember(log_path, commit=None):
    """ember.cpp's server at EMBER_COMMIT, built once per VM in EMBER_DIR/build (cloned into
    EMBER_DIR/src; Boost.Regex from apt when missing). An exclusive lock on EMBER_DIR/.lock makes post
    jobs that start together wait for one build. Returns the build's record (EMBER_DIR/build/
    ember_build.json: bin, commit, cuda, seconds) plus built_here, or {"error"} when it cannot be built."""
    import fcntl  # POSIX: the VM, and WSL for the CPU test
    commit = commit or EMBER_COMMIT  # a jobs file may pin its own (post --ember-commit; amendment 5)
    src, build = os.path.join(EMBER_DIR, "src"), os.path.join(EMBER_DIR, "build")
    marker = os.path.join(build, "ember_build.json")
    binary = os.path.join(build, "tinyllama_server")
    os.makedirs(EMBER_DIR, exist_ok=True)
    with open(os.path.join(EMBER_DIR, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(marker) and os.path.exists(binary):
            m = jload(marker)
            if m.get("commit") == commit:
                return {**m, "built_here": False}
        t0 = time.time()
        log = open(log_path, "a", encoding="utf-8")

        def sh(argv, **kw):
            log.write(f"$ {' '.join(argv)}\n")
            log.flush()
            p = subprocess.run(argv, capture_output=True, text=True, **kw)
            log.write(p.stdout[-20000:] + p.stderr[-20000:])
            log.flush()
            return p

        try:
            # The package, not the header: an image can carry Boost's headers without libboost_regex.
            have = subprocess.run(["dpkg", "-s", "libboost-regex-dev"], capture_output=True).returncode == 0 \
                if shutil.which("dpkg") else os.path.exists("/usr/include/boost/regex.hpp")
            if not have and os.environ.get("TINY_CHAT_EMBER_APT", "1") != "0":
                sh(["apt-get", "update", "-qq"])
                p = sh(["apt-get", "install", "-y", "-q", "libboost-regex-dev"])
                if p.returncode != 0:
                    return {"error": f"apt-get install libboost-regex-dev failed (rc {p.returncode})"}
            head = sh(["git", "-C", src, "rev-parse", "HEAD"]) if os.path.isdir(os.path.join(src, ".git")) else None
            if not head or head.returncode != 0:
                shutil.rmtree(src, ignore_errors=True)
                p = sh(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", EMBER_REPO, src])
                if p.returncode != 0:
                    return {"error": f"git clone {EMBER_REPO} failed (rc {p.returncode}): {p.stderr[-400:]}"}
            p = sh(["git", "-C", src, "checkout", "-q", "--detach", commit])
            got = sh(["git", "-C", src, "rev-parse", "HEAD"]).stdout.strip()
            if p.returncode != 0 or got != commit:
                return {"error": f"ember checkout of {commit} failed (HEAD {got!r}): {p.stderr[-400:]}"}
            p = sh(["cmake", "-S", src, "-B", build, "-DCMAKE_BUILD_TYPE=Release", *EMBER_CMAKE])
            if p.returncode != 0:
                return {"error": f"ember cmake failed (rc {p.returncode}): {p.stderr[-400:]}"}
            cuda = "CUDA toolkit and cuBLAS found" in p.stdout + p.stderr
            jobs = os.environ.get("TINY_CHAT_EMBER_JOBS", str(os.cpu_count() or 4))
            p = sh(["make", "-C", build, f"-j{jobs}", "tinyllama_server"])
            if p.returncode != 0 or not os.path.exists(binary):
                return {"error": f"ember build failed (rc {p.returncode}): {(p.stdout + p.stderr)[-600:]}"}
            m = {"bin": binary, "commit": got, "repo": EMBER_REPO, "cuda": cuda, "cmake_args": EMBER_CMAKE,
                 "seconds": round(time.time() - t0, 1)}
            jsave(marker, m)
            return {**m, "built_here": True}
        except OSError as e:
            return {"error": f"ember build: {type(e).__name__}: {e}"}
        finally:
            log.close()


def gguf_f32(path):
    """True when PATH is a GGUF whose tensors are all F32 (ggml type 0)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(root, "spec", "tools"))  # keramos
    import ir_check
    try:
        _, tensors, *_ = ir_check.read_gguf(path)
    except (OSError, ValueError):
        return False
    return bool(tensors) and all(t[2] == 0 for t in tensors)


def ensure_f32_gguf(spec_path, weights, log_path):
    """The SFT model's F32 GGUF beside its safetensors (the SFT spec exports it; re-exported with
    `mtstudio export` when missing or not F32). Returns (path, sha256) or (None, why)."""
    spec = jload(spec_path)
    g = os.path.join(spec["out_dir"], spec["name"] + ".gguf")
    if not gguf_f32(g):
        with open(log_path, "a", encoding="utf-8") as log:
            p = subprocess.run([BIN, "export", spec_path, "--from", weights, "--formats", "gguf"], stdout=log,
                               stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0 or not gguf_f32(g):
            return None, f"mtstudio export --formats gguf failed or not F32 (rc {p.returncode})"
    return g, sha256(g)


def gen_records(work):
    p = os.path.join(work, "m", "gen.jsonl")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(x) for x in f if x.strip())}


def compare_gens(a_recs, b_recs, ids):
    """Per item, every reply (five self-history turns, gold, twin, noctx) on SPOT_FIELDS."""
    rows, same_items, same_replies, total = [], 0, 0, 0
    for i in ids:
        ra, rb = a_recs.get(i), b_recs.get(i)
        if ra is None or rb is None:
            rows.append({"id": i, "missing": [n for n, r in (("ember", ra), ("mtstudio", rb)) if r is None]})
            total += 8
            continue
        pa = [(f"self{k + 1}", x) for k, x in enumerate(ra["self"])] + [(m, ra[m]) for m in ("gold", "twin", "noctx")]
        pb = dict([(f"self{k + 1}", x) for k, x in enumerate(rb["self"])] + [(m, rb[m]) for m in ("gold", "twin", "noctx")])
        diff = []
        for name, x in pa:
            y = pb.get(name)
            total += 1
            if y is not None and all(x.get(f) == y.get(f) for f in SPOT_FIELDS):
                same_replies += 1
            else:
                diff.append({"reply": name, "ember": {f: x.get(f) for f in SPOT_FIELDS},
                             "mtstudio": {f: (y or {}).get(f) for f in SPOT_FIELDS}})
        same_items += not diff
        if diff:
            rows.append({"id": i, "diff": diff[:3]})
    return {"items": len(ids), "items_identical": same_items, "replies": total, "replies_identical": same_replies,
            "differing": rows[:10]}


def spot_check(spec_path, items, work, ember, env, tag):
    """Generate the fixed spot-check items with ember (into WORK/ember_TAG) and with `mtstudio chat
    --kv-cache` (WORK/mtstudio, generated once), no scoring; compare every reply."""
    sub = ["--every", str(SPOT_EVERY), "--limit", str(SPOT_N)]
    res = {"every": SPOT_EVERY, "n": SPOT_N, "fields": list(SPOT_FIELDS), "env": {k: env.get(k) for k in
                                                                                  ("EMBER_GEMM_TF32", "EMBER_IR_DEVICE")}}
    if env.get("EMBER_DETERMINISTIC"):  # the Carry-12 cells' post jobs; earlier receipts keep their keys
        res["env"]["EMBER_DETERMINISTIC"] = env["EMBER_DETERMINISTIC"]
    out = {}
    os.makedirs(work, exist_ok=True)
    for eng, d in (("mtstudio", os.path.join(work, "mtstudio")), ("ember", os.path.join(work, f"ember_{tag}"))):
        timing = os.path.join(work, f"{os.path.basename(d)}_timing.json")
        if os.path.exists(timing):  # generated already (mtstudio's side serves every ember attempt)
            t = jload(timing)
        else:
            t0 = time.time()
            p = subprocess.run(carry5_argv(spec_path, items, d, eng, port_for(d), ember=ember, scorer="none", extra=sub),
                               capture_output=True, text=True, env=env)
            with open(os.path.join(work, f"{os.path.basename(d)}.log"), "w", encoding="utf-8") as f:
                f.write(p.stdout[-20000:] + p.stderr[-20000:])
            t = {"seconds": round(time.time() - t0, 1), "rc": p.returncode}
            if p.returncode != 0:
                t["error"] = p.stderr[-800:]
            else:
                jsave(timing, t)
        res[f"{eng}_seconds"], res[f"{eng}_rc"] = t["seconds"], t["rc"]
        if "error" in t:
            res[f"{eng}_error"] = t["error"]
        out[eng] = gen_records(d)
    ids = [r["id"] for r in read_items(items)[::SPOT_EVERY][:SPOT_N]]
    res.update(compare_gens(out["ember"], out["mtstudio"], ids))
    res["identical"] = res["items_identical"]
    res["n"] = len(ids)  # SPOT_N on Carry-5 itself (1,032 items); fewer only on a shorter items file (tests)
    res["pass"] = bool(ids) and res["items_identical"] == len(ids) and res.get("ember_rc") == 0 \
        and res.get("mtstudio_rc") == 0
    return res


def read_items(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def choose_engine(spec_path, weights, items, out, custom, commit=None, ember_env=None):
    """OUT/gen_engine.json: the generation engine for this run's Carry-5 and why. ember when its build
    works, the F32 GGUF exists and the spot-check is N/N (for llama_recursive a failed spot-check on
    ember's CUDA interpreter is retried with EMBER_IR_DEVICE=cpu, recorded and flagged); otherwise
    mtstudio for the whole run, flagged. Decided once per run: a rerun reads the file. ember_env: the
    server's environment (default EMBER_ENV; the Carry-12 cells add EMBER_DETERMINISTIC=1)."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "gen_engine.json")
    if os.path.exists(path):
        return jload(path)
    fam = ember_family(custom)
    commit = commit or EMBER_COMMIT
    ember_env = dict(ember_env or EMBER_ENV)
    d = {"engine": "mtstudio", "family": fam, "ember_commit_pinned": commit, "ember_env": dict(ember_env),
         "flag": None, "ir_device": None}
    t0 = time.time()
    ember = ensure_ember(os.path.join(out, "ember_build.log"), commit)
    d["ember_build"] = ember
    if "error" in ember:
        d["flag"] = f"ember unavailable ({ember['error'][:300]}); generated with mtstudio"
    else:
        g, why = ensure_f32_gguf(spec_path, weights, os.path.join(out, "mtstudio.log"))
        if g is None:
            d["flag"] = f"no F32 GGUF ({why}); generated with mtstudio"
        else:
            d["gguf_sha256"] = why
            eb = {"bin": ember["bin"], "gpu_layers": EMBER_GPU_LAYERS}
            d["gpu_layers"] = EMBER_GPU_LAYERS
            work = os.path.join(out, "spot")
            env = {**os.environ, **ember_env}
            env.pop("EMBER_IR_DEVICE", None)
            spot = spot_check(spec_path, items, work, eb, env, "default")
            d["spot"] = spot
            if spot["pass"]:
                d["engine"] = "ember"
            elif fam == "llama_recursive" and ember.get("cuda"):
                # ember's IR interpreter on CUDA is not yet gated for this family: this spot-check is its gate.
                cpu = spot_check(spec_path, items, work, eb, {**env, "EMBER_IR_DEVICE": "cpu"}, "ir_cpu")
                d["spot_ir_cpu"] = cpu
                if cpu["pass"]:
                    d.update(engine="ember", ir_device="cpu",
                             flag=f"llama_recursive: ember's IR on CUDA failed the spot-check "
                                  f"({spot['identical']}/{spot['n']}); EMBER_IR_DEVICE=cpu passed ({cpu['identical']}/"
                                  f"{cpu['n']}) and generated the run")
                    d["ember_env"]["EMBER_IR_DEVICE"] = "cpu"
            if d["engine"] != "ember" and not d["flag"]:
                d["flag"] = (f"ember spot-check {spot['identical']}/{spot['n']}"
                             + (f", EMBER_IR_DEVICE=cpu {d['spot_ir_cpu']['identical']}/{d['spot_ir_cpu']['n']}"
                                if "spot_ir_cpu" in d else "")
                             + "; generated with mtstudio throughout")
            if d["engine"] == "ember" and fam == "llama_recursive" and not d["ir_device"]:
                d["ir_device"] = "gpu" if ember.get("cuda") and EMBER_GPU_LAYERS != "0" else "cpu"
            if fam == "llama_mor":
                d["ir_device"] = "cpu"  # ember serves MoR on its CPU path, also in CUDA builds
            d["ember"] = eb
    d["choose_seconds"] = round(time.time() - t0, 1)
    jsave(path, d)
    return d


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


SFT_TOKENS = 32 * 512  # section 1: 32 rows of T 512; the Carry-12 cells keep it at 16 rows of T 1,024
# override: tests only (a CPU toy at T 1,024 cannot hold 16 rows); every registered job uses SFT_TOKENS
SFT_TOKENS_PER_STEP = int(os.environ.get("TINY_CHAT_SFT_TOKENS_PER_STEP", SFT_TOKENS))


def sft_batching(mode, shape, sft_T=512):
    """The SFT stage's (batch, accum). "depth" (the default, every registered arm's): batching() by effective
    depth, 16 x 2 from 16 layers, else 32 x 1. "spec" (amendment 6, the 150M cells, whose width needs smaller
    micro-batches; the Carry-12 cells at T 1,024): the shape spec's pretraining (batch, accum). Both keep
    SFT_TOKENS_PER_STEP tokens per step: 32 rows at the SFT template's T 512, 16 rows at T 1,024."""
    if mode == "spec":
        tr = shape["train"]
        b, acc = tr["batch"], tr.get("accum", 1)
        if b * acc * sft_T != SFT_TOKENS_PER_STEP:
            raise SystemExit(f"--sft-batching spec: batch {b} x accum {acc} x T {sft_T} is not section 1's "
                             f"{SFT_TOKENS_PER_STEP} tokens per step")
        return b, acc
    b, acc = batching(effective_layers(shape["arch"]["custom"]))
    if SFT_TOKENS_PER_STEP != SFT_TOKENS:  # tests only: the same layout with fewer rows
        b = max(1, b * SFT_TOKENS_PER_STEP // SFT_TOKENS)
    return b, acc


def effective_layers(custom):
    """Layer applications per forward: prelude + block x repeats + coda with arch.custom.recursion
    (A3.3; `layers` then counts stored layers), else layers. Batching and activation memory follow it."""
    r = custom.get("recursion")
    if not r:
        return custom["layers"]
    return r.get("prelude", 0) + r["block"] * r.get("repeats", 1) + r.get("coda", 0)


def provenance():
    """The VM's provenance.json (boot_jobs.sh: provider, GPU, driver, CUDA, image), or None on a VM booted
    before it existed. Every summary and result this file writes carries it as `provenance`
    (PREREGISTRATION.md, implementation note of 24 Oct 2026)."""
    try:
        return jload(os.path.join(os.environ.get("JOBS_CHAT_DIR", "/content/chat"), "provenance.json"))
    except (OSError, ValueError):
        return None


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


def log_tail(path, n=4000):
    """The end of an mtstudio log, so a failure's cause comes home in the summary (Stage 0's first
    attempt kept its logs on the VM only)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return None


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def screen_status(s):
    """A screen's state under section 4's rule as clarified by amendment 2. DIVERGED: a logged
    training loss went non-finite, or the run finished and its final validation loss is missing,
    non-finite or above the first training loss. FINISHED: the run finished otherwise. FAILED: the
    process ended non-zero with finite losses (a crash, not a property of the lr); it is rerun,
    unchanged, and carries no information about the rate. Works on summaries written before the
    distinction existed (their `diverged` flag also marked crashes)."""
    losses = [s.get("first_loss"), s.get("last_loss")]
    if any(x is not None and not finite(x) for x in losses) or s.get("nonfinite_loss"):
        return "DIVERGED"
    if s.get("rc") == 0:
        v, f0 = s.get("final_val"), s.get("first_loss")
        if s.get("diverged") or not finite(v) or (finite(f0) and v > f0):  # rc 0: the flag can only mean divergence
            return "DIVERGED"
        return "FINISHED"
    return "FAILED"


# ---------------------------------------------------------------- the CUDA gate (amendment 3, A3.3)
def validate_plan(script):
    """(make targets, [(env, binary)]) read from tools/colab_cuda_validate.sh, the one list of the
    device legs (so the gate cannot drift from it): the `make` line's targets, and every line of the
    form `[VAR=value ...] ./test_x` in order."""
    import re
    lines, buf = [], ""
    with open(script, encoding="utf-8") as f:
        for raw in f:
            s = raw.rstrip("\n").rstrip()
            if s.lstrip().startswith("#"):
                continue
            if s.endswith(chr(92)):  # a continued line
                buf += s[:-1] + " "
                continue
            lines.append(buf + s)
            buf = ""
    targets, legs = [], []
    leg_re = re.compile(r"^((?:[A-Z_][A-Z0-9_]*=\S*\s+)*)\./([\w.+-]+)\s*$")
    for s in lines:
        t = s.strip()
        if t.startswith("make "):
            targets = [x for x in t.split()[1:] if not x.startswith("-")]
        m = leg_re.match(t)
        if m:
            env = dict(kv.split("=", 1) for kv in m.group(1).split())
            legs.append((env, m.group(2)))
    if "test_recursion" not in targets or not any(b == "test_recursion" for _, b in legs):
        raise SystemExit(f"{script}: no test_recursion target or leg; the recursion arm's gate needs it")
    return targets, legs


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_cuda_gate(out, script=CUDA_SCRIPT, build=None):
    """Build the validation targets in mtstudio's own build directory (configured by boot_jobs.sh) and
    run every leg of the script on this VM's GPU, each in its own environment with the job's
    MICROTORCH_* variables removed (jobs_vm.py sets them for every job; the legs set their own). Then
    test_recursion once more with the kernels mtstudio trains with here (the job's MICROTORCH_* minus
    step residency and deferred downloads, which its leg 8 switches on itself). Every leg runs even
    after a failure, so one failure does not hide the others. Writes OUT/cuda_gate.json:
    {"ok", "legs": [{"binary", "env", "rc", "seconds", "output" (test_recursion, whole), "log_tail"}], ...}."""
    build = build or os.path.dirname(BIN)
    os.makedirs(out, exist_ok=True)
    targets, legs = validate_plan(script)
    job_env = {k: v for k, v in os.environ.items() if k.startswith("MICROTORCH_")}
    base = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    prod = {k: v for k, v in job_env.items() if k not in ("MICROTORCH_STEP_RESIDENCY", "MICROTORCH_DEFER_DOWNLOADS")}
    prod.setdefault("MICROTORCH_DEVICE", "cuda")
    prod.setdefault("MICROTORCH_DEVICE_OPS", "1")
    legs = legs + [(prod, "test_recursion")]
    res = {"mode": "cuda_gate", "script": script, "build_dir": build, "commit": git_head(), "gpu": gpu_name(),
           "targets": targets, "job_env": job_env, "legs": [], "provenance": provenance()}
    log = os.path.join(out, "cuda_gate.log")
    make = os.environ.get("TINY_CHAT_MAKE", "make")  # override: tests only
    t0 = time.time()
    try:
        b = subprocess.run([make, f"-j{os.cpu_count() or 4}", *targets], cwd=build, capture_output=True, text=True,
                           timeout=7200)
        brc, btext = b.returncode, b.stdout + b.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        brc, btext = -1, f"{type(e).__name__}: {e}"
    with open(log, "a", encoding="utf-8") as f:
        f.write(btext)
    res["build"] = {"rc": brc, "seconds": round(time.time() - t0, 1)}
    if brc != 0:
        res["build"]["log_tail"] = btext[-4000:]
    else:
        for k, (env, binary) in enumerate(legs):
            t1 = time.time()
            try:
                p = subprocess.run([os.path.join(build, binary)], cwd=build, env={**base, **env}, capture_output=True,
                                   text=True, timeout=3600)
                rc, text = p.returncode, p.stdout + p.stderr
            except (OSError, subprocess.TimeoutExpired) as e:
                rc, text = -1, f"{type(e).__name__}: {e}"
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n=== leg {k}: {env} ./{binary}: rc {rc}\n{text}")
            leg = {"binary": binary, "env": env, "rc": rc, "seconds": round(time.time() - t1, 1),
                   "production_kernels": k == len(legs) - 1}
            if binary == "test_recursion":
                leg["output"] = text[-20000:]  # its measured worst errors are the tolerance evidence
            if rc != 0:
                leg["log_tail"] = text[-4000:]
            res["legs"].append(leg)
    res["ok"] = brc == 0 and bool(res["legs"]) and all(x["rc"] == 0 for x in res["legs"])
    res["failed"] = [f"{x['env']} ./{x['binary']}" for x in res["legs"] if x["rc"] != 0]
    res["seconds"] = round(time.time() - t0, 1)
    jsave(os.path.join(out, "cuda_gate.json"), res)
    return res


def cmd_cuda_gate(a):
    res = run_cuda_gate(a.out, a.script, a.build_dir)
    print(json.dumps({k: res[k] for k in ("ok", "failed", "gpu", "commit", "seconds")}))
    return 0 if res["ok"] else 1


def rebuild_data(jobs_path, corpus, log_path):
    """A fresh VM lacks the corpus when the data job came home on the VM it replaced (the runner marks
    home jobs FINISHED there, so it is not rerun): run that jobs file's data job here, into the
    corpus's directory, exactly as jobs_vm.py would (it is idempotent and pinned)."""
    j = next(x for x in jload(jobs_path)["jobs"] if x["name"] == "data")
    chat = os.environ.get("JOBS_CHAT_DIR", "/content/chat")
    out = os.path.dirname(os.path.abspath(corpus))
    argv = [x.replace("{out}", out).replace("{upload}", os.path.join(chat, "upload")) for x in j["cmd"]]
    p = subprocess.run(argv, capture_output=True, text=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(p.stdout[-20000:] + p.stderr[-20000:])
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def vm_prereqs(a):
    """None, or why this probe or screen must not start: the CUDA gate has not passed on this VM, or
    the corpus is missing and could not be rebuilt."""
    if getattr(a, "cuda_gate", None):
        g = os.path.join(a.cuda_gate, "cuda_gate.json")
        if not os.path.exists(g):  # a fresh VM: the gate job is home, not here; validate this GPU first
            try:
                run_cuda_gate(a.cuda_gate)
            except (OSError, subprocess.SubprocessError, SystemExit) as e:
                return f"the CUDA gate could not run on this VM: {type(e).__name__}: {e}"
        r = jload(g)
        if not r.get("ok"):
            return f"the CUDA gate did not pass on this VM ({g}; failed legs {r.get('failed')})"
    if getattr(a, "data_job", None) and not os.path.exists(a.corpus):
        rc, tail = rebuild_data(a.data_job, a.corpus, os.path.join(a.out, "data_rebuild.log"))
        if rc != 0 or not os.path.exists(a.corpus):
            return f"{a.corpus} missing and the data job failed to rebuild it (rc {rc}): {tail}"
    return None


def refuse(a, why):
    s = {"mode": a.mode, "spec": a.spec, "rc": None, "refused": why, "status": "FAILED", "gpu": gpu_name(),
         "provenance": provenance()}
    if a.mode == "screen":
        s.update({"lr": a.lr, "seed": a.seed, "steps": a.steps, "final_val": None, "diverged": False})
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps(s))
    return 1


# ---------------------------------------------------------------- modes
def cmd_probe(a):
    why = vm_prereqs(a)
    if why:
        return refuse(a, why)
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
         "other_mtstudio_at_start": others, "params": res.get("params"), "provenance": provenance()}
    if rc != 0:
        s["log_tail"] = log_tail(os.path.join(a.out, "mtstudio.log"))
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps(s))
    return 0 if rc == 0 else 1


def cmd_screen(a):
    why = vm_prereqs(a)
    if why:
        return refuse(a, why)
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
    s = {"mode": "screen", "spec": a.spec, "rc": rc, "lr": a.lr, "seed": a.seed, "steps": a.steps,
         "final_val": val, "first_loss": losses[0] if losses else None,
         "last_loss": losses[-1] if losses else None, "last_step": max((e.get("step", 0) for e in ev if e.get("event") == "step"), default=None),
         "nonfinite_loss": any(not finite(x) for x in losses), "tok_per_s_steady": rate,
         "other_mtstudio_at_end": concurrent_mtstudio(), "gpu": gpu_name(), "provenance": provenance()}
    s["status"] = screen_status(s)
    s["diverged"] = s["status"] == "DIVERGED"
    if rc != 0:
        s["log_tail"] = log_tail(os.path.join(a.out, "mtstudio.log"))
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps(s))
    return 0 if rc == 0 else 1


PANEL_WINDOWS = int(os.environ.get("TINY_CHAT_PANEL_WINDOWS", 512))  # override: tests only (CPU toys)


def zero_rate_panels(shape, weights, out, panel_kvs, prefix="panel"):
    """{name: nats/token} for each NAME=IDS panel: 1 step at lr 0 from the weights, 512 held-out
    windows (the weights stay byte-identical), in OUT/<prefix>_<name>. Finished panels are not rerun."""
    res = {}
    for kv in panel_kvs:
        name, path = kv.split("=", 1)
        d = os.path.join(out, f"{prefix}_{name}")
        if not os.path.exists(os.path.join(d, "result.json")):
            shutil.rmtree(d, ignore_errors=True)  # init_from applies to fresh runs only
            spec = {"name": f"{prefix}_{name}", "arch": shape["arch"],
                    "data": {**shape["data"], "corpus": path, "max_tokens": 0},
                    "train": {"steps": 1, "lr": 0.0, "batch": 1, "accum": 1, "eval_every": 1, "eval_windows": PANEL_WINDOWS,
                              "checkpoint_every": 10 ** 7, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
                              "init_from": weights},
                    "export": {"formats": []}}
            rc, _ = run_mtstudio(spec, d, os.path.join(out, "mtstudio.log"))
            if rc != 0:
                raise SystemExit(f"panel {name} failed (rc {rc}): {log_tail(os.path.join(out, 'mtstudio.log'), 1500)}")
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
        try:
            return carry5_read(c5b)
        except SystemExit as e:
            if required:
                raise
            return {"error": str(e)}
    if required:
        raise SystemExit(f"carry5_eval (base) failed (rc {pb.returncode}): {pb.stderr[-500:]}")
    return {"error": pb.stderr[-500:]}


def rung_metrics(c5b):
    """The rung predictors (amendment 1): base-model forced-choice accuracy with gold history
    (acc_gold) and Y_base = acc_gold - acc_twin, both teacher-forced, over the primary metric's
    families (amendment 2: the instruction family out); the all-items values beside them, and the
    twin and question-alone accuracies on the context families (the floor's quantities)."""
    try:
        g, t = float(c5b["pool"]["gold"]), float(c5b["pool"]["twin"])
    except (KeyError, TypeError, ValueError):
        return {"acc_gold": None, "Y_base": None}
    out = {"acc_gold": g, "Y_base": g - t}
    try:
        out["acc_gold_all"] = float(c5b["acc"]["gold"])
        out["Y_base_all"] = out["acc_gold_all"] - float(c5b["acc"]["twin"])
    except (KeyError, TypeError, ValueError):
        pass
    out["twin_ctx"] = c5b.get("ctx", {}).get("twin")
    out["noctx_ctx"] = c5b.get("ctx", {}).get("noctx")
    return out


def is_mor(custom):
    return (custom.get("recursion") or {}).get("schedule") == "mor"


def realised_depth(evs):
    """A4.5's m_s from a MoR pretraining run's events: the mean over its steps of each step's mean depth
    (sum_i i f_i over the step's depth fractions, each step weighted equally). A step logged twice (a
    relay resumed past it) counts once, its last record. Also the mean depth fractions and, from them,
    the active fraction a_k at each recursion step k (the tokens whose depth exceeds k), which the
    realised FLOPs per token use. None without depth fractions."""
    by = {}
    for e in evs:
        if e.get("event") == "step" and e.get("mor_frac"):
            by[e["step"]] = e["mor_frac"]
    if not by:
        return None
    R = len(next(iter(by.values())))
    steps = sorted(by)
    m = sum(sum((i + 1) * f for i, f in enumerate(by[t])) for t in steps) / len(steps)
    fm = [sum(by[t][i] for t in steps) / len(steps) for i in range(R)]
    return {"m": m, "steps_logged": len(steps), "first_step": steps[0], "last_step": steps[-1],
            "frac_mean": fm, "active": [sum(fm[k:]) for k in range(R)]}


def mor_depth(spec_path, items, out_dir):
    """H-MoR2's depth trace (tools/carry5_depth.py) on the SFT model; its depth.json, or {"error"}."""
    argv = [sys.executable, MOR_DEPTH_TOOL, "run", "--model", spec_path, "--mtstudio", BIN, "--out", out_dir,
            "--set", items, "--families", ",".join(CONTEXT_FAMILIES)]
    p = subprocess.run(argv, capture_output=True, text=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "depth.log"), "w", encoding="utf-8") as f:
        f.write(p.stdout[-20000:] + p.stderr[-20000:])
    dp = os.path.join(out_dir, "depth.json")
    if p.returncode != 0 or not os.path.exists(dp):
        return {"error": f"carry5_depth failed (rc {p.returncode}): {p.stderr[-800:]}"}
    return jload(dp)


def preflight(items):
    missing = [p for p in (CARRY5_TOOL, items) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Carry-5 is not in this commit: missing {missing}")


# ---------------------------------------------------------------- Carry-12 in the post job (the long programme)
# experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md sections 2, 6, 7 and 10: after Carry-5, the SFT model
# is scored on Carry-12 by tools/carry12_eval.py in the post job's registered configuration: forced choice in gold,
# self, near, twin and noctx by `mtstudio score`; self history played to the target turn only at --max-new 48 by
# ember's server (the same build as Carry-5's, at the jobs file's --ember-commit; EMBER_GEMM_TF32=0 and
# EMBER_DETERMINISTIC=1, which carry12_eval sets) after its spot-check against `mtstudio chat`; no near_self;
# every reply through the parity gate. --carry12-gen none scores forced choice only (cell W512's re-run SFT).
CARRY12_TOOL = os.environ.get("TINY_CHAT_CARRY12", "tools/carry12_eval.py")  # override: tests only
CARRY12_MAX_NEW = 48


def carry12_read(out_dir):
    """What the post summary keeps of carry12_eval's summary (OUT/carry12/summary.json comes home whole): the
    pools and derived quantities (Y12, the decomposition, the validity gate), the curves analyze.py reads
    (by family and distance, by distance), overflow, HL0's slopes, the engine, the parity gate and the timing."""
    s = jload(os.path.join(out_dir, "summary.json"))
    m = s["models"]["m"]
    comp = (m.get("gen") or {}).get("complies", {})
    return {"meta": s["meta"], "context": m.get("context"), "items": m.get("items"), "derived": m["derived"],
            "pools": m["pools"], "family_distance": m["fc"].get("family_distance"),
            "distance": {mode: m["fc"]["acc"][mode]["distance"] for mode in ("gold", "near", "self", "twin")},
            "over_ctx": m["fc"]["over_ctx"], "hl0": m.get("hl0"), "gen_engine": m.get("gen_engine"),
            "gate": m.get("gate"), "server_errors": (m.get("gen") or {}).get("server_errors"),
            "instruction_complies": {mode: v["family"].get("instruction") for mode, v in comp.items()},
            "timing": m.get("timing")}


def run_carry12(a, spec_path, weights):
    """Carry-12 on the SFT model (SPEC_PATH, its safetensors WEIGHTS): OUT/carry12 by carry12_eval, resumable;
    returns carry12_read's record plus the run's settings, or {"error"} (the job then exits 1 after its summary)."""
    out = os.path.join(a.out, "carry12")
    rec = {"items": a.carry12_items, "items_sha256": sha256(a.carry12_items), "gen_requested": a.carry12_gen,
           "max_new": CARRY12_MAX_NEW, "self_scope": "target", "near_self": False, "spot": a.carry12_spot}
    gen, extra = a.carry12_gen, []
    if gen != "none":
        g, why = ensure_f32_gguf(spec_path, weights, os.path.join(a.out, "mtstudio.log"))
        if g is None:
            return {**rec, "error": f"Carry-12: {why}"}
    if gen == "ember":
        eb = ensure_ember(os.path.join(a.out, "ember_build.log"), a.ember_commit)
        rec["ember_build"] = eb
        if "error" in eb:
            gen = "mtstudio"
            rec["flag"] = f"ember unavailable ({eb['error'][:300]}); Carry-12 generated with mtstudio"
        else:
            extra = ["--ember-bin", eb["bin"], "--gpu-layers", str(EMBER_GPU_LAYERS), "--spot", str(a.carry12_spot)]
    argv = [sys.executable, CARRY12_TOOL, "run", "--model", f"m={spec_path}", "--out", out, "--set", a.carry12_items,
            "--scorer", "mtstudio", "--mtstudio", BIN, "--gen", gen, "--self-scope", "target",
            "--max-new", str(CARRY12_MAX_NEW), "--no-near-self", "--port", str(port_for(out)), *extra]
    rec["argv"] = argv[1:]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True)
    rec["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "carry12.log"), "w", encoding="utf-8") as f:
        f.write(p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0 or not os.path.exists(os.path.join(out, "summary.json")):
        return {**rec, "error": f"carry12_eval failed (rc {p.returncode}): {p.stderr[-800:]}"}
    return {**rec, **carry12_read(out)}


def cmd_post(a):
    done = os.path.join(a.out, "summary.json")
    shape = shape_spec(a.spec)
    custom = shape["arch"]["custom"]
    mor = is_mor(custom)
    c5_on = not getattr(a, "no_carry5", False)          # off only for cell W512's re-run (Carry-12 alone)
    c12_on = bool(getattr(a, "carry12_items", None))    # the long programme's cells: Carry-12 after Carry-5
    if not c5_on and not c12_on:
        raise SystemExit("post --no-carry5 needs --carry12-items: the job would score nothing")
    if os.path.exists(done):
        prev = jload(done)
        depth_ok = not mor or (bool(prev.get("mor_depth")) and "error" not in prev["mor_depth"])
        c5_ok = not c5_on or prev.get("primary") is not None
        c12_ok = not c12_on or (bool(prev.get("carry12")) and "error" not in prev["carry12"])
        sp_ok = not getattr(a, "sft_panels", False) or bool(prev.get("sft_panels"))
        if c5_ok and depth_ok and c12_ok and sp_ok:
            print(f"{done}: present")
            return 0
    preflight(a.carry5_items)
    if c12_on and not os.path.exists(CARRY12_TOOL):
        raise SystemExit(f"Carry-12 is not in this commit: missing {CARRY12_TOOL}")
    sft = jload(a.sft_template)
    sft.pop("_comment", None)
    b, acc = sft_batching(a.sft_batching, shape, sft["data"]["T"])
    if not os.path.exists(a.pt_weights):
        raise SystemExit(f"pretrained weights not found: {a.pt_weights}")
    s = {"mode": "post", "cell": a.cell, "share": a.share, "seed": a.seed, "spec": a.spec, "arch": custom,
         "shape": jload(a.spec).get("_shape"), "pt_lr": a.pt_lr if a.pt_lr is not None else shape["train"]["lr"],
         "pt_weights_sha256": sha256(a.pt_weights),
         "gpu": gpu_name(), "provenance": provenance()}
    if a.sft_batching != "depth":  # amendment 6; the default path writes the summary as before
        s["sft_batching"] = {"mode": a.sft_batching, "batch": b, "accum": acc}
    # Pretraining receipts, when this VM ran the job (on a fresh VM they are home already).
    if os.path.exists(os.path.join(a.pt_dir, "result.json")):
        s["pt_result"] = jload(os.path.join(a.pt_dir, "result.json"))
    s["pt_evals"] = [(e["step"], e["val_loss"]) for e in events(a.pt_dir) if e.get("event") == "eval"]
    # 1. Panels: the zero-rate pass (1 step at lr 0 from the weights) reads the panel's held-out windows.
    s["panels"] = zero_rate_panels(shape, a.pt_weights, a.out, a.panel)
    # 2. SFT from the pretrained weights (resumes from its own checkpoint on this VM).
    model = f"tc_sft_{a.cell}_r{a.seed}"
    sd = os.path.join(a.out, "sft")
    spec = {"name": model, "arch": shape["arch"], "data": sft["data"],
            "train": {**sft["train"], "batch": b, "accum": acc, "seed": a.seed, "init_from": a.pt_weights},
            "export": sft["export"]}
    t0 = time.time()
    if not os.path.exists(os.path.join(sd, "result.json")):
        rc, rate = run_mtstudio(spec, sd, os.path.join(a.out, "mtstudio.log"), t_from=20)
        if rc != 0:
            raise SystemExit(f"SFT failed (rc {rc}): {log_tail(os.path.join(a.out, 'mtstudio.log'), 1500)}")
        s["sft_tok_per_s_steady"] = rate
    s["sft_seconds"] = round(time.time() - t0, 1)
    s["sft_result"] = jload(os.path.join(sd, "result.json"))
    s["sft_evals"] = [(e["step"], e["val_loss"]) for e in events(sd) if e.get("event") == "eval"]
    sw = os.path.join(sd, model + ".safetensors")
    s["sft_weights_sha256"] = sha256(sw)
    s["sft_data_sha256"] = sha256(sft["data"]["sft_examples"])
    if getattr(a, "sft_panels", False):  # the long programme's B, D and T: the same panels at the SFT weights
        s["sft_panels"] = zero_rate_panels(shape, sw, a.out, a.panel, prefix="sft_panel")
    # 3. Carry-5: the SFT model with generated (self) history (required), then the base model
    #    teacher-forced only (descriptive; a failure is recorded, not fatal).
    #    Generation: --gen ember picks the engine by choose_engine (ember's CUDA server when its spot-check
    #    against mtstudio chat is N/N, else mtstudio for the whole run, flagged); --gen mtstudio as before.
    ember_env = {**EMBER_ENV, "EMBER_DETERMINISTIC": "1"} if getattr(a, "ember_deterministic", False) else None
    if c5_on:
        carry5_stage(a, s, sd, sw, custom, mor, shape, ember_env)
    else:
        s["primary"] = None
        s["carry5"] = {"skipped": "post --no-carry5 (Carry-12 alone)"}
    # 4. Carry-12 after Carry-5 (the long programme's cells).
    if c12_on:
        s["carry12"] = run_carry12(a, sd + "_spec.json", sw)
        d = s["carry12"].get("derived") or {}
        s["Y12"], s["Y12_gold"] = d.get("Y12_self"), d.get("Y12_gold")
    s["other_mtstudio_at_end"] = concurrent_mtstudio()
    jsave(os.path.join(a.out, "summary.json"), s)
    print(json.dumps({k: s.get(k) for k in ("cell", "seed", "primary", *(("Y12",) if c12_on else ()), "panels")}))
    if mor and "error" in s.get("mor_depth", {}):
        print(s["mor_depth"]["error"], file=sys.stderr)
        return 1
    if c12_on and "error" in s["carry12"]:
        print(s["carry12"]["error"], file=sys.stderr)
        return 1
    return 0


def carry5_stage(a, s, sd, sw, custom, mor, shape, ember_env=None):
    """3. Carry-5 on the SFT model (self history generated), then the base model teacher-forced; fills S."""
    c5 = os.path.join(a.out, "carry5")
    env, ember = dict(os.environ), None
    if a.gen == "ember":
        eng = choose_engine(sd + "_spec.json", sw, a.carry5_items, a.out, custom, a.ember_commit, ember_env)
        s["gen_engine"] = eng
        if eng["flag"]:
            s["gen_flag"] = eng["flag"]
            print(f"generation: {eng['flag']}", file=sys.stderr)
        if eng["engine"] == "ember":
            ember = eng["ember"]
            env.update(eng["ember_env"])
    else:
        s["gen_engine"] = {"engine": "mtstudio", "requested": "mtstudio", "flag": None}
    t0 = time.time()
    p = subprocess.run(carry5_argv(sd + "_spec.json", a.carry5_items, c5, "ember" if ember else "mtstudio",
                                   port_for(a.out), ember=ember), capture_output=True, text=True, env=env)
    s["carry5_seconds"] = round(time.time() - t0, 1)
    open(os.path.join(a.out, "carry5.log"), "w", encoding="utf-8").write(p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0 or not os.path.exists(os.path.join(c5, "summary.json")):
        raise SystemExit(f"carry5_eval failed (rc {p.returncode}): {p.stderr[-500:]}")
    s["carry5"] = carry5_read(c5)
    if s["carry5"]["meta"].get("gen") != s["gen_engine"]["engine"]:
        raise SystemExit(f"{c5} was generated with {s['carry5']['meta'].get('gen')}, this run chose "
                         f"{s['gen_engine']['engine']}: engines are never mixed; remove {c5} and rerun")
    # ember returns token ids, so every self-history reply is also scored by `mtstudio score` as a
    # continuation (carry5_eval's parity gate): the whole run's argmax agreement, beside the spot-check.
    s["gen_engine"]["full_gate"] = s["carry5"]["summary"].get("gate")
    s["primary"] = primary(s["carry5"])
    s["primary_all_items"] = primary_all_items(s["carry5"])  # section 3's frozen definition, reported beside it
    if mor:  # amendment 4: m_s from the pretraining events (when this VM ran it) and H-MoR2's depth trace
        s["pt_mor"] = realised_depth(events(a.pt_dir))
        s["mor_depth"] = mor_depth(sd + "_spec.json", a.carry5_items, os.path.join(a.out, "mor_depth"))
    s["carry5_base"] = carry5_base(shape, a.pt_weights, a.carry5_items, a.out)
    s["final_rung"] = rung_metrics(s["carry5_base"])  # the 100% rung of the early-ranking analysis


PICK_RULE = "highest primary metric (forced choice, self history, multi minus twin); tie -> lower seed"


def pick_row(d):
    """What pick reads of a post job's run directory d: its summary.json's seed and primary, nothing else."""
    p = os.path.join(d, "summary.json")
    sm = jload(p) if os.path.exists(p) else {}
    return {"run": d, "seed": sm.get("seed"), "primary": sm.get("primary"), "summary": os.path.exists(p)}


def pick_best(rows):
    """PICK_RULE over pick_row rows: the highest primary, a tie to the lower seed; None when no row has a
    primary. runner_multi.py applies this function to the home summaries on a replacement VM, so only the
    post job pick can still choose is rerun there (the others' summary.json goes up instead)."""
    ok = [r for r in rows if r["primary"] is not None]
    return max(ok, key=lambda r: (r["primary"], -r["seed"])) if ok else None


def cmd_pick(a):
    rows = [pick_row(d) for d in a.run]
    best = pick_best(rows)
    out = {"rule": PICK_RULE, "runs": rows, "best": best}
    rc = 1
    if best:
        sd = os.path.join(best["run"], "sft")
        # The SFT model's own files, named after its spec (24 Oct 2026: a glob over *.safetensors
        # picked the optimiser file, so the pilot's pick brought home AdamW state as its "model").
        name = jload(sd + "_spec.json").get("name") if os.path.exists(sd + "_spec.json") else None
        w = os.path.join(sd, f"{name}.safetensors") if name else ""
        g = os.path.join(sd, f"{name}.gguf") if name else ""
        want = jload(os.path.join(best["run"], "summary.json")).get("sft_weights_sha256")
        if name and os.path.exists(w) and os.path.exists(g):
            got = sha256(w)
            if want and got != want:
                out["weights"] = f"{w} has sha256 {got}, not the post job's SFT weights {want}; not copied"
            else:
                shutil.copyfile(w, os.path.join(a.out, "model.safetensors"))
                shutil.copyfile(g, os.path.join(a.out, "model.gguf"))
                shutil.copyfile(sd + "_spec.json", os.path.join(a.out, "spec.json"))
                out["weights_sha256"] = got
                out["weights_match_summary"] = bool(want)
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
    s["provenance"] = provenance()
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
            raise SystemExit(f"mtstudio failed (rc {rc}): {log_tail(os.path.join(a.out, 'mtstudio.log'), 1500)}")
    ev = events(run_dir)
    losses = [e["loss"] for e in ev if e.get("event") == "step"]
    val = final_eval(run_dir)
    diverged = val is None or not math.isfinite(val) or bool(losses and val > losses[0])
    s = {"mode": "asha_train", "config": a.config, "round": a.round, "budget": a.budget, "spec": a.spec,
         "arch": shape["arch"]["custom"], "lr": a.lr, "warmup_frac": a.warmup_frac, "share": a.share, "seed": a.seed,
         "final_val": val, "diverged": diverged, "first_loss": losses[0] if losses else None,
         "last_loss": losses[-1] if losses else None, "tok_per_s_steady": rate,
         "evals": [(e["step"], e["val_loss"]) for e in ev if e.get("event") == "eval"], "gpu": gpu_name(),
         "acc_gold": None, "Y_base": None, "provenance": provenance()}
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
    if a.need_pilot:  # amendment 3: E5/E1 at 20M and the recursion arm need PASS; E5 at 60M needs FAIL
        v = jload(a.pilot_verdict).get("verdict") if a.pilot_verdict and os.path.exists(a.pilot_verdict) else None
        if v != a.need_pilot:
            problems.append(f"floor pilot verdict {v} ({a.pilot_verdict}), this block needs {a.need_pilot} "
                            "(PREREGISTRATION.md amendment 3)")
    if a.need_cuda_gate:  # A3.3: the recursion arm's runs need the device legs validated on an L4
        g = jload(a.need_cuda_gate) if os.path.exists(a.need_cuda_gate) else {}
        if not g.get("ok"):
            problems.append(f"{a.need_cuda_gate}: no passed CUDA gate (Stage 0's cuda_gate.json, copied home and "
                            "committed)")
    for f in a.need_ok or []:  # amendment 5: home receipts (the plain replay's bit-identity, ember's gate)
        r = jload(f) if os.path.exists(f) else {}
        if r.get("ok") is not True:
            problems.append(f"{f}: ok is {r.get('ok')}; this block needs that receipt passed and committed")
    for f in getattr(a, "need_file", None) or []:  # the Carry-12 cells: the frozen PREREGISTRATION.md
        if not os.path.exists(f):
            problems.append(f"{f} missing: this block launches only once that file is committed")
    if a.need_rec_r8:  # A3.3's R8 rule, read at home (analyze.py --rec-decision) and committed
        d = jload(a.need_rec_r8) if os.path.exists(a.need_rec_r8) else {}
        if d.get("run_r8") is not True:
            problems.append(f"{a.need_rec_r8}: run_r8 is {d.get('run_r8')}; R8 runs only on A3.3's rule")
    if a.stage0:
        if a.need_carry5:
            problems += [f"{p} missing: the rung metrics cannot be computed" for p in (CARRY5_TOOL, a.carry5_items)
                         if not os.path.exists(p)]
        os.makedirs(a.out, exist_ok=True)
        jsave(os.path.join(a.out, "gate.json"), {"ok": not problems, "problems": problems})
        print(json.dumps({"ok": not problems, "problems": problems}))
        return 1 if problems else 0
    choice = jload(a.choice)["lr"] if os.path.exists(a.choice) else None
    if a.pilot_lr and os.path.exists(a.pilot_lr):  # amendment 3: the pilot's own rate, for its shape only
        pl = jload(a.pilot_lr)
        if choice is None or pl["shape"] not in choice:
            choice = {**(choice or {}), pl["shape"]: pl["lr"]}
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
    wins. A shape whose screens all diverged stops the programme (no rate is chosen). Amendment 2:
    a FAILED screen (screen_status: a crash with finite losses) is no evidence about its rate and is
    rerun unchanged; --art may be given several times (the first attempt, then its reruns) and each
    screen is read from the one folder that holds it FINISHED or DIVERGED. A shape is chosen only
    when all its screens are FINISHED or DIVERGED; otherwise nothing is written."""
    grid = tuple(getattr(a, "grid", None) or SCREEN_LRS_EXPECTED)
    drop = bool(getattr(a, "drop_diverged", False))
    found = {}
    for k, art in enumerate(a.art):
        for p in sorted(glob.glob(os.path.join(art, "screen_*", "summary.json"))):
            job = os.path.basename(os.path.dirname(p))
            s = jload(p)
            found.setdefault(job, []).append((k, art, s, screen_status(s)))
    by, problems = {}, []
    for job, rows in sorted(found.items()):
        sid = job[len("screen_"):].rsplit("_lr", 1)[0]
        valid = [r for r in rows if r[3] != "FAILED"]
        if len(valid) > 1:
            problems.append(f"{job}: valid in {[r[1] for r in valid]}; a finished or diverged screen is never rerun")
            continue
        if not valid:
            problems.append(f"{job}: FAILED in every folder ({[r[1] for r in rows]}); rerun it")
            by.setdefault(sid, [])
            continue
        _, art, s, st = valid[0]
        by.setdefault(sid, []).append({**s, "status": st, "art": art, "job": job,
                                       "attempts": [{"art": r[1], "status": r[3], "rc": r[2].get("rc")} for r in rows]})
    for sid, ss in by.items():
        if len(ss) != len(grid) and not any(p.startswith(f"screen_{sid}_") for p in problems):
            problems.append(f"{sid}: {len(ss)} screens found, {len(grid)} registered")
        elif len(ss) == len(grid) and sorted(float(s["lr"]) for s in ss) != sorted(grid):
            problems.append(f"{sid}: screened at {sorted(s['lr'] for s in ss)}, registered {sorted(grid)}")
    if problems:
        raise SystemExit("no rate is chosen:\n  " + "\n  ".join(problems))
    lr, detail, dropped = {}, {}, {}
    for sid, ss in sorted(by.items()):
        good = [s for s in ss if s["status"] == "FINISHED"]
        if not good and drop:  # A3.3: a recursion cell whose four screens all diverge is dropped and reported
            dropped[sid] = f"all {len(ss)} screens diverged (A3.3: the cell is dropped)"
            detail[sid] = [{k: s.get(k) for k in ("lr", "final_val", "status", "first_loss", "last_loss", "job", "art")}
                           for s in sorted(ss, key=lambda x: x["lr"])]
            continue
        if not good:
            raise SystemExit(f"{sid}: every screen diverged; no rate is chosen (section 4: one amendment may "
                             "extend the grid downwards)")
        best = min(s["final_val"] for s in good)
        tied = [s for s in good if s["final_val"] <= best + SCREEN_TIE]
        lr[sid] = min(s["lr"] for s in tied)
        detail[sid] = [{k: s.get(k) for k in ("lr", "final_val", "status", "first_loss", "last_loss",
                                              "tok_per_s_steady", "job", "art", "attempts")}
                       for s in sorted(ss, key=lambda x: x["lr"])]
    out = {"rule": " ".join(cmd_choose.__doc__.split()), "tie_nats": SCREEN_TIE, "lr": {}, "screens": {}}
    if drop:
        out.update({"grid": list(grid), "dropped": {},
                    "rule_rec": "A3.3: the same rule over the four-rate grid; a cell whose screens all diverged "
                                "is dropped and reported"})
    if os.path.exists(a.out):  # the 60M stage adds its shapes to the 20M choice; a shape is chosen once
        out = jload(a.out)
        again = sorted((set(lr) | set(dropped)) & (set(out["lr"]) | set(out.get("dropped", {}))))
        if again:
            raise SystemExit(f"{a.out} already holds a choice for {again}")
    out["lr"].update(lr)
    out["screens"].update(detail)
    if drop:
        out.setdefault("dropped", {}).update(dropped)
    lr = out["lr"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jsave(a.out, out)
    print(json.dumps({"lr": lr, **({"dropped": out.get("dropped", {})} if drop else {})}))
    return 0


def screen_rule(ss):
    """Section 4's rule among FINISHED screens: the lowest final validation loss; within SCREEN_TIE
    nats of it, the smallest lr. None when no screen finished."""
    good = [s for s in ss if s["status"] == "FINISHED"]
    if not good:
        return None
    best = min(s["final_val"] for s in good)
    return min(s["lr"] for s in good if s["final_val"] <= best + SCREEN_TIE)


def cmd_pilot_lr(a):
    """The floor pilot's learning rate (amendment 3), for --shape (20m_L8). (0) When the shape's
    three screens are all FINISHED or DIVERGED, section 4's rule exactly as choose applies it
    ("registered"); if all three diverged, 2.5e-4, the first rate of section 4's downward extension
    ("all_diverged"). Otherwise the pilot waits for the rerun session; once it has ended
    (--session-ended), the registered fallback, first that applies: (F1) section 4's rule over the
    shape's FINISHED screens; (F2) the smaller of the rates the same rule gives the shape's
    neighbours in depth at its size (20m_L4 and 20m_L16 for 20m_L8), over their FINISHED screens;
    (F3) the smallest rate among the size's FINISHED screens. The rate is the pilot's only: it never
    enters lr_choice.json."""
    found = {}
    for k, art in enumerate(a.art):
        for p in sorted(glob.glob(os.path.join(art, "screen_*", "summary.json"))):
            job = os.path.basename(os.path.dirname(p))
            s = jload(p)
            found.setdefault(job, []).append({**s, "status": screen_status(s), "art": art, "job": job})
    by = {}
    for job, rows in sorted(found.items()):
        sid = job[len("screen_"):].rsplit("_lr", 1)[0]
        valid = [r for r in rows if r["status"] != "FAILED"]
        if len(valid) > 1:
            raise SystemExit(f"{job}: valid in {[r['art'] for r in valid]}; a finished or diverged screen is never rerun")
        by.setdefault(sid, []).append(valid[0] if valid else rows[-1])
    size, depth = a.shape.split("_L")
    own = by.get(a.shape, [])
    resolved = len(own) == len(SCREEN_LRS_EXPECTED) and all(s["status"] != "FAILED" for s in own)
    lr, source = None, None
    if resolved:
        lr = screen_rule(own)
        source = "registered" if lr is not None else "all_diverged"
        if lr is None:
            lr = SCREEN_LRS_EXPECTED[0] / 2
    elif not a.session_ended:
        raise SystemExit(f"{a.shape}: {len(own)} of {len(SCREEN_LRS_EXPECTED)} screens resolved; the pilot waits for "
                         "the rerun session (rerun with --session-ended once it has ended)")
    else:
        lr, source = screen_rule(own), "F1_finished_screens"
        if lr is None:
            # E1's grid doubles the depth: the neighbours of L are L/2 and 2L (20m_L4, 20m_L16 for 20m_L8)
            nb = [n for n in (f"{size}_L{int(depth) // 2}", f"{size}_L{int(depth) * 2}") if n in by]
            got = [x for x in (screen_rule(by[n]) for n in nb) if x is not None]
            lr, source = (min(got), f"F2_neighbours_{'_'.join(nb)}") if got else (None, None)
        if lr is None:
            fin = [s["lr"] for sid, ss in by.items() if sid.startswith(size + "_L") for s in ss
                   if s["status"] == "FINISHED"]
            lr, source = (min(fin), f"F3_smallest_finished_{size}") if fin else (None, None)
        if lr is None:
            raise SystemExit(f"no {size} screen FINISHED: the pilot has no rate (amendment 3: Jonathan decides)")
    out = {"shape": a.shape, "lr": lr, "source": source, "pilot_only": True, "arts": a.art,
           "rule": " ".join(cmd_pilot_lr.__doc__.split()),
           "screens": {sid: [{k: s.get(k) for k in ("lr", "final_val", "status", "job", "art")} for s in ss]
                       for sid, ss in sorted(by.items())}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jsave(a.out, out)
    print(json.dumps({"shape": a.shape, "lr": lr, "source": source}))
    return 0


def vm_flags(p):
    p.add_argument("--cuda-gate", default=None, help="A3.3: refuse unless DIR/cuda_gate.json passed on this VM "
                                                     "(a fresh VM runs the gate into DIR first)")
    p.add_argument("--data-job", default=None, help="a jobs file whose data job rebuilds a missing corpus here")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--steps", type=int, default=300)
    vm_flags(p)
    p = sub.add_parser("screen")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    vm_flags(p)
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
    p.add_argument("--gen", choices=["ember", "mtstudio"], default="mtstudio",
                   help="Carry-5's generation: ember = ember.cpp's server at EMBER_COMMIT when the per-run spot-check "
                        "against mtstudio chat is N/N, else mtstudio for the whole run (flagged)")
    p.add_argument("--sft-batching", choices=["depth", "spec"], default="depth",
                   help="SFT (batch, accum): depth = by effective depth (every registered arm); spec = the shape spec's "
                        "pretraining micro-batching (amendment 6, the 150M cells)")
    p.add_argument("--ember-deterministic", action="store_true",
                   help="EMBER_DETERMINISTIC=1 for Carry-5's ember servers too (the Carry-12 cells' post jobs)")
    p.add_argument("--carry12-items", default=None,
                   help="the long programme: score the SFT model on Carry-12 after Carry-5 (tools/carry12_eval.py)")
    p.add_argument("--carry12-gen", choices=["ember", "mtstudio", "none"], default="ember",
                   help="Carry-12's generation (none: forced choice only, cell W512's re-run)")
    p.add_argument("--carry12-spot", type=int, default=20, help="Carry-12's ember vs mtstudio spot-check items")
    p.add_argument("--no-carry5", action="store_true",
                   help="skip Carry-5 and base-model Carry-5 (cell W512's re-run SFT: Carry-12 alone)")
    p.add_argument("--sft-panels", action="store_true",
                   help="also read the panels at the SFT weights (summary sft_panels; Carry-12 cells B, D and T)")
    p.add_argument("--ember-commit", default=None,
                   help="the ember commit to generate with (default EMBER_COMMIT; amendment 5 pins its own in the jobs file)")
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
    p.add_argument("--pilot-lr", default="", help="amendment 3: the floor pilot's rate (pilot_lr.json), used for its "
                                                  "shape only when the choice lacks that shape")
    p.add_argument("--pilot-verdict", default="", help="amendment 3: the floor pilot's verdict.json")
    p.add_argument("--need-pilot", choices=["PASS", "FAIL"], default=None,
                   help="amendment 3: refuse unless the pilot verdict is this")
    p.add_argument("--need-cuda-gate", default="", help="A3.3: refuse unless this cuda_gate.json passed")
    p.add_argument("--need-file", action="append", default=[],
                   help="refuse unless this file exists (the Carry-12 cells: their frozen PREREGISTRATION.md)")
    p.add_argument("--need-rec-r8", default="", help="A3.3: refuse unless this rec_decision.json has run_r8 true")
    p.add_argument("--need-ok", action="append", default=[],
                   help="amendment 5: refuse unless this JSON receipt has ok true (repeatable)")
    p = sub.add_parser("cuda_gate")
    p.add_argument("--out", required=True)
    p.add_argument("--script", default=CUDA_SCRIPT)
    p.add_argument("--build-dir", default=None, help="default: mtstudio's build directory (dirname of JOBS_MTSTUDIO)")
    p = sub.add_parser("choose_rec")
    p.add_argument("--art", action="append", required=True, help="the recursion Stage 0 folders, first attempt first")
    p.add_argument("--out", required=True)
    p.set_defaults(grid=REC_SCREEN_LRS, drop_diverged=True)
    p = sub.add_parser("pilot_lr")
    p.add_argument("--art", action="append", required=True, help="Stage 0 folders, first attempt first (as choose)")
    p.add_argument("--shape", default="20m_L8")
    p.add_argument("--session-ended", action="store_true",
                   help="the rerun session has ended: allow the registered fallback (amendment 3)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--art", action="append", required=True,
                   help="a Stage 0 artefact folder; repeat for reruns (amendment 2), first attempt first")
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
    if a.mode not in ("choose", "pilot_lr", "choose_rec"):
        os.makedirs(a.out, exist_ok=True)
    return {"probe": cmd_probe, "screen": cmd_screen, "post": cmd_post, "pick": cmd_pick, "gate": cmd_gate,
            "choose": cmd_choose, "rung": cmd_rung, "asha_train": cmd_asha_train,
            "asha_promote": cmd_asha_promote, "pilot_lr": cmd_pilot_lr, "cuda_gate": cmd_cuda_gate,
            "choose_rec": cmd_choose}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
