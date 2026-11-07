#!/usr/bin/env python3
"""The arm's whole read path before its first training step (PROTOCOL.md section 31). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b2_precheck.py --out OUT --jobs tools/colab_chat/tc_b1_r1.json --train pt_b1_r1 --rung rung_b1_s1000 --ember SHA

Stage 0 never ran the rung's GGUF export on CUDA, so a bad export could first show on a raced model, and differ between the arms. This job
proves the path on the arm's own shape and mix, in a scratch directory, and nothing from it reaches the run:

1. a short run of the arm's real training spec, with its real environment, to a transactional checkpoint and an evaluation;
2. that trainer stopped with SIGSTOP and still holding its GPU memory, as jobs_vm.py stops it for an exclusive rung, and the arm's registered rung
   command on a snapshot of the checkpoint, unchanged apart from where it reads and writes: both panels, the F32 GGUF, Carry-5. The trainer
   is then released, and must reach its next checkpoint, as a raced trainer must after a rung;
3. the GGUF must declare the arm's state tokens;
4. greedy parity of ember's server (CUDA and CPU, the commit the final benchmarks score with) against mtstudio on a few frozen prompts, and
   ember's POST /score, the path of the race's LAMBADA read, against `mtstudio score` on multi-turn ids that carry state blocks;
5. panel time and GPU memory measured, with the rung inside section 30's 30 minutes.

Writes OUT/precheck.json. Exit 1 when anything fails: the job is halt_on_failure, the arm halts with its volume, and pt_* never starts. Rung 1,000 then confirms; it is not the first proof.
"""
import argparse
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
import tc_b2_probe as P  # noqa: E402

STEPS = 60                                  # the checkpoint and evaluation the rung reads; the trainer is stopped soon after
TRAIN_STEPS = 3 * STEPS                     # never reached: the run must still be alive when it is stopped
PARITY_N = 5
TRAIN_TIMEOUT_S = 900
PARITY_TIMEOUT_S = 2400
RESUME_TIMEOUT_S = 900
JOB_TIMEOUT_S = 7200                        # the whole job: `mtstudio score` and the ember build have no timeout of their own
EMBER_TRIES = 3
EMBER_DIR = "/content/ember_score"          # the bench job's own build directory, so the build is not paid twice
SPEC_VM = "/content/chat/precheck_spec.json"  # chat_parity_vm.py reads CHAT/JOB_spec.json
COST_RUNG_S = 900                           # race_costs.py's RUNG_H: over this is a cost to record
MAX_RUNG_S = 1800                           # section 30 stops an arm by hand over 30 minutes
SCORE_N = 5
SCORE_TOL = 0.002                           # nats, on any one token, between engines
PORT = 8197
ON_GPU = "IR interpreter (CUDA;"            # ember's line (api.cpp), a warning that reaches only ./debugging.log: a CUDA build falls to the CPU without an error


def short_spec(spec, run):
    """The arm's training spec, cut to a short run in a scratch directory. Shape, mix, batch, precision and seed stay the arm's."""
    s = json.loads(json.dumps(spec))
    s["name"] = "precheck"
    s["out_dir"] = run
    s["train"].update(steps=TRAIN_STEPS, checkpoint_every=STEPS, eval_every=STEPS, lr_schedule={"warmup_steps": 10, "decay": "cosine", "min_lr_frac": 0.1})
    s["export"] = {"formats": ["safetensors"]}
    return s


def rung_argv(cmd, out, weights, pt, step):
    """The registered command, reading and writing elsewhere and nothing else."""
    a = [out if x == "{out}" else x for x in cmd]
    for flag, value in (("--weights", weights), ("--pt-dir", pt), ("--step", str(step))):
        a[a.index(flag) + 1] = value
    return a


def parity_ok(s, n):
    if not s:
        return False
    return all(s.get(leg, {}).get(k) == n for leg in ("cuda", "cpu") for k in ("n", "reply_identical", "token_count", "stop_reason", "prompt_tokens"))


def score_ok(s, n):
    if not s or not s.get("on_gpu"):
        return False
    return all(s.get(leg, {}).get("n") == n and s[leg].get("argmax_equal") is True and s[leg].get("max_nats_per_token") is not None
               and s[leg]["max_nats_per_token"] <= SCORE_TOL for leg in ("cuda", "cpu"))


def verdict(rep):
    """Pure, for the tests: every bar on the measured numbers."""
    bad = [f"the check stopped: {rep['error']}"] if rep.get("error") else []
    if rep.get("train_rc") != 0:
        bad.append(f"the short run failed ({rep.get('train_rc')})")
    if rep.get("rung_rc") != 0:
        bad.append(f"the rung path failed (rc {rep.get('rung_rc')})")
    if rep.get("trainer_held") is not True:
        bad.append("the trainer did not stay stopped through the rung")
    if rep.get("resumed_step") is None:
        bad.append("the trainer did not reach its next checkpoint after the rung")
    if not isinstance(rep.get("val_loss"), (int, float)) or not math.isfinite(rep["val_loss"]):
        bad.append(f"val_loss {rep.get('val_loss')}")
    if rep.get("acc_gold") is None:
        bad.append("no Carry-5 reading")
    panels = rep.get("panels") or {}
    if not panels or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in panels.values()):
        bad.append(f"panels {panels}")
    if rep.get("gguf_state_k") != rep.get("want_state_k") or not rep.get("gguf_state_k"):
        bad.append(f"the GGUF declares {rep.get('gguf_state_k')} state tokens, not {rep.get('want_state_k')}")
    if not rep.get("parity_ok"):
        bad.append("ember chat parity")
    if not rep.get("score_ok"):
        bad.append("ember /score parity, or ember not on the GPU")
    if not rep.get("peak_train_mib") or not rep.get("peak_total_mib"):
        bad.append("no memory reading")
    elif rep["peak_total_mib"] > P.DEVICE_MIB:
        bad.append(f"{rep['peak_total_mib']} MiB beside the stopped trainer is over {P.DEVICE_MIB}")
    s = rep.get("rung_seconds")
    if s is None or s > MAX_RUNG_S:
        bad.append(f"the rung took {s} s, past {MAX_RUNG_S} s")
    return bad


def phase_times(rung_out, t0):
    """Seconds from the rung's start at which each stage's own output appeared: cumulative, not durations."""
    def at(path):
        p = os.path.join(rung_out, path)
        return round(os.path.getmtime(p) - t0, 1) if os.path.exists(p) else None
    return {"panel_fineweb": at("panel_fineweb/result.json"), "panel_dialogue": at("panel_dialogue/result.json"), "export": at("model.gguf"),
            "carry5": at("carry5_base/summary.json")}


class Peak:
    """nvidia-smi over one phase: the highest reading."""

    def __enter__(self):
        self.seen, self.stop = [], threading.Event()
        self.t = threading.Thread(target=P.sampler, args=(self.stop, self.seen), daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(15)
        self.mib = max(self.seen) if self.seen else None


def gguf_kv(path):
    import llama3_bpe
    return llama3_bpe.read_gguf_kv(path, {"paperkiln.state_tokens.k", "tokenizer.ggml.tokens"})


def gguf_state_k(path):
    try:
        return int(gguf_kv(path).get("paperkiln.state_tokens.k") or 0)
    except (OSError, ValueError):
        return None


def read_ids(path, n):
    """The first n ids of a PKIDS001 file."""
    with open(path, "rb") as f:
        if f.read(8) != b"PKIDS001":
            raise ValueError(f"{path} is not an ids file")
        width = struct.unpack("<IIQ", f.read(16))[0]
        fmt = {1: "B", 2: "H", 4: "I"}[width]
        raw = f.read(n * width)
    return list(struct.unpack(f"<{len(raw) // width}{fmt}", raw))


def score_requests(gguf, panel, n=SCORE_N):
    """Several exchanges of panel text, laid out as the engines lay them, so a state block sits in every prompt; the continuation is the last reply."""
    import state_tokens
    kv = gguf_kv(gguf)
    toks, k = kv["tokenizer.ggml.tokens"], int(kv["paperkiln.state_tokens.k"])
    eot, user, assistant = (toks.index(t) for t in ("<|endoftext|>", "<|user|>", "<|assistant|>"))
    text = [t for t in read_ids(panel, 20000) if t < len(toks) and t not in (eot, user, assistant)]
    out, pos = [], 0
    for i in range(n):
        ids = []
        for _ in range(3 + i % 3):
            ids += [user] + text[pos:pos + 30] + [assistant] + text[pos + 30:pos + 70] + [eot]
            pos += 70
        ids += [user] + text[pos:pos + 30] + [assistant]
        cont = text[pos + 30:pos + 78]
        pos += 78
        full = state_tokens.insert(ids, len(toks), k, eot, user, assistant)[0]
        if sum(t >= len(toks) for t in full) < k or len(cont) != 48:
            raise ValueError("score request without a state block or a whole continuation")
        out.append({"key": f"s{i}", "prompt_ids": full, "continuation_ids": cont})
    return out


def diffs(ref, res):
    """Token by token against the reference, so that errors cannot cancel: the largest difference in nats, and whether every argmax agrees.
    max_nats_per_token is None when anything is missing or over the context."""
    worst, same = 0.0, len(res) == len(ref)
    for r in res:
        a = ref.get(r.get("key"))
        if a is None or r.get("over_ctx") or a.get("over_ctx") or not a.get("logprobs") or len(r.get("logprobs") or []) != len(a["logprobs"]):
            return {"max_nats_per_token": None, "argmax_equal": False}
        d = [abs(x - y) for x, y in zip(r["logprobs"], a["logprobs"])]
        if not all(math.isfinite(x) for x in d):
            return {"max_nats_per_token": None, "argmax_equal": False}
        worst, same = max(worst, *d), same and r.get("argmax") == a.get("argmax")
    return {"max_nats_per_token": worst if len(res) == len(ref) else None, "argmax_equal": same}


def serve(ember, gguf, layers, cwd, env):
    """Ember in a directory of its own: a server start truncates ./debugging.log, the only place its device line goes."""
    os.makedirs(cwd, exist_ok=True)
    p = subprocess.Popen([ember, gguf, str(PORT), "127.0.0.1", "-", layers], stdout=open(os.path.join(cwd, "server.log"), "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, env=env, cwd=cwd)
    for _ in range(900):
        if p.poll() is not None:
            raise RuntimeError(f"ember exited {p.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            return p
        except OSError:
            time.sleep(1)
    p.kill()
    raise RuntimeError("ember did not come up")


def score_parity(out, gguf, spec_path, panel, ember):
    """`mtstudio score`, then ember's /score on the same ids, on both of ember's devices. {"cuda", "cpu", "on_gpu"}; "error" when a leg could not run."""
    import carry5_eval
    os.makedirs(out, exist_ok=True)
    reqs = score_requests(gguf, panel)
    spec = json.load(open(spec_path))
    for k in ("precision", "attention_kernel"):      # both engines serve in FP32, as chat_parity_vm.py has it
        spec.get("train", {}).pop(k, None)
    sp, inp, raw = (os.path.join(out, x) for x in ("score_spec.json", "score_in.jsonl", "score_out.jsonl"))
    with open(sp, "w") as f:
        json.dump(spec, f, indent=1)
    with open(inp, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    carry5_eval.mtstudio_score(P.BIN, sp, inp, raw)
    ref = {}
    for line in open(raw):
        r = json.loads(line)
        ref[r["key"]] = {"logprobs": r.get("logprobs"), "argmax": r.get("argmax"), "over_ctx": r.get("over_ctx"), "sum": r.get("sum")}
    res = {"on_gpu": False, "reference": {k: v["sum"] for k, v in ref.items()}}
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    env.update(EMBER_GEMM_TF32="0", EMBER_DETERMINISTIC="1")
    body = json.dumps({"requests": reqs}).encode()
    for leg, layers in (("cuda", "-1"), ("cpu", "0")):
        cwd = os.path.join(out, f"ember_{leg}")
        p = serve(ember, gguf, layers, cwd, env)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/score", data=body, headers={"Content-Type": "application/json"})
            got = json.load(urllib.request.urlopen(req, timeout=900))["results"]
        finally:
            p.terminate()
            p.wait(30)
        res[leg] = {"n": len(got), **diffs(ref, got)}
        if leg == "cuda":
            res["on_gpu"] = ON_GPU in tail(os.path.join(cwd, "debugging.log"), 10 ** 7)
    return res


def proc_state(pid):
    """The process's state letter: T when stopped."""
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def timed_out(*_):
    raise TimeoutError(f"the job passed {JOB_TIMEOUT_S} s")


def live_step(live):
    try:
        return int(open(os.path.join(live, "state.txt")).readline().strip() or 0)
    except (OSError, ValueError):
        return None


def tail(path, n=1500):
    try:
        return open(path, errors="replace").read()[-n:]
    except OSError:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--ember", required=True)
    a = ap.parse_args(argv)
    os.setsid()                                # this job's children are one group, so a failure leaves nothing on the GPU
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(JOB_TIMEOUT_S)
    rep = {"producer": "Claude Opus 5.5, 7 Nov 2026", "train": a.train, "rung": a.rung, "ember": a.ember, "steps": STEPS, "max_rung_s": MAX_RUNG_S,
           "parity_n": PARITY_N, "score_n": SCORE_N, "cost_rung_s": COST_RUNG_S, "device_mib": P.DEVICE_MIB}

    def finish():
        rep["failed"] = verdict(rep)
        rep["ok"] = not rep["failed"]
        rep["over_cost_allowance"] = bool(rep.get("rung_seconds") and rep["rung_seconds"] > COST_RUNG_S)
        with open(os.path.join(a.out, "precheck.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "producer"}), flush=True)
        if rep["failed"]:                       # only the job's log comes home from a failed job
            for f in ("train.log", "rung.log", "parity.log"):
                print(f"--- {f}\n{tail(os.path.join(a.out, f))}", flush=True)
        return 0 if rep["ok"] else 1

    os.makedirs(a.out, exist_ok=True)
    try:
        run_check(a, rep)
    except (Exception, SystemExit) as e:
        rep["error"] = repr(e)
    signal.alarm(0)
    code = finish()
    if code:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(0, signal.SIGTERM)           # a timed-out rung's mtstudio, or parity's servers
    else:
        shutil.rmtree(os.path.join(a.out, "run"), ignore_errors=True)   # proved: nothing of it stays, and nothing of it goes home
        for f in ("model.gguf",):
            os.remove(os.path.join(a.out, "rung", f))
        for d in ("panel_fineweb", "panel_dialogue"):
            shutil.rmtree(os.path.join(a.out, "rung", d), ignore_errors=True)
    return code


def run_check(a, rep):
    """Fills rep; returns early when a later step could only repeat the failure."""
    by = {j["name"]: j for j in json.load(open(a.jobs))["jobs"]}
    train, rung = by[a.train], by[a.rung]
    run, rung_out, par, sc = (os.path.join(a.out, x) for x in ("run", "rung", "parity", "score"))
    for d in (run, rung_out, par, sc):
        shutil.rmtree(d, ignore_errors=True)   # a replacement pod starts clean
    spec = short_spec(train["spec"], run)
    path = os.path.join(a.out, "precheck_train.json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
    rep["want_state_k"] = train["spec"]["arch"]["state_tokens"]["k"]
    env = {**os.environ, **{k: str(v) for k, v in train.get("env", {}).items()}}
    live = os.path.join(run, "ckpt_gen", "live")
    t0 = time.time()
    log = open(os.path.join(a.out, "train.log"), "w")
    trainer = subprocess.Popen([P.BIN, "run", path], stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        with Peak() as pk:
            step = None
            while time.time() - t0 < TRAIN_TIMEOUT_S and trainer.poll() is None and not step:
                got = live_step(live)
                step = got if got and got >= STEPS else None
                time.sleep(0.5)
        rep["train_seconds"], rep["peak_train_mib"] = round(time.time() - t0, 1), pk.mib
        if not step or trainer.poll() is not None:
            rep["train_rc"] = f"no live trainer at a checkpoint (rc {trainer.poll()})"
            return
        os.kill(trainer.pid, signal.SIGSTOP)   # jobs_vm.py's exclusive rung: stopped, memory held
        time.sleep(1)
        if proc_state(trainer.pid) != "T":
            rep["train_rc"] = f"the trainer is not stopped (state {proc_state(trainer.pid)})"
            return
        rep["train_rc"] = 0
        step = live_step(live)
        snap = os.path.join(run, f"rung_{step}")
        os.makedirs(snap)
        for f in ("model.safetensors", "state.txt"):   # jobs_vm.py's snapshot: the weights and their step
            shutil.copyfile(os.path.join(os.path.realpath(live), f), os.path.join(snap, f))
        argv_r = rung_argv(rung["cmd"], rung_out, os.path.join(snap, "model.safetensors"), run, step)
        t1 = time.time()
        renv = {**os.environ, **{k: str(v) for k, v in rung.get("env", {}).items()}}
        with Peak() as pk, open(os.path.join(a.out, "rung.log"), "w") as rlog:
            try:
                rep["rung_rc"] = subprocess.run(argv_r, stdout=rlog, stderr=subprocess.STDOUT, env=renv, timeout=MAX_RUNG_S).returncode
            except subprocess.TimeoutExpired:
                rep["rung_rc"] = "timeout"
        rep["rung_seconds"], rep["peak_total_mib"], rep["phases_s"] = round(time.time() - t1, 1), pk.mib, phase_times(rung_out, t1)
        rep["trainer_held"] = proc_state(trainer.pid) == "T"
        os.kill(trainer.pid, signal.SIGCONT)   # a raced trainer goes on after a rung
        t1 = time.time()
        while time.time() - t1 < RESUME_TIMEOUT_S and trainer.poll() is None:
            got = live_step(live)
            if got and got >= step + STEPS:
                rep["resumed_step"], rep["resume_seconds"] = got, round(time.time() - t1, 1)
                break
            time.sleep(0.5)
    finally:
        if trainer.poll() is None:
            trainer.kill()                     # SIGKILL ends a stopped process too
        trainer.wait()
        log.close()
    done = os.path.join(rung_out, "rung.json")
    if os.path.exists(done):
        r = json.load(open(done))
        rep.update(acc_gold=r.get("acc_gold"), panels=r.get("panels"), gguf_sha256=r.get("gguf_sha256"), val_loss=r.get("val_loss"))
    gguf = os.path.join(snap, "model.gguf")
    rep["gguf_state_k"] = gguf_state_k(gguf) if os.path.exists(gguf) else None
    if rep["rung_rc"] != 0 or rep["gguf_state_k"] != rep["want_state_k"]:
        return
    shutil.copyfile(os.path.join(rung_out, "export_spec.json"), SPEC_VM)
    os.makedirs(par, exist_ok=True)
    ember_env = {**os.environ, "TINY_CHAT_EMBER_DIR": EMBER_DIR}
    build = os.path.join(par, "ember_build.json")
    for k in range(EMBER_TRIES):               # the build's clone has no retry of its own, and one 504 must not halt an arm again
        if subprocess.run([sys.executable, os.path.join(HERE, "ensure_ember_vm.py"), build, os.path.join(par, "ember_build.log"), a.ember], env=ember_env).returncode == 0:
            break
        time.sleep(60 * (k + 1))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "chat_parity_vm.py"), par, "precheck", str(PARITY_N), a.ember], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=ember_env, timeout=PARITY_TIMEOUT_S)
        open(os.path.join(a.out, "parity.log"), "w").write(p.stdout)
    except subprocess.TimeoutExpired:
        pass
    rep["parity_seconds"] = round(time.time() - t0, 1)
    s = os.path.join(par, "parity_summary.json")
    summary = json.load(open(s)) if os.path.exists(s) else None
    rep["parity_ok"] = parity_ok(summary, PARITY_N) and ON_GPU in tail(os.path.join(par, "ember_cuda", "debugging.log"), 10 ** 7)
    if summary:
        rep["parity"] = {leg: summary.get(leg) for leg in ("cuda", "cpu")}
        rep["ember_commit"] = summary.get("ember", {}).get("commit")
    t0 = time.time()
    try:
        panel = next(x.split("=", 1)[1] for x in rung["cmd"] if x.startswith("dialogue="))
        rep["score"] = score_parity(sc, gguf, os.path.join(rung_out, "export_spec.json"), panel, json.load(open(build))["bin"])
        rep["score_ok"] = score_ok(rep["score"], SCORE_N)
    except (OSError, ValueError, KeyError, RuntimeError, SystemExit) as e:
        rep["score"], rep["score_ok"] = {"error": repr(e)}, False
    rep["score_seconds"] = round(time.time() - t0, 1)


if __name__ == "__main__":
    sys.exit(main())
