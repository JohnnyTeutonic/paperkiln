#!/usr/bin/env python3
"""`mtstudio score --batch-tokens` against the unbatched scorer on a real post job's Carry-5 and Carry-12 forced
choice (LOCAL_LLM_WORK E5.2, 28 Oct 2026), on the GPU it runs on.

    python3 tools/score_batch_bench.py run --weights W.safetensors --spec-json S.json --vocab V.gguf --mtstudio BIN
        --out DIR [--c5-every 4] [--c12-every 6] [--c12-ctx 1024] [--batch-tokens 16384]
        [--variants unbatched,batched,batched_tiled]
    python3 tools/score_batch_bench.py table DIR/bench.json

`run` scores the model (a pick's weights and spec, as tools/carry12_l0.py takes them) with tools/carry5_eval.py at
the model's own context and tools/carry12_eval.py at --c12-ctx (Carry-12's cost runs serve the 60M picks at 1,024),
history from the item set only (--gen none: gold, twin, noctx, and Carry-12's near), every N-th item, once per
variant, each a whole eval run in its own folder through the post job's own path (carry5_eval.score_all: chunks of
256 requests, one `mtstudio score` process per chunk):
  unbatched      the registered scorer: one request per forward, the job's environment (jobs_vm.py's ENV);
  batched        --score-batch-tokens N, the same environment (in the batch's step window the GEMMs are cuBLAS's);
  batched_tiled  the same with MICROTORCH_CUBLAS removed (the window's GEMMs are the tiled kernel, whose rows sum
                 in the order they would alone).
Per variant: the wall time and requests per second; against unbatched, per request: identical bits, the largest
|d log-prob| and |d sum|, IR_SPEC 4.3's ROUNDING test on each request's log-probs (|d| <= 5e-5 max(1, |ref|) on
CUDA, 1e-5 on CPU), argmax positions equal, and per item and mode the forced choice (acc, acc_norm, the chosen
candidate) and every summary rate. Writes DIR/bench.json after each variant (rerunning resumes) and prints the table.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsave(p, d):
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(p + ".tmp", p)


def model_dir(a, out, T):
    """DIR/model_T<T>: the weights (symlink), the spec at data.T = T, its local spec and the GGUF the evals
    tokenise with (mtstudio export, as carry12_l0.py's cost runs do)."""
    import carry12_l0
    d = os.path.join(out, f"model_T{T}")
    os.makedirs(d, exist_ok=True)
    w = os.path.join(d, "model.safetensors")
    if not os.path.exists(w):
        os.symlink(os.path.abspath(a.weights), w)
    sp = jload(a.spec_json)
    sp["data"] = {**sp["data"], "T": T}
    jsave(os.path.join(d, "spec.json"), sp)
    spec_path, _ = carry12_l0.local_spec(d, a.vocab, d)
    if not os.path.exists(os.path.join(d, "model.gguf")):
        p = subprocess.run([a.mtstudio, "export", spec_path, "--from", w, "--formats", "gguf"], capture_output=True,
                           text=True)
        if p.returncode != 0:
            sys.exit(f"mtstudio export failed (rc {p.returncode}): {p.stderr[-800:]}")
    return spec_path


def scores_of(run_dir):
    res = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "score", "chunk*.json"))):
        res.update(jload(p)["results"])
    return res


def items_of(run_dir):
    out = {}
    for p in glob.glob(os.path.join(run_dir, "*", "items.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r.get("fc", {})
    return out


def compare(ref_dir, run_dir, tol):
    a, b = scores_of(ref_dir), scores_of(run_dir)
    c = {"requests": len(a), "keys_equal": set(a) == set(b), "over_ctx_equal": True, "identical": 0,
         "rounding_pass": 0, "max_abs_dlogprob": 0.0, "max_abs_dsum": 0.0, "argmax_equal": 0, "argmax_total": 0,
         "argmax_diffs": [], "rounding_fails": []}
    for k, r in a.items():
        s = b.get(k)
        if s is None:
            continue
        if r["over_ctx"] != s["over_ctx"]:
            c["over_ctx_equal"] = False
            continue
        if r["over_ctx"]:
            c["identical"] += 1
            c["rounding_pass"] += 1
            continue
        lr, ls = r["logprobs"], s["logprobs"]
        same = lr == ls and r["argmax"] == s["argmax"] and r["sum"] == s["sum"]
        c["identical"] += same
        d = max(abs(x - y) for x, y in zip(lr, ls))
        ok = len(lr) == len(ls) and d <= tol * max(1.0, max(abs(x) for x in lr))
        c["rounding_pass"] += ok
        if not ok and len(c["rounding_fails"]) < 20:
            c["rounding_fails"].append({"key": k, "max_abs_d": d})
        c["max_abs_dlogprob"] = max(c["max_abs_dlogprob"], d)
        c["max_abs_dsum"] = max(c["max_abs_dsum"], abs(r["sum"] - s["sum"]))
        for j, (x, y) in enumerate(zip(r["argmax"], s["argmax"])):
            c["argmax_total"] += 1
            c["argmax_equal"] += x == y
            if x != y and len(c["argmax_diffs"]) < 20:
                c["argmax_diffs"].append({"key": k, "pos": j, "ref": x, "got": y})
    ia, ib = items_of(ref_dir), items_of(run_dir)
    n = diff = 0
    diffs = []
    for iid, fa in ia.items():
        fb = ib.get(iid, {})
        for mode, ra in fa.items():
            n += 1
            rb = fb.get(mode, {})
            if any(ra.get(k) != rb.get(k) for k in ("acc", "acc_norm", "chose", "over_ctx")):
                diff += 1
                if len(diffs) < 20:
                    diffs.append({"id": iid, "mode": mode, "ref": {k: ra.get(k) for k in ("acc", "acc_norm", "chose")},
                                  "got": {k: rb.get(k) for k in ("acc", "acc_norm", "chose")}})
    c["items_modes"] = n
    c["choice_diffs"] = diff
    c["choice_diff_rows"] = diffs
    def fc(d):
        p = os.path.join(d, "summary.json")
        return jload(p).get("models", {}).get("m", {}).get("fc") if os.path.exists(p) else None
    sa, sb = fc(ref_dir), fc(run_dir)
    c["summary_fc_equal"] = sa is not None and sa == sb
    return c


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bench.json")
    res = jload(path) if os.path.exists(path) else {}
    trained_T = jload(a.spec_json)["data"]["T"]
    cuda = os.environ.get("MICROTORCH_DEVICE") == "cuda"
    tol = 5e-5 if cuda else 1e-5
    res.update({"gpu": gpu_name(), "device": "cuda" if cuda else "cpu", "rounding_tol": tol,
                "env": {k: v for k, v in os.environ.items() if k.startswith("MICROTORCH_")},
                "batch_tokens": a.batch_tokens, "weights": os.path.abspath(a.weights), "spec": a.spec_json})
    res.setdefault("evals", {})
    evals = [("carry5", "carry5_eval.py", trained_T, a.c5_every), ("carry12", "carry12_eval.py", a.c12_ctx,
                                                                      a.c12_every)]
    variants = a.variants.split(",")
    if variants[0] != "unbatched":
        sys.exit("--variants: unbatched first (the reference)")
    for name, tool, T, every in evals:
        if every <= 0:
            continue
        spec_path = model_dir(a, a.out, T)
        e = res["evals"].setdefault(name, {"T": T, "every": every, "variants": {}})
        for v in variants:
            vd = os.path.join(a.out, name, v)
            if v in e["variants"] and e["variants"][v].get("done"):
                continue
            env = dict(os.environ)
            argv = [sys.executable, os.path.join(HERE, tool), "run", "--model", f"m={spec_path}", "--out", vd,
                    "--scorer", "mtstudio", "--mtstudio", a.mtstudio, "--gen", "none", "--every", str(every)]
            if v != "unbatched":
                argv += ["--score-batch-tokens", str(a.batch_tokens)]
            if v == "batched_tiled":
                env.pop("MICROTORCH_CUBLAS", None)
            print(json.dumps({"start": f"{name}/{v}", "t": time.strftime("%H:%M:%S")}), flush=True)
            t0 = time.time()
            p = subprocess.run(argv, capture_output=True, text=True, env=env)
            wall = time.time() - t0
            with open(vd + ".log", "w", encoding="utf-8") as f:
                f.write(p.stdout[-20000:] + p.stderr[-20000:])
            rec = {"rc": p.returncode, "wall_s": round(wall, 1), "done": True}
            if p.returncode == 0:
                n = len(scores_of(vd))
                rec.update({"requests": n, "requests_per_s": round(n / wall, 3) if wall else None})
            else:
                rec["log_tail"] = (p.stdout + p.stderr)[-1500:]
            e["variants"][v] = rec
            jsave(path, res)
        base = e["variants"].get("unbatched", {})
        for v in variants[1:]:
            r = e["variants"].get(v, {})
            if base.get("rc") == 0 and r.get("rc") == 0:
                r["vs_unbatched"] = compare(os.path.join(a.out, name, "unbatched"), os.path.join(a.out, name, v), tol)
                r["speedup"] = round(base["wall_s"] / r["wall_s"], 3) if r["wall_s"] else None
        jsave(path, res)
    print(table(res))
    bad = [f"{n}/{v}" for n, e in res["evals"].items() for v, r in e["variants"].items() if r.get("rc") != 0]
    return 1 if bad else 0


def table(res):
    rows = ["| eval | variant | requests | wall s | req/s | speed-up | identical | ROUNDING pass | max abs dlogprob | "
            "argmax equal | forced-choice rows differing | summary equal |", "|" + "---|" * 12]
    for name, e in res.get("evals", {}).items():
        for v, r in e["variants"].items():
            c = r.get("vs_unbatched") or {}
            rows.append("| " + " | ".join(str(x) for x in [
                f"{name} (T {e['T']}, every {e['every']})", v, r.get("requests"), r.get("wall_s"), r.get("requests_per_s"),
                r.get("speedup", "-"), f"{c['identical']}/{c['requests']}" if c else "-",
                f"{c['rounding_pass']}/{c['requests']}" if c else "-",
                f"{c['max_abs_dlogprob']:.2e}" if c else "-",
                f"{c['argmax_equal']}/{c['argmax_total']}" if c else "-",
                f"{c['choice_diffs']}/{c['items_modes']}" if c else "-",
                c.get("summary_fc_equal", "-") if c else "-"]) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--weights", required=True)
    r.add_argument("--spec-json", required=True)
    r.add_argument("--vocab", required=True)
    r.add_argument("--mtstudio", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--c5-every", type=int, default=4)
    r.add_argument("--c12-every", type=int, default=6)
    r.add_argument("--c12-ctx", type=int, default=1024)
    r.add_argument("--batch-tokens", type=int, default=16384)
    r.add_argument("--variants", default="unbatched,batched,batched_tiled")
    t = sub.add_parser("table")
    t.add_argument("bench")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    print(table(jload(a.bench)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
