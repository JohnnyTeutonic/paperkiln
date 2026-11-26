#!/usr/bin/env python3
"""SAT's CUDA smoke on the VM (tools/colab_chat/make_sat.py --smoke; the author's research tree,
AI_ML/chat_screens_claude/base2/token_ladder/symbols_as_targets/SMOKE.md). Claude Opus 5.5, 26 Nov 2026.

    sat_smoke_vm.py watch  --job-dir DIR --status FILE --out OUT [--interval S] [--limit-s S]
    sat_smoke_vm.py checks --main-dir DIR --spec-json SPEC --out OUT --watch GENS.jsonl
                           [--base-commit SHA] [--repo /content/pk] [--bin MTSTUDIO] [--base-bin MTSTUDIO]

watch: while the smoke's training job runs, records each generation that becomes live in DIR/ckpt_gen (step, the
model file's sha256, and the sha256 of every tensor of the auxiliary heads and their optimiser in optim.safetensors)
to OUT/gens.jsonl; ends when FILE reads FINISHED or FAILED, or after --limit-s.

checks: after the training job, on the same VM:
  1. the comparator trainer: a CUDA build of --base-commit (the ladder's launch commit 672501bb) from --repo's history,
     started in the background (or --base-bin, already built);
  2. a reference run: the smoke's spec, uninterrupted from step 0 in OUT/ref, with the windows logged; each live
     generation's digests recorded, and the generation at the cut step copied to OUT/gen_cut;
  3. the identity runs from that copy: A, the launch build under the SAT spec; B, the comparator build under the plain
     spec (symbol_targets removed: N1B's), both to the spec's last step;
  4. the comparisons, written to OUT/smoke_vm.json: the main run (with its forced session loss) against the reference,
     step by step and generation by generation (item c); A against B and against the reference after the cut, and A's
     windows against the trainer's draw rule (item e); the scaffold's auxiliary losses and the natural and twin rows of
     every step against the schedule (item b); the evals inside the scaffold (item d); scaffold against plain step
     times (item f, from MICROTORCH_STEP_TIMING's per-phase times).
Exits 0 when it could compute every comparison (the smoke's evidence is the record, read at home), 1 otherwise.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")
BASE_COMMIT = "672501bb1d09f707fdd20a5c7280120940734327"
HEAD_PREFIX = "sat_heads"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_digests(path, prefix=None):
    """{name: sha256 of the tensor's bytes} from a safetensors file, optionally only names starting with prefix."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in sorted(head.items()):
            if k == "__metadata__" or (prefix and not k.startswith(prefix)):
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = hashlib.sha256(f.read(b - a)).hexdigest()
    return out


def live_generation(job_dir):
    """(step, generation directory) of DIR/ckpt_gen/live, or None."""
    live = os.path.join(job_dir, "ckpt_gen", "live")
    if not os.path.islink(live):
        return None
    g = os.path.realpath(live)
    try:
        step = int(open(os.path.join(g, "state.txt")).readline().strip())
    except (OSError, ValueError):
        return None
    if not os.path.exists(os.path.join(g, "MANIFEST.json")):
        return None
    return step, g


def gen_record(step, g):
    rec = {"step": step, "model_sha256": sha(os.path.join(g, "model.safetensors")), "at": time.time()}
    op = os.path.join(g, "optim.safetensors")
    rec["heads"] = tensor_digests(op, HEAD_PREFIX) if os.path.exists(op) else {}
    return rec


def watch_dir(job_dir, sink, stop, interval, on_gen=None):
    seen = set()
    while True:
        lg = live_generation(job_dir)
        if lg and lg[0] not in seen:
            try:
                if on_gen:
                    on_gen(*lg)
                rec = gen_record(*lg)
            except OSError:
                rec = None                       # replaced while being read: the next poll sees the newer one
            if rec:
                seen.add(lg[0])
                sink(rec)
        if stop():
            return seen
        time.sleep(interval)


def cmd_watch(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "gens.jsonl")
    t0 = time.time()

    def sink(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def stop():
        try:
            st = open(a.status).read().strip()
        except OSError:
            st = ""
        return st in ("FINISHED", "FAILED") or time.time() - t0 > a.limit_s
    watch_dir(a.job_dir, sink, stop, a.interval)
    time.sleep(a.interval)
    lg = live_generation(a.job_dir)                  # the last one, in case it went live with the status
    if lg:
        recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
        if lg[0] not in {r["step"] for r in recs}:
            sink(gen_record(*lg))
    return 0


# ---------------------------------------------------------------- runs

def write_spec(spec, out_dir, path, sat=True):
    s = json.loads(json.dumps(spec))
    s["out_dir"] = out_dir
    if not sat:
        s["train"].pop("symbol_targets", None)
    with open(path, "w") as f:
        json.dump(s, f, indent=1)
    return path


def run_trainer(binary, spec_path, log_path, env):
    t0 = time.time()
    with open(log_path, "w") as log:
        r = subprocess.run([binary, "run", spec_path], stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"returncode": r.returncode, "seconds": time.time() - t0, "binary": binary, "binary_sha256": sha(binary)}


def copy_generation(src_gen, dst_dir, src_events=None):
    """A generation as the live one of a fresh run directory, as the courier restores one: dst_dir/ckpt_gen/<name>,
    live pointing at it as the trainer writes it (a relative name), and the run's events.jsonl cut after that step's
    checkpoint event. The trainer's resume reads its events: without them a resume is not the uninterrupted run (on the
    CPU builds of claude/sat-pt and of 672501bb alike, 26 Nov 2026)."""
    cg = os.path.join(dst_dir, "ckpt_gen")
    os.makedirs(cg, exist_ok=True)
    name = os.path.basename(src_gen)
    shutil.copytree(src_gen, os.path.join(cg, name))
    os.symlink(name, os.path.join(cg, "live"))
    if src_events:
        step = int(open(os.path.join(src_gen, "state.txt")).readline())
        out = []
        for ln in open(src_events):
            out.append(ln if ln.endswith("\n") else ln + "\n")
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("event") == "checkpoint" and e.get("step") == step:
                break
        else:
            raise RuntimeError(f"{src_events} has no checkpoint event for step {step}")
        with open(os.path.join(dst_dir, "events.jsonl"), "w") as f:
            f.writelines(out)


def build_base(repo, commit, dst, log_path):
    t0 = time.time()
    with open(log_path, "w") as log:
        def sh(*argv, cwd=None):
            subprocess.run(list(argv), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not os.path.isdir(dst):
            sh("git", "-C", repo, "worktree", "add", "--detach", dst, commit)
        b = os.path.join(dst, "build_cuda")
        os.makedirs(b, exist_ok=True)
        sh("cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DMICROTORCH_CUDA=ON", "-DCUDA_ENABLED=ON",
           "-DCMAKE_CUDA_ARCHITECTURES=89", cwd=b)
        sh("make", f"-j{os.cpu_count() or 4}", "mtstudio", cwd=b)
    exe = os.path.join(dst, "build_cuda", "mtstudio")
    head = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"exe": exe, "commit": head, "seconds": time.time() - t0, "exe_sha256": sha(exe),
            "mtstudio_cpp_sha256": sha(os.path.join(dst, "tools", "mtstudio.cpp"))}


# ---------------------------------------------------------------- reading the events

def events(path, kind):
    out = []
    if os.path.exists(path):
        for ln in open(path):
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("event") == kind:
                out.append(e)
    return out


def by_step(evs, micro=False):
    d = {}
    for e in evs:
        d[(e["step"], e.get("micro")) if micro else e["step"]] = e       # a restarted run's last line wins
    return d


STEP_KEYS = ("loss", "grad_norm", "lr")
SAT_KEYS = ("w", "role_loss", "pointer_loss", "heads_grad_norm", "role_targets", "pointer_targets", "natural_rows",
            "twin_rows", "natural_role_targets", "natural_pointer_targets", "twin_role_targets", "twin_pointer_targets",
            "pointer_targets_dropped")


def step_view(e):
    v = {k: e.get(k) for k in STEP_KEYS}
    st = e.get("symbol_targets") or {}
    v.update({f"sat.{k}": st.get(k) for k in SAT_KEYS if k in st})
    return v


def compare_steps(a, b, steps):
    """Equality of the step events of two runs over steps: the count equal, the first differing step and field, and the
    largest absolute difference of the loss."""
    da, db = by_step(a), by_step(b)
    missing = [s for s in steps if s not in da or s not in db]
    first, n_eq, maxd = None, 0, 0.0
    for s in steps:
        if s in missing:
            continue
        va, vb = step_view(da[s]), step_view(db[s])
        if va == vb:
            n_eq += 1
        elif first is None:
            first = {"step": s, "fields": sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k)),
                     "a": va, "b": vb}
        la, lb = va.get("loss"), vb.get("loss")
        if isinstance(la, (int, float)) and isinstance(lb, (int, float)):
            maxd = max(maxd, abs(la - lb))
    return {"steps": [steps[0], steps[-1]] if steps else [], "n": len(steps), "missing": missing[:20],
            "n_missing": len(missing), "equal": n_eq, "all_equal": n_eq == len(steps) and not missing,
            "first_difference": first, "max_abs_loss_difference": maxd}


def compare_evals(a, b, kind, keys):
    da, db = by_step(a), by_step(b)
    rows = []
    for s in sorted(set(da) | set(db)):
        x, y = da.get(s), db.get(s)
        rows.append({"step": s, "a": {k: x.get(k) for k in keys} if x else None, "b": {k: y.get(k) for k in keys} if y else None,
                     "equal": bool(x and y and all(x.get(k) == y.get(k) for k in keys))})
    return {"kind": kind, "rows": rows, "all_equal": bool(rows) and all(r["equal"] for r in rows)}


def compare_gens(a, b):
    da, db = {r["step"]: r for r in a}, {r["step"]: r for r in b}
    rows = []
    for s in sorted(set(da) & set(db)):
        rows.append({"step": s, "model_equal": da[s]["model_sha256"] == db[s]["model_sha256"],
                     "heads_tensors": len(da[s]["heads"]), "heads_equal": da[s]["heads"] == db[s]["heads"]})
    return {"common_steps": [r["step"] for r in rows], "only_a": sorted(set(da) - set(db)), "only_b": sorted(set(db) - set(da)),
            "rows": rows, "all_equal": bool(rows) and all(r["model_equal"] and r["heads_equal"] for r in rows)}


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def scaffold_report(steps_ev, spec):
    """Item b: every scaffold step's auxiliary losses finite, and its natural and twin rows those of the schedule:
    global row g = (step - 1) * rows + micro * batch + b, a twin row when g % every == 0 (before the cut)."""
    tr = spec["train"]
    st = tr["symbol_targets"]
    rows = tr["batch"] * tr["accum"]
    every = (st.get("twins") or {}).get("every", 0)
    cut = st["cut_step"]
    d = by_step(steps_ev)
    bad, n, aux_missing = [], 0, []
    for s in range(1, cut):
        e = d.get(s)
        if e is None:
            bad.append({"step": s, "why": "no step event"})
            continue
        x = e.get("symbol_targets") or {}
        n += 1
        want_twin = sum(1 for g in range((s - 1) * rows, s * rows) if every and g % every == 0)
        if x.get("twin_rows") != want_twin or x.get("natural_rows") != rows - want_twin:
            bad.append({"step": s, "why": "rows", "natural": x.get("natural_rows"), "twin": x.get("twin_rows"),
                        "want_twin": want_twin})
        for k in ("role_loss", "pointer_loss"):
            if k not in x:
                aux_missing.append((s, k))
            elif not finite(x[k]):
                bad.append({"step": s, "why": f"{k} not finite", "value": x[k]})
        if not finite(e.get("loss")):
            bad.append({"step": s, "why": "LM loss not finite"})
    after = [s for s in range(cut, tr["steps"] + 1) if s in d and d[s].get("symbol_targets")]
    return {"scaffold_steps_read": n, "problems": bad[:20], "n_problems": len(bad), "aux_missing": aux_missing[:20],
            "n_aux_missing": len(aux_missing), "steps_after_cut_with_symbol_targets": after[:20],
            "role_loss_first_last": [d[1]["symbol_targets"].get("role_loss") if 1 in d else None,
                                     d[cut - 1]["symbol_targets"].get("role_loss") if cut - 1 in d else None],
            "pointer_loss_first_last": [d[1]["symbol_targets"].get("pointer_loss") if 1 in d else None,
                                        d[cut - 1]["symbol_targets"].get("pointer_loss") if cut - 1 in d else None],
            "ok": not bad and not after}


def windows_report(win_ev, spec, steps):
    """The logged windows of the given steps against the trainer's draw rule: row g's draw is the g-th output of
    std::mt19937(123 + 1000003 * seed), its offset the draw modulo the main limit, or the twin limit for a twin row
    (g % every == 0 before the cut)."""
    sys.path.insert(0, HERE)
    import sat_windows as SW
    tr = spec["train"]
    st = tr.get("symbol_targets") or {}
    rows, batch = tr["batch"] * tr["accum"], tr["batch"]
    every = (st.get("twins") or {}).get("every", 0)
    cut = st.get("cut_step", 0)
    d = by_step(win_ev, micro=True)
    last = max(steps)
    draws = SW.mt19937_raw(123 + 1000003 * tr["seed"], last * rows).tolist()
    bad, n = [], 0
    for s in steps:
        for k in range(tr["accum"]):
            e = d.get((s, k))
            if e is None:
                bad.append({"step": s, "micro": k, "why": "no windows event"})
                continue
            for b in range(batch):
                g = (s - 1) * rows + k * batch + b
                twin = bool(every and s < cut and g % every == 0)
                lim = e["twin_lim"] if twin else e["lim"]
                ok = (e["raw"][b] == draws[g] and bool(e["twin"][b]) == twin and e["at"][b] == draws[g] % lim)
                n += 1
                if not ok:
                    bad.append({"step": s, "micro": k, "b": b, "raw": e["raw"][b], "want": draws[g],
                                "at": e["at"][b], "twin": e["twin"][b], "want_twin": twin})
    return {"rows_checked": n, "problems": bad[:20], "n_problems": len(bad), "ok": n > 0 and not bad}


def windows_equal(a_ev, b_ev, steps):
    da, db = by_step(a_ev, micro=True), by_step(b_ev, micro=True)
    keys = [k for k in sorted(set(da) & set(db)) if k[0] in steps]
    fields = ("raw", "at", "twin", "lim", "twin_lim", "src", "labels", "pointers")
    diff = [k for k in keys if any(da[k].get(f) != db[k].get(f) for f in fields)]
    return {"micro_batches": len(keys), "differ": [list(k) for k in diff[:20]], "n_differ": len(diff),
            "all_equal": bool(keys) and not diff}


def timing(steps_ev, cut, skip):
    """Wall time a step from MICROTORCH_STEP_TIMING's per-phase times (the phases summed), scaffold steps against plain
    ones, leaving out the steps in skip (evals, checkpoints, the first steps of a session)."""
    def total(e):
        t = e.get("t_ms")
        return sum(v for v in t.values() if isinstance(v, (int, float))) if isinstance(t, dict) else None
    sc = [total(e) for e in steps_ev if e["step"] < cut and e["step"] not in skip and total(e) is not None]
    pl = [total(e) for e in steps_ev if e["step"] >= cut and e["step"] not in skip and total(e) is not None]

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None
    out = {"scaffold_steps": len(sc), "plain_steps": len(pl), "scaffold_median_ms": med(sc), "plain_median_ms": med(pl),
           "scaffold_mean_ms": sum(sc) / len(sc) if sc else None, "plain_mean_ms": sum(pl) / len(pl) if pl else None}
    if out["scaffold_median_ms"] and out["plain_median_ms"]:
        out["ratio_median"] = out["scaffold_median_ms"] / out["plain_median_ms"]
    return out


def cmd_checks(a):
    spec = json.loads(a.spec_json)
    tr = spec["train"]
    cut, last = tr["symbol_targets"]["cut_step"], tr["steps"]
    os.makedirs(a.out, exist_ok=True)
    rec = {"producer": "sat_smoke_vm.py checks, Claude Opus 5.5, 26 Nov 2026", "spec_sha256":
           hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), "cut_step": cut, "steps": last,
           "launch_build": {"exe": a.bin, "exe_sha256": sha(a.bin)}}
    env = {**os.environ, "MICROTORCH_CHECKPOINT_TXN": "1", "MICROTORCH_STEP_TIMING": "1", "MICROTORCH_LOG_WINDOWS": "1"}
    # 1. the comparator build, in the background
    base = {}

    def do_build():
        try:
            base.update(build_base(a.repo, a.base_commit, a.base_dir, os.path.join(a.out, "base_build.log")))
        except Exception as e:  # noqa: BLE001 recorded, and the identity runs are then skipped
            base["error"] = f"{type(e).__name__}: {e}"
    if a.base_bin:
        base.update({"exe": a.base_bin, "exe_sha256": sha(a.base_bin), "prebuilt": True})
        bt = None
    else:
        bt = threading.Thread(target=do_build, daemon=True)
        bt.start()
    # 2. the reference run, its generations recorded and the cut generation copied
    ref = os.path.join(a.out, "ref")
    shutil.rmtree(ref, ignore_errors=True)
    shutil.rmtree(os.path.join(a.out, "gen_cut"), ignore_errors=True)
    ref_gens, done = [], threading.Event()

    def on_gen(step, g):
        if step == cut and not os.path.exists(os.path.join(a.out, "gen_cut")):
            shutil.copytree(g, os.path.join(a.out, "gen_cut", os.path.basename(g)))
    wt = threading.Thread(target=watch_dir, args=(ref, ref_gens.append, done.is_set, 0.2, on_gen), daemon=True)
    wt.start()
    rec["reference_run"] = run_trainer(a.bin, write_spec(spec, ref, os.path.join(a.out, "ref.spec.json")),
                                       os.path.join(a.out, "ref.log"), env)
    done.set()
    wt.join(timeout=60)
    rec["identity_from"] = "the cut generation, copied when it went live"
    if not os.path.exists(os.path.join(a.out, "gen_cut")):
        # Missed while live: the oldest whole generation the run kept at or after the cut (the trainer keeps the
        # live one and the one before it).
        kept = []
        cg = os.path.join(ref, "ckpt_gen")
        for n in (os.listdir(cg) if os.path.isdir(cg) else []):
            g = os.path.join(cg, n)
            if n != "live" and os.path.isdir(g) and os.path.exists(os.path.join(g, "MANIFEST.json")):
                try:
                    kept.append((int(open(os.path.join(g, "state.txt")).readline()), g))
                except (OSError, ValueError):
                    pass
        kept = sorted(k for k in kept if k[0] >= cut)
        if kept:
            shutil.copytree(kept[0][1], os.path.join(a.out, "gen_cut", os.path.basename(kept[0][1])))
            rec["identity_from"] = f"the reference run's kept generation at step {kept[0][0]} (the cut's was not caught live)"
    rec["reference_generations"] = ref_gens
    main_ev = os.path.join(a.main_dir, "events.jsonl")
    ref_ev = os.path.join(ref, "events.jsonl")
    main_steps, ref_steps = events(main_ev, "step"), events(ref_ev, "step")
    main_gens = [json.loads(x) for x in open(a.watch)] if a.watch and os.path.exists(a.watch) else []
    drill = sorted(e["step"] for e in events(main_ev, "symbol_targets_resume")) or None
    rec["c_resume"] = {
        "main_resumed_at": drill,
        "main_resume_events": [e for e in map(json.loads, open(main_ev)) if e.get("event") in
                               ("resume", "symbol_targets_resume", "symbol_targets_counts")] if os.path.exists(main_ev) else [],
        "steps_before_resume": compare_steps(main_steps, ref_steps, list(range(1, (drill or [1])[0] + 1))),
        "steps_after_resume": compare_steps(main_steps, ref_steps, list(range((drill or [0])[0] + 1, last + 1))),
        "evals": compare_evals(events(main_ev, "eval"), events(ref_ev, "eval"), "eval", ("val_loss",)),
        "scaffold_evals": compare_evals(events(main_ev, "symbol_targets_eval"), events(ref_ev, "symbol_targets_eval"),
                                        "symbol_targets_eval", ("role_acc", "pointer_acc", "recency_acc", "majority_acc")),
        "generations": compare_gens(main_gens, ref_gens),
        "end": {"symbol_targets_end_main": by_step(events(main_ev, "symbol_targets_end")).get(cut),
                "symbol_targets_end_ref": by_step(events(ref_ev, "symbol_targets_end")).get(cut)}}
    # items b, d, f on the main run (the run under the runner, with its session loss)
    rec["b_scaffold"] = scaffold_report(main_steps, spec)
    rec["b_scaffold"]["windows_reference"] = windows_report(events(ref_ev, "windows"), spec, list(range(1, last + 1)))
    rec["d_scaffold_evals"] = [e for e in events(main_ev, "symbol_targets_eval")]
    # Left out of the step times: a session's first steps, eval and checkpoint steps and the step after each.
    marks = {s for s in range(1, last + 1) if s % tr["eval_every"] == 0 or s % tr["checkpoint_every"] == 0}
    starts = {1} | {s + 1 for s in (drill or [])}
    skip = {s + k for s in starts for k in range(0, 3)} | marks | {s + 1 for s in marks}
    rec["f_timing_main"] = timing(main_steps, cut, skip)
    rec["f_timing_reference"] = timing(ref_steps, cut, skip)
    # 3. the identity runs from the cut generation
    if bt is not None:
        bt.join()
    rec["comparator_build"] = base
    gen_cut = os.path.join(a.out, "gen_cut")
    if "exe" not in base or not os.path.isdir(gen_cut):
        rec["e_identity"] = {"skipped": base.get("error") or "no generation at the cut step was copied"}
    else:
        g = os.path.join(gen_cut, os.listdir(gen_cut)[0])
        g_step = int(open(os.path.join(g, "state.txt")).readline())
        runs = {}
        for name, binary, sat in (("A", a.bin, True), ("B", base["exe"], False)):
            d = os.path.join(a.out, name)
            shutil.rmtree(d, ignore_errors=True)
            copy_generation(g, d, ref_ev)
            runs[name] = run_trainer(binary, write_spec(spec, d, os.path.join(a.out, f"{name}.spec.json"), sat=sat),
                                     os.path.join(a.out, f"{name}.log"), env)
        ev = {n: os.path.join(a.out, n, "events.jsonl") for n in ("A", "B")}
        after = list(range(g_step + 1, last + 1))

        def export(n):
            p = os.path.join(a.out, n, spec["name"] + ".safetensors")
            return sha(p) if os.path.exists(p) else None
        gens_end = {n: live_generation(os.path.join(a.out, n)) for n in ("A", "B")}
        rec["e_identity"] = {
            "from_generation_step": g_step, "runs": runs,
            "A_vs_B_steps": compare_steps(events(ev["A"], "step"), events(ev["B"], "step"), after),
            "A_vs_B_evals": compare_evals(events(ev["A"], "eval"), events(ev["B"], "eval"), "eval", ("val_loss",)),
            "A_vs_reference_steps": compare_steps(events(ev["A"], "step"), ref_steps, after),
            "A_vs_main_steps": compare_steps(events(ev["A"], "step"), main_steps, after),
            "export_sha256": {"A": export("A"), "B": export("B"), "reference": export("ref"),
                              "main": (lambda p: sha(p) if os.path.exists(p) else None)(
                                  os.path.join(a.main_dir, spec["name"] + ".safetensors"))},
            "end_generation_model_sha256": {n: (sha(os.path.join(v[1], "model.safetensors")) if v else None)
                                            for n, v in gens_end.items()},
            "A_windows_rule": windows_report(events(ev["A"], "windows"), spec, after),
            "A_vs_reference_windows": windows_equal(events(ev["A"], "windows"), events(ref_ev, "windows"), set(after)),
            "B_windows_logged_after_start": len([w for w in events(ev["B"], "windows") if w["step"] > g_step]),
            "B_note": "the comparator trainer (672501bb) has no MICROTORCH_LOG_WINDOWS: its windows show only through its "
                      "losses and weights"}
        ex = rec["e_identity"]["export_sha256"]
        rec["e_identity"]["exports_A_B_equal"] = ex["A"] is not None and ex["A"] == ex["B"]
    with open(os.path.join(a.out, "smoke_vm.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    # compact copies of the three runs' events beside it, without the windows' label lists
    for n in ("ref", "A", "B"):
        p = os.path.join(a.out, n, "events.jsonl")
        if os.path.exists(p):
            with open(p) as fi, open(os.path.join(a.out, f"events_{n}.jsonl"), "w") as fo:
                for ln in fi:
                    try:
                        e = json.loads(ln)
                    except ValueError:
                        continue
                    if e.get("event") == "windows":
                        e.pop("labels", None)
                        e.pop("pointers", None)
                    fo.write(json.dumps(e) + "\n")
    ok = rec["reference_run"]["returncode"] == 0 and "skipped" not in rec.get("e_identity", {})
    print(json.dumps({"ok": ok, "c_after": rec["c_resume"]["steps_after_resume"]["all_equal"],
                      "e_A_vs_B": rec.get("e_identity", {}).get("A_vs_B_steps", {}).get("all_equal")}))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--job-dir", required=True)
    w.add_argument("--status", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--interval", type=float, default=5.0)
    w.add_argument("--limit-s", type=float, default=7200)
    c = sub.add_parser("checks")
    c.add_argument("--main-dir", required=True)
    c.add_argument("--spec-json", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--watch", default=None)
    c.add_argument("--base-commit", default=BASE_COMMIT)
    c.add_argument("--repo", default="/content/pk")
    c.add_argument("--base-dir", default="/content/pk_base")
    c.add_argument("--bin", default=BIN)
    c.add_argument("--base-bin", default=None)
    a = ap.parse_args(argv)
    return cmd_watch(a) if a.cmd == "watch" else cmd_checks(a)


if __name__ == "__main__":
    sys.exit(main())
