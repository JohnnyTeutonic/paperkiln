#!/usr/bin/env python3
"""SAT's sat_check job on the VM (tools/colab_chat/make_sat.py; the registration's section 7). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/sat_check_vm.py --out DIR --windows 8 --spec-json '<SAT training spec>'

Writes the training spec with out_dir DIR/run (a fresh directory: the check runs at the initial weights), runs the
VM's trainer (JOBS_MTSTUDIO, as jobs_vm.py) with MICROTORCH_SAT_CHECK=<windows>: the scaffold step's forward on the
first <windows> training windows with labels A (the spec's), B (every role moved, every pointer moved to another
candidate) and none, and the plain step's forward, every logit compared bit for bit by its sha256. It also requires
that the check could have failed, on each configured source apart (role and pointer targets under A and roles and
pointer golds moved under B, in natural rows and, with twins configured, in twin rows), finite auxiliary losses that
differ, the exact expected (step, micro) batches for every variant with equal hashes and LM losses at each, and the
LM's logits and loss unchanged after the auxiliary terms (evaluate() lists the conditions). Writes
DIR/sat_check.json (each condition named) and exits 0 only when every condition holds; anything else (a difference,
a refusal, a missing result) exits non-zero, and the job's halt_on_failure stops the session before training.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")


VARIANTS = ("A", "B", "none", "plain")
SOURCES = ("natural", "twin")


def expected_keys(windows, batch, accum, from_step=1):
    """The (step, micro) batches the trainer's check takes for `windows` windows: rows in training order from from_step,
    `batch` a micro-batch, `accum` micro-batches a step, until `windows` rows are taken."""
    keys, taken, step = [], 0, from_step
    while taken < windows:
        for k in range(accum):
            if taken >= windows:
                break
            keys.append((step, k))
            taken += batch
        step += 1
    return keys


def evaluate(spec, windows, checks, r, returncode, n_results):
    """Every condition the check must meet, each recomputed from the logged events where it can be (Explore 10, GPT 4,
    GPT delta review 2 point 11):
    - the trainer exited 0, reported once, with equal logits and its own pass;
    - every variant ran exactly the expected (step, micro) batches in order, and at each batch the four variants'
      logits hashes and LM losses are identical, and the LM was unchanged after the auxiliary terms;
    - each configured source (natural rows; twin rows when twins are configured) had role and pointer targets under A
      and moved roles and pointer golds under B; without twins no twin target appears;
    - the auxiliary losses under A and B are logged wherever there are targets, finite, sum to the trainer's reported
      sums, and differ between A and B."""
    st = spec["train"]["symbol_targets"]
    twins = bool(st.get("twins"))
    keys = expected_keys(windows, spec["train"].get("batch", 1), spec["train"].get("accum", 1))
    by = {v: [e for e in checks if e.get("variant") == v] for v in VARIANTS}
    order_ok = all([(e["step"], e["micro"]) for e in by[v]] == keys for v in VARIANTS)
    at = {v: {(e["step"], e["micro"]): e for e in by[v]} for v in VARIANTS}
    same_hash = order_ok and all(len({at[v][k]["logits_sha256"] for v in VARIANTS}) == 1 for k in keys)
    same_loss = order_ok and all(len({repr(at[v][k]["loss"]) for v in VARIANTS}) == 1 for k in keys)

    def total(v, field, src, kind):
        return sum(e.get(field, {}).get(src, {}).get(kind, 0) for e in by[v])

    need = SOURCES if twins else ("natural",)
    exercised = all(total("A", "targets_by_source", s, k) > 0 for s in need for k in ("role", "pointer"))
    perturbed = all(total("B", "moved_by_source", s, k) > 0 for s in need for k in ("role", "pointer"))
    no_stray = twins or all(total(v, f, "twin", k) == 0 for v, f in (("A", "targets_by_source"), ("B", "moved_by_source"))
                            for k in ("role", "pointer"))
    reported = r.get("a_targets_by_source", {})
    counts_agree = all(reported.get(s, {}).get(k) == total("A", "targets_by_source", s, k) for s in SOURCES
                       for k in ("role", "pointer"))

    def aux_logged(v):
        for e in by[v]:
            for n, l in (("role_n", "role_loss"), ("ptr_n", "ptr_loss")):
                if e.get(n, 0) > 0 and not (isinstance(e.get(l), (int, float)) and math.isfinite(e[l])):
                    return False
        return bool(by[v])

    sums = {v: {"role": sum(e.get("role_loss", 0.0) for e in by[v]), "pointer": sum(e.get("ptr_loss", 0.0) for e in by[v])}
            for v in ("A", "B")}
    rep = r.get("aux_loss_sums", {})
    sums_agree = all(math.isclose(sums[v][k], rep.get(v, {}).get(k, float("nan")), rel_tol=1e-12, abs_tol=0.0)
                     for v in ("A", "B") for k in ("role", "pointer"))
    return {
        "trainer_exit_0": returncode == 0,
        "one_result": n_results == 1,
        "logits_equal": r.get("equal") is True,
        "trainer_pass": r.get("pass") is True,
        "twins_as_configured": r.get("twins_configured") is twins,
        "batch_keys_as_expected": order_ok and r.get("from_step") == 1 and r.get("windows") == windows,
        "hashes_equal_at_each_batch": same_hash,
        "lm_losses_equal_at_each_batch": same_loss,
        "after_aux_identical": r.get("after_aux_identical") is True and bool(checks) and
                               all(e.get("after_aux_identical") is True for e in checks),
        "targets_exercised_per_source": exercised and no_stray,
        "targets_perturbed_per_source": perturbed,
        "target_counts_agree": counts_agree,
        "aux_losses_logged_finite": aux_logged("A") and aux_logged("B") and r.get("aux_finite") is True,
        "aux_sums_agree": sums_agree,
        "aux_differ": all(sums["A"][k] != sums["B"][k] for k in ("role", "pointer")) and
                      r.get("aux_differ") == {"role": True, "pointer": True},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, required=True)
    ap.add_argument("--spec-json", required=True)
    ap.add_argument("--bin", default=BIN)
    a = ap.parse_args(argv)
    spec = json.loads(a.spec_json)
    if "symbol_targets" not in spec.get("train", {}):
        raise SystemExit("sat_check: the spec has no train.symbol_targets")
    run_dir = os.path.join(a.out, "run")
    shutil.rmtree(run_dir, ignore_errors=True)
    spec["out_dir"] = run_dir
    os.makedirs(a.out, exist_ok=True)
    spec_path = os.path.join(a.out, "check_spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    env = {**os.environ, "MICROTORCH_SAT_CHECK": str(a.windows)}
    env.pop("MICROTORCH_CHECKPOINT_TXN", None)
    p = subprocess.run([a.bin, "run", spec_path], env=env, capture_output=True, text=True)
    events = []
    ev_path = os.path.join(run_dir, "events.jsonl")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
    checks = [e for e in events if e.get("event") == "sat_check"]
    result = [e for e in events if e.get("event") == "sat_check_result"]
    r = result[0] if len(result) == 1 else {}
    conditions = evaluate(spec, a.windows, checks, r, p.returncode, len(result))
    ok = all(conditions.values())
    rec = {"ok": ok, "conditions": conditions, "returncode": p.returncode, "result": result, "checks": checks,
           "symbol_targets": [e for e in events if e.get("event") == "symbol_targets"],
           "stderr_tail": p.stderr[-2000:]}
    with open(os.path.join(a.out, "sat_check.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    shutil.rmtree(run_dir, ignore_errors=True)
    print(json.dumps({"sat_check": "pass" if ok else "FAIL", "returncode": p.returncode}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
