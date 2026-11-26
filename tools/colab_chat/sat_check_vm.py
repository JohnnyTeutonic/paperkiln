#!/usr/bin/env python3
"""SAT's sat_check job on the VM (tools/colab_chat/make_sat.py; the registration's section 7). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/sat_check_vm.py --out DIR --windows 8 --spec-json '<SAT training spec>'

Writes the training spec with out_dir DIR/run (a fresh directory: the check runs at the initial weights), runs the
VM's trainer (JOBS_MTSTUDIO, as jobs_vm.py) with MICROTORCH_SAT_CHECK=<windows>: the scaffold step's forward on the
first <windows> training windows with labels A (the spec's), B (every role moved, every pointer moved to another
candidate) and none, and the plain step's forward, every logit compared bit for bit by its sha256. It also requires
that the check could have failed (targets of both kinds under A, roles and pointer golds moved under B, finite
auxiliary losses that differ) and that the LM's logits and loss are unchanged after the auxiliary terms. Writes
DIR/sat_check.json (each condition named) and exits 0 only when every condition holds; anything else (a difference,
a refusal, a missing result) exits non-zero, and the job's halt_on_failure stops the session before training.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")


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
    variants = sorted({e["variant"] for e in checks})
    under_a = [e for e in checks if e["variant"] == "A"]
    r = result[0] if len(result) == 1 else {}
    # The check must be one that could fail (Explore 10, GPT 4): role and pointer targets under A, at least one role and
    # one pointer gold moved under B, finite auxiliary losses that differ between A and B, and the LM's logits and loss
    # recomputed after the auxiliary terms bit for bit those before. The trainer requires the same ("pass"); this
    # wrapper checks each condition again from the events.
    conditions = {
        "trainer_exit_0": p.returncode == 0,
        "one_result": len(result) == 1,
        "logits_equal": r.get("equal") is True,
        "trainer_pass": r.get("pass") is True,
        "four_variants": variants == ["A", "B", "none", "plain"],
        "a_role_targets": sum(e.get("role_n", 0) for e in under_a) > 0 and r.get("a_role_targets", 0) > 0,
        "a_pointer_targets": sum(e.get("ptr_n", 0) for e in under_a) > 0 and r.get("a_pointer_targets", 0) > 0,
        "b_moved_roles": sum(r.get("b_roles_moved", {}).values()) > 0,
        "b_moved_pointers": sum(r.get("b_pointers_moved", {}).values()) > 0,
        "aux_finite": r.get("aux_finite") is True,
        "aux_differ": r.get("aux_differ") == {"role": True, "pointer": True},
        "after_aux_identical": r.get("after_aux_identical") is True and all(e.get("after_aux_identical") is True for e in checks),
    }
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
