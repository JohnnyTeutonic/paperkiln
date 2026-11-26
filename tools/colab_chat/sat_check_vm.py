#!/usr/bin/env python3
"""SAT's sat_check job on the VM (tools/colab_chat/make_sat.py; the registration's section 7). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/sat_check_vm.py --out DIR --windows 8 --spec-json '<SAT training spec>'

Writes the training spec with out_dir DIR/run (a fresh directory: the check runs at the initial weights), runs the
VM's trainer (JOBS_MTSTUDIO, as jobs_vm.py) with MICROTORCH_SAT_CHECK=<windows>: the scaffold step's forward on the
first <windows> training windows with labels A (the spec's), B (every role moved, every pointer moved to another
candidate) and none, and the plain step's forward, every logit compared bit for bit by its sha256. Writes
DIR/sat_check.json and exits 0 only when the trainer reported equal logits; anything else (a difference, a refusal, a
missing result) exits non-zero, and the job's halt_on_failure stops the session before training.
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
    ok = (p.returncode == 0 and len(result) == 1 and result[0].get("equal") is True
          and variants == ["A", "B", "none", "plain"] and sum(e.get("role_n", 0) for e in checks if e["variant"] == "A") > 0)
    rec = {"ok": ok, "returncode": p.returncode, "result": result, "checks": checks,
           "symbol_targets": [e for e in events if e.get("event") == "symbol_targets"],
           "stderr_tail": p.stderr[-2000:]}
    with open(os.path.join(a.out, "sat_check.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    shutil.rmtree(run_dir, ignore_errors=True)
    print(json.dumps({"sat_check": "pass" if ok else "FAIL", "returncode": p.returncode}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
