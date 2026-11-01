"""Explicit, local-only continuation mapping into the unchanged S-full finalizer.

A prospectively hash-pinned contract supplies exactly three selected campaigns.
No runner, provider or frozen scientific code is modified. View creation is
exclusive and refuses existing destinations. Views are references, not copies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from runpod_api import ledger_usd
import sfull_checkpoint_v2 as checkpoint

NUMERICAL = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
CELLS = ("B", "Sgist", "Sfull")
SEEDS = (1, 2, 3)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pins_check(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("empty source inventory")
    for name, digest in pins.items():
        p = Path(name)
        if not p.is_absolute() or not p.is_file() or p.is_symlink() or sha(p) != digest:
            raise ValueError("pinned input/source differs: " + name)


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in ("..", ".") for x in p.parts) or "\\" in name:
        raise ValueError("unsafe baseline path")
    return p


def validate_ledger(state, carried, protected, now):
    if state.get("campaign_home") is not True or state.get("endpoint"):
        raise ValueError("campaign incomplete or endpoint still owned")
    jobs = state.get("jobs")
    if not isinstance(jobs, dict) or not jobs or any(
        not isinstance(j, dict) or j.get("home") is not True or j.get("status") != "FINISHED"
        for j in jobs.values()
    ):
        raise ValueError("all jobs must be finished and home")
    r = state.get("runpod", {})
    if r.get("pod") or r.get("volume") or r.get("retired_volumes") or r.get("creating"):
        raise ValueError("campaign still owns provider resources")
    ledger = r["ledger"]
    if set(ledger) != {"pods", "volumes"} or set(carried) != {"pods", "volumes"}:
        raise ValueError("ledger inventory differs")
    if not set(protected).issubset({v["id"] for v in carried["volumes"]}):
        raise ValueError("protected volume must belong to preserved original ledger")
    for kind in ("pods", "volumes"):
        rows = ledger[kind]
        if len({v["id"] for v in rows}) != len(rows):
            raise ValueError("duplicate ledger resource")
        if any(v not in rows for v in carried[kind]):
            raise ValueError("prior spend omitted or changed")
    if any(not p.get("stop") for p in ledger["pods"]):
        raise ValueError("unstopped pod ledger")
    for kind, start, end, rate in (("pods", "start", "stop", "usd_per_hr"), ("volumes", "created", "deleted", "gb")):
        for item in ledger[kind]:
            values = [item[start], item.get(end) or now, item[rate]]
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in values) or values[1] < values[0] or values[2] < 0:
                raise ValueError("invalid ledger time/rate")
            if kind == "volumes" and not item.get(end) and item["id"] not in protected:
                raise ValueError("unaccounted undeleted volume")
    spent = ledger_usd(ledger, now)
    if not math.isfinite(spent) or spent > 7:
        raise ValueError("seed cumulative spend exceeds seven dollars")
    return spent


def validate(contract, now):
    if contract["numerical_commit"] != NUMERICAL or contract["seed_cap_usd"] != 7 or contract["aggregate_cap_usd"] != 21:
        raise ValueError("registered source/budget differs")
    pins_check(contract["pins"])
    freeze = read(contract["original_source_freeze"])
    if not freeze.get("source_hashes") or any(contract["pins"].get(k) != v for k, v in freeze["source_hashes"].items()):
        raise ValueError("original complete source freeze not bound")
    pins_check(freeze["source_hashes"])
    if contract["frozen_finalizer"] not in freeze["source_hashes"] or contract["original_manifest"] not in freeze["source_hashes"]:
        raise ValueError("finalizer/manifest must belong to original source freeze")
    manifest = read(contract["original_manifest"])
    baselines = manifest["baseline_artifact_sha256"]
    wanted = {f"c12_{c}_r{s}" for c in CELLS for s in SEEDS}
    roots = contract["artifact_roots"]
    if set(roots) != wanted or len({str(Path(v).resolve()) for v in roots.values()}) != 9:
        raise ValueError("exact unique nine-cell mapping required")
    for name, root in roots.items():
        p = Path(root)
        if not p.is_absolute() or not p.is_dir() or p.is_symlink():
            raise ValueError("artifact root must be an absolute real directory")
        cell, seed = name[len("c12_"):].rsplit("_r", 1)
        summaries = list(p.glob("post_c12_*/summary.json"))
        expected = p / f"post_c12_{cell}_r{seed}" / "summary.json"
        if summaries != [expected]:
            raise ValueError("missing or duplicate selected summary")
        row = read(expected)
        if row.get("cell") != cell or type(row.get("seed")) is not int or row["seed"] != int(seed):
            raise ValueError("selected summary identity differs")
        if cell == "Sfull":
            required = [f"post_c12_Sfull_r{seed}/carry12/m/items.jsonl", f"post_c12_Sfull_r{seed}/carry12/m/gen.jsonl", f"pt_c12_Sfull_r{seed}/pt_c12_Sfull_r{seed}_spec.json", f"post_c12_Sfull_r{seed}/sft_spec.json", "data/data.ledger.json", "gate/gate.json", "cuda_gate/cuda_gate.json", "sfull_inputs/verified_inputs.json"]
            for rel in required:
                file = p / rel
                if not file.is_file() or file.is_symlink():
                    raise ValueError("incomplete selected Sfull output: " + rel)
            for relative, step in ((f"pt_c12_Sfull_r{seed}", 10000), (f"post_c12_Sfull_r{seed}/sft", 1500)):
                root = p / relative / ".checkpoint_v2"
                with checkpoint.locked(root, False):
                    _, generation = checkpoint.current_unlocked(root)
                if generation["step"] != step or generation["source_commit"] != NUMERICAL or generation["rng"]["recipe_seed"] != int(seed) or generation["rng"]["batch"] != 16 or generation["rng"]["accum"] != 1:
                    raise ValueError("final complete checkpoint source/recipe differs")
    for name, digest in baselines.items():
        rel = safe_relative(name)
        if rel.parts[0] not in roots or rel.parts[0].startswith("c12_Sfull_"):
            raise ValueError("baseline inventory contains unexpected cell")
        path = Path(roots[rel.parts[0]]).joinpath(*rel.parts[1:])
        if path.is_symlink() or path.resolve().is_relative_to(Path(roots[rel.parts[0]]).resolve()) is False or sha(path) != digest:
            raise ValueError("audited baseline changed")
    campaigns = contract["campaigns"]
    if len(campaigns) != 3 or {c["seed"] for c in campaigns} != set(SEEDS) or any(type(c["seed"]) is not int for c in campaigns):
        raise ValueError("exact three seed campaigns required")
    costs = []
    for c in campaigns:
        if not c["operational_source_pins"] or not c["original_admission_pins"]:
            raise ValueError("old and continuation source/admission pins required")
        pins_check(c["operational_source_pins"])
        pins_check(c["original_admission_pins"])
        checkout = Path(c["operational_checkout"])
        commit = c["operational_commit"]
        if not re.fullmatch("[0-9a-f]{40}", commit):
            raise ValueError("invalid operational source commit")
        if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != commit:
            raise ValueError("operational source commit differs")
        subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--"], check=True, capture_output=True)
        required_control = ["runner_multi.py", "sfull_operational_v2.py", "sfull_checkpoint_v2.py", "sfull_checkpoint_transport_v2.py", "boot_sfull_v2_20261101.sh", "sfull_verify_inputs_v2.py", "sfull_sft_template_v2.json", f"carry12_sfull_v2_r{c['seed']}.json"]
        if any(str(checkout / "tools/colab_chat" / n) not in c["operational_source_pins"] for n in required_control):
            raise ValueError("incomplete operational source inventory")
        if any(contract["pins"].get(k) != v for k, v in {**c["operational_source_pins"], **c["original_admission_pins"]}.items()):
            raise ValueError("campaign source inventory not prospectively bound")
        for key in ("state", "carried_ledger"):
            if str(Path(c[key])) not in contract["pins"]:
                raise ValueError("final campaign state/ledger not bound")
        state = read(c["state"])
        required = {"cuda_gate", "data", "gate", "sfull_inputs", f"pt_c12_Sfull_r{c['seed']}", f"post_c12_Sfull_r{c['seed']}"}
        if set(state.get("jobs", {})) != required:
            raise ValueError("campaign job inventory differs")
        costs.append(validate_ledger(state, read(c["carried_ledger"]), c["protected_original_volumes"], now))
    if sum(costs) > 21:
        raise ValueError("aggregate cumulative spend exceeds twenty-one dollars")
    for name in ("original_source_freeze", "original_manifest", "frozen_finalizer", "cpu_budget"):
        if contract[name] not in contract["pins"]:
            raise ValueError("required anchor not pinned: " + name)
    return {"seed_costs_usd_at_validation": costs, "original_scientific_predicates_unchanged": True}


def create_views(contract, destination):
    destination = Path(destination)
    if not destination.is_absolute() or destination.is_symlink() or destination.exists():
        raise ValueError("view destination must be new absolute directory")
    destination.mkdir(parents=False)
    art, states = destination / "art", destination / "states"
    art.mkdir(); states.mkdir()
    for name, root in contract["artifact_roots"].items():
        (art / name).symlink_to(Path(root).resolve(), target_is_directory=True)
    for c in contract["campaigns"]:
        (states / f"pk_c12_sfull_r{c['seed']}_runner").symlink_to(Path(c["state"]).resolve().parent, target_is_directory=True)
    return art, states


def main():
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sha(args.contract) != args.contract_sha256:
        raise ValueError("prospective mapping contract changed")
    contract = read(args.contract)
    before = validate(contract, time.time())
    if not args.execute:
        print(json.dumps({"status": "READY_FOR_MAPPED_FINALIZER", **before}))
        return 0
    for p in (args.out, args.state_dir):
        if not p.is_absolute() or p.exists():
            raise ValueError("new absolute output and finalizer state required")
    art, states = create_views(contract, args.view)
    command = [sys.executable, contract["frozen_finalizer"], "--once", "--art", str(art), "--campaign-state-root", str(states), "--source-freeze", contract["original_source_freeze"], "--cpu-budget", contract["cpu_budget"], "--out", str(args.out), "--state-dir", str(args.state_dir)]
    result = subprocess.run(command, check=False)
    validate(contract, time.time())
    if result.returncode != 0:
        raise RuntimeError("unchanged finalizer refused or remained incomplete: " + str(result.returncode))
    print(json.dumps({"status": "UNCHANGED_FINALIZER_COMPLETED", "contract_sha256": args.contract_sha256, "command": command, **before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
