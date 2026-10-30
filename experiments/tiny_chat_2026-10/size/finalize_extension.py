"""Durable, local-only completion of the registered matched-seed extension.

The existing Windows watchdog may own this light polling driver. It never rents
compute, judges text, edits Markdown, or commits. Reports publish atomically only
after provenance and exact-five-seed gates pass. HALTED waits for an operator.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parents[1]
CPU_BUDGET = REPO.parent / "scripts/cpu_budget.py"
SOURCE_FILES = (
    Path(__file__), HERE / "check_extension_results.py", HERE / "analyze_size.py",
    EXPERIMENT / "analyze.py", EXPERIMENT / "analyze_x5.py",
    EXPERIMENT / "PREREGISTRATION.md", EXPERIMENT / "design.json",
    HERE / "depth_choice.json", EXPERIMENT / "stage0/lr_choice_mor.json",
    EXPERIMENT / "asha_plan.json", CPU_BUDGET,
)
SEEDS = [1, 2, 3, 4, 5]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DriverLock(AbstractContextManager):
    """OS lock: a crash releases it; another watchdog process cannot finalize twice."""
    def __init__(self, home):
        self.path = Path(home) / "driver.lock"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            if self.handle.seek(0, os.SEEK_END) == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *_):
        if self.handle:
            self.handle.close()


class Finalizer:
    def __init__(self, art, state_dir, output=None, max_hours=24):
        self.art = Path(art).resolve()
        self.home = Path(state_dir).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.output = Path(output).resolve() if output else self.art / "size_extension_final"
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.home / "state.json"
        self.max_hours = max_hours

    def log(self, text):
        with (self.home / "runner.log").open("a", encoding="utf-8") as f:
            f.write(now() + " " + text + "\n")

    def state(self):
        if self.state_path.exists():
            value = load_json(self.state_path)
            if value["art"] != str(self.art) or value["output"] != str(self.output):
                raise ValueError("state directory belongs to another artifact/output directory")
            return value
        value = {"schema": 1, "created": time.time(), "art": str(self.art),
                 "output": str(self.output), "status": "WAITING", "python": sys.version,
                 "source_hashes": {str(p.resolve()): sha(p) for p in SOURCE_FILES}}
        atomic_json(self.state_path, value)
        return value

    def verify_sources(self, state):
        changed = [p for p, digest in state["source_hashes"].items()
                   if not Path(p).is_file() or sha(p) != digest]
        if changed:
            raise ValueError("frozen source/configuration changed: " + ", ".join(changed))

    def inputs(self, checker, size, receipt):
        """Read all fifteen selected reports, refusing extra seeds or duplicate conflicts."""
        files, unique, seen = {}, set(), {}
        depth = load_json(HERE / "depth_choice.json")["depth"]
        for family in size.SIZES:
            data = size.load_size(str(self.art), family, depth)
            for block, cell in size.sources(str(self.art), family, depth):
                for path in (self.art / block).glob("post_*/summary.json"):
                    report = load_json(path)
                    if report.get("cell") == cell:
                        key = (family, report.get("seed"))
                        if key in seen and seen[key] != report:
                            raise ValueError("conflicting duplicate summary: " + str(path))
                        seen[key] = report
                        files[str(path)] = sha(path)
            if family == "20m":
                for row in receipt["summaries"]:
                    if row["size"] != "20m":
                        continue
                    report = load_json(row["path"])
                    seed = report["seed"]
                    if seed in data and data[seed] != report:
                        raise ValueError("conflicting original/extension 20M report")
                    data[seed] = report
                    files[row["path"]] = row["sha256"]
            if set(data) != set(SEEDS) or any(
                    type(data[s].get("seed")) is not int or data[s]["seed"] != s
                    or type(data[s].get("primary")) not in (int, float)
                    or not math.isfinite(data[s]["primary"]) for s in SEEDS):
                raise ValueError(f"{family} must have exactly five complete registered seeds")
            unique.update((family, s) for s in data)
        if len(unique) != 15:
            raise ValueError("expected fifteen matched summaries")
        for row in receipt["ledgers"]:
            files[row["path"]] = row["sha256"]
            files[row["reference"]] = row["reference_sha256"]
        return {"selected_size_seed_pairs": sorted([list(x) for x in unique]), "sha256": files}

    @staticmethod
    def verify_outputs(folder):
        size = load_json(Path(folder) / "analysis_size.json")
        main = load_json(Path(folder) / "analysis_tiny_chat.json")
        if (size.get("seeds") != SEEDS or not all(type(s) is int for s in size["seeds"])
                or size.get("missing") != []):
            raise ValueError("size analysis is not a complete five-seed result")
        if main.get("partial") is not False or main.get("E1_150m") != size:
            raise ValueError("main analysis does not reproduce the same complete size block")
        if "PARTIAL" in str(size.get("label", "")).upper():
            raise ValueError("partial size analysis cannot be published")

    def run_analyses(self, staging):
        calls = [
            [sys.executable, str(HERE / "analyze_size.py"), "--art", str(self.art),
             "--depth-choice", str(HERE / "depth_choice.json"), "--out", str(staging / "analysis_size.json")],
            [sys.executable, str(EXPERIMENT / "analyze.py"), "--art", str(self.art),
             "--out", str(staging / "analysis_tiny_chat.json")],
        ]
        atomic_json(staging / "commands.json", calls)
        with (staging / "analysis.log").open("w", encoding="utf-8") as log:
            for call in calls:
                wrapped = [sys.executable, str(CPU_BUDGET), "run", "--session", "paperkiln",
                           "--job", "registered size-extension final analysis", "--cores", "1",
                           "--resume", "restart existing size finalizer watchdog entry", "--", *call]
                subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)

    def tick(self):
        state = self.state()
        if state["status"] == "HALTED":
            return "HALTED"
        try:
            self.verify_sources(state)
            published = self.output / "reports"
            if published.exists():
                completion = load_json(published / "completion.json")
                self.verify_outputs(published)
                if completion["source_hashes"] != state["source_hashes"]:
                    raise ValueError("existing publication has different frozen sources")
                for path, digest in completion["input_manifest"]["sha256"].items():
                    if not Path(path).is_file() or sha(path) != digest:
                        raise ValueError("published input changed: " + path)
                for name, digest in completion["report_hashes"].items():
                    if sha(published / name) != digest:
                        raise ValueError("existing published report changed: " + name)
                if state["status"] != "COMPLETE":
                    state.update(status="COMPLETE", completed=completion["completed"])
                    atomic_json(self.state_path, state)
                # Also repair a crash between saving COMPLETE and logging the
                # watchdog's terminal marker; otherwise it would restart forever.
                self.log("all jobs finished and home: registered five-seed analysis published")
                return "COMPLETE"
            if time.time() - state["created"] > self.max_hours * 3600:
                raise ValueError("extension completion deadline exceeded; operator review required")
            checker = load_module("extension_inputs", HERE / "check_extension_results.py")
            receipt = checker.audit(self.art)
            if receipt["errors"]:
                raise ValueError("extension input mismatch: " + "; ".join(receipt["errors"]))
            if not receipt["ready"]:
                count = len(receipt["pending"])
                if state.get("pending_count") != count:
                    state.update(status="WAITING", pending_count=count, checked=now())
                    atomic_json(self.state_path, state)
                    self.log(f"waiting for {count} extension inputs; no analysis performed")
                return "WAITING"
            size = load_module("extension_size", HERE / "analyze_size.py")
            before = self.inputs(checker, size, receipt)
            self.verify_sources(state)
            checker.merge_20m(self.art, receipt)
            # Merged report paths also become part of the stable input manifest.
            before = self.inputs(checker, size, receipt)
            state.update(status="ANALYSING", input_manifest=before, checked=now())
            atomic_json(self.state_path, state)
            staging = self.output / ".incoming"
            staging.mkdir(exist_ok=True)
            atomic_json(staging / "input_receipt.json", receipt)
            self.run_analyses(staging)
            self.verify_outputs(staging)
            self.verify_sources(state)
            fresh = checker.audit(self.art)
            if fresh["errors"] or not fresh["ready"] or self.inputs(checker, size, fresh) != before:
                raise ValueError("extension inputs changed during analysis")
            report_names = ("analysis_size.json", "analysis_tiny_chat.json", "input_receipt.json", "commands.json")
            completion = {"completed": now(), "source_hashes": state["source_hashes"],
                          "input_manifest": before,
                          "report_hashes": {name: sha(staging / name) for name in report_names}}
            atomic_json(staging / "completion.json", completion)
            staging.rename(published)
            state.update(status="COMPLETE", completed=completion["completed"])
            atomic_json(self.state_path, state)
            self.log("all jobs finished and home: registered five-seed analysis published")
            return "COMPLETE"
        except (Exception, SystemExit) as exc:
            state.update(status="HALTED", error=str(exc), checked=now())
            atomic_json(self.state_path, state)
            (self.home / "HALTED").write_text(str(exc) + "\n", encoding="utf-8")
            self.log("HALTED: " + str(exc))
            return "HALTED"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--poll-seconds", type=float, default=240)
    ap.add_argument("--max-hours", type=float, default=24)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    driver = Finalizer(args.art, args.state_dir, args.out, args.max_hours)
    try:
        with DriverLock(driver.home):
            while True:
                result = driver.tick()
                if result == "COMPLETE":
                    return 0
                if args.once:
                    return 1 if result == "HALTED" else 2
                time.sleep(max(10, args.poll_seconds))
    except (BlockingIOError, PermissionError):
        print("another extension finalizer already owns the state directory", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
