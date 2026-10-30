"""Durable, local-only completion of the registered Carry-12 core.

The existing Windows watchdog may own this light polling driver. It never rents
compute, judges text, edits Markdown, or commits. Reports publish atomically only
after provenance and exact-three-seed Carry-12 gates pass. HALTED waits for an operator.
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
REPO = HERE.parents[1]
E1 = HERE.parent / "tiny_chat_2026-10"
CPU_BUDGET = REPO.parent / "scripts/cpu_budget.py"
SOURCE_FILES = (Path(__file__), HERE / "analyze.py", HERE / "PREREGISTRATION.md",
    E1 / "analyze.py", E1 / "analyze_x5.py", E1 / "size/depth_choice.json", CPU_BUDGET,
    REPO / "tools/colab_chat/tiny_chat_data.py",
    *sorted((HERE / "specs").glob("*.json")),
    *sorted((REPO / "tools/colab_chat").glob("carry12_[BSPX]*_r[123].json")))
SEEDS = [1, 2, 3]
CELLS = ["W512", "B", "D", "TS", "Sgist", "P16", "X"]
LAYOUTS = ["B", "Sgist", "P16", "X"]


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


def data_ledger(value):
    """Only operational keys and documented no-op argument defaults normalize.

    tiny_chat_data.py parser defaults (carry12/template_sft store_true, long_sft
    default empty) document these no-op values; that source is frozen too.
    Never discard hashes, counts, source/mix/window settings or template content.
    `out` values include temporary filenames differing across old/new builders.
    """
    if isinstance(value, dict):
        result = {k: data_ledger(v) for k, v in value.items()
                  if k not in ("out", "path")}
        if "args" in result:
            for key, default in (("carry12", False), ("template_sft", False), ("long_sft", "")):
                result["args"].setdefault(key, default)
        return result
    if isinstance(value, list):
        return [data_ledger(v) for v in value]
    return value


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
        self.output = Path(output).resolve() if output else self.art / "carry12_core_final"
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

    def inputs(self):
        """Exact frozen core only; absence waits, malformed or competing input halts."""
        pending, files, pairs = [], {}, []
        analysis = load_module("carry12_registered", HERE / "analyze.py")
        normalizer = data_ledger
        depth = load_json(E1 / "size/depth_choice.json")["depth"]
        if depth != 8:
            raise ValueError("registered frozen core requires L*=8 and exactly 21 reports")
        expected = {}
        for seed in SEEDS:
            for cell in CELLS:
                machine = "B" if cell in ("W512", "B", "D", "TS") else cell
                path = self.art / f"c12_{machine}_r{seed}" / f"post_c12_{cell}_r{seed}" / "summary.json"
                expected[path] = (cell, seed)
        actual = set(self.art.glob("*/post_c12_*/summary.json")) | set(self.art.glob("post_c12_*/summary.json"))
        extra = actual - set(expected)
        if extra:
            raise ValueError("unexpected/duplicate/conditional post summaries: " + str(sorted(map(str, extra))))
        for path, (cell, seed) in expected.items():
            if not path.is_file():
                pending.append(str(path))
                continue
            row = load_json(path)
            if (row.get("cell") != cell or type(row.get("seed")) is not int
                    or row["seed"] != seed or row.get("mode") != "post"):
                raise ValueError("wrong report identity: " + str(path))
            metric = analysis.get(row, "Y12_gold" if cell == "W512" else "Y12")
            if type(metric) not in (int, float) or not math.isfinite(metric) or "error" in row.get("carry12", {}):
                raise ValueError("incomplete/nonfinite Carry-12 summary: " + str(path))
            files[str(path)] = sha(path)
            pairs.append([cell, seed])
        # Preserve each registered layout's full data receipt. B alone adds TSFT;
        # the T512 build must reproduce E1's data, not the T1024 build.
        for layout in LAYOUTS + ["data512"]:
            reference = (self.art / f"e1_60m_L{depth}" / "data/data.ledger.json" if layout == "data512"
                         else self.art / f"c12_{layout}_r1" / "data/data.ledger.json")
            if not reference.is_file():
                pending.append(str(reference))
                continue
            files[str(reference)] = sha(reference)
            wanted = normalizer(load_json(reference))
            for seed in SEEDS:
                block = f"c12_B_r{seed}" if layout == "data512" else f"c12_{layout}_r{seed}"
                leaf = "data512" if layout == "data512" else "data"
                path = self.art / block / leaf / "data.ledger.json"
                if not path.is_file():
                    pending.append(str(path))
                    continue
                if normalizer(load_json(path)) != wanted:
                    raise ValueError("data ledger differs from reference: " + str(path))
                files[str(path)] = sha(path)
        for layout in LAYOUTS:
            for seed in SEEDS:
                path = self.art / f"c12_{layout}_r{seed}" / "gate/gate.json"
                if not path.is_file():
                    pending.append(str(path))
                    continue
                gate = load_json(path)
                if gate.get("ok") is not True or gate.get("problems") != []:
                    raise ValueError("registered execution gate failed: " + str(path))
                files[str(path)] = sha(path)
        # Descriptive B-chat is deliberately optional, but any home results read
        # by the unchanged analysis are part of the immutable input snapshot.
        for seed in SEEDS:
            folder = self.art / f"c12_B_r{seed}" / f"chat_c12_r{seed}"
            for pattern in ("chat_eval_results.tar.gz", "*/summary.json", "*/qa_summary.json"):
                for path in folder.glob(pattern):
                    files[str(path)] = sha(path)
        # Reproduction checks read these E1 SFT hashes; wait for all references.
        for seed in SEEDS:
            path = self.art / f"e1_60m_L{depth}" / f"post_e1_60m_L{depth}_r{seed}" / "summary.json"
            if path.is_file():
                files[str(path)] = sha(path)
            else:
                pending.append(str(path))
        return {"selected_cell_seed_pairs": sorted(pairs), "sha256": files,
                "pending": sorted(set(pending)), "ready": not pending}

    @staticmethod
    def verify_outputs(folder):
        report = load_json(Path(folder) / "analysis_carry12.json")
        core = report.get("carry12", {})
        if (report.get("partial") is not False or core.get("seeds") != SEEDS
                or any(type(s) is not int for s in core.get("seeds", []))
                or core.get("missing") != [] or set(core.get("cells", {})) != set(CELLS)
                or core.get("label") != "REGISTERED (PREREGISTRATION.md section 9)"):
            raise ValueError("not a complete registered three-seed Carry-12 core")

    def run_analyses(self, staging):
        call = [sys.executable, str(HERE / "analyze.py"), "--art", str(self.art),
                "--e1-home", str(self.art / "e1_60m_L8"),
                "--out", str(staging / "analysis_carry12.json")]
        atomic_json(staging / "commands.json", [call])
        wrapped = [sys.executable, str(CPU_BUDGET), "run", "--session", "paperkiln",
                   "--job", "registered Carry-12 core final analysis", "--cores", "1",
                   "--resume", "restart existing Carry-12 finalizer watchdog entry", "--", *call]
        with (staging / "analysis.log").open("w", encoding="utf-8") as log:
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
                if self.inputs() != completion["input_manifest"]:
                    raise ValueError("published input inventory changed")
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
                self.log("all jobs finished and home: registered three-seed Carry-12 analysis published")
                return "COMPLETE"
            if time.time() - state["created"] > self.max_hours * 3600:
                raise ValueError("Carry-12 completion deadline exceeded; operator review required")
            before = self.inputs()
            observed = state.get("observed_inputs", {})
            for path, digest in observed.items():
                if before["sha256"].get(path) != digest:
                    raise ValueError("previously observed input changed/disappeared: " + path)
            if observed != before["sha256"]:
                state["observed_inputs"] = before["sha256"]
                atomic_json(self.state_path, state)
            if not before["ready"]:
                count = len(before["pending"])
                if state.get("pending_count") != count:
                    state.update(status="WAITING", pending_count=count, checked=now())
                    atomic_json(self.state_path, state)
                    self.log(f"waiting for {count} Carry-12 inputs; no analysis performed")
                return "WAITING"
            if len(before["selected_cell_seed_pairs"]) != 21:
                raise ValueError("expected exactly 21 core cell/seed reports")
            self.verify_sources(state)
            state.update(status="ANALYSING", input_manifest=before, checked=now())
            atomic_json(self.state_path, state)
            staging = self.output / ".incoming"
            staging.mkdir(exist_ok=True)
            atomic_json(staging / "input_receipt.json", before)
            self.run_analyses(staging)
            self.verify_outputs(staging)
            self.verify_sources(state)
            fresh = self.inputs()
            if fresh != before:
                raise ValueError("Carry-12 inputs changed during analysis")
            report_names = ("analysis_carry12.json", "input_receipt.json", "commands.json")
            completion = {"completed": now(), "source_hashes": state["source_hashes"],
                          "input_manifest": before,
                          "report_hashes": {name: sha(staging / name) for name in report_names}}
            atomic_json(staging / "completion.json", completion)
            staging.rename(published)
            state.update(status="COMPLETE", completed=completion["completed"])
            atomic_json(self.state_path, state)
            self.log("all jobs finished and home: registered three-seed Carry-12 analysis published")
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
        print("another Carry-12 finalizer already owns the state directory", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
