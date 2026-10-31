"""Fixed source-bound ABBA layernorm comparison; no adaptive timing decisions."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/receipts/cpu_layernorm_20261031/protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kwargs):
    p = subprocess.run(list(map(str, argv)), capture_output=True, timeout=120, **kwargs)
    if p.returncode:
        raise RuntimeError(str(argv) + "\n" + p.stderr.decode(errors="replace"))
    return p


def store(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def snapshot():
    return {
        p: sha(ROOT / p)
        for p in [
            "src/ops.cpp",
            "tools/bench_layernorm_cpu.cpp",
            "tools/compare_layernorm_cpu.py",
            "CMakeLists.txt",
        ]
    }


def install(build, art, name):
    directory = art / name
    directory.mkdir()
    for executable in ["mtstudio", "bench_layernorm_cpu"]:
        shutil.copyfile(build / executable, directory / executable)
        (directory / executable).chmod(0o755)
    raw = run([directory / "bench_layernorm_cpu", "dump"]).stdout
    with (directory / "bits.json.gz").open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0))
    dump = json.loads(raw)
    if not all(dump["malformed_dy_refused"]):
        raise ValueError("malformed derivative was not refused")
    store(
        directory / "identity.json",
        {
            "sources": snapshot(),
            "binary_sha256": {
                p: sha(directory / p) for p in ["mtstudio", "bench_layernorm_cpu"]
            },
            "bitdump_sha256": sha(directory / "bits.json.gz"),
            "cases": len(dump["cases"]),
            "models": len(dump["models"]),
        },
    )


def prepare(build, art):
    art.mkdir()
    shutil.copyfile(PROTOCOL, art / "protocol.json")
    install(build, art, "baseline")
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
    from test_thirdparty_ir import spec_of

    spec = spec_of(
        art / "fixture",
        "model",
        "gpt2-nano",
        {
            "d": 16,
            "layers": 2,
            "heads": 2,
            "d_ff": 24,
            "norm": "layernorm",
            "activation": "gelu",
            "position": "learned",
        },
        steps=3,
        formats=("safetensors",),
    )
    p = run([art / "baseline/mtstudio", "run", spec])
    (art / "training.log").write_bytes(p.stdout + p.stderr)
    case = {"prompt_ids": [1, 2, 3, 4], "continuation_ids": [5, 6, 7, 8, 9, 10, 11, 12]}
    (art / "requests.jsonl").write_text((json.dumps(case) + "\n") * 12000)
    store(
        art / "fixture.json",
        {
            "spec": str(spec),
            "sha256": {
                str(p): sha(p)
                for p in [
                    spec,
                    art / "requests.jsonl",
                    art / "fixture/model/model.safetensors",
                    art / "fixture/v.gguf",
                    art / "fixture/c.txt",
                ]
            },
        },
    )


def compare(build, art):
    if sha(PROTOCOL) != sha(art / "protocol.json"):
        raise ValueError("protocol drift")
    install(build, art, "candidate")
    if gzip.decompress((art / "baseline/bits.json.gz").read_bytes()) != gzip.decompress(
        (art / "candidate/bits.json.gz").read_bytes()
    ):
        store(art / "HALTED.json", {"reason": "operation/model bitdump mismatch"})
        raise ValueError("bitwise operation/model mismatch; no timing performed")
    plan = json.loads(PROTOCOL.read_text())
    sources = snapshot()
    fixture = json.loads((art / "fixture.json").read_text())
    spec = Path(fixture["spec"])
    for path, h in fixture["sha256"].items():
        if sha(path) != h:
            raise ValueError("fixture drift")
    expected = {(r, c, b) for r, c in plan["operation_shapes"] for b in [False, True]}
    canonical = None

    def score(name, label, retain=False):
        output = art / (label + ".scores.jsonl")
        logits = art / (label + ".logits.f32")
        start = time.perf_counter_ns()
        p = run(
            [
                art / name / "mtstudio",
                "score",
                spec,
                "--in",
                art / "requests.jsonl",
                "--out",
                output,
                "--logits",
                logits,
            ]
        )
        elapsed = time.perf_counter_ns() - start
        rows = [json.loads(s) for s in output.read_text().splitlines()]
        if len(rows) != 12000 or any(
            a["over_ctx"]
            or len(a["argmax"]) != 8
            or any(not math.isfinite(v) for v in a["logprobs"])
            for a in rows
        ):
            raise ValueError("incomplete score trial")
        value = {
            "wall_ns": elapsed,
            "score_sha256": sha(output),
            "logits_sha256": sha(logits),
            "stderr": p.stderr.decode(errors="replace"),
        }
        if canonical is not None and canonical != (
            value["score_sha256"],
            value["logits_sha256"],
        ):
            store(
                art / "HALTED.json",
                {
                    "reason": "score/logit mismatch",
                    "trial": label,
                    "observed": value,
                    "expected": canonical,
                },
            )
            raise ValueError("score/logit mismatch; raw first mismatch preserved")
        if not retain:
            output.unlink()
            logits.unlink()
        return value

    # Warmups are retained canonical correctness output, excluded from timing statistics.
    for name in ["baseline", "candidate"]:
        v = score(name, "warmup-" + name, True)
        if canonical is None:
            canonical = (v["score_sha256"], v["logits_sha256"])
        if canonical != (v["score_sha256"], v["logits_sha256"]):
            raise ValueError("actual score/logits bit mismatch")
    trials = []
    checksums = None
    with (art / "trials.jsonl").open("x") as journal:
        for block in range(plan["blocks"]):
            for position, variant in enumerate(plan["order_per_block"]):
                name = "baseline" if variant == "A" else "candidate"
                p = run([art / name / "bench_layernorm_cpu", "bench"])
                op = json.loads(p.stdout)
                if (
                    len(op) != 16
                    or {(a["rows"], a["cols"], a["backward"]) for a in op} != expected
                    or any(
                        a["iterations"] != 1024
                        or a["warmup"] != 32
                        or not math.isfinite(a["ns_per_call"])
                        or a["ns_per_call"] <= 0
                        or not math.isfinite(a["checksum"])
                        for a in op
                    )
                ):
                    raise ValueError("invalid operation trial")
                observed_checksums = {
                    (a["rows"], a["cols"], a["backward"]): a["checksum"] for a in op
                }
                if checksums is None:
                    checksums = observed_checksums
                if observed_checksums != checksums:
                    store(
                        art / "HALTED.json",
                        {
                            "reason": "operation checksum drift",
                            "block": block,
                            "position": position,
                            "operation": op,
                        },
                    )
                    raise ValueError("operation checksum mismatch")
                scored = score(name, f"b{block}-p{position}")
                if canonical != (scored["score_sha256"], scored["logits_sha256"]):
                    raise ValueError("trial numerical drift")
                record = {
                    "block": block,
                    "position": position,
                    "variant": variant,
                    "operation": op,
                    "score": scored,
                }
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                trials.append(record)
    for name in ["baseline", "candidate"]:
        identity = json.loads((art / name / "identity.json").read_text())
        if any(sha(art / name / p) != h for p, h in identity["binary_sha256"].items()):
            raise ValueError("binary drift during trials")
    if any(sha(p) != h for p, h in fixture["sha256"].items()):
        raise ValueError("fixture drift during trials")
    if sources != snapshot():
        raise ValueError("source drift during trials")
    ratios = []
    score_ratios = []
    for b in range(8):
        records = [a for a in trials if a["block"] == b]

        def mean(case, variant):
            return np.mean(
                [
                    next(
                        v["ns_per_call"]
                        for v in a["operation"]
                        if (v["rows"], v["cols"], v["backward"]) == case
                    )
                    for a in records
                    if a["variant"] == variant
                ]
            )

        ratios.append(
            float(
                np.exp(
                    np.mean(
                        [
                            math.log(mean(c, "A") / mean(c, "B"))
                            for c in sorted(expected)
                        ]
                    )
                )
            )
        )
        score_ratios.append(
            float(
                np.mean([a["score"]["wall_ns"] for a in records if a["variant"] == "A"])
                / np.mean(
                    [a["score"]["wall_ns"] for a in records if a["variant"] == "B"]
                )
            )
        )
    samples = np.random.default_rng(20261031).integers(0, 8, size=(10000, 8))

    def estimate(values):
        logs = np.log(values)
        boots = np.exp(logs[samples].mean(axis=1))
        return {
            "block_ratios": values,
            "geometric_mean": float(np.exp(logs.mean())),
            "bootstrap95_ci": np.quantile(boots, [0.025, 0.975]).tolist(),
        }

    op = estimate(ratios)
    score_result = estimate(score_ratios)
    store(
        art / "results.json",
        {
            "status": "COMPLETE",
            "adopt": op["bootstrap95_ci"][0] > 1
            and score_result["bootstrap95_ci"][0] > 1,
            "operation_aggregate": op,
            "actual_score": score_result,
            "trials": len(trials),
            "sources": sources,
            "protocol_sha256": sha(PROTOCOL),
            "trials_sha256": sha(art / "trials.jsonl"),
            "limits": "Tiny CPU teacher-forced score, startup included; fixed equal-weight operation cases. Contiguous blocks may be correlated; no general Transformer or GPU claim.",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "compare"])
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--art", type=Path, required=True)
    a = ap.parse_args()
    (prepare if a.action == "prepare" else compare)(a.build, a.art)


if __name__ == "__main__":
    main()
