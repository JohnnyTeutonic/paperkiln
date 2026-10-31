"""Replay retained LayerNorm comparison bits, identities, and fixed block statistics."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

PROTOCOL_SHA256 = "b944e18372bac7486f5d8be674bea609603cbc9b5665ba1710fe095f806946b3"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replay(directory):
    directory = Path(directory)

    def read(name):
        return json.loads((directory / name).read_text())

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    require(
        sha((directory / "protocol.json").read_bytes()) == PROTOCOL_SHA256,
        "frozen protocol hash mismatch",
    )
    plan = read("protocol.json")
    result = read("results.json")
    require(
        sha((directory / "protocol.json").read_bytes()) == result["protocol_sha256"],
        "sha((directory / 'protocol.json').read_bytes()) == result['protocol_sha256']",
    )
    require(
        sha((directory / "trials.jsonl").read_bytes()) == result["trials_sha256"],
        "sha((directory / 'trials.jsonl').read_bytes()) == result['trials_sha256']",
    )
    require(
        gzip.decompress((directory / "baseline_bits.json.gz").read_bytes())
        == gzip.decompress((directory / "candidate_bits.json.gz").read_bytes()),
        "gzip.decompress((directory / 'baseline_bits.json.gz').read_bytes()) == gzip.decompress((directory / 'candidate_bits.json.gz').read_bytes())",
    )
    for variant in ["baseline", "candidate"]:
        require(
            sha((directory / (variant + "_bits.json.gz")).read_bytes())
            == read(variant + "_identity.json")["bitdump_sha256"],
            "sha((directory / (variant + '_bits.json.gz')).read_bytes()) == read(variant + '_identity.json')['bitdump_sha256']",
        )
    canonical = (
        sha(gzip.decompress((directory / "canonical_scores.jsonl.gz").read_bytes())),
        sha(gzip.decompress((directory / "canonical_logits.f32.gz").read_bytes())),
    )
    trials = [
        json.loads(line)
        for line in (directory / "trials.jsonl").read_text().splitlines()
    ]
    require(
        len(trials) == plan["blocks"] * 4 == 32,
        "len(trials) == plan['blocks'] * 4 == 32",
    )
    cases = {(r, c, b) for r, c in plan["operation_shapes"] for b in [False, True]}
    op_blocks = []
    score_blocks = []
    for block in range(plan["blocks"]):
        rows = [r for r in trials if r["block"] == block]
        require(
            len(rows) == 4 and [r["position"] for r in rows] == list(range(4)),
            "len(rows) == 4 and [r['position'] for r in rows] == list(range(4))",
        )
        require(
            [r["variant"] for r in rows] == plan["order_per_block"],
            "[r['variant'] for r in rows] == plan['order_per_block']",
        )
        for r in rows:
            require(
                len(r["operation"]) == 16
                and {(a["rows"], a["cols"], a["backward"]) for a in r["operation"]}
                == cases,
                "len(r['operation']) == 16 and {(a['rows'], a['cols'], a['backward']) for a in r['operation']} == cases",
            )
            require(
                canonical == (r["score"]["score_sha256"], r["score"]["logits_sha256"]),
                "canonical == (r['score']['score_sha256'], r['score']['logits_sha256'])",
            )
            require(
                math.isfinite(r["score"]["wall_ns"]) and r["score"]["wall_ns"] > 0,
                "math.isfinite(r['score']['wall_ns']) and r['score']['wall_ns'] > 0",
            )
            for a in r["operation"]:
                require(
                    a["iterations"] == plan["timed_calls"]
                    and a["warmup"] == plan["warmup_calls"],
                    "a['iterations'] == plan['timed_calls'] and a['warmup'] == plan['warmup_calls']",
                )
                require(
                    math.isfinite(a["ns_per_call"]) and a["ns_per_call"] > 0,
                    "math.isfinite(a['ns_per_call']) and a['ns_per_call'] > 0",
                )
        case_ratios = []
        for case in sorted(cases):
            times = {"A": [], "B": []}
            for r in rows:
                a = next(
                    (
                        a
                        for a in r["operation"]
                        if (a["rows"], a["cols"], a["backward"]) == case
                    )
                )
                times[r["variant"]].append(a["ns_per_call"])
            case_ratios.append(
                statistics.mean(times["A"]) / statistics.mean(times["B"])
            )
        op_blocks.append(math.exp(statistics.mean(map(math.log, case_ratios))))
        score_blocks.append(
            statistics.mean(
                (r["score"]["wall_ns"] for r in rows if r["variant"] == "A")
            )
            / statistics.mean(
                (r["score"]["wall_ns"] for r in rows if r["variant"] == "B")
            )
        )
    indices = np.random.default_rng(20261031).integers(0, 8, size=(10000, 8))

    def summary(values):
        draws = [
            math.exp(statistics.mean((math.log(values[i]) for i in sample)))
            for sample in indices
        ]
        return (
            math.exp(statistics.mean(map(math.log, values))),
            np.quantile(draws, [0.025, 0.975]).tolist(),
        )

    verified = {}
    for name, values in [
        ("operation_aggregate", op_blocks),
        ("actual_score", score_blocks),
    ]:
        value, interval = summary(values)
        require(
            math.isclose(value, result[name]["geometric_mean"], rel_tol=1e-12),
            "math.isclose(value, result[name]['geometric_mean'], rel_tol=1e-12)",
        )
        require(
            np.allclose(interval, result[name]["bootstrap95_ci"], rtol=1e-12, atol=0),
            "np.allclose(interval, result[name]['bootstrap95_ci'], rtol=1e-12, atol=0)",
        )
        verified[name] = {"speedup": value, "bootstrap95_ci": interval}
    require(
        result["adopt"] == all((v["bootstrap95_ci"][0] > 1 for v in verified.values())),
        "result['adopt'] == all((v['bootstrap95_ci'][0] > 1 for v in verified.values()))",
    )
    return {
        "status": "PASS",
        "trials": len(trials),
        "bit_dumps_equal": True,
        "canonical_score_and_logits_match_all_trials": True,
        **verified,
        "adopt": result["adopt"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(replay(parser.parse_args().directory), indent=2))
