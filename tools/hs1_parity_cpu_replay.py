"""Replay retained CPU diagnostic rows; this does not recover historical GPU logits."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def replay(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "cpu_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("artifact hash differs:" + name)
    requests = json.loads((folder / "requests.json").read_text())
    results = []
    for run in dict.fromkeys(row["run"] for row in requests):
        models = {}
        for mode in ("full", "kv"):
            prefix = run + "_" + mode
            rows = [
                json.loads(x)
                for x in (folder / (prefix + ".scores.jsonl")).read_text().splitlines()
            ]
            raw = gzip.decompress((folder / (prefix + ".logits.f32.gz")).read_bytes())
            logits = np.frombuffer(raw, dtype="<f4").reshape(-1, 16384)
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite diagnostic logits")
            offset = 0
            selected = [x for x in requests if x["run"] == run]
            if len(rows) != len(selected):
                raise ValueError("request count differs")
            for request, scored in zip(selected, rows):
                ids = request["request"]["continuation_ids"]
                if (
                    scored["key"] != request["request"]["key"]
                    or scored["continuation_ids"] != ids
                    or scored["over_ctx"]
                ):
                    raise ValueError("request identity/scope differs")
                positions = []
                for j, chosen in enumerate(ids):
                    values = logits[offset + j].copy()
                    values[manifest["banned_ids"]] = -np.inf
                    best, second = map(int, np.argsort(-values, kind="stable")[:2])
                    if best != scored["argmax"][j]:
                        raise ValueError("native argmax differs from raw logits")
                    positions.append(
                        {
                            "position_zero_based": j,
                            "saved_chosen_id": chosen,
                            "argmax_id": best,
                            "runner_up_id": second,
                            "argmax_logit": float(values[best]),
                            "runner_up_logit": float(values[second]),
                            "saved_chosen_logit": float(values[chosen]),
                            "top2_margin": float(values[best]) - float(values[second]),
                            "argmax_minus_saved_chosen": float(values[best])
                            - float(values[chosen]),
                            "saved_chosen_matches": best == chosen,
                        }
                    )
                offset += len(ids)
                models[mode, scored["key"]] = positions
            if offset != len(logits):
                raise ValueError("logit row count differs")
        for request in [x for x in requests if x["run"] == run]:
            key = request["request"]["key"]
            results.append(
                {
                    "run": run,
                    "key": key,
                    "original_gpu_failed_token_position": None,
                    "historical_score_vectors_unavailable": True,
                    "full": models["full", key],
                    "kv": models["kv", key],
                }
            )
    if len(results) != 6:
        raise ValueError("expected exactly six historical affected replies")
    return {
        "status": "VERIFIED_CPU_DIAGNOSTIC",
        "scope": "New CPU diagnostic only; not recovered original GPU margins or a demonstrated historical cause",
        "rows": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(replay(args.folder), indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")
