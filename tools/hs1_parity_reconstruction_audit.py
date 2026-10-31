"""Audit landed reconstruction evidence without compute, network or scientific-result edits."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np

V = 16384
EXPECTED_MODELS = {"c12_B_r1", "c12_B_r3", "c12_Sgist_r1", "c12_Sgist_r4"}
EXPECTED_KEYS = {
    "carry12_v1:recall:0017|gate_self|4",
    "carry12_v1:correction:0046|gate_self|2",
    "carry12_v1:correction:0224|gate_self|2",
    "carry12_v1:correction:0306|gate_self|2",
    "carry12_v1:correction:0111|gate_self|3",
    "carry12_v1:recall:0365|gate_self|8",
}


def audit_chunk(requests, original, diagnostic, raw, targets):
    if len(original) != len(requests) or original != diagnostic:
        raise ValueError("original/tap request inventory or numerical results differ")
    if len({r["key"] for r in requests}) != len(requests):
        raise ValueError("duplicate request identity")
    result = []
    values = np.frombuffer(raw, dtype="<f4").reshape(-1, V)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite target logits")
    offset = 0
    for request, scored in zip(requests, original):
        key, chosen = request["key"], request["continuation_ids"]
        expected_over = len(request["prompt_ids"]) + len(chosen) > 1025
        if (
            scored.get("key") != key
            or scored.get("continuation_ids") != chosen
            or scored.get("over_ctx") is not expected_over
        ):
            raise ValueError("request key/order/continuation/context differs")
        if not expected_over:
            if (
                len(scored["argmax"]) != len(chosen)
                or len(scored["logprobs"]) != len(chosen)
                or not math.isfinite(scored["sum"])
                or not all(math.isfinite(x) for x in scored["logprobs"])
            ):
                raise ValueError("scorer row finite/inventory differs")
        if key not in targets:
            continue
        if expected_over:
            raise ValueError("target unexpectedly exceeds context")
        positions = []
        for j, token in enumerate(chosen):
            row = values[offset + j].copy()
            row[0] = -np.inf
            best, second = map(int, np.argsort(-row, kind="stable")[:2])
            if best != scored["argmax"][j]:
                raise ValueError(
                    "raw target logits disagree with untouched scorer argmax"
                )
            positions.append(
                {
                    "position_zero_based": j,
                    "saved_generation_id": token,
                    "reconstructed_scorer_argmax": best,
                    "runner_up": second,
                    "argmax_logit": float(row[best]),
                    "runner_up_logit": float(row[second]),
                    "saved_chosen_logit": float(row[token]),
                    "top2_margin": float(row[best]) - float(row[second]),
                    "argmax_minus_saved": float(row[best]) - float(row[token]),
                    "matches_saved_generation": best == token,
                }
            )
        offset += len(chosen)
        result.append(
            {
                "key": key,
                "positions": positions,
                "historical_failed_position_unknown": True,
            }
        )
    if offset != len(values) or {x["key"] for x in result} != set(targets):
        raise ValueError("target logit inventory differs")
    return result


def audit(archive, stage):
    stage = Path(stage)
    expected = (stage / "hs1_inputs.json").read_bytes()
    manifest = json.loads(expected)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if (
            len(names) != len(set(names))
            or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names)
            or sum(x.file_size for x in z.infolist()) > 512 << 20
        ):
            raise ValueError("unsafe/duplicate/oversized archive")
        data = {n: z.read(n) for n in names}
    summary = json.loads(data["summary.json"])
    if (
        summary["status"] != "SCORER_RECONSTRUCTION_COMPLETE"
        or summary["errors"]
        or any(
            summary.get(k) is not True
            for k in (
                "original_source_restored",
                "spec_delta_path_only_verified",
                "exact_historical_chunk_reconstruction_verified",
            )
        )
    ):
        raise ValueError(
            "reconstruction incomplete or failed; preserve negative evidence"
        )
    if (
        data["input_manifest.json"] != (json.dumps(manifest, indent=2) + "\n").encode()
        or summary["input_manifest_sha256"] != hashlib.sha256(expected).hexdigest()
    ):
        raise ValueError("prospective manifest binding differs")
    for name, wanted in manifest["files"].items():
        if hashlib.sha256((stage / name).read_bytes()).hexdigest() != wanted:
            raise ValueError("staged input changed:" + name)
    for name, wanted in summary["files_sha256"].items():
        if hashlib.sha256(data[name]).hexdigest() != wanted:
            raise ValueError("landed evidence hash differs:" + name)
    if (
        data["diagnostic_source.cpp"]
        != (stage / "mtstudio_diagnostic.cpp").read_bytes()
    ):
        raise ValueError("diagnostic source differs")
    if (
        manifest["score_batch_tokens"] != 16384
        or manifest["chunk_size"] != 256
        or set(manifest["models"]) != EXPECTED_MODELS
        or set(summary["models"]) != EXPECTED_MODELS
    ):
        raise ValueError("registered model/geometry identity differs")
    if any(c["returncode"] != 0 for c in summary["commands"]):
        raise ValueError("command failure retained")
    commands = {x["name"]: x["argv"] for x in summary["commands"]}
    result, count = [], 0
    for run, model in manifest["models"].items():
        for chunk in model["chunks"]:
            prefix = run + "_chunk" + str(chunk["index"])
            for mode in ("original", "diagnostic"):
                command = commands[prefix + "_" + mode]
                if (
                    command[1] != "score"
                    or command[command.index("--batch-tokens") + 1] != "16384"
                    or "--kv-cache" in command
                ):
                    raise ValueError("scorer execution geometry differs")
            requests = [
                json.loads(x) for x in (stage / chunk["file"]).read_text().splitlines()
            ]
            if (
                len(requests) != 256
                or hashlib.sha256((stage / chunk["file"]).read_bytes()).hexdigest()
                != chunk["sha256"]
            ):
                raise ValueError("exact original chunk differs")
            original = [
                json.loads(x)
                for x in data[prefix + "_original.jsonl"].decode().splitlines()
            ]
            diagnostic = [
                json.loads(x)
                for x in data[prefix + "_diagnostic.jsonl"].decode().splitlines()
            ]
            result.extend(
                audit_chunk(
                    requests,
                    original,
                    diagnostic,
                    data[prefix + ".f32"],
                    chunk["target_keys"],
                )
            )
            count += len(requests)
    if (
        count != 1536
        or {x["key"] for x in result} != EXPECTED_KEYS
        or sum(len(x["positions"]) for x in result) != 89
    ):
        raise ValueError("complete reconstruction scope differs")
    if (
        manifest.get("source_archive_inventory")
        and summary.get("exact_source_inventory_verified") is not True
    ):
        raise ValueError("strict extracted source inventory not verified")
    generation = []
    if "generation_skipped" not in summary:
        if (
            "CUDA Enabled. Will define HAS_CUDA"
            not in data["ember_configure.log"].decode()
        ):
            raise ValueError("Ember CUDA build not proved")
        for run, model in manifest["models"].items():
            proof = summary["models"][run]
            if (
                proof["export"] != "EXACT_HASH"
                or proof["gguf_sha256"] != model["original_gguf_sha256"]
                or proof.get("selected_cuda_runtime_proved") is not True
                or proof["health"].get("deterministic") is not True
            ):
                raise ValueError(
                    "exact generation source/backend identity not established"
                )
            rows = proof["generation_reconstruction"]
            expected_cases = {
                (c["request"]["key"], repeat)
                for c in model["cases"]
                for repeat in (0, 1)
            }
            if (
                len(rows) != len(expected_cases)
                or {(x["key"], x["repeat"]) for x in rows} != expected_cases
            ):
                raise ValueError("generation repeat inventory differs")
            for row in rows:
                case = next(
                    c for c in model["cases"] if c["request"]["key"] == row["key"]
                )
                if row["response"].get("prompt_ids") != case["request"]["prompt_ids"]:
                    raise ValueError("generation prompt IDs differ")
                expected_body = {
                    "user_input": case["user"],
                    "history": [
                        {"user": u, "assistant": a} for u, a in case["history"]
                    ],
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": 7,
                    "max_new_tokens": 48,
                    "return_ids": True,
                }
                if row["request"] != expected_body:
                    raise ValueError("generation request contract differs")
                generation.append(
                    {
                        "run": run,
                        "key": row["key"],
                        "repeat": row["repeat"],
                        "ids_match_saved": row["response"].get("ids")
                        == case["expected"]["ids"],
                        "stop_match_saved": row["response"].get("stop_reason")
                        == case["expected"]["stop"],
                        "response": row["response"],
                    }
                )
        if len(generation) != 12:
            raise ValueError("expected twelve generation responses")
    return {
        "status": "RECONSTRUCTION_AUDITED",
        "scope": "New exact-source reconstruction, not recovered historical logits, research validity pass or parity waiver",
        "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(expected).hexdigest(),
        "requests_numerically_identical": count,
        "reply_positions": 89,
        "rows": result,
        "generation": generation,
        "generation_skip_reason": summary.get("generation_skipped"),
        "source_binary_sha256": {
            "original": summary["original_binary_sha256"],
            "diagnostic": summary["diagnostic_binary_sha256"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.stage)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
