"""Retrospective scorer-only audit of an overall failed reconstruction. Never a full pass."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from hs1_parity_reconstruction_audit import EXPECTED_KEYS, EXPECTED_MODELS, audit_chunk


def verify_partial_boundary(summary):
    if (
        summary["status"] != "ERROR"
        or summary["errors"] != ["ember_checkout timed out"]
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


def audit_component(archive, stage):
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
    verify_partial_boundary(summary)
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
    if "CUDA enabled - version" not in data["configure.log"].decode():
        raise ValueError("native CUDA configure not established")
    if any(
        "generation_reconstruction" in model or "health" in model
        for model in summary["models"].values()
    ):
        raise ValueError("unexpected generation evidence in scorer-only audit")
    exports = {}
    for run, model in manifest["models"].items():
        actual = summary["models"][run]
        if (
            actual["export"] != "EXACT_HASH"
            or actual["gguf_sha256"] != model["original_gguf_sha256"]
        ):
            raise ValueError("original GGUF byte identity not established")
        exports[run] = actual["gguf_sha256"]
    return {
        "status": "SCORER_COMPONENT_AUDITED_OVERALL_ERROR",
        "overall_run_status": summary["status"],
        "overall_run_errors": summary["errors"],
        "scope": "Retrospective audit of completed native scorer component only. No generation reconstruction, historical cause, full gate, validity pass or parity waiver.",
        "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(expected).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path(__file__).with_name("hs1_parity_reconstruction_audit.py").read_bytes()
        ).hexdigest(),
        "requests_numerically_identical": count,
        "reply_positions": 89,
        "rows": result,
        "gguf_exports_exact": exports,
        "generation": [],
        "generation_skip_reason": "NOT_EXECUTED: historical Ember checkout timed out before configure/build/server",
        "source_binary_sha256": {
            "original": summary["original_binary_sha256"],
            "diagnostic": summary["diagnostic_binary_sha256"],
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit_component(a.archive, a.stage)
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
