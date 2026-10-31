"""Versioned summary/detail correction; V1 failures and all original hashes remain.

Detailed report bytes are recorded, not claimed equal to an unavailable historical
raw report. Explicit host path overrides change locations only, never expectations.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

REFERENCE_SHA = "acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def normal(value):
    # Exact existing registered finalizer's operational path/no-op-default contract.
    if isinstance(value, dict):
        r = {k: normal(v) for k, v in value.items() if k not in ("out", "path")}
        if "args" in r:
            for k, v in [("carry12", False), ("template_sft", False), ("long_sft", "")]:
                r["args"].setdefault(k, v)
        return r
    if isinstance(value, list):
        return [normal(v) for v in value]
    return value


def verify(data, reference, repo=None, *, vocab_path=None, human_dir=None):
    data, reference = Path(data), Path(reference)
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    if sha(reference) != REFERENCE_SHA:
        raise ValueError("reference ledger hash differs")
    ref = json.loads(reference.read_text())
    actual = json.loads((data / "data.ledger.json").read_text())
    if normal(actual) != normal(ref):
        raise ValueError(
            "complete registered data/source/decontamination ledger differs"
        )
    checked = {}
    for name, wanted in ref["outputs"].items():
        path = data / name
        if (
            Path(name).name != name
            or not path.is_file()
            or path.stat().st_size != wanted["bytes"]
            or sha(path) != wanted["sha256"]
        ):
            raise ValueError("data output differs:" + name)
        checked[name] = wanted["sha256"]
    vocab = (
        Path(vocab_path) if vocab_path is not None else Path(actual["args"]["vocab"])
    )
    human = Path(human_dir) if human_dir is not None else data / "human"
    if sha(vocab) != ref["vocab_sha256"]:
        raise ValueError("vocabulary differs")
    for name, wanted in ref["human"]["files"].items():
        p = human / name
        if not p.resolve().is_relative_to(human.resolve()) or sha(p) != wanted:
            raise ValueError("raw human source differs:" + name)
    for name, key in [
        ("carry5_v1.jsonl", "carry5_v1_sha256"),
        ("carry12_v1.jsonl", "carry12_v1_sha256"),
    ]:
        if sha(repo / "corpus/eval" / name) != ref["human"][key]:
            raise ValueError("evaluation source differs:" + name)
    report = json.loads((data / "human_decontam.json").read_text())
    validate_detail(report, ref["steps"]["decontam"])
    return {
        "verified": True,
        "validation_version": 2,
        "detailed_decontam_sha256": sha(data / "human_decontam.json"),
        "raw_detailed_historical_byte_identity_claimed": False,
        "path_only_overrides": {
            "vocab_path": str(vocab) if vocab_path is not None else None,
            "human_dir": str(human) if human_dir is not None else None,
        },
        "reference_sha256": REFERENCE_SHA,
        "ledger_sha256": sha(data / "data.ledger.json"),
        "outputs_sha256": checked,
        "vocab_sha256": ref["vocab_sha256"],
        "raw_human_sources_verified": len(ref["human"]["files"]),
        "evaluation_sources_verified": 2,
        "no_data_mutation": True,
    }


FROZEN_REMOTE = "f616a0f85447f95f67d0f9bdf6dfef513dc632a3"
V1_SOURCE_SHA = "59c27187e31a282fa394ccab7730a13a4bc148b86c3ce251b26a0cd7db9f979b"
VM_WITNESS_SHA = "c9177f35110d8a08640bee6da386a98aca742e9d2e7669978baa202cf9273039"
VM_SCRIPT_SHA = "0557808b649c75dc1c657d6d218b9babe7dfe96be70155060816634c1baed777"
VM_PATCHED_SOURCE_SHA = (
    "853896010b1a1e970790d1fff4871fe390071a38bfbf6cd65c7e0ef369282f39"
)
PROTOCOL_SHA = "3640b1595afb5847e66081b7a6fc10766caed7c4fa5b9cb61b2e3957fb1b4ea9"
VM_RAW_REPORT_SHA = "0213ca6898f2589ac4b704b9240a70052ea5d522773f12ec65633de8cde936cf"
BEFORE = 'if normal(report) != normal(ref["steps"]["decontam"]):'
AFTER = 'if not isinstance(report.get("summary"), dict) or normal(report["summary"]) != normal(ref["steps"]["decontam"]):'


def validate_detail(report, expected_summary):
    if type(report) is not dict or set(report) != {
        "summary",
        "dropped",
        "flagged",
        "eval_hits",
    }:
        raise ValueError("detailed decontamination shape differs")
    if (
        type(report["summary"]) is not dict
        or type(report["dropped"]) is not list
        or type(report["flagged"]) is not list
        or type(report["eval_hits"]) is not dict
        or any(
            type(row) is not dict
            for key in ("dropped", "flagged")
            for row in report[key]
        )
        or any(
            type(key) is not str or type(value) is not int or value < 0
            for key, value in report["eval_hits"].items()
        )
    ):
        raise ValueError("detailed decontamination types differ")
    if normal(report["summary"]) != normal(expected_summary):
        raise ValueError("decontamination summary differs")
    if any(
        len(report[key]) != expected_summary[count]
        for key, count in [
            ("dropped", "rows_dropped"),
            ("flagged", "answer_flags"),
            ("eval_hits", "evals_hit"),
        ]
    ):
        raise ValueError("decontamination detail lengths differ from summary")


def verify_witness(witness, reference, local_result):
    result = witness.get("corrected_contract_result", {})
    if (
        witness.get("kind") != "SUPPLEMENTAL_READ_ONLY_SUMMARY_SHAPE_DIAGNOSTIC_V2"
        or witness.get("source_commit") != FROZEN_REMOTE
        or witness.get("original_verifier_sha256") != V1_SOURCE_SHA
        or witness.get("diagnostic_source_sha256") != VM_PATCHED_SOURCE_SHA
        or witness.get("only_source_change") != {"before": BEFORE, "after": AFTER}
        or witness.get("original_gate_pass") is not False
        or witness.get("files_or_job_states_modified") is not False
        or witness.get("raw_detailed_historical_byte_identity_claimed") is not False
        or witness.get("raw_report_keys")
        != sorted(["summary", "dropped", "flagged", "eval_hits"])
        or witness.get("raw_decontam_sha256") != VM_RAW_REPORT_SHA
        or local_result["detailed_decontam_sha256"] != VM_RAW_REPORT_SHA
        or normal(witness.get("summary")) != normal(reference["steps"]["decontam"])
    ):
        raise ValueError("VM diagnostic source or summary witness differs")
    expected = {
        "verified": True,
        "reference_sha256": REFERENCE_SHA,
        "ledger_sha256": local_result["ledger_sha256"],
        "outputs_sha256": local_result["outputs_sha256"],
        "vocab_sha256": reference["vocab_sha256"],
        "raw_human_sources_verified": 13,
        "evaluation_sources_verified": 2,
        "no_data_mutation": True,
    }
    if result != expected or witness.get("five_products") != {
        name: {"bytes": value["bytes"], "sha256": value["sha256"]}
        for name, value in reference["outputs"].items()
    }:
        raise ValueError("VM complete verification result differs")
    raw = witness.get("raw_human_witness")
    if (
        type(raw) is not dict
        or len(raw) != 13
        or set(raw) != set(reference["human"]["files"])
    ):
        raise ValueError("VM raw-source witness inventory differs")
    for name, wanted in reference["human"]["files"].items():
        if raw[name] != {"expected_sha256": wanted, "actual_sha256": wanted}:
            raise ValueError("VM raw human witness differs")
    summary = reference["steps"]["decontam"]
    if witness.get("detailed_lengths") != {
        "dropped": summary["rows_dropped"],
        "flagged": summary["answer_flags"],
        "eval_hits": summary["evals_hit"],
    }:
        raise ValueError("VM detailed lengths differ")


def verify_relay(
    data,
    reference,
    *,
    witness_path,
    diagnostic_source_path,
    protocol_manifest_path,
    v1_state_path,
    repo=None,
    vocab_path=None,
    human_dir=None,
):
    """Supplemental host verification; never changes a V1 job or gate marker."""
    for path, expected in [
        (witness_path, VM_WITNESS_SHA),
        (diagnostic_source_path, VM_SCRIPT_SHA),
        (protocol_manifest_path, PROTOCOL_SHA),
    ]:
        if sha(path) != expected:
            raise ValueError("reviewed VM witness/source/protocol hash differs")
    protocol = json.loads(Path(protocol_manifest_path).read_text())
    if protocol.get("remote_commit") != FROZEN_REMOTE:
        raise ValueError("protocol remote source differs")
    controller_root = Path(protocol["controller_root"])
    for name, expected in protocol["source_sha256"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or sha(controller_root / relative) != expected
        ):
            raise ValueError("reviewed controller source inventory differs")
    state_before = sha(v1_state_path)
    state = json.loads(Path(v1_state_path).read_text())
    if (
        state.get("jobs", {}).get("sfull_inputs", {}).get("status") != "FAILED"
        or state.get("jobs", {}).get("data", {}).get("status") != "FINISHED"
        or state["jobs"]["data"].get("home") is not True
    ):
        raise ValueError("preserved V1 failed/home state required")
    result = verify(data, reference, repo, vocab_path=vocab_path, human_dir=human_dir)
    ref = json.loads(Path(reference).read_text())
    verify_witness(json.loads(Path(witness_path).read_text()), ref, result)
    if sha(v1_state_path) != state_before:
        raise ValueError("V1 state changed during supplemental verification")
    return {
        "status": "SUPPLEMENTAL_V2_INPUTS_VERIFIED_V1_GATE_FAILED",
        "validation": result,
        "v1_gate_pass": False,
        "v1_state_sha256": state_before,
        "vm_witness_sha256": VM_WITNESS_SHA,
        "vm_diagnostic_script_sha256": VM_SCRIPT_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "historical_raw_report_identity_claimed": False,
        "scientific_outcome": False,
        "no_data_or_status_mutation": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--repo")
    p.add_argument("--vocab-path")
    p.add_argument("--human-dir")
    p.add_argument(
        "--out", required=True, help="New immutable JSON report; never a V1 marker"
    )
    p.add_argument("--witness")
    p.add_argument("--diagnostic-source")
    p.add_argument("--protocol-manifest")
    p.add_argument("--v1-state")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        p.error("output already exists")
    paths = [a.witness, a.diagnostic_source, a.protocol_manifest, a.v1_state]
    if any(paths) and not all(paths):
        p.error("relay mode requires all four witness/source/protocol/state paths")
    kwargs = {"repo": a.repo, "vocab_path": a.vocab_path, "human_dir": a.human_dir}
    if all(paths):
        result = verify_relay(
            a.data,
            a.reference,
            witness_path=a.witness,
            diagnostic_source_path=a.diagnostic_source,
            protocol_manifest_path=a.protocol_manifest,
            v1_state_path=a.v1_state,
            **kwargs,
        )
    else:
        result = verify(a.data, a.reference, **kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp." + str(os.getpid()))
    with temp.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temp, out)  # atomic no-replace publication; an existing result is preserved
    temp.unlink()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
