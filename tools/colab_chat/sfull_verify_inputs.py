"""Fail closed before S-full training unless rebuilt data equals original registered S-gist data."""

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


def verify(data, reference, repo=None):
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
    vocab = Path(actual["args"]["vocab"])
    if sha(vocab) != ref["vocab_sha256"]:
        raise ValueError("vocabulary differs")
    for name, wanted in ref["human"]["files"].items():
        p = data / "human" / name
        if (
            not p.resolve().is_relative_to((data / "human").resolve())
            or sha(p) != wanted
        ):
            raise ValueError("raw human source differs:" + name)
    for name, key in [
        ("carry5_v1.jsonl", "carry5_v1_sha256"),
        ("carry12_v1.jsonl", "carry12_v1_sha256"),
    ]:
        if sha(repo / "corpus/eval" / name) != ref["human"][key]:
            raise ValueError("evaluation source differs:" + name)
    report = json.loads((data / "human_decontam.json").read_text())
    if normal(report) != normal(ref["steps"]["decontam"]):
        raise ValueError("raw decontamination report differs")
    return {
        "verified": True,
        "reference_sha256": REFERENCE_SHA,
        "ledger_sha256": sha(data / "data.ledger.json"),
        "outputs_sha256": checked,
        "vocab_sha256": ref["vocab_sha256"],
        "raw_human_sources_verified": len(ref["human"]["files"]),
        "evaluation_sources_verified": 2,
        "no_data_mutation": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = verify(a.data, a.reference)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    temp = out / "verified_inputs.json.tmp"
    temp.write_text(json.dumps(r, indent=2) + "\n")
    os.replace(temp, out / "verified_inputs.json")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
