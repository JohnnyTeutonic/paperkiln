"""Check registered extension provenance before the frozen five-seed analysis.

No scoring or scientific rule changes. The registration's 28 Oct implementation
note requires ledger agreement and copying the two 20M post summaries into E5.
    python check_extension_results.py --art C:/ml_artifacts/tiny_chat --out RECEIPT.json
    python check_extension_results.py --art ... --out NEW.json --merge-20m
Exit 2 means inputs pending; exit 1 means invalid inputs; exit 0 means ready.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRATION = HERE.parent / "PREREGISTRATION.md"
FROZEN = HERE.parent / "analyze_x5.py"
spec = importlib.util.spec_from_file_location("registered_x5_ledger", FROZEN)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
CELLS = (
    ("20m", "e5", "e5_s35_r{seed}", "post_e5_s35_r{seed}", "s35"),
    ("60m", "e1_60m_L8", "e1_60m_L8_r{seed}", "post_e1_60m_L8_r{seed}", "L8"),
    ("150m", "e1_150m_r1", "e1_150m_r{seed}", "post_e1_150m_L8_r{seed}", "L8"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(art):
    art = Path(art)
    result = {"kind": "REGISTERED_EXTENSION_INPUT_CHECK_NOT_A_SCIENTIFIC_RESULT",
              "registration_sha256": sha(REGISTRATION), "auditor_sha256": sha(Path(__file__)),
              "ledger_normalizer_sha256": sha(FROZEN), "pending": [], "errors": [],
              "ledgers": [], "summaries": []}
    for size, reference, campaign, post, cell in CELLS:
        ref = art / reference / "data/data.ledger.json"
        if not ref.is_file():
            result["errors"].append(f"missing reference ledger: {ref}")
            continue
        wanted = frozen.strip_ledger(jload(ref))
        for seed in (4, 5):
            folder = art / campaign.format(seed=seed)
            ledger = folder / "data/data.ledger.json"
            if not ledger.is_file():
                result["pending"].append(str(ledger))
            else:
                got = frozen.strip_ledger(jload(ledger))
                equal = got == wanted
                result["ledgers"].append({"size": size, "seed": seed, "path": str(ledger),
                    "sha256": sha(ledger), "reference": str(ref), "reference_sha256": sha(ref),
                    "matches": equal})
                if not equal:
                    result["errors"].append(f"{size} seed {seed}: data ledger differs from reference")
            summary = folder / post.format(seed=seed) / "summary.json"
            if not summary.is_file():
                result["pending"].append(str(summary))
                continue
            s = jload(summary)
            primary = s.get("primary")
            valid = (type(s.get("seed")) is int and s["seed"] == seed and s.get("cell") == cell
                     and type(primary) in (int, float) and math.isfinite(primary))
            result["summaries"].append({"size": size, "seed": seed, "path": str(summary),
                                      "sha256": sha(summary), "valid": valid})
            if not valid:
                result["errors"].append(f"{size} seed {seed}: wrong cell/seed or incomplete primary")
    result["ready"] = not result["pending"] and not result["errors"]
    return result


def merge_20m(art, receipt):
    """Only the two tiny reports are copied, as explicitly required by registration.

    Model/corpus/checkpoint files retain one home. Never replace a different report.
    All source/destination checks precede the first write.
    """
    if not receipt["ready"]:
        raise ValueError("extension inputs are pending or invalid; no summaries copied")
    copies = []
    for item in receipt["summaries"]:
        if item["size"] != "20m":
            continue
        source = Path(item["path"])
        target = Path(art) / "e5" / source.parent.name / "summary.json"
        if sha(source) != item["sha256"]:
            raise ValueError(f"source changed after audit: {source}")
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"refusing to replace a different summary: {target}")
        copies.append((source, target))
    if len(copies) != 2:
        raise ValueError("receipt must contain both registered 20M extension reports")
    for source, target in copies:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.incoming")
            temporary.write_bytes(source.read_bytes())
            temporary.replace(target)
    return [str(target) for _, target in copies]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-20m", action="store_true")
    args = ap.parse_args(argv)
    output = Path(args.out)
    if output.exists():
        ap.error("receipt output already exists; use a new filename")
    receipt = audit(args.art)
    if args.merge_20m:
        receipt["merged_20m"] = merge_20m(args.art, receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("ready", "pending", "errors")}, indent=1))
    return 1 if receipt["errors"] else (0 if receipt["ready"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
