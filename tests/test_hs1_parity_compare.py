"""Offline report-join contracts; synthetic GPU rows are never scientific outcomes."""

import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location(
    "compare", ROOT / "tools/hs1_parity_compare.py"
)
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)


def fixtures():
    receipt = (
        ROOT / "experiments/tiny_chat_long_2026-10/receipts/parity_diagnosis_20261031"
    )
    cpu = json.loads((receipt / "cpu_per_position.json").read_text())
    fp64 = json.loads((receipt / "fp64_report.json").read_text())
    gpu = {
        "status": "RECONSTRUCTION_AUDITED",
        "generation": [],
        "generation_skip_reason": "synthetic test only",
        "rows": [
            {
                "key": r["key"],
                "positions": [
                    {
                        "position_zero_based": p["position_zero_based"],
                        "saved_generation_id": p["saved_chosen_id"],
                        "matches_saved_generation": True,
                    }
                    for p in r["positions"]
                ],
            }
            for r in cpu["rows"]
        ],
    }
    return cpu, fp64, gpu


def test_join_preserves_all_positions_and_historical_unknown():
    report = g.compare(*fixtures())
    assert report["positions"] == 89
    assert sum(len(r["positions"]) for r in report["rows"]) == 89
    assert all(
        p["original_failed_position_unknown"]
        for r in report["rows"]
        for p in r["positions"]
    )
    assert sum(len(r["new_cpu_full_disagreements"]) for r in report["rows"]) == 2


@pytest.mark.parametrize("mutation", ["status", "partial", "chosen"])
def test_reject_operational_or_identity_mismatch(mutation):
    cpu, fp64, gpu = fixtures()
    if mutation == "status":
        gpu["status"] = "ERROR"
    if mutation == "partial":
        gpu["rows"].pop()
    if mutation == "chosen":
        gpu["rows"][0]["positions"][0]["saved_generation_id"] += 1
    with pytest.raises(ValueError):
        g.compare(cpu, fp64, gpu)
