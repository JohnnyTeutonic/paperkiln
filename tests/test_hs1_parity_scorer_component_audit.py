"""No full-gate conversion: only the explicitly identified post-scoring operational failure."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hs1_parity_scorer_component_audit as audit


def complete_component():
    return {
        "status": "ERROR",
        "errors": ["ember_checkout timed out"],
        "original_source_restored": True,
        "spec_delta_path_only_verified": True,
        "exact_historical_chunk_reconstruction_verified": True,
    }


def test_exact_partial_failure_boundary():
    audit.verify_partial_boundary(complete_component())


@pytest.mark.parametrize(
    "change", ["success", "other_failure", "source_not_restored", "chunk_not_proved"]
)
def test_other_status_or_missing_proof_cannot_be_component_pass(change):
    s = complete_component()
    if change == "success":
        s.update(status="SCORER_RECONSTRUCTION_COMPLETE", errors=[])
    if change == "other_failure":
        s["errors"] = ["build_original failed"]
    if change == "source_not_restored":
        s["original_source_restored"] = False
    if change == "chunk_not_proved":
        s["exact_historical_chunk_reconstruction_verified"] = False
    with pytest.raises(ValueError):
        audit.verify_partial_boundary(s)
