"""Offline recovery boundaries; fake dispatch only, never loads credentials or HTTP."""

import ast
import hashlib
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_campaign_v2 as campaign


def minimal():
    return dict(
        kind="FROZEN_FAILSTOP_API_CAMPAIGN_V2_REVIEWED_RECOVERY",
        source_sha256={},
        profile_sha256=campaign.api.digest(campaign.api.PROFILE),
        max_new_attempts=5514,
        initial_training_attempts=480,
        initial_valid_api_rankings=526,
        max_usd=200,
        gate_reserve_usd=25,
        reserve_per_call_usd=0.25,
        batch_calls=24,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_new_attempts", 5515),
        ("initial_training_attempts", 479),
        ("initial_valid_api_rankings", 525),
        ("max_usd", 201),
        ("gate_reserve_usd", 24),
        ("reserve_per_call_usd", 0.24),
        ("batch_calls", 25),
    ],
)
def test_changed_recovery_limits_refuse_before_artifact_or_dispatch(
    monkeypatch, field, value
):
    p = minimal()
    p[field] = value
    monkeypatch.setattr(campaign, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="limits"):
        campaign.inspect(p)


def anchors(tmp_path):
    a = {}
    for name in ("old_protocol", "old_halt", "retention_receipt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        a[name] = {"path": str(path), "sha256": campaign.api.file_hash(path)}
    cache = tmp_path / "cache"
    cache.write_bytes(b"old-success\n")
    a.update(
        resolved_pending_id="aafafd1531f74959936e5422388ff30e",
        cache_prefix_bytes=len(cache.read_bytes()),
        cache_prefix_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
    )
    return {"recovery_anchors": a, "cache": str(cache)}


def test_prior_success_cache_may_only_append(tmp_path):
    p = anchors(tmp_path)
    campaign.verify_recovery_anchors(p)
    with Path(p["cache"]).open("ab") as f:
        f.write(b"new-success\n")
    campaign.verify_recovery_anchors(p)
    Path(p["cache"]).write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        campaign.verify_recovery_anchors(p)


@pytest.mark.parametrize("name", ["old_protocol", "old_halt", "retention_receipt"])
def test_original_anchor_mutation_refused(tmp_path, name):
    p = anchors(tmp_path)
    Path(p["recovery_anchors"][name]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        campaign.verify_recovery_anchors(p)


def test_old_durable_campaign_and_launch_bodies_unchanged():
    old = ast.parse(
        Path(campaign.__file__).with_name("chat_judge_campaign.py").read_text()
    )
    new = ast.parse(Path(campaign.__file__).read_text())
    for name in ("campaign", "main"):
        a = next(
            n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        b = next(
            n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert ast.dump(a) == ast.dump(b)


def test_reviewed_phase_completion_and_no_duplicate_dispatch(tmp_path):
    p = minimal()
    p.update(
        cache="unused",
        legacy_cache="unused",
        rubric="unused",
        ledger="unused",
        env_file="unused",
        prior_api_ledgers=[],
    )
    calls = []

    def inspect(_):
        return [], {
            "missing_rankings": 0 if calls else 1,
            "attempts_since_start": len(calls),
        }

    def dispatch(*a, **kw):
        calls.append(1)
        return {"calls": 1, "failed_rankings": 0, "pending_reservations": 0}

    state = tmp_path / "v2"
    assert campaign.campaign(p, state, dispatch, inspect) == 0
    assert campaign.campaign(p, state, dispatch, inspect) == 0 and len(calls) == 1


def test_new_unknown_halts_and_restart_cannot_retry(tmp_path):
    p = minimal()
    p.update(
        cache="unused",
        legacy_cache="unused",
        rubric="unused",
        ledger="unused",
        env_file="unused",
        prior_api_ledgers=[],
    )
    calls = []

    def inspect(_):
        return [], {"missing_rankings": 1, "attempts_since_start": 0}

    def dispatch(*a, **kw):
        calls.append(1)
        raise RuntimeError("synthetic transport failure")

    state = tmp_path / "v2"
    assert campaign.campaign(p, state, dispatch, inspect) == 2
    assert campaign.campaign(p, state, dispatch, inspect) == 2 and len(calls) == 1
    assert (state / "HALTED").exists()


def test_watchdog_stop_file_is_exact_new_state_halt():
    import json

    path = (
        Path('/home/jonat/pk_api_transport_diagnostic_20261031')
        / "experiments/chat_general/receipts/anthropic_api_recovery_v2_20261031/watchdog_proposal.json"
    )
    entry = json.loads(path.read_text())["entry"]
    assert entry["stop_file"] == entry["state_dir"] + "/HALTED"
    assert (
        entry["state_dir"] == "/mnt/c/ml_artifacts/pref_api_20261031/full_campaign_v2"
    )
    assert (
        entry["stop_file"]
        != "/mnt/c/ml_artifacts/pref_api_20261031/full_campaign/HALTED"
    )
