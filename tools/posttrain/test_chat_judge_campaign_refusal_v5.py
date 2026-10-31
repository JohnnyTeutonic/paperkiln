"""Offline bounded diagnostic phase:24attempt fence, no automatic recovery, pending gate."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_campaign_refusal_v5 as campaign


def protocol():
    return {
        "max_new_attempts": 4348,
        "batch_calls": 24,
        "cache": "unused",
        "legacy_cache": "unused",
        "rubric": "unused",
        "ledger": "unused",
        "env_file": "unused",
        "reserve_per_call_usd": 0.25,
        "max_usd": 200,
        "gate_reserve_usd": 25,
        "prior_api_ledgers": [],
        "reviewed_failure_exclusion": {
            "key": "fixture-preserved-refusal",
            "paired_unattempted_key": "fixture-paired-unattempted",
        },
    }


def test_all24_success_stop_never_continues_full_judging(tmp_path):
    p = protocol()
    calls = []

    def inspect(_):
        return [], {
            "missing_rankings": 4350 - len(calls),
            "eligible_missing_rankings": 4348 - len(calls),
            "attempts_since_start": len(calls),
        }

    def dispatch(*a, **kw):
        assert kw["excluded_failed_keys"] == ("fixture-preserved-refusal",)
        assert kw["excluded_unattempted_keys"] == ("fixture-paired-unattempted",)
        calls.extend([1] * a[6])
        return {"calls": a[6], "failed_rankings": 0, "pending_reservations": 0}

    state = tmp_path / "v3"
    assert campaign.campaign(p, state, dispatch, inspect) == 0
    assert (
        len(calls) == 4348
        and (state / "STOPPED").exists()
        and not (state / "HALTED").exists()
    )
    import json

    stopped = json.loads((state / "STOPPED").read_text())
    assert stopped["eligible_judging_complete"] is True
    assert stopped["full_judging_complete"] is False
    assert stopped["missing_rankings"] == 2
    assert not (state / "COMPLETE").exists()
    assert campaign.campaign(p, state, dispatch, inspect) == 0 and len(calls) == 4348


def test_partial_success_then_restart_only_remaining_reviewed_attempts(tmp_path):
    p = protocol()
    calls = []

    def inspect(_):
        return [], {
            "missing_rankings": 4350 - len(calls),
            "eligible_missing_rankings": 4348 - len(calls),
            "attempts_since_start": len(calls),
        }

    def interrupted(*a, **kw):
        calls.extend([1] * 3)
        raise KeyboardInterrupt

    state = tmp_path / "v3"
    with pytest.raises(KeyboardInterrupt):
        campaign.campaign(p, state, interrupted, inspect)

    def dispatch(*a, **kw):
        assert a[6] == min(24, 4348 - len(calls))
        assert kw["excluded_failed_keys"] == ("fixture-preserved-refusal",)
        assert kw["excluded_unattempted_keys"] == ("fixture-paired-unattempted",)
        calls.extend([1] * a[6])
        return {"calls": a[6], "failed_rankings": 0, "pending_reservations": 0}

    assert campaign.campaign(p, state, dispatch, inspect) == 0 and len(calls) == 4348


def test_unknown_failure_halts_and_stop_file_blocks_restart(tmp_path):
    p = protocol()
    calls = []

    def inspect(_):
        return [], {
            "missing_rankings": 4350,
            "eligible_missing_rankings": 4348,
            "attempts_since_start": 0,
        }

    def fail(*a, **kw):
        calls.append(1)
        raise RuntimeError("fixture transport failure")

    state = tmp_path / "v3"
    assert campaign.campaign(p, state, fail, inspect) == 2
    assert (state / "HALTED").exists() and (state / "STOPPED").exists()
    assert campaign.campaign(p, state, fail, inspect) == 2 and len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_new_attempts", 4350),
        ("initial_training_attempts", 1645),
        ("initial_valid_api_rankings", 1689),
        ("max_usd", 201),
        ("gate_reserve_usd", 24),
    ],
)
def test_diagnostic_limits_cannot_expand(monkeypatch, field, value):
    p = protocol()
    p.update(
        kind="FROZEN_FAILSTOP_API_REFUSAL_V5_4348",
        source_sha256={},
        profile_sha256=campaign.api.digest(campaign.api.PROFILE),
        initial_training_attempts=1646,
        initial_valid_api_rankings=1690,
    )
    p[field] = value
    monkeypatch.setattr(campaign, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="limits"):
        campaign.inspect(p)


def test_bad_launch_fenced_without_overwriting_prior_negative(tmp_path):
    import subprocess

    path = tmp_path / "bad_protocol.json"
    path.write_text("{}")
    state = tmp_path / "v3"
    cmd = [
        sys.executable,
        str(Path(campaign.__file__)),
        "--protocol",
        str(path),
        "--protocol-sha256",
        "bad",
        "--state-dir",
        str(state),
    ]
    result = subprocess.run(cmd, capture_output=True)
    assert (
        result.returncode == 2
        and (state / "HALTED").exists()
        and (state / "STOPPED").exists()
    )
    before = {p.name: p.read_bytes() for p in state.iterdir()}
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    assert {p.name: p.read_bytes() for p in state.iterdir()} == before


def pair_fixture():
    requests = [{"prompt": "fixture", "candidates": ["one", "two"], "uri": "fixture"}]
    version = "fixture-version"
    planned = [
        (
            o,
            campaign.api.judge.key_of(
                r["prompt"], r["candidates"], o, campaign.api.profile_id(), version
            ),
        )
        for r, o in campaign.api.judge.jobs(requests)
    ]
    gate = {
        "candidate_case_sha256": campaign.api.digest(requests[0]),
        "planned_orders": [o for o, _ in planned],
        "key": planned[0][1],
        "paired_unattempted_key": planned[1][1],
    }
    return {"reviewed_failure_exclusion": gate}, requests, version


def test_pair_is_derived_from_fixed_request_and_planner():
    p, requests, version = pair_fixture()
    campaign.verify_reviewed_pair(p, requests, version, [], set())


@pytest.mark.parametrize("change", ["case", "key", "orders", "attempted", "ranked"])
def test_pair_changes_or_previously_attempted_reverse_refused(change):
    p, requests, version = pair_fixture()
    gate = p["reviewed_failure_exclusion"]
    history, done = [], set()
    if change == "case":
        gate["candidate_case_sha256"] = "wrong"
    if change == "key":
        gate["paired_unattempted_key"] = "foreign"
    if change == "orders":
        gate["planned_orders"] = list(reversed(gate["planned_orders"]))
    if change == "attempted":
        history = [{"event": "reserve", "key": gate["paired_unattempted_key"]}]
    if change == "ranked":
        done = {gate["paired_unattempted_key"]}
    with pytest.raises(ValueError):
        campaign.verify_reviewed_pair(p, requests, version, history, done)
