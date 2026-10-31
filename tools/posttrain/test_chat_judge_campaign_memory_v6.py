"""Offline bounded diagnostic phase:24attempt fence, no automatic recovery, pending gate."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_campaign_memory_v6 as campaign


@pytest.fixture(autouse=True)
def synthetic_headroom(monkeypatch):
    monkeypatch.setattr(campaign, "memory_available_bytes", lambda: 2 * 1024**3)


def protocol():
    return {
        "max_new_attempts": 4252,
        "batch_calls": 24,
        "minimum_mem_available_bytes": 1024**3,
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
            "missing_rankings": 4254 - len(calls),
            "valid_api_rankings": 1786 + len(calls),
            "eligible_missing_rankings": 4252 - len(calls),
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
        len(calls) == 4252
        and (state / "STOPPED").exists()
        and not (state / "HALTED").exists()
    )
    import json

    stopped = json.loads((state / "STOPPED").read_text())
    assert stopped["eligible_judging_complete"] is True
    assert stopped["full_judging_complete"] is False
    assert stopped["missing_rankings"] == 2
    assert stopped["valid_api_rankings"] == 6038
    assert not (state / "COMPLETE").exists()
    assert campaign.campaign(p, state, dispatch, inspect) == 0 and len(calls) == 4252


def test_partial_success_then_restart_only_remaining_reviewed_attempts(tmp_path):
    p = protocol()
    calls = []

    def inspect(_):
        return [], {
            "missing_rankings": 4254 - len(calls),
            "valid_api_rankings": 1786 + len(calls),
            "eligible_missing_rankings": 4252 - len(calls),
            "attempts_since_start": len(calls),
        }

    def interrupted(*a, **kw):
        calls.extend([1] * 3)
        raise KeyboardInterrupt

    state = tmp_path / "v3"
    with pytest.raises(KeyboardInterrupt):
        campaign.campaign(p, state, interrupted, inspect)

    def dispatch(*a, **kw):
        assert a[6] == min(24, 4252 - len(calls))
        assert kw["excluded_failed_keys"] == ("fixture-preserved-refusal",)
        assert kw["excluded_unattempted_keys"] == ("fixture-paired-unattempted",)
        calls.extend([1] * a[6])
        return {"calls": a[6], "failed_rankings": 0, "pending_reservations": 0}

    assert campaign.campaign(p, state, dispatch, inspect) == 0 and len(calls) == 4252


def test_unknown_failure_halts_and_stop_file_blocks_restart(tmp_path):
    p = protocol()
    calls = []

    def inspect(_):
        return [], {
            "missing_rankings": 4254,
            "eligible_missing_rankings": 4252,
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
        ("max_new_attempts", 4254),
        ("initial_training_attempts", 1741),
        ("initial_valid_api_rankings", 1785),
        ("max_usd", 201),
        ("gate_reserve_usd", 24),
    ],
)
def test_diagnostic_limits_cannot_expand(monkeypatch, field, value):
    p = protocol()
    p.update(
        kind="FROZEN_FAILSTOP_API_MEMORY_V6_4252",
        source_sha256={},
        profile_sha256=campaign.api.digest(campaign.api.PROFILE),
        initial_training_attempts=1742,
        initial_valid_api_rankings=1786,
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


def test_low_memory_wait_never_dispatches(monkeypatch, tmp_path):
    calls = []
    readings = iter([0, 1024**3 - 1, 1024**3, 1024**3, 1024**3])
    sleeps = []
    monkeypatch.setattr(campaign, "memory_available_bytes", lambda: next(readings))

    def sleep(seconds):
        assert not calls
        import json

        assert (
            json.loads((tmp_path / "memory_admission.json").read_text())["status"]
            == "WAITING_MEMORY"
        )
        sleeps.append(seconds)

    monkeypatch.setattr(campaign.time, "sleep", sleep)

    def inspect(_):
        return [], {
            "attempts_since_start": len(calls),
            "missing_rankings": 3 - len(calls),
            "eligible_missing_rankings": 1 - len(calls),
            "valid_api_rankings": 6037 + len(calls),
        }

    def dispatch(*a, **kw):
        assert sleeps == [10, 10]
        calls.append(1)
        return {
            "calls": 1,
            "rankings_added": 1,
            "failed_rankings": 0,
            "pending_reservations": 0,
        }

    assert campaign.campaign(protocol(), tmp_path, dispatch, inspect) == 0
    assert calls == [1]


@pytest.mark.parametrize("where", ["inspect", "dispatch", "progress_publish"])
def test_oserror_phase_errno_and_existing_pending_are_preserved(
    monkeypatch, tmp_path, where
):
    import json

    dispatches = []

    def inspect(_):
        if where == "inspect":
            raise OSError(12, "PRIVATE_FIXTURE")
        return [], {
            "attempts_since_start": 0,
            "missing_rankings": 3,
            "eligible_missing_rankings": 1,
        }

    def dispatch(*a, **kw):
        dispatches.append(1)
        (tmp_path / "pending_fixture.json").write_text('{"unchanged":true}')
        raise OSError(12, "PRIVATE_FIXTURE")

    original_save = campaign.save

    def save(path, value):
        if where == "progress_publish" and Path(path).name == "progress.json":
            raise OSError(12, "PRIVATE_FIXTURE")
        original_save(path, value)

    monkeypatch.setattr(campaign, "save", save)
    assert campaign.campaign(protocol(), tmp_path, dispatch, inspect) == 2
    halt = json.loads((tmp_path / "HALTED").read_text())
    assert halt["phase"] == where and halt["errno"] == 12
    assert "PRIVATE_FIXTURE" not in (tmp_path / "HALTED").read_text()
    assert (tmp_path / "STOPPED").exists()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert campaign.campaign(protocol(), tmp_path, dispatch, inspect) == 2
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert len(dispatches) == (1 if where == "dispatch" else 0)
    if where == "dispatch":
        assert (tmp_path / "pending_fixture.json").read_text() == '{"unchanged":true}'


def test_unknown_dispatch_failure_remains_halted_without_errno_or_retry(tmp_path):
    import json

    calls = []

    def inspect(_):
        return [], {
            "attempts_since_start": 0,
            "missing_rankings": 3,
            "eligible_missing_rankings": 1,
        }

    def dispatch(*a, **kw):
        calls.append(1)
        (tmp_path / "unknown_ceiling_fixture").write_text("preserved")
        raise RuntimeError("PRIVATE_PROVIDER_TEXT")

    assert campaign.campaign(protocol(), tmp_path, dispatch, inspect) == 2
    halt = json.loads((tmp_path / "HALTED").read_text())
    assert halt["phase"] == "dispatch" and halt["errno"] is None
    assert (tmp_path / "unknown_ceiling_fixture").read_text() == "preserved"
    assert (
        campaign.campaign(protocol(), tmp_path, dispatch, inspect) == 2
        and len(calls) == 1
    )


@pytest.mark.parametrize("bad", [0, True, 1024**3 - 1])
def test_memory_threshold_cannot_be_relaxed(tmp_path, bad):
    with pytest.raises(ValueError, match="threshold"):
        campaign.wait_for_memory(tmp_path, bad, reader=lambda: 2 * 1024**3)


def test_missing_memory_measurement_fails_before_inspect(monkeypatch, tmp_path):
    import json

    def broken():
        raise OSError(5, "PRIVATE_MEMORY_READER")

    monkeypatch.setattr(campaign, "memory_available_bytes", broken)
    calls = []
    assert (
        campaign.campaign(
            protocol(),
            tmp_path,
            lambda *a, **kw: calls.append(1),
            lambda _: calls.append(1),
        )
        == 2
    )
    assert not calls
    halt = json.loads((tmp_path / "HALTED").read_text())
    assert halt["phase"] == "memory_admission_before_inspect" and halt["errno"] == 5


def test_missing_protocol_records_launch_phase_errno_without_exception_text(
    monkeypatch, tmp_path
):
    import json

    state = tmp_path / "newstate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--protocol",
            str(tmp_path / "PRIVATE_MISSING_FILE"),
            "--protocol-sha256",
            "not-a-hash",
            "--state-dir",
            str(state),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2
    halt = json.loads((state / "HALTED").read_text())
    assert halt["phase"] == "launch_validation" and halt["errno"] == 2
    assert "PRIVATE_MISSING_FILE" not in (state / "HALTED").read_text()


def test_non_numeric_errno_is_not_serialized():
    error = RuntimeError("PRIVATE_EXCEPTION")
    error.errno = "PRIVATE_ERRNO"
    assert campaign.safe_error_fields(error, "inspect") == {
        "error_type": "RuntimeError",
        "phase": "inspect",
        "errno": None,
    }
