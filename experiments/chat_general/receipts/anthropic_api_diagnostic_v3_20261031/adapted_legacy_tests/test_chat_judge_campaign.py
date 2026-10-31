"""Offline durability tests. Fake HTTP only; no credentials or paid calls."""

import hashlib
import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_campaign as campaign  # noqa: E402
from test_chat_judge_api import files as api_files, requests, fake, response  # noqa: E402,F401

api = campaign.api


@pytest.fixture
def files(tmp_path):
    return api_files.__wrapped__(tmp_path)


def protocol(files):
    api.run(requests(), **files, max_calls=0)
    req = files["cache"].parent / "requests.jsonl"
    for r in requests():
        api.append(req, r)
    raw = files["ledger"].read_bytes()
    return dict(
        kind="FROZEN_FAILSTOP_API_CAMPAIGN_V1",
        source_sha256=campaign.source_hashes(),
        profile_sha256=api.digest(api.PROFILE),
        max_new_attempts=5993,
        max_usd=200,
        gate_reserve_usd=25,
        reserve_per_call_usd=0.25,
        batch_calls=24,
        cache=str(files["cache"]),
        ledger=str(files["ledger"]),
        requests=str(req),
        rubric=str(files["rubric"]),
        legacy_cache=str(files["legacy_cache"]),
        env_file=str(files["env_file"]),
        prior_api_ledgers=[],
        binding_sha256=api.file_hash(str(files["cache"]) + ".binding.json"),
        ledger_prefix_bytes=len(raw),
        ledger_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        requests_sha256=api.digest(requests()),
        rubric_sha256=api.file_hash(files["rubric"]),
        initial_training_attempts=0,
    )


def test_clean_completion_and_restart_skip_paid_cache(files, tmp_path, monkeypatch):
    p = protocol(files)
    post = fake()
    monkeypatch.setattr(api, "http_post", post)
    state = tmp_path / "state"
    assert campaign.campaign(p, state) == 0
    assert len(post.calls) == 2 and (state / "COMPLETE").exists()
    assert campaign.campaign(p, state) == 0 and len(post.calls) == 2
    assert (
        json.loads((state / "progress.json").read_text())["attempts_since_start"] == 2
    )


def test_unknown_charge_halts_and_watchdog_restart_cannot_retry(
    files, tmp_path, monkeypatch
):
    p = protocol(files)
    calls = []

    def broken(key, body, **_diagnostic_trace):
        calls.append(body)
        raise urllib.error.URLError("fixture")

    monkeypatch.setattr(api, "http_post", broken)
    state = tmp_path / "state"
    assert campaign.campaign(p, state) == 2 and len(calls) == 1
    assert (state / "HALTED").exists()
    assert campaign.campaign(p, state) == 2 and len(calls) == 1
    b = api.Budget(files["ledger"], api.legacy_account(files["legacy_cache"]))
    pending, paid = b.state()
    assert len(pending) == 1 and paid == 0
    assert next(iter(pending.values()))["reserve_usd"] == 0.25


def test_process_death_after_settled_checkpoint_resumes_without_duplicate(
    files, tmp_path, monkeypatch
):
    p = protocol(files)
    post = fake()
    monkeypatch.setattr(api, "http_post", post)

    def interrupted(*args, **kwargs):
        a = list(args)
        a[6] = 1
        api.run(*a, **kwargs)
        raise KeyboardInterrupt

    state = tmp_path / "state"
    with pytest.raises(KeyboardInterrupt):
        campaign.campaign(p, state, dispatch=interrupted)
    assert not (state / "HALTED").exists() and len(post.calls) == 1
    assert campaign.campaign(p, state) == 0 and len(post.calls) == 2


def test_failed_parse_is_paid_preserved_and_not_retried(files, tmp_path, monkeypatch):
    p = protocol(files)
    post = fake(response(content=[]))
    monkeypatch.setattr(api, "http_post", post)
    state = tmp_path / "state"
    assert campaign.campaign(p, state) == 2
    assert len(api.rows(str(files["cache"]) + ".failures.jsonl")) == 1
    assert campaign.campaign(p, state) == 2 and len(post.calls) == 1


@pytest.mark.parametrize(
    "field", ["source", "requests", "rubric", "binding", "ledger", "legacy", "limit"]
)
def test_provenance_drift_stops_before_dispatch(files, tmp_path, monkeypatch, field):
    p = protocol(files)
    post = fake()
    monkeypatch.setattr(api, "http_post", post)
    if field == "source":
        p["source_sha256"] = {}
    elif field == "requests":
        api.append(p["requests"], requests()[0])
    elif field == "rubric":
        files["rubric"].write_text("changed")
    elif field == "binding":
        Path(p["cache"] + ".binding.json").write_text("{}")
    elif field == "ledger":
        files["ledger"].write_text("")
    elif field == "legacy":
        api.append(files["legacy_cache"], {"cost_usd": 1})
    else:
        p["max_usd"] = 201
    assert campaign.campaign(p, tmp_path / "state") == 2 and not post.calls


def test_state_identity_refuses_different_campaign(files, tmp_path):
    p = protocol(files)
    state = tmp_path / "state"
    state.mkdir()
    campaign.save(state / "campaign_identity.json", {"protocol_sha256": "other"})
    assert campaign.campaign(p, state) == 2
    assert json.loads((state / "HALTED").read_text())["requires_review"]


def test_exhausted_bound_does_not_dispatch(files, tmp_path):
    p = protocol(files)

    def checked(_):
        return requests(), {"missing_rankings": 1, "attempts_since_start": 5993}

    def forbidden(*a, **kw):
        raise AssertionError("must never dispatch")

    assert (
        campaign.campaign(p, tmp_path / "state", dispatch=forbidden, inspector=checked)
        == 2
    )


def test_error_text_and_stderr_never_copied(files, tmp_path):
    p = protocol(files)
    secret = "fixture-secret-must-not-be-logged"

    def broken(*args, **kwargs):
        print(secret, file=sys.stderr)
        raise ValueError(secret)

    state = tmp_path / "state"
    assert campaign.campaign(p, state, dispatch=broken) == 2
    for path in state.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert json.loads((state / "HALTED").read_text())[
        "stderr_present_content_suppressed"
    ]


def test_cli_bad_protocol_hash_creates_watchdog_stop_file(tmp_path):
    import subprocess

    state = tmp_path / "state"
    p = tmp_path / "protocol.json"
    p.write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(campaign.__file__)),
            "--protocol",
            str(p),
            "--protocol-sha256",
            "bad",
            "--state-dir",
            str(state),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (state / "HALTED").exists()
    assert not result.stderr
