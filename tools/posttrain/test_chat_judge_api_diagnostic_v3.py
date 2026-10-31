"""Fake transport only: same judge profile/body, unknowns retained, durable trace hashes."""

import json
from pathlib import Path
import sys
import urllib.error
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_api as original
import chat_judge_api_diagnostic_v3 as api
from test_chat_judge_api import files as api_files, requests, fake


@pytest.fixture
def files(tmp_path):
    return api_files.__wrapped__(tmp_path)


def test_exact_original_profile_request_and_cache_identity():
    assert api.PROFILE == original.PROFILE and api.profile_id() == original.profile_id()
    assert api.encoded(api.request_body("system", "user")) == original.encoded(
        original.request_body("system", "user")
    )
    assert api.API_ROOT == original.API_ROOT and api.API_VERSION == original.API_VERSION


def test_success_retains_matching_trace_hash_without_headers_body_or_key(files):
    post = fake()
    result = api.run(requests(), **files, max_calls=2, post=post)
    assert result["rankings_added"] == 2 and len(post.calls) == 2
    for row in api.rows(files["cache"]):
        receipt = row["transport_receipt"]
        path = Path(receipt["transport_trace"])
        assert api.file_hash(path) == receipt["transport_trace_sha256"]
        trace = json.loads(path.read_text())
        assert trace["binding"]["reservation_id"] == row["budget_reservation_id"]
        assert trace["binding"]["request_sha256"] == receipt["request_sha256"]
        assert (
            "sk-fixture" not in path.read_text() and "headers" not in path.read_text()
        )


def test_unknown_never_retries_or_releases_ceiling_and_binds_trace(files):
    calls = []

    def broken(key, body):
        calls.append(1)
        raise urllib.error.URLError(OSError(101, "SECRET"))

    with pytest.raises(RuntimeError):
        api.run(requests(), **files, max_calls=2, post=broken)
    assert len(calls) == 1
    budget = api.Budget(files["ledger"], api.legacy_account(files["legacy_cache"]))
    pending, paid = budget.state()
    assert len(pending) == 1 and paid == 0
    ident = next(iter(pending))
    error = json.loads(
        (files["ledger"].with_suffix(".raw") / (ident + ".error.json")).read_text()
    )
    path = Path(error["transport_trace"])
    assert (
        api.file_hash(path) == error["transport_trace_sha256"]
        and not json.loads(path.read_text())["http_write_started"]
    )
    assert (
        "SECRET" not in path.read_text()
        and error["charge"] == "unknown_reservation_retained"
    )
