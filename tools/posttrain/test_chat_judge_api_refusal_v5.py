"""Offline refusal exclusions: never rejudge; unchanged provider profile and accounting."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_api_refusal_v5 as api
import chat_judge_api_diagnostic_v3 as original
from test_chat_judge_api import files as old_files, requests, fake, response


@pytest.fixture
def files(tmp_path):
    return old_files.__wrapped__(tmp_path)


def refuse(files):
    post = fake(
        response(
            stop_reason="refusal",
            content=[],
            usage={"input_tokens": 1000, "output_tokens": 0},
        )
    )
    r = api.run(requests(), **files, max_calls=2, post=post, stop_on_failure=True)
    assert r["calls"] == r["failed_rankings"] == 1 and r["rankings_added"] == 0
    assert r["pending_reservations"] == 0
    return api.rows(str(files["cache"]) + ".failures.jsonl")[0]


def test_profile_body_and_pricing_unchanged():
    assert api.PROFILE == original.PROFILE
    assert api.encoded(api.request_body("s", "u")) == original.encoded(
        original.request_body("s", "u")
    )
    assert api.profile_id() == original.profile_id()


def test_exact_failed_key_is_not_rejudged_other_order_is_ranked(files):
    f = refuse(files)
    before = Path(str(files["cache"]) + ".failures.jsonl").read_bytes()
    post = fake()
    r = api.run(
        requests(),
        **files,
        max_calls=2,
        post=post,
        stop_on_failure=True,
        excluded_failed_keys=(f["key"],),
    )
    assert r["calls"] == r["rankings_added"] == 1 and r["pending_reservations"] == 0
    assert len(post.calls) == 1
    assert api.rows(files["cache"])[0]["key"] != f["key"]
    assert Path(str(files["cache"]) + ".failures.jsonl").read_bytes() == before
    assert (
        api.run(
            requests(),
            **files,
            max_calls=2,
            post=post,
            excluded_failed_keys=(f["key"],),
        )["calls"]
        == 0
    )


@pytest.mark.parametrize("bad", [(), ("wrong-key",)])
def test_unreviewed_or_wrong_exclusion_refuses_before_http(files, bad):
    refuse(files)
    post = fake()
    before = files["ledger"].read_bytes()
    with pytest.raises(ValueError, match="exclusions"):
        api.run(requests(), **files, max_calls=2, post=post, excluded_failed_keys=bad)
    assert not post.calls and files["ledger"].read_bytes() == before


def test_new_refusal_cannot_be_silently_excluded_on_restart(files):
    f = refuse(files)
    post = fake(
        response(
            stop_reason="refusal",
            content=[],
            usage={"input_tokens": 1000, "output_tokens": 0},
        )
    )
    r = api.run(
        requests(),
        **files,
        max_calls=2,
        post=post,
        stop_on_failure=True,
        excluded_failed_keys=(f["key"],),
    )
    assert r["calls"] == r["failed_rankings"] == 1 and r["pending_reservations"] == 0
    before = files["ledger"].read_bytes()
    with pytest.raises(ValueError, match="exclusions"):
        api.run(
            requests(),
            **files,
            max_calls=2,
            post=post,
            excluded_failed_keys=(f["key"],),
        )
    assert len(post.calls) == 1 and files["ledger"].read_bytes() == before
