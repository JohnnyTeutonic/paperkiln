"""A pinned download whose host's TLS certificate has expired is fetched without verification and accepted only
at the pinned sha256 (dl.fbaipublicfiles.com, 13 Nov 2026). An unpinned one, or any other failure, is not."""
import hashlib
import io
import os
import ssl
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus", "fetchers"))
import human_dialogue as H  # noqa: E402

BODY = b"pinned archive bytes"
GOOD = hashlib.sha256(BODY).hexdigest()


class Resp(io.BytesIO):
    headers = {"Content-Length": str(len(BODY))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def expired():
    return urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed: certificate has expired"))


def fake_urlopen(calls):
    def urlopen(req, timeout=None, context=None):
        calls.append(context)
        if context is None or context.verify_mode != ssl.CERT_NONE:
            raise expired()
        return Resp(BODY)
    return urlopen


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(H.C.time, "sleep", lambda s: None)


def test_a_pinned_download_with_an_expired_certificate_is_fetched_unverified_and_checked(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(H.urllib.request, "urlopen", fake_urlopen(calls))
    path = H.fetch_url("https://example.invalid/a.tgz", str(tmp_path), GOOD)
    assert open(path, "rb").read() == BODY
    assert calls[0] is None and calls[-1].verify_mode == ssl.CERT_NONE


def test_the_pinned_hash_still_decides(tmp_path, monkeypatch):
    monkeypatch.setattr(H.urllib.request, "urlopen", fake_urlopen([]))
    with pytest.raises(H.C.HashMismatch):
        H.fetch_url("https://example.invalid/b.tgz", str(tmp_path), "0" * 64)


def test_an_unpinned_download_is_never_fetched_unverified(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(H.urllib.request, "urlopen", fake_urlopen(calls))
    with pytest.raises(urllib.error.URLError):
        H.fetch_url("https://example.invalid/c.tgz", str(tmp_path), None)
    assert all(c is None for c in calls)


def test_a_transient_failure_keeps_the_six_tries_of_the_retry_policy(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(context)
        raise TimeoutError("timed out")
    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    with pytest.raises(TimeoutError):
        H.fetch_url("https://example.invalid/e.tgz", str(tmp_path), GOOD)
    assert len(calls) == len(H.C.RETRY_WAITS) + 1


def test_a_final_http_error_is_tried_once(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(context)
        raise urllib.error.HTTPError("https://example.invalid/f.tgz", 404, "Not Found", {}, None)
    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        H.fetch_url("https://example.invalid/f.tgz", str(tmp_path), GOOD)
    assert len(calls) == 1


def test_an_expired_certificate_after_a_transient_failure_still_falls_back(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(context)
        if len(calls) == 1:
            raise TimeoutError("timed out")
        if context is None:
            raise expired()
        return Resp(BODY)
    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    path = H.fetch_url("https://example.invalid/g.tgz", str(tmp_path), GOOD)
    assert open(path, "rb").read() == BODY
    assert calls[-1].verify_mode == ssl.CERT_NONE


def test_another_tls_failure_is_not_bypassed(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(context)
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed certificate"))
    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.URLError):
        H.fetch_url("https://example.invalid/d.tgz", str(tmp_path), GOOD)
    assert all(c is None for c in calls)
