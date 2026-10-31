"""No network: ordered fallback, write fence, immutability and secret-free diagnostics."""

import errno
import json
from pathlib import Path
import socket
import ssl
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_transport_diagnostic as d


class Sock:
    def __init__(self, failure=None):
        self.failure = failure
        self.closed = False
        self.sent = []

    def settimeout(self, value):
        self.timeout = value

    def bind(self, value):
        self.bound = value

    def setsockopt(self, *a):
        pass

    def close(self):
        self.closed = True

    def connect(self, addr):
        self.addr = addr
        if self.failure:
            raise self.failure

    def sendall(self, data):
        self.sent.append(data)


def candidates(*args):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
    ]


def test_every_endpoint_failure_preserved_and_no_http(tmp_path, monkeypatch):
    trace = d.Trace(tmp_path / "trace.json")
    socks = [
        Sock(ConnectionRefusedError(errno.ECONNREFUSED, "SECRET")),
        Sock(OSError(errno.ENETUNREACH, "SECRET")),
    ]
    queue = iter(socks)
    monkeypatch.setattr(d.socket, "getaddrinfo", candidates)
    monkeypatch.setattr(d.socket, "socket", lambda *a: next(queue))
    with pytest.raises(d.EndpointConnectionError) as error:
        trace.create_connection(("api.anthropic.com", 443), 3)
    assert error.value.errno == 101
    saved = json.loads(trace.path.read_text())
    assert not saved["http_write_started"]
    assert [a["failure"]["errno"] for a in saved["attempts"]] == [111, 101]
    assert all(s.closed for s in socks) and "SECRET" not in trace.path.read_text()


def test_first_success_skips_extra_address_and_preserves_bind_timeout(monkeypatch):
    trace = d.Trace()
    s = Sock()
    calls = []
    monkeypatch.setattr(d.socket, "getaddrinfo", candidates)

    def factory(*a):
        calls.append(a)
        return s

    monkeypatch.setattr(d.socket, "socket", factory)
    assert trace.create_connection(("api.anthropic.com", 443), 4, ("127.0.0.1", 0)) is s
    assert len(calls) == 1 and s.timeout == 4 and s.bound == ("127.0.0.1", 0)


def test_fallback_success_is_same_ordered_socket_semantics(monkeypatch):
    trace = d.Trace()
    first = Sock(OSError(111, "x"))
    second = Sock()
    queue = iter((first, second))
    monkeypatch.setattr(d.socket, "getaddrinfo", candidates)
    monkeypatch.setattr(d.socket, "socket", lambda *a: next(queue))
    assert trace.create_connection(("api.anthropic.com", 443)) is second
    assert first.closed and not second.closed and len(trace.data["attempts"]) == 2


def test_dns_failure_is_pre_http_and_secret_free(monkeypatch):
    trace = d.Trace()

    def broken(*a):
        raise socket.gaierror(-2, "SECRET")

    monkeypatch.setattr(d.socket, "getaddrinfo", broken)
    with pytest.raises(socket.gaierror):
        trace.create_connection(("api.anthropic.com", 443))
    assert (
        trace.data["stage"] == "dns_failed_before_http_write"
        and not trace.data["http_write_started"]
    )
    assert "SECRET" not in trace.encoded().decode()


def test_http_write_fence_durable_before_send_and_never_logs_bytes(
    tmp_path, monkeypatch
):
    trace = d.Trace(tmp_path / "trace.json")
    conn = d.TrackingHTTPSConnection("api.anthropic.com", trace=trace)
    sock = Sock()

    def connect():
        conn.sock = sock
        trace.stage("tls_ready")

    monkeypatch.setattr(conn, "connect", connect)

    def send(data):
        assert json.loads(trace.path.read_text())["http_write_started"] is True
        sock.sent.append(data)

    monkeypatch.setattr(sock, "sendall", send)
    conn.send(b"x-api-key:SECRET\r\n\r\nPRIVATE_BODY")
    assert (
        sock.sent
        and "SECRET" not in trace.path.read_text()
        and "PRIVATE_BODY" not in trace.path.read_text()
    )


def test_connect_failure_cannot_reach_http_send(tmp_path, monkeypatch):
    trace = d.Trace(tmp_path / "trace.json")
    conn = d.TrackingHTTPSConnection("api.anthropic.com", trace=trace)

    def fail():
        raise OSError(101, "SECRET")

    monkeypatch.setattr(conn, "connect", fail)
    with pytest.raises(OSError):
        conn.send(b"SECRET")
    assert not json.loads(trace.path.read_text())["http_write_started"]


def test_existing_diagnostic_artifact_never_overwritten(tmp_path):
    path = tmp_path / "trace.json"
    path.write_bytes(b"original negative")
    with pytest.raises(FileExistsError):
        d.Trace(path)
    assert path.read_bytes() == b"original negative"


def test_tls_failure_is_pre_http(tmp_path, monkeypatch):
    trace = d.Trace(tmp_path / "trace.json")
    conn = d.TrackingHTTPSConnection("api.anthropic.com", trace=trace)
    sock = Sock()

    def connect(*a, **kw):
        trace.stage("tcp_connected_before_tls_or_tunnel")
        return sock

    monkeypatch.setattr(conn, "_create_connection", connect)

    class Context:
        def wrap_socket(self, *a, **kw):
            raise ssl.SSLError("SECRET")

    conn._context = Context()
    with pytest.raises(ssl.SSLError):
        conn.send(b"PRIVATE_BODY")
    data = json.loads(trace.path.read_text())
    assert (
        not data["http_write_started"] and data["connect_failure"]["type"] == "SSLError"
    )
    assert "SECRET" not in trace.path.read_text()


@pytest.mark.parametrize("failure", ["file_fsync", "parent_fsync"])
def test_durability_failure_prevents_every_http_byte(tmp_path, monkeypatch, failure):
    trace = d.Trace(tmp_path / "trace.json")
    conn = d.TrackingHTTPSConnection("api.anthropic.com", trace=trace)
    sock = Sock()
    conn.sock = sock

    def fail(*args):
        raise OSError(errno.EIO, "fixture durability failure")

    if failure == "file_fsync":
        monkeypatch.setattr(d.os, "fsync", fail)
    else:
        monkeypatch.setattr(trace, "fsync_parent", fail)
    with pytest.raises(OSError):
        conn.send(b"x-api-key:SECRET")
    assert sock.sent == []
    assert "SECRET" not in trace.path.read_text()


def test_actual_proxy_connect_is_fenced_before_any_connect_bytes(tmp_path, monkeypatch):
    import io

    trace = d.Trace(tmp_path / "trace.json")
    conn = d.TrackingHTTPSConnection("proxy.fixture", trace=trace)
    conn.set_tunnel("api.anthropic.com", 443, headers={"Proxy-Authorization": "SECRET"})
    sock = Sock()
    sock.makefile = lambda *a, **kw: io.BytesIO(
        b"HTTP/1.0 200 Connection established\r\n\r\n"
    )

    def send(data):
        assert json.loads(trace.path.read_text())["http_write_started"] is True
        sock.sent.append(data)

    sock.sendall = send
    monkeypatch.setattr(conn, "_create_connection", lambda *a, **kw: sock)

    class Context:
        def wrap_socket(self, value, server_hostname):
            assert server_hostname == "api.anthropic.com"
            return value

    conn._context = Context()
    conn.connect()
    assert sock.sent and sock.sent[0].startswith(b"CONNECT api.anthropic.com:443")
    assert trace.data["http_write_started"] and "SECRET" not in trace.path.read_text()
