"""Unused diagnostic HTTPS adapter: same stdlib fallback, credential-free durable stages.

No caller in the frozen API phases imports this module. It does not dispatch requests
or authorise a retry. A future integration must bind its source and per-reservation
trace before HTTP; absence of HTTP-write evidence never rewrites old charge history.
"""

import http.client
import json
import os
import re
from pathlib import Path
import socket
import urllib.request


class EndpointConnectionError(OSError):
    """All original address candidates failed; detailed safe errors live in trace."""


class Trace:
    def __init__(self, path=None, binding=None):
        self.path = Path(path) if path is not None else None
        self.data = {
            "schema": 1,
            "http_write_started": False,
            "stage": "created",
            "dns": [],
            "attempts": [],
        }
        if binding is not None:
            if (
                set(binding) != {"reservation_id", "request_sha256"}
                or not re.fullmatch(r"[0-9a-f]{32}", binding["reservation_id"])
                or not re.fullmatch(r"[0-9a-f]{64}", binding["request_sha256"])
            ):
                raise ValueError(
                    "trace binding must contain only reservation/request hashes"
                )
            self.data["binding"] = dict(binding)
        if self.path is not None:
            with self.path.open("xb") as f:
                f.write(self.encoded())
                f.flush()
                os.fsync(f.fileno())
            self.fsync_parent()

    def fsync_parent(self):
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def encoded(self):
        return (json.dumps(self.data, sort_keys=True) + "\n").encode()

    def save(self):
        if self.path is None:
            return
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("xb") as f:
            f.write(self.encoded())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self.path)
        self.fsync_parent()

    def stage(self, stage):
        self.data["stage"] = stage
        self.save()

    @staticmethod
    def failure(error):
        # Never exception text, headers, body, key, proxy URL or TLS peer certificate.
        return {"type": type(error).__name__, "errno": getattr(error, "errno", None)}

    def create_connection(
        self, address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None
    ):
        """Equivalent ordered getaddrinfo fallback; not an HTTP/SDK retry."""
        host, port = address
        self.stage("resolving")
        try:
            candidates = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        except OSError as error:
            self.data["dns_failure"] = self.failure(error)
            self.stage("dns_failed_before_http_write")
            raise
        self.data["dns"] = [
            {"family": int(f), "ip": a[0], "port": a[1]} for f, t, p, c, a in candidates
        ]
        self.stage("connecting")
        last = None
        for family, kind, proto, canon, sockaddr in candidates:
            sock = None
            attempt = {"family": int(family), "ip": sockaddr[0], "port": sockaddr[1]}
            self.data["attempts"].append(attempt)
            self.save()
            try:
                sock = socket.socket(family, kind, proto)
                if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                    sock.settimeout(timeout)
                if source_address:
                    sock.bind(source_address)
                sock.connect(sockaddr)
            except OSError as error:
                last = error
                if sock is not None:
                    sock.close()
                attempt["failure"] = self.failure(error)
                self.save()
                continue
            attempt["tcp_connected"] = True
            try:
                self.stage("tcp_connected_before_tls_or_tunnel")
            except BaseException:
                sock.close()
                raise
            return sock
        self.stage("all_endpoints_failed_before_http_write")
        if last is None:
            raise EndpointConnectionError("getaddrinfo returned no addresses")
        raise EndpointConnectionError(
            last.errno, "all connection endpoints failed; see credential-free trace"
        ) from None

    def before_http_write(self):
        self.data["http_write_started"] = True
        self.stage("http_write_may_begin")


class TrackingHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, *, trace, **kwargs):
        self.trace = trace
        super().__init__(host, **kwargs)
        self._create_connection = trace.create_connection

    def connect(self):
        try:
            super().connect()
        except OSError as error:
            self.trace.data["connect_failure"] = self.trace.failure(error)
            connected = [
                a for a in self.trace.data["attempts"] if a.get("tcp_connected")
            ]
            if connected:
                self.trace.data["connect_failure_endpoint"] = {
                    k: connected[-1][k] for k in ("family", "ip", "port")
                }
            self.trace.stage("connection_or_tls_failed")
            raise
        self.trace.stage("tls_ready")

    def send(self, data):
        # stdlib send would connect lazily. Connect first so a failed TCP/TLS leg
        # cannot falsely be recorded as an HTTP write. Proxy CONNECT counts as
        # HTTP write too: a conservative fence, never a claim about API charging.
        if self.sock is None:
            if self.auto_open:
                self.connect()
            else:
                raise http.client.NotConnected()
        self.trace.before_http_write()
        return super().send(data)


class TrackingHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, trace, **kwargs):
        self.trace = trace
        super().__init__(**kwargs)

    def https_open(self, request):
        def connection(host, **kwargs):
            return TrackingHTTPSConnection(host, trace=self.trace, **kwargs)

        return self.do_open(connection, request, context=self._context)
