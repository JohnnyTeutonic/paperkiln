"""tools/iftiny_eval.py's POST to a /chat server: a reply comes back as JSON, and an HTTP error raises with the
server's own message (pk-bench2's IF-tiny job logged only "HTTP Error 500" on 26 Oct 2026; the cause was in the
body). Offline: a local http.server, no model."""
import http.server
import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import iftiny_eval  # noqa: E402

ERR = "Internal server error: [json.exception.type_error.316] incomplete UTF-8 string; last byte: 0xD7"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["user_input"] == "fail":
            code, out = 500, {"error": ERR}
        else:
            code, out = 200, {"reply": body["user_input"].upper(), "stop_reason": "eos"}
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    s = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()
    s.server_close()


def test_post_json_returns_the_reply(server):
    assert iftiny_eval.post_json(server + "/chat", {"user_input": "hi"}) == {"reply": "HI", "stop_reason": "eos"}


def test_post_json_raises_with_the_servers_message(server):
    with pytest.raises(RuntimeError) as e:
        iftiny_eval.post_json(server + "/chat", {"user_input": "fail"})
    assert "HTTP 500" in str(e.value) and "type_error.316" in str(e.value) and '"user_input": "fail"' in str(e.value)
