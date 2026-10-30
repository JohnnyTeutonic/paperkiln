"""Actual Studio HTTP lifecycle with an owned mock worker, never an API call."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = os.environ.get("MTSTUDIO_BIN", str(ROOT / "build/mtstudio"))
pytestmark = pytest.mark.skipif(not Path(BINARY).is_file(), reason="set MTSTUDIO_BIN")


@pytest.fixture
def studio(tmp_path, request):
    worker = tmp_path / "mock.py"
    worker.write_text("""import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('id');p.add_argument('--out');p.add_argument('--budget-usd');p.add_argument('--tag');p.add_argument('--parent-pid');a=p.parse_args()
Path(a.out+'.pid').write_text(str(os.getpid()))
if a.id == '1111.11111': time.sleep(30)
if a.id == '2222.22222':
 Path(a.out+'.error.json').write_text(json.dumps({'error':'mock provider refused'}));raise SystemExit(1)
Path(a.out).write_text(json.dumps({'arxiv_id':a.id,'title':'<img src=x onerror=bad()>','fields':{'d_model':{'value':16,'verdict':'used','evidence':'dimension 16'}},'llm':{'verification':{'mock':True}},'budget_passthrough':float(a.budget_usd)}))
""")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    out = tmp_path / "out"
    out.mkdir()
    configured = getattr(request, "param", True)
    env = dict(os.environ, MTSTUDIO_UI=str(ROOT / "studio/index.html"), MTSTUDIO_EXTRACT=str(worker),
               MTSTUDIO_PYTHON=sys.executable, ANTHROPIC_API_KEY="mock-only-not-sent" if configured else "")
    proc = subprocess.Popen([BINARY, "serve", str(out), str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def request(path, value=None, headers=None):
        body = None if value is None else (json.dumps(value) if not isinstance(value, str) else value).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/{path}", data=body, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()
    try:
        for _ in range(100):
            try:
                if request("extractstatus")[0] == 200:
                    break
            except OSError:
                time.sleep(.03)
        else:
            pytest.fail("Studio did not start")
        yield request, out, port
    finally:
        try:
            request("extractcancel", {})
        except OSError:
            pass
        proc.terminate()
        proc.wait(timeout=5)


def wait_state(request, expected):
    for _ in range(100):
        value = json.loads(request("extractstatus")[1])
        if value["state"] == expected:
            return value
        time.sleep(.03)
    pytest.fail(f"did not reach {expected}: {value}")


def test_success_actual_result_then_failure_does_not_reuse_previous(studio):
    request, out, _ = studio
    assert request("extractresult")[0] == 409
    assert request("extract", {"arxiv_id":"1234.56789", "budget_usd":1.23456789})[0] == 202
    wait_state(request, "done")
    result = json.loads(request("extractresult")[1])
    assert result["arxiv_id"] == "1234.56789"
    assert result["budget_passthrough"] == 1.23456789
    assert result["llm"]["verification"]["mock"] is True
    assert request("extract", {"arxiv_id":"2222.22222", "budget_usd":1})[0] == 202
    status = wait_state(request, "failed")
    assert status["error"] == "mock provider refused"
    assert request("extractresult")[0] == 409
    assert len(list(out.glob("*.extraction.json"))) == 1
    assert not (out / "arch.json").exists()  # no silent rule fallback or builder application


def test_cancel_owned_worker_and_reject_duplicate(studio):
    request, out, _ = studio
    payload = {"arxiv_id":"1111.11111", "budget_usd":1}
    assert request("extract", payload)[0] == 202
    wait_state(request, "extracting")
    assert request("extract", payload)[0] == 409
    for _ in range(100):
        pids = list(out.glob("*.pid"))
        if pids:
            break
        time.sleep(.01)
    assert pids
    pid = int(pids[0].read_text())
    assert request("extractcancel", {}) == (200, "cancelled")
    assert json.loads(request("extractstatus")[1])["state"] == "cancelled"
    assert request("extractresult")[0] == 409
    assert not list(out.glob("*.extraction.json"))
    if os.name != "nt":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_validation_and_cross_origin_rejected_before_worker(studio):
    request, out, port = studio
    valid = {"arxiv_id":"1234.56789", "budget_usd":1}
    for origin in ("https://foreign.example", "null", "http://127.0.0.1:1"):
        assert request("extract", valid, {"Origin":origin})[0] == 403
        assert request("extractcancel", {}, {"Origin":origin})[0] == 403
    assert request("extract", valid, {"Host":"foreign.example"})[0] == 403
    for payload in ({}, {"arxiv_id":"not a paper", "budget_usd":1}, {"arxiv_id":"1234.56789", "budget_usd":0}, {"arxiv_id":"1234.56789", "budget_usd":6}):
        assert request("extract", payload)[0] == 400
    assert not list(out.iterdir())
    assert request("extract", valid, {"Origin":f"http://127.0.0.1:{port}"})[0] == 202


@pytest.mark.parametrize("studio", [False], indirect=True)
def test_no_key_fails_before_worker_and_status_never_exposes_key(studio):
    request, out, _ = studio
    status = json.loads(request("extractstatus")[1])
    assert status["configured"] is False
    assert request("extract", {"arxiv_id":"1234.56789", "budget_usd":1})[0] == 409
    assert not list(out.iterdir())
