"""Transport subprocess limits with tiny owned local processes, never provider/SSH calls."""
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/colab_chat"))
import runpod_transport as rt  # noqa: E402


@pytest.fixture
def transport(tmp_path, monkeypatch):
    t = rt.RunPodTransport("L4", "SECURE", 10, None, None, None, "c", "test", str(tmp_path), lambda m: None)
    monkeypatch.setattr(rt, "OPERATION_POLL_S", 0.02)
    return t


def test_normal_large_stdin_stdout_stderr_survive_poll_retries(transport):
    payload = "a" * 250000
    rc, out = transport._run([sys.executable, "-c",
        "import sys,time;time.sleep(.08);x=sys.stdin.read();print(len(x));print('stderr',file=sys.stderr)"],
        timeout=3, stdin_text=payload)
    assert rc == 0 and "250000" in out and "stderr" in out


def test_fixed_timeout_preserves_output_and_reaps_owned_process(transport, tmp_path):
    pidfile = tmp_path / "pid"
    script = "import os,time,pathlib;pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid()));print('partial',flush=True);time.sleep(30)"
    started = time.monotonic()
    rc, out = transport._run([sys.executable, "-c", script], timeout=.3)
    assert rc == 124 and "TIMEOUT" in out and "partial" in out
    assert time.monotonic() - started < 5
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)


def test_deadline_interrupts_transfer_without_waiting_fixed_timeout(transport):
    transport.deadline = time.time() + .15
    start = time.monotonic()
    with pytest.raises(rt.api.Refused, match="deadline"):
        transport._run([sys.executable, "-c", "import time;time.sleep(30)"], timeout=30)
    assert time.monotonic() - start < 5


def test_budget_expiry_aborts_upload_no_retries_or_partial_cleanup(transport, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_text("upload")
    partial = tmp_path / "remote.part00"
    partial.write_text("partial progress")
    transport.st = SimpleNamespace()
    started = time.monotonic()
    monkeypatch.setattr(transport, "over_budget", lambda st: "spend cap reached" if time.monotonic() - started > .15 else None)
    monkeypatch.setattr(transport, "_remote_shas", lambda paths: {})
    attempts = []
    def put(src, dst, timeout):
        attempts.append(dst)
        return transport._run([sys.executable, "-c", "import time;time.sleep(30)"], timeout=timeout)
    monkeypatch.setattr(transport, "put", put)
    with pytest.raises(rt.api.Refused, match="spend cap"):
        transport.upload_whole(str(source), "remote")
    assert attempts == ["remote.part00"]
    assert partial.read_text() == "partial progress"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership proof")
def test_expiry_kills_only_owned_process_group_no_running_child_leak(transport, tmp_path):
    childfile = tmp_path / "child"
    script = ("import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
              "pathlib.Path(" + repr(str(childfile)) + ").write_text(str(p.pid));time.sleep(30)")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        transport.deadline = time.time() + .4
        with pytest.raises(rt.api.Refused):
            transport._run([sys.executable, "-c", script], timeout=30)
        assert unrelated.poll() is None
        child = int(childfile.read_text())
        stat = Path(f"/proc/{child}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
