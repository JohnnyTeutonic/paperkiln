"""Studio worker verifier/publication tests: no model calls or credentials."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "papers"))
import studio_extract as worker  # noqa: E402
from test_llm_extract import LLAMA_TEX, llama_output  # noqa: E402


def test_actual_raw_response_reverified_not_asserted_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-only-never-sent")
    raw = llama_output()
    forged = {"fields": {"d_model": {"value": 999}}, "source": {"tex_sha256": "forged"},
              "llm": {"output": raw, "verification": {"forged": True}}}
    extract = Mock(return_value=forged)
    client = object()
    path = tmp_path / "extraction.json"
    actual = worker.extract("1234.56789", path, 2, "studio-mock", source_loader=lambda _: LLAMA_TEX,
                            extractor=extract, client_factory=lambda: client)
    assert actual["fields"]["d_model"]["value"] == 2048
    assert actual["source"]["tex_sha256"] == worker.lx.sha256_text(LLAMA_TEX)
    assert actual["llm"]["verification"] != {"forged": True}
    assert json.loads(path.read_text()) == actual
    assert extract.call_args.args[2:5] == (client, "studio-mock", 2)


@pytest.mark.parametrize("budget", [0, -1, 6, float("nan"), float("inf")])
def test_invalid_budget_before_any_source_or_client_work(tmp_path, monkeypatch, budget):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-only")
    source = Mock()
    with pytest.raises(ValueError, match="spend cap"):
        worker.extract("1234.56789", tmp_path / "x", budget, "tag", source_loader=source)
    source.assert_not_called()


def test_missing_key_no_credential_search_no_source_fetch(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    source = Mock()
    with pytest.raises(ValueError, match="not configured"):
        worker.extract("1234.56789", tmp_path / "x", 1, "tag", source_loader=source)
    source.assert_not_called()


def test_bad_raw_quote_stays_unverified_in_published_review(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-only")
    raw = llama_output()
    raw["flavours"]["norm"]["quote"] = "Invented evidence not in this paper"
    actual = worker.extract("1234.56789", tmp_path / "x", 1, "tag", source_loader=lambda _: LLAMA_TEX,
        extractor=lambda *args: {"llm": {"output": raw}}, client_factory=lambda: object())
    assert actual["fields"]["norm"]["verdict"] == "unverified"
    assert any(field["field"] == "norm" for field in actual["unverified"])


def test_atomic_publication_failure_preserves_previous_bytes(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_bytes(b"previous exact bytes")
    def denied(*args):
        raise OSError("injected publication error")
    monkeypatch.setattr(worker.os, "replace", denied)
    with pytest.raises(OSError):
        worker.atomic_json(path, {"new": True})
    assert path.read_bytes() == b"previous exact bytes"
    assert list(tmp_path.iterdir()) == [path]


def test_errors_redact_inherited_key_without_rule_fallback(tmp_path, monkeypatch):
    key = "private-test-key-never-sent"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    def failure(*args):
        raise RuntimeError("provider error " + key)
    monkeypatch.setattr(worker, "extract", failure)
    out = tmp_path / "x.json"
    assert worker.main(["1234.56789", "--out", str(out), "--budget-usd", "1", "--tag", "tag"]) == 1
    assert not out.exists()
    error = Path(str(out) + ".error.json").read_text()
    assert key not in error
    assert "[redacted]" in error


@pytest.mark.skipif(sys.platform != "linux" and sys.platform != "win32", reason="Linux/Windows parent-death execution probe")
def test_worker_stops_after_owned_parent_exits(tmp_path):
    import os
    import signal
    import subprocess
    import time
    heartbeat = tmp_path / "heartbeat"
    child_code = ("import sys,os,time; from pathlib import Path; "
                  + "sys.path.insert(0," + repr(str(ROOT / "tools")) + "); "
                  + "import studio_extract; studio_extract.watch_parent(os.getppid()); "
                  + "p=Path(" + repr(str(heartbeat)) + "); "
                  + "exec('while True:\\n p.write_text(str(time.time_ns()))\\n time.sleep(.03)')")
    parent_code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c'," + repr(child_code) + "]);print(p.pid,flush=True);time.sleep(30)"
    parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    handle = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, child_pid)
        assert handle
    def live():
        if handle:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        path = Path(f"/proc/{child_pid}/stat")
        return path.exists() and path.read_text().split(") ", 1)[1].split()[0] != "Z"
    try:
        for _ in range(100):
            if heartbeat.exists():
                break
            time.sleep(.02)
        assert heartbeat.exists()
        parent.terminate()
        parent.wait(timeout=3)
        for _ in range(100):
            if not live():
                break
            time.sleep(.02)
        assert not live(), "worker outlived its owning server process"
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=3)
        if live():
            os.kill(child_pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
        if handle:
            kernel.CloseHandle(handle)


@pytest.mark.parametrize("identifier", ["../../private", "https://foreign.example/x", "", "1234.56789; echo bad"])
def test_invalid_identifier_before_source_access(tmp_path, monkeypatch, identifier):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-only")
    source = Mock()
    with pytest.raises(ValueError, match="arXiv identifier"):
        worker.extract(identifier, tmp_path / "x", 1, "tag", source_loader=source)
    source.assert_not_called()
