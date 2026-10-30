"""Pinned command-only boot must preserve the default training boot protocol."""
import builtins
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runner_boot", ROOT / "tools/colab_chat/runner_multi.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("script", ["boot_jobs.sh", "boot_commands.sh", "boot_commands_sparse.sh"])
def test_production_boot_snippet_fetches_selected_pinned_script(tmp_path, monkeypatch, script):
    urls, launches = [], []
    target = tmp_path / "boot.sh"
    real_open = builtins.open

    def fetch(url):
        urls.append(url)
        return io.BytesIO(b"#!/bin/bash\necho fixture\n")

    def fopen(path, *args, **kwargs):
        assert path == "/content/boot_jobs.sh"
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: launches.append((command, kwargs)))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(runner.boot_launch_code("abc123", "tools/jobs.json", "tools/colab_chat/" + script),
             {"open": fopen})
    assert urls == ["https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/abc123/tools/colab_chat/" + script]
    assert target.read_bytes() == b"#!/bin/bash\necho fixture\n"
    assert launches == [("nohup bash /content/boot_jobs.sh abc123 tools/jobs.json > /dev/null 2>&1 &", {"shell": True})]
    assert output.getvalue() == "BOOTED\n"


def test_default_is_unchanged():
    assert runner.BOOT_SCRIPT == "tools/colab_chat/boot_jobs.sh"
    assert runner.boot_launch_code("abc", "jobs.json") == runner.boot_launch_code(
        "abc", "jobs.json", "tools/colab_chat/boot_jobs.sh")


@pytest.mark.parametrize("path", ["../../bad.sh", "https://example.com/a.sh", "tools/colab_chat/boot_commands.sh; evil"])
def test_unknown_scripts_refused(path):
    with pytest.raises(ValueError):
        runner.boot_launch_code("abc", "jobs.json", path)


@pytest.mark.parametrize("jobs,accepted", [([], False), ([{"spec": "train.json"}], False),
    ([{"cmd": ["echo", "ok"]}, {"spec": "train.json"}], False), ([{"cmd": ["echo", "ok"]}], True)])
def test_command_boot_rejects_training_jobs(tmp_path, jobs, accepted):
    import json
    import sys
    source = (ROOT / "tools/colab_chat/boot_commands.sh").read_text()
    validation = source.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": jobs}))
    old = sys.argv
    try:
        sys.argv = ["fixture", str(path)]
        if accepted:
            exec(validation, {})
        else:
            with pytest.raises(SystemExit):
                exec(validation, {})
    finally:
        sys.argv = old
