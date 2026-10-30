"""Resume archives must never overwrite a run already finished on the VM."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil

import pytest

SPEC = importlib.util.spec_from_file_location(
    "transfer_runner", Path(__file__).resolve().parents[1] / "tools/colab_transfer_runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_run(root, name, files):
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    for filename, value in files.items():
        (target / filename).write_text(value)
    return target


@pytest.mark.parametrize("local_finished", [False, True])
@pytest.mark.parametrize("remote_finished", [False, True])
def test_resume_preserves_finished_remote_run(tmp_path, monkeypatch, local_finished, remote_finished):
    local, remote = tmp_path / "local", tmp_path / "remote"
    old = {"state.txt": "2900\n", "model.safetensors": "old-model",
           "optim.safetensors": "old-optim", "events.jsonl": "old-events\n"}
    if local_finished:
        old["result.json"] = '{"step":2900}'
    write_run(local / ("runs" if local_finished else "partial"), "run1", old)
    finished = {"result.json": '{"step":5000}', "events.jsonl": "final-events\n"}
    dst = write_run(remote / "runs", "run1", finished if remote_finished else {})
    archive = tmp_path / "uploaded.zip"
    monkeypatch.setattr(runner, "TMP", str(tmp_path / "driver"))

    def upload(session, source, target):
        shutil.copyfile(source, archive)
        return True

    def exec_remote(session, code, timeout):
        # Execute the production VM snippet against a filesystem fixture.
        code = code.replace("/content/tr_resume.zip", archive.as_posix())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, "resume_vm", "exec"), {})
        return 0, output.getvalue()

    monkeypatch.setattr(runner, "upload", upload)
    monkeypatch.setattr(runner, "exec_py", exec_remote)
    result = runner.push_resume("fixture", str(local), str(remote))
    assert result == ((1, 0) if local_finished else (0, 1))
    observed = {p.name: p.read_text() for p in dst.iterdir()}
    assert observed == (finished if remote_finished else old)
