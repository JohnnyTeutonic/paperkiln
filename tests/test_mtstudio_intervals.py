"""Reject invalid modulo intervals before model/data/output work."""

import json
import os
import subprocess
import pytest

BIN = os.environ.get("MTSTUDIO_BIN")
pytestmark = pytest.mark.skipif(not BIN, reason="set MTSTUDIO_BIN to actual CLI binary")


@pytest.mark.parametrize("field", ["eval_every", "checkpoint_every"])
@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("command", ["plan", "run"])
def test_invalid_interval_is_rejected_before_outputs(tmp_path, field, value, command):
    out = tmp_path / "run"
    spec = {
        "arch": {"preset": "gpt2-nano"},
        "data": {"corpus": "missing.txt"},
        "train": {field: value},
        "out_dir": str(out),
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    result = subprocess.run(
        [BIN, command, str(path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode > 0, result.stdout + result.stderr
    assert "train." + field + " must be >= 1" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    "train",
    [
        {},
        {"eval_every": 1, "checkpoint_every": 1},
        {"eval_every": 3, "checkpoint_every": 5},
    ],
)
def test_positive_or_default_intervals_can_be_planned(tmp_path, train):
    out = tmp_path / "run"
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {"arch": {"preset": "gpt2-nano"}, "train": train, "out_dir": str(out)}
        )
    )
    result = subprocess.run(
        [BIN, "plan", str(path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not out.exists()
