"""Operational wrapper refusals; no GPU or external service is contacted."""

import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "flex_gate", ROOT / "tools/flex_precision_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def manifest():
    return {
        "schema": "plain-flex-bf16-l4-v1",
        "science_commit": "3db619690a0f08051224fa174484a2a22dc1f98a",
        "limits": {
            "max_usd": 1,
            "max_hours": 1,
            "gate_seconds": 1500,
            "gpu": "NVIDIA L4",
            "gpus": 1,
        },
        "bounds": {
            "normalized_element": 0.0003,
            "tensor_relative_norm": 0.03,
            "norm_floor": 1e-10,
        },
        "files": {
            name: "a" * 64
            for name in [
                "source.tgz",
                "flex_precision_gate.py",
                "protocol.json",
                "cpu_fixture.json",
            ]
        },
        "source_files": {
            name: "b" * 64
            for name in [
                "CMakeLists.txt",
                "studio/chat.html",
                "tools/parity_model.hpp",
                "tools/mtstudio.cpp",
                "tests/flex_precision_gate.cpp",
                "tools/replay_flex_precision.py",
                "tools/flex_precision_gate.py",
            ]
        },
    }


def test_registered_manifest():
    gate.validate_manifest(manifest())


@pytest.mark.parametrize(
    "field,value",
    [("schema", "other"), ("science_commit", "0" * 40), ("limits", {}), ("bounds", {})],
)
def test_registration_drift(field, value):
    m = manifest()
    m[field] = value
    with pytest.raises(ValueError):
        gate.validate_manifest(m)


@pytest.mark.parametrize(
    "name",
    [
        "studio/chat.html",
        "tests/flex_precision_gate.cpp",
        "tools/replay_flex_precision.py",
        "tools/mtstudio.cpp",
    ],
)
def test_missing_critical_source(name):
    m = manifest()
    del m["source_files"][name]
    with pytest.raises(ValueError):
        gate.validate_manifest(m)


@pytest.mark.parametrize(
    "name", ["/outside", "../outside", "foo/../outside", "foo\\outside"]
)
def test_unsafe_source_name(name):
    m = manifest()
    m["source_files"][name] = "a" * 64
    with pytest.raises(ValueError):
        gate.validate_manifest(m)


@pytest.mark.parametrize("digest", [None, "z" * 64, "a" * 63, "A" * 64])
def test_bad_digest(digest):
    m = manifest()
    m["files"]["source.tgz"] = digest
    with pytest.raises(ValueError):
        gate.validate_manifest(m)


def test_wrong_manifest_preserves_failure(tmp_path):
    upload, out = tmp_path / "upload", tmp_path / "out"
    upload.mkdir()
    (upload / "inputs.json").write_text(json.dumps(manifest()))
    assert gate.run(upload, "0" * 64, out) == 1
    report = json.loads((out / "gate_summary.json").read_text())
    assert report["pass"] is False and report["errors"] == ["manifest differs"]
    assert report["commands"] == [] and (out / "out.zip").is_file()
    with pytest.raises(ValueError, match="immutable"):
        gate.run(upload, "0" * 64, out)


def test_uploaded_drift_stops_before_hardware(tmp_path):
    upload, out = tmp_path / "upload", tmp_path / "out"
    upload.mkdir()
    m = manifest()
    for name in m["files"]:
        (upload / name).write_text("deliberately wrong")
    raw = json.dumps(m).encode()
    (upload / "inputs.json").write_bytes(raw)
    assert gate.run(upload, hashlib.sha256(raw).hexdigest(), out) == 1
    report = json.loads((out / "gate_summary.json").read_text())
    assert report["pass"] is False and report["commands"] == []
    assert report["errors"] == ["uploaded file drift"]
