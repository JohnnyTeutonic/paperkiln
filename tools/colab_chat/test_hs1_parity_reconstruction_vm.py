"""Offline fail-closed checks; no model execution or provider calls."""

import importlib.util
import json
from pathlib import Path
import pytest

p = Path(__file__).with_name("hs1_parity_reconstruction_vm.py")
s = importlib.util.spec_from_file_location("reconstruction", p)
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)


def test_tap_requires_every_numerical_field_equal():
    row = [
        {
            "key": "case",
            "argmax": [1, 2],
            "logprobs": [-1.0, -2.0],
            "sum": -3.0,
            "over_ctx": False,
        }
    ]
    g.instrumented_equivalent(row, row)
    for field, value in [
        ("argmax", [1, 3]),
        ("logprobs", [-1.0, -2.0001]),
        ("sum", -3.0001),
        ("over_ctx", True),
    ]:
        altered = json.loads(json.dumps(row))
        altered[0][field] = value
        with pytest.raises(ValueError, match="changed"):
            g.instrumented_equivalent(row, altered)
    with pytest.raises(ValueError):
        g.instrumented_equivalent(row, [])


def test_existing_negative_receipt_preserved(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "summary.json"
    marker.write_bytes(b'{"status":"ERROR","cause":"historical"}')
    before = marker.read_bytes()
    with pytest.raises(FileExistsError):
        g.run(out)
    assert marker.read_bytes() == before
    assert list(out.iterdir()) == [marker]


def test_missing_inputs_leave_durable_negative(tmp_path):
    upload = tmp_path / "upload"
    upload.mkdir()
    out = tmp_path / "out"
    assert g.run(out, upload, tmp_path / "source", tmp_path / "build") == 0
    report = json.loads((out / "summary.json").read_text())
    assert report["status"] == "ERROR" and report["errors"]
    assert (out / "out.zip").is_file()
    assert report["commands"] == []


def test_source_inventory_refuses_missing_extra_or_changed(tmp_path):
    p = tmp_path / "source.cpp"
    p.write_bytes(b"exact historical bytes")
    expected = {"source.cpp": g.sha(p)}
    g.verify_source_inventory(tmp_path, expected)
    with pytest.raises(ValueError, match="absent"):
        g.verify_source_inventory(tmp_path, {})
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        g.verify_source_inventory(tmp_path, expected)
    p.write_bytes(b"exact historical bytes")
    (tmp_path / "extra.cpp").write_bytes(b"extra")
    with pytest.raises(ValueError, match="differs"):
        g.verify_source_inventory(tmp_path, expected)
    p.unlink()
    with pytest.raises(ValueError, match="differs"):
        g.verify_source_inventory(tmp_path, expected)
