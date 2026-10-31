"""Replay integrity against independent synthetic raw float rows, without inference."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hs1_parity_cpu_replay as replay  # noqa: E402


def bind(folder):
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.iterdir()
        if p.name != "cpu_manifest.json"
    }
    (folder / "cpu_manifest.json").write_text(
        json.dumps({"files_sha256": files, "banned_ids": [0]})
    )


@pytest.fixture
def saved(tmp_path):
    requests = []
    for i in range(6):
        run, key = f"run{i}", f"key{i}"
        requests.append({"run": run, "request": {"key": key, "continuation_ids": [2]}})
        row = np.zeros((1, 16384), dtype="<f4")
        row[0, 0], row[0, 2], row[0, 3] = 100.0, 1.0, 0.5
        for mode in ("full", "kv"):
            (tmp_path / f"{run}_{mode}.scores.jsonl").write_text(
                json.dumps(
                    {
                        "key": key,
                        "continuation_ids": [2],
                        "argmax": [2],
                        "over_ctx": False,
                    }
                )
                + "\n"
            )
            (tmp_path / f"{run}_{mode}.logits.f32.gz").write_bytes(
                gzip.compress(row.tobytes(), mtime=0)
            )
    (tmp_path / "requests.json").write_text(json.dumps(requests))
    bind(tmp_path)
    return tmp_path


def test_replay_uses_ban_and_raw_rows(saved):
    result = replay.replay(saved)
    assert len(result["rows"]) == 6
    assert all(
        x["full"][0]["argmax_id"] == 2 and x["full"][0]["top2_margin"] == 0.5
        for x in result["rows"]
    )


def test_artifact_tampering_refused(saved):
    (saved / "run0_full.scores.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        replay.replay(saved)


def test_native_argmax_must_match_raw_logits_even_with_rebound_manifest(saved):
    p = saved / "run0_full.scores.jsonl"
    row = json.loads(p.read_text())
    row["argmax"] = [3]
    p.write_text(json.dumps(row))
    bind(saved)
    with pytest.raises(ValueError, match="argmax"):
        replay.replay(saved)


def test_nonfinite_logits_refused(saved):
    p = saved / "run0_full.logits.f32.gz"
    row = np.frombuffer(gzip.decompress(p.read_bytes()), dtype="<f4").copy()
    row[2] = np.nan
    p.write_bytes(gzip.compress(row.tobytes(), mtime=0))
    bind(saved)
    with pytest.raises(ValueError, match="nonfinite"):
        replay.replay(saved)
