"""tools/precision_gate.py noise_verdict: the pre-stated rule of bf16_gate_v2
(docs/receipts/bf16_gate_v2_prereg_20261025.md) on synthetic run summaries."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import precision_gate as pg  # noqa: E402

STEPS = 100


def summ(val, offset=0.0, sps=1.0, rec=None):
    loss = [5.0 - 0.01 * t + offset for t in range(STEPS)]
    rec = rec or {}
    return {"loss": loss, "val": val, "s_per_step": sps, "model": dict(rec), "result": dict(rec),
            "gguf_precision_key": "precision" in rec}


def base():
    # FP32 at three seeds: final val range 0.04, curve offsets 0, 0.02, 0.04 (range 0.04 everywhere)
    return {1: summ(4.00, 0.00), 2: summ(4.02, 0.02), 3: summ(4.04, 0.04)}


def test_within_the_floor_passes():
    want = {"precision": "bf16"}
    runs = {s: summ(b["val"] + 0.03, (s - 1) * 0.02 + 0.03, 0.5, want) for s, b in base().items()}
    v = pg.noise_verdict(base(), runs, STEPS, want)
    assert v["floor_val_range"] == pytest.approx(0.04)
    assert v["val_diff_mean"] == pytest.approx(0.03) and v["primary_ok"]
    assert v["floor_curve_range_mean"] == pytest.approx(0.04) and v["curve_diff_mean_abs"] == pytest.approx(0.03)
    assert v["secondary_ok"] and v["ok"] and v["verdict"] == "PASS" and v["speedup"] == pytest.approx(2.0)


def test_shift_past_the_floor_fails_either_rule():
    want = {"precision": "bf16"}
    # final val shifted by 0.05 > range 0.04: primary fails
    runs = {s: summ(b["val"] + 0.05, (s - 1) * 0.02, 1.0, want) for s, b in base().items()}
    v = pg.noise_verdict(base(), runs, STEPS, want)
    assert not v["primary_ok"] and v["secondary_ok"] and v["verdict"] == "FAIL"
    # the curve shifted by 0.05 throughout, final val on the floor: secondary fails
    runs = {s: summ(b["val"], (s - 1) * 0.02 + 0.05, 1.0, want) for s, b in base().items()}
    v = pg.noise_verdict(base(), runs, STEPS, want)
    assert v["primary_ok"] and not v["secondary_ok"] and not v["ok"]


def test_unrecorded_setting_fails_and_one_seed_reports_only():
    runs = {s: summ(b["val"], (s - 1) * 0.02) for s, b in base().items()}  # no precision in the events
    v = pg.noise_verdict(base(), runs, STEPS, {"precision": "bf16"})
    assert v["primary_ok"] and v["secondary_ok"] and not v["recorded"] and not v["ok"]
    one = pg.noise_verdict({1: summ(4.0)}, {1: summ(4.1, 0.1, rec={"attention_kernel": "flash"})}, STEPS,
                           {"attention_kernel": "flash"})
    assert one["ok"] is None and "report only" in one["verdict"] and one["val_diff"] == [pytest.approx(0.1)]
