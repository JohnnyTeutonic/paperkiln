import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location(
    "cli", ROOT / "tools/mamba1_cli_lifecycle.py"
)
cli = importlib.util.module_from_spec(s)
s.loader.exec_module(cli)
D = ROOT / "docs/receipts/mamba1_cli_lifecycle_20261031/cpu_control"
EVENTS = [
    json.loads(line)
    for line in (D / "depth1/trained/events.jsonl").read_text().splitlines()
]


def synthetic_gpu():
    events = copy.deepcopy(EVENTS)
    for e in events:
        if e.get("event") == "step":
            e["gpu_n"] = {
                op: 1
                for op in [
                    "conv1d_causal_fwd",
                    "conv1d_causal_bwd",
                    "selective_scan_fwd",
                    "selective_scan_bwd",
                ]
            }
            e["gpu_ms"] = {op: 0.1 for op in e["gpu_n"]}
    return events


def test_cpu():
    cli.check_events(EVENTS, 1, False)


def test_gpu_parser_only():
    cli.check_events(synthetic_gpu(), 1, True)


@pytest.mark.parametrize("kind", range(13))
def test_event_mutations(kind):
    events = synthetic_gpu()
    model = next(e for e in events if e.get("event") == "model")
    step = next(e for e in events if e.get("event") == "step")
    if kind == 0:
        model["ssm"]["kind"] = "s4"
    if kind == 1:
        model["d"] = 16
    if kind == 2:
        model["layers"] = True
    if kind == 3:
        step["grad_norm"] = 0
    if kind == 4:
        step["loss"] = float("nan")
    if kind == 5:
        step["gpu_n"].pop("conv1d_causal_fwd")
    if kind == 6:
        step["gpu_n"].pop("conv1d_causal_bwd")
    if kind == 7:
        step["gpu_n"].pop("selective_scan_fwd")
    if kind == 8:
        step["gpu_n"].pop("selective_scan_bwd")
    if kind == 9:
        step["gpu_n"]["selective_scan_fwd"] = True
    if kind == 10:
        events.remove(step)
    if kind == 11:
        step["step"] = True
    if kind == 12:
        step["gpu_ms"]["selective_scan_fwd"] = float("nan")
    with pytest.raises(ValueError):
        cli.check_events(events, 1, True)


def test_cpu_cannot_substitute():
    with pytest.raises(ValueError):
        cli.check_events(EVENTS, 1, True)


@pytest.mark.parametrize("kind", range(5))
def test_full_row_mutations(kind):
    ref = np.arange(56, dtype=np.float64).reshape(8, 7) / 100
    actual = ref.copy()
    if kind == 0:
        actual = actual.reshape(7, 8)
    if kind == 1:
        actual[0, 0] = float("nan")
    if kind == 2:
        actual[0, 0] += 0.01
    if kind == 3:
        actual = np.zeros((8, 7))
    if kind == 4:
        actual = np.zeros((8, 1))
    with pytest.raises(ValueError):
        cli.compare(actual, ref, (8, 7))


def test_full_rows_pass():
    cli.compare(np.arange(56).reshape(8, 7), np.arange(56).reshape(8, 7), (8, 7))
