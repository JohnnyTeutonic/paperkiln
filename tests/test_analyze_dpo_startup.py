import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/colab_chat/analyze_dpo_startup.py'
spec = importlib.util.spec_from_file_location('startup_analysis', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def markers(total=35, step=1):
    labels = mod.ORDER[:3] + ['reference_cache_progress'] * 3 + mod.ORDER[3:]
    counts = [0, 0, 0, 1, 32, total, total, total, total]
    return [dict(phase=p, elapsed_seconds=i / 10, pairs_completed=n,
                 pairs_total=total, step=step if i >= 7 else 0, diagnostic_only=True)
            for i, (p, n) in enumerate(zip(labels, counts))]


def encode(rows):
    return ('ordinary stderr\n' + ''.join(mod.PREFIX + json.dumps(r) + '\n' for r in rows)).encode()


def test_complete_reports_observations_without_training_or_checkpoint_verdict():
    result = mod.analyze(encode(markers()), 35)
    assert result['first_iteration_end_observed'] is True
    assert result['reference_pairs_observed_complete'] == 35
    assert result['completed_phase_seconds']['reference_cache'] == pytest.approx(.4)
    assert result['checkpoint_proof'] is False and result['adoption_verdict'] is None


@pytest.mark.parametrize('length', range(10))
def test_valid_prefix_is_observation_not_assumed_halt_cause(length):
    result = mod.analyze(encode(markers()[:length]), 35)
    assert result['markers'] == length
    assert result['first_iteration_end_observed'] is (length == 9)
    assert result['checkpoint_proof'] is False


def test_real_compiled_cli_marker_log():
    raw = (ROOT / 'docs/receipts/dpo_startup_phases_20261101/on_stderr.txt').read_bytes()
    result = mod.analyze(raw, 35)
    assert result['first_iteration_end_observed'] is True
    assert result['markers'] == 9


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '1'])
def test_bad_clock_rejected(value):
    rows = markers(); rows[4]['elapsed_seconds'] = value
    with pytest.raises(ValueError): mod.analyze(encode(rows), 35)


@pytest.mark.parametrize('change', [
    lambda r: r[4].update(elapsed_seconds=.1),
    lambda r: r[4].update(pairs_completed=1),
    lambda r: r[4].update(pairs_completed=36),
    lambda r: r[4].update(pairs_completed=True),
    lambda r: r[4].update(pairs_total=36),
    lambda r: r[4].update(diagnostic_only=1),
    lambda r: r[4].update(phase='unknown'),
    lambda r: r[4].update(extra='unrecognised'),
    lambda r: r[6].update(pairs_completed=34),
    lambda r: r[7].update(step=2),
    lambda r: r[1].update(step=1),
    lambda r: r.pop(5),
    lambda r: r.insert(1, dict(r[0])),
    lambda r: r.extend(markers()),
])
def test_malformed_or_mixed_process_stream_rejected(change):
    rows = markers(); change(rows)
    with pytest.raises(ValueError): mod.analyze(encode(rows), 35)


def test_resume_step_explicitly_bound():
    assert mod.analyze(encode(markers(step=11)), 35, start_step=10)['start_step'] == 10


def test_duplicate_key_and_truncated_marker_rejected():
    for raw in [b'DPO_STARTUP_PHASE {"phase":0,"phase":1}\n', b'DPO_STARTUP_PHASE {']:
        with pytest.raises(ValueError): mod.analyze(raw, 35)


@pytest.mark.parametrize('total,step', [(True, 0), (0, 0), (35, True), (35, -1)])
def test_invalid_expected_contract(total, step):
    with pytest.raises(ValueError): mod.analyze(b'', total, step)


def test_bounded_input(monkeypatch):
    monkeypatch.setattr(mod, 'MAX_BYTES', 2)
    with pytest.raises(ValueError): mod.analyze(b'123', 35)


def test_cli_exclusive_output_and_original_preservation(tmp_path):
    log = tmp_path / 'stderr'; log.write_bytes(encode(markers()))
    output = tmp_path / 'result.json'
    command = [sys.executable, str(SCRIPT), str(log), '--expected-pairs', '35', '--output', str(output)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
    same = command[:-1] + [str(log)]
    assert subprocess.run(same, capture_output=True).returncode != 0
    assert log.read_bytes() == encode(markers())
