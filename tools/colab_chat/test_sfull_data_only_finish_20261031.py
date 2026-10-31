import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('data_finish', Path(__file__).with_name('sfull_data_only_finish_20261031.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    for job in ('data', 'sfull_inputs'):
        (tmp_path / ('STATUS_' + job)).write_text('FINISHED\n')
    folder = tmp_path / 'sfull_inputs'
    folder.mkdir()
    report = {'verified': True, 'raw_human_sources_verified': 13,
              'evaluation_sources_verified': 2,
              'reference_sha256': 'acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4'}
    (folder / 'verified_inputs.json').write_text(json.dumps(report))
    return report


def test_only_complete_original_verification_succeeds(tmp_path):
    fixture(tmp_path)
    assert module.finish(tmp_path, 0) == 0
    assert (tmp_path / 'STATUS').read_text() == 'FINISHED\n'
    assert not (tmp_path / 'STATUS.tmp').exists()


@pytest.mark.parametrize('job', ['data', 'sfull_inputs'])
@pytest.mark.parametrize('status', ['FAILED', 'WAITING', 'RUNNING', 'FINISHED_extra', None])
def test_zero_jobs_exit_does_not_hide_failed_or_missing_job(tmp_path, job, status):
    fixture(tmp_path)
    p = tmp_path / ('STATUS_' + job)
    if status is None:
        p.unlink()
    else:
        p.write_text(status)
    assert module.finish(tmp_path, 0) == 1
    assert (tmp_path / 'STATUS').read_text() == 'JOBS_FAILED\n'


@pytest.mark.parametrize('field,value', [
    ('verified', False), ('verified', 1), ('verified', None),
    ('raw_human_sources_verified', 12), ('evaluation_sources_verified', 1),
    ('reference_sha256', 'different'),
])
def test_incomplete_or_different_verifier_report_fails(tmp_path, field, value):
    report = fixture(tmp_path)
    report[field] = value
    (tmp_path / 'sfull_inputs/verified_inputs.json').write_text(json.dumps(report))
    assert module.finish(tmp_path, 0) == 1


@pytest.mark.parametrize('content', ['not json', '[]', 'null', '{}', None])
def test_malformed_or_missing_verifier_report_fails(tmp_path, content):
    fixture(tmp_path)
    p = tmp_path / 'sfull_inputs/verified_inputs.json'
    if content is None:
        p.unlink()
    else:
        p.write_text(content)
    assert module.finish(tmp_path, 0) == 1


def test_failed_process_cannot_succeed_with_old_finished_statuses(tmp_path):
    fixture(tmp_path)
    assert module.finish(tmp_path, 1) == 1
