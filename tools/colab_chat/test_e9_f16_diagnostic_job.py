"""Execute the exact command-job Python heredoc on successful/incomplete transport."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

JOB = Path(__file__).with_name('ember_e9_f16_diagnosis_20261031.json')


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('missing', [None, 'vm.log', 'out.zip', 'bad_zip', 'bad_report'])
def test_actual_embedded_transport_json(tmp_path, complete, missing):
    shell = json.loads(JOB.read_text())['jobs'][0]['cmd'][2]
    assert shell.startswith('set -euo pipefail\n')
    body = shell.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    report = {'status': 'DIAGNOSTIC_COMPLETE', 'e9_closed': False}
    if complete:
        (tmp_path/'diagnosis_report.json').write_text(json.dumps(report)+'\n')
    if missing == 'bad_report':
        (tmp_path/'diagnosis_report.json').write_text('{invalid')
    if missing != 'vm.log':
        (tmp_path/'vm.log').write_text('original log\n')
    if missing != 'out.zip':
        if missing == 'bad_zip':
            (tmp_path/'out.zip').write_bytes(b'not a zip')
        else:
            with zipfile.ZipFile(tmp_path/'out.zip', 'w') as archive:
                archive.writestr('diagnosis/diagnosis_report.json', json.dumps(report)+'\n')
    result = subprocess.run([sys.executable, '-c', body, str(tmp_path), '0' if complete else '124'],
                            capture_output=True)
    assert (result.returncode == 0) is (missing is None)
    if missing is None:
        recorded = json.loads((tmp_path/'diagnosis_report.json').read_text())
        execution = json.loads((tmp_path/'execution.json').read_text())
        assert execution['complete'] is complete
        assert execution['e9_closed'] is False
        assert recorded['status'] == ('DIAGNOSTIC_COMPLETE' if complete else 'INCOMPLETE')
    if missing == 'bad_report':
        assert (tmp_path/'diagnosis_report.json').read_text() == '{invalid'


def test_complete_outer_report_must_equal_archive(tmp_path):
    shell = json.loads(JOB.read_text())['jobs'][0]['cmd'][2]
    body = shell.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    (tmp_path/'diagnosis_report.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE'}))
    (tmp_path/'vm.log').write_text('original log')
    with zipfile.ZipFile(tmp_path/'out.zip', 'w') as archive:
        archive.writestr('diagnosis/diagnosis_report.json', '{}')
    result = subprocess.run([sys.executable,'-c',body,str(tmp_path),'0'],capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path/'execution.json').exists()