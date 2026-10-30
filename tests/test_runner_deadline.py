import argparse
import importlib.util
from pathlib import Path
import sys

import pytest

spec=importlib.util.spec_from_file_location('deadline_runner',Path(__file__).resolve().parents[1]/'tools/colab_chat/runner_multi.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('value',['2026-10-30','2026-10-30T13:17:00','tomorrow','2026-99-30T13:17:00Z','2026-10-30 13:17:00Z'])
def test_invalid_or_naive_deadline_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_deadline_utc(value)


def test_timezone_offsets_normalize():
    assert runner.parse_deadline_utc('2026-10-30T13:17:00Z')==runner.parse_deadline_utc('2026-10-31T00:17:00+11:00')


def test_earliest_deadline_and_default():
    assert runner.resolve_deadline(1,None,100)==3700
    assert runner.resolve_deadline(1,10000,100)==3700
    assert runner.resolve_deadline(1,1000,100)==1000
    assert runner.resolve_deadline(1,1000,700)==1000
    with pytest.raises(ValueError,match='expired'):
        runner.resolve_deadline(1,1000,1000)


def test_expired_main_refuses_before_any_api_or_filesystem(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(runner,'make_transport',lambda *a:calls.append('transport'))
    monkeypatch.setattr(runner,'listed',lambda *a:calls.append('listed'))
    monkeypatch.setattr(runner,'launch',lambda *a:calls.append('launch'))
    monkeypatch.setattr(sys,'argv',['runner','--backend','runpod','--commit','frozen','--jobs','missing.json',
        '--name','expired','--art',str(tmp_path/'art'),'--state-dir',str(tmp_path/'state'),
        '--deadline-utc','2000-01-01T00:00:00Z'])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code==2 and calls==[]
    assert not (tmp_path/'state').exists() and not (tmp_path/'art').exists()


def test_expired_launch_never_stops_or_creates(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(runner,'T',SimpleNamespace(deadline=0,stop=lambda st:calls.append('stop'),create=lambda st:calls.append('create')))
    monkeypatch.setattr(runner,'log',lambda m:None)
    assert runner.launch(None,'c','j',[],[]) is False
    assert calls==[]
