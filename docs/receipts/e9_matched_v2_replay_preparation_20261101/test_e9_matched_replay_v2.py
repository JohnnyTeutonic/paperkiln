"""Synthetic receipts exercise the actual unchanged frozen checker, without compute."""
import contextlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile
import pytest
import e9_matched_replay_v2 as r

STAGE = Path('/mnt/c/ml_artifacts/kvq/stage-e9-matched-v2-20261101')


def rows(mean=1):
    return [{'name': name, 'positions': 8192, 'formats': {'f16': {'contexts': {
        c: {'all': {'positions': int(c), 'max_rel': .002, 'mean_rel': .00004 * mean, 'ppl_rel_diff': .00001}}
        for c in ('2048', '4096', '8192')}}}} for name in ('book', 'chat')]


def put(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def pack(raw, result):
    for n in ('gate_summary.json', 'execution.json'):
        (result / n).write_bytes((raw / n).read_bytes())
    with zipfile.ZipFile(result / 'out.zip', 'w') as z:
        for f in raw.rglob('*'):
            if f.is_file():
                z.write(f, f.relative_to(raw).as_posix())


def fixture(tmp_path, mean=1, mode='normal'):
    raw, result = tmp_path / 'raw', tmp_path / 'results'
    raw.mkdir(); result.mkdir()
    mod = r.frozen_checker(STAGE)
    (raw / 'models.sha256').write_text(''.join(f'{v}  /x/{k}\n' for k, v in mod.EXPECTED.items()))
    (raw / 'STAGING.json').write_bytes((STAGE / 'STAGING.json').read_bytes())
    for model in mod.MODELS:
        p = raw / f'eval_{model}'
        put(p / 'meta.json', {'device': 'cuda'})
        put(p / 'tf_prefill.json', rows())
        put(p / 'tf_decode.json', rows(mean))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(raw)
    summary = json.loads(buf.getvalue())
    execution = {'gate_pass': rc == 0, 'process_exit_code': rc, 'complete_marker': rc == 0}
    if rc:
        summary['errors'] += ['gate process failed, timed out, or lacks completion marker', 'registered matched criteria']
        summary['classification'] = 'FAILED'
    if mode == 'timeout':
        execution.update(gate_pass=False, process_exit_code=124, complete_marker=False)
        summary['pass'] = False
        summary['errors'].append('gate process failed, timed out, or lacks completion marker')
    if mode == 'setup':
        (raw / 'eval_qwen3/tf_decode.json').unlink()
        summary = {'pass': False, 'errors': ['missing gate summary']}
        execution = {'gate_pass': False, 'process_exit_code': None, 'complete_marker': False}
    put(raw / 'gate_summary.json', summary); put(raw / 'execution.json', execution)
    pack(raw, result)
    return raw, result


@pytest.mark.parametrize('mean,mode,want', [(1,'normal','PASS'),(1.06,'normal','COMPLETE_SCIENTIFIC_NEGATIVE'),(1,'timeout','OPERATIONAL_OR_VERDICT_NEGATIVE'),(1.06,'timeout','OPERATIONAL_OR_VERDICT_NEGATIVE'),(1,'setup','SETUP_OR_INCOMPLETE_NEGATIVE')])
def test_actual_frozen_checker_replays_and_classifies(tmp_path, mean, mode, want):
    _, result = fixture(tmp_path, mean, mode)
    report = r.replay(result, STAGE)
    assert report['classification'] == want
    assert report['overall_pass'] is (want == 'PASS')
    assert report['frozen_checker_sha256'] == r.CHECKER_SHA
    assert report['provider_calls'] == report['gpu_calls'] == 0
    if want == 'COMPLETE_SCIENTIFIC_NEGATIVE':
        assert report['complete_scientific_matrix']
        assert len(report['local_summary']['cells']) == 12
        assert all(not c['c1_mean'] for c in report['local_summary']['cells'])
        assert report['local_summary']['errors'] != json.loads((result/'gate_summary.json').read_text())['errors']


@pytest.mark.parametrize('bad', ['external_summary','external_execution','criteria','cells','flag_type','return_type','manifest'])
def test_mismatch_and_malformed_binding_refused(tmp_path,bad):
    raw,result=fixture(tmp_path)
    if bad.startswith('external'):
        p=result/('gate_summary.json' if bad=='external_summary' else 'execution.json')
        d=json.loads(p.read_text());d['changed']=True;put(p,d)
    else:
        d=json.loads((raw/'gate_summary.json').read_text());e=json.loads((raw/'execution.json').read_text())
        if bad=='criteria':d['criteria']['mean_ratio']=2
        elif bad=='cells':d['cells'][0]['prefill']['mean_rel']=.5
        elif bad=='flag_type':d['pass']=1
        elif bad=='return_type':e['process_exit_code']=False
        elif bad=='manifest':(raw/'STAGING.json').write_bytes((raw/'STAGING.json').read_bytes()+b'\n')
        put(raw/'gate_summary.json',d);put(raw/'execution.json',e);pack(raw,result)
    with pytest.raises(ValueError):r.replay(result,STAGE)


@pytest.mark.parametrize('bad',['summary_false','execution_false','missing_marker','return_code'])
def test_each_gate_predicate_required_even_with_passing_matrix(tmp_path,bad):
    raw,result=fixture(tmp_path)
    d=json.loads((raw/'gate_summary.json').read_text());e=json.loads((raw/'execution.json').read_text())
    if bad=='summary_false':d['pass']=False
    elif bad=='execution_false':e['gate_pass']=False
    elif bad=='missing_marker':e['complete_marker']=False
    elif bad=='return_code':e['process_exit_code']=1
    put(raw/'gate_summary.json',d);put(raw/'execution.json',e);pack(raw,result)
    report=r.replay(result,STAGE)
    assert not report['overall_pass'] and report['classification']=='OPERATIONAL_OR_VERDICT_NEGATIVE'


@pytest.mark.parametrize('kind',['parent','absolute','backslash','drive','dot','duplicate','collision','symlink'])
def test_unsafe_archive_inventory_refused(tmp_path,kind):
    archive=tmp_path/'bad.zip';dest=tmp_path/'extract';dest.mkdir()
    names={'parent':'../escape','absolute':'/escape','backslash':'a\\b','drive':'C:escape','dot':'a/./b'}
    with zipfile.ZipFile(archive,'w') as z:
        if kind in names:z.writestr(names[kind],b'x')
        elif kind=='duplicate':
            z.writestr('x',b'a')
            with pytest.warns(UserWarning):z.writestr('x',b'b')
        elif kind=='collision':z.writestr('x',b'a');z.writestr('x/y',b'b')
        else:
            info=zipfile.ZipInfo('link');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,'target')
    with pytest.raises(ValueError):r.unpack(archive,dest)
    assert list(dest.iterdir())==[]


@pytest.mark.parametrize('bound',['MAX_MEMBERS','MAX_MEMBER_BYTES','MAX_TOTAL_BYTES'])
def test_archive_size_bounds(tmp_path,monkeypatch,bound):
    a=tmp_path/'a.zip';dest=tmp_path/'extract';dest.mkdir()
    with zipfile.ZipFile(a,'w') as z:z.writestr('a',b'123');z.writestr('b',b'123')
    monkeypatch.setattr(r,bound,1 if bound!='MAX_TOTAL_BYTES' else 5)
    with pytest.raises(ValueError):r.unpack(a,dest)
    assert not list(dest.iterdir())


def test_receipt_immutable_and_no_fake_scientific_pass(tmp_path):
    _,result=fixture(tmp_path,1.06)
    out=tmp_path/'replay.json'
    cmd=[sys.executable,r.__file__,'--results',str(result),'--stage',str(STAGE),'--receipt',str(out)]
    done=subprocess.run(cmd,capture_output=True,text=True)
    assert done.returncode==1 and json.loads(out.read_text())['classification']=='COMPLETE_SCIENTIFIC_NEGATIVE'
    original=out.read_bytes()
    again=subprocess.run(cmd,capture_output=True,text=True)
    assert again.returncode!=0 and out.read_bytes()==original


def test_duplicate_json_key_refused(tmp_path):
    raw,result=fixture(tmp_path)
    (raw/'gate_summary.json').write_text('{"pass":true,"pass":false}')
    pack(raw,result)
    with pytest.raises(ValueError,match='duplicate JSON'):r.replay(result,STAGE)


def test_boolean_substitution_in_transported_cells_refused(tmp_path):
    raw,result=fixture(tmp_path)
    d=json.loads((raw/'gate_summary.json').read_text())
    d['cells'][0]['c1_mean']=1
    put(raw/'gate_summary.json',d);pack(raw,result)
    with pytest.raises(ValueError,match='criteria/cells'):r.replay(result,STAGE)


@pytest.mark.parametrize('mean',[1,1.06])
def test_unexplained_wrapper_error_cannot_be_pass_or_clean_scientific_negative(tmp_path,mean):
    raw,result=fixture(tmp_path,mean)
    d=json.loads((raw/'gate_summary.json').read_text());d['errors'].append('unexplained fatal failure')
    put(raw/'gate_summary.json',d);pack(raw,result)
    assert r.replay(result,STAGE)['classification']=='OPERATIONAL_OR_VERDICT_NEGATIVE'


def test_missing_raw_criterion_error_is_not_clean_scientific_negative(tmp_path):
    raw,result=fixture(tmp_path,1.06)
    d=json.loads((raw/'gate_summary.json').read_text());d['errors']=['registered matched criteria']
    put(raw/'gate_summary.json',d);pack(raw,result)
    assert r.replay(result,STAGE)['classification']=='OPERATIONAL_OR_VERDICT_NEGATIVE'


def test_timestamp_valid_unverified_bytecode_is_never_executed(tmp_path):
    import importlib.util,py_compile,os
    stage=tmp_path/'stage';stage.mkdir()
    for name in ('e9m_check.py','STAGING.json'):(stage/name).write_bytes((STAGE/name).read_bytes())
    source=stage/'e9m_check.py';raw=source.read_bytes();stamp=int(source.stat().st_mtime)
    malicious=b"raise RuntimeError('unverified bytecode ran')\n".replace(b'\\n',b'\n')
    malicious=malicious+b'#'*(len(raw)-len(malicious))
    source.write_bytes(malicious);os.utime(source,(stamp,stamp));py_compile.compile(str(source),doraise=True)
    source.write_bytes(raw);os.utime(source,(stamp,stamp))
    # CPython normally accepts this timestamp/size-valid cache; our loader must ignore it.
    with pytest.raises(RuntimeError,match='unverified bytecode'):
        spec=importlib.util.spec_from_file_location('unsafe_cache_control',source)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    checker=r.frozen_checker(stage)
    assert checker.PINNED_COMMIT=='3ada969788f4d5ea65f5b81b29c22c30b7f25c97'
