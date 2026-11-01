import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gate', ROOT/'tools/colab_chat/host_read_boundary_gate_v1.py')
gate = importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def fixture(folder):
    for leg in gate.legs():
        cases = gate.GEMM_CASES if leg['family']=='gemm' else gate.LEGACY_CASES
        rows = []
        for case in sorted(cases):
            row = {'case':case,'pass':leg['expected_pass'],'actual_cuda':True,'synthetic_authority':False}
            if not row['pass']:row['error']='independent numerical reference differs'
            rows.append(gate.PREFIX[leg['family']]+json.dumps(row)+'\n')
        (folder/(leg['name']+'.txt')).write_text(''.join(rows))
        (folder/(leg['name']+'.rc')).write_text('0\n' if leg['expected_pass'] else '1\n')


def modify(folder, name, change):
    path=folder/(name+'.txt');prefix=gate.PREFIX[name.split('_')[0]]
    rows=[json.loads(line[len(prefix):]) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text(''.join(prefix+json.dumps(row)+'\n' for row in rows))


def test_complete_matrix(tmp_path):
    fixture(tmp_path);r=gate.replay(tmp_path)
    assert r['pass'] is True and r['rows']==90 and r['legs']==18


def test_complete_fixed_numerical_negative_is_not_incomplete(tmp_path):
    fixture(tmp_path)
    modify(tmp_path,'legacy_fixed_active_b10',lambda rows:rows[0].update({'pass':False,'error':'independent numerical reference differs'}))
    (tmp_path/'legacy_fixed_active_b10.rc').write_text('1\n')
    r=gate.replay(tmp_path)
    assert r['complete'] is True and r['pass'] is False and len(r['failures'])==1


def test_old_unexpected_pass_does_not_establish_causal_proof(tmp_path):
    fixture(tmp_path)
    def passing(rows):
        for row in rows:row.update({'pass':True});row.pop('error')
    modify(tmp_path,'legacy_old_active_b10',passing)
    (tmp_path/'legacy_old_active_b10.rc').write_text('0\n')
    assert gate.replay(tmp_path)['pass'] is False


@pytest.mark.parametrize('change',[
    lambda rows:rows.pop(),
    lambda rows:rows.append(dict(rows[0])),
    lambda rows:rows[0].update({'case':rows[1]['case']}),
    lambda rows:rows[0].update({'actual_cuda':False}),
    lambda rows:rows[0].update({'actual_cuda':1}),
    lambda rows:rows[0].update({'synthetic_authority':True}),
    lambda rows:rows[0].update({'pass':1}),
    lambda rows:rows[0].update({'pass':False,'error':'CUDA allocation failed'}),
    lambda rows:rows[0].update({'error':'unexpected'}),
    lambda rows:rows[0].update({'unregistered':'field'}),
    lambda rows:rows[0].update({'case':['not','a','name']}),
])
def test_incomplete_or_operational_failure_rejected(tmp_path,change):
    fixture(tmp_path);modify(tmp_path,'legacy_fixed_active_b10',change)
    with pytest.raises(ValueError):gate.replay(tmp_path)


@pytest.mark.parametrize('rc',['2','-9','true','0.0'])
def test_bad_exit_status(tmp_path,rc):
    fixture(tmp_path);(tmp_path/'legacy_fixed_active_b10.rc').write_text(rc)
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_exit_status_disagreement(tmp_path):
    fixture(tmp_path);(tmp_path/'legacy_fixed_active_b10.rc').write_text('1')
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_duplicate_keys_rejected(tmp_path):
    fixture(tmp_path);p=tmp_path/'legacy_fixed_active_b10.txt'
    p.write_text(p.read_text().replace('"pass": true','"pass": true, "pass": false',1))
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_missing_and_symlinked_raw_evidence(tmp_path):
    fixture(tmp_path);p=tmp_path/'legacy_fixed_active_b10.txt';raw=p.read_bytes();p.unlink()
    with pytest.raises(ValueError):gate.replay(tmp_path)
    other=tmp_path/'other';other.write_bytes(raw);p.symlink_to(other)
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_truncated_log(tmp_path):
    fixture(tmp_path);p=tmp_path/'legacy_fixed_active_b10.txt';p.write_text(p.read_text()[:-5])
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_gemm_real_cuda_restores_zero_only(tmp_path):
    fixture(tmp_path)
    modify(tmp_path,'gemm_fixed_active',lambda rows:rows[0].update({'restores':0}))
    assert gate.replay(tmp_path)['pass'] is True
    modify(tmp_path,'gemm_fixed_active',lambda rows:rows[0].update({'restores':True}))
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_nonfinite_unused_json_rejected(tmp_path):
    fixture(tmp_path);p=tmp_path/'gemm_fixed_active.txt'
    p.write_text(p.read_text().replace('"pass": true','"restores": NaN, "pass": true',1))
    with pytest.raises(ValueError):gate.replay(tmp_path)


def test_vm_driver_refuses_local_execution_before_gpu_probe(tmp_path):
    import subprocess,sys
    driver=ROOT/'tools/colab_chat/run_host_read_boundaries_l4_v1.py'
    done=subprocess.run([sys.executable,str(driver),str(tmp_path/'out'),'--expected-source','a'*40],capture_output=True,text=True)
    assert done.returncode!=0 and 'only in an admitted remote' in done.stderr
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('raw',[b'a'*40,b'a'*40+b'\n'])
def test_uploaded_full_source_pin(tmp_path,monkeypatch,raw):
    monkeypatch.syspath_prepend(str(ROOT/'tools/colab_chat'))
    from run_host_read_boundaries_l4_v1 import read_source_pin
    p=tmp_path/'HEAD';p.write_bytes(raw)
    assert read_source_pin(p)=='a'*40


@pytest.mark.parametrize('raw',[b'',b'a'*39,b'a'*41,b'a'*40+b'\r\n',b'F'*40,b'g'*40,b'a'*40+b'\nextra',b'a'*39+b' '])
def test_malformed_uploaded_source_pin(tmp_path,monkeypatch,raw):
    monkeypatch.syspath_prepend(str(ROOT/'tools/colab_chat'))
    from run_host_read_boundaries_l4_v1 import read_source_pin
    p=tmp_path/'HEAD';p.write_bytes(raw)
    with pytest.raises(ValueError):read_source_pin(p)


def test_source_pin_symlink_refused(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools/colab_chat'))
    from run_host_read_boundaries_l4_v1 import read_source_pin
    p=tmp_path/'HEAD';p.write_bytes(b'a'*40);link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):read_source_pin(link)
