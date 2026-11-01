import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import sys
import pytest
from test_host_read_boundary_gate_v1 import gate,fixture,modify

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/colab_chat'))
spec=importlib.util.spec_from_file_location('packet',ROOT/'tools/colab_chat/replay_host_read_boundary_packet_v1.py')
packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
HEAD='a'*40


def make_packet(folder,negative=False):
    fixture(folder)
    if negative:
        modify(folder,'legacy_fixed_active_b10',lambda rows:rows[0].update({'pass':False,'error':'independent numerical reference differs'}))
        (folder/'legacy_fixed_active_b10.rc').write_text('1')
    result=gate.replay(folder)
    blobs={name:(ROOT/name).read_bytes() for name in packet.SOURCE_FILES}
    hardware={'source_commit':HEAD,'hardware':['NVIDIA L4'],'source_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in blobs.items()}}
    execution={'kind':'HOST_READ_BOUNDARY_EXECUTION_V1','expected_source':HEAD,'diagnostic_package_complete':True,'gate_pass':result['pass'],'operational_error':None,'binary_sha256':{name:'b'*64 for name in ['gemm_fallback_old','gemm_fallback_fixed','legacy_matmul_old','legacy_matmul_fixed']}}
    for name,data in [('execution.json',execution),('gate_summary.json',result),('source_hardware.json',hardware)]:
        (folder/name).write_text(json.dumps(data))
    (folder/'configure.txt').write_text('configuration log')
    (folder/'build.txt').write_text('build log')
    return {f.name:f.read_bytes() for f in folder.iterdir()},blobs


def archive(files,extra=None):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name,raw in files.items():
            item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
        if extra:
            item,data=extra
            tar.addfile(item,io.BytesIO(data) if data is not None else None)
    return stream.getvalue()


@pytest.mark.parametrize('negative',[False,True])
def test_complete_packet_replayed_from_frozen_source(tmp_path,negative):
    files,blobs=make_packet(tmp_path,negative)
    r=packet.verify(archive(files),HEAD,blobs)
    assert r['complete'] is True and r['pass'] is (not negative) and r['rows']==90


@pytest.mark.parametrize('name,field,value',[
    ('execution.json','gate_pass',False),
    ('execution.json','gate_pass',1),
    ('execution.json','diagnostic_package_complete',False),
    ('execution.json','operational_error','timeout'),
    ('execution.json','expected_source','c'*40),
    ('execution.json','binary_sha256',{}),
    ('source_hardware.json','hardware',['NVIDIA T4']),
    ('source_hardware.json','source_commit','c'*40),
    ('source_hardware.json','source_sha256',{}),
    ('gate_summary.json','pass',False),
    ('gate_summary.json','pass',1),
    ('gate_summary.json','complete',1),
    ('gate_summary.json','rows',90.0),
])
def test_transported_verdict_or_provenance_drift_rejected(tmp_path,name,field,value):
    files,blobs=make_packet(tmp_path);r=json.loads(files[name]);r[field]=value;files[name]=json.dumps(r).encode()
    with pytest.raises(ValueError):packet.verify(archive(files),HEAD,blobs)


@pytest.mark.parametrize('name',['../escape','/absolute','unknown','execution.json'])
def test_unknown_paths_and_duplicate_members(tmp_path,name):
    files,blobs=make_packet(tmp_path);item=tarfile.TarInfo(name);item.size=1
    with pytest.raises(ValueError):packet.verify(archive(files,(item,b'x')),HEAD,blobs)


@pytest.mark.parametrize('kind',[tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE])
def test_nonregular_archive_members(tmp_path,kind):
    files,blobs=make_packet(tmp_path);del files['execution.json']
    item=tarfile.TarInfo('execution.json');item.type=kind;item.linkname='elsewhere'
    with pytest.raises(ValueError):packet.verify(archive(files,(item,None)),HEAD,blobs)


def test_missing_member(tmp_path):
    files,blobs=make_packet(tmp_path);del files['build.txt']
    with pytest.raises(ValueError):packet.verify(archive(files),HEAD,blobs)


def test_limits_include_decompressed_metadata(tmp_path,monkeypatch):
    files,blobs=make_packet(tmp_path);raw=archive(files)
    monkeypatch.setattr(packet,'MAX_EXPANDED_ARCHIVE',100)
    with pytest.raises(ValueError):packet.verify(raw,HEAD,blobs)
    monkeypatch.setattr(packet,'MAX_COMPRESSED',10)
    with pytest.raises(ValueError):packet.verify(raw,HEAD,blobs)


def test_source_blob_mutation_rejected(tmp_path):
    files,blobs=make_packet(tmp_path);blobs['src/device.cpp']+=b'\n'
    with pytest.raises(ValueError):packet.verify(archive(files),HEAD,blobs)


def test_duplicate_json_key(tmp_path):
    files,blobs=make_packet(tmp_path)
    files['execution.json']=files['execution.json'].replace(b'"gate_pass": true',b'"gate_pass": true, "gate_pass": false')
    with pytest.raises(ValueError):packet.verify(archive(files),HEAD,blobs)
