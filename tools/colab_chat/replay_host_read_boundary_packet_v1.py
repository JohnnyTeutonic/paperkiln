#!/usr/bin/env python3
"""Bounded local replay of a returned, separately admitted host-read packet."""
from pathlib import Path
import argparse
import hashlib
import gzip
import io
import json
import subprocess
import tarfile
import tempfile

from host_read_boundary_gate_v1 import legs, unique

SOURCE_FILES = ['src/device.cpp', 'tests/test_gemm_fallback_materialization.cpp',
                'tests/gemm_fallback_baseline.cpp', 'tests/test_legacy_matmul_materialization.cpp',
                'docs/receipts/legacy_matmul_materialization_20261102/baseline_device.cpp',
                'tools/colab_chat/host_read_boundary_gate_v1.py',
                'tools/colab_chat/run_host_read_boundaries_l4_v1.py', 'CMakeLists.txt']
MAX_COMPRESSED = 16 * 1024 * 1024
MAX_EXPANDED_ARCHIVE = 64 * 1024 * 1024 + 128 * 1024


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def members(raw, leg_spec=None):
    if len(raw) > MAX_COMPRESSED:
        raise ValueError('compressed packet exceeds 16 MiB')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(MAX_EXPANDED_ARCHIVE+1)
    if len(expanded) > MAX_EXPANDED_ARCHIVE:
        raise ValueError('expanded tar stream exceeds bounds, including metadata')
    expected = {leg['name']+suffix for leg in (legs() if leg_spec is None else leg_spec) for suffix in ['.txt', '.rc']}
    expected |= {'source_hardware.json', 'execution.json', 'gate_summary.json', 'configure.txt', 'build.txt'}
    result, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(expanded), mode='r:') as archive:
        for item in archive:
            if item.name not in expected or item.name in result or not item.isfile():
                raise ValueError('unexpected, duplicate or nonregular packet member')
            total += item.size
            if item.size < 0 or item.size > 16*1024*1024 or total > 64*1024*1024:
                raise ValueError('expanded packet exceeds limits')
            with archive.extractfile(item) as stream:
                data = stream.read(item.size+1)
            if len(data) != item.size:
                raise ValueError('incomplete packet member')
            result[item.name] = data
    if set(result) != expected:
        raise ValueError('incomplete packet inventory')
    return result


def verify(raw, expected_source, blobs):
    if type(expected_source) is not str or len(expected_source)!=40 or any(c not in '0123456789abcdef' for c in expected_source):
        raise ValueError('full expected source identifier required')
    if set(blobs) != set(SOURCE_FILES):
        raise ValueError('source blob inventory differs')
    namespace = {'__name__':'frozen_host_read_checker','__file__':'host_read_boundary_gate_v1.py'}
    # Execute the prospectively pinned Git blob, never a transported source or pyc.
    checker = blobs['tools/colab_chat/host_read_boundary_gate_v1.py']
    exec(compile(checker,'frozen_host_read_checker','exec'),namespace)
    files = members(raw, namespace['legs']())
    hardware = strict_json(files['source_hardware.json'])
    expected_hashes = {name:hashlib.sha256(data).hexdigest() for name,data in blobs.items()}
    if hardware != {'source_commit':expected_source,'hardware':['NVIDIA L4'],'source_sha256':expected_hashes}:
        raise ValueError('worker source/hardware binding differs')
    execution = strict_json(files['execution.json'])
    if execution.get('kind')!='HOST_READ_BOUNDARY_EXECUTION_V1' or execution.get('expected_source')!=expected_source or execution.get('diagnostic_package_complete') is not True or execution.get('operational_error') is not None:
        raise ValueError('execution is operationally incomplete')
    binary_hashes = execution.get('binary_sha256')
    targets = {'gemm_fallback_old','gemm_fallback_fixed','legacy_matmul_old','legacy_matmul_fixed'}
    if not isinstance(binary_hashes,dict) or set(binary_hashes)!=targets or any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in binary_hashes.values()):
        raise ValueError('binary hash inventory invalid')
    if type(execution.get('gate_pass')) is not bool:
        raise ValueError('execution gate flag is not boolean')
    with tempfile.TemporaryDirectory(prefix='host_boundary_replay_') as td:
        folder=Path(td)
        for name,data in files.items():
            (folder/name).write_bytes(data)
        local=namespace['replay'](folder)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if canonical(local) != canonical(strict_json(files['gate_summary.json'])) or execution['gate_pass'] is not local['pass']:
        raise ValueError('transported flags/verdict differ from frozen local replay')
    return {'kind':'LOCAL_HOST_READ_PACKET_REPLAY_V1','complete':True,'pass':local['pass'],
            'outcome':'PASS' if local['pass'] else 'COMPLETE_MATRIX_NEGATIVE',
            'rows':local['rows'],'source_commit':expected_source,'packet_sha256':hashlib.sha256(raw).hexdigest(),
            'failures':local['failures'],'limits':'Source/hardware/binary identities are hash-bound worker attestations; Python does not recompute GPU output arithmetic or prove remote attestation against compromise.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet',type=Path)
    parser.add_argument('--expected-source',required=True)
    parser.add_argument('--source-repo',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    with args.packet.open('rb') as stream:
        raw=stream.read(16*1024*1024+1)
    blobs={name:subprocess.check_output(['git','-C',str(args.source_repo),'show',args.expected_source+':'+name]) for name in SOURCE_FILES}
    result=verify(raw,args.expected_source,blobs)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))
    return 0 if result['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
