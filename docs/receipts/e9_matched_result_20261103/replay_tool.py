"""Offline replay of returned E9 matched v2 receipts; never launches compute."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import types
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

CHECKER_SHA = '45885f2b55b6c99c1fca2c478cae7c0d5fe72d661477072b303bc551ea89cc8e'
STAGING_SHA = '33cb93cb56b7ed5f49be96c8404e714d1cb152bd1b2d33a826e9a861270eb221'
MAX_MEMBERS = 4096
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def strict_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError('duplicate JSON key: ' + k)
        d[k] = v
    return d


def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=strict_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))


def regular(path):
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError('regular non-symlink file required')
    return p


def unpack(archive, destination):
    """Validate the entire inventory before writing bounded regular members."""
    destination = Path(destination)
    records = {}
    with zipfile.ZipFile(regular(archive)) as z:
        entries = z.infolist()
        if len(entries) > MAX_MEMBERS:
            raise ValueError('archive member count exceeds bound')
        total = 0
        names = set()
        files = set()
        for entry in entries:
            name = entry.filename
            rel = PurePosixPath(name)
            parts = name.rstrip('/').split('/')
            mode = entry.external_attr >> 16
            if (not name or '\\' in name or '\x00' in name or rel.is_absolute()
                    or ':' in name or any(x in ('', '.', '..') for x in parts)):
                raise ValueError('unsafe archive path')
            key = '/'.join(parts)
            if key in names:
                raise ValueError('duplicate archive member')
            names.add(key)
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('nonregular archive member')
            if entry.flag_bits & 1:
                raise ValueError('encrypted archive member')
            if not entry.is_dir():
                files.add(key)
            if entry.file_size > MAX_MEMBER_BYTES:
                raise ValueError('archive member exceeds bound')
            total += entry.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError('archive total exceeds bound')
        if any('/'.join(name.split('/')[:i]) in files for name in names for i in range(1, len(name.split('/')))):
            raise ValueError('archive file/directory collision')
        for entry in entries:
            if entry.is_dir():
                continue
            out = destination / entry.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            remaining = entry.file_size
            digest = hashlib.sha256()
            with z.open(entry) as source, out.open('xb') as target:
                while True:
                    block = source.read(min(1024**2, remaining + 1))
                    if not block:
                        break
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError('archive data exceeds declared size')
                    digest.update(block)
                    target.write(block)
            if remaining:
                raise ValueError('archive data shorter than declared size')
            records[entry.filename] = {'bytes': entry.file_size, 'sha256': digest.hexdigest()}
    return records


def frozen_checker(stage):
    stage = Path(stage)
    checker, manifest = regular(stage / 'e9m_check.py'), regular(stage / 'STAGING.json')
    source_bytes, manifest_bytes = checker.read_bytes(), manifest.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != CHECKER_SHA or hashlib.sha256(manifest_bytes).hexdigest() != STAGING_SHA:
        raise ValueError('frozen v2 checker or manifest differs')
    # Execute the verified bytes, never a timestamp-valid unverified .pyc.
    module = types.ModuleType('e9_frozen_v2_replay')
    module.__file__ = str(checker)
    exec(compile(source_bytes, str(checker), 'exec'), module.__dict__)
    return module


def replay(results, stage):
    results, stage = Path(results), Path(stage)
    checker = frozen_checker(stage)
    for name in ('out.zip', 'gate_summary.json', 'execution.json'):
        regular(results / name)
    input_hashes = {n: sha(results / n) for n in ('out.zip', 'gate_summary.json', 'execution.json')}
    external_summary, execution = load(results / 'gate_summary.json'), load(results / 'execution.json')
    if type(external_summary.get('pass')) is not bool or type(execution.get('gate_pass')) is not bool:
        raise ValueError('both verdict flags must be explicit booleans')
    if type(execution.get('complete_marker')) is not bool:
        raise ValueError('completion marker must be explicit boolean')
    transported_errors = external_summary.get('errors')
    if not isinstance(transported_errors, list) or any(not isinstance(x, str) for x in transported_errors):
        raise ValueError('explicit string error inventory required')
    rc = execution.get('process_exit_code')
    if rc is not None and type(rc) is not int:
        raise ValueError('process exit code must be integer or null')
    with tempfile.TemporaryDirectory(prefix='e9-v2-replay-') as scratch:
        scratch = Path(scratch)
        inventory = unpack(results / 'out.zip', scratch)
        for name in ('gate_summary.json', 'execution.json'):
            if sha(regular(scratch / name)) != sha(results / name):
                raise ValueError('external and archived receipt differ: ' + name)
        # Missing manifest is a setup/incomplete receipt, never a scientific verdict.
        manifest_present = (scratch / 'STAGING.json').is_file()
        if manifest_present and sha(scratch / 'STAGING.json') != STAGING_SHA:
            raise ValueError('returned staging manifest differs from frozen v2')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            checker_rc = checker.main(scratch)
        local = json.loads(output.getvalue(), object_pairs_hook=strict_pairs)
    complete_matrix = (manifest_present and len(local.get('cells', [])) == 12
                       and not any(str(x).startswith('invalid receipts:') for x in local.get('errors', [])))
    comparable = 'criteria' in external_summary and 'cells' in external_summary
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    matrix_agrees = comparable and canonical(external_summary['criteria']) == canonical(local['criteria']) and canonical(external_summary['cells']) == canonical(local['cells'])
    if comparable and not matrix_agrees:
        raise ValueError('transported criteria/cells differ from frozen checker replay')
    # Wrapper errors are deliberately not compared: it appends execution failures.
    process_complete = rc == 0 and execution['complete_marker'] is True
    overall = (complete_matrix and matrix_agrees and checker_rc == 0 and local['pass'] is True
               and external_summary['pass'] is True and execution['gate_pass'] is True and process_complete and not transported_errors)
    known_wrapper_errors = {'gate process failed, timed out, or lacks completion marker', 'registered matched criteria'}
    raw_errors = local.get('errors', [])
    error_inventory_agrees = (all(x in transported_errors for x in raw_errors)
                              and all(x in raw_errors or x in known_wrapper_errors for x in transported_errors))
    criterion_failure_exit = (error_inventory_agrees and rc == 1 and execution['complete_marker'] is False
                             and any('registered matched criteria' in str(x) for x in external_summary.get('errors', [])))
    if overall:
        classification = 'PASS'
    elif complete_matrix and matrix_agrees and local['pass'] is False and external_summary['pass'] is False and execution['gate_pass'] is False and criterion_failure_exit:
        classification = 'COMPLETE_SCIENTIFIC_NEGATIVE'
    elif not complete_matrix:
        classification = 'SETUP_OR_INCOMPLETE_NEGATIVE'
    else:
        classification = 'OPERATIONAL_OR_VERDICT_NEGATIVE'
    if any(sha(results / n) != h for n, h in input_hashes.items()):
        raise ValueError('returned input changed during replay')
    return {'kind': 'LOCAL_E9_MATCHED_V2_REPLAY', 'classification': classification,
            'overall_pass': overall, 'complete_scientific_matrix': complete_matrix,
            'transported_matrix_agrees': bool(matrix_agrees), 'process_completed_successfully': process_complete,
            'execution': execution, 'transported_gate_pass': external_summary['pass'],
            'local_checker_exit_code': checker_rc, 'local_summary': local,
            'frozen_checker_sha256': CHECKER_SHA, 'frozen_staging_sha256': STAGING_SHA,
            'replay_helper_sha256': sha(__file__),
            'returned_file_sha256': input_hashes,
            'archive_members': inventory, 'provider_calls': 0, 'gpu_calls': 0,
            'automatic_production_adoption': False,
            'qualification': 'Replays reported raw metrics and recorded model/input hashes; model binaries and actual device execution are not independently rerun.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, type=Path)
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--receipt', required=True, type=Path)
    args = p.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise ValueError('replay receipt is immutable; choose a new path')
    result = replay(args.results, args.stage)
    with args.receipt.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'classification': result['classification'], 'overall_pass': result['overall_pass'], 'receipt': str(args.receipt)}))
    return 0 if result['overall_pass'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
