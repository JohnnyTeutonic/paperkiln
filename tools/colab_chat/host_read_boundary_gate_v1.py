#!/usr/bin/env python3
"""Replay the prospective 90-row actual-CUDA host-boundary matrix.

Read-only: no compilation, model, device, provider or API calls. The admitted VM
driver owns source/hardware/binary provenance; this checker checks its raw legs.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

GEMM_CASES = {'NN', 'NT', 'TN', 'TT', 'saved_parent_backward'}
LEGACY_CASES = {'both_rectangular', 'left_only', 'right_only', 'dot', 'odd_rectangular'}
PREFIX = {'gemm': 'GEMM_FALLBACK_JSON ', 'legacy': 'LEGACY_MATMUL_JSON '}


def legs():
    for family, variant, transition in itertools.product(['gemm', 'legacy'], ['old', 'fixed'], ['active', 'cpu', 'stepoff']):
        for b1 in ([False, True] if family == 'legacy' else [False]):
            name = f'{family}_{variant}_{transition}' + (f'_b1{int(b1)}' if family == 'legacy' else '')
            yield {'name': name, 'family': family, 'variant': variant, 'transition': transition, 'b1': b1,
                   'expected_pass': variant == 'fixed' or (family == 'gemm' and transition == 'active')}


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read_regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing/nonregular raw leg: ' + path.name)
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('oversized raw leg')
    return raw


def replay(folder):
    folder = Path(folder)
    inventory, failures, total = {}, [], 0
    for leg in legs():
        log = read_regular(folder / (leg['name'] + '.txt'))
        raw_rc = read_regular(folder / (leg['name'] + '.rc'))
        try:
            rc_text = raw_rc.decode('ascii').strip()
            if rc_text not in ('0', '1'):
                raise ValueError('non-scientific process status')
            rc = int(rc_text)
            rows = [json.loads(line[len(PREFIX[leg['family']]):], object_pairs_hook=unique,
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
                    for line in log.decode('utf-8').splitlines() if line.startswith(PREFIX[leg['family']])]
        except (UnicodeError, ValueError) as error:
            raise ValueError(leg['name'] + ': invalid/incomplete execution') from error
        expected_cases = GEMM_CASES if leg['family'] == 'gemm' else LEGACY_CASES
        if len(rows) != 5 or any(not isinstance(row, dict) or type(row.get('case')) is not str for row in rows) or {row.get('case') for row in rows} != expected_cases:
            raise ValueError(leg['name'] + ': incomplete/duplicate case inventory')
        for row in rows:
            required = {'case', 'pass', 'actual_cuda', 'synthetic_authority'}
            optional = {'error'} | ({'restores'} if leg['family'] == 'gemm' else set())
            if not required <= set(row) or not set(row) <= required | optional:
                raise ValueError('unexpected raw result fields')
            if 'restores' in row and (type(row['restores']) is not int or row['restores'] != 0):
                raise ValueError('synthetic restoration count in CUDA result')
            if row.get('actual_cuda') is not True or row.get('synthetic_authority') is not False or type(row.get('pass')) is not bool:
                raise ValueError(leg['name'] + ': actual CUDA provenance flags differ')
            if row['pass']:
                if 'error' in row:
                    raise ValueError('passing row has an error')
            elif row.get('error') != 'independent numerical reference differs':
                raise ValueError(leg['name'] + ': setup/runtime failure is not numerical evidence')
            if row['pass'] is not leg['expected_pass']:
                failures.append({'leg': leg['name'], 'case': row['case'], 'expected_pass': leg['expected_pass'], 'actual_pass': row['pass']})
        if rc != (0 if all(row['pass'] for row in rows) else 1):
            raise ValueError(leg['name'] + ': exit status and case results disagree')
        inventory[leg['name']] = {'log_sha256': hashlib.sha256(log).hexdigest(), 'rc_sha256': hashlib.sha256(raw_rc).hexdigest()}
        total += len(rows)
    return {'kind': 'HOST_READ_BOUNDARY_MATRIX_V1', 'complete': True, 'pass': not failures,
            'rows': total, 'legs': len(inventory), 'failures': failures, 'inventory': inventory,
            'scope': 'GEMM fallback 30 rows and legacy Matrix matmul 60 rows; C++ checks every finite output against an independent double reference. This checker does not recompute those numeric outputs or prove source/hardware provenance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.folder)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': result['complete'], 'pass': result['pass'], 'rows': result['rows']}))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
