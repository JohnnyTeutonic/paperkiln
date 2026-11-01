"""Pinned operational boot recovery; delegates all campaign bounds to original admission."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outside_boot(source):
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'boot_launch_code']
    if len(matches) != 1:
        raise ValueError('expected exactly one boot-launch function')
    n = matches[0]
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:n.lineno-1] + lines[n.end_lineno:])


def validate(record):
    if record['kind'] != 'ROOT_SFULL_OPERATIONAL_BOOT_RECOVERY_V1' or not record['adopted']:
        raise ValueError('inactive boot recovery')
    for path, digest in record['source_pins'].items():
        if sha(path) != digest:
            raise ValueError('recovery source differs: ' + path)
    old_guard = record['original_guard']
    old_admission = record['original_admission']
    if old_guard not in record['source_pins'] or old_admission not in record['source_pins']:
        raise ValueError('original admission must remain pinned')
    spec = importlib.util.spec_from_file_location('original_sfull_root_guard', old_guard)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = json.loads(Path(old_admission).read_text())
    argv = module.validate(original)
    base = Path(argv[1])
    modified = Path(record['modified_runner'])
    helper = record['recovery_helper']
    if str(base) not in record['source_pins'] or str(modified) not in record['source_pins'] or helper not in record['source_pins']:
        raise ValueError('runner/helper binding missing')
    if outside_boot(base.read_text()) != outside_boot(modified.read_text()):
        raise ValueError('change extends beyond boot-launch function')
    if argv[argv.index('--state-dir')+1] != record['state_dir']:
        raise ValueError('campaign state differs')
    if argv[argv.index('--art')+1] != record['artifact_dir']:
        raise ValueError('campaign artifacts differ')
    if original['deadline_utc'] != record['deadline_utc']:
        raise ValueError('deadline differs')
    if original['per_seed_cap_usd'] != 7 or original['aggregate_cap_usd'] != 21:
        raise ValueError('caps differ')
    if original['source_control_commit'] != '1af22cfd66f95cd40dcd98cf4ffae57cb3de4c61':
        raise ValueError('control revision differs')
    if original['numerical_commit'] != '1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1':
        raise ValueError('numerical revision differs')
    argv[1] = str(modified)
    return argv


def main():
    admission = Path(sys.argv[1])
    record = json.loads(admission.read_text())
    argv = validate(record)
    if os.environ.get('S_FULL_OWNER_PROBE') == '1':
        print('OWNER_PROBE_VALIDATED_NO_PROVIDER_CALLS', flush=True)
        time.sleep(30)
        return
    # Sibling imports remain the original frozen modules, not publication HEAD.
    original_runner = json.loads(Path(record['original_admission']).read_text())['runner_argv'][1]
    sys.path.insert(0, str(Path(original_runner).parent))
    import runpod_api
    import runpod_transport
    runpod_api.MAX_PODS = 1
    if runpod_transport.api is not runpod_api:
        raise RuntimeError('provider module identity differs')
    sys.argv = argv[1:]
    runpy.run_path(argv[1], run_name='__main__')


if __name__ == '__main__':
    main()
