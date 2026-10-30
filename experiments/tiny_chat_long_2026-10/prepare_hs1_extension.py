"""Build additive HS1 seed-4/5 jobs from the exact registered core commit; never launch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PIN = '3dc818b2476b0957b1337b4c9af22474b76350b5'
REL = 'experiments/tiny_chat_long_2026-10'
CELLS = ('B', 'Sgist')
SEEDS = (4, 5)


def frozen(path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:{path}'])


def build(cell, seed):
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError('Only registered HS1 cells B/Sgist and extension seeds 4/5')
    source = f'tools/colab_chat/carry12_{cell}_r1.json'
    old = json.loads(frozen(source))
    keep = {'cuda_gate', 'data', 'gate', f'pt_c12_{cell}_r1', f'post_c12_{cell}_r1'}
    doc = copy.deepcopy(old)
    doc['jobs'] = [j for j in doc['jobs'] if j['name'] in keep]
    path = f'tools/colab_chat/carry12_hs1_{cell}_r{seed}.json'
    # Rename only seed-dependent model/job/output identifiers. Corpus seed stays 1.
    doc = json.loads(json.dumps(doc).replace(f'_{cell}_r1', f'_{cell}_r{seed}'))
    for job in doc['jobs']:
        if job['name'] == 'gate':
            i = job['cmd'].index('--jobs')
            job['cmd'][i + 1] = path
        if job['name'] == f'pt_c12_{cell}_r{seed}':
            job['spec']['train']['seed'] = seed
        if job['name'] == f'post_c12_{cell}_r{seed}':
            i = job['cmd'].index('--seed')
            job['cmd'][i + 1] = str(seed)
    doc['_comment'] = (f'Registered HS1 extension only, {cell} seed {seed}. Derived from {PIN}:{source}; '
                       'only seed-dependent identifiers/training seeds and gate jobs path change; '
                       'B omits unrelated D/TS/W512/chat jobs. Original data recipe including corpus seed 1, '
                       'all CUDA/data gates and post outputs remain unchanged. No new selection or S-full. '
                       'RunPod L4, cumulative cap $6, 80GB volume, max-hours 23.5; launch through existing watchdog only.')
    assert len(doc['jobs']) == 5 and {j['name'] for j in doc['jobs']} == {
        'cuda_gate', 'data', 'gate', f'pt_c12_{cell}_r{seed}', f'post_c12_{cell}_r{seed}'}
    return path, doc, source


def trigger(report):
    core = report['carry12']
    if report['partial'] is not False or core['seeds'] != [1, 2, 3] or core['missing']:
        raise ValueError('complete original three-seed report required')
    h = core['HS1_Sgist_minus_B']
    if (h['verdict'] != 'NOT_RESOLVED' or h['extend_to_5_seeds'] is not True
            or len(h['per_seed']) != 3 or not all(x < 0 for x in h['per_seed'])
            or not all(core['validity_gate'].values()) or core['s_full_condition']['run_s_full']):
        raise ValueError('registered HS1 trigger not established')
    return copy.deepcopy(h)


def write(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / REL / 'receipts/core_2026-10-30/analysis_carry12.json'
    raw = report_path.read_bytes()
    proof = trigger(json.loads(raw))
    inventory = {}
    for cell in CELLS:
        for seed in SEEDS:
            path, doc, source = build(cell, seed)
            target = out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (json.dumps(doc, indent=1) + '\n').encode()
            if target.exists() and target.read_bytes() != data:
                raise ValueError('refuse overwrite: ' + str(target))
            target.write_bytes(data)
            inventory[path] = {'sha256': hashlib.sha256(data).hexdigest(), 'source': source,
                               'source_sha256': hashlib.sha256(frozen(source)).hexdigest(), 'cell': cell, 'seed': seed}
    manifest = {'parent': PIN, 'rule': 'PREREGISTRATION.md section 9: primary NOT_RESOLVED, same sign all3, seeds4/5 once',
                'trigger_report_sha256': hashlib.sha256(raw).hexdigest(), 'HS1_trigger': proof,
                'S_full': False, 'jobs': inventory, 'expected_gpu_usd': 18.16, 'total_cumulative_cap_usd': 24,
                'model_or_recipe_changes': False, 'data_seed': 1,
                'excluded_jobs': ['D', 'TS', 'W512', 'chat', 'Sfull'],
                'analysis_scope': 'B/Sgist five-seed HS1 only; unchanged global core remains three-seed'}
    dest = out / REL / 'hs1_extension_manifest.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(write(a.out), indent=1))
