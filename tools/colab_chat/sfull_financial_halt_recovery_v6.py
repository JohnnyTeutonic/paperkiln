"""Exact financial-transport halt recovery; original V4/V5 evidence stays immutable."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import time

CONTROL = "c9b82fd6718497a4ace4fba62f6e23b8ed2346a7"
NUMERICAL = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
DEADLINE = "2026-11-02T11:41:15Z"
FINANCIAL = "RunPod spend $7.00 reached --max-usd 7.0; stopping"
V4_HALT = {"kind": "S_FULL_CHECKPOINT_V2_FAILURE", "error": FINANCIAL}
V5_STOP = {"kind": "SFULL_FINANCIAL_CONTINUATION_REFUSED", "error": "home or halted continuation cannot relaunch"}
PREFLIGHTS = {"cuda_gate", "data", "gate", "sfull_inputs"}
JOBS = PREFLIGHTS | {"pt_c12_Sfull_r1", "post_c12_Sfull_r1"}


class Refused(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def flag(argv, key):
    if argv.count(key) != 1 or argv.index(key) + 1 >= len(argv):
        raise Refused('unique flag required: ' + key)
    return argv[argv.index(key) + 1]


def pin_check(r):
    for name, digest in r['pins'].items():
        p = Path(name)
        if not p.is_absolute() or p.is_symlink() or not p.is_file() or sha(p) != digest:
            raise Refused('pinned evidence/source differs: ' + name)


def classify_financial_halt(halt, stopped, driver, runner_log):
    if halt != V4_HALT or stopped != V5_STOP:
        raise Refused('not the exact reviewed financial transport halt/refusal')
    exceptions = [line for line in driver.splitlines() if re.match(r'^[\w.]+(?:Error|Refused): ', line)]
    if exceptions != ['runpod_api.Refused: ' + FINANCIAL, 'ValueError: ' + V5_STOP['error']]:
        raise Refused('mixed, additional or uncorroborated terminal failures')
    if driver.count('Traceback (most recent call last):') != 2:
        raise Refused('terminal traceback inventory differs')
    lines = runner_log.splitlines()
    if len(lines) < 2 or 'terminated (RunPod ledger $7.00)' not in lines[-2] or not lines[-1].endswith('session stopped; runner exiting'):
        raise Refused('terminal budget stop is not corroborated')
    return 'EXACT_FINANCIAL_TRANSPORT_HALT'


def ledger_check(ledger):
    if set(ledger) != {'pods', 'volumes'}:
        raise Refused('ledger kinds differ')
    for kind in ('pods', 'volumes'):
        if len({x['id'] for x in ledger[kind]}) != len(ledger[kind]):
            raise Refused('duplicate ledger resource')
        for x in ledger[kind]:
            start = x['start' if kind == 'pods' else 'created']
            end = x.get('stop' if kind == 'pods' else 'deleted')
            rate = x['usd_per_hr' if kind == 'pods' else 'gb']
            if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in [start, rate]):
                raise Refused('invalid ledger price/time')
            if start > time.time() + 1:
                raise Refused('future ledger start')
            if end is not None and (type(end) not in (int,float) or not math.isfinite(end) or not start <= end <= time.time() + 1):
                raise Refused('invalid ledger closure')
    if any(not p.get('stop') for p in ledger['pods']):
        raise Refused('all inherited pod charges must be closed')


def retain_ledger(before, after):
    ledger_check(before); ledger_check(after)
    for kind in ('pods', 'volumes'):
        if any(x not in after[kind] for x in before[kind]):
            raise Refused('inherited cumulative spending omitted or changed')


def predecessor_state(r):
    old = read(r['old_state'])
    if old.get('campaign_home') or old.get('endpoint') or old['runpod'].get('pod') or old['runpod'].get('creating'):
        raise Refused('predecessor is home or still owns compute')
    if set(old['jobs']) != JOBS or any(old['jobs'][n].get('status') != 'FINISHED' or old['jobs'][n].get('home') is not True for n in PREFLIGHTS):
        raise Refused('original preflight/job evidence differs')
    if any(j.get('status') not in ('FINISHED','RUNNING','WAITING',None) for j in old['jobs'].values()):
        raise Refused('scientific or operational job failure remains')
    owned = old['runpod']
    if owned.get('retired_volumes') or (owned.get('volume') or {}).get('id') != r['working_volume_id']:
        raise Refused('working volume ownership differs')
    if owned['volume'].get('dc') != r['working_volume_dc'] or owned['volume'].get('gb') != 80 or owned['volume'].get('deleted'):
        raise Refused('working volume geometry/lifecycle differs')
    if r['working_volume_id'] in r['protected_volume_ids']:
        raise Refused('protected volume cannot become owned')
    ledger_check(owned['ledger'])
    if sum(v['id'] == r['working_volume_id'] and not v.get('deleted') for v in owned['ledger']['volumes']) != 1:
        raise Refused('working volume priced history missing')
    return old


def safe_path(path):
    p=Path(path)
    if not p.is_absolute():raise Refused('absolute path required')
    if any(x.is_symlink() for x in [p,*p.parents]):raise Refused('symlinked path/ancestor refused')
    return p.resolve()


def overlap(a,b):
    return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def path_safety(r):
    destination=safe_path(r['state_dir']).parent
    safe_path(r['registry_path'])
    preserved=[r['source_checkout'],r['art_dir'],str(Path(r['old_state']).parent)]
    preserved += [p for p in r['pins'] if Path(p).resolve()!=Path(__file__).resolve()]
    if any(overlap(destination,safe_path(p)) for p in preserved):
        raise Refused('new output overlaps preserved evidence/source')
    if overlap(destination,safe_path(r['registry_path'])):
        raise Refused('output overlaps registry')


def configuration(review, digest, supplied, now=None):
    if Path(review).is_symlink() or sha(review) != digest:
        raise Refused('review digest differs')
    r = read(review)
    if r['kind'] != 'SFULL_FINANCIAL_HALT_RECOVERY_V6' or r['status'] != 'ADMITTED':
        raise Refused('not admitted')
    if r['seed'] != 1 or r['per_seed_cap_usd'] != 12 or r['aggregate_cap_usd'] != 36 or r['seeds2_and3_admitted'] is not False:
        raise Refused('financial-only scope differs')
    mandatory = [r[k] for k in ['old_state','old_halt','v5_stopped','old_driver_output','old_runner_log','original_admission','financial_amendment']]
    mandatory.append(str(Path(__file__).resolve()))
    if any(p not in r['pins'] for p in mandatory):
        raise Refused('required terminal/source evidence unpinned')
    pin_check(r)
    original = read(r['original_admission'])
    if original['kind'] != 'SFULL_CONTINUATION_ADMISSION_V4' or original['source_control_commit'] != CONTROL or original['numerical_commit'] != NUMERICAL or original['deadline_utc'] != DEADLINE or original['per_seed_cap_usd'] != 7 or original['aggregate_cap_usd'] != 21:
        raise Refused('original frozen admission differs')
    if any(r['pins'].get(path) != digest for path, digest in original['pins'].items()):
        raise Refused('complete original source/input inventory omitted')
    if r['source_checkout'] != original['source_checkout'] or r['art_dir'] != original['art_dir'] or r['old_state'] != str(Path(original['state_dir'])/'state.json'):
        raise Refused('historical source/artifact/state binding differs')
    if r['deadline_utc'] != DEADLINE or r['source_control_commit'] != CONTROL or r['numerical_commit'] != NUMERICAL:
        raise Refused('source or absolute deadline changed')
    now = now or dt.datetime.now(dt.timezone.utc)
    if now >= dt.datetime.fromisoformat(DEADLINE.replace('Z','+00:00')):
        raise Refused('original absolute deadline expired')
    wanted = list(original['runner_argv'])
    for key,value in [('--max-usd','12'),('--state-dir',r['state_dir']),('--name',r['name'])]:
        wanted[wanted.index(key)+1] = value
    if r['runner_argv'] != wanted or supplied != wanted[1:]:
        raise Refused('nonfinancial source/recipe/argv override')
    path_safety(r)
    new = Path(r['state_dir'])
    if not new.is_absolute() or new.resolve() == Path(original['state_dir']).resolve() or new.is_relative_to(Path(original['state_dir'])) or Path(original['state_dir']).is_relative_to(new):
        raise Refused('new state must not overlap historical state')
    if (new/'HALTED').exists() or (Path(review).parent/'STOPPED').exists():
        raise Refused('new recovery halt is never waived')
    amendment = read(r['financial_amendment'])
    if amendment.get('per_seed_cumulative_cap_usd') != 12 or amendment.get('three_seed_planning_reservation_usd') != 36 or amendment.get('seed') != 1 or amendment.get('seeds2_and3_admitted') is not False or amendment.get('numerical_commit') != NUMERICAL or not amendment.get('authority') or not amendment.get('adopted_at_utc'):
        raise Refused('financial amendment authority differs')
    classify_financial_halt(read(r['old_halt']), read(r['v5_stopped']),Path(r['old_driver_output']).read_text(),Path(r['old_runner_log']).read_text())
    predecessor_state(r)
    return r


def validate_checkpoint(r):
    folder = Path(r['source_checkout'])/'tools/colab_chat'
    sys.path.insert(0,str(folder))
    import sfull_checkpoint_v2 as cp
    import sfull_checkpoint_transport_v2 as tr
    jobs = read(folder/'carry12_sfull_v2_r1.json')['jobs']
    validated = []
    for job in jobs:
        if not job.get('full_checkpoint_v2'):
            continue
        root = tr.root_for(r['art_dir'],job)
        if not (root/'CURRENT').is_symlink():
            if job['name'] == 'pt_c12_Sfull_r1':
                raise Refused('complete pretraining checkpoint absent')
            continue
        with cp.locked(root,False):
            _, manifest = cp.current_unlocked(root)
            tr.check_manifest(job,manifest)
            if (root/'INFLIGHT').is_symlink():
                _, pinned = cp.current_unlocked(root,'INFLIGHT')
                tr.check_manifest(job,pinned)
            if job['name'] == 'pt_c12_Sfull_r1' and manifest['step'] < r['minimum_home_step']:
                raise Refused('checkpoint regressed below reviewed complete state')
            validated.append({'job':job['name'],'step':manifest['step'],'manifest_sha256':cp.sha(root/'CURRENT/manifest.json')})
    return validated


def write_atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.v6.pending')
    with temporary.open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare_state(r, provider_pods, predecessor_alive, registry_path, registry_runner_alive):
    """Offline/local transfer only. All registered drivers must be stopped first."""
    if provider_pods or predecessor_alive:
        raise Refused('provider and predecessor must be conclusively empty/dead')
    path_safety(r)
    if safe_path(registry_path)!=safe_path(r['registry_path']):raise Refused('registry path differs from review')
    old = predecessor_state(r)
    checkpoint = validate_checkpoint(r)
    state_dir=Path(r['state_dir']);root=state_dir.parent
    if state_dir.exists() or (root/'transfer.json').exists():
        raise Refused('new state or transfer already exists; inspect rather than overwrite')
    registry_path=Path(registry_path)
    with (root/'registry_transfer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=read(registry_path)
        if any(registry_runner_alive(name) for name in before):
            raise Refused('registered driver still alive')
        old_name=str(Path(r['old_state']).parent)
        if old_name not in before or r['state_dir'] in before:
            raise Refused('registry predecessor/new-owner inventory differs')
        for name in before:
            if name==old_name:continue
            p=Path(name)/'state.json'
            if p.is_file():
                state=read(p);owned=state.get('runpod',{})
                volumes=[owned.get('volume'),*owned.get('retired_volumes',[])]
                if any(isinstance(v,dict) and v.get('id')==r['working_volume_id'] for v in volumes):
                    raise Refused('another registry owner records working volume')
        # Preserve exact terminal evidence, ledger and registry before transfer.
        proof={'kind':'SFULL_V6_EXACT_FINANCIAL_HALT_TRANSFER','original_state_sha256':sha(r['old_state']),'old_halt_sha256':sha(r['old_halt']),'v5_stopped_sha256':sha(r['v5_stopped']),'checkpoint':checkpoint,'initial_ledger':old['runpod']['ledger'],'registry_before':before,'new_state_dir':r['state_dir'],'working_volume_id':r['working_volume_id']}
        write_atomic(root/'transfer.json',proof)
        new=copy.deepcopy(old)
        new['booted']=False  # frozen launch also clears this after acquiring a replacement pod
        new['gone_ticks']=0
        state_dir.mkdir()
        write_atomic(state_dir/'state.json',new)
        after=copy.deepcopy(before);after.pop(old_name)
        after[r['state_dir']]={'name':r['name'],'registered':time.time()}
        if read(registry_path)!=before or sha(r['old_state'])!=proof['original_state_sha256']:
            raise Refused('registry or predecessor changed during offline transfer')
        write_atomic(registry_path,after)
        write_atomic(root/'transfer_anchor.json',{'transfer_sha256':sha(root/'transfer.json')})
    return proof


def validate_owner(r):
    registry=read(safe_path(r['registry_path']))
    old_name=str(Path(r['old_state']).parent)
    if old_name in registry or r['state_dir'] not in registry:
        raise Refused('registry ownership transfer missing or reverted')
    if live_state_pid(old_name):raise Refused('predecessor driver is alive')
    for name in registry:
        if name==r['state_dir']:continue
        path=Path(name)/'state.json'
        if path.is_file():
            owned=read(path).get('runpod',{})
            if any(isinstance(v,dict) and v.get('id')==r['working_volume_id'] for v in [owned.get('volume'),*owned.get('retired_volumes',[])]):
                raise Refused('another registry owner records working volume')


def validate_prepared(r):
    validate_owner(r)
    root=Path(r['state_dir']).parent
    if read(root/'transfer_anchor.json')!={'transfer_sha256':sha(root/'transfer.json')}:
        raise Refused('transfer anchor differs')
    proof=read(root/'transfer.json')
    if sha(r['old_state'])!=proof['original_state_sha256'] or sha(r['old_halt'])!=proof['old_halt_sha256'] or sha(r['v5_stopped'])!=proof['v5_stopped_sha256']:
        raise Refused('historical terminal evidence changed')
    if proof.get('kind') != 'SFULL_V6_EXACT_FINANCIAL_HALT_TRANSFER' or proof.get('new_state_dir') != r['state_dir'] or proof.get('working_volume_id') != r['working_volume_id'] or proof.get('initial_ledger') != read(r['old_state'])['runpod']['ledger']:
        raise Refused('transfer identity or original cumulative ledger differs')
    state=read(Path(r['state_dir'])/'state.json')
    if state.get('campaign_home'):
        return state
    if state.get('endpoint') or state['runpod'].get('pod') or state['runpod'].get('creating'):
        raise Refused('restart must wait for owned compute cleanup')
    if state['runpod'].get('retired_volumes') or state['runpod'].get('volume')!=read(r['old_state'])['runpod']['volume']:
        raise Refused('working volume or retired ownership changed')
    retain_ledger(proof['initial_ledger'],state['runpod']['ledger'])
    return state


def live_state_pid(state_dir):
    try:
        pid=int((Path(state_dir)/'runner.pid').read_text())
        if pid <= 0: return False
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0] != 'Z'
    except (OSError,ValueError,IndexError):
        return False


def record_refusal(root,error):
    try:
        with (Path(root)/'STOPPED').open('x') as f:
            json.dump({'kind':'SFULL_FINANCIAL_HALT_RECOVERY_REFUSED','error':str(error)},f)
            f.flush();os.fsync(f.fileno())
    except FileExistsError:
        pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--review-sha256',required=True);p.add_argument('argv',nargs=argparse.REMAINDER)
    a=p.parse_args();supplied=a.argv[1:] if a.argv[:1]==['--'] else a.argv
    root=Path(__file__).resolve().parent
    with (root/'launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('DEFER_V6_LAUNCH_LOCK_HELD',flush=True);return
        r=configuration(a.review,a.review_sha256,supplied)
        if live_state_pid(r['state_dir']):
            print('DEFER_V6_OWNER_ALIVE',flush=True);return
        validate_checkpoint(r)
        state=validate_prepared(r)
        if state.get('campaign_home'):return
        if not r.get('dispatch_authorized'):raise Refused('prepared/probe record cannot launch')
        source=Path(r['source_checkout'])/'tools/colab_chat';sys.path.insert(0,str(source))
        import runpod_api as api
        if Path(api.__file__).resolve()!=source/'runpod_api.py':raise Refused('provider module differs')
        if api.list_pods():raise Refused('GPU campaign is not serial')
        spent=api.ledger_usd(state['runpod']['ledger'])
        if not math.isfinite(spent) or not 7 <= spent < 12:raise Refused('outside revised cumulative financial bound')
        api.MAX_PODS=1
        sys.argv=r['runner_argv'][1:]
        runpy.run_path(r['runner_argv'][1],run_name='__main__')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        record_refusal(Path(__file__).resolve().parent,error)
        raise
