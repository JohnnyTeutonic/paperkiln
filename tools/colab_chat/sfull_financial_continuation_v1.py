"""Financial-only continuation after a natural S-full reservation stop.

No live-process mutation, checkpoint-guard override or credential inspection.
The existing watchdog invokes this source-pinned guard, never a second driver.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time


class Refused(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def flag(args, name):
    if args.count(name) != 1 or args.index(name) + 1 >= len(args):
        raise Refused('unique argument required: ' + name)
    return args[args.index(name) + 1]


def process_identity(pid):
    if type(pid) is not int or pid <= 0:
        raise Refused('invalid PID')
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = text.rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return int(fields[19])
    except FileNotFoundError:
        return None


def unique_ledger(ledger):
    for kind in ('pods', 'volumes'):
        entries = ledger[kind]
        ids = [x['id'] for x in entries]
        if len(ids) != len(set(ids)):
            raise Refused('duplicate ledger resource')
        for x in entries:
            values = ([x['usd_per_hr'], x['start']] if kind == 'pods'
                      else [x['gb'], x['created']])
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise Refused('invalid priced ledger entry')
            start = x['start' if kind == 'pods' else 'created']
            end = x.get('stop' if kind == 'pods' else 'deleted')
            if start > time.time() + 1:
                raise Refused('future ledger start')
            if end is not None and (type(end) not in (int, float) or not math.isfinite(end)
                                    or not start <= end <= time.time() + 1):
                raise Refused('invalid ledger closure')


def retain_ledger(before, after):
    """Only monotonic, priced lifecycle closure may change old entries."""
    unique_ledger(before)
    unique_ledger(after)
    for kind, closure in [('pods', 'stop'), ('volumes', 'deleted')]:
        current = {x['id']: x for x in after[kind]}
        for old in before[kind]:
            new = current.get(old['id'])
            if new is None:
                raise Refused('historical resource omitted')
            for key, value in old.items():
                if new.get(key) != value and not (key == closure and value is None):
                    raise Refused('historical ledger field changed: ' + key)
            value = new.get(closure)
            if value is not None:
                start = old['start' if kind == 'pods' else 'created']
                if type(value) not in (int, float) or not start <= value <= time.time() + 1:
                    raise Refused('invalid closure timestamp')


def predecessor_gate(r, state, log, identity, pods, spent):
    if identity == r['predecessor_start_ticks']:
        return 'DEFER_PREDECESSOR_ALIVE'
    if not math.isfinite(spent):
        raise Refused('nonfinite spending')
    jobs = state.get('jobs', {})
    if set(jobs) != set(r['expected_jobs']) or any(
            job.get('status') not in ('FINISHED', 'RUNNING', 'WAITING', None) for job in jobs.values()):
        raise Refused('job inventory or operational status differs')
    if any(jobs[name].get('status') != 'FINISHED' or jobs[name].get('home') is not True
           for name in r['completed_preflight_jobs']):
        raise Refused('preflight evidence incomplete')
    if state.get('campaign_home'):
        owned = state['runpod']
        if (not state.get('jobs') or not all(j.get('home') is True for j in state['jobs'].values())
                or pods or owned.get('pod') or state.get('endpoint') or owned.get('creating')):
            raise Refused('incoherent home state')
        return 'ALREADY_HOME'
    owned = state['runpod']
    if pods or owned.get('pod') or state.get('endpoint') or owned.get('creating'):
        raise Refused('predecessor/provider resource not conclusively stopped')
    if not all(p.get('stop') for p in owned['ledger']['pods']):
        raise Refused('unclosed predecessor pod ledger')
    if spent < 7 or spent >= 12:
        raise Refused('outside financial transition interval')
    lines = log.splitlines()
    stop = [i for i, line in enumerate(lines) if 'reached --max-usd 7; stopping' in line]
    if not stop or not any('session stopped; runner exiting' in x for x in lines[stop[-1] + 1:]):
        raise Refused('natural budget exit not proven')
    if any('HALTED' in x or 'Traceback' in x for x in lines[stop[-1]:]):
        raise Refused('failure after budget exit')
    working = owned.get('volume') or {}
    if working.get('id') != r['working_volume_id'] or working.get('dc') != r['working_volume_dc']:
        raise Refused('working volume ownership changed')
    if working.get('deleted') or working.get('id') == r['protected_original_volume_id']:
        raise Refused('working volume invalid')
    return 'INITIALIZE'


def configuration(review, digest, supplied):
    path = Path(review)
    if path.is_symlink() or sha(path) != digest:
        raise Refused('review binding differs')
    r = read(path)
    if r['kind'] != 'SFULL_FINANCIAL_CONTINUATION_V5' or r['status'] != 'ADMITTED':
        raise Refused('not admitted')
    if (type(r['per_seed_cap_usd']) is not int or r['per_seed_cap_usd'] != 12
            or type(r['aggregate_cap_usd']) is not int or r['aggregate_cap_usd'] != 36):
        raise Refused('financial contract differs')
    if r.get('seeds2_and3_admitted') is not False or r['seed'] != 1:
        raise Refused('seed scope differs')
    mandatory = [r[key] for key in ('original_admission', 'original_guard', 'original_reservation',
                                    'financial_amendment', 'initial_ledger')]
    mandatory.append(str(Path(__file__).resolve()))
    if any(name not in r['pins'] for name in mandatory):
        raise Refused('required admission anchor not pinned')
    for name, pin in r['pins'].items():
        p = Path(name)
        if p.is_symlink() or not p.is_file() or sha(p) != pin:
            raise Refused('pin differs: ' + name)
    old = read(r['original_admission'])
    if r['original_reservation'] != old['reservation_path']:
        raise Refused('original reservation path differs')
    original_source = Path(r['original_guard']).read_bytes()
    namespace = {'__file__': r['original_guard'], '__name__': 'frozen_original_guard'}
    exec(compile(original_source, r['original_guard'], 'exec'), namespace)
    if not read(Path(r['state_dir']) / 'state.json').get('campaign_home'):
        namespace['validate'](r['original_admission'], r['pins'][r['original_admission']], old['runner_argv'][1:])
    wanted = list(old['runner_argv'])
    wanted[wanted.index('--max-usd') + 1] = '12'
    if r['runner_argv'] != wanted or supplied != wanted[1:]:
        raise Refused('nonfinancial runner argument changed')
    if r['state_dir'] != old['state_dir'] or r['deadline_utc'] != old['deadline_utc']:
        raise Refused('state or absolute deadline changed')
    end = dt.datetime.fromisoformat(r['deadline_utc'].replace('Z', '+00:00'))
    if dt.datetime.now(dt.timezone.utc) >= end:
        raise Refused('absolute deadline expired')
    if (Path(r['state_dir']) / 'HALTED').exists():
        raise Refused('existing halt is never removed or waived')
    financial = read(r['financial_amendment'])
    if (financial['per_seed_cumulative_cap_usd'] != 12
            or financial['three_seed_planning_reservation_usd'] != 36
            or financial['seeds2_and3_admitted'] is not False
            or financial['status'] != 'ADOPTED_FOR_SEED1_PENDING_SAFE_CAP_BOUNDARY_TRANSITION'):
        raise Refused('financial authority differs')
    if (financial.get('seed') != 1 or financial.get('numerical_commit') != old['numerical_commit']
            or not financial.get('authority') or not financial.get('adopted_at_utc')):
        raise Refused('financial amendment source/authority differs')
    return r, namespace, old


def snapshot_once(root, state_path, log_path, r, ledger, step):
    terminal = root / 'terminal_v4'
    temporary = root / 'terminal_v4.pending'
    if terminal.exists() or temporary.exists():
        raise Refused('terminal snapshot already exists or was interrupted')
    if read(state_path)['runpod']['ledger'] != ledger:
        raise Refused('terminal state changed during transition')
    temporary.mkdir()
    names = {'state.json': state_path, 'runner.log': log_path,
             'admission.json': Path(r['original_admission']),
             'guard.py': Path(r['original_guard']),
             'reservation.json': Path(r['original_reservation'])}
    hashes = {}
    for name, source in names.items():
        destination = temporary / name
        with destination.open('xb') as f:
            f.write(source.read_bytes())
            f.flush()
            os.fsync(f.fileno())
        hashes[name] = sha(destination)
    proof = {'kind': 'SFULL_V4_TERMINAL_BUDGET_TRANSITION', 'hashes': hashes,
             'ledger': ledger, 'validated_home_step': step,
             'initialized_unix': time.time(), 'review_sha256': r['_review_sha256']}
    with (temporary / 'transition.json').open('x') as f:
        json.dump(proof, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temporary, terminal)
    with (root / 'terminal_anchor.json').open('x') as f:
        json.dump({'transition_sha256': sha(terminal / 'transition.json')}, f)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return proof


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', required=True)
    p.add_argument('--review-sha256', required=True)
    p.add_argument('argv', nargs=argparse.REMAINDER)
    args = p.parse_args()
    supplied = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    root = Path(args.review).parent
    if Path(args.review).is_symlink() or sha(args.review) != args.review_sha256:
        raise Refused('review binding differs')
    preview = read(args.review)
    deadline = dt.datetime.fromisoformat(preview['deadline_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None or dt.datetime.now(dt.timezone.utc) >= deadline:
        raise Refused('absolute deadline expired before defer/dispatch')
    state_dir = Path(preview['state_dir'])
    with (root / 'launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('DEFER_ANOTHER_V5_OWNER', flush=True)
            return
        if process_identity(preview['predecessor_pid']) == preview['predecessor_start_ticks']:
            print('DEFER_PREDECESSOR_ALIVE', flush=True)
            return
        # This guard is never a second driver and cannot take over a new live PID.
        pidfile = state_dir / 'runner.pid'
        if pidfile.exists():
            pid = int(pidfile.read_text())
            if pid != os.getpid() and process_identity(pid) is not None:
                print('DEFER_ANOTHER_STATE_OWNER', flush=True)
                return
        r, original, old = configuration(args.review, args.review_sha256, supplied)
        r['_review_sha256'] = args.review_sha256
        source = Path(old['source_checkout']) / 'tools/colab_chat'
        sys.path.insert(0, str(source))
        import runpod_api as api
        state_path = state_dir / 'state.json'
        state = read(state_path)
        terminal = root / 'terminal_v4'
        if not terminal.exists():
            result = predecessor_gate(r, state, (state_dir / 'runner.log').read_text(),
                                      process_identity(r['predecessor_pid']), api.list_pods(),
                                      api.ledger_usd(state['runpod']['ledger']))
            if result == 'ALREADY_HOME':
                (root / 'STOPPED').write_text('COMPLETED_WITHOUT_CONTINUATION\n')
                print(result, flush=True)
                return
            if result != 'INITIALIZE':
                print(result, flush=True)
                return
            retain_ledger(read(r['initial_ledger']), state['runpod']['ledger'])
            step = original['validate_checkpoint'](old)
            if step < r['minimum_home_step']:
                raise Refused('checkpoint regressed below reviewed home evidence')
            # The frozen validator's current_unlocked is full payload/RNG validation,
            # not metadata_only. Do not substitute a header or bare model restore.
            snapshot_once(root, state_path, state_dir / 'runner.log', r, state['runpod']['ledger'], step)
        anchor = read(root / 'terminal_anchor.json')
        if anchor != {'transition_sha256': sha(terminal / 'transition.json')}:
            raise Refused('transition anchor differs')
        proof = read(terminal / 'transition.json')
        if proof['review_sha256'] != args.review_sha256:
            raise Refused('transition belongs to another admission')
        for name, digest in proof['hashes'].items():
            if sha(terminal / name) != digest:
                raise Refused('historical transition snapshot changed')
        retain_ledger(proof['ledger'], read(state_path)['runpod']['ledger'])
        state = read(state_path)
        if state.get('campaign_home'):
            return
        owned = state['runpod']
        if owned.get('pod') or state.get('endpoint') or owned.get('creating'):
            raise Refused('restart state still owns active resources')
        working = owned.get('volume') or {}
        if (working.get('id') != r['working_volume_id'] or working.get('dc') != r['working_volume_dc']
                or working.get('deleted')):
            raise Refused('restart working volume differs')
        if api.list_pods():
            raise Refused('GPU admission requires an empty provider slot')
        spent = api.ledger_usd(state['runpod']['ledger'])
        if not math.isfinite(spent) or spent >= 12:
            raise Refused('revised cumulative reservation exhausted')
        if original['validate_checkpoint'](old) < proof['validated_home_step']:
            raise Refused('checkpoint regressed below terminal snapshot')
        api.MAX_PODS = 1
        previous_argv = sys.argv
        try:
            sys.argv = r['runner_argv'][1:]
            runpy.run_path(r['runner_argv'][1], run_name='__main__')
        finally:
            sys.argv = previous_argv


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        marker = Path(__file__).resolve().parent / 'STOPPED'
        try:
            with marker.open('x') as f:
                json.dump({'kind': 'SFULL_FINANCIAL_CONTINUATION_REFUSED', 'error': str(error)}, f)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            pass
        raise
