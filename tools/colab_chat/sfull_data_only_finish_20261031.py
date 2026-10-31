"""Publish success only after both data jobs and the original verifier succeeded."""
from pathlib import Path
import argparse
import json
import os


def finish(root, jobs_exit):
    root = Path(root)
    reasons = []
    if jobs_exit != 0:
        reasons.append(f'jobs_vm exit {jobs_exit}')
    for job in ('data', 'sfull_inputs'):
        p = root / ('STATUS_' + job)
        try:
            status = p.read_text().strip() if p.is_file() else 'MISSING'
        except (OSError, UnicodeError) as error:
            status = 'UNREADABLE_' + type(error).__name__
        if status != 'FINISHED':
            reasons.append(f'{job}: {status}')
    try:
        report = json.loads((root / 'sfull_inputs/verified_inputs.json').read_text())
        if report.get('verified') is not True or report.get('raw_human_sources_verified') != 13 or report.get('evaluation_sources_verified') != 2:
            reasons.append('original verifier did not establish complete input identity')
        if report.get('reference_sha256') != 'acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4':
            reasons.append('unexpected reference identity')
    except (OSError, ValueError, TypeError, AttributeError):
        reasons.append('missing or malformed original verifier report')
    status = 'JOBS_FAILED' if reasons else 'FINISHED'
    temp = root / 'STATUS.tmp'
    temp.write_text(status + '\n', encoding='utf-8')
    os.replace(temp, root / 'STATUS')
    print(json.dumps({'status': status, 'reasons': reasons}), flush=True)
    return 1 if reasons else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/content/chat')
    parser.add_argument('--jobs-exit', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(finish(args.root, args.jobs_exit))
