"""One frozen engineering admission, inactive until root finalizes the review."""
import argparse, datetime, hashlib, json
from pathlib import Path
import runpy, sys

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def validate(review_path,review_sha,args):
 p=Path(review_path)
 if digest(p)!=review_sha:raise ValueError('review hash mismatch')
 review=json.loads(p.read_text())
 if review['status']!='APPROVED_ONE_ATTEMPT_AFTER_SERIAL_ADMISSION':raise ValueError('root admission is not approved')
 approved=review['driver_argv'][1:]
 if args!=approved:raise ValueError('driver arguments differ from reviewed launch')
 deadline=review['bounds']['deadline_utc']
 parsed=datetime.datetime.fromisoformat(deadline.replace('Z','+00:00'))
 if parsed.tzinfo is None or parsed<=datetime.datetime.now(datetime.timezone.utc):raise ValueError('invalid or expired absolute deadline')
 if approved[approved.index('--deadline-utc')+1]!=deadline:raise ValueError('deadline binding')
 if approved[approved.index('--max-usd')+1]!='1' or approved[approved.index('--max-hours')+1]!='1':raise ValueError('resource bounds differ')
 if review['bounds']['scientific_seconds']!=1800:raise ValueError('science timeout differs')
 for name,expected in review['pins'].items():
  if digest(name)!=expected:raise ValueError('operational/upload source changed: '+name)
 if digest(review['registered_job_path'])!=review['registered_job_sha256']:raise ValueError('registered job changed')
 for i,value in enumerate(approved):
  if value=='--upload':
   name,source=approved[i+1].split('=',1);path=Path(source)
   if '/' in name or '\\' in name or not path.is_file() or path.is_symlink() or str(path.resolve())!=source:raise ValueError('upload source must resolve to flat regular file')
 return approved

def main():
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--review-sha256',required=True);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
 args=a.args[1:] if a.args[:1]==['--'] else a.args
 approved=validate(a.review,a.review_sha256,args)
 runner=Path(approved[0]).resolve();sys.path.insert(0,str(runner.parent))
 import runpod_api
 if Path(runpod_api.__file__).resolve()!=runner.parent/'runpod_api.py':raise ValueError('provider import binding')
 runpod_api.MAX_PODS=1;sys.argv=approved;runpy.run_path(str(runner),run_name='__main__')
if __name__=='__main__':main()
