"""Root-reviewed E9 matched v2 serial admission; no credentials are inspected here."""
import argparse, datetime, hashlib, json, os, re, runpy, sys, time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(1048576),b''):h.update(block)
 return h.hexdigest()

def validate(review_path,review_sha,argv):
 if sha(review_path)!=review_sha:raise ValueError('launch review hash differs')
 r=json.loads(Path(review_path).read_text())
 if r['kind']!='ROOT_E9_MATCHED_SERIAL_ADMISSION_V2' or r['status']!='APPROVED_AFTER_OFFLINE_AND_SERIAL_CHECKS':raise ValueError('not approved')
 approved=r['driver_argv'][1:]
 if argv!=approved:raise ValueError('driver argv differs')
 for flag,value in {'--backend':'runpod','--gpu':'L4','--cloud':'SECURE','--volume-gb':'30','--max-usd':'2','--max-hours':'2'}.items():
  if approved.count(flag)!=1 or approved[approved.index(flag)+1]!=value:raise ValueError('resource or boot binding differs')
 if approved.count('--commit')!=1 or approved[approved.index('--commit')+1]!=r['runner_source_commit'] or not re.fullmatch('[0-9a-f]{40}',r['runner_source_commit']):raise ValueError('frozen source commit required')
 for flag,value in {'--jobs':r['jobs_relative'],'--boot-script':'tools/colab_chat/boot_commands.sh','--art':r['art_dir']}.items():
  if approved.count(flag)!=1 or approved[approved.index(flag)+1]!=value:raise ValueError('source job/boot/art binding differs')
 if approved[0]!=str(Path(r['source'])/'tools/colab_chat/runner_multi.py'):raise ValueError('runner path differs')
 if approved.count('--state-dir')!=1 or approved[approved.index('--state-dir')+1]!=r['state_dir']:raise ValueError('state directory differs')
 if approved.count('--deadline-utc')!=1 or approved[approved.index('--deadline-utc')+1]!=r['deadline_utc']:raise ValueError('deadline differs')
 deadline=datetime.datetime.fromisoformat(r['deadline_utc'].replace('Z','+00:00'))
 if deadline.tzinfo is None or deadline<=datetime.datetime.now(datetime.timezone.utc):raise ValueError('absolute deadline expired')
 admitted=datetime.datetime.fromisoformat(r['admitted_at_utc'].replace('Z','+00:00'))
 if admitted.tzinfo is None or not 0<(deadline-admitted).total_seconds()<=7200:raise ValueError('absolute admission window exceeds two hours')
 if admitted>datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=5):raise ValueError('admission starts in future')
 attestation=Path(r['fresh_paths_attestation'])
 if sha(attestation)!=r['fresh_paths_attestation_sha256']:raise ValueError('fresh paths attestation changed')
 att=json.loads(attestation.read_text())
 if att!={'state_dir':r['state_dir'],'art_dir':r['art_dir'],'both_absent_at_prepare':True}:raise ValueError('fresh paths attestation differs')
 anchor=Path(r['state_dir'])/'E9_ADMISSION_ANCHOR.json'
 if anchor.exists():
  if anchor.is_symlink() or json.loads(anchor.read_text())!={'review_sha256':review_sha}:raise ValueError('admission anchor differs')
 else:
  for p in [Path(r['state_dir']),Path(r['art_dir'])]:
   if p.exists():raise ValueError('fresh campaign directory already exists')
 for path,digest in r['pins'].items():
  if not Path(path).is_file() or Path(path).is_symlink() or str(Path(path).resolve())!=path:raise ValueError('pinned regular canonical source required')
  if sha(path)!=digest:raise ValueError('pinned source/input changed: '+path)
 seen=set()
 for i,arg in enumerate(approved):
  if arg=='--upload':
   name,source=approved[i+1].split('=',1);p=Path(source)
   if name in seen or Path(name).name!=name or '\\' in name or not p.is_file() or p.is_symlink() or str(p.resolve())!=source:raise ValueError('flat regular unique upload required')
   if r['uploads'].get(name)!=source or source not in r['pins']:raise ValueError('upload source is not pinned')
   seen.add(name)
 if seen!=set(r['upload_names']):raise ValueError('upload inventory differs')
 if approved[approved.index('--art')+1]!=r['art_dir']:raise ValueError('artifact directory differs')
 state=Path(r['state_dir'])/'state.json'
 if state.exists():
  saved=json.loads(state.read_text())
  if saved.get('campaign_home'):raise ValueError('completed campaign cannot relaunch')
 return approved

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--review',required=True);parser.add_argument('--review-sha256',required=True);parser.add_argument('argv',nargs=argparse.REMAINDER);a=parser.parse_args()
 argv=a.argv[1:] if a.argv[:1]==['--'] else a.argv
 state=Path(json.loads(Path(a.review).read_text())['state_dir'])
 if (state/'HALTED').exists():raise SystemExit('preserved root admission halt requires review')
 try:
  approved=validate(a.review,a.review_sha256,argv)
 except Exception as error:
  state.mkdir(parents=True,exist_ok=True)
  try:
   with (state/'HALTED').open('x') as handle:handle.write(json.dumps({'kind':'E9_ROOT_ADMISSION_REFUSED','error':str(error)})+'\n')
  except FileExistsError:pass
  raise

 if os.environ.get('E9_OWNER_PROBE')=='1':
  print('E9_OWNER_PROBE_VALIDATED_NO_PROVIDER',flush=True);time.sleep(30);return
 record=json.loads(Path(a.review).read_text())
 state=Path(record['state_dir']);anchor=state/'E9_ADMISSION_ANCHOR.json'
 if not anchor.exists():
  state.mkdir(parents=True,exist_ok=False)
  with anchor.open('x') as handle:json.dump({'review_sha256':a.review_sha256},handle)
 runner=Path(approved[0]).resolve();sys.path.insert(0,str(runner.parent))
 import runpod_api
 if Path(runpod_api.__file__).resolve()!=runner.parent/'runpod_api.py':raise ValueError('provider import binding differs')
 runpod_api.MAX_PODS=1;sys.argv=approved;runpy.run_path(str(runner),run_name='__main__')
if __name__=='__main__':main()
