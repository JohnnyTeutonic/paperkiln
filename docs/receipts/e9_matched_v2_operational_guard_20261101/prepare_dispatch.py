"""Offline dispatch record constructor. Does not enable watchdog or call provider.
Root must supply a fresh serial-check receipt after previous GPU campaign cleanup.
The absolute two-hour window is set here, at actual dispatch, never during queue preparation.
"""
from pathlib import Path
import argparse,datetime,hashlib,json
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(now):return now.astimezone(datetime.timezone.utc).isoformat()
def prepare(plan,output,serial,guard,now=None):
 now=now or datetime.datetime.now(datetime.timezone.utc)
 output=Path(output);guard=Path(guard).resolve();serial=Path(serial).resolve()
 serial_record=json.loads(serial.read_text())
 checked=datetime.datetime.fromisoformat(serial_record['checked_at_utc'].replace('Z','+00:00'))
 if checked.tzinfo is None or not 0<=(now-checked).total_seconds()<=120:raise ValueError('serial check must be within two minutes')
 if (serial_record.get('gpu_campaigns_active')!=0 or serial_record.get('previous_campaign_home') is not True
     or serial_record.get('previous_owned_resources_deleted') is not True):raise ValueError('serial cleanup not verified')
 for name in ['state_dir','art_dir']:
  p=Path(plan[name])
  if p.exists() or p.is_symlink():raise ValueError('fresh campaign directory already exists: '+name)
 for path,digest in plan['pins'].items():
  p=Path(path)
  if not p.is_file() or p.is_symlink() or str(p.resolve())!=path or sha(path)!=digest:raise ValueError('source/input pin differs: '+path)
 if output.exists():raise ValueError('dispatch record output already exists')
 output.mkdir(parents=True)
 att=output/'fresh_paths.json';att.write_text(json.dumps({k:plan[k] for k in ['state_dir','art_dir']}|{'both_absent_at_prepare':True},indent=2)+'\n')
 deadline=stamp(now+datetime.timedelta(hours=2));source=Path(plan['source']);runner=source/'tools/colab_chat/runner_multi.py'
 argv=['python3',str(runner),'--backend','runpod','--gpu','L4','--cloud','SECURE','--volume-gb','30','--max-usd','2','--max-hours','2','--commit',plan['runner_source_commit'],'--jobs','tools/colab_chat/ember_e9_matched_20261101.json','--boot-script','tools/colab_chat/boot_commands.sh','--name','e9-matched-v2-20261101','--state-dir',plan['state_dir'],'--art',plan['art_dir'],'--deadline-utc',deadline]
 for key,value in plan['uploads'].items():argv+=['--upload',key+'='+value]
 pins=dict(plan['pins']);pins.pop(plan['guard'],None);pins[str(guard)]=sha(guard);pins[str(serial)]=sha(serial)
 record=dict(plan,kind='ROOT_E9_MATCHED_SERIAL_ADMISSION_V2',status='APPROVED_AFTER_OFFLINE_AND_SERIAL_CHECKS',driver_argv=argv,upload_names=list(plan['uploads']),pins=pins,admitted_at_utc=stamp(now),deadline_utc=deadline,jobs_relative='tools/colab_chat/ember_e9_matched_20261101.json',fresh_paths_attestation=str(att),fresh_paths_attestation_sha256=sha(att),serial_check=str(serial),guard=str(guard))
 review=output/'review.json';review.write_text(json.dumps(record,indent=2)+'\n')
 result={'review':str(review),'review_sha256':sha(review),'guard_argv':['python3',str(guard),'--review',str(review),'--review-sha256',sha(review),'--']+argv[1:],'admitted_at_utc':stamp(now),'deadline_utc':deadline,'provider_called':False,'watchdog_enabled':False}
 (output/'dispatch.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--serial-check',required=True);p.add_argument('--guard',default=str(BASE/'proposed_guarded_runner.py'));a=p.parse_args()
 print(json.dumps(prepare(json.loads(Path(a.plan).read_text()),a.output,a.serial_check,a.guard),indent=2))
