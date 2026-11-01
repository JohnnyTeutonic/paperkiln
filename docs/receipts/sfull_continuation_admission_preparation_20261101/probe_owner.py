from pathlib import Path
import json,hashlib,subprocess,os,sys
r=Path('/home/jonat/sfull_continuation_admission_v4_20261101');review=r/'probe_review.json';v=json.loads(review.read_text());h=hashlib.sha256(review.read_bytes()).hexdigest();state=Path(v['state_dir'])
sys.path.insert(0,'/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools/colab_chat');import runpod_sweep
cmd=['python3',str(r/'guarded_continuation.py'),'--review',str(review),'--review-sha256',h,'--']+v['runner_argv'][1:]
p=subprocess.Popen(cmd,env=os.environ|{'SFULL_OWNER_PROBE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
try:
 lines=[]
 for line in p.stdout:
  lines.append(line)
  if 'SFULL_OWNER_PROBE_VALIDATED_NO_PROVIDER' in line:break
 else:raise AssertionError(''.join(lines)[-3000:])
 (state/'runner.pid').write_text(str(p.pid)+'\n');assert runpod_sweep.runner_alive(str(state))
 (state/'runner.pid').write_text(str(os.getpid())+'\n');assert not runpod_sweep.runner_alive(str(state))
 (state/'runner.pid').write_text(str(p.pid)+'\n');p.terminate();p.wait(timeout=10);assert not runpod_sweep.runner_alive(str(state))
 out={'pass':True,'live_owner_recognized':True,'unrelated_and_dead_pid_refused':True,'complete_checkpoint_validated':True,'provider_calls':0,'gpu_calls':0,'review_sha256':h};(r/'owner_probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=10)
 pid=state/'runner.pid'
 if pid.exists() and pid.read_text().strip() in [str(p.pid),str(os.getpid())]:pid.unlink()
