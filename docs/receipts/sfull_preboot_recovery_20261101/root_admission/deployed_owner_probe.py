"""Actual admitted guard process, environment-only no-provider branch. No real PID-file writes."""
import sys,subprocess,json,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch')
sys.path.insert(0,str(ROOT/'tools/colab_chat'))
import runpod_sweep as s
HERE=Path(__file__).resolve().parent
GUARD='/home/jonat/sfull_boot_recovery_root_20261101/guarded_recovery.py'
ADMISSION='/home/jonat/sfull_boot_recovery_root_20261101/recovery_admission.json'
STATE='/home/jonat/sfull_v2_states_20261101/pk_c12_sfull_r1_runner'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
known=s.ADMITTED_ROOT_GUARDS[GUARD]
assert sha(GUARD)==known['sha256'] and sha(ADMISSION)==known['admission_sha256']
record=json.loads(Path(ADMISSION).read_text());assert record['adopted'] is True
match=GUARD+' '+ADMISSION;pattern='['+match[0]+']'+match[1:]
def matching():
 p=subprocess.run(['pgrep','-f',pattern],capture_output=True,text=True)
 if p.returncode not in [0,1]:raise ValueError('pgrep failed')
 return [int(value) for value in p.stdout.split()]
assert matching()==[], 'another real guard exists; leave it alone'
files=set(record['source_pins'])|{GUARD,ADMISSION,record['original_guard'],record['original_admission']}
before={name:sha(name) for name in files}
pidfile=Path(STATE)/'runner.pid';pid_before=pidfile.read_bytes() if pidfile.exists() else None
process=subprocess.Popen(['env','S_FULL_OWNER_PROBE=1','PYTHONDONTWRITEBYTECODE=1','python3',GUARD,ADMISSION],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
 marker=process.stdout.readline().strip();assert marker=='OWNER_PROBE_VALIDATED_NO_PROVIDER_CALLS'
 snapshot=s.process_snapshot(process.pid)
 assert snapshot['args']==['python3',GUARD,ADMISSION]
 assert s.admitted_root_guard_pid_alive(process.pid,STATE)
 assert matching()==[process.pid]
 assert not s.admitted_root_guard_pid_alive(__import__('os').getpid(),STATE)
 stdout,stderr=process.communicate(timeout=35)
 assert process.returncode==0 and not stdout.strip() and not stderr.strip()
 assert not s.admitted_root_guard_pid_alive(process.pid,STATE)
 assert matching()==[]
 assert before=={name:sha(name) for name in files}
 assert pid_before==(pidfile.read_bytes() if pidfile.exists() else None)
 result=dict(status='PASS_ACTUAL_NO_PROVIDER_OWNER_PROBE',guard_sha256=sha(GUARD),admission_sha256=sha(ADMISSION),sweeper_sha256=sha(ROOT/'tools/colab_chat/runpod_sweep.py'),process_pid=process.pid,process_start_ticks=snapshot['start_ticks'],actual_argv=snapshot['args'],predicate_live=True,predicate_after_exit=False,bracketed_watchdog_match=pattern,matching_live_pid_count=1,matching_after_exit_count=0,provider_calls=0,credential_accesses=0,gpu_jobs=0,original_pidfile_unchanged=True,all_pinned_files_unchanged=True,deployed_sweeper_changed=False,configuration_changed=False,pinned_files_sha256=before)
 with (HERE/'actual_owner_probe.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({key:result[key] for key in ['status','process_pid','predicate_live','predicate_after_exit','matching_live_pid_count','original_pidfile_unchanged']}))
finally:
 if process.poll() is None:process.terminate();process.wait(timeout=5)
 process.stdout.close();process.stderr.close()
