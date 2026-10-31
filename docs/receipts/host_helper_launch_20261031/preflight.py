#!/usr/bin/env python3
"""Offline operational preflight: imports/parses real drivers, never creates resources."""
import argparse,hashlib,importlib.util,json,os,runpy,shutil,subprocess,sys
from pathlib import Path
from unittest import mock
STAGE=Path('/home/jonat/host_helper_launch_v1_20261031')
SCIENCE=Path('/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031')
REPO=Path('/home/jonat/pk_host_helper_launch_20261031')
OUT=STAGE/'preflight'
OUT.mkdir()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
guard=load('owned_guard',STAGE/'host_helper_guarded_runner.py')
review=json.loads((STAGE/'launch_review.json').read_text())
argv=review['driver_argv'][1:]
api=guard.install_ceiling(argv[0])
if api.MAX_PODS!=2:raise ValueError('ceiling not installed')
refusals=[]
for n in (2,3,13):
    with mock.patch.object(api,'list_pods',return_value=[{}]*n),mock.patch.object(api,'balance',side_effect=RuntimeError('unexpected balance query')),mock.patch.object(api,'rest',side_effect=RuntimeError('unexpected POST')):
        try:api.create_pod({})
        except api.Refused:refusals.append(n)
        else:raise ValueError('pod ceiling not enforced')
with mock.patch.object(sys,'argv',['guard']+argv),mock.patch.object(guard.runpy,'run_path') as dispatch:
    guard.main()
    if dispatch.call_count!=1:raise ValueError('dispatch not exact')
with mock.patch.object(sys,'argv',['guard']+argv+['--tamper']):
    try:guard.main()
    except ValueError:pass
    else:raise ValueError('argv mutation accepted')
key=next(iter(guard.PINS));original=guard.PINS[key];guard.PINS[key]='0'*64
try:
    try:guard.install_ceiling(argv[0])
    except ValueError:pass
    else:raise ValueError('source mutation accepted')
finally:guard.PINS[key]=original
class Parsed(Exception):pass
real_parse=argparse.ArgumentParser.parse_args
parsed={}
def parse(parser,*args,**kw):
    ns=real_parse(parser,*args,**kw);parsed.update(vars(ns));raise Parsed()
with mock.patch.object(sys,'argv',argv),mock.patch.object(argparse.ArgumentParser,'parse_args',parse):
    try:runpy.run_path(argv[0],run_name='__main__')
    except Parsed:pass
if parsed['jobs']!='tools/colab_chat/host_helper_gate_20261031.json' or parsed['max_usd']!=.75 or parsed['deadline_utc']!=__import__('datetime').datetime(2026,10,31,11,15,tzinfo=__import__('datetime').timezone.utc).timestamp():raise ValueError('parsed operational values differ')
runner=runpy.run_path(argv[0])
boot=runner['boot_launch_code'](parsed['commit'],parsed['jobs'],parsed['boot_script'])
if parsed['commit'] not in boot or parsed['jobs'] not in boot:raise ValueError('boot binding differs')
subprocess.run(['bash','-n',str(REPO/parsed['boot_script'])],check=True)
# Run the actual jobs manager and actual frozen gate wrapper with a deliberately invalid manifest.
chat=OUT/'manager_chat';chat.mkdir();(chat/'upload').mkdir()
jobs=json.loads((REPO/parsed['jobs']).read_text());cmd=jobs['jobs'][0]['cmd'];cmd[cmd.index('/content/chat/upload/host_helper_gate.py')]=str(SCIENCE/'host_helper_gate.py');cmd[cmd.index('--upload')+1]=str(SCIENCE);cmd[cmd.index('--manifest-sha256')+1]='0'*64
jpath=OUT/'refusal_jobs.json';jpath.write_text(json.dumps(jobs))
env=dict(os.environ,JOBS_CHAT_DIR=str(chat),JOBS_REPO=str(REPO),JOBS_TICK='.05')
p=subprocess.run(['python3',str(REPO/'tools/colab_chat/jobs_vm.py'),str(jpath)],env=env,capture_output=True,text=True,timeout=20)
(OUT/'manager.stdout').write_text(p.stdout);(OUT/'manager.stderr').write_text(p.stderr)
gate=json.loads((chat/'host_helper_gate/gate_summary.json').read_text())
if p.returncode!=0 or gate['pass'] is not False or gate['errors']!=['manifest differs'] or gate['commands']:raise ValueError('real wrapper refusal preflight failed')
# Execute actual pinned CPU replay, then require rejection of CPU-as-CUDA through the actual host guard.
host=load('host_guard',STAGE/'host_replay_guard.py');host.verify()
control=OUT/'host_control';control.mkdir()
fixture=REPO/'tests/fixtures/host_helper_gate_cpu.json'
for mode in ('cpu','cuda'):shutil.copyfile(fixture,control/(mode+'.json'))
host.LAND=control;host.OUT=OUT/'host_control_replay'
try:host.main()
except ValueError as error:
    if str(error)!='cuda replay failed':raise
else:raise ValueError('CPU substituted for CUDA')
cpu=json.loads((host.OUT/'cpu.stdout').read_text())
if cpu['pass'] is not True or cpu['actual_cuda'] is not False:raise ValueError('CPU replay failed')
host.verify()
key=next(iter(host.PINS));original=host.PINS[key];host.PINS[key]='0'*64
try:
    try:host.verify()
    except ValueError:pass
    else:raise ValueError('host pin mutation accepted')
finally:host.PINS[key]=original
record={'pass':True,'no_network_no_local_GPU':True,'synthetic_pod_refusals_before_balance_or_POST':refusals,'installed_process_local_cap':api.MAX_PODS,'exact_dispatch_mocked':True,'argv_and_driver_hash_mutations_rejected':True,'actual_runner_parser':parsed,'actual_boot_launch_code':boot,'bash_syntax_pass':True,'real_jobs_vm_with_real_frozen_wrapper_refusal':{'returncode':p.returncode,'expected_gate_pass':False,'expected_error':gate['errors'],'commands_executed':gate['commands']},'actual_host_CPU_replay':cpu,'actual_host_CPU_as_CUDA_refused':True,'host_pin_mutation_refused':True,'scientific_manifest_unchanged':hashlib.sha256((SCIENCE/'inputs.json').read_bytes()).hexdigest(),'guard_bindings':guard.PINS,'status':'UNLAUNCHED_NO_WATCHDOG_ENTRY'}
(OUT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS offline operational preflight; no network or GPU')
