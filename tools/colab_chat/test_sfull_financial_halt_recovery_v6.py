import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import sfull_financial_halt_recovery_v6 as g


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value))
    return str(p)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'new';self.root.mkdir()
        ledger={'pods':[{'id':'oldpod','start':1.,'stop':2.,'usd_per_hr':1.}], 'volumes':[{'id':'working','created':1.,'gb':80}]}
        jobs={n:{'status':'FINISHED','home':True} for n in g.PREFLIGHTS}
        jobs.update(pt_c12_Sfull_r1={'status':'RUNNING','home':False},post_c12_Sfull_r1={'status':'WAITING','home':False})
        self.old={'jobs':jobs,'booted':True,'gone_ticks':5,'endpoint':None,'runpod':{'pod':None,'creating':None,'retired_volumes':[], 'volume':{'id':'working','dc':'dc','gb':80},'ledger':ledger}}
        self.r={'old_state':put(self.base/'old/state.json',self.old),'old_halt':put(self.base/'old/HALTED',g.V4_HALT),'v5_stopped':put(self.base/'v5/STOPPED',g.V5_STOP),'state_dir':str(self.root/'runner'),'working_volume_id':'working','working_volume_dc':'dc','protected_volume_ids':['protected'],'name':'new','source_checkout':str(self.base/'source'),'art_dir':str(self.base/'art'),'registry_path':put(self.base/'registry.json',{str(self.base/'old'):{'name':'old'},str(self.base/'other'):{'name':'unrelated'}})}
        self.r['pins']={self.r[k]:g.sha(self.r[k]) for k in ['old_state','old_halt','v5_stopped']}
        self.driver='Traceback (most recent call last):\nrunpod_api.Refused: '+g.FINANCIAL+'\nTraceback (most recent call last):\nValueError: '+g.V5_STOP['error']+'\n'
        self.log='terminated (RunPod ledger $7.00)\nsession stopped; runner exiting\n'

    def transfer(self):
        with patch.object(g,'validate_checkpoint',return_value=[{'step':7500}]):
            return g.prepare_state(self.r,[],False,self.r['registry_path'],lambda _:False)

    def configured(self):
        r=copy.deepcopy(self.r)
        original={'kind':'SFULL_CONTINUATION_ADMISSION_V4','source_control_commit':g.CONTROL,'numerical_commit':g.NUMERICAL,'deadline_utc':g.DEADLINE,'per_seed_cap_usd':7,'aggregate_cap_usd':21,'source_checkout':r['source_checkout'],'art_dir':r['art_dir'],'state_dir':str(self.base/'old'),'pins':{},'runner_argv':['python3',str(self.base/'source/tools/colab_chat/runner_multi.py'),'--max-usd','7','--state-dir',str(self.base/'old'),'--name','old','--checkpoint-interval','25','--checkpoint-gap-seconds','180']}
        r['original_admission']=put(self.base/'original.json',original)
        r['financial_amendment']=put(self.base/'financial.json',{'per_seed_cumulative_cap_usd':12,'three_seed_planning_reservation_usd':36,'seed':1,'seeds2_and3_admitted':False,'numerical_commit':g.NUMERICAL,'authority':'human','adopted_at_utc':'2026-11-02T00:00:00Z'})
        r['old_driver_output']=str(self.base/'old/driver.out');Path(r['old_driver_output']).write_text(self.driver)
        r['old_runner_log']=str(self.base/'old/runner.log');Path(r['old_runner_log']).write_text(self.log)
        r.update(kind='SFULL_FINANCIAL_HALT_RECOVERY_V6',status='ADMITTED',seed=1,per_seed_cap_usd=12,aggregate_cap_usd=36,seeds2_and3_admitted=False,deadline_utc=g.DEADLINE,source_control_commit=g.CONTROL,numerical_commit=g.NUMERICAL)
        r['runner_argv']=list(original['runner_argv'])
        for key,value in [('--max-usd','12'),('--state-dir',r['state_dir']),('--name',r['name'])]:r['runner_argv'][r['runner_argv'].index(key)+1]=value
        for key in ['original_admission','financial_amendment','old_driver_output','old_runner_log']:r['pins'][r[key]]=g.sha(r[key])
        r['pins'][str(Path(g.__file__).resolve())]=g.sha(g.__file__)
        return r

    def check_config(self,r):
        import datetime as dt
        review=self.root/'admission.json';put(review,r)
        return g.configuration(str(review),g.sha(review),r['runner_argv'][1:],now=dt.datetime(2026,11,2,10,tzinfo=dt.timezone.utc))

    def test_configuration_financial_only_and_new_halt(self):
        r=self.configured();self.assertEqual(self.check_config(r),r)
        for key,value in [('per_seed_cap_usd',13),('aggregate_cap_usd',37),('numerical_commit','other'),('deadline_utc','2026-11-03T00:00:00Z'),('seeds2_and3_admitted',True)]:
            x=copy.deepcopy(r);x[key]=value
            with self.subTest(key=key),self.assertRaises(g.Refused):self.check_config(x)
        x=copy.deepcopy(r);x['runner_argv'][-1]='181'
        with self.assertRaises(g.Refused):self.check_config(x)
        x=copy.deepcopy(r);x['pins'].pop(r['old_state'])
        with self.assertRaises(g.Refused):self.check_config(x)
        put(Path(r['state_dir'])/'HALTED',{'error':'new failure'})
        with self.assertRaises(g.Refused):self.check_config(r)

    def test_original_source_and_financial_authority_bound(self):
        r=self.configured()
        original=g.read(r['original_admission']);original['numerical_commit']='other';put(Path(r['original_admission']),original)
        r['pins'][r['original_admission']]=g.sha(r['original_admission'])
        with self.assertRaises(g.Refused):self.check_config(r)
        r=self.configured();amendment=g.read(r['financial_amendment']);amendment['seed']=2;put(Path(r['financial_amendment']),amendment)
        r['pins'][r['financial_amendment']]=g.sha(r['financial_amendment'])
        with self.assertRaises(g.Refused):self.check_config(r)

    def test_complete_checkpoint_payload_validation(self):
        import sfull_checkpoint_v2 as cp
        import sfull_checkpoint_transport_v2 as tr
        from test_sfull_checkpoint_v2 import flat
        folder=self.base/'source/tools/colab_chat';folder.mkdir(parents=True)
        job={'name':'pt_c12_Sfull_r1','full_checkpoint_v2':{'root':'checkpoint','source_commit':g.NUMERICAL,'seed':1,'batch':16,'accum':1}}
        # Use the actual registered shape, not a substitute manifest contract.
        actual=g.read(Path(g.__file__).parent/'carry12_sfull_v2_r1.json')['jobs']
        job=next(j for j in actual if j['name']=='pt_c12_Sfull_r1')
        put(folder/'carry12_sfull_v2_r1.json',{'jobs':[job]})
        self.r['minimum_home_step']=7500
        root=tr.root_for(self.r['art_dir'],job);flat(root.parent,7500)
        cp.publish(root.parent,7500,1,16,1,g.NUMERICAL)
        self.assertEqual(g.validate_checkpoint(self.r)[0]['step'],7500)
        generation,_=cp.current_unlocked(root)
        data=bytearray((generation/'optim.safetensors').read_bytes());data[-1]^=1;(generation/'optim.safetensors').write_bytes(data)
        with self.assertRaises(ValueError):g.validate_checkpoint(self.r)

    def test_exact_classifier(self):
        self.assertEqual(g.classify_financial_halt(g.V4_HALT,g.V5_STOP,self.driver,self.log),'EXACT_FINANCIAL_TRANSPORT_HALT')
        for halt,driver,log in [(dict(g.V4_HALT,extra=True),self.driver,self.log),(dict(g.V4_HALT,error='budget'),self.driver,self.log),(g.V4_HALT,self.driver+'RuntimeError: corruption\n',self.log),(g.V4_HALT,self.driver.replace('Traceback','Other',1),self.log),(g.V4_HALT,self.driver,self.log+'later failure\n')]:
            with self.subTest(halt=halt,driver=driver,log=log),self.assertRaises(g.Refused):g.classify_financial_halt(halt,g.V5_STOP,driver,log)

    def test_ledger_mutations(self):
        original=self.old['runpod']['ledger'];g.ledger_check(original)
        mutations=[]
        for key,value in [('start',time.time()+99),('stop',time.time()+99),('usd_per_hr',float('nan')),('stop',None)]:
            x=copy.deepcopy(original);x['pods'][0][key]=value;mutations.append(x)
        x=copy.deepcopy(original);x['pods']*=2;mutations.append(x)
        for x in mutations:
            with self.subTest(x=x),self.assertRaises(g.Refused):g.ledger_check(x)
        for kind in ['pods','volumes']:
            x=copy.deepcopy(original);x[kind]=[]
            with self.assertRaises(g.Refused):g.retain_ledger(original,x)
        x=copy.deepcopy(original);x['pods'][0]['usd_per_hr']=2
        with self.assertRaises(g.Refused):g.retain_ledger(original,x)

    def test_transfer_preserves_and_moves_one_owner(self):
        old_bytes=Path(self.r['old_state']).read_bytes();before=g.read(self.r['registry_path'])
        proof=self.transfer();new=g.validate_prepared(self.r)
        self.assertFalse(new['booted']);self.assertEqual(new['gone_ticks'],0)
        self.assertEqual(new['runpod'],self.old['runpod']);self.assertEqual(Path(self.r['old_state']).read_bytes(),old_bytes)
        after=g.read(self.r['registry_path']);self.assertNotIn(str(self.base/'old'),after)
        self.assertEqual(after[str(self.base/'other')],before[str(self.base/'other')]);self.assertIn(self.r['state_dir'],after)
        self.assertEqual(proof['registry_before'],before)
        with self.assertRaises(g.Refused):self.transfer()
        new['runpod']['ledger']['pods']=[];put(Path(self.r['state_dir'])/'state.json',new)
        with self.assertRaises(g.Refused):g.validate_prepared(self.r)

    def test_dispatch_owner_and_predecessor_pid_recheck(self):
        self.transfer();g.validate_prepared(self.r)
        registry=g.read(self.r['registry_path']);registry[str(self.base/'old')]={'name':'old'};put(Path(self.r['registry_path']),registry)
        with self.assertRaises(g.Refused):g.validate_prepared(self.r)
        registry.pop(str(self.base/'old'));put(Path(self.r['registry_path']),registry)
        (self.base/'old/runner.pid').write_text(str(os.getpid()))
        with self.assertRaises(g.Refused):g.validate_prepared(self.r)
        (self.base/'old/runner.pid').unlink()
        put(self.base/'other/state.json',{'runpod':{'retired_volumes':[{'id':'working'}]}})
        with self.assertRaises(g.Refused):g.validate_prepared(self.r)

    def test_transfer_refuses_live_or_duplicate_owner(self):
        for pods,alive,registry_alive in [([{}],False,False),([],True,False),([],False,True)]:
            with patch.object(g,'validate_checkpoint',return_value=[]),self.assertRaises(g.Refused):g.prepare_state(self.r,pods,alive,self.r['registry_path'],lambda _:registry_alive)
        put(self.base/'other/state.json',{'runpod':{'volume':{'id':'working'}}})
        with self.assertRaises(g.Refused):self.transfer()

    def test_path_aliases_and_evidence_overlap(self):
        g.path_safety(self.r)
        for destination in [self.base/'old/new',self.base/'source/new',self.base/'art/new',self.base]:
            r=dict(self.r,state_dir=str(destination/'runner'))
            with self.subTest(destination=destination),self.assertRaises(g.Refused):g.path_safety(r)
        (self.base/'alias').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(g.Refused):g.path_safety(dict(self.r,state_dir=str(self.base/'alias/runner')))
        alias=self.base/'registry_alias';alias.symlink_to(self.r['registry_path'])
        with self.assertRaises(g.Refused):g.path_safety(dict(self.r,registry_path=str(alias)))
        with patch.object(g,'validate_checkpoint',return_value=[]),self.assertRaises(g.Refused):g.prepare_state(self.r,[],False,str(self.base/'other_registry'),lambda _:False)

    def test_refusal_marker_exclusive(self):
        g.record_refusal(self.root,'first');first=(self.root/'STOPPED').read_bytes()
        g.record_refusal(self.root,'second');self.assertEqual((self.root/'STOPPED').read_bytes(),first)

    def test_live_pid(self):
        state=self.root/'runner';state.mkdir();(state/'runner.pid').write_text(str(os.getpid()))
        self.assertTrue(g.live_state_pid(state));(state/'runner.pid').write_text('-1');self.assertFalse(g.live_state_pid(state))

    def test_launch_lock_defers_before_configuration(self):
        script=self.root/'guard.py';script.write_bytes(Path(g.__file__).read_bytes())
        with (self.root/'launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=subprocess.run([sys.executable,str(script),'--review','missing','--review-sha256','missing'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('DEFER_V6_LAUNCH_LOCK_HELD',result.stdout);self.assertFalse((self.root/'STOPPED').exists())


if __name__=='__main__':unittest.main()
