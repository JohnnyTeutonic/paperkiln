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
import sfull_deadline_recovery_v7 as g


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
        import datetime as dt
        r=copy.deepcopy(self.r)
        Path(r["old_halt"]).unlink();r["pins"].pop(r["old_halt"])
        original={'kind':'SFULL_FINANCIAL_HALT_RECOVERY_V6','seed':1,'source_control_commit':g.CONTROL,'numerical_commit':g.NUMERICAL,'deadline_utc':g.DEADLINE,'per_seed_cap_usd':12,'source_checkout':r['source_checkout'],'art_dir':r['art_dir'],'state_dir':str(self.base/'old'),'pins':{},'runner_argv':['python3',str(self.base/'source/tools/colab_chat/runner_multi.py'),'--max-usd','12','--state-dir',str(self.base/'old'),'--name','old','--deadline-utc',g.DEADLINE,'--max-hours','23.5']}
        r['original_admission']=put(self.base/'original.json',original)
        r['old_halt']=put(self.base/'old/STOPPED',{'kind':'SFULL_FINANCIAL_HALT_RECOVERY_REFUSED','error':'original absolute deadline expired'})
        self.driver='Traceback (most recent call last):\nRefused: original absolute deadline expired\n'
        self.log='11-02 22:41:15 pod old terminated (RunPod ledger $10.89)\n11-02 22:41:15 session stopped; runner exiting\n'
        r['old_driver_output']=str(self.base/'old/driver.out');Path(r['old_driver_output']).write_text(self.driver)
        r['old_runner_log']=str(self.base/'old/runner.log');Path(r['old_runner_log']).write_text(self.log)
        r.update(kind='SFULL_DEADLINE_RECOVERY_V7',status='ADMITTED',seed=1,per_seed_cap_usd=13.5,seeds2_and3_admitted=False,deadline_utc='2026-11-04T00:00:00Z',source_control_commit=g.CONTROL,numerical_commit=g.NUMERICAL)
        amendment={'kind':'PROSPECTIVE_SFULL_V7_DEADLINE_AND_COST_AMENDMENT','authority':'human3Nov','seed':1,'seeds2_and3_admitted':False,'numerical_commit':g.NUMERICAL,'admitted_at_utc':'2026-11-03T00:00:00Z','deadline_utc':r['deadline_utc'],'per_seed_cumulative_cap_usd':13.5,'cap_calculation':{'current_cumulative_usd':11,'remaining_cost_components_usd':{'eval':2},'carried_state_sha256':g.sha(r['old_state']),'evidence':[{'path':r['original_admission'],'sha256':g.sha(r['original_admission'])}],'proxy_limitations':'different matched cell'}}
        r['financial_amendment']=put(self.base/'amendment.json',amendment)
        r['runner_argv']=list(original['runner_argv'])
        for key,value in [('--max-usd','13.5'),('--state-dir',r['state_dir']),('--name',r['name']),('--deadline-utc',r['deadline_utc']),('--max-hours','24')]:r['runner_argv'][r['runner_argv'].index(key)+1]=value
        for key in ['original_admission','financial_amendment','old_halt','old_driver_output','old_runner_log']:r['pins'][r[key]]=g.sha(r[key])
        r['pins'][str(Path(g.__file__).resolve())]=g.sha(g.__file__)
        return r

    def check_config(self,r):
        import datetime as dt
        review=self.root/'admission.json';put(review,r)
        return g.configuration(str(review),g.sha(review),r['runner_argv'][1:],now=dt.datetime(2026,11,3,1,tzinfo=dt.timezone.utc))

    def test_prospective_configuration_and_recipe_refusal(self):
        r=self.configured();self.assertEqual(self.check_config(r),r)
        for key,value in [('per_seed_cap_usd',20.1),('deadline_utc','2026-11-05T00:00:00Z'),('numerical_commit','wrong'),('seeds2_and3_admitted',True)]:
            x=copy.deepcopy(r);x[key]=value
            with self.subTest(key=key),self.assertRaises(g.Refused):self.check_config(x)
        x=copy.deepcopy(r);x['runner_argv']+=['--steps','1600']
        with self.assertRaises(g.Refused):self.check_config(x)
        put(Path(r['state_dir'])/'HALTED',{'failure':'no waiver'})
        with self.assertRaises(g.Refused):self.check_config(r)

    def test_cap_formula_and_evidence_tampering(self):
        r=self.configured();amendment=g.read(r['financial_amendment'])
        for change in ['historical','unpin','ledger','margin']:
            x=copy.deepcopy(r);a=copy.deepcopy(amendment)
            if change=='historical':a['cap_calculation']['current_cumulative_usd']=50
            elif change=='unpin':a['cap_calculation']['evidence'][0]['sha256']='bad'
            elif change=='ledger':a['cap_calculation']['carried_state_sha256']='bad'
            else:a['per_seed_cumulative_cap_usd']=13.49;x['per_seed_cap_usd']=13.49
            put(Path(x['financial_amendment']),a);x['pins'][x['financial_amendment']]=g.sha(x['financial_amendment'])
            with self.subTest(change=change),self.assertRaises(g.Refused):self.check_config(x)

    def test_deadline_classifier(self):
        r=self.configured();stop=g.read(r['old_halt'])
        self.assertEqual(g.classify_deadline_stop(stop,self.driver,self.log),'EXACT_V6_DEADLINE_STOP')
        for stopped,driver,log in [(dict(stop,error='different'),self.driver,self.log),(stop,self.driver+'RuntimeError: corruption\n',self.log),(stop,self.driver,self.log+'laterfailure\n')]:
            with self.assertRaises(g.Refused):g.classify_deadline_stop(stopped,driver,log)

    def test_full_endpoint_and_payload_corruption(self):
        import sfull_checkpoint_v2 as cp
        import sfull_checkpoint_transport_v2 as tr
        from test_sfull_checkpoint_v2 import flat
        folder=self.base/'source/tools/colab_chat';folder.mkdir(parents=True)
        jobs=g.read(Path(g.__file__).parent/'carry12_sfull_v2_r1.json')['jobs']
        jobs=[j for j in jobs if j.get('full_checkpoint_v2')]
        put(folder/'carry12_sfull_v2_r1.json',{'jobs':jobs});self.r['minimum_home_step']=10000
        for job in jobs:
            root=tr.root_for(self.r['art_dir'],job);step=job['full_checkpoint_v2']['steps'];flat(root.parent,step);cp.publish(root.parent,step,1,16,1,g.NUMERICAL)
        self.assertEqual([x['step'] for x in g.validate_checkpoint(self.r)],[10000,1500])
        root=tr.root_for(self.r['art_dir'],jobs[-1]);generation,_=cp.current_unlocked(root)
        payload=generation/'optim.safetensors';data=bytearray(payload.read_bytes());data[-1]^=1;payload.write_bytes(data)
        with self.assertRaises(ValueError):g.validate_checkpoint(self.r)

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
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('DEFER_V7_LAUNCH_LOCK_HELD',result.stdout);self.assertFalse((self.root/'STOPPED').exists())


if __name__=='__main__':unittest.main()
