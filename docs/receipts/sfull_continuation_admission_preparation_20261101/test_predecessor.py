from pathlib import Path
import importlib.util,copy,json
r=Path('/home/jonat/sfull_continuation_admission_v4_20261101');s=importlib.util.spec_from_file_location('prepare',r/'prepare_dispatch.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
base={'campaign_home':True,'endpoint':None,'runpod':{'pod':None,'volume':None,'retired_volumes':[],'ledger':{'pods':[{'id':'old','stop':1}],'volumes':[{'id':'oldv','deleted':1}]}},'jobs':{'train':{'home':True}}};m.predecessor_ready(base);count=1
for case in ['not_home','endpoint','pod','volume','retired','pod_unstopped','volume_undeleted','outputs_not_home']:
 v=copy.deepcopy(base)
 if case=='not_home':v['campaign_home']=False
 if case=='endpoint':v['endpoint']='remote'
 if case in ['pod','volume']:v['runpod'][case]={'id':'active'}
 if case=='retired':v['runpod']['retired_volumes']=[{'id':'oldv'}]
 if case=='pod_unstopped':v['runpod']['ledger']['pods'][0].pop('stop')
 if case=='volume_undeleted':v['runpod']['ledger']['volumes'][0].pop('deleted')
 if case=='outputs_not_home':v['jobs']['train']['home']=False
 try:m.predecessor_ready(v)
 except ValueError:count+=1
 else:raise AssertionError(case)
(r/'predecessor_validation.json').write_text(json.dumps({'cases_passed':count,'provider_calls':0,'gpu_calls':0},indent=2)+'\n');print('Serial predecessor tests: '+str(count)+' pass')
