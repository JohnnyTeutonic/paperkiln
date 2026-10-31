"""Independent offline raw-log replay for the four-case LayerNorm CUDA gate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

CASES=('forward_x','forward_gamma','forward_beta','backward_gamma')
FORWARD='std::fabs(y->data(i, j) - expected[i * C + j]) / (1 + std::fabs(expected[i * C + j])) < 2e-5'
BACKWARD='std::fabs(x0->grad(i, j) - dx[i * C + j]) / (1 + std::fabs(dx[i * C + j])) < 2e-5'
PINS={'src/ops.cpp':'94e4d38b17066362dc9e40d7ea64705ccb3834b89034756f3d929453236dd81a','tests/test_layernorm_materialization.cpp':'e13d9a0cc7e8013ed908db448327dcb2e71d3abf6d050764488dbf94ea251535','CMakeLists.txt':'217b24c7e3278d723b98e0aa59ce982eac72ea66abeddbd257f33a0bb9f76550'}
BASELINE='9578ee8cfc3cd5c98f201df71834291a37b73585a114733a335cf7779ccf2abd'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,reason):
 if not value:raise ValueError(reason)
def rows(text):return [json.loads(x) for x in text.splitlines() if x.startswith('{')]
def audit(home,stage):
 home,stage=Path(home),Path(stage);r=json.loads((home/'gate_summary.json').read_text());m=json.loads((stage/'layernorm_inputs.json').read_text())
 require(r['pass'] is True and r['errors']==[] and r['source_restored'] is True and r['baseline_expected_numerical_failure'] is True,'verdict/restoration differs')
 require(r['inputs']==m,'manifest differs')
 for name,h in m['files'].items():require(sha(stage/name)==h,'uploaded input differs '+name)
 expected_logs={'hardware.log','nvcc.log','configure.log','baseline_build.log','fixed_build.log'}|{p+'_'+c+'.log' for p in ('baseline','fixed') for c in CASES}
 require(set(r['logs'])==expected_logs,'log inventory differs')
 for name,v in r['logs'].items():require(sha(home/name)==v['sha256'] and (home/name).stat().st_size==v['bytes'],'raw log hash/size differs '+name)
 expected_commands=['hardware','nvcc','configure','baseline_build']+['baseline_'+c for c in CASES]+['fixed_build']+['fixed_'+c for c in CASES]
 require([c['name'] for c in r['commands']]==expected_commands,'command inventory differs')
 require([c['returncode'] for c in r['commands']]==[0]*4+[1]*4+[0]*5,'returncode inventory differs')
 hardware=(home/'hardware.log').read_text().strip().splitlines();require(len(hardware)==1 and hardware[0].startswith('NVIDIA L4,'),'actual L4 absent')
 require(all(x in r['commands'][2]['argv'] for x in ('-DMICROTORCH_CUDA=ON','-DMICROTORCH_DEVCHECK=ON','-DCMAKE_CUDA_ARCHITECTURES=89')),'build contract differs')
 require(set(r['baseline'])==set(CASES) and set(r['fixed'])==set(CASES),'case inventory differs')
 for case in CASES:
  for phase in ('baseline','fixed'):
   c=next(c for c in r['commands'] if c['name']==phase+'_'+case);require(c['argv'][-3:]==['--require-cuda','--case',case],'case CUDA command differs')
  baseline=(home/('baseline_'+case+'.log')).read_text();proof={'case':case,'actual_cuda':True,'stale_operand_proved':True,'host_poison':True,'ops_disabled_before_fallback':True};require(proof in rows(baseline),'actual stale/fallback proof absent')
  expression=BACKWARD if case=='backward_gamma' else FORWARD
  require(re.search(r'^CHECK failed .*test_layernorm_materialization\.cpp:[0-9]+: '+re.escape(expression)+r'$',baseline,re.MULTILINE),'wrong baseline numeric assertion')
  require(r['baseline'][case]=={'returncode':1,'registered_numerical_failure':True},'registered baseline verdict differs')
  fixed=(home/('fixed_'+case+'.log')).read_text();values=rows(fixed);final=values[-1]
  require(final==r['fixed'][case] and all(final.get(k) is True for k in ('pass','cuda_integration','devcheck_enabled')) and type(final.get('cases')) is int and final['cases']==1,'fixed final proof differs')
  require(any(v.get('case')==case and v.get('pass') is True and v.get('cuda_integration') is True and v.get('forward_backward_reference') is True for v in values),'fixed case proof absent')
 require(r['baseline_ops_sha256']==sha(stage/'layernorm_baseline_ops.cpp')==BASELINE,'baseline optimized source differs')
 with tarfile.open(stage/'layernorm_source.tgz') as archive:
  for name,h in PINS.items():require(hashlib.sha256(archive.extractfile(name).read()).hexdigest()==h,'fixed source archive differs '+name)
  require(archive.getmember('studio/chat.html').size>0,'configure asset absent')
 require(r['fixed_ops_sha256']==PINS['src/ops.cpp'] and r['baseline_binary_sha256']!=r['fixed_binary_sha256'],'fixed source/binary identity differs')
 return {'pass':True,'scope':'Offline registered four-case source/manifest/log/command/proof replay; no fresh device work','report_sha256':sha(home/'gate_summary.json'),'manifest_sha256':sha(stage/'layernorm_inputs.json'),'raw_report':r,'raw_test_logs':{n:(home/n).read_text() for n in sorted(expected_logs) if n.startswith(('baseline_','fixed_')) and not n.endswith('build.log')},'limitations':['Only explicitly isolated LayerNorm fallback cells were tested; broad host-loop coherence audit remains open.','No training, whole-model, throughput or endurance claim. Historical pointer-loop timing belongs to1b0102709/d960e78bc; this materialization binary has not been retimed.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();result=audit(a.home,a.stage)
 if a.output:
  if a.output.exists():raise FileExistsError('immutable replay output exists')
  a.output.parent.mkdir(parents=True,exist_ok=True);t=a.output.with_suffix('.partial');t.write_text(json.dumps(result,indent=1)+'\n');t.replace(a.output)
 print(json.dumps({'pass':True,'report_sha256':result['report_sha256']}))
