"""Prospective deterministic HS1-only five-seed analysis; no compute rental or model calls."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import finalize_core as common
import prepare_hs1_extension as preparation

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SEEDS=[1,2,3,4,5]
CELLS=['B','Sgist']
REPORT='analysis_hs1.json'
SOURCES=(*common.SOURCE_FILES,Path(__file__),HERE/'prepare_hs1_extension.py',HERE/'hs1_extension_manifest.json',
         HERE/'hs1_extension_launch_pin.json',HERE/'receipts/core_2026-10-30/analysis_carry12.json',
         HERE/'receipts/core_2026-10-30/completion.json',
         *sorted((REPO/'tools/colab_chat').glob('carry12_hs1_*_r[45].json')))


def calculate(art,out):
    """Invoke unchanged registered extraction, floor gate and paired decision helpers."""
    a=common.load_module('hs1_registered',HERE/'analyze.py')
    runs={}
    for cell in CELLS:
        for seed in SEEDS:
            p=Path(art)/f'c12_{cell}_r{seed}'/f'post_c12_{cell}_r{seed}'/'summary.json'
            runs[(cell,seed)]=common.load_json(p)
    floors={c:a.floor_ok(runs,c,SEEDS) for c in CELLS}
    if not all(floors.values()):
        result={'verdict':'FLOOR','cells':floors}
    else:
        result=a.decide(a.paired(runs,'Sgist','B',SEEDS,'Y12'),a.SESOI,-1)
    common.atomic_json(out,{'scope':'REGISTERED HS1 EXTENSION ONLY (PREREGISTRATION.md section 9)',
        'partial':False,'seeds':SEEDS,'cells':CELLS,'missing':[], 'validity_gate':floors,
        'cell_means':a.cell_table(runs,CELLS,SEEDS),'HS1_Sgist_minus_B':result,
        's_full_condition':{'run_s_full':result['verdict']=='SUPPORTED','HS1_verdict':result['verdict']},
        'global_core_report_unchanged':True,'no_further_seed_extension':True})


class Finalizer(common.Finalizer):
    def state(self):
        if self.state_path.exists():
            return super().state()
        s={'schema':1,'created':time.time(),'art':str(self.art),'output':str(self.output),'status':'WAITING',
           'python':sys.version,'source_hashes':{str(p.resolve()):common.sha(p) for p in SOURCES}}
        common.atomic_json(self.state_path,s)
        return s

    def inputs(self):
        original=HERE/'receipts/core_2026-10-30/analysis_carry12.json'
        preparation.trigger(common.load_json(original))
        pending=[];files={str(original):common.sha(original)};pairs=[]
        original_completion=common.load_json(HERE/'receipts/core_2026-10-30/completion.json')
        for path,digest in original_completion['input_manifest']['sha256'].items():
            # Original inputs remain exactly the ones supporting the three-seed trigger.
            # Resolve their artifact-relative tails so an offline fixture can be isolated.
            marker='/tiny_chat/'
            if marker not in path:raise ValueError('unknown original artifact path: '+path)
            p=self.art/path.split(marker,1)[1]
            if not p.is_file():pending.append(str(p));continue
            if common.sha(p)!=digest:raise ValueError('original core input changed: '+str(p))
            files[str(p)]=digest
        a=common.load_module('hs1_registered_inputs',HERE/'analyze.py')
        pin=common.load_json(HERE/'hs1_extension_launch_pin.json')['commit']
        wanted={(c,s) for c in CELLS for s in SEEDS}
        found={}
        for p in self.art.glob('*/post_c12_*/summary.json'):
            row=common.load_json(p);key=(row.get('cell'),row.get('seed'))
            if key in wanted:
                if key in found:raise ValueError('duplicate HS1 report: '+str(key))
                found[key]=p
        for cell in CELLS:
            ref=self.art/f'c12_{cell}_r1/data/data.ledger.json'
            if not ref.is_file():pending.append(str(ref));continue
            reference=common.data_ledger(common.load_json(ref))
            for seed in SEEDS:
                base=self.art/f'c12_{cell}_r{seed}'
                p=base/f'post_c12_{cell}_r{seed}/summary.json'
                if not p.is_file():pending.append(str(p))
                else:
                    row=common.load_json(p)
                    metric=a.get(row,'Y12')
                    if (row.get('cell')!=cell or type(row.get('seed')) is not int or row['seed']!=seed
                        or row.get('mode')!='post' or type(metric) not in (int,float) or not math.isfinite(metric)):
                        raise ValueError('invalid HS1 summary identity/metric: '+str(p))
                    if seed>=4 and row.get('provenance',{}).get('paperkiln_commit')!=pin:
                        raise ValueError('extension source pin mismatch: '+str(p))
                    if row.get('carry12',{}).get('context')!=1024:
                        raise ValueError('wrong context: '+str(p))
                    files[str(p)]=common.sha(p);pairs.append([cell,seed])
                ledger=base/'data/data.ledger.json';gate=base/'gate/gate.json'
                for path in (ledger,gate):
                    if not path.is_file():pending.append(str(path));continue
                    value=common.load_json(path)
                    if path==ledger and common.data_ledger(value)!=reference:
                        raise ValueError('data ledger differs: '+str(path))
                    if path==gate and (value.get('ok') is not True or value.get('problems')!=[]):
                        raise ValueError('execution gate failed: '+str(path))
                    files[str(path)]=common.sha(path)
        return {'ready':not pending,'pending':pending,'selected_cell_seed_pairs':pairs,'sha256':files}

    @staticmethod
    def verify_outputs(folder):
        r=common.load_json(Path(folder)/REPORT)
        if (r.get('scope')!='REGISTERED HS1 EXTENSION ONLY (PREREGISTRATION.md section 9)'
            or r.get('partial') is not False or r.get('seeds')!=SEEDS or r.get('cells')!=CELLS
            or any(type(seed) is not int for seed in r.get('seeds',[]))
            or r.get('missing')!=[] or r.get('no_further_seed_extension') is not True):
            raise ValueError('not exact complete HS1 five-seed output')
        h=r.get('HS1_Sgist_minus_B',{})
        if h.get('verdict')!='FLOOR' and (h.get('n')!=5 or len(h.get('per_seed',[]))!=5):
            raise ValueError('HS1 must contain all five paired seed differences')

    def tick(self):
        s=self.state()
        if s['status']=='HALTED':return 'HALTED'
        try:
            self.verify_sources(s)
            published=self.output/'reports'
            if published.exists():
                c=common.load_json(published/'completion.json');self.verify_outputs(published)
                if c['source_hashes']!=s['source_hashes'] or c['input_manifest']!=self.inputs():
                    raise ValueError('published input/source inventory changed')
                for n,h in c['report_hashes'].items():
                    if common.sha(published/n)!=h:raise ValueError('published report changed: '+n)
                s.update(status='COMPLETE',completed=c['completed']);common.atomic_json(self.state_path,s)
                self.log('all jobs finished and home: HS1 five-seed analysis published');return 'COMPLETE'
            if time.time()-s['created']>self.max_hours*3600:raise ValueError('HS1 deadline exceeded')
            before=self.inputs()
            for p,h in s.get('observed_inputs',{}).items():
                if before['sha256'].get(p)!=h:raise ValueError('observed input changed: '+p)
            s['observed_inputs']=before['sha256']
            if not before['ready']:
                s.update(status='WAITING',pending_count=len(before['pending']),checked=common.now())
                common.atomic_json(self.state_path,s);return 'WAITING'
            if len(before['selected_cell_seed_pairs'])!=10:raise ValueError('expected exactly10 HS1 reports')
            s.update(status='ANALYSING');common.atomic_json(self.state_path,s)
            stage=self.output/'.incoming';stage.mkdir(exist_ok=True)
            common.atomic_json(stage/'input_receipt.json',before)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--calculate','--art',str(self.art),'--out',str(stage/REPORT)]
            common.atomic_json(stage/'commands.json',[cmd])
            wrapper=[sys.executable,str(common.CPU_BUDGET),'run','--session','paperkiln','--job','registered HS1 five-seed contrast',
                     '--cores','1','--resume','existing HS1 finalizer watchdog entry','--',*cmd]
            with (stage/'analysis.log').open('w') as log:subprocess.run(wrapper,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            self.verify_sources(s);self.verify_outputs(stage)
            if self.inputs()!=before:raise ValueError('inputs changed during analysis')
            c={'completed':common.now(),'source_hashes':s['source_hashes'],'input_manifest':before,
               'report_hashes':{n:common.sha(stage/n) for n in (REPORT,'input_receipt.json','commands.json')}}
            common.atomic_json(stage/'completion.json',c);stage.rename(published)
            s.update(status='COMPLETE',completed=c['completed']);common.atomic_json(self.state_path,s)
            self.log('all jobs finished and home: HS1 five-seed analysis published');return 'COMPLETE'
        except (Exception,SystemExit) as e:
            s.update(status='HALTED',error=str(e),checked=common.now());common.atomic_json(self.state_path,s)
            (self.home/'HALTED').write_text(str(e)+'\n');self.log('HALTED: '+str(e));return 'HALTED'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--art',required=True);p.add_argument('--out',required=True)
    p.add_argument('--state-dir');p.add_argument('--calculate',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=240);p.add_argument('--max-hours',type=float,default=24)
    a=p.parse_args()
    if a.calculate:calculate(a.art,a.out);return 0
    if not a.state_dir:p.error('--state-dir required for driver')
    f=Finalizer(a.art,a.state_dir,a.out,a.max_hours)
    with common.DriverLock(f.home):
        while True:
            result=f.tick()
            if result=='COMPLETE':return 0
            if a.once:return 1 if result=='HALTED' else 2
            time.sleep(max(10,a.poll_seconds))

if __name__=='__main__':raise SystemExit(main())
