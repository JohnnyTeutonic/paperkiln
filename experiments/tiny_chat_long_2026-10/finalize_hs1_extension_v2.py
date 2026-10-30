"""HS1 finalizer v2: one audited cached-ledger omission, with v1's halt preserved."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import finalize_hs1_extension as v1

common,preparation,HERE,SEEDS,CELLS=v1.common,v1.preparation,v1.HERE,v1.SEEDS,v1.CELLS
RECEIPTS=HERE/'receipts/hs1_environment_2026-10-30'
PROOF=RECEIPTS/'cached_data_audit_20261030.json'
EXPECTED_LEDGER='cf509e0bbaed13b57c00b42408527275da34d086c0001bf5a69c761de47c9ad3'
EXPECTED_REFERENCE='acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4'
EXPECTED_PROOF='d6a29a19a64432e1c6ac4607c8a5af9db7d369e9d7eb209251bcd1f4d043efd6'
LIVE_BUILDER=v1.REPO/'tools/colab_chat/tiny_chat_data.py'
HISTORICAL_BUILDER=RECEIPTS/'historical_tiny_chat_data.py'
EXPECTED_BUILDER='48df84d69ef7e309b3eee1701653584625a426bca0ea6a0b214cba82d2eef465'
# Only the operational builder is replaced by its exact frozen-job blob. It is not
# executed by the analysis; preserving this artifact lets future builder fixes proceed.
SOURCES=(*(p for p in v1.SOURCES if p.resolve()!=LIVE_BUILDER.resolve()),Path(__file__),PROOF,HISTORICAL_BUILDER,*sorted(RECEIPTS.glob('*')))


def verify_proof(proof,reference,value):
    """No field is ignored generally. This proof covers only the exact known omissions."""
    if (proof.get('all_checks_pass') is not True or proof.get('no_data_or_ledger_mutation') is not True
        or proof.get('pin')!='7baf6d95f79b9cc543705e9b510bcbfeb5c18a63'
        or proof.get('reference_ledger_sha256')!=EXPECTED_REFERENCE
        or proof.get('resumed_ledger_sha256')!=EXPECTED_LEDGER
        or proof.get('exact_omissions')!=['human','steps.decontam','steps.human_fetch']):
        raise ValueError('cached-ledger proof identity mismatch')
    if proof.get('source_dialogues_sha256')!=reference['human']['files']:
        raise ValueError('cached source dialogues differ')
    if proof.get('decontam_report')!=reference['steps']['decontam']:
        raise ValueError('cached decontamination differs')
    if set(proof.get('outputs',{}))!=set(reference['outputs']):raise ValueError('cached output inventory differs')
    for name,wanted in reference['outputs'].items():
        actual=proof['outputs'][name]
        if any(actual.get(k)!=wanted[k] for k in ('bytes','sha256')):raise ValueError('cached output differs: '+name)
    if proof.get('eval_sha256')!={'carry5_v1.jsonl':reference['human']['carry5_v1_sha256'],
                               'carry12_v1.jsonl':reference['human']['carry12_v1_sha256']}:
        raise ValueError('cached evaluation source differs')
    fetched=proof.get('fetch_ledger',{}).get('sources',{})
    if set(fetched)!=set(reference['human']['sources']):raise ValueError('fetch source inventory differs')
    if any(fetched[n].get('id')!='human_'+n+'_v1' for n in fetched):raise ValueError('fetch source identity differs')
    for name in ('decontam_report_sha256','fetch_ledger_sha256'):
        if not isinstance(proof.get(name),str) or len(proof[name])!=64:raise ValueError('missing raw provenance hash')
    if 'human' in value or any(k in value.get('steps',{}) for k in ('decontam','human_fetch')):
        raise ValueError('not the exact cached omissions')
    restored=copy.deepcopy(value);restored['human']=reference['human']
    for key in ('decontam','human_fetch'):restored['steps'][key]=reference['steps'][key]
    if common.data_ledger(restored)!=common.data_ledger(reference):raise ValueError('additional ledger differences')


def cached_exception(path,reference_path,value,reference):
    if (common.sha(path)!=EXPECTED_LEDGER or common.sha(reference_path)!=EXPECTED_REFERENCE
        or not PROOF.is_file() or common.sha(PROOF)!=EXPECTED_PROOF
        or not HISTORICAL_BUILDER.is_file() or common.sha(HISTORICAL_BUILDER)!=EXPECTED_BUILDER):
        raise ValueError('exact cached ledger/reference/proof hash required')
    verify_proof(common.load_json(PROOF),reference,value)


class Finalizer(v1.Finalizer):
    def state(self):
        if self.state_path.exists():return super().state()
        s={'schema':2,'created':time.time(),'art':str(self.art),'output':str(self.output),'status':'WAITING',
           'python':sys.version,'source_hashes':{str(p.resolve()):common.sha(p) if p.is_file() else 'ABSENT' for p in SOURCES},
           'operational_exception':'Sgist5 cached ledger only, exact hash plus raw-input/output proof'}
        common.atomic_json(self.state_path,s);return s

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
                    if seed>=4:
                        c12=row.get('carry12',{});meta=c12.get('meta',{});engine=c12.get('gen_engine',{})
                        if (meta.get('items')!=1452 or meta.get('max_new')!=48
                            or meta.get('self_scope')!='target' or meta.get('near_self') is not False
                            or meta.get('scorer')!='mtstudio' or meta.get('spot')!=20
                            or meta.get('sha256')!='de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797'
                            or 'error' in c12):
                            raise ValueError('incomplete/different registered Carry-12 evaluation: '+str(p))
                        if (engine.get('engine') not in ('ember','mtstudio')
                            or (engine['engine']=='ember' and (engine.get('spot') or {}).get('pass') is not True)
                            or (engine['engine']=='mtstudio' and not engine.get('flag'))):
                            raise ValueError('missing checked generation engine/fallback receipt: '+str(p))
                    if row.get('carry12',{}).get('context')!=1024:
                        raise ValueError('wrong context: '+str(p))
                    files[str(p)]=common.sha(p);pairs.append([cell,seed])
                ledger=base/'data/data.ledger.json';gate=base/'gate/gate.json'
                for path in (ledger,gate):
                    if not path.is_file():pending.append(str(path));continue
                    value=common.load_json(path)
                    if path==ledger and common.data_ledger(value)!=reference:
                        if cell!='Sgist' or seed!=5:raise ValueError('data ledger differs: '+str(path))
                        cached_exception(path,ref,value,common.load_json(ref))
                        files[str(PROOF)]=common.sha(PROOF)
                    if path==gate and (value.get('ok') is not True or value.get('problems')!=[]):
                        raise ValueError('execution gate failed: '+str(path))
                    files[str(path)]=common.sha(path)
        return {'ready':not pending,'pending':pending,'selected_cell_seed_pairs':pairs,'sha256':files,'operational_exception':{'ledger_sha256':EXPECTED_LEDGER,'proof_sha256':EXPECTED_PROOF,'v1_halt_preserved':True}}




def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--art',required=True);p.add_argument('--out',required=True)
    p.add_argument('--state-dir',required=True);p.add_argument('--once',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=240);p.add_argument('--max-hours',type=float,default=24)
    a=p.parse_args();f=Finalizer(a.art,a.state_dir,a.out,a.max_hours)
    with common.DriverLock(f.home):
        while True:
            result=f.tick()
            if result=='COMPLETE':return 0
            if a.once:return 1 if result=='HALTED' else 2
            time.sleep(max(10,a.poll_seconds))

if __name__=='__main__':raise SystemExit(main())
