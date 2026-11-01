"""Offline inventory/verdict replay, not a replacement for the C++ numeric reference."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);a=p.parse_args()
expected_names={'NN','NT','TN','TT','saved_parent_backward'}
rows=[]
for variant in ['old','fixed']:
 for transition in ['active','cpu','stepoff']:
  stem=variant+'_'+transition
  lines=(a.outputs/(stem+'.txt')).read_text().splitlines()
  data=[json.loads(line.split('GEMM_FALLBACK_JSON ',1)[1]) for line in lines if line.startswith('GEMM_FALLBACK_JSON ')]
  assert len(data)==5 and {r['case'] for r in data}==expected_names,stem
  wanted=variant=='fixed' or transition=='active'
  assert int((a.outputs/(stem+'.rc')).read_text())==(0 if wanted else 1),stem
  for row in data:
   assert row['actual_cuda'] is True and row['synthetic_authority'] is False,stem
   assert type(row['pass']) is bool and row['pass'] is wanted,stem
   if not wanted:assert row.get('error')=='independent numerical reference differs',stem
  rows.extend(data)
print(json.dumps({'kind':'GEMM_FALLBACK_L4_VERDICT_REPLAY','pass':True,'rows':len(rows),'numerical_checks':'C++ independent double reference, not re-evaluated by this inventory checker'}))
