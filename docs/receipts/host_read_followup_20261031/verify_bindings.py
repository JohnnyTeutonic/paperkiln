import hashlib,json,subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[3]
temp=Path(__file__).resolve().parent
def verify(commit,path,h):
 raw=subprocess.check_output(['git','show',str(commit)+':'+path],cwd=repo)
 if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('binding mismatch '+str(commit)+':'+path)
a=json.loads((temp/'paperkiln_host_read_audit_20261031.json').read_text());count=0
for path,h in a['source_sha256'].items():verify(a['source_commit'],path,h);count+=1
h=json.loads((temp/'grpo_historical_applicability_20261031.json').read_text())
for path,digest in h['source_hashes'].items():verify(h['commit'],path,digest);count+=1
for row in h['tracked_spec_inventory']:verify(h['commit'],row['path'],row['sha256']);count+=1
for row in h['grpo_run_inventory']:
 verify(h['commit'],row['spec'],row['sha256']);verify(h['commit'],row['events'],row['events_sha256']);count+=2
for commit,files in h['historical_licence_source_bindings'].items():
 for path,info in files.items():verify(commit,path,info['sha256']);count+=1
print(json.dumps({'pass':True,'source_and_receipt_bindings':count,'historical_runs':h['grpo_spec_count'],'runtime_exposure':'UNRESOLVED'}))
