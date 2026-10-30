from pathlib import Path
import json,hashlib
art=Path('/mnt/c/ml_artifacts/tiny_chat');out=[];counts={}
for c in ['B','Sgist']:
 for seed in range(1,6):
  base=art/f'c12_{c}_r{seed}'/f'post_c12_{c}_r{seed}/carry12/m';items=[json.loads(x) for x in (base/'items.jsonl').read_text().splitlines()];gens={x['id']:x for x in map(json.loads,(base/'gen.jsonl').read_text().splitlines())};passed=sum(sum(v is True for v in x['gate']) for x in items);total=sum(len(x['gate']) for x in items);counts[c+str(seed)]={'pass':passed,'total':total}
  for r in items:
   for k,ok in enumerate(r['gate']):
    if ok:continue
    g=gens[r['id']];assert not g.get('near_self') and len(r['gate'])==len(g['self']);reply=g['self'][k]
    out.append({'cell':c,'seed':seed,'id':r['id'],'family':r['family'],'gate_turn':k+1,'target_turn':r['target_turn'],'before_target':k+1<r['target_turn'],'at_target':k+1==r['target_turn'],'stop':reply.get('stop'),'reply':reply.get('reply'),'ids':reply.get('ids'),'stop_id':reply.get('stop_id'),'dropped':reply.get('dropped'),'primary_family':r['family'] in ['recall','correction','topic','load','knowledge'],'item_file_sha256':hashlib.sha256((base/'items.jsonl').read_bytes()).hexdigest(),'gen_file_sha256':hashlib.sha256((base/'gen.jsonl').read_bytes()).hexdigest()})
r={'scope':'Read-only audit of retained complete HS1 history argmax flags; mismatch causes unavailable without original score chunks/logit rows. No threshold, history or registered metric altered.','counts':counts,'nonidentical':out};p=Path('/mnt/c/Users/jonat/AppData/Local/Temp/hs1_history_argmax_audit.json');p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
