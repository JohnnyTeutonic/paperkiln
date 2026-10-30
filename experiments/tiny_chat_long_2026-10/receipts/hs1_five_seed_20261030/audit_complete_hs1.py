import json,hashlib,math,statistics,pathlib,sys
art=pathlib.Path('/mnt/c/ml_artifacts/tiny_chat')
folder=art/'carry12_hs1_final_v2/reports'
if not folder.exists():
 print('PENDING: complete HS1 report not yet published');sys.exit(2)
load=lambda p:json.loads(pathlib.Path(p).read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c=load(folder/'completion.json');r=load(folder/'analysis_hs1.json')
checks={}
for kind,records in [('sources',c['source_hashes']),('inputs',c['input_manifest']['sha256']),('reports',{str(folder/k):v for k,v in c['report_hashes'].items()})]:
 bad=[p for p,h in records.items() if not pathlib.Path(p).is_file() or sha(p)!=h]
 if bad:raise ValueError((kind,bad))
 checks[kind]=len(records)
assert r['seeds']==[1,2,3,4,5] and r['cells']==['B','Sgist'] and r['partial'] is False and r['missing']==[]
assert c['input_manifest']['ready'] is True and c['input_manifest']['pending']==[]
assert sorted(c['input_manifest']['selected_cell_seed_pairs'])==sorted([[c,s] for c in ['B','Sgist'] for s in range(1,6)])
runs={}
for cell in ['B','Sgist']:
 for seed in range(1,6):
  p=art/f'c12_{cell}_r{seed}'/f'post_c12_{cell}_r{seed}/summary.json'
  x=load(p);assert x['cell']==cell and type(x['seed']) is int and x['seed']==seed
  runs[cell,seed]=x
# Independent raw-pool arithmetic, without importing registered statistical helpers.
def pool(x,which,mode):return x['carry12']['pools']['acc'][which][mode]['mean']
means={}
for cell in ['B','Sgist']:
 xs=[runs[cell,s] for s in range(1,6)]
 self_=statistics.mean(pool(x,'primary','self') for x in xs)
 twin=statistics.mean(pool(x,'primary','twin') for x in xs)
 gap=statistics.mean(pool(x,'context','twin')-pool(x,'context','noctx') for x in xs)
 gate=twin>=.35 and gap>=.10
 assert gate==r['validity_gate'][cell]
 for x in xs:assert math.isclose(x['Y12'],pool(x,'primary','self')-pool(x,'primary','twin'),abs_tol=1e-12)
 means[cell]={'self':self_,'twin':twin,'Y12':self_-twin,'twin_minus_noctx_context':gap,'validity_gate':gate}
ds=[runs['Sgist',s]['Y12']-runs['B',s]['Y12'] for s in range(1,6)]
m=statistics.mean(ds);sd=statistics.stdev(ds);se=sd/math.sqrt(5);t=m/se
# Frozen df4 rounded t cutoffs; scope explicitly matches registered table precision.
crit=2.776;c90=2.132;lo=m-c90*se;hi=m+c90*se
v='SUPPORTED' if abs(t)>crit and m<0 and abs(m)>=.03 else 'CONTRADICTED' if abs(t)>crit and m>0 else 'EQUIVALENT' if -.03<lo and hi<.03 else 'SIGNIFICANT_BELOW_SESOI' if abs(t)>crit else 'NOT_RESOLVED'
h=r['HS1_Sgist_minus_B']
for k,x in [('mean',m),('sd',sd),('t',t),('crit',crit)]:assert math.isclose(h[k],x,rel_tol=1e-12,abs_tol=1e-12),(k,h[k],x)
assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(h['per_seed'],ds))
assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(h['ci90'],[lo,hi]))
assert h['verdict']==v and h['n']==5 and h['extend_to_5_seeds'] is False
assert r['s_full_condition']['run_s_full']==(v=='SUPPORTED')
result={'audit':'independent input/source hashes and raw-pool five-pair arithmetic','completion':c['completed'],'inventories_verified':checks,'cell_means':means,'paired':{'per_seed':ds,'mean':m,'sd':sd,'t':t,'crit':crit,'ci90':[lo,hi],'verdict':v},'report_sha256':sha(folder/'analysis_hs1.json')}
print(json.dumps(result,indent=2))
