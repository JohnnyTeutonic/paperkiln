import pathlib,json,hashlib,math,statistics,sys
repo=pathlib.Path('/home/jonat/pk_publication_20261030');corpus=repo/'corpus/eval/carry12_v1.jsonl';assert hashlib.sha256(corpus.read_bytes()).hexdigest()=='de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797'
items={x['id']:x for x in map(json.loads,corpus.read_text().splitlines())};primary={'recall','correction','topic','load','knowledge'};context=primary-{'knowledge'}
result=[]
for filename in sys.argv[1:]:
 p=pathlib.Path(filename);rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==1452 and len({r['id'] for r in rows})==1452 and {r['id'] for r in rows}==set(items)
 checked=0
 for r in rows:
  it=items[r['id']];assert r['family']==it['family']
  for mode,v in r['fc'].items():
   if v.get('over_ctx'):expected=0.0
   else:
    raw=v['raw'];assert len(raw)==4 and all(math.isfinite(x) for x in raw)
    top=max(raw);tied=[i for i,s in enumerate(raw) if s==top];expected=1/len(tied) if it['gold_index'] in tied else 0.0
   assert v['acc']==expected,(p,r['id'],mode);checked+=1
 def pool(mode,fams):return statistics.mean(r['fc'][mode]['acc'] for r in rows if r['family'] in fams)
 d=[r['fc']['gold']['acc']-r['fc']['near']['acc'] for r in rows if r['family'] in context];m=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(len(d))
 result.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'items':len(rows),'raw_fc_credits_recomputed':checked,'self_primary':pool('self',primary),'twin_primary':pool('twin',primary),'Y12':pool('self',primary)-pool('twin',primary),'context_twin_minus_noctx':pool('twin',context)-pool('noctx',context),'hl0':{'n':len(d),'mean':round(m,5),'se':round(se,5),'ci95':[round(m-1.959964*se,5),round(m+1.959964*se,5)],'ci90':[round(m-1.644854*se,5),round(m+1.644854*se,5)]}})
print(json.dumps({'scope':'Independent replay of raw summed log-prob credits and item-pool arithmetic; no model inference replay','corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),'runs':result},indent=2))
