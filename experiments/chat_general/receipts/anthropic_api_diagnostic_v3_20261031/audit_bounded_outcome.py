from pathlib import Path
import json,sys,hashlib,collections
W=Path('/home/jonat/pk_api_transport_diagnostic_20261031');sys.path.insert(0,str(W/'tools/posttrain'))
import chat_judge_campaign_diagnostic_v3 as c
pfile=W/'experiments/chat_general/receipts/anthropic_api_diagnostic_v3_20261031/diagnostic_v3_protocol.json';p=json.loads(pfile.read_text());_,status=c.inspect(p)
rows=[json.loads(x) for x in Path(p['ledger']).read_bytes()[p['ledger_prefix_bytes']:].splitlines()];res={r['id']:r for r in rows if r['event']=='reserve'};settles=[r for r in rows if r['event']=='settle'];assert len(res)==len(settles)==24
out=[]
for r in settles:
 q=r['receipt'];t=json.loads(Path(q['transport_trace']).read_text());assert c.api.file_hash(q['transport_trace'])==q['transport_trace_sha256'];assert t['binding']=={'reservation_id':r['id'],'request_sha256':q['request_sha256']};assert t['http_write_started'] is True;assert q['http_status']==200
 for name in ('request','response'):assert c.api.file_hash(q[name])==q[name+'_sha256']
 assert any(x.get('tcp_connected') is True for x in t['attempts'])
 out.append({'id':r['id'],'request_sha256':q['request_sha256'],'response_sha256':q['response_sha256'],'trace_sha256':q['transport_trace_sha256'],'http_status':q['http_status'],'cost_usd':r['cost_usd'],'attempts':t['attempts'],'http_write_started':t['http_write_started']})
d=Path(p['state_dir']);stop=json.loads((d/'STOPPED').read_text());assert not(d/'HALTED').exists();assert stop['kind']=='BOUNDED_DIAGNOSTIC_FINISHED_REVIEW_REQUIRED';assert stop['full_judging_complete'] is False;assert stop['protocol_sha256']==c.api.digest(p);assert status['attempts_since_start']==24 and status['missing_rankings']==5475
receipt={'status':'PASS_BOUNDED_24_ONLY','protocol_file_sha256':c.api.file_hash(pfile),'source_sha256':c.source_hashes(),'stop_sha256':c.api.file_hash(d/'STOPPED'),'stop':stop,'status_replayed':status,'pending_reservations':0,'new_successful_rankings':24,'valid_rankings_total':len(c.api.rows(p['cache'])),'training_attempts_total':520,'new_api_spend_usd':sum(r['cost_usd'] for r in settles),'trace_count':24,'traces':out,'interpretation':'24 successful calls with durable endpoint traces. No historical network cause established; no full continuation authorised by this receipt.'}
assert receipt['valid_rankings_total']==565
folder=pfile.parent;(folder/'bounded_outcome.json').write_text(json.dumps(receipt,indent=2)+'\n');(folder/'bounded_STOPPED.json').write_bytes((d/'STOPPED').read_bytes());print(json.dumps({k:v for k,v in receipt.items() if k not in ('traces','source_sha256','stop')}))
