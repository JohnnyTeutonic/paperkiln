"""Read-only projection of reviewed retention; never changes campaign state or dispatches."""
import copy,json,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]/'tools/posttrain'))
import chat_judge_campaign_continuation_v8 as c
api=c.api

def forbidden(*a,**k):raise RuntimeError('FORBIDDEN_DISPATCH_CREDENTIAL_OR_MUTATION')
for name in ['run','load_key','http_post','append','atomic_blob']:setattr(api,name,forbidden)
p=json.loads((HERE/'full_campaign_v8_protocol.json').read_text());g=p['fresh_dispatch_gate']
watched=[p['ledger'],p['cache'],p['cache']+'.failures.jsonl',g['v7_halt']['path'],g['v7_stopped']['path'],g['request']['path'],g['error']['path'],g['transport']['path']]
before={x:api.file_hash(x) for x in watched};original_rows=api.rows
history=original_rows(p['ledger']);ret=json.loads((HERE/'retention_authorization.template.json').read_text());reservation=[r for r in history if r.get('event')=='reserve' and r.get('id')==c.PENDING_ID][0]
event=dict(event='retain_unknown_ceiling',id=c.PENDING_ID,ceiling_usd=.25,reservation_sha256=api.digest(reservation),raw_request_sha256=g['request']['sha256'],error_sha256=g['error']['sha256'],error_type='URLError',authorization_sha256=api.digest(ret),authorization=ret,charge='unknown_full_ceiling_retained_not_actual_spend',reason=ret['reason'])
api.verify_retained(event,reservation,p['ledger'])
projected=history+[event]
api.rows=lambda path:copy.deepcopy(projected) if str(path)==p['ledger'] else original_rows(path)
with tempfile.TemporaryDirectory(prefix='v8_projection_') as scratch:
 for key,name in [('retention_authorization','retention_authorization.template.json'),('fresh_authorization','fresh_dispatch_authorization.template.json')]:
  q=Path(scratch)/name;q.write_bytes((HERE/name).read_bytes());g[key]['path']=str(q)
 assert c.verify_fresh_gate(p) is True
 requests,status=c.inspect(p)
 assert status['attempts_since_start']==0 and status['valid_api_rankings']==2777 and status['eligible_missing_rankings']==3259
 assert status['retained_unknown_ceiling_usd']==1.0 and status['api_actual_spend_usd']==41.874124
 assert len([r for r in history if r.get('event')=='reserve' and r.get('key')==c.FRESH_KEY])==1
 assert before=={x:api.file_hash(x) for x in watched}
 result={'kind':'READONLY_V8_PREPARED_PROJECTION','status':'PASS','source_sha256':c.source_hashes(),'protocol_sha256':api.file_hash(HERE/'full_campaign_v8_protocol.json'),'evidence_sha256':before,'projected_retention_only_not_appended':True,'actual_state_unchanged':True,'explicit_first_dispatch_remaining':True,'projected_status':status,'provider_calls':0,'credential_accesses':0,'disposition_performed':False,'activation':False}
 print(json.dumps(result,indent=2))
