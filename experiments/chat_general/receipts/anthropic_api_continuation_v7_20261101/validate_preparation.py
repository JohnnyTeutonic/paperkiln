"""Read-only actual-state inspection; no provider or credential accessor calls."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/posttrain'))
import chat_judge_campaign_continuation_v7 as c
out=Path(__file__).parent
p=json.loads((out/'full_campaign_v7_protocol.json').read_text())
# Hard-fail if a read-only preflight ever reaches either dispatch or credential loading.
def forbidden(*args, **kwargs): raise AssertionError("provider/credential access forbidden during preparation")
c.api.http_post=forbidden
c.api.load_key=forbidden
paths=[p['cache'],p['ledger'],p['cache']+'.failures.jsonl']+[p['recovery_anchors'][k]['path'] for k in ['v6_halt','v6_stopped']]
before={path:c.api.file_hash(path) for path in paths}
requests,status=c.inspect(p)
assert before=={path:c.api.file_hash(path) for path in paths}
assert status['valid_api_rankings']==2568 and status['missing_rankings']==3472 and status['eligible_missing_rankings']==3468
assert status['preserved_failed_rankings']==2 and status['unattempted_paired_exclusions']==2
report=dict(status=status,provider_calls=0,credential_accessor_calls=0,prior_state_unchanged=True,protocol_file_sha256=c.api.file_hash(out/'full_campaign_v7_protocol.json'),dataset_decision_sha256=p['dataset_decision_sha256'],source_sha256=c.source_hashes())
(out/'actual_readonly_inspect.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
