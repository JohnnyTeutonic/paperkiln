#!/usr/bin/env python3
"""Independent calibration replay. No production judge imports, network or credentials."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    p=Path(path)
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.exists() else []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def render(prompt,candidates,order):
    assert prompt.startswith('<|user|>') and prompt.endswith('<|assistant|>')
    chunks=prompt.split('<|user|>')[1:]
    lines=[]
    if len(chunks)>1:
        lines.append('Conversation so far:')
        for chunk in chunks[:-1]:
            user,assistant=chunk.split('<|assistant|>',1)
            assert assistant.endswith('<|endoftext|>')
            lines.extend(['User: '+user,'Assistant: '+assistant[:-len('<|endoftext|>')]])
        lines.append('')
    lines.extend(['User message: '+chunks[-1][:-len('<|assistant|>')],'','Candidate replies:'])
    lines.extend(chr(65+i)+': '+candidates[index] for i,index in enumerate(order))
    return '\n'.join(lines)


def provider_cost(response):
    u=response['usage']
    assert response['model']=='claude-opus-5-5'
    assert u.get('inference_geo','global')=='global'
    assert u.get('speed','standard')=='standard' and u.get('service_tier','standard')=='standard'
    names=('input_tokens','output_tokens','cache_read_input_tokens','cache_creation_input_tokens')
    counts={k:u.get(k,0) for k in names}
    assert all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in counts.values())
    assert counts['output_tokens']<=1024
    write=u.get('cache_creation',{})
    one=write.get('ephemeral_1h_input_tokens',0)
    five=write.get('ephemeral_5m_input_tokens',counts['cache_creation_input_tokens']-one)
    assert one+five==counts['cache_creation_input_tokens']
    return (Decimal(counts['input_tokens'])*4+Decimal(counts['output_tokens'])*20+
            Decimal(counts['cache_read_input_tokens'])*Decimal('.2')+Decimal(five)*5+Decimal(one)*8)/1_000_000


def ranking(response,order):
    text=''.join(b['text'] for b in response['content'] if b['type']=='text')
    try:
        d=json.loads(text[text.index('{'):text.rindex('}')+1])
        labels=d['ranking']
        assert sorted(labels)==[chr(65+i) for i in range(len(order))]
        assert isinstance(d['rationale'],str) and d['rationale'].strip()
        if response['stop_reason']!='end_turn':return None
        return [order[ord(label)-65] for label in labels]
    except (ValueError,KeyError,AssertionError,TypeError):
        return None


def replay(art,candidates,rubric,protocol_dir):
    hist=rows(art/'budget.jsonl')
    contract=hist[0]['contract']
    binding=load(art/'api_cache.jsonl.binding.json')
    allreq=rows(candidates)
    assert binding==contract['binding']
    assert stable(allreq)==binding['requests_sha256']
    assert sha(rubric)==binding['rubric_sha256']
    profile=load(protocol_dir/'prospective_readiness.json')['plan']['profile']
    assert profile==contract['profile'] and stable(profile)==binding['profile_sha256']
    reserves={};settled={};unknown={}
    for event in hist[1:]:
        ident=event['id']
        if event['event']=='reserve':
            assert ident not in reserves
            reserves[ident]=event
        elif event['event']=='settle':
            assert ident in reserves and ident not in settled and ident not in unknown
            settled[ident]=event
        elif event['event']=='retain_unknown_ceiling':
            assert ident in reserves and ident not in settled and ident not in unknown
            assert event['ceiling_usd']==reserves[ident]['reserve_usd']
            assert event['reservation_sha256']==stable(reserves[ident])
            assert event['charge']=='unknown_full_ceiling_retained_not_actual_spend'
            assert stable(event['authorization'])==event['authorization_sha256']
            assert event['authorization']['kind']=='ROOT_AUTHORISED_RETAIN_UNKNOWN_CEILING'
            unknown[ident]=event
        else:raise AssertionError('Unexpected journal event')
    cache=rows(art/'api_cache.jsonl');failed=rows(art/'api_cache.jsonl.failures.jsonl')
    byid={r['budget_reservation_id']:r for r in cache+failed}
    assert len(byid)==len(cache)+len(failed) and set(byid)==set(settled)
    candidate_by_uri={r['uri']:r for r in allreq}
    costs=[];inputs=[];outputs=[];thinking=[];stops=Counter();pair_rows={}
    raw_sha={}
    for ident,event in settled.items():
        rec=byid[ident];reservation=reserves[ident]
        request=art/'budget.raw'/(ident+'.request.json')
        response=art/'budget.raw'/(ident+'.response.json')
        rr=load(request);rp=load(response)
        receipt=event['receipt']
        assert sha(request)==receipt['request_sha256'] and sha(response)==receipt['response_sha256']
        raw_sha[request.name]=sha(request);raw_sha[response.name]=sha(response)
        assert rec['transport_receipt']==receipt and rr['profile']==profile
        assert rr['cache_key']==rec['key']==reservation['key']
        assert rr['semantic_request_key']==rec['semantic_request_key']
        r=candidate_by_uri[rec['uri']];order=rec['order']
        body=rr['body']
        expected={'model':'claude-opus-5-5','max_tokens':1024,'thinking':{'type':'adaptive'},
          'output_config':{'effort':'medium'},'service_tier':'standard_only','inference_geo':'global',
          'system':rubric.read_text(encoding='utf-8'),'messages':[{'role':'user','content':render(r['prompt'],r['candidates'],order)}],'stream':False}
        assert body==expected
        for model,key in ((profile['model'],'semantic_request_key'),(rec['judge_id'],'key')):
            blob=json.dumps([r['prompt'],r['candidates'],order,model,rec['rubric_version']],ensure_ascii=False).encode()
            assert hashlib.sha256(blob).hexdigest()==rec[key]
        cost=provider_cost(rp)
        assert float(cost)==rec['cost_usd']==event['cost_usd'] and cost<=Decimal(str(reservation['reserve_usd']))
        costs.append(cost);inputs.append(rp['usage']['input_tokens']);outputs.append(rp['usage']['output_tokens'])
        thinking.append(rp['usage'].get('output_tokens_details',{}).get('thinking_tokens',0))
        stops[rp['stop_reason']]+=1
        rk=ranking(rp,order)
        if 'ranking' in rec:
            assert rk==rec['ranking']
            pair_rows.setdefault(rec['uri'],[]).append(rk)
        else:assert rk is None
    for ident,event in unknown.items():
        request=art/'budget.raw'/(ident+'.request.json');error=art/'budget.raw'/(ident+'.error.json')
        assert sha(request)==event['raw_request_sha256'] and sha(error)==event['error_sha256']
        assert load(error)['error_type']==event['error_type']
        assert not (art/'budget.raw'/(ident+'.response.json')).exists()
        raw_sha[request.name]=sha(request);raw_sha[error.name]=sha(error)
    pending=set(reserves)-set(settled)-set(unknown)
    pair_counts=Counter();fullmatch=0
    for uri,ranks in pair_rows.items():
        if len(ranks)!=2:pair_counts['unranked_or_missing_direction']+=1;continue
        a,b=ranks
        fullmatch+=a==b
        if (a[0],a[-1])!=(b[0],b[-1]):pair_counts['position_disagreement']+=1;continue
        pair_counts['best_and_worst_agree']+=1
        texts=candidate_by_uri[uri]['candidates']
        sizes=[max(1,len(texts[i].split())) for i in (a[0],a[-1])]
        pair_counts['length_matched_pairs' if max(sizes)/min(sizes)<=1.5 else 'length_mismatched']+=1
    known=sum(costs,Decimal(0));ceiling=sum(Decimal(str(r['ceiling_usd'])) for r in unknown.values())
    legacy=contract['legacy']
    baseline=sum(Decimal(str(legacy.get(k,0))) for k in ('known_spend_usd','unknown_failure_reserve_usd','prior_api_spend_usd','prior_api_unknown_reserve_usd'))
    wall=sum(load(art/name)['wall_seconds'] for name in ('calibration_exit.json','calibration_phase2_exit.json') if (art/name).exists())
    n=len(settled);attempts=len(reserves);remaining=2*len(allreq)-len(cache)
    mean_cost=known/n if n else None
    normal_projection=baseline+ceiling+known+remaining*mean_cost+Decimal(str(contract['gate_reserve_usd'])) if n else None
    sample_unknown_projection=baseline+known+ceiling+remaining*(known+ceiling)/max(1,attempts)+Decimal(str(contract['gate_reserve_usd']))
    allbyte=0
    for r in allreq:
        for order in (list(range(len(r['candidates']))),list(range(len(r['candidates'])-1,-1,-1))):
            upper=len(rubric.read_bytes())+len(render(r['prompt'],r['candidates'],order).encode())+4096
            allbyte+=(Decimal(upper)*4+Decimal(1024)*20)/1_000_000
    result={'kind':'INDEPENDENT_CALIBRATION_REPLAY_NOT_API_CLI_EQUIVALENCE',
      'attempts':attempts,'known_settled_calls':n,'valid_rankings':len(cache),'failed_rankings':len(failed),
      'unresolved_pending_calls':len(pending),'retained_unknown_calls':len(unknown),
      'actual_usage_priced_cost_usd':float(known),'retained_unknown_ceiling_usd':float(ceiling),
      'calibration_accounted_actual_plus_ceiling_usd':float(known+ceiling),
      'prior_cli_api_cost_and_reserve_usd':float(baseline),'heldout_gate_reserve_usd':contract['gate_reserve_usd'],
      'input_tokens':{'sum':sum(inputs),'mean':statistics.mean(inputs) if inputs else None,'max':max(inputs,default=0)},
      'output_tokens_including_thinking':{'sum':sum(outputs),'mean':statistics.mean(outputs) if outputs else None,'max':max(outputs,default=0)},
      'thinking_tokens_reported':sum(thinking),'stop_reasons':dict(stops),
      'pair_counts':dict(pair_counts),'full_ranking_both_orders_agree':fullmatch,
      'api_runner_wall_seconds_excluding_manual_recovery_pause':wall,
      'remaining_rankings_in_full_api_profile':remaining,
      'projections_not_guarantees':{'normal_future_calls_at_observed_known_mean_total_including_gate_usd':float(normal_projection) if normal_projection else None,
       'observed_unknown_ceiling_rate_total_including_gate_usd':float(sample_unknown_projection),
       'runner_text_byte_admission_bound_all_one_attempt_calls_plus_baseline_and_gate_usd':float(allbyte+baseline+Decimal(str(contract['gate_reserve_usd']))),
       'remaining_serial_hours_at_observed_attempt_wall_rate':wall/max(1,attempts)*remaining/3600,
       'qualification':'First 24 candidate requests, one operational failure and a single recovery. Not representative proof of full-dataset cost, latency, parse rate or transport equivalence. Hard cap stops if actual/reserved expenditure grows.'},
      'checks':{'raw_requests_and_responses_verified':True,'actual_model_and_settings_verified':True,'usage_and_cost_independently_recomputed':True,'cache_keys_and_rankings_replayed':True},
      'raw_file_hashes':raw_sha,'network_or_model_calls_made_by_replay':0}
    assert attempts<=48 and known+ceiling<=2 and baseline+known+ceiling+Decimal(str(contract['gate_reserve_usd']))<=200
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--art',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--rubric',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();r=replay(a.art,a.candidates,a.rubric,Path(__file__).resolve().parent)
    with a.out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=1);f.write('\n')
    print(json.dumps({k:r[k] for k in ('attempts','valid_rankings','actual_usage_priced_cost_usd','retained_unknown_ceiling_usd','stop_reasons','pair_counts','projections_not_guarantees')},indent=1))