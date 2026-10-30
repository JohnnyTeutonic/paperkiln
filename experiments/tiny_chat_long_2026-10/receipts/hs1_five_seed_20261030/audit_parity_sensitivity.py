import json,pathlib,statistics,math
art=pathlib.Path('/mnt/c/ml_artifacts/tiny_chat');tmp=pathlib.Path('/mnt/c/Users/jonat/AppData/Local/Temp');audit=json.loads((tmp/'hs1_history_argmax_audit.json').read_text());bad={}
for r in audit['nonidentical']:bad.setdefault((r['cell'],r['seed']),set()).add(r['id'])
primary={'recall','correction','topic','load','knowledge'};ranges={};denominators={}
for cell in ['B','Sgist']:
 for seed in range(1,6):
  root=art/f'c12_{cell}_r{seed}'/f'post_c12_{cell}_r{seed}';rows=[json.loads(x) for x in (root/'carry12/m/items.jsonl').read_text().splitlines()];xs=[r for r in rows if r['family'] in primary];n=len(xs);assert n==1380
  affected=bad.get((cell,seed),set());assert affected<=set(r['id'] for r in xs)
  unaffected=sum(r['fc']['self']['acc'] for r in xs if r['id'] not in affected);twin=round(sum(r['fc']['twin']['acc'] for r in xs)/n,4)
  lo=round(unaffected/n,4)-twin;hi=round((unaffected+len(affected))/n,4)-twin
  original=json.loads((root/'summary.json').read_text())['Y12'];assert lo-1e-12<=original<=hi+1e-12
  ranges[cell,seed]=(lo,hi);denominators[cell+str(seed)]={'primary_items':n,'affected_distinct_items':len(affected),'min_Y12':lo,'max_Y12':hi,'original_Y12':original}
report=json.loads((art/'carry12_hs1_final_v2/reports/analysis_hs1.json').read_text());d=report['HS1_Sgist_minus_B']['per_seed'];lo=[ranges['Sgist',s][0]-ranges['B',s][1] for s in range(1,6)];hi=[ranges['Sgist',s][1]-ranges['B',s][0] for s in range(1,6)]
b=[max(abs(l-x),abs(h-x)) for l,h,x in zip(lo,hi,d)];sd_upper=statistics.stdev(d)+math.sqrt(sum(x*x for x in b))/math.sqrt(4);se_upper=sd_upper/math.sqrt(5);mean_low,mean_high=statistics.mean(lo),statistics.mean(hi);t_abs_lower=-mean_high/se_upper
assert mean_high<-.03 and t_abs_lower>2.776
result={'status':'POSTHOC_SENSITIVITY_ONLY_NOT_PARITY_REPAIR','assumption':'Only the six identified per-item self-history credits may vary arbitrarily in[0,1]; all other credits, twins, item weighting and registered four-decimal pool rounding stay fixed. This does not bound unrecorded parity defects or validate generation.','denominator':1380,'affected_items':6,'cell_bounds':denominators,'paired_per_seed_bounds':[[l,h] for l,h in zip(lo,hi)],'original_mean':statistics.mean(d),'mean_interval':[mean_low,mean_high],'largest_abs_change_in_mean':max(abs(mean_low-statistics.mean(d)),abs(mean_high-statistics.mean(d))),'sd_upper_bound':sd_upper,'abs_t_lower_bound':t_abs_lower,'registered_crit':2.776,'ci90_envelope':[mean_low-2.132*se_upper,mean_high+2.132*se_upper],'all_bounded_variants_meet_numerical_supported_rule':True,'proof':'For componentwise delta bounds b, sampleSD(d+delta)<=sampleSD(d)+norm(b)/sqrt(n-1), since centering is an orthogonal projection. Mean extrema use exact rounded per-cell endpoint pools; all five paired differences remain negative.','historical_parity_violation_remains':True,'canonical_data_and_analysis_modified':False}
(tmp/'hs1_parity_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['mean_interval','largest_abs_change_in_mean','abs_t_lower_bound','ci90_envelope','all_bounded_variants_meet_numerical_supported_rule']},indent=2))
