"""Operational V8 migration preserves the adopted scientific cohort and pair recipe."""
import copy,json
from pathlib import Path
import pytest
import materialize_preference_v8 as m
from test_materialize_preference_v7 import fixture as original_fixture
from pref_artifact_io_v1 import sha


def fixture(tmp_path):
    old,proposal,adopt,vocab,status=original_fixture(tmp_path)
    protocol=tmp_path/'v8.json';p=json.loads(old.read_text());p['fresh_dispatch_gate']={'v7_protocol':{'path':str(old)}};protocol.write_text(json.dumps(p))
    op=dict(kind='ROOT_ADOPTED_V8_PROVEN_UNTRANSMITTED_CONTINUATION',adopted=True,protocol_file_sha256=sha(protocol),source_sha256=m.campaign.source_hashes(),original_dataset_adoption_sha256=sha(adopt),eligible_target_rankings=6036,eligible_candidate_sets=3018,preserve_accepted_mate=True,no_new_candidate_exclusions=True,training_recipe_and_five_part_acceptance_unchanged=True,future_unknown_auto_retry=False)
    operational=tmp_path/'operational.json';operational.write_text(json.dumps(op))
    return protocol,proposal,adopt,operational,vocab,status


@pytest.mark.parametrize('field',['adopted','protocol_file_sha256','source_sha256','original_dataset_adoption_sha256','eligible_target_rankings','preserve_accepted_mate','no_new_candidate_exclusions','future_unknown_auto_retry'])
def test_operational_migration_cannot_redefine_dataset_or_source(tmp_path,field):
    p,proposal,a,o,v,s=fixture(tmp_path);op=json.loads(o.read_text());op[field]='wrong'
    with pytest.raises(ValueError):m.verify_adoption(p,proposal,json.loads(a.read_text()),op,a)


def test_materialization_keeps_agreement_length_stop_and_last_five_percent(monkeypatch,tmp_path):
    p,proposal,a,o,v,s=fixture(tmp_path);monkeypatch.setattr(m.campaign,'inspect',lambda p:([],s))
    records=[dict(prompt='p'+str(i),chosen='good reply',rejected='bad reply',chosen_stop='length',rejected_stop='eos') for i in range(20)]
    def pairs(req,cache,jid,rubric,out):Path(out).write_bytes(m.encode_rows(records));return dict(pairs=20,unranked=2,out=out)
    monkeypatch.setattr(m.api.judge,'pairs',pairs)
    result=m.materialize(p,a,o,v,tmp_path/'output',encode=lambda s:s.split())
    assert result['readiness']['eligible_candidate_sets']==3018
    assert result['token_filter_report']['train_pairs']==19 and result['token_filter_report']['val_pairs']==1
    assert m.api.rows(tmp_path/'output/dpo_pairs.jsonl')==records
    assert result['scientific_cohort_version']=='V7_6036' and result['chosen_finished_is_breakdown_only']
    assert str(o) in result['input_sha256'] and result['original_full_valid_completeness'] is False


def test_unknown_or_incomplete_rankings_never_materialize(monkeypatch,tmp_path):
    p,proposal,a,o,v,s=fixture(tmp_path);s['eligible_missing_rankings']=1
    monkeypatch.setattr(m.campaign,'inspect',lambda p:([],s));monkeypatch.setattr(m.api.judge,'pairs',lambda *a:pytest.fail('incomplete reached pair extraction'))
    with pytest.raises(ValueError):m.materialize(p,a,o,v,tmp_path/'output',encode=lambda s:s.split())
