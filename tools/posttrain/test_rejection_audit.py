"""Selection (rejection_sample.py) and the audit round trip (audit_sample.py).
Pure Python: no model, no binary."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_sample as au  # noqa: E402
import rejection_sample as rs  # noqa: E402


def rec(pi, si, reward, kind="exact_match", accepted=None):
    r = {"prompt": f"p{pi}", "completion": f"the answer is {si}", "reward": reward,
         "rationale": "why" if reward >= 0 else "no reference", "evidence_kind": kind,
         "evidence_uri": f"t/{pi}", "audited": False, "prompt_index": pi, "sample_index": si,
         "verifier": f"{kind}:v1", "stop": "eos"}
    if accepted is not None:
        r["accepted"] = accepted
    return r


def test_top_k_keeps_best_per_prompt_ties_to_lower_index_never_abstains():
    recs = [rec(0, 0, 0.0), rec(0, 1, 1.0), rec(0, 2, 1.0), rec(0, 3, -1.0),
            rec(1, 0, -1.0), rec(1, 1, -1.0)]
    out = rs.select(recs, rs.parse_keep("top1"))
    assert [r["accepted"] for r in out] == [False, True, False, False, False, False]
    out = rs.select(recs, rs.parse_keep("top3"))
    assert [r["accepted"] for r in out] == [True, True, True, False, False, False]
    assert recs[0].get("accepted") is None          # selection never mutates its input


def test_threshold_and_bad_keep():
    recs = [rec(0, 0, 0.4), rec(0, 1, 0.6), rec(0, 2, -1.0)]
    assert [r["accepted"] for r in rs.select(recs, rs.parse_keep("ge:0.5"))] == [False, True, False]
    for bad in ("top0", "best", "ge:x"):
        with pytest.raises(ValueError):
            rs.parse_keep(bad)


def test_summary_counts():
    out = rs.select([rec(0, 0, 1.0), rec(0, 1, 0.0), rec(1, 0, -1.0)], ("top", 1))
    s = rs.summarise(out, ("top", 1))
    assert s["records"] == 3 and s["abstained"] == 1 and s["accepted"] == 1 and s["mean_reward"] == 0.5
    assert s["prompts_with_an_accepted_record"] == 1


def scored_fixture():
    recs = []
    for pi in range(20):
        for si in range(4):
            r = 1.0 if si == 0 else (0.0 if si < 3 else -1.0)
            recs.append(rec(pi, si, r, kind="judge" if pi % 2 else "exact_match"))
    return rs.select(recs, ("top", 1))


def test_allocation_covers_every_stratum_and_sums_to_n():
    sizes = {"a": 50, "b": 3, "c": 1}
    al = au.allocate(sizes, 10)
    assert sum(al.values()) == 10 and all(al[k] >= 1 for k in sizes) and al["c"] == 1
    with pytest.raises(ValueError):
        au.allocate(sizes, 2)


def test_audit_round_trip(tmp_path):
    scored = scored_fixture()
    items, alloc = au.draw(scored, 12, seed=3)
    assert len(items) == 12 and len(alloc) == len({au.stratum(r) for r in scored})
    assert items == au.draw(scored, 12, seed=3)[0]            # seeded, reproducible
    au.write_jsonl(str(tmp_path / "audit.jsonl"), items)
    # A person edits the file by hand: two audits, one disagreeing with the machine.
    edited = au.read_jsonl(str(tmp_path / "audit.jsonl"))
    edited[0].update(audited=True, audit_reward=int(edited[0]["machine_reward"] >= 0.5),
                     audit_rationale="checked the sum", annotator="jr")
    edited[1].update(audited=True, audit_reward=1 - int(max(edited[1]["machine_reward"], 0) >= 0.5),
                     audit_rationale="the machine misread it", annotator="jr")
    au.write_jsonl(str(tmp_path / "audit.jsonl"), edited)
    out, human, agree = au.merge(scored, au.read_jsonl(str(tmp_path / "audit.jsonl")))
    assert len(human) == 2 and all(h["rationale"] and h["annotator"] == "jr" for h in human)
    flagged = [r for r in out if r.get("audited")]
    assert len(flagged) == 2 and all("audit_rationale" in r for r in flagged)
    assert not any(r.get("audited") for r in scored)          # input untouched
    # The flag survives a write and a read.
    au.write_jsonl(str(tmp_path / "scored.audited.jsonl"), out)
    back = au.read_jsonl(str(tmp_path / "scored.audited.jsonl"))
    assert sum(1 for r in back if r.get("audited")) == 2
    # human_audit.jsonl is in the HumanAuditVerifier line format.
    assert set(human[0]) == {"prompt", "completion", "reward", "rationale", "annotator"}


@pytest.mark.parametrize("patch,msg", [
    ({"audit_reward": None}, "0 or 1"), ({"audit_reward": 2}, "0 or 1"), ({"audit_reward": True}, "0 or 1"),
    ({"audit_rationale": "  "}, "rationale"), ({"annotator": ""}, "annotator"),
    ({"record_sha256": "0" * 64}, "does not match")])
def test_merge_refuses_incomplete_or_mismatched_audits(patch, msg):
    scored = scored_fixture()
    items, _ = au.draw(scored, 12, seed=3)
    items[0].update(audited=True, audit_reward=1, audit_rationale="ok", annotator="jr")
    items[0].update(patch)
    with pytest.raises(SystemExit, match=msg):
        au.merge(scored, items)
