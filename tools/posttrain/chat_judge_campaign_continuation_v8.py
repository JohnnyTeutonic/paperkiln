#!/usr/bin/env python3
"""Separately reviewed recovery phase; no dispatch until its new protocol is authorised."""

import argparse
from contextlib import redirect_stderr
import io
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat_judge_api_refusal_v5 as api  # noqa: E402


def save(path, value):
    api.atomic_blob(Path(path), api.encoded(value))


def source_hashes():
    return {
        name: api.file_hash(HERE / name)
        for name in (
            "chat_judge_campaign_continuation_v8.py",
            "chat_judge_campaign_continuation_v7.py",
            "chat_judge_api_refusal_v5.py",
            "chat_judge_transport_diagnostic.py",
            "chat_judge.py",
        )
    }


def verify_recovery_anchors(protocol):
    a = protocol["recovery_anchors"]
    for name in (
        "old_protocol",
        "old_halt",
        "v2_protocol",
        "v2_halt",
        "first_retention_receipt",
        "v3_protocol",
        "v3_stopped",
        "v3_outcome",
        "v4_protocol",
        "v4_halt",
        "v4_stopped",
        "v4_halt_audit",
        "v5_protocol",
        "v5_halt",
        "v5_stopped",
        "v5_halt_audit",
        "v6_protocol", "v6_halt", "v6_stopped", "v6_halt_audit",
    ):
        if api.file_hash(a[name]["path"]) != a[name]["sha256"]:
            raise ValueError("prior halt/recovery anchor changed: " + name)
    raw = Path(protocol["cache"]).read_bytes()
    if (
        hashlib.sha256(raw[: a["cache_prefix_bytes"]]).hexdigest()
        != a["cache_prefix_sha256"]
    ):
        raise ValueError("prior successful cache prefix changed")
    gate = protocol["pending_disposition_gate"]
    if gate["pending_id"] != "7ef677f4605a4b489f00c4bfde5b625c":
        raise ValueError("pending disposition identity differs")
    for name in ("request", "error"):
        if api.file_hash(gate[name]["path"]) != gate[name]["sha256"]:
            raise ValueError("pending disposition raw evidence changed")
    retained = [
        r
        for r in api.rows(protocol["ledger"])
        if r.get("event") == "retain_unknown_ceiling"
        and r.get("id") == gate["pending_id"]
    ]
    if (
        len(retained) != 1
        or retained[0].get("ceiling_usd") != 0.25
        or api.digest(retained[0]) != gate["retention_event_sha256"]
    ):
        raise ValueError(
            "pending unknown disposition requires explicit root review before V3"
        )
    # Budget.state/verify_retained below verifies exact reservation, root intent,
    # raw hashes, absence of a response and full unknown ceiling; no inferred usage.



PENDING_ID='9ec8e16cff374d32811248dc4ca16140'
FRESH_KEY='449227ef031a12d087690832c6b488a4785243046941b9579ef9ce352e4fc184'
MATE_KEY='48af36ab02e12b264b0b3ba9ce73345d081f3980dd6461e44ccfa9df9943a6eb'
CASE_SHA='b0f912d18f4a52b245a259f849a4650fcdb12f3de287938869dc3ae31a4f6fa9'


def prove_untransmitted(trace,error,reservation,request_sha256):
    if (reservation.get('id')!=PENDING_ID or reservation.get('key')!=FRESH_KEY or reservation.get('reserve_usd')!=.25
        or type(trace.get('schema')) is not int or trace.get('schema')!=1 or trace.get('stage')!='connection_or_tls_failed'
        or trace.get('http_write_started') is not False
        or trace.get('binding')!={'reservation_id':PENDING_ID,'request_sha256':request_sha256}
        or trace.get('connect_failure')!={'type':'EndpointConnectionError','errno':101}
        or error.get('error_type')!='URLError' or error.get('reason_type')!='EndpointConnectionError'
        or error.get('errno')!=101 or error.get('charge')!='unknown_reservation_retained'):
        raise ValueError('exact reviewed request is not proven untransmitted')
    dns=trace.get('dns',[]);attempts=trace.get('attempts',[])
    def address(x):return (x.get('family'),x.get('ip'),x.get('port'))
    if (not dns or len(attempts)!=len(dns) or [address(x) for x in dns]!=[address(x) for x in attempts]
        or any(x.get('failure')!={'type':'OSError','errno':101} or x.get('port')!=443 or x.get('family') not in [2,10] for x in attempts)):
        raise ValueError('all resolved endpoints must have reviewed network-unreachable failures')


def fresh_authorization_expected(protocol):
    g=protocol['fresh_dispatch_gate']
    return {'kind':'ROOT_AUTHORISED_ONE_PROVEN_UNTRANSMITTED_DISPATCH_V8','authorised':True,
      'pending_id':PENDING_ID,'key':FRESH_KEY,'accepted_mate_key':MATE_KEY,'candidate_case_sha256':CASE_SHA,
      'request_sha256':g['request']['sha256'],'transport_sha256':g['transport']['sha256'],
      'pre_disposition_ledger_sha256':g['pre_disposition_ledger_sha256'],
      'retention_authorization_sha256':g['retention_authorization']['sha256'],
      'source_sha256':source_hashes(),'max_fresh_dispatches':1,'eligible_target_rankings':6036,
      'preserve_accepted_mate':True,'future_unknown_auto_retry':False,'unknown_ceiling_retained_usd':.25}


def verify_fresh_authorization(auth,protocol):
    expected=fresh_authorization_expected(protocol)
    if any(auth.get(k)!=v or type(auth.get(k))!=type(v) for k,v in expected.items()):raise ValueError('explicit one-dispatch root authorization differs')


def fresh_history(history,cache_rows):
    reserves=[r for r in history if r.get('event')=='reserve' and r.get('key')==FRESH_KEY]
    if len(reserves) not in [1,2] or reserves[0]['id']!=PENDING_ID:raise ValueError('fresh exception reserve count differs; no third dispatch')
    old=[r for r in history if r.get('event')=='retain_unknown_ceiling' and r.get('id')==PENDING_ID]
    if len(old)!=1 or old[0].get('ceiling_usd')!=.25:raise ValueError('old unknown ceiling must remain retained')
    done=[r for r in cache_rows if r.get('key')==FRESH_KEY]
    if len(reserves)==1:
        if done:raise ValueError('no response may predate explicit fresh dispatch')
        return True
    second=reserves[1]
    settled=[r for r in history if r.get('event')=='settle' and r.get('id')==second['id']]
    if len(done)!=1 or done[0].get('budget_reservation_id')!=second['id'] or len(settled)!=1:
        raise ValueError('fresh dispatch did not settle one accepted ranking; no repeat')
    return False


def verify_fresh_gate(protocol):
    g=protocol['fresh_dispatch_gate']
    fixed={'kind':'ONE_SOURCE_BOUND_PROVEN_UNTRANSMITTED_EXCEPTION_V8','pending_id':PENDING_ID,'key':FRESH_KEY,'accepted_mate_key':MATE_KEY,'candidate_case_sha256':CASE_SHA}
    if any(g.get(k)!=v for k,v in fixed.items()):raise ValueError('fresh exception gate identity differs')
    history=api.rows(protocol['ledger'])
    for name in ['v7_protocol','v7_halt','v7_stopped','request','error','transport','retention_authorization','fresh_authorization']:
        if api.file_hash(g[name]['path'])!=g[name]['sha256']:raise ValueError('explicit fresh evidence/authorization changed: '+name)
    v7=json.loads(Path(g['v7_protocol']['path']).read_text())
    if (v7.get('kind')!='FROZEN_FAILSTOP_API_PAIRED_V7_3468' or v7.get('reviewed_failure_exclusions')!=protocol['reviewed_failure_exclusions']
        or any(v7[k]!=protocol[k] for k in ['cache','ledger','requests','requests_sha256','rubric','rubric_sha256','profile_sha256','dataset_decision_file','dataset_decision_sha256'])):
        raise ValueError('V8 changes the reviewed dataset or judging contract')
    cached=Path(protocol['cache']).read_bytes()
    if hashlib.sha256(cached[:g['cache_prefix_bytes']]).hexdigest()!=g['cache_prefix_sha256']:raise ValueError('V7 accepted cache prefix changed')
    raw=Path(protocol['ledger']).read_bytes();prefix=raw[:g['pre_disposition_ledger_prefix_bytes']]
    if hashlib.sha256(prefix).hexdigest()!=g['pre_disposition_ledger_sha256']:raise ValueError('V7 settled ledger prefix changed')
    old=[r for r in history if r.get('event')=='reserve' and r.get('id')==PENDING_ID]
    if len(old)!=1 or api.digest(old[0])!=g['reservation_sha256']:raise ValueError('reviewed pending reserve differs')
    request_path=Path(g['request']['path']);response=request_path.with_name(PENDING_ID+'.response.json')
    if response.exists():raise ValueError('original reservation has a response; fresh exception forbidden')
    error=json.loads(Path(g['error']['path']).read_text());trace=json.loads(Path(g['transport']['path']).read_text())
    if error.get('transport_trace')!=g['transport']['path'] or error.get('transport_trace_sha256')!=g['transport']['sha256']:raise ValueError('error trace binding differs')
    prove_untransmitted(trace,error,old[0],g['request']['sha256'])
    verify_fresh_authorization(json.loads(Path(g['fresh_authorization']['path']).read_text()),protocol)
    retained=[r for r in history if r.get('event')=='retain_unknown_ceiling' and r.get('id')==PENDING_ID]
    if len(retained)!=1 or retained[0].get('authorization_sha256')!=api.digest(json.loads(Path(g['retention_authorization']['path']).read_text())):
        raise ValueError('explicit root retained-ceiling disposition missing')
    cache_rows=api.rows(protocol['cache'])
    if len([r for r in cache_rows if r.get('key')==MATE_KEY])!=1:raise ValueError('accepted mate must remain present')
    remaining=fresh_history(history,cache_rows)
    _,version,_=api.judge.rubric_identity(protocol['rubric']);requests=api.rows(protocol['requests'])
    matches=[r for r in requests if api.digest(r)==CASE_SHA]
    if len(matches)!=1:raise ValueError('candidate case differs')
    keys={api.judge.key_of(r['prompt'],r['candidates'],o,api.profile_id(),version) for r,o in api.judge.jobs(matches)}
    if keys!={FRESH_KEY,MATE_KEY}:raise ValueError('explicit fresh candidate/order differs')
    done={r['key'] for r in cache_rows};blocked={g[k] for g in protocol['reviewed_failure_exclusions'] for k in ['key','paired_unattempted_key']}
    todo=[api.judge.key_of(r['prompt'],r['candidates'],o,api.profile_id(),version) for r,o in api.judge.jobs(requests) if api.judge.key_of(r['prompt'],r['candidates'],o,api.profile_id(),version) not in done|blocked]
    if remaining and (not todo or todo[0]!=FRESH_KEY):raise ValueError('first isolated dispatch must be the reviewed key')
    return remaining


def verify_reviewed_pairs(protocol, requests, version, history, done):
    """Both orders excluded for each failed set; no ranking or outcome determines selection."""
    gates = protocol["reviewed_failure_exclusions"]
    if len(gates) != 2 or len({g["candidate_case_sha256"] for g in gates}) != 2:
        raise ValueError("exactly two disjoint failed candidate sets required")
    blocked = set()
    attempted = {r["key"] for r in history if r.get("event") == "reserve"}
    for gate in gates:
        matches = [r for r in requests if api.digest(r) == gate["candidate_case_sha256"]]
        if len(matches) != 1:
            raise ValueError("reviewed candidate case differs from fixed requests")
        planned = [(order, api.judge.key_of(r["prompt"], r["candidates"], order,
                    api.profile_id(), version)) for r, order in api.judge.jobs(matches)]
        keys = {key for _, key in planned}
        if (len(planned) != 2 or len(keys) != 2 or keys != {gate["key"], gate["paired_unattempted_key"]}
                or [order for order, _ in planned] != gate["planned_orders"] or blocked & keys):
            raise ValueError("reviewed paired exclusions differ from frozen planner")
        if gate["key"] in done or gate["paired_unattempted_key"] in attempted | done:
            raise ValueError("paired exclusion was already attempted or ranked")
        blocked |= keys
    # The only unranked retained key admitted is the separately reviewed untransmitted exception.
    reserves = {r["id"]: r for r in history if r.get("event") == "reserve"}
    unknown_keys = {reserves[r["id"]]["key"] for r in history if r.get("event") == "retain_unknown_ceiling"}
    exception=protocol['fresh_dispatch_gate']['key']
    if not unknown_keys <= done | {exception}:
        raise ValueError("unranked historical unknown requires separate reviewed disposition")
    return blocked


def inspect(protocol):
    """Replay every settlement before any new dispatch, including restart."""
    p = protocol
    if (
        p.get("kind") != "FROZEN_EXPLICIT_UNTRANSMITTED_API_V8_3259"
        or source_hashes() != p["source_sha256"]
    ):
        raise ValueError("campaign/source protocol differs")
    if p["profile_sha256"] != api.digest(api.PROFILE):
        raise ValueError("campaign profile differs")
    if (
        p["max_new_attempts"] != 3259
        or p["initial_training_attempts"] != 2735
        or p["initial_valid_api_rankings"] != 2777
        or p["max_usd"] != 200
        or p["gate_reserve_usd"] != 25
        or p["reserve_per_call_usd"] != 0.25
        or p["batch_calls"] != 24
        or p["minimum_mem_available_bytes"] != 1024**3
    ):
        raise ValueError("campaign limits differ from reviewed continuation")
    verify_recovery_anchors(p)
    fresh_remaining=verify_fresh_gate(p)
    if api.file_hash(p["dataset_decision_file"]) != p["dataset_decision_sha256"]:
        raise ValueError("proposed symmetric dataset decision changed")
    cache, ledger = Path(p["cache"]), Path(p["ledger"])
    if api.file_hash(Path(str(cache) + ".binding.json")) != p["binding_sha256"]:
        raise ValueError("campaign binding changed")
    raw = ledger.read_bytes()
    if (
        hashlib.sha256(raw[: p["ledger_prefix_bytes"]]).hexdigest()
        != p["ledger_prefix_sha256"]
    ):
        raise ValueError("historical ledger prefix changed")
    requests = api.rows(p["requests"])
    if api.digest(requests) != p["requests_sha256"]:
        raise ValueError("fixed campaign requests changed")
    _, version, rsha = api.judge.rubric_identity(p["rubric"])
    if rsha != p["rubric_sha256"]:
        raise ValueError("campaign rubric changed")
    binding = api.verified_binding(cache, ledger, requests, rsha, version)
    budget = api.Budget(
        ledger,
        api.legacy_account(p["legacy_cache"], p["prior_api_ledgers"]),
        p["max_usd"],
        p["gate_reserve_usd"],
        binding,
    )
    api.reconcile(cache, ledger, requests, rsha, version)
    pending, paid = budget.state()
    if pending:
        raise ValueError("pending unknown dispatch requires explicit reviewed recovery")
    failures = api.rows(str(cache) + ".failures.jsonl")
    reviewed = p["reviewed_failure_exclusions"]
    if (len(failures) != 2 or api.file_hash(str(cache) + ".failures.jsonl") != p["failure_file_sha256"]
            or {(f["key"], f["budget_reservation_id"]) for f in failures}
            != {(g["key"], g["reservation_id"]) for g in reviewed}):
        raise ValueError("preserved reviewed failures changed; no automatic recovery")
    history = api.rows(ledger)
    training = [
        r
        for r in history
        if r.get("event") == "reserve" and r["stage"] == "training_judge"
    ]
    attempts = len(training) - p["initial_training_attempts"]
    if not 0 <= attempts <= p["max_new_attempts"]:
        raise ValueError("campaign training attempt count changed")
    cache_rows = api.rows(cache)
    done = {r["key"] for r in cache_rows}
    if len(cache_rows) != len(done) or len(done) != p["initial_valid_api_rankings"] + attempts:
        raise ValueError("valid cache/unique-key/attempt counts drifted")
    verify_reviewed_pairs(p, requests, version, history, done)
    todo = sum(
        api.judge.key_of(r["prompt"], r["candidates"], o, api.profile_id(), version)
        not in done
        for r, o in api.judge.jobs(requests)
    )
    retained = sum(
        r["ceiling_usd"] for r in history if r.get("event") == "retain_unknown_ceiling"
    )
    return requests, {
        "explicit_fresh_dispatch_remaining":fresh_remaining,
        "attempts_since_start": attempts,
        "missing_rankings": todo,
        "valid_api_rankings": len(done),
        "eligible_missing_rankings": todo - 4,
        "preserved_failed_rankings": 2,
        "unattempted_paired_exclusions": 2,
        "api_actual_spend_usd": paid,
        "retained_unknown_ceiling_usd": retained,
        "baseline_usd": budget.legacy,
        "gate_reserve_usd": budget.gate,
    }


def memory_available_bytes():
    """Read Linux host headroom without loading model, cache or ledger artifacts."""
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise ValueError("invalid MemAvailable unit")
                value = int(fields[1])
                if value < 0:
                    raise ValueError("invalid MemAvailable value")
                return value * 1024
    raise ValueError("MemAvailable is unavailable")


def wait_for_memory(state, minimum, reader=None, sleeper=None):
    """Wait only before read-heavy work/dispatch, never retry any provider call."""
    if type(minimum) is not int or minimum != 1024**3:
        raise ValueError("memory admission threshold differs from reviewed 1 GiB")
    reader, sleeper = reader or memory_available_bytes, sleeper or time.sleep
    while True:
        available = reader()
        if type(available) is not int or available < 0:
            raise ValueError("invalid memory admission measurement")
        admitted = available >= minimum
        save(
            Path(state) / "memory_admission.json",
            {
                "status": "ADMITTED" if admitted else "WAITING_MEMORY",
                "minimum_available_bytes": minimum,
                "observed_available_bytes": available,
                "checked_unix": time.time(),
                "provider_call_or_retry": False,
            },
        )
        if admitted:
            return
        sleeper(10)


def safe_error_fields(error, phase):
    """Only fixed phase/type and numeric errno; never exception text."""
    errnum = getattr(error, "errno", None)
    if type(errnum) is not int:
        errnum = None
    return {"error_type": type(error).__name__, "phase": phase, "errno": errnum}


def campaign(protocol, state_dir, dispatch=None, inspector=None):
    """All restarts use this same cap; the watchdog only owns process liveness."""
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    dispatch, inspector = dispatch or api.run, inspector or inspect
    identity = api.digest(protocol)
    with api.exclusive(state / "campaign.lock"):
        if (state / "HALTED").exists():
            return 2
        stderr = io.StringIO()
        phase = "identity"
        try:
            marker = state / "campaign_identity.json"
            if marker.exists() and json.loads(marker.read_text()) != {
                "protocol_sha256": identity
            }:
                raise ValueError("state directory belongs to another campaign")
            if not marker.exists():
                save(marker, {"protocol_sha256": identity})
            if (state / "STOPPED").exists():
                stopped = json.loads((state / "STOPPED").read_text())
                if stopped.get("protocol_sha256") != identity:
                    raise ValueError("continuation stop identity differs")
                return 0
            while True:
                phase = "memory_admission_before_inspect"
                wait_for_memory(state, protocol["minimum_mem_available_bytes"])
                phase = "inspect"
                requests, status = inspector(protocol)
                phase = "progress_publish"
                save(
                    state / "progress.json",
                    dict(status, protocol_sha256=identity, checked_unix=time.time()),
                )
                left = protocol["max_new_attempts"] - status["attempts_since_start"]
                if left <= 0 and status["eligible_missing_rankings"] > 0:
                    raise ValueError("reviewed attempt limit exhausted before eligible completeness")
                if status["eligible_missing_rankings"] == 0:
                    phase = "terminal_publish"
                    save(
                        state / "STOPPED",
                        dict(
                            status,
                            protocol_sha256=identity,
                            kind="PAIRED_CONTINUATION_V8_STOPPED",
                            full_judging_complete=status["missing_rankings"] == 0,
                            eligible_judging_complete=status[
                                "eligible_missing_rankings"
                            ]
                            == 0,
                        ),
                    )
                    print(
                        "V8 stopped; preserved refusal and paired exclusion keep full valid judging incomplete",
                        flush=True,
                    )
                    return 0
                phase = "memory_admission_before_dispatch"
                wait_for_memory(state, protocol["minimum_mem_available_bytes"])
                phase = "batch_start_publish"
                api.append(
                    state / "campaign_events.jsonl",
                    {
                        "event": "batch_start",
                        "status": status,
                        "protocol_sha256": identity,
                        "max_calls": (1 if status["explicit_fresh_dispatch_remaining"] else min(protocol["batch_calls"], left)),
                    },
                )
                phase = "dispatch"
                with redirect_stderr(stderr):
                    result = dispatch(
                        requests,
                        protocol["cache"],
                        protocol["legacy_cache"],
                        protocol["rubric"],
                        protocol["ledger"],
                        protocol["env_file"],
                        (1 if status["explicit_fresh_dispatch_remaining"] else min(protocol["batch_calls"], left)),
                        protocol["reserve_per_call_usd"],
                        protocol["max_usd"],
                        protocol["gate_reserve_usd"],
                        "training_judge",
                        prior_api_ledgers=protocol["prior_api_ledgers"],
                        stop_on_failure=True,
                        excluded_failed_keys=tuple(g["key"] for g in protocol["reviewed_failure_exclusions"]),
                        excluded_unattempted_keys=tuple(g["paired_unattempted_key"] for g in protocol["reviewed_failure_exclusions"]),
                    )
                phase = "batch_return_publish"
                api.append(
                    state / "campaign_events.jsonl",
                    {
                        "event": "batch_return",
                        "result": result,
                        "stderr_present_content_suppressed": bool(stderr.getvalue()),
                        "protocol_sha256": identity,
                    },
                )
                phase = "batch_result_check"
                if result["failed_rankings"] or result["pending_reservations"]:
                    raise ValueError(
                        "batch contains failed ranking or pending charge; review required"
                    )
                if not result["calls"]:
                    raise ValueError("batch made no progress with missing rankings")
        except Exception as error:
            # Persist the ledger and all raw evidence; never release or retry a charge here.
            message = (
                "review ledger and raw receipts; exception text and stderr suppressed"
            )
            save(
                state / "HALTED",
                {
                    **safe_error_fields(error, phase),
                    "reason": message,
                    "protocol_sha256": identity,
                    "requires_review": True,
                    "stderr_present_content_suppressed": bool(stderr.getvalue()),
                },
            )
            save(
                state / "STOPPED",
                {"kind": "HALTED_REVIEW_REQUIRED", "protocol_sha256": identity},
            )
            print("campaign HALTED; explicit review required", flush=True)
            return 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--state-dir", type=Path, required=True)
    a = ap.parse_args()
    already_stopped = (a.state_dir / "HALTED").exists() or (
        a.state_dir / "STOPPED"
    ).exists()
    try:
        if api.file_hash(a.protocol) != a.protocol_sha256:
            raise ValueError("frozen protocol bytes differ from launch command")
        protocol = json.loads(a.protocol.read_text())
        if str(a.state_dir.resolve()) != protocol["state_dir"]:
            raise ValueError("state directory differs from protocol")
    except Exception as error:
        if already_stopped:
            raise SystemExit(2) from None
        a.state_dir.mkdir(parents=True, exist_ok=True)
        save(
            a.state_dir / "HALTED",
            {
                **safe_error_fields(error, "launch_validation"),
                "reason": "launch protocol validation failed; review required",
                "requires_review": True,
            },
        )
        save(
            a.state_dir / "STOPPED",
            {
                "kind": "LAUNCH_VALIDATION_HALTED_REVIEW_REQUIRED",
                "protocol_file_sha256_requested": a.protocol_sha256,
            },
        )
        raise SystemExit(2) from None
    raise SystemExit(campaign(protocol, a.state_dir))


if __name__ == "__main__":
    main()
