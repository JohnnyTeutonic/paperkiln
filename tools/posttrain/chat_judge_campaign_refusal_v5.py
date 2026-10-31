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
            "chat_judge_campaign_refusal_v5.py",
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


def verify_reviewed_pair(protocol, requests, version, history, done):
    """Derive both orders from the frozen planner; never rerank refused content."""
    gate = protocol["reviewed_failure_exclusion"]
    matches = [r for r in requests if api.digest(r) == gate["candidate_case_sha256"]]
    if len(matches) != 1:
        raise ValueError("reviewed candidate case differs from fixed requests")
    planned = [
        (
            order,
            api.judge.key_of(
                r["prompt"], r["candidates"], order, api.profile_id(), version
            ),
        )
        for r, order in api.judge.jobs(matches)
    ]
    if (
        len(planned) != 2
        or {key for _, key in planned} != {gate["key"], gate["paired_unattempted_key"]}
        or [order for order, _ in planned] != gate["planned_orders"]
    ):
        raise ValueError("reviewed paired exclusions differ from frozen planner")
    attempted = {r["key"] for r in history if r.get("event") == "reserve"}
    if gate["paired_unattempted_key"] in attempted | done:
        raise ValueError("paired exclusion was already attempted or ranked")


def inspect(protocol):
    """Replay every settlement before any new dispatch, including restart."""
    p = protocol
    if (
        p.get("kind") != "FROZEN_FAILSTOP_API_REFUSAL_V5_4348"
        or source_hashes() != p["source_sha256"]
    ):
        raise ValueError("campaign/source protocol differs")
    if p["profile_sha256"] != api.digest(api.PROFILE):
        raise ValueError("campaign profile differs")
    if (
        p["max_new_attempts"] != 4348
        or p["initial_training_attempts"] != 1646
        or p["initial_valid_api_rankings"] != 1690
        or p["max_usd"] != 200
        or p["gate_reserve_usd"] != 25
        or p["reserve_per_call_usd"] != 0.25
        or p["batch_calls"] != 24
    ):
        raise ValueError("campaign limits differ from reviewed continuation")
    verify_recovery_anchors(p)
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
    reviewed = p["reviewed_failure_exclusion"]
    if (
        len(failures) != 1
        or api.file_hash(str(cache) + ".failures.jsonl")
        != reviewed["failure_file_sha256"]
        or failures[0]["key"] != reviewed["key"]
        or failures[0]["budget_reservation_id"] != reviewed["reservation_id"]
        or reviewed["key"]
        != "5bc86514bca85e5299e967378ad87cc23ad98e29942f9937c3a7fe755675790d"
        or reviewed["paired_unattempted_key"]
        != "078c6b9cc209e59b4fc8c9c829fbd6388560dc37ba54ade31cedf11cabc8a07c"
    ):
        raise ValueError("preserved reviewed failure changed; no automatic recovery")
    history = api.rows(ledger)
    training = [
        r
        for r in history
        if r.get("event") == "reserve" and r["stage"] == "training_judge"
    ]
    attempts = len(training) - p["initial_training_attempts"]
    if not 0 <= attempts <= p["max_new_attempts"]:
        raise ValueError("campaign training attempt count changed")
    done = {r["key"] for r in api.rows(cache)}
    verify_reviewed_pair(p, requests, version, history, done)
    todo = sum(
        api.judge.key_of(r["prompt"], r["candidates"], o, api.profile_id(), version)
        not in done
        for r, o in api.judge.jobs(requests)
    )
    retained = sum(
        r["ceiling_usd"] for r in history if r.get("event") == "retain_unknown_ceiling"
    )
    return requests, {
        "attempts_since_start": attempts,
        "missing_rankings": todo,
        "eligible_missing_rankings": todo - 2,
        "preserved_failed_rankings": 1,
        "unattempted_paired_exclusions": 1,
        "api_actual_spend_usd": paid,
        "retained_unknown_ceiling_usd": retained,
        "baseline_usd": budget.legacy,
        "gate_reserve_usd": budget.gate,
    }


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
                requests, status = inspector(protocol)
                save(
                    state / "progress.json",
                    dict(status, protocol_sha256=identity, checked_unix=time.time()),
                )
                left = protocol["max_new_attempts"] - status["attempts_since_start"]
                if status["eligible_missing_rankings"] == 0 or left <= 0:
                    save(
                        state / "STOPPED",
                        dict(
                            status,
                            protocol_sha256=identity,
                            kind="REFUSAL_CONTINUATION_V5_STOPPED",
                            full_judging_complete=status["missing_rankings"] == 0,
                            eligible_judging_complete=status[
                                "eligible_missing_rankings"
                            ]
                            == 0,
                        ),
                    )
                    print(
                        "V5 stopped; preserved refusal means full valid judging remains incomplete",
                        flush=True,
                    )
                    return 0
                api.append(
                    state / "campaign_events.jsonl",
                    {
                        "event": "batch_start",
                        "status": status,
                        "protocol_sha256": identity,
                        "max_calls": min(protocol["batch_calls"], left),
                    },
                )
                with redirect_stderr(stderr):
                    result = dispatch(
                        requests,
                        protocol["cache"],
                        protocol["legacy_cache"],
                        protocol["rubric"],
                        protocol["ledger"],
                        protocol["env_file"],
                        min(protocol["batch_calls"], left),
                        protocol["reserve_per_call_usd"],
                        protocol["max_usd"],
                        protocol["gate_reserve_usd"],
                        "training_judge",
                        prior_api_ledgers=protocol["prior_api_ledgers"],
                        stop_on_failure=True,
                        excluded_failed_keys=(
                            protocol["reviewed_failure_exclusion"]["key"],
                        ),
                        excluded_unattempted_keys=(
                            protocol["reviewed_failure_exclusion"][
                                "paired_unattempted_key"
                            ],
                        ),
                    )
                api.append(
                    state / "campaign_events.jsonl",
                    {
                        "event": "batch_return",
                        "result": result,
                        "stderr_present_content_suppressed": bool(stderr.getvalue()),
                        "protocol_sha256": identity,
                    },
                )
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
                    "error_type": type(error).__name__,
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
                "error_type": type(error).__name__,
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
