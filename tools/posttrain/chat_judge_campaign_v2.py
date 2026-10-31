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
import chat_judge_api as api  # noqa: E402


def save(path, value):
    api.atomic_blob(Path(path), api.encoded(value))


def source_hashes():
    return {
        name: api.file_hash(HERE / name)
        for name in ("chat_judge_campaign_v2.py", "chat_judge_api.py", "chat_judge.py")
    }


def verify_recovery_anchors(protocol):
    """Preserve original halt, explicit disposition and all prior successful/cache bytes."""
    a = protocol["recovery_anchors"]
    for name in ("old_protocol", "old_halt", "retention_receipt"):
        if api.file_hash(a[name]["path"]) != a[name]["sha256"]:
            raise ValueError("reviewed recovery anchor changed: " + name)
    if a["resolved_pending_id"] != "aafafd1531f74959936e5422388ff30e":
        raise ValueError("recovery pending identity differs")
    raw = Path(protocol["cache"]).read_bytes()
    if (
        hashlib.sha256(raw[: a["cache_prefix_bytes"]]).hexdigest()
        != a["cache_prefix_sha256"]
    ):
        raise ValueError("pre-recovery successful cache prefix changed")


def inspect(protocol):
    """Replay every settlement before any new dispatch, including restart."""
    p = protocol
    if (
        p.get("kind") != "FROZEN_FAILSTOP_API_CAMPAIGN_V2_REVIEWED_RECOVERY"
        or source_hashes() != p["source_sha256"]
    ):
        raise ValueError("campaign/source protocol differs")
    if p["profile_sha256"] != api.digest(api.PROFILE):
        raise ValueError("campaign profile differs")
    if (
        p["max_new_attempts"] != 5514
        or p["initial_training_attempts"] != 480
        or p["initial_valid_api_rankings"] != 526
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
    if api.rows(str(cache) + ".failures.jsonl"):
        raise ValueError("failed ranking requires explicit review; no automatic retry")
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
            while True:
                requests, status = inspector(protocol)
                save(
                    state / "progress.json",
                    dict(status, protocol_sha256=identity, checked_unix=time.time()),
                )
                if status["missing_rankings"] == 0:
                    save(state / "COMPLETE", dict(status, protocol_sha256=identity))
                    print("all jobs finished and home", flush=True)
                    return 0
                left = protocol["max_new_attempts"] - status["attempts_since_start"]
                if left <= 0:
                    raise ValueError(
                        "reviewed dispatch limit exhausted with missing rankings"
                    )
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
            print("campaign HALTED; explicit review required", flush=True)
            return 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--state-dir", type=Path, required=True)
    a = ap.parse_args()
    try:
        if api.file_hash(a.protocol) != a.protocol_sha256:
            raise ValueError("frozen protocol bytes differ from launch command")
        protocol = json.loads(a.protocol.read_text())
        if str(a.state_dir.resolve()) != protocol["state_dir"]:
            raise ValueError("state directory differs from protocol")
    except Exception as error:
        a.state_dir.mkdir(parents=True, exist_ok=True)
        save(
            a.state_dir / "HALTED",
            {
                "error_type": type(error).__name__,
                "reason": "launch protocol validation failed; review required",
                "requires_review": True,
            },
        )
        raise SystemExit(2) from None
    raise SystemExit(campaign(protocol, a.state_dir))


if __name__ == "__main__":
    main()
