"""Read-only V5 halt audit. No credentials, provider calls or historical mutation."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frozen-repo",
        type=Path,
        default=Path("/home/jonat/pk_api_refusal_continuation_v5_20261031"),
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    mem = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if mem < 1024 * 1024:
        raise SystemExit(
            "Audit deferred: less than 1 GiB MemAvailable; no historical state changed"
        )
    sys.path.insert(0, str(args.frozen_repo / "tools/posttrain"))
    import chat_judge_campaign_refusal_v5 as campaign

    api = campaign.api
    protocol_file = (
        args.frozen_repo
        / "experiments/chat_general/receipts/anthropic_api_refusal_v5_20261031/full_campaign_v5_protocol.json"
    )
    protocol_raw = protocol_file.read_bytes()
    assert (
        sha(protocol_raw)
        == "a576a57a7796440b2936ede0176f748372655551d83f3a57e09ab7e1e6e8b763"
    )
    p = json.loads(protocol_raw)
    assert campaign.source_hashes() == p["source_sha256"]
    campaign.verify_recovery_anchors(p)
    state = Path(p["state_dir"])
    snapshots = {
        name: (state / name).read_bytes()
        for name in ("HALTED", "STOPPED", "progress.json")
    }
    halt, stopped, progress = (
        json.loads(snapshots[name]) for name in ("HALTED", "STOPPED", "progress.json")
    )
    assert (
        halt["error_type"] == "OSError" and stopped["kind"] == "HALTED_REVIEW_REQUIRED"
    )
    identity = api.digest(p)
    assert all(x["protocol_sha256"] == identity for x in (halt, stopped, progress))
    event_raw = (state / "campaign_events.jsonl").read_bytes()
    events = [json.loads(line) for line in event_raw.splitlines()]
    returns = [e for e in events if e["event"] == "batch_return"]
    assert len(returns) == 4 and events[-1]["event"] == "batch_return"
    assert all(
        e["result"]["calls"] == e["result"]["rankings_added"] == 24
        and e["result"]["failed_rankings"] == e["result"]["pending_reservations"] == 0
        for e in returns
    )
    assert progress["attempts_since_start"] == 72
    ledger_raw = Path(p["ledger"]).read_bytes()
    delta = [
        json.loads(line) for line in ledger_raw[p["ledger_prefix_bytes"] :].splitlines()
    ]
    reserves = {e["id"]: e for e in delta if e["event"] == "reserve"}
    settled = {e["id"]: e for e in delta if e["event"] == "settle"}
    assert len(reserves) == len(settled) == 96 and set(reserves) == set(settled)
    assert len(delta) == 192
    _, status = campaign.inspect(p)
    assert status["attempts_since_start"] == 96 and status["valid_api_rankings"] == 1786
    assert (
        status["eligible_missing_rankings"] == 4252
        and status["missing_rankings"] == 4254
    )
    history = api.rows(p["ledger"])
    training_attempts = sum(
        e.get("event") == "reserve" and e.get("stage") == "training_judge"
        for e in history
    )
    assert training_attempts == 1742
    failures = api.rows(p["cache"] + ".failures.jsonl")
    assert len(failures) == 1
    excluded = {
        p["reviewed_failure_exclusion"][n] for n in ("key", "paired_unattempted_key")
    }
    assert all(e["key"] not in excluded for e in reserves.values())
    assert (
        api.file_hash(p["cache"] + ".failures.jsonl")
        == p["reviewed_failure_exclusion"]["failure_file_sha256"]
    )
    inventory = []
    for ident, settlement in settled.items():
        q = settlement["receipt"]
        trace_raw = Path(q["transport_trace"]).read_bytes()
        trace = json.loads(trace_raw)
        assert sha(trace_raw) == q["transport_trace_sha256"]
        assert trace["binding"] == {
            "reservation_id": ident,
            "request_sha256": q["request_sha256"],
        }
        assert (
            trace["http_write_started"] is True
            and q["http_status"] == 200
            and q["stop_reason"] == "end_turn"
        )
        assert any(x.get("tcp_connected") is True for x in trace["attempts"])
        assert not Path(q["request"]).with_name(ident + ".error.json").exists()
        for name in ("request", "response"):
            assert api.file_hash(q[name]) == q[name + "_sha256"]
        inventory.append(
            {
                "id": ident,
                "key": reserves[ident]["key"],
                "request_sha256": q["request_sha256"],
                "response_sha256": q["response_sha256"],
                "trace_sha256": q["transport_trace_sha256"],
                "cost_usd": settlement["cost_usd"],
            }
        )
    assert abs(status["api_actual_spend_usd"] - 26.847668) < 1e-9
    config = json.loads(
        Path(
            "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/scripts/colab_driver.json"
        ).read_text()
    )
    entries = [e for e in config["drivers"] if e.get("state_dir") == p["state_dir"]]
    assert len(entries) == 1 and entries[0]["enabled"] is False
    source = Path(campaign.__file__).read_text().splitlines()
    lines = {
        needle: next(i + 1 for i, line in enumerate(source) if needle in line)
        for needle in (
            "requests, status = inspector(protocol)",
            'state / "progress.json"',
            '"event": "batch_start"',
            '"event": "batch_return"',
        )
    }
    report = {
        "status": "AUDITED_LOCAL_OSERROR_AFTER_FOUR_SUCCESSFUL_BATCHES",
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_file_sha256": sha(protocol_raw),
        "protocol_object_sha256": identity,
        "source_sha256": campaign.source_hashes(),
        "terminal_snapshots": {
            name: {"sha256": sha(raw), "value": json.loads(raw)}
            for name, raw in snapshots.items()
        },
        "actual_readonly_inspect": status,
        "new_reservations": 96,
        "new_settlements": 96,
        "new_valid_rankings": 96,
        "global_pending_reservations": 0,
        "valid_api_rankings_total": 1786,
        "training_attempts_total": training_attempts,
        "preserved_refusals": 1,
        "unattempted_paired_exclusions": 1,
        "never_attempted_eligible_remaining": 4252,
        "new_API_errors_or_failures": 0,
        "full_cache_failure_raw_accounting_reconcile": "PASS",
        "verified_V5_trace_count": 96,
        "verified_V5_inventory_sha256": api.digest(inventory),
        "ledger_snapshot": {"bytes": len(ledger_raw), "sha256": sha(ledger_raw)},
        "events_snapshot": {"bytes": len(event_raw), "sha256": sha(event_raw)},
        "failure_file_sha256": api.file_hash(p["cache"] + ".failures.jsonl"),
        "halt_phase_bound": {
            "last_event": "batch_return",
            "last_batch_calls": 24,
            "last_batch_valid": 24,
            "last_batch_pending": 0,
            "stale_progress_attempts": 72,
            "settled_attempts": 96,
            "next_batch_reservations": 0,
            "possible_source_region": "next read-only inspect or progress publication before the next batch_start",
            "source_lines": lines,
            "exact_errno_retained": False,
            "exact_stack_retained": False,
            "OSError_cause": "not conclusively recoverable from preserved wrapper diagnostics; reported host memory pressure is context, not recovered errno proof",
        },
        "watchdog_entry_disabled": True,
        "old_HALTs_and_paired_exclusion_preserved": True,
        "full_valid_complete": False,
        "DPO_and_fixed_gate": "BLOCKED_PENDING_SEPARATE_DATASET_DECISION",
        "proposed_continuation_counts": {
            "initial_training_attempts": 1742,
            "initial_valid_rankings": 1786,
            "max_new_attempts": 4252,
            "maximum_valid_if_eligible_succeed": 6038,
            "refused": 1,
            "unattempted_paired_exclusion": 1,
        },
        "no_dispatch_or_disposition_by_audit": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, raw in snapshots.items():
        target = args.output / ("v5_" + name.replace(".json", "") + ".json")
        if target.exists():
            assert target.read_bytes() == raw
        else:
            with target.open("xb") as handle:
                handle.write(raw)
    for name, value in (
        ("halt_audit.json", report),
        ("settled_inventory.json", inventory),
    ):
        with (args.output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "new_valid_rankings",
                    "valid_api_rankings_total",
                    "global_pending_reservations",
                    "never_attempted_eligible_remaining",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
