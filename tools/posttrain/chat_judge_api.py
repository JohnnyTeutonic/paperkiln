#!/usr/bin/env python3
"""Prospective Anthropic judge: disjoint CLI/API profiles and durable cost reservations.

Runs on POSIX (flock), serially, with one HTTP attempt per request. Old CLI
verdicts are never silently reused; their costs and unknown charges still count.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat_judge as judge  # noqa: E402

MODEL = "claude-opus-5-5"
API_ROOT = "https://api.anthropic.com/v1"
API_VERSION = "2026-10-21"
PRICE = {"input": 4, "output": 20, "cache_read": .20, "cache_write_5m": 5, "cache_write_1h": 8}
PROFILE = {"provider": "anthropic", "endpoint": API_ROOT + "/messages", "api_version": API_VERSION,
           "model": MODEL, "max_tokens": 1024, "thinking": {"type": "adaptive"},
           "output_config": {"effort": "medium"}, "service_tier": "standard_only", "inference_geo": "global",
           "temperature": "omitted_model_default", "tools": "none", "stream": False,
           "automatic_retries": 0, "price_usd_per_million": PRICE,
           "compatibility": "CLI equivalence unestablished; disjoint profile/cache; no CLI verdict reuse"}


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(obj):
    return hashlib.sha256(encoded(obj)).hexdigest()


def profile_id():
    return MODEL + "@anthropic-messages/" + digest(PROFILE)[:16]


def nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid " + label)
    return float(value)


def token(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid usage " + label)
    return value


def usage_cost(response):
    """Provider output_tokens includes hidden thinking; never count only text."""
    if response.get("model") != MODEL:
        raise ValueError("returned model differs from requested exact model")
    u = response.get("usage")
    if not isinstance(u, dict):
        raise ValueError("missing provider usage")
    inp, out = token(u.get("input_tokens"), "input_tokens"), token(u.get("output_tokens"), "output_tokens")
    read = token(u.get("cache_read_input_tokens", 0), "cache_read_input_tokens")
    writes = token(u.get("cache_creation_input_tokens", 0), "cache_creation_input_tokens")
    detail = u.get("cache_creation") or {}
    onehour = token(detail.get("ephemeral_1h_input_tokens", 0), "ephemeral_1h_input_tokens")
    fivemin = token(detail.get("ephemeral_5m_input_tokens", writes - onehour), "ephemeral_5m_input_tokens")
    if fivemin + onehour != writes:
        raise ValueError("cache-creation usage does not reconcile")
    if any((u.get("server_tool_use") or {}).values()):
        raise ValueError("unexpected server-tool charges")
    if out > PROFILE["max_tokens"]:
        raise ValueError("provider output exceeds recorded token cap")
    if u.get("service_tier", "standard") != "standard" or u.get("speed", "standard") != "standard":
        raise ValueError("unexpected premium service pricing")
    if u.get("inference_geo", "global") != "global":
        raise ValueError("unexpected regional pricing")
    values = ((inp, "input"), (out, "output"), (read, "cache_read"),
              (fivemin, "cache_write_5m"), (onehour, "cache_write_1h"))
    return float(sum(Decimal(n) * Decimal(str(PRICE[k])) for n, k in values) / Decimal(1_000_000))


def request_body(system, user):
    return {"model": MODEL, "max_tokens": PROFILE["max_tokens"], "thinking": PROFILE["thinking"],
            "output_config": PROFILE["output_config"], "service_tier": PROFILE["service_tier"],
            "inference_geo": PROFILE["inference_geo"], "system": system,
            "messages": [{"role": "user", "content": user}], "stream": False}


def load_key(path):
    """Read only the explicitly designated ANTHROPIC_API_KEY line."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == "ANTHROPIC_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    raise ValueError("designated ANTHROPIC_API_KEY is missing or empty")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_post(key, body):
    req = urllib.request.Request(API_ROOT + "/messages", data=encoded(body), method="POST", headers={
        "x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
    # Fixed first-party endpoint, no SDK retries or redirected credentials.
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_account(cache):
    """Unknown historical costs retain a dearest-call reserve of at least $0.25."""
    if not Path(cache).is_file():
        raise ValueError("designated historical legacy cache is missing")
    paths = [Path(cache), Path(str(cache) + ".failures.jsonl")]
    records = [r for p in paths for r in rows(p)]
    known = [nonnegative(r["cost_usd"], "legacy cost") for r in records if r.get("cost_usd") is not None]
    ceiling = max(known + [.25])
    unknown = sum(r.get("cost_usd") is None for r in records)
    return {"known_spend_usd": math.fsum(known), "unknown_failure_reserve_usd": unknown * ceiling,
            "unknown_cost_records": unknown,
            "files": {str(p.resolve()): file_hash(p) for p in paths if p.exists()}}


def append(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_blob(path, raw):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def exclusive(path):
    import fcntl
    with Path(path).open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Budget:
    def __init__(self, path, baseline, cap=200, gate_reserve=25):
        self.path = Path(path)
        self.cap = nonnegative(cap, "cap")
        self.gate = nonnegative(gate_reserve, "held-out gate reserve")
        if self.cap > 200 or self.gate == 0:
            raise ValueError("campaign cap must be <=200 with a positive future gate reserve")
        records = rows(path)
        contract = {"profile": PROFILE, "legacy": baseline, "cap_usd": self.cap, "gate_reserve_usd": self.gate}
        if records:
            if records[0].get("contract") != contract:
                raise ValueError("budget/profile/legacy snapshot changed; audit before resuming")
        else:
            append(path, {"event": "contract", "contract": contract})
        self.legacy = baseline["known_spend_usd"] + baseline["unknown_failure_reserve_usd"]

    def state(self):
        pending, paid = {}, []
        for row in rows(self.path)[1:]:
            if row["event"] == "reserve":
                if row["id"] in pending:
                    raise ValueError("duplicate budget reservation")
                pending[row["id"]] = row
            elif row["event"] == "settle":
                if row["id"] not in pending:
                    raise ValueError("settlement lacks reservation")
                reservation = pending.pop(row["id"])
                cost = nonnegative(row["cost_usd"], "settled cost")
                if cost > reservation["reserve_usd"] + 1e-12:
                    raise ValueError("actual cost exceeded reservation; stop and reconcile")
                paid.append(cost)
            else:
                raise ValueError("unknown budget event")
        return pending, math.fsum(paid)

    def reserve(self, key, amount, stage):
        amount = nonnegative(amount, "call reserve")
        if amount <= 0:
            raise ValueError("positive per-call reserve required")
        pending, paid = self.state()
        if pending:
            raise ValueError("unresolved call remains reserved; reconcile before another dispatch")
        if stage == "probe":
            history = rows(self.path)[1:]
            stages = {r["id"]: r["stage"] for r in history if r["event"] == "reserve"}
            previous = math.fsum(r["cost_usd"] for r in history
                                 if r["event"] == "settle" and stages[r["id"]] == "probe")
            if previous + amount > .25 + 1e-12:
                raise ValueError("cumulative availability-probe allowance would exceed $0.25")
        keep = 0 if stage == "heldout_gate" else self.gate
        if self.legacy + paid + amount + keep > self.cap + 1e-12:
            raise ValueError("cumulative cap/future gate reserve would be exceeded")
        ident = uuid.uuid4().hex
        append(self.path, {"event": "reserve", "id": ident, "key": key, "stage": stage,
                           "reserve_usd": amount, "created_unix": time.time()})
        return ident

    def settle(self, ident, cost, receipt):
        pending, _ = self.state()
        if ident not in pending or nonnegative(cost, "cost") > pending[ident]["reserve_usd"] + 1e-12:
            raise ValueError("invalid settlement; reservation retained")
        append(self.path, {"event": "settle", "id": ident, "cost_usd": cost, "receipt": receipt})


def run(requests, cache, legacy_cache, rubric, ledger, env_file, max_calls, reserve=.25,
        cap=200, gate_reserve=25, stage="training_judge", post=None):
    """Serial API execution; profile/campaign compatibility needs prospective review."""
    if max_calls < 0 or stage not in ("probe", "training_judge", "heldout_gate"):
        raise ValueError("invalid call count/stage")
    if stage == "probe" and (max_calls > 1 or reserve > .25):
        raise ValueError("availability probe is at most one request/reservation <=$0.25")
    cache, ledger = Path(cache), Path(ledger)
    if rows(cache) and not ledger.is_file():
        raise ValueError("existing API cache requires its original cumulative budget ledger")
    if cache.resolve() == Path(legacy_cache).resolve():
        raise ValueError("API and historical CLI caches must be separate")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    blobs = ledger.with_suffix(".raw")
    blobs.mkdir(exist_ok=True)
    system, version, rsha = judge.rubric_identity(str(rubric))
    jid = profile_id()
    for record in rows(cache):
        if record.get("judge_id") != jid or record.get("transport_profile_sha256") != digest(PROFILE):
            raise ValueError("API cache contains a different/unproven transport profile")
    done = {r["key"] for r in rows(cache)}
    key = None
    calls = accepted = failures = 0
    with exclusive(str(ledger) + ".lock"):
        budget = Budget(ledger, legacy_account(legacy_cache), cap, gate_reserve)
        for request, order in judge.jobs(requests):
            semantic_key = judge.key_of(request["prompt"], request["candidates"], order, MODEL, version)
            cache_key = judge.key_of(request["prompt"], request["candidates"], order, jid, version)
            if cache_key in done:
                continue
            if calls >= max_calls:
                break
            body = request_body(system, judge.render(request["prompt"], request["candidates"], order))
            # Conservative text-byte admission plus explicit hidden-thinking/output cap.
            # Oversized requests need a separately reviewed token bound, never silent dispatch.
            input_ceiling = len(system.encode()) + len(body["messages"][0]["content"].encode()) + 4096
            bound = (input_ceiling * PRICE["input"] + PROFILE["max_tokens"] * PRICE["output"]) / 1_000_000
            if bound > reserve:
                raise ValueError("per-call reserve is below conservative request/output bound")
            if key is None:
                key = load_key(env_file)
            ident = budget.reserve(cache_key, reserve, stage)
            request_path = blobs / (ident + ".request.json")
            atomic_blob(request_path, encoded({"profile": PROFILE, "body": body, "semantic_request_key": semantic_key,
                                              "cache_key": cache_key, "rubric_sha256": rsha}))
            try:
                status, raw = (post or http_post)(key, body)
                calls += 1
            except Exception as error:
                atomic_blob(blobs / (ident + ".error.json"), encoded({"error_type": type(error).__name__,
                                                                     "charge": "unknown_reservation_retained"}))
                raise RuntimeError("API dispatch outcome unknown; reservation retained") from None
            redacted = raw.replace(key.encode(), b"[REDACTED_CREDENTIAL]")
            response_path = blobs / (ident + ".response.json")
            atomic_blob(response_path, redacted)
            if status != 200:
                atomic_blob(blobs / (ident + ".error.json"), encoded({"http_status": status,
                             "charge": "unknown_reservation_retained", "raw_sha256": hashlib.sha256(raw).hexdigest()}))
                raise RuntimeError("API HTTP status " + str(status) + "; reservation retained")
            try:
                response = json.loads(raw)
                cost = usage_cost(response)
            except (ValueError, TypeError, KeyError):
                raise RuntimeError("API usage/model/pricing unverifiable; reservation retained") from None
            receipt = {"request": str(request_path), "response": str(response_path), "http_status": status,
                       "response_sha256": hashlib.sha256(raw).hexdigest(), "credential_redacted": redacted != raw,
                       "model_returned": response["model"], "message_id": response.get("id"),
                       "usage": response["usage"], "stop_reason": response.get("stop_reason")}
            budget.settle(ident, cost, receipt)
            text = "".join(b["text"] for b in response.get("content", []) if b.get("type") == "text")
            try:
                ranking, why = judge.parse_ranking(text, order)
            except (ValueError, TypeError, AttributeError):
                ranking, why = None, ""
            if response.get("stop_reason") != "end_turn":
                ranking = None
            record = {"key": cache_key, "semantic_request_key": semantic_key, "uri": request.get("uri", ""),
                      "order": order, "judge_id": jid, "model": MODEL, "rubric_version": version,
                      "rubric_sha256": rsha, "transport_profile_sha256": digest(PROFILE),
                      "transport_receipt": receipt, "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "cost_usd": cost, "budget_reservation_id": ident}
            if ranking is None:
                append(str(cache) + ".failures.jsonl", record)
                failures += 1
            else:
                append(cache, dict(record, ranking=ranking, rationale=why))
                done.add(cache_key)
                accepted += 1
        pending, paid = budget.state()
    return {"calls": calls, "rankings_added": accepted, "failed_rankings": failures,
            "api_actual_spend_usd": paid, "legacy_spend_and_reserve_usd": budget.legacy,
            "pending_reservations": len(pending), "heldout_gate_reserve_usd": gate_reserve,
            "profile": PROFILE, "judge_id": jid}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("plan", "run", "pairs"))
    ap.add_argument("--requests", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--legacy-cache", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--ledger")
    ap.add_argument("--env-file")
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--reserve-per-call", type=float, default=.25)
    ap.add_argument("--max-usd", type=float, default=200)
    ap.add_argument("--gate-reserve-usd", type=float, default=25)
    ap.add_argument("--stage", choices=("probe", "training_judge", "heldout_gate"), default="training_judge")
    ap.add_argument("--out")
    a = ap.parse_args()
    requests = rows(a.requests)
    if a.cmd == "plan":
        _, version, _ = judge.rubric_identity(a.rubric)
        done = {r["key"] for r in rows(a.cache)}
        todo = sum(judge.key_of(r["prompt"], r["candidates"], o, profile_id(), version) not in done
                   for r, o in judge.jobs(requests))
        result = {"requests": len(requests), "calls_needed": todo, "judge_id": profile_id(), "profile": PROFILE,
                  "legacy_accounting": legacy_account(a.legacy_cache), "historical_verdict_reuse": False}
    elif a.cmd == "pairs":
        if not a.out:
            ap.error("pairs needs --out")
        result = judge.pairs(requests, a.cache, profile_id(), a.rubric, a.out)
    else:
        if not a.env_file or not a.ledger:
            ap.error("run needs --env-file and --ledger")
        result = run(requests, a.cache, a.legacy_cache, a.rubric, a.ledger, a.env_file, a.max_calls,
                     a.reserve_per_call, a.max_usd, a.gate_reserve_usd, a.stage)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()