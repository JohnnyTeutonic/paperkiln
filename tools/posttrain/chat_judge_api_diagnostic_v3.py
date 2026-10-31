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
from chat_judge_transport_diagnostic import Trace, TrackingHTTPSHandler  # noqa: E402

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


def http_post(key, body, *, trace):
    req = urllib.request.Request(API_ROOT + "/messages", data=encoded(body), method="POST", headers={
        "x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
    # Fixed first-party endpoint, no SDK retries or redirected credentials.
    opener = urllib.request.build_opener(NoRedirect, TrackingHTTPSHandler(trace))
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


def legacy_account(cache, prior_api_ledgers=()):
    """Unknown historical costs retain a dearest-call reserve of at least $0.25."""
    if not Path(cache).is_file():
        raise ValueError("designated historical legacy cache is missing")
    paths = [Path(cache), Path(str(cache) + ".failures.jsonl")]
    records = [r for p in paths for r in rows(p)]
    known = [nonnegative(r["cost_usd"], "legacy cost") for r in records if r.get("cost_usd") is not None]
    ceiling = max(known + [.25])
    unknown_records = [r for r in records if r.get("cost_usd") is None]
    unknown = len(unknown_records)
    historical_reserve = math.fsum(max([ceiling] + [nonnegative(r[k], "historical failed-call reserve")
                        for k in ("reserved_usd", "reserve_usd", "cost_reservation_usd", "reserved_cost_usd") if k in r])
                        for r in unknown_records)
    prior_paid, prior_reserve, prior_files = [], [], {}
    for ledger in prior_api_ledgers:
        ledger = Path(ledger)
        history = rows(ledger)
        if not history or history[0].get("event") != "contract":
            raise ValueError("prior API ledger lacks its contract")
        profile = history[0]["contract"]["profile"]
        if profile != PROFILE:
            raise ValueError("prior API pricing/profile requires a separate audited migration")
        reserved = {}
        for row in history[1:]:
            if row["event"] == "reserve":
                if row["id"] in reserved:
                    raise ValueError("duplicate prior API reservation")
                reserved[row["id"]] = row
            elif row["event"] == "settle":
                before = reserved.pop(row["id"])
                receipt = row["receipt"]
                raw_path = Path(receipt["response"])
                raw = raw_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]:
                    raise ValueError("prior API raw response hash mismatch")
                actual = usage_cost(json.loads(raw))
                if actual != row["cost_usd"] or actual > before["reserve_usd"]:
                    raise ValueError("prior API cost cannot be replayed")
                prior_paid.append(actual)
                prior_files[str(raw_path.resolve())] = file_hash(raw_path)
                request_path = Path(receipt["request"])
                prior_files[str(request_path.resolve())] = file_hash(request_path)
            elif row["event"] == "retain_unknown_ceiling":
                before = reserved.pop(row["id"])
                verify_retained(row, before, ledger)
                prior_reserve.append(row["ceiling_usd"])
            else:
                raise ValueError("unknown prior API event")
        prior_reserve.extend(nonnegative(r["reserve_usd"], "prior API reserve") for r in reserved.values())
        prior_files[str(ledger.resolve())] = file_hash(ledger)
    return {"known_spend_usd": math.fsum(known), "unknown_failure_reserve_usd": historical_reserve,
            "unknown_cost_records": unknown, "prior_api_spend_usd": math.fsum(prior_paid),
            "prior_api_unknown_reserve_usd": math.fsum(prior_reserve), "prior_api_files": prior_files,
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


def verify_retained(row, reservation, ledger):
    ident = row["id"]
    rawdir = Path(ledger).with_suffix(".raw")
    request, error = rawdir / (ident + ".request.json"), rawdir / (ident + ".error.json")
    authorization = row.get("authorization") or {}
    prefix = bytearray()
    for line in Path(ledger).read_bytes().splitlines(keepends=True):
        event = json.loads(line)
        if event.get("event") == "retain_unknown_ceiling" and event.get("id") == ident:
            break
        prefix.extend(line)
    contract = rows(ledger)[0]["contract"]
    if (authorization.get("kind") != "ROOT_AUTHORISED_RETAIN_UNKNOWN_CEILING"
            or digest(authorization) != row.get("authorization_sha256")
            or authorization.get("ledger_sha256") != hashlib.sha256(prefix).hexdigest()
            or authorization.get("binding_sha256") != digest(contract.get("binding"))
            or authorization.get("pending_id") != ident
            or row.get("reservation_sha256") != digest(reservation)
            or row.get("ceiling_usd") != reservation["reserve_usd"]
            or row.get("raw_request_sha256") != file_hash(request)
            or row.get("error_sha256") != file_hash(error)
            or row.get("error_type") != json.loads(error.read_text()).get("error_type")
            or (rawdir / (ident + ".response.json")).exists()
            or row.get("charge") != "unknown_full_ceiling_retained_not_actual_spend"
            or not row.get("authorization_sha256")):
        raise ValueError("retained unknown charge no longer matches its authorised evidence")



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
    def __init__(self, path, baseline, cap=200, gate_reserve=25, binding=None):
        self.path = Path(path)
        self.cap = nonnegative(cap, "cap")
        self.gate = nonnegative(gate_reserve, "held-out gate reserve")
        if self.cap > 200 or self.gate == 0:
            raise ValueError("campaign cap must be <=200 with a positive future gate reserve")
        records = rows(path)
        if records and binding is None:
            binding = records[0]["contract"].get("binding")
        self.binding = binding
        contract = {"profile": PROFILE, "legacy": baseline, "cap_usd": self.cap,
                    "gate_reserve_usd": self.gate, "binding": binding}
        if records:
            if records[0].get("contract") != contract:
                raise ValueError("budget/profile/legacy snapshot changed; audit before resuming")
        else:
            append(path, {"event": "contract", "contract": contract})
        self.legacy = math.fsum(baseline.get(k, 0) for k in (
            "known_spend_usd", "unknown_failure_reserve_usd", "prior_api_spend_usd", "prior_api_unknown_reserve_usd"))

    def state(self):
        pending, paid, seen = {}, [], set()
        for row in rows(self.path)[1:]:
            if row["event"] == "reserve":
                if row["id"] in seen:
                    raise ValueError("duplicate budget reservation")
                seen.add(row["id"])
                pending[row["id"]] = row
            elif row["event"] == "settle":
                if row["id"] not in pending:
                    raise ValueError("settlement lacks reservation")
                reservation = pending.pop(row["id"])
                cost = nonnegative(row["cost_usd"], "settled cost")
                if cost > reservation["reserve_usd"] + 1e-12:
                    raise ValueError("actual cost exceeded reservation; stop and reconcile")
                paid.append(cost)
            elif row["event"] == "retain_unknown_ceiling":
                if row["id"] not in pending:
                    raise ValueError("unknown-charge resolution is duplicate or lacks pending reservation")
                reservation = pending.pop(row["id"])
                verify_retained(row, reservation, self.path)
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
        history = rows(self.path)[1:]
        stages = {r["id"]: r["stage"] for r in history if r["event"] == "reserve"}
        retained = math.fsum(r["ceiling_usd"] for r in history if r["event"] == "retain_unknown_ceiling")
        previous = math.fsum((r["cost_usd"] if r["event"] == "settle" else r["ceiling_usd"])
                            for r in history if r["event"] in ("settle", "retain_unknown_ceiling")
                            and stages[r["id"]] == stage)
        attempts = sum(r["event"] == "reserve" and r["stage"] == stage for r in history)
        limits = (self.binding or {}).get("stage_limits", {"probe": {"attempts": 1, "usd": .25}})
        if stage not in limits:
            raise ValueError("stage is not prospectively registered")
        if attempts >= limits[stage]["attempts"]:
            raise ValueError("registered stage attempt limit reached")
        if previous + amount > limits[stage]["usd"] + 1e-12:
            raise ValueError("registered stage dollar allowance would be exceeded")
        keep = self.gate
        if self.legacy + paid + retained + amount + keep > self.cap + 1e-12:
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


    def retain_unknown(self, authorization):
        """Explicit reviewed disposition; retain money, never invent usage or verdict."""
        if authorization.get("kind") != "ROOT_AUTHORISED_RETAIN_UNKNOWN_CEILING":
            raise ValueError("explicit root-authorised recovery receipt required")
        if (authorization.get("ledger_sha256") != file_hash(self.path)
                or authorization.get("binding_sha256") != digest(self.binding)):
            raise ValueError("recovery authorization does not bind this exact pending ledger")
        pending, _ = self.state()
        ident = authorization["pending_id"]
        if set(pending) != {ident}:
            raise ValueError("authorization does not identify the one unresolved call")
        reservation = pending[ident]
        rawdir = self.path.with_suffix(".raw")
        error = json.loads((rawdir / (ident + ".error.json")).read_text())
        event = {"event": "retain_unknown_ceiling", "id": ident, "ceiling_usd": reservation["reserve_usd"],
                 "reservation_sha256": digest(reservation),
                 "raw_request_sha256": file_hash(rawdir / (ident + ".request.json")),
                 "error_sha256": file_hash(rawdir / (ident + ".error.json")),
                 "error_type": error.get("error_type"), "authorization_sha256": digest(authorization),
                 "authorization": authorization,
                 "charge": "unknown_full_ceiling_retained_not_actual_spend",
                 "reason": authorization.get("reason", "")}
        if (event["raw_request_sha256"] != authorization.get("raw_request_sha256")
                or event["error_sha256"] != authorization.get("error_sha256")
                or event["error_type"] != authorization.get("error_type")
                or not event["reason"]):
            raise ValueError("recovery authorization evidence differs")
        verify_retained(event, reservation, self.path)
        append(self.path, event)
        return event


def binding_identity(cache, ledger, requests, rsha, version):
    return {"schema": 1, "cache": str(cache.resolve()), "ledger": str(ledger.resolve()),
            "profile_sha256": digest(PROFILE), "requests_sha256": digest(requests), "rubric_sha256": rsha,
            "rubric_version": version,
            "stage_limits": {"probe": {"attempts": 1, "usd": .25},
                             "calibration": {"attempts": 48, "usd": 2},
                             "training_judge": {"attempts": 2 * len(requests), "usd": 200}},
            "heldout_gate": "reserved; new held-out dataset/transport needs explicit reviewed stage migration"}


def verified_binding(cache, ledger, requests, rsha, version):
    expected = binding_identity(cache, ledger, requests, rsha, version)
    path = Path(str(cache) + ".binding.json")
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("immutable cache/ledger/request/rubric binding differs; reviewed migration required")
    return expected


def bind_cache(cache, ledger, requests, rsha, version):
    """Called under both path locks before any journal or cache operation."""
    path = Path(str(cache) + ".binding.json")
    if path.exists():
        return verified_binding(cache, ledger, requests, rsha, version)
    if rows(cache) or rows(str(cache) + ".failures.jsonl") or rows(ledger):
        raise ValueError("existing cache or ledger lacks verified binding; reviewed migration required")
    identity = binding_identity(cache, ledger, requests, rsha, version)
    atomic_blob(path, encoded(identity))
    return identity


def reconcile(cache, ledger, requests, rsha, version):
    """Every scientific cache row must have one durable, replayable paid settlement."""
    records = rows(ledger)
    if not records or records[0].get("event") != "contract":
        raise ValueError("ledger lacks budget contract")
    binding = records[0]["contract"].get("binding")
    if not binding or binding["cache"] != str(cache.resolve()) or binding["ledger"] != str(ledger.resolve()):
        raise ValueError("ledger and cache are not bound")
    reserves, settled, retained = {}, {}, {}
    for row in records[1:]:
        if row["event"] == "reserve":
            if row["id"] in reserves:
                raise ValueError("duplicate reservation identity")
            reserves[row["id"]] = row
        elif row["event"] == "settle":
            if row["id"] in settled or row["id"] in retained or row["id"] not in reserves:
                raise ValueError("duplicate/unreserved settlement identity")
            settled[row["id"]] = row
        elif row["event"] == "retain_unknown_ceiling":
            if row["id"] in settled or row["id"] in retained or row["id"] not in reserves:
                raise ValueError("duplicate/unreserved unknown-charge resolution")
            verify_retained(row, reserves[row["id"]], ledger)
            retained[row["id"]] = row
        else:
            raise ValueError("unknown journal event")
    valid = {(judge.key_of(r["prompt"], r["candidates"], o, MODEL, version), tuple(o)): r
             for r, o in judge.jobs(requests)}
    used = set()
    accepted_rows = [(True, row) for row in rows(cache)]
    failed_rows = [(False, row) for row in rows(str(cache) + ".failures.jsonl")]
    for accepted, row in accepted_rows + failed_rows:
        ident = row.get("budget_reservation_id")
        if ident in used or ident not in settled:
            raise ValueError("cached/failed row has no unique settled reservation")
        used.add(ident)
        settlement, reservation = settled[ident], reserves[ident]
        receipt = settlement["receipt"]
        if (row.get("judge_id") != profile_id() or row.get("transport_profile_sha256") != digest(PROFILE)
                or row.get("rubric_sha256") != rsha or row.get("rubric_version") != version
                or row.get("key") != reservation["key"]
                or row.get("cost_usd") != settlement["cost_usd"] or row.get("transport_receipt") != receipt):
            raise ValueError("cache/settlement profile, cost, key or receipt mismatch")
        request_path, response_path = Path(receipt["request"]), Path(receipt["response"])
        expected = ledger.with_suffix(".raw")
        if (request_path != expected / (ident + ".request.json")
                or response_path != expected / (ident + ".response.json")):
            raise ValueError("receipt path escapes bound journal namespace")
        raw_request, raw_response = request_path.read_bytes(), response_path.read_bytes()
        if (hashlib.sha256(raw_request).hexdigest() != receipt["request_sha256"]
                or hashlib.sha256(raw_response).hexdigest() != receipt["response_sha256"]):
            raise ValueError("raw request/response hash mismatch")
        req = json.loads(raw_request)
        if (req["profile"] != PROFILE or req["cache_key"] != row["key"]
                or req["rubric_sha256"] != rsha or req["semantic_request_key"] != row["semantic_request_key"]):
            raise ValueError("raw request identity mismatch")
        body = req["body"]
        if hashlib.sha256(body["system"].encode()).hexdigest() != rsha:
            raise ValueError("raw system prompt differs from bound rubric")
        order = row["order"]
        original = valid.get((row["semantic_request_key"], tuple(order)))
        if original is None:
            raise ValueError("raw request is not a member of the fixed campaign")
        if (body != request_body(body["system"], judge.render(original["prompt"], original["candidates"], order))
                or judge.key_of(original["prompt"], original["candidates"], order, profile_id(), row["rubric_version"])
                != row["key"]):
            raise ValueError("payload differs from fixed request rendering")
        response = json.loads(raw_response)
        if (usage_cost(response) != settlement["cost_usd"] or response["usage"] != receipt["usage"]
                or settlement["cost_usd"] > reservation["reserve_usd"]
                or receipt["http_status"] != 200 or receipt["model_returned"] != response["model"]
                or receipt["stop_reason"] != response.get("stop_reason")
                or receipt["message_id"] != response.get("id")):
            raise ValueError("provider usage/cost/metadata replay mismatch")
        text = "".join(b["text"] for b in response.get("content", []) if b.get("type") == "text")
        try:
            ranking, why = judge.parse_ranking(text, order)
        except (ValueError, TypeError, AttributeError):
            ranking, why = None, ""
        if hashlib.sha256(text.encode()).hexdigest() != row["response_sha256"]:
            raise ValueError("cached text hash differs from raw response")
        if accepted:
            if (response.get("stop_reason") != "end_turn" or ranking != row.get("ranking")
                    or not ranking or why != row.get("rationale")):
                raise ValueError("cached ranking differs from raw provider response")
        elif "ranking" in row or (ranking is not None and response.get("stop_reason") == "end_turn"):
            raise ValueError("failure row does not represent a failed ranking")
    if used != set(settled):
        raise ValueError("settled call missing cache/failure record; recover from raw receipt before dispatch")


def run(requests, cache, legacy_cache, rubric, ledger, env_file, max_calls, reserve=.25,
        cap=200, gate_reserve=25, stage="training_judge", post=None, prior_api_ledgers=(), stop_on_failure=False):
    """Serial API execution; profile/campaign compatibility needs prospective review."""
    if max_calls < 0 or stage not in ("probe", "calibration", "training_judge"):
        raise ValueError("invalid call count/stage")
    if stage == "probe" and (max_calls > 1 or reserve > .25):
        raise ValueError("availability probe is at most one request/reservation <=$0.25")
    cache, ledger = Path(cache), Path(ledger)
    if cache.resolve() == Path(legacy_cache).resolve():
        raise ValueError("API and historical CLI caches must be separate")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    blobs = ledger.with_suffix(".raw")
    blobs.mkdir(exist_ok=True)
    system, version, rsha = judge.rubric_identity(str(rubric))
    jid = profile_id()
    key = None
    calls = accepted = failures = 0
    with exclusive(str(cache) + ".lock"), exclusive(str(ledger) + ".lock"):
        binding = bind_cache(cache, ledger, requests, rsha, version)
        budget = Budget(ledger, legacy_account(legacy_cache, prior_api_ledgers), cap, gate_reserve, binding)
        reconcile(cache, ledger, requests, rsha, version)
        done = {r["key"] for r in rows(cache)}
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
            trace_path = blobs / (ident + ".transport.json")
            try:
                trace = Trace(trace_path, binding={"reservation_id": ident, "request_sha256": file_hash(request_path)})
                status, raw = post(key, body) if post is not None else http_post(key, body, trace=trace)
                calls += 1
            except Exception as error:
                cause = getattr(error, "reason", None)
                atomic_blob(blobs / (ident + ".error.json"), encoded({"error_type": type(error).__name__,
                            "reason_type": type(cause).__name__, "errno": getattr(cause, "errno", None),
                            "charge": "unknown_reservation_retained",
                            "transport_trace": str(trace_path),
                            "transport_trace_sha256": file_hash(trace_path) if trace_path.exists() else None}))
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
                       "request_sha256": file_hash(request_path),
                       "response_sha256": hashlib.sha256(raw).hexdigest(), "credential_redacted": redacted != raw,
                       "model_returned": response["model"], "message_id": response.get("id"),
                       "usage": response["usage"], "stop_reason": response.get("stop_reason"),
                       "transport_trace": str(trace_path), "transport_trace_sha256": file_hash(trace_path)}
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
                if stop_on_failure:
                    break
            else:
                append(cache, dict(record, ranking=ranking, rationale=why))
                done.add(cache_key)
                accepted += 1
        pending, paid = budget.state()
    return {"calls": calls, "rankings_added": accepted, "failed_rankings": failures,
            "api_actual_spend_usd": paid, "legacy_spend_and_reserve_usd": budget.legacy,
            "pending_reservations": len(pending), "heldout_gate_reserve_usd": gate_reserve,
            "api_retained_unknown_ceiling_usd": math.fsum(r["ceiling_usd"] for r in rows(ledger)
                                                         if r.get("event") == "retain_unknown_ceiling"),
            "profile": PROFILE, "judge_id": jid}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("plan", "run", "pairs", "retain-unknown"))
    ap.add_argument("--requests", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--legacy-cache", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--ledger")
    ap.add_argument("--authorization-file")
    ap.add_argument("--prior-api-ledger", action="append", default=[])
    ap.add_argument("--env-file")
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--reserve-per-call", type=float, default=.25)
    ap.add_argument("--max-usd", type=float, default=200)
    ap.add_argument("--gate-reserve-usd", type=float, default=25)
    ap.add_argument("--stage", choices=("probe", "calibration", "training_judge"), default="training_judge")
    ap.add_argument("--out")
    a = ap.parse_args()
    requests = rows(a.requests)
    if a.cmd == "plan":
        _, version, rsha = judge.rubric_identity(a.rubric)
        if rows(a.cache) or rows(a.cache + ".failures.jsonl"):
            if not a.ledger:
                ap.error("existing API state needs --ledger for verified planning")
            verified_binding(Path(a.cache), Path(a.ledger), requests, rsha, version)
            reconcile(Path(a.cache), Path(a.ledger), requests, rsha, version)
        done = {r["key"] for r in rows(a.cache)}
        todo = sum(judge.key_of(r["prompt"], r["candidates"], o, profile_id(), version) not in done
                   for r, o in judge.jobs(requests))
        result = {"requests": len(requests), "calls_needed": todo, "judge_id": profile_id(), "profile": PROFILE,
                  "legacy_accounting": legacy_account(a.legacy_cache, a.prior_api_ledger), "historical_verdict_reuse": False}
    elif a.cmd == "retain-unknown":
        if not a.ledger or not a.authorization_file:
            ap.error("retain-unknown needs --ledger and --authorization-file")
        _, version, rsha = judge.rubric_identity(a.rubric)
        with exclusive(a.cache + ".lock"), exclusive(a.ledger + ".lock"):
            binding = verified_binding(Path(a.cache), Path(a.ledger), requests, rsha, version)
            budget = Budget(a.ledger, legacy_account(a.legacy_cache, a.prior_api_ledger),
                            a.max_usd, a.gate_reserve_usd, binding)
            reconcile(Path(a.cache), Path(a.ledger), requests, rsha, version)
            result = budget.retain_unknown(json.loads(Path(a.authorization_file).read_text()))
    elif a.cmd == "pairs":
        if not a.out or not a.ledger:
            ap.error("pairs needs --out and --ledger for verified cache provenance")
        _, version, rsha = judge.rubric_identity(a.rubric)
        verified_binding(Path(a.cache), Path(a.ledger), requests, rsha, version)
        reconcile(Path(a.cache), Path(a.ledger), requests, rsha, version)
        result = judge.pairs(requests, a.cache, profile_id(), a.rubric, a.out)
    else:
        if not a.env_file or not a.ledger:
            ap.error("run needs --env-file and --ledger")
        result = run(requests, a.cache, a.legacy_cache, a.rubric, a.ledger, a.env_file, a.max_calls,
                     a.reserve_per_call, a.max_usd, a.gate_reserve_usd, a.stage, prior_api_ledgers=a.prior_api_ledger)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()