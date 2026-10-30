#!/usr/bin/env python3
"""Listwise LLM-judge rankings of chat replies, cached with receipts, and the
preference pairs they license (charter 0001 §6: an offline judge, its verdicts
cached and auditable; experiments/chat_general/PLAN.md).

    python tools/posttrain/chat_judge.py plan  --requests R.jsonl --cache C.jsonl --rubric RUBRIC --judge-id MODEL
    python tools/posttrain/chat_judge.py run   --requests R.jsonl --cache C.jsonl --rubric RUBRIC --judge-id MODEL \\
        --claude-bin PATH --max-calls N
    python tools/posttrain/chat_judge.py pairs --requests R.jsonl --cache C.jsonl --rubric RUBRIC --judge-id MODEL \\
        --out PAIRS.jsonl

Requests are {"prompt": templated chat prompt (tools/chat_data.py), "candidates":
[reply, ...], "uri"} lines, 2-8 candidates each. The judge is Claude through the
logged-in Claude Code CLI in print mode, so no API key is read: the rubric is
the whole system prompt, tools and MCP servers are off, one turn, JSON output.

Bias controls:
- position: every prompt is ranked twice, candidates in the given order and
  reversed; a prompt yields a pair only if both rankings agree on the best and
  the worst reply;
- length: a pair is kept only if chosen and rejected are within 1.5x of each
  other in words (judges favour longer replies);
- identity: the cache key is the prompt, the candidates, the order, the judge
  id and the rubric version (file name + sha256 prefix), so editing a rubric
  or a candidate can never reuse a stale verdict.
Every verdict is appended to the cache with its rationale, the response hash
and the cost the CLI reports; failures (no JSON, not a permutation) go to
<cache>.failures.jsonl and yield nothing. --max-calls is a hard cap.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from judge_sidecar import read_jsonl, rubric_identity, sha256_bytes  # noqa: E402
from chat_sample_ember import split_prompt  # noqa: E402

LABELS = "ABCDEFGH"


def key_of(prompt, candidates, order, judge_id, rubric_version):
    blob = json.dumps([prompt, candidates, order, judge_id, rubric_version], ensure_ascii=False)
    return sha256_bytes(blob.encode("utf-8"))


def render(prompt, candidates, order):
    """The user message the judge sees: the conversation, then the labelled replies in `order`."""
    hist, user = split_prompt(prompt)
    lines = []
    if hist:
        lines.append("Conversation so far:")
        for t in hist:
            lines += [f"User: {t['user']}", f"Assistant: {t['assistant']}"]
        lines.append("")
    lines += [f"User message: {user}", "", "Candidate replies:"]
    for lab, i in zip(LABELS, order):
        lines.append(f"{lab}: {candidates[i]}")
    return "\n".join(lines)


def parse_ranking(text, order):
    """JSON {"ranking": [labels], "rationale"} -> candidate indices best first, or None."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    rk, why = d.get("ranking"), str(d.get("rationale", "")).strip()
    labs = list(LABELS[:len(order)])
    if not isinstance(rk, list) or sorted(map(str, rk)) != labs or not why:
        return None, why
    return [order[labs.index(str(x))] for x in rk], why


class JudgeStopped(SystemExit):
    """A non-verdict CLI response may still report a billable call."""

    def __init__(self, response, cost_usd):
        self.response, self.cost_usd = response, cost_usd
        super().__init__(f"claude CLI: {response[:200]!r}; stopped, resolve the CLI error before resuming")


def claude_cli_transport(claude_bin, model, workdir=None):
    """The CLI runs in an empty directory of its own: started inside a repository it loads that
    repository's instruction files and project memory into every call (27 Oct 2026, from the
    microtorch root: both AGENTS.md files, 15,610 prompt tokens and $0.13 a call against 1,177 and
    $0.012 from an empty directory), so the judge would see them and the cost would be tenfold."""
    import tempfile
    workdir = workdir or tempfile.mkdtemp(prefix="chat_judge_")
    if os.listdir(workdir):
        raise SystemExit(f"{workdir}: the judge's working directory must be empty")

    def call(system, user):
        p = subprocess.run([claude_bin, "-p", user, "--system-prompt", system, "--model", model, "--tools", "",
                            "--strict-mcp-config", "--no-session-persistence", "--max-turns", "1",
                            "--output-format", "json"], capture_output=True, text=True, timeout=300,
                           stdin=subprocess.DEVNULL, cwd=workdir)
        d = json.loads(p.stdout)
        text, cost = d.get("result", ""), float(d.get("total_cost_usd", 0.0))
        # An account limit is no verdict: stop the run instead of logging a failure per call
        # (28 Oct 2026: 121 calls answered "You've hit your session limit" at $0). The verdicts
        # already cached stay; a rerun after the reset continues with the rest.
        if d.get("is_error") or (cost == 0.0 and re.search(r"\b(session|usage|weekly|rate) limit\b", text)):
            raise JudgeStopped(text, cost)
        return text, cost
    return call


def jobs(requests):
    for r in requests:
        n = len(r["candidates"])
        if not 2 <= n <= len(LABELS):
            raise SystemExit(f"{r.get('uri')}: need 2-{len(LABELS)} candidates, got {n}")
        fwd = list(range(n))
        for order in (fwd, fwd[::-1]):
            yield r, order


def spent(cache_path):
    """Dollars already recorded for this cache: verdicts and failures alike."""
    return sum(float(c.get("cost_usd", 0.0)) for p in (cache_path, cache_path + ".failures.jsonl")
               for c in read_jsonl(p))


def run(requests, cache_path, judge_id, rubric_path, max_calls, transport, now=None, max_usd=None, workers=1):
    """max_usd caps the cache's TOTAL recorded spend (earlier runs included): no call starts when
    the spend, the calls in flight and one more, each at the dearest call so far, would pass it, so
    repeated runs cannot pass it either. workers > 1 keeps that many calls in flight (threads); the
    verdicts are the same, appended in the order they return."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    rubric, version, rsha = rubric_identity(rubric_path)
    done = {c["key"] for c in read_jsonl(cache_path)}
    before = spent(cache_path)
    calls, cost, failures = 0, 0.0, 0
    dearest = max((float(c.get("cost_usd", 0.0))
                   for p in (cache_path, cache_path + ".failures.jsonl") for c in read_jsonl(p)), default=0.0)
    stop_err = None
    with open(cache_path, "a", encoding="utf-8", newline="\n") as cache, \
            open(cache_path + ".failures.jsonl", "a", encoding="utf-8", newline="\n") as fail, \
            ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = {}

        def settle(fut):
            nonlocal calls, cost, failures, dearest, stop_err
            k, order, r = pending.pop(fut)
            stopped = False
            try:
                text, c = fut.result()
            except JudgeStopped as e:
                stop_err = stop_err or e
                if e.cost_usd <= 0.0:
                    return  # a free account-limit response has no billable call or verdict
                text, c, stopped = e.response, e.cost_usd, True
            except SystemExit as e:  # a transport without a billable-response receipt
                stop_err = stop_err or e
                return
            calls += 1
            cost += c
            dearest = max(dearest, c)
            ranking, why = (None, "") if stopped else parse_ranking(text, order)
            rec = {"key": k, "uri": r.get("uri", ""), "order": order, "judge_id": judge_id, "rubric_version": version,
                   "rubric_sha256": rsha, "response_sha256": sha256_bytes(text.encode("utf-8")), "cost_usd": c,
                   "created": now or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            if ranking is None:
                failed = dict(rec, response=text)
                if stopped:
                    failed["transport_stopped"] = True
                fail.write(json.dumps(failed, ensure_ascii=False) + "\n")
                fail.flush()
                failures += 1
                return
            cache.write(json.dumps(dict(rec, ranking=ranking, rationale=why), ensure_ascii=False) + "\n")
            cache.flush()
            done.add(k)

        for r, order in jobs(requests):
            k = key_of(r["prompt"], r["candidates"], order, judge_id, version)
            if k in done:
                continue
            # One call at a time until a call's cost is known (the cap reserves the dearest per call).
            while len(pending) >= (max(1, workers) if dearest > 0 else 1):
                for f in wait(list(pending), return_when=FIRST_COMPLETED).done:
                    settle(f)
            if stop_err is not None or calls + len(pending) >= max_calls or (
                    max_usd is not None and before + cost + dearest * (len(pending) + 1) > max_usd + 1e-9):
                break
            fut = pool.submit(transport, rubric, render(r["prompt"], r["candidates"], order))
            pending[fut] = (k, order, r)
        while pending:
            for f in wait(list(pending), return_when=FIRST_COMPLETED).done:
                settle(f)
    if stop_err is not None:
        print(json.dumps({"calls": calls, "cost_usd": round(cost, 4), "failures": failures,
                          "total_spent_usd": round(before + cost, 4)}))
        raise stop_err
    return {"calls": calls, "cost_usd": round(cost, 4), "failures": failures,
            "total_spent_usd": round(before + cost, 4)}


def stops(r, chosen, rejected):
    """{"chosen_stop", "rejected_stop"} from a sampled request (sample_candidates.py's sampler.runs:
    how each candidate's first run stopped: "eos", "length" or "turn"), or {} for a request without
    them. mtstudio's provenance_po gives a reply its end-of-text only when it stopped on one."""
    s = r.get("sampler") or {}
    if "runs" not in s or "candidate_runs" not in s:
        return {}
    return {"chosen_stop": s["runs"][s["candidate_runs"][chosen][0]]["stop"],
            "rejected_stop": s["runs"][s["candidate_runs"][rejected][0]]["stop"]}


def pairs(requests, cache_path, judge_id, rubric_path, out_path, per_prompt=1):
    """Best vs worst where both orders agree on them; with per_prompt=2 (four
    or more candidates) also second-best vs second-worst where both orders
    agree on those positions too. Each pair must be length-matched."""
    _, version, _ = rubric_identity(rubric_path)
    cache = {c["key"]: c for c in read_jsonl(cache_path)}
    n_pairs = disagree = unranked = long_short = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as g:
        for r in requests:
            n = len(r["candidates"])
            fwd = list(range(n))
            got = [cache.get(key_of(r["prompt"], r["candidates"], o, judge_id, version)) for o in (fwd, fwd[::-1])]
            if not all(got):
                unranked += 1
                continue
            a, b = got[0]["ranking"], got[1]["ranking"]
            if a[0] != b[0] or a[-1] != b[-1]:
                disagree += 1
                continue
            ev = {"evidence_kind": "judge", "judge_id": judge_id, "rubric_version": version,
                  "evidence_uri": f"{cache_path}#{got[0]['key']},{got[1]['key']}"}
            ranks = [(0, -1)]
            if per_prompt >= 2 and n >= 4 and a[1] == b[1] and a[-2] == b[-2]:
                ranks.append((1, -2))
            for hi, lo in ranks:
                best, worst = r["candidates"][a[hi]], r["candidates"][a[lo]]
                lb, lw = max(1, len(best.split())), max(1, len(worst.split()))
                if max(lb, lw) / min(lb, lw) > 1.5:
                    long_short += 1
                    continue
                rec = {"prompt": r["prompt"], "chosen": best, "rejected": worst, "uri": r.get("uri", ""),
                       "rank_pair": [hi, lo],
                       "chosen_evidence": dict(ev, rationale=got[0]["rationale"]),
                       "rejected_evidence": dict(ev, rationale=got[1]["rationale"])}
                rec.update(stops(r, a[hi], a[lo]))
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
    return {"pairs": n_pairs, "position_disagreements": disagree, "unranked": unranked,
            "length_mismatched": long_short, "out": out_path}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "run", "pairs"])
    ap.add_argument("--requests", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--judge-id", required=True)
    ap.add_argument("--claude-bin")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-usd", type=float, help="cap on the cache's total recorded spend (run needs it)")
    ap.add_argument("--pairs-per-prompt", type=int, default=1, choices=[1, 2])
    ap.add_argument("--workers", type=int, default=1, help="calls in flight at once (run)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reqs = read_jsonl(a.requests)
    if a.cmd == "plan":
        _, version, _ = rubric_identity(a.rubric)
        done = {c["key"] for c in read_jsonl(a.cache)}
        todo = sum(key_of(r["prompt"], r["candidates"], o, a.judge_id, version) not in done for r, o in jobs(reqs))
        print(json.dumps({"requests": len(reqs), "calls_needed": todo}))
    elif a.cmd == "run":
        if not a.claude_bin or a.max_calls is None or a.max_usd is None:
            ap.error("run needs --claude-bin, --max-calls and --max-usd")
        print(json.dumps(run(reqs, a.cache, a.judge_id, a.rubric, a.max_calls,
                             claude_cli_transport(a.claude_bin, a.judge_id), max_usd=a.max_usd,
                             workers=a.workers)))
    else:
        if not a.out:
            ap.error("pairs needs --out")
        print(json.dumps(pairs(reqs, a.cache, a.judge_id, a.rubric, a.out, per_prompt=a.pairs_per_prompt)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
