#!/usr/bin/env python3
"""Score chat models on Carry-12 (corpus/eval/carry12_v1.jsonl; tools/carry12_build.py).

    python tools/carry12_eval.py run --model tag=spec.json [--model ...] --out DIR
        [--scorer mtstudio|ir|uniform|none] [--mtstudio BIN]
        [--gen ember|mtstudio|none] [--ember-bin BIN | --chat-url URL] [--port 8094] [--spot 40]
        [--self-scope target|whole] [--no-near-self] [--max-new 24]
        [--limit N] [--every N] [--families recall,correction,...]
    python tools/carry12_eval.py report DIR          # re-aggregate a run's records

The long-carry companion of tools/carry5_eval.py, built on its internals (tokeniser, /chat servers,
log-prob backends, ties, intervals, the state-token template), which it imports. Experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md sections 2 and 7.

FORCED CHOICE (raw summed log-prob `acc`; per-token `acc_norm`; a tie at the top earns 1/(tied)) at the
target turn, in six history modes:
  gold       the item's canned history to the target;
  self       the model's own greedy replies to the earlier user turns (needs --gen);
  near       the item's event turns and question without the small talk, canned replies;
  near_self  the near rendering with the model's own replies (needs --gen; off with --no-near-self);
  twin       the statements and the question in one user turn;
  noctx      the question alone (chance by construction of the set).
twin -> near is the cost of splitting the facts into turns, near -> gold the cost of the distance
(HL0: gold - near < 0 over the context families), gold -> self the cost of the model's own replies.
A rendering longer than the model's context (T + 1 ids) is `over_ctx`: credit 0 in `acc` (as Carry-5),
left out of `acc_fit`, counted per mode. Nothing here assumes a T: models at T 512 or 1,024 are scored at
their own context.

GREEDY GENERATION through a /chat server (ember.cpp's or `mtstudio chat --kv-cache`), as the tiny-chat
post job does: with --gen ember a spot-check first generates --spot items (spread over the set) with both
engines and compares every reply (text, stop reason, history used); ember generates the run only when all
match, otherwise mtstudio generates the whole run, flagged (the engines are never mixed; DIR/TAG/
gen_engine.json records the decision). ember runs with EMBER_GEMM_TF32=0. --self-scope target plays each
conversation to its target turn (Stage L0), whole plays every turn. Replies are matched as Carry-5
(contains / exclusive / exact); the instruction family by its machine check (opener: the first word is
the required form; closer: the last words are). A /chat request the server refuses (HTTP 4xx/5xx) is
recorded as an empty reply with stop "server_error" and counted (gen.server_errors). PARITY GATE: every self and near_self reply is scored
again by the log-prob backend as a continuation; its argmax must equal the reply at every position.

CURVES: forced choice by distance (target minus the needed fact's turn, 1-13), target turn (10-14),
interference, distractor window, correction kind, family x distance; the self conversation by turn
(1-14): clean stop, empty, repeats, max-new stops, history dropped, words; the HL0 slope (item-level gold
accuracy, and gold minus near, on distance, ordinary least squares over the context families).

Resumable: generation appends one record per item (gen.jsonl), scoring runs in atomic chunks; rerunning
the same command continues. Writes summary.json, summary.md and items.jsonl per model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carry12_build as c12  # noqa: E402
import carry5_eval as ce  # noqa: E402
import chat_eval  # noqa: E402

SET = c12.OUT
MODES = ("gold", "self", "near", "near_self", "twin", "noctx")
GEN_MODES = MODES
PRIMARY_FAMILIES = ("recall", "correction", "topic", "load", "knowledge")
CONTEXT_FAMILIES = ("recall", "correction", "topic", "load")
# ember fe01b35 or later: EMBER_DETERMINISTIC=1 makes decode bit-identical run to run on one GPU model and build
# (ember df43463); earlier builds ignore it. CPU servers are deterministic either way.
EMBER_ENV = {"EMBER_GEMM_TF32": "0", "EMBER_DETERMINISTIC": "1"}
MAX_TURNS = 14


def write_atomic(path, text):
    ce.write_atomic(path, text)


# ---------------------------------------------------------------- per-turn state tokens (the S cells)
# carry5_eval's: (vocab size, k) from a GGUF that declares state tokens, and a rendered prompt's ids with the state
# block after every <|endoftext|> that closes an exchange (tools/state_tokens.py); one definition for both tools.
state_template = ce.state_template
prompt_ids = ce.prompt_ids


# ---------------------------------------------------------------- renderings
def gold_history(it):
    tt = it["target_turn"]
    return [(t["user"], t["gold"]) for t in it["turns"][:tt - 1]], it["turns"][tt - 1]["user"]


def near_history(it):
    return [(t["user"], t["gold"]) for t in it["near"][:-1]], it["near"][-1]["user"]


def verdicts(it, reply):
    """Carry-5's reply verdicts (contains / exact / exclusive / names_distractor); the instruction family's
    compliance by Carry-12's machine check."""
    v = ce.reply_verdicts(dict(it, check=None), reply)
    if it.get("check"):
        v["complies"] = c12.check_instruction(it["check"], reply)
    return v


# ---------------------------------------------------------------- generation
def generate(a, items, tok, spec_path, gguf, work):
    """work/gen.jsonl: per item the self conversation (to the target, or whole), the near conversation
    with own replies, and the target reply in gold / near / twin / noctx."""
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "gen.jsonl")
    done = {r["id"] for r in ce.read_jsonl(path)} if os.path.exists(path) else set()
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return
    srv = ce.ChatServer(a, spec_path, gguf)
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for it in todo:
                def one(history, user):
                    try:
                        r = srv.chat(history, user, a.max_new)
                    except urllib.error.HTTPError as e:
                        # e.g. `mtstudio chat` cannot encode a reply that ends inside a UTF-8 sequence
                        # (HTTP 400): recorded and counted, never a crash mid-run and never a reply.
                        return {"reply": "", "stop": "server_error", "history_used": len(history),
                                "dropped": False, "error": f"{e.code}: {e.read()[:300].decode('utf-8', 'replace')}"}
                    used = r.get("history_turns_used", len(history))
                    rec = {"reply": r["reply"].strip(), "stop": r["stop_reason"], "history_used": used,
                           "dropped": used < len(history)}
                    if "ids" in r:
                        rec["ids"] = r["ids"]
                        rec["stop_id"] = r.get("stop_id")
                        if not rec["dropped"] and r.get("prompt_ids") != prompt_ids(tok, ce.template(history, user)):
                            sys.exit(f"{it['id']}: the server's prompt ids differ from this tokeniser's encoding")
                    return rec

                def play(users):
                    hist, recs = [], []
                    for u in users:
                        rec = one(hist, u)
                        recs.append(rec)
                        hist.append((u, rec["reply"]))
                    return recs
                tt = it["target_turn"]
                upto = tt if a.self_scope == "target" else it["n_turns"]
                out = {"id": it["id"], "self_scope": a.self_scope,
                       "self": play([t["user"] for t in it["turns"][:upto]]),
                       "gold": one(*gold_history(it)), "near": one(*near_history(it)),
                       "twin": one([], it["twin"]), "noctx": one([], it["noctx"])}
                if not a.no_near_self:
                    out["near_self"] = play([t["user"] for t in it["near"]])
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        srv.close()


def gen_records(work):
    p = os.path.join(work, "gen.jsonl")
    return {r["id"]: r for r in ce.read_jsonl(p)} if os.path.exists(p) else {}


SPOT_FIELDS = ("reply", "stop", "history_used")


def replies_of(r):
    out = [(f"self{k + 1}", x) for k, x in enumerate(r["self"])]
    out += [(f"near_self{k + 1}", x) for k, x in enumerate(r.get("near_self", []))]
    return out + [(m, r[m]) for m in ("gold", "near", "twin", "noctx")]


def compare(ra, rb, ids):
    """Per item, every reply of the ember (ra) and mtstudio (rb) records on SPOT_FIELDS."""
    rows, same_items, same, total = [], 0, 0, 0
    for i in ids:
        x, y = ra.get(i), rb.get(i)
        if x is None or y is None:
            rows.append({"id": i, "missing": [n for n, r in (("ember", x), ("mtstudio", y)) if r is None]})
            continue
        yb = dict(replies_of(y))
        diff = []
        for name, p in replies_of(x):
            q = yb.get(name)
            total += 1
            if q is not None and all(p.get(k) == q.get(k) for k in SPOT_FIELDS):
                same += 1
            else:
                diff.append({"reply": name, "ember": {k: p.get(k) for k in SPOT_FIELDS},
                             "mtstudio": {k: (q or {}).get(k) for k in SPOT_FIELDS}})
        same_items += not diff and len(yb) == len(replies_of(x))
        if diff:
            rows.append({"id": i, "diff": diff[:3]})
    return {"items": len(ids), "items_identical": same_items, "replies": total, "replies_identical": same,
            "differing": rows[:10]}


def spot_items(items, n):
    if n <= 0 or not items:
        return []
    n = min(n, len(items))
    return [items[(k * len(items)) // n] for k in range(n)]


def choose_engine(a, items, tok, spec_path, gguf, work):
    """work/gen_engine.json: ember when its spot-check is N/N, else mtstudio for the whole run (flagged);
    --gen mtstudio: mtstudio. Decided once per model and output folder."""
    path = os.path.join(work, "gen_engine.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    d = {"requested": a.gen, "engine": a.gen, "flag": None, "ember_env": dict(EMBER_ENV) if a.gen == "ember" else {},
         "spot": None}
    if a.gen == "ember" and a.spot > 0 and not a.chat_url:
        spot = spot_items(items, a.spot)
        res = {}
        for eng in ("mtstudio", "ember"):
            b = copy.copy(a)
            b.gen, b.out = eng, os.path.join(work, "spot", eng)
            b.port = a.port + (1 if eng == "mtstudio" else 0)
            generate(b, spot, tok, spec_path, gguf, os.path.join(work, "spot", eng))
            res[eng] = gen_records(os.path.join(work, "spot", eng))
        cmp = compare(res["ember"], res["mtstudio"], [it["id"] for it in spot])
        cmp["pass"] = bool(spot) and cmp["items_identical"] == len(spot)
        d["spot"] = cmp
        if not cmp["pass"]:
            d["engine"] = "mtstudio"
            d["flag"] = (f"ember spot-check {cmp['items_identical']}/{len(spot)} items identical; generated "
                         f"with mtstudio throughout")
    elif a.gen == "ember" and a.spot <= 0:
        d["flag"] = "no spot-check (--spot 0): ember unchecked against mtstudio before the run"
    write_atomic(path, json.dumps(d, indent=1, ensure_ascii=False))
    return d


# ---------------------------------------------------------------- log-prob requests
def stop_ids(tok, r):
    return {"eos": [tok.eot], "turn": [r["stop_id"]] if r.get("stop_id") is not None else []}.get(r["stop"], [])


def requests(items, tok, gens):
    """[(key, prompt_ids, continuation_ids)]: four candidates per item and mode, plus the gate rows."""
    out = []
    for it in items:
        tt = it["target_turn"]
        cont = [tok.encode(it["frame"].format(c)) + [tok.eot] for c in it["candidates"]]
        prompts = {"gold": ce.template(*gold_history(it)), "near": ce.template(*near_history(it)),
                   "twin": ce.template([], it["twin"]), "noctx": ce.template([], it["noctx"])}
        g = gens.get(it["id"])
        if g:
            for mode, users, recs in (("self", [t["user"] for t in it["turns"]], g["self"]),
                                      ("near_self", [t["user"] for t in it["near"]], g.get("near_self"))):
                if not recs:
                    continue
                hist = [(u, r["reply"]) for u, r in zip(users, recs)]
                last = tt if mode == "self" else it["near_target"]
                if len(recs) >= last:
                    prompts[mode] = ce.template(hist[:last - 1], users[last - 1])
                for k, r in enumerate(recs):
                    if "ids" not in r or r["dropped"]:
                        continue
                    c = r["ids"] + stop_ids(tok, r)
                    if c:
                        out.append((f"{it['id']}|gate_{mode}|{k + 1}", prompt_ids(tok, ce.template(hist[:k], users[k])), c))
        for mode, p in prompts.items():
            pids = prompt_ids(tok, p)
            for c, ids in enumerate(cont):
                out.append((f"{it['id']}|{mode}|{c}", pids, ids))
    return out


# ---------------------------------------------------------------- per-item records
def item_records(items, gens, scores, tok):
    recs = []
    for it in items:
        r = {"id": it["id"], "family": it["family"], "slot": it["slot"], "distance": it["distance"],
             "interference": it["interference"], "target_turn": it["target_turn"], "n_turns": it["n_turns"],
             "fact_turn": it["fact_turn"], "near_target": it["near_target"],
             "distractor_window": it.get("distractor_window"), "correction_kind": it.get("correction_kind"),
             "fc": {}, "gen": {}}
        gi = it["gold_index"]
        dis = set(it["distractor_values"]) | ({it["old_value"]} if "old_value" in it else set())
        for mode in MODES:
            rows = [scores.get(f"{it['id']}|{mode}|{c}") for c in range(4)]
            if any(x is None for x in rows):
                continue
            if any(x["over_ctx"] for x in rows):
                r["fc"][mode] = {"over_ctx": True, "acc": 0.0, "acc_norm": 0.0}
                continue
            raw = [x["sum"] for x in rows]
            nrm = [x["sum"] / len(x["logprobs"]) for x in rows]
            acc, ch = ce.choose(raw, gi)
            accn, _ = ce.choose(nrm, gi)
            r["fc"][mode] = {"over_ctx": False, "acc": acc, "acc_norm": accn, "raw": raw, "norm": nrm,
                             "chose": it["candidates"][ch], "chose_distractor": it["candidates"][ch] in dis,
                             "chose_old": it["candidates"][ch] == it.get("old_value")}
        g = gens.get(it["id"])
        if g:
            tt = it["target_turn"]
            target = {"gold": g["gold"], "near": g["near"], "twin": g["twin"], "noctx": g["noctx"]}
            if len(g["self"]) >= tt:
                target["self"] = g["self"][tt - 1]
            if g.get("near_self"):
                target["near_self"] = g["near_self"][-1]
            for mode, rec in target.items():
                r["gen"][mode] = {"reply": rec["reply"], "stop": rec["stop"], "clean": ce.clean(rec),
                                  "repeats": chat_eval.repeats(rec["reply"]), "dropped": rec["dropped"],
                                  **verdicts(it, rec["reply"])}
            allrecs = list(g["self"]) + list(g.get("near_self") or []) + [g[m] for m in ("gold", "near", "twin", "noctx")]
            r["server_errors"] = sum(1 for x in allrecs if x["stop"] == "server_error")
            r["turns"] = [{"turn": k + 1, "clean": ce.clean(x), "repeats": chat_eval.repeats(x["reply"]),
                           "words": len(x["reply"].split()), "length": x["stop"] == "length",
                           "dropped": x["dropped"]} for k, x in enumerate(g["self"])]
            gate = []
            for mode, played in (("self", g["self"]), ("near_self", g.get("near_self") or [])):
                for k, x in enumerate(played):
                    s = scores.get(f"{it['id']}|gate_{mode}|{k + 1}")
                    if s is not None:
                        gate.append(not s["over_ctx"] and s["argmax"] == x["ids"] + stop_ids(tok, x))
            r["gate"] = gate
        recs.append(r)
    return recs


# ---------------------------------------------------------------- aggregation
BY = {"family": lambda r: r["family"], "distance": lambda r: r["distance"],
      "interference": lambda r: r["interference"], "target_turn": lambda r: r["target_turn"],
      "distractor_window": lambda r: r["distractor_window"], "correction_kind": lambda r: r["correction_kind"],
      "slot": lambda r: f"{r['family']}:{r['slot']}"}


def breakdown(recs, get, by):
    out = {}
    for r in recs:
        v, k = get(r), by(r)
        if v is None or k is None:
            continue
        out.setdefault(str(k), []).append(v)
    return {k: ce.rate(v) for k, v in sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0]))}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ols_slope(pts):
    """Least-squares slope of y on x, and n; None with fewer than 2 distinct x."""
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    if len({x for x, _ in pts}) < 2:
        return {"slope": None, "n": len(pts)}
    mx, my = mean([x for x, _ in pts]), mean([y for _, y in pts])
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return {"slope": round(sum((x - mx) * (y - my) for x, y in pts) / sxx, 5), "n": len(pts)}


def pools(recs, kind="acc"):
    """Mode means over the primary and the context families (over_ctx = 0 credit), and per family."""
    out = {"primary": {}, "context": {}, "family": {}}
    for mode in MODES:
        vals = {}
        for r in recs:
            if mode in r["fc"]:
                vals.setdefault(r["family"], []).append(float(r["fc"][mode][kind]))
        if not vals:
            continue
        for pool, fams in (("primary", PRIMARY_FAMILIES), ("context", CONTEXT_FAMILIES)):
            xs = [v for f in fams for v in vals.get(f, [])]
            out[pool][mode] = {"mean": round(sum(xs) / len(xs), 4) if xs else None, "n": len(xs)}
        out["family"][mode] = {f: {"mean": round(sum(v) / len(v), 4), "n": len(v)} for f, v in sorted(vals.items())}
    return out


def derived(p):
    """Y12 and the decomposition from the pools (forced choice, acc); None where a mode is missing."""
    def g(pool, mode):
        return (p[pool].get(mode) or {}).get("mean")

    def diff(x, y):
        return None if x is None or y is None else round(x - y, 4)
    tw, nc = g("primary", "twin"), g("context", "noctx")
    out = {"Y12_self": diff(g("primary", "self"), tw), "Y12_gold": diff(g("primary", "gold"), tw),
           "split_cost_gold": diff(g("context", "near"), g("context", "twin")),
           "distance_cost_gold": diff(g("context", "gold"), g("context", "near")),
           "split_cost_self": diff(g("context", "near_self"), g("context", "twin")),
           "distance_cost_self": diff(g("context", "self"), g("context", "near_self")),
           "own_replies_cost": diff(g("context", "self"), g("context", "gold")),
           "validity_gate": {"twin_primary": tw, "twin_primary_at_least_0.35": None if tw is None else tw >= 0.35,
                             "twin_minus_noctx_context": diff(g("context", "twin"), nc),
                             "twin_minus_noctx_at_least_0.10":
                                 None if diff(g("context", "twin"), nc) is None else
                                 diff(g("context", "twin"), nc) >= 0.10}}
    return out


def aggregate(recs):
    s = {"items": len(recs), "fc": {}, "gen": {}}
    for kind in ("acc", "acc_norm"):
        s["fc"][kind] = {}
        for mode in MODES:
            def get(r, mode=mode, kind=kind):
                return r["fc"][mode][kind] if mode in r["fc"] else None
            s["fc"][kind][mode] = {"all": ce.rate([get(r) for r in recs]),
                                   **{name: breakdown(recs, get, f) for name, f in BY.items()}}
    s["fc"]["acc_fit"] = {}
    for mode in MODES:
        def fit(r, mode=mode):
            x = r["fc"].get(mode)
            return None if not x or x["over_ctx"] else x["acc"]
        s["fc"]["acc_fit"][mode] = {"all": ce.rate([fit(r) for r in recs]),
                                    "family": breakdown(recs, fit, BY["family"])}
    s["fc"]["over_ctx"] = {m: sum(1 for r in recs if r["fc"].get(m, {}).get("over_ctx")) for m in MODES}
    s["fc"]["wrong_chose_distractor"] = {
        m: ce.rate([r["fc"][m]["chose_distractor"] for r in recs if m in r["fc"] and not r["fc"][m]["over_ctx"]
                    and r["fc"][m]["acc"] == 0 and (r["interference"] or r["family"] == "correction")]) for m in MODES}
    s["fc"]["correction_wrong_chose_old"] = {
        m: ce.rate([r["fc"][m]["chose_old"] for r in recs if r["family"] == "correction" and m in r["fc"]
                    and not r["fc"][m]["over_ctx"] and r["fc"][m]["acc"] == 0]) for m in MODES}
    ctx = [r for r in recs if r["family"] in CONTEXT_FAMILIES]
    s["fc"]["family_distance"] = {
        mode: {f: breakdown([r for r in ctx if r["family"] == f],
                            lambda r, mode=mode: r["fc"][mode]["acc"] if mode in r["fc"] else None, BY["distance"])
               for f in CONTEXT_FAMILIES} for mode in ("gold", "near", "self", "twin")}

    def acc(r, m):
        return r["fc"][m]["acc"] if m in r["fc"] else None
    s["hl0"] = {"slope_gold_on_distance": ols_slope([(r["distance"], acc(r, "gold")) for r in ctx]),
                "slope_gold_minus_near_on_distance": ols_slope(
                    [(r["distance"], None if acc(r, "gold") is None or acc(r, "near") is None
                      else acc(r, "gold") - acc(r, "near")) for r in ctx]),
                "slope_self_on_distance": ols_slope([(r["distance"], acc(r, "self")) for r in ctx])}
    s["pools"] = {"acc": pools(recs, "acc"), "acc_norm": pools(recs, "acc_norm")}
    s["derived"] = derived(s["pools"]["acc"])
    if any(r["gen"] for r in recs):
        for metric in ("contains", "exclusive", "exact", "complies", "clean", "repeats"):
            s["gen"][metric] = {}
            for mode in GEN_MODES:
                def get(r, mode=mode, metric=metric):
                    g = r["gen"].get(mode)
                    return g.get(metric) if g else None
                s["gen"][metric][mode] = {"all": ce.rate([get(r) for r in recs]),
                                          "family": breakdown(recs, get, BY["family"]),
                                          "distance": breakdown(recs, get, BY["distance"]),
                                          "target_turn": breakdown(recs, get, BY["target_turn"])}
        s["gen"]["dropped_history"] = {m: sum(1 for r in recs if r["gen"].get(m, {}).get("dropped")) for m in GEN_MODES}
        s["decay"] = {}
        for k in range(MAX_TURNS):
            ts = [r["turns"][k] for r in recs if len(r.get("turns", [])) > k]
            if not ts:
                continue
            s["decay"][str(k + 1)] = {"n": len(ts), "clean_stop": ce.rate([t["clean"] for t in ts])["mean"],
                                      "repeats": ce.rate([t["repeats"] for t in ts])["mean"],
                                      "stop_length": ce.rate([t["length"] for t in ts])["mean"],
                                      "empty": ce.rate([t["words"] == 0 for t in ts])["mean"],
                                      "history_dropped": ce.rate([t["dropped"] for t in ts])["mean"],
                                      "words": round(sum(t["words"] for t in ts) / len(ts), 2)}
        s["gen"]["server_errors"] = sum(r.get("server_errors", 0) for r in recs)
        gates = [g for r in recs for g in r.get("gate", [])]
        s["gate"] = {"replies": len(gates), "argmax_identical": sum(gates)}
    return s


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def markdown(summaries, meta):
    fams = ["recall", "correction", "topic", "load", "instruction", "knowledge"]
    L = ["# Carry-12 results", "",
         f"Set `{meta['set']}` (sha256 {meta['sha256'][:16]}), {meta['items']} items; scorer {meta['scorer']}, "
         f"generation {meta['gen']} (greedy, max {meta['max_new']} new tokens, self history to the "
         f"{meta['self_scope']}). Forced choice over 4 candidates (chance 0.25), acc = raw log-prob. Modes: gold "
         "(canned history), self (own replies), near (events without small talk), near_self, twin (one turn), "
         "noctx (question alone).", ""]
    for tag, s in summaries.items():
        L += [f"## {tag}", "", f"Generation engine: {s.get('gen_engine', {}).get('engine')}"
              + (f" (flag: {s['gen_engine']['flag']})" if s.get("gen_engine", {}).get("flag") else ""), "",
              "| mode | all | " + " | ".join(fams) + " | over ctx |", "|---|---:|" + "---:|" * (len(fams) + 1)]
        for mode in MODES:
            a = s["fc"]["acc"][mode]
            if not a["all"]["n"]:
                continue
            cells = [fmt(a["family"].get(f, {}).get("mean")) for f in fams]
            L.append(f"| {mode} | {fmt(a['all']['mean'])} | " + " | ".join(cells) + f" | {s['fc']['over_ctx'][mode]} |")
        d = s["derived"]
        L += ["", "Pools (acc): " + ", ".join(f"{k} {fmt(v) if not isinstance(v, dict) else ''}".strip()
                                              for k, v in d.items() if k != "validity_gate"),
              f"Validity gate: {json.dumps(d['validity_gate'])}",
              f"HL0 slopes: {json.dumps(s['hl0'])}", "",
              "By distance (context families; acc): ", "",
              "| distance | gold | near | self | twin | n |", "|---|---:|---:|---:|---:|---:|"]
        dist = s["fc"]["acc"]["gold"]["distance"]
        for k, v in dist.items():
            L.append(f"| {k} | {fmt(v['mean'])} | {fmt(s['fc']['acc']['near']['distance'].get(k, {}).get('mean'))} | "
                     f"{fmt(s['fc']['acc']['self']['distance'].get(k, {}).get('mean'))} | "
                     f"{fmt(s['fc']['acc']['twin']['distance'].get(k, {}).get('mean'))} | {v['n']} |")
        L += ["", "By target turn (acc):", "", "| target | gold | near | self | n |", "|---|---:|---:|---:|---:|"]
        for k, v in s["fc"]["acc"]["gold"]["target_turn"].items():
            L.append(f"| {k} | {fmt(v['mean'])} | {fmt(s['fc']['acc']['near']['target_turn'].get(k, {}).get('mean'))} | "
                     f"{fmt(s['fc']['acc']['self']['target_turn'].get(k, {}).get('mean'))} | {v['n']} |")
        if s.get("gen"):
            g = s["gen"]
            L += ["", "Greedy generation at the target:", "",
                  "| mode | contains | exclusive | complies (instruction) | clean stop | repeats |",
                  "|---|---:|---:|---:|---:|---:|"]
            for mode in GEN_MODES:
                if not g["contains"][mode]["all"]["n"]:
                    continue
                L.append(f"| {mode} | {fmt(g['contains'][mode]['all']['mean'])} | "
                         f"{fmt(g['exclusive'][mode]['all']['mean'])} | {fmt(g['complies'][mode]['all']['mean'])} | "
                         f"{fmt(g['clean'][mode]['all']['mean'])} | {fmt(g['repeats'][mode]['all']['mean'])} |")
            L += ["", "Self-history conversation by turn:", "",
                  "| turn | n | clean stop | empty | repeats | hit max-new | history dropped | words |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for k, v in s["decay"].items():
                L.append(f"| {k} | {v['n']} | {fmt(v['clean_stop'])} | {fmt(v['empty'])} | {fmt(v['repeats'])} | "
                         f"{fmt(v['stop_length'])} | {fmt(v['history_dropped'])} | {v['words']} |")
            L += ["", f"Parity gate: the scorer's argmax equals the server's greedy reply on "
                      f"{s['gate']['argmax_identical']}/{s['gate']['replies']} self and near_self replies.", ""]
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- commands
def load_items(a):
    items = ce.read_jsonl(a.set)
    if a.families:
        keep = set(a.families.split(","))
        items = [it for it in items if it["family"] in keep]
    if a.every > 1:
        items = items[::a.every]
    return items[:a.limit] if a.limit else items


def run_model(a, tag, spec_path, items):
    work = os.path.join(a.out, tag)
    os.makedirs(work, exist_ok=True)
    engine = {"engine": "none"}
    if spec_path == "uniform":
        tok, gguf, ctx = None, None, None
    else:
        chat_eval.check_rope_heads(spec_path)
        spec, gguf, vocab = ce.model_paths(spec_path)
        ctx = spec.get("data", {}).get("T")
        tok = ce.Tok(vocab)
        tok.state = ce.check_state_spec(spec_path, gguf)
    # Wall time by phase (the post-job cost measurement, PREREGISTRATION_DRAFT.md section 10): this call's
    # own work only, so a resumed run reports what it did after the resume.
    timing = {"spot_s": 0.0, "gen_s": 0.0, "score_s": 0.0}
    if tok and a.gen != "none":
        os.environ.update(EMBER_ENV)  # the servers inherit the environment
        t0 = time.time()
        engine = choose_engine(a, items, tok, spec_path, gguf, work)
        timing["spot_s"] = round(time.time() - t0, 1)
        if not a.no_generate:
            b = copy.copy(a)
            b.gen = engine["engine"]
            t0 = time.time()
            generate(b, items, tok, spec_path, gguf, work)
            timing["gen_s"] = round(time.time() - t0, 1)
    gens = gen_records(work)
    timing["replies"] = sum(len(replies_of(r)) + sum(1 for m in ("gold", "near", "twin", "noctx") if m in r)
                            for r in gens.values())
    if spec_path == "uniform":
        scores = {f"{it['id']}|{m}|{c}": {"over_ctx": False, "sum": 0.0, "logprobs": [0.0], "argmax": []}
                  for it in items for m in MODES for c in range(4)}
    elif a.scorer != "none":
        t0 = time.time()
        scores = ce.score_all(a, requests(items, tok, gens), spec_path, gguf, work)
        timing["score_s"] = round(time.time() - t0, 1)
    else:
        scores = {}
    recs = item_records(items, gens, scores, tok)
    with open(os.path.join(work, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = aggregate(recs)
    s["gen_engine"] = engine
    s["context"] = ctx
    s["timing"] = timing
    return s


def set_meta(a, items):
    return {"set": os.path.relpath(a.set, ROOT).replace(os.sep, "/"),
            "sha256": hashlib.sha256(open(a.set, "rb").read()).hexdigest(), "items": len(items),
            "scorer": a.scorer, "gen": a.gen, "max_new": a.max_new, "self_scope": a.self_scope,
            "near_self": not a.no_near_self, "spot": a.spot, "tool": "tools/carry12_eval.py"}


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    items = load_items(a)
    meta = set_meta(a, items)
    cfg = os.path.join(a.out, "config.json")
    conf = {**meta, "models": a.model, "families": a.families, "limit": a.limit, "every": a.every}
    if os.path.exists(cfg) and json.load(open(cfg, encoding="utf-8")) != conf:
        sys.exit(f"{cfg} records a different configuration; use a new --out")
    write_atomic(cfg, json.dumps(conf, indent=1))
    summaries = {}
    for m in a.model:
        tag, spec = m.split("=", 1)
        summaries[tag] = run_model(a, tag, spec, items)
    write_atomic(os.path.join(a.out, "summary.json"), json.dumps({"meta": meta, "models": summaries}, indent=1))
    md = markdown(summaries, meta)
    write_atomic(os.path.join(a.out, "summary.md"), md)
    print(md)
    return 0


def cmd_report(a):
    conf = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    for k in ("scorer", "gen", "max_new", "self_scope", "spot"):
        setattr(a, k, conf[k])
    a.no_near_self = not conf["near_self"]
    a.set = os.path.join(ROOT, conf["set"])
    a.model, a.families, a.limit, a.every = conf["models"], conf["families"], conf["limit"], conf["every"]
    a.no_generate = True  # re-aggregate what gen.jsonl holds
    if a.scorer == "mtstudio" and not a.mtstudio:
        sys.exit("report: pass --mtstudio for any chunk still unscored")
    return cmd_run(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--mtstudio", default="")
        p.add_argument("--scorer", default="mtstudio", choices=["mtstudio", "ir", "uniform", "none"])
        p.add_argument("--port", type=int, default=8094)
        p.add_argument("--no-kv-cache", action="store_true",
                       help="mtstudio chat without its KV cache (the default uses it for the llama family)")
        p.add_argument("--max-new", type=int, default=24)
        p.add_argument("--ember-bin", default="")
        p.add_argument("--chat-url", default="")
        p.add_argument("--gpu-layers", default=os.environ.get("EMBER_GPU_LAYERS", "0"),
                       help="ember's GPU layer count: 0 (CPU) by default")
        if name == "report":
            continue
        p.add_argument("--model", action="append", required=True,
                       help="tag=spec.json (tag=uniform: the chance baseline, no model)")
        p.add_argument("--set", default=SET)
        p.add_argument("--families", default="")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--every", type=int, default=1, help="score every N-th item only (a subset)")
        p.add_argument("--gen", default="ember", choices=["ember", "mtstudio", "none"])
        p.add_argument("--spot", type=int, default=40, help="ember vs mtstudio spot-check items (0: none)")
        p.add_argument("--self-scope", default="target", choices=["target", "whole"])
        p.add_argument("--no-near-self", action="store_true")
        p.add_argument("--no-generate", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.gen == "ember" and not a.ember_bin and not a.chat_url:
        ap.error("--gen ember needs --ember-bin or --chat-url")
    if a.cmd == "run" and a.scorer == "mtstudio" and not a.mtstudio and not all(m.endswith("=uniform") for m in a.model):
        ap.error("--scorer mtstudio needs --mtstudio")
    if a.cmd == "run" and a.gen != "none" and not a.mtstudio and (a.gen == "mtstudio" or a.spot > 0):
        ap.error("generation with mtstudio (or the ember spot-check) needs --mtstudio")
    return {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
