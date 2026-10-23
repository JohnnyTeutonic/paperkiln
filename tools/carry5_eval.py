#!/usr/bin/env python3
"""Score chat models on Carry-5 (corpus/eval/carry5_v1.jsonl; tools/carry5_build.py).

    python tools/carry5_eval.py run --model tag=spec.json [--model ...] --out DIR
        [--scorer mtstudio|ir|uniform|none] [--mtstudio BIN]
        [--gen ember|mtstudio|none] [--ember-bin BIN | --chat-url URL] [--port 8093]
        [--max-new 24] [--limit N] [--every N] [--families recall,correction,...]
    python tools/carry5_eval.py report DIR          # re-aggregate a run's checkpoints
    python tools/carry5_eval.py gate --mtstudio BIN --model tag=spec.json --out DIR [--n 40]

Two scorers, both deterministic:

LOG-PROB FORCED CHOICE. At the target turn each of the item's four candidate replies (the answer
frame filled with each candidate, then <|endoftext|>) is scored as the assistant reply after the
conversation so far, in the served chat template (<|user|>U<|assistant|>A<|endoftext|> per past
turn). Reported: `acc` (raw summed log-probability; the frame is shared, so the comparison is the
value's tokens in context) and `acc_norm` (mean log-probability per reply token); a tie at the top
earns 1/(number tied), so the `uniform` scorer is exactly chance (0.25). Four history modes:
  gold   teacher-forced: earlier assistant turns are the item's short canned replies;
  self   the model's own greedy replies to the earlier user turns (needs --gen);
  twin   the single-turn twin (the same statements and the question in one user turn);
  noctx  the question alone (the model's prior over the candidates; chance by construction of the
         set, so a noctx far from 0.25 means a prior the set's balance is cancelling).
gold minus self is the cost of the model's own replies; twin minus gold is the carry-over gap.
Backends: `mtstudio score` (tools/mtstudio.cpp: the spec's safetensors, the same forward pass as
`mtstudio chat`) or `ir` (tools/ir_numpy.py on the GGUF's IR: slower, independent of mtstudio).

GREEDY GENERATION. The model's argmax reply at the target turn through a /chat server (ember.cpp's
server or `mtstudio chat`; LOCAL_LLM_STACK.md section 3), in the same four modes, matched against
the accepted answers as tools/qa_eval.py does (word boundary, case-insensitive): `contains`, and
`exclusive` (contains the gold and no other candidate's answer). The instruction family is scored
by its machine check instead (one_word: one word; short: 1-5 words; yes_no: "yes" or "no" plus at
most one word; number: one number), with content correctness reported beside it. The self mode
plays all five turns, which gives the per-turn curves: clean-stop rate (end-of-text, no role
marker), repetition (tools/chat_eval.py repeats()), reply length and context overflow by turn.

PARITY GATE. Every self-mode reply is scored again by the log-prob backend as a continuation (its
ids plus its stop token): the backend's argmax must equal the reply at every position, so the
generation server and the scorer run the same model. `gate` checks the same against `mtstudio
chat` directly (greedy, no KV cache) on N gold-history target prompts.

Resumable: generation appends one record per item (gen.jsonl), scoring runs in chunks whose
outputs are written atomically; rerunning the same command continues. Writes summary.json,
summary.md and items.jsonl (per item, per mode: scores, choice, reply, verdicts) per model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bpe  # noqa: E402
import chat_eval  # noqa: E402
import qa_eval  # noqa: E402

SET = os.path.join(ROOT, "corpus", "eval", "carry5_v1.jsonl")
MODES = ("gold", "self", "twin", "noctx")
EOT, USER, ASSISTANT = "<|endoftext|>", "<|user|>", "<|assistant|>"
CHUNK = 256
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                   "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                   "hundred".split())


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def template(history, user):
    return "".join(f"{USER}{u}{ASSISTANT}{a}{EOT}" for u, a in history) + f"{USER}{user}{ASSISTANT}"


# ---------------------------------------------------------------- tokeniser
class Tok:
    """tools/bpe.py for paperkiln's byte-level BPE vocabularies; tools/llama3_bpe.py (HF tokenizers)
    for the 16k Llama 3-scheme vocabulary. Special tokens in text map to their ids."""

    def __init__(self, gguf):
        try:
            t = bpe.read_gguf(gguf)
            self.encode, self.decode = t.encode, t.decode
            self.ids = {s: i for i, s in enumerate(bpe.SPECIALS)}
        except Exception:
            import llama3_bpe
            t = llama3_bpe.from_gguf(gguf)
            self.encode = lambda s: t.encode(s).ids
            self.decode = lambda ids: t.decode(ids, skip_special_tokens=False)
            self.ids = {s: t.token_to_id(s) for s in ("<unk>", EOT, USER, ASSISTANT)}
        self.eot, self.user, self.assistant = self.ids[EOT], self.ids[USER], self.ids[ASSISTANT]


def model_paths(spec_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    gguf = os.path.join(spec["out_dir"], spec["name"] + ".gguf")
    vocab = gguf if os.path.exists(gguf) else spec["data"]["vocab"]
    return spec, gguf, vocab


# ---------------------------------------------------------------- reply checks
def words_of(text):
    return re.findall(r"[A-Za-z0-9']+", text)


def check_instruction(kind, reply):
    w = words_of(reply)
    if kind == "one_word":
        return len(w) == 1
    if kind == "short":
        return 1 <= len(w) <= 5
    if kind == "yes_no":
        return 1 <= len(w) <= 2 and w[0].lower() in ("yes", "no")
    if kind == "number":
        return len(w) == 1 and (w[0].isdigit() or w[0].lower() in NUMBER_WORDS)
    raise ValueError(kind)


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def reply_verdicts(it, reply):
    gi = it["gold_index"]
    v = {"contains": qa_eval.correct(reply, it["answers"])}
    v["exact"] = norm(reply) in {norm(it["frame"].format(it["candidates"][gi]))} | {norm(a) for a in it["answers"]}
    if it.get("candidate_answers"):
        others = [a for k, acc in enumerate(it["candidate_answers"]) if k != gi for a in acc]
        v["exclusive"] = v["contains"] and not qa_eval.correct(reply, others)
        v["names_distractor"] = qa_eval.correct(reply, [x.lower() for x in it["distractor_values"] +
                                                        ([it["old_value"]] if "old_value" in it else [])]) \
            if (it["distractor_values"] or "old_value" in it) else None
    if it.get("check"):
        v["complies"] = check_instruction(it["check"]["kind"], reply)
    return v


def clean(rec):
    """A non-empty reply that ended at end-of-text with no role marker inside."""
    return rec["stop"] == "eos" and bool(rec["reply"]) and USER not in rec["reply"] and ASSISTANT not in rec["reply"]


# ---------------------------------------------------------------- /chat servers
class ChatServer:
    """ember.cpp's server or `mtstudio chat` on one model, or an already running server (--chat-url)."""

    def __init__(self, a, spec_path, gguf):
        self.proc = None
        self.url = a.chat_url or f"http://127.0.0.1:{a.port}"
        if a.chat_url:
            return
        try:
            urllib.request.urlopen(self.url + "/health", timeout=2).read()
            sys.exit(f"port {a.port} already answers /health: another server is there; pick a free --port")
        except OSError:
            pass
        if a.gen == "ember":
            www = os.path.join(a.out, "www")
            os.makedirs(www, exist_ok=True)
            argv = [a.ember_bin, gguf, str(a.port), "127.0.0.1", www, str(a.gpu_layers)]
            want = os.path.realpath(gguf)
        else:
            argv = [a.mtstudio, "chat", spec_path, "--port", str(a.port)]
            # The KV cache (gated: logits within 2e-7 of the range vs recompute, greedy 40/40,
            # tests/test_kv_cache.cpp) for the llama family: without it every token recomputes the
            # whole conversation, and the floor pilot's post job spent over 2 h generating (24 Oct 2026).
            spec = json.load(open(spec_path, encoding="utf-8"))
            if not getattr(a, "no_kv_cache", False) and spec.get("arch", {}).get("preset", "").startswith("llama"):
                argv.append("--kv-cache")
            want = spec["name"]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        health = None
        for _ in range(300):
            try:
                health = json.loads(urllib.request.urlopen(self.url + "/health", timeout=2).read())
                break
            except OSError:
                if self.proc.poll() is not None:
                    sys.exit(f"{argv[0]} exited before serving")
                time.sleep(1)
        got = health and health.get("model", "")
        if not health or (os.path.realpath(got) if a.gen == "ember" else got) != want:
            self.close()
            sys.exit(f"/health names {got!r}, not {want!r}")
        self.context = health.get("context")

    def chat(self, history, user, max_new):
        body = {"user_input": user, "history": [{"user": u, "assistant": r} for u, r in history],
                "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 7, "max_new_tokens": max_new,
                "return_ids": True}
        req = urllib.request.Request(self.url + "/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def close(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


def generate(a, items, tok, spec_path, gguf, work):
    """gen.jsonl: per item the five self-history turns and the target reply in gold/twin/noctx."""
    path = os.path.join(work, "gen.jsonl")
    done = {r["id"] for r in read_jsonl(path)} if os.path.exists(path) else set()
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return
    srv = ChatServer(a, spec_path, gguf)
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for it in todo:
                def one(history, user):
                    r = srv.chat(history, user, a.max_new)
                    full = tok.encode(template(history, user))
                    used = r.get("history_turns_used", len(history))
                    rec = {"reply": r["reply"].strip(), "stop": r["stop_reason"], "history_used": used,
                           "dropped": used < len(history)}
                    if "ids" in r:
                        rec["ids"] = r["ids"]
                        rec["stop_id"] = r.get("stop_id")
                        if not rec["dropped"] and r.get("prompt_ids") != full:
                            sys.exit(f"{it['id']}: the server's prompt ids differ from this tokeniser's encoding")
                    return rec
                hist, selfrecs = [], []
                for t in it["turns"]:
                    rec = one(hist, t["user"])
                    selfrecs.append(rec)
                    hist.append((t["user"], rec["reply"]))
                tt = it["target_turn"]
                gold_hist = [(t["user"], t["gold"]) for t in it["turns"][:tt - 1]]
                out = {"id": it["id"], "self": selfrecs, "gold": one(gold_hist, it["turns"][tt - 1]["user"]),
                       "twin": one([], it["twin"]), "noctx": one([], it["noctx"])}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        srv.close()


# ---------------------------------------------------------------- log-prob requests and backends
def requests(items, tok, gens):
    """[(key, prompt_ids, continuation_ids)]: four candidates per item and mode, plus the gate rows."""
    out = []
    for it in items:
        tt = it["target_turn"]
        cont = [tok.encode(it["frame"].format(c)) + [tok.eot] for c in it["candidates"]]
        prompts = {"gold": template([(t["user"], t["gold"]) for t in it["turns"][:tt - 1]], it["turns"][tt - 1]["user"]),
                   "twin": template([], it["twin"]), "noctx": template([], it["noctx"])}
        g = gens.get(it["id"])
        if g:
            hist = [(t["user"], r["reply"]) for t, r in zip(it["turns"], g["self"])]
            prompts["self"] = template(hist[:tt - 1], it["turns"][tt - 1]["user"])
            for k, r in enumerate(g["self"]):
                if "ids" not in r or r["dropped"]:
                    continue
                stop = {"eos": [tok.eot], "turn": [r["stop_id"]] if r.get("stop_id") is not None else []}.get(
                    r["stop"], [])
                if r["ids"] + stop:
                    out.append((f"{it['id']}|gate|{k + 1}", tok.encode(template(hist[:k], it["turns"][k]["user"])),
                                r["ids"] + stop))
        for mode, p in prompts.items():
            pids = tok.encode(p)
            for c, ids in enumerate(cont):
                out.append((f"{it['id']}|{mode}|{c}", pids, ids))
    return out


def log_softmax_row(x):
    m = max(x)
    return m + math.log(sum(math.exp(v - m) for v in x))


class IrBackend:
    def __init__(self, gguf):
        import numpy as np
        import ir_check
        import ir_numpy
        self.np, self.ir = np, ir_numpy
        meta, tensors, data_start, raw, _ = ir_check.read_gguf(gguf)
        self.doc = json.loads(meta["paperkiln.ir"][1])
        if all(t[2] == 0 for t in tensors):
            self.W = {name: np.frombuffer(raw, dtype=np.float32, count=int(np.prod(dims)), offset=data_start + off)
                      .reshape(list(reversed(dims))) for name, dims, _, off in tensors}
        else:
            self.W = ir_numpy.load_weights(gguf)
        self.ctx = self.doc["model"]["ctx"]
        sp = self.doc["tokenizer"]["specials"]
        self.banned = [sp["unk"]] if "unk" in sp else []

    def score(self, p, c):
        np = self.np
        ids = p + c
        if len(ids) > self.ctx + 1:
            return {"over_ctx": True}
        lg = self.ir.forward(self.doc, self.W, ids[:-1]).astype(np.float64)
        rows = lg[len(p) - 1:]
        mx = rows.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(rows - mx).sum(axis=1))
        lp = [float(rows[j, c[j]] - lse[j]) for j in range(len(c))]
        rb = rows.copy()
        rb[:, self.banned] = -np.inf
        return {"over_ctx": False, "logprobs": lp, "argmax": [int(x) for x in rb.argmax(axis=1)], "sum": sum(lp)}


def score_all(a, reqs, spec_path, gguf, work):
    """{key: result}: chunks of CHUNK requests, each written atomically, skipped when already present."""
    res = {}
    if a.scorer == "uniform":
        return {k: {"over_ctx": False, "sum": 0.0, "logprobs": [0.0] * len(c), "argmax": list(c)} for k, _, c in reqs}
    os.makedirs(os.path.join(work, "score"), exist_ok=True)
    ir = None
    for n in range(0, len(reqs), CHUNK):
        part = reqs[n:n + CHUNK]
        sig = json.dumps([k for k, _, _ in part])
        outp = os.path.join(work, "score", f"chunk{n // CHUNK:04d}.json")
        if os.path.exists(outp):
            got = json.load(open(outp, encoding="utf-8"))
            if got["keys"] == sig:
                res.update(got["results"])
                continue
        if a.scorer == "mtstudio":
            inp = outp + ".in.jsonl"
            raw = outp + ".out.jsonl"
            with open(inp, "w", encoding="utf-8", newline="\n") as f:
                for k, p, c in part:
                    f.write(json.dumps({"key": k, "prompt_ids": p, "continuation_ids": c}) + "\n")
            subprocess.run([a.mtstudio, "score", spec_path, "--in", inp, "--out", raw], check=True,
                           capture_output=True)
            got = {r["key"]: {x: r[x] for x in ("over_ctx", "logprobs", "argmax", "sum") if x in r}
                   for r in read_jsonl(raw)}
            os.remove(inp)
            os.remove(raw)
        else:
            ir = ir or IrBackend(gguf)
            got = {k: ir.score(p, c) for k, p, c in part}
        if set(got) != {k for k, _, _ in part}:
            sys.exit(f"scorer returned {len(got)} of {len(part)} rows for chunk {n // CHUNK}")
        write_atomic(outp, json.dumps({"keys": sig, "results": got}))
        res.update(got)
        print(f"scored {min(n + CHUNK, len(reqs))}/{len(reqs)}", flush=True)
    return res


# ---------------------------------------------------------------- aggregation
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def choose(scores, gold):
    """Credit for the gold candidate (1/(tied) on a tie at the top) and the chosen index."""
    top = max(scores)
    tied = [k for k, s in enumerate(scores) if s == top]
    return (1.0 / len(tied) if gold in tied else 0.0), tied[0]


def item_records(items, gens, scores, tok):
    recs = []
    for it in items:
        r = {"id": it["id"], "family": it["family"], "slot": it["slot"], "distance": it["distance"],
             "interference": it["interference"], "target_turn": it["target_turn"], "fc": {}, "gen": {}}
        gi = it["gold_index"]
        for mode in MODES:
            rows = [scores.get(f"{it['id']}|{mode}|{c}") for c in range(4)]
            if any(x is None for x in rows):
                continue
            if any(x["over_ctx"] for x in rows):
                r["fc"][mode] = {"over_ctx": True, "acc": 0.0, "acc_norm": 0.0}
                continue
            raw = [x["sum"] for x in rows]
            nrm = [x["sum"] / len(x["logprobs"]) for x in rows]
            acc, ch = choose(raw, gi)
            accn, chn = choose(nrm, gi)
            dis = set(it["distractor_values"]) | ({it["old_value"]} if "old_value" in it else set())
            r["fc"][mode] = {"over_ctx": False, "acc": acc, "acc_norm": accn, "raw": raw, "norm": nrm,
                             "chose": it["candidates"][ch], "chose_distractor": it["candidates"][ch] in dis}
        g = gens.get(it["id"])
        if g:
            tt = it["target_turn"]
            for mode, rec in (("gold", g["gold"]), ("self", g["self"][tt - 1]), ("twin", g["twin"]),
                              ("noctx", g["noctx"])):
                r["gen"][mode] = {"reply": rec["reply"], "stop": rec["stop"], "clean": clean(rec),
                                  "repeats": chat_eval.repeats(rec["reply"]), "dropped": rec["dropped"],
                                  **reply_verdicts(it, rec["reply"])}
            r["turns"] = [{"clean": clean(x), "repeats": chat_eval.repeats(x["reply"]),
                           "words": len(x["reply"].split()), "length": x["stop"] == "length",
                           "dropped": x["dropped"]} for x in g["self"]]
            gate = []
            for k, x in enumerate(g["self"]):
                s = scores.get(f"{it['id']}|gate|{k + 1}")
                if s is None:
                    continue
                stop = {"eos": [tok.eot], "turn": [x["stop_id"]] if x.get("stop_id") is not None else []}.get(
                    x["stop"], [])
                gate.append(not s["over_ctx"] and s["argmax"] == x["ids"] + stop)
            r["gate"] = gate
        recs.append(r)
    return recs


def rate(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    k = sum(vals)
    return {"mean": round(k / n, 4) if n else None, "n": n, "ci95": wilson(k, n)}


def breakdown(recs, get, by):
    out = {}
    for r in recs:
        v = get(r)
        if v is None:
            continue
        out.setdefault(str(by(r)), []).append(v)
    return {k: rate(v) for k, v in sorted(out.items())}


def aggregate(recs):
    s = {"items": len(recs), "fc": {}, "gen": {}}
    for kind in ("acc", "acc_norm"):
        s["fc"][kind] = {}
        for mode in MODES:
            def get(r, mode=mode, kind=kind):
                return r["fc"][mode][kind] if mode in r["fc"] else None
            s["fc"][kind][mode] = {"all": rate([get(r) for r in recs]),
                                   "family": breakdown(recs, get, lambda r: r["family"]),
                                   "distance": breakdown(recs, get, lambda r: r["distance"]),
                                   "interference": breakdown(recs, get, lambda r: r["interference"]),
                                   "target_turn": breakdown(recs, get, lambda r: r["target_turn"]),
                                   "slot": breakdown(recs, get, lambda r: f"{r['family']}:{r['slot']}")}
    over = {m: sum(1 for r in recs if r["fc"].get(m, {}).get("over_ctx")) for m in MODES}
    s["fc"]["over_ctx"] = over
    # When wrong with distractors present: how often the choice was a distractor (or the old value).
    s["fc"]["wrong_chose_distractor"] = {
        m: rate([r["fc"][m]["chose_distractor"] for r in recs if m in r["fc"] and not r["fc"][m]["over_ctx"]
                 and r["fc"][m]["acc"] == 0 and (r["interference"] or r["family"] == "correction")]) for m in MODES}
    if any(r["gen"] for r in recs):
        for metric in ("contains", "exclusive", "exact", "complies", "clean", "repeats"):
            s["gen"][metric] = {}
            for mode in MODES:
                def get(r, mode=mode, metric=metric):
                    g = r["gen"].get(mode)
                    return g.get(metric) if g else None
                s["gen"][metric][mode] = {"all": rate([get(r) for r in recs]),
                                          "family": breakdown(recs, get, lambda r: r["family"]),
                                          "distance": breakdown(recs, get, lambda r: r["distance"]),
                                          "interference": breakdown(recs, get, lambda r: r["interference"]),
                                          "slot": breakdown(recs, get, lambda r: f"{r['family']}:{r['slot']}")}
        s["gen"]["dropped_history"] = {m: sum(1 for r in recs if r["gen"].get(m, {}).get("dropped")) for m in MODES}
        s["decay"] = {}
        for k in range(5):
            ts = [r["turns"][k] for r in recs if "turns" in r]
            s["decay"][str(k + 1)] = {"clean_stop": rate([t["clean"] for t in ts])["mean"],
                                      "repeats": rate([t["repeats"] for t in ts])["mean"],
                                      "stop_length": rate([t["length"] for t in ts])["mean"],
                                      "empty": rate([t["words"] == 0 for t in ts])["mean"],
                                      "history_dropped": rate([t["dropped"] for t in ts])["mean"],
                                      "words": round(sum(t["words"] for t in ts) / len(ts), 2) if ts else None}
        gates = [g for r in recs for g in r.get("gate", [])]
        s["gate"] = {"replies": len(gates), "argmax_identical": sum(gates)}
    s["gaps"] = {}
    for kind in ("acc", "acc_norm"):
        m = {mode: s["fc"][kind][mode]["all"]["mean"] for mode in MODES}
        if m["twin"] is not None and m["gold"] is not None:
            s["gaps"][f"{kind}_twin_minus_gold"] = round(m["twin"] - m["gold"], 4)
        if m["gold"] is not None and m["self"] is not None:
            s["gaps"][f"{kind}_gold_minus_self"] = round(m["gold"] - m["self"], 4)
    return s


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def markdown(summaries, meta):
    L = ["# Carry-5 results", "",
         f"Set `{meta['set']}` (sha256 {meta['sha256'][:16]}), {meta['items']} items; scorer {meta['scorer']}, "
         f"generation {meta['gen']} (greedy, max {meta['max_new']} new tokens). Forced choice over 4 candidates "
         "(chance 0.25): acc = raw log-prob, acc_norm = per token. Modes: gold = teacher-forced canned history, "
         "self = the model's own greedy history, twin = one-turn twin, noctx = the question alone.", ""]
    fams = ["recall", "correction", "instruction", "topic", "knowledge"]
    for tag, s in summaries.items():
        L += [f"## {tag}", "", "Forced choice (acc / acc_norm):", "",
              "| mode | all | " + " | ".join(fams) + " |", "|---|---:|" + "---:|" * len(fams)]
        for mode in MODES:
            a, n = s["fc"]["acc"][mode], s["fc"]["acc_norm"][mode]
            if not a["all"]["n"]:
                continue
            cells = [f"{fmt(a['family'].get(f, {}).get('mean'))} / {fmt(n['family'].get(f, {}).get('mean'))}"
                     for f in fams]
            L.append(f"| {mode} | {fmt(a['all']['mean'])} / {fmt(n['all']['mean'])} | " + " | ".join(cells) + " |")
        L += ["", "By distance (target turn minus fact turn; acc, gold / self / twin):", "",
              "| distance | gold | self | twin | n |", "|---|---:|---:|---:|---:|"]
        for d, v in s["fc"]["acc"]["gold"]["distance"].items():
            L.append(f"| {d} | {fmt(v['mean'])} | {fmt(s['fc']['acc']['self']['distance'].get(d, {}).get('mean'))} | "
                     f"{fmt(s['fc']['acc']['twin']['distance'].get(d, {}).get('mean'))} | {v['n']} |")
        L += ["", "By interference (distractor facts; acc, gold / self / twin):", "",
              "| distractors | gold | self | twin | n |", "|---|---:|---:|---:|---:|"]
        for d, v in s["fc"]["acc"]["gold"]["interference"].items():
            L.append(f"| {d} | {fmt(v['mean'])} | "
                     f"{fmt(s['fc']['acc']['self']['interference'].get(d, {}).get('mean'))} | "
                     f"{fmt(s['fc']['acc']['twin']['interference'].get(d, {}).get('mean'))} | {v['n']} |")
        wd = s["fc"]["wrong_chose_distractor"]["gold"]
        L += ["", f"Wrong with a distractor or old value present (gold history): the choice was a distractor "
                  f"{fmt(wd['mean'])} of {wd['n']} (1/3 to 2/3 by chance, by the number of distractors).",
              f"Gaps: {json.dumps(s['gaps'])}; over the context: {json.dumps(s['fc']['over_ctx'])}.", ""]
        if s.get("gen"):
            g = s["gen"]
            L += ["Greedy generation (contains / exclusive; instruction family: complies):", "",
                  "| mode | contains | exclusive | complies (instruction) | clean stop | repeats |",
                  "|---|---:|---:|---:|---:|---:|"]
            for mode in MODES:
                L.append(f"| {mode} | {fmt(g['contains'][mode]['all']['mean'])} | "
                         f"{fmt(g['exclusive'][mode]['all']['mean'])} | {fmt(g['complies'][mode]['all']['mean'])} | "
                         f"{fmt(g['clean'][mode]['all']['mean'])} | {fmt(g['repeats'][mode]['all']['mean'])} |")
            L += ["", "| family | contains gold | contains self | contains twin |", "|---|---:|---:|---:|"]
            for f in fams:
                L.append(f"| {f} | " + " | ".join(fmt(g["contains"][m]["family"].get(f, {}).get("mean"))
                                                  for m in ("gold", "self", "twin")) + " |")
            L += ["", "Self-history conversation by turn:", "",
                  "| turn | clean stop | empty | repeats | hit max-new | history dropped | words |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for k, v in s["decay"].items():
                L.append(f"| {k} | {fmt(v['clean_stop'])} | {fmt(v['empty'])} | {fmt(v['repeats'])} | "
                         f"{fmt(v['stop_length'])} | "
                         f"{fmt(v['history_dropped'])} | {v['words']} |")
            L += ["", f"Parity gate: the scorer's argmax equals the server's greedy reply on "
                      f"{s['gate']['argmax_identical']}/{s['gate']['replies']} self-history replies.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- commands
def load_items(a):
    items = read_jsonl(a.set)
    if a.families:
        keep = set(a.families.split(","))
        items = [it for it in items if it["family"] in keep]
    if getattr(a, "every", 1) > 1:  # every N-th item: a subset that keeps the families' mix
        items = items[::a.every]
    return items[:a.limit] if a.limit else items


def run_model(a, tag, spec_path, items):
    work = os.path.join(a.out, tag)
    os.makedirs(work, exist_ok=True)
    if spec_path == "uniform":
        tok, gguf = None, None
    else:
        chat_eval.check_rope_heads(spec_path)
        _, gguf, vocab = model_paths(spec_path)
        tok = Tok(vocab)
    if tok and a.gen != "none" and not getattr(a, "no_generate", False):
        generate(a, items, tok, spec_path, gguf, work)
    gp = os.path.join(work, "gen.jsonl")
    gens = {r["id"]: r for r in read_jsonl(gp)} if os.path.exists(gp) else {}
    if spec_path == "uniform":
        scores = {f"{it['id']}|{m}|{c}": {"over_ctx": False, "sum": 0.0, "logprobs": [0.0], "argmax": []}
                  for it in items for m in MODES for c in range(4)}
    elif a.scorer != "none":
        scores = score_all(a, requests(items, tok, gens), spec_path, gguf, work)
    else:
        scores = {}
    recs = item_records(items, gens, scores, tok)
    with open(os.path.join(work, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return aggregate(recs)


def set_meta(a, items):
    import hashlib
    return {"set": os.path.relpath(a.set, ROOT).replace(os.sep, "/"),
            "sha256": hashlib.sha256(open(a.set, "rb").read()).hexdigest(), "items": len(items),
            "scorer": a.scorer, "gen": a.gen, "max_new": a.max_new}


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
    write_atomic(os.path.join(a.out, "summary.md"), markdown(summaries, meta))
    print(markdown(summaries, meta))
    return 0


def cmd_report(a):
    conf = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    for k in ("scorer", "gen", "max_new"):
        setattr(a, k, conf[k])
    a.set = os.path.join(ROOT, conf["set"])
    a.model, a.families, a.limit, a.every = conf["models"], conf["families"], conf["limit"], conf.get("every", 1)
    a.no_generate = True  # the report re-aggregates what gen.jsonl holds
    if a.scorer == "mtstudio" and not a.mtstudio:
        sys.exit("report: pass --mtstudio for any chunk still unscored")
    return cmd_run(a)


def cmd_gate(a):
    """mtstudio chat's greedy reply (no KV cache) vs the log-prob backend's argmax, N target prompts."""
    tag, spec_path = a.model[0].split("=", 1)
    _, gguf, vocab = model_paths(spec_path)
    tok = Tok(vocab)
    items = load_items(a)[:a.n]
    os.makedirs(a.out, exist_ok=True)
    a.gen, a.chat_url = "mtstudio", ""
    srv = ChatServer(a, spec_path, gguf)
    rows = []
    try:
        for it in items:
            tt = it["target_turn"]
            hist = [(t["user"], t["gold"]) for t in it["turns"][:tt - 1]]
            r = srv.chat(hist, it["turns"][tt - 1]["user"], a.max_new)
            rows.append((it["id"], tok.encode(template(hist, it["turns"][tt - 1]["user"])), r))
    finally:
        srv.close()
    reqs, meta = [], []
    for iid, p, r in rows:
        stop = [tok.eot] if r["stop_reason"] == "eos" else []
        c = tok.encode(r["reply"]) + stop
        if c:
            reqs.append((f"{iid}|chatgate", p, c))
            meta.append((iid, r, c))
    scores = score_all(a, reqs, spec_path, gguf, os.path.join(a.out, tag + "_gate"))
    same = []
    for iid, r, c in meta:
        s = scores[f"{iid}|chatgate"]
        ok = not s["over_ctx"] and s["argmax"] == c
        same.append({"id": iid, "reply": r["reply"], "stop": r["stop_reason"], "argmax_identical": ok,
                     "first_diff": None if ok else next((j for j, (x, y) in enumerate(zip(s.get("argmax", []), c))
                                                         if x != y), None)})
    out = {"model": tag, "scorer": a.scorer, "prompts": len(same), "argmax_identical": sum(x["argmax_identical"]
                                                                                        for x in same), "rows": same}
    write_atomic(os.path.join(a.out, f"gate_{tag}.json"), json.dumps(out, indent=1, ensure_ascii=False))
    print(f"gate {tag} ({a.scorer} vs mtstudio chat greedy): {out['argmax_identical']}/{out['prompts']}")
    return 0 if out["argmax_identical"] == out["prompts"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--mtstudio", default="")
        p.add_argument("--scorer", default="mtstudio", choices=["mtstudio", "ir", "uniform", "none"])
        p.add_argument("--port", type=int, default=8093)
        p.add_argument("--no-kv-cache", action="store_true",
                       help="mtstudio chat without its KV cache (the default uses it for the llama family)")
        p.add_argument("--max-new", type=int, default=24)
        if name == "report":
            continue
        p.add_argument("--model", action="append", required=True,
                       help="tag=spec.json (tag=uniform: the chance baseline, no model)")
        p.add_argument("--set", default=SET)
        p.add_argument("--families", default="")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--every", type=int, default=1, help="score every N-th item only (a subset)")
        p.add_argument("--gen", default="ember", choices=["ember", "mtstudio", "none"])
        p.add_argument("--ember-bin", default="")
        p.add_argument("--chat-url", default="")
        p.add_argument("--gpu-layers", default=os.environ.get("EMBER_GPU_LAYERS", "0"),
                       help="ember's GPU layer count: 0 (CPU) by default")
        p.add_argument("--n", type=int, default=40, help="gate: prompts")
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.gen == "ember" and not a.ember_bin and not a.chat_url:
        ap.error("--gen ember needs --ember-bin or --chat-url")
    if a.scorer == "mtstudio" and a.cmd != "report" and not a.mtstudio and \
            not all(m.endswith("=uniform") for m in a.model):
        ap.error("--scorer mtstudio needs --mtstudio")
    return {"run": cmd_run, "report": cmd_report, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
