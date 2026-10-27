#!/usr/bin/env python3
"""Score chat models on IF-tiny v1 and instruction compliance by turn distance (tools/iftiny_build.py).

    python tools/iftiny_eval.py run --gen ember --ember-bin BIN --model tag=spec.json [--model ...] --out DIR
        [--port 8095] [--max-new 64] [--passes 2] [--mtstudio BIN]    (the parity check, below)
    python tools/iftiny_eval.py run --gen hf --hf tag=REPO@REVISION --out DIR      (third-party references)
    python tools/iftiny_eval.py report DIR [DIR ...] [--md OUT.md]

Generation is greedy (top_k 1) through a /chat server (ember's, contract point 4: system messages go in the
`system` field and the server folds them into the first user message, conversation schema v1), or HF
transformers' generate (do_sample False) with the model's own chat template and a native system turn. Every
distinct prompt is generated once per pass (the bare prompts repeat across families); the replies are checked by
tools/iftiny_checks.py.

--passes 2 replays every prompt in a fresh server process and counts identical replies (ember's CUDA decode is
not bit-reproducible: logits move by about 1e-5 run to run, so a near tie can flip a greedy token).
--mtstudio BIN re-scores every pass-1 reply that ended on a stop token with `mtstudio score` (the prompt ids the
server reports, the reply ids and the stop id): its argmax must equal the reply at every position (the same
gate as tools/carry5_eval.py's), so the server and paperkiln's own forward pass run the same model.

Outputs per model: gen_pass{N}.jsonl (append-only, resumable), summary.json (IF-tiny per family: compliance with
its bare base rate and Wilson 95% intervals, content accuracy where the item has an answer; turn distance per
condition and distance, per family), parity.json, items.jsonl.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec", "tools"))
sys.path.insert(0, HERE)
import iftiny_checks  # noqa: E402

IFTINY = os.path.join(ROOT, "corpus", "eval", "iftiny_v1.jsonl")
IFDIST = os.path.join(ROOT, "corpus", "eval", "ifdist_v1.jsonl")


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_atomic(path, text):
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / d
    return [max(0.0, round(c - h, 4)), min(1.0, round(c + h, 4))]


def pkey(system, history, user):
    return json.dumps([system, [list(h) for h in history], user], ensure_ascii=False)


def prompts(st, dist):
    """Every distinct (system, history, user) the two sets ask, in a fixed order."""
    seen, out = set(), []
    for it in st:
        for u in (it["prompt"], it["bare"]):
            k = pkey(None, [], u)
            if k not in seen:
                seen.add(k)
                out.append((k, None, [], u))
    for it in dist:
        k = pkey(it["system"], it["history"], it["user"])
        if k not in seen:
            seen.add(k)
            out.append((k, it["system"], it["history"], it["user"]))
    return out


# ---------------------------------------------------------------- generators
def post_json(url, body, timeout=600):
    """POST body as JSON; an HTTP error raises with the server's own message (ember's /chat returns
    {"error": ...}), which urllib's HTTPError leaves out: pk-bench2's IF-tiny job logged only "HTTP Error 500"
    on 26 Oct 2026 (the cause, ember's JSON refusing a reply cut inside a UTF-8 character, was in the body)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"{url}: HTTP {e.code}: {msg} (request {json.dumps(body)[:300]})") from None


class EmberGen:
    def __init__(self, a, spec_path, work):
        import carry5_eval
        _, gguf, _ = carry5_eval.model_paths(spec_path)
        ns = argparse.Namespace(chat_url=None, port=a.port, gen="ember", out=work, ember_bin=a.ember_bin,
                                gpu_layers=a.gpu_layers, no_kv_cache=False)
        self.srv = carry5_eval.ChatServer(ns, spec_path, gguf)
        self.health = json.loads(urllib.request.urlopen(self.srv.url + "/health", timeout=30).read())

    def __call__(self, system, history, user, max_new):
        body = {"user_input": user, "history": [{"user": u, "assistant": r} for u, r in history],
                "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 7, "max_new_tokens": max_new,
                "return_ids": True}
        if system:
            body["system"] = system
        x = post_json(self.srv.url + "/chat", body)
        return {"reply": x["reply"], "stop": x["stop_reason"], "ids": x.get("ids"), "stop_id": x.get("stop_id"),
                "prompt_ids": x.get("prompt_ids"), "history_used": x.get("history_turns_used", len(history))}

    def close(self):
        self.srv.close()


class HfGen:
    def __init__(self, a, repo_rev, work):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo, rev = repo_rev.split("@", 1)
        self.torch = torch
        self.t = AutoTokenizer.from_pretrained(repo, revision=rev)
        self.m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32).to(a.device).eval()
        self.dev = a.device
        self.health = {"engine": "hf", "repo": repo, "revision": rev, "torch": torch.__version__,
                       "transformers": __import__("transformers").__version__, "dtype": "float32",
                       "system": "native system turn of the model's chat template"}

    def __call__(self, system, history, user, max_new):
        msgs = [{"role": "system", "content": system}] if system else []
        for u, r in history:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        msgs.append({"role": "user", "content": user})
        ids = self.t.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.m.generate(ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
                                  pad_token_id=self.t.pad_token_id if self.t.pad_token_id is not None
                                  else self.t.eos_token_id)
        new = out[0, ids.shape[1]:].tolist()
        e = self.m.generation_config.eos_token_id
        e = e if isinstance(e, list) else [e]
        eos = {x for x in e + [self.t.eos_token_id] if x is not None}
        stop = "eos" if new and new[-1] in eos else "length"
        body = new[:-1] if stop == "eos" else new
        return {"reply": self.t.decode(body, skip_special_tokens=True), "stop": stop, "ids": body,
                "stop_id": new[-1] if stop == "eos" else None, "prompt_ids": ids[0].tolist(),
                "history_used": len(history)}

    def close(self):
        pass


def generate(a, kind, src, work, plist, path, label):
    done = {r["key"] for r in read_jsonl(path)}
    todo = [p for p in plist if p[0] not in done]
    if not todo:
        return
    gen = EmberGen(a, src, work) if kind == "pk" else HfGen(a, src, work)
    write_atomic(os.path.join(work, f"engine_{label}.json"), json.dumps(gen.health, indent=1))
    t0 = time.time()
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for n, (k, system, history, user) in enumerate(todo):
                r = gen(system, history, user, a.max_new)
                r["key"] = k
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if (n + 1) % 200 == 0:
                    print(f"{label}: {n + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        gen.close()


def parity(a, spec_path, work, gens):
    """mtstudio score's argmax against every reply that ended on a stop token."""
    reqs = []
    for k, g in gens.items():
        if g.get("ids") is None or g["stop"] not in ("eos", "turn") or g.get("stop_id") is None:
            continue
        if g["history_used"] < len(json.loads(k)[1]):
            continue
        reqs.append((k, g["prompt_ids"], g["ids"] + [g["stop_id"]]))
    inp, raw = os.path.join(work, "parity.in.jsonl"), os.path.join(work, "parity.out.jsonl")
    with open(inp, "w", encoding="utf-8", newline="\n") as f:
        for n, (k, p, c) in enumerate(reqs):
            f.write(json.dumps({"key": str(n), "prompt_ids": p, "continuation_ids": c}) + "\n")
    r = subprocess.run([a.mtstudio, "score", spec_path, "--in", inp, "--out", raw], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-800:]}
    got = {x["key"]: x for x in read_jsonl(raw)}
    eq = pos = pos_eq = 0
    bad = []
    for n, (k, p, c) in enumerate(reqs):
        x = got[str(n)]
        if x.get("over_ctx"):
            continue
        same = [int(u == v) for u, v in zip(x["argmax"], c)]
        pos += len(c)
        pos_eq += sum(same)
        if all(same):
            eq += 1
        elif len(bad) < 10:
            bad.append({"prompt": json.loads(k)[2][:80], "first_diff": same.index(0)})
    os.remove(inp)
    os.remove(raw)
    return {"replies_checked": len(reqs), "replies_equal": eq, "positions": pos, "positions_equal": pos_eq,
            "examples": bad}


def check_prompts(spec_path, gens):
    """The server's prompt ids against this side's rendering (tools/chat_template.py, schema v1: the system
    message folded into the first user message) for every reply whose history was not cut: a server that
    ignored the `system` field (ember before the contract's system message) is caught here."""
    import carry5_eval
    import chat_template
    _, _, vocab = carry5_eval.model_paths(spec_path)
    tok = carry5_eval.Tok(vocab)
    n = bad = sysn = 0
    ex = []
    for k, g in gens.items():
        system, history, user = json.loads(k)
        if g.get("prompt_ids") is None or g["history_used"] < len(history):
            continue
        want = tok.encode(chat_template.prompt([tuple(h) for h in history], user, system=system))
        n += 1
        sysn += bool(system)
        if want != g["prompt_ids"]:
            bad += 1
            if len(ex) < 5:
                ex.append({"system": bool(system), "user": user[:60], "server": len(g["prompt_ids"]),
                           "expected": len(want)})
    return {"checked": n, "with_system": sysn, "mismatched": bad, "examples": ex}


# ---------------------------------------------------------------- scoring
def rate(vals):
    n = len(vals)
    k = sum(vals)
    return {"n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def contains(reply, answers):
    import qa_eval
    return qa_eval.correct(reply, answers)


def score(st, dist, gens):
    recs = []
    fam = {}
    for it in st:
        g, b = gens[pkey(None, [], it["prompt"])], gens[pkey(None, [], it["bare"])]
        c, cb = iftiny_checks.check(it["check"], g["reply"]), iftiny_checks.check(it["check"], b["reply"])
        r = {"set": "iftiny", "id": it["id"], "family": it["family"], "comply": c, "bare_comply": cb,
             "reply": g["reply"], "stop": g["stop"], "bare_reply": b["reply"]}
        if it.get("answer"):
            r["correct"] = contains(g["reply"], it["answer"])
            r["bare_correct"] = contains(b["reply"], it["answer"])
        recs.append(r)
        fam.setdefault(it["family"], []).append(r)
    st_sum = {}
    for f, rs in fam.items():
        d = {"comply": rate([r["comply"] for r in rs]), "bare": rate([r["bare_comply"] for r in rs])}
        d["delta"] = round(d["comply"]["rate"] - d["bare"]["rate"], 4)
        if "correct" in rs[0]:
            d["correct"] = rate([r["correct"] for r in rs])
            d["bare_correct"] = rate([r["bare_correct"] for r in rs])
        d["length_stops"] = sum(r["stop"] == "length" for r in rs)
        st_sum[f] = d
    allr = [r for r in recs if r["set"] == "iftiny"]
    overall = {"micro": rate([r["comply"] for r in allr]), "micro_bare": rate([r["bare_comply"] for r in allr]),
               "macro": round(sum(v["comply"]["rate"] for v in st_sum.values()) / len(st_sum), 4),
               "macro_bare": round(sum(v["bare"]["rate"] for v in st_sum.values()) / len(st_sum), 4)}
    cells = {}
    for it in dist:
        g = gens[pkey(it["system"], it["history"], it["user"])]
        c = iftiny_checks.check(it["check"], g["reply"])
        r = {"set": "ifdist", "id": it["id"], "family": it["family"], "distance": it["distance"],
             "condition": it["condition"], "comply": c, "reply": g["reply"], "stop": g["stop"],
             "dropped": g["history_used"] < len(it["history"])}
        recs.append(r)
        cells.setdefault((it["condition"], it["distance"]), []).append(r)
    dsum = {}
    for (cond, d), rs in sorted(cells.items()):
        x = rate([r["comply"] for r in rs])
        x["dropped_turns"] = sum(r["dropped"] for r in rs)
        x["per_family"] = {}
        for r in rs:
            x["per_family"].setdefault(r["family"], []).append(r["comply"])
        x["per_family"] = {f: round(sum(v) / len(v), 4) for f, v in sorted(x["per_family"].items())}
        dsum.setdefault(cond, {})[f"d{d}"] = x
    return {"iftiny": {"overall": overall, "families": st_sum}, "ifdist": dsum}, recs


def cmd_run(a):
    st, dist = read_jsonl(a.iftiny), read_jsonl(a.ifdist)
    plist = prompts(st, dist)
    if a.limit:
        keep = {p[0] for p in plist[:a.limit]}
        st = [it for it in st if pkey(None, [], it["prompt"]) in keep and pkey(None, [], it["bare"]) in keep]
        dist = [it for it in dist if pkey(it["system"], it["history"], it["user"]) in keep]
        plist = plist[:a.limit]
    models = [("pk", m) for m in a.model or []] + [("hf", m) for m in a.hf or []]
    for kind, m in models:
        tag, src = m.split("=", 1)
        work = os.path.join(a.out, tag)
        os.makedirs(work, exist_ok=True)
        for p in range(1, a.passes + 1):
            generate(a, kind, src, work, plist, os.path.join(work, f"gen_pass{p}.jsonl"), f"{tag} pass {p}")
        gens = {r["key"]: r for r in read_jsonl(os.path.join(work, "gen_pass1.jsonl"))}
        summary, recs = score(st, dist, gens)
        summary["meta"] = {"tag": tag, "source": src, "gen": a.gen, "max_new": a.max_new, "prompts": len(plist),
                           "iftiny_sha256": sha256(a.iftiny), "ifdist_sha256": sha256(a.ifdist),
                           "engine": json.load(open(os.path.join(work, "engine_" + f"{tag} pass 1.json"),
                                                    encoding="utf-8"))
                           if os.path.exists(os.path.join(work, f"engine_{tag} pass 1.json")) else None}
        summary["stops"] = {s: sum(g["stop"] == s for g in gens.values()) for s in ("eos", "turn", "length")}
        if a.passes > 1:
            g2 = {r["key"]: r for r in read_jsonl(os.path.join(work, "gen_pass2.jsonl"))}
            same = sum(g2[k]["reply"] == gens[k]["reply"] for k in gens if k in g2)
            same_ids = sum(g2[k].get("ids") == gens[k].get("ids") for k in gens if k in g2)
            diff = [k for k in gens if k in g2 and g2[k]["reply"] != gens[k]["reply"]]
            flips = 0
            for it in st:
                for u in (it["prompt"], it["bare"]):
                    k = pkey(None, [], u)
                    if k in diff and iftiny_checks.check(it["check"], gens[k]["reply"]) != \
                            iftiny_checks.check(it["check"], g2[k]["reply"]):
                        flips += 1
            summary["replay"] = {"prompts": len(g2), "replies_identical": same, "ids_identical": same_ids,
                                 "iftiny_verdict_flips": flips,
                                 "examples": [{"prompt": json.loads(k)[2][:80], "pass1": gens[k]["reply"][:80],
                                               "pass2": g2[k]["reply"][:80]} for k in diff[:8]]}
        if kind == "pk":
            summary["prompt_check"] = check_prompts(src, gens)
        if a.mtstudio and kind == "pk":
            summary["parity"] = parity(a, src, work, gens)
        write_atomic(os.path.join(work, "summary.json"), json.dumps(summary, indent=1, ensure_ascii=False))
        with gzip.open(os.path.join(work, "items.jsonl.gz"), "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        o = summary["iftiny"]["overall"]
        print(json.dumps({"tag": tag, "iftiny_macro": o["macro"], "bare_macro": o["macro_bare"],
                          "dist_instr_user": {d: v["rate"] for d, v in summary["ifdist"].get("instr_user", {}).items()},
                          "replay": summary.get("replay", {}).get("replies_identical"),
                          "parity": summary.get("parity", {}).get("replies_equal")}), flush=True)


def sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def cmd_report(a):
    rows = []
    for d in a.dirs:
        for tag in sorted(os.listdir(d)):
            p = os.path.join(d, tag, "summary.json")
            if os.path.exists(p):
                rows.append((tag, json.load(open(p, encoding="utf-8"))))
    fams = sorted({f for _, r in rows for f in r["iftiny"]["families"]})
    L = ["IF-tiny v1: compliance (bare base rate in brackets), macro mean over 15 families", "",
         "| model | macro | macro bare | micro [95% CI] | " + " | ".join(fams) + " |",
         "|---|---|---|---|" + "---|" * len(fams)]
    for tag, r in rows:
        o = r["iftiny"]["overall"]
        lo, hi = o["micro"]["ci95"]
        L.append(f"| {tag} | {o['macro']:.3f} | {o['macro_bare']:.3f} | {o['micro']['rate']:.3f} [{lo:.3f}, {hi:.3f}] | "
                 + " | ".join(f"{r['iftiny']['families'][f]['comply']['rate']:.2f} ({r['iftiny']['families'][f]['bare']['rate']:.2f})"
                              if f in r["iftiny"]["families"] else "-" for f in fams) + " |")
    L += ["", "Compliance by turn distance d (instruction at turn 1, checked at turn 1 + d; 84 conversations per "
          "cell: 7 families x 12 sequences)", ""]
    conds = ("instr_user", "instr_system", "nohint", "bare")
    ds = ("d0", "d1", "d2", "d4")
    L.append("| model | condition | " + " | ".join(ds) + " |")
    L.append("|---|---|" + "---|" * len(ds))
    for tag, r in rows:
        for c in conds:
            if c not in r["ifdist"]:
                continue
            cells = []
            for d in ds:
                x = r["ifdist"][c].get(d)
                cells.append(f"{x['rate']:.3f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]" if x else "-")
            L.append(f"| {tag} | {c} | " + " | ".join(cells) + " |")
    L += ["", "| model | replay identical | verdict flips | parity (replies equal / checked) | stops eos/turn/length |",
          "|---|---|---|---|---|"]
    bad = [(t, r["prompt_check"]) for t, r in rows if r.get("prompt_check", {}).get("mismatched")]
    if bad:
        L.insert(0, "WARNING: server prompt ids differ from the schema v1 rendering: " +
                 ", ".join(f"{t} {c['mismatched']}/{c['checked']}" for t, c in bad) + "\n")
    for tag, r in rows:
        rp, pa, s = r.get("replay"), r.get("parity"), r.get("stops", {})
        L.append(f"| {tag} | {rp['replies_identical']}/{rp['prompts'] if rp else '-'} | "
                 f"{rp['iftiny_verdict_flips'] if rp else '-'} | "
                 f"{(str(pa.get('replies_equal')) + '/' + str(pa.get('replies_checked'))) if pa else '-'} | "
                 f"{s.get('eos')}/{s.get('turn')}/{s.get('length')} |" if rp else
                 f"| {tag} | - | - | {(str(pa.get('replies_equal')) + '/' + str(pa.get('replies_checked'))) if pa else '-'} | "
                 f"{s.get('eos')}/{s.get('turn')}/{s.get('length')} |")
    text = "\n".join(L) + "\n"
    if a.md:
        write_atomic(a.md, text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gen", choices=("ember", "hf"), required=True)
    r.add_argument("--model", action="append", help="tag=spec.json (paperkiln; ember serves its GGUF)")
    r.add_argument("--hf", action="append", help="tag=REPO@REVISION")
    r.add_argument("--ember-bin")
    r.add_argument("--port", type=int, default=8095)
    r.add_argument("--gpu-layers", type=int, default=-1)
    r.add_argument("--device", default="cuda")
    r.add_argument("--max-new", type=int, default=64)
    r.add_argument("--passes", type=int, default=1)
    r.add_argument("--mtstudio")
    r.add_argument("--out", required=True)
    r.add_argument("--iftiny", default=IFTINY)
    r.add_argument("--ifdist", default=IFDIST)
    r.add_argument("--limit", type=int, default=0, help="the first N distinct prompts only (a smoke test)")
    p = sub.add_parser("report")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--md")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
