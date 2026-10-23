#!/usr/bin/env python3
"""H-MoR2's depth trace (experiments/tiny_chat_2026-10/PREREGISTRATION.md amendment 4, A4.5): the per-token
recursion depth of a Mixture-of-Recursions chat model (arch.custom.recursion schedule "mor") over Carry-5's
context families, teacher-forced with gold history, from `mtstudio score`'s per-position "depth".

    python tools/carry5_depth.py run --model spec.json --mtstudio BIN --out DIR
        [--set corpus/eval/carry5_v1.jsonl] [--families recall,correction,topic] [--limit N] [--every N]

The rendering is carry5_eval's, exactly: per item and mode, the prompt (gold: the item's earlier turns with
their canned gold replies, then the target user turn; twin: the single-turn twin; noctx: the question alone)
in the served chat template, tokenised whole, followed by the gold candidate's reply (the answer frame filled
with the gold value, tokenised on its own) and <|endoftext|>: the sequence carry5_eval scores for the gold
candidate. `mtstudio score` runs one forward pass over every id but the last and reports each context
position's depth (1 .. repeats; the router reads the token at that position).

Positions (position i predicts token i + 1, as MoR section 5.2 reads depth against the next token):
  answer positions          the positions whose next token is a token of the gold value inside the target
                            reply (a token whose characters overlap the value's span in the reply);
  other assistant positions every other position whose next token belongs to an assistant turn: the
                            tokens of each earlier canned reply and of the target reply, each turn's
                            closing <|endoftext|> included (the role markers <|user|> and <|assistant|> are
                            the template's, not a turn's text);
  all positions             every context position (H-MoR0's depth fractions).

Per mode: the pooled mean depth over answer positions and over other assistant positions and Delta, their
difference (H-MoR2a); the OLS slope of an item's mean answer-position depth on its fact distance (Carry-5's
`distance`, gold mode only: H-MoR2b); the depth histogram, fractions and normalised entropy H(f) / log R over
all positions (H-MoR0); per family and per distance. gold is registered; twin (distance 0) and noctx (the
question alone) are descriptive (A3.3's m3). Sequences past the context are skipped and counted.

Writes DIR/depth.json (the per-run summary) and DIR/items.jsonl (per item and mode: the answer-position
depths, the other assistant positions' count and sum, the all-position histogram). Scoring runs in chunks
written atomically (DIR/score/), so a rerun continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carry5_eval as ce  # noqa: E402

CONTEXT_FAMILIES = ("recall", "correction", "topic")  # A4.5: the 828 items H-MoR2 reads
MODES = ("gold", "twin", "noctx")
REGISTERED = "gold"
CHUNK = 256


# ---------------------------------------------------------------- rendering and positions
def gold_value(it):
    return it["candidates"][it["gold_index"]]


def history_of(it, mode):
    tt = it["target_turn"]
    if mode == "gold":
        return [(t["user"], t["gold"]) for t in it["turns"][:tt - 1]], it["turns"][tt - 1]["user"]
    return [], it[mode]


def encode_prompt(tok, history, user):
    """(ids, assistant flags): the template tokenised whole, as carry5_eval scores it, and for each id
    whether it belongs to an assistant turn (a canned reply's text or its closing <|endoftext|>). The
    flags come from tokenising the template piece by piece, which must give the same ids (the specials
    split the text), so the check below is what makes the flags those of the scored sequence."""
    ids, flag = [], []
    for u, a in history:
        for part, is_asst in (([tok.user], False), (tok.encode(u), False), ([tok.assistant], False),
                              (tok.encode(a), True), ([tok.eot], True)):
            ids += part
            flag += [is_asst] * len(part)
    for part in ([tok.user], tok.encode(user), [tok.assistant]):
        ids += part
        flag += [False] * len(part)
    whole = tok.encode(ce.template(history, user))
    if whole != ids:
        raise SystemExit("the template tokenised whole differs from its pieces: the assistant positions "
                         "cannot be located in the scored sequence")
    return ids, flag


def reply_tokens(tok, it):
    """(ids, answer flags) of the gold candidate's reply as carry5_eval scores it (frame filled with the
    gold value, tokenised on its own, then <|endoftext|>): a token is an answer token when its characters
    overlap the value's span in the reply."""
    frame, value = it["frame"], gold_value(it)
    reply = frame.format(value)
    start = frame.index("{}")
    end = start + len(value)
    if reply[start:end] != value:
        raise SystemExit(f"{it['id']}: the gold value is not where the frame puts it")
    ids = tok.encode(reply)
    if tok.decode(ids) != reply:
        raise SystemExit(f"{it['id']}: the reply does not decode back to itself; spans cannot be mapped")
    offs = [len(tok.decode(ids[:k])) for k in range(len(ids) + 1)]
    if any(b < a for a, b in zip(offs, offs[1:])) or offs[-1] != len(reply):
        raise SystemExit(f"{it['id']}: token offsets are not monotone over the reply")
    ans = [offs[k] < end and offs[k + 1] > start for k in range(len(ids))]
    if not any(ans):
        raise SystemExit(f"{it['id']}: no token of the reply covers the gold value {value!r}")
    return ids + [tok.eot], ans + [False]


def render(tok, it, mode):
    """{"prompt_ids", "continuation_ids", "answer_pos", "other_pos"}: the positions (indices into the
    context, ids[:-1]) whose next token is an answer token, and those whose next token is any other
    assistant-turn token."""
    hist, user = history_of(it, mode)
    p, pflag = encode_prompt(tok, hist, user)
    c, cans = reply_tokens(tok, it)
    asst = pflag + [True] * len(c)
    ans = [False] * len(p) + cans
    answer_pos = [j - 1 for j in range(1, len(asst)) if ans[j]]
    other_pos = [j - 1 for j in range(1, len(asst)) if asst[j] and not ans[j]]
    return {"prompt_ids": p, "continuation_ids": c, "answer_pos": answer_pos, "other_pos": other_pos}


# ---------------------------------------------------------------- statistics
def mean(xs):
    return sum(xs) / len(xs) if xs else None


def entropy_norm(hist):
    n = sum(hist)
    if n == 0 or len(hist) < 2:
        return None
    h = -sum(c / n * math.log(c / n) for c in hist if c > 0)
    return h / math.log(len(hist))


def ols(xs, ys):
    """(slope, intercept) of y on x, or (None, None) without spread in x."""
    if len(xs) < 2:
        return None, None
    xm, ym = mean(xs), mean(ys)
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    return b, ym - b * xm


def hist_of(depths, R):
    h = [0] * R
    for d in depths:
        if not 1 <= d <= R:
            raise SystemExit(f"depth {d} outside 1..{R}")
        h[d - 1] += 1
    return h


def summarise_mode(recs, mode, R):
    """The registered statistics over one mode's item records (items without that mode are skipped)."""
    rows = [r for r in recs if mode in r["modes"] and not r["modes"][mode].get("over_ctx")]
    ans = [d for r in rows for d in r["modes"][mode]["answer"]]
    oth_n = sum(r["modes"][mode]["other_n"] for r in rows)
    oth_s = sum(r["modes"][mode]["other_sum"] for r in rows)
    hist = [sum(r["modes"][mode]["hist"][k] for r in rows) for k in range(R)]
    tot = sum(hist)
    a_m, o_m = mean(ans), (oth_s / oth_n if oth_n else None)
    out = {"items": len(rows),
           "answer": {"n": len(ans), "mean": a_m, "hist": hist_of(ans, R)},
           "other_assistant": {"n": oth_n, "mean": o_m},
           "delta": a_m - o_m if a_m is not None and o_m is not None else None,
           "all_positions": {"n": tot, "hist": hist, "frac": [c / tot for c in hist] if tot else None,
                             "mean": sum((k + 1) * c for k, c in enumerate(hist)) / tot if tot else None,
                             "entropy_norm": entropy_norm(hist)}}
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    out["family"] = {}
    for f, rs in sorted(fam.items()):
        a = [d for r in rs for d in r["modes"][mode]["answer"]]
        on = sum(r["modes"][mode]["other_n"] for r in rs)
        os_ = sum(r["modes"][mode]["other_sum"] for r in rs)
        am, om = mean(a), (os_ / on if on else None)
        out["family"][f] = {"items": len(rs), "answer_mean": am, "other_assistant_mean": om,
                            "delta": am - om if am is not None and om is not None else None}
    if mode == "gold":
        xs = [r["distance"] for r in rows]
        ys = [mean(r["modes"][mode]["answer"]) for r in rows]
        b, a0 = ols(xs, ys)
        out["slope_answer_on_distance"] = {"slope": b, "intercept": a0, "items": len(xs),
                                           "distances": sorted(set(xs))}
        byd = {}
        for x, y in zip(xs, ys):
            byd.setdefault(str(x), []).append(y)
        out["distance"] = {k: {"items": len(v), "answer_item_mean": mean(v)} for k, v in sorted(byd.items())}
    return out


# ---------------------------------------------------------------- scoring
def score_depths(mtstudio, spec_path, reqs, work):
    """{key: {"over_ctx", "depth", "depth_margin_min"}} from `mtstudio score`, CHUNK requests per call,
    each chunk written atomically and skipped on a rerun."""
    os.makedirs(os.path.join(work, "score"), exist_ok=True)
    res = {}
    for n in range(0, len(reqs), CHUNK):
        part = reqs[n:n + CHUNK]
        sig = json.dumps([k for k, _, _ in part])
        outp = os.path.join(work, "score", f"chunk{n // CHUNK:04d}.json")
        if os.path.exists(outp):
            got = json.load(open(outp, encoding="utf-8"))
            if got["keys"] == sig:
                res.update(got["results"])
                continue
        inp, raw = outp + ".in.jsonl", outp + ".out.jsonl"
        with open(inp, "w", encoding="utf-8", newline="\n") as f:
            for k, p, c in part:
                f.write(json.dumps({"key": k, "prompt_ids": p, "continuation_ids": c}) + "\n")
        r = subprocess.run([mtstudio, "score", spec_path, "--in", inp, "--out", raw], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"mtstudio score failed (rc {r.returncode}): {r.stderr[-1500:]}")
        got = {}
        for x in ce.read_jsonl(raw):
            if x.get("over_ctx"):
                got[x["key"]] = {"over_ctx": True}
                continue
            if "depth" not in x:
                raise SystemExit("mtstudio score wrote no depth trace: the model is not schedule mor (or the "
                                 "binary predates the trace)")
            got[x["key"]] = {"over_ctx": False, "depth": x["depth"], "depth_margin_min": x.get("depth_margin_min")}
        os.remove(inp)
        os.remove(raw)
        if set(got) != {k for k, _, _ in part}:
            raise SystemExit(f"scorer returned {len(got)} of {len(part)} rows for chunk {n // CHUNK}")
        ce.write_atomic(outp, json.dumps({"keys": sig, "results": got}))
        res.update(got)
        print(f"depth-scored {min(n + CHUNK, len(reqs))}/{len(reqs)}", flush=True)
    return res


def item_record(it, rend, scores, R):
    r = {"id": it["id"], "family": it["family"], "distance": it["distance"], "modes": {}}
    for mode, x in rend.items():
        s = scores[f"{it['id']}|{mode}"]
        if s["over_ctx"]:
            r["modes"][mode] = {"over_ctx": True}
            continue
        d = s["depth"]
        if len(d) != len(x["prompt_ids"]) + len(x["continuation_ids"]) - 1:
            raise SystemExit(f"{it['id']}|{mode}: {len(d)} depths for a context of "
                             f"{len(x['prompt_ids']) + len(x['continuation_ids']) - 1} positions")
        oth = [d[i] for i in x["other_pos"]]
        r["modes"][mode] = {"answer": [d[i] for i in x["answer_pos"]], "other_n": len(oth), "other_sum": sum(oth),
                            "hist": hist_of(d, R), "margin_min": s.get("depth_margin_min")}
    return r


def analyse(recs, R):
    out = {"modes": {m: summarise_mode(recs, m, R) for m in MODES if any(m in r["modes"] for r in recs)}}
    g = out["modes"].get(REGISTERED)
    if g:
        out["registered"] = {"mode": REGISTERED, "delta": g["delta"],
                             "slope_answer_on_distance": g["slope_answer_on_distance"]["slope"],
                             "entropy_norm": g["all_positions"]["entropy_norm"],
                             "frac_all_positions": g["all_positions"]["frac"],
                             "answer_mean": g["answer"]["mean"], "other_assistant_mean": g["other_assistant"]["mean"],
                             "answer_n": g["answer"]["n"], "other_assistant_n": g["other_assistant"]["n"],
                             "items": g["items"]}
    margins = [m["margin_min"] for r in recs for m in r["modes"].values() if m.get("margin_min") is not None]
    out["depth_margin_min"] = min(margins) if margins else None
    out["over_ctx"] = {m: sum(1 for r in recs if r["modes"].get(m, {}).get("over_ctx")) for m in MODES}
    return out


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    spec, gguf, vocab = ce.model_paths(a.model)
    rec = (spec.get("arch", {}).get("custom") or {}).get("recursion") or {}
    if rec.get("schedule") != "mor":
        raise SystemExit(f"{a.model}: arch.custom.recursion schedule is {rec.get('schedule')!r}, not mor")
    R = int(rec["repeats"])
    tok = ce.Tok(vocab)
    items = ce.load_items(a)
    fams = a.families.split(",") if a.families else sorted({it["family"] for it in items})
    if any(it.get("distance") is None for it in items):
        raise SystemExit("an item without a fact distance (the knowledge family?): H-MoR2 reads the context families")
    rends = {it["id"]: {m: render(tok, it, m) for m in MODES} for it in items}
    reqs = [(f"{it['id']}|{m}", x["prompt_ids"], x["continuation_ids"]) for it in items
            for m, x in rends[it["id"]].items()]
    scores = score_depths(a.mtstudio, a.model, reqs, a.out)
    recs = [item_record(it, rends[it["id"]], scores, R) for it in items]
    res = analyse(recs, R)
    with open(a.set, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    out = {"tool": "tools/carry5_depth.py", "hypothesis": "H-MoR2 / H-MoR0 (PREREGISTRATION.md A4.5)",
           "model": spec.get("name"), "spec": a.model, "repeats": R, "recursion": rec,
           "set": os.path.relpath(a.set, ROOT).replace(os.sep, "/"), "set_sha256": sha, "families": fams,
           "items": len(items), "limit": a.limit, "every": a.every,
           "positions": "answer: next token inside the gold value in the target reply; other assistant: any other "
                        "next token of an assistant turn (canned replies, the target reply, each closing "
                        "<|endoftext|>); all: every context position",
           **res}
    with open(os.path.join(a.out, "items.jsonl.tmp"), "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(os.path.join(a.out, "items.jsonl.tmp"), os.path.join(a.out, "items.jsonl"))
    ce.write_atomic(os.path.join(a.out, "depth.json"), json.dumps(out, indent=1))
    reg = out.get("registered") or {}
    print(json.dumps({k: reg.get(k) for k in ("delta", "slope_answer_on_distance", "entropy_norm", "items")}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, help="the MoR model's spec.json (out_dir/name.safetensors)")
    p.add_argument("--mtstudio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", default=ce.SET)
    p.add_argument("--families", default=",".join(CONTEXT_FAMILIES))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--every", type=int, default=1)
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
