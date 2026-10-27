#!/usr/bin/env python3
"""pk-bench2 at home: HellaSwag from each first-campaign run's own chunks, the receipts, and ember's POST /score
against hfgguf and mtstudio, per item.

    python bench2.py hellaswag     receipts/bench/<tag>/: HellaSwag re-summarised with one request per key
    python bench2.py collect       receipts/bench/<pick>/ and receipts/ember_score/<tag>/ from runs2/
    python bench2.py ember         receipts/ember_score/compare_*.json (per item)
    python bench2.py table         the RESULTS.md rows from the receipts

HellaSwag validation's `ind` names two items 433 times. Before paperkiln fafc4750, `tools/bench_eval.py` stored
both under one key, so the merged results credited each first item with the second's sums. Every first item sits
in a different 2,048-request chunk from its second, so its own scores are on disk: `rekey` reads each chunk on its
own and gives each request the key `bench_eval.unique_ids` gives it now (the n-th item under an id, n >= 2,
becomes id#n). The request order does not depend on the keys, so position n of chunk c is the same request in
both keyings. `bench_eval.summarise` then aggregates HellaSwag as a run at fafc4750 would.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PK, "tools"))
import bench_eval as be  # noqa: E402

ITEMS = "C:/ml_artifacts/benchmarks/bench_v1.jsonl"
RUNS = "C:/ml_artifacts/benchmarks/runs"
RUNS2 = "C:/ml_artifacts/benchmarks/runs2"
REC = os.path.join(HERE, "receipts")

# first-campaign full runs (every item): tag -> run folder
OLD = {"pkbase": f"{RUNS}/score_hfgguf/x/bench/pkbase", "pksft": f"{RUNS}/score_hfgguf/x/bench/pksft",
       "live": f"{RUNS}/score_live/x/bench/live", "pick_e1_L16": f"{RUNS}/score_hfgguf/x/bench/pick_e1_L16",
       "pick_e1_L32": f"{RUNS}/score_hfgguf/x/bench/pick_e1_L32",
       "pick_pilot_s35": f"{RUNS}/score_hfgguf/x/bench/pick_pilot_s35",
       "pythia160m": f"{RUNS}/hf_ref/x/bench/pythia160m", "smollm2_135m": f"{RUNS}/hf_ref/x/bench/smollm2_135m"}
NEW = ("pick_e1_L4", "pick_e1_60m_L32", "pick_e5_s0")
EMBER = ("pksft", "live") + NEW


def rel(path):
    return os.path.relpath(path, "C:/ml_artifacts/benchmarks").replace(os.sep, "/")


def keys_in_order(items):
    """bench_eval.requests' keys, in its order, without a tokeniser."""
    out = []
    for it in items:
        n = len(it["choices"]) if it["kind"] == "mc" else len(it["contexts"]) if it["kind"] == "partial" else 1
        out += [f"{it['id']}|{k}" for k in range(n)]
    return out


def selected(meta):
    return be.select(be.unique_ids(be.read_jsonl(ITEMS)), meta["sets"], meta["every"], meta["blimp_per"])


def rekey(work):
    """{key at fafc4750: result} from a run's chunks, each chunk read on its own."""
    meta = json.load(open(os.path.join(work, "meta.json"), encoding="utf-8"))
    keys = keys_in_order(selected(meta))
    files = sorted(glob.glob(os.path.join(work, "score", "chunk*.json")))
    chunks = [json.load(open(f, encoding="utf-8"))["results"] for f in files]
    size = len(chunks[0])
    if sum(len(c) for c in chunks) != len(keys) and len(files) != math.ceil(len(keys) / size):
        sys.exit(f"{work}: {len(files)} chunks of {size} for {len(keys)} requests")
    res = {}
    for i, k in enumerate(keys):
        c = chunks[i // size]
        if k in c:
            res[k] = c[k]
            continue
        iid, ch = k.rsplit("|", 1)
        old = iid.split("#", 1)[0] + "|" + ch
        if old not in c:
            sys.exit(f"{work}: request {i} ({k}) not in chunk {i // size}")
        res[k] = c[old]
    return meta, res


def records(meta, res, sets=None):
    items = [it for it in selected(meta) if sets is None or it["set"] in sets]
    return items, be.item_records(items, res, {})


def cmd_hellaswag():
    log = {}
    for tag, work in OLD.items():
        meta, res = rekey(work)
        items = [it for it in selected(meta) if it["set"] == "hellaswag"]
        with tempfile.TemporaryDirectory() as tmp:
            be.UNFIT.clear()
            be.summarise(tmp, items, res, {**meta, "truncated": meta["truncated"]}, {})
            hs = json.load(open(os.path.join(tmp, "summary.json")))["sets"]["hellaswag"]
            with gzip.open(os.path.join(tmp, "items.jsonl.gz"), "rt", encoding="utf-8") as f:
                hrecs = [json.loads(x) for x in f]
        d = os.path.join(REC, "bench", tag)
        if not os.path.exists(os.path.join(d, "summary.json")):  # the live model's came home in its run only
            os.makedirs(d, exist_ok=True)
            for f in ("summary.json", "meta.json", "items.jsonl.gz"):
                shutil.copyfile(os.path.join(work, f), os.path.join(d, f))
        sp = os.path.join(d, "summary.json")
        s = json.load(open(sp, encoding="utf-8"))
        s["sets"]["hellaswag"] = hs
        be.write_atomic(sp, json.dumps(s, indent=1))
        ip = os.path.join(d, "items.jsonl.gz")
        with gzip.open(ip, "rt", encoding="utf-8") as f:
            recs = [json.loads(x) for x in f]
        rest = [r for r in recs if r["set"] != "hellaswag"]
        if len(rest) + len(hrecs) != len(recs):
            sys.exit(f"{tag}: {len(recs)} records, {len(hrecs)} HellaSwag now")
        with gzip.open(ip + ".tmp", "wt", encoding="utf-8") as f:
            for r in hrecs + rest:
                f.write(json.dumps(r) + "\n")
        os.replace(ip + ".tmp", ip)
        log[tag] = {"run": rel(work), "items": len(hrecs),
                    "acc": hs["acc"], "acc_norm": hs["acc_norm"]}
        print(tag, json.dumps(log[tag]))
    be.write_atomic(os.path.join(REC, "bench", "hellaswag_one_request_per_key.json"), json.dumps(
        {"method": "each first-campaign run's chunks read one at a time and keyed as tools/bench_eval.py keys them "
                   "at fafc4750 (unique_ids), then bench_eval.summarise on the 10,042 HellaSwag items "
                   "(experiments/benchmarks_2026-10-26/bench2.py hellaswag)", "runs": log}, indent=1))


def cmd_collect():
    for t in NEW:
        src = f"{RUNS2}/score_hfgguf/x/bench/{t}"
        d = os.path.join(REC, "bench", t)
        os.makedirs(d, exist_ok=True)
        for f in ("summary.json", "meta.json", "items.jsonl.gz"):
            shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
        shutil.copyfile(f"{RUNS2}/score_hfgguf/x/compare_{t}_mtstudio_vs_hfgguf.json",
                        os.path.join(REC, "bench", "xcheck", f"compare_{t}_mtstudio_vs_hfgguf.json"))
    for f in ("staged.jsonl", "provenance.json", "gpu.txt", "hf_versions.txt"):
        shutil.copyfile(f"{RUNS2}/score_hfgguf/x/{f}", os.path.join(REC, "bench", "xcheck", "bench2_" + f))
    e = os.path.join(REC, "ember_score")
    for t in EMBER:
        src = f"{RUNS2}/ember_score/x/bench/{t}"
        d = os.path.join(e, t)
        os.makedirs(d, exist_ok=True)
        # its per-item records stay in runs2 (every choice equals the reference's: compare_*.json)
        for f in ("summary.json", "meta.json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
        shutil.copyfile(f"{RUNS2}/ember_score/x/health_{t}.json", os.path.join(d, "health.json"))
    for f in ("ember_build.json", "staged.jsonl", "provenance.json", "gpu.txt"):
        shutil.copyfile(f"{RUNS2}/ember_score/x/{f}", os.path.join(e, f))
    print("collected")


def gap(sums):
    s = sorted(sums, reverse=True)
    return s[0] - s[1] if len(s) > 1 else None


def compare(a_work, b_work, a_name, b_name):
    """Per item, two engines on the same model: every request both scored."""
    ma, ra = rekey(a_work)
    mb, rb = rekey(b_work)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        sys.exit("no common requests")
    worst, tok, tot, arg_eq, arg_n = 0.0, 0, 0.0, 0, 0
    sdiff = []
    for k in keys:
        x, y = ra[k], rb[k]
        if x.get("over_ctx") or y.get("over_ctx"):
            continue
        if len(x["logprobs"]) != len(y["logprobs"]):
            sys.exit(f"{k}: continuations of different lengths")
        d = [abs(p - q) for p, q in zip(x["logprobs"], y["logprobs"])]
        worst = max(worst, max(d))
        tot += sum(d)
        tok += len(d)
        arg_n += len(x["argmax"])
        arg_eq += sum(p == q for p, q in zip(x["argmax"], y["argmax"]))
        sdiff.append(abs(x["sum"] - y["sum"]))
    sdiff.sort()
    have = {k.rsplit("|", 1)[0] for k in keys}
    items = [it for it in selected(ma) if it["id"] in have]
    items = [it for it in items if all(f"{it['id']}|{c}" in ra and f"{it['id']}|{c}" in rb
                                       for c in range(len(it["choices"]) if it["kind"] == "mc" else
                                                      len(it["contexts"]) if it["kind"] == "partial" else 1))]
    be.UNFIT.clear()
    for iid in ma.get("unfit_items", []):  # chance credit in both, as bench_eval gives it
        be.UNFIT.update(f"{iid}|{c}" for c in range(4))
    # LAMBADA is compared on the argmax lists below; its acc from item_records is not used here
    conts = {k: ra[k]["argmax"] for k in keys if k.startswith("lambada/")}
    A = {r["id"]: r for r in be.item_records(items, ra, conts)}
    B = {r["id"]: r for r in be.item_records(items, rb, conts)}
    per_set, dis = {}, []
    for it in items:
        x, y = A[it["id"]], B[it["id"]]
        s = per_set.setdefault(it["set"], {"items": 0, "pick_agree": 0, "pick_norm_agree": 0, "lastword_agree": 0})
        s["items"] += 1
        if it["kind"] == "lastword":
            # acc needs every target token to be the argmax: compare the argmax lists themselves
            k = f"{it['id']}|0"
            same = list(ra[k]["argmax"]) == list(rb[k]["argmax"])
            s["lastword_agree"] += same
            if not same:
                dis.append({"id": it["id"], "set": it["set"], "metric": "argmax",
                            "max_abs_logprob_diff": max(abs(p - q) for p, q in
                                                        zip(ra[k]["logprobs"], rb[k]["logprobs"]))})
            continue
        for m, field in (("pick", "sums"), ("pick_norm", None)):
            if m not in x:
                continue
            same = x[m] == y[m]
            s[m + "_agree"] += same
            if not same:
                if field:
                    ga, gb = gap(x["sums"]), gap(y["sums"])
                else:
                    lens = [max(len(c[1:] if c.startswith(" ") else c), 1) for c in it["choices"]]
                    ga = gap([v / n for v, n in zip(x["sums"], lens)])
                    gb = gap([v / n for v, n in zip(y["sums"], lens)])
                nreq = len(x["sums"])
                dsum = max(abs(ra[f"{it['id']}|{c}"]["sum"] - rb[f"{it['id']}|{c}"]["sum"]) for c in range(nreq))
                dis.append({"id": it["id"], "set": it["set"], "metric": m, a_name: x[m], b_name: y[m],
                            "gap_" + a_name: ga, "gap_" + b_name: gb, "max_abs_sum_diff": dsum,
                            "credit_" + a_name: x["acc" if m == "pick" else "acc_norm"],
                            "credit_" + b_name: y["acc" if m == "pick" else "acc_norm"]})
    for name, s in per_set.items():
        if name == "lambada":
            drop = ("pick_agree", "pick_norm_agree")
        else:
            drop = ("lastword_agree",) + (() if s["pick_norm_agree"] else ("pick_norm_agree",))
        for k in drop:
            s.pop(k)
    return {"a": f"{a_name} {rel(a_work)}", "b": f"{b_name} {rel(b_work)}",
            "requests": len(keys), "tokens": tok, "max_abs_logprob_diff": worst, "mean_abs_logprob_diff": tot / tok,
            "median_abs_sum_diff": sdiff[len(sdiff) // 2], "max_abs_sum_diff": sdiff[-1],
            "argmax_agree": f"{arg_eq}/{arg_n}", "items": len(items), "per_set": per_set, "disagreements": dis}


def cmd_ember():
    e = f"{RUNS2}/ember_score/x/bench"
    pairs = [("pksft", f"{RUNS}/score_hfgguf/x/bench/pksft", "hfgguf"),
             ("pksft", f"{RUNS}/score_hfgguf/x/xcheck_mtstudio/pksft", "mtstudio"),
             ("live", f"{RUNS}/score_live/x/bench/live", "mtstudio")]
    for t in NEW:
        pairs += [(t, f"{RUNS2}/score_hfgguf/x/bench/{t}", "hfgguf"),
                  (t, f"{RUNS2}/score_hfgguf/x/xcheck_mtstudio/{t}", "mtstudio")]
    out = {}
    for t, ref, name in pairs:
        c = compare(f"{e}/{t}", ref, "ember", name)
        out[f"{t} ember vs {name}"] = {k: v for k, v in c.items() if k not in ("disagreements",)}
        out[f"{t} ember vs {name}"]["disagreements"] = len(c["disagreements"])
        be.write_atomic(os.path.join(REC, "ember_score", f"compare_{t}_ember_vs_{name}.json"), json.dumps(c, indent=1))
        print(t, name, json.dumps({k: c[k] for k in ("requests", "max_abs_logprob_diff", "mean_abs_logprob_diff",
                                                      "argmax_agree", "items")}), "disagreements", len(c["disagreements"]))
        for d in c["disagreements"]:
            print("   ", json.dumps(d))
    be.write_atomic(os.path.join(REC, "ember_score", "compare_all.json"), json.dumps(out, indent=1))


def cmd_table():
    order = ["pkbase", "pksft", "live", "pick_e1_L16", "pick_e1_L32", "pick_pilot_s35", "pick_e1_L4",
             "pick_e1_60m_L32", "pick_e5_s0", "pythia160m", "smollm2_135m"]
    for t in order:
        s = json.load(open(os.path.join(REC, "bench", t, "summary.json"), encoding="utf-8"))["sets"]
        cells = [be.fmt_cell(s[x], be.HEADLINE[x]) for x in be.SETS]
        print(f"| {t} | " + " | ".join(cells) + f" | ppl {s['lambada']['ppl']} | hs acc {s['hellaswag']['acc']}")
    for t in EMBER:
        s = json.load(open(os.path.join(REC, "ember_score", t, "summary.json"), encoding="utf-8"))["sets"]
        print(f"| ember {t} | " + " | ".join(be.fmt_cell(s[x], be.HEADLINE[x]) for x in be.SETS) + " |")


if __name__ == "__main__":
    {"hellaswag": cmd_hellaswag, "collect": cmd_collect, "ember": cmd_ember, "table": cmd_table}[sys.argv[1]]()
