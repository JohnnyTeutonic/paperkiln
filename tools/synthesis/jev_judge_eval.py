"""Evaluate Jev as a replacement for the scoper's verdict step.

For each idea, Jev sees the proposal text and the retrieved papers the
Sonnet judge saw (ids/titles from results/scoped.jsonl, abstracts from
results/abstracts_cache.json, filled by scraping arxiv.org/abs pages because
the arXiv API refuses this machine) and answers
one Choice: TAKEN / ADJACENT / OPEN, with the judge rubric from pipeline.py as
the instructions. Jev cannot write the evidence quote or the "survives" line;
this measures only whether its verdict and confidence are usable for triage.

Scored against (a) the four controls' expected verdicts and (b) agreement
with the Sonnet verdict on a stratified sample. Writes results/jev_judge.jsonl.

Usage:  python jev_judge_eval.py [--n-per-class 8] [--cap 30] [--all]
"""
import argparse, json, os, random, sys, time, collections
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CACHE = os.path.join(HERE, "results", "abstracts_cache.json")


def load_env():
    raw = open(os.path.join(ROOT, "jev.env"), "rb").read()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    for line in txt.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

CRITERIA = {
    "TAKEN": "a paper in the set places an equivalent mechanism at the SAME component for the SAME purpose; "
             "differences of scale, training recipe, domain or framing do not save the proposal",
    "ADJACENT": "clear overlap but not the same purpose: same mechanism at that component for a different "
                "capability, or the same capability by a different mechanism, or the same mechanism at a "
                "neighbouring component",
    "OPEN": "nothing in the set does this",
}
EXPECT = {"ctl-kvdiff": ["TAKEN"], "ctl-pid": ["TAKEN"], "ctl-hebbffn": ["TAKEN", "ADJACENT"], "ctl-choquet": ["ADJACENT"]}


def desc_of(idea):
    return (f"NAME: {idea.get('name')}\nCLAIM: {idea.get('claim')}\nNEW CAPABILITY: {idea.get('new_capability')}\n"
            f"PLACEMENT: {idea.get('placement')}\nCOMPONENT: {idea.get('component')}; MECHANISM: {idea.get('mechanism')}\n"
            f"GENERATOR'S OWN NEAREST-KNOWN GUESS: {idea.get('nearest_known')}")


def select(scoped, n_per_class, seed, all_):
    controls = [r for r in scoped if str(r["id"]).startswith("ctl-")]
    others = [r for r in scoped if not str(r["id"]).startswith("ctl-") and r.get("verdict") in ("TAKEN", "ADJACENT", "OPEN")]
    if all_:
        return controls + others
    rng = random.Random(seed)
    sample = []
    for v in ("TAKEN", "ADJACENT", "OPEN"):
        pool = [r for r in others if r["verdict"] == v]
        rng.shuffle(pool)
        sample += pool[:n_per_class]
    return controls + sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--cap", type=int, default=60, help="papers per idea, in stored order (backstops first); 60 matches what the Sonnet judge saw")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--min-abstracts", type=float, default=0.6, help="skip ideas with fewer than this fraction of abstracts cached")
    args = ap.parse_args()

    load_env()
    from typesafe_sdk import TypeSafeClient, Choice
    client = TypeSafeClient(timeout=30.0)

    ideas = {r["id"]: r for r in jsonl(os.path.join(HERE, "results", "ideas.jsonl"))}
    scoped = [r for r in jsonl(os.path.join(HERE, "results", "scoped.jsonl")) if r.get("papers")]
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    todo = select(scoped, args.n_per_class, args.seed, args.all)
    print(f"ideas to judge {len(todo)}; abstracts cached {sum(1 for v in cache.values() if v)}")

    out_path = os.path.join(HERE, "results", "jev_judge.jsonl")
    results, tok_in, skipped = [], 0, 0
    with open(out_path, "w", encoding="utf-8") as out:
        for k, row in enumerate(todo, 1):
            idea = ideas.get(row["id"], {})
            papers = []
            for p in row["papers"][:args.cap]:
                a = cache.get(p.get("id") or "", "")
                papers.append({"id": p.get("id"), "date": p.get("published"), "title": p.get("title"),
                               "abstract": a[:1800] if a else "(abstract unavailable)"})
            frac = sum(1 for p in papers if not p["abstract"].startswith("(")) / max(1, len(papers))
            if frac < args.min_abstracts:
                skipped += 1
                print(f"[{k:3d}/{len(todo)}] {row['id']:<18} skipped: only {frac:.0%} of abstracts cached")
                continue
            state = {"proposal": desc_of(idea) if idea else row["id"], "candidate_papers": papers}
            t0 = time.time()
            try:
                r = client.system_one(state=state, questions={"verdict": Choice(instructions=P.JUDGE_SYSTEM, criteria=CRITERIA)})
                a = r.answers["verdict"]
                rec = {"id": row["id"], "sonnet": row.get("verdict"), "sonnet_conf": row.get("confidence"),
                       "jev": a.choice, "jev_conf": a.confidence, "jev_probs": a.probabilities,
                       "ms": round((time.time() - t0) * 1000), "input_tokens": r.usage.input_tokens,
                       "n_papers": len(papers), "abstract_frac": round(frac, 2), "model": r.model}
                tok_in += r.usage.input_tokens or 0
            except Exception as e:
                rec = {"id": row["id"], "sonnet": row.get("verdict"), "jev": "ERROR", "error": f"{type(e).__name__}: {str(e)[:160]}"}
            results.append(rec)
            out.write(json.dumps(rec) + "\n"); out.flush()
            flag = "" if rec.get("jev") == rec.get("sonnet") else "  <-- differs"
            print(f"[{k:3d}/{len(todo)}] {rec['id']:<18} sonnet={rec.get('sonnet'):<8} jev={rec.get('jev'):<8} "
                  f"conf={rec.get('jev_conf', 0):.2f} {rec.get('ms', 0):>5} ms  abs={rec.get('abstract_frac', 0):.0%}{flag}")

    print("\n=== CONTROLS (expected verdicts from pipeline.py) ===")
    for rec in results:
        if rec["id"] in EXPECT:
            ok = rec.get("jev") in EXPECT[rec["id"]]
            print(f"  {rec['id']:<14} expected {'/'.join(EXPECT[rec['id']]):<16} jev={rec.get('jev'):<9} "
                  f"conf={rec.get('jev_conf', 0):.2f}  {'PASS' if ok else 'FAIL'}   (sonnet said {rec.get('sonnet')})")

    good = [r for r in results if r.get("jev") in CRITERIA and not r["id"].startswith("ctl-")]
    agree = [r for r in good if r["jev"] == r["sonnet"]]
    print(f"\n=== AGREEMENT WITH SONNET on {len(good)} sampled ideas: {len(agree)}/{len(good)} = {len(agree)/max(1, len(good)):.0%} ===")
    cm = collections.Counter((r["sonnet"], r["jev"]) for r in good)
    labels = ("TAKEN", "ADJACENT", "OPEN")
    print("  sonnet \\ jev   " + "  ".join(f"{v:>8}" for v in labels))
    for sv in labels:
        print(f"  {sv:<14} " + "  ".join(f"{cm.get((sv, jv), 0):>8}" for jv in labels))
    if good:
        m = lambda x: sum(x) / len(x) if x else float("nan")
        ca = [r["jev_conf"] for r in agree]; cd = [r["jev_conf"] for r in good if r["jev"] != r["sonnet"]]
        print(f"  mean Jev confidence when agreeing {m(ca):.2f} (n={len(ca)}), when differing {m(cd):.2f} (n={len(cd)})")
        for thr in (0.6, 0.7, 0.8, 0.9):
            hi = [r for r in good if r["jev_conf"] >= thr]
            ha = sum(1 for r in hi if r["jev"] == r["sonnet"])
            print(f"  confidence >= {thr}: {len(hi)}/{len(good)} kept, agreement {ha}/{len(hi) or 1} = {ha/max(1, len(hi)):.0%}")
    ms = sorted(r["ms"] for r in results if "ms" in r)
    print(f"\n=== COST/LATENCY === input tokens {tok_in:,} -> ${tok_in/1e6*0.042:.3f}; median {ms[len(ms)//2] if ms else 0} ms; "
          f"errors {sum(1 for r in results if r.get('jev') == 'ERROR')}; skipped for missing abstracts {skipped}")
    print("written:", out_path)


if __name__ == "__main__":
    main()
