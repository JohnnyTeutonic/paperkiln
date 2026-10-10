"""Fill results/abstracts_cache.json with arXiv abstracts for the papers stored in
results/scoped.jsonl, by scraping arxiv.org/abs pages.

The arXiv export API returns 406 to this machine, Semantic Scholar rate-limits a
single request and OpenAlex times out; the abstract HTML pages answer in under
half a second. One request every three seconds, cache written every 25.

Usage:  python prefetch_abstracts.py [--n-per-class 8] [--cap 60] [--all] [--seed 20260922]
The selection must match jev_judge_eval.select() so the cache covers what the
harness will ask for.
"""
import argparse, html, json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jev_judge_eval as J  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (paperkiln-scoper; mailto:jonathanreich100@gmail.com)"}
MARK = 'name="citation_abstract" content="'


def fetch(arxiv_id):
    b = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/abs/{arxiv_id}", headers=UA),
                               timeout=30).read().decode("utf-8", "replace")
    j = b.find(MARK)
    if j < 0:
        return ""
    j += len(MARK)
    return html.unescape(b[j:b.find('"', j)]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--sleep", type=float, default=3.0)
    args = ap.parse_args()

    scoped = [r for r in J.jsonl(os.path.join(HERE, "results", "scoped.jsonl")) if r.get("papers")]
    todo = J.select(scoped, args.n_per_class, args.seed, args.all)
    ids = []
    for r in todo:
        for p in r["papers"][:args.cap]:
            if p.get("id") and p["id"] not in ids:
                ids.append(p["id"])
    cache = json.load(open(J.CACHE, encoding="utf-8")) if os.path.exists(J.CACHE) else {}
    need = [i for i in ids if i not in cache]
    print(f"ideas {len(todo)}, unique papers {len(ids)}, cached {len(ids) - len(need)}, to fetch {len(need)} "
          f"(~{int(len(need) * args.sleep // 60)} min)", flush=True)
    for k, i in enumerate(need, 1):
        try:
            cache[i] = fetch(i)
        except Exception as ex:
            cache[i] = ""
            print(f"  [{k}] {i} failed: {type(ex).__name__}", flush=True)
        if k % 25 == 0 or k == len(need):
            json.dump(cache, open(J.CACHE, "w", encoding="utf-8"))
            print(f"  {k}/{len(need)} fetched; cache holds {sum(1 for v in cache.values() if v)} abstracts", flush=True)
        time.sleep(args.sleep)
    json.dump(cache, open(J.CACHE, "w", encoding="utf-8"))
    print("PREFETCH DONE:", sum(1 for i in ids if cache.get(i)), "of", len(ids), "have abstracts", flush=True)


if __name__ == "__main__":
    main()
