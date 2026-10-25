"""HellaSwag's duplicated `ind` ids in bench_v1.jsonl: per full bench run, the reported acc/acc_norm,
and the same with each first copy scored by its own requests where its chunk still holds them."""
import json
import os
import glob
import sys
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import bench_eval  # noqa: E402
items = [it for it in bench_eval.read_jsonl(r"C:/ml_artifacts/benchmarks/bench_v1.jsonl") if it["set"] == "hellaswag"]
pos = {}
for j, it in enumerate(items):
    pos.setdefault(it["id"], []).append(j)
dup = {k: v for k, v in pos.items() if len(v) > 1}
print("hellaswag items", len(items), "ids with two items", len(dup))
runs = sorted(glob.glob(r"C:/ml_artifacts/benchmarks/runs/*/x/bench/*/meta.json"))
for mp in runs:
    meta = json.load(open(mp))
    if meta.get("every") != 1 or "hellaswag" not in meta["sets"] or meta["sets"][0] != "hellaswag":
        continue
    work = os.path.dirname(mp)
    chunks = {}
    for f in sorted(glob.glob(f"{work}/score/chunk*.json")):
        chunks[int(os.path.basename(f)[5:10])] = json.load(open(f))["results"]
    CH = meta.get("chunk", 2048)
    res = {}
    for n in sorted(chunks):
        res.update(chunks[n])
    def acc(own):
        k = kn = 0.0
        lost = 0
        for j, it in enumerate(items):
            src = res
            if own and it["id"] in dup and dup[it["id"]][0] == j:
                c0, c1 = (4 * j) // CH, (4 * dup[it["id"]][1]) // CH
                if c0 != c1:
                    src = chunks[c0]
                else:
                    lost += 1
            sums = [src[f"{it['id']}|{c}"]["sum"] for c in range(4)]
            lens = [max(len(ch[1:] if ch.startswith(" ") else ch), 1) for ch in it["choices"]]
            k += bench_eval.choose(sums, it["gold"])[0]
            kn += bench_eval.choose([s / n for s, n in zip(sums, lens)], it["gold"])[0]
        return k / len(items), kn / len(items), lost
    a0, n0, _ = acc(False)
    a1, n1, lost = acc(True)
    print(f"{os.path.relpath(work, 'C:/ml_artifacts/benchmarks/runs')}: reported acc {a0:.4f} acc_norm {n0:.4f}; "
          f"first copies own scores: acc {a1:.4f} acc_norm {n1:.4f} (d {a1 - a0:+.4f}, {n1 - n0:+.4f}); "
          f"first copies unrecoverable (same chunk) {lost}")
