import collections
import json
import types
import sys
import glob
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import bench_eval
import carry5_eval  # noqa: E402
allitems = bench_eval.read_jsonl(r"C:/ml_artifacts/benchmarks/bench_v1.jsonl")
items = bench_eval.select(allitems, list(bench_eval.SETS), 400, 100)
sel_ids = [it["id"] for it in items]
ids_all = [it["id"] for it in allitems]
c = collections.Counter(ids_all)
print("selected items with a duplicated id:", [i for i in sel_ids if c[i] > 1])
gguf = r"C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf"
t = carry5_eval.Tok(gguf)
tok = types.SimpleNamespace(encode=t.encode, eot=t.eot, ctx=512)
dups = [it for it in allitems if it["id"] == "hellaswag/17234"]
ir = carry5_eval.IrBackend(gguf)
base = r"C:/ml_artifacts/benchmarks/runs/score_hfgguf/x"
cuda = json.load(open(f"{base}/xcheck_mtstudio/pick_e1_L16/score/chunk00000.json"))["results"]
hf = {}
for f in glob.glob(f"{base}/bench/pick_e1_L16/score/chunk*.json"):
    hf.update(json.load(open(f))["results"])
for n, it in enumerate(dups):
    reqs, _ = bench_eval.requests([it], tok)
    for k, p, cc in reqs:
        r = ir.score(p, cc)
        dc = max(abs(a - b) for a, b in zip(r["logprobs"], cuda[k]["logprobs"])) if len(cuda[k]["logprobs"]) == len(cc) else None
        dh = max(abs(a - b) for a, b in zip(r["logprobs"], hf[k]["logprobs"])) if len(hf[k]["logprobs"]) == len(cc) else None
        print(f"copy {n} {k} len {len(cc)}: IR vs CUDA xcheck {dc}, IR vs hfgguf {dh}; lens cuda {len(cuda[k]['logprobs'])} hf {len(hf[k]['logprobs'])}")
