"""Durable inputs for the CUDA verification job (C:/ml_artifacts/score_audit/inputs)."""
import json
import types
import os
import sys
import shutil
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import bench_eval
import carry5_eval  # noqa: E402
SA = os.environ.get("SCORE_AUDIT_WORK", "C:/ml_artifacts/score_audit/work")
OUT = r"C:/ml_artifacts/score_audit/inputs"
os.makedirs(OUT, exist_ok=True)
gguf = r"C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf"
t = carry5_eval.Tok(gguf)
tok = types.SimpleNamespace(encode=t.encode, eot=t.eot, ctx=512)
allitems = bench_eval.unique_ids(bench_eval.read_jsonl(r"C:/ml_artifacts/benchmarks/bench_v1.jsonl"))
sel = bench_eval.select(allitems, list(bench_eval.SETS), 400, 100)
reqs, _ = bench_eval.requests(sel, tok)
second = [it for it in allitems if it["id"] == "hellaswag/17234#2"]
r2, _ = bench_eval.requests(second, tok)
# The pod's xcheck chunk 0 (its first 128 requests, 17234's first copy at rows 12-15), then the second copy.
rows = reqs[:128] + r2
with open(f"{OUT}/bench_xcheck_chunk0_plus_17234b.in.jsonl", "w", newline="\n") as f:
    for k, p, c in rows:
        f.write(json.dumps({"key": k, "prompt_ids": p, "continuation_ids": c}) + "\n")
print("bench rows", len(rows))
for run in ("ctl_r1", "x5_r1"):
    shutil.copy(f"{SA}/sa_{run}/chunk0000.in.jsonl", f"{OUT}/carry5_{run}_chunk0000.in.jsonl")
VOC = "/content/chat/upload/vocab_llama3_bpe.gguf"
s = json.load(open(r"C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/spec.json"))
s["out_dir"], s["data"]["vocab"] = "/content/sa/pick_e1_L16", VOC
json.dump(s, open(f"{OUT}/spec_pick_e1_L16.json", "w"), indent=1)
s = json.load(open(r"C:/ml_artifacts/tiny_chat/e1_x5_r1/post_e1_x5_X5_r1/sft_spec.json"))
s["out_dir"], s["data"]["vocab"] = "/content/sa/x5_r1", VOC
json.dump(s, open(f"{OUT}/spec_x5_r1.json", "w"), indent=1)
print(sorted(os.listdir(OUT)))
