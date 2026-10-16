set -e
mkdir -p ~/pt_bridge && cd ~/pt_bridge && rm -rf run
python3 - <<'PY'
import json
s=json.load(open("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/experiments/atlas_stage3_rope_all/receipts/runs/run_000_c00_s1/spec.json"))
s["train"]["steps"]=100; s["out_dir"]="/home/jonat/pt_bridge/run"
json.dump(s,open("/home/jonat/pt_bridge/spec.json","w"))
PY
timeout 1500 ~/mtrel/mtstudio run ~/pt_bridge/spec.json > log.txt 2>&1
python3 - <<'PY'
import json
def ev(p): return [json.loads(l) for l in open(p)]
old=ev("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/experiments/atlas_stage3_rope_all/receipts/runs/run_000_c00_s1/events.jsonl")
new=ev("/home/jonat/pt_bridge/run/events.jsonl")
o=[(e["step"],e["loss"]) for e in old if e["event"]=="step"][:100]
n=[(e["step"],e["loss"]) for e in new if e["event"]=="step"][:100]
same=sum(a==b for a,b in zip(o,n)); print("step losses bit-identical:",same,"of",len(n))
oe=[e for e in old if e["event"]=="eval"][0]; ne=[e for e in new if e["event"]=="eval"][0]; print("eval@100", oe["val_loss"], ne["val_loss"], oe["val_loss"]==ne["val_loss"])
om=[e for e in old if e["event"]=="model"][0]; nm=[e for e in new if e["event"]=="model"][0]; print("model event identical:", om==nm)
od=[e for e in old if e["event"]=="data"][0]; nd=[e for e in new if e["event"]=="data"][0]; print("data event identical:", od==nd)
PY
