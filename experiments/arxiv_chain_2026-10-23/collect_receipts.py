"""Copy a chain run's small receipts (WSL ext4 run directory) into this folder.

    python3 collect_receipts.py RUN_DIR OUT_DIR

Writes: spec_report, the two specs, manifest, sft_fit, parity.jsonl, chat.jsonl, the IR, the
export and mtstudio chat logs, the IR's expr / rope nodes (ir_nodes.json) and a training summary
(training_summary.json) from each stage's result.json and events.jsonl. Weights and the GGUF stay
in the run directory (sha256 in the manifest)."""
import glob
import json
import os
import shutil
import sys

run, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in ("spec_report.json", "pretrain_spec.json", "sft_spec.json", "manifest.json", "sft_fit.json",
          "parity.jsonl", "chat.jsonl", "export.log", "mtstudio_chat.log", "gate.json"):
    if os.path.exists(os.path.join(run, f)):
        shutil.copyfile(os.path.join(run, f), os.path.join(out, f))
for p in glob.glob(os.path.join(run, "export", "*.ir.json")):
    shutil.copyfile(p, os.path.join(out, os.path.basename(p)))
    ir = json.load(open(p))
    nodes = {sub: [n for n in ir[sub]["nodes"] if n["op"] in ("expr", "rope") or "_gain" in str(n.get("weight"))]
             for sub in ("embed", "block", "head")}
    json.dump({"block_ops": [n["op"] for n in ir["block"]["nodes"]],
               "head_ops": [n["op"] for n in ir["head"]["nodes"]], "nodes": nodes},
              open(os.path.join(out, "ir_nodes.json"), "w"), indent=1)

summary = {}
for stage in ("pretrain", "sft"):
    d = os.path.join(run, stage)
    if not os.path.exists(os.path.join(d, "result.json")):
        continue
    r = json.load(open(os.path.join(d, "result.json")))
    ev = [json.loads(x) for x in open(os.path.join(d, "events.jsonl")) if x.strip()]
    steps = [e for e in ev if e["event"] == "step"]
    summary[stage] = {
        "steps": r["final_step"], "batch": r["batch"], "T": r["T"], "params": r["params"], "lr": r["lr"],
        "wall_seconds": round(r["wall_seconds"], 1), "tokens_per_second": round(r["tokens_per_second"], 1),
        "train_loss_first": round(steps[0]["loss"], 4), "train_loss_last": round(steps[-1]["loss"], 4),
        "train_loss_mean_last_50": round(sum(e["loss"] for e in steps[-50:]) / len(steps[-50:]), 4),
        "val_loss": [{"step": e["step"], "val_loss": round(e["val_loss"], 4)} for e in ev if e["event"] == "eval"],
        "best_val": round(r["best_val"], 4),
        "expr_gradcheck": [e for e in ev if e["event"] == "expr_gradcheck"]}
json.dump(summary, open(os.path.join(out, "training_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
