import json
import os
SA = os.environ.get("SCORE_AUDIT_WORK", "C:/ml_artifacts/score_audit/work")
TC = "/mnt/c/ml_artifacts/tiny_chat"
out = {}
for r in (1, 2, 3):
    s = json.load(open(f"C:/ml_artifacts/tiny_chat/e1_x5_r{r}/post_e1_x5_X5_r{r}/sft_spec.json"))
    s["out_dir"] = f"{TC}/e1_x5_r{r}/post_e1_x5_X5_r{r}/sft"
    s["data"]["vocab"] = f"{s['out_dir']}/{s['name']}.gguf"
    out[f"x5_r{r}"] = s
s = json.load(open("C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/spec.json"))
s["out_dir"] = "/home/jonat/sa_models/ctl_r1"
s["data"]["vocab"] = f"{s['out_dir']}/{s['name']}.gguf"
out["ctl_r1"] = s
for k, v in out.items():
    json.dump(v, open(f"{SA}/sa_spec_{k}.json", "w"), indent=1)
    print(k, v["name"], v["out_dir"], v["arch"].get("rope_heads"))
