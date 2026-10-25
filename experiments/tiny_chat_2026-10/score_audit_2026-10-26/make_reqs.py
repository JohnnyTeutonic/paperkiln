"""Rebuild carry5_eval's request list for a stored VM run and write the subset Y needs
(self and twin modes, the four primary families) as chunked mtstudio score inputs."""
import json
import os
import sys
import hashlib
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import carry5_eval  # noqa: E402
SA = os.environ.get("SCORE_AUDIT_WORK", "C:/ml_artifacts/score_audit/work")
TC = r"C:/ml_artifacts/tiny_chat"
RUNS = {
    "ctl_r1": (f"{TC}/e1_x5_ctl/post_e1_x5_ctl_L16_r1", f"{TC}/e1/pick_e1_L16/model.gguf"),
    "x5_r1": (f"{TC}/e1_x5_r1/post_e1_x5_X5_r1", f"{TC}/e1_x5_r1/post_e1_x5_X5_r1/sft/tc_sft_X5_r1.gguf"),
    "x5_r2": (f"{TC}/e1_x5_r2/post_e1_x5_X5_r2", f"{TC}/e1_x5_r2/post_e1_x5_X5_r2/sft/tc_sft_X5_r2.gguf"),
    "x5_r3": (f"{TC}/e1_x5_r3/post_e1_x5_X5_r3", f"{TC}/e1_x5_r3/post_e1_x5_X5_r3/sft/tc_sft_X5_r3.gguf"),
}
PRIMARY = ("recall", "correction", "topic", "knowledge")
CH = 256
which = sys.argv[1:] or list(RUNS)
items = carry5_eval.read_jsonl(carry5_eval.SET)
print("set sha", hashlib.sha256(open(carry5_eval.SET, "rb").read()).hexdigest()[:16], len(items))
for run in which:
    post, gguf = RUNS[run]
    tok = carry5_eval.Tok(gguf)  # vocab_llama3_bpe, the same for every run here
    gens = {r["id"]: r for r in carry5_eval.read_jsonl(f"{post}/carry5/m/gen.jsonl")}
    reqs = carry5_eval.requests(items, tok, gens)
    fam = {it["id"]: it["family"] for it in items}
    keep = [(k, p, c) for k, p, c in reqs
            if fam[k.split("|")[0]] in PRIMARY and k.split("|")[1] in ("self", "twin")]
    out = f"{SA}/sa_{run}"
    os.makedirs(out, exist_ok=True)
    for n in range(0, len(keep), CH):
        with open(f"{out}/chunk{n // CH:04d}.in.jsonl", "w", newline="\n") as f:
            for k, p, c in keep[n:n + CH]:
                f.write(json.dumps({"key": k, "prompt_ids": p, "continuation_ids": c}) + "\n")
    L = [len(p) + len(c) for _, p, c in keep]
    print(run, "requests", len(reqs), "kept", len(keep), "chunks", (len(keep) + CH - 1) // CH,
          "tokens", sum(L), "max", max(L), "over T+1", sum(x > 513 for x in L))
