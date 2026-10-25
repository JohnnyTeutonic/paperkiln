"""The numpy IR's records (key, over_ctx, logprobs, argmax, sum) for the first N requests of an mtstudio score
input file: the reference the CUDA job's score_sequence_check.py --ref compares with.
    python ir_ref.py GGUF IN.jsonl OUT.jsonl N"""
import json
import os
import sys
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import carry5_eval  # noqa: E402
gguf, inp, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ir = carry5_eval.IrBackend(gguf)
with open(out + ".tmp", "w", newline="\n") as f:
    for r in carry5_eval.read_jsonl(inp)[:n]:
        f.write(json.dumps({"key": r["key"], **ir.score(r["prompt_ids"], r["continuation_ids"])}) + "\n")
os.replace(out + ".tmp", out)
