#!/usr/bin/env python3
"""Side-by-side evaluation of chat checkpoints on the frozen held-out prompts.

    python tools/chat_eval.py --mtstudio build/mtstudio \\
        --model base=runs/base/spec.json --model sft=runs/sft/spec.json [--model po=...] \\
        --prompts corpus/eval/tinychat_hf_prompts_v1.jsonl --out runs/chat_eval [--n 2 --temp 0.7]

Every model gets the same prompts in the served chat template
(<|user|>P<|assistant|>) and the same seeds. Per model it reports:
  clean_stop   reply ended at <|endoftext|> within the budget, with no role
               marker inside (the turn protocol: what SFT exists to teach)
  distinct2    distinct bigrams / bigrams in the reply (1 = no repetition)
  words        mean reply length in words
  ref_bpb      bits per byte of the REFERENCE reply under the model, via a
               masked-SFT evaluation pass (how well it predicts a good answer)
and writes side_by_side.md: each prompt, the reference, and every model's
reply, for reading. Metrics are descriptive; a pre-registered comparison
reads them from summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def distinct2(text):
    w = text.split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg) if bg else 1.0


def ref_bpb(mtstudio, spec_path, prompts, work, tag):
    """Bits per byte of the reference replies: one 1-step SFT run at lr 0 whose
    held-out evaluation covers every example (masked CE, nats/token), turned
    into bits per byte with the reference's own byte and token counts."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    ck = os.path.join(spec["out_dir"], spec["name"] + ".safetensors")
    vocab = spec["data"]["vocab"]
    tok = bpe.read_gguf(vocab)
    ex = [{"prompt": f"<|user|>{p['prompt']}<|assistant|>", "completion": p["reference"]} for p in prompts]
    # 19 copies of the first example per real one, then the real ones: the
    # held-out tail mtstudio scores (1/20 of the file) is exactly the real set.
    path = os.path.join(work, f"{tag}_ref.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in [ex[0]] * (19 * len(ex)) + ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    ev = {"name": f"{tag}_refeval", "arch": spec["arch"],
          "data": {"vocab": vocab, "vocab_cap": spec["data"].get("vocab_cap", 8192), "T": spec["data"]["T"],
                   "sft_examples": path},
          "train": {"steps": 1, "lr": 0.0, "batch": 1, "eval_every": 1, "checkpoint_every": 1000,
                    "gradmap_every": 0, "seed": 1, "objective": "sft", "init_from": ck},
          "export": {"formats": []}, "out_dir": os.path.join(work, f"{tag}_refeval")}
    sp = os.path.join(work, f"{tag}_refeval.json")
    json.dump(ev, open(sp, "w"))
    subprocess.run([mtstudio, "run", sp], check=True, capture_output=True)
    evs = [json.loads(x) for x in open(os.path.join(ev["out_dir"], "events.jsonl"), encoding="utf-8")]
    nats = next(e["val_loss"] for e in evs if e["event"] == "eval")
    held = ex
    toks = sum(len(tok.encode(e["completion"])) + 1 for e in held)
    byts = sum(len(e["completion"].encode()) + 1 for e in held)
    return nats * toks / byts / math.log(2), len(held)


def check_rope_heads(spec_path):
    """A spec over bare weights (no events.jsonl in out_dir) with no
    arch.rope_heads loads as legacy "first" (mtstudio resolve_rope_heads),
    which samples a model trained with "all" under the wrong architecture
    and corrupts every metric but ref_bpb. Refuse it."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec.get("arch", {}).get("rope_heads"):
        return
    if not os.path.exists(os.path.join(spec["out_dir"], "events.jsonl")):
        sys.exit(f"{spec_path}: out_dir has no events.jsonl and arch.rope_heads is unset, so mtstudio would load "
                 "the weights as rope_heads=\"first\"; state arch.rope_heads or copy the run's events.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--model", action="append", required=True, help="tag=spec.json")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(argv)
    for m in a.model:
        check_rope_heads(m.split("=", 1)[1])
    os.makedirs(a.out, exist_ok=True)
    prompts = read_jsonl(a.prompts)[:a.limit]
    pf = os.path.join(a.out, "prompts_templated.jsonl")
    with open(pf, "w", encoding="utf-8", newline="\n") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": f"<|user|>{p['prompt']}<|assistant|>", "uri": p["uri"]}) + "\n")
    summary, replies = {}, {}
    for m in a.model:
        tag, spec = m.split("=", 1)
        out = os.path.join(a.out, f"{tag}_samples.jsonl")
        subprocess.run([a.mtstudio, "sample-set", spec, "--prompts", pf, "--out", out, "--n", str(a.n),
                        "--temp", repr(a.temp), "--seed", "7", "--max-new", str(a.max_new)], check=True,
                       capture_output=True)
        recs = read_jsonl(out)
        clean = [r["stop"] == "eos" and not any(t in r["completion"] for t in ("<|user|>", "<|assistant|>"))
                 for r in recs]
        bpb, nref = ref_bpb(a.mtstudio, spec, prompts, a.out, tag)
        summary[tag] = {"clean_stop": sum(clean) / len(recs),
                        "distinct2": sum(distinct2(r["completion"]) for r in recs) / len(recs),
                        "words": sum(len(r["completion"].split()) for r in recs) / len(recs),
                        "ref_bpb": bpb, "ref_examples": nref, "samples": len(recs)}
        replies[tag] = {}
        for r in recs:
            replies[tag].setdefault(r["prompt_index"], r["completion"])
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    L = ["# Chat side by side", "", "| model | clean stop | distinct-2 | words | ref bits/byte |", "|---|---:|---:|---:|---:|"]
    for t, s in summary.items():
        L.append(f"| {t} | {s['clean_stop']:.2f} | {s['distinct2']:.2f} | {s['words']:.1f} | {s['ref_bpb']:.3f} |")
    L.append("")
    for i, p in enumerate(prompts):
        L += [f"### {i + 1}. {p['prompt']}", f"- reference: {p['reference']}"]
        for t in summary:
            L.append(f"- **{t}**: {replies[t].get(i, '')}")
        L.append("")
    open(os.path.join(a.out, "side_by_side.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
