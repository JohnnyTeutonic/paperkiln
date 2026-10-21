#!/usr/bin/env python3
"""Write the next base's Colab jobs file (LOCAL_LLM_STACK.md section 7):
lever 2's dialogue text re-fetched on the VM, the mix and its .ids
(next_base_data.sh), a tiny gate model on the new vocabulary for ember's
parity gate, the base itself, and lever 1's SFT set rebuilt for the new
vocabulary and context.

    python tools/colab_chat/make_next_base.py --lever2 C:/ml_artifacts/capacity/lever2 \
        --fineweb-bytes 3460000000 --total-bytes 4200000000 --steps 120000 --out tools/colab_chat/next_base_v1.json

Ratios come from the dialogue sources' real sizes (each dialogue_*/text.txt
brought home by the lever2 job): dialogue makes up --dialogue-main of the
main part and --dialogue-tail of the annealing tail, split across the
dialogue sources in proportion to their bytes, so every source is used to
the same fraction of itself and none runs dry (at most --max-use of any
source, leaving room for the mixer's one-document overshoot). Refuses a
total the sources cannot supply.
"""
from __future__ import annotations

import argparse
import json
import os

DIALOGUE = ["soda", "ultrachat", "oasst", "everyday"]
C = "/content/chat"
FW_TEXT = "/content/pk/corpus/data/fineweb_edu_shard{}_v1/pretrain_train.txt"
VOCAB_KEY = "vocab_llama3_bpe.gguf"


def ratios(avail, fw_bytes, total, d_main, d_tail, tail_frac, max_use):
    dsum = sum(avail.values())
    main = {"fineweb": 1.0 - d_main, **{s: d_main * avail[s] / dsum for s in DIALOGUE}}
    tail = {"fineweb": 1.0 - d_tail, **{s: d_tail * avail[s] / dsum for s in DIALOGUE}}
    need = {s: total * ((1 - tail_frac) * main[s] + tail_frac * tail[s]) for s in main}
    have = {"fineweb": fw_bytes, **avail}
    over = {s: (need[s], have[s]) for s in need if need[s] > max_use * have[s]}
    if over:
        raise SystemExit(f"--total-bytes {total} needs more than {max_use:.0%} of: {over}")
    return main, tail, {s: need[s] / have[s] for s in need}


def fmt(r):
    return ",".join(f"{s}={r[s]:.6f}" for s in sorted(r))


def lever2_job():
    srcs = ["soda", "oasst", "ultrachat", "everyday"]
    return {"name": "lever2", "cmd": ["python3", "corpus/fetchers/dialogue_pt.py", "all", "--out-dir", "{out}"],
            "outputs": ["dialogue_pt_v1.ledger.json"] + [f"dialogue_{s}_v1/report.json" for s in srcs]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lever2", required=True, help="home copy of the lever2 job's outputs")
    ap.add_argument("--fineweb-bytes", type=int, required=True, help="bytes of FineWeb text over the shards used")
    ap.add_argument("--fineweb-shards", type=int, default=1, help="FineWeb-Edu shards 0..K-1 (1-3 fetched and "
                    "pinned on the VM, their hashes brought home)")
    ap.add_argument("--total-bytes", type=int, required=True)
    ap.add_argument("--dialogue-main", type=float, default=0.27)
    ap.add_argument("--dialogue-tail", type=float, default=0.47)
    ap.add_argument("--tail-fraction", type=float, default=0.15)
    ap.add_argument("--max-use", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--extra-jobs", action="append", default=[],
                    help="a jobs file whose jobs ride in this launch (e.g. arxiv_to_chat.py --emit-jobs with "
                         "--vm-after data --vm-sft-after lever1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for v in (a.d, a.d_ff):
        if v % 256:
            raise SystemExit(f"d and d_ff must be multiples of 256 (llama.cpp k-quant rows): {v}")
    avail = {s: os.path.getsize(os.path.join(a.lever2, f"dialogue_{s}_v1", "text.txt")) for s in DIALOGUE}
    main, tail, use = ratios(avail, a.fineweb_bytes, a.total_bytes, a.dialogue_main, a.dialogue_tail,
                             a.tail_fraction, a.max_use)
    vocab = f"{C}/upload/{VOCAB_KEY}"
    fetch = [{"name": f"fw{n}", "cmd": ["sh", "-c", f"python3 corpus/fetchers/fineweb_edu.py shard{n}_v1 --pin && "
                                                  f"cp corpus/hashes/fineweb_edu_shard{n}_v1.sha256 {{out}}/"],
              "outputs": [f"fineweb_edu_shard{n}_v1.sha256"]} for n in range(1, a.fineweb_shards)]
    data = {"name": "data", "after": ["lever2"] + [j["name"] for j in fetch],
            "cmd": ["bash", "tools/colab_chat/next_base_data.sh", "{out}", vocab, f"{C}/lever2",
                    *[FW_TEXT.format(n) for n in range(a.fineweb_shards)]],
            "env": {"MIX_MAIN": fmt(main), "MIX_TAIL": fmt(tail), "MIX_BYTES": a.total_bytes,
                    "TAIL_FRACTION": a.tail_fraction, "MIX_SEED": a.seed},
            "outputs": ["mix.ledger.json", "ids.json"]}
    corpus = {"corpus": f"{C}/data/mix.main.ids", "vocab": vocab, "vocab_cap": a.vocab_size, "max_tokens": 0}
    sched = {"warmup_steps": 2000, "decay": "cosine", "min_lr_frac": 0.1}
    export = {"formats": ["safetensors", "gguf", "gguf_q8_0_all"]}
    # ember's 40/40 gate needs a llama3-vocabulary export early, not a good model.
    gate = {"name": "gate", "model_name": "gate_llama3", "after": "data",
            "spec": {"arch": {"preset": "llama-tiny", "custom": {"d": 256, "layers": 4, "heads": 4, "d_ff": 768}},
                     "data": {**corpus, "T": 256, "max_tokens": 20000000},
                     "train": {"steps": 400, "lr": 0.001, "batch": 16, "eval_every": 200, "checkpoint_every": 200,
                               "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
                     "export": export}}
    base = {"name": "base", "model_name": "pk_base_v1", "after": "data", "relay_every_steps": 5000,
            "spec": {"arch": {"preset": "llama-tiny",
                              "custom": {"d": a.d, "layers": a.layers, "heads": a.heads, "d_ff": a.d_ff}},
                     "data": {**corpus, "T": a.T,
                              "anneal": {"corpus": f"{C}/data/mix.tail.ids", "from_fraction": 1 - a.tail_fraction}},
                     "train": {"steps": a.steps, "lr": 0.0006, "batch": a.batch, "eval_every": 1000,
                               "checkpoint_every": 1000, "gradmap_every": 0, "seed": 1, "optimizer": "adamw",
                               "lr_schedule": sched},
                     "export": export}}
    # Lever 1 on the new vocabulary and context, built on spare CPUs while
    # the base trains, for the base's SFT (a later launch).
    sft_data = {"name": "lever1", "delay_s": 1800,
                "cmd": ["python3", "tools/lever1_sft.py", "--out-dir", "{out}", "--vocab", vocab, "--T", str(a.T),
                        "--jobs", "4"],
                "outputs": ["lever1_sft_v1.jsonl", "lever1_sft_v1.ledger.json", "decontam_smol.json",
                            "decontam_tinychat.json", "decontam_everyday.json"]}
    tokens = a.steps * a.batch * a.T
    doc = {"_comment": (f"Next base (pk_base_v1): llama d{a.d} L{a.layers} d_ff {a.d_ff}, llama3-scheme {a.vocab_size} "
                        f"vocabulary, T {a.T}, {a.steps} steps x {a.batch} x {a.T} = {tokens / 1e9:.2f}B tokens on a "
                        f"{a.total_bytes / 1e9:.2f} GB mix ({a.dialogue_main:.0%} dialogue in the main part, "
                        f"{a.dialogue_tail:.0%} in the last {a.tail_fraction:.0%}, read from step "
                        f"{int((1 - a.tail_fraction) * a.steps)}). Generated by tools/colab_chat/make_next_base.py; "
                        f"source use: " + ", ".join(f"{s} {u:.0%}" for s, u in sorted(use.items())) +
                        f". Upload {VOCAB_KEY} with the runner's --upload."),
           "max_parallel": 5 + len(fetch), "jobs": [lever2_job(), *fetch, data, gate, base, sft_data]}
    for path in a.extra_jobs:
        with open(path, encoding="utf-8") as f:
            extra = json.load(f)["jobs"]
        names = {j["name"] for j in doc["jobs"]}
        clash = [j["name"] for j in extra if j["name"] in names]
        if clash:
            raise SystemExit(f"{path}: job names already in the launch: {clash}")
        doc["jobs"] += extra
        doc["_comment"] += f" Extra jobs from {os.path.basename(path)}: " + ", ".join(j["name"] for j in extra) + "."
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print(doc["_comment"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
