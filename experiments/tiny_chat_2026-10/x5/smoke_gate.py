#!/usr/bin/env python3
"""Ember gate files for X5's Stage 0 smoke model (amendment 5, A5.4): the arm's shape on the real 16k vocabulary.

    python3 experiments/tiny_chat_2026-10/x5/smoke_gate.py --smoke ART/s0_x5/smoke_20m_X5 \
        --vocab /mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf --mtstudio BIN --out GATE_DIR

Prompts: the target user turn of 40 Carry-5 items (every 26th, the post jobs' spot-check items), rendered in the
served chat template with gold history and tokenised by the model's own GGUF vocabulary. keramos's numpy
interpreter decodes 8 greedy tokens per prompt over the F32 GGUF (id 0 excluded, as `mtstudio score`'s argmax);
`mtstudio score` scores the same ids. Written: the GGUF, its IR, score_in.jsonl, score_out.jsonl, the local spec
and manifest.json (sha256s, numpy vs mtstudio worst log-prob error over max(1, |ref|), greedy agreement), the
layout of tests/test_llama_attnres_ir.py's ATTNRES_GATE_DIR.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))
import carry5_depth as cd  # noqa: E402
import carry5_eval as ce  # noqa: E402
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=8)
    a = ap.parse_args(argv)
    smoke, out = Path(a.smoke), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((smoke / f"{smoke.name}_spec.json").read_text())
    name = spec["name"]
    g = smoke / f"{name}.gguf"
    spec = {**spec, "out_dir": str(smoke), "data": {**spec["data"], "vocab": a.vocab}}
    sp = out / "spec.json"
    sp.write_text(json.dumps(spec, indent=1))
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(str(g))
    W = {t[0]: np.frombuffer(raw, dtype=np.float32, count=int(np.prod(t[1])), offset=data_start + t[3])
         .reshape(list(reversed(t[1]))) for t in tensors}
    doc = json.loads(meta["paperkiln.ir"][1])
    tok = ce.Tok(str(g))
    items = ce.read_jsonl(str(ROOT / "corpus" / "eval" / "carry5_v1.jsonl"))[::26][:40]
    rows, cont = [], []
    for it in items:
        hist, user = cd.history_of(it, "gold")
        p = tok.encode(ce.template(hist, user))
        ids, outs, lps = list(p), [], []
        for _ in range(a.new):
            lg = ir_numpy.forward(doc, W, ids, last_only=True)[-1].astype(np.float64)
            ban = lg.copy()
            ban[0] = -np.inf
            t = int(np.argmax(ban))
            lps.append(lg[t] - (lg.max() + np.log(np.exp(lg - lg.max()).sum())))
            outs.append(t)
            ids.append(t)
        rows.append({"prompt_ids": p, "continuation_ids": outs})
        cont.append((outs, lps))
    req, res = out / "score_in.jsonl", out / "score_out.jsonl"
    req.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = subprocess.run([a.mtstudio, "score", str(sp), "--in", str(req), "--out", str(res)], capture_output=True,
                       text=True, timeout=3600)
    if r.returncode != 0:
        raise SystemExit(f"mtstudio score failed: {r.stderr[-2000:]}")
    got = [json.loads(x) for x in res.read_text().splitlines() if x.strip()]
    worst, agree = 0.0, 0
    for (outs, lps), gg in zip(cont, got):
        ref = np.array(gg["logprobs"])
        worst = max(worst, float(np.max(np.abs(np.array(lps) - ref)) / max(1.0, float(np.max(np.abs(ref))))))
        agree += gg["argmax"] == outs
    for f in (g, smoke / f"{name}.ir.json"):
        shutil.copyfile(f, out / f.name)
    files = [g.name, f"{name}.ir.json", "score_in.jsonl", "score_out.jsonl", "spec.json"]
    man = {"model": name, "family_hint": doc.get("family_hint"), "ir_version": doc.get("ir_version"),
           "arch_custom": spec["arch"]["custom"], "train": spec["train"], "vocab": a.vocab,
           "prompts": "Carry-5 items 0, 26, ..., 1014: the target user turn with gold history, served template",
           "numpy_vs_mtstudio": {"worst": worst, "agree": agree, "n": len(got)},
           "files": {f: sha256(out / f) for f in files},
           "_note": "score_in.jsonl: prompt_ids and keramos ir_numpy's greedy continuation over the F32 GGUF; "
                    "score_out.jsonl: mtstudio score over the same ids. Ember gate: greedy n/40 and log-probabilities "
                    "within ROUNDING on CPU, CUDA and WASM."}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["numpy_vs_mtstudio"]))
    return 0 if agree == len(got) and worst <= 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
