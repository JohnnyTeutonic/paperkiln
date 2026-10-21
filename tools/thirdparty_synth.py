#!/usr/bin/env python3
"""Synthetic third-party gate (LOCAL_LLM_STACK.md section 7), for graph features no small official
checkpoint exercises or that we cannot download: a tiny randomly initialised HF transformers model of
the family, converted by llama.cpp's own convert_hf_to_gguf.py (f32; a placeholder vocabulary, since
the graph is under test, not tokenisation), then tools/gguf_ir.py's IR on tools/ir_numpy.py against
HF's PyTorch forward of the same random weights (eager attention, f32) on random token sequences
longer than the attention window.

    python3 tools/thirdparty_synth.py --family mistral_swa|gemma3 --llamacpp ~/llamacpp_b6000 \
        --work DIR --out RECEIPT.jsonl [--n 8] [--len 48] [--max-new 16] [--seed 0]

Per sequence: max |dlogit| over the range at every position (rel), argmax agreement at every
position, and greedy continuations (IR recomputed from position 0 each step vs the argmax of HF's
cached forward, thirdparty_parity.hf_greedy).
Also, for gemma3, the same check with the norm weights stored as HF's raw w and weight_offset 1 on
every rmsnorm (the load-time transform). Writes RECEIPT.jsonl (one row per sequence),
RECEIPT.summary.json and RECEIPT.ir.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402
import thirdparty_parity  # noqa: E402

VOCAB = 512


def hf_model(family, seed):
    import torch
    torch.manual_seed(seed)
    if family == "mistral_swa":
        from transformers import MistralConfig, MistralForCausalLM
        cfg = MistralConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                            sliding_window=8, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
                            initializer_range=0.2, attn_implementation="eager")
        model = MistralForCausalLM(cfg)
    elif family == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        cfg = Gemma3TextConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=32, query_pre_attn_scalar=32,
                               max_position_embeddings=256, sliding_window=8, rms_norm_eps=1e-6, rope_theta=1e6,
                               rope_local_base_freq=10000.0, rope_scaling=None, final_logit_softcapping=None,
                               attn_logit_softcapping=None, tie_word_embeddings=True, initializer_range=0.2,
                               attn_implementation="eager")
        model = Gemma3ForCausalLM(cfg)
    else:
        raise SystemExit(f"thirdparty_synth: unknown family {family}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:  # non-trivial norm weights (gemma's init is 0, llama's 1)
                p.add_(0.2 * torch.randn_like(p))
    return model.float().eval()


def convert(model, family, llamacpp, work):
    """save_pretrained, then llama.cpp's convert_hf_to_gguf.py --outtype f32, its set_vocab replaced
    by a placeholder vocabulary of VOCAB tokens (our tiny models have no tokenizer)."""
    hf_dir = os.path.join(work, "hf")
    model.save_pretrained(hf_dir, safe_serialization=True)
    out = os.path.join(work, f"{family}.gguf")
    sys.path.insert(0, llamacpp)
    sys.path.insert(0, os.path.join(llamacpp, "gguf-py"))
    import convert_hf_to_gguf as conv
    import gguf

    def placeholder_vocab(self):
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list([f"<t{i}>" for i in range(VOCAB)])
        self.gguf_writer.add_token_scores([0.0] * VOCAB)
        self.gguf_writer.add_token_types([gguf.TokenType.NORMAL] * VOCAB)

    cls = conv.ModelBase.from_model_architecture(model.config.architectures[0])
    cls.set_vocab = placeholder_vocab
    if family == "mistral_swa":
        # llama.cpp's convert drops Mistral's sliding_window; write it as <arch>.attention.sliding_window.
        base = cls.set_gguf_parameters

        def with_window(self):
            base(self)
            self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        cls.set_gguf_parameters = with_window
    old = sys.argv
    sys.argv = ["convert_hf_to_gguf.py", hf_dir, "--outtype", "f32", "--outfile", out]
    try:
        conv.main()
    finally:
        sys.argv = old
    return out


def compare(doc, W, model, seqs, max_new):
    import torch
    rows = []
    for k, ids in enumerate(seqs):
        ref = model(torch.tensor([ids])).logits[0].numpy()
        mine = ir_numpy.forward(doc, W, ids)
        rng = (ref.max(axis=1) - ref.min(axis=1))
        rel = float((np.abs(ref - mine).max(axis=1) / rng).max())
        top1 = int((ref.argmax(axis=1) == mine.argmax(axis=1)).sum())
        _, hf_new = thirdparty_parity.hf_greedy(model, ids, max_new, set())  # plain argmax, no generate
        seq, ir_new = list(ids), []
        for _ in range(max_new):
            t = int(ir_numpy.forward(doc, W, seq, last_only=True)[-1].argmax())
            ir_new.append(t)
            seq.append(t)
        rows.append({"i": k, "len": len(ids), "max_rel_all_positions": rel, "top1_same": top1, "positions": len(ids),
                     "greedy_same": hf_new == ir_new, "first_diff": next((j for j, (a, b) in enumerate(zip(hf_new, ir_new))
                                                                           if a != b), None)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True, choices=("mistral_swa", "gemma3"))
    ap.add_argument("--llamacpp", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--len", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    torch.set_grad_enabled(False)
    os.makedirs(a.work, exist_ok=True)
    model = hf_model(a.family, a.seed)
    path = convert(model, a.family, os.path.expanduser(a.llamacpp), a.work)
    meta, shapes = gguf_ir.read_meta(path)
    doc = gguf_ir.build(meta, shapes)
    json.dump(doc, open(a.out + ".ir.json", "w", encoding="utf-8"), indent=1)
    W = ir_numpy.load_weights(path)
    rng = np.random.default_rng(a.seed)
    seqs = [rng.integers(0, VOCAB, size=a.len).tolist() for _ in range(a.n)]
    rows = compare(doc, W, model, seqs, a.max_new)
    for r in rows:
        r["variant"] = "gguf"
    summ = {"family": a.family, "gguf_arch": meta["general.architecture"], "sequences": len(rows), "len": a.len,
            "worst_rel_all_positions": max(r["max_rel_all_positions"] for r in rows),
            "top1_same": sum(r["top1_same"] for r in rows), "positions": sum(r["positions"] for r in rows),
            "greedy_same": sum(r["greedy_same"] for r in rows), "max_new": a.max_new, "seed": a.seed}
    win = [n for n in doc["block"]["nodes"] if n["op"] == "attention"][0]["window"]
    summ["window"] = win
    # The window must matter: the same IR with every window off has to disagree with HF.
    off = ir_numpy.forward(doc, W, seqs[0], no_window=True)
    ref0 = model(torch.tensor([seqs[0]])).logits[0].numpy()
    summ["window_off_rel"] = float((np.abs(ref0 - off).max(axis=1) / (ref0.max(axis=1) - ref0.min(axis=1))).max())
    if a.family == "gemma3":
        W2, doc2 = dict(W), json.loads(json.dumps(doc))
        for part in ("block", "head"):
            for nd in doc2[part]["nodes"]:
                if nd["op"] == "rmsnorm":
                    nd["weight_offset"] = 1.0
        for k in W2:
            if k.endswith("norm.weight"):
                W2[k] = (W[k] - np.float32(1)).astype(np.float32)
        rows2 = compare(doc2, W2, model, seqs[:2], a.max_new)
        for r in rows2:
            r["variant"] = "raw_w_weight_offset_1"
        rows += rows2
        summ["weight_offset_rel"] = max(r["max_rel_all_positions"] for r in rows2)
        summ["weight_offset_greedy_same"] = sum(r["greedy_same"] for r in rows2)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
