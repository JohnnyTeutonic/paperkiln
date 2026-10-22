"""ember runs flex's expr norm from the IR: a tiny flex model with Primer's
modified layer norm (x(x - mu) in place of (x - mu)^2, as the row expression
of norm "expr:<e>", times a learned gain plus a bias), rope positions and a
squared-ReLU FFN trains on the CPU (mtstudio), exports a GGUF whose IR has
the expr norm nodes, and ember's CPU server answers the 40 frozen prompts
greedily; the replies must equal mtstudio /chat's 40/40.

Reuses tools/arxiv_to_chat.py's stages (specs, training, export, the /chat
parity harness), without an extraction: the spec is written here.

    python3 expr_norm_gate.py RUN_DIR --mtstudio ... --ember-bin ... --ember-www ...
        --corpus ... --vocab ... --sft-examples ... --prompts ...
"""
import argparse
import json
import os
import sys

MT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(MT, "tools"))
import arxiv_to_chat as A  # noqa: E402

PRIMER_NORM = "(x - rowmean(x)) / sqrt(abs(rowmean(x * (x - rowmean(x)))))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--ember-bin", required=True)
    ap.add_argument("--ember-www", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--sft-examples", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--pretrain-steps", type=int, default=400)
    ap.add_argument("--sft-steps", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    x = ap.parse_args()
    a = argparse.Namespace(
        arxiv_id="expr_norm_gate", vocab=x.vocab, vocab_cap=8192, T=64, corpus=x.corpus, max_tokens=400000,
        pretrain_steps=x.pretrain_steps, sft_steps=x.sft_steps, lr=3e-3, sft_lr=1e-3, batch=4, seed=1,
        checkpoint_every=100, lr_warmup=50, sft_lr_warmup=20, lr_decay="cosine", sft_examples=x.sft_examples,
        sft_limit=2000, mtstudio=x.mtstudio, threads=x.threads, prompts=x.prompts, parity_n=40, port_mts=8121,
        port_engine=8122, engine="ember", ember_bin=x.ember_bin, ember_www=x.ember_www, max_new=40,
        chat=["Hello! What is your name?"], chat_greedy=True)
    run = os.path.abspath(x.run_dir)
    os.makedirs(run, exist_ok=True)
    arch = {"custom": {"norm": "expr:" + PRIMER_NORM, "activation": "expr:relu(x)^2", "position": "rope",
                       "residual": "residual", "attention": "exact", "d": 128, "layers": 2, "heads": 4}}
    name, pre, sft = A.make_specs(a, run, "flex", arch)
    for which, s in (("pretrain", pre), ("sft", sft)):
        p = os.path.join(run, f"{which}_spec.json")
        if not os.path.exists(p):
            A.write_json(p, s)
    A.stage_train(a, run, "pretrain", name)
    A.stage_train(a, run, "sft", name)
    gguf = A.stage_export(a, run, name)
    ir = A.read_json(os.path.join(run, "export", name + ".ir.json"))
    parity, turns = A.stage_chat(a, run, name, gguf, {})
    parity["ember_bin"], parity["ember_bin_sha256"] = a.ember_bin, A.sha256(a.ember_bin)
    out = {"model": "flex d128 L2 H4 T64: norm expr (Primer) * gain + bias, rope, squared ReLU",
           "norm_expr": PRIMER_NORM, "gguf_sha256": A.sha256(gguf),
           "block_ops": [n["op"] for n in ir["block"]["nodes"]], "head_ops": [n["op"] for n in ir["head"]["nodes"]],
           "sft_fit": A.read_json(os.path.join(run, "sft_fit.json")),
           "pretrain": A.read_json(os.path.join(run, "pretrain", "result.json")),
           "sft": A.read_json(os.path.join(run, "sft", "result.json")), "parity": parity, "chat": turns}
    A.write_json(os.path.join(run, "gate.json"), out)
    print(json.dumps({k: out[k] for k in ("block_ops", "head_ops")}), "\nparity pass:", parity["pass"])
    return 0 if parity["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
