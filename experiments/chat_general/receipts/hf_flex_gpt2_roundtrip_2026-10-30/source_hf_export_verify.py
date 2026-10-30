"""Verify HF export using exact greedy token IDs and raw output logits.

The same encoded prompt and HF continuation are teacher-forced through mtstudio
score. Its per-position argmax must match every HF next-token ID; raw logits
must also match within the recorded numerical tolerance. No text comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def resolve_spec(out_dir, explicit=None):
    if explicit:
        return Path(explicit).resolve()
    root = Path(out_dir)
    candidates = [p for p in (root / "spec.json", root / "sft_spec.json") if p.is_file()]
    if len(candidates) != 1:
        raise ValueError("spec is absent or ambiguous; pass --spec explicitly")
    return candidates[0].resolve()


def assess(hf_ids, mt, hf_logits, mt_logits, atol, rtol):
    import numpy as np
    if mt.get("over_ctx") or mt.get("argmax") != hf_ids:
        raise ValueError("greedy token-ID parity failed")
    if hf_logits.shape != mt_logits.shape or not np.isfinite(mt_logits).all():
        raise ValueError("invalid mtstudio output logits")
    delta = float(np.max(np.abs(hf_logits - mt_logits)))
    if not np.allclose(hf_logits, mt_logits, atol=atol, rtol=rtol):
        raise ValueError(f"output logits differ (max absolute error {delta})")
    return delta


def source_prompt_ids(source_gguf, text):
    """Use an independent supported source encoder; never echo exported IDs."""
    from llama3_bpe import from_gguf, read_gguf_kv
    kind = read_gguf_kv(str(source_gguf), {"tokenizer.ggml.model"}).get("tokenizer.ggml.model")
    if kind == "gpt2":
        return from_gguf(str(source_gguf)).encode(text, add_special_tokens=False).ids
    if kind == "bpe":
        import bpe
        return bpe.read_gguf(str(source_gguf)).encode(text)
    if kind in (None, "word", "wordlevel"):
        from hf_export import legacy_word_vocab, legacy_word_ids, legacy_word_tokenizer
        tokens = legacy_word_vocab(source_gguf)
        legacy_word_tokenizer(tokens)  # same explicit supported-vocabulary contract
        return legacy_word_ids(tokens, text)
    raise ValueError("source tokenizer is unsupported for independent token-ID parity: " + repr(kind))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--spec")
    ap.add_argument("--prompt", default="once upon a time")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--mtstudio", default=os.path.expanduser("~/mtrel/mtstudio"))
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--score-without-chat-template", action="store_true",
                    help="omit only the unused chat field when scoring raw token IDs on older mtstudio")
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    args = ap.parse_args()
    if args.tokens < 1 or args.atol < 0 or args.rtol < 0:
        ap.error("positive token count and nonnegative tolerances required")
    if args.receipt and args.receipt.exists():
        ap.error("receipt already exists")
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    spec = resolve_spec(args.out_dir, args.spec)
    tok = Tokenizer.from_file(str(args.hf_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, attn_implementation="eager")
    model.eval()
    from safetensors.numpy import load_file
    from hf_export import map_weights, model_event, start_name
    # The original recorded model/weights must map bit-for-bit to the export.
    # Mixed logs may be passed as the export source, so the export records that path.
    provenance = json.loads((args.hf_dir / "export_receipt.json").read_text())
    events = provenance.get("events")
    recorded = model_event(args.out_dir, events)
    source = args.out_dir / (start_name(args.out_dir, events) + ".safetensors")
    mapped = map_weights(load_file(str(source)), recorded)
    exported = load_file(str(args.hf_dir / "model.safetensors"))
    if set(mapped) != set(exported) or any(not np.array_equal(w, exported[k]) for k, w in mapped.items()):
        raise ValueError("exported tensor mapping differs from original source weights")
    loaded = model.state_dict()
    for k, w in mapped.items():
        if k not in loaded or not np.array_equal(w, loaded[k].float().numpy()):
            raise ValueError("HF loaded tensor differs from mapped source: " + k)
    if recorded["family"] == "llama":
        allowed = set(mapped) | {"lm_head.weight"}
        if set(loaded) != allowed or not np.array_equal(loaded["lm_head.weight"].numpy(), loaded["model.embed_tokens.weight"].numpy()):
            raise ValueError("unexpected HF tensor inventory or untied Llama output head")
    elif set(loaded) != set(mapped) or model.config.tie_word_embeddings:
        raise ValueError("unexpected HF GPT2 tensor inventory or tied output head")
    prompt = tok.encode(args.prompt, add_special_tokens=False).ids
    source_gguf = source.with_suffix(".gguf")
    source_ids = source_prompt_ids(source_gguf, args.prompt)
    if source_ids != prompt:
        raise ValueError("exported prompt tokenizer IDs differ from source GGUF")
    if not prompt or len(prompt) + args.tokens > model.config.max_position_embeddings:
        raise ValueError("empty prompt or continuation exceeds the recorded context")
    vocab = tok.get_vocab()
    banned = [vocab[t] for t in ("<unk>", "<s>", "</s>", "<pad>") if t in vocab]
    ids, generated, logits_rows = list(prompt), [], []
    with torch.no_grad():
        for _ in range(args.tokens):
            logits = model(torch.tensor([ids]), use_cache=False).logits[0, -1].float()
            logits_rows.append(logits.numpy().copy())
            ranked = logits.clone()
            ranked[banned] = -float("inf")
            nxt = int(torch.argmax(ranked))
            generated.append(nxt)
            ids.append(nxt)
    with tempfile.TemporaryDirectory(prefix="pk_hf_verify_") as tmp:
        root = Path(tmp)
        request, response, raw = root / "in.jsonl", root / "out.jsonl", root / "logits.f32"
        request.write_text(json.dumps({"prompt_ids": prompt, "continuation_ids": generated}) + "\n")
        original_spec = json.loads(spec.read_text())
        delta_spec = {}
        score_spec = spec
        if args.score_without_chat_template:
            scored_spec = dict(original_spec)
            removed = scored_spec.pop("chat", None)
            delta_spec = {"removed_chat": removed}
            score_spec = root / "numeric_spec.json"
            score_spec.write_text(json.dumps(scored_spec))
        cmd = [args.mtstudio, "score", str(score_spec), "--in", str(request), "--out", str(response), "--logits", str(raw)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise ValueError("mtstudio score failed: " + result.stderr[-4000:])
        rows = response.read_text().splitlines()
        if len(rows) != 1:
            raise ValueError("expected exactly one mtstudio score record")
        mt = json.loads(rows[0])
        hf_logits = np.asarray(logits_rows)
        mt_logits = np.fromfile(raw, dtype="<f4").reshape(hf_logits.shape)
        delta = assess(generated, mt, hf_logits, mt_logits, args.atol, args.rtol)
    receipt = {"status": "HF-PARITY-OK", "prompt_ids": prompt, "continuation_ids": generated,
               "mtstudio_argmax": mt["argmax"], "max_abs_logit_error": delta,
               "atol": args.atol, "rtol": args.rtol, "spec": str(spec),
               "scope": "raw token IDs, source tensor mapping and output logits; not chat rendering",
               "score_spec_delta": delta_spec, "mapped_tensors": len(mapped),
               "source_prompt_tokenizer_ids_checked": True,
               "versions": {"torch": torch.__version__},
               "sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), Path(__file__).with_name("hf_export.py"),
                                    Path(__file__).with_name("llama3_bpe.py"), source_gguf, source, spec, Path(events) if events else args.out_dir / "events.jsonl", args.hf_dir / "model.safetensors", args.hf_dir / "config.json",
                                    args.hf_dir / "tokenizer.json", Path(args.mtstudio)]}}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
