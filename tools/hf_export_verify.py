"""Verify HF export using exact greedy token IDs and raw output logits.

The same encoded prompt and HF continuation are teacher-forced through mtstudio
score. Its per-position argmax must match every HF next-token ID; raw logits
must also match within the recorded numerical tolerance. No text comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def hashes(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def snapshot_provenance(hf_dir, out_dir):
    """Parse exactly the captured bytes, then bind them to the later input snapshot."""
    provenance_path = Path(hf_dir) / "export_receipt.json"
    provenance_bytes = provenance_path.read_bytes()
    events = json.loads(provenance_bytes).get("events")
    events_path = Path(events) if events else Path(out_dir) / "events.jsonl"
    events_bytes = events_path.read_bytes()
    records = [json.loads(line) for line in events_bytes.decode("utf-8").splitlines()
               if line.lstrip().startswith("{")]
    try:
        recorded = next(row for row in records if row.get("event") == "model")
        name = next(row["name"] for row in records if row.get("event") == "start")
    except StopIteration as error:
        raise ValueError("missing recorded model/start event") from error
    return recorded, Path(out_dir) / (name + ".safetensors"), events_path, {
        str(provenance_path.resolve()): hashlib.sha256(provenance_bytes).hexdigest(),
        str(events_path.resolve()): hashlib.sha256(events_bytes).hexdigest()}


def assess(hf_ids, mt, hf_logits, mt_logits, atol, rtol):
    import numpy as np
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("finite nonnegative tolerances required")
    if mt.get("over_ctx") or mt.get("argmax") != hf_ids:
        raise ValueError("greedy token-ID parity failed")
    if (hf_logits.ndim != 2 or not hf_logits.size or hf_logits.shape != mt_logits.shape
            or hf_logits.shape[0] != len(hf_ids) or not np.isfinite(hf_logits).all()
            or not np.isfinite(mt_logits).all()):
        raise ValueError("invalid output logits")
    delta = float(np.max(np.abs(hf_logits - mt_logits)))
    if not np.allclose(hf_logits, mt_logits, atol=atol, rtol=rtol):
        raise ValueError(f"output logits differ (max absolute error {delta})")
    return delta


def publish_evidence(path, receipt, hf_logits, mt_logits, banned):
    """Retain rows for offline replay; never overwrite an existing receipt or evidence."""
    import numpy as np
    path = Path(path)
    evidence = path.with_name(path.name + ".evidence")
    if path.exists():
        raise FileExistsError(path)
    evidence.mkdir()  # exclusive reservation, including concurrent writers
    raw = evidence / "logits.npz"
    temporary = None
    try:
        with raw.open("xb") as stream:
            np.savez(stream, hf_logits=hf_logits, mt_logits=mt_logits,
                     banned=np.asarray(banned, dtype=np.int64))
        value = dict(receipt, evidence={"schema": 1, "file": evidence.name + "/logits.npz",
                     "bytes": raw.stat().st_size,
                     "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()})
        serialized = json.dumps(value, indent=1, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + "-", suffix=".incoming",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic complete-file publication, fails if destination exists
        return value
    except BaseException:
        try:
            published = temporary is not None and path.exists() and path.samefile(temporary)
        except OSError:
            published = True  # uncertain publication must preserve evidence
        if not published:
            if raw.exists():
                raw.unlink()
            evidence.rmdir()
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replay_evidence(path):
    """Recompute numerical and masked greedy gates without Torch, HF, or model calls."""
    import numpy as np
    path = Path(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    record = receipt.get("evidence", {})
    expected = path.name + ".evidence/logits.npz"
    if record.get("schema") != 1 or record.get("file") != expected:
        raise ValueError("missing or invalid retained row evidence")
    raw = path.parent / expected
    if (raw.stat().st_size != record.get("bytes")
            or hashlib.sha256(raw.read_bytes()).hexdigest() != record.get("sha256")):
        raise ValueError("retained row evidence hash/size mismatch")
    with np.load(raw, allow_pickle=False) as arrays:
        if set(arrays.files) != {"hf_logits", "mt_logits", "banned"}:
            raise ValueError("unexpected evidence inventory")
        hf, mt, banned = arrays["hf_logits"], arrays["mt_logits"], arrays["banned"]
    if hf.dtype != np.dtype("float32") or mt.dtype != np.dtype("float32"):
        raise ValueError("expected retained float32 rows")
    ids = receipt["continuation_ids"]
    native_ids = receipt["mtstudio_argmax"]
    if (not isinstance(ids, list) or any(type(i) is not int for i in ids)
            or not isinstance(native_ids, list) or any(type(i) is not int for i in native_ids)
            or banned.ndim != 1 or banned.dtype.kind not in "iu"):
        raise ValueError("invalid retained token IDs")
    delta = assess(ids, {"argmax": native_ids}, hf, mt,
                   receipt["atol"], receipt["rtol"])
    if (any(i < 0 or i >= hf.shape[1] for i in ids)
            or any(i < 0 or i >= hf.shape[1] for i in banned)
            or len(set(banned.tolist())) >= hf.shape[1]):
        raise ValueError("invalid banned/generated token range")
    for rows in (hf, mt):
        ranked = rows.copy()
        ranked[:, banned] = -np.inf
        if np.argmax(ranked, axis=1).tolist() != ids:
            raise ValueError("raw masked greedy IDs differ")
    if receipt.get("status") != "HF-PARITY-OK" or delta != receipt["max_abs_logit_error"]:
        raise ValueError("recorded parity result differs from retained rows")
    return {"status": "HF-PARITY-REPLAY-OK", "rows": len(ids),
            "max_abs_logit_error": delta, "scope": "retained rows and IDs; no model execution or source replay"}


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
    ap.add_argument("hf_dir", type=Path, nargs="?")
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--replay", type=Path, help="replay retained raw rows without loading either model")
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
    if args.replay:
        if args.hf_dir or args.out_dir or args.receipt:
            ap.error("--replay accepts no model directories or receipt output")
        print(json.dumps(replay_evidence(args.replay), indent=1))
        return 0
    if not args.hf_dir or not args.out_dir:
        ap.error("hf_dir and out_dir are required for model verification")
    if args.tokens < 1 or any(not math.isfinite(x) or x < 0 for x in (args.atol, args.rtol)):
        ap.error("positive token count and nonnegative tolerances required")
    if args.receipt and args.receipt.exists():
        ap.error("receipt already exists")
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    spec = resolve_spec(args.out_dir, args.spec)
    from safetensors.numpy import load_file
    from hf_export import map_weights
    # The original recorded model/weights must map bit-for-bit to the export.
    # Mixed logs may be passed as the export source, so the export records that path.
    recorded, source, events_path, early_hashes = snapshot_provenance(args.hf_dir, args.out_dir)
    source_gguf = source.with_suffix(".gguf")
    evidence_inputs = [Path(__file__), Path(__file__).with_name("hf_export.py"),
                       Path(__file__).with_name("llama3_bpe.py"), Path(__file__).with_name("bpe.py"),
                       source_gguf, source, spec, events_path,
                       Path(args.mtstudio), *sorted(p for p in args.hf_dir.iterdir() if p.is_file())]
    before = hashes(evidence_inputs)
    if any(before[path] != value for path, value in early_hashes.items()):
        raise ValueError("verification provenance changed while reading inputs")
    tok = Tokenizer.from_file(str(args.hf_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, attn_implementation="eager")
    model.eval()
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
    else:
        expected_tied = recorded["family"] == "flex" and recorded.get("tie_head", False)
        if set(loaded) != set(mapped) or model.config.tie_word_embeddings is not expected_tied:
            raise ValueError("unexpected HF tensor inventory or output-head tying")
        if expected_tied and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
            raise ValueError("HF did not retain shared input/output parameters")
    prompt = tok.encode(args.prompt, add_special_tokens=False).ids
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
    if hashes(evidence_inputs) != before:
        raise ValueError("verification input changed during execution; no receipt published")
    receipt = {"status": "HF-PARITY-OK", "prompt_ids": prompt, "continuation_ids": generated,
               "mtstudio_argmax": mt["argmax"], "max_abs_logit_error": delta,
               "atol": args.atol, "rtol": args.rtol, "spec": str(spec),
               "scope": "raw token IDs, source tensor mapping and output logits; not chat rendering",
               "score_spec_delta": delta_spec, "mapped_tensors": len(mapped),
               "source_prompt_tokenizer_ids_checked": True,
               "versions": {"torch": torch.__version__},
               "sha256": before, "verification_inputs_unchanged": True}
    if args.receipt:
        receipt = publish_evidence(args.receipt, receipt, hf_logits, mt_logits, banned)
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
