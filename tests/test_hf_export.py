"""HF export preserves tokenizer semantics and verifies IDs plus output tensors."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
import hf_export
import hf_export_verify as verify


def test_weight_mapping_transposes_and_permutes_each_head():
    q = np.arange(32).reshape(4, 8).astype(np.float32)
    source = {"layers.0.self_attn.q_proj.weight": q,
              "layers.0.mlp.down_proj.weight": np.arange(12).reshape(3, 4),
              "norm.weight": np.ones((1, 4)), "embed_tokens.weight": np.ones((7, 4))}
    got = hf_export.convert_weights(source, 2)
    expected = q.T.reshape(2, 4, 4)[:, [0, 2, 1, 3], :].reshape(8, 4)
    np.testing.assert_array_equal(got["model.layers.0.self_attn.q_proj.weight"], expected)
    np.testing.assert_array_equal(got["model.layers.0.mlp.down_proj.weight"], source["layers.0.mlp.down_proj.weight"].T)
    assert got["model.norm.weight"].shape == (4,)
    np.testing.assert_array_equal(got["model.embed_tokens.weight"], source["embed_tokens.weight"])


def test_native_bpe_refused_instead_of_silent_wordlevel(tmp_path):
    import bpe
    vocab = tmp_path / "v.gguf"
    bpe.write_gguf(str(vocab), bpe.train("hello", 262))
    with pytest.raises(ValueError, match="no faithful"):
        hf_export.write_tokenizer(tmp_path, [], str(vocab))


def test_llama_bpe_preserves_unicode_spaces_and_control_ids(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from test_llama3_tokenizer import _write_gguf, LLAMA3, SPECIALS
    from tokenizers import Regex
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["hello  café 😀\nworld" * 20], trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIALS))
    data = json.loads(tok.to_str())
    tokens = [t for t, _ in sorted(tok.get_vocab().items(), key=lambda x: x[1])]
    merges = [" ".join(m) if isinstance(m, list) else m for m in data["model"]["merges"]]
    vocab = tmp_path / "v.gguf"
    _write_gguf(vocab, tokens, merges, [2, 3, 3, 3] + [1] * (len(tokens) - 4))
    hf_export.write_tokenizer(tmp_path, tokens, str(vocab))
    exported = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    for text in ("hello  café 😀\nworld", "x<|user|>hello", " 123\t中文"):
        assert exported.encode(text).ids == tok.encode(text).ids
        assert exported.decode(exported.encode(text).ids, skip_special_tokens=False) == text


def test_verifier_resolves_real_spec_and_rejects_ambiguity(tmp_path):
    (tmp_path / "spec.json").write_text("{}")
    assert verify.resolve_spec(tmp_path) == (tmp_path / "spec.json").resolve()
    (tmp_path / "sft_spec.json").write_text("{}")
    with pytest.raises(ValueError, match="ambiguous"):
        verify.resolve_spec(tmp_path)


@pytest.mark.parametrize("mt,other", [({"argmax": [3]}, [[1., 2.]]),
                                    ({"argmax": [2]}, [[1., 3.]]),
                                    ({"argmax": [2], "over_ctx": True}, [[1., 2.]])])
def test_verifier_rejects_id_tensor_and_context_failures(mt, other):
    with pytest.raises(ValueError):
        verify.assess([2], mt, np.array([[1., 2.]]), np.array(other), 1e-5, 1e-5)


def test_verifier_accepts_exact_ids_and_output_logits():
    logits = np.array([[1., 2.]])
    assert verify.assess([2], {"argmax": [2]}, logits, logits, 1e-5, 1e-5) == 0


def test_export_uses_training_rms_epsilon(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps({"event": "start", "name": "tiny"}) + "\n" +
        json.dumps({"event": "model", "family": "llama", "rope_heads": "all", "d": 4,
                    "heads": 1, "layers": 1, "vocab": 2, "T": 16}) + "\n")
    monkeypatch.setattr(hf_export, "load_file", lambda p: {"embed_tokens.weight": np.ones((2, 4))})
    monkeypatch.setattr(hf_export, "gguf_tokens", lambda p: ["<unk>", "a"])
    monkeypatch.setattr(hf_export, "write_tokenizer", lambda *a: ({"<unk>": 0, "a": 1}, "<unk>"))
    import llama3_bpe
    monkeypatch.setattr(llama3_bpe, "read_gguf_kv", lambda *a: {})
    monkeypatch.setattr(sys, "argv", ["hf_export.py", str(run)])
    hf_export.main()
    cfg = json.loads((run / "hf/config.json").read_text())
    assert cfg["rms_norm_eps"] == 1e-5


@pytest.mark.parametrize("extra", [
    {"thirdparty": {"kv_heads": 2}}, {"kv_heads": 2}, {"qk_norm": True},
    {"rope_pairing": "neox"}, {"rope_dims": 2}, {"norm_placement": "post"},
    {"recursion": {"repeats": 2}}, {"attn_res": {"form": "full"}},
    {"moe": {"experts": 2}}, {"state_tokens": {"k": 2}},
    {"tie_embeddings": False}, {"tie_head": False}, {"rope_theta": float("nan")},
])
def test_refuse_unmapped_llama_variants(extra):
    with pytest.raises(ValueError):
        hf_export.validate_model({"heads": 4, "d": 32, **extra})


def test_nondefault_rope_theta_is_preserved():
    assert hf_export.validate_model({"heads": 4, "d": 32, "rope_theta": 500000.}) == 500000.
