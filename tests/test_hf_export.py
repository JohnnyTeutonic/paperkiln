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



def test_llama_bpe_preserves_unicode_spaces_and_control_ids(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from test_llama3_tokenizer import _write_gguf, LLAMA3, SPECIALS
    from tokenizers import Regex
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["hello  cafÃƒÂ© Ã°Å¸Ëœâ‚¬\nworld" * 20], trainers.BpeTrainer(
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


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory):
    import bpe
    from test_bpe import TEXT
    path = tmp_path_factory.mktemp("native") / "native.gguf"
    bpe.write_gguf(str(path), bpe.train(TEXT, 700))
    return path


def native_cases():
    import random
    pieces = ["a", "Z", "ab", " ", "  ", "   ", "9", "12345", "3.14", "'", "'s", "'LL",
              "!!", "...", "<unk>", "<|user|>", "<|assistant|>", "<|endoftext|>", "<|", "|>",
              "x<|assistant|>y", "\t", "\r", "\n", "\r\n", "\x0b", "\x0c", "\x00", "\x7f",
              "\u00e9", "e\u0301", "\u4e2d\u6587", "\U0001f600", "\u00a0", "\u3000", "\u2028"]
    rng = random.Random(300)
    cases = ["", "  a", " \ta", "\t a", "\n  a", "'a'", "<unk>", "x<|user|>y", "123456",
             "".join(chr(i) for i in range(128))]
    cases += ["".join(rng.choice(pieces) for _ in range(rng.randrange(1, 45))) for _ in range(2000)]
    rng = random.Random(301)
    for _ in range(1000):
        cases.append("".join(chr(rng.choice([rng.randrange(0, 128), rng.randrange(128, 0xd800),
                                           rng.randrange(0xe000, 0x110000)]))
                             for _ in range(rng.randrange(1, 80))))
    return cases


def test_native_serialized_ids_and_decoding_match_independent_reference(native_fixture, tmp_path):
    import bpe
    from tokenizers import Tokenizer
    original = bpe.read_gguf(str(native_fixture))
    hf_export.write_tokenizer(tmp_path, [], str(native_fixture))
    tok = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    for text in native_cases():
        assert tok.encode(text, add_special_tokens=False).ids == original.encode(text), repr(text)
        assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text, repr(text)
    assert 0 not in tok.encode("<unk>").ids
    assert tok.encode("<|user|>").ids == [2]


def test_native_cpp_reference_if_available(native_fixture):
    import os
    import subprocess
    import bpe
    binary = os.environ.get("BPE_ENCODE_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("set BPE_ENCODE_BIN for independent C++ parity")
    # The existing binary frames records by newline; include embedded-newline cases
    # by splitting exactly as getline does, preserving every input byte.
    stream = "\n".join(native_cases()) + "\n"
    lines = [s + "\n" for s in stream.split("\n")[:-1]]
    result = subprocess.run([binary, str(native_fixture)], input=stream.encode(), capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode()
    cpp = [list(map(int, row.split())) for row in result.stdout.decode().splitlines()]
    native = bpe.read_gguf(str(native_fixture))
    tok = hf_export.native_bpe_tokenizer(str(native_fixture))
    assert len(cpp) == len(lines)
    for text, ids in zip(lines, cpp):
        assert ids == native.encode(text) == tok.encode(text, add_special_tokens=False).ids, repr(text)


def test_native_uses_ranked_merges_not_whole_token_shortcut(tmp_path):
    import bpe
    base = [s.encode() for s in bpe.SPECIALS] + [bytes([i]) for i in range(256)]
    native = bpe.BPE(base + [b"ab", b"bc", b"abc"], [(101, 102), (102, 103), (101, 261)])
    path = tmp_path / "ranked.gguf"
    bpe.write_gguf(str(path), native)
    tok = hf_export.native_bpe_tokenizer(str(path))
    assert native.encode("abc") == [260, 103]
    assert tok.encode("abc").ids == [260, 103]


@pytest.mark.parametrize("bad", ["layout", "merge"])
def test_native_rejects_invalid_layout_or_ranked_merge(native_fixture, monkeypatch, bad):
    import bpe
    native = bpe.read_gguf(str(native_fixture))
    if bad == "layout":
        native.tokens[4] = b"wrong"
    else:
        native.merges[0] = (999999, 0)
    monkeypatch.setattr(bpe, "read_gguf", lambda _: native)
    with pytest.raises(ValueError, match="invalid native"):
        hf_export.native_bpe_tokenizer(str(native_fixture))


@pytest.mark.parametrize("kind", [None, "word", "wordlevel"])
def test_legacy_word_level_export_preserves_source_token_ids(tmp_path, monkeypatch, kind):
    # Native ASCII word encoding yields [hello, !, !, it's]; HF Whitespace
    # previously produced [UNK, UNK, it, UNK, s] for this same example.
    import llama3_bpe
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"<unk>": 0, "hello": 1, "!": 2, "it's": 3, "it": 4, "s": 5}
    old = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    old.pre_tokenizer = pre_tokenizers.Whitespace()
    assert old.encode("Hello!! it's").ids != [1, 2, 2, 3]
    monkeypatch.setattr(llama3_bpe, "read_gguf_kv", lambda *a: {"tokenizer.ggml.model": kind, "tokenizer.ggml.tokens": list(vocab)})
    hf_export.write_tokenizer(tmp_path, list(vocab), "legacy.gguf")
    assert Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode("Hello!! it's").ids == [1, 2, 2, 3]


def test_explicit_legacy_vocabulary_without_gguf_is_supported(tmp_path):
    from tokenizers import Tokenizer
    hf_export.write_tokenizer(tmp_path, ["<unk>", "hello"])
    assert Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode("HELLO").ids == [1]


@pytest.mark.parametrize("kind", ["unknown", "sentencepiece"])
def test_verifier_refuses_unsupported_source_encoder(monkeypatch, kind):
    import llama3_bpe
    monkeypatch.setattr(llama3_bpe, "read_gguf_kv", lambda *a: {"tokenizer.ggml.model": kind})
    with pytest.raises(ValueError, match="unsupported for independent token-ID parity"):
        verify.source_prompt_ids("unused.gguf", "Hello!! it's")


LEGACY_TOKENS = ["<unk>", "hello", "world", "it's", "123", "a", "z", "<|endoftext|>", "caf\u00e9", "\u201ci", "\U0001f600"] + list(".,!?;:()<>|/-_+=$[]{}") + ["\x00", "\x7f"]


def legacy_decode(tokens, ids):
    out = ""
    for i in ids:
        token = tokens[i]
        if out and not (len(token) == 1 and token in ".,!?;:)") and not out.endswith("("):
            out += " "
        out += token
    return out


@pytest.mark.parametrize("with_eos", [False, True])
def test_legacy_serialized_unicode_bytes_punctuation_controls_and_digits(tmp_path, with_eos):
    from tokenizers import Tokenizer
    import random
    tokens = [t for t in LEGACY_TOKENS if with_eos or t != "<|endoftext|>"]
    tok = hf_export.legacy_word_tokenizer(tokens)
    path = tmp_path / "legacy.json"
    tok.save(str(path))
    loaded = Tokenizer.from_file(str(path))
    cases = native_cases() + ["HELLO!! it's", "hello<|endoftext|>world", "<|ENDOFTEXT|>", "(hello,world!)", "123 12345"]
    for text in cases:
        want = hf_export.legacy_word_ids(tokens, text)
        assert loaded.encode(text, add_special_tokens=False).ids == want, repr(text)
        assert loaded.decode(want, skip_special_tokens=False) == legacy_decode(tokens, want), repr(text)
    rng = random.Random(840)
    for _ in range(1000):
        ids = [rng.randrange(len(tokens)) for _ in range(rng.randrange(0, 30))]
        assert loaded.decode(ids, skip_special_tokens=False) == legacy_decode(tokens, ids)


@pytest.mark.parametrize("bad", [[], [""], ["a", "a"], ["<unk>", "a b"], ["<unk>", "!!"]])
def test_legacy_nonrepresentable_vocabulary_remains_refused(bad):
    with pytest.raises(ValueError, match="legacy word-level HF export requires"):
        hf_export.legacy_word_tokenizer(bad)


@pytest.fixture(scope="module")
def legacy_cpp_reference(tmp_path_factory):
    import shutil
    import subprocess
    compiler = shutil.which("g++") or shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler needed for independent C-locale legacy reference")
    root = Path(__file__).resolve().parents[1]
    out = tmp_path_factory.mktemp("legacy_cpp") / "word_reference"
    result = subprocess.run([compiler, "-std=c++17", "-O1", "-I", str(root / "include"),
                             str(root / "tests/word_tokenizer_reference.cpp"), "-o", str(out)],
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return out


@pytest.mark.parametrize("with_eos", [False, True])
def test_legacy_serialized_ids_and_decode_match_cpp_bytes(tmp_path, legacy_cpp_reference, with_eos):
    import subprocess
    from tokenizers import Tokenizer
    tokens = [t for t in LEGACY_TOKENS if with_eos or t != "<|endoftext|>"]
    vocab = tmp_path / "tokens.hex"
    vocab.write_text("".join(t.encode("utf-8").hex() + "\n" for t in tokens))
    cases = native_cases() + ["HELLO!! it's", "hello<|endoftext|>world", "<|ENDOFTEXT|>", "123 12345"]
    result = subprocess.run([str(legacy_cpp_reference), str(vocab)],
                            input="".join(t.encode("utf-8").hex() + "\n" for t in cases).encode(),
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    path = tmp_path / "tokenizer.json"
    hf_export.legacy_word_tokenizer(tokens).save(str(path))
    tok = Tokenizer.from_file(str(path))
    rows = result.stdout.decode("ascii").splitlines()
    assert len(rows) == len(cases)
    for text, row in zip(cases, rows):
        encoded, decoded = row.split("\t")
        ids = [int(i) for i in encoded.split(",")] if encoded else []
        assert ids == hf_export.legacy_word_ids(tokens, text) == tok.encode(text).ids, repr(text)
        assert tok.decode(ids, skip_special_tokens=False).encode("utf-8").hex() == decoded, repr(text)


@pytest.mark.parametrize("tokens", [[b"\xff"], ["good", b"\xc3"]])
def test_legacy_invalid_utf8_source_tokens_never_silently_replace(monkeypatch, tokens):
    import llama3_bpe
    monkeypatch.setattr(llama3_bpe, "read_gguf_kv", lambda *a: {"tokenizer.ggml.tokens": tokens})
    with pytest.raises(ValueError, match="valid UTF-8 GGUF token strings"):
        hf_export.legacy_word_vocab("unused.gguf")



def test_legacy_auto_tokenizer_reload_keeps_original_spacing_and_control_semantics(tmp_path):
    transformers = pytest.importorskip("transformers")
    hf_export.write_tokenizer(tmp_path, LEGACY_TOKENS)
    tok = transformers.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    for text in ["HELLO!! it's", "hello<|endoftext|>world", "<|ENDOFTEXT|>", "\u4e2d\U0001f600", "\x00\x7f", "(hello,world!)"]:
        ids = hf_export.legacy_word_ids(LEGACY_TOKENS, text)
        assert tok.encode(text, add_special_tokens=False) == ids
        assert tok.decode(ids, skip_special_tokens=False) == legacy_decode(LEGACY_TOKENS, ids)
    assert tok.clean_up_tokenization_spaces is False



def test_legacy_arbitrary_generated_ids_decode_like_cpp(tmp_path, legacy_cpp_reference):
    import random
    import subprocess
    tokens = LEGACY_TOKENS
    vocab = tmp_path / "tokens.hex"
    vocab.write_text("".join(t.encode("utf-8").hex() + "\n" for t in tokens))
    rng = random.Random(841)
    sequences = [[rng.randrange(len(tokens)) for _ in range(rng.randrange(0, 40))] for _ in range(1000)]
    result = subprocess.run([str(legacy_cpp_reference), str(vocab)],
                            input="".join("d:" + ",".join(map(str, ids)) + "\n" for ids in sequences).encode(),
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    tok = hf_export.legacy_word_tokenizer(tokens)
    for ids, row in zip(sequences, result.stdout.decode("ascii").splitlines(), strict=True):
        assert tok.decode(ids, skip_special_tokens=False).encode("utf-8").hex() == row.split("\t")[1]



def gpt2_fixture():
    model = {"family": "gpt2", "d": 4, "heads": 2, "layers": 2, "vocab": 7, "T": 16}
    state = {"wte.weight": np.arange(28, dtype=np.float32).reshape(7, 4),
             "wpe.weight": np.arange(64, dtype=np.float32).reshape(16, 4),
             "head.weight": np.arange(28, dtype=np.float32).reshape(4, 7),
             "ln_f.weight": np.ones((1, 4), dtype=np.float32), "ln_f.bias": np.zeros((1, 4), dtype=np.float32)}
    for b in range(2):
        for module, layer, dim_in, dim_out in (("attn", "c_attn", 4, 12), ("attn", "c_proj", 4, 4),
                                               ("mlp", "c_fc", 4, 16), ("mlp", "c_proj", 16, 4)):
            state[f"{module}_{b}.{layer}.weight"] = np.arange(dim_in*dim_out, dtype=np.float32).reshape(dim_in, dim_out)
            state[f"{module}_{b}.{layer}.bias"] = np.arange(dim_out, dtype=np.float32).reshape(1, dim_out)
        for norm in ("ln1", "ln2"):
            state[f"{norm}_{b}.weight"] = np.ones((1, 4), dtype=np.float32)
            state[f"{norm}_{b}.bias"] = np.zeros((1, 4), dtype=np.float32)
    return model, state


def test_gpt2_mapping_keeps_conv1d_layout_and_transposes_only_untied_head():
    model, state = gpt2_fixture()
    mapped = hf_export.map_weights(state, model)
    assert len(mapped) == 29
    np.testing.assert_array_equal(mapped["transformer.h.0.attn.c_attn.weight"], state["attn_0.c_attn.weight"])
    np.testing.assert_array_equal(mapped["transformer.h.1.mlp.c_proj.weight"], state["mlp_1.c_proj.weight"])
    np.testing.assert_array_equal(mapped["lm_head.weight"], state["head.weight"].T)
    np.testing.assert_array_equal(mapped["transformer.h.0.attn.c_attn.bias"], state["attn_0.c_attn.bias"].reshape(-1))
    assert mapped["transformer.h.0.ln_1.weight"].shape == (4,)
    assert mapped["transformer.ln_f.bias"].shape == (4,)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "nan"])
def test_gpt2_incomplete_or_incompatible_tensor_inventory_is_refused(change):
    model, state = gpt2_fixture()
    if change == "missing":
        del state["head.weight"]
    elif change == "extra":
        state["optimizer.adam_m"] = np.zeros((1,), dtype=np.float32)
    elif change == "shape":
        state["attn_0.c_attn.weight"] = state["attn_0.c_attn.weight"].T
    else:
        state["head.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="tensor"):
        hf_export.map_weights(state, model)


@pytest.mark.parametrize("extra", [
    {"attention": "swa"}, {"attention": "kimi"}, {"attention": "recurrence"},
    {"norm": "rmsnorm"}, {"norm_placement": "post"}, {"final_norm": False},
    {"activation": "relu"}, {"gelu_approx": "none"}, {"position": "rope"},
    {"residual": "plain"}, {"residual": "highway"}, {"state_tokens": {"k": 4}},
    {"recursion": {"repeats": 2}}, {"attn_res": True}, {"moe": {"experts": 2}},
    {"tie_head": True}, {"tie_embeddings": True}, {"layers": 3}, {"d_ff": 8}, {"heads": 3},
    {"layer_norm_epsilon": 1e-6},
])
def test_gpt2_export_refuses_unmapped_architectures(extra):
    model, _ = gpt2_fixture()
    with pytest.raises(ValueError):
        hf_export.validate_gpt2({**model, **extra})


def test_gpt2_export_config_and_hf_loaded_tensors_are_faithful(tmp_path, monkeypatch):
    from safetensors.numpy import load_file
    transformers = pytest.importorskip("transformers")
    model, state = gpt2_fixture()
    model.update(event="model", attention="exact", activation="gelu", norm="layernorm", position="learned")
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps({"event": "start", "name": "tiny"}) + "\n" + json.dumps(model) + "\n")
    monkeypatch.setattr(hf_export, "load_file", lambda _: state)
    monkeypatch.setattr(hf_export, "gguf_tokens", lambda _: ["<unk>", "a", "b", "c", "d", "e", "f"])
    monkeypatch.setattr(hf_export, "write_tokenizer", lambda *a: ({str(i): i for i in range(7)}, "<unk>"))
    import llama3_bpe
    monkeypatch.setattr(llama3_bpe, "read_gguf_kv", lambda *a: {})
    monkeypatch.setattr(sys, "argv", ["hf_export.py", str(run)])
    hf_export.main()
    cfg = json.loads((run / "hf/config.json").read_text())
    assert cfg["activation_function"] == "gelu_new" and cfg["tie_word_embeddings"] is False
    assert cfg["resid_pdrop"] == cfg["embd_pdrop"] == cfg["attn_pdrop"] == 0
    assert cfg["n_positions"] == 16 and cfg["n_inner"] == 16
    loaded = transformers.AutoModelForCausalLM.from_pretrained(run / "hf").state_dict()
    mapped = load_file(str(run / "hf/model.safetensors"))
    assert set(loaded) == set(mapped)
    for key, value in mapped.items():
        np.testing.assert_array_equal(loaded[key].numpy(), value)


def flex_gpt2_fixture():
    """Native FlexLM's layer registration, including non-ParityLM geometry."""
    model, old = gpt2_fixture()
    model.update(family="flex", layers=3, d_ff=10,
                 norm_placement={"attn": "pre", "ffn": "pre"}, attention="exact",
                 norm="layernorm", activation="gelu", position="learned", residual="residual", final_norm=True)
    state = {k: v.copy() for k, v in old.items() if k.startswith(("wte.", "wpe.", "head.", "ln_f."))}
    for b in range(3):
        for module, layer, dim_in, dim_out in (("attn", "c_attn", 4, 12), ("attn", "c_proj", 4, 4),
                                               ("mlp", "c_fc", 4, 10), ("mlp", "c_proj", 10, 4)):
            state[f"layers.{b}.{module}.{layer}.weight"] = np.arange(dim_in*dim_out, dtype=np.float32).reshape(dim_in, dim_out)
            state[f"layers.{b}.{module}.{layer}.bias"] = np.arange(dim_out, dtype=np.float32).reshape(1, dim_out)
        for norm in ("ln_1", "ln_2"):
            state[f"layers.{b}.{norm}.weight"] = np.ones((1, 4), dtype=np.float32)
            state[f"layers.{b}.{norm}.bias"] = np.zeros((1, 4), dtype=np.float32)
    return model, state


def test_plain_flex_maps_every_layer_and_nonstandard_mlp_width():
    model, state = flex_gpt2_fixture()
    mapped = hf_export.map_weights(state, model)
    assert len(mapped) == 41
    np.testing.assert_array_equal(mapped["transformer.h.2.mlp.c_fc.weight"], state["layers.2.mlp.c_fc.weight"])
    assert mapped["transformer.h.1.ln_1.weight"].shape == (4,)
    assert mapped["transformer.h.2.mlp.c_fc.bias"].shape == (10,)
    np.testing.assert_array_equal(mapped["lm_head.weight"], state["head.weight"].T)


@pytest.mark.parametrize("field", ["attention", "norm", "activation", "position", "residual", "final_norm", "norm_placement", "d_ff"])
def test_flex_missing_resolved_flavour_record_is_refused(field):
    model, _ = flex_gpt2_fixture()
    del model[field]
    with pytest.raises(ValueError, match="resolved flavour record"):
        hf_export.validate_gpt2(model)


@pytest.mark.parametrize("extra", [
    {"attention": "swa"}, {"activation": "gelu_erf"}, {"norm": "rmsnorm"},
    {"position": "sinusoidal"}, {"residual": "plain"}, {"residual": "highway"},
    {"norm_placement": "pre"}, {"norm_placement": {"attn": "pre"}},
    {"norm_placement": {"attn": "post", "ffn": "pre"}},
    {"norm_placement": {"attn": "pre", "ffn": "pre", "other": "pre"}},
    {"final_norm": False}, {"qkv_conv": 3}, {"qk_norm": "per_head"},
    {"embed": {"expr": "expr:x*2"}}, {"logits": "expr:x*2"},
    {"activation_inputs": [{"kind": "up"}, {"kind": "scalar"}]},
    {"norm_inputs": [{"kind": "row"}, {"kind": "scalar"}]},
    {"tie_head": True}, {"state_tokens": {"k": 4}}, {"recursion": {"repeats": 2}},
    {"attn_res": True}, {"moe": {"experts": 2}}, {"ssm": {"kind": "mamba1"}},
    {"d_ff": 0}, {"d_ff": True}, {"layers": 0}, {"heads": 3},
])
def test_flex_modified_or_unrecorded_geometry_is_refused(extra):
    model, _ = flex_gpt2_fixture()
    with pytest.raises(ValueError):
        hf_export.validate_gpt2({**model, **extra})


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "nan", "wrong_family_names"])
def test_flex_tensor_mapping_never_silently_discards_or_reshapes(change):
    model, state = flex_gpt2_fixture()
    if change == "missing":
        del state["layers.2.attn.c_proj.bias"]
    elif change == "extra":
        state["layers.0.attn.q_norm.weight"] = np.ones((1, 2), dtype=np.float32)
    elif change == "shape":
        state["layers.0.mlp.c_fc.weight"] = np.zeros((4, 16), dtype=np.float32)
    elif change == "nan":
        state["head.weight"][0, 0] = np.nan
    else:
        state["ln1_0.weight"] = state.pop("layers.0.ln_1.weight")
    with pytest.raises(ValueError, match="tensor"):
        hf_export.map_weights(state, model)


def test_plain_flex_config_and_complete_hf_loaded_inventory(tmp_path, monkeypatch):
    from safetensors.numpy import load_file
    transformers = pytest.importorskip("transformers")
    model, state = flex_gpt2_fixture()
    model.update(event="model", attention="exact", activation="gelu", norm="layernorm", position="learned")
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps({"event": "start", "name": "tiny"}) + "\n" + json.dumps(model) + "\n")
    monkeypatch.setattr(hf_export, "load_file", lambda _: state)
    monkeypatch.setattr(hf_export, "gguf_tokens", lambda _: ["<unk>", "a", "b", "c", "d", "e", "f"])
    monkeypatch.setattr(hf_export, "write_tokenizer", lambda *a: ({str(i): i for i in range(7)}, "<unk>"))
    import llama3_bpe
    monkeypatch.setattr(llama3_bpe, "read_gguf_kv", lambda *a: {})
    monkeypatch.setattr(sys, "argv", ["hf_export.py", str(run)])
    hf_export.main()
    cfg = json.loads((run / "hf/config.json").read_text())
    assert cfg["n_layer"] == 3 and cfg["n_inner"] == 10
    assert cfg["activation_function"] == "gelu_new" and cfg["tie_word_embeddings"] is False
    loaded = transformers.AutoModelForCausalLM.from_pretrained(run / "hf").state_dict()
    mapped = load_file(str(run / "hf/model.safetensors"))
    assert set(loaded) == set(mapped)
    for key, value in mapped.items():
        np.testing.assert_array_equal(loaded[key].numpy(), value)
