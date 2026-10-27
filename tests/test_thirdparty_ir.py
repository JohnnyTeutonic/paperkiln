"""The third-party ops in training, end to end (LOCAL_LLM_STACK.md section 11 item 4.4): grouped-query
attention, NEOX and partial RoPE, per-head q/k RMSNorm (the llama family's arch.custom.kv_heads, rope_pairing,
rope_dims, qk_norm, rope_theta) and Mamba-1 (the mamba preset's arch.custom.ssm "mamba1": conv1d_causal and
selective_scan).

Each model trains 40 steps on CPU and exports F32 (and, for the llama cases, Q8_0) GGUFs. Checks: the files
pass spec/tools/ir_check.py (keramos's validator and the file's tensors); the architecture is "llama" for
grouped-query attention alone (llama.cpp's llama graph has head_count_kv) and "paperkiln" otherwise; the IR's
nodes carry the thirdparty attributes (rope kv_heads / pairing / n_rot, rmsnorm group, the kv slot's kv_heads;
conv1d_causal and selective_scan with their recurrent slots); spec/tools/ir_numpy.py's forward over the GGUF
equals mtstudio's own forward (`mtstudio score`) on 40 prompts: log-probabilities of numpy's greedy
continuation within ROUNDING (1e-5 of max(1, max |ref|)) and mtstudio's argmax equal to numpy's greedy token
at every scored position (40/40 greedy). The single ops against the numpy reference's own expected values:
the conformance suite's op cases through tools/ir_eval (paperkiln's training ops). Bad specs are refused by
name.

TP_GATE_DIR=DIR also writes ember's gate files (F32 GGUF, IR, Q8_0 GGUF where exported, vocabulary,
score_in/score_out, spec and a manifest with sha256) into DIR/<name>/.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import WORDS, corpus, events, run  # noqa: E402
from test_recursion_ir import gguf_exports, parity  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

# name -> (preset, arch.custom, GGUF architecture)
CASES = {
    "tp_gqa": ("llama-tiny", {"d": 32, "heads": 4, "kv_heads": 2}, "llama"),
    "tp_qwen": ("llama-tiny", {"d": 32, "heads": 4, "kv_heads": 2, "rope_pairing": "neox", "qk_norm": True,
                               "rope_theta": 1000000.0}, "paperkiln"),
    "tp_mqa_partial": ("llama-tiny", {"d": 32, "heads": 4, "kv_heads": 1, "rope_dims": 4}, "paperkiln"),
    "tp_neox_partial": ("llama-tiny", {"d": 32, "heads": 4, "rope_pairing": "neox", "rope_dims": 4}, "paperkiln"),
    "tp_mamba1": ("mamba-tiny", {"d": 32, "layers": 2, "ssm": "mamba1", "d_state": 8}, "paperkiln"),
    # QK-norm with a learned per-head temperature (arch.custom.qk_temp, the init value).
    "tp_qk_temp": ("llama-tiny", {"d": 32, "heads": 4, "qk_norm": True, "qk_temp": 2.75}, "paperkiln"),
}


def spec_of(tmp, name, preset, custom, steps=40, formats=("safetensors", "gguf", "gguf_q8_0"), extra=None):
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    arch = {"preset": preset, "custom": dict(custom)}
    arch.update(extra or {})
    spec = {"name": name, "arch": arch,
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": {"steps": steps, "lr": 0.003, "eval_every": 20, "checkpoint_every": 20, "batch": 2, "seed": 3,
                      "gradmap_every": 0},
            "export": {"formats": list(formats)}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gate(name, spec, res, custom, family):
    out = Path(os.environ["TP_GATE_DIR"]) / name
    out.mkdir(parents=True, exist_ok=True)
    ex = gguf_exports(spec)
    tmp = Path(spec).parent
    f32 = Path(next(e["path"] for e in ex if e["format"] == "gguf"))
    files = [f32, f32.with_suffix(".ir.json")]
    files += [Path(e["path"]) for e in ex if e["format"] == "gguf_q8_0"]
    files += [tmp / "v.gguf", tmp / "score_in.jsonl", tmp / "score_out.jsonl"]
    for f in files:
        shutil.copyfile(f, out / f.name)
    shutil.copyfile(spec, out / "spec.json")
    files.append(out / "spec.json")
    worst, agree, n = res
    manifest = {"model": name, "family_hint": family, "arch_custom": custom,
                "spec": json.loads(Path(spec).read_text()),
                "numpy_vs_mtstudio": {"worst": worst, "agree": agree, "n": n},
                "files": {f.name: sha256(out / f.name) for f in files},
                "_note": "score_in.jsonl: prompt_ids and spec/tools/ir_numpy.py's greedy continuation over the F32 GGUF; "
                         "score_out.jsonl: mtstudio score over the same ids (logprobs, argmax). ember gate: "
                         "greedy 40/40 against mtstudio and log-probabilities within ROUNDING on CPU and CUDA "
                         "(LOCAL_LLM_STACK.md section 3 point 1)."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def check_nodes(name, custom, doc):
    nodes = doc["block"]["nodes"]
    if name == "tp_mamba1":
        ops = [n["op"] for n in nodes]
        assert ops == ["rmsnorm", "linear", "split", "conv1d_causal", "silu", "linear", "split", "linear",
                       "selective_scan", "silu", "mul", "linear", "add"], ops
        assert [s["op"] for s in doc["block"]["state"]] == ["conv1d_causal", "selective_scan"]
        assert "pos" not in [i["name"] for i in doc["inputs"]]
        return
    H, hd = custom["heads"], custom["d"] // custom["heads"]
    kv = custom.get("kv_heads", H)
    rope = next(n for n in nodes if n["op"] == "rope")
    att = next(n for n in nodes if n["op"] == "attention")
    assert att["kv_heads"] == kv and doc["block"]["state"][0]["kv_heads"] == kv
    assert rope.get("kv_heads", H) == kv
    assert rope["pairing"] == custom.get("rope_pairing", "interleaved")
    assert rope.get("n_rot", hd) == custom.get("rope_dims", hd)
    assert rope["theta"] == custom.get("rope_theta", 10000.0)
    norms = [n for n in nodes if n["op"] == "rmsnorm" and n.get("group")]
    if custom.get("qk_norm"):
        assert [(n["in"], n["group"], n["weight"]) for n in norms] == [
            (["q"], hd, "blk.{i}.attn_q_norm.weight"), (["k"], hd, "blk.{i}.attn_k_norm.weight")]
        i_rope = nodes.index(rope)
        assert all(nodes.index(n) < i_rope for n in norms)  # before RoPE
    else:
        assert norms == []
    scales = [n for n in nodes if n["op"] == "scale"]
    if custom.get("qk_temp"):
        # q times its head's temperature, after q's norm and before RoPE
        assert [(n["in"], n["weight"], n["out"]) for n in scales] == [(["q"], "blk.{i}.attn_q_temp.weight", "q")]
        assert nodes.index(norms[0]) < nodes.index(scales[0]) < nodes.index(rope)
    else:
        assert scales == []


@needs_bin
@pytest.mark.parametrize("name", list(CASES))
def test_thirdparty_export_and_numpy_parity(tmp_path, name):
    preset, custom, arch = CASES[name]
    llama = preset.startswith("llama")
    spec = spec_of(tmp_path, name, preset, custom, formats=("safetensors", "gguf", "gguf_q8_0") if llama else
                   ("safetensors", "gguf"))
    run(spec)
    ev = events(spec)
    steps = [e["loss"] for e in ev if e["event"] == "step"]
    assert steps[-1] < steps[0] - 0.3, steps  # it trains
    model = next(e for e in ev if e["event"] == "model")
    if name == "tp_mamba1":
        assert model["ssm"]["kind"] == "mamba1" and model["ssm"]["dt_rank"] == 2
    elif name == "tp_gqa":
        assert model["thirdparty"]["kv_heads"] == 2
    elif name == "tp_qk_temp":
        assert model["thirdparty"]["qk_temp"] == 2.75
    ex = gguf_exports(spec)
    assert {e["format"] for e in ex} == ({"gguf", "gguf_q8_0"} if llama else {"gguf"})
    for e in ex:
        assert ir_check.check(e["path"]) == [], ir_check.check(e["path"])
        meta, tensors, *_ = ir_check.read_gguf(e["path"])
        assert meta["general.architecture"][1] == arch
        if llama:
            assert meta[arch + ".attention.head_count_kv"][1] == custom.get("kv_heads", custom["heads"])
            names = {t[0] for t in tensors}
            assert ("blk.0.attn_q_norm.weight" in names) == bool(custom.get("qk_norm"))
            assert ("blk.0.attn_q_temp.weight" in names) == bool(custom.get("qk_temp"))
            k = next(t for t in tensors if t[0] == "blk.0.attn_k.weight")
            assert list(k[1]) == [custom["d"], custom.get("kv_heads", custom["heads"]) * custom["d"] // custom["heads"]]
    doc = json.loads(ir_check.read_gguf(ex[0]["path"])[0]["paperkiln.ir"][1])
    assert doc["family_hint"] == ("mamba1" if name == "tp_mamba1" else "llama")
    check_nodes(name, custom, doc)
    worst, agree, n, _ = parity(spec, tmp_path)
    print(f"{name}: numpy IR vs mtstudio log-probs worst {worst:.2e} of max(1, |ref|); greedy {agree}/{n}")
    assert n == 40 and agree == 40
    assert worst <= 1e-5
    if os.environ.get("TP_GATE_DIR"):
        write_gate(name, spec, (worst, agree, n), custom, doc["family_hint"])


@needs_bin
def test_thirdparty_single_ops_match_numpy():
    """The conformance suite's op cases for these ops, paperkiln's training ops against numpy's expected values."""
    ie = Path(BIN).parent / "ir_eval"
    if not ie.is_file():
        pytest.skip("no ir_eval beside mtstudio")
    import ir_conformance
    import ir_conformance_mtstudio
    seen = {}
    for key in ("op/rope", "op/rmsnorm", "op/attention", "op/conv1d_causal", "op/selective_scan"):
        counts = ir_conformance.run_suite(ir_conformance_mtstudio.MtstudioAdapter(BIN), key, None)
        seen[key] = counts
        assert counts["FAIL"] == 0, (key, counts)
    passed = sum(c["PASS"] for c in seen.values())
    # rope 4 (interleaved, neox, neox_partial, grouped_query), rmsnorm 2 (row, group), attention 4 (causal,
    # grouped_query, position_bias, position_bias_grouped_query: T5's bias through ops::relpos_attention since
    # e8f66a10), conv1d 1, selective_scan 1; skipped: freq_scale, weight_offset, windows (not trained here).
    assert passed == 12, seen


@needs_bin
@pytest.mark.parametrize("preset,custom,extra,msg", [
    ("llama-tiny", {"d": 32, "heads": 4, "kv_heads": 3}, None, "must divide heads"),
    ("llama-tiny", {"d": 32, "heads": 4, "rope_dims": 3}, None, "rope_dims must be even"),
    ("llama-tiny", {"d": 32, "heads": 4, "rope_dims": 10}, None, "at most the head width"),
    ("llama-tiny", {"d": 32, "heads": 4, "rope_pairing": "half"}, None, "interleaved or neox"),
    ("llama-tiny", {"d": 32, "heads": 4, "kv_heads": 2, "attn_res": {"form": "full"}}, None, "not wired with attn_res"),
    ("gpt2-nano", {"d": 32, "heads": 4, "kv_heads": 2}, None, "need the llama family"),
    ("llama-tiny", {"d": 32, "heads": 4, "ssm": "mamba1"}, None, "need the mamba preset"),
    ("mamba-tiny", {"d": 32, "d_conv": 3}, None, "need ssm mamba1"),
    ("mamba-tiny", {"d": 32, "ssm": "mamba2"}, None, "must be s4 or mamba1"),
    ("llama-tiny", {"d": 32, "heads": 4, "qk_temp": 2.0}, None, "qk_temp must be positive and needs qk_norm"),
    ("llama-tiny", {"d": 32, "heads": 4, "qk_norm": True, "qk_temp": -1.0}, None, "qk_temp must be positive"),
])
def test_bad_thirdparty_specs_are_refused(tmp_path, preset, custom, extra, msg):
    spec = spec_of(tmp_path, "bad", preset, custom, formats=("safetensors",), extra=extra)
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and msg in r.stderr, r.stderr


@needs_bin
def test_muon_refuses_grouped_query(tmp_path):
    spec = spec_of(tmp_path, "muon", "llama-tiny", {"d": 32, "heads": 4, "kv_heads": 2}, steps=2,
                   formats=("safetensors",))
    j = json.loads(spec.read_text())
    j["train"]["optimizer"] = "muon"
    spec.write_text(json.dumps(j))
    r = subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "not wired with it" in (r.stderr + r.stdout)
