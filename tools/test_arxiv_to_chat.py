"""Offline tests for tools/arxiv_to_chat.py: cached extraction -> spec ->
refusal, and the stage bookkeeping with mtstudio mocked (no training, no
network, no servers). Runs in about a second.

    python3 -m pytest tools/test_arxiv_to_chat.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_to_chat as A  # noqa: E402

CACHE = os.path.join(HERE, "arxiv_chat")
SIZE = {"d": 64, "layers": 2, "heads": 2, "d_ff": 0, "T": 64}


def cached(pid):
    return os.path.join(CACHE, f"{pid}.extraction.json")


def run_cli(tmp_path, pid, *extra):
    return A.main([pid, "--run-dir", str(tmp_path / pid), "--extraction", cached(pid), *extra])


def test_llama_paper_maps_to_llama_family(tmp_path):
    assert run_cli(tmp_path, "2302.13971", "--stop-after", "spec", "--d", "64", "--heads", "2") == 0
    run = tmp_path / "2302.13971"
    rep = json.loads((run / "spec_report.json").read_text())
    assert rep["family"] == "llama"
    pre = json.loads((run / "pretrain_spec.json").read_text())
    sft = json.loads((run / "sft_spec.json").read_text())
    assert pre["arch"]["rope_heads"] == "all"
    assert pre["arch"]["custom"] == {"position": "rope", "d": 64, "layers": 2, "heads": 2}
    assert sft["train"]["objective"] == "sft"
    assert sft["train"]["init_from"].endswith("pretrain/arxiv_2302_13971_base.safetensors")
    buckets = {b for b, _ in rep["report"]}
    assert {"MATCHED", "DEFAULT", "SIZED"} <= buckets
    man = json.loads((run / "manifest.json").read_text())
    assert man["stages"]["spec"]["family"] == "llama" and len(man["stages"]["extract"]["extraction_sha256"]) == 64


def test_gpt3_like_paper_maps_to_flex(tmp_path):
    # Cerebras-GPT inherits GPT-3's flavours and says it uses dense attention
    # ("unlike GPT-3 ..."): the sparse-attention cue is noted, not blocking.
    assert run_cli(tmp_path, "2304.03208", "--stop-after", "spec") == 0
    rep = json.loads((tmp_path / "2304.03208" / "spec_report.json").read_text())
    assert rep["family"] == "flex"
    custom = json.loads((tmp_path / "2304.03208" / "pretrain_spec.json").read_text())["arch"]["custom"]
    assert (custom["norm"], custom["activation"], custom["position"], custom["attention"]) == \
        ("layernorm", "gelu", "learned", "exact")
    assert any(b == "NOTED" and "sparse" in d for b, d in rep["report"])


@pytest.mark.parametrize("pid,why", [("2312.00752", "state-space model"),
                                     ("2005.14165", "sparse or banded attention")])
def test_refusals(tmp_path, pid, why, capsys):
    assert run_cli(tmp_path, pid) == A.EXIT_REFUSED
    ref = json.loads((tmp_path / pid / "refusal.json").read_text())
    assert any(why in r for r in ref["reasons"])
    assert not (tmp_path / pid / "pretrain_spec.json").exists()
    assert "REFUSED" in capsys.readouterr().out
    assert json.loads((tmp_path / pid / "manifest.json").read_text())["stages"]["spec"]["refused"]


def ext_with(**fields):
    f = {k: {"value": v, "evidence": "x", "verdict": "used", "score": 3.0} for k, v in fields.items()}
    return {"arxiv_id": "0000.00000", "fields": f, "mentions": {}, "mechanism_scan": []}


def test_unresolved_flavour_refused_until_asserted():
    ext = ext_with(activation="gelu", positional="learned")
    with pytest.raises(A.Refused, match="norm is unresolved"):
        A.build_spec(ext, SIZE)
    fam, arch, rep = A.build_spec(ext, SIZE, {"norm": "layernorm"})
    assert fam == "flex" and arch["custom"]["norm"] == "layernorm"
    assert any(b == "ASSERTED" for b, _ in rep)


def test_contested_and_unsupported_refused():
    ext = ext_with(norm="layernorm", activation="gelu", positional="learned")
    ext["fields"]["norm"]["verdict"] = "contested"
    with pytest.raises(A.Refused, match="contested"):
        A.build_spec(ext, SIZE)
    with pytest.raises(A.Refused, match="no engine support"):
        A.build_spec(ext_with(norm="rmsnorm", activation="geglu", positional="rope"), SIZE)
    with pytest.raises(A.Refused, match="llama family"):
        A.build_spec(ext_with(norm="layernorm", activation="gelu", positional="rope"), SIZE)
    with pytest.raises(A.Refused, match="n_kv_heads"):
        A.build_spec(ext_with(norm="rmsnorm", activation="swiglu", positional="rope", n_heads=32,
                              n_kv_heads=8), SIZE)


def test_swa_window_scaled_and_sinks_explicit():
    ext = ext_with(norm="layernorm", activation="gelu", positional="learned", attention="swa", window_size=4096)
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["window"] == 16 and arch["custom"]["sinks"] == 0
    assert any(b == "SCALED" for b, _ in rep)
    del ext["fields"]["window_size"]
    with pytest.raises(A.Refused, match="window"):
        A.build_spec(ext, SIZE)


def test_missing_mechanism_scan_refused_unless_skipped():
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional="rope")
    del ext["mechanism_scan"]
    with pytest.raises(A.Refused, match="mechanism scan"):
        A.build_spec(ext, SIZE)
    _, _, rep = A.build_spec(ext, SIZE, skip_scan=True)
    assert any(b == "UNCHECKED" for b, _ in rep)


def test_stages_with_mtstudio_mocked_and_restartable(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    sft_src = tmp_path / "sft.jsonl"
    sft_src.write_text("".join(json.dumps({"prompt": f"<|user|>q{i}<|assistant|>", "completion": "a"}) + "\n"
                               for i in range(5)))
    calls = []

    def fake(a, args, log_path):
        calls.append(args[0:2])
        spec = json.loads(open(args[1]).read())
        os.makedirs(spec["out_dir"], exist_ok=True)
        open(os.path.join(spec["out_dir"], spec["name"] + ".safetensors"), "wb").write(b"w" + spec["name"].encode())
        json.dump({"steps": spec["train"]["steps"]}, open(os.path.join(spec["out_dir"], "result.json"), "w"))
        json.dump(spec, open(os.path.join(spec["out_dir"], "spec.json"), "w"))

    monkeypatch.setattr(A, "mtstudio", fake)
    args = ["--corpus", str(corpus), "--vocab", str(corpus), "--sft-examples", str(sft_src), "--sft-limit", "3",
            "--stop-after", "sft", "--emit-jobs", str(tmp_path / "jobs.json")]
    assert run_cli(tmp_path, "2302.13971", *args) == 0
    run = tmp_path / "2302.13971"
    assert len((run / "sft_examples.jsonl").read_text().splitlines()) == 3
    man = json.loads((run / "manifest.json").read_text())
    assert len(man["stages"]["sft"]["sft_slice_sha256"]) == 64 and man["stages"]["pretrain"]["corpus_sha256"]
    assert calls == [["run", str(run / "pretrain_spec.json")], ["run", str(run / "sft_spec.json")]]
    # Restart: every stage's output exists, so nothing runs again.
    assert run_cli(tmp_path, "2302.13971", *args) == 0
    assert len(calls) == 2
    jobs = json.loads((tmp_path / "jobs.json").read_text())["jobs"]
    assert [j["name"] for j in jobs] == ["base_2302_13971", "sft_2302_13971"]
    assert jobs[1]["after"] == ["base_2302_13971"] and jobs[1]["init_job"] == "base_2302_13971"
    assert "after" not in jobs[0]
    assert jobs[0]["spec"]["data"]["corpus"].startswith("/content/pk/")
    assert "init_from" not in jobs[1]["spec"]["train"] and "out_dir" not in jobs[0]["spec"]
    # Riding in another launch: absolute VM paths, an .ids corpus read whole, extra dependencies.
    args2 = [a if a != str(tmp_path / "jobs.json") else str(tmp_path / "jobs2.json") for a in args]
    assert run_cli(tmp_path, "2302.13971", *args2, "--vm-corpus", "/content/chat/data/mix.main.ids",
                   "--vm-sft-upload", "/content/chat/lever1/lever1_sft_v1.jsonl", "--vm-after", "data",
                   "--vm-sft-after", "lever1") == 0
    j2 = json.loads((tmp_path / "jobs2.json").read_text())["jobs"]
    assert j2[0]["after"] == ["data"] and j2[1]["after"] == ["base_2302_13971", "lever1"]
    assert j2[0]["spec"]["data"]["corpus"] == "/content/chat/data/mix.main.ids"
    assert j2[0]["spec"]["data"]["max_tokens"] == 0
    assert j2[1]["spec"]["data"]["sft_examples"] == "/content/chat/lever1/lever1_sft_v1.jsonl"
