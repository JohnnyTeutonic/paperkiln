"""Offline HF-config comparisons preserve quoted extraction and never select variants."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fetch import Arch, Finding, crosscheck_hf_config, emit_html, to_json


def paper():
    return Arch("fixture", fields={"d_model": Finding(256, "Our hidden size is 256."),
                                  "n_heads": Finding(8, "We use eight heads."),
                                  "n_layers": Finding(4, "Four layers.")},
                variants=[{"d_model": 256, "n_heads": 8}, {"d_model": 512, "n_heads": 16}])


def test_matches_supported_aliases_without_mutating_paper_or_config():
    arch = paper()
    cfg = {"n_embd": 256, "n_head": 8, "n_layer": 4, "n_inner": None,
           "num_key_value_heads": 2, "num_local_experts": 8, "num_experts_per_tok": 2}
    before, original = copy.deepcopy(arch), copy.deepcopy(cfg)
    report = crosscheck_hf_config(arch, cfg)
    assert report["primary"]["differences"] == []
    assert report["primary"]["fields"]["d_model"]["status"] == "match"
    assert report["primary"]["fields"]["n_kv_heads"]["status"] == "paper_not_stated"
    assert report["primary"]["fields"]["n_experts"]["paper"] is None
    assert arch == before and cfg == original
    assert "hf_config_crosscheck" not in to_json(arch)


def test_all_variants_reported_without_selection_or_quote_override():
    arch = paper()
    report = crosscheck_hf_config(arch, {"hidden_size": 512, "num_attention_heads": 16})
    assert report["primary"]["differences"] == ["d_model", "n_heads"]
    assert report["variants"][0]["differences"] == ["d_model", "n_heads"]
    assert report["variants"][1]["differences"] == []
    assert report["variant_selected"] is None
    assert report["paper_fields_modified"] is False
    assert report["primary"]["fields"]["d_model"]["paper_evidence"] == "Our hidden size is 256."
    assert arch.fields["d_model"].value == 256


@pytest.mark.parametrize("cfg", [[], None, {"hidden_size": True}, {"hidden_size": "256"},
                                  {"hidden_size": 256.0}, {"hidden_size": -1},
                                  {"hidden_size": 256, "n_embd": 512}])
def test_bad_inputs_and_conflicting_aliases_rejected_without_mutation(cfg):
    arch = paper()
    original = copy.deepcopy(arch)
    with pytest.raises(ValueError):
        crosscheck_hf_config(arch, cfg)
    assert arch == original


def test_identical_aliases_are_not_ambiguous_and_no_defaults_are_inferred():
    report = crosscheck_hf_config(paper(), {"hidden_size": 256, "d_model": 256})
    fields = report["primary"]["fields"]
    assert fields["d_model"]["hf_aliases"] == {"hidden_size": 256, "d_model": 256}
    assert fields["n_heads"]["status"] == "hf_not_stated"
    assert fields["n_kv_heads"]["hf"] is None
    assert fields["d_ff"]["hf"] is None


def test_unresolved_paper_numbers_are_not_asserted_and_nested_config_not_selected():
    arch = paper()
    arch.unresolved.append("d_model")
    report = crosscheck_hf_config(arch, {"text_config": {"hidden_size": 512}})
    assert report["primary"]["fields"]["d_model"]["paper"] is None
    assert report["primary"]["fields"]["d_model"]["status"] == "paper_unresolved"
    assert "nested text_config was not selected" in report["note"]


def test_json_and_html_surface_comparison_without_unescaped_input():
    arch = paper()
    arch.hf_config_crosscheck = crosscheck_hf_config(arch, {"hidden_size": 512})
    arch.hf_config_crosscheck["source"] = {"path": "<script>alert(1)</script>"}
    assert to_json(arch)["hf_config_crosscheck"]["primary"]["differences"] == ["d_model"]
    html = emit_html(arch)
    assert "HF config cross-check" in html and "difference" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_cli_local_files_only_and_original_quotes_preserved(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text("We use a hidden size of 256 and 4 layers.")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"hidden_size": 512}))
    result, html = tmp_path / "result.json", tmp_path / "result.html"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("fetch.py")), "fixture",
                          "--tex", str(tex), "--hf-config", str(config), "--json", str(result),
                          "--emit-html", str(html)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    data = json.loads(result.read_text())
    assert data["fields"]["d_model"]["value"] == 256
    assert data["hf_config_crosscheck"]["primary"]["fields"]["d_model"]["hf"] == 512
    assert data["hf_config_crosscheck"]["source"]["sha256"]
    assert "paper fields unchanged" in run.stdout
    assert "HF config cross-check" in html.read_text()


def test_empty_config_does_not_claim_comparable_matches():
    from fetch import hf_config_summary
    report = crosscheck_hf_config(paper(), {})
    assert report["primary"]["compared_fields"] == []
    assert "0 comparable" in hf_config_summary(report)


def test_cli_bad_config_rejected_before_output(tmp_path):
    tex, config, result = tmp_path / "paper.tex", tmp_path / "config.json", tmp_path / "result.json"
    tex.write_text("We use a hidden size of 256.")
    config.write_text('{"hidden_size":true}')
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("fetch.py")), "fixture",
                          "--tex", str(tex), "--hf-config", str(config), "--json", str(result)],
                         capture_output=True, text=True)
    assert run.returncode == 2 and "nonnegative integer" in run.stderr
    assert not result.exists()


def test_package_cli_supports_same_local_comparison(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    tex, config, result = tmp_path / "paper.tex", tmp_path / "config.json", tmp_path / "result.json"
    tex.write_text("We use a hidden size of 256.")
    config.write_text('{"hidden_size":512}')
    run = subprocess.run([sys.executable, "-m", "paperkiln_fetch.cli", "fixture", "--tex", str(tex),
                          "--hf-config", str(config), "--json", str(result)],
                         env={**os.environ, "PYTHONPATH": str(root / "paperkiln_fetch/src")},
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    data = json.loads(result.read_text())
    assert data["hf_config_crosscheck"]["primary"]["differences"] == ["d_model"]
    assert data["fields"]["d_model"]["value"] == 256


def test_moe_numeric_comparison_preserves_explicit_quotes():
    arch = Arch("moe", fields={"n_experts": Finding(8, "We use eight experts."),
                               "top_k": Finding(2, "Two experts are active per token.")})
    report = crosscheck_hf_config(arch, {"num_local_experts": 8, "num_experts_per_tok": 4})
    assert report["primary"]["fields"]["n_experts"]["status"] == "match"
    assert report["primary"]["differences"] == ["top_k"]
    assert arch.fields["top_k"].value == 2


@pytest.mark.parametrize("verdict", ["unverified", "quote_only", "contested", "not_stated"])
def test_hf_agreement_cannot_verify_an_ineligible_paper_finding(verdict):
    arch = Arch("unverified", fields={"d_model": Finding(256, "A numerical mention: 256.", verdict=verdict)})
    report = crosscheck_hf_config(arch, {"hidden_size": 256})
    row = report["primary"]["fields"]["d_model"]
    assert row["status"] == "paper_" + verdict
    assert row["paper"] is None and row["paper_reported_value"] == 256
    assert row["paper_verdict"] == verdict
    assert row["paper_evidence"] == "A numerical mention: 256."
    assert report["primary"]["compared_fields"] == []
    assert arch.fields["d_model"].verdict == verdict
