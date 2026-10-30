"""Cautious rule-based MoE fields and fail-closed simple emitters (no network)."""
from pathlib import Path
import sys

import pytest

from fetch import emit_cpp, emit_spec, extract


def fields(text):
    return extract("moe-fixture", text)


def test_explicit_adoption_counts_and_quotes():
    text = "We adopt a mixture-of-experts architecture with 8 experts and top-2 routing."
    arch = fields(text)
    assert arch.fields["moe"].value == "true"
    assert arch.fields["n_experts"].value == 8 and arch.fields["top_k"].value == 2
    for name in ("moe", "n_experts", "top_k"):
        assert arch.fields[name].evidence == text
        assert arch.fields[name].verdict == "used"


def test_routing_out_of_pool():
    arch = fields("Our MoE model routes each token to 2 out of 8 experts.")
    assert arch.fields["n_experts"].value == 8
    assert arch.fields["top_k"].value == 2


@pytest.mark.parametrize("text", [
    r"\section{Related Work} We use a MoE with 64 experts and top-4 routing.",
    "Prior work uses MoE with 64 experts and top-4 routing.",
    "We use a MoE with 64 experts and top-4 routing as a baseline.",
    "We do not use MoE; our baseline has 64 experts.",
    "We use 8 human experts for annotation.",
    "We use feed-forward layers and GELU activations per token.",
    "Our model uses a gated feed-forward network.",
])
def test_mentions_negation_human_experts_and_dense_ffn_do_not_assert_moe(text):
    arch = fields(text)
    assert not ({"moe", "n_experts", "top_k"} & set(arch.fields))
    assert "moe" not in arch.unresolved


def test_architecture_single_value_table():
    text = r"""\section{Architecture}
\begin{tabular}{ll}
Number of experts & 8 \\
Top-k & 2 \\
\end{tabular}
"""
    arch = fields(text)
    assert arch.fields["n_experts"].value == 8
    assert arch.fields["top_k"].value == 2
    assert arch.fields["n_experts"].evidence == "Number of experts & 8"


def test_active_expert_row_not_mistaken_for_pool_size():
    text = r"""\section{Architecture}
\begin{tabular}{ll}
Routed experts & 8 \\
Active experts & 2 \\
\end{tabular}
"""
    arch = fields(text)
    assert arch.fields["n_experts"].value == 8
    assert arch.fields["top_k"].value == 2


def test_baseline_table_and_variant_columns_do_not_select_counts():
    text = r"""\section{Architecture}
\begin{tabular}{lll}
Baseline & A & B \\
Experts & 8 & 16 \\
Top-k & 2 & 4 \\
\end{tabular}
"""
    arch = fields(text)
    assert "n_experts" not in arch.fields and "top_k" not in arch.fields


@pytest.mark.parametrize("text", [
    "We adopt a MoE with 8 experts and top-2 routing. Our MoE model uses 16 experts and top-2 routing.",
    "We adopt a MoE with 8 experts and top-16 routing.",
    "We adopt a MoE with 0 experts and top-2 routing.",
    "We compare our MoE candidate with 8 experts and top-2 routing.",
])
def test_conflicts_invalid_relationships_and_ambiguous_adoption_abstain(text):
    arch = fields(text)
    assert "n_experts" not in arch.fields
    assert {"n_experts", "top_k"} & set(arch.unresolved)


def test_active_count_does_not_invent_pool_size():
    arch = fields("We activate 2 experts per token in our MoE.")
    assert "n_experts" not in arch.fields
    assert arch.fields["top_k"].value == 2


@pytest.mark.parametrize("emit", [emit_cpp, emit_spec, lambda a: emit_spec(a, house_dims=True)])
def test_simple_emitters_refuse_moe_even_with_known_counts(emit):
    with pytest.raises(ValueError, match="MoE"):
        emit(fields("We adopt a MoE with 8 experts and top-2 routing."))


def test_packaged_hf_emitter_refuses_moe():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paperkiln_fetch/src"))
    from paperkiln_fetch.hf_emit import build_hf_config
    with pytest.raises(ValueError, match="MoE"):
        build_hf_config(fields("We adopt a MoE with 8 experts and top-2 routing."))


def test_ambiguous_own_moe_cannot_fall_through_to_dense_emission():
    arch = fields("We compare our MoE candidate with 8 experts and top-2 routing.")
    assert "moe" in arch.unresolved
    with pytest.raises(ValueError, match="MoE"):
        emit_spec(arch)


@pytest.mark.parametrize("text", [
    "We adopt a MoE with 8 experts and top-2 routing following prior work.",
    "We use a MoE with 8 experts, not 64 experts, and top-2 routing.",
    "We use a MoE with 8 experts without auxiliary loss and top-2 routing.",
])
def test_ambiguous_modifier_cannot_remove_moe_safety_gate(text):
    arch = fields(text)
    assert "n_experts" not in arch.fields
    with pytest.raises(ValueError, match="MoE"):
        emit_spec(arch)


@pytest.mark.parametrize("ending", [
    "We do not use MoE in the final model.",
    "We abandon MoE for a dense architecture.",
    "We replace MoE with dense layers.",
    "We switched to a dense architecture.",
    "Our final model abandons it.",
    "Our final model uses dense layers.",
    "We no longer use MoE.",
])
def test_global_own_retraction_vetoes_earlier_moe_counts(ending):
    arch = fields("We use MoE with 8 experts and top-2 routing. " + ending)
    assert not ({"moe", "n_experts", "top_k"} & set(arch.fields))
    assert "moe" in arch.unresolved
    assert any(m["status"] == "own_retraction" for m in arch.mentions["moe"])
    with pytest.raises(ValueError, match="MoE"):
        emit_spec(arch)


def test_preliminary_runs_not_promoted_to_final_architecture():
    arch = fields("We use MoE with 8 experts in preliminary runs. We do not use MoE in the final model.")
    assert not ({"moe", "n_experts", "top_k"} & set(arch.fields))
    assert "moe" in arch.unresolved


def test_top_n_performance_metric_not_misread_as_expert_routing():
    arch = fields("We use MoE with 8 experts and report top-1 accuracy of 90%.")
    assert arch.fields["n_experts"].value == 8
    assert "top_k" not in arch.fields and "top_k" in arch.unresolved


@pytest.mark.parametrize("flag", ["--emit-spec", "--emit-cpp"])
def test_cli_refusal_does_not_create_dense_or_empty_artifacts(tmp_path, flag):
    import subprocess
    tex, output = tmp_path / "moe.tex", tmp_path / "model.out"
    tex.write_text("We adopt a MoE with 8 experts and top-2 routing.")
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("fetch.py")), "fixture",
                          "--tex", str(tex), flag, str(output)], capture_output=True, text=True)
    assert run.returncode == 2 and "MoE" in run.stderr
    assert not output.exists()


def test_cli_can_report_ambiguous_counts_as_metadata(tmp_path):
    import subprocess
    import json
    tex, output = tmp_path / "moe.tex", tmp_path / "extraction.json"
    tex.write_text("We adopt a MoE with 8 experts and top-16 routing.")
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("fetch.py")), "fixture",
                          "--tex", str(tex), "--json", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    data = json.loads(output.read_text())
    assert data["mentions"]["n_experts"][0]["value"] == 8
    assert "top_k" in data["unresolved"]


def test_package_cli_refuses_hf_dense_fallback(tmp_path):
    import os
    import subprocess
    tex, output = tmp_path / "moe.tex", tmp_path / "config.json"
    tex.write_text("We adopt a MoE with 8 experts and top-2 routing.")
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "-m", "paperkiln_fetch.cli", "fixture", "--tex", str(tex),
                          "--emit-hf", str(output)], env={**os.environ, "PYTHONPATH": str(root / "paperkiln_fetch/src")},
                         capture_output=True, text=True)
    assert run.returncode == 2 and "MoE" in run.stderr
    assert not output.exists()


@pytest.mark.parametrize("metric", ["top-1 routing accuracy", "top-1 expert accuracy", "top-k=1 token accuracy"])
def test_explicit_router_metric_still_does_not_define_top_k(metric):
    arch = fields("We use MoE with 8 experts and report " + metric + " of 90%.")
    assert arch.fields["n_experts"].value == 8
    assert "top_k" not in arch.fields


@pytest.mark.parametrize("subject,verb", [("model", "is"), ("architecture", "is"), ("layers", "are")])
def test_copular_own_moe_cannot_fall_through_to_dense(subject, verb):
    text = f"Our {subject} {verb} a mixture-of-experts with 8 experts and top-2 routing."
    arch = fields(text)
    assert arch.fields["moe"].value == "true"
    assert arch.fields["n_experts"].value == 8 and arch.fields["top_k"].value == 2
    assert arch.fields["moe"].evidence == text
    for emitter in (emit_spec, emit_cpp):
        with pytest.raises(ValueError, match="MoE"):
            emitter(arch)


@pytest.mark.parametrize("text", [
    "We propose a dense model instead of a MoE with 8 experts and top-2 routing.",
    "We adopt dense FFNs rather than mixture-of-experts with 8 experts and top-2 routing.",
    "We propose a dense alternative to MoE with 8 experts and top-2 routing.",
])
def test_rejected_alternative_counts_are_not_own_adoption(text):
    arch = fields(text)
    assert not ({"moe", "n_experts", "top_k"} & set(arch.fields))
    assert "moe" in arch.unresolved
    assert any(m["status"] == "rejected_alternative" for m in arch.mentions["moe"])
    for emitter in (emit_spec, emit_cpp):
        with pytest.raises(ValueError, match="MoE"):
            emitter(arch)


def test_copular_comparison_is_not_automatically_own_adoption():
    arch = fields("Our model is compared to a MoE baseline with 8 experts and top-2 routing.")
    assert not ({"moe", "n_experts", "top_k"} & set(arch.fields))
