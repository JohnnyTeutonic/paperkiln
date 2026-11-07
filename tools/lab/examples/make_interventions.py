"""Writes interventions.json: the Level 3 library. Priors are readings, not measurements. Claude Opus 5.5, 7 Nov 2026. Noise is a point-scale standard deviation near a20_bars.json's 7.45 items of 108 for a one-seed null."""
import json
from pathlib import Path

R = "Claude Opus 5.5's reading, 7 Nov 2026"
COMMON = {"frozen_evaluation": ["T_E, once built", "chatflow_v001", "contrast_v002"], "guards": ["faithfulness", "abstention", "identity", "role"], "registration_status": "unregistered", "review_status": "none",
          "stop_criteria": "a guard fails, or the budget is met", "continue_criteria": "own registration; no iterative campaign"}


def e(id_, cls, group, hyp, pre, hours, start, control, facet, primary, mean, sd, noise=7.0):
    return dict(COMMON, id=id_, group=group, **{"class": cls}, hypothesis=hyp, prerequisites=pre, estimated_l4_hours=hours, starting_checkpoint=start, control_condition=control, changed_facet=facet, primary_endpoint=primary,
                effect_prior={"mean_points": mean, "sd_points": sd, "noise_sd_points": noise, "producer": R})


SHARED = ["t_e_rows_built", "noise_floor_measured"]
L = [
    e("noise_pair_pk_sft_v1", "measurement", "pk_task", "Two seeds of the balanced fixed curriculum, excluded from any comparison, set the noise floor and show whether any family stays above 0.15 nats excess", ["non_weak_base_available", "t_m_rows", "t_c_all_families", "dev_read_on_vm", "packet_writer", "t_e_rows_built"], 3.04, "pk_sft_v1", "itself, second seed", "seed", "same-allocation spread on T_E", 0.0, 7.0, 4.0),
    e("fixed_balanced_stronger", "sft", "stronger_task", "Design A: a balanced fixed mixture beats the recipe on the stronger base", ["stronger_base_selected", "fixed_baseline_on_stronger_base", "t_c_all_families", "packet_writer"] + SHARED, 1.23, "B-SFT", "S34b", "mixture", "told recall and request scorecard", 4.0, 6.0),
    e("adaptive_three_pairs_pk_sft_v1", "sft", "pk_task", "Between-leg adaptive allocation beats the fixed curriculum at matched exposure", ["non_weak_base_available", "mixed_specialist_on_pk_sft_v1", "t_c_all_families", "dev_read_on_vm", "packet_writer", "dev_signal_checked"] + SHARED, 9.12, "pk_sft_v1", "the fixed pair", "allocation policy", "T_E task points", 1.0, 4.0),
    e("adaptive_three_pairs_stronger", "sft", "stronger_task", "The same, on the selected base", ["stronger_base_selected", "fixed_baseline_on_stronger_base", "t_c_all_families", "dev_read_on_vm", "packet_writer", "dev_signal_checked"] + SHARED, 8.43, "B-SFT", "fixed curriculum", "allocation policy", "T_E task points", 1.5, 4.0, 5.0),
    e("error_directed_pk_sft_v1", "sft", "pk_task", "Rows selected from cells the model fails beat random verified rows", ["non_weak_base_available", "s54_j1_read", "t_m_rows", "t_c_mixed_families", "packet_writer", "t_e_rows_built", "noise_floor_measured"], 2.35, "S54", "random verified rows", "row selection", "T_E told recall", 1.0, 3.5),
    e("error_directed_stronger", "sft", "stronger_task", "The same, on the selected base", ["stronger_base_selected", "fixed_baseline_on_stronger_base", "t_m_rows", "t_c_mixed_families", "packet_writer", "t_e_rows_built", "noise_floor_measured"], 1.85, "fixed recipe on B-SFT", "random verified rows", "row selection", "T_E told recall", 1.5, 3.5),
    e("context_annotations", "architecture", "stronger_task", "Typed per-turn state supervises attribution and supersession", ["stronger_base_selected", "dev_signal_checked"], 4.0, "B-SFT", "state tokens without targets", "state supervision", "attribution and correction cells", 3.0, 6.0),
    e("pairwise_preference", "loss", "stronger_task", "Verified chosen and rejected pairs raise context pairs", ["stronger_base_selected", "preference_gate_r0"], 2.2, "B-SFT", "equal-budget SFT on the chosen replies", "objective", "context pairs", 2.0, 5.0),
    e("dpo", "loss", "stronger_task", "DPO after SFT raises relevance at chat scale", ["stronger_base_selected", "dpo_reference_cache_repaired"], 3.0, "best SFT", "SFT alone", "objective", "J1 relevance", 1.0, 4.0),
    e("more_pretraining", "base", "stronger_base_read", "More exposure lifts knowledge and carries into chat", ["stronger_base_selected"], 14.0, "the winner's last checkpoint", "the winner", "exposure", "LAMBADA, then chat", 4.0, 7.0),
    e("depth_for_width", "architecture", "stronger_task", "Depth at matched parameters helps once the base is strong", ["stronger_base_selected"], 5.6, "B-SFT", "the other shape", "depth", "registered two-by-two", 2.0, 6.0),
    e("larger_architecture", "architecture", "stronger_task", "A larger model buys chat competence", ["stronger_base_selected", "sft_saturated"], 30.0, "fresh", "the selected base", "size", "registered in its own amendment", 5.0, 8.0),
]
Path(__file__).with_name("interventions.json").write_text(json.dumps({"schema": "lab-interventions-v1", "producer": "Claude Opus 5.5, 7 Nov 2026", "interventions": L}, indent=1) + "\n", encoding="utf-8", newline="\n")
