"""The chain's spec stage (tools/arxiv_to_chat.py build_spec) on E3's papers, each extraction re-verified by
today's verifier against its cached LaTeX, without and with the operator's asserts and waivers. Offline, no
API. Writes spec_results.json beside this file.

    python experiments/arxiv_chain_2026-10-28_e3/spec_results.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "papers"))
import arxiv_to_chat as C  # noqa: E402

E26 = os.path.join(ROOT, "experiments", "arxiv_chain_2026-10-26")
R = os.path.join(ROOT, "papers", "llm_extract_receipts")
SIZE = {"d": 64, "layers": 2, "heads": 4, "d_ff": None, "T": 64}
RUNS = [
    ("2109.08668", HERE, [], []),  # Primer under prompt 7512599b+: positions refused by rule
    ("2312.00752", E26, [], []),  # Mamba
    ("2312.00752", E26, ["d_conv=4", "ssm_inner_norm=none"], []),
    ("2505.09388", E26, [], []),  # Qwen3 (headline MoE, q/k norm of unstated width)
    ("2505.09388", E26, ["qk_norm=per_head", "moe_gate_norm=renorm"],
     ["fine-grained expert segmentation", "no shared experts", "load balancing", "yarn and dual chunk"],
     {"d": 128, "heads": 16}),  # 16 query heads per kv head at the paper's ratio
    ("2401.04088", R, [], []),  # Mixtral
    ("2401.04088", R, ["norm=rmsnorm", "positional=rope"], []),
    ("2501.00656", HERE, [], []),  # OLMo 2 (q/k norm, full width by the operator)
    ("2501.00656", HERE, ["qk_norm=full"], []),
    ("2204.02311", R, [], []),  # PaLM: the logit scale 1/sqrt(d_model)
    ("2408.00118", R, [], []),  # Gemma 2: the final soft cap
    ("1706.03762", HERE, ["family=decoder_only"], []),  # the Transformer: sqrt(d_model) on the embedding
]

out = []
for pid, folder, asserts, waive, *size in RUNS:
    ext = C.reverify(json.load(open(os.path.join(folder, f"{pid}.extraction.json"), encoding="utf-8")))
    row = {"arxiv_id": pid, "extraction": os.path.relpath(os.path.join(folder, f"{pid}.extraction.json"), ROOT),
           "prompt_sha256": ext["source"].get("prompt_sha256", "")[:8], "asserts": asserts, "waive": waive}
    try:
        fam, arch, rep = C.build_spec(ext, {**SIZE, **(size[0] if size else {})}, C.parse_asserts(asserts),
                                      waive=waive)
        row.update(verdict="builds", family=fam, arch=arch, report=rep)
    except C.Refused as r:
        row.update(verdict="refused", reasons=r.args[0])
    out.append(row)
    print(f"{pid} {asserts or ''} {waive or ''}: {row['verdict']}"
          + (f" ({row['family']})" if row["verdict"] == "builds" else f", {len(row['reasons'])} reason(s)"))
with open(os.path.join(HERE, "spec_results.json"), "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
