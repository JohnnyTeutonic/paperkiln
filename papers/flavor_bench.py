#!/usr/bin/env python3
"""Contribution-vs-mention benchmark for the flavor scorer (AUROC).

    python papers/flavor_bench.py            # download (cached) + evaluate
    python papers/flavor_bench.py --offline  # cached papers only

A paper MENTIONS many flavor alternatives; it USES one. This benchmark
measures whether score_flavors() separates the two, on real papers whose
architectures are public knowledge. For every (paper, field) with ground
truth, every candidate the text contains becomes one (score, label) pair
— label 1 iff the candidate is what the paper actually uses. AUROC over
those pairs is the discrimination number; the naive baseline is the old
first-match rule (score = match priority, what fetch.py did before).

Papers where the true value is NOT in our lattice (Primer: squared ReLU)
contribute only negative labels — the scorer's job there is to assert
nothing. Verdict accuracy is reported alongside AUROC: top-1 correctness
of what extract() would actually apply ("used"/"contested" fields).

Ground truth sources: the papers themselves (each architecture statement
is quotable) — Vaswani §3, LLaMA §2, PaLM §2, BLOOM §3.1, GPT-NeoX §2.1,
Pythia §2.1, RoFormer, ALiBi §3, Primer §4 (squared ReLU), T5 §2.1.

SAMPLE-SIZE CAVEAT (docs/STUDIO_PLAN.md §13.1): 10 papers / 31 pairs is
enough to kill the observed failure modes, not enough for a stable
AUROC estimate — grow TRUTH toward 30-50 papers and report bootstrap
CIs (resampling papers, the independent unit) before quoting these
numbers anywhere load-bearing.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from fetch import (FLAVOR, FLAVOR_FAMILIES, extract, fetch_source,
                   score_flavors, split_sections)

CACHE = pathlib.Path(__file__).parent / ".cache"

# paper -> field -> the value this paper USES (None = the true value is
# outside the candidate lattice, so every candidate is a negative —
# abstention is the only correct output). Truth = what the paper's own
# text asserts about ITS model; fields we are not certain of are simply
# omitted rather than guessed.
TRUTH: dict[str, dict[str, str | None]] = {
    # ---- original ten (2026-08-01) ----
    "1706.03762": {"norm": "layernorm", "activation": "relu",
                   "positional": "sinusoidal"},          # Transformer
    "2302.13971": {"norm": "rmsnorm", "activation": "swiglu",
                   "positional": "rope"},                # LLaMA
    "2104.09864": {"positional": "rope"},                # RoFormer
    "2108.12409": {"positional": "alibi"},               # ALiBi
    "2109.08668": {"norm": "rmsnorm",
                   "activation": None},                  # Primer (squared ReLU)
    "2204.02311": {"activation": "swiglu",
                   "positional": "rope"},                # PaLM
    "2211.05100": {"norm": "layernorm", "activation": "gelu",
                   "positional": "alibi"},               # BLOOM
    "2204.06745": {"norm": "layernorm", "activation": "gelu",
                   "positional": "rope"},                # GPT-NeoX
    "2304.01373": {"norm": "layernorm",
                   "positional": "rope"},                # Pythia
    "1910.10683": {"activation": "relu"},                # T5
    # ---- growth batch (2026-08-01, STUDIO_PLAN 13.1) ----
    "1810.04805": {"norm": "layernorm", "activation": "gelu",
                   "positional": "learned"},             # BERT
    "1907.11692": {"norm": "layernorm", "activation": "gelu",
                   "positional": "learned"},             # RoBERTa
    "1909.11942": {"norm": "layernorm", "activation": "gelu",
                   "positional": "learned"},             # ALBERT
    "2005.14165": {"norm": "layernorm",
                   "positional": "learned"},             # GPT-3
    "2205.01068": {"norm": "layernorm", "activation": "relu",
                   "positional": "learned"},             # OPT
    "2211.09085": {"activation": "gelu",
                   "positional": "learned"},             # Galactica
    "2006.03654": {"norm": "layernorm", "activation": "gelu",
                   "positional": None},                  # DeBERTa (relative)
    "1906.08237": {"norm": "layernorm",
                   "positional": None},                  # XLNet (rel. two-stream)
    "2004.05150": {"norm": "layernorm", "activation": "gelu",
                   "positional": "learned"},             # Longformer
    "2307.09288": {"norm": "rmsnorm", "activation": "swiglu",
                   "positional": "rope"},                # Llama 2
    "2401.02385": {"norm": "rmsnorm", "activation": "swiglu",
                   "positional": "rope"},                # TinyLlama
    "2309.16609": {"norm": "rmsnorm", "activation": "swiglu",
                   "positional": "rope"},                # Qwen
    "2401.02954": {"norm": "rmsnorm", "activation": "swiglu",
                   "positional": "rope"},                # DeepSeek LLM
    "2403.08295": {"norm": "rmsnorm", "activation": "geglu",
                   "positional": "rope"},                # Gemma (GeGLU!)
    "2311.16867": {"norm": "layernorm", "activation": "gelu",
                   "positional": "rope"},                # Falcon
    "2112.11446": {"norm": "rmsnorm",
                   "positional": None},                  # Gopher (relative)
}


def get_tex(arxiv_id: str, offline: bool) -> str | None:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{arxiv_id}.tex"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    if offline:
        return None
    try:
        tex = fetch_source(arxiv_id)
    except Exception as e:                                # noqa: BLE001
        print(f"  [skip] {arxiv_id}: {e}")
        return None
    p.write_text(tex, encoding="utf-8", errors="replace")
    return tex


def auroc(pairs: list[tuple[float, int]]) -> float | None:
    """Rank-based AUROC with tie handling (average ranks)."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for pv in pos:
        for nv in neg:
            wins += 1.0 if pv > nv else 0.5 if pv == nv else 0.0
    return wins / (len(pos) * len(neg))


def bootstrap_ci(per_paper, stat, n_boot=2000, seed=0):
    """95% percentile CI resampling PAPERS (the independent unit — the
    STUDIO_PLAN 13.1 protocol), not pairs. per_paper: paper_id ->
    whatever `stat` consumes for a resampled list of papers."""
    import random
    rng = random.Random(seed)
    ids = list(per_paper.keys())
    vals = []
    for _ in range(n_boot):
        sample = [per_paper[rng.choice(ids)] for _ in ids]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None  # too many degenerate resamples to trust a CI
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scored: list[tuple[float, int]] = []   # the scorer
    postveto: list[tuple[float, int]] = []  # scorer, vetoed candidates dropped
    naive: list[tuple[float, int]] = []    # old first-match priority
    per_field: dict[str, list[tuple[float, int]]] = {}
    # Grouped = one AUROC per (paper, field), averaged. This is the
    # DEPLOYED decision ("in this paper, does the used flavor outrank
    # the mentioned ones?"); pooled additionally measures cross-paper
    # score calibration, which the used/contested threshold relies on.
    grouped: list[tuple[list[tuple[float, int]], list[tuple[float, int]]]] = []
    # paper -> {"pairs": [(score,label)...], "groups": [[(s,l)...] per
    # field]} — the resampling unit for the bootstrap CIs.
    by_paper: dict[str, dict] = {}
    verdict_ok, verdict_bad, contested = 0, [], 0
    n_papers = 0
    n_inherited_ok, n_inherited_bad, n_family = 0, [], 0
    inherit_papers = []

    for arxiv_id, truths in TRUTH.items():
        tex = get_tex(arxiv_id, args.offline)
        if tex is None:
            continue
        n_papers += 1
        cands_by_field = score_flavors(split_sections(tex))
        arch = extract(arxiv_id, tex)
        if arch.inherits:
            inherit_papers.append((arxiv_id, arch.inherits["ancestor"]))
        for fieldname, true_val in truths.items():
            cands = cands_by_field.get(fieldname, [])
            prio = {v: -i for i, (v, _) in enumerate(FLAVOR[fieldname])}
            g_new, g_old = [], []
            for c in cands:
                label = 1 if c["value"] == true_val else 0
                scored.append((c["score"], label))
                naive.append((prio[c["value"]], label))
                per_field.setdefault(fieldname, []).append((c["score"], label))
                g_new.append((c["score"], label))
                g_old.append((prio[c["value"]], label))
                # Post-veto view: the deployed system can never assert a
                # candidate the paper explicitly rejected, so a ranking
                # metric that still contains them measures a scorer we do
                # not ship. Drop them (Falcon's SwiGLU, Primer's).
                if "rejection-elsewhere" not in c.get("cues", []):
                    postveto.append((c["score"], label))
            grouped.append((g_new, g_old))
            bp = by_paper.setdefault(arxiv_id, {"pairs": [], "groups": []})
            bp["pairs"].extend(g_new)
            if g_new:
                bp["groups"].append(g_new)
            if args.verbose:
                pretty = ", ".join(f"{c['value']}:{c['score']}" for c in cands)
                print(f"  {arxiv_id} {fieldname:11} truth={true_val}  [{pretty}]")
            # verdict accuracy: what extract() would actually apply
            f = arch.fields.get(fieldname)
            if f is None or f.verdict is None:
                applied = None
            elif f.verdict == "contested":
                contested += 1
                applied = None
            elif f.verdict == "family":
                # Family-level assertion is CORRECT iff the truth is a
                # member of that family (naming disagreement resolved
                # honestly rather than by coin flip).
                n_family += 1
                fam = FLAVOR_FAMILIES.get(fieldname, {})
                applied = f.value if fam.get(true_val) == f.value else f.value
                if fam.get(true_val) == f.value:
                    applied = true_val   # counts as correct
            elif f.verdict == "inherited":
                applied = f.value
                if applied == true_val:
                    n_inherited_ok += 1
                else:
                    n_inherited_bad.append((arxiv_id, fieldname, applied, true_val))
            else:
                applied = f.value
            if applied == true_val or (applied is None and true_val is None):
                verdict_ok += 1
            elif applied is None and true_val is not None:
                # abstained where a truth existed: counted separately —
                # honest but not wrong
                verdict_ok += 0
            else:
                verdict_bad.append((arxiv_id, fieldname, applied, true_val))

    print(f"\n== flavor benchmark: {n_papers} papers, "
          f"{len(scored)} (candidate,label) pairs ==")
    ga_new = [auroc(g) for g, _ in grouped]
    ga_old = [auroc(g) for _, g in grouped]
    ga_new = [a for a in ga_new if a is not None]
    ga_old = [a for a in ga_old if a is not None]
    g_new = sum(ga_new) / len(ga_new) if ga_new else None
    g_old = sum(ga_old) / len(ga_old) if ga_old else None
    a_new, a_old = auroc(scored), auroc(naive)
    def stat_grouped(sample):
        vals = [auroc(g) for p in sample for g in p["groups"]]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def stat_pooled(sample):
        return auroc([pr for p in sample for pr in p["pairs"]])

    ci_g = bootstrap_ci(by_paper, stat_grouped)
    ci_p = bootstrap_ci(by_paper, stat_pooled)
    fmt = lambda ci: f"  [95% CI {ci[0]:.3f}–{ci[1]:.3f}]" if ci else ""
    print(f"grouped AUROC (per paper+field, the deployed decision): "
          f"scorer {g_new:.3f}{fmt(ci_g)}   naive {g_old:.3f}   "
          f"({len(ga_new)} groups)")
    print(f"pooled AUROC (cross-paper calibration):                "
          f"scorer {a_new:.3f}{fmt(ci_p)}   naive {a_old:.3f}")
    a_pv = auroc(postveto)
    if a_pv is not None:
        print(f"pooled AUROC, POST-VETO (the ranking we actually ship): "
              f"{a_pv:.3f}   ({len(postveto)}/{len(scored)} candidates)")
    for fieldname, pairs in sorted(per_field.items()):
        a = auroc(pairs)
        print(f"  {fieldname:11} pooled "
              f"{'n/a (single class)' if a is None else f'{a:.3f}'} "
              f"({len(pairs)} pairs)")
    total = sum(len(t) for aid, t in TRUTH.items()
                if (CACHE / f"{aid}.tex").exists())
    print(f"verdicts: {verdict_ok}/{total} correct, {contested} contested, "
          f"{len(verdict_bad)} WRONG")
    print(f"  inheritance: {len(inherit_papers)} papers resolved an ancestor "
          f"({', '.join(f'{a}<-{b}' for a, b in inherit_papers[:6])}"
          f"{'...' if len(inherit_papers) > 6 else ''}); "
          f"inherited fields {n_inherited_ok} correct, "
          f"{len(n_inherited_bad)} wrong")
    for aid, fieldname, applied, true_val in n_inherited_bad:
        print(f"    inherited-wrong: {aid} {fieldname}: {applied} vs {true_val}")
    if n_family:
        print(f"  family-level assertions (GLU naming soup): {n_family}")
    for aid, fieldname, applied, true_val in verdict_bad:
        print(f"  WRONG: {aid} {fieldname}: applied {applied}, truth {true_val}")
    if g_new is not None and g_old is not None and g_new < g_old:
        print("REGRESSION: grouped AUROC under naive baseline")
        return 1
    if verdict_bad:
        print("FAIL: wrong assertions exist (worse than abstaining)")
        return 1
    print("BENCH-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
