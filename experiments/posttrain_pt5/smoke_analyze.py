#!/usr/bin/env python3
"""D3 smoke test for PT5: every verdict fires on fabricated results, and the
pair construction keeps its promises."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402
import run_pt5 as P  # noqa: E402


def fab(gap_j_minus_r=0.0, gap_mu_minus_mw=0.0, truth_md_minus_mw=0.0, noise=0.01, seed=0):
    rng = random.Random(seed)
    res = []
    for s in range(301, 311):
        m = {k: {"truth_acc": 0.4, "judge_truth_gap_25": 0.1, "distractor_frac": 0.02} for k in A.MODELS}
        m["J"]["judge_truth_gap_25"] += gap_j_minus_r + rng.gauss(0, noise)
        m["Mu"]["judge_truth_gap_25"] += gap_mu_minus_mw + rng.gauss(0, noise)
        m["Md"]["truth_acc"] += truth_md_minus_mw + rng.gauss(0, noise)
        res.append({"seed": s, "N": 100, "composition": {}, "models": m})
    return res


def expect(label, res, key, want):
    got = A.decide(res)[key]["verdict"]
    assert got == want, (label, got)
    print(f"  {label:42s} {key}: {got}")


def main():
    expect("judge pairs fool the judge more", fab(gap_j_minus_r=0.05), "F_PT5a", "SUPPORTED")
    expect("judge pairs fool it less", fab(gap_j_minus_r=-0.05), "F_PT5a", "CONTRADICTED")
    expect("no difference", fab(noise=0.02), "F_PT5a", "NOT_RESOLVED")
    expect("weighting helps, no truth cost", fab(gap_mu_minus_mw=0.05), "F_PT5b", "PROVENANCE_WEIGHTING_EARNED")
    expect("weighting inert", fab(), "F_PT5b", "WEIGHTING_INERT")
    expect("weighting helps but dropping wins on truth", fab(gap_mu_minus_mw=0.05, truth_md_minus_mw=0.08),
           "F_PT5b", "DROP_BEATS_WEIGHT")
    try:
        A.decide(fab()[:9])
    except SystemExit as e:
        print(f"  refused: {e}")
    else:
        raise AssertionError("decide accepted an incomplete design")

    # Pair construction on fabricated samples: 40 prompts x 8 samples.
    rng = random.Random(4)
    prompt_of, text, exact, judge = [], [], [], []
    for p in range(40):
        for k in range(8):
            e = int(rng.random() < 0.4)
            j = e if e else int(rng.random() < 0.25)
            prompt_of.append(f"p{p}")
            text.append(f"the answer is {k} .")
            exact.append(e)
            judge.append(j)
    pl = P.pools(prompt_of, text, exact, judge, 301)
    arms, N, comp = P.build_arms(pl, exact, 301)
    assert all(len(v) == N for v in arms.values()) and N % 2 == 0
    assert arms == P.build_arms(pl, exact, 301)[0]                       # deterministic
    assert all(exact[a] == 1 and exact[b] == 0 for a, b, _ in arms["E"])
    assert all(judge[a] == 1 and judge[b] == 0 for a, b, _ in arms["J"])
    assert sum(src == "E" for _, _, src in arms["Mu"]) == N // 2
    assert arms["Mw"] is arms["Mu"] and arms["Md"] is arms["Mu"]
    # R carries J's error composition (in expectation): ties and inversions.
    r_tie = sum(src == "Rtie" for _, _, src in arms["R"]) / N
    r_inv = sum(1 for a, b, src in arms["R"] if src != "Rtie" and exact[a] == 0 and exact[b] == 1) / N
    assert abs(r_tie - comp["f_tie"]) < 0.02 and abs(r_inv - comp["f_inv"]) < 0.08, (r_tie, comp, r_inv)
    print(f"  pairs: N={N}, J f_tie={comp['f_tie']:.3f} f_inv={comp['f_inv']:.3f}; R tie={r_tie:.3f} inv={r_inv:.3f}")
    print("smoke_analyze: every verdict fires; pair construction holds")


if __name__ == "__main__":
    main()
