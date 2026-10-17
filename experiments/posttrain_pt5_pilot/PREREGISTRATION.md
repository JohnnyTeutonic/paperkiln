# posttrain_pt5_pilot — choosing β so DPO keeps the model answering (pre-registration)

*A calibration pilot for PT5, in the sense of charter 0001 §3.2. Its runs
are excluded from every confirmatory analysis and produce no registry row.
The licence anchor is the commit adding this file, `run_pilot.sh` and
`analyze_pilot.py`.*

## Why

The PT5 driver smoke run (60-step base, 20 DPO steps) showed several arms
collapsing out of the answer format after DPO: `hack_frac` fell from 0.94
at the base to 0.00 in Mu and Mw. Every pair differs only in its number
token, so this is DPO's likelihood displacement. Pushing down the rejected
answer moves probability onto tokens no pair showed. An arm that stops
answering would score a low judge–truth gap because the judge has nothing
to accept, which PT5's primary metric would misread. Whether this happens
on a real base is the pilot's question, and β is the lever: a larger β
holds the policy closer to the reference.

## Design

- **Seed:** 391. One base: the calibration recipe, R = 20, 3000 steps.
- **Grid:** β ∈ {0.1, 0.5, 2.0}.
- **Arms:** E, J, R and Mu, built by the PT5 driver as specified in its
  pre-registration (format-matched pools, equal N).
- **Training:** `provenance_po` from the base, 400 steps, lr 1e-3, batch 8
  pairs. The base and its samples are trained once and shared across β.
- **Measurement:** `answer_rate`, the fraction of test samples passing the
  format verifier (8 per test prompt, temperature 1.0, seed 17), for every
  arm and for the base. Descriptive alongside it: `truth_acc` and
  `judge_truth_gap_25`.

## Decision rule (`analyze_pilot.py`)

- **Choose the smallest β** in the grid at which every arm's `answer_rate`
  is at least 0.90.
- **If no β qualifies:** outcome `NO_BETA`. Halt. PT5 is not licensed until
  a dated amendment adds a remedy chosen in writing, such as an NLL term
  on the chosen completions.
- **PT5 is licensed with the chosen β.** Its pre-registration records the
  pilot and this outcome.

## Round 1 result (17 Oct 2026): NO_BETA

`answer_rate` on the test set (base 1.000):

| β | E | J | R | Mu |
|---:|---:|---:|---:|---:|
| 0.1 | 0.159 | 0.994 | 0.160 | 0.002 |
| 0.5 | 0.002 | 0.000 | 0.539 | 0.396 |
| 2.0 | 0.330 | 0.608 | 0.787 | 0.000 |

The training trajectories (`receipts/round1/`) show the mechanism:
- **At β = 0.1**, training margins climb to 5–8 nats while held-out
  preference accuracy stays near 0.72. The margin grows by pushing both
  completions down, the rejected faster: likelihood displacement.
- **At β = 2.0**, gradient norms of 10–20 against a clip of 1 make the
  optimisation erratic.

So plain DPO at this scale and learning rate does not keep the model
answering at any β in the grid. As the rule requires, PT5 is not licensed
on round 1.

## Amendment 1 (17 Oct 2026, after round 1, before round 2)

**The remedy chosen in writing:** an NLL term on the chosen completions
(RPO; Pang et al. 2024, "Iterative Reasoning Preference Optimization"; to
be checked against the paper before external citation):

  loss = DPO + α · Σ_p w_p · mean_t(−log p(chosen_t)) / Σ_p w_p

- **Weights:** the NLL term uses the same pair weights as the DPO term, so
  the Mu, Mw and Md arms still differ only in weighting.
- **Implementation:** `train.po.nll`, as `ops::policy_gradient_kl` with
  zero KL weight. That op is already gradchecked.

**Round 2:**
- **Fixed:** β = 0.1 and lr 1e-3, as in the PT5 draft.
- **Grid:** α ∈ {0.2, 1.0, 5.0}.
- **Data:** the same seed, base and samples as round 1 (`run_pilot2.sh`).
- **Rule:** choose the smallest α at which every arm has answer_rate ≥ 0.90
  (`analyze_pilot.py WORK --round2`).
- **If no α qualifies:** NO_BETA again. Halt, and PT5 waits for a further
  amendment.

PT5 is licensed with β = 0.1 and the chosen α.

## Round 2 result (17 Oct 2026): CHOSEN, α = 0.2

| α | E | J | R | Mu |
|---:|---:|---:|---:|---:|
| 0.2 | 0.996 | 0.994 | 1.000 | 0.989 |
| 1.0 | 0.996 | 1.000 | 0.998 | 0.998 |
| 5.0 | 0.996 | 1.000 | 0.998 | 0.998 |

The NLL term restores the answer format at every α. The rule picks the
smallest, 0.2.

Descriptively, `truth_acc` at α = 0.2 sits just below the base for every
arm (base 0.310; E 0.289). At α = 5 the NLL term dominates and acts mostly
as SFT on the chosen answers (E 0.511). Receipts: `receipts/round2/`.
PT5 is licensed with β = 0.1 and α = 0.2.
