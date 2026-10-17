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
