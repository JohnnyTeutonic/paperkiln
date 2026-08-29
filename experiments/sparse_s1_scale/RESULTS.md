# Rung B RESULT (width-only, Amendment 1) — 13 Aug 2026

Local CPU execution per the EXECUTION NOTE; receipts/ holds every
run's events.jsonl + result.json. Guards green (d=256, layers=2, lane
from each run's own model event); no early stops; trace vs result.json
best_val agree on all 10 runs.

**H-SCALE-SHRINK: SUPPORTED at Rung B — also two-tailed.**
- shrink_s = Delta_s(1200) - Delta_s(400), per-seed all positive:
  [+0.0069, +0.0210, +0.0458, +0.0488, +0.0446]
- mean +0.0334, t = 4.03, df = 4 (one-tailed crit 2.132, two-tailed
  2.776). The budget-conditionality effect recurs at d=256: SWA's
  early advantage shrinks as budget grows.

**D-SIGNS (descriptive):** Delta(400) = -0.0613 (5/5 seeds negative,
t = -6.51); Delta(1200) = -0.0279 (5/5 seeds still negative,
t = -3.36). NOTE the shape difference from d=128: at this width SWA
is ahead at BOTH budgets — the gap shrinks with budget but the
crossover to an exact-attention advantage has not happened by 1200
steps at d=256. The shrink direction is the pre-registered claim and
it held; the crossover point moving with width is a new observation,
descriptive only, a candidate question for the next rung.

Scope: TinyStories slice, d=256, 2 layers, T=256, 1200 steps, 5
paired seeds, house protocol. The depth rung (d=256, L=4, exact vs
swa — capability-ready since deep SWA landed) is future work under
its own pre-registration.
