# Carry-5 results

Set `corpus/eval/carry5_v1.jsonl` (sha256 de5f8b6803efd334), 1032 items; scorer mtstudio, generation none (greedy, max 24 new tokens). Forced choice over 4 candidates (chance 0.25): acc = raw log-prob, acc_norm = per token. Modes: gold = teacher-forced canned history, self = the model's own greedy history, twin = one-turn twin, noctx = the question alone.

## live

Forced choice (acc / acc_norm):

| mode | all | recall | correction | instruction | topic | knowledge |
|---|---:|---:|---:|---:|---:|---:|
| gold | 0.336 / 0.203 | 0.241 / 0.215 | 0.241 / 0.241 | 0.854 / 0.021 | 0.261 / 0.250 | 0.350 / 0.283 |
| twin | 0.346 / 0.237 | 0.259 / 0.280 | 0.213 / 0.255 | 0.875 / 0.007 | 0.289 / 0.256 | 0.350 / 0.367 |
| noctx | 0.341 / 0.220 | 0.259 / 0.250 | 0.236 / 0.241 | 0.826 / 0.000 | 0.250 / 0.239 | 0.417 / 0.400 |

By distance (target turn minus fact turn; acc, gold / self / twin):

| distance | gold | self | twin | n |
|---|---:|---:|---:|---:|
| 1 | 0.357 | - | 0.357 | 168 |
| 2 | 0.328 | - | 0.315 | 372 |
| 3 | 0.350 | - | 0.380 | 300 |
| 4 | 0.295 | - | 0.341 | 132 |
| None | 0.350 | - | 0.350 | 60 |

By interference (distractor facts; acc, gold / self / twin):

| distractors | gold | self | twin | n |
|---|---:|---:|---:|---:|
| 0 | 0.446 | - | 0.469 | 480 |
| 1 | 0.265 | - | 0.272 | 276 |
| 2 | 0.217 | - | 0.206 | 276 |

Wrong with a distractor or old value present (gold history): the choice was a distractor 0.566 of 475 (1/3 to 2/3 by chance, by the number of distractors).
Gaps: {"acc_twin_minus_gold": 0.0097, "acc_norm_twin_minus_gold": 0.0339}; over the context: {"gold": 0, "self": 0, "twin": 0, "noctx": 0}.

## chance

Forced choice (acc / acc_norm):

| mode | all | recall | correction | instruction | topic | knowledge |
|---|---:|---:|---:|---:|---:|---:|
| gold | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 |
| self | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 |
| twin | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 |
| noctx | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 | 0.250 / 0.250 |

By distance (target turn minus fact turn; acc, gold / self / twin):

| distance | gold | self | twin | n |
|---|---:|---:|---:|---:|
| 1 | 0.250 | 0.250 | 0.250 | 168 |
| 2 | 0.250 | 0.250 | 0.250 | 372 |
| 3 | 0.250 | 0.250 | 0.250 | 300 |
| 4 | 0.250 | 0.250 | 0.250 | 132 |
| None | 0.250 | 0.250 | 0.250 | 60 |

By interference (distractor facts; acc, gold / self / twin):

| distractors | gold | self | twin | n |
|---|---:|---:|---:|---:|
| 0 | 0.250 | 0.250 | 0.250 | 480 |
| 1 | 0.250 | 0.250 | 0.250 | 276 |
| 2 | 0.250 | 0.250 | 0.250 | 276 |

Wrong with a distractor or old value present (gold history): the choice was a distractor - of 0 (1/3 to 2/3 by chance, by the number of distractors).
Gaps: {"acc_twin_minus_gold": 0.0, "acc_gold_minus_self": 0.0, "acc_norm_twin_minus_gold": 0.0, "acc_norm_gold_minus_self": 0.0}; over the context: {"gold": 0, "self": 0, "twin": 0, "noctx": 0}.

