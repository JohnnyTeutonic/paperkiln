# Carry-5 results

Set `corpus/eval/carry5_v1.jsonl` (sha256 de5f8b6803efd334), 148 items; scorer mtstudio, generation ember (greedy, max 24 new tokens). Forced choice over 4 candidates (chance 0.25): acc = raw log-prob, acc_norm = per token. Modes: gold = teacher-forced canned history, self = the model's own greedy history, twin = one-turn twin, noctx = the question alone.

## live

Forced choice (acc / acc_norm):

| mode | all | recall | correction | instruction | topic | knowledge |
|---|---:|---:|---:|---:|---:|---:|
| gold | 0.290 / 0.182 | 0.194 / 0.177 | 0.226 / 0.258 | 0.809 / 0.000 | 0.160 / 0.280 | 0.333 / 0.111 |
| self | 0.338 / 0.216 | 0.274 / 0.242 | 0.258 / 0.290 | 0.857 / 0.000 | 0.120 / 0.240 | 0.444 / 0.222 |
| twin | 0.351 / 0.257 | 0.274 / 0.258 | 0.258 / 0.387 | 0.809 / 0.000 | 0.280 / 0.280 | 0.333 / 0.333 |
| noctx | 0.318 / 0.236 | 0.226 / 0.226 | 0.258 / 0.355 | 0.762 / 0.000 | 0.200 / 0.240 | 0.444 / 0.444 |

By distance (target turn minus fact turn; acc, gold / self / twin):

| distance | gold | self | twin | n |
|---|---:|---:|---:|---:|
| 1 | 0.280 | 0.280 | 0.320 | 25 |
| 2 | 0.309 | 0.345 | 0.345 | 55 |
| 3 | 0.262 | 0.333 | 0.381 | 42 |
| 4 | 0.294 | 0.353 | 0.353 | 17 |
| None | 0.333 | 0.444 | 0.333 | 9 |

By interference (distractor facts; acc, gold / self / twin):

| distractors | gold | self | twin | n |
|---|---:|---:|---:|---:|
| 0 | 0.443 | 0.471 | 0.471 | 70 |
| 1 | 0.154 | 0.205 | 0.231 | 39 |
| 2 | 0.154 | 0.231 | 0.256 | 39 |

Wrong with a distractor or old value present (gold history): the choice was a distractor 0.534 of 73 (1/3 to 2/3 by chance, by the number of distractors).
Gaps: {"acc_twin_minus_gold": 0.0609, "acc_gold_minus_self": -0.0473, "acc_norm_twin_minus_gold": 0.0744, "acc_norm_gold_minus_self": -0.0338}; over the context: {"gold": 0, "self": 0, "twin": 0, "noctx": 0}.

Greedy generation (contains / exclusive; instruction family: complies):

| mode | contains | exclusive | complies (instruction) | clean stop | repeats |
|---|---:|---:|---:|---:|---:|
| gold | 0.020 | 0.000 | 0.000 | 0.993 | 0.054 |
| self | 0.027 | 0.000 | 0.000 | 0.932 | 0.061 |
| twin | 0.007 | 0.008 | 0.000 | 0.993 | 0.027 |
| noctx | 0.034 | 0.000 | 0.000 | 1.000 | 0.000 |

| family | contains gold | contains self | contains twin |
|---|---:|---:|---:|
| recall | 0.000 | 0.000 | 0.000 |
| correction | 0.000 | 0.000 | 0.032 |
| instruction | 0.143 | 0.191 | 0.000 |
| topic | 0.000 | 0.000 | 0.000 |
| knowledge | 0.000 | 0.000 | 0.000 |

Self-history conversation by turn:

| turn | clean stop | empty | repeats | hit max-new | history dropped | words |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.987 | 0.013 | 0.007 | 0.000 | 0.000 | 11.0 |
| 2 | 1.000 | 0.000 | 0.047 | 0.000 | 0.000 | 11.95 |
| 3 | 0.993 | 0.000 | 0.054 | 0.007 | 0.000 | 12.51 |
| 4 | 0.980 | 0.007 | 0.061 | 0.013 | 0.000 | 12.66 |
| 5 | 0.885 | 0.115 | 0.034 | 0.000 | 0.000 | 11.07 |

Parity gate: the scorer's argmax equals the server's greedy reply on 740/740 self-history replies.

