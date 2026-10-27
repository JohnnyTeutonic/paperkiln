Placement (574 IF-tiny items, greedy, 64 new tokens): compliance per condition, micro over items [Wilson 95%] and macro over 15 families

| model | user | system | bare | macro user | macro system | macro bare | system - user [95%] | system only / user only |
|---|---|---|---|---|---|---|---|---|
| pksft_v1 | 0.124 [0.099, 0.153] | 0.139 [0.113, 0.170] | 0.065 [0.047, 0.088] | 0.118 | 0.133 | 0.062 | +0.016 [-0.006, +0.037] | 25 / 16 |
| pksft_v2 | 0.160 [0.133, 0.193] | 0.138 [0.112, 0.168] | 0.065 [0.047, 0.088] | 0.153 | 0.133 | 0.062 | -0.023 [-0.054, +0.009] | 36 / 49 |

Per family: user / system / bare

| family | pksft_v1 | pksft_v2 |
|---|---|---|
| avoid_word | 0.00 / 0.05 / 0.25 (n 40) | 0.12 / 0.35 / 0.12 (n 40) |
| end_full_stop | 0.70 / 0.82 / 0.38 (n 40) | 0.75 / 0.85 / 0.47 (n 40) |
| end_with_word | 0.05 / 0.05 / 0.00 (n 40) | 0.12 / 0.00 / 0.00 (n 40) |
| include_word | 0.42 / 0.33 / 0.00 (n 40) | 0.42 / 0.12 / 0.00 (n 40) |
| lowercase | 0.00 / 0.00 / 0.00 (n 40) | 0.00 / 0.00 / 0.00 (n 40) |
| max_words | 0.05 / 0.05 / 0.00 (n 40) | 0.05 / 0.03 / 0.00 (n 40) |
| number_only | 0.00 / 0.00 / 0.00 (n 24) | 0.00 / 0.04 / 0.00 (n 24) |
| numbered_list | 0.12 / 0.07 / 0.00 (n 40) | 0.05 / 0.07 / 0.03 (n 40) |
| one_word | 0.03 / 0.03 / 0.00 (n 40) | 0.05 / 0.03 / 0.00 (n 40) |
| quoted | 0.00 / 0.00 / 0.00 (n 40) | 0.00 / 0.00 / 0.00 (n 40) |
| repeat_phrase | 0.30 / 0.53 / 0.30 (n 40) | 0.65 / 0.20 / 0.30 (n 40) |
| repeat_word | 0.00 / 0.00 / 0.00 (n 30) | 0.00 / 0.00 / 0.00 (n 30) |
| start_with | 0.10 / 0.03 / 0.00 (n 40) | 0.07 / 0.05 / 0.00 (n 40) |
| uppercase | 0.00 / 0.00 / 0.00 (n 40) | 0.00 / 0.00 / 0.00 (n 40) |
| yes_no | 0.00 / 0.05 / 0.00 (n 40) | 0.00 / 0.25 / 0.00 (n 40) |

Turn distance (84 conversations per cell): instr_user, instr_system, nohint; system - user paired

| model | d | user | system | nohint | system - user [95%] |
|---|---|---|---|---|---|
| pksft_v1 | d0 | 0.000 | 0.024 | 0.000 | +0.024 [-0.01, +0.06] |
| pksft_v1 | d1 | 0.190 | 0.214 | 0.167 | +0.024 [-0.02, +0.07] |
| pksft_v1 | d2 | 0.310 | 0.298 | 0.250 | -0.012 [-0.09, +0.07] |
| pksft_v1 | d4 | 0.548 | 0.524 | 0.476 | -0.024 [-0.08, +0.03] |
| pksft_v2 | d0 | 0.012 | 0.000 | 0.000 | -0.012 [-0.04, +0.01] |
| pksft_v2 | d1 | 0.369 | 0.405 | 0.321 | +0.036 [-0.04, +0.11] |
| pksft_v2 | d2 | 0.595 | 0.667 | 0.560 | +0.071 [-0.02, +0.16] |
| pksft_v2 | d4 | 0.702 | 0.714 | 0.643 | +0.012 [-0.01, +0.04] |

| model | prompt check (mismatched / checked, with system) | stops eos / turn / length |
|---|---|---|
| pksft_v1 | 0 / 2158 (798) | 1772 / 0 / 386 |
| pksft_v2 | 0 / 2158 (798) | 1864 / 0 / 294 |

Tool calls (corpus/eval/tool_calls_v1.jsonl, greedy, 96 new tokens)

| model | call made | right name | arguments JSON | arguments equal | false calls | stops eos / turn / length |
|---|---|---|---|---|---|---|
| pksft_v2 | 198/200 | 198/200 | 195/200 | 172/200 | 2/100 | 298 / 0 / 2 |
