# Carry-12 results

Set `corpus/eval/carry12_v1.jsonl` (sha256 de24f2c726a1006e), 1452 items; scorer mtstudio, generation ember (greedy, max 24 new tokens, self history to the target). Forced choice over 4 candidates (chance 0.25), acc = raw log-prob. Modes: gold (canned history), self (own replies), near (events without small talk), near_self, twin (one turn), noctx (question alone).

## pick_e1_150m_L8

Generation engine: ember

| mode | all | recall | correction | topic | load | instruction | knowledge | over ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold | 0.441 | 0.423 | 0.322 | 0.381 | 0.833 | 0.819 | 0.417 | 0 |
| self | 0.425 | 0.406 | 0.386 | 0.372 | 0.808 | 0.306 | 0.500 | 0 |
| near | 0.490 | 0.473 | 0.361 | 0.436 | 0.867 | 0.833 | 0.550 | 0 |
| near_self | 0.466 | 0.465 | 0.386 | 0.447 | 0.817 | 0.306 | 0.550 | 0 |
| twin | 0.484 | 0.485 | 0.417 | 0.411 | 0.892 | 0.444 | 0.550 | 0 |
| noctx | 0.256 | 0.244 | 0.278 | 0.214 | 0.250 | 0.222 | 0.533 | 0 |

Pools (acc): Y12_self -0.055, Y12_gold -0.065, split_cost_gold -0.015, distance_cost_gold -0.047, split_cost_self -0.013, distance_cost_self -0.043, own_replies_cost 0.007
Validity gate: {"twin_primary": 0.4862, "twin_primary_at_least_0.35": true, "twin_minus_noctx_context": 0.2378, "twin_minus_noctx_at_least_0.10": true}
HL0 slopes: {"slope_gold_on_distance": {"slope": 0.01026, "n": 1320}, "slope_gold_minus_near_on_distance": {"slope": 0.00509, "n": 1320}, "slope_self_on_distance": {"slope": 0.00373, "n": 1320}}

By distance (context families; acc): 

| distance | gold | near | self | twin | n |
|---|---:|---:|---:|---:|---:|
| 1 | 0.611 | 0.389 | 0.722 | 0.556 | 18 |
| 2 | 0.333 | 0.500 | 0.444 | 0.444 | 18 |
| 3 | 0.361 | 0.306 | 0.333 | 0.361 | 36 |
| 4 | 0.306 | 0.389 | 0.389 | 0.361 | 36 |
| 5 | 0.167 | 0.306 | 0.250 | 0.417 | 36 |
| 6 | 0.369 | 0.441 | 0.381 | 0.476 | 84 |
| 7 | 0.380 | 0.513 | 0.380 | 0.449 | 158 |
| 8 | 0.495 | 0.516 | 0.457 | 0.516 | 188 |
| 9 | 0.484 | 0.561 | 0.471 | 0.504 | 244 |
| 10 | 0.419 | 0.458 | 0.386 | 0.496 | 236 |
| 11 | 0.511 | 0.494 | 0.477 | 0.511 | 178 |
| 12 | 0.471 | 0.510 | 0.385 | 0.490 | 104 |
| 13 | 0.482 | 0.446 | 0.393 | 0.375 | 56 |

By target turn (acc):

| target | gold | near | self | n |
|---|---:|---:|---:|---:|
| 10 | 0.457 | 0.520 | 0.457 | 300 |
| 11 | 0.435 | 0.489 | 0.413 | 276 |
| 12 | 0.443 | 0.457 | 0.400 | 300 |
| 13 | 0.420 | 0.493 | 0.409 | 276 |
| 14 | 0.447 | 0.490 | 0.443 | 300 |

Greedy generation at the target:

| mode | contains | exclusive | complies (instruction) | clean stop | repeats |
|---|---:|---:|---:|---:|---:|
| gold | 0.081 | 0.077 | 0.208 | 0.995 | 0.037 |
| self | 0.082 | 0.079 | 0.000 | 0.968 | 0.085 |
| near | 0.154 | 0.154 | 0.194 | 0.976 | 0.035 |
| near_self | 0.174 | 0.167 | 0.000 | 0.895 | 0.151 |
| twin | 0.304 | 0.270 | 0.056 | 0.672 | 0.304 |
| noctx | 0.025 | 0.012 | 0.000 | 0.785 | 0.170 |

Self-history conversation by turn:

| turn | n | clean stop | empty | repeats | hit max-new | history dropped | words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1452 | 0.931 | 0.000 | 0.036 | 0.069 | 0.000 | 10.47 |
| 2 | 1452 | 0.847 | 0.000 | 0.117 | 0.153 | 0.000 | 11.07 |
| 3 | 1452 | 0.886 | 0.000 | 0.105 | 0.114 | 0.000 | 10.31 |
| 4 | 1452 | 0.920 | 0.000 | 0.094 | 0.080 | 0.000 | 9.76 |
| 5 | 1452 | 0.926 | 0.000 | 0.089 | 0.074 | 0.000 | 9.76 |
| 6 | 1452 | 0.940 | 0.000 | 0.064 | 0.060 | 0.000 | 9.34 |
| 7 | 1452 | 0.948 | 0.000 | 0.076 | 0.052 | 0.000 | 8.89 |
| 8 | 1452 | 0.961 | 0.000 | 0.067 | 0.039 | 0.000 | 8.76 |
| 9 | 1452 | 0.958 | 0.000 | 0.073 | 0.042 | 0.000 | 8.69 |
| 10 | 1452 | 0.957 | 0.000 | 0.063 | 0.043 | 0.000 | 8.41 |
| 11 | 1152 | 0.965 | 0.000 | 0.076 | 0.035 | 0.000 | 8.33 |
| 12 | 876 | 0.969 | 0.000 | 0.080 | 0.031 | 0.000 | 8.13 |
| 13 | 576 | 0.958 | 0.000 | 0.080 | 0.042 | 0.000 | 8.07 |
| 14 | 300 | 0.980 | 0.000 | 0.100 | 0.020 | 0.000 | 7.35 |

Parity gate: the scorer's argmax equals the server's greedy reply on 22956/22956 self and near_self replies.


