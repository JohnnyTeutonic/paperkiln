# posttrain_pt5_pilot — analysis output

Floor: every arm's answer_rate >= 0.9; choose the smallest qualifying beta.

| beta | base | E | J | R | Mu |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1.000 | 0.159 | 0.994 | 0.160 | 0.002 |
| 0.5 | 1.000 | 0.002 | 0.000 | 0.539 | 0.396 |
| 2.0 | 1.000 | 0.330 | 0.608 | 0.787 | 0.000 |

Descriptive (truth_acc / judge_truth_gap_25):

- beta 0.1: base 0.310/0.162; E 0.011/0.097; J 0.091/0.271; R 0.019/0.063; Mu 0.000/0.045
- beta 0.5: base 0.310/0.162; E 0.000/0.007; J 0.000/0.028; R 0.067/0.043; Mu 0.032/0.131
- beta 2.0: base 0.310/0.162; E 0.103/0.080; J 0.121/0.183; R 0.144/0.220; Mu 0.000/0.035

**Outcome: NO_BETA**
