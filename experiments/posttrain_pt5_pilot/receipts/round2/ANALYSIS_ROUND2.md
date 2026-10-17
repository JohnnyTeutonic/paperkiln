# posttrain_pt5_pilot — analysis output

Floor: every arm's answer_rate >= 0.9; choose the smallest qualifying beta.

| nll | base | E | J | R | Mu |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 1.000 | 0.996 | 0.994 | 1.000 | 0.989 |
| 1.0 | 1.000 | 0.996 | 1.000 | 0.998 | 0.998 |
| 5.0 | 1.000 | 0.996 | 1.000 | 0.998 | 0.998 |

Descriptive (truth_acc / judge_truth_gap_25):

- nll 0.2: base 0.310/0.162; E 0.289/0.222; J 0.306/0.203; R 0.278/0.200; Mu 0.280/0.181
- nll 1.0: base 0.310/0.162; E 0.396/0.129; J 0.347/0.149; R 0.403/0.127; Mu 0.410/0.140
- nll 5.0: base 0.310/0.162; E 0.511/0.127; J 0.375/0.138; R 0.418/0.119; Mu 0.424/0.118

**Outcome: CHOSEN**: nll = 0.2
