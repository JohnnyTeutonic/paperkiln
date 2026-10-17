# PT2 — analysis output

level R = 20; commit e10514e57ede94076b88e097561a8c7243962953

- **H1 (q=0.10)** judge_truth_gap_10: J10 - R10: mean +0.0043, SE 0.0051, t +0.84, signs +7/-2 -> **NOT_RESOLVED**
  per seed: +0.007, +0.015, +0.002, +0.021, +0.009, +0.000, +0.015, +0.017, -0.034, -0.009
- **H1 (q=0.25)** judge_truth_gap_25: J25 - R25: mean +0.0050, SE 0.0098, t +0.51, signs +5/-5 -> **NOT_RESOLVED**
  per seed: +0.037, -0.022, -0.006, +0.002, -0.002, +0.013, +0.060, -0.052, -0.002, +0.022
- **H2 (q=0.10)** truth_acc: E - J10: mean +0.0588, SE 0.0186, t +3.17, signs +7/-3 -> **SUPPORTED**
  per seed: -0.015, +0.054, -0.002, +0.121, +0.080, +0.060, +0.123, +0.073, -0.034, +0.127
- **H2 (q=0.25)** truth_acc: E - J25: mean +0.0965, SE 0.0158, t +6.10, signs +9/-1 -> **SUPPORTED**
  per seed: +0.090, +0.101, +0.106, +0.097, +0.082, +0.080, +0.196, +0.131, -0.007, +0.090
- **H3 (q=0.10)** truth_acc: R10 - J10: mean -0.0069, SE 0.0125, t -0.55, signs +5/-5 -> **NOT_RESOLVED**
  per seed: +0.004, +0.037, -0.019, -0.026, +0.006, +0.006, -0.004, -0.060, -0.071, +0.058
- **H3 (q=0.25)** truth_acc: R25 - J25: mean +0.0052, SE 0.0156, t +0.34, signs +7/-3 -> **NOT_RESOLVED**
  per seed: +0.028, +0.026, +0.041, +0.007, -0.028, +0.011, +0.082, +0.017, -0.034, -0.099

Descriptive:
- truth_acc: E - X: mean +0.1433, SE 0.0196, t +7.30
- hack_frac: E - base: mean -0.0071, SE 0.0052, t -1.38
- distractor_frac: E - base: mean +0.0131, SE 0.0059, t +2.24
- hack_frac: J10 - base: mean -0.0074, SE 0.0051, t -1.45
- distractor_frac: J10 - base: mean -0.0012, SE 0.0046, t -0.26
- hack_frac: R10 - base: mean +0.0036, SE 0.0048, t +0.74
- distractor_frac: R10 - base: mean +0.0006, SE 0.0048, t +0.12
- hack_frac: J25 - base: mean -0.0068, SE 0.0034, t -2.01
- distractor_frac: J25 - base: mean +0.0051, SE 0.0054, t +0.93
- hack_frac: R25 - base: mean +0.0009, SE 0.0039, t +0.23
- distractor_frac: R25 - base: mean +0.0060, SE 0.0054, t +1.10
- hack_frac: X - base: mean -0.0057, SE 0.0046, t -1.24
- distractor_frac: X - base: mean -0.0080, SE 0.0032, t -2.49
- K per seed: [2355, 1804, 1920, 1315, 1977, 1264, 1441, 2293, 1958, 2560]
