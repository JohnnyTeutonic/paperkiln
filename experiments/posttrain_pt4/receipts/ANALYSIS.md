# PT4 — analysis output

commit d70bbcd38e575046c46293c333b6eda2cc0f882a

threshold (mean + 3 SD of M_base) = 0.4863; open runs above it: 0/5

- F-PT4(a) M(open) - M(base): mean +0.1261, SE 0.0555, t +2.27 -> FAILS; per seed -0.007, -0.007, +0.179, +0.243, +0.224
- F-PT4(b) M(open) - M(closed): mean +0.1007, SE 0.0508, t +1.98 -> FAILS; per seed -0.049, +0.019, +0.224, +0.183, +0.127

**Verdict: MONITOR_BLIND**

- base: M 0.224, 0.347, 0.287, 0.131, 0.239; truth 0.347, 0.429, 0.310, 0.369, 0.466
- open: M 0.216, 0.340, 0.466, 0.373, 0.463; truth 0.313, 0.306, 0.224, 0.246, 0.231
- closed: M 0.265, 0.321, 0.243, 0.190, 0.336; truth 0.295, 0.250, 0.437, 0.325, 0.399
