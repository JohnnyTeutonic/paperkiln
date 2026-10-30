# Carry-12 registered core — 30 October 2026

The complete seven-cell core has reported seeds 1, 2 and 3. The unchanged registered
analysis ran through the existing watchdog finalizer, with `partial: false`, no missing
cells and every validity gate passing. Independent verification rehashed all 36 frozen
sources, 55 inputs and three published reports without a mismatch. The exact
[analysis and completion receipts](receipts/core_2026-10-30/) preserve the source and
input inventory, command, output hashes and independent audit. Earlier partial analyses
remain historical records.

The clear registered result is **HT-SFT: adding the disjoint human-written template SFT
improved Carry-12's carry metric by 9.16 percentage points over B**. All three paired
differences were positive (10.29, 8.26 and 8.92 points). This is a result on this small
model, training recipe and benchmark, not a state-of-the-art or general dialogue claim.
The secondary trained/untrained family analyses do not resolve whether that improvement
transfers to structures absent from the templates.

## Cell means and the metric

Y12 is self-history forced-choice accuracy minus twin accuracy over recall, correction,
topic, load and knowledge. A less negative value means a smaller history cost relative
to each model's own twin accuracy; raw accuracy is therefore reported alongside it.
All entries below are percentages or percentage-point differences, averaged over three
seeds. W512 was registered for gold-history scoring, so it has no self-history Y12.

| Cell | Self accuracy | Twin accuracy | Y12 | Gold-history Y12 |
|---|---:|---:|---:|---:|
| W512, original context 512 | — | 49.61 | — | −5.99 |
| B, context 1024 | 42.90 | 52.37 | −9.47 | −8.48 |
| D, long human SFT weighting | 42.83 | 52.13 | −9.30 | −8.12 |
| TS, explicit template SFT | 57.22 | 57.54 | −0.32 | −0.58 |
| S-gist, per-turn state | 31.84 | 52.76 | −20.92 | −19.42 |
| P16, plain depth-16 control | 45.53 | 54.30 | −8.77 | −7.51 |
| X, Attention Residuals | 46.23 | 52.10 | −5.87 | −4.11 |

The validity gates require mean twin accuracy ≥35% and twin minus question-alone
accuracy ≥10 points over context families. Every cell passes. W512 reproduced E1's
pretraining and SFT weight hashes in all three seeds, with zero FineWeb panel difference.

## Frozen decisions

These are the unchanged analysis's registered verdicts, not judgments made from the
table's rounded means. SUPPORTED requires the registered direction, a two-sided 5%
paired t test and the registered effect-size threshold; equivalence uses a 90% interval.
The intervals below are the analysis's 90% intervals in percentage points.

| Contrast | Mean | 90% interval | Registered verdict |
|---|---:|---:|---|
| HT: B − W512, gold-history Y12 | −2.49 | [−7.44, 2.46] | NOT_RESOLVED |
| HD: D − B | 0.17 | [−3.26, 3.60] | NOT_RESOLVED |
| HT-SFT: TS − B | 9.16 | [7.41, 10.90] | SUPPORTED |
| HS1: S-gist − B, predicted negative | −11.45 | [−19.73, −3.16] | NOT_RESOLVED |
| HX-main: X − P16 | 2.89 | [−5.53, 11.32] | NOT_RESOLVED |

S-gist is lower than B in every seed (−13.33, −15.14 and −5.87 points), but its
absolute t statistic is 4.035, below the frozen 4.303 cutoff. Its 90% interval does
not replace that registered test. The analysis flags a five-seed extension; **S-full
does not trigger**, because HS1 is not SUPPORTED. No conditional paid campaign was
launched from this result.

HX's two distance tests are also NOT_RESOLVED: the far-minus-near recall interaction
is −0.14 points, and the correction-distance slope is −0.20 points per turn. The
registered joint H-gap is consequently NOT_RESOLVED. X's FineWeb panel loss is lower
than P16's by 0.05883 nats (the secondary test is SUPPORTED), but this does not establish
a carry benefit. Its isoplot residual is unresolved and extrapolates beyond the two
plain cells' loss range in every seed; the analysis flags those limitations.

## What the TS result does and does not establish

TS improves both absolute self accuracy (57.22% versus B's 42.90%) and the carry metric;
the result is not merely a reduction in twin accuracy. The secondary gains in trained
families (recall/correction: 10.08 points) and untrained families (topic/load: 7.92 points)
are individually NOT_RESOLVED, as is their difference. All three seeds have positive
gains in each pool, and the shared decision helper emits extension flags for both.
Section 9 limits the registered extension rule to primary contrasts: those secondary
flags do not authorize additional campaigns or establish transfer.

The checks beside HT-SFT do not show supported degradation: Carry-5's difference is
+5.11 points but unresolved; SFT dialogue loss changes by −0.01838 nats and is unresolved;
FineWeb loss changes by −0.01188 nats, contradicting the registered worsening direction.
Descriptive chat checks are mixed: clean stops are unchanged on chat and 2.45 points
higher on chat2, while repetition rises by about 2 points and distinct bigrams fall by
0.76/0.91 points. QA's overall accuracy is unchanged at 16.67%. Instruction compliance
remains near zero (B 0%, TS 0.46%). These are narrow diagnostics, not evidence that TS
has become a broadly useful conversational model.

## Remaining work

The three-seed core is complete. HS1 alone meets section 9's primary-contrast extension
rule (NOT_RESOLVED and one sign in all three seeds): it requires B and S-gist at seeds
4 and 5, once. The shared helper also flags secondary comparisons, but those are outside
that launch rule. Exact flags and per-seed values remain in the JSON; follow-up scope
and spending require review before launch. Stage L0 on the registered 150M pick and the
matched five-seed size extension are separate campaigns and were still running when
this core receipt was written.
