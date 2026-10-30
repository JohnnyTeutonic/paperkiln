# Registered size extension: five matched seeds

Completed 30 October 2026. This is the final registered A6.5 extension of the Carry-5 size arm: 20M, 60M and 150M non-embedding parameters, depth 8, seeds 1–5. The initial three-seed files and the original 150M pick are preserved. No model was reselected after the extension.

The primary size slope is **EQUIVALENT** under the frozen rule: mean +0.002099 Y per doubling of non-embedding parameters, 90% CI[-0.004249,+0.008446], entirely inside the registered ±0.01 practical-equivalence interval. This means a practically negligible slope in this protocol; it does not establish an exact zero effect or a general limit to scaling. The secondary 60M→150M step remains **NOT_RESOLVED**:−0.009396 Y per doubling,90% CI[-0.030160,+0.011368]. Its direction agrees with the registered positive prediction in only 1/5 seeds; the primary slope is positive in 4/5.

| Non-embedding parameters | Self-history accuracy | Twin-history accuracy | Y=self−twin | Gold-history Y | FineWeb loss | Dialogue loss | SFT validation loss |
|---|---:|---:|---:|---:|---:|---:|---:|
|19,848,640|0.46329|0.53018|−0.06689|−0.05248|3.83479|3.23580|3.02371|
|60,175,104|0.46171|0.51104|−0.04932|−0.04167|3.75574|3.15484|3.15949|
|150,065,344|0.45225|0.51396|−0.06171|−0.04505|3.69612|3.09715|3.19124|

Every entry is the mean over the same five seeds. FineWeb and dialogue panel losses decrease monotonically with size, while the carry score does not: the 60M mean is less negative than either 20M or 150M. Self-history accuracy itself also does not improve. All three registered task floors pass. The negative carry means indicate that self histories score below their twin-history controls; lower language-model loss has not translated into a positive carry advantage here.

The scope is fixed 163.84M pretraining tokens, the registered human-only corpus/mix and training/post-training recipes, and the selected depth 8. That gives approximately 8.2/2.7/1.1 tokens per non-embedding parameter at 20M/60M/150M. Consequently these results cannot distinguish under-training of larger models from a wider limit of scale, and do not compare with state-of-the-art systems. The descriptive isoplot uses extrapolation outside the 20M panel-loss range and is not an additional confirmatory test. Stage L0 on the unchanged150M pick remains a separate descriptive evaluation; the Carry-12 HS1 B/S-gist extension is a different registered comparison.

## Verification and records

The deterministic finalizer published at08:18:07 UTC after all 15 selected size×seed summaries were available. Independent verification rehashed all 11 frozen sources, 29 input paths and 4 reports; the exact seeds are1–5 with no missing cells, and the complete main analysis reproduces the standalone size block. The primary OLS slopes and 90%interval were independently recomputed directly from the 15 selected summaries.

Canonical records: [analysis_size.json](receipts/extension_20261030/analysis_size.json), [complete main analysis](receipts/extension_20261030/analysis_tiny_chat.json), [completion/source/input/report manifest](receipts/extension_20261030/completion.json), [input audit](receipts/extension_20261030/input_receipt.json), [exact commands](receipts/extension_20261030/commands.json), and [independent audit](receipts/extension_20261030/independent_audit.json). Original artifacts remain at `C:/ml_artifacts/tiny_chat/size_extension_final/reports/`.
