# Configurable parity depth: CPU integration receipts

The production source hashes and actual compiled binary hashes are in suite_validation.json.
The full CPU+DEVCHECK CTest suite passes all 52 cases. The dedicated native unit checks depth
1/2/3/4, current default-versus-explicit-two equivalence, manual block composition, stacked
sequence isolation and 36 selected whole-model finite differences.

The independent CLI checks pass 21 cases across two recorded invocations: six three-step
trained models (Kimi/SRD/recurrence at depth 1 and 3), twelve early invalid-depth refusals,
and three actual historical-default-two comparisons. The first executed CLI source is
preserved as cli_source_attempt1.py; the final test source includes the historical cases.
Each invocation has its exact source hash, stdout and launcher in cli_validation.json.

The actual trained source score rows, independent NumPy IR rows, and Ember CPU prefill/decode
rows agree within the unchanged 1e-5 bound (observed worst 2.98e-8). Native source scored greedy
decisions and independently free-running Ember greedy choices agree for three prompts and
three generated tokens per model and mode. Independent parameter formula, exported tensor
inventory, model/card depth/counts and recurrent application state counts also pass.

Historical preservation uses the exact pre-depth author-Flex binary recorded by its receipt.
At omitted/default-two depth, all three families match before/after in every named safetensor
dtype, shape and raw value byte, native scored full-row and greedy-row float32 bits, argmax IDs,
and float64 log-probability bits. Only nonsemantic timestamp/path fields are excluded; weights
and substantive scores are not normalized. Corpus/tokenizer input hashes are equal.
This is an actual old-versus-new binary comparison, separate from the current-constructor unit.

The initial native unit failed by reading IR n_layers at the document root instead of
document.model.n_layers. The initial failed log is preserved; correcting that path did not
change production code or weaken numerical assertions.

All runs are CPU only. The historical name Kimi refers to the existing ELU+1 channelwise
mixer, not published Kimi Linear/KDA. Toy three-step runs establish integration and parity,
not useful training, convergence, throughput, GPU support or model quality. Trained models,
corpora and checkpoints have one canonical home outside the repository; cli/ contains raw
result evidence and inventories, not duplicate model files.
