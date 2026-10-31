# S-full data-only admission packet

This is a prepared, unlaunched operational gate. It rebuilds the original frozen data and runs the original full input verifier. It does not build a model, train, evaluate model quality, waive a historical parity failure, or authorise the S-full experiment.

The remote source is f616a0f85447f95f67d0f9bdf6dfef513dc632a3. Relative to frozen 7edb0ca1b it adds exactly four operational files; all prior computational files are unchanged. The modern local controller is based on d95664546. Its runner change adds only the explicit custom boot allowlist entry. The local job and boot bytes must equal the remote Git blobs.

Admission requires root review, published remote source, a fresh manual balance-accessor check, provider cleanup confirmation, and hash-bound home/cleanup states for the preceding ternary and E10 campaigns. No admission has been supplied. The template is deliberately invalid. Root must set a fixed absolute deadline no later than 75 minutes after admission; restarts cannot extend it. The cumulative campaign ceiling is $1.25. A process-local two-pod fence supplements existing global limits without changing them. This wrapper uses the existing watchdog, never a second manual driver.

The relay contains five data products, the full ledger, decontamination report and original verified_inputs report. Local acceptance rechecks every product hash/length and the original full ledger normalisation, plus the original 13 raw-source and two evaluation-source proofs. A job reaching home is insufficient: the original verifier and local relay checks must pass. All failures remain fail-stop with immutable markers and artifacts. No automatic policy or hash correction is allowed.

Runtime dependencies may be unavailable or may produce changed data. The 75-minute ceiling is not a completion guarantee; the original FineWeb shard, human sources and tokenization can exceed it. A mismatch must be retained. This packet makes no claim that remote source availability or rebuilt inputs have already passed.

Run the controller with the proposal argv only after admission. For preparation only, replace the admission option with `--check`; this checks source, exact commands and vocabulary without provider calls. No provider resource or watchdog entry was created during preparation.

Operational integration note, 1 November: the packet is executed from its frozen isolated controller location, not directly from this publication checkout. The watchdog entry must specify a unique process match for the actual admission argument; its default --state-dir match does not occur on this controller command. A duplicate start is rejected by the exclusive controller lock. Preserve the frozen proposal and use a separately reviewed live entry. The packet contains data preparation and original input verification only, with no training follow-up.

The orphan sweeper now separately recognises this exact source-pinned in-process controller only after verifying its complete manifest, queried state and anchored root admission. Direct-runner ownership and deletion requirements are unchanged; a valid live owner remains recognised during deadline-triggered stop/relay. See ../sfull_owner_recognition_20261101/ for portable negative controls and the real-source binding. No keep-list exemption or disabled orphan cleanup is used.
