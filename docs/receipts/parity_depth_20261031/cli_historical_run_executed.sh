#!/bin/bash
set -euo pipefail
cd /home/jonat/pk_host_read_final_20261031
R=docs/receipts/parity_depth_20261031
export MTSTUDIO_BIN=/home/jonat/pk_host_read_final_build_20261031/mtstudio
export MTSTUDIO_BASE_BIN=/home/jonat/pk_studio_probes_build_20261031/mtstudio
export EMBER_IR_LOGITS_BIN=/home/jonat/ember_publication_build_20261030/ir_logits
export PARITY_DEPTH_PROOF_DIR=/home/jonat/pk_depth_cli_artifacts_20261031
python3 -m pytest tests/test_parity_depth_cli.py -q -k historical > "$R/cli_historical_attempt1.log" 2>&1
tail -4 "$R/cli_historical_attempt1.log"
