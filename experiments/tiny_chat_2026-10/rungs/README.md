# Rungs lost and rescored (the early-rank analysis, amendment 1)

`analyze.py`'s `rung_analysis` reads each block's `rung_<job>/rung.json` from the artifacts
(C:/ml_artifacts/tiny_chat/<block>/). This folder records the rungs that have none, and why.

## Lost to VM replacement: missing, never imputed

Ten rungs (`lost_rungs.json`). Before paperkiln 11fc9317 (25 Oct 2026 09:49 +1100) a rung snapshot lived
only on the VM that took it, so a replacement VM failed every unscored rung job of a run it resumed
("rung lost: no snapshot of ... on this VM"). As amendment 1 says, each is recorded as lost and never
imputed; `rung_analysis` counts it as missing (a rung with fewer than max(8, 80%) of the block's runs is
INCOMPLETE).

| block | run | lost steps |
|---|---|---|
| E1 (20M) | pt_e1_L16_r1 | 1000, 2500, 5000 |
| E1 (60M) | pt_e1_60m_L32_r1 | 2500, 5000 |
| E5 | pt_e5_s17_r2 | 1000, 2500, 5000 |
| E5 | pt_e5_s35_r2 | 500, 1000 |

From 11fc9317 on, snapshots come home and go back to a replacement VM.

## Lost to a failure: rescored

The MoR block's MoR3 r1 rungs at steps 500, 1000 and 2500 failed when `mtstudio score` ran out of GPU
memory beside the trainer (the vendored transformer_core GEMM buffer cache, fixed in 86a978ca). Their
snapshots were home (C:/ml_artifacts/tiny_chat/e1_mor/pt_e1_mor_MoR3_r1/rung_snapshots/), so they are
rescored at the MoR campaign's pin: `tools/colab_chat/tiny_chat_e1_mor_rescore.json` runs the campaign's
own data job and its three rung jobs argv for argv at branch rescore-mor3-r1 (2047b511 = 86a978ca + that
jobs file), on one RunPod L4 (driver pk-tc-mor-rescore). The rung.json files go to
C:/ml_artifacts/tiny_chat/e1_mor_rescore/; the copies analyze.py reads are in
C:/ml_artifacts/tiny_chat/e1_mor/rung_e1_mor_MoR3_r1_s{500,1000,2500}/rung.json, beside each failed job's log,
and in `mor3_r1_rescore/` here.

Run 26 Oct 2026, 02:18-03:43 (+1100), $0.84. The data job's ledger is identical to the campaign's; the
staged weights' sha256 equal the runner state's records of the snapshots (`mor3_r1_rescore/staged.sha256`),
and each rung.json records them again.

| step | val loss | acc_gold | Y_base | twin (ctx) | panel FineWeb | panel dialogue |
|---|---|---|---|---|---|---|
| 500 | 5.151 | 0.238 | -0.001 | 0.236 | 5.584 | 4.657 |
| 1000 | 4.413 | 0.420 | -0.041 | 0.464 | 4.858 | 4.086 |
| 2500 | 3.929 | 0.445 | -0.027 | 0.469 | 4.356 | 3.675 |
| 5000 (the campaign's own) | 3.660 | 0.464 | -0.038 | 0.500 | 4.078 | 3.446 |

Each rung.json's provenance names paperkiln 2047b511 (86a978ca's tree plus the jobs file).
