# transfer_s2 stage 1 results (selection rule applied by select_lr.py, verbatim output)

## d = 256 (banked 11 Sep 2026 15:33)

Sources: 1e-3 from transfer_s1 arm S (identical spec); 2e-3, 5e-4, 2.5e-4 from sweep_lr_S.json (receipts/lr_S). Rule: PREREG_DRAFT.md stage 1.

```
width 256: stage-1 table (exact lane, seeds (21, 22, 23))
       lr seed  best_val best_step    final regime
    0.002   21    3.8456      3400   3.8617 ok
    0.002   22    3.8059      3600   3.8059 ok
    0.002   23    3.8217      3600   3.8217 ok
    0.001   21    3.4075      3600   3.4075 ok
    0.001   22    3.3690      3600   3.3690 ok
    0.001   23    3.2599      3500   3.2833 ok
   0.0005   21    3.2057      3100   3.2647 FAIL
   0.0005   22    3.2376      2800   3.2518 FAIL
   0.0005   23    3.2144      3600   3.2144 ok
  0.00025   21    3.2280      3600   3.2280 ok
  0.00025   22    3.2283      3600   3.2283 ok
  0.00025   23    3.2217      3600   3.2217 ok

lr 0.002: median best val 3.8217, median best step 3600, regime 3/3 PASS
lr 0.001: median best val 3.3690, median best step 3600, regime 3/3 PASS
lr 0.0005: median best val 3.2144, median best step 3100, regime 1/3 fail
lr 0.00025: median best val 3.2280, median best step 3600, regime 3/3 PASS

lr*(256) = 0.00025   [lowest median best val among regime-passing grid points]
```

Reading: the s1 protocol rate 1e-3 is not the best rate even at the base width (median best val 3.369 against 3.228 at 2.5e-4). 5e-4 has the lowest median best val but fails the regime check in 2 of 3 seeds (best val before the last three evals), so the rule passes it over. Per PREREG_DRAFT.md, lr*(256) = 2.5e-4 means arm S is re-run at that rate in stage 2 so every arm is on the same footing.

## d = 512 (banked 11 Sep 2026 18:03)

Sources: 1e-3 from transfer_s1 arm M; 5e-4 from the transfer_s1 Threat 4 cell M_lr; 2e-3 and 2.5e-4 from sweep_lr_M.json (receipts/lr_M). Rule: PREREG_DRAFT.md stage 1.

```
width 512: stage-1 table (exact lane, seeds (21, 22, 23))
       lr seed  best_val best_step    final regime
    0.002   21    4.0181      3500   4.0293 ok
    0.002   22    3.9914      2100   3.9936 FAIL
    0.002   23    4.0169      3400   4.0193 ok
    0.001   21    3.7216      2800   3.7260 FAIL
    0.001   22    3.7398      2400   3.7644 FAIL
    0.001   23    3.7369      3400   3.7697 ok
   0.0005   21    3.2256      3600   3.2256 ok
   0.0005   22    3.2247      3500   3.2349 ok
   0.0005   23    3.1775      3400   3.1834 ok
  0.00025   21    3.2117      3200   3.2444 FAIL
  0.00025   22    3.1427      2700   3.2083 FAIL
  0.00025   23    3.1945      2600   3.2368 FAIL

lr 0.002: median best val 4.0169, median best step 3400, regime 2/3 PASS
lr 0.001: median best val 3.7369, median best step 2800, regime 1/3 fail
lr 0.0005: median best val 3.2247, median best step 3500, regime 3/3 PASS
lr 0.00025: median best val 3.1945, median best step 2700, regime 0/3 fail

lr*(512) = 0.0005   [lowest median best val among regime-passing grid points]
```
