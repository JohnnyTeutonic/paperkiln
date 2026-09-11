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
