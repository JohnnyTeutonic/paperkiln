# transfer_s2 stage 1 results (selection rule applied by select_lr.py, verbatim output)

## d = 256 (banked 11 Sep 2026 15:33)

Sources: 1e-3 from transfer_s1 arm S (identical spec); 2e-3, 5e-4, 2.5e-4 from sweep_lr_S.json (receipts/lr_S). Rule: PREREGISTRATION.md stage 1.

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

Reading: the s1 protocol rate 1e-3 is not the best rate even at the base width (median best val 3.369 against 3.228 at 2.5e-4). 5e-4 has the lowest median best val but fails the regime check in 2 of 3 seeds (best val before the last three evals), so the rule passes it over. Per PREREGISTRATION.md, lr*(256) = 2.5e-4 means arm S is re-run at that rate in stage 2 so every arm is on the same footing.

## d = 512 (banked 11 Sep 2026 18:03)

Sources: 1e-3 from transfer_s1 arm M; 5e-4 from the transfer_s1 Threat 4 cell M_lr; 2e-3 and 2.5e-4 from sweep_lr_M.json (receipts/lr_M). Rule: PREREGISTRATION.md stage 1.

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

## d = 1024 (banked 14 Sep 2026 00:22)

Sources: 1e-3 from transfer_s1 arm L (exact lane, seeds 21-23); 5e-4, 2.5e-4 and 1.25e-4 from sweep_lr_L.json (receipts/lr_L). Rule: PREREGISTRATION.md stage 1, applied as written.

```
width 1024: stage-1 table (exact lane, seeds (21, 22, 23))
       lr seed  best_val best_step    final regime
    0.001   21    3.9109      3100   3.9332 FAIL
    0.001   22    3.8936      2400   3.9302 FAIL
    0.001   23    3.9342      3000   3.9464 FAIL
   0.0005   21    3.6320      3100   3.6516 FAIL
   0.0005   22    3.6287      3000   3.6574 FAIL
   0.0005   23    3.6102      3400   3.6209 ok
  0.00025   21    3.1489      3100   3.2701 FAIL
  0.00025   22    3.1579      2700   3.2611 FAIL
  0.00025   23    3.1650      2400   3.2741 FAIL
 0.000125   21    3.1414      2800   3.2277 FAIL
 0.000125   22    3.1461      2700   3.2251 FAIL
 0.000125   23    3.1546      2800   3.1977 FAIL

lr 0.001: median best val 3.9109, median best step 3000, regime 0/3 fail
lr 0.0005: median best val 3.6287, median best step 3100, regime 1/3 fail
lr 0.00025: median best val 3.1579, median best step 2700, regime 0/3 fail
lr 0.000125: median best val 3.1461, median best step 2800, regime 0/3 fail

lr*(1024) = 0.0005   [no grid point passes the regime check; latest median best-val step]
```

Reading, and a note the paper must carry. No grid point passes the regime check at d = 1024: at every rate the best validation loss falls before the last three evals, i.e. the 16x model overfits the slice inside 3600 steps at every rate tried (best step 2400-3400). The rule's fallback clause, "latest median best-val step, ties to the larger rate", therefore decides, and it selects **5e-4** (median best step 3100) over 1.25e-4 (2800) and 2.5e-4 (2700). The fallback was written to prefer the rate still improving when nothing has converged; here the rate it prefers peaks later because it learns more slowly, not because it is healthier: its median best val is 3.63 against 3.15-3.16 at the two lower rates, a gap larger than the whole S-M spread at stage 1. Consequence for stage 2: at 5e-4 the L arm reaches the S-arm milestones down to about 3.6 and not the lower ones, so the matched band at L will again be narrower than at M (whose selected rate reaches 3.22). The rule is applied because it was fixed before any run; whether to amend the fallback (a dated amendment, before the stage-2 L arm is read) is the author's decision and is recorded here so the choice is visible either way. If amended to "lowest median best val, regime advisory", the selection would be 1.25e-4 (3.146), which sits at the grid's lower edge and would itself argue for one more grid point below it.

lr*(1024) = 5e-4 (rule as written).

