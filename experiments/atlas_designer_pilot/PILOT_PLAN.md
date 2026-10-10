# ATLAS designer: prospective Colab pilot preparation

28 September 2026. DRAFT operational pilot, not a preregistration or permission
to launch. No sessions created. Read microtorch/AGENTS.md before execution.

Question: can the implemented contrast model use fresh matched-seed observations
to propose informative normalisation x activation x residual experiments?
This extends ATLAS; it does not repeat any completed study.

## Concrete initial design

Finite FlexLM space: layernorm/rmsnorm, GELU/ReLU, residual/highway (8 combinations).
Fixed exact attention, learned positions, width 128, 2 layers, 4 heads, context
128, batch 4, AdamW at 0.001. Use the runner's existing TinyStories corpus and
chat7b vocabulary capped at 4096. Evaluation metric: logged validation loss at
step 1200. No best-checkpoint selection. These are component substitutions at
fixed width, not parameter/FLOP-matched architectures: highway adds parameters.
Record parameter counts and timings rather than claiming matched capacity.

1. **Timing/operations probe:** 2 configurations (baseline and rmsnorm+ReLU+highway),
   seed 301, 200 steps, checkpoints every 50, evaluation every 50. One L4 session,
   one job, two CPU threads on the VM, 2-hour maximum driver allocation. This
   bounds the initial attempt; it is NOT a claim that it will take two hours or
   that two probes estimate every configuration's pace. Probe data do not enter
   the scientific model. Establish actual run times and durable receipt/checkpoint
   relay before increasing the allocation. Do not force a reclaim to retest the
   author's already-validated resume implementation.
2. **Initial information batch:** baseline plus each single substitution,
   seeds 311–313: 4 configurations x 3 seeds = 12 runs, 1200 steps each,
   checkpoints every 100, evaluation every 100. The main effects are observed;
   pairwise interactions deliberately remain uncertain. This is a small pilot,
   not the twelve-seed confirmatory study. Single-substitution designs cannot
   support empirical claims about interactions before the next batch arrives.
3. **Adaptive batch, not generated yet:** select at most three configurations
   plus their common baseline on three NEW seeds (321–323), at most 12 runs.
   Use measured per-configuration cost estimates and the frozen target list.
   A planned matched-seed baseline is charged and measured again in the new
   block. Preserve within-seed covariance. Do not select an outcome-dependent
   noise model or alter the basis after seeing results without documenting it.

Potential maximum: 2 probe + 12 initial + 12 adaptive = 26 new runs. Only the
first two have a proposed operational allocation today. Initial/adaptive GPU
hours are UNKNOWN until the probe: scale observed seconds/step to 1200, include
evaluation/checkpoint overhead, build/upload overhead and reclaim contingency.
If projected work exceeds a few GPU hours, obtain the author's budget decision
with GPU type, one-session count and expected hours, as AGENTS.md requires.

## Launch prerequisites and commands

Use the existing `tools/colab_transfer_runner.py`; no new launcher. It clones
remote master and fingerprints its build inputs. New local sweeps are currently
unpublished, so **remote availability of the exact sweep and compatible trainer
must be resolved before any session is allocated**. Do not push unrelated or
uncommitted work implicitly. Capture the actual source/binary/data identity in
receipts. No staging or publishing was done in this preparation.

Run the documented session/orphan/adoption-list preflight from a saved .sh file
in WSL, not a variable-filled inline PowerShell command. Leave unrelated sessions
alone, never update the 0.6.0 CLI, never remove HALTED, plan for at most three
account sessions (this pilot uses ONE), and never use the laptop GPU.

After those prerequisites, the probe driver command is:

```sh
python3 tools/colab_transfer_runner.py \
  --sweep experiments/atlas_designer_pilot/sweep_probe.json \
  --session atlas-dprobe --local-out /mnt/c/ml_artifacts/atlas_designer_pilot/probe \
  --expect 2 --gpu L4 --jobs 1 --omp 2 --max-hours 2 --partial-every 300
```

Run it in a foreground WSL terminal that stays open. Do not use nohup, a detached
Windows launcher, or a periodic heartbeat as a substitute for the live driver.
Keep its log under /mnt/c/ml_artifacts/atlas_designer_pilot/, not /tmp or OneDrive.
Check that the driver elapsed time increases and its log advances a few minutes
after launch. A completion count on LOCAL DISK is progress; remote log output
alone is not. Slow exec calls are not proof the session is dead. Session-key
refresh, adoption, chunked transfer, partial relay and resume stay with the
existing driver. On completion, confirm owned session cleanup and copy receipts
into this study before analysing them.

## Remaining scientific preparation

Before the initial batch: fix a prospective target list (the three pairwise
difference-in-differences at declared reference contexts), prior/noise assumptions
and sensitivity reporting; write an adapter from NEW run receipts to scoped seed
blocks. The operational pilot does not establish uncertainty calibration or
superiority over a conventional design. A budget-matched comparison needs a
separate pre-specified allocation and fresh evaluation seeds; it is not silently
included in the 26-run ceiling above. No "Colab-ready" completion claim until
source delivery, data identity, foreground ownership and allocation are resolved.
