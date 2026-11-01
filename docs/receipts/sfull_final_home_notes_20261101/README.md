# S-full final-home operational notes N1 and N2

Missing exact-final-step checkpoints now halt after three FINISHED observations instead of waiting for the campaign deadline. The first two counters are persisted in runner state, so ordinary watchdog restarts do not reset the bound. A valid final full checkpoint clears the counter. Corruption is refused immediately. Final validation errors are logged with the job and cause before an exclusive HALTED marker is written and the runner exits.

Independent Codex read-only review: APPROVE, no blocking findings; actual FINISHED branch and relay order inspected, transport tests independently passed (21). Combined checkpoint/transport/operational/DPO transport suite: 87 passed. No GPU calls. Numerical source, 25-step checkpoint interval and 180-second guard unchanged. This addresses Claude's optional N1/N2 recommendations on ac304241b; historical halt evidence remains preserved.
