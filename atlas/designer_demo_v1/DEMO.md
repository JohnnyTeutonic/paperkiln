# ATLAS synthetic design loop

Synthetic software demonstration; no empirical architectural finding.

The initial observations vary normalisation and activation separately.
The requested target is their difference-in-differences interaction.

Planted synthetic interaction: -0.8. Initial posterior mean: 0.000000; SD: 1.000000.
Planner selects 3 configurations plus a shared baseline over two new seeds:

- `{'norm': 'rmsnorm', 'activation': 'relu', 'residual': 'residual'}`
- `{'norm': 'rmsnorm', 'activation': 'gelu', 'residual': 'residual'}`
- `{'norm': 'layernorm', 'activation': 'relu', 'residual': 'residual'}`

Cost: 8/8 declared synthetic run units.
Expected interaction variance: 1.000000 -> 0.013110.
After generating only the selected *synthetic* observations and ingesting them:
posterior mean -0.789023, SD 0.114497, variance 0.013110.

The second plan uses fresh seed IDs. Shared-baseline covariance is retained.
All uncertainty is conditional on the declared Gaussian model/prior/noise.
No real datasets, checkpoints, completed studies, training processes or external services were used.

sweep.json demonstrates the mtsweep interface. Its corpus path is deliberately a placeholder;
a real prospective protocol and data/tokenizer configuration are required before execution.
