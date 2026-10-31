# Mamba1 actual CLI lifecycle CUDA gate

The frozen L4 gate passed both depth-one and depth-two tiny Mamba1 LM cases. Each ran three real CLI training steps, saved model and optimizer checkpoints, and reproduced the F32 GGUF bytes after loading the saved checkpoint. Every step recorded positive causal-convolution and selective-scan forward/backward kernel counters.

The required host replay used the pinned NumPy 1.26.4 equations and CPU Ember binary. All eight-by-seven score rows per case passed the prospectively fixed normalized error bound of 1e-5 and every full-row argmax matched. The largest CLI-versus-NumPy error was 2.2351741790771484e-8; CPU Ember prefill/decode maxima were 1.4901161193847656e-8. Source, helper and Ember pins passed before and after the exact reviewed replay command.

The raw ZIP, all nested hashes, logs, full-row outputs, host replay receipt and resource-cleanup proof are retained here. The owned L4 pod and volume were deleted and the watchdog entry disabled. The scientific source and thresholds were frozen before device outcomes. This proves only the registered tiny LM lifecycle; mixed host/device boundaries remain. It does not establish SFT support, model quality, complete device residency, or speed.
