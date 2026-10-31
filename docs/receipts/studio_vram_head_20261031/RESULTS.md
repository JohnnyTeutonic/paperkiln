# CLI GPT2 parameter inventory correction

The bounded Studio estimator previously counted a tied token/head table, and its C++ reference instantiated `nn::GPT2`. The actual CLI plain GPT2 path uses `parity::ParityLM` at depth two and `parity::FlexLM` at other depths; both have an independent head. The estimator now includes its `vocab * d` parameters. The memory formula otherwise remains unchanged and remains a conservative estimate, not a measured peak-memory guarantee.

An independent constructor probe enumerates actual storage for six GPT2 geometries spanning depths one through five plus two tied-head Llama configurations. All eight JavaScript tests pass with the probe enabled. Four actual one-step CLI runs independently confirm the ParityLM/FlexLM dispatch and inventories. The smallest two-layer case has 1,904 parameters, previously estimated as 1,848. The existing four-layer reference had 15,376 actual parameters. The tied Llama counts stay unchanged.

Reproduce by compiling tests/studio_vram_parameter_probe.cpp against the library and tool/header dependencies, then set STUDIO_VRAM_PARAMETER_PROBE to its executable when running node --test tests/test_studio_vram.js. Raw constructor inventories and CLI model events are retained here. This does not calibrate BF16 attention or other unsupported planner paths.
