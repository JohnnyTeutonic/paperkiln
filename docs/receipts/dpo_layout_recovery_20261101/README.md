# DPO bare-weight layout recovery

The first DPO campaign passed source/input preflight but failed before training: its bare SFT reference weights had no run record and the generated spec omitted the explicit RoPE-head layout. This setup failure remains negative; no optimizer, cadence or preference-quality result was produced. The owned pod and volume were cleaned up.

The canonical original SFT export explicitly records `rope_heads=all`; its model width, depth, head count and head dimension match the original training spec. Root derives an immutable base spec by adding only `arch.rope_heads=all`, with the original spec, IR and weight hashes retained. This clarifies the existing model's semantics; it does not select a layout from evaluation outcomes or alter the numerical training recipe. Missing/invalid layout is now rejected during recipe construction before GPU admission.

An actual CPU LoadedLM initialization regression uses the original97M SFT weights and vocabulary with an empty scoring input. The historical missing-layout control fails with the exact error; the explicit-layout case loads successfully, without training or changing weights. The same LoadedLM constructor initializes the DPO reference. This is not a CUDA-training or quality result.

Ninety-seven materializer/packet/checkpoint/transport/manager/boot tests pass. The reviewed S-full diagnostic changes are also ported to DPO's separate checkpoint core: no per-event journal fsync, ordinary diagnostic failures best-effort, retained integrity failures distinct/fatal. Full checkpoint validation, recipe and180second guard remain. Held-out source/policy and dataset remain unchanged.

A fresh source/input archive and serial paid admission are still required. No previous result or source packet is overwritten. All five original acceptance gates and final-checkpoint selection remain required.
