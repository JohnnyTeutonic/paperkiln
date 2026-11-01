# DPO V3 first-checkpoint gap: preserved negative

The exact-source upload and on-VM preflight pass. CUDA build completes and actual L4 initialization accepts the original SFT weights with rope_heads=all. The unchanged committed-full-checkpoint guard then halts before a completed-step event or first checkpoint is recorded. This is an operational negative; no DPO quality or acceptance-gate verdict exists. Downstream parity/evaluation/sampling jobs did not run.

All thirteen diagnostic members are home, regular and readable, with independently verified archive/inventory hashes. Original halt evidence and preclosure state are retained locally. The watchdog entry is disabled; only the owned pod and its 30GB volume were deleted. Protected original S-full volumes were untouched. Operational home status explicitly means negative evidence home, not FINISHED training; closure.json records scientific_completion=false.

Source inspection finds a substantial uninstrumented region: loading the frozen reference, then serially computing margins for all1445pairs before the first training step. Captured logs end at initialization, so this is a concrete hypothesis, not exact live-phase proof. No OMP oversubscription was established (four threads configured). A separately bounded opt-in phase diagnostic is being prepared; no guard/cadence/cap relaxation or blind retry has been adopted.
