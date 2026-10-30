# BF16/flash per-shape profile: complete

The bounded descriptive profile completed all 40 cells on one NVIDIA L4 at exact source pin `346354e8d34b1b9beac536f72ca1595cbf60edc9`: ten shapes × four configurations, each with a clean pass and an instrumented pass,30 steps each, warmup 8. This closes the previously unfinished per-shape profile. It does not rerun or replace the separately completed noise-floor/loss gate.

|Shape|FP32 tok/s|BF16 tok/s|Flash tok/s|Combined tok/s|BF16/FP32|Flash/FP32|Combined/FP32|FP32 attention/GPU spans|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|20m_L2|36307|68038|36650|69973|1.87×|1.01×|1.93×|16.0%|
|20m_L4|34154|54508|35551|57922|1.60×|1.04×|1.70×|20.7%|
|20m_L8|29753|44993|31187|47372|1.51×|1.05×|1.59×|26.2%|
|20m_L16|26628|35978|28534|38933|1.35×|1.07×|1.46×|33.9%|
|20m_L32|22650|27867|24197|30270|1.23×|1.07×|1.34×|33.8%|
|60m_L4|15299|30759|15733|32026|2.01×|1.03×|2.09×|16.2%|
|60m_L8|15151|24995|15655|26207|1.65×|1.03×|1.73×|22.2%|
|60m_L16|13720|20722|14508|22099|1.51×|1.06×|1.61×|28.0%|
|60m_L32|11080|15093|11733|16444|1.36×|1.06×|1.48×|33.7%|
|pk_base_v1|10575|17979|11148|19047|1.70×|1.05×|1.80×|24.7%|

These are single descriptive measurements per cell, not replicated estimates, significance tests or universal speedup guarantees. Clean throughput is measured using subprocess step-line arrival intervals; those arrival timestamps are not retained in raw event JSONL. Speedup ratios above are recomputed from the recorded clean throughput. The instrumented phase, op, wall, GPU and idle arithmetic was independently reconstructed from raw events for all 40 cells. Instrumented overhead and timed-span sums should not be interpreted as physical GPU utilization.

Across this matrix BF16 gives 1.23–2.01×, flash 1.01–1.07×, and combined 1.34–2.09× the recorded FP32 throughput. Attention forward/backward spans account for 16.0–33.9% of baseline instrumented GPU-span time. Thus the roadmap's condition for investigating BF16 inside attention products is met: attention is a material measured component. This is evidence to motivate a separate bounded implementation/parity investigation, not proof that this additional change will improve end-to-end throughput. No such implementation or extra paid run is part of this profile.

Independent verification checked all 4,718 recorded source-file hashes against immutable Git objects at the pin, all 40 completed-cell hashes, all 80 raw-event/spec pairs, exact requested precision/kernel model events, 30 finite-loss steps per pass (2,400 total), 22 timed steps per instrumented pass, and no negative idle cells. The synthetic 8M-ID corpus hash was independently reconstructed from its deterministic seed/header without retaining another corpus copy. All recorded source/binary/vocabulary/corpus/GPU metadata are in the raw archive's `source_manifest.json`; the archived binary hash is a recorded run-time receipt, not a local rehash of a retained binary.

Records: [profile.json](profile.json), [completion.json](completion.json), [independent audit and summary](independent_audit.json), and [raw events/specs/manifests](raw_events.tar.gz). Original artifacts and CUDA preflight logs remain at `C:/ml_artifacts/bf16_profile_20261030/step_profile_bounded/`. The raw archive is 710KB and excludes models/corpus; no legacy campaign result was changed.
