# Retained HF raw-row verification

The existing clean-source tiny untied Llama-flavour flex fixture was verified again without training or changing its weights. All three prompts, all24 generated IDs and each maximum logit difference exactly reproduce the historical records. The new receipts retain float32 HF/native rows plus banned-token IDs; each can be independently replayed with `python tools/hf_export_verify.py --replay <receipt>` using NumPy alone. The original row files were not retained and are still unavailable; this follow-up does not rewrite that historical limitation.

Receipt/source hashes were captured before execution and checked after it. Source helper snapshots are exact bytes used by these new runs; the native clean binary is unchanged. The JSON evidence publication is exclusive and atomic, including an interrupt immediately after link publication; old receipts are never overwritten. Offline replay verifies saved rows/IDs, numerical bounds and recorded maxima. It does not run the models again or prove chat-template rendering or downstream quality.

`summary.json` binds all local evidence/source snapshots. The original forward outputs are in `C:/ml_artifacts/hf_flex_llama_roundtrip_20261030/evidence_replay_final/`. Historical model/tensor/source receipts remain in `experiments/chat_general/receipts/hf_flex_llama_roundtrip_2026-10-30/`.
