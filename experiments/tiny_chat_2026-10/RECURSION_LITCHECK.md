# Recursion arm for E1: literature check (23 Oct 2026)

Question: does a shared block applied several times (recursion) carry facts across turns better than plain depth or width at ~20M non-embedding parameters, scored by Carry-5's Y? Every claim below was read from the arXiv abstract or HTML (or the paper PDF) on 23 Oct 2026. mtstudio already implements fixed recursion (`include/microtorch/llama.hpp`, `arch.custom.recursion`: prelude + block x repeats + coda, a KV cache per application, full backprop through every application, no input injection, schedule `fixed` only).

## 1. The sources

| Work | What is recursed | Matching | Stability and training tricks | Small-scale finding; recall vs knowledge |
|---|---|---|---|---|
| Universal Transformers (1807.03819) | One encoder/decoder layer, weights tied across steps; per-position ACT halting | Not parameter-matched | A sinusoidal timestep embedding added at every step | bAbI (10k, single): 0.23 average error vs Transformer 15.2; the plain Transformer "always overfits" there. bAbI is story fact-retrieval, the nearest old analogue of carry-over |
| ALBERT (1909.11942) | All layers share one set (encoder, MLM) | Same layers, fewer params | None needed | All-shared costs ~2.5 points at E=768, ~1.5 at E=128; FFN sharing causes most of the loss, attention sharing little |
| MobileLLM-LS (2402.14905) | Each layer applied twice (immediate block-wise) | Iso-param (2x compute) | None | 125M: +0.4 (immediate), +0.6 (repeat-all-over), +0.2 (reverse) zero-shot average; final +0.7 at 125M, +0.8 at 350M |
| Relaxed Recursive (2410.20672) | One block looped, from pretrained weights; depth-wise LoRA relaxes the tie | Iso-param vs smaller vanilla | Initialisation from the pretrained stack | 1B-2B only |
| MoEUT (2405.16039) | Groups of G=2 MoE layers recurred | Iso-param | Peri-layernorm (no norm on the main residual path) | 44M: C4 perplexity 18.30 vs 18.97 baseline; BLiMP 78.2 vs 73.5; CBT 81.1 vs 81.3. Dense single shared layer reported weak |
| Saunshi et al. (2502.17416) | k-layer block looped L times; also "middle looping" | Both iso-param (k x 1) and iso-FLOP (24 x 1) | Vanilla looping, no injection; a cosine regulariser tying layer blocks | 1B on Pile: looped worse perplexity than iso-FLOP; closed-book QA tracks perplexity; open-book QA (includes QuAC, CoQA, DROP, SQuADv2), math and reasoning primitives gain. Open-book avg: base 4x1 13.8, loop 4x6 26.2, 24x1 33.9. At equal perplexity, looped models score higher on context tasks |
| Huginn (2502.05171) | Prelude 2 + core 4 recurred + coda 2 | Compute vs a 50B-equivalent | Input injection (concat + adapter; addition as good at small size), random initial state, random recurrence count, truncated backprop to last 8, sandwich norm, low lr | At 3.5B, bad runs collapsed (token-correlation 1.0) or ignored the state; "at small scale most choices worked". Reasoning keeps improving with recurrence, easy tasks saturate ~8 |
| HRM (2506.21734) | Two modules (slow H, fast L) recurring | 27M, puzzles, ~1,000 examples | Input added each step, one-step gradient, deep supervision, ACT via Q-learning | No language modelling |
| TRM (2510.04871) | One 2-layer net; latent z updated n=6 times, answer y refined, T=3 | 5-7M | Full backprop through the last recursion (1-step gradient: 56.5% vs 87.4%), EMA 0.999 (without: 79.9%), deep supervision up to 16 steps | 2 layers beat 4 at matched effective depth (87.4% vs 61.6% at k=3), attributed to overfitting with ~1k examples. No language modelling |
| Mixture-of-Recursions (2507.10524) | Middle-Cycle (unique first/last layer, shared middle) with per-token routers | Iso-FLOP and equal-data, 135M-1.7B | Expert-choice routing needs an auxiliary loss (non-causal otherwise); token-choice struggles to balance load | Middle-Cycle had the lowest loss at 135M and 360M. **At 135M MoR trails vanilla** (avg 38.11 vs 42.15 at 16.5e18 FLOPs), attributed to a recursive capacity bottleneck; N_r=4 worse than N_r=2 at 360M. A per-recursion learned scale changed little. No recall evaluation |
| Ouro (2510.25741) | Whole 24-layer stack x 4 | Iso-param vs larger | 8 steps gave loss spikes, cut to 4; lower lr than param-matched transformers | Synthetic 1M-40M models: looping does **not** raise knowledge capacity (~2 bits/param with or without loops); gains come from knowledge manipulation |
| Looped GPT-BERT (2609.09691) | 4 layers x {1,3,6,12}, injection z_{t+1}=F(x+z_t) | Iso-param among loops (12.18M); the 12-layer baseline has 29.9M | 4x1 unstable early | BabyLM Strict-small, single seed: BLiMP gains from 4x1 to 4x3, then diminishing; **entity tracking does not improve with loops** (1:1 mix: 4x1 17.5, 4x3 16.6, 4x6 13.1, 4x12 17.2; non-loop 38.7) |
| MixerLoop (2608.18230) | Repeat only the sequence mixer, FFN once | 15M and 110M | n/a | At 15M beats full-block looping on CORE |
| Looped Models Done Right (2610.23036) | Ouro-style vs prelude-core-coda | Matched params, logical depth, tokens; 730M only, single seed | Injection as a diagonal write before each pass | Sandwich topology helps context extraction and reasoning (DROP +2.6, GSM8K +7.1); injection helps prompt-dependent tasks (DROP, BBH, code) and hurts math |

## 2. What the literature predicts for E1

- **Iso-param, more passes:** looping a block beats the same block applied once (Saunshi, MobileLLM-LS, GPT-BERT 4x1 to 4x3), with diminishing returns after 3-4 passes (GPT-BERT, MoR N_r=4, Ouro's spikes at 8).
- **Recursion vs unique layers at equal parameters and equal effective depth** (the contrast E1 can run for free): small-scale evidence leans against recursion (MoR's 135M bottleneck; GPT-BERT entity tracking; ALBERT's FFN-sharing cost). MoEUT's 44M win needed MoE layers.
- **Recall vs knowledge:** recursion does not add stored knowledge (Ouro); its inductive bias is towards using the context (Saunshi's open-book and isoplot result). Carry-5's Y is a context-use quantity, so this is the direction most worth registering.
- **Puzzle-regime results (TRM, HRM) do not transfer directly:** their "2 layers beat 4" rests on overfitting with ~1,000 examples; E1 trains 8 tokens per parameter.

## 3. Step 1 design recommendation

**Block and counts.** Recurse the whole 20m_L4 stack (d 640, 4 layers, 20.08M non-embedding), r = 2, 4, 8: effective depths 8, 16, 32. Each cell then has an exact E1 twin at equal non-embedding parameters and equal effective depth (20m_L8, L16, L32), and r = 1 is E1's own L4 cell. No new control runs. Secondary, one cell: Middle-Cycle 1 + [2 shared] x 7 + 1 at the same d 640 (effective depth 16, the same parameters and FLOPs as [4] x 4). It tests topology and is step 2's fixed-depth baseline, since MoR's routers run on Middle-Cycle. mtstudio's prelude/block/coda already expresses both.

**Matching.** Non-embedding parameters, as E1. Recursion multiplies compute by r, so register no iso-FLOP verdict. Report FLOPs per token, Y per L4-hour (S10), and, descriptively, the 60M cells nearest in FLOPs. The embedding confound reverses here: the recursion cells carry 10.49M embedding against 7.34/5.24/3.15M for the twins. So a recursion win over the twins is ambiguous in the same way as a width win.

**Training.**
- No input injection in the primary cells. Saunshi, MoR and MobileLLM work without it at up to 8 passes, and mtstudio has none. If anything, register one injection ablation at r = 4 (additive re-injection of the token embedding before each pass, zero parameters, as GPT-BERT and HRM). Done Right is the reason: injection helped prompt-dependent tasks.
- No per-step embeddings or LoRA relaxation (both break exact matching; MoR's per-recursion scale did little).
- Full backprop through every application (mtstudio's default; TRM's 1-step gradient collapsed). No deep supervision.
- Own Stage 0 lr screens per recursion cell, with the grid extended downwards to 2.5e-4. Shared weights receive gradients summed over r applications, and Ouro and Huginn needed lower lr.
- Batch 16 x 2 accumulation at effective depth >= 16, as section 1.

**Predictions to register** (SESOI as E1):
- **HR1, iso-param recursion gain:** the slope of Y on log2(r) over r = 1, 2, 4, 8 at d 640 is > 0.
- **HR2, recursion vs depth** (pooled paired contrast Y(L4 x r) - Y(L_4r)): the literature-led prediction is "not above SESOI" (EQUIVALENT or negative). If Jonathan holds that recursion wins, register that as the stated alternative.
- **HR3, inductive bias** (Saunshi's isoplot): at matched FineWeb panel loss, recursion cells have higher Y than the line fitted through E1's plain cells.
- **HR4, diminishing passes:** the quadratic contrast over log2(r) is negative (descriptive).

**Cost** (rough, extrapolated from A2.1's probes; probes replace it). About 2.5, 4 and 7 h of pretraining per run at r = 2, 4 and 8. That is about 50 L4 VM-hours (~240 units) for 9 runs, plus ~15 for the Middle-Cycle cell. r = 8 is nearly half of it. A pre-registered conditional (run r = 8 only if r = 4 beats r = 2) would halve the risk.

**Step 2 (MoR), conditional on HR1.** Use Middle-Cycle with expert-choice routing and the auxiliary loss, MoR's best. It is a new `schedule` in mtstudio. Expect it to trail at 20M: MoR itself lost to vanilla at 135M.

## 4. Failure modes to watch

- Capacity bottleneck: shared FFNs cost the most (ALBERT, MoR at 135M).
- The model ignores the recurrence, or hidden states collapse (Huginn). Check that the panel loss differs between r = 2 and r = 8.
- Loss spikes at 8 passes (Ouro), handled by the per-cell lr screen.
- Entity tracking not improving with loops (GPT-BERT), which is the nearest published sign for Carry-5.
- Compute and embedding confounds (section 3).
- Inference: each application has its own KV cache, so ember's cache grows with effective depth as for a plain model of that depth. Export to llama.cpp would need unrolled duplicate tensors or a layer map (an inference from the architecture, not tested).

## 5. Open vs taken (conservative)

**Taken.**
- Looping or recursion at iso-param and iso-FLOP at 100M and above (Saunshi, MoR, MobileLLM-LS, Relaxed, Ouro, Huginn, Done Right).
- Recursion at 60M parameters or fewer, on puzzles (TRM, HRM), on perplexity and zero-shot tasks (MoEUT 44M, MixerLoop 15M), on BabyLM with entity tracking (GPT-BERT 12M), and on synthetic knowledge capacity (Ouro 1-40M).
- The findings that recursion does not add knowledge capacity and that it favours context use over memorisation.
- Per-token adaptive recursion (UT's ACT, MoR).

**Open (none found in targeted searches).** Recursion vs plain depth vs width at matched parameters, at 60M or fewer, measured on multi-turn carry-over after SFT, multi-seed, with a single-turn twin. The closest sources are GPT-BERT's entity tracking (single seed, not dialogue, no deep-thin control at matched parameters) and Saunshi's open-book QA including QuAC and CoQA (1B, one scale). The contribution would be the measurement and the scale, not the mechanism. The search covered arXiv and web search on these terms and could miss a very recent paper.
