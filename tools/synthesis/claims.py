#!/usr/bin/env python3
"""The re-aim (17 Sep 2026): scope CLAIMS, not mechanisms.

For a two-person team with a from-scratch trainer and a seed-honest,
matched-budget, pre-registered protocol, the splash shape is not a new
mechanism (that race goes to whoever has the cluster) but a phenomenon
found by looking carefully at small scale, or the falsification of a
claim the field repeats without a controlled test (grokking, double
descent, attention sinks, "emergent abilities are a mirage").

This module enumerates the folk claims of transformer practice, each
with where the belief came from and the controlled test we would run,
then asks the scoper a different question: not "is this idea taken" but
"has this claim been tested with a controlled, matched-budget,
multi-seed design, at what scale, and what does the evidence actually
license". The ranking is belief strength x weakness of evidence x
testability on paperkiln today.

    python tools/synthesis/claims.py list
    python tools/synthesis/claims.py scope [--limit N]   # -> results/claims_scoped.jsonl
    python tools/synthesis/claims.py report              # -> results/claims_ranked.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

OUT = P.OUT

# paperkiln slots switchable today (tools/atlas_taxonomy.py, 17 Sep 2026):
# family gpt2/llama/flex; attention exact/kimi/srd/attnres/swa (+sinks,
# window); optimizer adamw/muon; d, layers, heads, T, lr, batch; norm
# layernorm/rmsnorm; residual residual/highway/plain; position
# learned/sinusoidal/rope; activation gelu/relu/swiglu. Planned: stream
# pre-norm/post-norm/attnres.

# Each claim: statement as believed; component; origin (what installed the
# belief); evidence_scale (what the origin actually showed); test (our
# controlled design in one sentence); knobs (paperkiln slots needed);
# testable_now; belief (1-5, how widely repeated); stakes (what changes if
# it falls); queries (3-4 fielded arXiv queries in the field's words).
CLAIMS = [
    # ---------------------------------------------------------- normalisation
    dict(id="rmsnorm-better", component="normalisation",
         claim="RMSNorm matches or beats LayerNorm in quality and is faster, so LayerNorm is obsolete for LLMs",
         origin="Zhang & Sennrich 2019 (RMSNorm); adopted by T5, LLaMA; repeated as settled",
         evidence_scale="RMSNorm paper: MT and RC tasks, few seeds; LLM adoption by choice, no controlled quality comparison published at scale",
         test="matched-FLOPs nano panel, layernorm vs rmsnorm, 12 seeds, gpt2 and llama families, report the sign per seed and the loss delta with bootstrap band",
         knobs=["norm"], testable_now=True, belief=5,
         stakes="a null or a LayerNorm win at small scale contradicts a default every modern model inherits",
         queries=['all:RMSNorm AND all:LayerNorm AND all:comparison', 'abs:"root mean square layer normalization" AND abs:transformer', 'all:normalization AND all:"language model" AND all:ablation AND all:seeds']),
    dict(id="prenorm-better", component="normalisation",
         claim="Pre-norm trains more stably and reaches equal or better loss than post-norm, so post-norm is dead",
         origin="Xiong et al. 2020 (On Layer Normalization in the Transformer Architecture); GPT-2 practice",
         evidence_scale="theory on gradient scale at init plus BERT/MT experiments; post-norm with warmup is often better at convergence (Liu et al. 2020 Admin), a fact rarely repeated",
         test="pre vs post placement at matched budget with and without warmup, 12 seeds, loss and gradient-norm receipts",
         knobs=["stream"], testable_now=False, belief=5,
         stakes="post-norm winning at convergence would reverse a universal default",
         queries=['all:"pre-norm" AND all:"post-norm" AND all:transformer', 'abs:"layer normalization" AND abs:"warm-up" AND abs:transformer', 'all:"post-LN" AND all:"pre-LN" AND all:convergence']),
    # ------------------------------------------------------------- activation
    dict(id="swiglu-better", component="activation",
         claim="SwiGLU beats GELU and ReLU MLPs at matched parameters",
         origin="Shazeer 2020 (GLU Variants Improve Transformer); PaLM, LLaMA adoption",
         evidence_scale="Shazeer: T5-base scale, one seed per variant, small deltas; the note admits no explanation",
         test="gelu vs relu vs swiglu at matched params (d_ff adjusted), 12 seeds, both families",
         knobs=["activation"], testable_now=True, belief=5,
         stakes="a single-seed 2020 result installed a default; a seed-honest null would be news",
         queries=['all:SwiGLU AND all:GELU AND all:comparison', 'abs:"GLU variants" AND abs:transformer', 'all:activation AND all:"language model" AND all:ablation AND all:"matched parameters"']),
    dict(id="relu-dead-llm", component="activation",
         claim="ReLU MLPs are strictly worse than smooth activations for language models",
         origin="folklore since GPT-2 (GELU) and BERT",
         evidence_scale="no controlled multi-seed comparison at matched budget in wide circulation; ReLU revival papers (sparsity) report parity",
         test="relu vs gelu at matched params, 12 seeds, plus activation sparsity as a free measurement",
         knobs=["activation"], testable_now=True, belief=4,
         stakes="parity would reopen ReLU sparsity for inference at no quality cost",
         queries=['all:ReLU AND all:GELU AND all:"language model" AND all:comparison', 'abs:"ReLU strikes back"', 'all:activation AND all:sparsity AND all:"large language model" AND all:quality']),
    # ---------------------------------------------------------------- position
    dict(id="rope-better-learned", component="positional-encoding",
         claim="RoPE beats learned and sinusoidal absolute positions in-distribution, not just for extrapolation",
         origin="Su et al. 2021 (RoFormer); LLaMA adoption",
         evidence_scale="RoFormer: MT and GLUE with few seeds; most later evidence is about extrapolation",
         test="learned vs sinusoidal vs rope at fixed T, matched params, 12 seeds, in-distribution loss only",
         knobs=["position"], testable_now=True, belief=5,
         stakes="in-distribution parity would mean RoPE's case is extrapolation alone",
         queries=['all:RoPE AND all:"learned positional" AND all:comparison', 'abs:"rotary position embedding" AND abs:ablation', 'all:"positional encoding" AND all:"language model" AND all:comparison AND all:seeds']),
    dict(id="nope-causal", component="positional-encoding",
         claim="Causal decoders need no positional encoding: NoPE matches RoPE",
         origin="Haviv et al. 2022; Kazemnejad et al. 2023 (The Impact of Positional Encoding on Length Generalization)",
         evidence_scale="small models, specific tasks; contested at scale (NoPE degrades on long context in later reports)",
         test="none vs rope at matched budget, 12 seeds, in-distribution loss and a length-generalisation probe",
         knobs=["position"], testable_now=False, belief=3,
         stakes="a clean small-scale answer either way settles a live dispute",
         queries=['all:NoPE AND all:"positional encoding" AND all:causal', 'abs:"no positional encoding" AND abs:transformer', 'all:"length generalization" AND all:"positional encoding" AND all:decoder']),
    # --------------------------------------------------------------- attention
    dict(id="sinks-necessary", component="attention-scores",
         claim="Attention sinks emerge in every trained decoder and are necessary for stable long-context attention",
         origin="Xiao et al. 2023 (StreamingLLM); Gu et al. 2025 (When Attention Sink Emerges)",
         evidence_scale="observations on pretrained LLMs; Gu et al. train small models and report window-size dependence",
         test="sink emergence across norm/activation/position choices at nano scale, 12 seeds; does any configuration train without a sink, and does its window-attention lane then fail",
         knobs=["attention", "norm", "position"], testable_now=True, belief=4,
         stakes="a sink-free configuration with intact sliding-window quality would reframe sinks as an artefact of defaults",
         queries=['all:"attention sink" AND all:emergence', 'abs:"attention sinks" AND abs:"language model" AND abs:training', 'all:"massive activations" AND all:"attention sink"']),
    dict(id="window-loses-longrange", component="attention-topology",
         claim="Sliding-window attention loses long-range dependencies that full attention captures, at every scale",
         origin="Longformer, Mistral practice; assumed in every sparse-attention paper",
         evidence_scale="measured on long-context benchmarks at scale; at nano scale the transfer study found the window lane LEADS early and trails late, with the crossing moving with width",
         test="window vs exact at matched budget across T and width, milestone-matched, 12 seeds; where is the crossing and does it exist at T=256",
         knobs=["attention"], testable_now=True, belief=5,
         stakes="already partly measured (transfer_s2); the position-dependence is the phenomenon",
         queries=['all:"sliding window attention" AND all:"long-range" AND all:comparison', 'abs:"local attention" AND abs:"full attention" AND abs:"language model"', 'all:"sparse attention" AND all:quality AND all:"matched compute"']),
    dict(id="gqa-free", component="multi-head",
         claim="Grouped-query attention loses nothing measurable versus full multi-head attention",
         origin="Ainslie et al. 2023 (GQA); adoption by LLaMA 2 and after",
         evidence_scale="uptrained from MHA checkpoints at scale, few seeds; from-scratch small-scale comparison rarely reported",
         test="MHA vs GQA vs MQA from scratch at matched params (heads/d adjusted), 12 seeds",
         knobs=["heads"], testable_now=False, belief=4,
         stakes="a from-scratch cost at small scale would mark GQA as an inference trade, not free",
         queries=['all:"grouped-query attention" AND all:"multi-head" AND all:comparison', 'abs:"multi-query attention" AND abs:quality AND abs:"from scratch"', 'all:GQA AND all:MQA AND all:ablation']),
    dict(id="qknorm-stabilises", component="attention-scores",
         claim="QK-normalisation removes attention-logit growth and is free in quality",
         origin="Dehghani et al. 2023 (ViT-22B); Wortsman et al. 2023 (Small-scale proxies for large-scale instabilities)",
         evidence_scale="Wortsman et al. is the rare small-scale-proxy study; quality cost at small scale not the focus",
         test="with/without QK-norm at matched budget, 12 seeds, loss and logit-growth receipts across lr",
         knobs=["attention"], testable_now=False, belief=3,
         stakes="a quality cost at small scale would qualify a default",
         queries=['all:"QK normalization" OR all:"QK-norm"', 'abs:"attention logit growth" AND abs:instability', 'abs:"small-scale proxies" AND abs:instabilities']),
    dict(id="softmax-temperature-scale", component="attention-scores",
         claim="The 1/sqrt(d_k) attention scaling is the right temperature at every width",
         origin="Vaswani et al. 2017; never revisited in practice (muP uses 1/d)",
         evidence_scale="a variance argument at init; muP (Yang et al.) argues 1/d for width transfer",
         test="1/sqrt(d) vs 1/d vs learned temperature across widths at matched budget, 12 seeds",
         knobs=["attention", "d"], testable_now=False, belief=4,
         stakes="a width-dependent optimum would connect to the transfer study's lr-per-width finding",
         queries=['all:"attention temperature" AND all:scaling AND all:width', 'abs:"1/d" AND abs:attention AND abs:"maximal update"', 'all:"attention scaling" AND all:"query key" AND all:temperature']),
    # --------------------------------------------------------------- optimiser
    dict(id="adamw-default", component="optimiser-state",
         claim="AdamW is the best general optimiser for transformer pretraining; alternatives win only on tuned benchmarks",
         origin="practice since 2018; Kaddour et al. 2023 (No Train No Gain) for the sceptical version",
         evidence_scale="mostly large-scale reports; Muon (Jordan 2024; Liu et al. 2025 Moonshot) claims ~2x token efficiency",
         test="adamw vs muon at matched wall-clock and matched tokens, 12 seeds, both families, lr swept per optimiser",
         knobs=["optimizer"], testable_now=True, belief=4,
         stakes="already switchable in paperkiln; a seed-honest Muon gain or null at nano scale is a clean data point either way",
         queries=['all:Muon AND all:optimizer AND all:AdamW AND all:comparison', 'abs:"orthogonalized" AND abs:momentum AND abs:"language model"', 'all:optimizer AND all:"language model pretraining" AND all:benchmark AND all:seeds']),
    dict(id="warmup-necessary", component="lr-schedule",
         claim="Learning-rate warmup is necessary for transformer training stability",
         origin="Vaswani et al. 2017 schedule; Xiong et al. 2020 attribute it to post-norm gradients",
         evidence_scale="pre-norm models train without warmup in Xiong et al.; practice keeps warmup anyway",
         test="warmup 0 vs 2 percent vs 10 percent at matched tokens, pre-norm, 12 seeds, divergence counts and final loss",
         knobs=["lr"], testable_now=True, belief=4,
         stakes="a null under pre-norm would remove a universal knob",
         queries=['all:"learning rate warmup" AND all:transformer AND all:necessary', 'abs:warmup AND abs:"pre-norm" AND abs:stability', 'all:"warm-up" AND all:Adam AND all:variance']),
    dict(id="cosine-best", component="lr-schedule",
         claim="Cosine decay to ~10 percent is the best schedule; alternatives are within noise",
         origin="practice since GPT-3; Hägele et al. 2024 (Scaling laws and compute-optimal training beyond fixed durations) for WSD",
         evidence_scale="WSD matches cosine at scale; at nano scale rarely compared with seeds",
         test="cosine vs constant vs WSD at matched tokens, 12 seeds",
         knobs=["lr"], testable_now=True, belief=3,
         stakes="schedule-invariance at small scale would say the schedule is a large-scale phenomenon",
         queries=['all:"cosine schedule" AND all:"warmup-stable-decay"', 'abs:"learning rate schedule" AND abs:"language model" AND abs:comparison', 'all:"constant learning rate" AND all:cooldown AND all:pretraining']),
    dict(id="weight-decay-needed", component="training-objective",
         claim="Weight decay (0.1) improves language-model pretraining loss, not just generalisation",
         origin="GPT-3 and LLaMA recipes; Andriushchenko et al. 2023 (Why do we need weight decay in modern deep learning)",
         evidence_scale="Andriushchenko: LLM weight decay lowers TRAINING loss via lr-schedule interaction; small-scale seeds present",
         test="wd 0 vs 0.1 at matched tokens, 12 seeds, training and validation loss separately",
         knobs=["lr"], testable_now=False, belief=4,
         stakes="mostly settled by Andriushchenko; a nano-scale replication is cheap and would bound the effect",
         queries=['abs:"weight decay" AND abs:"language model" AND abs:"training loss"', 'all:"weight decay" AND all:AdamW AND all:mechanism', 'all:"why do we need weight decay"']),
    dict(id="lr-scales-mup", component="lr-schedule",
         claim="The optimal learning rate falls with width as 1/width under standard parameterisation, and muP makes it constant",
         origin="Yang et al. 2022 (Tensor Programs V, muTransfer)",
         evidence_scale="muP paper: shown on transformers to GPT-3 scale; at nano scale the transfer study found lr* 2.5e-4, 5e-4, 1.25e-4 for widths 256, 512, 1024 under SP, which is NOT monotone",
         test="lr sweep per width under SP, 12 seeds, matched tokens; is lr*(width) monotone at nano scale",
         knobs=["d", "lr"], testable_now=True, belief=5,
         stakes="already partly observed (transfer_s2 stage 1): a non-monotone lr*(width) at small scale contradicts the folk reading of muP",
         queries=['all:"maximal update parametrization" AND all:"learning rate" AND all:width', 'abs:muTransfer AND abs:"learning rate"', 'all:"optimal learning rate" AND all:width AND all:"standard parameterization"']),
    dict(id="grad-clip-needed", component="gradient",
         claim="Gradient clipping at 1.0 is necessary to avoid loss spikes",
         origin="GPT-3 recipe; universal default",
         evidence_scale="no controlled small-scale study in wide circulation",
         test="clip 1.0 vs none vs 5.0 at matched tokens, 12 seeds, spike counts and final loss",
         knobs=["lr"], testable_now=False, belief=4,
         stakes="a null would remove a default; a difference would locate where spikes come from at nano scale",
         queries=['all:"gradient clipping" AND all:transformer AND all:"loss spikes"', 'abs:"gradient clipping" AND abs:"language model" AND abs:ablation', 'all:"loss spike" AND all:pretraining AND all:cause']),
    dict(id="batch-critical", component="data-order",
         claim="Larger batches are free up to a critical batch size that grows with loss, so small-batch training is wasteful",
         origin="McCandlish et al. 2018 (An Empirical Model of Large-Batch Training)",
         evidence_scale="gradient-noise-scale theory with experiments across domains; nano-scale LM checks rare",
         test="batch 4 to 256 at matched tokens with lr scaled, 12 seeds; where is the critical batch at 10M params",
         knobs=["batch", "lr"], testable_now=True, belief=4,
         stakes="a critical batch far below practice at nano scale would say small-model sweeps are mis-batched",
         queries=['all:"critical batch size" AND all:"language model"', 'abs:"gradient noise scale" AND abs:"batch size"', 'all:"batch size" AND all:"learning rate" AND all:scaling AND all:transformer']),
    # ---------------------------------------------------------- depth and width
    dict(id="deeper-beats-wider", component="layer-schedule",
         claim="At fixed parameters, deeper models beat wider ones for language modelling",
         origin="Kaplan et al. 2020 (shape matters little); Levine et al. 2020 (depth-width optimum); practice favours depth",
         evidence_scale="Kaplan: shape is a weak effect across a range; Levine: an optimal depth per width",
         test="fixed params, layers x d grid, matched tokens, 12 seeds; is there an optimum at 10M to 60M",
         knobs=["layers", "d"], testable_now=True, belief=4,
         stakes="a sharp optimum or its absence at nano scale is a phenomenon either way",
         queries=['all:depth AND all:width AND all:transformer AND all:"fixed parameters"', 'abs:"depth-width" AND abs:"language model"', 'all:"scaling laws" AND all:shape AND all:transformer']),
    dict(id="parallel-blocks-free", component="residual-stream",
         claim="Parallel attention and MLP blocks (GPT-J, PaLM) cost nothing in quality",
         origin="GPT-J (Wang & Komatsuzaki 2021); PaLM reports a small loss at 8B, none at 62B",
         evidence_scale="PaLM: degradation at small scale, vanishing at large; rarely tested with seeds",
         test="sequential vs parallel blocks at matched params, 12 seeds, nano scale",
         knobs=["residual"], testable_now=False, belief=3,
         stakes="a measurable small-scale cost confirms PaLM's remark and bounds it",
         queries=['all:"parallel attention" AND all:MLP AND all:transformer AND all:PaLM', 'abs:"parallel layers" AND abs:"language model"', 'all:"parallel formulation" AND all:transformer AND all:quality']),
    dict(id="residual-necessary", component="residual-stream",
         claim="Residual connections are necessary; plain deep stacks do not train",
         origin="He et al. 2016; transformer practice",
         evidence_scale="true at depth; at 2 to 4 layers untested and possibly false",
         test="residual vs highway vs plain at 2, 4, 8 layers, matched params, 12 seeds",
         knobs=["residual", "layers"], testable_now=True, belief=5,
         stakes="a plain 4-layer decoder that matches its residual twin would be a small surprise; the depth at which it fails is a phenomenon",
         queries=['all:"residual connections" AND all:transformer AND all:shallow AND all:ablation', 'abs:"without residual" AND abs:transformer AND abs:training', 'all:highway AND all:residual AND all:transformer AND all:comparison']),
    # --------------------------------------------------------------- embeddings
    dict(id="tied-embeddings-free", component="output-head",
         claim="Tying input and output embeddings costs nothing at small scale and saves parameters",
         origin="Press & Wolf 2017; GPT-2 practice; untied in large models since LLaMA",
         evidence_scale="Press & Wolf: LSTM LMs; transformer evidence anecdotal; large models untie for a reason nobody states with numbers",
         test="tied vs untied at matched params (d adjusted), 12 seeds, both families",
         knobs=["family"], testable_now=False, belief=4,
         stakes="a small-scale cost of tying would explain the silent large-scale switch",
         queries=['all:"tied embeddings" AND all:"language model" AND all:comparison', 'abs:"weight tying" AND abs:transformer AND abs:ablation', 'all:"untied" AND all:embeddings AND all:"language model"']),
    dict(id="vocab-bigger-better", component="tokeniser",
         claim="Larger vocabularies help at every scale",
         origin="Tao et al. 2024 (Scaling laws with vocabulary); LLaMA 3 128k vocab",
         evidence_scale="Tao et al.: optimal vocab grows with compute, i.e. small models want SMALL vocabularies",
         test="vocab 1k to 32k at matched params (embedding counted), matched bytes, 12 seeds, bits per byte",
         knobs=["vocab_cap"], testable_now=True, belief=3,
         stakes="a small-scale optimum far below practice, measured in bits per byte, is a clean data point",
         queries=['all:"vocabulary size" AND all:"scaling laws" AND all:"language model"', 'abs:"vocabulary" AND abs:"bits per byte" AND abs:transformer', 'all:tokenizer AND all:"vocabulary size" AND all:"compute-optimal"']),
    # ------------------------------------------------------------------ data
    dict(id="repetition-harms", component="data-order",
         claim="Repeating data beyond about four epochs harms language models",
         origin="Muennighoff et al. 2023 (Scaling Data-Constrained Language Models)",
         evidence_scale="shown at 100M to 9B; the four-epoch figure is quoted as universal",
         test="1 to 32 epochs on a fixed slice at matched steps, 12 seeds, nano scale",
         knobs=["data"], testable_now=True, belief=4,
         stakes="a different threshold or none at 10M params would qualify a widely quoted number",
         queries=['all:"data-constrained" AND all:"language models" AND all:epochs', 'abs:"repeated data" AND abs:"language model" AND abs:degradation', 'all:"multiple epochs" AND all:pretraining AND all:transformer']),
    dict(id="curriculum-useless", component="data-order",
         claim="Data ordering and curricula do not help language-model pretraining",
         origin="folklore; Campos 2021; mixed results since",
         evidence_scale="mixed at scale; at nano scale untested with seeds",
         test="random vs easy-to-hard (by unigram loss) vs hard-to-easy, matched tokens, 12 seeds",
         knobs=["data"], testable_now=True, belief=3,
         stakes="a reliable small-scale effect either way is a phenomenon",
         queries=['all:"curriculum learning" AND all:"language model pretraining"', 'abs:"data ordering" AND abs:pretraining AND abs:transformer', 'all:curriculum AND all:"language model" AND all:"no benefit"']),
    dict(id="dropout-unnecessary", component="training-objective",
         claim="Dropout is unnecessary for LLM pretraining and only hurts",
         origin="practice since GPT-3 (dropout 0); Liu et al. 2023 (Dropout Reduces Underfitting) argues early dropout helps",
         evidence_scale="mixed; at nano scale with a small corpus dropout may matter",
         test="dropout 0 vs 0.1 vs early-only, matched tokens, 12 seeds, single-epoch and multi-epoch regimes",
         knobs=["train"], testable_now=False, belief=4,
         stakes="a regime where dropout helps at nano scale is useful for every small-model practitioner",
         queries=['all:dropout AND all:"language model pretraining"', 'abs:"dropout reduces underfitting"', 'all:dropout AND all:transformer AND all:"single epoch"']),
    # ------------------------------------------------------------------ MoE
    dict(id="aux-loss-needed", component="moe-router",
         claim="MoE routers collapse without a load-balancing auxiliary loss",
         origin="Shazeer et al. 2017; Switch Transformer 2021",
         evidence_scale="collapse observed at scale; loss-free balancing (DeepSeek 2024) shows a bias suffices; small-scale seeds rare",
         test="no balancing vs aux loss vs bias at matched budget, 12 seeds; dead-expert counts",
         knobs=["moe"], testable_now=False, belief=5,
         stakes="already softened by loss-free balancing; a nano-scale map of when collapse happens is a phenomenon",
         queries=['all:"load balancing" AND all:"mixture of experts" AND all:collapse', 'abs:"auxiliary loss" AND abs:"mixture-of-experts" AND abs:"expert collapse"', 'all:"routing collapse" AND all:MoE']),
    # -------------------------------------------------------------- phenomena
    dict(id="grokking-needs-wd", component="training-objective",
         claim="Grokking requires weight decay",
         origin="Power et al. 2022; Nanda et al. 2023",
         evidence_scale="algorithmic tasks; contested (grokking without wd reported with other regularisers)",
         test="modular arithmetic on the paperkiln stack, wd 0 vs 0.1 vs 1.0, 12 seeds, time-to-generalise",
         knobs=["lr", "data"], testable_now=True, belief=3,
         stakes="a wd-free grokking recipe on a standard stack is a data point",
         queries=['all:grokking AND all:"weight decay"', 'abs:grokking AND abs:"without weight decay"', 'all:grokking AND all:mechanism AND all:regularization']),
    dict(id="double-descent-lm", component="training-objective",
         claim="Epoch-wise double descent does not occur in language-model pretraining",
         origin="Nakkiran et al. 2019 showed it in vision and small transformers on MT; LM folklore says no",
         evidence_scale="untested with seeds on decoders at nano scale",
         test="fixed small corpus, many epochs, label noise sweep, 12 seeds, validation loss vs epoch",
         knobs=["data"], testable_now=True, belief=3,
         stakes="a double descent in a nano LM would be a phenomenon paper",
         queries=['all:"double descent" AND all:"language model"', 'abs:"epoch-wise double descent" AND abs:transformer', 'all:"double descent" AND all:"next-token prediction"']),
    dict(id="scaling-law-bottom", component="training-objective",
         claim="Loss follows a clean power law in parameters down to the smallest models",
         origin="Kaplan et al. 2020; Hoffmann et al. 2022",
         evidence_scale="Kaplan's smallest models are ~100k params and already deviate; nano-scale seed variance rarely shown",
         test="params 100k to 60M at matched tokens per param, 12 seeds, fit with seed bands; where does the power law break",
         knobs=["d", "layers"], testable_now=True, belief=4,
         stakes="the seed band at the bottom of the scaling curve is a number nobody reports",
         queries=['all:"scaling laws" AND all:"small models" AND all:deviation', 'abs:"scaling law" AND abs:"seed variance"', 'all:"neural scaling laws" AND all:"parameter count" AND all:"lower bound"']),
    dict(id="val-loss-proxy", component="training-objective",
         claim="Validation loss is a sufficient proxy for downstream quality at small scale",
         origin="universal practice in small-model research",
         evidence_scale="known to break for emergent abilities at scale; at nano scale untested",
         test="rank 12 configurations by val loss and by a fixed synthetic task battery, 12 seeds; rank correlation",
         knobs=["eval"], testable_now=True, belief=4,
         stakes="a low rank correlation would undermine every small-scale ablation including ours",
         queries=['all:"validation loss" AND all:"downstream performance" AND all:correlation AND all:"language model"', 'abs:"perplexity" AND abs:"downstream" AND abs:"small models"', 'all:"loss-to-performance" AND all:"scaling"']),
    dict(id="seeds-dont-matter", component="training-objective",
         claim="Seed variance is negligible for architecture comparisons",
         origin="the one-seed habit of the architecture literature",
         evidence_scale="Bouthillier et al. 2021 and Picard 2021 show large seed effects in vision; for nano LMs the transfer study measured F1 pass rate 16 percent at three seeds",
         test="already partly measured; extend to every panel comparison: what fraction of single-seed sign calls flip under 12 seeds",
         knobs=["seed"], testable_now=True, belief=4,
         stakes="a flip rate for the standard knobs at nano scale is a methodological result with teeth",
         queries=['all:"random seed" AND all:variance AND all:"architecture comparison"', 'abs:"seed" AND abs:"language model" AND abs:"statistical significance"', 'all:"accounting for variance" AND all:"machine learning benchmarks"']),
    dict(id="init-small-better", component="weights-at-inference",
         claim="Smaller initialisation (0.02 and depth-scaled output projections) is required for deep transformers",
         origin="GPT-2 init; Megatron scaling by 1/sqrt(2L)",
         evidence_scale="justified by depth; at 2 to 8 layers the effect is unmeasured",
         test="init scale sweep at 4 and 8 layers, 12 seeds, matched tokens",
         knobs=["init"], testable_now=False, belief=3,
         stakes="init-invariance at nano depth would say the rule is a deep-model rule",
         queries=['all:initialization AND all:transformer AND all:depth AND all:scaling', 'abs:"depth-scaled" AND abs:initialization AND abs:transformer', 'all:"initialization" AND all:"language model" AND all:ablation']),
    dict(id="bias-unnecessary", component="ffn",
         claim="Bias terms in linear layers are unnecessary in transformers",
         origin="PaLM, LLaMA remove them; Dettmers et al. note no loss",
         evidence_scale="removed at scale by choice; small-scale seeded test rare",
         test="bias on vs off at matched params, 12 seeds, nano scale",
         knobs=["family"], testable_now=False, belief=4,
         stakes="a small-scale effect would say biases are a small-model tool",
         queries=['all:"bias terms" AND all:transformer AND all:removed', 'abs:"no bias" AND abs:"language model" AND abs:layers', 'all:bias AND all:"dense layers" AND all:PaLM']),
    dict(id="muon-2x", component="optimiser-state",
         claim="Muon is about twice as token-efficient as AdamW",
         origin="Jordan 2024 (modded-nanoGPT speedruns); Liu et al. 2025 (Muon is Scalable)",
         evidence_scale="speedrun single runs; Moonshot at scale; seed-honest matched-lr comparison at nano scale rare",
         test="adamw vs muon, lr swept per optimiser, 12 seeds, tokens-to-target-loss",
         knobs=["optimizer"], testable_now=True, belief=3,
         stakes="a measured factor with seed bands at nano scale, whatever it is",
         queries=['all:Muon AND all:"token efficiency"', 'abs:"Muon is scalable"', 'all:"modded-nanoGPT" OR all:"speedrun" AND all:optimizer']),
    dict(id="sparse-transfer", component="attention-topology",
         claim="Architectural conclusions drawn at small scale transfer to large scale",
         origin="the premise of every nano-scale ablation, including ours",
         evidence_scale="transfer_s2 (Sep 2026): sign structure of a sparse-attention panel did NOT transfer from d=256 to d=512 under per-width tuning; agreement decays with position on the loss curve",
         test="the finding exists; the claim-level test is whether other knobs (norm, activation, position) transfer across width under the same protocol",
         knobs=["norm", "activation", "position", "d"], testable_now=True, belief=5,
         stakes="a general non-transfer of small-scale ablations across a 4x width range would be the field-level version of the transfer study",
         queries=['all:"scale transfer" AND all:ablation AND all:transformer', 'abs:"small-scale" AND abs:"large-scale" AND abs:"architecture" AND abs:transfer', 'all:"scaling" AND all:"architectural choices" AND all:"do not transfer"']),
]


def all_claims():
    return CLAIMS


JUDGE_SYSTEM = """You are a sceptical methodologist deciding how well a widely repeated claim about transformer training has actually been tested. You are given the claim, where the belief came from, and candidate papers (id, date, title, abstract). Decide:
- TESTED-SOLID: at least one paper tests this claim with a controlled design (matched parameters or compute, multiple seeds or a variance estimate, the alternative actually run) at a scale relevant to the claim, and the result supports the claim.
- TESTED-WEAK: the claim has been tested, but with single seeds, unmatched budgets, a different scale than the claim is repeated at, or a confounded comparison.
- CONTESTED: controlled evidence exists on both sides.
- UNTESTED: the retrieved papers assert, adopt or cite the claim but none tests it under a controlled design; or the only tests are at large scale with no small-scale replication.
Be strict and grounded: quote from each cited abstract the phrase that shows what was actually done (seeds, scale, matched budget). Return JSON: {"verdict": ..., "evidence": [{"id": ..., "title": ..., "quote": ..., "what_it_shows": one sentence}] (up to 4), "gap": one sentence: what a 12-seed matched-budget nano-scale test would add that the literature lacks, or null if nothing, "confidence": 0-1}. Return only the JSON."""


def scope_one(c):
    seen = {}
    for q in c["queries"][:4]:
        for p in P.arxiv_search(q, n=10):
            if p["id"] and p["id"] not in seen:
                seen[p["id"]] = p
        time.sleep(P.ARXIV_SLEEP)
    papers = list(seen.values())[:40]
    if not papers:
        return {"id": c["id"], "verdict": "ERROR", "error": "no retrieval", "n_papers": 0}
    desc = (f"CLAIM: {c['claim']}\nCOMPONENT: {c['component']}\nORIGIN OF THE BELIEF: {c['origin']}\n"
            f"WHAT THE ORIGIN SHOWED (our prior): {c['evidence_scale']}\nOUR PROPOSED TEST: {c['test']}")
    plist = "\n\n".join(f"[{p['id']}] ({p['published']}) {p['title']}\n{p['abstract']}" for p in papers)
    j = P.parse_json(P.ask(P.JUDGE_MODEL, JUDGE_SYSTEM, "CLAIM:\n" + desc + "\n\nPAPERS:\n" + plist, max_tokens=1500))
    return {"id": c["id"], "n_papers": len(papers),
            "papers": [{k: p[k] for k in ("id", "title", "published")} for p in papers],
            "verdict": j.get("verdict", "UNTESTED"), "evidence": j.get("evidence", []),
            "gap": j.get("gap"), "confidence": j.get("confidence")}


def stage_scope(limit):
    path = os.path.join(OUT, "claims_scoped.jsonl")
    done = {r["id"] for r in P.read_jsonl(path) if r.get("verdict") != "ERROR"}
    todo = [c for c in CLAIMS if c["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"scope(claims): {len(todo)} to do ({len(done)} cached)", flush=True)
    for k, c in enumerate(todo, 1):
        try:
            s = scope_one(c)
        except Exception as e:
            s = {"id": c["id"], "verdict": "ERROR", "error": str(e)[:200]}
        if s.get("verdict") != "ERROR":
            P.append_jsonl(path, [s])
        print(f"  {k}/{len(todo)} {c['id']:26s} {s['verdict']:13s} {c['claim'][:70]}", flush=True)


def stage_report():
    scoped = {r["id"]: r for r in P.read_jsonl(os.path.join(OUT, "claims_scoped.jsonl"))}
    weak = {"UNTESTED": 3, "CONTESTED": 2, "TESTED-WEAK": 2, "TESTED-SOLID": 0}
    rows = []
    for c in CLAIMS:
        s = scoped.get(c["id"])
        v = s["verdict"] if s else "UNSCOPED"
        score = c["belief"] * weak.get(v, 1) * (2 if c["testable_now"] else 1)
        rows.append((score, v, c, s))
    rows.sort(key=lambda t: -t[0])
    lines = ["# Claims: ranked by belief x weakness of evidence x testability", "",
             "Score = belief(1-5) x evidence-weakness (UNTESTED 3, CONTESTED/WEAK 2, SOLID 0) x 2 if paperkiln can run it today.", "",
             "| # | score | verdict | testable now | claim | origin | gap |", "|---|---|---|---|---|---|---|"]
    for k, (sc, v, c, s) in enumerate(rows, 1):
        gap = ((s or {}).get("gap") or "").replace("|", "/")
        lines.append(f"| {k} | {sc} | {v} | {'yes' if c['testable_now'] else 'no'} | {c['claim'].replace('|','/')} | {c['origin'].replace('|','/')} | {gap} |")
    lines += ["", "## Evidence, per claim", ""]
    for sc, v, c, s in rows:
        lines += [f"### {c['id']}: {c['claim']}", "", f"Verdict {v}; our prior: {c['evidence_scale']}", ""]
        for e in (s or {}).get("evidence", []):
            lines.append(f"- [{e.get('id')}] {e.get('title')}: \"{e.get('quote')}\" ({e.get('what_it_shows')})")
        lines += [f"- Proposed test: {c['test']} (knobs: {', '.join(c['knobs'])})", ""]
    open(os.path.join(OUT, "claims_ranked.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"report: {len(rows)} claims; verdicts " + str({v: sum(1 for r in rows if r[1] == v) for v in set(r[1] for r in rows)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "scope", "report"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "list":
        for c in CLAIMS:
            print(f"{c['id']:26s} belief {c['belief']} now={'y' if c['testable_now'] else 'n'}  {c['claim']}")
        print(len(CLAIMS), "claims;", sum(1 for c in CLAIMS if c["testable_now"]), "testable today")
    elif a.stage == "scope":
        stage_scope(a.limit)
    else:
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
