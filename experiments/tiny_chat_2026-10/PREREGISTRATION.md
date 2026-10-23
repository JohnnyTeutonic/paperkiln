# Tiny chat from scratch: E5 (the data bar) and E1 (the shape sweep), pre-registration

*Frozen 23 Oct 2026, before any run. Licensed by the commit that adds this file together with
`make_jobs.py` and `analyze.py` (this folder), `tools/colab_chat/tiny_chat_run.py`,
`tools/colab_chat/tiny_chat_data.py`, the jobs files `tools/colab_chat/tiny_chat_{s0,e5,e1}.json`
(and the 60M pair) and their tests, each exercised end to end on CPU at toy size before licence.
Rules change only by a dated amendment at the end of this file, written before the data it
concerns. Literature: `experiments/tiny_chat_litcheck_2026-10-23.md` (E1 and E5 there).*

## 0. Objective and constraints

- **Objective (Jonathan, 23 Oct 2026):** basic multi-turn chat from scratch. The model needs a
  little knowledge; carrying a fact from an earlier turn is the ability measured.
- **Human-written data only, no distillation.** No model-generated text enters training (no SODA,
  UltraChat, everyday-conversations, smol-smoltalk, TinyChat or any teacher output). Section 2
  says how each source meets the rule.
- **Measurement before mechanism.** E5 and E1 fix the data mix and the shape. Role embeddings,
  turn positions and state tokens (E2-E4) are later deltas measured against the gap E5 opens.

## 1. Fixed for every run

| item | value |
|---|---|
| family | llama: RMSNorm, RoPE on every head, SwiGLU, no biases, tied input/output embedding, head dim 64 |
| vocabulary | the 16,384-entry Llama 3-scheme byte-level BPE (`C:/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf`, sha256 0b880aa073a7e27c...), uploaded to every VM |
| context | T = 512 for pretraining, SFT and evaluation |
| pretraining budget | 10,000 steps x 16,384 tokens = **163.84M tokens** per run; batch 32 x 512, or 16 x 512 with 2-step accumulation at L >= 16 (activation memory; the same tokens and update) |
| optimiser | AdamW, clip 1.0, warmup 500 steps, cosine to 0.1 x lr at step 10,000; the lr per shape from Stage 0 (section 4) |
| SFT | from the pretrained weights; human dialogues in mtstudio's segments form (one row per dialogue, every assistant turn trained, prompts masked); 1,500 steps x 32 rows x 512; AdamW lr 3e-4, warmup 100, cosine to 0.1x; the same set and recipe for every run |
| seeds | 1, 2, 3 (`train.seed`: initialisation and data order; the same seed in every cell, so runs pair by seed); seeds 4-5 only under the extension rule; screens use seed 101 |
| evaluation | Carry-5 on the SFT model (section 3); two fixed text panels on the base model; SFT held-out loss |

**Why AdamW, not Muon.** Atlas S2/S3-muon found Muon better at 1-3M parameters on CPU. In mtstudio
Muon reads gradients on the host, which closes the device-resident step on CUDA, so at 20M it
would cost wall-clock this budget cannot absorb, and Muon x lr interacts (S3-lrxopt). A Muon arm
belongs to a later pre-registration on the winning shape.

**Why no WSD schedule.** mtstudio has warmup + constant or cosine only. WSD's benefit (cheap
intermediate-budget models) is not used here: every claim is scoped to the one budget (section 9).

**Vocabulary note.** At this scale the tied embedding is 14-42% of all parameters (section 6).
The 16k Llama 3-scheme vocabulary stays because (a) winners must export to ember and llama.cpp,
which tokenise this scheme exactly; (b) Carry-5 and the panels are tokenised once, so losses in
nats per token compare across every run; (c) a smaller vocabulary trades embedding parameters
for longer sequences, a different experiment. **E0 (optional, not scheduled):** an 8,192-entry
vocabulary trained by the same recipe on the same sample, at 20m_L8 on the dE1 mix, 3 seeds,
compared on Carry-5 and on bits per byte (not loss per token). It runs only if E1 finds the
shallow-wide end ahead, where the embedding share is the obvious alternative explanation.

## 2. Data, and how each source meets the human-only rule

- **Human dialogues:** `corpus/fetchers/human_dialogue.py default` (paperkiln a9ed70e6; survey in
  `corpus/HUMAN_DIALOGUE_SURVEY.md`): the DEFAULT set, permissively licensed and documented as
  written by people: oasst, ConvAI2, Wizard of Wikipedia, Blended Skill Talk, Taskmaster 1-3,
  MultiWOZ, Self-dialogue. The opt-in sets (share-alike, non-commercial, LLM risk) await
  Jonathan's licence decision and enter only by an amendment before Stage 0 (they change the
  dialogue levels below). `tiny_chat_data.py` normalises to alternating user/assistant turns,
  runs `corpus/decontam.py`, and splits by sha256 of the content key: **5% val** (the dialogue
  panel), **10% SFT**, **85% pretraining**. The SFT dialogues never appear in any pretraining mix.
- **The corpus is small.** On the laptop, oasst, BST and Self-dialogue are 8.9M tokens of the 16k
  vocabulary (40 MB rendered); the whole default set is expected at roughly 30-60M tokens. A
  163.84M-token run cannot hold 50% unrepeated human dialogue, so the design plans for repeated
  passes (at most four) and sets its dialogue levels from the measured size (section 5).
- **General text: FineWeb-Edu**, the pinned sample-10BT shard 0 already on every VM. It is
  human-written web text; Llama 3 annotations trained the classifier that *selected* it, but no
  model wrote it. Its 2023-24 crawls can hold some machine-written pages. Under the rule as
  stated it is admitted; if Jonathan rules strictly, a dump filter (crawls up to CC-MAIN-2022-40)
  is added to `fineweb_edu.py` as a new corpus id and recorded here as an amendment before
  Stage 0.
- **The knowledge arm runs with or without Simple English Wikipedia.** The general text of every
  E5 and E1 run is FineWeb-Edu, so nothing here needs Wikipedia. **E5w** (optional) is the arm
  that uses it: levels 0 and E1's level with Simple English Wikipedia in place of FineWeb-Edu,
  3 seeds (6 runs), run only if Jonathan allows share-alike sources (`human_dialogue.py
  simplewiki --allow-sa`; 157 MB of parquet, the 20231101 dump, which postdates ChatGPT, so some
  edits may hold model text: survey section 3) and E5's primary contrast resolved. It checks
  that the dialogue effect does not depend on the general-text source; being several times
  smaller than the budget, it repeats in the 0% cell (the fetcher measures by how much; more than
  four passes is reported as such). Without Wikipedia, E5w is not run and nothing else
  changes. If FineWeb-Edu were ruled out too, the strict pre-2023 dump filter above keeps a
  human general-text source.
- **The vocabulary** was trained on FineWeb-Edu plus lever 2's dialogue text, which includes
  synthetic sources. A vocabulary is a segmentation; no model output enters any training corpus
  through it.
- **Mixes:** 800 MB per level (~182M tokens; the 95% training part covers one budget), share =
  dialogue fraction by bytes (`corpus/mix_pretrain.py`'s unit, as lever 2's ratios), documents
  interleaved by seed, no annealing tail. The realised token share is in each mix's ledger. The
  dialogue source is k concatenated copies of the pretraining split, the fewest that cover the
  level (FineWeb-Edu is never repeated); k <= 4 (Muennighoff et al. 2023: up to four epochs cost
  little). The level rule (section 5) keeps every level within four copies; a VM-measured split
  smaller than the home estimate falls back to the largest share four copies cover, and the
  ledger and every table then show `share_effective`.

## 3. The benchmark and the metrics

**Carry-5** (benchmark agent: `corpus/eval/carry5_v1.jsonl`, `tools/carry5_eval.py`): templated
5-turn dialogues; a fact planted at turn 1-3 is queried at turn 4-5; 0-2 same-type distractors;
entities and values from held-out pools; each item has a single-turn twin (fact and question in one
turn). Scored by log-prob forced choice over candidate values and by greedy exact match, with the
earlier assistant turns either **gold** (reference replies) or **self** (the model's own greedy
replies).

**Primary metric, per run: Y = forced-choice accuracy with SELF history on the multi-turn items
minus forced-choice accuracy on their single-turn twins**, on the SFT model, over all items.
Higher is better (0 means no loss from spreading the exchange over turns).

**Validity gate (floor).** A cell's Y is interpreted only if its mean twin accuracy is at least
chance + 10 points; otherwise a model that fails everything would score Y near 0 and look good.
A contrast with a floor cell is reported as FLOOR, not as a verdict.

**Secondary metrics (descriptive unless named in a hypothesis):**
- S1 multi-turn accuracy (self), S2 twin accuracy, S3 Y with GOLD history (carry-over without
  error propagation), S4 greedy exact match (multi and twin);
- S5 breakdowns by distance, family and interference;
- S6 panel losses on the base model, nats/token, 512 held-out windows of 512 tokens each by the
  zero-rate pass (1 step at lr 0 from the weights; the weights stay byte-identical): FineWeb-Edu
  held-out text (the knowledge and fluency proxy) and human dialogue val;
- S7 SFT held-out loss (the same 5% of the same SFT set for every run);
- S8 pretraining validation trajectory (32 windows per eval; each mix's own held-out part);
- S9 throughput: tokens/s per shape alone on the GPU (Stage 0 probes) and in the runs (shared
  VM), peak GPU memory, wall-clock; S10 Y per L4-hour;
- S11 base-model Carry-5 with gold history (only meaningful for mixes that contain dialogue).

**Smallest effect of interest (SESOI):** 3 points of Y for a pairwise contrast (E5); 1 point of
Y per doubling of depth for the E1 slope (4 points from L2 to L32); 0.01 nats per doubling, or
0.02 nats pairwise, for panel losses.

## 4. Stage 0: throughput probes and learning-rate screens (`tiny_chat_s0.json`)

- **Probes:** each shape alone on the L4, 300 steps on the dE1 mix (section 5); steady tokens/s timed
  between step 50 and the last step from mtstudio's own step events; peak GPU memory.
- **Screens:** each shape at lr {5e-4, 1e-3, 2e-3}, 1,500 steps (warmup 150, cosine to 0.1x),
  seed 101, dE1 mix; final validation loss on 128 held-out windows.
- **Rule (fixed now; `tiny_chat_run.py choose`):** per shape, among screens that did not diverge
  (non-finite loss, or final validation loss above the first training loss), the lowest final
  validation loss; screens within 0.01 nats of it are tied and the smallest tied lr wins. The
  choice is written to `stage0/lr_choice.json` and committed before any E run. The E jobs files
  are regenerated from it (`make_jobs.py`); their gate job refuses on the VM unless every
  pretraining job's lr equals the choice and Carry-5 is present.
- If all three screens of a shape diverge: one amendment may extend the grid downwards by halves
  (2.5e-4, 1.25e-4); if those diverge too, the shape is dropped and reported.
- Stage 0 may change the schedule (sessions, concurrency), never the token budget or the design.

## 5. E5: the data bar (`tiny_chat_e5.json`)

**Design.** Shape 20m_L8 (d 448, 8 layers, 7 heads, d_ff 1,248; 19.85M non-embedding + 7.34M
embedding = 27.19M). Four pretraining dialogue levels d0 < d1 < d2 < d3 of the fixed budget, the
rest FineWeb-Edu; the same SFT; 3 seeds: 12 runs.

**The level rule (fixed now; `make_jobs.py dialogue_levels`, tested).** Let P be the rendered
bytes of the pretraining split, 95% of it usable per copy, and M = 800 MB the mix size.
- If four copies cover a whole mix (4 x 0.95 P >= M): the nominal levels **0 / 25 / 50 / 100%**.
- Otherwise **0 and one, two and four passes** over the pretraining dialogues:
  share_k = floor(100 k x 0.95 P / M)% for k = 1, 2, 4 (for example P = 110 MB gives
  0 / 13 / 26 / 52%; the laptop's three corpora alone give 0 / 4 / 8 / 16%).
- **dE1**, the level E1 trains on (and E5's cell that doubles as E1's L8 cell), is the non-zero
  level nearest 50% (ties: the smaller): 50% in the nominal case, the four-pass level otherwise.
- P comes from the default set measured whole (`make_jobs.py --dialogue-pt-bytes`, from the
  built corpora's `text.txt` sizes x 0.85, or the data job's `human_pt_bytes`), written to
  `design.json` with `final: true` and committed before Stage 0. Every gate job refuses a
  provisional design (`final: false`, written while sources are missing), so no run uses levels
  set from part of the corpus. Levels that are not distinct (a corpus under ~10 MB) stop E5.

**Hypotheses and directions, recorded now** (d0 = 0%; dtop = d3, the most dialogue):
- **H5 (registered prediction; TurnWise 2603.16759 and 2510.20358 find that dialogue data moves
  multi-turn ability; SmolLM2 2502.02737 is data-centric):** dialogue in pretraining raises
  carry-over after the same SFT. **Y(dE1) - Y(d0) > 0.** Ordering prediction (descriptive): Y
  rises from d0 to dE1.
- **H5-knowledge (2510.20358: dialogue-only pretraining is weak on general suites):** the
  FineWeb-Edu panel loss is higher at dtop than at d0. **L(dtop) - L(d0) > 0.** The direction is
  near-certain; its size is the knowledge price set against the Y gain, which is what the data bar
  weighs.
- **H5-saturation** (only when dtop differs from dE1, i.e. the nominal levels): dropping general
  text entirely buys no further carry-over: **Y(dtop) - Y(dE1) not above the SESOI** (predicted
  EQUIVALENT or NOT_RESOLVED, not SUPPORTED).
- **Counter-hypothesis H5-0 (format is learnt in SFT; superficial alignment):** Y(dE1) - Y(d0)
  inside +-3 points (EQUIVALENT). With a small human corpus this is the live alternative: the
  same dialogues might do as much in SFT alone as in pretraining plus SFT.
- **Counter-hypothesis H5-neg (expository text teaches entity tracking that chit-chat does not):**
  Y(dE1) - Y(d0) < 0.

**Falsification.** H5 is falsified by CONTRADICTED (Y(dE1) below Y(d0) beyond the critical
value) or by EQUIVALENT (the 90% interval inside +-3 points). H5-knowledge is falsified if the
FineWeb panel loss at dtop is not above d0 (CONTRADICTED or EQUIVALENT within 0.02 nats).
H5-saturation is falsified if Y(dtop) - Y(dE1) is SUPPORTED.

**The data bar** (what E5 hands to future work): the share with the highest mean Y among shares
whose twin accuracy is within 3 points of the best cell's and whose FineWeb panel loss is within
0.10 nats of the 0% cell's; shares within 3 points of that Y are tied and the smallest wins
(it keeps the most general text).

## 6. E1: the shape sweep (`tiny_chat_e1.json`; 60M: `tiny_chat_e1_60m.json`, on Jonathan's go)

**Design.** Non-embedding parameters matched (~20M; ~60M in the second stage) plus the tied
16,384 x d embedding; d a multiple of 64 (head dim 64, heads = d/64), the d whose matching
d_ff/d is nearest 3, d_ff a multiple of 32 bringing the count nearest the target
(`make_jobs.py shape_for`). The same 163.84M tokens, the dE1 mix (section 5; fixed before Stage 0, so E1 runs beside E5
and its L8 cell is E5's dE1 cell), the same SFT, 3 seeds.

| shape | layers | d | heads | d_ff | d_ff/d | non-emb | emb (V x d) | total | emb share | est. tok/s | PT h | SFT h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20m_L2 | 2 | 896 | 14 | 2528 | 2.82 | 20.02M | 14.68M | 34.70M | 42% | 28.1k | 1.6 | 0.24 |
| 20m_L4 | 4 | 640 | 10 | 1760 | 2.75 | 20.08M | 10.49M | 30.56M | 34% | 29.3k | 1.6 | 0.23 |
| 20m_L8 | 8 | 448 | 7 | 1248 | 2.79 | 19.85M | 7.34M | 27.19M | 27% | 14.6k | 3.1 | 0.47 |
| 20m_L16 | 16 | 320 | 5 | 864 | 2.70 | 19.84M | 5.24M | 25.08M | 21% | 7.3k | 6.2 | 0.93 |
| 20m_L32 | 32 | 192 | 3 | 832 | 4.33 | 20.07M | 3.15M | 23.21M | 14% | 3.7k | 12.4 | 1.87 |
| 60m_L4 | 4 | 1088 | 17 | 3136 | 2.88 | 59.89M | 17.83M | 77.72M | 23% | 12.5k | 3.6 | 0.55 |
| 60m_L8 | 8 | 768 | 12 | 2240 | 2.92 | 60.18M | 12.58M | 72.76M | 17% | 13.2k | 3.5 | 0.52 |
| 60m_L16 | 16 | 512 | 8 | 1760 | 3.44 | 60.05M | 8.39M | 68.44M | 12% | 7.3k | 6.2 | 0.93 |
| 60m_L32 | 32 | 384 | 6 | 1120 | 2.92 | 60.19M | 6.29M | 66.48M | 9% | 3.7k | 12.4 | 1.87 |

**The embedding confound, stated before data.** Matching non-embedding parameters (Kaplan's
convention) gives the shallow-wide shapes far more embedding: 14.7M at L2 against 3.2M at L32, so
L2 has 1.5x L32's total parameters. A depth win is therefore conservative against this confound;
a width win is ambiguous between width and lexical capacity (E0 is the check). Every table
reports total parameters and the embedding share. A second difference sits at the deepest 20M
shape: d_ff/d is 4.33 at L32 (2.70-2.82 elsewhere), because d = 192 is the multiple of 64 whose
matching ratio is nearest 3.

**Hypotheses and directions, recorded now:**
- **H1 (Jonathan's position): depth serves sequential reasoning, width serves parallel capacity.**
  (a) Carry-over improves with depth: **the slope of Y on log2(layers) > 0.**
  (b) Knowledge and fluency improve with width: **the FineWeb panel loss rises with depth (slope
  > 0)** and twin accuracy falls with depth (slope < 0).
- **H-lit (TinyStories 2305.07759: context tracking depends more on layers, facts on width;
  MobileLLM 2402.14905: deeper-thinner wins at fixed parameters, ~30 layers near-optimal at 125M;
  Gemstones 2502.06857: deeper wins per FLOP):** agrees with H1(a); MobileLLM predicts the panel
  loss does **not** rise with depth, against H1(b). The sign of the panel-loss slope separates H1
  from H-lit.
- **H-recall (counter; induction heads form at two layers, Olsson et al. 2022; Zoology 2312.04927:
  recall is an attention-width property):** Carry-5's carry-over is in-context retrieval, which
  two to four layers suffice for; predicts a Y slope inside +-1 point per doubling, or negative.
- **H-Kaplan (counter; 2001.08361: shape matters little at fixed N):** every slope inside its
  SESOI.
- **H-interior (MobileLLM's optimum; a 192-wide model may be too thin):** Y peaks at an interior
  depth: **the quadratic contrast of Y over log2(layers) is negative.** Descriptive argmax.
- **Throughput (Gemstones: wider wins per GPU-hour):** per token, L32 takes **at least 3x** L2's
  time (the probes measure it); so even if depth wins per token, the ranking per L4-hour (S10) may
  differ, and both are reported.
- **60M stage:** the sign of the Y slope at 60M equals the sign at 20M (the Atlas lesson T1/T2:
  signs can transfer while magnitudes do not; registered as a sign prediction only).

**Falsification.** H1(a) is falsified by a CONTRADICTED or EQUIVALENT Y slope. H1(b) is falsified
by a FineWeb panel-loss slope that is CONTRADICTED (loss falls with depth) or EQUIVALENT; a
CONTRADICTED panel slope together with a SUPPORTED Y slope is the H-lit pattern. H-recall is
falsified by a SUPPORTED Y slope. H-interior is falsified by a quadratic contrast that is
CONTRADICTED or EQUIVALENT.

## 7. Analysis plan (`analyze.py`, tested on synthetic summaries)

- **Unit:** one number per (cell, seed). Cells pair by seed (same seed, same initialisation where
  the architecture is the same, the same SFT order).
- **Contrasts per seed, then a one-sample t over seeds** (Atlas house style; two-tailed, alpha
  0.05, df = n - 1; n = 3 gives critical 4.303):
  - E5 primary: D(s) = Y(dE1, s) - Y(d0, s). Also Y(dtop) - Y(dE1) (nominal levels only), the
    FineWeb panel L(dtop) - L(d0), and descriptively each other level minus d0 and S3's
    gold-history version of the primary.
  - E1 primary: the OLS slope of Y on log2(layers) within each seed (weights over L2...L32 sum to
    zero; units: points of Y per doubling). Also the slopes of the FineWeb panel loss and of twin
    accuracy, the quadratic contrast of Y, and the gold-history slope.
- **Verdicts** (`decide`): SUPPORTED = the registered direction, |t| above the critical value and
  |mean| >= SESOI; CONTRADICTED = the opposite direction with |t| above the critical value;
  EQUIVALENT = the 90% interval inside +-SESOI; SIGNIFICANT_BELOW_SESOI; otherwise NOT_RESOLVED.
  Every result is reported with its per-seed values, mean, SD, t, 90% interval and sign agreement.
- **Extension rule (stopping rule 1):** a primary contrast NOT_RESOLVED at three seeds with the
  same sign on all three adds seeds 4 and 5 to the cells it uses, once. The five-seed result
  (critical 2.776) is final and labelled "resolved at the second look"; with two looks the
  familywise false-positive rate of that label is at most 0.10, and it is reported as such.
- **No peeking:** `analyze.py` refuses a contrast whose cells are incomplete; `--partial` prints
  descriptives labelled DESCRIPTIVE, PARTIAL and no verdict. Interim reads during the runs are
  limited to failure monitoring (losses, divergence, throughput).
- **Seed-lottery discipline:** no single-seed result is quoted as a direction; sign agreement is
  reported beside every mean.

## 8. Stopping and failure rules

1. The extension rule above.
2. **Divergence of a main run** (non-finite loss, or final validation loss above its first
   training loss): rerun once, same seed, at half the chosen lr, recorded per run; a second
   divergence marks the cell failed and the contrasts that use it are not computed.
3. **Floor:** the validity gate (section 3); a floor cell ends that contrast as FLOOR.
4. **Reclaims:** a pretraining run resumes from its last relayed weights with a cold optimiser
   (the runner's weights-only relay every 2,500 steps); each resume is recorded per run, and the
   primary contrasts are also reported without resumed runs when any exist. A lost post job is
   rerun from the pretrained weights.
5. **Budget:** if Stage 0's probes put the core programme (Stage 0 + E5 + E1 at 20M) above
   80 L4 VM-hours, Jonathan decides before E1 launches whether to proceed as designed; the token
   budget per run and the shapes do not change silently.
6. **E1 at 60M and E5w** launch only on Jonathan's go, after the 20M results.

## 9. Threats, stated before data

- **One budget.** 163.84M tokens is 8 tokens per non-embedding parameter at 20M and under 3 at
  60M (data-limited). Atlas S1c showed a comparison's sign can depend on the budget; every claim
  is scoped to this budget.
- **Learning rates chosen at 1,500 steps**, extrapolated to 10,000; the screen picks among three.
- **The SFT recipe is fixed across shapes**, not tuned per shape.
- **Embedding and d_ff confounds** (section 6).
- **Shared GPUs.** Throughput in the runs is measured beside other jobs; the probes measure it
  alone, and the run-level numbers are reported with the count of concurrent jobs.
- **Template exposure.** Mixes with dialogue teach the chat template tokens in pretraining; the 0%
  cell meets them only in SFT. That is part of what E5 measures, not a nuisance to remove.
- **Carry-5 is a templated proxy.** Forced choice can reward surface copying; S3 and S4 bound the
  reading; a Carry-5 gain is not a claim of general chat quality.
- **The human dialogue corpus** sets the register (crowd chit-chat vs assistant-style); E5's
  answer holds for this corpus.
- **Three seeds** give little power for pairwise contrasts (E5); the E1 slope pools five shapes
  per seed and is better powered. The extension rule is the remedy, registered now.
- **Cold-optimiser resumes** (rule 4).

## 10. Compute budget and schedule

**Throughput model** (`make_jobs.py`; Stage 0 replaces it with measurements). Receipts on the L4:
arxiv_2302_13971_base (llama d384 L6, T 512, batch 16, 16k vocabulary) 19,388 tokens/s beside
pk_base_v1; pk_base_v1 (d768 L12, same T and batch) 0.84-1.01 s/step (8.1-9.8k tokens/s, from
the runner's relay times 22-23 Oct). Both give about **0.07 s per layer per 8,192 tokens**,
independent of width between d 384 and 768, so the model is tokens/s = min(117k / layers,
6.0e12 / FLOPs per token). It makes depth expensive: if the per-layer cost is not the binding one
for thin models, the deep shapes cost up to ~3x less than below. A VM running 3-4 tiny jobs is
assumed to deliver 2.5 jobs' worth (anchors were measured with two jobs sharing).

| block | jobs | job-hours | L4 VM-hours |
|---|---|---|---|
| Stage 0, 20M (5 probes + 15 screens) | 20 | 12 | 5 |
| E5 (4 dialogue levels x 3 seeds at 20m_L8) | 12 | 43 | 17 |
| E1 20M (L2, L4, L16, L32 x 3 seeds; L8 from E5) | 12 | 75 | 30 |
| Stage 0, 60M (on go) | 16 | 12 | 5 |
| E1 60M (4 shapes x 3 seeds, on go) | 12 | 89 | 36 |
| **core (Stage 0 + E5 + E1 20M)** | | **130** | **52** |
| with the 60M stage | | 232 | 93 |

Plus about an hour per session for the data job (CPU on the VM) and the evaluations. At Colab's
L4 rate (about 4.8 compute units per hour; confirm on the resources page) the core is ~250 units
of the 600 a month, the 60M stage another ~200. The next base (pk_base_v1) is 15-25 L4-hours for
scale. **T4:** unmeasured; FLOP-bound it would be 3-4x slower, but the d-independence above hints
at a bound T4 shares; a 20-minute T4 probe would tell whether T4 sessions (cheaper per hour) can
carry the 20M runs. The runner asks for an L4 today.

**Schedule (1-2 concurrent sessions; the main session launches; at most 4 Colab sessions in all):**
1. Fix the dialogue levels (`make_jobs.py --dialogue-pt-bytes`, section 5). Stage 0: one session, ~6 hours (`pk-tc-s0`). At home: `tiny_chat_run.py choose`, commit
   `stage0/lr_choice.json`, rerun `make_jobs.py`, commit, push paperkiln.
2. E5 (`pk-tc-e5`, max 3 jobs, ~17-18 h) and E1 at 20M (`pk-tc-e1`, max 4 jobs, ~23 h; the L32
   runs are the critical path) in parallel. E1 may outlast the runner's `--max-hours 23.5`; the
   watchdog relaunch resumes it from the relays.
3. `analyze.py`, results file, registry rows.
4. On Jonathan's go: Stage 0 at 60M, then E1 at 60M; E5w if E5's primary resolved.

**Home footprint:** per pretraining run, the relayed weights plus the final safetensors and GGUF
(~0.3 GB at 20M); once a run is home its relay copy (`model.safetensors`) can go, leaving ~0.2 GB.
Post jobs bring a summary only; each cell's best SFT model comes home through its pick job.

## 11. Interfaces this design depends on

- **Carry-5** (`tools/carry5_eval.py`, benchmark agent; reconciled 23 Oct): `tiny_chat_run.py
  carry5_argv` calls `carry5_eval.py run --model m=SFT_SPEC --scorer mtstudio --mtstudio BIN
  --gen mtstudio --set corpus/eval/carry5_v1.jsonl --port P --out DIR` (self history played by
  `mtstudio chat` on a per-job port, since post jobs share a VM; log-probs by `mtstudio score`,
  which the benchmark agent adds to mtstudio) and reads `summary.json` `models.m.fc.acc.<mode>.
  all.mean` for gold, self, twin and noctx; chance is 0.25 (four candidates). Y = acc(self) -
  acc(twin). The base model is scored teacher-forced (`--gen none`). Still needed from the
  benchmark: the frozen items file in the commit; items at most ~400 tokens in this vocabulary
  with self replies capped (`--max-new` 24 by default); enough items that the item-level SE of Y
  is under ~1.5 points; a run time of minutes for a 27M model on the VM; and Carry-5's
  decontamination (`tools/carry5_build.py decontam`) run over the human corpora before Stage 0.
- **Human dialogues** (`corpus/fetchers/human_dialogue.py default --out-dir DIR`, paperkiln
  a9ed70e6): `human_<source>_v1/dialogues.jsonl` of `{"messages", "uri"}`, alternating from a
  user turn, pinned, no model-written source. The measured size of the whole default set fixes
  the dialogue levels (section 5); the nominal 0/25/50/100% levels need about 250 MB of rendered
  dialogue (about 210 MB in the pretraining split).
- **Runner** (`runner_multi.py`, main session): pretraining as training jobs (relays, cold
  resume), post/pick/probe/screen as data jobs; `init_job` on a post job brings its pretrained
  weights to a fresh VM.

## 12. Launch commands (not run; for the main session)

```
# once the default human set is home (C:/ml_artifacts/human_dialogue; copied into corpus/data):
python experiments/tiny_chat_2026-10/make_jobs.py --dialogue-pt-bytes P   # P = 0.85 x the default set's text.txt bytes
# commit design.json and the regenerated files, push paperkiln, then:
python3 -u tools/colab_chat/runner_multi.py --commit SHA --jobs tools/colab_chat/tiny_chat_s0.json \
  --name pk-tc-s0 --art /mnt/c/ml_artifacts/tiny_chat/s0 --state-dir /home/jonat/pk_tc_s0_runner \
  --max-hours 23.5 --upload vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf
python3 tools/colab_chat/tiny_chat_run.py choose --art /mnt/c/ml_artifacts/tiny_chat/s0 \
  --out experiments/tiny_chat_2026-10/stage0/lr_choice.json
python3 experiments/tiny_chat_2026-10/make_jobs.py      # then commit, push paperkiln, and launch:
#   --jobs tools/colab_chat/tiny_chat_e5.json --name pk-tc-e5 --art /mnt/c/ml_artifacts/tiny_chat/e5 ...
#   --jobs tools/colab_chat/tiny_chat_e1.json --name pk-tc-e1 --art /mnt/c/ml_artifacts/tiny_chat/e1 ...
python experiments/tiny_chat_2026-10/analyze.py --art C:/ml_artifacts/tiny_chat --out experiments/tiny_chat_2026-10/analysis.json
```

Preconditions for the Stage 0 launch: `design.json` final (the whole default human set measured;
Stage 0's gate job checks it), Carry-5's decontamination of the human corpora, and Jonathan's
ruling on FineWeb-Edu and on the opt-in dialogue licences (section 2). The data job fetches the
default set again on the VM (~2.7 GB of downloads).
Preconditions for E5/E1: `stage0/lr_choice.json` committed and Carry-5 in the commit (the gate
job checks both on the VM).

## Amendments

### Amendment 1 (23 Oct 2026, before any run): the data, the 60M stage, and an early-stopping study

Written after Jonathan's decisions of 23 Oct 2026 (`LOCAL_LLM_STACK.md` section 9) and before any
Stage 0, E5, E1 or successive-halving job ran. Sections 0-12 stand except where this amendment
says otherwise.

**A1.1 Data.**
- Human-written text only, as section 0. The dialogue pool of E5 and E1 (the pretraining
  dialogue, the SFT set and the dialogue panel) is **13 sources**: the default set (oasst,
  convai2, wow, bst, taskmaster, multiwoz, self_dialogue), the share-alike Topical-Chat, Dolly and
  QuAC, and the non-commercial EmpatheticDialogues, DailyDialog and No Robots (Jonathan:
  share-alike and non-commercial sources are in). OpenAssistant stays in (Jonathan: clean enough).
- **Aya English stays out** (`aya_en`, behind `--allow-llm-risk`): it is the one source with
  evidence of ChatGPT-written text (survey section 3), and the decision covered licences and
  OpenAssistant, not the LLM-risk gate.
- **FineWeb-Edu counts as human-written** (Jonathan): the general text of every run, no dump filter.
- **Simple English Wikipedia** (`human_simplewiki_v1`, 224,597 articles, 79.5M tokens of the 16k
  vocabulary) is not dialogue and enters no E5 or E1 mix. Its licence condition for E5w (section 2)
  is met; E5w's other condition stands (E5's primary contrast resolved), and its jobs are made then.
- **Taskmaster is Taskmaster-1 and -2.** The v1 reader maps the speaker labels `USER` and
  `ASSISTANT`; Taskmaster-3 writes them in lower case, so its 23,757 dialogues drop
  (`unknown_speaker`) and the pinned corpus holds 28,791 dialogues from TM-1 and TM-2. TM-3's agent
  turns are filled from templates. Adding it would be a new corpus id and a further amendment
  before Stage 0, with P and the levels measured again.
- **Pins.** All 15 human corpora are pinned (`corpus/hashes/human_*_v1.sha256`; the ParlAI archives
  by sha256 in `human_dialogue.py`). The VM data job rebuilds the 13 training sources and refuses
  unless every file equals its pin (`tiny_chat_data.py --require-pinned`), so every session trains
  on the bytes measured at home.
- **Carry-5 decontamination.** `tools/carry5_build.py decontam` over the 13 sources and Simple
  English Wikipedia (395,169 rows; `carry5_decontam_human_v1.json` in this folder): **no item
  leaks** (an item's fact sentence and its question in one row); 44 items' fact sentences occur
  somewhere (everyday sentences such as "I work as a chef."); 6 Carry-5 user turns occur as whole
  user turns. The data job's decontamination now adds Carry-5 to the frozen sets (its user turns as
  prompts, its gold replies as references; `corpus/decontam.py`'s rules) and drops those **52 of
  170,572 dialogues** (convai2 42, wow 3, taskmaster 3, topical_chat 2, multiwoz 1, self_dialogue 1).
  The corpus builds themselves keep the frozen sets only: their v1 pins predate Carry-5
  (`human_dialogue.py --carry5` opts in for a new corpus id).

**A1.2 Dialogue levels (section 5's rule, applied).**
- **P = 149,473,532 bytes**, measured by `tiny_chat_data.py --measure-only --require-pinned` over
  the 13 pinned sources with the data job's own normalisation, decontamination and split (170,520
  dialogues: val 8,564, SFT 16,977, pretraining 144,979; `design_p.ledger.json`). Identical bytes
  and code make it the data job's `human_pt_bytes`, so no level falls back to `share_effective`.
- Four copies cover 71% of a mix, under 100%, so the levels are 0 and one, two and four passes:
  **0 / 17 / 35 / 70%** (`design.json`, final).
- **dE1 = 35%** (two passes). Section 5's operative rule is "the non-zero level nearest 50% (ties:
  the smaller)": 35 is 15 points from 50, 70 is 20. Its parenthetical "the four-pass level
  otherwise" described the small corpus measured then, where four passes was nearest 50%. E1 trains
  on the 35% mix and its L8 cell is E5's 35% cell.
- dtop = 70% differs from dE1, so **H5-saturation is tested** as registered (Y(70) - Y(35)).
- The SFT set and the dialogue panel are the 10% and 5% splits of the same 13-source pool.

**A1.3 The 60M stage** is authorised now (Jonathan: train now, the 60M stage included). Section 8
rule 6 changes: Stage 0 at 60M may launch beside Stage 0 at 20M, and E1 at 60M after its own
learning-rate choice, without waiting for the 20M results. The 60M sign prediction (section 6) is
unchanged. Section 8 rule 5 (Jonathan decides before E1 if Stage 0 puts the core above 80 L4
VM-hours) stands.

**A1.4 The early-stopping study.** Jonathan asked whether an early-stopping mechanism can show
whether fast learning happens. It has three parts; none changes a primary contrast.

*(a) The registered cells run to full length.* No E5, E1 or 60M run is stopped, shortened or
dropped for its early performance. Stopping slow starters would favour configurations that learn
fast early, the very thing this study asks about, and would bias E1's depth slope in particular
(deep, thin models take longer per step and may start slower). The only stops remain section 8's
divergence and failure rules.

*(b) Rung logging, every E5, E1 (20M) and E1 (60M) pretraining run.* Rungs at **5, 10, 25 and 50%**
of the 10,000 steps (steps 500, 1,000, 2,500, 5,000); **100%** is the post job.
- The VM keeps the weights at each rung (`jobs_vm.py` `snapshot_steps` copies that step's
  checkpoint before the next save overwrites it) and a light rung job (its own slot, so it never
  holds back a training job) scores them: the two zero-rate panels (S6: FineWeb-Edu and human
  dialogue val, 512 windows) and **base-model Carry-5, teacher-forced** (forced choice with gold
  history, twin and noctx; no generation). Each rung's `rung.json` comes home. The 100% point is the
  post job's base-model Carry-5 and panels (`final_rung`).
- Validation every 500 steps (was 1,000), so every rung has a validation loss. mtstudio draws its
  evaluation windows from a fixed generator of their own: the training data order and every update
  are unchanged.
- **Rung predictors:** Y_base = acc_gold - acc_twin (the base model's gold-history analogue of the
  primary metric), acc_gold, and the two panel losses.
- A rung lost to a replaced VM (a snapshot lives only on the VM that ran that step) is recorded as
  lost and never imputed.
- **Registered analysis** (`analyze.py` `rung_analysis`). Blocks: E5 (12 runs), E1 at 20M (15,
  with E5's 35% cell), E1 at 60M (12), and **pooled 20M** (E5 plus E1's other depths, 24 runs). At
  each rung, for the pairs Y_base -> Y, acc_gold -> Y (Y = the primary metric: SFT model, self
  history), FineWeb panel -> final FineWeb panel and dialogue panel -> final dialogue panel:
  Spearman's rho across runs, its 90% percentile bootstrap interval (2,000 resamples of runs, seed
  0), and the **top-third check**: keeping the ceil(cells / 3) best cells by the predictor's cell
  means, is the cell with the best final mean kept?
- **Threshold, fixed now:** early ranking is **PREDICTIVE** at a rung when rho >= 0.80, the
  interval's lower end >= 0.50, and the top-third check keeps the final best cell; otherwise
  NOT_PREDICTIVE; INCOMPLETE when fewer than max(8, 80% of the block's runs) have that rung. The
  answer reported first is the **earliest predictive rung for Y_base -> Y, pooled 20M**.
- Discipline: rung files are analysed with their block's final summaries (`--partial` labels the
  output DESCRIPTIVE, PARTIAL). Interim reads remain failure monitoring.
- **Predictions (descriptive, not tested):** the panel losses rank early (predictive by 25%);
  Carry-5 ranks later than the losses, because carry-over forms late and SFT reorders models; the
  deepest 20M shapes start slowest, so a stopping rule at 5-10% would have cut them.

*(c) A successive-halving arm, EXPLORATORY, not confirmatory* (`tools/colab_chat/tiny_chat_asha.json`;
configurations and costs in `asha_plan.json`).
- **Space:** 27 configurations drawn with seed 20261023: 20M non-embedding shapes at depths 2, 3,
  4, 6, 8, 12 and 16 (`shape_for`; E1 covers L32 at 3-4x the cost per step), peak lr log-uniform in
  [2.5e-4, 4e-3], warmup 2, 5 or 10% of the budget, dialogue share one of E5's levels (0, 17, 35,
  70%), seed 1000 + index. Everything else as section 1.
- **Rounds** (synchronous successive halving, eta = 3): 27 configurations x 1,000 steps, the top 9
  x 3,000, the top 3 x 10,000 (the full budget). Each round trains from scratch to its budget with
  its own warmup and cosine to 0.1x, so the finalists are full runs comparable with the E cells and
  a replaced VM never strands a promotion. The finalists get the E cells' post job (the same SFT,
  Carry-5 with self history, the panels).
- **Promotion rule, fixed now** (`tiny_chat_run.py` `promote_rank`): rank the previous round's
  candidates by **base-model Carry-5 acc_gold** at the end of their budget (higher first), ties by
  the dialogue panel loss (lower first), then config id; a diverged or failed run ranks last and is
  never promoted; promote the top 9, then the top 3. Promotion runs on the VM; a candidate not
  promoted ends as a FAILED job whose log says "not promoted".
- **Kept:** every run's `rung.json` (final validation loss, a 10-point validation curve, panels,
  base-model Carry-5) and its `events.jsonl` (the training curve) come home, killed or not. Killed
  runs' weights are not kept.
- **Budget cap: 20 L4 VM-hours** (about 96 compute units), enforced as the session's
  `--max-hours 20` (one VM at a time). The throughput model's worst case is 19.5 VM-hours (the
  dearest 9 and 3 promoted), about 13 expected. Whatever is unfinished at the cap is reported as such.
- **Claims:** descriptive only. The arm can show which configurations learn fast at 1,000 and 3,000
  steps and where the finalists' Y falls against the E cells; it tests no hypothesis (one seed per
  configuration, selection on the metric, configurations that interact with budget). The promotion
  metric favours dialogue-rich mixes by construction (base-model Carry-5 reads the chat template),
  so a dialogue-heavy finalist is not evidence for H5.

**A1.5 Interfaces, as changed before any run.** `tiny_chat_data.py`: `--sources`,
`--require-pinned`, `--measure-only`, Carry-5 in the decontamination. `jobs_vm.py`:
`snapshot_steps`, `rung_of`, light slots, `after_terminal`, `promoted_by`. `tiny_chat_run.py`:
`rung`, `asha_train`, `asha_promote`; `post` records `final_rung`. `make_jobs.py`: validation every
500 steps, the rung jobs, the 13 sources, `asha_plan.json` and `tiny_chat_asha.json`.
`tools/colab_chat/tiny_chat_asha.py`: the home side of the arm (`drive` keeps
`ART/asha_state.json`, every result and decision already home, current for the runner's upload, so
a replaced VM can still promote). `analyze.py`: `rung_analysis`, `asha_summary`.

**A1.6 Compute and launch** (section 12's commands with these files; the main session launches).

| block | L4 VM-hours (throughput model) | session | depends on |
|---|---|---|---|
| Stage 0, 20M | 5 | `pk-tc-s0` | this commit |
| Stage 0, 60M | 5 | `pk-tc-s0-60m` | this commit |
| successive halving (exploratory) | 13 expected, cap 20 | `pk-tc-asha` | this commit |
| E5 | 17 | `pk-tc-e5` | Stage 0 20M chosen, committed, pushed |
| E1 20M | 30 | `pk-tc-e1` | the same |
| E1 60M | 36 | `pk-tc-e1-60m` | Stage 0 60M chosen, committed, pushed |
| **total** | **about 106-113** | | |

Rung jobs add about 3 minutes of light-slot work per rung (144 rungs), beside the training jobs.
At about 4.8 compute units an hour the total is about 510-540 units, most of a month's 600 beside
the next base model's sessions; Stage 0's probes replace the model's numbers, and section 8 rule 5
applies to them. With at most four Colab sessions at once: Stage 0 (both sizes) and the
successive-halving arm first, beside `pk-base`; E5 and E1 when the two Stage 0 sessions end; E1 at
60M when the next session ends.

```
U="--upload vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
R="python3 -u tools/colab_chat/runner_multi.py"
# this commit (SHA1):
$R --commit SHA1 --jobs tools/colab_chat/tiny_chat_s0.json --name pk-tc-s0 --art /mnt/c/ml_artifacts/tiny_chat/s0 \
   --state-dir /home/jonat/pk_tc_s0_runner --max-hours 23.5 $U
$R --commit SHA1 --jobs tools/colab_chat/tiny_chat_s0_60m.json --name pk-tc-s0-60m \
   --art /mnt/c/ml_artifacts/tiny_chat/s0_60m --state-dir /home/jonat/pk_tc_s0_60m_runner --max-hours 23.5 $U
python3 -u tools/colab_chat/tiny_chat_asha.py drive --art /mnt/c/ml_artifacts/tiny_chat/asha \
   --state-dir /home/jonat/pk_tc_asha_runner -- $R --commit SHA1 --jobs tools/colab_chat/tiny_chat_asha.json \
   --name pk-tc-asha --art /mnt/c/ml_artifacts/tiny_chat/asha --state-dir /home/jonat/pk_tc_asha_runner \
   --max-hours 20 $U --upload asha_state.json=/mnt/c/ml_artifacts/tiny_chat/asha/asha_state.json
# after each Stage 0, at home (60M adds its shapes to the same file):
python3 tools/colab_chat/tiny_chat_run.py choose --art /mnt/c/ml_artifacts/tiny_chat/s0 \
   --out experiments/tiny_chat_2026-10/stage0/lr_choice.json
python3 tools/colab_chat/tiny_chat_run.py choose --art /mnt/c/ml_artifacts/tiny_chat/s0_60m \
   --out experiments/tiny_chat_2026-10/stage0/lr_choice.json
python3 experiments/tiny_chat_2026-10/make_jobs.py   # commit lr_choice.json and the jobs files, push: SHA2
$R --commit SHA2 --jobs tools/colab_chat/tiny_chat_e5.json --name pk-tc-e5 --art /mnt/c/ml_artifacts/tiny_chat/e5 \
   --state-dir /home/jonat/pk_tc_e5_runner --max-hours 23.5 $U
$R --commit SHA2 --jobs tools/colab_chat/tiny_chat_e1.json --name pk-tc-e1 --art /mnt/c/ml_artifacts/tiny_chat/e1 \
   --state-dir /home/jonat/pk_tc_e1_runner --max-hours 23.5 $U
$R --commit SHA2 --jobs tools/colab_chat/tiny_chat_e1_60m.json --name pk-tc-e1-60m \
   --art /mnt/c/ml_artifacts/tiny_chat/e1_60m --state-dir /home/jonat/pk_tc_e1_60m_runner --max-hours 23.5 $U
python experiments/tiny_chat_2026-10/analyze.py --art C:/ml_artifacts/tiny_chat --out experiments/tiny_chat_2026-10/analysis.json
```

### Amendment 2 (23 Oct 2026, after Stage 0's first attempt, before any E5, E1 or successive-halving run): failed screens, one GPU job at a time, the measured budget, and the instruction family

Written after both Stage 0 sessions came home (`pk-tc-s0`, 07:10-09:56, and `pk-tc-s0-60m`, 07:10-08:06, from
amendment 1's commit) and after the Carry-5 sanity run on the live model (`experiments/carry5_2026-10-23/README.md`),
before any E5, E1 or successive-halving job ran and before this programme produced any Carry-5 number. Sections 0-12
and amendment 1 stand except where this amendment says otherwise. Receipts: `stage0/attempt1/` (every Stage 0 job's
`summary.json`, the data ledgers, both runner logs).

**A2.1 Stage 0, attempt 1.** Probes, each shape alone on the L4 (300 steps; steady rate from step 50; 16,384 tokens
per step; peak memory is the whole device's, from `nvidia-smi`):

| shape | tokens/s | s/step | peak GPU MiB | | shape | tokens/s | s/step | peak GPU MiB |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 20m_L2 | 33,942 | 0.483 | 8,668 | | 60m_L4 | 13,803 | 1.187 | 13,940 |
| 20m_L4 | 32,202 | 0.509 | 9,214 | | 60m_L8 | 13,799 | 1.187 | 15,906 |
| 20m_L8 | 27,948 | 0.586 | 10,492 | | 60m_L16 | 12,517 | 1.309 | 10,628 |
| 20m_L16 | 25,383 | 0.645 | 6,660 | | 60m_L32 | 10,308 | 1.590 | 13,244 |
| 20m_L32 | 21,856 | 0.750 | 8,132 | | | | | |

- All nine probes finished. Per token, 20m_L32 takes **1.55x** 20m_L2's time (60M: L32 1.34x L4). Section 6's
  throughput prediction (at least 3x) does not hold: depth costs far less on this GPU than section 10's per-layer
  model assumed. S10 (Y per L4-hour) is reported as registered.
- **Screens: 5 of 27 finished, 22 failed** (exit code 1). Finished, final validation loss (128 windows): 20m_L2 at
  5e-4, 4.439; 20m_L4 at 5e-4, 4.338; 20m_L16 at 1e-3, 4.241, and at 2e-3, 4.476; 20m_L32 at 2e-3, 4.282. Failed:
  20m_L2 and 20m_L4 at 1e-3 and 2e-3, all three 20m_L8 screens, 20m_L16 at 5e-4, 20m_L32 at 5e-4 and 1e-3, and all
  twelve 60M screens.
- **None of the 22 is a divergence** under section 4's definition. No logged training loss is non-finite. Five
  failed screens logged losses before stopping, falling from 9.74-9.77 to 4.65-5.85 (one stopped near its last
  step; one after its first step); seventeen stopped before their first step.
- **Likely cause: GPU memory, with three jobs on one L4** (the Stage 0 files' `max_parallel` 3). Each mtstudio
  process caches freed device blocks up to three quarters of the device (`src/device_pool.cuh`) and, when an
  allocation fails, returns only its own cache. Alone, the 20M shapes held 6,660-10,492 MiB and the 60M shapes
  10,628-15,906 MiB of the L4's ~23,000. Each 20M crash before a first step came as a third job started while the
  three jobs' solo footprints summed past the device (24,396, 25,088 and 26,550 MiB); no two 60M footprints fit
  together, and all twelve 60M screens failed within one runner poll of the probes ending. mtstudio's own log stayed
  on the VM (only `summary.json` came home), so the cause is inferred from these facts, not read from a log; screen and
  probe summaries now carry the log's tail on any failure.
- **Sharing bought no throughput.** The screens' own timings ran at 0.30-0.38 of their shapes' solo rates with two
  others beside them: 0.93-1.05 jobs' worth per VM in all, against section 10's assumed 2.5. One job saturates the L4.
- The registered `choose` refuses for both sizes (no shape has three usable screens; 20m_L8, E5's shape, has none),
  so no learning rate is chosen and `stage0/lr_choice.json` does not exist. The E gate jobs refuse until it does.

**A2.2 Failed screens are rerun (section 4, clarified).** Section 4's rule knows screens that diverged and screens
that did not; a crashed process is neither, and says nothing about its rate.
- A screen is FINISHED (exit 0, a finite final validation loss not above its first training loss), DIVERGED
  (section 4's definition: a non-finite training loss, or a finished run whose final validation loss is missing,
  non-finite or above its first training loss) or FAILED (a non-zero exit with finite losses)
  (`tiny_chat_run.py screen_status`).
- Every FAILED screen is rerun once, unchanged (the same spec, lr, seed 101, 1,500 steps and 35% mix, rebuilt on the
  VM from the pinned sources), one GPU job at a time: `tools/colab_chat/tiny_chat_s0_rerun.json` (the ten 20M
  failures) and `tiny_chat_s0_60m_rerun.json` (the twelve 60M screens). The five finished screens stand: a run's
  arithmetic does not depend on the processes beside it.
- `choose` reads attempt 1 and then the rerun (`--art` given twice) and takes each screen from the one folder that
  holds it FINISHED or DIVERGED; a finished or diverged screen found twice is refused, so no choice is ever made among
  reruns. A shape is chosen only when all three of its screens are FINISHED or DIVERGED, by section 4's rule as written
  (lowest final validation loss; within 0.01 nats, the smaller lr).
- A screen that FAILS again, alone on the GPU, is reported with its log tail; its shape stays unchosen until a further
  amendment, written before any E run uses that shape.
- Nothing else changes: the token budget, the grid, the shapes, the screen recipe and every step count stand. This
  amendment changes no learning rate.

**A2.3 One GPU job at a time** (section 4: Stage 0 may change the schedule and the concurrency). Every jobs file
written from now on (the reruns, E5, E1, E1 at 60M and the successive-halving arm) has `max_parallel` 1:
pretraining, post, screen and successive-halving training jobs run one after another on a VM. Light jobs (rungs,
promotion steps) keep their own slot beside the GPU job; rung jobs run with `MICROTORCH_POOL_MB=2048`, so their cache
stays small. By A2.1 this costs no VM-hours; each block takes longer on the clock, and blocks run as parallel
sessions (at most four Colab sessions in all). The Stage 0 files as launched (`tiny_chat_s0.json`,
`tiny_chat_s0_60m.json`) are kept as they ran.

**A2.4 The budget from the probes** (section 10's model replaced, as it said Stage 0 would; section 8 rule 5).
`make_jobs.py` reads the probes in `stage0/attempt1/`. A run's pretraining and SFT run at its shape's solo rate (the
successive-halving depths 3, 6 and 12 interpolated in log2 depth between probed neighbours), one job per VM, plus
allowances not yet measured: 0.30 h per post job beyond its SFT (panels, Carry-5 with generation, base-model
Carry-5), 0.08 h per rung job, and 0.25 h per session for the gate and data jobs (attempt 1: 13-19 minutes).
Compute units at 4.8 per L4 hour (section 10's figure).

| block | runs | L4 VM-hours | compute units |
|---|---:|---:|---:|
| Stage 0 20M, attempt 1 (as run) | 20 | 2.8 | 13 |
| Stage 0 20M, rerun of the 10 failed screens | 10 | 2.7 | 13 |
| E5 (4 levels x 3 seeds at 20m_L8) | 12 | 30.2 | 145 |
| E1 20M (L2, L4, L16, L32 x 3 seeds) | 12 | 30.6 | 147 |
| **core (section 8 rule 5)** | | **66.2** | **318** |
| Stage 0 60M, attempt 1 (as run) | 16 | 0.9 | 4 |
| Stage 0 60M, rerun of the 12 screens | 12 | 6.8 | 33 |
| E1 60M (4 shapes x 3 seeds) | 12 | 58.2 | 279 |
| successive halving (worst case; cap 20) | 27 | 19.2 | 92 |
| everything | | 151.4 | 727 |

- Per run (pretraining + SFT + allowances, VM-hours): 20m_L2 2.16, L4 2.25, L8 2.49, L16 2.68, L32 3.01; 60m_L4
  4.41, L8 4.41, L16 4.80, L32 5.70.
- **Rule 5 is not triggered: the core (Stage 0 at 20M, both attempts, E5 and E1 at 20M) is 66.2 L4 VM-hours, under
  80.** Without the evaluation allowances it is 51.3; it reaches 80 only if a run's post and rung jobs together take
  more than 1.20 h (allowed: 0.62 h). The first E5 post jobs measure it: if the first three E5 runs' post and rung
  jobs average more than 1.20 h per run, rule 5 applies then, before E1 launches.
- The whole programme, 151 VM-hours (727 units), is more than one month's 600 units; E1 at 60M is the largest block.
- E5 and E1 take about 30 VM-hours each and E1 at 60M about 58, past the runner's `--max-hours 23.5`: each session
  ends at least once mid-block, and a run in flight resumes from its last relay with a cold optimiser (section 8
  rule 4: recorded per run, and the primary contrasts reported without resumed runs as well).

**A2.5 The instruction family and the question-alone floor** (section 3; before any Carry-5 number from this
programme). On the live model (`chat_sftfinal_v3`, 14.6M; `experiments/carry5_2026-10-23/README.md`) the instruction
family's compliant answer is always the shortest of the four candidates, so raw forced choice rewards a length prior:
gold 0.854, twin 0.875 and **the question alone 0.826**, while the model complied in 0 of 21 generated replies. On the
memory families it was at chance in every mode, the twin included.
- **(a) Primary metric, amended: Y = forced-choice accuracy (raw log-prob, `acc`) with SELF history minus that on
  the single-turn twins, over the recall, correction, topic-return and knowledge families (888 of the 1,032 items).
  The instruction family (144 items) is out of Y and of every contrast.** Section 3's all-items Y is reported beside
  it (`Y_all`), descriptively. Instruction-following is reported separately and descriptively as generation-scored
  compliance (carry5_eval's machine check on the greedy target reply: one word, one to five words, yes or no, one
  number), with self history and in the twin (S12, new). Its forced-choice accuracy appears in the family table,
  marked as length-confounded.
- **(b) The question alone** (carry5_eval's `noctx` mode: the target question with no statement and no history) is
  reported beside the twin for every family and every cell (S5), on the SFT model and at every rung.
- **The validity gate, restated.** A cell's Y is interpreted only if, on the cell's means over seeds, **(i)** twin
  accuracy over the primary families is at least chance + 10 points (0.35), and **(ii)** twin accuracy minus
  question-alone accuracy over the context families (recall, correction, topic return: 828 items) is **at least 10
  points (0.10)**. Knowledge is left out of (ii) because its twin is its question alone, word for word. A contrast
  that uses a cell failing either part is FLOOR, as before. Why this margin: the set's balance puts the question
  alone near chance (0.251 on the context families for the live model), so (ii) binds where a model's prior over the
  candidates lifts its twin above chance without its reading the statement; 10 points is the same margin as (i) and
  about four item-level standard errors of the paired difference at 828 items. The live model fails both parts:
  twin 0.260 over the primary families, and twin minus question alone 0.002 on the context families.
- **(c) Everything built from Y or the twin uses the same items:** S1-S3 (multi-turn and twin accuracy, gold-history
  Y), H1(b)'s twin slope and the data bar's twin condition over the primary families; the rung predictors Y_base and
  acc_gold (amendment 1) and the successive-halving promotion metric acc_gold over the primary families, with the
  all-items values kept beside them. No successive-halving job has run, so its promotion rule changes before its
  data.
- SESOIs, hypotheses, directions, decision rules and the extension rule are unchanged.

**A2.6 Interfaces.** `tiny_chat_run.py`: `carry5_pools` (exact pools from carry5_eval's `items.jsonl`: per mode over
the primary and the context families, per family, and instruction compliance by generation; checked against the
sanity receipts), `carry5_read` (requires the item records), the amended `primary` with `primary_all_items`,
`rung_metrics` over the primary families (all-items values and the floor's quantities beside them), `screen_status`,
the screen summary's `status` and `log_tail`, `choose --art` repeated. `make_jobs.py`: the probes, `failed_screens`,
the two rerun files, `max_parallel` 1, the light jobs' cache cap, the measured budget (`budget_rows`). `analyze.py`:
the amended Y and validity gate, `Y_all`, the family table with the question alone, instruction compliance. Tests:
`tests/test_tiny_chat_analyze.py`, `test_tiny_chat_design.py`, `test_tiny_chat_pipeline.py`.

**A2.7 Launch** (the main session launches). SHA2 = this commit: the two reruns, and the successive-halving arm, which
needs no Stage 0 choice. E5 and E1 launch from the commit that adds the 20M choice (SHA3); E1 at 60M from the commit
that adds the 60M choice (SHA3 if both are home, else SHA4).

```
U="--upload vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
R="python3 -u tools/colab_chat/runner_multi.py"
# SHA2 (this commit):
$R --commit SHA2 --jobs tools/colab_chat/tiny_chat_s0_rerun.json --name pk-tc-s0r --art /mnt/c/ml_artifacts/tiny_chat/s0_rerun \
   --state-dir /home/jonat/pk_tc_s0r_runner --max-hours 23.5 $U
$R --commit SHA2 --jobs tools/colab_chat/tiny_chat_s0_60m_rerun.json --name pk-tc-s0r-60m \
   --art /mnt/c/ml_artifacts/tiny_chat/s0_60m_rerun --state-dir /home/jonat/pk_tc_s0r_60m_runner --max-hours 23.5 $U
python3 -u tools/colab_chat/tiny_chat_asha.py drive --art /mnt/c/ml_artifacts/tiny_chat/asha \
   --state-dir /home/jonat/pk_tc_asha_runner -- $R --commit SHA2 --jobs tools/colab_chat/tiny_chat_asha.json \
   --name pk-tc-asha --art /mnt/c/ml_artifacts/tiny_chat/asha --state-dir /home/jonat/pk_tc_asha_runner \
   --max-hours 20 $U --upload asha_state.json=/mnt/c/ml_artifacts/tiny_chat/asha/asha_state.json
# after each rerun, at home (the 60M call adds its shapes to the same file):
python3 tools/colab_chat/tiny_chat_run.py choose --art /mnt/c/ml_artifacts/tiny_chat/s0 \
   --art /mnt/c/ml_artifacts/tiny_chat/s0_rerun --out experiments/tiny_chat_2026-10/stage0/lr_choice.json
python3 tools/colab_chat/tiny_chat_run.py choose --art /mnt/c/ml_artifacts/tiny_chat/s0_60m \
   --art /mnt/c/ml_artifacts/tiny_chat/s0_60m_rerun --out experiments/tiny_chat_2026-10/stage0/lr_choice.json
python3 experiments/tiny_chat_2026-10/make_jobs.py   # commit lr_choice.json and the jobs files, push: SHA3
$R --commit SHA3 --jobs tools/colab_chat/tiny_chat_e5.json --name pk-tc-e5 --art /mnt/c/ml_artifacts/tiny_chat/e5 \
   --state-dir /home/jonat/pk_tc_e5_runner --max-hours 23.5 $U
$R --commit SHA3 --jobs tools/colab_chat/tiny_chat_e1.json --name pk-tc-e1 --art /mnt/c/ml_artifacts/tiny_chat/e1 \
   --state-dir /home/jonat/pk_tc_e1_runner --max-hours 23.5 $U
$R --commit SHA3 --jobs tools/colab_chat/tiny_chat_e1_60m.json --name pk-tc-e1-60m \
   --art /mnt/c/ml_artifacts/tiny_chat/e1_60m --state-dir /home/jonat/pk_tc_e1_60m_runner --max-hours 23.5 $U
```

### Amendment 3 (23 Oct 2026, before any E5, E1, successive-halving or recursion run): the floor pilot, the branch it decides, and the recursion arm

Written after Jonathan's go of 23 Oct 2026 (E5 and E1 at 20M, the 60M stage, E5w on its condition, the
successive-halving arm after `pk-base`, a recursion arm in E1; complete autonomy, paced across the month
boundary). It was written while the Stage 0 reruns were running (`pk-tc-s0r`, `pk-tc-s0r-60m`), before any of their screens
came home, before any E5, E1, successive-halving or recursion job ran, and before this programme produced
any Carry-5 number. Sections 0-12 and amendments 1-2 stand except where this amendment says otherwise. The
recursion design follows `RECURSION_LITCHECK.md` (this folder, paperkiln 1602a3ee).

**A3.1 The floor pilot.** The live 14.6M model fails both parts of the validity gate (A2.5). If 20m_L8
lands on the floor at this budget, every E5 and E1 contrast at 20M ends FLOOR, and those blocks' 61 VM-hours
buy no verdict. One run settles the question before that money is spent.
- **The run:** one complete 20m_L8 run at E1's level (35%), by section 1's recipe without change: 10,000
  pretraining steps, the same SFT, the post job (panels, Carry-5 with self history, base-model Carry-5) and the
  four rung jobs (5/10/25/50%) logged as for every E run. **Seed 99**, declared here before it runs; it is
  outside 1-5 and is not the screen seed 101. File: `tools/colab_chat/tiny_chat_pilot.json` (the data job at
  35% only, the gate, one pretraining job, four light rung jobs with the 2 GB cache cap, the post job and a
  pick); one GPU job per VM; about 2.7 L4 VM-hours (13 units).
- **Excluded from all analysis.** It lives in `ART/pilot`, which no analysis block reads; `analyze.py` refuses
  a seed-99 summary found inside an E block. It never counts as a seed, a cell, a rung point, an extension
  seed or data-bar evidence. Its verdict file holds only the gate's quantities (twin accuracy over the
  primary families, chance, twin and question-alone accuracy over the context families). Its Y and
  multi-turn accuracies are not read or reported in any result of this programme; they remain in its
  `summary.json` on disk.
- **Its learning rate, fixed now** (`tiny_chat_run.py pilot_lr`; `make_jobs.py pilot_rate`):
  (0) Stage 0's choice for 20m_L8 when `stage0/lr_choice.json` holds it. Otherwise `pilot_lr` writes
  `stage0/pilot_lr.json`: when 20m_L8's three screens are all FINISHED or DIVERGED, section 4's rule exactly
  as `choose` applies it (so `choose` refusing the whole file over another shape does not hold the pilot
  up); if all three diverged, 2.5e-4, the first rate of section 4's downward extension. Otherwise the pilot
  waits until the `pk-tc-s0r` session has ended (its runner exits when every job is terminal or at
  `--max-hours` 23.5, so the wait is bounded), and then takes the first of these that applies:
  (F1) section 4's rule over 20m_L8's FINISHED screens; (F2) the smaller of the rates the same rule gives
  20m_L4 and 20m_L16 (half and twice the depth) over their FINISHED screens; (F3) the smallest rate among the
  FINISHED 20M screens. If no 20M screen finished, the pilot has no rate and Jonathan decides.
- The pilot's rate never enters `lr_choice.json`. E5 and E1 still train on Stage 0's choice (A2.2: an L8
  screen that fails again alone leaves 20m_L8 unchosen until a further amendment). If the choice later
  differs from a fallback rate the pilot ran at, the verdict stands and the pilot is not rerun.
- **The verdict, fixed now** (`analyze.py --art ART --pilot-verdict pilot/verdict.json`, committed before
  any E5 or E1 job): the A2.5 gate on the pilot's SFT model. **PASS** if (i) twin accuracy over the primary
  families is at least chance + 10 points (0.35) **and** (ii) twin minus question-alone accuracy over the
  context families is at least 10 points; **FAIL** otherwise. With one run, the cell mean over seeds is
  that run's value.
- A pilot interrupted by a reclaim resumes as section 8 rule 4 says. If its post job fails twice, or no
  verdict can be computed, there is no default: Jonathan decides. The E5 and E1 gate jobs at 20M refuse
  unless `pilot/verdict.json` says PASS.

**A3.2 What runs after the pilot.**
- **PASS:** E5 and E1 at 20M launch as registered (`tiny_chat_e5.json`, `tiny_chat_e1.json`), from the commit
  holding `lr_choice.json` with the 20M shapes and the verdict. E1 at 60M, E5w at 20m_L8 (on its condition),
  the successive-halving arm (after `pk-base`) and the recursion arm at 20M (A3.3) follow as registered.
  `tiny_chat_e5_60m.json` is not run.
- **FAIL: the primary size moves to 60M before any 20M E5 or E1 spend.**
  - *What runs:* E1 at 60M as registered (`tiny_chat_e1_60m.json`: L4, L8, L16, L32 x 3 seeds on its own
    Stage 0 choice). **E5 at 60M** (`tiny_chat_e5_60m.json`): 60m_L8 at 0, 17 and 70% x 3 seeds (9 runs).
    Its 35% cell is E1 at 60M's L8 cell (the same runs), just as E1's L8 cell at 20M is E5's 35% cell. Its gate
    needs the verdict FAIL and 60m_L8's choice. Section 5 applies with 60m_L8 in place of 20m_L8: the
    hypotheses, SESOIs, data bar, contrasts and extension rule. E5w, on its condition, moves to 60m_L8. The
    60M sign prediction (section 6) has no 20M sign to compare with and is reported as not testable.
  - *The 20M blocks:* E5 and E1 at 20M are not run. Their files stay as generated and their gates refuse. No
    20M E contrast is reported. The pilot's FAIL is reported as one run's observation at this budget (the
    gate's quantities), not as a test. Running them later needs a new amendment.
  - The successive-halving arm (20M) is held: its promotion metric would rank configurations near chance. If
    it is already running, it finishes and every finalist is reported with the gate's quantities beside it.
  - The recursion arm: A3.3, last item.
  - No second pilot at 60M. The 60M cells meet the validity gate as registered, and a floor cell is FLOOR.
- **Budget** (`make_jobs.py budget_a3`; amendment 2's measured model): PASS branch 154.2 L4 VM-hours (740
  units: amendment 2's 727 and the pilot's 13); FAIL branch 114.1 VM-hours (548 units: plus the pilot and
  E5 at 60M, 40.0; minus E5 and E1 at 20M and the held arm). Both exclude the recursion arm.

**A3.3 The recursion arm (E1-R), step 1: fixed recursion.** Jonathan's question: does a shared block applied
several times carry facts across turns better than plain depth or width at E1's ~20M non-embedding
parameters?
- **Mechanism:** mtstudio's `arch.custom.recursion` = {prelude P, block B, repeats r, coda C, schedule
  `fixed`} stores P + B + C distinct layers and applies the prelude once, the block r times and the coda once.
  Each application has its own KV cache, backprop runs through every application, and there is no input
  injection. At the time of writing its owner is adding it to mtstudio. The arm's jobs are generated from
  the commit that holds it.
- **Cells,** all at 20m_L4's shape: d 640, 10 heads, d_ff 1,760; 20.08M non-embedding, 10.49M embedding,
  30.56M total. Every cell trains on the 35% mix with section 1's budget, SFT, Carry-5, panels and rungs,
  on seeds 1-3.

  | cell | recursion (P, B x r, C) | effective depth | FLOPs/token (training) | plain twin in E1 |
  |---|---|---:|---:|---|
  | (r = 1) | E1's 20m_L4 cell, no new runs | 4 | 0.19G | - |
  | R2 | 0, 4 x 2, 0 | 8 | 0.32G | 20m_L8 (0.17G) |
  | R4 | 0, 4 x 4, 0 | 16 | 0.58G | 20m_L16 (0.17G) |
  | R8 | 0, 4 x 8, 0 (conditional, below) | 32 | 1.09G | 20m_L32 (0.16G) |
  | MC7 | 1, 2 x 7, 1 (Middle-Cycle topology) | 16 | 0.58G | 20m_L16; R4 |

- **Training:** section 1 except the learning rate.
  - Each cell gets its own Stage 0: a 300-step probe alone on the GPU (throughput, peak memory) and screens
    at **{2.5e-4, 5e-4, 1e-3, 2e-3}**, 1,500 steps, seed 101, on the 35% mix. The extra low rate is there
    because shared weights sum their gradients over r applications, and Ouro and Huginn needed lower rates.
  - The rate is chosen by section 4's rule with A2.2's statuses. A cell whose four screens all diverge is
    dropped and reported.
  - Batch 16 x 2 at effective depth 16 or more (R4, R8, MC7); 32 x 1 at R2.
  - None of: input injection, per-step embeddings, LoRA relaxation, deep supervision or EMA (all break
    exact matching or are untested here).
- **Matching and confounds.** The cells are matched on non-embedding parameters only, as E1 is. Recursion
  multiplies compute (table), so **no compute-matched verdict is registered**. FLOPs per token and Y per
  L4-hour (S10) are reported per cell, and the 60M cells nearest in FLOPs are shown descriptively. **The
  embedding confound is the reverse of E1's:** the recursion cells carry 10.49M embedding against the twins'
  7.34, 5.24 and 3.15M, so a recursion win over its twin is ambiguous between recursion and lexical
  capacity, as a width win is in E1.
- **Hypotheses** (`decide`, three seeds, verdicts as section 7; the validity gate applies to every recursion
  cell, and a contrast that uses a FLOOR cell is FLOOR):
  - **HR1 (registered prediction; Saunshi et al. 2502.17416, MobileLLM-LS, GPT-BERT 2609.09691): Y rises
    with passes.** The within-seed OLS slope of Y on log2(r) over r = 1, 2, 4 (E1's L4, R2, R4) is > 0; SESOI
    1 point per doubling. With R8, the slope over r = 1, 2, 4, 8 is secondary.
    **HR1-knowledge** (Saunshi; Ouro 2510.25741: looping adds no stored knowledge): the FineWeb panel loss
    does not improve with passes. Its slope on log2(r) over the same cells is predicted not SUPPORTED in the
    falling direction (EQUIVALENT within 0.01 nats per doubling, or rising). It is falsified by a SUPPORTED
    fall.
  - **HR2, recursion against plain depth at equal parameters and effective depth.** Per seed, the mean of
    Y(R2) - Y(20m_L8) and Y(R4) - Y(20m_L16); with R8, the three-pair mean (adding Y(R8) - Y(20m_L32)) is
    secondary. SESOI 3 points. *Literature-led prediction* (MoR at 135M, ALBERT, GPT-BERT entity tracking):
    the difference is not above the SESOI (EQUIVALENT or CONTRADICTED), falsified by SUPPORTED. *Jonathan's
    view, registered as the alternative HR2-J:* recursion helps sequential carry-over, so the difference is
    SUPPORTED; HR2-J is falsified by EQUIVALENT or CONTRADICTED. NOT_RESOLVED decides neither.
  - **HR3, inductive bias (Saunshi's isoplot): at equal FineWeb panel loss, recursion cells have higher Y.**
    Per seed, the OLS line of Y on the FineWeb panel loss through E1's five plain 20M cells. Each of R2 and
    R4 (R8 secondary) gets its residual from that line at its own panel loss, and the per-seed mean residual is
    > 0; SESOI 3 points. A cell whose panel loss falls outside the plain cells' range is flagged as an
    extrapolation.
  - **HR4 (descriptive): diminishing returns.** The quadratic contrast of Y over log2(r) at r = 1, 2, 4, 8
    is negative. It is computed only when R8 runs, with no verdict.
  - Secondary, no prediction: MC7 - R4 (topology at equal parameters, effective depth and FLOPs) and
    MC7 - 20m_L16, labelled SECONDARY.
- **R8 runs only if 4 passes beats 2 (registered; halves the cost risk).** R8's screens and runs launch only
  if R2 and R4 both pass the validity gate on their three seeds and the mean over seeds of Y(R4) - Y(R2) is at
  least 1 point (the SESOI per doubling). The check is read at home once both cells' post jobs are home,
  written to `rec_decision.json` and committed before any R8 job. It reads only that number and the two
  cells' gates.
- **Step 2, Mixture-of-Recursions-style adaptive recursion (Bae et al. 2507.10524): only if HR1 is
  SUPPORTED.** Design: MC7's Middle-Cycle topology with per-token expert-choice routing and MoR's auxiliary
  loss, a new mtstudio schedule. **Registered measurement (not a run date),** on the SFT model, teacher-forced
  with gold history over Carry-5's primary families, per seed:
  - (m1) mean recursions allocated to answer tokens (the gold value's tokens in the target reply) minus the
    mean over the other assistant tokens;
  - (m2) the OLS slope of answer-token recursions on fact distance (Carry-5's turns between plant and query);
  - (m3) the same allocations on the single-turn twins (distance 0), beside them.
  These results are descriptive. The literature check expects MoR to trail fixed recursion at this size (MoR
  trailed vanilla at 135M). The router's capacity, auxiliary-loss weight and maximum recursion count take
  MoR's best 135M setting and are recorded in a dated note before any step-2 run.
- **Failure rules:** section 8 (divergence reruns at half the rate; resumes recorded).
- **Cost** (the literature check's extrapolation from A2.1's probes; the arm's probes replace it).
  - Pretraining per run: about 2.5, 4 and 7 h at r = 2, 4 and 8. With SFT and evaluations, about 52 L4
    VM-hours for R2, R4 and R8 x 3 seeds, and about 15 for MC7.
  - Screens: about 6 VM-hours, plus about 4 for R8's. That makes about 48 VM-hours without R8 and about 78
    with it.
  - Jonathan's go approves the arm at about 65 VM-hours of runs. If the probes put the runs (R8 included)
    above 80 VM-hours, Jonathan decides before they launch, in section 8 rule 5's form.
- **Files:** `tiny_chat_s0_rec.json` (probes and screens; R8's screens are a separate file, made on the R8
  rule) and `tiny_chat_e1_rec.json`. Both are written by `make_jobs.py` from the commit that holds mtstudio's
  recursion. Their gates need the pilot's PASS and each cell's choice. `analyze.py`'s E1-R block (HR1-HR4 and
  the R8 rule) lands with its tests before any run of the arm. None of this changes a rule above.
- **On the pilot's FAIL branch** the arm is not run at 20M, because its twins and its r = 1 cell are 20M E1
  cells that then do not run. Its 60M transfer recurses the whole 60m_L4 stack (d 1,088) 2 and 4 times,
  against the twins 60m_L8 and 60m_L16, with r = 1 as E1 at 60M's L4 cell. R8 and MC7 are not registered at
  60M. The rules are those above with the 60M cells. It runs if its probes put it within 65 L4 VM-hours
  (estimated at about 75: r times 60m_L4's cost per token); otherwise Jonathan decides.

**A3.4 Interfaces.** `tiny_chat_run.py`: `pilot_lr`, `screen_rule`, and the gate's `--pilot-lr` (the pilot's
rate, used only for a shape that the choice lacks), `--pilot-verdict` and `--need-pilot PASS|FAIL`.
`make_jobs.py`: `PILOT_SEED`, `pilot_rate`, `pilot_doc`, `e5_60m_cells` and `budget_a3`;
`tiny_chat_pilot.json` and `tiny_chat_e5_60m.json`; the E5 and E1 gates at 20M need PASS. `analyze.py`:
`pilot_verdict` and `--pilot-verdict`, and `load` refuses the pilot seed. Tests: `test_tiny_chat_design.py`,
`test_tiny_chat_pipeline.py`, `test_tiny_chat_analyze.py`.

**A3.5 Launch** (the main session launches). This commit holds the machinery. The pilot launches from the
commit that adds its rate (SHA_P); E5 and E1 at 20M, or E5 at 60M, launch from the commit that adds the
verdict (SHA_V).

```
U="--upload vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
R="python3 -u tools/colab_chat/runner_multi.py"
# once pk-tc-s0r has brought 20m_L8's screens home (A2.7's choose first; it may refuse over other shapes):
python3 tools/colab_chat/tiny_chat_run.py choose --art /mnt/c/ml_artifacts/tiny_chat/s0 \
   --art /mnt/c/ml_artifacts/tiny_chat/s0_rerun --out experiments/tiny_chat_2026-10/stage0/lr_choice.json
# only if lr_choice.json lacks 20m_L8 (add --session-ended once the pk-tc-s0r runner has exited):
python3 tools/colab_chat/tiny_chat_run.py pilot_lr --art /mnt/c/ml_artifacts/tiny_chat/s0 \
   --art /mnt/c/ml_artifacts/tiny_chat/s0_rerun --out experiments/tiny_chat_2026-10/stage0/pilot_lr.json
python3 experiments/tiny_chat_2026-10/make_jobs.py   # commit the rate file(s) and the jobs files, push: SHA_P
$R --commit SHA_P --jobs tools/colab_chat/tiny_chat_pilot.json --name pk-tc-pilot --art /mnt/c/ml_artifacts/tiny_chat/pilot \
   --state-dir /home/jonat/pk_tc_pilot_runner --max-hours 8 $U
# once post_pilot_s35_r99 is home:
python experiments/tiny_chat_2026-10/analyze.py --art C:/ml_artifacts/tiny_chat \
   --pilot-verdict experiments/tiny_chat_2026-10/pilot/verdict.json   # commit it, push: SHA_V
# PASS: A2.7's E5 and E1 commands with --commit SHA_V. FAIL:
$R --commit SHA_V --jobs tools/colab_chat/tiny_chat_e5_60m.json --name pk-tc-e5-60m \
   --art /mnt/c/ml_artifacts/tiny_chat/e5_60m --state-dir /home/jonat/pk_tc_e5_60m_runner --max-hours 23.5 $U
```

On the FAIL branch, `analyze.py` gains its E5-at-60M block (the 35% cell read from E1 at 60M's L8 runs) with
tests before the first of those runs comes home.
