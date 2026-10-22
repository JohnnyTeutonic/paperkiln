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

None.
