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

### Amendment 4 (23 Oct 2026, before any recursion, E5 or E1 data, and before mtstudio's schedule "mor" exists): Mixture-of-Recursions runs now, as causal token choice

Written after Jonathan's instruction of 23 Oct 2026 ("Mixture of Recursions I have an intuition is a missing
ingredient": step 2 of the recursion arm registered now and built as soon as fixed recursion passes its gate). Fixed
recursion has passed its CPU gates (paperkiln 567a441d; ember runs weight_index, ember 6a6e655). No recursion job has
run (`tiny_chat_s0_rec.json` and `tiny_chat_e1_rec.json` are not launched), no E5, E1, pilot or successive-halving job
has produced data, and this programme has produced no Carry-5 number. The commit that adds this amendment holds no
engine code for the new schedule: its implementation follows in later commits. Sections 0-12 and amendments 1-3 stand
except where this amendment says otherwise. Sources: the MoR paper (Bae et al., arXiv 2507.10524, read from its arXiv
HTML on 23 Oct 2026: section 2.2, appendix A.2-A.3, section 4.2 and Table 4, appendix G), `RECURSION_LITCHECK.md`, and
the five constraints in `include/microtorch/ir.hpp` agreed with ember.

**A4.1 What changes in A3.3.**
- A3.3's bullet "Step 2, Mixture-of-Recursions-style adaptive recursion ...: only if HR1 is SUPPORTED" is replaced by
  this amendment. **Step 2 is no longer conditional on HR1; it runs alongside the fixed arm** (the MoR block, A4.3),
  under the arm's gates: the pilot's PASS and each cell's own Stage 0 choice.
- Its design changes from expert choice to **causal token choice** (A4.2). A3.3's measurements (m1)-(m3) become
  H-MoR2's registered statistics (A4.5); A3.3 had them descriptive.
- A3.3's other rules (HR1-HR4, the R8 rule, MC7, the cost rule) are unchanged. On the pilot's FAIL branch the MoR
  block is not run at 20M (its comparison cells are 20M cells); a 60M MoR block would need a further amendment.

**A4.2 The mechanism (mtstudio `arch.custom.recursion` schedule "mor"; registered here, implemented after this commit).**
- **Topology:** Middle-Cycle, as MoR and MC7: prelude P = 1, shared block B = 2, coda C = 1 (4 stored layers);
  `repeats` = N_r, the largest number of recursion steps a token can take. Recursion step k (k = 0 .. N_r - 1) is the
  B applications P + kB .. P + kB + B - 1.
- **Router (token choice, MoR section 2.2.1):** one linear map W [N_r, d], no bias (GGUF `rec.0.router.weight`, written
  in full), reads the token's own hidden state h at the input of step 0 (after the prelude: MoR's H^1 = h^1 under
  Middle-Cycle). p = softmax(W h) over the N_r depths; the token's depth n = 1 + the index of the largest p (ties to the
  lower index, the shallower depth), n in 1 .. N_r; its gate g = p[n - 1]. The depth is a constant (top-1, as in MoE),
  not differentiated; g is.
- **Update:** the token takes steps 0 .. n - 1. At each step it takes, x <- x + g (S(x) - x), S(x) the step's B
  applications; at the steps it skips, x is unchanged. Every token passes step 0, so every token's gate receives the
  language-model gradient. `gate: "none"` (x <- S(x), no scaling) exists for tests only.
- **KV policy `kv_rows: "active"` (MoR's recursion-wise caching, its default):** at every application of step k the
  active tokens (n > k) of each sequence are the queries and the keys; attention is causal by absolute position among
  them, RoPE at absolute positions; a token that has exited writes no k/v row at that application or any later one.
  Decode keeps, per application, exactly those rows (ir.hpp constraint 2). Recursive KV sharing (`kv_from`) stays
  reserved and is not built.
- **Causality (ir.hpp constraint 1):** a token's depth is a function of its own prefix only, so prefilling a prompt and
  decoding it token by token give the same depths and, to rounding, the same logits.
- **Auxiliary losses (MoR appendix A.2 and G.1), coefficients fixed now:** balancing loss alpha sum_i f_i P_i with
  f_i = (N_r / T) #(tokens choosing depth i) and P_i = the mean of p_i over the micro-batch's tokens (1 at perfect
  balance), **alpha = 0.1**; router z-loss mean_t (logsumexp_i W h)^2, **coefficient 1e-3**.
- **MoR's settings, recorded as A3.3 required ("a dated note before any step-2 run"):** the paper's router ablation is at
  360M with N_r = 3 (Table 4, right). Linear routers with balancing loss 0.1 had MaxVio 0.266-0.492 (MLP: 0.200; the
  0.01 coefficient: 0.682; loss-free: 0.852-1.281), and section 4.2's final design keeps the balancing loss and adds
  the z-loss back at 1e-3 (appendix G.3). We take balancing loss 0.1 (A4.5's compute matching needs balance), softmax,
  a linear router (the one an IR linear expresses; mtstudio has no MLP router), z-loss 1e-3, **N_r = 3** (the paper's
  iso-FLOP and ablation setting, and its 135M MoR, 1 + 10 + 1 layers) and recursion-wise caching.
- **What this is not.** This tests **causal token-choice MoR**, not the paper's expert-choice variant. Expert choice
  ranks tokens across the sequence at every step (top-k per step), so a token's depth depends on later tokens and it
  cannot be served token by token; the paper's remedies (an auxiliary router, or a BCE loss pushing the router towards
  the top-k) make inference causal only approximately. At N_r = 3 and 16.5e18 FLOPs the paper reports expert choice
  42.6% and token choice 40.0% few-shot (Table 3). **The lower expected accuracy is a modelling cost accepted for an
  exactly servable model** (prefill equals decode, one token at a time, in ember).
- **An interpretation, stated.** The paper's equation 2.2 as printed gates g_t^r f(H^r) at every step and adds H^1 only
  at the exit step. We use the residual form x + g (S(x) - x) at every step taken, with g the chosen depth's
  probability (the top-1 MoE convention, and the residual form of its equation 2.1).
- **Compute:** mtstudio runs each step's applications on the active rows only (gathered per sequence), so a model's
  training FLOPs follow its tokens' depths.

**A4.3 The MoR block (cells).** All at 20m_L4's shape (d 640, 10 heads, d_ff 1,760; 4 stored layers, 20.08M
non-embedding, 10.49M embedding; the MoR router adds N_r d = 1,920 parameters).

| cell | recursion (P, B x r, C) | schedule | layer applications per token | training FLOPs/token |
|---|---|---|---:|---:|
| MoR3 | 1, 2 x {1, 2, 3}, 1 | mor, N_r = 3 | 4, 6 or 8; mean 6 at perfect balance | about 0.26G at balance (0.19-0.32G) |
| MC2 | 1, 2 x 2, 1 | fixed | 6 | 0.26G |
| MC3 | 1, 2 x 3, 1 (MoR3's largest depth) | fixed | 8 | 0.32G |

FLOPs as A3.3's table (about 0.0325G per application plus 0.06G for the head; attention under kv_rows active is
lower still for MoR3: at balance its three steps cost 1 + 4/9 + 1/9, about 1.56, full-width attention steps against
MC2's 2). Every cell trains on the 35% mix with section 1's budget (163.84M tokens), SFT, Carry-5, panels and the rung
logging of amendment 1 (b) (rungs at 5, 10, 25 and 50%), on seeds 1-3, batch 32 x 1 (effective depth at most 8). The
validity gate (A2.5) applies to every cell; a contrast that uses a FLOOR cell is FLOOR. None of input injection,
per-step embeddings, LoRA relaxation, deep supervision or EMA (as A3.3).

**A4.4 Training and Stage 0.** Each cell gets its own Stage 0 as A3.3: a 300-step probe alone on the GPU and screens at
**{2.5e-4, 5e-4, 1e-3, 2e-3}**, 1,500 steps, seed 101, the 35% mix; the rate by section 4's rule with A2.2's statuses.
A cell whose four screens all diverge is dropped and reported; if MoR3 is dropped, no H-MoR contrast is computed; if
MC2 or MC3 is dropped, H-MoR1 and H-MoR0 are not computed (H-MoR2 needs MoR3 alone). The router's depth fractions,
balancing loss and z-loss are logged at every step (mtstudio's step events), and the depth fractions at every
validation. Files: `tools/colab_chat/tiny_chat_s0_mor.json` (probes and screens) and
`tools/colab_chat/tiny_chat_e1_mor.json` (the nine runs with their post and rung jobs), separate from A3.3's files;
their gates need the pilot's PASS and each cell's choice. Section 8's failure rules apply.

**A4.5 Hypotheses** (three seeds, `decide` and verdicts as section 7; two-tailed, n = 3, critical 4.303).
- **Realised depth.** m_s = the mean number of recursion steps per training token over seed s's whole pretraining run
  (the mean over its 10,000 steps of each step's depth fractions, each step weighted equally; 1 <= m_s <= 3; perfect
  balance gives 2).
- **The fixed line at that depth (the compute matching).** Y_fix(s) = Y(MC2, s) + (m_s - 2) (Y(MC3, s) - Y(MC2, s))
  when m_s >= 2, and Y(MC2, s) when m_s < 2 (MC2 then spends more block compute than MoR3: conservative). MoR3's block
  FLOPs per token equal fixed recursion's at m_s steps (its attention FLOPs are lower); the linear interpolation in Y
  between 6 and 8 applications is the assumption that makes the comparison FLOP-matched at matched parameters. The
  same construction gives L_fix(s) for the FineWeb panel loss.
- **H-MoR1 (Jonathan's intuition; registered prediction):** at matched parameters and matched training FLOPs, causal
  token-choice MoR beats fixed recursion on Y. **D1(s) = Y(MoR3, s) - Y_fix(s) > 0**, SESOI 3 points (the pairwise
  SESOI). Literature-led alternative (MoR's own token choice below its expert choice; MoR below vanilla at 135M;
  `RECURSION_LITCHECK.md`): D1 is not above the SESOI (EQUIVALENT or CONTRADICTED). H-MoR1 is falsified by EQUIVALENT
  or CONTRADICTED; NOT_RESOLVED decides neither. Secondary (SECONDARY, no verdict claimed for the hypothesis):
  Y(MoR3) - Y(MC2), Y(MoR3) - Y(MC3), and the panel contrast L(MoR3) - L_fix; each seed's m_s is reported beside D1.
- **H-MoR2 (the router reads the task).** From the per-token depth trace of each MoR3 seed's SFT model, teacher-forced
  with gold history over Carry-5's context families (recall, correction, topic return: 828 items), in the multi-turn
  rendering carry5_eval scores. *Answer positions*: the positions whose next token is a token of the gold value inside
  the target reply (the positions that predict the answer, as MoR section 5.2 reads depth against the next token).
  *Other assistant positions*: every other position whose next token belongs to an assistant turn.
  - **H-MoR2a:** Delta(s) = the mean depth over answer positions minus the mean over other assistant positions (each
    pooled over the items) **> 0**, SESOI 0.10 recursion steps (5% of the 1-3 range).
  - **H-MoR2b:** per seed, the OLS slope of an item's mean answer-position depth on its fact distance (Carry-5's
    `distance`, 1-4 turns) **> 0**, SESOI 0.05 steps per turn.
  - H-MoR2 is SUPPORTED when both are; it is falsified when either is CONTRADICTED or EQUIVALENT. Descriptive beside
    them (A3.3's m3): the same allocations on the single-turn twins (distance 0) and on the question alone.
- **H-MoR0 (counter: causal routing collapses).** *Collapse criterion*, per seed, on all positions of the H-MoR2 trace:
  the normalised entropy of the depth fractions H(f) / log 3 < 0.35 (one depth takes about 90% of positions), or
  |Delta(s)| < 0.05 steps (half H-MoR2a's SESOI: routing does not separate answers from other tokens). **H-MoR0 is SUPPORTED when the collapse criterion holds in every seed and D1 is EQUIVALENT** (the 90%
  interval inside +-3 points): MoR is then fixed recursion with extra steps. It is falsified when D1 is SUPPORTED or
  CONTRADICTED, or when no seed collapses.
- Extension rule (section 7) applies to D1 alone.

**A4.6 Engine gates, before any MoR job runs** (CPU here; CUDA on Colab through a gate job before the Stage 0 launch):
(1) causality: prefill equals token-by-token decode, logits to rounding and depths token for token; (2) a
finite-difference gradcheck through the routed model (gate, router and auxiliary losses included); (3) with every token
forced to the largest depth and `gate: "none"`, the model equals fixed recursion (MC3's topology) bit for bit; (4) the
IR: numpy over the F32 GGUF against `mtstudio score` within ROUNDING, greedy 40/40, and the depth trace identical token
for token; (5) `tools/colab_cuda_validate.sh` with the MoR leg passes on Colab. `analyze.py`'s MoR block (H-MoR0-2,
m_s, the interpolation) lands with its tests before any MoR run comes home.

**A4.7 Budget** (amendment 2's measured model with A3.3's per-application estimate: about 0.098 s per application per
16,384-token step at d 640, plus 0.117 s; probes replace it). Per run (pretraining + SFT + 0.62 h of post and rung
jobs): MC2 2.9, MC3 3.5, MoR3 4.1 L4 VM-hours (MoR3 costed at its largest depth plus 20% for gathering and
per-sequence attention, until its probe measures it). The nine runs: about **31 VM-hours**; Stage 0 (3 probes, 12
screens, the session's data and gate jobs): about **5**; the MoR block **about 36 VM-hours** (about 175 compute units).
With A3.3's arm (about 48 VM-hours without R8, 78 with it) the recursion arms come to **about 84, or 114 with R8: past
80**. Jonathan approved "go for all" (23 Oct 2026). Under section 8 rule 5's form: the MoR Stage 0 (about 5 VM-hours)
runs on that go; **the nine MoR runs launch after the probes have replaced these estimates and Jonathan has confirmed
the measured total**. No cell, seed or token budget changes silently.

**A4.8 Interfaces.** mtstudio: `arch.custom.recursion` = {prelude, block, repeats, coda, schedule "mor", inject "none",
"mor": {route "token_choice", gate "score", balance 0.1, z_loss 0.001, kv_rows "active"}}; step events carry the
depth fractions and the router losses; `mtstudio score` writes each prompt's per-token depth trace. IR (`ir.hpp`): the
`depth_route` node and the kv slot's `kv_rows: "active"`, normative from the implementing commit. Jobs:
`tiny_chat_s0_mor.json`, `tiny_chat_e1_mor.json`. Analysis: `analyze.py` MoR block, as A4.6 says.

### Amendment 5 (25 Oct 2026, before any X5 run): arm X5, Attention Residuals on the llama base, on Carry-5

Written after Jonathan approved arm X on 25 Oct 2026 ("execute the plan to completion"): his cross-layer idea, that
tokens should choose which earlier activations to strengthen beyond the fixed residual sum, run on Carry-5 as a
main-effect read. When it was written no X5 job had run and no llama_attnres model had been trained on the
programme's data. E1's three 20m_L16 runs were home (`INTERIM_2026-10-25.md`). Sections 0-12 and amendments 1-4
stand. This amendment adds one arm and changes nothing registered for any other. Sources:
`../tiny_chat_long_2026-10/CROSSLAYER_LITCHECK.md` (sections 2, 4, 5) and the X cell of
`../tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md` (sections 6 and 8).

**A5.1 The mechanism: `llama_attnres`** (paperkiln c48ca962; normative in the IR as keramos v1.1.3's family
`llama_attnres`, IR_SPEC section 9). The llama block of section 1 (RMSNorm, RoPE on every head, SwiGLU, tied
embedding) with Attention Residuals (Kimi, arXiv 2603.15031) in full form at sublayer level:
- **Sources:** the token embedding and every attention output and every FFN output, in order: 2L + 1 = 33 at
  20m_L16. Each sublayer's input, and the head's, is a softmax over the sources it can see, with one head,
  computed by `ops::depth_attend`. The logits are a pseudo-query (one [d] vector per sublayer, plus one for the
  head) dotted with the parameter-free RMSNormed sources. The values are the raw sources.
- **Scale "sources":** each mixture is multiplied by its source count S (the IR's `depth_attend` followed by
  `affine(S, 0)`). Under the following RMSNorm this is Kimi's convex mixture with epsilon eps / S^2.
- **Zero-initialised queries:** the uniform mixture times S is the plain residual stream, so at initialisation
  the model is the plain llama of the same seed to rounding. The queries draw no random numbers, so every other
  tensor starts bit for bit as in E1's L16 run of that seed. 32 queries of d 320 add 10,240 parameters (0.04% of all parameters, 0.05% of the non-embedding ones).
  Application 0's attention query sees one source and is never read (exported as zeros).
- **`block_size`, resolved from the code:** in `nn::AttnResStack`, `block_size` 0 is the full form and
  `block_size` 1 equals it exactly (each block is one layer; `tests/test_attn_res.cpp` and
  `python/attn_res_reference.py` pin Block(S = 1) == Full). `llama_attnres` has no `block_size`:
  `arch.custom.attn_res` = {"form": "full", "scale": "sources"} applies the full form over the 2L sublayers.
  That is AttnResStack's `block_size` 0 with the sublayers as its layers, equivalently `block_size` 1 counted in
  sublayers. The literature check's two wordings name the same function. mtstudio's GPT-2 `AttnResLM` is
  different: `block_size` 2, one source per transformer block.
- **Checks on CPU before this amendment** (`tests/test_llama_attnres.cpp`, `tests/test_llama_attnres_reference.py`,
  `tests/test_llama_attnres_ir.py`):
  - The zero-init equivalence: logits within 3.0e-7 relative of the plain llama's at 16 layers, plain-parameter
    gradients within 1.1e-6, and the step-1 loss equal to 7 digits.
  - Finite-difference gradchecks of the op and of 3-layer and recursed models (worst 6.6e-3 at a step of 1e-2).
  - The fused op against python/attn_res_reference.py's `_depth_attend` under torch autograd in float64: output,
    query and source gradients within 7.1e-6 of the largest magnitude, the worst case at S 33 and d 320.
  - Decode against prefill: 3.1e-6.
  - The arm's exact shape (`ar_x5`: d 320, 16 layers, 5 heads, d_ff 864, RoPE on every head), trained 40 steps:
    keramos's numpy interpreter over its F32 GGUF against `mtstudio score` within 2.7e-6, greedy 40/40, and its
    IR (`ir_version` 1, only `embedding`, `list`, `depth_attend`, `affine`, `rmsnorm`, the llama attention and
    SwiGLU, `append`) accepted by keramos v1.1.3's `ir_validate.py` with both GGUFs.
  - The CUDA kernels against the host loop on an L4: the `test_llama_attnres` leg of
    `tools/colab_cuda_validate.sh` passed at paperkiln 86a978ca (21/21 legs). X5's Stage 0 repeats the gate at this
    amendment's commit.
  - No keramos change was needed.

**A5.2 Arm X5.** One cell, **X5 = 20m_L16 + llama_attnres**, seeds 1-3, everything else E1's L16 cell's:
- the 35% mix (s35), 10,000 steps x 16,384 tokens (batch 16 x 2), the same data order (`train.seed`), the same
  SFT recipe and set, Carry-5, the panels, the rungs, the post job and its generation path (A5.4);
- **learning rate:** E1 L16's chosen rate, 1e-3 (`stage0/lr_choice.json`; written for X5 to
  `stage0/lr_choice_x5.json`). This is Kimi's convention and conservative against the arm. There is no screen.
- `tests/test_tiny_chat_x5.py` checks that every X5 pretraining spec equals E1's L16 spec of its seed except
  `arch.custom.attn_res` and the job names, and that every post command is E1's except its spec path and names.

**Stage 0** (`tools/colab_chat/tiny_chat_s0_x5.json`, one RunPod L4), in order, one GPU job at a time:
1. the CUDA gate (every leg of `tools/colab_cuda_validate.sh`);
2. the gate (design, the floor pilot's PASS) and the data job;
3. 300-step probes of 20m_L16 and 20m_X5 alone on the GPU (tokens/s);
4. the plain replay (A5.3);
5. a 300-step X5 smoke run (seed 101, warmup 30, exported), whose F32 GGUF goes to ember's gate at the real
   vocabulary.

**The runs launch only when** Stage 0's CUDA gate passed (`x5/cuda_gate_x5.json`), the plain replay is
bit-identical (`x5/plain_bitid.json`), ember's gate passed (`x5/ember_gate.json`), and the probe's X5/L16
tokens/s ratio is at least 0.3 (the draft's costing). Below 0.3, Jonathan decides. The gate job of
`tiny_chat_e1_x5.json` refuses without the first three receipts.

**A5.3 The control.** E1's three home 20m_L16 runs (`ART/e1`, paperkiln dc29b5e6), paired by seed.
- **The plain path at this amendment's commit.** E1's pt_e1_L16_r1 spec trains again for its first 500 steps.
  500 steps is the warmup, so the learning rate up to there does not depend on `train.steps`. It runs as a
  training job in Stage 0 (`replay_e1_L16_r1`, the job environment E1's runs had). `x5/bitid.py` compares it
  with E1's log and passes only if:
  - the specs are equal except steps, name and out_dir;
  - the model events are equal;
  - every step record (loss, gradient norm, learning rate) and the step-500 validation loss equal E1's exactly.
  The floor pilot showed pretraining bit-identical across L4 providers and commits.
  The arm's runs launch from a later commit that adds Stage 0's receipts. `git diff` between the two commits
  touches no file under `src/`, `include/` or `tests/`, no CMake file, and no tool except the jobs files.
- **The control's item records.** HX5-main is item-level, and E1's post jobs kept their per-item records on
  their VMs. So E1's three L16 post jobs are re-run at the arm's commit from E1's home pretrained weights
  (`tiny_chat_e1_x5_ctl.json`). The command is E1's post command, but for where the weights are uploaded, and the
  outputs include `carry5/m/items.jsonl`.
  - `analyze_x5.py` compares each re-run with its home summary: the pretrained weights' sha256, every Carry-5
    pool, Y, Y_all, the SFT weights' sha256 and the panels.
  - The test pairs X5's items with the re-runs' records whether or not a re-run reproduces exactly: the same
    pretrained model, through the post pipeline X5's runs use at the same commit. Each seed's reproduction is
    reported beside the verdict.
  - The validity gate reads E1's home summaries, as registered for E1's cell.

**A5.4 Generation.** Carry-5's self-history replies for X5's and the re-run's post jobs are generated by ember
under the implementation note of 24 Oct 2026: the per-run 40-item spot-check against `mtstudio chat`, and
mtstudio for the whole run otherwise.
- **The pin.** One ember commit for both files, written into every post command (`post --ember-commit`, from
  `make_jobs.py`'s `X5_EMBER_COMMIT`). It is the commit named by ember's gate receipt, `x5/ember_gate.json`.
- **What the receipt must show at that commit:**
  - the X5-shaped model `ar_x5` passes greedy 40/40 against mtstudio, with log-probabilities against
    `mtstudio score`, on CPU, CUDA and WASM;
  - Stage 0's smoke model (the real vocabulary) passes the same on CPU. Its CUDA leg on an L4 is each X5 run's
    own 40-item spot-check, which falls back to mtstudio for the whole run on any difference;
  - on the control's own SFT model (E1's pick, `pick_e1_L16/model.gguf`, seed 1), the commit generates exactly
    as ember cbdcb915d26a did. E1's L16 posts were generated at cbdcb915d26a (all three summaries'
    `gen_engine`: TF32 off, all layers on the GPU).
- **If that check fails,** nothing launches until Jonathan or the main session decides.
- The commit and the receipt are recorded here, in the commit that adds `x5/ember_gate.json`, before any X5
  run. `tests/test_tiny_chat_x5.py` requires the jobs files' pin to equal the receipt's.

**A5.5 The registered prediction, HX5-main (Jonathan's hypothesis).** Y(X5) - Y(L16) > 0 beyond the SESOI of 3
points. Y is the primary metric as amended (A2.5): forced choice, self history, multi-turn minus twin, over the
888 items of the recall, correction, topic-return and knowledge families.
- **Test:** the item-level paired differences d(i, s) = [self(X5) - twin(X5)] - [self(L16) - twin(L16)] for the
  same item i and seed s, pooled over the three seeds (3 x 888 = 2,664). Their mean is Y(X5) - Y(L16) averaged
  over seeds. The test reports the mean with normal-approximation 95% and 90% intervals.
- **Verdicts, as in Stage L0:**
  - SUPPORTED: the 95% lower bound > 0 and the difference >= 0.03;
  - SIGNIFICANT_BELOW_SESOI: the 95% lower bound > 0 and the difference < 0.03;
  - CONTRADICTED: the 95% upper bound < 0;
  - EQUIVALENT: the 90% interval inside +-0.03;
  - otherwise NOT_RESOLVED.
- **The validity gate** (A2.5) applies to both cells on their means over seeds. A FLOOR cell ends the contrast as
  FLOOR.
- No extension rule: the unit is the item, and three seeds are registered.
- The programme's seed-level t (section 7) is printed beside it, labelled DESCRIPTIVE.
- `experiments/tiny_chat_2026-10/analyze_x5.py` (run by `analyze.py` as `E1_x5_20m`) implements it.
  `tests/test_tiny_chat_x5.py` tests it on synthetic summaries and items, for each verdict, FLOOR, the
  reproduction report, incomplete inputs and items that do not belong to their summary.
- `analyze_x5.py` refuses when a machine's data ledger differs from E1's build (the split note of 25 Oct).

**A5.6 The registered counter-hypothesis (the literature's default).** Any gain is a general language-model
improvement. At matched FineWeb panel loss, X5 sits on the plain cells' loss-to-Y line. This is the isoplot,
descriptive and reported beside the verdict:
- the run-level OLS line of Y on the FineWeb panel loss through all home plain 20M runs (E5's four mixes and E1's
  depths), and through the s35 runs alone (the arm's own mix);
- X5's residual per seed and its mean, with extrapolations flagged.
Reading: a positive HX5-main with residuals near zero is the counter-hypothesis's pattern. A positive residual
beyond the plain runs' residual spread is the pattern of an arm-specific gain.

**A5.7 Descriptive (no verdicts).**
- Y with gold history, as a paired item-level difference.
- The paired difference by family and by distance.
- Panel losses (FineWeb, dialogue) and SFT loss, X5 minus L16 per seed.
- Tokens/s from Stage 0's probes (X5 against 20m_L16 on the same L4) and Y per L4-hour.
- **Depth-attention weights at the answer token** against their mean over the other assistant positions,
  singling out the embedding and the early sources: the direct test of "selective strengthening". Also the
  **routing entropy per mixture** (the collapse check: near-uniform weights are the plain model). Both come from
  `tools/carry5_attnres_trace.py`, which runs at home on the post jobs' SFT GGUFs, teacher-forced with gold
  history over Carry-5's context families. It is written and tested before any X5 post job is home.
- Stage L0 on X5's pick, as on the E picks (`STAGE_L0.md`; descriptive here).

**A5.8 What this arm does not test.** The gap hypothesis (HX: a benefit that grows with distance) stays in the
Carry-12 pre-registration (`../tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md`), where distances 6-13 exist.
Within Carry-5 the cost of carry-over does not grow with distance (INTERIM point 4), so Carry-5 cannot separate a
gap effect from a main effect.

**A5.9 Failure rules.**
- Section 8 applies: a divergence reruns once at half the rate; resumes are recorded.
- A reclaimed post job reruns from the pretrained weights.
- A Stage 0 failure (CUDA gate, replay, smoke) stops the arm before any run, and the cause is reported. No
  design choice is changed to make it pass.

**A5.10 Schedule and budget.**
- **Machines:** separate RunPod L4s. The seeds run on three machines at once (`pk-tc-x5-r1`, `-r2`, `-r3`):
  - Each machine runs `tiny_chat_e1_x5.json` at the arm's commit, builds its own data and gate, and owns one
    seed's pretraining, rungs and post job. The 25 Oct split note's pattern is applied by seed rather than by
    cell.
  - The cell's pick does not run on a machine. It runs at home: `tiny_chat_run.py pick --out ART/e1_x5/pick_e1_x5_X5
    --run ART/e1_x5_r1/post_e1_x5_X5_r1 --run ... r2 --run ... r3`. The pick's rule is unchanged, and it checks
    the copied weights against the summary's sha256.
  - X5's post jobs bring home their SFT model (safetensors, F32 GGUF, spec) for that pick, plus the per-item
    records and the replies. `tests/test_tiny_chat_x5.py` tests the home pick.
  - Stage 0 (`pk-tc-x5-s0`) runs on its own L4 first. The control re-run (`pk-tc-x5-ctl`) runs on another L4
    beside the seeds, from the same commit, once A5.4's pin is fixed.
- **Estimates:** Stage 0 about 1.5 L4 VM-hours. The re-run about 5 (E1's L16 post jobs took 1.4-1.6 h each).
  X5 about 4.5-5 per seed at the build's estimated 0.65x throughput (pretraining 3.1 h, post 1.6 h). In all,
  about 21 L4 VM-hours.
- **Caps:** per-campaign dollar caps on RunPod: Stage 0 $3, the re-run $5, each seed $6.

**A5.11 Interfaces.**
- `make_jobs.py`: `x5_shapes`, `x5_lr`, `x5_spec`, `s0_x5_doc`, `ctl_x5_doc`, `e1_x5_doc`, `budget_x5`; the
  files `x5/specs/shape_20m_X5.json`, `stage0/lr_choice_x5.json`, `tools/colab_chat/tiny_chat_s0_x5.json`,
  `tiny_chat_e1_x5_ctl.json` and `tiny_chat_e1_x5.json`.
- `tiny_chat_run.py gate --need-ok FILE`.
- `x5/bitid.py`; `analyze_x5.py`; `tools/carry5_attnres_trace.py` (A5.7).
- Tests: `tests/test_tiny_chat_x5.py`, `tests/test_llama_attnres_reference.py` and the `ar_x5` case of
  `tests/test_llama_attnres_ir.py`.

**A5.12 Stage 0's receipts and the pin** (26 Oct 2026, before any X5 run; Stage 0 `pk-tc-x5-s0` at paperkiln
585206a8, one RunPod L4 in EU-RO-1, 00:30-01:17 AEDT). The arm's runs launch from the commit that adds this
section.
- **CUDA gate:** 22/22 legs of `tools/colab_cuda_validate.sh` on the L4 (`x5/cuda_gate_x5.json`). The
  `test_llama_attnres` device leg: depth_attend at S 33, T 64, d 320, kernels against the host loop, output
  3.4e-6, gradients 1.5e-5; four device training steps against the host, losses within 7.0e-7.
- **Throughput** (`x5/stage0/`, 300-step probes alone on the GPU, steady rate from step 50):
  - 20m_L16 26,007 tokens/s, peak 6,604 MiB;
  - 20m_X5 17,378 tokens/s, peak 7,568 MiB;
  - **X5/L16 = 0.668** (above A5.2's 0.3), and +15% peak memory.
  - X5's pretraining is about 2.6 h a run.
- **The plain path** (`x5/plain_bitid.json`): the replay of pt_e1_L16_r1 at 585206a8 equals E1's log at dc29b5e6
  in all 500 step records (loss, gradient norm, learning rate) and in the step-500 validation loss,
  5.281066030263901. The specs are equal but steps, and the model events are equal. **Bit-identical.**
- **The smoke run** (seed 101, 300 steps): loss 9.769 to 5.676, validation 5.601. Its F32 GGUF and IR pass keramos
  v1.1.3's validator, and paperkiln's numpy interpreter against `mtstudio score` on the 40 spot-check items is
  within 1.9e-6, greedy 40/40 (`x5/smoke_gate_manifest.json`).
- **A5.4's pin: ember e115fd3d23ad5cee1ab7fb3bd0c8bb249ef69555** (receipts ember d74e052,
  `ember.cpp/experiments/attnres_x5_2026-10-25/`; `x5/ember_gate.json`). No engine change was needed.
  - **ar_x5:** worst log-probability error against `mtstudio score`:
    - CPU 40/40, 5.6e-7, decode equal to prefill bit for bit;
    - WASM 40/40, 5.6e-7;
    - CUDA 40/40, 6.7e-7 (on a T4, as no L4 was free);
    - Q8_0 40/40, 7.1e-5;
    - the never-read query set to NaN leaves the logits bit-equal.
  - **The control's generation path:** e115fd3 against cbdcb915 on the control's own SFT GGUF
    (`pick_e1_L16/model.gguf`, seed 1, sha256 96f6afdc), TF32 off, 360 requests (40 chat prompts and the 40
    Carry-5 spot-check items):
    - CPU: bit-identical at all 4,599 sampled positions; replies and ids 360/360.
    - CUDA: replies and ids 360/360, and first-token logits bit-equal. Decode logits differ by up to 1e-5. That
      is each build's own run-to-run spread (8.6e-6 for e115fd3, 1.3e-5 for cbdcb915): ember's CUDA decode uses
      atomics and is not bit-reproducible run to run, at about 1e-5, in both builds.
    - The control's posts were generated under the same property, so the comparison stays symmetric, and every
      run's 40-item spot-check guards it. Ember's planned deterministic mode is not enabled for X5, because the
      control did not have it.
  - **The smoke model, on ember's CPU:** greedy 40/40, worst 1.6e-6, decode equal to prefill bit for bit, and
    the chat server's path on the spot-check items 40/40 with logits bit-equal to `ir_logits`
    (`smoke/x5s_gate_cpu.json`, `x5s_chat_cpu.json`). Ember's WASM leg finished after the runs launched:
    40/40, 1.7e-6, top-1 3,699/3,699 (ember cfc5c1c, `smoke/`). Ember's later EMBER_DETERMINISTIC option
    (fe01b35) is not in the pin.
  - e115fd3's first build for the L4 (sm_89, `ensure_ember`'s command) is the first X5 post job's own; its
    spot-check is that build's gate.
- **Commits:** the runs' commit differs from 585206a8 only in documents (this file, README, ROADMAP,
  CHANGELOG), the files under `x5/` (the receipts and `smoke_gate.py`), the pin in `make_jobs.py`, and the
  regenerated `tiny_chat_e1_x5.json` and `tiny_chat_e1_x5_ctl.json` (`git diff --stat 585206a8`). Nothing under
  `src/`, `include/`, `tests/` or `tools/` changes except those two jobs files.

### Amendment 6 (26 Oct 2026, before any 150M job): a size arm on Carry-5 at 150M

**Registered 26 Oct 2026, before any 150M job ran,** on Jonathan's decisions of that day: (1) tokens fixed at
163.84M per run (A6.3); (2) Stage 0 at all four candidate depths at once, on four L4s (A6.4's first launch option).
Its machinery and Stage 0 files are those of paperkiln cd72fe17; the registering commit changes this file only.

Written on 26 Oct 2026 under Jonathan's change of tactics for the programme (`TACTICS_2026-10-26.md`): five
interventions at 20M (dialogue share, depth, recursion, MoR and Attention Residuals, X5 EQUIVALENT) have left
Carry-5's carry-over cost where it was, and model size is the one lever with a signal (descriptive, from the
home runs: E1 at 60M's L32 cell, mean Y -0.043 over three seeds, against 20M's L32 cell, -0.079, with no overlap
between the seeds). When it was written, no 150M job had run; E1 at 60M had its L32 cell home (3 seeds), L4 and L8
with two post jobs each and L16 with none; E5 and E1 at 20M were not complete. Sections 0-12 and amendments 1-5
stand. This amendment adds one arm and changes nothing registered for any other: E1 at 60M's hypotheses, analysis
and verdicts are untouched, and the depth rule below reads its cell means without deciding anything for it.
Machinery: `size/make_size_jobs.py` (shapes, specs, the Stage 0 files, the budget, the depth rule),
`size/analyze_size.py` and `tests/test_tiny_chat_size.py`.

**A6.1 Shape: 150M non-embedding parameters at the depth E1 at 60M favours.**

| shape | layers | d | heads | d_ff | d_ff/d | non-emb | emb | total | emb share | micro-batch |
|---|---|---|---|---|---|---|---|---|---|---|
| 150m_L4 | 4 | 1728 | 27 | 4928 | 2.85 | 149.98M | 28.31M | 178.29M | 16% | 16 x 2 |
| 150m_L8 | 8 | 1216 | 19 | 3520 | 2.89 | 150.07M | 19.92M | 169.99M | 12% | 16 x 2 |
| 150m_L16 | 16 | 832 | 13 | 2656 | 3.19 | 150.40M | 13.63M | 164.03M | 8% | 16 x 2 |
| 150m_L32 | 32 | 576 | 9 | 1952 | 3.39 | 150.44M | 9.44M | 159.88M | 6% | 8 x 4 |

Shapes by `make_jobs.py shape_for` (section 6's rule: d a multiple of 64, d_ff/d nearest 3, d_ff a multiple of 32).
- **The depth rule, fixed now** (`make_size_jobs.py depth`, tested). It reads only once all twelve E1 at 60M runs
  (L4, L8, L16, L32 x seeds 1-3) have their post summaries home. The 150M depth L* is the 60M depth with the highest
  mean Y (A2.5's primary) over seeds 1-3 among the cells that pass the validity gate. Depths within 1 point
  (0.01) of that mean are tied, and the tied depth with the higher 60M probe rate (A2.1) wins: the cheaper run. The
  receipt `size/depth_choice.json` is committed before any 150M run; it is written once. If every 60M cell is
  FLOOR, no depth is chosen and Jonathan decides.
- **Why the rule and not L32 now.** The signal was seen at L32, but 20m_L32 is the sweep's thinnest shape (d 192,
  d_ff/d 4.33) and its worst 20M cell so far, so a slope anchored there could carry 20m_L32's own deficit. The rule
  takes the depth the 60M models favour, which every 60M depth's 20M twin allows (E1's L8 cell is E5's 35% cell).
- **Selection and the test.** The 60M cell is chosen for its high Y. In HS1's slope over log2 non-embedding
  parameters the three cells weigh -0.353, +0.021 and +0.333 (L32; -0.354/+0.020/+0.334 at L4, -0.353/+0.022/+0.331
  at L8, -0.352/+0.021/+0.331 at L16), so the chosen cell's selection bias reaches HS1 at 2% of its size. HS2 uses
  the 60M cell directly; the selection raises Y(60M), which biases HS2 towards zero, against the arm's prediction.
- **Confounds, stated before data.** Size changes width at fixed depth: at L* the 150M model is wider, and its
  embedding share is lower (L32: 14%, 9% and 6% at 20M, 60M and 150M; L4: 34%, 23%, 16%). Every table reports total
  parameters and the slope over log2 total parameters beside the registered one.

**A6.2 Training.** Section 1 for everything not named here; the cell is E1's recipe at a third size.
- The 35% mix (E1's level), 10,000 steps x 16,384 tokens (163.84M), AdamW, warmup 500, cosine to 0.1x, the rate
  from A6.4; seeds 1, 2, 3, paired by seed with the 20M and 60M cells (`train.seed` sets the data order).
- **Micro-batching:** 16 x 512 with 2-step accumulation at L <= 16 and 8 x 512 with 4 steps at L32 (activation
  memory; the same 32 x 512 tokens per step and the same update, as section 1). The probes measure the peak.
- **SFT** by section 1's recipe at the same micro-batching. `tiny_chat_run.py post` sets SFT batching by depth alone
  (32 x 1 below 16 layers), which is too large at 150M, so the post jobs read (batch, accum) from the shape spec
  (`post --sft-batching spec`; the flag, with a test that every existing jobs file's command is unchanged, lands
  before the runs' commit).
- `train.fused_head` and BF16 stay off, as in every registered arm, so the 150M cell differs from the 20M and 60M
  cells in size alone.
- Rungs at 5, 10, 25 and 50% as amendment 1 (b), descriptive here. The post job is E1's (panels, Carry-5 with ember
  generation and its 40-item spot-check, base-model Carry-5) and also brings home the per-item records, the replies
  and the SFT model, because the seeds run on separate machines and the pick runs at home (A5.10's pattern).

**A6.3 Tokens: fixed at 163.84M, not fixed per parameter.**
- *Fixed tokens* (chosen) keeps every registered 20M and 60M cell as this arm's comparison without new runs, and
  answers the programme's practical question: with this human-only corpus and budget, does a larger model carry facts
  across turns better? Its cost: 163.84M tokens is 8.2 tokens per non-embedding parameter at 20M, 2.7 at 60M and 1.1
  at 150M, so a flattening at 150M can be under-training, not a limit of scale (HS-0 below states it).
- *Fixed tokens per parameter* (8.2 per non-embedding parameter: about 1.23B tokens at 150M) would need the 60M cell
  re-run at about 490M tokens, and 35% dialogue over 1.23B tokens is about 430M dialogue tokens, about 13 passes over
  the pretraining dialogues: three times section 2's limit of four (Muennighoff et al.), with new mixes beyond the
  800 MB of section 2. It would also cost about 7.5 times the pretraining. The human-only corpus rules it out here.

**A6.4 Stage 0** (`tools/colab_chat/tiny_chat_s0_150m_L<4|8|16|32>.json`, one RunPod L4 per depth; one GPU job at a
time):
1. the CUDA gate (every leg of `tools/colab_cuda_validate.sh`) on that machine's GPU;
2. the gate (design final, the floor pilot's PASS) and the 35% data job;
3. a 300-step probe alone on the GPU: steady tokens/s from step 50 and peak memory at A6.2's micro-batching;
4. lr screens at **{2.5e-4, 5e-4, 1e-3, 2e-3}**, 1,500 steps (warmup 150, cosine to 0.1x), seed 101, on the 35% mix.
   The grid is A3.3's four-rate grid, one step below section 4's: at 60M the two widest shapes chose 5e-4 and 2e-3
   was their worst rate by 1.02 and 1.06 nats (`stage0/lr_choice.json`), and the 150M shapes are wider still
   (d 576-1,728). The screens wait for the probe, so a probe that fails for memory holds them back;
5. in the L4 file only, **the plain replay** `replay_e1_60m_L32_r2`: E1 at 60M's pt_e1_60m_L32_r2 spec at this
   arm's commit for its first 500 steps (the warmup), compared with its log at a90ec0e2 by `x5/bitid.py` into
   `size/plain_bitid.json` (A5.3's test: specs equal but for the steps, model events equal, every step record and
   the step-500 validation loss equal). The replay's spec equals the registered one but for the steps
   (`tests/test_tiny_chat_size.py`).
- **The rate, fixed now:** `tiny_chat_run.py choose_rec` over L*'s four screens (section 4's rule with A2.2's
  statuses: the lowest final validation loss; within 0.01 nats, the smaller rate), into `stage0/lr_choice_150m.json`.
  A choice at either end of the grid stands and is reported as an edge choice. If all four screens diverge, no rate
  is chosen and Jonathan decides. The screens of the other depths are reported descriptively (validation loss by
  depth and rate at 150M) and choose nothing.
- **Launch options** (the main session's call): all four depths at once, on four L4s, so the runs can launch as soon
  as the depth rule fires (about 25.5 L4 VM-hours, $15), or L*'s file alone after the rule fires (6-7.5 VM-hours,
  $3.40-4.40, about 7 hours later on the clock). The jobs files are the same either way.

**A6.5 Hypotheses** (three seeds per cell, paired by seed; `decide` and the verdicts of section 7; two-tailed,
n = 3, critical 4.303; x = log2 of the non-embedding parameter count of each cell at L*).
- **HS1 (registered prediction): the carry-over cost falls with size.** The within-seed OLS slope of Y on x over the
  20M, 60M and 150M cells at L* is **> 0; SESOI 1 point of Y per doubling** (E1's SESOI per doubling of depth;
  20M to 150M is 2.9 doublings, so the SESOI over the range is 2.9 points, about the pairwise SESOI).
  - Counter-hypothesis **HS-0** (at a fixed 163.84M tokens the larger models are data-starved, 1.1 tokens per
    parameter at 150M, and carry-over is limited by the data's carry-over signal rather than by capacity): the
    slope is inside +-1 point per doubling.
  - HS1 is falsified by EQUIVALENT or CONTRADICTED; NOT_RESOLVED decides neither.
- **HS2 (registered, secondary): the cost is still falling between 60M and 150M.** b2(s) = [Y(150M, s) -
  Y(60M, s)] / log2(N_150 / N_60) (1.32 doublings) **> 0, SESOI 1 point per doubling**.
  - Counter **HS2-sat**: EQUIVALENT (the fall stops by 150M at this budget).
  - Falsified by EQUIVALENT or CONTRADICTED.
  - A6.1's selection biases b2 towards zero. With the seed SDs of the home L32 cells (about 0.01 at each size), a
    true 2 points per doubling gives HS1 a t near 7 at three seeds but HS2 a t near 3, so HS2 will likely need the
    extension rule.
- **The validity gate** (A2.5) applies to each of the three cells on its means over seeds; a FLOOR cell ends both
  contrasts as FLOOR.
- **The extension rule** (section 7) applies to HS1 and HS2: seeds 4 and 5 for the three cells they use, once (new
  runs at 20M and 60M as well as 150M, unless E1's own extension has run them).
- **Descriptive (no verdicts):** the 20M -> 60M step b1 and the change of step b2 - b1; the slopes of Y with gold
  history, of twin and multi-turn accuracy and of the panel and SFT losses; Y's slope over log2 total parameters; the
  family table; the isoplot (A5.6's line of Y on the FineWeb panel loss through every home plain 20M run, with the
  150M runs' residuals and an extrapolation flag); Y per L4-hour; Stage L0 on the 150M pick (Carry-12 v1 at T 512 by
  `STAGE_L0.md`'s commands; descriptive).
- `size/analyze_size.py` implements it, on synthetic summaries in `tests/test_tiny_chat_size.py`: the slope's weights,
  SUPPORTED and EQUIVALENT readings, FLOOR, the L8 cell read from E5's s35 runs, the refusal of incomplete cells and
  `--partial`.

**A6.6 Failure rules.** Section 8 applies (a divergence reruns once at half the rate; resumes recorded; a lost post
job reruns from the pretrained weights). A probe whose peak exceeds 19,500 MiB (too little of the L4 left for the
light rung job) or that fails for memory doubles that depth's accumulation (the same tokens and update) and is
repeated before any run: a schedule change, recorded here before the run. A Stage 0 failure of the CUDA gate or
the replay stops the arm before any run, with the cause reported.

**A6.7 Budget and the launch criterion** (`make_size_jobs.py --tables`). Planning rates: the lower of the fit through
A2.1's nine probes (time per token in training FLOPs and layers) and 60M's probe at the same depth scaled by FLOPs
per token; 1.08 for evaluations and checkpoints in the run (E1 at 60M ran 6-11% under its probe rate); 2.0 h per
post job beyond its SFT (E1 at 60M: 0.78-2.88 h, median 1.36). At $0.59 per L4-hour (RunPod).

| block | tok/s (planning) | L4 VM-hours | USD | runner cap |
|---|---:|---:|---:|---:|
| Stage 0 150m_L4 (CUDA gate, data, probe, 4 screens, the replay) | 5,950 | 6.0 | 3.51 | $5 |
| Stage 0 150m_L8 | 5,990 | 5.7 | 3.37 | $5 |
| Stage 0 150m_L16 | 5,320 | 6.3 | 3.75 | $5 |
| Stage 0 150m_L32 | 4,430 | 7.5 | 4.43 | $6 |
| runs at 150m_L4, 3 seeds (PT 8.3 h, SFT 1.2 h, post 2.0 h each) | | 36.2 | 21.33 | $10 per seed |
| runs at 150m_L8, 3 seeds | | 36.0 | 21.23 | $10 per seed |
| runs at 150m_L16, 3 seeds | | 39.5 | 23.31 | $11 per seed |
| runs at 150m_L32, 3 seeds (PT 11.1 h each) | | 45.9 | 27.10 | $12 per seed |
| **the arm, Stage 0 at all depths + runs at L*** | | **61-71** | **$36-42** | |

- The caps are 1.3 times each runner's estimate, rounded up. The runs go one seed per L4 (A5.10's pattern), about
  12-15 hours on the clock.
- **The runs launch** when (a) `size/depth_choice.json` is committed; (b) L*'s Stage 0 CUDA gate passed; (c)
  `size/plain_bitid.json` is ok; (d) `stage0/lr_choice_150m.json` holds a rate for 150m_L*; (e) L*'s probe peaked at
  or under 19,500 MiB; and (f) the runs' total from L*'s own probe (`make_size_jobs.py` with the probe home) is at most
  1.5 times the estimate above (54, 54, 59 or 69 VM-hours at L4, L8, L16, L32). Then the main session launches them;
  otherwise Jonathan decides. The runs' commit differs from Stage 0's (cd72fe17) only in documents, receipts, the
  arm's jobs files, its own tools (`size/make_size_jobs.py`, `size/analyze_size.py`, `analyze.py`'s `E1_150m` block)
  and `tiny_chat_run.py post --sft-batching` with their tests: nothing under `src/`, `include/` or CMake, and no
  change to any existing jobs file or to the post job's default path.
- **(c)'s model events (Jonathan's ruling, 27 Oct 2026, after the replay came home).** `x5/bitid.py` compares the
  replay's model event with E1 at 60M's on the fields both commits log: every shared field equal, and no field in
  E1 at 60M's event alone. The replay runs at cd72fe17 and E1 at 60M ran at a90ec0e2; between them, paperkiln
  c6365b88 added two fields to mtstudio's logged model event, `final_norm` and `norm_placement`, whose values are
  fixed for every non-flex family (true and "pre"; this arm and E1 are llama). They are listed in `bitid.py`
  (`LOGGED_LATER`) and checked against those values, not required to be absent; any other field in the replay's
  event alone fails (c). They describe the model the spec already fixes, so they are a logging change, not a
  difference in the run ("that's just a logging change, why bother?"). The receipt records them under
  `model_event`. Amendment 5's receipt (`x5/plain_bitid.json`, both commits before c6365b88) is unchanged by the
  rule: rerun, it is byte-identical. `tests/test_tiny_chat_x5.py` tests the rule.

**A6.8 Interfaces.** Now: `size/make_size_jobs.py` (it imports `make_jobs.py` and writes none of its files),
`size/specs/shape_150m_L*.json`, `tools/colab_chat/tiny_chat_s0_150m_L*.json`, `size/analyze_size.py`,
`tests/test_tiny_chat_size.py`. For the runs, in place before the depth is chosen: `tiny_chat_run.py post
--sft-batching spec` (default `depth`, every registered arm's path, its summary unchanged); the runs' jobs files
`tools/colab_chat/tiny_chat_e1_150m_r<1|2|3>.json`, one per seed and machine, which `make_size_jobs.py` writes once
`size/depth_choice.json` exists, at the PROVISIONAL rate until `stage0/lr_choice_150m.json` holds L\*'s (their gate
refuses until then, and until L\*'s home CUDA gate `size/stage0/cuda_gate_150m_L*.json` and `size/plain_bitid.json`
are committed); the pick at home over the three post folders; `make_size_jobs.py`'s A6.7 (e)/(f) check once L\*'s
probe is home; and `analyze.py`'s `E1_150m` block (`size/analyze_size.py`, `--size-depth-choice`).

**A6.9 Launch** (the main session launches; Stage 0 at paperkiln cd72fe17, SHA_S below: the registering commit
changes documents only, so Stage 0's code and jobs files are the same):

```
U="--upload vocab_llama3_bpe.gguf=/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
R="python3 -u tools/colab_chat/runner_multi.py --backend runpod --gpu L4"
SHA_S=cd72fe171a7a4b101eef75503f7bd7155cb648c0
for L in 4 8 16 32; do   # all four at once (Jonathan, 26 Oct 2026)
  $R --commit SHA_S --jobs tools/colab_chat/tiny_chat_s0_150m_L$L.json --name pk-tc-s150-L$L \
     --art /mnt/c/ml_artifacts/tiny_chat/s0_150m_L$L --state-dir /home/jonat/pk_tc_s150_L${L}_runner \
     --max-hours 12 --volume-gb 40 --max-usd CAP_L $U
done
# at home, once E1 at 60M is home:
python experiments/tiny_chat_2026-10/size/make_size_jobs.py depth --art C:/ml_artifacts/tiny_chat
python3 tools/colab_chat/tiny_chat_run.py choose_rec --art /mnt/c/ml_artifacts/tiny_chat/s0_150m_L$LSTAR \
   --out experiments/tiny_chat_2026-10/stage0/lr_choice_150m.json
python experiments/tiny_chat_2026-10/x5/bitid.py --replay ART/s0_150m_L4/replay_e1_60m_L32_r2 \
   --e1 ART/e1_60m/pt_e1_60m_L32_r2 --commit SHA_S --e1-commit a90ec0e2b2aaf6052a5ff953acd1e0a40a9a222c \
   --out experiments/tiny_chat_2026-10/size/plain_bitid.json
```

The root `scripts/colab_driver.json` holds the four Stage 0 entries (`pk-tc-s150-L4`, `-L8`, `-L16`, `-L32`, caps
$5, $5, $5 and $6), enabled on 26 Oct 2026.

`bitid.py`'s receipt `size/plain_bitid.json` applies (c)'s model-event rule (A6.7; Jonathan's ruling, 27 Oct 2026)
with the arguments above.

## Implementation notes

### 24 Oct 2026, before any E5, E1, recursion, MoR or successive-halving post job: Carry-5's generation engine

An implementation note, not an amendment: no hypothesis, metric, cell, seed, item, threshold or analysis changes.

**What changes.** Carry-5's greedy replies (the five self-history turns of every item, and the gold, twin and
question-alone target replies) are generated by ember.cpp's server instead of `mtstudio chat`. Forced-choice
scoring, and with it the primary metric, stays with `mtstudio score`, unchanged. Reason: the floor pilot's post job
took 4.3 h on an L4, over 2 h of it generating through `mtstudio chat` (about 0.1-0.2 s per token); section 10
allowed 0.3 h per post job beyond its SFT, and E5, E1 and E1 at 60M hold about 39 post jobs.

**Why it is not a design change.** Greedy generation is engine-independent by the stack's parity contract
(`LOCAL_LLM_STACK.md`: ember and mtstudio agree greedy 40/40 on every family, CUDA included). The run enforces it
per job rather than assuming it:

1. *Build.* Every post job uses ember at the commit `tiny_chat_run.EMBER_COMMIT` names, cloned from GitHub and built
   once per VM (`/content/ember/build`, Release, CUDA architecture 89) by the first post job under a file lock; the
   commit is recorded in each post job's `gen_engine.json` and `summary.json`. Campaigns launched from paperkiln
   dc29b5e6 to 2d42395c (E5, E1 and E1-R at 20M, E1 at 60M) pin `cbdcb915d26a` (main on 24 Oct 2026: weight_index
   recursion and MoR). Later commits, to 0c802f86, pin `69d7ca26d5cd` (ember's CUDA prefill rmsnorm reduced over a
   power-of-two block: at d 192, 20m_L32's width, `cbdcb915d26a` dropped a third of the sum of squares, so 20m_L32's
   spot-check failed 0/40 and its post jobs generate with `mtstudio chat`, as rule 3 provides; every other width in
   the sweep is unaffected). Commits after 0c802f86 pin `aa50a6112dfd` (ember main on 25 Oct 2026, conformant with
   `spec/IR_SPEC.md` v1: it reads `ir_version` 1, which paperkiln's exporters write from 5349ee19 on, and
   which `69d7ca26d5cd` refuses). Its parity receipts are byte-identical to `69d7ca26d5cd`'s, so generation is
   unchanged, and the per-run spot-check still gates every run.
2. *Server.* The SFT model's F32 GGUF (the SFT spec exports it; `mtstudio export` re-exports it when missing or not
   F32), all layers on the GPU, `EMBER_GEMM_TF32=0` (TF32 GEMMs can flip greedy tokens against mtstudio's f32),
   flash-attention-2 on (parity-gated in ember).
3. *Spot-check.* Before generating, the job generates 40 fixed items (Carry-5 items 0, 26, ..., 1014: 17 recall, 8
   correction, 7 topic return, 6 instruction, 2 knowledge) with ember and with `mtstudio chat --kv-cache`, and
   compares all eight replies of each item on text, stop reason and history used. Only 40/40 lets ember generate the
   run. Otherwise `mtstudio chat` generates the whole run and `summary.json` carries `gen_flag`; the engines are never
   mixed within a run (carry5_eval refuses a resumed run whose generator differs).
4. *Families.* llama_recursive runs on ember's IR interpreter, on CUDA in CUDA builds, whose device run ember has not
   yet gated: the first recursion post job's spot-check is that gate. When it fails, the spot-check is repeated with
   `EMBER_IR_DEVICE=cpu`; if that passes, ember generates the run on its CPU interpreter, recorded and flagged, else
   mtstudio generates it. MoR (llama_mor) runs on ember's CPU path, also in CUDA builds.
5. *Whole-run check (descriptive).* ember returns token ids, so carry5_eval's parity gate also scores every
   self-history reply with `mtstudio score` as a continuation (the argmax must equal the reply at every position):
   recorded per run as `gen_engine.full_gate` (5,160 replies).
6. The floor pilot's jobs file keeps mtstudio generation: it ran before this note.

*Checked on CPU before launch* (WSL, ember at the pinned commit built by `tiny_chat_run.ensure_ember` with
`-DHAS_CUDA=OFF`): on the floor pilot's pretrained 20m_L8 model (27M parameters, the Llama 3-scheme 16k vocabulary)
the spot-check gave 40/40 items and 320/320 replies identical between ember and `mtstudio chat --kv-cache` (3,414
generated tokens; 136 s for ember, 458 s for mtstudio). The post job end to end on toy llama, fixed-recursion and
MoR models (`tests/test_tiny_chat_ember.py`): spot-check 3/3 items (24/24 replies) and the whole-run gate 283/283
on each family, ember generating. The CUDA spot-checks are the first E post jobs' own.

### 24 Oct 2026, before any E5, E1, recursion, MoR or successive-halving post job: the SFT loss on the device

An implementation note, not an amendment: no hypothesis, metric, cell, seed, item, threshold, analysis or SFT recipe
changes. It changes the speed of the SFT stage, not its loss.

**What changes.** `ops::cross_entropy_masked` (the SFT objective) gains a device path. Until now it had none: on
every SFT step it downloaded the [16,384, 16,384] logits (1 GiB at batch 32 x T 512 x the 16k vocabulary), ran the
softmax and the gradient on the host and uploaded the 1 GiB gradient. The floor pilot's SFT ran at 3,490 tok/s on an
L4 against 25,113 tok/s for pretraining the same 27M model at the same tokens per step (1.96 h per post job; section
10 budgets SFT at the pretraining rate). The device path runs the same softmax, nll and (P - onehot) g kernels as
`ops::cross_entropy`, with masked rows contributing an exact 0 to the loss sum and an exact 0 gradient, and the mean
over the kept rows. paperkiln commit: the one carrying this note.

**Why the loss is the same.** (1) A mask of all ones still returns `ops::cross_entropy` itself, so it is equal bit for
bit (`tests/test_posttrain_losses.cpp`, and on the device `tests/test_cuda_ops.cpp` leg 9, write-through and
deferred). (2) The unmasked kernels are template instantiations whose masked branch is compiled out, so
`ops::cross_entropy` (pretraining) is unchanged. (3) On a kept row the masked kernel executes the unmasked kernel's
instructions, so its softmax and gradient equal device `cross_entropy` on the kept rows bit for bit (leg 9); the
loss differs only in the order of one fp32 sum (leg 9 bound: 1e-6 relative to device `cross_entropy` on the kept
rows, 1e-5 relative to the host op; gradient within 1e-6 of the host op). (4) The host path, used on CPU, is
unchanged: its bit-for-bit equality with `cross_entropy` on the kept rows and its finite-difference gradcheck (worst
relative error 5.8e-6) pass as before. The CUDA legs run on Colab in `tools/colab_cuda_validate.sh` before the first
post job; they were compiled, not run, locally.

**Measured locally (CPU).** The host path at half the SFT shape (8,192 x 16,384, two threads) took 2.35 s per step
(forward 0.87 s, backward 1.48 s), about 4.7 s at the full shape before any bus traffic, against the observed 4.04 s
per-step gap between SFT and pretraining on the L4 (whose VM has more cores for the forward's parallel softmax; the
backward's loops are serial). Expected SFT rate with the device path: about the pretraining rate, ~25k tok/s, so
1,500 SFT steps in about 16 minutes instead of 1.96 h; the first post job's step log measures it.

### 24 Oct 2026, before any run outside Colab: a second L4 provider

An implementation note, not an amendment: no hypothesis, metric, cell, seed, item, threshold or analysis changes.

**What changes.** Runs may execute on an NVIDIA L4 rented from RunPod as well as on Colab's L4 (`runner_multi.py
--backend runpod`; the same jobs files, `tiny_chat_run.py` and checkpoint relay). Reason: Colab grants three L4s at
once, and E1 at 60M waits for a slot behind E5, E1 and the recursion arm.

**The GPU rule.** microtorch/AGENTS.md makes the L4 the study GPU and forbids mixing GPU types inside an arm. A
RunPod L4 is the same GPU (Ada, sm89, 24 GB), so an arm may hold runs from both providers; no other GPU type (A100,
RTX 4090 and the like) enters a claim-carrying arm. A run's pretraining and its post job may sit on different
providers (the post job reads the pretrained weights); a relaunch resumes a run only on an L4.

**Why the registered metrics carry over.** Y, the validity gate, the panels and the validation losses are computed
from the models' outputs. Different L4 instances (already a different Colab VM for nearly every run) differ only in
the rounding of the same kernels. Before any arm uses RunPod, the floor pilot (20m_L8, 35% dialogue, seed 99, lr
1e-3, its post job and Carry-5) is repeated there and compared with its Colab run; the comparison is recorded in
experiments/tiny_chat_2026-10/pilot/ beside the Colab verdict.

**What does not carry over: wall-clock.** S9 (throughput in the runs) and S10 (Y per L4-hour) use each run's own
wall-clock, which depends on the host (CPU, bus, power cap) as well as the GPU. Every run records its provider, GPU
model, driver and CUDA version and image digest (`provenance` in its result and summary files). S9 and S10 are
reported with each run's provider beside them, and a cell whose runs span providers has its S10 flagged. Stage 0's
per-shape probe rates, all measured on Colab, stay the reference for per-token time.

### 25 Oct 2026, before the MoR Stage 0 and before any R8 decision: decisions delegated by Jonathan

An implementation note, not an amendment: no hypothesis, metric, cell, seed, item, threshold or analysis changes.

Jonathan delegated four decisions to the main session (2e) on 25 Oct 2026, before leaving it to work unattended:
the MoR block's launch, including A4.7's confirmation of the measured total; R8 if A3.3's rule fires; the Carry-12
MSC rulings (recorded in the long-carry pre-registration); and the RunPod pod cap (raised from 2 to 4).

**A4.7, the criterion fixed before the probes report.** The nine MoR runs launch when the MoR Stage 0 is home and
(a) its CUDA gate passed, (b) choose_rec chose a learning rate for every cell (no cell dropped), and (c) the measured
total for the nine runs, from the block's own probes through `make_jobs.py`, is at most 47 L4 VM-hours (1.5 times
A4.7's 31-hour estimate). Above 47, or with a dropped cell, the runs wait for Jonathan. The runs execute on an L4
(Colab or RunPod; the provider note of 24 Oct applies).

**R8.** If `analyze.py --rec-decision` writes `run_r8: true` (A3.3's rule on R2 and R4), R8's Stage 0 and its three
runs launch without a further confirmation, though the arm then exceeds 80 L4 VM-hours (about 81); the decision file
is committed before any R8 job, as A3.3 requires.

### 25 Oct 2026, with E1 at 60M and E1-R under way: arms split by cell across machines

An implementation note, not an amendment: no hypothesis, metric, cell, seed, item, threshold or analysis changes.

**What changes.** On Jonathan's go, an arm may run on several L4 machines at once, divided by cell. A machine owns
whole cells: each cell's three pretraining runs, their rungs and post jobs, and its pick, so a pick's inputs all live
on the machine that picks. Every machine runs the arm's own jobs file at the arm's pinned commit and builds its own
`data` and `gate`; a machine skips the cells it does not own (its runner records them as run elsewhere). Split on
25 Oct 2026, at about 19:30 AEDT:
- E1 at 60M (`tiny_chat_e1_60m.json`, a90ec0e2): L32 stays on its RunPod L4 (`pk-tc-e1-60m`); L16, L8 and L4 each
  run on their own RunPod L4 (`pk-tc-e1-60m-L16`, `-L8`, `-L4`). No job of L16, L8 or L4 had started.
- E1-R (`tiny_chat_e1_rec.json`, 11c5baf0): R4 and MC7 stay on Colab (`pk-tc-e1-rec`); R2 runs on a RunPod L4
  (`pk-tc-e1-rec-R2`). No job of R2 had started.
- The MoR block (`tiny_chat_e1_mor.json`, 86a978ca): MoR3 and MC3 stay on their RunPod L4 (`pk-tc-e1-mor`); MC2
  runs on its own RunPod L4 (`pk-tc-e1-mor-MC2`). No job of MC2 had started.

**Why the registered quantities carry over.** Each machine's data job runs the same command at the same commit; its
`data.ledger.json` (corpus and panel hashes) is compared with the arm's first data build before that machine's
results are analysed, and a mismatch holds the cell's results for Jonathan. Everything else follows the provider
note of 24 Oct: the L4 is the study GPU on either provider, and wall-clock measures carry each run's provider.

**Provenance.** E1-R's pinned commit predates the `provenance` stamp in result and summary files, on Colab and RunPod
alike, so for E1-R each run's provider and GPU come from its runner's ledger (RunPod pod records) or Colab session
log; the analysis records them beside S9 and S10. Each cell's artefacts come home to their own directory
(`/mnt/c/ml_artifacts/tiny_chat/e1_60m_<cell>`, `.../e1_rec_R2`) and join the arm's directory before analysis.
