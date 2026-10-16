# Ledger 0000 — legacy selections (TinyStories, the chat7b vocabulary)

*Covers `tinystories_slice_v1`, `vocab_chat7b_cap4096_v1` and
`tinystories_valid_v1`. These were chosen before the ledger existed. This
entry was written on 16 Oct 2026 from git history in this repository and in
`transformer_cpp`. Each statement is marked **recovered** (a contemporaneous
record exists and is cited) or **unknown** (no record exists). Not knowing
why a corpus was chosen is itself a finding.*

## What is recovered

- **TinyStories itself, in `transformer_cpp`, July 2026.**
  `transformer_cpp/docs/TINYSTORIES_TRAINING.md` gives the reason: "TinyStories
  suits the word-level tokenizer well (simple vocabulary)". The same document
  ranks three instruction/dialogue datasets by fit to a 20,000-word word-level
  cap: TinyStories-Instruct (best), then DailyDialog, then Dolly ("richest
  vocabulary, most `<unk>`"). **Recovered:** the selection criterion was
  vocabulary fit to a word-level tokenizer, and those three were the
  alternatives ranked by it.
- **The slice.** `transformer_cpp/scripts/prepare_tinystories.py` wrote
  `train-0-small.txt` on 12 Jul 2026. It is the first 5,000 non-empty stories
  of the train split, a "smoke-test" variant (`--small`) of the 300,000-story
  training files. **Recovered:** the slice was built for quick runs, not chosen
  as a research corpus.
- **The vocabulary.** `chat7b.gguf` is a `transformer_cpp` model trained on
  TinyStories-Instruct (`CHAT_EXPERIMENTS.md`, chat7b row). Its 20,000-entry
  vocabulary was built by that repository's whitespace tokenizer, so
  punctuation stays attached to words (`day,`, `said.`).
- **The bytes.** `corpus/fetchers/tinystories.py slice_v1` rebuilds the slice
  from `roneneldan/TinyStories@f54c09fd`. Its sha256 equals the hash of the
  retained historical file (`672622b8…`, also recorded in the transfer paper's
  data note on 30 Sep 2026). The slice has no end-of-story token and has CRLF
  line endings.

## What is unknown

- **Why the smoke-test slice became the research corpus.** It enters
  paperkiln on 30 Jul 2026 (`4a4dd1e`, the SRD work) through an absolute path
  into `transformer_cpp`. No commit, document or decision records it. Every
  later study kept that path for comparability.
- **Why the chat7b vocabulary was paired with a TinyStories corpus**, rather
  than a vocabulary built from the corpus itself. No record exists. The
  pairing has consequences, measured in registry row
  `C0-legacy-vocab-coverage` and in
  `experiments/transfer_s2/posthoc/token_coverage.py` (9 Oct 2026):
  - 19.1% of trained tokens map to `<unk>`;
  - 14.6 points of that are ASCII punctuation absent from the vocabulary
    (the full stop, comma and double quotation mark above all);
  - 1,503 of the 4,095 non-`<unk>` entries are strings the mtstudio
    tokenizer can never produce.
- **Why 400,000 tokens.** That is mtstudio's default `data.max_tokens`. The
  studies therefore trained on the first 41% of the slice: 400,001 of its
  972,728 tokens, 1.72 MB of 4.17 MB. No study chose this explicitly.

## The seven questions

**1. Task.** Next-token prediction on simple English narrative.

**2. Target capability.** For the research corpus, none in itself: the
corpus is a fixed substrate on which architecture and attention lanes are
compared on identical batches. For `tinystories_valid_v1`, the quickstart
model, it is coherent children's-story continuation.

**3. Alternatives considered.** Recovered for TinyStories as a family
(TinyStories-Instruct, DailyDialog, Dolly, ranked by vocabulary fit). Unknown
for the slice and for the vocabulary pairing. No alternative to a narrative
corpus is on record.

**4. Reasons for rejection.**
- DailyDialog: worse fit to a word-level vocabulary cap; it also taught
  register but not answers (`CHAT_EXPERIMENTS.md`).
- Dolly: the richest vocabulary and the most `<unk>` under the cap.
- TinyStories-Instruct: not rejected; it was used for the chat7b model. The
  vocabulary came from it, the corpus did not.

**5. Contamination risk.** No public benchmark is evaluated in this
repository. The validation signal is the trainer's own held-out tail (the
final 5% of the token stream), taken from the same 5,000-story slice. It is
**not a separate split**: it is the last ~20,000 tokens of the same text. No
TinyStories train/validation overlap measurement exists. It is pending as an
n-gram overlap line on the card (Phase H).

**6. Licence.** The pinned revision's dataset card declares
`cdla-sharing-1.0`. Redistributing the text carries that licence. What it
says about trained models (the licence's "Results") is quoted from the
licence text into the card before any model release relies on it. It is
not stated here from memory.

**7. Measured baseline the next corpus must beat.** None comparable exists.
The recorded losses (for example best_val 3.40 at d=256, 3,600 steps in
transfer_s1) are in nats per token on this corpus's own tail under a 19%
`<unk>` vocabulary. They cannot be compared with any other corpus or
tokenizer. The comparable baseline is bits per byte on the fixed evaluation
panel, measured by the C1 pilot under `tinystories_slice_v1` +
`vocab_chat7b_cap4096_v1` at d=128, L=2 before any other corpus is trained.
That pilot writes the number here.
