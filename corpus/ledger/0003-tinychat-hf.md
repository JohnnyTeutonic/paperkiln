# Ledger 0003 — TinyChat (starhopp3r) for the chat model

*Written 18 Oct 2026, before any model is trained on it. Covers
`tinychat_hf_full_v1`, `tinychat_hf_sample_v1` (the first 20 MiB, used to
train the tokenizer), `vocab_tinychat_bpe8k_v1` and the frozen
`tinychat_hf_prompts_v1`.*

**1. Task.** Short, multi-turn user/assistant conversation in plain English.

**2. Target capability.** A paperkiln-trained model you can chat with: it
answers the turn it is given, stays on topic, and stops. The base model
learns the language and the content of conversation. Post-training teaches
the turn protocol (SFT) and sharpens the replies (preference optimisation).
The gain from each stage is what the chat server should make visible.

**3. Alternatives considered.**
- TinyStories (with or without dialogue mixed in)
- UltraChat-200k
- everyday-conversations-llama3.1-2k
- TinyDialogues (styfeng)
- SmolTalk

**4. Reasons for rejection.**
- **TinyStories:** narrative prose, not conversation. A model trained on it
  learns to tell stories, not to take turns. It was briefly in the plan, as
  a pretraining mix, and was rightly removed.
- **UltraChat-200k:** open-domain dialogue with long, technical answers, far
  beyond what a 15–30M-parameter model trained on about 100M tokens can
  learn. Kept only as a possible later source.
- **everyday-conversations:** the right register, but 2,005 dialogues, which
  is far too few to pretrain on. It remains an SFT candidate.
- **TinyDialogues:** child-directed, often multi-party speech. A different
  register from user/assistant chat.
- **SmolTalk:** the dataset card declares no licence.

**5. Contamination.**
- Train, val and test are disjoint by construction: each conversation is
  assigned by a hash of its text.
- The frozen evaluation prompts come from the test split only.
- Duplicate conversations across splits would be caught by the hash only
  if they are byte-identical. A near-duplicate check is pending (Phase H
  MinHash).
- No public benchmark is evaluated.

**Known defect: bare [/INST] turns.** About 0.3% of source conversations
(90 of 27,587 in sample_v1) separate turns with `[/INST]` alone, so the
parser reads several turns as one assistant reply that still contains the
markers. full_v1 is pinned as fetched: those markers appear in its
pretraining text (561 in sample_v1's 19 MB), and a base model can echo
them. `tools/chat_data.py` drops such dialogues whole from SFT data and
reports the count (`dropped_dialogues_with_markers`).

**6. Licence.** CC-BY-NC-4.0, per the dataset card. Use is non-commercial,
and any model trained on it inherits that: a public release must carry the
NC terms. The data is synthetic (generated with GPT-4o-mini, per the card).

**7. Baseline to beat.** Bits per byte of the base model on `dialogues_val`
reference replies, and on the frozen prompts: the rate of single-turn
replies that end cleanly (end-of-text, no role spill) and the repetition
rate. Each post-training stage must beat the base on these, and the
numbers are written here when the base exists.
