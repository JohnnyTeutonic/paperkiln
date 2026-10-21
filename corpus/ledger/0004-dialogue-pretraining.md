# 0004 — dialogue text in pretraining (lever 2)

Corpus ids: `dialogue_soda_v1`, `dialogue_oasst_v1`, `dialogue_ultrachat_v1`,
`dialogue_everyday_v1` (fetcher `corpus/fetchers/dialogue_pt.py`), mixed with FineWeb-Edu
by `corpus/mix_pretrain.py`. Decided with Jonathan on 21 Oct 2026 (LOCAL_LLM_STACK.md
sections 7 and 8).

**1. Task.** The next from-scratch base model reads about 70% FineWeb-Edu and 30%
conversation, with a higher conversation share in a final annealing part, so that
conversation is a native register of the base and not only a thin SFT layer.

**2. Target.** The FineWeb base models learnt facts and an encyclopaedic register from
0.2-0.5B tokens of expository prose, but saw dialogue only in a 48k-example SFT. They answer
like a textbook and loop. The target is a base that takes turns in the chat template before
any SFT.

**3. Alternatives considered.**
- smol-smoltalk (Apache-2.0)
- DailyDialog
- TinyChat-HF (`0003-tinychat-hf.md`)
- LMSYS-Chat-1M
- ShareGPT dumps
- UltraChat in full (stingning/ultrachat)

**4. Reasons for rejection.**
- **smol-smoltalk:** it is lever 1's SFT set, and its test split is the preference-prompt
  pool. Reading it in pretraining would blur which lever moved what, and could put
  preference prompts into the base.
- **DailyDialog:** CC BY-NC-SA.
- **TinyChat-HF:** CC-BY-NC-4.0, and already the SFT register.
- **LMSYS-Chat-1M:** gated behind a licence agreement, and it holds real users' text.
- **ShareGPT dumps:** no clear licence or provenance.
- **Full UltraChat:** HuggingFaceH4/ultrachat_200k is its filtered subset at a pinned
  revision, already the source of `chat_prompts_v1`.

**Kept:**
- SODA (CC-BY-4.0): two-speaker social dialogue. The first speaker renders as the user, the
  second as the assistant.
- OpenAssistant oasst2 plus the oasst1 trees it lacks (Apache-2.0): the best-ranked English
  branch of each tree.
- UltraChat-200k (MIT): its own source and ratio knob, at Jonathan's decision. Its text is
  ChatGPT output, so it carries OpenAI's terms on outputs. Each mix's ledger records whether
  it read UltraChat, so runs with and without it can be compared.
- everyday-conversations (Apache-2.0), train split.

**5. Contamination.**
- Every dialogue goes through `corpus/decontam.py` against `simple_qa_v1`,
  `tinychat_hf_prompts_v1` and `chat_prompts_v1`. A row is dropped on a shared word 13-gram,
  on a user turn equal to an eval prompt of four or more words, or on an 8-12 word eval prompt
  as a span. A simple-QA knowledge answer near its question's words is flagged, not dropped.
- `chat_prompts_v1` comes from UltraChat's and everyday's test splits, which these corpora do
  not read.
- Duplicates are removed within each source by a hash of the dialogue's words.
- System messages: a dialogue whose system message gives an instruction is dropped, not
  stripped. Only `_common.BOILERPLATE_SYSTEM` text is stripped. Dialogues containing a
  chat-template marker are dropped.
- Every count, by reason and per source, is in each corpus's `report.json` and in
  `dialogue_pt_v1.ledger.json`. The corpora are not built yet.

**6. Licence.**
- SODA: CC-BY-4.0. Attribution: Kim et al., 2023, "SODA: Million-scale Dialogue Distillation
  with Social Commonsense Contextualization", allenai/soda.
- OpenAssistant: Apache-2.0.
- everyday-conversations: Apache-2.0.
- UltraChat-200k: MIT, with the OpenAI-output provenance above.
- No non-commercial or unclear licence is used.

**7. Baseline to beat.** `ctrl_sft`: a FineWeb-only base (dense fw_base plus 10k steps of
unseen FineWeb) with the QA+chat SFT. A base of the same size and token count trained on the
mix, given the same SFT, must beat it on each of these sets:
- `simple_qa_v1`: ctrl_sft scores 0.12 overall.
- `tinychat_hf_prompts_v1` (100 TinyChat prompts): reference F1 0.123 for ctrl_sft (the live
  model scores 0.162).
- `chat_prompts_v1`, UltraChat half only: reference F1 and reference bits per byte from
  `chat_eval.py`'s per-source report. ctrl_sft has no measurement on this set yet; 2e's eval
  turn measures it, and the numbers are written here when they exist. The everyday half is
  excluded from the gate: all 100 of its items are greetings whose reference, "Hello! How can I
  help you today?", is an assistant turn 2,005 times in everyday-conversations' training data,
  which both this corpus and the SFT read. On that half, F1 measures a memorised greeting.
