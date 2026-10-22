# 0005 — human-written dialogue (no distillation)

Corpus ids: `human_<source>_v1` (fetcher `corpus/fetchers/human_dialogue.py`). The source
survey, licences, sizes and role mapping are in `corpus/HUMAN_DIALOGUE_SURVEY.md`; this entry
answers the seven questions.

**1. Task.** Dialogue text for the tiny-chat experiments
(`experiments/tiny_chat_litcheck_2026-10-23.md`): pretraining share and SFT for 5M-150M models
judged on multi-turn carry-over (Carry-5).

**2. Target.** Jonathan's rule of 23 Oct 2026: for this research, no text written by a large
language model enters training data. Lever 2's SODA, UltraChat and everyday-conversations, lever
1's smol-smoltalk and TinyChat are all model-generated (their dataset cards say so), so the tiny-chat
runs need a dialogue corpus written by people.

**3. Alternatives considered.** SODA, UltraChat-200k, smol-smoltalk, everyday-conversations,
TinyChat, Synthetic-Persona-Chat, ConvAI2's human-bot evaluation logs, Reddit-derived sets, the
Cornell Movie-Dialogs corpus, CANDOR, Switchboard, the Ubuntu Dialogue Corpus, Schema-Guided
Dialogue, CoQA, Commonsense-Dialogues, CMU_DoG, Taskmaster-4.

**4. Reasons for rejection.**
- **Model-generated:** SODA (InstructGPT), UltraChat (ChatGPT), smol-smoltalk (SmolTalk's
  synthetic sets), everyday-conversations (Llama-3.1-70B-Instruct), TinyChat (GPT-4o mini),
  Synthetic-Persona-Chat (LLM generator-critic), ConvAI2 human-bot logs (bot turns).
- **Terms:** Reddit (the Data API terms forbid training a model on user content without the
  rightsholders' permission), Cornell Movie-Dialogs (no licence; film scripts), CANDOR (access by
  application, research use), Switchboard (LDC user agreement, fee), Ubuntu Dialogue Corpus and
  CMU_DoG (no data licence found).
- **Other:** Schema-Guided Dialogue (simulator outlines paraphrased by crowd workers; the dialogue
  flow is machine-made; CC BY-SA), CoQA (licence differs by domain; answers are terse spans of an
  unseen passage), Commonsense-Dialogues (CC BY-NC 4.0; not implemented, a further candidate
  for the non-commercial flag), Taskmaster-4 (crowd self-dialogues for a 2024 paper, no stated control
  against LLM use).

**Kept (default, permissive):** OpenAssistant oasst2 + oasst1 (Apache-2.0; synthetic messages and
ChatGPT tells removed), ConvAI2/PersonaChat, Wizard of Wikipedia and Blended Skill Talk (CC BY 4.0
per ParlAI), Taskmaster-1/2/3 (CC BY 4.0), MultiWOZ 2.2 (MIT), the Self-dialogue Corpus (BSD-3).
**Opt-in, off by default:** share-alike Topical-Chat, Dolly, QuAC, Simple English Wikipedia;
non-commercial EmpatheticDialogues, DailyDialog, No Robots; LLM-risk Aya (English originals).

**5. Contamination.**
- `corpus/decontam.py` against `simple_qa_v1`, `tinychat_hf_prompts_v1`, `chat_prompts_v1`, and
  `carry5_v1` (user turns and gold replies) once it exists; Carry-5's own template and item-leak
  check is `tools/carry5_build.py decontam` on each `text.txt`.
- Within-source duplicates removed by `_common.dialogue_key`; duplicates across sources counted in
  `human_dialogue_v1.ledger.json`, not removed.
- Only train splits are read where a source defines them.
- LLM-era sources (oasst, Dolly, No Robots, Aya) drop dialogues holding a ChatGPT tell phrase.

**6. Licence.** Per source in the survey. The default set carries attribution duties only (CC BY
4.0, Apache-2.0, MIT, BSD-3). Share-alike and non-commercial sources need Jonathan's decision
before a released model reads them.

**7. Baseline to beat.** `ctrl_sft` (ledger 0004's baseline): simple_qa_v1 0.12,
tinychat_hf_prompts_v1 reference F1 0.123. For the tiny-chat experiments, the E5 data bar: a
model with the human-dialogue share must beat the same model with 0% dialogue on Carry-5's
multi-turn accuracy at equal tokens.
