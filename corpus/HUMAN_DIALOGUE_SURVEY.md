# Human-written dialogue for tiny chat: source survey (23 Oct 2026)

**Rule (Jonathan, 23 Oct 2026).** For the tiny-chat research, no text written by a large language
model enters training data: no distillation. Every claim below was read on 23 Oct 2026 from the
primary page named in section 8 (dataset card, paper, licence file or terms page), not from memory.
Fetcher: `corpus/fetchers/human_dialogue.py`; selection record: `ledger/0005-human-dialogue.md`.

## 1. The sets

The tiny-chat programme trains on the default set and the six share-alike and non-commercial
dialogue sources (Jonathan, 23 Oct 2026; PREREGISTRATION.md amendment 1): 13 sources, all built
and pinned. Aya stays out; Simple English Wikipedia is built for the knowledge arm (E5w).

On by default (permissive licence, human authorship documented):

| Source | Written by | Licence | Built | Dialogues kept | Turns | 16k tokens | 8k tokens |
|---|---|---|---|---:|---:|---:|---:|
| OpenAssistant oasst2 + oasst1 | Volunteers, both roles | Apache-2.0 | laptop, pinned | 5,328 | 16,490 | 2,425,273 | 2,909,441 |
| Blended Skill Talk | Paired crowd workers | CC BY 4.0 | laptop, pinned | 4,819 | 54,034 | 1,014,819 | 1,075,603 |
| Self-dialogue Corpus | One crowd worker, both sides | BSD-3-Clause | laptop, pinned | 24,133 | 372,450 | 5,527,322 | 5,919,493 |
| ConvAI2 / PersonaChat | Paired crowd workers | CC BY 4.0 | Colab, pinned | 8,951 | 131,530 | 1,883,518 | 1,999,124 |
| Wizard of Wikipedia | Crowd apprentice + wizard | CC BY 4.0 | Colab, pinned | 18,428 | 148,166 | 3,466,236 | 3,750,181 |
| Taskmaster-1, -2 (v1: TM-3 not read, section 6) | Crowd users with call-centre operators; crowd self-dialogues | CC BY 4.0 | Colab, pinned | 28,791 | 508,774 | 7,599,378 | 8,516,118 |
| MultiWOZ 2.2 | Crowd workers, both sides (Wizard-of-Oz) | MIT | Colab, pinned | 8,292 | 111,676 | 2,209,246 | 2,449,865 |
| **Default set** | | | | **98,742** | **1,343,120** | **24,125,792** | **26,619,825** |

Gated (`--allow-sa`, `--allow-nc`, `--allow-llm-risk`); the share-alike and non-commercial
dialogue sources are in the tiny-chat pool, Aya is not:

| Source | Gate | Written by | Licence | Built | Dialogues kept | Turns | 16k tokens | 8k tokens |
|---|---|---|---|---|---:|---:|---:|---:|
| Topical-Chat | share-alike | Paired crowd workers | CDLA-Sharing-1.0 | Colab, pinned | 8,605 | 181,884 | 5,037,915 | 5,459,288 |
| databricks-dolly-15k | share-alike | Databricks employees | CC BY-SA 3.0 | laptop, pinned | 14,945 | 29,890 | 2,921,410 | 3,372,933 |
| QuAC | share-alike | Crowd student + teacher | CC BY-SA 4.0 | laptop, pinned | 11,567 | 138,218 | 2,571,107 | 2,920,693 |
| Simple English Wikipedia | share-alike | Wikipedia editors | CC BY-SA, GFDL | Colab, pinned | 224,597 articles | documents | 79,523,627 | 96,655,832 |
| EmpatheticDialogues | non-commercial | Paired crowd workers | CC BY-NC 4.0 | laptop, pinned | 17,780 | 73,258 | 1,361,911 | 1,432,972 |
| DailyDialog | non-commercial | Human-written (card) | CC BY-NC-SA 4.0 | laptop, pinned | 10,235 | 76,770 | 1,273,731 | 1,347,962 |
| No Robots | non-commercial | Skilled annotators (card) | CC BY-NC 4.0 | laptop, pinned | 8,698 | 17,396 | 2,765,925 | 3,164,023 |
| Aya, English originals | LLM risk | Volunteer annotators | Apache-2.0 | laptop, pinned | 2,103 | 4,206 | 207,879 | 242,728 |

Token counts are of each corpus's `text.txt` in the stack's chat template (markers included):
"16k" is `vocab_llama3_bpe.gguf` (Llama 3 scheme, 16,384 entries,
C:/ml_artifacts/capacity/vocab/), "8k" is `fineweb_tinychat_bpe8k.gguf` (paperkiln BPE, 8,192).
Rows marked "Colab" were built by `tools/colab_chat/human_dialogue_v1.json` (session pk-hd, 23 Oct
2026). The 13 tiny-chat sources hold 170,572 dialogues, 176.0 MB of rendered text and 40.1M tokens
of the 16k vocabulary.

## 2. Candidates surveyed

Contamination column: "pre-ChatGPT" means collected before ChatGPT's release (30 Nov 2022) by
people with no access to a capable chat model; see section 3 for the later ones.

| Source | Who wrote it | Licence (primary page) | Size | Depth | Content | LLM contamination | Verdict |
|---|---|---|---|---|---|---|---|
| OpenAssistant oasst1 + oasst2 | Volunteers on open-assistant.io, both roles (13,500+ for oasst1) | Apache-2.0 (card, LICENSE) | oasst2: 13,854 ready trees, 135,174 messages; oasst1: 10,364 ready trees | Trees; best branch 2-6 turns | Assistant-style Q&A, many topics | Guidelines forbade pasting ChatGPT; offenders banned; tell phrases searched; the paper says "these mechanisms cannot remove all ChatGPT-generated content". 8,576 synthetic messages in the full dump (none in the HF parquet; checked). Hundreds of "as an AI language model" assistant messages remain in the parquet | DEFAULT, with the synthetic flag, a tell filter and tell-avoiding branch choice; residual risk flagged |
| ConvAI2 / PersonaChat | Paired Mechanical Turk workers playing assigned personas (2018) | CC BY 4.0 (ParlAI ConvAI2 README; PersonaChat's own README names none) | 10,907 dialogues, 162,064 utterances (PersonaChat paper) | ~15 turns | Getting-to-know-you chit-chat | Pre-ChatGPT | DEFAULT (Colab: 430 MB archive) |
| ConvAI2 human-bot logs (HF conv_ai_2) | Volunteers talking to 2018 bots | Unknown (card) | 3.5k dialogues | | Human-to-bot | Bot turns are model output | EXCLUDED |
| Synthetic-Persona-Chat (google) | LLM generator-critic | CC BY 4.0 | 20k conversations | | PersonaChat-style | LLM-generated (card) | EXCLUDED |
| Wizard of Wikipedia | Crowd workers: apprentice and a wizard who reads Wikipedia sentences | CC BY 4.0 (ParlAI task README; the HF mirror chujiezheng says CC BY-NC 4.0, its own label) | 22k dialogues, 201k utterances (ParlAI README) | ~9 turns | Knowledge-grounded chat over 1,365 topics | Pre-ChatGPT | DEFAULT (Colab: 972 MB archive) |
| Blended Skill Talk | Paired crowd workers; the guided worker may adopt a suggestion from retrieval models over ConvAI2/ED/WoW utterances (20.5% of guided turns, 2004.08449) | CC BY 4.0 (ParlAI task README; HF card says unknown) | 7k conversations (train 4,819) | ~11 turns | Persona + empathy + knowledge | Pre-ChatGPT; suggestions are retrieved human utterances, not generated | DEFAULT (seed utterances from other datasets left out) |
| Topical-Chat | Paired Mechanical Turk workers with reading sets | CDLA-Sharing-1.0 (repo DATALICENSE; section 3.5: no obligations on Results) | 10,784 conversations, 235,281 utterances (train 8,628) | 21.8 turns | Knowledge-grounded chat, 8 topics | Pre-ChatGPT | OPT-IN (share-alike on the data only) |
| DailyDialog | "Human-written" daily-life dialogues (card; the card does not name the writers) | CC BY-NC-SA 4.0 (card) | 13,118 dialogues (train 11,118) | 7.5 turns (measured) | Everyday life | Pre-ChatGPT | OPT-IN (NC, SA) |
| EmpatheticDialogues | Paired crowd workers: speaker describes a situation, listener responds | CC BY-NC 4.0 (repo LICENSE and HF card); ParlAI README says CC BY 4.0: conflict, the stricter reading taken | 25k conversations (76,673 train utterances) | 4.1 turns (measured) | Emotional situations | Pre-ChatGPT | OPT-IN (NC) |
| MultiWOZ 2.2 | Crowd workers, Wizard-of-Oz (both sides human) | MIT (repo README and LICENSE; HF card says Apache-2.0) | 10,438 dialogues (train 8,438) | Multi-turn | Multi-domain booking and information (README) | Pre-ChatGPT | DEFAULT (Colab: 172 MB of JSON) |
| Taskmaster-1, -2, -3 | TM-1: two-person spoken WOz (crowd users, trained call-centre operators) and written self-dialogues; TM-2: spoken WOz; TM-3: self-dialogues | CC BY 4.0 (each README) | 13,215 + 17,289 + 23,789 dialogues | Multi-turn | Ordering, booking, search, movie tickets | Pre-ChatGPT (2019-2020) | DEFAULT (Colab: 0.99 GB of JSON); TM-1 dev/test ids excluded; v1 reads TM-1 and TM-2 only, TM-3 is `human_taskmaster3_v1` (section 9) |
| Taskmaster-4 | Crowd self-dialogues for a 2024 paper | CC BY 4.0 | 6,500 dialogues | | Coffee ordering | Crowd text of the LLM era with no stated control (section 3) | EXCLUDED |
| Schema-Guided Dialogue | Simulator outlines paraphrased by crowd workers | CC BY-SA 4.0 (card) | 16,142 train dialogues | | Task-oriented | Dialogue flow machine-made (not an LLM) | EXCLUDED (template flow, SA) |
| Self-dialogue Corpus | One Mechanical Turk worker writes both sides (2017) | BSD-3-Clause (repo LICENSE) | 24,165 conversations, 141,945 turns, 3.65M words (README) | 15.4 utterances (measured; the README counts turns differently) | Movies, music, sport | Pre-ChatGPT | DEFAULT |
| QuAC | Crowd workers: a student asks, a teacher answers with spans of a hidden Wikipedia section | CC BY-SA 4.0 (quac.ai; HF card says MIT) | 11,567 train dialogues | 11.9 turns after removals (measured) | Wikipedia Q&A | Pre-ChatGPT | OPT-IN (SA) |
| CoQA | Crowd workers about passages | Per domain: CC BY-SA 4.0, MSR-LA, Apache, others (card) | 8k conversations | | Passage Q&A with terse answers | Pre-ChatGPT | EXCLUDED (mixed licences; answers refer to an unseen passage) |
| Commonsense-Dialogues (alexa) | One crowd worker writes both roles, 4-6 turns (card) | CC BY-NC 4.0 (repo LICENSE) | ~11k dialogues (train 9,058) | 4-6 turns | Social situations from SocialIQA | Pre-ChatGPT | Not implemented; a further NC candidate |
| CMU_DoG | Crowd workers about movie articles | No licence file in the repo | 4,112 conversations, 21.4 turns | | Document-grounded | Pre-ChatGPT | EXCLUDED (no licence) |
| databricks-dolly-15k | Databricks employees | CC BY-SA 3.0 (card) | 15,011 records | Single turn | Instructions in 8 categories; some context copied from Wikipedia | LLM era (written "to exhibit the magical interactivity of ChatGPT"), but contributors were "explicitly instructed to avoid using generative AI" (card) | OPT-IN (SA); tell filter |
| No Robots | "Skilled human annotators" (card) | CC BY-NC 4.0 (card) | 10,000 (train 9,500) | Mostly single turn (Chat 850) | InstructGPT-style categories | LLM era (the card benchmarks against MT-Bench and AlpacaEval); its premise is no GPT text, but it states no control mechanism | OPT-IN (NC); tell filter |
| Aya dataset (English originals) | Volunteer annotators on the Aya platform (dataset paper 2402.06619) | Apache-2.0 (card) | 2,164 English originals (+1,780 re-annotations of automatically generated pairs) | Single turn | Mixed Q&A | LLM era; neither card nor paper (2402.06619) states a control; sampled replies read like ChatGPT ("plays a vital role ... meticulously crafted") | OPT-IN (llm-risk); re-annotations never used |
| Cornell Movie-Dialogs | Screenwriters (617 film scripts) | None stated (ConvoKit page) | 220,579 exchanges, 10,292 character pairs | | Film dialogue | Pre-ChatGPT | EXCLUDED (no licence; copyrighted scripts) |
| CANDOR | Recorded conversations, transcribed (BetterUp) | Research use; access by application (BetterUp page; ConvoKit does not redistribute) | 1,600+ conversations, 850+ hours | Long | Natural spoken conversation | Human speech | EXCLUDED (application, research-only) |
| Switchboard-1 | Telephone conversations, transcribed | LDC User Agreement for Non-Members, fee (LDC catalogue) | ~2,400 conversations, 260 hours | Long | Telephone speech | Human speech | EXCLUDED (paid licence) |
| Reddit (pushshift and derived sets) | Reddit users | Reddit Data API Terms: no right to use user content "for training a machine learning or AI model, without the express permission of rightsholders" | Very large | Threads | Everything | Post-2022 text may hold LLM output | EXCLUDED (terms) |
| Ubuntu Dialogue Corpus | IRC users | Unknown for the data (HF card); creator script Apache-2.0 | ~1M dialogues | Multi-party | Ubuntu tech support | Pre-ChatGPT | EXCLUDED (no data licence; multi-party IRC) |
| SODA, UltraChat-200k, smol-smoltalk, everyday-conversations, TinyChat | InstructGPT; ChatGPT; SmolTalk's synthetic sets; Llama-3.1-70B-Instruct; GPT-4o mini (each card) | CC BY 4.0; MIT; Apache-2.0; Apache-2.0; CC BY-NC 4.0 | | | | Model-generated | EXCLUDED (the rule) |
| Simple English Wikipedia (knowledge) | Wikipedia editors | CC BY-SA 3.0 and GFDL (card) | 157 MB parquet (20231101.simple) | Documents | Encyclopaedia in plain English | The 20231101 dump postdates ChatGPT; edits made after it may hold LLM text | OPT-IN (SA): "a bit of knowledge" |

## 3. LLM contamination: the rules applied

1. **Model-generated datasets** are excluded outright, by their own cards.
2. **Collected before 30 Nov 2022** (ChatGPT's release): treated as human-written. Retrieval
   suggestions over human utterances (BST) are human text.
3. **Collected after it.** Veselovsky, Horta Ribeiro and West (2306.07899) estimate that "33-46% of
   crowd workers used LLMs" on a text-production task. So later crowd text is used only when the
   collector states a control against LLM use (oasst: guidelines, bans, tell searches; Dolly:
   contributors instructed not to use generative AI), and even then every dialogue holding a
   ChatGPT tell phrase is dropped (`TELLS`: "as an AI language model", "as a large language model",
   "knowledge cutoff", "September 2021", "OpenAI", "ChatGPT" and near variants, after the oasst
   paper's own filters). Later crowd text with no stated control (Aya, Taskmaster-4) is excluded or
   held behind `--allow-llm-risk`.
4. **oasst residual risk.** The tell filter catches only text that announces itself. Paraphrased or
   lightly edited ChatGPT answers in oasst cannot be detected by phrase; the paper concedes this. If
   the rule is read strictly, oasst moves behind `--allow-llm-risk` too (decision 3, section 7).

## 4. Role mapping (2-party human dialogue onto user/assistant)

The chat template (contract point 3) is `<|user|>U<|assistant|>A<|endoftext|>` per exchange.

- **Asymmetric roles keep their meaning:** oasst prompter/assistant; Taskmaster USER/ASSISTANT;
  MultiWOZ USER/SYSTEM; Wizard of Wikipedia apprentice/wizard (the wizard holds the knowledge);
  QuAC student/teacher; EmpatheticDialogues speaker/listener; BST free/guided worker; instruction
  sets prompt/response. When the assistant side opens a dialogue (WoW has wizard-first dialogues;
  ParlAI's teacher handles them), its leading turn is dropped and counted
  (`leading_assistant_turns_trimmed`), so every dialogue starts with a user turn.
- **Symmetric chit-chat** (ConvAI2, Topical-Chat, self-dialogue, DailyDialog): the first speaker is
  the user and the speakers alternate. The "assistant" there is a peer who also asks questions and
  talks about itself; that is the register these sources teach. An alternative, rendering each
  dialogue twice with the roles swapped, would double these sources and is not done.
- **Same speaker twice in a row:** the turns are merged with a newline (`turns_merged_same_speaker`,
  counted per source; BST had one; oasst trees alternate by construction).
- **Three or more speakers:** the dialogue is dropped (`multi_party`). No default source has them;
  the multi-party sets (Ubuntu IRC) are excluded.
- **Short turns** ("ok", "lol", "Yes."): kept; they are how people talk, and their share is in each
  report (`short_turns_le2_words`). A dialogue needs one full exchange (`too_short`), and a trailing
  unanswered user turn is trimmed (`_common.normalise_dialogue`).
- **Light normalisation:** ConvAI2's lowercased, space-tokenised text is detokenised and its sentence
  starts and the pronoun "I" capitalised (proper nouns stay lower case); DailyDialog is detokenised;
  ParlAI's `__SILENCE__` and empty utterances are removed; EmpatheticDialogues' `_comma_` becomes a
  comma; Dolly's bracketed citation numbers are removed (its card's advice); QuAC's unanswerable
  questions are removed with their CANNOTANSWER reply, and its first question carries the page and
  section title.

## 5. Processing and records

Per source, in order: reader (train split only where the source defines splits), role mapping, the
licence and LLM-risk gate, `_common.normalise_dialogue` (template markers and instruction-bearing
system turns drop the dialogue), the tell filter for LLM-era sources, within-source duplicate check
(`_common.dialogue_key`), `corpus/decontam.py` against `simple_qa_v1`, `tinychat_hf_prompts_v1` and
`chat_prompts_v1`. Carry-5 (`carry5_v1`, its user turns and gold replies) joins only with
`--carry5`, because the v1 corpora were pinned before it was frozen; the tiny-chat data job applies
it where the corpora become training data.
Every drop has a reason in `report.json`; a run of several sources writes
`human_dialogue_v1.ledger.json` with exact duplicates across sources (counted, not removed).
Downloads are deleted after a build (`--keep-raw` keeps them); Hugging Face files stay in the HF
cache. Pins: `corpus/hashes/human_<source>_v1.sha256`.

`tools/carry5_build.py decontam` over the 13 tiny-chat sources and Simple English Wikipedia
(395,169 rows; `experiments/tiny_chat_2026-10/carry5_decontam_human_v1.json`): no item leaks, 44
items' fact sentences somewhere (everyday sentences), 6 Carry-5 user turns as whole user turns in
52 dialogues, which the tiny-chat data job drops (`tools/colab_chat/tiny_chat_data.py`).

## 6. Build status

**Built on the laptop and pinned** (Windows Python under `scripts/cpu_budget.py`, 1 core, about
15 minutes in all; about 350 MB of downloads, deleted after the builds except the HF cache), nine
corpora, 88 MB of `text.txt`. Drops by reason (full lists in each `report.json`):

| Corpus | Rows in | Written | Dropped (reason: count) |
|---|---:|---:|---|
| oasst | 13,854 trees | 5,328 | not English or not ready 8,453; llm_tell 50; too_short 15; decontaminated 8 |
| bst | 4,819 | 4,819 | none (1 empty utterance removed, 1 same-speaker merge) |
| self_dialogue | 24,249 | 24,133 | rejected or blocked worker 84; duplicate 22; too_short 9; empty_turn 1 |
| dolly | 15,011 | 14,945 | llm_tell 47; duplicate 17; decontaminated 2 |
| quac | 11,567 | 11,567 | none (14,459 unanswerable QA pairs removed inside dialogues) |
| empathetic | 17,844 | 17,780 | too_short 64 |
| dailydialog | 11,118 | 10,235 | duplicate 878; decontaminated 5 |
| no_robots | 9,500 | 8,698 | system_instruction 794; llm_tell 5; decontaminated 2; non_alternating 1 |
| aya_en | 202,362 | 2,103 | not English 198,418; re-annotation 1,780; duplicate 56; llm_tell 3; decontaminated 2 |

Exact duplicates across sources (`human_dialogue_v1.ledger.json`): 1 dolly+oasst, 6 aya_en+dolly
(Aya "original" items that are Dolly records, another sign of copied rather than written text).
Answer flags (a simple-QA fact near its question's words; reported, not dropped): dolly 221,
no_robots 185, oasst 124, dailydialog 42, aya_en 36, self_dialogue 33, bst 25, quac 18,
empathetic 12. Short turns of two words or fewer: self_dialogue 19,085 of 372,450, dailydialog
4,008, quac 4,662. Measured depth: self_dialogue 15.4 turns per dialogue, quac 11.9, dailydialog
7.5, bst 11.2, empathetic 4.1, oasst 3.1; the instruction sets are one exchange.

**Built on Colab and pinned** (`tools/colab_chat/human_dialogue_v1.json`, session pk-hd, 23 Oct
2026): ConvAI2, Wizard of Wikipedia, Taskmaster, MultiWOZ, Topical-Chat and Simple English
Wikipedia. The VM's rebuilds of the laptop's corpora equal their pins byte for byte. The files stay
where the runner put them, outside OneDrive (`C:/ml_artifacts/human_dialogue/hd_default` and
`hd_optin`), pinned there with `human_dialogue.py pin SOURCE --out-dir DIR`; the two ParlAI archives
are pinned by sha256 (`ARCHIVE_SHA256`). Drops: ConvAI2 8,908 perspective duplicates (each dialogue
appears from both speakers) and 19 decontaminated; Wizard of Wikipedia 2 decontaminated; MultiWOZ
132 decontaminated and 13 duplicates; Topical-Chat 23 decontaminated; Simple English Wikipedia
17,173 articles under 20 words.

**Taskmaster v1 holds Taskmaster-1 and -2 only.** Taskmaster-3 labels its speakers `user` and
`assistant` in lower case and the reader maps `USER` and `ASSISTANT`, so all 23,757 TM-3 dialogues
drop as `unknown_speaker`. Taskmaster-3 has its own corpus id, `human_taskmaster3_v1` (section 9):
its free self-dialogues, without the "Auto template" dialogues whose agent turns come fixed from the
collectors' templates. Using it in a pre-registered study needs that study's amendment.

The tiny-chat data job (`tools/colab_chat/tiny_chat_data.py`) builds the 13 sources on the VM with
`--allow-sa --allow-nc` and refuses unless every file equals its pin (`--require-pinned`).

## 7. Decisions (Jonathan, 23 Oct 2026)

Share-alike and non-commercial sources are in; FineWeb-Edu counts as human-written; OpenAssistant
counts as clean enough; Aya stays behind `--allow-llm-risk` and out of the tiny-chat pool; first
speaker as user stands. The questions as they were put:

1. **Share-alike.** Topical-Chat's CDLA-Sharing-1.0 puts no obligation on models (section 3.5); the
   share-alike binds only a republished copy of the data. Recommendation: switch it on; it is the
   largest pre-ChatGPT open-domain human chat set with a usable licence. Dolly, QuAC and Simple
   English Wikipedia are CC BY-SA: whether a trained model is an adaptation of its training text is
   unsettled; many released models train on Wikipedia. Simple English Wikipedia is the only
   "bit of knowledge" source in the survey.
2. **Non-commercial.** EmpatheticDialogues, DailyDialog, No Robots: fine for research runs; a model
   trained on them inherits NC terms if released (as with TinyChat today).
3. **oasst's residual ChatGPT risk** (section 3, point 4): keep it in the default set (current) or move
   it behind `--allow-llm-risk`.
4. **Aya English originals:** behind `--allow-llm-risk` (current) or dropped.
5. **Role mapping for symmetric chit-chat:** first speaker as user (current), or both orders.
6. **Launch** `tools/colab_chat/human_dialogue_v1.json` (CPU data jobs; can ride along with the next
   Colab session).

## 8. Primary sources read (23 Oct 2026)

- Hugging Face dataset cards at the pinned revisions in `human_dialogue.py`: OpenAssistant/oasst1,
  oasst2; allenai/soda; HuggingFaceH4/ultrachat_200k; HuggingFaceTB/smol-smoltalk;
  HuggingFaceTB/everyday-conversations-llama3.1-2k; starhopp3r/TinyChat; google/Synthetic-Persona-Chat;
  convai-challenge/conv_ai_2; ParlAI/blended_skill_talk; facebook/empathetic_dialogues;
  li2017dailydialog/daily_dialog; pfb30/multi_woz_v22; google-research-datasets/taskmaster1-3 and
  schema_guided_dstc8; databricks/databricks-dolly-15k; HuggingFaceH4/no_robots;
  CohereLabs/aya_dataset; allenai/quac; stanfordnlp/coqa; gretelai/commonsense-dialogues;
  ubuntu-dialogs-corpus/ubuntu_dialogs_corpus; stanfordnlp/craigslist_bargains;
  chujiezheng/wizard_of_wikipedia; bavard/personachat_truecased; wikimedia/wikipedia.
- ParlAI task READMEs (github.com/facebookresearch/ParlAI, parlai/tasks/{personachat, convai2,
  wizard_of_wikipedia, blended_skill_talk, empathetic_dialogues, cornell_movie}/README.md).
- Licence files: alexa/Topical-Chat DATALICENSE; alexa/Commonsense-Dialogues LICENSE;
  facebookresearch/EmpatheticDialogues LICENSE; budzianowski/multiwoz LICENSE and README;
  google-research-datasets/Taskmaster TM-1..TM-4 READMEs; jfainberg/self_dialogue_corpus LICENSE
  and README; festvox/datasets-CMU_DoG (no licence file); rkadlec/ubuntu-ranking-dataset-creator
  LICENSE; quac.ai.
- Papers: OpenAssistant Conversations (2304.07327, section 4 and appendix D); PersonaChat
  (1801.07243); Wizard of Wikipedia (1811.01241); Blended Skill Talk (2004.08449); Aya Dataset
  (2402.06619); Veselovsky et al. (2306.07899).
- Terms: Reddit Data API Terms (redditinc.com/policies/data-api-terms, sections 2.4 and 3.2);
  BetterUp CANDOR page and ConvoKit's CANDOR page; ConvoKit's Cornell Movie-Dialogs page; LDC
  catalogue LDC97S62 (Switchboard-1 Release 2).
- Read on 24 Oct 2026 for section 9: google-research-datasets/Taskmaster TM-3-2020/README.md and
  splits/{train,dev,test}/README.md at commit d92cb6af, and TM-3-2020/data/data_00.json itself;
  "Beyond Goldfish Memory" (Xu, Szlam and Weston, arXiv 2107.07567 and ACL 2022 long 356, section 3,
  tables 1-2, appendix B, ethics statement); ParlAI parlai/tasks/msc/{build.py, agents.py},
  parlai/tasks/task_list.py (msc entry), projects/msc/README.md, the repository README and LICENSE
  (MIT, "ParlAI is MIT licensed"), parlai/tasks/{wizard_of_internet, casino}/README.md, and the
  msc_v0.1 archive's file list; ABCD (Chen et al., arXiv 2104.00783, sections 4-5, appendix A, ethics
  statement) and asappresearch/abcd README and LICENSE (MIT) at commit 6b8700ce;
  google-research-datasets/ccpe README (CC BY 4.0) at commit 2c9cd30f; Persuasion for Good (Wang et
  al., arXiv 1906.06725, section 3 and table 2) and gitlab.com/ucdavisnlp/persuasionforgood README,
  LICENSE (Apache-2.0) and data/FullData/readme.md at commit 90a3fd7b; Wizard of the Internet
  (Komeili et al., arXiv 2107.07566, section 4 and table 1); Conversation Chronicles (Jang et al.,
  arXiv 2310.13420, section 3 and footnote 2) and the jihyoung/ConversationChronicles card; LoCoMo
  (Maharana et al., arXiv 2402.17753, abstract and section 3) and snap-research/locomo README;
  DuLeMon (Xu et al., arXiv 2203.05797). Hugging Face mirrors of MSC (gonced8/multi-session_chat
  tagged GPL-3.0, nayohan/multi_session_chat untagged) carry their uploaders' labels, not the
  authors' licence.

## 9. Long human dialogue (24 Oct 2026)

For the long-carry programme (`experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md`
section 4, option 3): sources whose dialogues reach 12 or more turns (a turn is one exchange, as
`tools/dialogue_lengths.py` counts). Same rules as above, with Jonathan's decisions of 23 Oct:
share-alike and non-commercial licences are allowed; anything written by a language model is
excluded; human-written text only, no distillation; separate dialogues are never concatenated into
one conversation. Fetcher group `long` of `human_dialogue.py` (outside default/optin/all, so the
v1 corpora and their pins are unchanged; a rebuild of bst, dailydialog and empathetic with the new
code equals their pins byte for byte); ledger `human_dialogue_long_v1.ledger.json`; Colab job
`tools/colab_chat/human_dialogue_long_v1.json`.

| Source | Who wrote it | Licence (primary page) | Size | Depth | Content | LLM contamination | Verdict |
|---|---|---|---|---|---|---|---|
| Taskmaster-3 (TicketTalk) | One crowd worker writes both the customer and the ticketing agent (self-dialog). "Auto template" dialogues (scenario field, either case) are partly pre-written: the agent's turns are fixed lines of the collectors' template with slot values ("force assistant" in the instructions) and the worker writes only the customer's turns (README; checked in data_00.json) | CC BY 4.0 (TM-3 README) | 23,789 dialogues (README); data_00.json: 1,188, of which 344 Auto template (29%) | data_00.json, free dialogues as built: median 10 turns, max 15; 46 of 836 reach 12, 3 reach 14, none 20 | Buying movie tickets | Pre-ChatGPT (2020; instructions mention COVID-19 closures) | PASS for the free self-dialogues: `human_taskmaster3_v1`, default gate; Auto template dialogues drop (`auto_template`: their agent turns are template text, not the worker's), dev/test ids drop (`held_out_split`, from splits/dev and splits/test). Built by the Colab job (0.4 GB with the split files) |
| Multi-Session Chat (MSC) | Mechanical Turk workers (over 1,000, English-speaking, US; about six months, released July 2021). Session 1 is PersonaChat; sessions 2-5 were written later by workers playing the same two personas, shown the earlier sessions' crowd-written summaries and dialogue; "these crowdworkers may not be the same ones that played those characters in previous sessions" (paper, section 3). Session openings are human annotations (paper, figure 4). No model text in the dialogue files; the archive's `summaries_subsample5.json` files are model-predicted summaries (ParlAI agents.py) and are not read | **Not stated.** No licence in the paper, the archive, the ParlAI task folder (no README, unlike ConvAI2, WoW and BST, CC BY 4.0) or the task list; the ParlAI LICENSE (MIT) covers the code. Session 1 is PersonaChat (CC BY 4.0, ParlAI ConvAI2 README) | Train 4,000 episodes (2,999 end at session 3, 1,001 at session 4); valid 500 and test 501 five-session episodes (msc_v0.1, 51 MB, sha256 pinned by ParlAI's build.py) | Built: median 20 turns, max 32; all 3,989 reach 12 and 14, 2,306 reach 20 | Persona chit-chat, picking up earlier sessions after hours or days | Pre-ChatGPT | Human-written: PASS. `human_msc_v1`, behind `--allow-unstated-licence` until Jonathan rules on the licence. Train episodes only. Built and pinned on the laptop |
| ABCD (Action-Based Conversations) | Expert Live Chat: a trained crowd agent (qualification exams, guidelines, live coaching) and a crowd customer, synchronous, "all participants are human – no wizard" (paper, section 4.2); customer scenarios and their synthetic names and order numbers come from the collectors | MIT (repository LICENSE) | 10,042 conversations (train 8,034), 22.1 utterances on average with the agent's action lines (paper, table 2) | Built: median 6 turns, max 22 (texting style: 35,486 consecutive messages by one speaker merged); 242 reach 12, 70 reach 14, 1 reaches 20 | Retail customer service | Pre-ChatGPT (NAACL 2021) | PASS: `human_abcd_v1` (default gate); action lines removed. Built and pinned on the laptop |
| CCPE-M | Two paid crowd workers, Wizard-of-Oz: the "assistant" types, eliciting movie preferences; the "user" speaks and is transcribed (README) | CC BY 4.0 (README) | 502 dialogues, 11,971 utterances | Built: median 9 turns, max 25; 119 reach 12, 52 reach 14, 6 reach 20 | Movie preferences | Pre-ChatGPT (SIGDIAL 2019) | PASS: `human_ccpe_v1` (default gate). The assistant opens 492 of 502 dialogues; that question is trimmed by the role rule (section 4). Built and pinned on the laptop |
| PersuasionForGood | Paired Mechanical Turk workers, a persuader and a persuadee, at least 10 turns each; collected over two months (paper, section 3), in 2018 (dialogue ids) | Apache-2.0 (repository LICENSE) | 1,017 dialogues (paper, table 2) | Built: median 10 turns, max 15; 25 reach 12, 1 reaches 14 | Persuasion to donate to Save the Children | Pre-ChatGPT | PASS: `human_p4g_v1` (default gate); persuader (speaks first) = user. Built and pinned on the laptop |
| Taskmaster-2 | Spoken Wizard-of-Oz | CC BY 4.0 | 17,289 dialogues | | Seven domains | Pre-ChatGPT | Already in `human_taskmaster_v1` and in the 6,429 below: no new supply |
| Wizard of the Internet | Crowd apprentice and a wizard who searches the internet | CC BY 4.0 (ParlAI task README) | 9,633 dialogues; 1.3 GB archive | 5-6 turns by design, 9.6 utterances per dialogue (paper, table 1) | Knowledge-grounded chat | Pre-ChatGPT (2021) | Passes the rule, adds no 12+-turn dialogue: not implemented |
| Conversation Chronicles | ChatGPT ("We use ChatGPT", paper, footnote 2) | CC BY 4.0 (card) | 200,000 episodes of five sessions | Long | Multi-session chat | Model-generated | EXCLUDED (the rule) |
| LoCoMo | LLM agents generate, annotators edit for consistency (paper, abstract) | None in the README | 50 conversations (10 released), about 300 turns each | Very long | Multi-session chat with images | Model-generated | EXCLUDED (the rule) |
| DuLeMon | Long-term persona chat (paper) | | | Multi-session | Chinese | | EXCLUDED (Chinese; the tokeniser, template and evaluations are English) |
| CANDOR, Switchboard-1, Reddit threads, SODA | Section 2 | Section 2 | | Long | | | Still EXCLUDED for the section 2 reasons (access by application and research-only; paid LDC licence; Reddit's terms and multi-party threads; model-generated); length does not change a licence |

**Joining MSC sessions (decision for Jonathan).** An episode is the record of its last session,
which carries sessions 1..k-1 in `previous_dialogs`: the same two characters continuing, each later
session written as a continuation of the earlier ones. The sessions are joined in order with no
marker between them; the time gaps (hours or days) stay in the source's metadata, because any marker
would be text no participant wrote. Speakers are positional, as ParlAI's msc teacher reads them:
utterances 0, 2, ... of every session are Speaker 1, who opens each session (464 of the 4,000
session-2 records give the opener a numeric id instead of "Speaker 1"; the rest alternate). Speaker 1
is the user. When a session ends on Speaker 1 (an odd number of utterances), Speaker 1 also opens the
next: the two messages become one user turn joined by a newline (303 joins in 4,000 episodes,
`msc_session_joins_same_speaker`). In 47 places a session's text differs slightly between its own
record and the later record's copy (case, a removed phrase); the last record's copy is used. Three
points for Jonathan:

1. **Is a multi-session episode one conversation?** The characters are the same and each session
   continues the last, but the workers behind a character may change between sessions (paper,
   section 3). The build treats it as one conversation; the alternative is to train on single
   sessions only, which leaves no long dialogue (a session is about 6 turns; session 1 about 7).
2. **The licence is unstated.** Comparable ParlAI datasets of the same group are CC BY 4.0, but
   nothing says so for MSC. Until he rules, `human_msc_v1` sits behind `--allow-unstated-licence`
   (CMU_DoG, with no licence file, is excluded in section 2).
3. **Session 1 duplicates ConvAI2.** 3,987 of the 3,989 MSC episodes open with a dialogue already
   in `human_convai2_v1` (first two messages equal after word normalisation). Using both shows that
   text twice and lets it fall on both sides of the data job's hash split (the MSC episode and the
   ConvAI2 dialogue hash differently). One option: where MSC is used, drop the matching ConvAI2
   dialogues in the data job. MSC's valid and test episodes (1,001, five sessions, about 33 turns)
   are not read, by the train-split rule.

**Counts.** A turn is one exchange; `tools/dialogue_lengths.py` normalises as the data job does.
Measured on the laptop (`C:/ml_artifacts/human_dialogue/hd_long/dialogue_lengths_long_laptop_v1.json`,
16k vocabulary) except Taskmaster-3, which is an estimate from one of its twenty data files:

| Corpus | Dialogues | Turns p50 / p90 / max | Reach 12 | Reach 14 | Reach 20 | SFT split reach 12 | 16k tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| msc (measured) | 3,989 | 20 / 26 / 32 | 3,989 | 3,989 | 2,306 | 447 | 4,267,234 |
| abcd (measured) | 8,019 | 6 / 10 / 22 | 242 | 70 | 1 | 24 | 1,720,656 |
| ccpe (measured) | 502 | 9 / 14 / 25 | 119 | 52 | 6 | 10 | 199,292 |
| p4g (measured) | 1,011 | 10 / 11 / 15 | 25 | 1 | 0 | 2 | 467,402 |
| **four measured** | **13,521** | | **4,375** | **4,112** | **2,313** | **483** | **6,654,584** |
| taskmaster3 (ESTIMATE: data_00.json, 836 built, 46 / 3 / 0, times 20 files times the train share 0.8) | about 13,400 | 10 / 11 / 15 | about 740 | about 50 | 0 | | |
| **current 13 sources** | 170,572 | | 6,429 | 2,814 | 306 | 668 | |
| **with the four measured** | 184,093 | | 10,804 | 6,926 | 2,619 | 1,151 | |
| **without msc** | 180,104 | | 6,815 | 2,937 | 313 | 704 | |

The real Taskmaster-3 counts come from the Colab job's `dialogue_lengths_long_v1.json`. MSC is
nearly all of the gain, and almost every MSC turn from 12 on lies in sessions 2-4: the long-range supply
depends on decisions 1-3 above.
