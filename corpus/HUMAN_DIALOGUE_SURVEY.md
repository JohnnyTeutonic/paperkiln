# Human-written dialogue for tiny chat: source survey (23 Oct 2026)

**Rule (Jonathan, 23 Oct 2026).** For the tiny-chat research, no text written by a large language
model enters training data: no distillation. Every claim below was read on 23 Oct 2026 from the
primary page named in section 8 (dataset card, paper, licence file or terms page), not from memory.
Fetcher: `corpus/fetchers/human_dialogue.py`; selection record: `ledger/0005-human-dialogue.md`.

## 1. Recommended set

On by default (permissive licence, human authorship documented):

| Source | Written by | Licence | Built | Dialogues kept | Turns | 16k tokens | 8k tokens |
|---|---|---|---|---:|---:|---:|---:|
| OpenAssistant oasst2 + oasst1 | Volunteers, both roles | Apache-2.0 | laptop, pinned | 5,328 | 16,490 | 2,425,273 | 2,909,441 |
| Blended Skill Talk | Paired crowd workers | CC BY 4.0 | laptop, pinned | 4,819 | 54,034 | 1,014,819 | 1,075,603 |
| Self-dialogue Corpus | One crowd worker, both sides | BSD-3-Clause | laptop, pinned | 24,133 | 372,450 | 5,527,322 | 5,919,493 |
| ConvAI2 / PersonaChat | Paired crowd workers | CC BY 4.0 | Colab | (10,907 dialogues, 162,064 utterances, all splits) | | | |
| Wizard of Wikipedia | Crowd apprentice + wizard | CC BY 4.0 | Colab | (22k dialogues, 201k utterances, all splits) | | | |
| Taskmaster-1, -2, -3 | Crowd users with call-centre operators; crowd self-dialogues | CC BY 4.0 | Colab | (54,293 dialogues) | | | |
| MultiWOZ 2.2 | Crowd workers, both sides (Wizard-of-Oz) | MIT | Colab | (8,438 train dialogues) | | | |
| **Built so far** | | | | **34,280** | **442,974** | **8,967,414** | **9,904,537** |

Off by default, for Jonathan's decision (section 7):

| Source | Gate | Written by | Licence | Built | Dialogues kept | Turns | 16k tokens | 8k tokens |
|---|---|---|---|---|---:|---:|---:|---:|
| Topical-Chat | share-alike | Paired crowd workers | CDLA-Sharing-1.0 | Colab | (8,628 train conversations, 188,378 utterances) | | | |
| databricks-dolly-15k | share-alike | Databricks employees | CC BY-SA 3.0 | laptop, pinned | 14,945 | 29,890 | 2,921,410 | 3,372,933 |
| QuAC | share-alike | Crowd student + teacher | CC BY-SA 4.0 | laptop, pinned | 11,567 | 138,218 | 2,571,107 | 2,920,693 |
| Simple English Wikipedia | share-alike | Wikipedia editors | CC BY-SA, GFDL | Colab | (157 MB parquet) | documents | | |
| EmpatheticDialogues | non-commercial | Paired crowd workers | CC BY-NC 4.0 | laptop, pinned | 17,780 | 73,258 | 1,361,911 | 1,432,972 |
| DailyDialog | non-commercial | Human-written (card) | CC BY-NC-SA 4.0 | laptop, pinned | 10,235 | 76,770 | 1,273,731 | 1,347,962 |
| No Robots | non-commercial | Skilled annotators (card) | CC BY-NC 4.0 | laptop, pinned | 8,698 | 17,396 | 2,765,925 | 3,164,023 |
| Aya, English originals | LLM risk | Volunteer annotators | Apache-2.0 | laptop, pinned | 2,103 | 4,206 | 207,879 | 242,728 |

Token counts are of each corpus's `text.txt` in the stack's chat template (markers included):
"16k" is `vocab_llama3_bpe.gguf` (Llama 3 scheme, 16,384 entries,
C:/ml_artifacts/capacity/vocab/), "8k" is `fineweb_tinychat_bpe8k.gguf` (paperkiln BPE, 8,192).
Rows marked "Colab" are prepared as `tools/colab_chat/human_dialogue_v1.json` and not launched;
their sizes are the dataset's own counts until the job reports.

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
| Taskmaster-1, -2, -3 | TM-1: two-person spoken WOz (crowd users, trained call-centre operators) and written self-dialogues; TM-2: spoken WOz; TM-3: self-dialogues | CC BY 4.0 (each README) | 13,215 + 17,289 + 23,789 dialogues | Multi-turn | Ordering, booking, search, movie tickets | Pre-ChatGPT (2019-2020) | DEFAULT (Colab: 0.99 GB of JSON); TM-1 dev/test ids excluded |
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
(`_common.dialogue_key`), `corpus/decontam.py` against `simple_qa_v1`, `tinychat_hf_prompts_v1`,
`chat_prompts_v1` and `carry5_v1` (its user turns and gold replies, read when the file exists).
Every drop has a reason in `report.json`; a run of several sources writes
`human_dialogue_v1.ledger.json` with exact duplicates across sources (counted, not removed).
Downloads are deleted after a build (`--keep-raw` keeps them); Hugging Face files stay in the HF
cache. Pins: `corpus/hashes/human_<source>_v1.sha256`.

Carry-5 (`corpus/eval/carry5_v1.jsonl`) did not exist when the local corpora were built and pinned;
when it is frozen, run `python tools/carry5_build.py decontam --out REPORT.json
corpus/data/human_*_v1/text.txt` (its template-span and item-leak rules), and the Colab job reads it
automatically if it is committed before launch.

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

**Prepared for Colab, not launched:** `tools/colab_chat/human_dialogue_v1.json`, two CPU data jobs.
`hd_default` builds the default set (~2.7 GB of downloads: Wizard of Wikipedia 0.97 GB, Taskmaster
0.99 GB, ConvAI2 0.43 GB, MultiWOZ 0.17 GB); `hd_optin` builds the gated sources (~0.4 GB, Topical-Chat
and Simple English Wikipedia new). Both count tokens with both vocabularies. The VM rebuilds the
laptop's nine corpora too; their files must match the laptop's pins. At home: copy each
`human_<source>_v1/` into `corpus/data/`, `python corpus/fetchers/human_dialogue.py pin SOURCE`
for the new ones, and write the two ParlAI archives' sha256 (in `report.json` inputs) into
`ARCHIVE_SHA256`. The readers for the Colab-only sources are tested on synthetic archives
(`tests/test_human_dialogue.py`), not yet on the real files.

The tiny-chat data job (`tools/colab_chat/tiny_chat_data.py`) calls `human_dialogue.py all --out-dir
{human}`: with no `--allow-*` flag, `all` builds the default set only.

## 7. Decisions for Jonathan

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
