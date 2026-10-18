# corpus/eval — frozen evaluation sets

*Every file here is frozen at first generation: its sha256 is pinned in `corpus/hashes/<set>.sha256` and
`tests/test_tinyarith.py` fails if a byte changes. A changed set is a new version (`_v2`); v1 stays.
Regenerate and verify with `python corpus/fetchers/tinyarith.py`.*

Contamination is measured, not asserted (16 Oct 2026):

- **fact overlap**: test/dev arithmetic facts (op, a, b) that occur in training: measured below; 0 by construction of the split.
- **8-gram overlap**: fraction of a set's word 8-grams that occur anywhere in its level's `train.txt`. High by design: train and evaluation share templates and names, and only the facts differ. This is why fact overlap, not n-gram overlap, is the contamination that matters here.
- Against other corpora: the sets contain no text from TinyStories, TinyChat or any public benchmark (generated; templates written here).

| set | items | sha256 (first 16) | fact overlap with train | 8-gram overlap with train |
|---|---:|---|---:|---:|
| `tinyarith_r10_dev_v1` | 18 | `2ff3a989e1ee3667` | 0/18 | 0.425 |
| `tinyarith_r10_test_v1` | 18 | `4fd27f8865fe2303` | 0/18 | 0.427 |
| `tinyarith_r10_hackprobe_v1` | 10 | `61046f68f82c23ce` | 0/10 | 0.285 |
| `tinyarith_r20_dev_v1` | 67 | `cb5d72bcc34102a7` | 0/67 | 0.433 |
| `tinyarith_r20_test_v1` | 67 | `e8e17726e2cee4c2` | 0/67 | 0.445 |
| `tinyarith_r20_hackprobe_v1` | 42 | `6584db0624577ae9` | 0/42 | 0.281 |
| `tinyarith_r50_dev_v1` | 392 | `657bc4d2f9b60875` | 0/392 | 0.525 |
| `tinyarith_r50_test_v1` | 392 | `c813393356889953` | 0/392 | 0.544 |
| `tinyarith_r50_hackprobe_v1` | 263 | `d622a881690b53fe` | 0/263 | 0.306 |

`*_hackprobe_v1` items are addition facts from the test split, phrased with a distractor quantity.
Each carries four canned completions and the verdict each verifier must give; `tests/test_tinyarith.py`
checks those verdicts against the C++ verifiers. Two of the four are format-pass, truth-fail: the
reward-hacking signal on fixed inputs, which is the negative control for the PT4 monitor.

## Held-out text (bits-per-byte panel)

Built by `corpus/fetchers/eval_heldout.py`; overlap measured into `heldout_overlap_v1.json`.

| set | what | exact-line overlap (worst corpus) | 8-gram overlap (worst corpus) |
|---|---|---:|---:|
| `tinystories_heldout_v1` | TinyStories train stories 100000-100499 (shard 0, rev f54c09fd); no registered corpus draws on them | 0.000 | 0.020 (`tinystories_valid_v1`) |
| `tinychat_heldout_v1` | 500 TinyChat v2 dialogues at seed 999 (no corpus uses that seed) | 0.004 | 0.822 (`tinychat_v2_4000`) |

TinyChat is a closed world: 82% of the held-out set's 8-grams, and 0.4% of its dialogues verbatim,
occur in `tinychat_v2_4000`. On TinyChat, held-out loss measures how well the world was learned, not
generalisation beyond it. Read it that way.

**No instruction-following set exists yet.** It waits on the instruction corpus, which is blocked: TinyStories-Instruct
declares no licence (ledger 0002).

## Chat

| set | what |
|---|---|
| `tinychat_hf_prompts_v1` | first user turn and reference reply of the first 200 TinyChat (starhopp3r) test dialogues, built by `tinychat_hf.py eval_v1 --pin` from the pinned source; the chat model's frozen prompts (`tools/chat_eval.py`) |
| `simple_qa_v1` | 60 hand-written questions with accepted answers (25 everyday facts, 15 basic science, 10 small arithmetic, 10 conversational turns), written before any model saw them; scored by whether a reply contains an accepted answer. It asks what TinyChat cannot teach: whether general-text pretraining bought simple Q&A (`experiments/chat_general/PLAN.md`) |
