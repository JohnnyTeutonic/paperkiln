# Provenance: where the cited evidence lives

The research programme works on branches: its registry, protocols and run scripts are not merged
into master. A link to a branch shows whatever the branch holds today. A link to a commit shows a
file as it stood when the result was read, so the README cites commits. This page lists each pin,
what it holds and which branch it came from, then the live branches and the latest results.

## Pinned evidence

Each pin was checked on 13 November 2026: the commit exists on GitHub, the file is present at it,
and the commit is on the branch named (on that date it was the branch's head).

| What it holds | File | Commit | Branch |
|---|---|---|---|
| Base A's shape and schedule: the race's 8 × 768 and 12 × 768 arms, 60,000 steps of 16 × 1,024 positions | [`tools/colab_chat/make_tc_b1.py`](https://github.com/JohnnyTeutonic/paperkiln/blob/ab15db97f1d7d4f032c4b66915e808b69824247f/tools/colab_chat/make_tc_b1.py) | `ab15db97` (9 Nov 2026) | `claude/tc-b1-sft1500-20261109` |
| The programme registry: the model (section 2), measurement (3), every screen's verdict A6 to A29 (4), the judge run (4b), the capability frontier (7), the compute ledger (8), retractions (9), the next authorised step (10b) | [`experiments/tiny_chat_programme/TINY_CHAT_RESEARCH_PROGRAMME.md`](https://github.com/JohnnyTeutonic/paperkiln/blob/36523b27c93223a48cb10503adfc8bc6bf0a8c23/experiments/tiny_chat_programme/TINY_CHAT_RESEARCH_PROGRAMME.md) | `36523b27` (11 Nov 2026, A28's verdict) | `claude/chat-screens` |
| Base 2's arm N: 18 × 512, 69,751,296 parameters, 250,000 steps of 16,384 positions | [`tools/colab_chat/base2_N.json`](https://github.com/JohnnyTeutonic/paperkiln/blob/428b60f0329de3dced1606e1f83fa73acc66be93/tools/colab_chat/base2_N.json) | `428b60f0` (12 Nov 2026) | `claude/base2-commit-budget` |

A pinned commit stays reachable on GitHub while a branch or tag contains it. Each pin has a
tag, so it outlives its branch: `research/base-a-recipe-20261109`,
`research/programme-registry-20261111` and `research/base2-graph-20261113` (13 November 2026).
To check a pin again:

```bash
git fetch origin claude/chat-screens
git merge-base --is-ancestor 36523b27c93223a48cb10503adfc8bc6bf0a8c23 origin/claude/chat-screens && echo pinned
```

## Live branches

These move. Use them for what came after a pin, and cite a commit when quoting them.

- [`claude/chat-screens`](https://github.com/JohnnyTeutonic/paperkiln/tree/claude/chat-screens):
  the registry and the chat screens.
- [`claude/base2-commit-budget`](https://github.com/JohnnyTeutonic/paperkiln/tree/claude/base2-commit-budget):
  base 2.
- [`claude/tc-b1-sft1500-20261109`](https://github.com/JohnnyTeutonic/paperkiln/tree/claude/tc-b1-sft1500-20261109):
  base A and its first SFT.

## Latest results

- **The current chat models:** [CAPABILITY.md](CAPABILITY.md), the measured record, with every
  number's instrument, denominator and receipt.
- **The latest registered verdicts** (registry section 4 at `36523b27`): A26 `ready` (S71), A27
  `relevance_cost`, A28 `wide_only` (S71 at a second seed), A29 `base_up` (the same recipe on an
  open base).
- **Results kept in this tree:** [size and conversational memory](../experiments/tiny_chat_2026-10/size/RESULTS.md),
  [Carry-12](../experiments/tiny_chat_long_2026-10/RESULTS.md),
  [pk_base_v1 against references](../experiments/benchmarks_2026-10-26/RESULTS.md),
  [pk_sft_v1](../experiments/chat_general/RESULTS.md) and the
  [findings registry](../atlas/FINDINGS.md), retractions included.
- **The arXiv extractor:** [`2026-10-27_bench_scores.json`](../papers/llm_extract_receipts/2026-10-27_bench_scores.json),
  read in [FEATURES.md](FEATURES.md#what-the-extraction-benchmark-measured).
- **The ready-made browser model** that `quickstart_chat.sh` downloads: release
  [`chat-v1`](https://github.com/JohnnyTeutonic/paperkiln/releases/tag/chat-v1), asset
  `tinychat-v2-web.tar.gz`, 1,728,786 bytes, SHA-256
  `16ae917498bc3a30e816b140e2fbced1d6b8a62426607a077e9fb1f1e9dd0fb2`. The script pins that digest
  and checks it before unpacking.
