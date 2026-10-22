# Carry-5 v1: build, contamination and sanity receipts (23 Oct 2026)

The set: `corpus/eval/carry5_v1.jsonl`, 1,032 items, sha256 `de5f8b6803efd334990e7fa879571d9777509980baf28cd81f570c390abc41a5`
(pinned in `corpus/hashes/carry5_v1.sha256`), built by `tools/carry5_build.py build` (seed 20261023) and
scored by `tools/carry5_eval.py`. Design: `experiments/tiny_chat_litcheck_2026-10-23.md` section 3.

| family | items | cells |
|---|---:|---|
| recall | 432 | 6 slots x fact turn 1-3 x target 4-5 x 0-2 distractors, 4 draws each |
| correction | 216 | 6 slots x (plant, correction) turns x target x 0-2 distractors, 2 draws each |
| instruction | 144 | one word / five words / yes-no / number x instruction turn 1-3 x target, 6 draws each |
| topic return | 180 | 6 topics x (fact, target) turns x 0-2 distractors, 2 draws each |
| knowledge | 60 | each question once (none repeats `simple_qa_v1`) |

`build_v1.json`: counts per family by slot, target turn, distance, distractors and gold position.

## Contamination (`contamination_v1.json`)

`python tools/carry5_build.py decontam --out REPORT.json CORPUS...` over 24 local corpora, 2,031,617 rows:
TinyChat (hf sample train/val + pretraining text, v1_1500, v2_1500, v2_4000), everyday_v1, smol-smoltalk
train/val/test, the FineWeb-Edu sample, lever 1 (`lever1_sft_v1.jsonl`), lever 2 (oasst, SODA, UltraChat) and
the nine `corpus/data/human_*_v1/text.txt` corpora (aya_en, bst, dailydialog, dolly, empathetic, no_robots,
oasst, quac, self_dialogue). Rules: decontam.py's normalisation, 13-gram, user-turn equality (4+ words) and
8-12 word span; plus every template sentence as a span, every value-bearing fact sentence, and item leaks
(a fact sentence and its question in one training row).

- Item leaks: 0. 13-gram hits: 0. Span hits: 0.
- User turns equal to a training user turn: 5 (SODA 4, self_dialogue 1): the opener "Hi there! How are you
  today?" and generic fact statements ("My name is Tomas.", "I am 18 years old.", "I work as an engineer.",
  "I work as an accountant."), whose items ask for a value the training row cannot supply.
- Fact sentences found anywhere in training (without the question): 52 items, all generic statements.
- Eight content questions (instruction and knowledge) found verbatim in training in earlier drafts were
  replaced before freezing (square sides, seasons in a year, "Is ice cold?", "Is fire hot?", hexagon sides,
  bicycle wheels, dog legs, "What do you write with?").
- Pending: the Colab-built human corpora in `C:/ml_artifacts/human_dialogue/` (only ledgers there on
  23 Oct). When their text arrives: `python tools/carry5_build.py decontam --out
  experiments/carry5_2026-10-23/contamination_colab_human.json <their text.txt files>`.

## Sanity runs (`sanity/`)

The live model (`chat_sftfinal_v3`, 14.6M, 8k BPE, ctx 256; GGUF sha256 a9d5ba5e..., safetensors b29120f5...)
and the chance baseline (`chance=uniform`), CPU only, WSL; inputs hashed in `sanity/inputs.sha256`.

- `fc_all/`: forced choice on all 1,032 items with `mtstudio score` (gold, twin, question alone).
- `gen7/`: every 7th item (148; a stride coprime with the cell structure) with ember 338b24a (CPU) greedy
  generation, self history and the parity gate; `summary_ir.md` re-scores it with the numpy IR.
- `gate/`: `mtstudio chat` greedy vs `mtstudio score` argmax on 40 gold-history prompts: 40/40.
- `ir_crosscheck.json`: mtstudio score vs numpy IR on gen7: the same choice in 592/592 item-modes (raw and
  per token), max |difference of summed log-probs| 4.6e-5; both reproduce all 740 greedy replies.

Forced choice, raw log-prob accuracy (chance 0.25):

| run | mode | all | recall | correction | instruction | topic | knowledge |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_all (1,032) | gold | 0.336 | 0.241 | 0.241 | 0.854 | 0.261 | 0.350 |
| fc_all | twin | 0.346 | 0.259 | 0.213 | 0.875 | 0.289 | 0.350 |
| fc_all | question alone | 0.341 | 0.259 | 0.236 | 0.826 | 0.250 | 0.417 |
| gen7 (148) | gold | 0.291 | 0.194 | 0.226 | 0.810 | 0.160 | 0.333 |
| gen7 | self | 0.338 | 0.274 | 0.258 | 0.857 | 0.120 | 0.444 |
| gen7 | twin | 0.351 | 0.274 | 0.258 | 0.810 | 0.280 | 0.333 |
| chance | every mode | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 |

Reading: the live model carries nothing measurable. On the memory families (recall, correction, topic) it
is at chance in every mode, the single-turn twin included, so its failure is reading the fact at all, not
carrying it across turns; gold and self history do not differ beyond noise (gen7 gold 0.291 [0.223, 0.368],
self 0.338 [0.267, 0.417]). The instruction family's high raw accuracy is a length prior (the compliant
answer is the shortest candidate): the question alone gives 0.826, and per-token accuracy is near 0. Read
instruction-following from the generation check, where the live model complies 0/21 in every mode. Greedy
generation contains the gold answer in 1-3% of target replies ("My name is Assistant, ..."); clean stops
are 0.93-1.00; the fifth turn of the self conversation ends empty 11.5% of the time.
