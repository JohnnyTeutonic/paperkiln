# Recovered S-full raw human cache

The exact cache at `/home/jonat/r3_data/out/human` contains all thirteen
registered `dialogues.jsonl` files and their thirteen companion `text.txt` files.
Every dialogue hash equals the frozen S-full reference ledger; every dialogue
and text hash also equals its pin in archive `7edb0ca1`. This extends the earlier
eight-source exact-path presence check without changing or copying any data.

`presence_check.py.txt` is the read-only executed checker; `presence.json` binds
its bytes, the original reference and every inspected file. A second parent
replay used an explicit one-megabyte streaming SHA-256 loop, independently
rehashing all 26 raw files and verifying the saved checker digest: PASS.

The full runtime gate remains open. In this cache only `panel_dialogue.ids`
matches the registered derived outputs. `sft_human.jsonl` and
`panel_fineweb.ids` differ, and the long SFT and training mix are absent. There
is no original `data.ledger.json` here. The small FineWeb input also differs
from the original source hash and must not be used to reconstruct scientific
inputs. No provenance is inferred for these unmatched products.

No download, rebuild, training or paid resource was started. The scientific
recipe, expected products, decontamination checks and historical parity
qualification remain unchanged. This closes raw-source availability in this
cache, not complete runtime readiness or funding/scientific review.
