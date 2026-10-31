# Partial local S-full input check

Eight registered raw human-data files and the vocabulary match the original ledger.
Five exact raw source paths were absent from the checked local cache; the five
registered derived products were absent from each of two landed data folders.
The independent check rehashed every available file and verified the scoped absences.

This is a bounded local presence/hash check. It is not a machine-wide search,
the complete registered runtime/decontamination gate, or a resolution of historical
parity or funding. No source download, rebuild, training, paid resource or launch
occurred. `presence.json` names every checked path and expected digest;
`presence_check.py.txt` preserves the actual check source without test collection.
