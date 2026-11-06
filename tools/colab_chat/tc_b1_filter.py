#!/usr/bin/env python3
"""Filter B's dialogue sources before the mix (stronger_base_spec.md, section 10). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b1_filter.py --repo . --heldout human_val.txt human_sft.jsonl --out DIR \
        --source msc=long/human_msc_v1/dialogues.jsonl [--source ...]

The long sets and lever 2's dialogue sets were built without Carry-5 and Carry-12, and the two held-out splits of the human
sets (the dialogue panel and the SFT rows) know nothing of them: Multi-Session Chat's first session is PersonaChat, the
same text as ConvAI2's, so an episode can repeat a held-out dialogue. Each source's dialogues are therefore read once more:

- dropped when corpus/decontam.py's rules hit against the frozen eval sets, Carry-5 and Carry-12;
- dropped when the dialogue shares a word 13-gram with a held-out dialogue (val or SFT), the same 13 words as those rules.

Kept dialogues are rendered as their source's text.txt renders them. Outputs: DIR/<source>.txt and DIR/filter.ledger.json
(per source its licence, gate and whether its text is model-generated, counts by rule and sha256 of every input and output; no
text). Deterministic: rows in file order, nothing random.
"""
import argparse
import hashlib
import json
import os
import re
import sys

N = 13
SPLIT = re.compile(r"[^0-9a-z]+")
MARKERS = ("<|user|>", "<|assistant|>", "<|endoftext|>")
# (licence, model-generated), for the sources this mix reads from dialogue_pt.py; human_dialogue.py's own table has the rest.
LEVER2 = {"soda": ("CC BY 4.0", True), "ultrachat": ("MIT; ChatGPT output", True), "everyday": ("Apache-2.0; Llama-3.1-70B-Instruct", True)}


def words(text):
    return [w for w in SPLIT.split(text.lower()) if w]


def grams(ws):
    return zip(*[ws[i:] for i in range(N)])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def heldout_grams(val_txt, sft_jsonl):
    """Every word 13-gram of the rendered val split and of the SFT dialogues, over a dialogue's words alone. The markers must go
    first: words() would keep "user" and "assistant" in the grams, and a source's bare text could then never meet a gram that
    crosses a turn. A document ends at the separator with its newline, so a val dialogue stays whole across its exchanges."""
    table = set()
    with open(val_txt, encoding="utf-8") as f:
        for doc in f.read().split("<|endoftext|>\n"):
            for m in MARKERS:
                doc = doc.replace(m, " ")
            table.update(grams(words(doc)))
    with open(sft_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                table.update(grams(words("\n".join(m["content"] for m in json.loads(line)["messages"]))))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--heldout", nargs=2, required=True, metavar=("VAL_TXT", "SFT_JSONL"))
    ap.add_argument("--source", action="append", required=True, help="NAME=DIALOGUES_JSONL (repeat)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(a.repo, "corpus", "fetchers"))
    sys.path.insert(0, os.path.join(a.repo, "corpus"))
    import _common as C
    import human_dialogue
    os.makedirs(a.out, exist_ok=True)
    dc, carry5_sha = human_dialogue.decontaminator(carry5=True, carry12=True)
    if not carry5_sha:
        raise SystemExit("Carry-5 is missing")
    held = heldout_grams(*a.heldout)
    ledger = {"recipe": "tools/colab_chat/tc_b1_filter.py", "carry5_v1_sha256": carry5_sha,
              "carry12_v1_sha256": human_dialogue.carry12_sha(), "heldout_grams": len(held),
              "heldout": {os.path.basename(p): sha256(p) for p in a.heldout}, "sources": {}}
    for item in a.source:
        name, path = item.split("=", 1)
        out = os.path.join(a.out, f"{name}.txt")
        n = kept = drop_eval = drop_held = both = 0
        with open(path, encoding="utf-8") as f, open(out + ".tmp", "w", encoding="utf-8", newline="\n") as g:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                msgs = json.loads(line)["messages"]
                text = "\n".join(m["content"] for m in msgs)
                d, _ = dc.check([m["content"] for m in msgs if m["role"] == "user"], text)
                ws = words(text)
                h = len(ws) >= N and not held.isdisjoint(grams(ws))
                drop_eval += bool(d)
                drop_held += h
                both += bool(d) and h
                if d or h:
                    continue
                kept += 1
                g.write(C.render_dialogue(msgs))
        os.replace(out + ".tmp", out)
        gate, licence = "lever 2", LEVER2.get(name, (None, None))[0]
        generated = LEVER2.get(name, (None, False))[1]
        if name in human_dialogue.SOURCES:
            gate, licence = human_dialogue.SOURCES[name][0], human_dialogue.SOURCES[name][1]
        ledger["sources"][name] = {"input": os.path.basename(os.path.dirname(path)), "input_sha256": sha256(path),
                                   "gate": gate, "licence": licence, "model_generated": generated,
                                   "rows_in": n, "rows_kept": kept, "dropped_eval_rules": drop_eval,
                                   "dropped_heldout_13gram": drop_held, "dropped_both": both,
                                   "bytes": os.path.getsize(out), "sha256": sha256(out)}
        print(json.dumps({name: ledger["sources"][name]}), flush=True)
    with open(os.path.join(a.out, "filter.ledger.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(ledger, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
