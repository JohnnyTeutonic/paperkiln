#!/usr/bin/env python3
"""Standard small-model benchmarks as one item file (LOCAL_LLM_STACK.md section 11, item 5.2).

    python -I tools/bench_build.py fetch --raw DIR          # download the pinned files, check sha256
    python -I tools/bench_build.py build --raw DIR --out items.jsonl [--blimp-per 1000]
    python -I tools/bench_build.py pin --raw DIR            # write the sha256s into the pin file (once)

Sources (the pin file, experiments/benchmarks_2026-10-26/sources.json, holds every URL at a fixed
revision, its licence and the sha256 of each downloaded file and of the built item file):
  hellaswag    Rowan/hellaswag validation, 10,042 items, 4 endings (MIT)
  arc_easy     allenai/ai2_arc ARC-Easy test, 2,376 (CC BY-SA 4.0)
  arc_challenge allenai/ai2_arc ARC-Challenge test, 1,172 (CC BY-SA 4.0)
  piqa         ybisk/ybisk.github.io piqa/data valid, 1,838, 2 solutions (AFL 3.0)
  winogrande   allenai/winogrande winogrande_xl validation, 1,267 (Apache 2.0, the upstream repository)
  lambada      EleutherAI/lambada_openai test, 5,153 (OpenAI's modified MIT; LAMBADA CC BY 4.0)
  blimp        nyu-mll/blimp, 67 paradigms x 1,000 minimal pairs (CC BY 4.0)
The data files are downloaded data: they stay in their own directory (--raw), are read only by this
script under `python -I`, and are never committed (ARC's share-alike licence among them); the item
file is rebuilt from them on any machine and checked against the pinned sha256.

Items (one JSON object per line, tokeniser-free text; tools/bench_eval.py tokenises per model):
  {"id", "set", "kind": "mc", "context", "choices": [continuation...], "gold"}
      the choices are scored as continuations of the context (the delimiter is inside them);
  {"id", "set", "kind": "partial", "contexts": [ctx...], "continuation", "gold"}
      WinoGrande's partial scoring: one continuation after each filled-in context;
  {"id", "set", "kind": "lastword", "context", "target"}
      LAMBADA: correct when every target token is the argmax (greedy last-word accuracy).
Formatting follows lm-evaluation-harness's zero-shot tasks (hellaswag's preprocess, "Question: ...
\\nAnswer:" for ARC and PIQA, winogrande's partial evaluation, lambada_openai, blimp's whole-sentence
comparison as a choice between the two sentences with an empty context).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIN = os.path.join(ROOT, "experiments", "benchmarks_2026-10-26", "sources.json")

HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
GH = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"
REV = {"Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
       "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
       "ybisk/ybisk.github.io": "50ac265b6469095b0db861b0a8e20833ed5b25b7",
       "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
       "EleutherAI/lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
       "nyu-mll/blimp": "877fba0801ffb7cbd8c39c1ff314a46f053f6036"}
BLIMP = ("adjunct_island anaphor_gender_agreement anaphor_number_agreement animate_subject_passive animate_subject_trans "
         "causative complex_NP_island coordinate_structure_constraint_complex_left_branch "
         "coordinate_structure_constraint_object_extraction determiner_noun_agreement_1 determiner_noun_agreement_2 "
         "determiner_noun_agreement_irregular_1 determiner_noun_agreement_irregular_2 "
         "determiner_noun_agreement_with_adj_2 determiner_noun_agreement_with_adj_irregular_1 "
         "determiner_noun_agreement_with_adj_irregular_2 determiner_noun_agreement_with_adjective_1 "
         "distractor_agreement_relational_noun distractor_agreement_relative_clause drop_argument ellipsis_n_bar_1 "
         "ellipsis_n_bar_2 existential_there_object_raising existential_there_quantifiers_1 "
         "existential_there_quantifiers_2 existential_there_subject_raising expletive_it_object_raising inchoative "
         "intransitive irregular_past_participle_adjectives irregular_past_participle_verbs "
         "irregular_plural_subject_verb_agreement_1 irregular_plural_subject_verb_agreement_2 "
         "left_branch_island_echo_question left_branch_island_simple_question matrix_question_npi_licensor_present "
         "npi_present_1 npi_present_2 only_npi_licensor_present only_npi_scope passive_1 passive_2 "
         "principle_A_c_command principle_A_case_1 principle_A_case_2 principle_A_domain_1 principle_A_domain_2 "
         "principle_A_domain_3 principle_A_reconstruction regular_plural_subject_verb_agreement_1 "
         "regular_plural_subject_verb_agreement_2 sentential_negation_npi_licensor_present "
         "sentential_negation_npi_scope sentential_subject_island superlative_quantifiers_1 "
         "superlative_quantifiers_2 tough_vs_raising_1 tough_vs_raising_2 transitive wh_island "
         "wh_questions_object_gap wh_questions_subject_gap wh_questions_subject_gap_long_distance wh_vs_that_no_gap "
         "wh_vs_that_no_gap_long_distance wh_vs_that_with_gap wh_vs_that_with_gap_long_distance").split()


def files():
    """[(local name, url)] of every pinned file."""
    out = [("hellaswag/validation.parquet", HF.format(repo="Rowan/hellaswag", rev=REV["Rowan/hellaswag"],
                                                      path="data/validation-00000-of-00001.parquet"))]
    for s in ("ARC-Easy", "ARC-Challenge"):
        out.append((f"arc/{s}_test.parquet", HF.format(repo="allenai/ai2_arc", rev=REV["allenai/ai2_arc"],
                                                       path=f"{s}/test-00000-of-00001.parquet")))
    for f in ("valid.jsonl", "valid-labels.lst"):
        out.append((f"piqa/{f}", GH.format(repo="ybisk/ybisk.github.io", rev=REV["ybisk/ybisk.github.io"],
                                           path=f"piqa/data/{f}")))
    out.append(("winogrande/xl_validation.parquet",
                HF.format(repo="allenai/winogrande", rev=REV["allenai/winogrande"],
                          path="winogrande_xl/validation-00000-of-00001.parquet")))
    out.append(("lambada/lambada_test.jsonl", HF.format(repo="EleutherAI/lambada_openai",
                                                        rev=REV["EleutherAI/lambada_openai"],
                                                        path="data/lambada_test.jsonl")))
    for p in BLIMP:
        out.append((f"blimp/{p}.parquet", HF.format(repo="nyu-mll/blimp", rev=REV["nyu-mll/blimp"],
                                                     path=f"{p}/train-00000-of-00001.parquet")))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pin():
    return json.load(open(PIN, encoding="utf-8"))


def cmd_fetch(a):
    pin = load_pin()["files"] if os.path.exists(PIN) else {}
    bad = 0
    for name, url in files():
        dst = os.path.join(a.raw, name)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".tmp", "wb") as f:
                        f.write(r.read())
                    break
                except OSError as e:
                    if attempt == 3:
                        sys.exit(f"{name}: {url}: {e}")
            os.replace(dst + ".tmp", dst)
        want = pin.get(name, {}).get("sha256")
        got = sha256(dst)
        if want and want != got:
            print(f"{name}: sha256 {got} is not the pinned {want}", file=sys.stderr)
            bad += 1
    if bad:
        sys.exit(f"{bad} files differ from the pin")
    print(f"{len(files())} files in {a.raw}" + (", every sha256 as pinned" if pin else " (no pin yet)"))


# ---------------------------------------------------------------- per-set builders
def parquet_rows(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:  # python -I drops the user's site-packages, where pyarrow may live; the data dir stays off sys.path
        import site
        site.addsitedir(site.getusersitepackages())
        import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def hs_pre(text):
    """lm-evaluation-harness hellaswag.preprocess."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def build_hellaswag(raw):
    for r in parquet_rows(os.path.join(raw, "hellaswag/validation.parquet")):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        yield {"id": f"hellaswag/{r['ind']}", "set": "hellaswag", "kind": "mc",
               "context": hs_pre(r["activity_label"] + ": " + ctx),
               "choices": [" " + hs_pre(e) for e in r["endings"]], "gold": int(r["label"])}


def build_arc(raw, which):
    s = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}[which]
    for r in parquet_rows(os.path.join(raw, f"arc/{s}_test.parquet")):
        labels = list(r["choices"]["label"])
        yield {"id": f"{which}/{r['id']}", "set": which, "kind": "mc",
               "context": f"Question: {r['question']}\nAnswer:",
               "choices": [" " + t for t in r["choices"]["text"]], "gold": labels.index(r["answerKey"])}


def build_piqa(raw):
    rows = [json.loads(x) for x in open(os.path.join(raw, "piqa/valid.jsonl"), encoding="utf-8") if x.strip()]
    labels = [int(x) for x in open(os.path.join(raw, "piqa/valid-labels.lst"), encoding="utf-8").read().split()]
    if len(rows) != len(labels):
        sys.exit(f"piqa: {len(rows)} rows, {len(labels)} labels")
    for k, (r, g) in enumerate(zip(rows, labels)):
        yield {"id": f"piqa/{k}", "set": "piqa", "kind": "mc", "context": f"Question: {r['goal']}\nAnswer:",
               "choices": [" " + r["sol1"], " " + r["sol2"]], "gold": g}


def build_winogrande(raw):
    for k, r in enumerate(parquet_rows(os.path.join(raw, "winogrande/xl_validation.parquet"))):
        s = r["sentence"]
        i = s.index("_")
        yield {"id": f"winogrande/{k}", "set": "winogrande", "kind": "partial",
               "contexts": [s[:i] + r["option1"], s[:i] + r["option2"]],
               "continuation": " " + s[i + 1:].strip(), "gold": int(r["answer"]) - 1}


def build_lambada(raw):
    rows = [json.loads(x) for x in open(os.path.join(raw, "lambada/lambada_test.jsonl"), encoding="utf-8")
            if x.strip()]
    for k, r in enumerate(rows):
        ctx, last = r["text"].rsplit(" ", 1)
        yield {"id": f"lambada/{k}", "set": "lambada", "kind": "lastword", "context": ctx, "target": " " + last}


def build_blimp(raw, per):
    for p in BLIMP:
        for k, r in enumerate(parquet_rows(os.path.join(raw, f"blimp/{p}.parquet"))[:per]):
            yield {"id": f"blimp/{p}/{r['pair_id']}", "set": "blimp", "paradigm": p, "kind": "mc", "context": "",
                   "choices": [r["sentence_good"], r["sentence_bad"]], "gold": 0}


def build(raw, per):
    yield from build_hellaswag(raw)
    yield from build_arc(raw, "arc_easy")
    yield from build_arc(raw, "arc_challenge")
    yield from build_piqa(raw)
    yield from build_winogrande(raw)
    yield from build_lambada(raw)
    yield from build_blimp(raw, per)


def cmd_build(a):
    tmp = a.out + ".tmp"
    counts = {}
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for it in build(a.raw, a.blimp_per):
            counts[it["set"]] = counts.get(it["set"], 0) + 1
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    got = sha256(a.out)
    print(json.dumps({"items": a.out, "sha256": got, "counts": counts}))
    if os.path.exists(PIN) and a.blimp_per == 1000:
        want = load_pin().get("items_sha256")
        if want and want != got:
            sys.exit(f"items sha256 {got} is not the pinned {want}")


def cmd_pin(a):
    d = load_pin()
    d["files"] = {name: {"url": url, "sha256": sha256(os.path.join(a.raw, name))} for name, url in files()}
    tmp = os.path.join(a.raw, "_pin_items.jsonl")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        counts = {}
        for it in build(a.raw, 1000):
            counts[it["set"]] = counts.get(it["set"], 0) + 1
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    d["items_sha256"] = sha256(tmp)
    d["counts"] = counts
    os.remove(tmp)
    with open(PIN + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(PIN + ".tmp", PIN)
    print(json.dumps({"items_sha256": d["items_sha256"], "counts": counts}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--raw", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--blimp-per", type=int, default=1000, help="pairs per BLiMP paradigm (the first N)")
    p = sub.add_parser("pin")
    p.add_argument("--raw", required=True)
    a = ap.parse_args(argv)
    {"fetch": cmd_fetch, "build": cmd_build, "pin": cmd_pin}[a.cmd](a)


if __name__ == "__main__":
    main()
