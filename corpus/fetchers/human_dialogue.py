#!/usr/bin/env python3
"""Human-written dialogue for tiny-chat pretraining and SFT, with no text written by a large
language model (Jonathan's rule of 23 Oct 2026: no distillation in this research's training data).
Survey, licences and the role mapping: corpus/HUMAN_DIALOGUE_SURVEY.md.

    python corpus/fetchers/human_dialogue.py SOURCE[,SOURCE...]|default|optin|all [--allow-sa] [--allow-nc]
        [--allow-llm-risk] [--pin] [--out-dir DIR] [--count-vocab GGUF ...] [--keep-raw]
    python corpus/fetchers/human_dialogue.py pin SOURCE          # pin files brought home from a VM
    python corpus/fetchers/human_dialogue.py tokens SOURCE ... --count-vocab GGUF [...]

One corpus per source, id human_<source>_v1, in corpus/data/human_<source>_v1/ (or --out-dir):
  dialogues.jsonl   kept dialogues {"messages", "uri"} (documents.jsonl {"text", "uri"} for simplewiki)
  text.txt          one rendered document per dialogue (_common.render_dialogue), or per article
  report.json       counts by drop reason, inputs and their sha256, decontamination drops, answer
                    flags, turn statistics, token counts (--count-vocab)
A run of two or more sources also writes human_dialogue_v1.ledger.json beside the corpora.

Gates. DEFAULT (permissive licence, human authorship documented): oasst, convai2, wow, bst,
taskmaster, multiwoz, self_dialogue. SHARE-ALIKE (off; --allow-sa): topical_chat
(CDLA-Sharing-1.0), dolly (CC BY-SA 3.0), quac (CC BY-SA 4.0), simplewiki (CC BY-SA, GFDL).
NON-COMMERCIAL (off; --allow-nc): empathetic (CC BY-NC 4.0), dailydialog (CC BY-NC-SA 4.0),
no_robots (CC BY-NC 4.0). LLM RISK (off; --allow-llm-risk): aya_en (Apache-2.0, but crowd text of
2023 with no stated control against LLM use, and replies that read as ChatGPT's; survey section 3).

Role mapping (survey section 4). Sources with asymmetric roles keep them: oasst prompter/assistant,
Taskmaster and MultiWOZ USER/ASSISTANT|SYSTEM, Wizard of Wikipedia apprentice/wizard, QuAC
student/teacher, EmpatheticDialogues speaker/listener, BST free worker (who speaks first)/guided
worker, instruction sets prompt/response; a dialogue opened by the assistant side loses its leading
assistant turn(s) (counted). Symmetric chit-chat (ConvAI2, Topical-Chat, self-dialogue,
DailyDialog): the first speaker is the user and the speakers alternate. Every source: consecutive turns by one speaker are merged with a newline
(counted), empty utterances and ParlAI's __SILENCE__ are removed, a dialogue with three or more
speakers is dropped (multi_party), short turns ("ok", "lol") are kept, a dialogue needs one full
exchange (too_short). Only train splits are read where a source defines splits.

LLM-era sources (collected after ChatGPT's release, 30 Nov 2022: oasst, aya_en, dolly, no_robots)
drop any dialogue holding a ChatGPT tell phrase (TELLS; reason llm_tell); oasst also drops messages
flagged synthetic or carrying a model_name, and its branch choice skips replies with a tell.

Every dialogue then goes through _common.normalise_dialogue, a within-source duplicate check
(_common.dialogue_key), and corpus/decontam.py against the frozen eval sets. Carry-5
(corpus/eval/carry5_v1.jsonl) joins the targets only with --carry5: the v1 corpora were built and
pinned before Carry-5 was frozen, so a build that read it would no longer match its pin. Its rules
are applied where the corpora become training data (tools/colab_chat/tiny_chat_data.py), and
`tools/carry5_build.py decontam` scans the pinned outputs for its template and item-leak rules.

    python corpus/fetchers/human_dialogue.py pin SOURCE ... [--out-dir DIR]   # files in DIR/human_<s>_v1/

LONG DIALOGUE (24 Oct 2026; survey section 9, experiments/tiny_chat_long_2026-10 section 4): the group
`long` (taskmaster3, msc, abcd, ccpe, p4g) is outside default/optin/all, so those groups build what
they built before; a run of it writes human_dialogue_long_v1.ledger.json. taskmaster3 is Taskmaster-3
read with its lower-case speaker labels, without its "Auto template" dialogues (agent turns fixed by
the collectors' templates) and without the dialogues its splits put only in dev/test (2 at d92cb6af); msc (Multi-Session Chat) joins each
episode's sessions 1..k, the same two characters continuing, and is gated behind
--allow-unstated-licence (its authors state no data licence).
"""
from __future__ import annotations

import argparse
import codecs
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

sys.path.insert(0, C.CORPUS_DIR)
import decontam  # noqa: E402

ROOT = os.path.dirname(C.CORPUS_DIR)
CARRY5 = os.path.join(C.CORPUS_DIR, "eval", "carry5_v1.jsonl")
CARRY12 = os.path.join(C.CORPUS_DIR, "eval", "carry12_v1.jsonl")
LEDGER = "human_dialogue_v1.ledger.json"

# ---- pinned sources ---------------------------------------------------------
# Hugging Face: (repo, revision, files). Revisions are content addresses.
OASST2 = ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75",
          ["data/train-00000-of-00001-88ba0162028a73fc.parquet",
           "data/validation-00000-of-00001-1deeef95c3248fe0.parquet"])
OASST1 = ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b",
          ["data/train-00000-of-00001-b42a775f407cee45.parquet",
           "data/validation-00000-of-00001-134b8fd0c89408b6.parquet"])
BST = ("ParlAI/blended_skill_talk", "d7b0093243439fa5f0cd9663125cc47575ced2ea", ["data/train-00000-of-00001.parquet"])
AYA = ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2", ["data/train-00000-of-00001.parquet"])
DOLLY = ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", ["databricks-dolly-15k.jsonl"])
NO_ROBOTS = ("HuggingFaceH4/no_robots", "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b", ["data/train-00000-of-00001.parquet"])
# Script-era datasets: the Hub's own parquet conversion (refs/convert/parquet) at a pinned commit.
QUAC = ("allenai/quac", "768addbf2099da8efabd67d3a669f6483b813c1b", ["plain_text/train/0000.parquet"])
DAILYDIALOG = ("li2017dailydialog/daily_dialog", "469b7118f52336674f1d78693797e9fcc7e47d2a", ["default/train/0000.parquet"])
EMPATHETIC = ("facebook/empathetic_dialogues", "d5b57ae707b0b9a384af8ed50c043c608d597ca7", ["default/train/0000.parquet"])
SIMPLEWIKI = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", ["20231101.simple/train-00000-of-00001.parquet"])

# GitHub: (repo, commit). Files are read from raw.githubusercontent.com at the commit.
TASKMASTER = ("google-research-datasets/Taskmaster", "d92cb6af3005f1dc09c39e75e7daf4a04905e00b")
TM_FILES = (["TM-1-2019/self-dialogs.json", "TM-1-2019/woz-dialogs.json",
             "TM-1-2019/train-dev-test/dev.csv", "TM-1-2019/train-dev-test/test.csv"]
            + [f"TM-2-2020/data/{d}.json" for d in ("flights", "food-ordering", "hotels", "movies", "music",
                                                     "restaurant-search", "sports")]
            + [f"TM-3-2020/data/data_{k:02d}.json" for k in range(20)])
MULTIWOZ = ("budzianowski/multiwoz", "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b")
MWOZ_FILES = [f"data/MultiWOZ_2.2/train/dialogues_{k:03d}.json" for k in range(1, 18)]
SELF_DIALOGUE = ("jfainberg/self_dialogue_corpus", "113a1b0ff10308f1227c13a4aaded9bc85577816")
TOPICAL = ("alexa/Topical-Chat", "7c939229cbcf6f55f6977b341a5a2f2fe982d53f")

# ParlAI archives (no revision): pinned by sha256 (None until the first fetch records it).
CONVAI2_URL = "https://dl.fbaipublicfiles.com/parlai/convai2/convai2_fix_723.tgz"
WOW_URL = "https://dl.fbaipublicfiles.com/parlai/wizard_of_wikipedia/wizard_of_wikipedia.tgz"
ARCHIVE_SHA256 = {CONVAI2_URL: "d0ae89defe2fd0b0a4221eaa642a457d7d40cef475f54798119c7f3b8dd9361d",
                  WOW_URL: "2a549627a83fea745efa2076a41d1c0078ad002ab2b54eae6a4e3d3d66ae24b7"}  # pk-hd, 23 Oct 2026

# Long-dialogue sources (24 Oct 2026). Taskmaster-3 is read from the TASKMASTER commit above.
TM3_FILES = [f"TM-3-2020/data/data_{k:02d}.json" for k in range(20)]
TM3_HELD = [f"TM-3-2020/splits/{s}/{s}-a{c}.tsv" for s in ("dev", "test") for c in "abcd"]
# Taskmaster-3's splits are per turn (one context -> next-turn example per line, cut into 10,000-line
# shards aa, ab, ...), not per dialogue: at d92cb6af the train shards name 22,967 of the 23,757 dialogues
# and nearly every dev/test dialogue also has train turns. "Train splits only" therefore keeps a dialogue
# that train uses and drops only those whose ids appear in dev/test and never in train (2); the other
# 788 dialogues appear in no split file and are kept.
TM3_TRAIN = [f"TM-3-2020/splits/train/train-{a}{b}.tsv" for a, b in
             [("a", c) for c in "abcdefghijklmnopqrstuvwxyz"] + [("b", c) for c in "abc"]]
MSC_URL = "https://dl.fbaipublicfiles.com/parlai/msc/msc_v0.1.tar.gz"
ARCHIVE_SHA256[MSC_URL] = "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60"  # ParlAI build.py's pin; laptop, 24 Oct 2026
ABCD = ("asappresearch/abcd", "6b8700ce67c6b37b062dd7a60abc76d7ef832a97")
CCPE = ("google-research-datasets/ccpe", "2c9cd30f33f3a154b5a27d015333679262ff36f5")
P4G_URL = ("https://gitlab.com/ucdavisnlp/persuasionforgood/-/raw/90a3fd7b93437cec880e8f8192d26b144ecc493a/"
           "data/FullData/full_dialog.csv")
FILE_SHA256 = {"data/abcd_v1.1.json.gz": "2bdf53ac359543dcdc38d55bc6513e78df120363f8f44870716e909f4606de15",
               "data.json": "4ff051ea7ea60cf0f480c911c7e2cfed56434e2e2c9ea8965ac5e26365773f0a",
               P4G_URL: "3ecaa3e9f0577170b91c3306d2439c910150cf010b76e7637ef3bab724075a9f"}  # laptop, 24 Oct 2026

# Phrases that mark ChatGPT-style text: the OpenAssistant paper's own filters ("as a large language
# model", "knowledge cutoff after September 2021"; 2304.07327 appendix D) and their near variants.
TELLS = ("as an ai language model", "as a large language model", "as a language model", "as an ai assistant",
         "i am an ai", "i'm an ai", "as an ai,", "knowledge cutoff", "september 2021", "openai", "chatgpt",
         "gpt-3", "gpt-4")

SA, NC, RISK, DEFAULT = "share-alike", "non-commercial", "llm-risk", "default"
SOURCES = {
    # name: (gate, licence, role mapping, LLM-era collection, built where by default)
    "oasst": (DEFAULT, "Apache-2.0", "prompter=user, assistant=assistant", True, "local"),
    "convai2": (DEFAULT, "CC BY 4.0 (ParlAI task README)", "first speaker=user, alternating", False, "colab"),
    "wow": (DEFAULT, "CC BY 4.0 (ParlAI task README)", "apprentice=user, wizard=assistant", False, "colab"),
    "bst": (DEFAULT, "CC BY 4.0 (ParlAI task README)", "free worker (first)=user, guided worker=assistant", False, "local"),
    "taskmaster": (DEFAULT, "CC BY 4.0", "USER=user, ASSISTANT=assistant", False, "colab"),
    "multiwoz": (DEFAULT, "MIT (repository LICENSE; HF card says Apache-2.0)", "USER=user, SYSTEM=assistant", False, "colab"),
    "aya_en": (RISK, "Apache-2.0", "inputs=user, targets=assistant", True, "local"),
    "self_dialogue": (DEFAULT, "BSD-3-Clause (repository LICENSE)", "first speaker=user, alternating", False, "local"),
    "topical_chat": (SA, "CDLA-Sharing-1.0 (no obligations on Results, section 3.5)", "first speaker=user, alternating", False, "colab"),
    "dolly": (SA, "CC BY-SA 3.0", "instruction (+context)=user, response=assistant", True, "local"),
    "quac": (SA, "CC BY-SA 4.0 (quac.ai; HF card says MIT)", "student=user, teacher=assistant", False, "local"),
    "simplewiki": (SA, "CC BY-SA and GFDL (Wikipedia)", "plain documents, no roles", False, "colab"),
    "empathetic": (NC, "CC BY-NC 4.0 (repository LICENSE; ParlAI README says CC BY 4.0)", "speaker=user, listener=assistant", False, "local"),
    "dailydialog": (NC, "CC BY-NC-SA 4.0", "first speaker=user, alternating", False, "local"),
    "no_robots": (NC, "CC BY-NC 4.0", "user=user, assistant=assistant (system instructions drop)", True, "local"),
}
# Long-dialogue sources (survey section 9): the group `long`, never part of default/optin/all.
NOLIC = "licence-unstated"
LONG_SOURCES = {
    "taskmaster3": (DEFAULT, "CC BY 4.0 (TM-3 README)",
                    "user=user, assistant=assistant (lower-case labels); Auto template dialogues drop", False, "colab"),
    "msc": (NOLIC, "not stated by the authors (no ParlAI task README, nothing in the archive or paper); "
                   "session 1 is PersonaChat (CC BY 4.0 in ParlAI's ConvAI2 README)",
            "Speaker 1 (opens every session)=user, Speaker 2=assistant; sessions 1..k joined in order", False, "local"),
    "abcd": (DEFAULT, "MIT (repository LICENSE)", "customer=user, agent=assistant; action lines removed", False, "local"),
    "ccpe": (DEFAULT, "CC BY 4.0 (README)", "USER=user, ASSISTANT=assistant", False, "local"),
    "p4g": (DEFAULT, "Apache-2.0 (repository LICENSE)", "persuader (speaks first)=user, persuadee=assistant", False, "local"),
}
LONG_SET = list(LONG_SOURCES)
LONG_LEDGER = "human_dialogue_long_v1.ledger.json"
DEFAULT_SET = [s for s, v in SOURCES.items() if v[0] == DEFAULT]
OPTIN_SET = [s for s, v in SOURCES.items() if v[0] != DEFAULT]
SOURCES.update(LONG_SOURCES)


def names_for(source):
    return ["documents.jsonl", "text.txt"] if source == "simplewiki" else ["dialogues.jsonl", "text.txt"]


# ---- downloads --------------------------------------------------------------
INPUTS = []   # (uri, sha256) read by the current build


def _sha(path):
    return C.sha256_file(path)


def fetch_url(url, raw_dir, want=None):
    """Download url into raw_dir (atomic: a .part file, then rename); verify want if given."""
    os.makedirs(raw_dir, exist_ok=True)
    # short names: Windows paths stop at 260 characters
    dest = os.path.join(raw_dir, hashlib.sha256(url.encode()).hexdigest()[:16] + "_" + url.rsplit("/", 1)[-1])
    if not os.path.exists(dest):
        tmp = dest + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "paperkiln-corpus-fetcher"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        os.replace(tmp, dest)
    h = _sha(dest)
    if want and h != want:
        raise C.HashMismatch(f"{url}: sha256 {h} != pinned {want}")
    INPUTS.append((url, h))
    return dest


def github_file(repo_commit, path, raw_dir, want=None):
    repo, commit = repo_commit
    return fetch_url(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", raw_dir, want)


def github_tree(repo_commit, prefix, suffix):
    repo, commit = repo_commit
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1",
                                 headers={"User-Agent": "paperkiln-corpus-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = json.load(r)
    return sorted(x["path"] for x in tree["tree"]
                  if x["type"] == "blob" and x["path"].startswith(prefix) and x["path"].endswith(suffix))


def hf_rows(spec, columns=None):
    repo, rev, files = spec
    for fn in files:
        INPUTS.append((f"hf://{repo}@{rev}/{fn}", None))
    yield from C.parquet_rows(repo, rev, files, columns)


def iter_json_array(f, chunk=1 << 22):
    """The elements of a top-level JSON array, read incrementally from a binary file (WoW's
    train.json is too large for json.load beside a training job)."""
    utf8 = codecs.getincrementaldecoder("utf-8")()   # a tar stream is not seekable: no TextIOWrapper
    dec = json.JSONDecoder()
    buf, pos, eof = "", 0, False

    def more():
        nonlocal buf, pos, eof
        raw = f.read(chunk)
        eof = not raw
        buf, pos = buf[pos:] + utf8.decode(raw, final=eof), 0

    def skip(chars):
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos] in chars:
                pos += 1
            if pos < len(buf) or eof:
                return
            more()
    skip(" \t\r\n")
    if pos >= len(buf) or buf[pos] != "[":
        raise ValueError("not a JSON array")
    pos += 1
    while True:
        skip(" \t\r\n,")
        if pos >= len(buf):
            raise ValueError("JSON array not closed")
        if buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            more()
            continue
        yield obj
        pos = end


# ---- turn handling ------------------------------------------------------------
_DETOK = [(re.compile(r" ([.,!?;:%)\]])"), r"\1"), (re.compile(r"([(\[$]) "), r"\1"),
          (re.compile(r" ?(['’]) ?(s|m|re|ve|ll|d|t)\b", re.I), lambda m: m.group(1) + m.group(2).lower()), (re.compile(r" n(['’])t\b"), r"n\1t"),
          (re.compile(r"\s+"), " ")]


def detok(s, case=False):
    """Undo whitespace tokenisation ("i 'm fine , thanks ." -> "i'm fine, thanks."); with case,
    restore sentence-initial capitals and the pronoun I (ConvAI2 is lowercased)."""
    s = s.strip()
    for pat, rep in _DETOK:
        s = pat.sub(rep, s)
    if case:
        s = re.sub(r"\bi\b", "I", s)
        s = re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s


def has_tell(text):
    t = text.lower()
    return any(p in t for p in TELLS)


def to_messages(utts, mapping, counts):
    """utts: [(speaker, text)]. mapping: {speaker: role} or None (first speaker is the user, the
    other the assistant). Returns messages or (None, reason); counts gets merged/trimmed tallies."""
    clean = []
    for spk, text in utts:
        text = (text or "").replace("\r\n", "\n").strip()
        if not text or text == "__SILENCE__":
            counts["utterances_removed_empty_or_silence"] += 1
            continue
        clean.append((spk, text))
    if len({s for s, _ in clean}) > 2:
        return None, "multi_party"
    merged = []
    for spk, text in clean:
        if merged and merged[-1][0] == spk:
            merged[-1] = (spk, merged[-1][1] + "\n" + text)
            counts["turns_merged_same_speaker"] += 1
        else:
            merged.append((spk, text))
    if not merged:
        return None, "empty_turn"
    if mapping is None:
        first = merged[0][0]
        msgs = [{"role": "user" if s == first else "assistant", "content": t} for s, t in merged]
    else:
        if any(s not in mapping for s, _ in merged):
            return None, "unknown_speaker"
        msgs = [{"role": mapping[s], "content": t} for s, t in merged]
        lead = 0
        while lead < len(msgs) and msgs[lead]["role"] == "assistant":
            lead += 1
        if lead:
            counts["leading_assistant_turns_trimmed"] += lead
            msgs = msgs[lead:]
    return msgs, None


# ---- per-source raw dialogues: (uri, messages, None) or (uri, None, reason) ----
def _oasst_human(m):
    return not m.get("synthetic") and m.get("model_name") is None and not has_tell(m["text"])


def oasst_branch(root_id, children, msgs):
    """dialogue_pt.best_branch with one more condition on every message: written by a person
    (not flagged synthetic, no model_name, no ChatGPT tell), so a tainted reply gives way to the
    next-ranked one instead of ending the dialogue."""
    def ok(m):
        return m["lang"] == "en" and not m["deleted"] and m["review_result"] is not False and _oasst_human(m)

    def has_ok_reply(m):
        return any(ok(msgs[c]) for c in children.get(m["message_id"], ()))

    path, node = [], msgs[root_id]
    while node is not None and ok(node):
        path.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
        kids = [msgs[c] for c in children.get(node["message_id"], ()) if ok(msgs[c])]
        if node["role"] == "prompter":
            ranked = [k for k in kids if k["rank"] is not None]
            node = min(ranked, key=lambda k: (k["rank"], k["message_id"])) if ranked else (
                kids[0] if len(kids) == 1 else None)
        else:
            follow = sorted((k for k in kids if has_ok_reply(k)), key=lambda k: (k["created_date"], k["message_id"]))
            node = follow[0] if follow else None
    return path


def raw_oasst(ctx):
    cols = ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "review_result",
            "message_tree_id", "tree_state", "created_date", "synthetic", "model_name"]
    trees, origin = defaultdict(dict), {}
    for name, spec in (("oasst2", OASST2), ("oasst1", OASST1)):
        for _, row in hf_rows(spec, cols):
            t = row["message_tree_id"]
            if origin.get(t, name) != name:
                continue
            origin[t] = name
            trees[t][row["message_id"]] = row
    for t, msgs in sorted(trees.items()):
        roots = [m for m in msgs.values() if m["parent_id"] is None]
        uri = f"{origin[t]}:{t}"
        if len(roots) != 1 or roots[0]["tree_state"] != "ready_for_export" or roots[0]["lang"] != "en":
            yield uri, None, "not_english_or_not_ready"
            continue
        if not _oasst_human(roots[0]):
            yield uri, None, "llm_tell" if has_tell(roots[0]["text"]) else "synthetic"
            continue
        children = defaultdict(list)
        for m in msgs.values():
            if m["parent_id"] is not None:
                children[m["parent_id"]].append(m["message_id"])
        yield uri, oasst_branch(roots[0]["message_id"], children, msgs), None


def raw_convai2(ctx):
    """ConvAI2 train (PersonaChat's human-human dialogues), the no-persona view. Each line is
    "N partner<TAB>self[...]"; a dialogue starts at N == 1. Training data holds each dialogue from
    both speakers' perspectives, so dialogues sharing their first four turns keep the longest
    (the rest: perspective_duplicate)."""
    path = fetch_url(CONVAI2_URL, ctx["raw"], ARCHIVE_SHA256[CONVAI2_URL])
    with tarfile.open(path, "r:gz") as tar:
        names = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
        member = names.get("train_none_original_no_cands.txt") or names["train_none_original.txt"]
        ctx["counts"]["member:" + member.name] += 1
        lines = tar.extractfile(member).read().decode("utf-8").splitlines()
    dialogues, cur = [], None
    for line in lines:
        m = re.match(r"(\d+) (.*)", line)
        if not m:
            continue
        if m.group(1) == "1":
            cur = []
            dialogues.append(cur)
        fields = m.group(2).split("\t")
        if fields[0].startswith(("your persona:", "partner's persona:")):
            continue
        cur.append(("partner", fields[0]))
        if len(fields) > 1:
            cur.append(("self", fields[1]))
    yield from convai2_dialogues(dialogues, os.path.basename(member.name), ctx["counts"])


def convai2_dialogues(dialogues, name, counts):
    """(uri, messages, reason) per parsed ConvAI2 dialogue [(speaker, text)], keeping the longest
    of the dialogues that share their first four (detokenised, non-silent) turns."""
    cleaned = [[(s, detok(t, case=True)) for s, t in d if t.strip() and t.strip() != "__SILENCE__"] for d in dialogues]
    best = {}
    for i, utts in enumerate(cleaned):
        key = tuple(t.lower() for _, t in utts[:4])
        if key not in best or len(utts) > len(cleaned[best[key]]):
            best[key] = i
    keep = set(best.values())
    for i, utts in enumerate(cleaned):
        uri = f"convai2:{name}#{i}"
        if i not in keep:
            yield uri, None, "perspective_duplicate"
            continue
        msgs, reason = to_messages(utts, None, counts)
        yield uri, msgs, reason


def raw_wow(ctx):
    path = fetch_url(WOW_URL, ctx["raw"], ARCHIVE_SHA256[WOW_URL])
    with tarfile.open(path, "r|gz") as tar:   # streamed: train.json is read as it decompresses
        member = next(m for m in tar if os.path.basename(m.name) == "train.json")
        for i, ep in enumerate(iter_json_array(tar.extractfile(member))):
            utts = []
            for turn in ep.get("dialog", []):
                spk = turn.get("speaker", "")
                utts.append(("wizard" if "Wizard" in spk else "apprentice" if "Apprentice" in spk else spk,
                             turn.get("text", "")))
            msgs, reason = to_messages(utts, {"apprentice": "user", "wizard": "assistant"}, ctx["counts"])
            yield f"wow:train.json#{i}", msgs, reason


def raw_bst(ctx):
    """Blended Skill Talk train. The two seed utterances (previous_utterance, taken from ConvAI2,
    WoW or EmpatheticDialogues) are left out: they belong to other sources. Guided workers could
    pick a suggestion from retrieval models over those datasets' human utterances (20.5% of
    guided turns, 2004.08449 section 3); suggestions are human-written text."""
    for uri, row in hf_rows(BST, ["free_messages", "guided_messages"]):
        utts = []
        for k in range(max(len(row["free_messages"]), len(row["guided_messages"]))):
            if k < len(row["free_messages"]):
                utts.append(("free", row["free_messages"][k]))
            if k < len(row["guided_messages"]):
                utts.append(("guided", row["guided_messages"][k]))
        msgs, reason = to_messages(utts, {"free": "user", "guided": "assistant"}, ctx["counts"])
        yield uri, msgs, reason


def raw_taskmaster(ctx):
    """Taskmaster-1 (self-dialogs minus the dev/test ids, and the two-person WOz dialogs),
    Taskmaster-2 and Taskmaster-3. Taskmaster-4 (collected for a 2024 paper) is left out: crowd
    text of the LLM era with no stated control (survey section 3). Taskmaster-3 labels its
    speakers in lower case, which this v1 mapping does not read: its dialogues drop as
    unknown_speaker, so human_taskmaster_v1 is TM-1 and TM-2 (survey section 6)."""
    files = {p: github_file(TASKMASTER, p, ctx["raw"]) for p in TM_FILES}
    held = set()
    for p in ("TM-1-2019/train-dev-test/dev.csv", "TM-1-2019/train-dev-test/test.csv"):
        with open(files[p], encoding="utf-8") as f:
            held |= {line.split(",")[0].strip() for line in f if line.strip()}
    for p in TM_FILES:
        if not p.endswith(".json"):
            continue
        with open(files[p], "rb") as f:
            for conv in iter_json_array(f):
                cid = conv.get("conversation_id", "")
                uri = f"taskmaster:{p}#{cid}"
                if p.endswith("self-dialogs.json") and cid in held:
                    yield uri, None, "held_out_split"
                    continue
                utts = [(u.get("speaker"), u.get("text", "")) for u in conv.get("utterances", [])]
                msgs, reason = to_messages(utts, {"USER": "user", "ASSISTANT": "assistant"}, ctx["counts"])
                yield uri, msgs, reason


def raw_multiwoz(ctx):
    for p in MWOZ_FILES:
        with open(github_file(MULTIWOZ, p, ctx["raw"]), "rb") as f:
            for d in iter_json_array(f):
                utts = [(t.get("speaker"), t.get("utterance", "")) for t in d.get("turns", [])]
                msgs, reason = to_messages(utts, {"USER": "user", "SYSTEM": "assistant"}, ctx["counts"])
                yield f"multiwoz:{p}#{d.get('dialogue_id')}", msgs, reason


def raw_aya_en(ctx):
    """Aya's English ORIGINAL annotations only: re-annotations are human edits of automatically
    generated (templated) pairs, so their text is not wholly written by a person."""
    for uri, row in hf_rows(AYA, ["inputs", "targets", "language", "annotation_type"]):
        if row["language"] != "English":
            yield uri, None, "not_english"
        elif row["annotation_type"] != "original-annotations":
            yield uri, None, "re_annotation"
        else:
            yield uri, [{"role": "user", "content": row["inputs"]}, {"role": "assistant", "content": row["targets"]}], None


def raw_self_dialogue(ctx):
    """One Mechanical Turk worker wrote both sides (2017). Rejected assignments and the
    repository's blocked workers are skipped, as its get_data.py does."""
    blocked = set(open(github_file(SELF_DIALOGUE, "blocked_workers.txt", ctx["raw"]), encoding="utf-8").read().split())
    for p in github_tree(SELF_DIALOGUE, "corpus/", ".csv"):
        with open(github_file(SELF_DIALOGUE, p, ctx["raw"]), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                uri = f"self_dialogue:{p}#{row.get('AssignmentId')}"
                if row.get("Reject") or row.get("WorkerId") in blocked:
                    yield uri, None, "rejected_or_blocked_worker"
                    continue
                keys = sorted((k for k in row if k and k.startswith("Answer.sentence")), key=lambda k: int(k[15:]))
                sents = [row[k] for k in keys if row[k] and row[k] != "{}"]
                utts = [("a" if k % 2 == 0 else "b", s) for k, s in enumerate(sents)]
                msgs, reason = to_messages(utts, None, ctx["counts"])
                yield uri, msgs, reason


def raw_topical_chat(ctx):
    with open(github_file(TOPICAL, "conversations/train.json", ctx["raw"]), encoding="utf-8") as f:
        data = json.load(f)
    for cid in sorted(data):
        utts = [(t.get("agent"), t.get("message", "")) for t in data[cid].get("content", [])]
        msgs, reason = to_messages(utts, None, ctx["counts"])
        yield f"topical_chat:train.json#{cid}", msgs, reason


def raw_dolly(ctx):
    from huggingface_hub import hf_hub_download
    repo, rev, files = DOLLY
    INPUTS.append((f"hf://{repo}@{rev}/{files[0]}", None))
    with open(hf_hub_download(repo, files[0], repo_type="dataset", revision=rev), encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            ctx_text = re.sub(r"\[\d+\]", "", r.get("context") or "").strip()   # the card's advice
            user = r["instruction"].strip() + ("\n\n" + ctx_text if ctx_text else "")
            yield f"dolly#{i}", [{"role": "user", "content": user}, {"role": "assistant", "content": r["response"]}], None


def raw_quac(ctx):
    """Student questions and the teacher's answers (Wikipedia spans). An unanswerable question
    (CANNOTANSWER) is removed with its answer; the first question carries the page and section
    title, the only context the student saw besides a background paragraph."""
    for uri, row in hf_rows(QUAC, ["wikipedia_page_title", "section_title", "questions", "orig_answers"]):
        utts, removed = [], 0
        for q, a in zip(row["questions"], row["orig_answers"]["texts"]):
            if a.strip() == "CANNOTANSWER":
                removed += 1
                continue
            if not utts:
                q = f"{row['wikipedia_page_title']}, {row['section_title']}: {q}"
            utts += [("student", q), ("teacher", a)]
        ctx["counts"]["qa_pairs_removed_cannotanswer"] += removed
        msgs, reason = to_messages(utts, {"student": "user", "teacher": "assistant"}, ctx["counts"])
        yield uri, msgs, reason


def raw_dailydialog(ctx):
    for uri, row in hf_rows(DAILYDIALOG, ["dialog"]):
        utts = [("a" if k % 2 == 0 else "b", detok(t)) for k, t in enumerate(row["dialog"])]
        msgs, reason = to_messages(utts, None, ctx["counts"])
        yield uri, msgs, reason


def raw_empathetic(ctx):
    convs = defaultdict(list)
    for _, row in hf_rows(EMPATHETIC, ["conv_id", "utterance_idx", "speaker_idx", "utterance"]):
        convs[row["conv_id"]].append((row["utterance_idx"], row["speaker_idx"], row["utterance"].replace("_comma_", ",")))
    for cid in sorted(convs):
        turns = sorted(convs[cid])
        msgs, reason = to_messages([(s, t) for _, s, t in turns], None, ctx["counts"])
        yield f"empathetic:{cid}", msgs, reason


def raw_no_robots(ctx):
    for uri, row in hf_rows(NO_ROBOTS, ["messages", "category"]):
        yield uri, [dict(m) for m in row["messages"]], None


# ---- long-dialogue sources (24 Oct 2026; survey section 9) ----------------------
def raw_taskmaster3(ctx):
    """Taskmaster-3 (TicketTalk, 2020): self-dialogs, one crowd worker writing both the customer and
    the ticketing agent, speaker labels "user" and "assistant" in lower case (the v1 reader maps
    upper case, so human_taskmaster_v1 holds none of them). Dialogues whose scenario starts "Auto
    template" (any case) drop (auto_template): the worker filled in a partly written conversation
    whose agent turns came fixed from the collectors' template ("force assistant" lines with slot
    values), so those turns are not the worker's text (TM-3 README, COLLECTION METHODOLOGY). The
    splits are per turn (see TM3_TRAIN): a dialogue drops as held_out_split only when its id appears
    in the dev or test splits (splits/{dev,test}/*.tsv, third column) and in no train shard."""
    def split_ids(paths):
        out = set()
        for p in paths:
            with open(github_file(TASKMASTER, p, ctx["raw"]), encoding="utf-8") as f:
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) >= 3 and cols[2].startswith("dlg-"):
                        out.add(cols[2].strip())
        return out
    held = split_ids(TM3_HELD) - split_ids(TM3_TRAIN)
    for p in TM3_FILES:
        with open(github_file(TASKMASTER, p, ctx["raw"]), "rb") as f:
            for conv in iter_json_array(f):
                cid = conv.get("conversation_id", "")
                uri = f"taskmaster3:{p}#{cid}"
                if cid in held:
                    yield uri, None, "held_out_split"
                elif (conv.get("scenario") or "").lower().startswith("auto template"):
                    yield uri, None, "auto_template"
                else:
                    utts = [(u.get("speaker"), u.get("text", "")) for u in conv.get("utterances", [])]
                    msgs, reason = to_messages(utts, {"user": "user", "assistant": "assistant"}, ctx["counts"])
                    yield uri, msgs, reason


def raw_msc(ctx):
    """Multi-Session Chat train (Xu, Szlam and Weston 2021; ParlAI msc_v0.1). Session 1 is PersonaChat;
    sessions 2-4 were written later by crowd workers playing the same two personas, who read the
    earlier sessions (not always the workers of session 1: paper section 3). The record of an
    episode's last session holds the earlier sessions in previous_dialogs, so one episode is that
    record's sessions 1..k in order. Valid and test (500 + 501 five-session episodes) are not read."""
    path = fetch_url(MSC_URL, ctx["raw"], ARCHIVE_SHA256[MSC_URL])
    recs = defaultdict(dict)
    with tarfile.open(path, "r:gz") as tar:
        names = {m.name: m for m in tar.getmembers() if m.isfile()}
        for s in (2, 3, 4, 5):
            m = names.get(f"msc/msc_dialogue/session_{s}/train.txt")
            if m is None:
                continue
            for line in tar.extractfile(m).read().decode("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[r["metadata"]["initial_data_id"]][s] = r
    yield from msc_episodes(recs, ctx["counts"])


def msc_episodes(recs, counts):
    """(uri, messages, reason) per MSC episode. recs: {initial_data_id: {session: record}}. Speakers
    are positional in every session, as ParlAI's msc teacher reads them: utterance 0, 2, ... is
    Speaker 1, who opens each session (a session's first "id" is sometimes a number instead of
    "Speaker 1"). An odd-length session ends on Speaker 1, who also opens the next: to_messages merges
    the two into one user turn with a newline (counted: msc_session_joins_same_speaker). No marker is
    inserted between sessions; the time gaps stay in the source's metadata."""
    for iid in sorted(recs):
        by = recs[iid]
        last = max(by)
        rec = by[last]
        sessions = [pd.get("dialog", []) for pd in rec.get("previous_dialogs", [])] + [rec.get("dialog", [])]
        for s in range(2, last):
            if s in by and [u.get("text") for u in by[s].get("dialog", [])] != [u.get("text") for u in sessions[s - 1]]:
                counts["msc_session_text_differs_from_its_own_record"] += 1   # the last record's copy is used
        utts = []
        for sess in sessions:
            new = [("speaker1" if i % 2 == 0 else "speaker2", u.get("text") or "") for i, u in enumerate(sess)]
            prev = [s for s, t in utts if t.strip()]
            nxt = [s for s, t in new if t.strip()]
            if prev and nxt and prev[-1] == nxt[0]:
                counts["msc_session_joins_same_speaker"] += 1
            utts += new
        counts[f"msc_episodes_{len(sessions)}_sessions"] += 1
        msgs, reason = to_messages(utts, {"speaker1": "user", "speaker2": "assistant"}, counts)
        yield f"msc:session_{last}/train.txt#{iid}", msgs, reason


def raw_abcd(ctx):
    """ABCD train (Chen et al. 2021): Expert Live Chat, a trained crowd agent and a crowd customer
    chatting in real time. The agent's system actions ("action" lines, e.g. "Account has been pulled
    up for ...") are interface records, not dialogue: removed (counted), so the agent's messages on
    either side of one merge."""
    with gzip.open(github_file(ABCD, "data/abcd_v1.1.json.gz", ctx["raw"], FILE_SHA256["data/abcd_v1.1.json.gz"]),
                   "rt", encoding="utf-8") as f:
        data = json.load(f)
    for c in data["train"]:
        utts = [(s, t) for s, t in c.get("original", []) if s != "action"]
        ctx["counts"]["action_lines_removed"] += len(c.get("original", [])) - len(utts)
        msgs, reason = to_messages(utts, {"customer": "user", "agent": "assistant"}, ctx["counts"])
        yield f"abcd:train#{c.get('convo_id')}", msgs, reason


def raw_ccpe(ctx):
    """CCPE-M (Radlinski et al. 2019): two crowd workers, Wizard-of-Oz; the assistant elicits movie
    preferences, the user's speech is transcribed. No splits defined: all 502 dialogues."""
    with open(github_file(CCPE, "data.json", ctx["raw"], FILE_SHA256["data.json"]), encoding="utf-8") as f:
        data = json.load(f)
    for c in data:
        utts = [(u.get("speaker"), u.get("text", "")) for u in sorted(c.get("utterances", []), key=lambda u: u.get("index", 0))]
        msgs, reason = to_messages(utts, {"USER": "user", "ASSISTANT": "assistant"}, ctx["counts"])
        yield f"ccpe:data.json#{c.get('conversationId')}", msgs, reason


def raw_p4g(ctx):
    """PersuasionForGood (Wang et al. 2019): paired Mechanical Turk workers, one persuading the other
    to donate to a charity, at least 10 turns each. full_dialog.csv holds one sentence per row (Unit),
    with the turn index (Turn), the role (B4: 0 persuader, 1 persuadee) and the dialogue (B2); a
    speaker's sentences within one turn are joined with a space. No splits defined: all 1,017."""
    convs = {}
    with open(fetch_url(P4G_URL, ctx["raw"], FILE_SHA256[P4G_URL]), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            utts = convs.setdefault(row["B2"], [])
            key = (row["Turn"], row["B4"])
            text = (row["Unit"] or "").strip()
            if utts and utts[-1][0] == key:
                utts[-1] = (key, (utts[-1][1] + " " + text).strip())
            else:
                utts.append((key, text))
    for did, utts in convs.items():   # file order
        role = {"0": "persuader", "1": "persuadee"}
        msgs, reason = to_messages([(role.get(k[1], k[1]), t) for k, t in utts],
                                   {"persuader": "user", "persuadee": "assistant"}, ctx["counts"])
        yield f"p4g:full_dialog.csv#{did}", msgs, reason


RAW = {"oasst": raw_oasst, "convai2": raw_convai2, "wow": raw_wow, "bst": raw_bst, "taskmaster": raw_taskmaster,
       "multiwoz": raw_multiwoz, "aya_en": raw_aya_en, "self_dialogue": raw_self_dialogue,
       "topical_chat": raw_topical_chat, "dolly": raw_dolly, "quac": raw_quac, "dailydialog": raw_dailydialog,
       "empathetic": raw_empathetic, "no_robots": raw_no_robots,
       "taskmaster3": raw_taskmaster3, "msc": raw_msc, "abcd": raw_abcd, "ccpe": raw_ccpe, "p4g": raw_p4g}


# ---- decontamination targets --------------------------------------------------
def decontaminator(carry5=False, carry12=False):
    """The frozen eval sets, plus Carry-5 when asked for (and present): one row per item turn, the
    user turn as a prompt and the gold reply as its reference. carry12 adds Carry-12 v1 the same way
    (experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md section 2: the long programme's data job
    drops dialogues by those rules), its single-turn twins as prompts too; each distinct text once. Returns
    (Decontaminator, Carry-5's sha256 or None); carry12 requires the set (carry12_sha gives its hash)."""
    use5 = carry5 and os.path.exists(CARRY5)
    if carry12 and not os.path.exists(CARRY12):
        raise SystemExit(f"{CARRY12} is missing: Carry-12 cannot join the decontamination")
    if not use5 and not carry12:
        return decontam.Decontaminator(), None
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with tmp:
        if use5:
            for line in open(CARRY5, encoding="utf-8"):
                if line.strip():
                    for t in json.loads(line).get("turns", []):
                        tmp.write(json.dumps({"prompt": t.get("user", ""), "reference": t.get("gold", "")}) + "\n")
    extra = []
    if carry12:
        seen = set()
        tmp12 = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        with tmp12:
            for line in open(CARRY12, encoding="utf-8"):
                if not line.strip():
                    continue
                it = json.loads(line)
                rows = [(t.get("user", ""), t.get("gold", "")) for t in it.get("turns", [])]
                rows.append((it.get("twin", ""), ""))
                for row in rows:
                    if row[0] and row not in seen:
                        seen.add(row)
                        tmp12.write(json.dumps({"prompt": row[0], "reference": row[1]}) + "\n")
        extra = [tmp12.name]
    try:
        sets = decontam.CHAT_SETS + ([tmp.name] if use5 else []) + extra
        return decontam.Decontaminator(chat_sets=sets), _sha(CARRY5) if use5 else None
    finally:
        os.remove(tmp.name)
        for x in extra:
            os.remove(x)


def carry12_sha():
    return _sha(CARRY12) if os.path.exists(CARRY12) else None


# ---- build ----------------------------------------------------------------------
def _out(cid, name, out_dir):
    if out_dir is None:
        return C.out_path(cid, name)
    os.makedirs(os.path.join(out_dir, cid), exist_ok=True)
    return os.path.join(out_dir, cid, name)


def _stats(turns, st):
    st["dialogues"] += 1
    st["turns"] += len(turns)
    st["exchanges"] += len(turns) // 2
    st["words"] += sum(len(t["content"].split()) for t in turns)
    st["short_turns_le2_words"] += sum(len(t["content"].split()) <= 2 for t in turns)
    st["max_turns"] = max(st["max_turns"], len(turns))


def build(source, pin=False, dc=None, rows=None, out_dir=None, raw_dir=None, keep_raw=False):
    """Write one source's corpus; returns its report. rows: (uri, messages, reason) standing in
    for the download (tests). out_dir: a directory in place of corpus/data/ (a VM run; pin at home
    with the `pin` command)."""
    cid = f"human_{source}_v1"
    names = names_for(source)
    if rows is None and out_dir is None and C.already_done(cid, names):
        print(f"{cid}: present and verified")
        return json.load(open(C.out_path(cid, "report.json"), encoding="utf-8"))
    if dc is None:
        dc, _ = decontaminator()
    if source == "simplewiki":
        return build_simplewiki(cid, pin, dc, rows, out_dir)
    gate, licence, roles, llm_era, _ = SOURCES[source]
    raw_dir = raw_dir or os.path.join(out_dir or C.DATA_DIR, "_raw_human_dialogue", source)
    INPUTS.clear()
    counts, st, drops, flags, keys = Counter(), Counter(), [], [], set()
    ctx = {"raw": raw_dir, "counts": counts}
    d_path, t_path = _out(cid, "dialogues.jsonl", out_dir), _out(cid, "text.txt", out_dir)
    with open(d_path, "w", encoding="utf-8", newline="\n") as fd, open(t_path, "w", encoding="utf-8", newline="\n") as ft:
        for uri, msgs, reason in (RAW[source](ctx) if rows is None else rows):
            counts["rows_in"] += 1
            if msgs is None:
                counts[reason] += 1
                continue
            turns, reason = C.normalise_dialogue(msgs)
            if turns is None:
                counts[reason] += 1
                continue
            if llm_era and any(has_tell(t["content"]) for t in turns):
                counts["llm_tell"] += 1
                continue
            k = C.dialogue_key(turns)
            if k in keys:
                counts["duplicate"] += 1
                continue
            keys.add(k)
            d, f = dc.check([t["content"] for t in turns if t["role"] == "user"], "\n".join(t["content"] for t in turns))
            if d:
                counts["decontaminated"] += 1
                drops.append({"uri": uri, "matches": d})
                continue
            flags += [{"uri": uri, **x} for x in f]
            counts[reason] += 1
            _stats(turns, st)
            fd.write(json.dumps({"messages": turns, "uri": uri}, ensure_ascii=False) + "\n")
            ft.write(C.render_dialogue(turns))
    counts["written"] = counts["kept"] + counts["system_stripped"]
    report = {"id": cid, "source": source, "gate": gate, "licence": licence, "roles": roles,
              "llm_era_tell_filter": llm_era, "counts": dict(sorted(counts.items())), "stats": dict(sorted(st.items())),
              "bytes": os.path.getsize(t_path), "inputs": [{"uri": u, "sha256": h} for u, h in INPUTS],
              "decontam_targets": _targets(dc), "decontam_dropped": drops, "answer_flags": flags}
    _finish(cid, source, report, [d_path, t_path], pin, rows, out_dir)
    if rows is None and not keep_raw and os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir, ignore_errors=True)   # the pins and the report hold the record
    return report


def build_simplewiki(cid, pin, dc, rows, out_dir):
    """Simple English Wikipedia articles as plain documents (text + <|endoftext|>), the "bit of
    knowledge". Articles under 20 words are dropped; answer flags are expected (encyclopaedic
    facts) and only reported."""
    INPUTS.clear()
    counts, st, drops, flags, keys = Counter(), Counter(), [], [], set()
    d_path, t_path = _out(cid, "documents.jsonl", out_dir), _out(cid, "text.txt", out_dir)
    it = hf_rows(SIMPLEWIKI, ["title", "text"]) if rows is None else rows
    with open(d_path, "w", encoding="utf-8", newline="\n") as fd, open(t_path, "w", encoding="utf-8", newline="\n") as ft:
        for uri, row in it:
            counts["rows_in"] += 1
            text = (row["text"] or "").replace("<|endoftext|>", "").strip()
            if any(m in text for m in C.MARKERS):
                counts["markers"] += 1
                continue
            if len(text.split()) < 20:
                counts["too_short"] += 1
                continue
            k = hashlib.sha256(" ".join(decontam.words(text)).encode()).hexdigest()
            if k in keys:
                counts["duplicate"] += 1
                continue
            keys.add(k)
            d, f = dc.check([], text)
            if d:
                counts["decontaminated"] += 1
                drops.append({"uri": uri, "matches": d})
                continue
            flags += [{"uri": uri, **x} for x in f]
            counts["written"] += 1
            st["documents"] += 1
            st["words"] += len(text.split())
            fd.write(json.dumps({"text": text, "title": row["title"], "uri": uri}, ensure_ascii=False) + "\n")
            ft.write(text + "\n<|endoftext|>\n")
    report = {"id": cid, "source": "simplewiki", "gate": SA, "licence": SOURCES["simplewiki"][1],
              "roles": "plain documents", "counts": dict(sorted(counts.items())), "stats": dict(sorted(st.items())),
              "bytes": os.path.getsize(t_path), "inputs": [{"uri": u, "sha256": h} for u, h in INPUTS],
              "decontam_targets": _targets(dc), "decontam_dropped": drops, "answer_flags": flags}
    _finish(cid, "simplewiki", report, [d_path, t_path], pin, rows, out_dir)
    return report


def _targets(dc):
    return sorted({e.split(":")[0] for v in dc.patterns.values() for _, _, e in v} | {e.split(":")[0] for e in dc.prompts.values()})


def _finish(cid, source, report, paths, pin, rows, out_dir):
    with open(_out(cid, "report.json", out_dir), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"{cid}: {report['counts']}")
    if rows is None and out_dir is None:
        C.finish(cid, paths, f"{source}: human-written text (human_dialogue.py)", pin=pin)


# ---- token counts ----------------------------------------------------------------
def count_tokens(text_path, vocab):
    """Tokens of a text.txt under a vocabulary: llama.cpp-form "gpt2"/"llama-bpe" (the 16k
    Llama 3 scheme, tools/llama3_bpe.py) or paperkiln "bpe" (tools/bpe.py, e.g. the 8k). Documents
    are encoded separately; both encoders split at the special tokens first, so this equals
    encoding the whole file."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import llama3_bpe
    model = llama3_bpe.read_gguf_kv(vocab, {"tokenizer.ggml.model"}).get("tokenizer.ggml.model")
    with open(text_path, encoding="utf-8") as f:
        docs = [d + "<|endoftext|>\n" for d in f.read().split("<|endoftext|>\n") if d]
    if model == "gpt2":
        tok = llama3_bpe.from_gguf(vocab)
        return sum(len(e.ids) for i in range(0, len(docs), 2048)
                   for e in tok.encode_batch(docs[i:i + 2048], add_special_tokens=False))
    import bpe
    enc, memo, n = bpe.read_gguf(vocab), {}, 0
    for d in docs:
        for ch in bpe.pretokenize(d):
            c = memo.get(ch)
            if c is None:
                c = memo[ch] = len(enc.encode_chunk(ch))
            n += c
    return n


def add_tokens(cid, vocabs, out_dir):
    rp = _out(cid, "report.json", out_dir)
    report = json.load(open(rp, encoding="utf-8"))
    tp = _out(cid, "text.txt", out_dir)
    for v in vocabs:
        if not os.path.exists(v):
            print(f"{cid}: no vocabulary at {v}; skipped")
            continue
        report.setdefault("tokens", {})[os.path.basename(v)] = count_tokens(tp, v)
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"{cid}: tokens {report.get('tokens')}")
    return report


# ---- ledger ------------------------------------------------------------------------
def write_ledger(reports, out_dir, carry5_sha, name=LEDGER):
    """Per-source counts and sizes, plus exact duplicates ACROSS sources (reported, not dropped:
    each corpus is built and pinned alone)."""
    where = {}
    for s, r in reports.items():
        if "error" in r:
            continue
        p = _out(r["id"], names_for(s)[0], out_dir)
        with open(p, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                k = C.dialogue_key(row["messages"]) if "messages" in row else \
                    hashlib.sha256(" ".join(decontam.words(row["text"])).encode()).hexdigest()
                where.setdefault(k, set()).add(s)
    cross = Counter()
    for srcs in where.values():
        if len(srcs) > 1:
            cross["+".join(sorted(srcs))] += 1
    ledger = {"carry5_v1_sha256": carry5_sha, "cross_source_duplicates": dict(cross), "sources": {}}
    for s, r in reports.items():
        ledger["sources"][s] = r if "error" in r else {
            k: r[k] for k in ("id", "gate", "licence", "roles", "counts", "stats", "bytes") if k in r} | {
            "tokens": r.get("tokens", {}), "decontam_dropped": len(r["decontam_dropped"]),
            "answer_flags": len(r["answer_flags"])}
    with open(os.path.join(out_dir or C.DATA_DIR, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(ledger, f, indent=1)
    return ledger


def gate_ok(source, a):
    g = SOURCES[source][0]
    return g == DEFAULT or (g == SA and a.allow_sa) or (g == NC and a.allow_nc) or (g == RISK and a.allow_llm_risk) or (
        g == NOLIC and getattr(a, "allow_unstated_licence", False))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build human-written dialogue corpora from pinned sources.")
    ap.add_argument("what", help="a source (or several, comma-separated), default, optin, all, pin, or tokens")
    ap.add_argument("sources", nargs="*", help="for pin and tokens: the sources")
    ap.add_argument("--allow-sa", action="store_true", help="permit share-alike sources")
    ap.add_argument("--allow-nc", action="store_true", help="permit non-commercial sources")
    ap.add_argument("--allow-llm-risk", action="store_true", help="permit sources with a documented LLM-text risk")
    ap.add_argument("--allow-unstated-licence", action="store_true",
                    help="permit sources whose authors state no data licence (msc)")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--out-dir", help="write here instead of corpus/data/ (a VM run; pin at home)")
    ap.add_argument("--count-vocab", action="append", default=[], help="GGUF vocabulary to count tokens with (repeat)")
    ap.add_argument("--keep-raw", action="store_true", help="keep downloaded archives and GitHub files")
    ap.add_argument("--carry5", action="store_true",
                    help="add corpus/eval/carry5_v1.jsonl to the decontamination targets (changes the v1 outputs)")
    a = ap.parse_args(argv)
    if a.what in ("pin", "tokens"):
        for s in a.sources:
            cid = f"human_{s}_v1"
            if a.what == "pin":  # files in corpus/data/<id>/, or DIR/<id>/ (a VM's outputs where they landed)
                paths = [os.path.join(a.out_dir, cid, n) if a.out_dir else C.out_path(cid, n) for n in names_for(s)]
                C.finish(cid, paths, f"{s}: human-written text (human_dialogue.py)", pin=True)
            else:
                add_tokens(cid, a.count_vocab, a.out_dir)
        return 0
    groups = {"default": DEFAULT_SET, "optin": OPTIN_SET, "all": [s for s in SOURCES if s not in LONG_SET], "long": LONG_SET}
    if a.what in groups:
        todo = [s for s in groups[a.what] if gate_ok(s, a)]
        skipped = [s for s in groups[a.what] if not gate_ok(s, a)]
        if skipped:
            print(f"gate: skipped {skipped} (--allow-sa / --allow-nc / --allow-llm-risk / --allow-unstated-licence)")
    else:
        todo = a.what.split(",")
        for s in todo:
            if s not in SOURCES:
                C.die(f"unknown source {s!r}; one of {sorted(SOURCES)} (comma-separated) or default/optin/all/long/pin/tokens")
            if not gate_ok(s, a):
                C.die(f"{s} is {SOURCES[s][0]} ({SOURCES[s][1]}); pass --allow-sa, --allow-nc, --allow-llm-risk "
                      "or --allow-unstated-licence")
    dc, carry5_sha = decontaminator(a.carry5)
    print(f"decontamination: frozen eval sets{' + carry5_v1' if carry5_sha else ' (carry5_v1 not used)'}")
    reports = {}
    for s in todo:
        try:
            reports[s] = build(s, a.pin, dc, out_dir=a.out_dir, keep_raw=a.keep_raw)
            if a.count_vocab:
                reports[s] = add_tokens(reports[s]["id"], a.count_vocab, a.out_dir)
        except C.HashMismatch as e:
            C.die(str(e))
        except Exception as e:  # one source's failure must not cost the others on a VM
            if len(todo) == 1:
                raise
            print(f"human_{s}_v1: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            reports[s] = {"error": f"{type(e).__name__}: {e}"}
    if len(todo) > 1:
        long_run = all(s in LONG_SET for s in todo)
        write_ledger(reports, a.out_dir, carry5_sha, LONG_LEDGER if long_run else LEDGER)
    return 1 if any("error" in r for r in reports.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
