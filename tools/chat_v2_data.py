#!/usr/bin/env python3
"""The v2 SFT mix (conversation schema v2, LOCAL_LLM_STACK.md section 3, point 3g): conversations that keep
their system prompts, conversations with tool calls and tool results, and a replay slice of lever 1's
user/assistant set, as paperkiln/2 conversation rows, with a ledger of sources, licences and counts.

    python tools/chat_v2_data.py build --smol DIR --no-robots PARQUET --glaive JSON --lever1 JSONL \\
        --vocab VOCAB_V2.gguf --out DIR [--T 512] [--glaive-n 40000] [--replay-n 40000] [--seed 1]
    python tools/chat_v2_data.py prompts --out corpus/eval/chat_v2_prompts_v1.jsonl ...  (the parity set)

Sources (each pinned; licences from the dataset cards, 27 Oct 2026):

- smol-smoltalk (HuggingFaceTB/smol-smoltalk@f73fe857, Apache-2.0): the rows whose first message is a system
  message (sources openhermes-50k, smol-summarize-5k, smol-summarize-20k, smollm-rewrite-30k,
  explore-instruct-rewrite), train split only (smol_smoltalk.py's split by the first user turn, so its test
  split stays held out). Lever 1 (pk_sft_v1) read openhermes-50k with the system messages dropped.
- no_robots (HuggingFaceH4/no_robots@e6f9a4ac, CC BY-NC 4.0, human-written): its 795 Chat rows with a
  system message.
- glaive-function-calling-v2 (glaiveai/glaive-function-calling-v2@e7f4b645, Apache-2.0, synthetic): the
  SYSTEM text's function definitions become the conversation's `tools`, `<functioncall> {"name",
  "arguments": '...'}` an assistant tool call (arguments re-serialised by json.dumps), FUNCTION RESPONSE a
  tool message. Rows that do not parse into schema v2 are dropped and counted by reason.
- lever 1's conversation rows (lever1_conv_v1.jsonl, sha256 b7a43fb5...; pk_sft_v1's set), a seeded replay
  sample: user/assistant only, rendered as under v1.

Every row is checked by chat_template.validate under paperkiln/2; rows holding a control string are
dropped. Decontamination: corpus/decontam.py's rules against the frozen chat and QA eval sets, and the
measurement's held-out prompts: a row with a user turn or system message containing an IF-tiny / turn-distance
question (4 words or more) as a word sequence is dropped. Length: a row is kept when its first exchange fits
T + 1 tokens (mtstudio keeps the longest prefix of whole exchanges); the ledger counts rows mtstudio will cut.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "corpus"))
import chat_template as CT  # noqa: E402

SMOL_REV = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
NOROBOTS_REV = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
GLAIVE_REV = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
GLAIVE_SHA = "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad"
LEVER1_SHA = "b7a43fb5d622537094f7fea1eb5df244c64007f967faed597f9324954a479a32"
SMOL_SYSTEM_SOURCES = ("openhermes-50k", "smol-summarize-5k", "smol-summarize-20k", "smollm-rewrite-30k",
                       "explore-instruct-rewrite")
LICENCES = {"smol_system": "Apache-2.0 (HuggingFaceTB/smol-smoltalk)",
            "no_robots_system": "CC BY-NC 4.0 (HuggingFaceH4/no_robots; non-commercial)",
            "glaive_tools": "Apache-2.0 (glaiveai/glaive-function-calling-v2)",
            "lever1_replay": "Apache-2.0 / TinyChat sample (lever 1's sources, LOCAL_LLM_STACK.md section 8)"}
CONTROL = ["<|endoftext|>", "<|user|>", "<|assistant|>"] + CT.V2_CONTROL
_W = re.compile(r"[^0-9a-z]+")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_of(text):  # corpus/fetchers/smol_smoltalk.py's split, by the first user turn
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h == 0 else ("val" if h == 1 else "train")


def has_control(text):
    return any(c in text for c in CONTROL)


def norm_turns(msgs):
    """[system?] + alternating user/assistant from a user turn, contents stripped, ending on an assistant
    message; (messages, reason)."""
    out = []
    sys_msg = None
    for i, m in enumerate(msgs):
        role, text = m["role"], (m.get("content") or "").strip()
        if role == "system":
            if i != 0:
                return None, "system_not_first"
            sys_msg = text
            continue
        if role not in ("user", "assistant"):
            return None, "role"
        if not text:
            return None, "empty"
        if role != ("user" if len(out) % 2 == 0 else "assistant"):
            return None, "non_alternating"
        out.append({"role": role, "content": text})
    if out and out[-1]["role"] == "user":
        out.pop()
    if not out:
        return None, "too_short"
    if sys_msg:
        out = [{"role": "system", "content": sys_msg}] + out
    return out, "kept"


# ---------------------------------------------------------------- sources
def smol_rows(d):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(d, "train-*.parquet")))
    if len(files) != 4:
        sys.exit(f"{d}: expected smol-smoltalk's 4 train shards")
    counts = {}
    for fp in files:
        f = pq.ParquetFile(fp)
        row = 0
        for b in f.iter_batches(batch_size=4096, columns=["messages", "source"]):
            dd = b.to_pydict()
            for msgs, src in zip(dd["messages"], dd["source"]):
                row += 1
                if src not in SMOL_SYSTEM_SOURCES or not msgs or msgs[0]["role"] != "system":
                    continue
                m, why = norm_turns(msgs)
                key = f"smol/{src}/{why}"
                if m is None or m[0]["role"] != "system":
                    counts[key if m is None else f"smol/{src}/empty_system"] = counts.get(key, 0) + 1
                    continue
                if split_of(m[1]["content"]) != "train":
                    counts[f"smol/{src}/heldout_split"] = counts.get(f"smol/{src}/heldout_split", 0) + 1
                    continue
                counts[key] = counts.get(key, 0) + 1
                yield {"messages": m, "source": "smol_system", "origin": src,
                       "uri": f"HuggingFaceTB/smol-smoltalk@{SMOL_REV[:8]}/{os.path.basename(fp)}#{row}"}
    smol_rows.counts = counts


def norobots_rows(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pylist()
    for i, r in enumerate(t):
        if not r["messages"] or r["messages"][0]["role"] != "system":
            continue
        m, why = norm_turns(r["messages"])
        if m is None or m[0]["role"] != "system":
            continue
        yield {"messages": m, "source": "no_robots_system", "origin": r["category"],
               "uri": f"HuggingFaceH4/no_robots@{NOROBOTS_REV[:8]}/train#{i}"}


_SEG = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE): ")
_FC = re.compile(r'^\{"name":\s*"([^"\n]+)",\s*"arguments":\s*\'(.*)\'\}$', re.S)
_GLAIVE_SYS = "You are a helpful assistant with access to the following functions. Use them if required -"


def glaive_parse(r):
    """(conversation, reason)."""
    s = r["system"].strip()
    if s.startswith("SYSTEM: "):
        s = s[len("SYSTEM: "):]
    tools = []
    if s.startswith(_GLAIVE_SYS):
        rest = s[len(_GLAIVE_SYS):].strip()
        dec = json.JSONDecoder()
        while rest:
            try:
                obj, k = dec.raw_decode(rest)
            except ValueError:
                return None, "tool_definition_json"
            if not isinstance(obj, dict) or "name" not in obj:
                return None, "tool_definition_shape"
            tools.append({"type": "function", "function": obj})
            rest = rest[k:].strip()
        if not tools:
            return None, "no_tool_definition"
        system = "You are a helpful assistant with access to the following functions. Use them if required."
    else:
        system = s
    chat = r["chat"]
    marks = list(_SEG.finditer(chat))
    if not marks or marks[0].start() != 0:
        return None, "chat_shape"
    segs = []
    for i, mk in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(chat)
        text = chat[mk.end():end].strip()
        if text.endswith("<|endoftext|>"):
            text = text[:-len("<|endoftext|>")].strip()
        segs.append((mk.group(1), text))
    msgs = [{"role": "system", "content": system}] if system else []
    for role, text in segs:
        if role == "USER":
            msgs.append({"role": "user", "content": text})
        elif role == "FUNCTION RESPONSE":
            msgs.append({"role": "tool", "content": text})
        else:
            p = text.find("<functioncall>")
            if p < 0:
                msgs.append({"role": "assistant", "content": text})
                continue
            m = _FC.match(text[p + len("<functioncall>"):].strip())
            if not m:
                return None, "functioncall_shape"
            try:
                args = json.dumps(json.loads(m.group(2)), ensure_ascii=False)
            except ValueError:
                return None, "arguments_json"
            call = [{"type": "function", "function": {"name": m.group(1), "arguments": args}}]
            if (not text[:p].strip() and msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")
                    and len(msgs) >= 2 and msgs[-2]["role"] == "user"):
                # "ASSISTANT: Let me check. ASSISTANT: <functioncall> ...": one reply, its text then the call
                msgs[-1]["tool_calls"] = call
                continue
            msgs.append({"role": "assistant", "content": text[:p].strip(), "tool_calls": call})
    while msgs and msgs[-1]["role"] in ("user", "tool"):
        if msgs[-1]["role"] == "tool":
            return None, "ends_on_tool"
        msgs.pop()
    conv = {"messages": msgs}
    if tools:
        conv["tools"] = tools
    for m in msgs:
        if has_control(m["content"]) or any(has_control(c["function"]["name"] + c["function"]["arguments"])
                                            for c in m.get("tool_calls", [])):
            return None, "control_text"
    try:
        CT.validate(conv, prompt=False, tmpl=CT.V2)
        CT.render(msgs, CT.V2, tools=tools or None)
    except CT.TemplateError as e:
        reason = str(e)
        return None, "structure" if "expected" in reason or "ends on" in reason else (
            "float_in_definition" if "non-integer" in reason else "template:" + reason[:40])
    if not any(m["role"] == "user" for m in msgs):
        return None, "too_short"
    return conv, "tools" if any(m.get("tool_calls") for m in msgs) else ("definitions_only" if tools else "no_tools")


def glaive_rows(path):
    if sha256(path) != GLAIVE_SHA:
        sys.exit(f"{path}: sha256 differs from the pinned {GLAIVE_SHA}")
    data = json.load(open(path, encoding="utf-8"))
    counts = {}
    for i, r in enumerate(data):
        conv, why = glaive_parse(r)
        counts[why] = counts.get(why, 0) + 1
        if conv is None:
            continue
        conv.update({"source": "glaive_tools", "origin": why,
                     "uri": f"glaiveai/glaive-function-calling-v2@{GLAIVE_REV[:8]}#{i}"})
        yield conv
    glaive_rows.counts = counts


# ---------------------------------------------------------------- held-out checks
def words(t):
    return [w for w in _W.split(t.lower()) if w]


class Heldout:
    """IF-tiny and turn-distance questions (4 words or more) as word sequences."""

    def __init__(self):
        seqs = set()
        for fn in ("iftiny_v1.jsonl", "ifdist_v1.jsonl"):
            for line in open(os.path.join(ROOT, "corpus", "eval", fn), encoding="utf-8"):
                it = json.loads(line)
                for k in ("question", "bare", "user"):
                    if isinstance(it.get(k), str) and len(words(it[k])) >= 4:
                        seqs.add(" ".join(words(it[k])))
        self.seqs = seqs
        self.first = {}
        for s in seqs:
            self.first.setdefault(s.split()[0], []).append(s)

    def hit(self, text):
        w = words(text)
        joined = " " + " ".join(w) + " "
        for x in set(w):
            for s in self.first.get(x, ()):
                if " " + s + " " in joined:
                    return s
        return None


# ---------------------------------------------------------------- build
def first_exchange_tokens(tok, conv):
    msgs = conv["messages"]
    ex = CT.exchanges(msgs)
    head = msgs[:ex[0][1]]
    pieces = CT.render_pieces(head, CT.V2, tools=conv.get("tools"))
    full = CT.render_pieces(msgs, CT.V2, tools=conv.get("tools"))
    n_first = sum(len(tok.encode(p[0], add_special_tokens=False).ids) for p in pieces)
    n_all = sum(len(tok.encode(p[0], add_special_tokens=False).ids) for p in full)
    return n_first, n_all


def cmd_build(a):
    sys.path.insert(0, HERE)
    import llama3_bpe
    from decontam import Decontaminator
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "chat_v2_mix_v1.jsonl")
    if os.path.exists(out_path):
        sys.exit(f"{out_path} exists")
    if sha256(a.lever1) != LEVER1_SHA:
        sys.exit("lever1_conv_v1.jsonl differs from the pinned sha256")
    tok = llama3_bpe.from_gguf(a.vocab)
    dec = Decontaminator()
    held = Heldout()
    rng = random.Random(a.seed)
    led = {"set": "chat_v2_mix_v1", "T": a.T, "seed": a.seed, "template": CT.V2, "vocab": os.path.abspath(a.vocab),
           "vocab_sha256": sha256(a.vocab), "licences": LICENCES, "sources": {}, "dropped": {}}
    kept = {}

    def take(name, rows, cap=None):
        pool = []
        drop = led["dropped"].setdefault(name, {})
        for r in rows:
            msgs = r["messages"]
            texts = [m["content"] for m in msgs if m["role"] in ("user", "system")]
            if any(has_control(m["content"]) for m in msgs):
                drop["control_text"] = drop.get("control_text", 0) + 1
                continue
            h = next((held.hit(t) for t in texts if held.hit(t)), None)
            if h:
                drop["iftiny_heldout"] = drop.get("iftiny_heldout", 0) + 1
                continue
            users = [m["content"] for m in msgs if m["role"] == "user"]
            alltext = "\n".join([m["content"] for m in msgs] + [c["function"]["name"] + " " + c["function"]["arguments"]
                                                                for m in msgs for c in m.get("tool_calls", [])])
            drops, _flags = dec.check(users, alltext)
            if drops:
                rule = drops[0]["rule"]
                drop["decontam_" + rule] = drop.get("decontam_" + rule, 0) + 1
                continue
            pool.append(r)
        if cap is not None and len(pool) > cap:
            rng.shuffle(pool)
            pool = pool[:cap]
        out = []
        cut = 0
        for r in pool:
            n1, n = first_exchange_tokens(tok, r)
            if n1 > a.T + 1:
                drop["over_T"] = drop.get("over_T", 0) + 1
                continue
            cut += n > a.T + 1
            out.append(r)
        led["sources"][name] = {"rows": len(out), "rows_cut_by_T": cut, "licence": LICENCES[name]}
        kept[name] = out
        print(name, len(out), drop, flush=True)

    take("smol_system", smol_rows(a.smol))
    led["sources"]["smol_system"]["parse"] = smol_rows.counts
    take("no_robots_system", norobots_rows(a.no_robots))
    glaive = list(glaive_rows(a.glaive))
    led["glaive_parse"] = glaive_rows.counts
    # tool rows: conversations with calls, and with definitions the reply does not use; glaive's rows without
    # definitions (generic synthetic Q&A) are left out
    take("glaive_tools", [r for r in glaive if r["origin"] in ("tools", "definitions_only")], cap=a.glaive_n)
    lever = [json.loads(x) for x in open(a.lever1, encoding="utf-8") if x.strip()]
    lever = rng.sample(lever, min(a.replay_n, len(lever)))  # lever 1 was decontaminated when built
    for r in lever:
        r["source"] = "lever1_replay"
    take("lever1_replay", lever, cap=a.replay_n)
    rows = [r for v in kept.values() for r in v]
    rng.shuffle(rows)
    stats = {"rows": len(rows), "with_system": sum(r["messages"][0]["role"] == "system" for r in rows),
             "with_tool_calls": sum(any(m.get("tool_calls") for m in r["messages"]) for r in rows),
             "with_tool_definitions": sum(bool(r.get("tools")) for r in rows),
             "tool_messages": sum(m["role"] == "tool" for r in rows for m in r["messages"]),
             "assistant_messages": sum(m["role"] == "assistant" for r in rows for m in r["messages"])}
    with open(out_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out_path + ".tmp", out_path)
    led.update({"out": out_path, "out_sha256": sha256(out_path), "stats": stats,
                "inputs": {"glaive_sha256": GLAIVE_SHA, "lever1_sha256": LEVER1_SHA,
                           "smol_revision": SMOL_REV, "no_robots_revision": NOROBOTS_REV,
                           "no_robots_sha256": sha256(a.no_robots)}})
    with open(out_path + ".ledger.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(led, f, indent=1)
    # the leftovers for the parity prompt set: glaive rows not drawn, smol-smoltalk's test split
    used = {r["uri"] for r in kept["glaive_tools"]}
    with open(os.path.join(a.out, "glaive_unused.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in glaive:
            if r["uri"] not in used and r["origin"] == "tools":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--smol", required=True, help="smol-smoltalk's data/ folder (the 4 train parquet shards)")
    b.add_argument("--no-robots", required=True)
    b.add_argument("--glaive", required=True)
    b.add_argument("--lever1", required=True)
    b.add_argument("--vocab", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--T", type=int, default=512)
    b.add_argument("--glaive-n", type=int, default=40000)
    b.add_argument("--replay-n", type=int, default=40000)
    b.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
