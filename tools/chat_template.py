#!/usr/bin/env python3
"""Conversations and chat templates (LOCAL_LLM_STACK.md section 3, points 3-5): the one schema
for instruction tuning and chat, and the template as a named, versioned record. The C++ twin is
include/microtorch/chat_template.hpp; tests/test_chat_template.py gates the two against each
other and against the string builders they replaced.

    python tools/chat_template.py parse ROWS.jsonl --out CONVERSATIONS.jsonl   # + .ledger.json
    python tools/chat_template.py jinja [paperkiln/1]                          # the Jinja string
    python tools/chat_template.py record [paperkiln/1]                         # the record as JSON

A conversation is {"messages": [{"role": "system"|"user"|"assistant", "content": str,
"loss": bool?}, ...], "uri": ..., ...}: at most one system message, first; then user and
assistant strictly alternating from the user. A training conversation ends on an assistant
message, a prompt conversation on a user message. An instruction example is a conversation with
one exchange. The tool role is reserved (refused).

A template renders bos, then every message as open + content + close (the system message by
its mode), then generation_prompt when a reply is wanted. Content holding one of the record's
`reserved` strings (v1: the three role and end markers) is refused, here, in the C++ twin and
in `parse`, so no text can forge a turn.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

# Jonathan's decisions pending (design of 24 Oct 2026): each default sits behind one constant.
SYSTEM_MODE_V1 = "fold"        # "fold" (the system text opens the first user turn) or "none" (refused)
DEFAULT_SFT_TURNS = "all"      # train.sft_loss.turns default: "all", "final" or "each"

SCHEMA = "paperkiln.conversation/1"
ROLES = ("system", "user", "assistant")
RESERVED_ROLES = ("tool",)
SFT_TURNS = ("all", "final", "each")
V1_SPECIALS = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"]  # ids 0-3 of every paperkiln BPE vocabulary


class TemplateError(ValueError):
    pass


def _v1():
    return {"name": "paperkiln", "version": 1, "bos": "", "content": "raw",
            "system": {"mode": SYSTEM_MODE_V1, "join": "\n\n"},
            "user": {"open": "<|user|>", "close": ""},
            "assistant": {"open": "<|assistant|>", "close": "<|endoftext|>"},
            "generation_prompt": "<|assistant|>",
            "stop": {"eos": ["<|endoftext|>"], "turn": ["<|user|>", "<|assistant|>"]},
            "past_assistant": "keep",
            "reserved": ["<|endoftext|>", "<|user|>", "<|assistant|>"]}


def _legacy_word():
    # mtstudio chat_prompt's word-level form: "user: U assistant: A " per past turn, then
    # "user: X assistant:". Stops: the vocabulary's end-of-text word (eos is GGUF-defined) and
    # a generated "user:" / "assistant:" (multi-token sequences).
    return {"name": "paperkiln", "version": "legacy-word", "bos": "", "content": "raw",
            "system": {"mode": SYSTEM_MODE_V1, "join": "\n\n"},
            "user": {"open": "user: ", "close": " "},
            "assistant": {"open": "assistant: ", "close": " "},
            "generation_prompt": "assistant:",
            "stop": {"eos": [], "turn": ["user:", "assistant:"]},
            "past_assistant": "keep",
            "reserved": []}


# Third-party records (ember owns and gates them against HF apply_chat_template, G5). `match`:
# sha256 of the exact tokenizer.chat_template strings known for the family; `markers`: tokens that
# must each be a single vocabulary entry for the marker check (resolution step 4).
_QWEN_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _chatml(name, default_system, past, nothink):
    r = {"name": name, "version": 1, "bos": "", "content": "raw",
         "system": {"mode": "turn", "open": "<|im_start|>system\n", "close": "<|im_end|>\n"},
         "user": {"open": "<|im_start|>user\n", "close": "<|im_end|>\n"},
         "assistant": {"open": "<|im_start|>assistant\n", "close": "<|im_end|>\n"},
         "generation_prompt": "<|im_start|>assistant\n",
         "stop": {"eos": ["<|im_end|>", "<|endoftext|>"], "turn": []},
         "past_assistant": past,
         "reserved": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
         "markers": ["<|im_start|>", "<|im_end|>"]}
    if default_system:
        r["default_system"] = default_system
    if nothink:
        r["generation_prompt_nothink"] = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return r


def _gemma3():
    return {"name": "gemma3", "version": 1, "bos": "<bos>", "content": "trim",
            "system": {"mode": "fold", "join": "\n\n"},
            "user": {"open": "<start_of_turn>user\n", "close": "<end_of_turn>\n"},
            "assistant": {"open": "<start_of_turn>model\n", "close": "<end_of_turn>\n"},
            "generation_prompt": "<start_of_turn>model\n",
            "stop": {"eos": ["<end_of_turn>", "<eos>"], "turn": []},
            "past_assistant": "keep",
            "reserved": ["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>"],
            "markers": ["<start_of_turn>", "<end_of_turn>"]}


def _registry():
    q25 = _chatml("chatml-qwen2.5", _QWEN_SYS, "keep", False)
    q25["match"] = ["d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4"]  # Qwen2.5-0.5B-Instruct
    q25["architectures"] = ["qwen2"]
    q3 = _chatml("chatml-qwen3", "", "strip_think", True)
    q3["match"] = ["57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361"]  # Qwen3-0.6B
    q3["architectures"] = ["qwen3"]
    g3 = _gemma3()
    g3["match"] = ["7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"]  # gemma-3-270m-it chat_template.jinja
    g3["architectures"] = ["gemma3"]
    return {rid(r): r for r in (_v1(), _legacy_word(), q25, q3, g3)}


def rid(rec):
    """A record's id: name/version (paperkiln/1, paperkiln/legacy-word, chatml-qwen3/1)."""
    return f"{rec['name']}/{rec['version']}"


def get(name_or_record="paperkiln/1"):
    """A deep copy of a registry record (by id) or a checked copy of a record given as a dict."""
    if isinstance(name_or_record, dict):
        return check_record(copy.deepcopy(name_or_record))
    reg = _registry()
    if name_or_record not in reg:
        raise TemplateError(f"no chat template {name_or_record!r} (known: {', '.join(reg)})")
    return reg[name_or_record]


V1 = "paperkiln/1"
LEGACY_WORD = "paperkiln/legacy-word"

_KEYS = {"name", "version", "bos", "content", "system", "user", "assistant", "generation_prompt",
         "generation_prompt_nothink", "stop", "past_assistant", "default_system", "reserved", "markers",
         "match", "architectures"}


def check_record(rec):
    """Refuse a record with an unknown field or value (engines refuse, never guess)."""
    for k in rec:
        if k not in _KEYS and not k.startswith("_"):
            raise TemplateError(f"chat template field {k!r} is not implemented")
    for k in ("name", "version", "user", "assistant", "generation_prompt", "stop"):
        if k not in rec:
            raise TemplateError(f"chat template has no {k!r}")
    if rec.get("content", "raw") not in ("raw", "trim"):
        raise TemplateError(f"chat template content {rec['content']!r} is not implemented (raw, trim)")
    if rec.get("past_assistant", "keep") not in ("keep", "strip_think"):
        raise TemplateError(f"chat template past_assistant {rec['past_assistant']!r} is not implemented")
    sysm = rec.get("system", {"mode": "none"})
    if sysm.get("mode") not in ("fold", "turn", "none"):
        raise TemplateError(f"chat template system mode {sysm.get('mode')!r} is not implemented (fold, turn, none)")
    for side in ("user", "assistant"):
        if set(rec[side]) - {"open", "close"}:
            raise TemplateError(f"chat template {side} takes open and close only")
    return rec


# ------------------------------------------------------------------ conversations
def _structure(messages, prompt):
    """(system text or None, the other messages). prompt: the conversation ends on a user message
    (a reply is wanted); else on an assistant message."""
    if not isinstance(messages, list) or not messages:
        raise TemplateError("a conversation needs a non-empty messages list")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise TemplateError("every message needs a role and a string content")
        if m.get("role") in RESERVED_ROLES:
            raise TemplateError(f"role {m['role']!r} is reserved: no template defines it yet")
        if m.get("role") not in ROLES:
            raise TemplateError(f"unknown role {m.get('role')!r}")
    sys_text, rest = None, messages
    if messages[0]["role"] == "system":
        sys_text, rest = messages[0]["content"], messages[1:]
    if not rest:
        raise TemplateError("a system message needs a user message after it")
    for i, m in enumerate(rest):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] == "system":
            raise TemplateError("a system message is allowed only as the first message")
        if m["role"] != want:
            raise TemplateError(f"message {i + (sys_text is not None)} is {m['role']}, expected {want} "
                                "(user and assistant alternate from the user)")
        if "loss" in m and (m["role"] != "assistant" or not isinstance(m["loss"], bool)):
            raise TemplateError("loss is a bool on assistant messages only")
    last = rest[-1]["role"]
    if prompt and last != "user":
        raise TemplateError("a prompt conversation ends on a user message")
    if not prompt and last != "assistant":
        raise TemplateError("a training conversation ends on an assistant message")
    return sys_text, rest


def validate(conv, prompt=False):
    """A stored conversation: the structure, plus non-empty content everywhere. Returns messages."""
    msgs = conv["messages"] if isinstance(conv, dict) else conv
    _structure(msgs, prompt)
    for m in msgs:
        if not m["content"]:
            raise TemplateError(f"empty {m['role']} content")
    return msgs


def render_pieces(messages, tmpl=V1, generation_prompt=False, thinking=None):
    """[(text, role, kind, index)]: kind open | content | close | bos | generation; index = the
    message's position in `messages` (-1 for bos and the generation prompt). Empty pieces are
    left out. generation_prompt=True needs a conversation ending on a user message."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    sys_text, rest = _structure(messages, generation_prompt)
    off = 1 if sys_text is not None else 0
    trim = t.get("content", "raw") == "trim"
    sysm = t.get("system", {"mode": "none"})
    reserved = [r for r in t.get("reserved", []) if r]

    def body(text):
        if trim:
            text = text.strip()  # "trim": Python's str.strip(), as HF's Jinja trim (the C++ twin's strip_py_ws)
        for r in reserved:
            if r in text:
                raise TemplateError(f"content holds the template's reserved text {r!r}")
        return text

    out = []

    def add(text, role, kind, idx):
        if text:
            out.append((text, role, kind, idx))

    add(t.get("bos", ""), "", "bos", -1)
    fold = None
    sys_idx = 0 if sys_text is not None else -1  # -1: the template's default system text
    if sys_text is None and t.get("default_system"):
        sys_text = t["default_system"]
    if sys_text is not None:
        mode = sysm["mode"]
        if mode == "none":
            raise TemplateError("this model's template has no system role")
        if mode == "turn":
            add(sysm.get("open", ""), "system", "open", sys_idx)
            add(body(sys_text), "system", "content", sys_idx)
            add(sysm.get("close", ""), "system", "close", sys_idx)
        else:
            for r in reserved:
                if r in sys_text:
                    raise TemplateError(f"content holds the template's reserved text {r!r}")
            fold = sys_text + sysm.get("join", "")
    last_user = max(i for i, m in enumerate(rest) if m["role"] == "user")
    for i, m in enumerate(rest):
        side = t[m["role"]]
        text = body(m["content"])
        if m["role"] == "user" and fold is not None:
            text, fold = fold + text, None
        if (m["role"] == "assistant" and t.get("past_assistant") == "strip_think" and i < last_user
                and "</think>" in text):
            text = text.split("</think>")[-1].lstrip("\n")
        add(side["open"], m["role"], "open", i + off)
        add(text, m["role"], "content", i + off)
        add(side["close"], m["role"], "close", i + off)
    if generation_prompt:
        g = t["generation_prompt"]
        if thinking is False and "generation_prompt_nothink" in t:
            g = t["generation_prompt_nothink"]
        add(g, "", "generation", -1)
    elif t.get("past_assistant") == "strip_think":
        raise TemplateError("a strip_think template renders prompt conversations only (HF re-wraps the "
                            "reasoning of replies after the last user message; prompts never have one)")
    return out


def render(messages, tmpl=V1, generation_prompt=False, thinking=None):
    return "".join(p[0] for p in render_pieces(messages, tmpl, generation_prompt, thinking))


def render_document(messages, tmpl=V1):
    """A conversation as one pretraining document: its whole exchanges rendered, then "\\n"
    (the corpus rule of _common.render_dialogue; a trailing unanswered user message is left out)."""
    msgs = list(messages)
    if msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    return render(msgs, tmpl) + "\n"


def from_pairs(history, user=None, system=None):
    """Messages from (user, assistant) pairs, then the new user message when given."""
    msgs = [{"role": "system", "content": system}] if system is not None else []
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    if user is not None:
        msgs.append({"role": "user", "content": user})
    return msgs


def prompt(history, user, tmpl=V1, system=None):
    """The served prompt for a new user message after (user, assistant) pairs."""
    return render(from_pairs(history, user, system), tmpl, generation_prompt=True)


def from_prompt(p):
    """A prompt-set row ({"prompt", "system"?} or a bare string) as a prompt conversation."""
    if isinstance(p, str):
        return [{"role": "user", "content": p}]
    msgs = [{"role": "system", "content": p["system"]}] if p.get("system") is not None else []
    return msgs + [{"role": "user", "content": p["prompt"]}]


def from_carry_item(item, mode, replies=None):
    """A Carry-5 / Carry-12 item's target prompt conversation in one history mode: gold (canned
    replies), self (`replies`, the model's own replies to the earlier turns), twin or noctx."""
    tt = item["target_turn"]
    turns = item["turns"]
    if mode == "gold":
        return from_pairs([(t["user"], t["gold"]) for t in turns[:tt - 1]], turns[tt - 1]["user"])
    if mode == "self":
        if replies is None or len(replies) < tt - 1:
            raise TemplateError("self mode needs the model's replies to the earlier turns")
        return from_pairs([(t["user"], r) for t, r in zip(turns[:tt - 1], replies)], turns[tt - 1]["user"])
    if mode in ("twin", "noctx"):
        return [{"role": "user", "content": item[mode]}]
    raise TemplateError(f"unknown Carry mode {mode!r}")


# ------------------------------------------------------------------ loss policy
def trained(messages, turns=DEFAULT_SFT_TURNS):
    """Per-row lists of trained message indices under train.sft_loss.turns: "all" (every
    assistant message, one row), "final" (the last assistant message, one row) or "each" (one
    row per assistant message k, messages 0..k). A message with "loss": false is never trained.
    Returns [(messages of the row, [trained indices])]."""
    if turns not in SFT_TURNS:
        raise TemplateError(f"train.sft_loss.turns must be one of {SFT_TURNS}, got {turns!r}")
    asst = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("loss", True)]
    if turns == "all":
        return [(messages, asst)]
    if turns == "final":
        return [(messages, asst[-1:] if asst and asst[-1] == len(messages) - 1 else [])]
    return [(messages[:k + 1], [k]) for k in asst]


def to_segments(messages, tmpl=V1):
    """mtstudio's legacy {"segments"} row for a training conversation under "all": runs of
    untrained pieces (loss false) and each trained reply's content (loss true; mtstudio appends
    the end-of-text the template closes a reply with)."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    if t["assistant"]["close"] != "<|endoftext|>":
        raise TemplateError("segments rows need a template whose replies close with <|endoftext|>")
    segs = []
    for text, role, kind, idx in render_pieces(messages, t):
        if role == "assistant" and kind == "close" and messages[idx].get("loss", True):
            continue  # mtstudio appends it after the trained reply
        loss = role == "assistant" and kind == "content" and messages[idx].get("loss", True)
        if segs and not loss and not segs[-1]["loss"]:
            segs[-1]["text"] += text
        else:
            segs.append({"text": text, "loss": loss})
    return {"segments": segs}


# ------------------------------------------------------------------ parsing rendered v1 rows
def parse_v1(text):
    """Messages from rendered v1 text: whole exchanges, optionally followed by a prompt
    (<|user|>U<|assistant|>, generation prompt). Returns (messages, is_prompt). Refuses text that
    does not round-trip (a marker inside content cannot be told from structure)."""
    U, A, E = "<|user|>", "<|assistant|>", "<|endoftext|>"
    msgs, p, is_prompt = [], 0, False
    while p < len(text):
        if not text.startswith(U, p):
            raise TemplateError(f"expected <|user|> at offset {p}")
        a = text.find(A, p + len(U))
        if a < 0:
            raise TemplateError("a user turn without <|assistant|>")
        msgs.append({"role": "user", "content": text[p + len(U):a]})
        e = text.find(E, a + len(A))
        if e < 0:
            if a + len(A) != len(text):
                raise TemplateError("a reply without <|endoftext|>")
            is_prompt = True
            break
        msgs.append({"role": "assistant", "content": text[a + len(A):e]})
        p = e + len(E)
    if not msgs:
        raise TemplateError("no exchange")
    if render(msgs, V1, generation_prompt=is_prompt) != text:
        raise TemplateError("the text does not round-trip through the v1 template")
    return msgs, is_prompt


def parse_row(row):
    """A legacy rendered row as (conversation, sft_loss turns that reproduces its targets, form).
    {"prompt", "completion"}: the per-turn form, reproduced by "final". {"segments"}: "all" with
    "loss": false on replies the row did not train. {"text"}: a pretraining document."""
    if "segments" in row:
        text = "".join(s["text"] + ("<|endoftext|>" if s["loss"] else "") for s in row["segments"])
        msgs, is_prompt = parse_v1(text)
        if is_prompt:
            raise TemplateError("a segments row ends without a trained reply")
        trained_text = [s["text"] for s in row["segments"] if s["loss"]]
        k = 0
        for m in msgs:
            if m["role"] == "assistant":
                if k < len(trained_text) and m["content"] == trained_text[k]:
                    k += 1
                else:
                    m["loss"] = False
        if k != len(trained_text):
            raise TemplateError("a loss segment is not a whole reply")
        if to_segments(msgs) != {"segments": row["segments"]}:
            raise TemplateError("the segments do not round-trip")
        return {"messages": msgs}, "all", "segments"
    if "prompt" in row and "completion" in row:
        msgs, is_prompt = parse_v1(row["prompt"] + row["completion"] + "<|endoftext|>")
        if row["prompt"] != render(msgs[:-1], V1, generation_prompt=True):
            raise TemplateError("the prompt is not a v1 prompt")
        return {"messages": msgs}, "final", "prompt_completion"
    if "text" in row:
        t = row["text"]
        msgs, is_prompt = parse_v1(t[:-1] if t.endswith("\n") else t)
        if is_prompt:
            raise TemplateError("a document ends without a reply")
        return {"messages": msgs}, None, "document"
    raise TemplateError("a row needs segments, prompt + completion, or text")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_parse(a):
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; parse writes a new file")
    counts, refused, kept = {}, [], 0
    tmp = a.out + ".tmp"
    with open(a.rows, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8", newline="\n") as g:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                conv, turns, form = parse_row(row)
            except TemplateError as e:
                refused.append({"line": n, "reason": str(e)})
                continue
            for k in ("uri", "id", "source"):
                if k in row:
                    conv[k] = row[k]
            if turns:
                conv["_sft_loss_turns"] = turns
            g.write(json.dumps(conv, ensure_ascii=False) + "\n")
            counts[form] = counts.get(form, 0) + 1
            kept += 1
    os.replace(tmp, a.out)
    ledger = {"schema": SCHEMA, "template": V1, "source": os.path.abspath(a.rows),
              "source_sha256": sha256_file(a.rows), "out": os.path.abspath(a.out),
              "out_sha256": sha256_file(a.out), "rows": kept, "by_form": counts,
              "refused": len(refused), "refused_lines": refused[:100]}
    with open(a.out + ".ledger.json.tmp", "w", encoding="utf-8", newline="\n") as g:
        json.dump(ledger, g, indent=1)
    os.replace(a.out + ".ledger.json.tmp", a.out + ".ledger.json")
    return {k: ledger[k] for k in ("rows", "by_form", "refused", "out_sha256")}


# ------------------------------------------------------------------ Jinja (llama.cpp, LM Studio)
def _jstr(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def jinja(tmpl=V1):
    """tokenizer.chat_template for a paperkiln record: the same text as render() for HF-style
    messages and add_generation_prompt (tests/test_chat_template.py renders both with jinja2)."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    if t["name"] != "paperkiln":
        raise TemplateError("jinja() writes paperkiln records only; third-party files carry their own")
    if t.get("content", "raw") != "raw" or t.get("past_assistant", "keep") != "keep" or t.get("default_system"):
        raise TemplateError("jinja() covers raw content, kept replies and no default system")
    sysm = t.get("system", {"mode": "none"})
    u, a = t["user"], t["assistant"]
    j = []
    if t.get("bos"):
        j.append("{{- " + _jstr(t["bos"]) + " -}}")
    j.append("{%- if messages and messages[0]['role'] == 'system' -%}"
             "{%- set sys = messages[0]['content'] -%}{%- set msgs = messages[1:] -%}"
             "{%- else -%}{%- set sys = none -%}{%- set msgs = messages -%}{%- endif -%}")
    if sysm["mode"] == "none":
        j.append("{%- if sys is not none -%}{{ raise_exception('this model\\'s template has no system role') }}"
                 "{%- endif -%}")
    elif sysm["mode"] == "turn":
        j.append("{%- if sys is not none -%}{{- " + _jstr(sysm.get("open", "")) + " + sys + "
                 + _jstr(sysm.get("close", "")) + " -}}{%- endif -%}")
    j.append("{%- for m in msgs -%}"
             "{%- if (m['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
             "{{ raise_exception('user and assistant messages alternate from the user') }}{%- endif -%}"
             "{%- if m['role'] == 'user' -%}{{- " + _jstr(u["open"]) + " -}}")
    if sysm["mode"] == "fold":
        j.append("{%- if loop.first and sys is not none -%}{{- sys + " + _jstr(sysm.get("join", ""))
                 + " -}}{%- endif -%}")
    j.append("{{- m['content'] + " + _jstr(u["close"]) + " -}}"
             "{%- else -%}{{- " + _jstr(a["open"]) + " + m['content'] + " + _jstr(a["close"]) + " -}}"
             "{%- endif -%}{%- endfor -%}"
             "{%- if add_generation_prompt -%}{{- " + _jstr(t["generation_prompt"]) + " -}}{%- endif -%}")
    return "".join(j)


def modelfile(gguf_name, tmpl=V1):
    """An Ollama Modelfile for a paperkiln GGUF: FROM, the TEMPLATE in Go syntax (system folded
    or a turn, as the record says) and a stop parameter per stop string."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    if t["name"] != "paperkiln":
        raise TemplateError("modelfile() writes paperkiln records only")
    sysm = t.get("system", {"mode": "none"})
    u, a = t["user"], t["assistant"]

    def go(s):
        return s.replace("{{", "{{ \"{{\" }}")

    msgs = ("{{- range $i, $m := .Messages }}"
            "{{- if eq $m.Role \"user\" }}" + go(u["open"]))
    if sysm["mode"] == "fold":
        msgs += "{{- if and $.System (eq $i 0) }}{{ $.System }}" + go(sysm.get("join", "")) + "{{ end }}"
    msgs += ("{{ $m.Content }}" + go(u["close"]) +
             "{{- else if eq $m.Role \"assistant\" }}" + go(a["open"]) + "{{ $m.Content }}" + go(a["close"]) +
             "{{- end }}{{- end }}" + go(t["generation_prompt"]))
    head = go(t.get("bos", ""))
    if sysm["mode"] == "turn":
        head += "{{- if .System }}" + go(sysm.get("open", "")) + "{{ .System }}" + go(sysm.get("close", "")) + "{{ end }}"
    lines = [f"# Ollama Modelfile for {gguf_name}: chat template {rid(t)} (LOCAL_LLM_STACK.md section 3, point 3).",
             f"FROM ./{gguf_name}", 'TEMPLATE """' + head + msgs + '"""']
    for s in t["stop"]["eos"] + t["stop"]["turn"]:
        lines.append(f"PARAMETER stop {json.dumps(s)}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ resolution (section 3, point 3)
def resolve(meta):
    """(record or None, why) for a GGUF's metadata dict, in the engines' order: (1) a declared
    paperkiln.chat_template; (2) ids 0-3 are paperkiln's specials (8k "bpe" and 16k "llama3_bpe"
    alike): paperkiln/1; (3) a word-level vocabulary: paperkiln/legacy-word; (4) a third-party
    tokenizer.chat_template by sha256 of a known exact string, else the marker check (each of the
    record's markers a single token and general.architecture among its architectures);
    (5) no chat mode."""
    if "paperkiln.chat_template" in meta:
        return get(json.loads(meta["paperkiln.chat_template"])), "declared"
    tokens = meta.get("tokenizer.ggml.tokens") or []
    norm = [x.decode("latin-1") if isinstance(x, bytes) else x for x in tokens[:4]]
    if norm == V1_SPECIALS:
        return get(V1), "paperkiln specials at ids 0-3"
    if meta.get("tokenizer.ggml.model") == "word":
        return get(LEGACY_WORD), "word-level vocabulary"
    jt = meta.get("tokenizer.chat_template", "")
    reg = _registry()
    if jt:
        h = hashlib.sha256(jt.encode("utf-8")).hexdigest()
        for r in reg.values():
            if h in r.get("match", []):
                return copy.deepcopy(r), f"tokenizer.chat_template sha256 {h[:12]}"
    vocab = set(x for x in tokens if isinstance(x, str))
    arch = meta.get("general.architecture", "")
    for r in reg.values():
        if r.get("markers") and all(m in vocab for m in r["markers"]) and arch in r.get("architectures", []):
            return copy.deepcopy(r), f"markers {r['markers']} and architecture {arch}"
    return None, ("no declared template, no paperkiln specials, and no known tokenizer.chat_template or "
                  "markers: raw completion only")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("rows")
    p.add_argument("--out", required=True)
    j = sub.add_parser("jinja")
    j.add_argument("template", nargs="?", default=V1)
    r = sub.add_parser("record")
    r.add_argument("template", nargs="?", default=V1)
    a = ap.parse_args(argv)
    if a.cmd == "parse":
        print(json.dumps(cmd_parse(a)))
    elif a.cmd == "jinja":
        print(jinja(a.template))
    else:
        print(json.dumps(get(a.template), ensure_ascii=False))


if __name__ == "__main__":
    main()
