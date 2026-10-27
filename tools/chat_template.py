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
one exchange. Under a record without a `tool` field (paperkiln/1 and every third-party record)
the tool role is reserved (refused).

Schema v2 (paperkiln/2, LOCAL_LLM_STACK.md section 3, point 3g) adds tool turns: an assistant
message may carry "tool_calls": [{"id"?, "type"?: "function", "function": {"name", "arguments"}}]
(arguments a string, rendered verbatim), followed by exactly as many {"role": "tool", "content",
"tool_call_id"?} messages in call order, then an assistant message; and a conversation may carry
"tools": [JSON objects] (tool definitions), rendered inside the system turn. A prompt conversation
then ends on a user or a tool message.

A template renders bos, then every message as open + content + close (the system message by
its mode), then generation_prompt when a reply is wanted. Content holding one of the record's
`reserved` strings (v1: the three role and end markers; v2: those and its eight control tokens)
is refused, here, in the C++ twin and in `parse`, so no text can forge a turn.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

# v1's defaults, each behind one constant shared with include/microtorch/chat_template.hpp.
SYSTEM_MODE_V1 = "fold"        # "fold" (the system text opens the first user turn) or "none" (refused)
DEFAULT_SFT_TURNS = "all"      # train.sft_loss.turns default: "all", "final" or "each"

SCHEMA = "paperkiln.conversation/1"
SCHEMA_V2 = "paperkiln.conversation/2"
ROLES = ("system", "user", "assistant")
RESERVED_ROLES = ("tool",)
SFT_TURNS = ("all", "final", "each")
V1_SPECIALS = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"]  # ids 0-3 of every paperkiln BPE vocabulary
# paperkiln/2's control tokens, appended in this order after a vocabulary's last id (16k Llama 3-scheme
# vocabulary: ids 16384-16391), GGUF token type 3 (control); merges and every text id unchanged.
V2_CONTROL = ["<|system|>", "<|tool_call|>", "<|tool|>", "<|spare_3|>", "<|spare_4|>", "<|spare_5|>",
              "<|spare_6|>", "<|spare_7|>"]


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


def _v2():
    # Schema v2: the system message is its own turn behind its own control token (no fold), tool calls
    # ride in the assistant turn after <|tool_call|>, tool results are turns behind <|tool|>, and tool
    # definitions render inside the system turn. A user/assistant-only conversation renders as v1.
    return {"name": "paperkiln", "version": 2, "bos": "", "content": "raw",
            "system": {"mode": "turn", "open": "<|system|>", "close": ""},
            "user": {"open": "<|user|>", "close": ""},
            "assistant": {"open": "<|assistant|>", "close": "<|endoftext|>"},
            "tool": {"open": "<|tool|>", "close": ""},
            "tool_call": {"open": "<|tool_call|>", "join": "\n"},
            "tools": {"join": "\n\n", "sep": "\n"},
            "generation_prompt": "<|assistant|>",
            "stop": {"eos": ["<|endoftext|>"], "turn": ["<|user|>", "<|assistant|>", "<|system|>", "<|tool|>"]},
            "past_assistant": "keep",
            "reserved": ["<|endoftext|>", "<|user|>", "<|assistant|>"] + V2_CONTROL}


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
    return {rid(r): r for r in (_v1(), _v2(), _legacy_word(), q25, q3, g3)}


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
V2 = "paperkiln/2"
LEGACY_WORD = "paperkiln/legacy-word"

_KEYS = {"name", "version", "bos", "content", "system", "user", "assistant", "generation_prompt",
         "generation_prompt_nothink", "stop", "past_assistant", "default_system", "reserved", "markers",
         "match", "architectures", "tool", "tool_call", "tools"}


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
    for side in ("user", "assistant", "tool"):
        if side in rec and set(rec[side]) - {"open", "close"}:
            raise TemplateError(f"chat template {side} takes open and close only")
    if "tool_call" in rec and set(rec["tool_call"]) - {"open", "join"}:
        raise TemplateError("chat template tool_call takes open and join only")
    if "tools" in rec and set(rec["tools"]) - {"join", "sep"}:
        raise TemplateError("chat template tools takes join and sep only")
    if ("tool" in rec) != ("tool_call" in rec):
        raise TemplateError("chat template tool and tool_call come together")
    if "tools" in rec and "tool" not in rec:
        raise TemplateError("chat template tools needs tool and tool_call")
    if "tool" in rec and sysm.get("mode") != "turn":
        raise TemplateError("chat template tool turns need system mode turn")
    return rec


# ------------------------------------------------------------------ conversations
def has_tools(tmpl):
    """Whether a record defines tool turns (paperkiln/2)."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    return "tool" in t


def _calls(m):
    """An assistant message's tool calls as [(id or None, name, arguments)]; [] when it has none."""
    tc = m.get("tool_calls")
    if tc is None:
        return []
    if not isinstance(tc, list) or not tc:
        raise TemplateError("tool_calls is a non-empty list")
    out = []
    for c in tc:
        if not isinstance(c, dict) or not isinstance(c.get("function"), dict):
            raise TemplateError("a tool call is {id?, type?: function, function: {name, arguments}}")
        if c.get("type", "function") != "function":
            raise TemplateError(f"tool call type {c.get('type')!r} is not implemented (function)")
        f = c["function"]
        name, args = f.get("name"), f.get("arguments", "")
        if not isinstance(name, str) or not name or "\n" in name:
            raise TemplateError("a tool call's name is a non-empty string without a newline")
        if not isinstance(args, str):
            raise TemplateError("a tool call's arguments are a string (rendered verbatim)")
        if "id" in c and not isinstance(c["id"], str):
            raise TemplateError("a tool call's id is a string")
        out.append((c.get("id"), name, args))
    return out


def _structure(messages, prompt, tools_ok=False):
    """(system text or None, the other messages). prompt: the conversation ends on a user message
    (a reply is wanted), or under v2 on a tool message; else on an assistant message. tools_ok: the
    record defines tool turns (v2): an assistant message with k tool calls is followed by exactly k
    tool messages in call order, then an assistant message; an assistant message without calls by a
    user message."""
    if not isinstance(messages, list) or not messages:
        raise TemplateError("a conversation needs a non-empty messages list")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise TemplateError("every message needs a role and a string content")
        if m.get("role") in RESERVED_ROLES and not tools_ok:
            raise TemplateError(f"role {m['role']!r} is reserved: no template defines it yet")
        if m.get("role") not in ROLES + (RESERVED_ROLES if tools_ok else ()):
            raise TemplateError(f"unknown role {m.get('role')!r}")
        if "tool_calls" in m and (not tools_ok or m["role"] != "assistant"):
            raise TemplateError("tool_calls belong on assistant messages of a template with tool turns")
        if "tool_call_id" in m and m["role"] != "tool":
            raise TemplateError("tool_call_id belongs on tool messages")
    sys_text, rest = None, messages
    if messages[0]["role"] == "system":
        sys_text, rest = messages[0]["content"], messages[1:]
    if not rest:
        raise TemplateError("a system message needs a user message after it")
    want, pending = "user", []
    for i, m in enumerate(rest):
        if m["role"] == "system":
            raise TemplateError("a system message is allowed only as the first message")
        if m["role"] != want:
            if not tools_ok:
                raise TemplateError(f"message {i + (sys_text is not None)} is {m['role']}, expected {want} "
                                    "(user and assistant alternate from the user)")
            raise TemplateError(f"message {i + (sys_text is not None)} is {m['role']}, expected {want} (user, "
                                "then assistant; an assistant message with k tool calls, then k tool messages, "
                                "then assistant)")
        if "loss" in m and (m["role"] != "assistant" or not isinstance(m["loss"], bool)):
            raise TemplateError("loss is a bool on assistant messages only")
        if m["role"] == "user":
            want = "assistant"
        elif m["role"] == "assistant":
            pending = _calls(m)
            want = "tool" if pending else "user"
        else:
            cid = pending.pop(0)[0]
            if "tool_call_id" in m and m["tool_call_id"] != cid:
                raise TemplateError(f"tool_call_id {m['tool_call_id']!r} is not the call's id {cid!r} "
                                    "(tool messages answer the calls in order)")
            want = "tool" if pending else "assistant"
    last = rest[-1]["role"]
    if prompt and not (last == "user" or (last == "tool" and not pending)):
        raise TemplateError("a prompt conversation ends on a user message"
                            + (" or on the last tool message of a call" if tools_ok else ""))
    if not prompt and last != "assistant":
        raise TemplateError("a training conversation ends on an assistant message")
    return sys_text, rest


def validate(conv, prompt=False, tmpl=V1):
    """A stored conversation: the structure, plus non-empty content everywhere (an assistant message
    with tool calls may have empty content). Returns messages."""
    msgs = conv["messages"] if isinstance(conv, dict) else conv
    _structure(msgs, prompt, has_tools(tmpl))
    for m in msgs:
        if not m["content"] and not (m["role"] == "assistant" and m.get("tool_calls")):
            raise TemplateError(f"empty {m['role']} content")
    return msgs


def _has_float(x):
    if isinstance(x, float):
        return True
    if isinstance(x, dict):
        return any(_has_float(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_float(v) for v in x)
    return False


def tool_json(d):
    """A tool definition's text: Python's json.dumps(d, ensure_ascii=False) (separators ", " and ": ",
    member order kept), which is HF transformers' Jinja `tojson`; the C++ twin's detail::dump_py writes
    the same. Non-integer numbers are refused (the one place two serialisers could differ)."""
    if not isinstance(d, dict):
        raise TemplateError("a tool definition is a JSON object")
    if _has_float(d):
        raise TemplateError("a tool definition holds a non-integer number (refused: its text is not portable)")
    return json.dumps(d, ensure_ascii=False)


def render_pieces(messages, tmpl=V1, generation_prompt=False, thinking=None, tools=None):
    """[(text, role, kind, index)]: kind open | content | call_open | call | close | bos | generation;
    index = the message's position in `messages` (-1 for bos and the generation prompt). Empty pieces
    are left out. generation_prompt=True needs a prompt conversation (ending on a user message, or under
    v2 on a tool message). tools: tool definitions (v2), rendered inside the system turn."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    tools_ok = "tool" in t
    if tools and "tools" not in t:
        raise TemplateError("this model's template has no tool definitions")
    sys_text, rest = _structure(messages, generation_prompt, tools_ok)
    off = 1 if sys_text is not None else 0
    trim = t.get("content", "raw") == "trim"
    sysm = t.get("system", {"mode": "none"})
    reserved = [r for r in t.get("reserved", []) if r]

    def check(text):
        for r in reserved:
            if r in text:
                raise TemplateError(f"content holds the template's reserved text {r!r}")
        return text

    def body(text):
        if trim:
            text = text.strip()  # "trim": Python's str.strip(), as HF's Jinja trim (the C++ twin's strip_py_ws)
        return check(text)

    out = []

    def add(text, role, kind, idx):
        if text:
            out.append((text, role, kind, idx))

    add(t.get("bos", ""), "", "bos", -1)
    fold = None
    sys_idx = 0 if sys_text is not None else -1  # -1: the template's default system text or the tools alone
    if sys_text is None and t.get("default_system"):
        sys_text = t["default_system"]
    defs = [check(tool_json(d)) for d in (tools or [])]
    if defs:
        tl = t["tools"]
        head = body(sys_text) if sys_text is not None else ""
        sys_text = (head + tl.get("join", "") if head else "") + tl.get("sep", "").join(defs)
    if sys_text is not None:
        mode = sysm["mode"]
        if mode == "none":
            raise TemplateError("this model's template has no system role")
        if mode == "turn":
            add(sysm.get("open", ""), "system", "open", sys_idx)
            add(sys_text if defs else body(sys_text), "system", "content", sys_idx)
            add(sysm.get("close", ""), "system", "close", sys_idx)
        else:
            check(sys_text)
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
        if m["role"] == "assistant":
            for _cid, name, args in _calls(m):
                add(t["tool_call"]["open"], "assistant", "call_open", i + off)
                add(check(name) + t["tool_call"].get("join", "") + check(args), "assistant", "call", i + off)
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


def render(messages, tmpl=V1, generation_prompt=False, thinking=None, tools=None):
    return "".join(p[0] for p in render_pieces(messages, tmpl, generation_prompt, thinking, tools))


def exchanges(messages):
    """[(start, end)] message index ranges of the exchanges after the system message: a user message
    and everything before the next user message (assistant replies, tool calls, tool results). The
    context rule (section 3, point 5) drops the oldest whole exchange; the system turn is pinned."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "user":
            if out:
                out[-1][1] = i
            out.append([i, len(messages)])
    return [tuple(x) for x in out]


def parse_reply(ids, call_id, decode):
    """A generated reply (ids without the stop token) under v2, split on ids, never on text: the ids
    are cut at every `call_id` (the <|tool_call|> control token), and each part decoded. Part 0 is the
    content; every later part must be name + "\\n" + arguments with a non-empty name. Returns
    {"content", "tool_calls": [{"id": "call_<k>", "type": "function", "function": {"name", "arguments"}}]}
    (no tool_calls when the ids hold no call_id), or None when a part does not parse (the engine then
    returns the whole decoded reply as content). BPE pieces that spell "<|tool_call|>" are content."""
    parts, cur = [], []
    for x in ids:
        if x == call_id:
            parts.append(cur)
            cur = []
        else:
            cur.append(x)
    parts.append(cur)
    content = decode(parts[0])
    calls = []
    for k, p in enumerate(parts[1:]):
        text = decode(p)
        nl = text.find("\n")
        if nl <= 0:
            return None
        calls.append({"id": f"call_{k}", "type": "function",
                      "function": {"name": text[:nl], "arguments": text[nl + 1:]}})
    return {"content": content, "tool_calls": calls}


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


def _jinja_tools(t):
    """The Jinja of a record with tool turns (paperkiln/2): HF-style messages (tool_calls[i].function
    {name, arguments}), `tools` rendered through HF transformers' `tojson` (json.dumps(x,
    ensure_ascii=False), as tool_json), add_generation_prompt. The structure rules are the engines'."""
    sysm, u, a, tl, tc, tools = t["system"], t["user"], t["assistant"], t["tool"], t["tool_call"], t.get("tools")
    j = []
    if t.get("bos"):
        j.append("{{- " + _jstr(t["bos"]) + " -}}")
    j.append("{%- if messages and messages[0]['role'] == 'system' -%}"
             "{%- set sys = messages[0]['content'] -%}{%- set msgs = messages[1:] -%}"
             "{%- else -%}{%- set sys = none -%}{%- set msgs = messages -%}{%- endif -%}")
    if tools:
        j.append("{%- if tools -%}{%- set ns = namespace(d='') -%}{%- for x in tools -%}"
                 "{%- if not loop.first -%}{%- set ns.d = ns.d + " + _jstr(tools.get("sep", "")) + " -%}{%- endif -%}"
                 "{%- set ns.d = ns.d + (x | tojson) -%}{%- endfor -%}"
                 "{%- if sys -%}{%- set sys = sys + " + _jstr(tools.get("join", "")) + " + ns.d -%}"
                 "{%- else -%}{%- set sys = ns.d -%}{%- endif -%}{%- endif -%}")
    j.append("{%- if sys is not none -%}{{- " + _jstr(sysm.get("open", "")) + " + sys + "
             + _jstr(sysm.get("close", "")) + " -}}{%- endif -%}")
    j.append("{%- for m in msgs -%}"
             "{%- if m['role'] == 'user' -%}{{- " + _jstr(u["open"]) + " + m['content'] + " + _jstr(u["close"]) + " -}}"
             "{%- elif m['role'] == 'assistant' -%}{{- " + _jstr(a["open"]) + " + m['content'] -}}"
             "{%- for c in (m['tool_calls'] or []) -%}{{- " + _jstr(tc["open"]) + " + c['function']['name'] + "
             + _jstr(tc.get("join", "")) + " -}}"
             "{%- if c['function']['arguments'] is string -%}{{- c['function']['arguments'] -}}"
             "{%- else -%}{{- c['function']['arguments'] | tojson -}}{%- endif -%}{%- endfor -%}"
             "{{- " + _jstr(a["close"]) + " -}}"
             "{%- elif m['role'] == 'tool' -%}{{- " + _jstr(tl["open"]) + " + m['content'] + " + _jstr(tl["close"]) + " -}}"
             "{%- else -%}{{ raise_exception('roles: one leading system message, then user, assistant and tool') }}"
             "{%- endif -%}{%- endfor -%}"
             "{%- if add_generation_prompt -%}{{- " + _jstr(t["generation_prompt"]) + " -}}{%- endif -%}")
    return "".join(j)


def _modelfile_tools(gguf_name, t):
    """Ollama's TEMPLATE for a record with tool turns. System, user, assistant and tool turns are the
    record's text; tool definitions and call arguments go through Ollama's `json` function (Go's JSON),
    which is not the record's tojson text, so a tools request through Ollama renders differently."""
    sysm, u, a, tl, tc, tools = t["system"], t["user"], t["assistant"], t["tool"], t["tool_call"], t.get("tools") or {}

    def go(s):
        return s.replace("{{", "{{ \"{{\" }}")

    head = go(t.get("bos", ""))
    head += ("{{ if or .System .Tools }}" + go(sysm.get("open", "")) + "{{ if .System }}{{ .System }}{{ end }}"
             "{{ if .Tools }}{{ if .System }}" + go(tools.get("join", "")) + "{{ end }}"
             "{{ range $i, $t := .Tools }}{{ if $i }}" + go(tools.get("sep", "")) + "{{ end }}{{ json $t }}{{ end }}"
             "{{ end }}" + go(sysm.get("close", "")) + "{{ end }}")
    msgs = ("{{ range .Messages }}{{ if eq .Role \"user\" }}" + go(u["open"]) + "{{ .Content }}" + go(u["close"]) +
            "{{ else if eq .Role \"assistant\" }}" + go(a["open"]) + "{{ .Content }}"
            "{{ range .ToolCalls }}" + go(tc["open"]) + "{{ .Function.Name }}" + go(tc.get("join", "")) +
            "{{ json .Function.Arguments }}{{ end }}" + go(a["close"]) +
            "{{ else if eq .Role \"tool\" }}" + go(tl["open"]) + "{{ .Content }}" + go(tl["close"]) +
            "{{ end }}{{ end }}" + go(t["generation_prompt"]))
    lines = [f"# Ollama Modelfile for {gguf_name}: chat template {rid(t)} (LOCAL_LLM_STACK.md section 3, point 3).",
             f"FROM ./{gguf_name}", 'TEMPLATE """' + head + msgs + '"""']
    for s in t["stop"]["eos"] + t["stop"]["turn"]:
        lines.append(f"PARAMETER stop {json.dumps(s)}")
    return "\n".join(lines) + "\n"


def jinja(tmpl=V1):
    """tokenizer.chat_template for a paperkiln record: the same text as render() for HF-style
    messages and add_generation_prompt (tests/test_chat_template.py renders both with jinja2)."""
    t = tmpl if isinstance(tmpl, dict) else get(tmpl)
    if t["name"] != "paperkiln":
        raise TemplateError("jinja() writes paperkiln records only; third-party files carry their own")
    if t.get("content", "raw") != "raw" or t.get("past_assistant", "keep") != "keep" or t.get("default_system"):
        raise TemplateError("jinja() covers raw content, kept replies and no default system")
    if "tool" in t:
        return _jinja_tools(t)
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
    if "tool" in t:
        return _modelfile_tools(gguf_name, t)
    sysm = t.get("system", {"mode": "none"})
    u, a = t["user"], t["assistant"]

    def go(s):
        return s.replace("{{", "{{ \"{{\" }}")

    # No action trims whitespace ({{- / -}}): the record's text is emitted exactly. Ollama keeps a system
    # message in .Messages as well as in .System, so the fold goes to the first user message, which is
    # .Messages[1] then, not to index 0 (that index test dropped every system message, 26 Oct 2026).
    msgs = ("{{ $first := true }}{{ range .Messages }}"
            "{{ if eq .Role \"user\" }}" + go(u["open"]))
    if sysm["mode"] == "fold":
        msgs += "{{ if and $first $.System }}{{ $.System }}" + go(sysm.get("join", "")) + "{{ end }}"
    msgs += ("{{ $first = false }}{{ .Content }}" + go(u["close"]) +
             "{{ else if eq .Role \"assistant\" }}" + go(a["open"]) + "{{ .Content }}" + go(a["close"]) +
             "{{ end }}{{ end }}" + go(t["generation_prompt"]))
    head = go(t.get("bos", ""))
    if sysm["mode"] == "turn":
        head += "{{ if .System }}" + go(sysm.get("open", "")) + "{{ .System }}" + go(sysm.get("close", "")) + "{{ end }}"
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
