"""The chat profile's template rendering (IR_SPEC.md section 8.1), the reference for the conformance suite's
chat cases (conformance/chat/render.json) and the adapter method render_chat (section 11.3).

    check_record(record)                         -> record      ChatError for a field or value not implemented
    render(record, messages, generation_prompt=False, thinking=None) -> str   ChatError when it cannot render

Standard library only. The same rules as paperkiln's tools/chat_template.py (render_pieces) and its C++ twin
include/microtorch/chat_template.hpp, which ember vendors: section 8.1.1's structure, then bos, each message
as open + content + close, and the generation prompt.
"""
from __future__ import annotations

ROLES = ("system", "user", "assistant")
RESERVED_ROLES = ("tool",)
FIELDS = {"name", "version", "bos", "content", "system", "user", "assistant", "generation_prompt",
          "generation_prompt_nothink", "stop", "past_assistant", "default_system", "reserved", "markers",
          "match", "architectures"}
# Section 8.1.3, content "trim": these code points are stripped from both ends of a message's own text (the
# whitespace of Python's str.strip(), as HF's Jinja `trim`).
TRIM = frozenset([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680,
                  *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


class ChatError(ValueError):
    pass


def check_record(rec):
    """A record whose every field and value is implemented (section 8.1.4 step 1: refused, never guessed)."""
    if not isinstance(rec, dict):
        raise ChatError("a chat template record is a JSON object")
    for k in rec:
        if k not in FIELDS and not k.startswith("_"):
            raise ChatError(f"chat template field {k!r} is not implemented")
    for k in ("name", "version", "user", "assistant", "generation_prompt", "stop"):
        if k not in rec:
            raise ChatError(f"chat template has no {k!r}")
    if rec.get("content", "raw") not in ("raw", "trim"):
        raise ChatError(f"chat template content {rec['content']!r} is not implemented (raw, trim)")
    if rec.get("past_assistant", "keep") not in ("keep", "strip_think"):
        raise ChatError(f"chat template past_assistant {rec['past_assistant']!r} is not implemented")
    if rec.get("system", {"mode": "none"}).get("mode") not in ("fold", "turn", "none"):
        raise ChatError(f"chat template system mode {rec['system'].get('mode')!r} is not implemented")
    for side in ("user", "assistant"):
        if not isinstance(rec[side], dict) or set(rec[side]) - {"open", "close"}:
            raise ChatError(f"chat template {side} takes open and close only")
    return rec


def structure(messages, prompt):
    """(system text or None, the other messages): section 8.1.1. prompt: the conversation ends on a user
    message (a reply is wanted), else on an assistant message."""
    if not isinstance(messages, list) or not messages:
        raise ChatError("a conversation needs a non-empty messages list")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise ChatError("every message needs a role and a string content")
        if m.get("role") in RESERVED_ROLES:
            raise ChatError(f"role {m['role']!r} is reserved: no template defines it yet")
        if m.get("role") not in ROLES:
            raise ChatError(f"unknown role {m.get('role')!r}")
    sys_text, rest = None, messages
    if messages[0]["role"] == "system":
        sys_text, rest = messages[0]["content"], messages[1:]
    if not rest:
        raise ChatError("a system message needs a user message after it")
    for i, m in enumerate(rest):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] == "system":
            raise ChatError("a system message is allowed only as the first message")
        if m["role"] != want:
            raise ChatError(f"message {i + (sys_text is not None)} is {m['role']}, expected {want}")
        if "loss" in m and (m["role"] != "assistant" or not isinstance(m["loss"], bool)):
            raise ChatError("loss is a bool on assistant messages only")
    last = rest[-1]["role"]
    if prompt and last != "user":
        raise ChatError("a prompt conversation ends on a user message")
    if not prompt and last != "assistant":
        raise ChatError("a training conversation ends on an assistant message")
    return sys_text, rest


def _trim(text):
    b, e = 0, len(text)
    while b < e and ord(text[b]) in TRIM:
        b += 1
    while e > b and ord(text[e - 1]) in TRIM:
        e -= 1
    return text[b:e]


def render(rec, messages, generation_prompt=False, thinking=None):
    """The rendered text of a conversation under a template record (section 8.1.3)."""
    t = check_record(rec)
    sys_text, rest = structure(messages, generation_prompt)
    trim = t.get("content", "raw") == "trim"
    sysm = t.get("system", {"mode": "none"})
    reserved = [r for r in t.get("reserved", []) if r]

    def no_reserved(text):
        for r in reserved:
            if r in text:
                raise ChatError(f"content holds the template's reserved text {r!r}")
        return text

    def body(text):
        return no_reserved(_trim(text) if trim else text)

    out = [t.get("bos", "")]
    fold = None
    if sys_text is None and t.get("default_system"):
        sys_text = t["default_system"]
    if sys_text is not None:
        mode = sysm["mode"]
        if mode == "none":
            raise ChatError("this model's template has no system role")
        if mode == "turn":
            out += [sysm.get("open", ""), body(sys_text), sysm.get("close", "")]
        else:  # fold: the system text, untrimmed, plus join opens the first user message
            fold = no_reserved(sys_text) + sysm.get("join", "")
    last_user = max(i for i, m in enumerate(rest) if m["role"] == "user")
    for i, m in enumerate(rest):
        side = t[m["role"]]
        text = body(m["content"])
        if m["role"] == "user" and fold is not None:
            text, fold = fold + text, None
        if (m["role"] == "assistant" and t.get("past_assistant") == "strip_think" and i < last_user
                and "</think>" in text):
            text = text.split("</think>")[-1].lstrip("\n")
        out += [side["open"], text, side["close"]]
    if generation_prompt:
        g = t["generation_prompt"]
        if thinking is False and "generation_prompt_nothink" in t:
            g = t["generation_prompt_nothink"]
        out.append(g)
    elif t.get("past_assistant") == "strip_think":
        raise ChatError("a strip_think template renders prompt conversations only")
    return "".join(out)
