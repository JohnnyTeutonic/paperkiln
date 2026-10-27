"""The chat profile's template rendering (IR_SPEC.md section 8.1), the reference for the conformance suite's
chat cases (conformance/chat/render.json and parse_reply.json) and the adapter methods render_chat and
parse_reply (section 11.3).

    check_record(record)                         -> record      ChatError for a field or value not implemented
    render(record, messages, generation_prompt=False, thinking=None, tools=None) -> str   ChatError when it cannot
    parse_reply(ids, call_id, decode)            -> {"content", "tool_calls"} or None    (section 8.1.6)

Standard library only. The same rules as paperkiln's tools/chat_template.py (render_pieces, parse_reply) and its
C++ twin include/microtorch/chat_template.hpp, which ember vendors: section 8.1.1's structure (8.1.6's under a
record with tool turns, paperkiln/2), then bos, each message as open + content + close (an assistant's tool
calls before its close), and the generation prompt.
"""
from __future__ import annotations

import json

ROLES = ("system", "user", "assistant")
TOOL_ROLES = ("tool",)
FIELDS = {"name", "version", "bos", "content", "system", "user", "assistant", "generation_prompt",
          "generation_prompt_nothink", "stop", "past_assistant", "default_system", "reserved", "markers",
          "match", "architectures", "tool", "tool_call", "tools"}
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
    sysm = rec.get("system", {"mode": "none"})
    if not isinstance(sysm, dict) or sysm.get("mode") not in ("fold", "turn", "none"):
        raise ChatError(f"chat template system mode {sysm.get('mode') if isinstance(sysm, dict) else sysm!r} "
                        "is not implemented")
    for side, keys in (("user", {"open", "close"}), ("assistant", {"open", "close"}), ("tool", {"open", "close"}),
                       ("tool_call", {"open", "join"}), ("tools", {"join", "sep"})):
        if side in rec and (not isinstance(rec[side], dict) or set(rec[side]) - keys):
            raise ChatError(f"chat template {side} takes {' and '.join(sorted(keys))} only")
    # Section 8.1.6: tool turns come as a pair, definitions need them, and both need a system turn.
    if ("tool" in rec) != ("tool_call" in rec):
        raise ChatError("chat template tool and tool_call come together")
    if "tools" in rec and "tool" not in rec:
        raise ChatError("chat template tools needs tool and tool_call")
    if "tool" in rec and sysm.get("mode") != "turn":
        raise ChatError("chat template tool turns need system mode turn")
    return rec


def _calls(m):
    """An assistant message's tool calls as [(id or None, name, arguments)] (section 8.1.6); [] for none."""
    tc = m.get("tool_calls")
    if tc is None:
        return []
    if not isinstance(tc, list) or not tc:
        raise ChatError("tool_calls is a non-empty list")
    out = []
    for c in tc:
        if not isinstance(c, dict) or not isinstance(c.get("function"), dict):
            raise ChatError("a tool call is {id?, type?: function, function: {name, arguments}}")
        if c.get("type", "function") != "function":
            raise ChatError(f"tool call type {c.get('type')!r} is not implemented (function)")
        f = c["function"]
        name, args = f.get("name"), f.get("arguments", "")
        if not isinstance(name, str) or not name or "\n" in name:
            raise ChatError("a tool call's name is a non-empty string without a newline")
        if not isinstance(args, str):
            raise ChatError("a tool call's arguments are a string (rendered verbatim)")
        if "id" in c and not isinstance(c["id"], str):
            raise ChatError("a tool call's id is a string")
        out.append((c.get("id"), name, args))
    return out


def structure(messages, prompt, tools_ok=False):
    """(system text or None, the other messages): section 8.1.1, or 8.1.6 when the record has tool turns
    (tools_ok). prompt: the conversation ends on a user message (a reply is wanted), or under tool turns on the
    tool message that answers an assistant message's last call; else on an assistant message."""
    if not isinstance(messages, list) or not messages:
        raise ChatError("a conversation needs a non-empty messages list")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise ChatError("every message needs a role and a string content")
        if m.get("role") in TOOL_ROLES and not tools_ok:
            raise ChatError(f"role {m['role']!r} is refused: this template has no tool turns")
        if m.get("role") not in ROLES + (TOOL_ROLES if tools_ok else ()):
            raise ChatError(f"unknown role {m.get('role')!r}")
        if "tool_calls" in m and (not tools_ok or m["role"] != "assistant"):
            raise ChatError("tool_calls belong on assistant messages of a template with tool turns")
        if "tool_call_id" in m and m["role"] != "tool":
            raise ChatError("tool_call_id belongs on tool messages")
    sys_text, rest = None, messages
    if messages[0]["role"] == "system":
        sys_text, rest = messages[0]["content"], messages[1:]
    if not rest:
        raise ChatError("a system message needs a user message after it")
    want, pending = "user", []
    for i, m in enumerate(rest):
        if m["role"] == "system":
            raise ChatError("a system message is allowed only as the first message")
        if m["role"] != want:
            raise ChatError(f"message {i + (sys_text is not None)} is {m['role']}, expected {want}")
        if "loss" in m and (m["role"] != "assistant" or not isinstance(m["loss"], bool)):
            raise ChatError("loss is a bool on assistant messages only")
        if m["role"] == "user":
            want = "assistant"
        elif m["role"] == "assistant":
            pending = _calls(m)
            want = "tool" if pending else "user"
        else:  # a tool message answers the next pending call, in order
            cid = pending.pop(0)[0]
            if "tool_call_id" in m and m["tool_call_id"] != cid:
                raise ChatError(f"tool_call_id {m['tool_call_id']!r} is not the call's id {cid!r}")
            want = "tool" if pending else "assistant"
    last = rest[-1]["role"]
    if prompt and not (last == "user" or (last == "tool" and not pending)):
        raise ChatError("a prompt conversation ends on a user message"
                        + (" or on the answer to an assistant message's last call" if tools_ok else ""))
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


def _fractional(x):
    """A JSON number written with a fraction or an exponent (json.loads gives a float for exactly those)."""
    if isinstance(x, float):
        return True
    if isinstance(x, dict):
        return any(_fractional(v) for v in x.values())
    if isinstance(x, list):
        return any(_fractional(v) for v in x)
    return False


def tool_text(d):
    """A tool definition's text (section 8.1.6): Python's json.dumps(d, ensure_ascii=False), HF's `tojson`."""
    if not isinstance(d, dict):
        raise ChatError("a tool definition is a JSON object")
    if _fractional(d):
        raise ChatError("a tool definition holds a number with a fraction or an exponent (not portable)")
    return json.dumps(d, ensure_ascii=False)


def render(rec, messages, generation_prompt=False, thinking=None, tools=None):
    """The rendered text of a conversation under a template record (sections 8.1.3 and 8.1.6)."""
    t = check_record(rec)
    tools_ok = "tool" in t
    if tools is not None and not isinstance(tools, list):
        raise ChatError("tools is a list of tool definitions")
    if tools and "tools" not in t:
        raise ChatError("this model's template has no tool definitions")
    sys_text, rest = structure(messages, generation_prompt, tools_ok)
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
    defs = [no_reserved(tool_text(d)) for d in (tools or [])]
    if defs:  # the system text, join, then the definitions joined by sep; the definitions alone without text
        tl = t["tools"]
        head = body(sys_text) if sys_text is not None else ""
        sys_text = (head + tl.get("join", "") if head else "") + tl.get("sep", "").join(defs)
    if sys_text is not None:
        mode = sysm["mode"]
        if mode == "none":
            raise ChatError("this model's template has no system role")
        if mode == "turn":
            out += [sysm.get("open", ""), sys_text if defs else body(sys_text), sysm.get("close", "")]
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
        out += [side["open"], text]
        if m["role"] == "assistant":
            for _cid, name, args in _calls(m):
                out += [t["tool_call"]["open"], no_reserved(name) + t["tool_call"].get("join", "") + no_reserved(args)]
        out.append(side["close"])
    if generation_prompt:
        g = t["generation_prompt"]
        if thinking is False and "generation_prompt_nothink" in t:
            g = t["generation_prompt_nothink"]
        out.append(g)
    elif t.get("past_assistant") == "strip_think":
        raise ChatError("a strip_think template renders prompt conversations only")
    return "".join(out)


def parse_reply(ids, call_id, decode):
    """Section 8.1.6: a generated reply (ids without the stop) split at every call_id, never on text; part 0 is
    the content, every later part name + "\\n" + arguments with a non-empty name. {"content", "tool_calls"}, or
    None when a part does not parse (the engine then returns the whole decoded reply as content)."""
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
