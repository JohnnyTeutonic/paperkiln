#!/usr/bin/env python3
"""Conversation schema v2's evaluation sets and the tool-call evaluation (LOCAL_LLM_STACK.md section 3, point 3g).

    python tools/chat_v2_eval.py sets --glaive-unused DIR/glaive_unused.jsonl --out-dir corpus/eval
    python tools/chat_v2_eval.py tools --mtstudio BIN --model tag=spec.json --out DIR [--port 8096] [--max-new 96]

`sets` writes two frozen files from glaive-function-calling-v2 rows the v2 mix did not draw
(tools/chat_v2_data.py's glaive_unused.jsonl; Apache-2.0) and hand-written system prompts:

- tool_calls_v1.jsonl: 300 /chat requests, each a glaive conversation cut before its first assistant
  reply (system, tool definitions, history, the user message), with the gold reply's tool call (200 items)
  or none (100 items, the reply asks or answers without a call).
- chat_v2_prompts_v1.jsonl: the 40 v2 prompts of the greedy parity gate (ember vs mtstudio): 10 system +
  user, 10 system + history, 10 tool requests ending on a user message, 10 ending on a tool result.

`tools` serves a v2 model with `mtstudio chat --kv-cache`, sends tool_calls_v1 greedily and scores: a call
made (the reply parses to at least one call), the gold function's name, arguments that parse as JSON, and
arguments equal to the gold's as JSON values; on the no-call items, a call made is a false call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chat_template as CT  # noqa: E402

SYSTEMS = [
    ("You are a pirate. Answer every question in pirate speak.", "What is the capital of France?"),
    ("You are a helpful assistant. Answer in one short sentence.", "Why is the sky blue?"),
    ("You are a chef who only talks about food.", "What should I do this weekend?"),
    ("Reply only with yes or no.", "Is water wet?"),
    ("You are a patient maths tutor for children.", "What is seven times eight?"),
    ("You are Kiln, a friendly assistant made by paperkiln.", "Who are you?"),
    ("Always answer in French.", "How are you today?"),
    ("You are a poet. Answer in rhyme.", "Tell me about the sea."),
    ("Be brief.", "Explain photosynthesis."),
    ("You are a travel guide for Rome.", "Where should I eat dinner?"),
]
HISTORIES = [
    ("You are a helpful assistant.", [("Hi!", "Hello! How can I help you today?")], "Can you suggest a book?"),
    ("Be brief.", [("What is 2 + 2?", "4."), ("And 3 + 3?", "6.")], "And 4 + 4?"),
    ("You are a weather reporter.", [("Good morning", "Good morning! It is sunny today.")], "Will it rain tomorrow?"),
    ("You are a pirate.", [("Hello", "Ahoy, matey!")], "Where is your ship?"),
    ("Answer in one word.", [("Capital of Italy?", "Rome.")], "Capital of Spain?"),
    ("You are a chef.", [("I have eggs and cheese.", "You could make an omelette!")], "How long does it take?"),
    ("You are a gardener.", [("My tomatoes are yellow.", "They may need more nitrogen.")], "What fertiliser should I use?"),
    ("You are a kind teacher.", [("I failed my test.", "That is hard. What subject was it?")], "Maths."),
    ("Speak like a robot.", [("Hi", "GREETINGS HUMAN.")], "What do you eat?"),
    ("You are a museum guide.", [("Where is the Mona Lisa?", "In the Louvre, in Paris.")], "Who painted it?"),
]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def cut(row):
    """(request, gold) from a glaive conversation: the messages before its first assistant reply."""
    msgs = row["messages"]
    i = next(k for k, m in enumerate(msgs) if m["role"] == "assistant")
    before, reply = msgs[:i], msgs[i]
    if not before or before[-1]["role"] != "user":
        return None
    req = {"history": [m for m in before[:-1] if m["role"] != "system"], "user_input": before[-1]["content"]}
    if before[0]["role"] == "system":
        req["system"] = before[0]["content"]
    if row.get("tools"):
        req["tools"] = row["tools"]
    gold = reply["tool_calls"][0]["function"] if reply.get("tool_calls") else None
    return req, gold


def as_messages(req):
    m = [{"role": "system", "content": req["system"]}] if req.get("system") else []
    m += req.get("history", [])
    if "user_input" in req:
        m.append({"role": "user", "content": req["user_input"]})
    return m


def cmd_sets(a):
    rows = read_jsonl(a.glaive_unused)
    rng = random.Random(1)
    rng.shuffle(rows)
    pos, neg, after_tool = [], [], []
    for r in rows:
        c = cut(r)
        if c is None:
            continue
        req, gold = c
        CT.render(as_messages(req), CT.V2, generation_prompt=True, tools=req.get("tools"))
        if gold and len(pos) < 200:
            pos.append({"id": f"tool/call/{len(pos)}", "uri": r["uri"], "request": req, "gold": gold})
        elif gold is None and len(neg) < 100:
            neg.append({"id": f"tool/none/{len(neg)}", "uri": r["uri"], "request": req, "gold": None})
        # the parity set's tool-result prompts: the conversation through its first tool message
        msgs = r["messages"]
        t = next((k for k, m in enumerate(msgs) if m["role"] == "tool"), None)
        if t is not None and len(after_tool) < 10 and all(m["role"] != "tool" for m in msgs[t + 1:t + 1]):
            head = msgs[:t + 1]
            k = next(j for j in range(t, -1, -1) if head[j]["role"] == "assistant")
            if len(head[k].get("tool_calls", [])) == t - k:
                req2 = {"history": [m for m in head if m["role"] != "system"]}
                if head[0]["role"] == "system":
                    req2["system"] = head[0]["content"]
                if r.get("tools"):
                    req2["tools"] = r["tools"]
                CT.render(as_messages(req2), CT.V2, generation_prompt=True, tools=req2.get("tools"))
                after_tool.append({"id": f"v2/after_tool/{len(after_tool)}", "uri": r["uri"], "request": req2})
        if len(pos) == 200 and len(neg) == 100 and len(after_tool) == 10:
            break
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "tool_calls_v1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for x in pos + neg:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    par = []
    for k, (s, u) in enumerate(SYSTEMS):
        par.append({"id": f"v2/system/{k}", "request": {"system": s, "user_input": u}})
    for k, (s, h, u) in enumerate(HISTORIES):
        par.append({"id": f"v2/history/{k}", "request": {"system": s, "history": [{"user": x, "assistant": y}
                                                                                    for x, y in h], "user_input": u}})
    for k, x in enumerate(pos[:10]):
        par.append({"id": f"v2/tool_user/{k}", "uri": x["uri"], "request": x["request"]})
    par += after_tool
    for x in par:
        r = x["request"]
        hist = []
        for h in r.get("history", []):
            hist += [{"role": "user", "content": h["user"]}, {"role": "assistant", "content": h["assistant"]}] \
                if "user" in h else [h]
        m = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + hist + \
            ([{"role": "user", "content": r["user_input"]}] if "user_input" in r else [])
        x["prompt"] = CT.render(m, CT.V2, generation_prompt=True, tools=r.get("tools"))
    with open(os.path.join(a.out_dir, "chat_v2_prompts_v1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for x in par:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"tool_calls": len(pos), "no_call": len(neg), "parity_prompts": len(par)}))


def cmd_tools(a):
    import carry5_eval
    from iftiny_eval import post_json
    tag, spec = a.model.split("=", 1)
    items = read_jsonl(a.items)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{tag}_tool_calls.jsonl")
    done = {json.loads(x)["id"] for x in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    _, gguf, _ = carry5_eval.model_paths(spec)
    ns = argparse.Namespace(chat_url=None, port=a.port, gen="mtstudio", out=a.out, mtstudio=a.mtstudio,
                            no_kv_cache=False)
    srv = carry5_eval.ChatServer(ns, spec, gguf)
    t0 = time.time()
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for it in items:
                if it["id"] in done:
                    continue
                body = dict(it["request"], temperature=1.0, top_k=1, top_p=1.0, seed=7, max_new_tokens=a.max_new,
                            return_ids=True)
                x = post_json(srv.url + "/chat", body)
                f.write(json.dumps({"id": it["id"], "reply": x["reply"], "tool_calls": x.get("tool_calls", []),
                                    "stop": x["stop_reason"], "ids": x.get("ids")}, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        srv.close()
    got = {r["id"]: r for r in read_jsonl(path)}
    pos = [it for it in items if it["gold"]]
    neg = [it for it in items if not it["gold"]]

    def args_ok(c):
        try:
            json.loads(c["function"]["arguments"])
            return True
        except ValueError:
            return False

    def args_eq(c, g):
        try:
            return json.loads(c["function"]["arguments"]) == json.loads(g["arguments"])
        except ValueError:
            return False
    s = {"items": len(items), "seconds": round(time.time() - t0, 1),
         "call_made": sum(bool(got[i["id"]]["tool_calls"]) for i in pos),
         "name_correct": sum(bool(got[i["id"]]["tool_calls"]) and got[i["id"]]["tool_calls"][0]["function"]["name"]
                             == i["gold"]["name"] for i in pos),
         "arguments_json": sum(bool(got[i["id"]]["tool_calls"]) and args_ok(got[i["id"]]["tool_calls"][0]) for i in pos),
         "arguments_equal": sum(bool(got[i["id"]]["tool_calls"]) and args_eq(got[i["id"]]["tool_calls"][0], i["gold"])
                                for i in pos),
         "with_call_items": len(pos), "no_call_items": len(neg),
         "false_calls": sum(bool(got[i["id"]]["tool_calls"]) for i in neg),
         "call_token_unparsed": sum("<|tool_call|>" in got[i["id"]]["reply"] for i in items),
         "stops": {k: sum(r["stop"] == k for r in got.values()) for k in ("eos", "turn", "length")}}
    with open(os.path.join(a.out, f"{tag}_tool_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1)
    print(json.dumps({"tag": tag, **s}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--glaive-unused", required=True)
    s.add_argument("--out-dir", required=True)
    t = sub.add_parser("tools")
    t.add_argument("--mtstudio", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "corpus", "eval", "tool_calls_v1.jsonl"))
    t.add_argument("--out", required=True)
    t.add_argument("--port", type=int, default=8096)
    t.add_argument("--max-new", type=int, default=96)
    a = ap.parse_args(argv)
    {"sets": cmd_sets, "tools": cmd_tools}[a.cmd](a)


if __name__ == "__main__":
    main()
