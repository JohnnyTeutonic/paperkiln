#!/usr/bin/env python3
"""Write a plain-language model card for a trained paperkiln run.

    python tools/model_card.py OUT_DIR [--spec spec.json]
    python tools/model_card.py OUT_DIR --probe [--mtstudio build/mtstudio]
    python tools/model_card.py OUT_DIR --probe --chat-url http://127.0.0.1:8123/chat

Writes OUT_DIR/card.md (for people) and OUT_DIR/card.json (the same
content as fields, for a web UI). The card answers three questions a
non-expert asks before chatting: what kind of model is this, how big and
how was it trained, and what should I expect it to say.

Sources, in order of authority: the run's own `events.jsonl` (the
"model" and "data" events are what the engine actually built), then
`result.json`, then the spec (OUT_DIR/spec.json unless --spec is given).
A spec emitted by `papers/fetch.py --emit-spec` names its paper in
`_comment`; the card says which mechanisms came from it.

Without --probe the "what to expect" paragraph is a conservative reading
of corpus, size and loss. With --probe the fixed prompt set in
tools/chat_probes.json is run greedily (deterministic) and the results
drive the wording. The sampler is pluggable: `MtstudioSampleBackend`
shells out to `mtstudio sample`; `HttpChatBackend` posts to a `/chat`
endpoint. Standard library only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "synthesis"))

from atlas_taxonomy import TAXONOMY, canonical  # noqa: E402
from vocab import COMPONENTS  # noqa: E402

CARD_SCHEMA = "paperkiln.model_card/1"
DEFAULT_PROBES = HERE / "chat_probes.json"

# Mirrors PRESETS in tools/mtstudio.cpp (name -> d, layers, heads, T),
# used only when neither events.jsonl nor result.json is present.
PRESET_DIMS = {
    "gpt2-nano": (128, 2, 4, 128), "llama-tiny": (128, 2, 4, 128),
    "gpt2-small": (256, 4, 8, 256), "kimi-tiny": (128, 2, 4, 128),
    "srd-tiny": (128, 2, 4, 128), "attnres-tiny": (128, 2, 4, 128),
    "swa-tiny": (128, 2, 4, 128),
}

# The reference block a mechanism is "non-default" against: the GPT-2
# decoder. Anything else gets a one-sentence explanation on the card.
BASELINE = {"attention": "exact", "norm": "layernorm", "residual": "residual",
            "position": "learned", "activation": "gelu", "optimizer": "adamw"}
# What the llama block fixes (atlas_taxonomy CONSTRAINTS).
LLAMA_BLOCK = {"position": "rope", "norm": "rmsnorm", "activation": "swiglu"}

# Slot -> the synthesis-vocabulary component it is an instance of; the
# component's "what" is the card's technical gloss for the slot.
SLOT_COMPONENT = {
    "attention": "attention-scores", "position": "positional-encoding",
    "norm": "normalisation", "residual": "residual-stream",
    "activation": "ffn", "optimizer": "optimiser-state",
    "layers": "layer-schedule", "heads": "multi-head", "T": "context-window",
}

SLOT_NAME = {"family": "Family", "attention": "Attention", "norm": "Normalisation",
             "residual": "Layer shortcuts", "position": "Word positions",
             "activation": "Feed-forward switch", "optimizer": "Optimiser"}

# Display names for engine values (card.json keeps the engine value).
VALUE_LABEL = {
    "exact": "full", "swa": "sliding window", "kimi": "linear (Kimi)",
    "srd": "surprise-routed", "attnres": "attention residuals",
    "rope": "rotary", "learned": "learned", "sinusoidal": "sinusoidal",
    "rmsnorm": "RMSNorm", "layernorm": "LayerNorm", "gelu": "GELU",
    "relu": "ReLU", "swiglu": "SwiGLU", "residual": "residual",
    "highway": "highway gate", "plain": "none", "adamw": "AdamW", "muon": "Muon",
}

# One sentence per alternative, for a reader who has never met the term.
PLAIN = {
    "family": {
        "gpt2": "a small GPT-2-style decoder, the classic layout behind the first widely known text generators",
        "llama": "a small Llama-style decoder, the layout used by many current open models, which comes with rotary positions, RMS normalisation and a gated feed-forward layer built in",
        "flex": "a configurable decoder whose parts were chosen one at a time, often to match a paper",
    },
    "attention": {
        "exact": "every word can look back at every earlier word it can see (full attention)",
        "swa": "each word only looks back a fixed distance{window}, which is cheaper but forgets earlier context",
        "kimi": "instead of looking back at every earlier word, the model keeps a running summary (linear attention); cheaper on long text, but the summary blurs details",
        "srd": "a small predictor guesses each word in advance; words it finds surprising get full attention and predictable ones take a cheaper summarising path",
        "attnres": "each layer can draw on the outputs of all earlier layers, not just the one before it",
    },
    "norm": {
        "layernorm": "each word's internal numbers are re-centred and rescaled to keep them in a stable range (layer normalisation)",
        "rmsnorm": "each word's internal numbers are rescaled by their overall size only (RMS normalisation), a slightly cheaper way of keeping them in a stable range",
    },
    "residual": {
        "residual": "each layer adds its result on top of what came in, so information can pass straight through (a residual shortcut)",
        "highway": "each layer has a learned gate deciding how much of its own result to add and how much of its input to pass through unchanged",
        "plain": "there are no shortcuts: each layer's output replaces its input entirely, which makes the model harder to train",
    },
    "position": {
        "learned": "the model learns one marker for each position in the text",
        "sinusoidal": "word positions are marked with fixed wave patterns rather than learned markers, so nothing about position has to be learnt",
        "rope": "word positions are encoded by rotating internal vectors (rotary positions), so the model sees how far apart two words are rather than where each one sits",
    },
    "activation": {
        "gelu": "the feed-forward layers use a smooth on/off switch (GELU) to decide which internal features fire",
        "relu": "the feed-forward layers use the simplest on/off switch (ReLU) to decide which internal features fire",
        "swiglu": "the feed-forward layers use a gate (SwiGLU) that lets one set of features switch another on or off, as most modern models do",
    },
    "optimizer": {
        "adamw": "trained with AdamW, the standard optimiser, which adjusts each weight with its own step size",
        "muon": "trained with Muon, an optimiser that keeps each weight matrix's updates balanced across directions (AdamW handles the remaining weights)",
    },
}

# Corpus kinds: (match substrings, display name, expectation, can, cannot)
CORPUS_KINDS = [
    ("tinychat", ("tinychat",), "TinyChat (synthetic small talk)",
     "Short small-talk replies for a handful of everyday topics: greetings, how are you, food, drinks, weather, hobbies, pets, outings, invitations and goodbyes. It knows no facts and often answers a slightly different question from the one asked.",
     ["reply to simple small talk in lower-case English"],
     ["answer factual questions", "do arithmetic", "remember what you told it earlier",
      "follow instructions or write anything longer than a sentence or two"]),
    ("tinystories-instruct", ("tinystories-instruct", "tinystories_instruct"), "TinyStories-Instruct",
     "Children's-story English in lower case that loosely follows a story prompt; a short attention span, and names and pronouns that wander.",
     ["continue a simple story"],
     ["hold a conversation", "answer factual questions", "do arithmetic", "keep track of a plot for long"]),
    ("tinystories", ("tinystories", "story"), "TinyStories (children's stories)",
     "Children's-story English in lower case: simple sentences about characters such as lily and tom, a short attention span, and names and pronouns that wander. It continues text rather than answering it.",
     ["continue a simple story opening such as 'once upon a time'"],
     ["hold a conversation or answer questions", "state facts", "do arithmetic",
      "keep track of who is who for more than a sentence or two"]),
    ("dialogue", ("dailydialog", "dialogmix", "empathetic"), "everyday dialogue",
     "A conversational tone in everyday English, but replies are often off topic or generic; the topics are broader than a model this size can keep straight.",
     ["sound conversational"],
     ["stay on topic reliably", "answer factual questions", "remember earlier turns"]),
    ("instructions", ("dolly",), "Dolly instructions",
     "Instruction-style answers that sound plausible for a few words and then drift; at this size nothing it states should be trusted.",
     ["imitate the shape of an answer"],
     ["give correct answers", "do arithmetic", "follow multi-step instructions"]),
    ("wiki", ("wikitext", "wiki"), "WikiText (encyclopaedia text)",
     "Encyclopaedia-style prose fragments. Names, dates and facts it produces are unreliable and often invented.",
     ["produce encyclopaedia-flavoured phrases"],
     ["state facts reliably", "hold a conversation", "do arithmetic"]),
]
GENERIC_KIND = ("generic", (), "an unlabelled text corpus",
                "Text in the style of its training corpus; how coherent it is depends on the corpus, which this card could not identify.",
                ["continue text in the style of its training data"],
                ["answer factual questions reliably", "do arithmetic", "follow instructions"])


# --------------------------------------------------------------------------
# reading a run
# --------------------------------------------------------------------------

def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_events(path: Path) -> dict:
    """The model/data/done events and the eval curve from events.jsonl."""
    out: dict = {"evals": []}
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue  # a run killed mid-write leaves a torn last line
            kind = ev.get("event")
            if kind in ("model", "data", "done", "start"):
                out[kind] = ev
            elif kind == "eval" and "val_loss" in ev:
                out["evals"].append((ev.get("step"), ev["val_loss"]))
    return out


def run_spec(spec: dict | None) -> dict:
    """A single-run spec; a sweep spec contributes its `base`."""
    if not spec:
        return {}
    return spec["base"] if "base" in spec and "arch" not in spec else spec


def paper_provenance(spec: dict | None) -> dict | None:
    """papers/fetch.py --emit-spec writes 'Generated by papers/fetch.py
    from arXiv:ID (Title). notes...' into _comment. A future explicit
    `provenance` block is honoured first."""
    if not spec:
        return None
    if isinstance(spec.get("provenance"), dict):
        return dict(spec["provenance"])
    c = spec.get("_comment", "")
    m = re.search(r"papers/fetch\.py from arXiv:(\S+?)(?: \((.*?)\))?\.(?:\s|$)", c)
    if not m:
        return None
    prov = {"arxiv_id": m.group(1), "title": m.group(2) or ""}
    na = re.search(r"not applied: (.*?)\. (?:Omitted|$)", c)
    prov["not_applied"] = [s.strip() for s in na.group(1).split(";")] if na else []
    prov["house_dims"] = "house dims" in c
    custom = run_spec(spec).get("arch", {}).get("custom", {})
    prov["mechanisms"] = {k: custom[k] for k in
                          ("attention", "norm", "activation", "position", "residual", "window")
                          if k in custom}
    return prov


def architecture(spec: dict, result: dict, events: dict) -> dict:
    """Resolve the built architecture: events 'model' > result > spec."""
    arch = run_spec(spec).get("arch", {})
    custom = arch.get("custom", {})
    preset = arch.get("preset", "gpt2-nano")
    d, L, H, T = PRESET_DIMS.get(preset, (None, None, None, None))
    a = {"family": "llama" if preset.startswith("llama") else "gpt2",
         "attention": "exact", "d": d, "layers": L, "heads": H, "T": T}
    for pfx in ("kimi", "srd", "swa", "attnres"):
        if preset.startswith(pfx):
            a["attention"] = pfx
    for k in ("d", "layers", "heads", "attention", "norm", "activation",
              "position", "residual", "window", "sinks", "d_ff"):
        if k in custom:
            a[k] = custom[k]
    data = run_spec(spec).get("data", {})
    if "T" in data:
        a["T"] = data["T"]
    if any(k in custom for k in ("norm", "activation", "position", "residual")) \
            and a.get("position") != "rope":
        a["family"] = "flex"
    a["optimizer"] = run_spec(spec).get("train", {}).get("optimizer", "adamw")
    for src in (result or {}, events.get("model", {})):
        for k in ("family", "attention", "d", "layers", "heads", "T", "norm",
                  "activation", "position", "residual", "window", "sinks",
                  "d_ff", "params", "optimizer"):
            if src.get(k) not in (None, ""):
                a[k] = src[k]
    if a["family"] == "llama":
        for k, v in LLAMA_BLOCK.items():
            a.setdefault(k, v)
    for k, v in BASELINE.items():
        a.setdefault(k, v)
    for slot in ("family", "attention", "norm", "residual", "position",
                 "activation", "optimizer"):
        a[slot] = canonical(slot, str(a[slot]).lower())
    return a


def corpus_kind(path: str, override: str = ""):
    if override:
        for row in CORPUS_KINDS + [GENERIC_KIND]:
            if row[0] == override:
                return row
        raise SystemExit(f"unknown --corpus-kind {override}")
    low = path.lower().replace("\\", "/")
    for row in CORPUS_KINDS:
        if any(m in low for m in row[1]):
            return row
    return GENERIC_KIND


def load_run(out_dir: Path, spec_path: Path | None = None) -> dict:
    spec_file = spec_path or out_dir / "spec.json"
    spec = _load_json(spec_file) if spec_file.exists() else None
    result = _load_json(out_dir / "result.json") or {}
    events = read_events(out_dir / "events.jsonl")
    if not spec and not result and "model" not in events:
        raise SystemExit(f"{out_dir}: no spec.json, result.json or events.jsonl")
    return {"out_dir": out_dir, "spec_path": spec_file if spec else None,
            "spec": spec, "result": result, "events": events}


# --------------------------------------------------------------------------
# card sections
# --------------------------------------------------------------------------

def _human(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} billion"
    if n >= 1e5:
        return f"{n / 1e6:.2f} million"
    if n >= 1e3:
        return f"{n / 1e3:.0f} thousand"
    return f"{n:.0f}"


def _bytes(n: int) -> str:
    for unit, k in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= k:
            return f"{n / k:.1f} {unit}"
    return f"{n} bytes"


def explain(slot: str, value: str, a: dict) -> str:
    text = PLAIN.get(slot, {}).get(value)
    if text is None:
        return f"{value} (no plain-language description recorded for this option)"
    if "{window}" in text:
        w = a.get("window")
        win = f" ({w} words)" if w else ""
        text = text.replace("{window}", win)
        if a.get("sinks"):
            n = a["sinks"]
            text += (f"; {n} anchor position{'s stay' if n != 1 else ' stays'}"
                     " visible to every word")
    return text


def what_it_is(a: dict, prov: dict | None) -> dict:
    comps = []
    for slot in ("family", "attention", "position", "norm", "activation",
                 "residual", "optimizer"):
        v = a[slot]
        fam_fixed = a["family"] == "llama" and LLAMA_BLOCK.get(slot) == v
        is_default = slot == "family" or BASELINE.get(slot) == v
        comp = COMPONENTS.get(SLOT_COMPONENT.get(slot, ""), {})
        known = v in (TAXONOMY.get(slot, {}).get("alternatives") or [])
        comps.append({
            "slot": slot, "name": SLOT_NAME[slot], "value": v,
            "label": VALUE_LABEL.get(v, v),
            "default": is_default, "part_of_family": fam_fixed,
            "recognised": known,
            "plain": explain(slot, v, a),
            "technical": comp.get("what", ""),
            "from_paper": bool(prov and prov.get("mechanisms", {}).get(slot) == v),
        })
    dims = []
    if a.get("layers"):
        dims.append(f"{a['layers']} layer{'s' if a['layers'] != 1 else ''} stacked one on another")
    if a.get("d"):
        dims.append(f"each word held as a list of {a['d']} numbers")
    if a.get("heads"):
        dims.append(f"{a['heads']} attention head{'s' if a['heads'] != 1 else ''} looking at the text in parallel")
    ctx = (f"It reads at most {a['T']} words at a time; anything earlier is invisible to it."
           if a.get("T") else "")
    fam_txt = PLAIN["family"].get(a["family"], a["family"])
    summary = (f"This is {fam_txt}: {', '.join(dims)}. {ctx}").replace(": .", ".").strip()
    unusual = [c for c in comps if not c["default"] and c["slot"] != "family"]
    if unusual:
        summary += (" It departs from the standard GPT-2 layout in "
                    f"{len(unusual)} place{'s' if len(unusual) != 1 else ''}, explained below.")
    else:
        summary += " Every other part is the standard GPT-2 layout."
    paper = None
    if prov:
        mech = [f"{k}={v}" for k, v in prov.get("mechanisms", {}).items()]
        title = prov.get("title") or ""
        paper = {
            "arxiv_id": prov.get("arxiv_id", ""), "title": title,
            "mechanisms": prov.get("mechanisms", {}),
            "not_applied": prov.get("not_applied", []),
            "house_dims": bool(prov.get("house_dims")),
            "text": (f"The design comes from arXiv:{prov.get('arxiv_id', '?')}"
                     + (f" ({title})" if title else "")
                     + (f". Mechanisms taken from the paper: {', '.join(mech)}." if mech
                        else ". No mechanism from the paper could be applied; engine defaults were used.")
                     + (" The size is paperkiln's small house size, not the paper's."
                        if prov.get("house_dims") else "")
                     + (f" Not applied: {'; '.join(prov['not_applied'])}." if prov.get("not_applied") else "")),
        }
    return {"summary": summary, "components": comps, "paper": paper}


def size_and_training(run: dict, a: dict, kind_override: str = "") -> dict:
    spec = run_spec(run["spec"])
    result, events = run["result"], run["events"]
    data = spec.get("data", {})
    corpus_path = data.get("corpus", "")
    kind = corpus_kind(corpus_path, kind_override)
    size = None
    if corpus_path and os.path.isfile(corpus_path):
        size = os.path.getsize(corpus_path)
    tokens = events.get("data", {}).get("tokens")
    vocab = events.get("data", {}).get("vocab") or events.get("model", {}).get("vocab") \
        or data.get("vocab_cap")
    best = result.get("best_val", events.get("done", {}).get("best_val"))
    if best is None and events["evals"]:
        best = min(v for _, v in events["evals"])
    steps = result.get("final_step", events.get("done", {}).get("final_step"))
    params = a.get("params")
    return {
        "params": params,
        "params_plain": _human(params) + " parameters" if params else "unknown",
        "corpus": {"kind": kind[0], "name": kind[2],
                   "path": corpus_path, "file": os.path.basename(corpus_path),
                   "bytes": size, "tokens": tokens},
        "vocab": vocab,
        "steps": steps,
        "steps_requested": result.get("steps_requested", spec.get("train", {}).get("steps")),
        "early_stopped": result.get("early_stopped", events.get("done", {}).get("early_stopped")),
        "best_val_loss": best,
        "perplexity": math.exp(best) if best is not None and best < 30 else None,
        "wall_seconds": result.get("wall_seconds", events.get("done", {}).get("wall_seconds")),
        "context": a.get("T"),
    }


def loss_reading(loss: float | None) -> str:
    if loss is None:
        return "No validation loss was recorded, so fluency cannot be estimated."
    if loss < 1.5:
        tier = "fluent within its narrow domain, often reproducing whole training phrases"
    elif loss < 2.5:
        tier = "mostly grammatical short phrases within its domain"
    elif loss < 3.5:
        tier = "recognisable phrases that drift within a sentence or two"
    elif loss < 5.0:
        tier = "short runs of plausible words; sentences rarely hold together"
    else:
        tier = "mostly jumbled words; training barely got started"
    return (f"Its best validation loss of {loss:.2f} suggests {tier} "
            "(a rough guide only: loss is not comparable across corpora or vocabularies).")


def expectation_heuristic(st: dict) -> dict:
    kind = corpus_kind("", st["corpus"]["kind"]) if st["corpus"]["kind"] != "generic" \
        else GENERIC_KIND
    parts = [kind[3], loss_reading(st["best_val_loss"])]
    if st["params"]:
        parts.append(f"With {_human(st['params'])} parameters it is tiny: the assistants "
                     "people chat with every day are at least a thousand times larger.")
    return {"source": "heuristic", "summary": " ".join(parts),
            "can": list(kind[4]), "cannot": list(kind[5])}


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

TURN_MARKERS = re.compile(r"\b(user|assistant):|<\|endoftext\|>")


class MtstudioSampleBackend:
    """Greedy replies from `mtstudio sample` (temp 0, top-k 1)."""

    name = "mtstudio-sample"

    def __init__(self, mtstudio: str, spec_path: str, tokens: int = 24,
                 timeout: float = 300.0):
        self.cmd = list(mtstudio) if isinstance(mtstudio, (list, tuple))             else shlex.split(mtstudio, posix=os.name != "nt")
        self.spec_path = spec_path
        self.tokens = tokens
        self.timeout = timeout

    @staticmethod
    def format_prompt(turns: list[dict]) -> str:
        p = " ".join(f"{t['role']}: {t['text']}" for t in turns)
        return p + " assistant:"

    def reply(self, turns: list[dict]) -> str:
        prompt = self.format_prompt(turns)
        argv = self.cmd + ["sample", self.spec_path, "--prompt", prompt,
                           "--tokens", str(self.tokens), "--temp", "0",
                           "--topk", "1", "--seed", "1"]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"mtstudio sample failed: {r.stderr.strip()[-300:]}")
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        text = lines[-1] if lines else ""
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        k = text.rfind("assistant:")
        return text[k + len("assistant:"):].strip() if k >= 0 else text.strip()


class HttpChatBackend:
    """POST {url} {user_input, history, temperature, top_k, max_new_tokens} -> {reply}.

    The `mtstudio chat` server's request shape (tools/mtstudio.cpp); top_k=1
    makes the decoding greedy, so the card is deterministic."""

    name = "http-chat"

    def __init__(self, url: str, tokens: int = 24, timeout: float = 120.0):
        self.url = url
        self.tokens = tokens
        self.timeout = timeout

    def reply(self, turns: list[dict]) -> str:
        history = [{"role": t["role"], "content": t["text"]} for t in turns[:-1]]
        body = json.dumps({"user_input": turns[-1]["text"], "history": history,
                           "temperature": 0, "top_k": 1, "max_new_tokens": self.tokens}).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return str(json.loads(resp.read().decode("utf-8")).get("reply", "")).strip()


def _choices(perplexity: float) -> str:
    """Perplexity as 'about how many equally likely words': one decimal below 10."""
    return f"{perplexity:.1f}" if perplexity < 10 else f"{perplexity:.0f}"


def load_probes(path: Path = DEFAULT_PROBES) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def turns_for(users: list[str], replies: list[str]) -> list[dict]:
    """User turns interleaved with the model's own earlier replies."""
    out: list[dict] = []
    for i, u in enumerate(users):
        out.append({"role": "user", "text": u})
        if i < len(replies):
            out.append({"role": "assistant", "text": replies[i]})
    return out


def cut_reply(raw: str) -> tuple[str, bool]:
    """The reply up to the next turn marker, and whether the model kept
    going into an invented next turn."""
    m = TURN_MARKERS.search(raw)
    return (raw[:m.start()].strip(), True) if m else (raw.strip(), False)


def first_sentence(text: str) -> str:
    m = re.search(r"[.!?]", text)
    return text[:m.end()].strip() if m else text.strip()


def classify(text: str, intents: dict) -> tuple[str, dict]:
    s = first_sentence(text).lower()
    scores = {name: sum(1 for p in spec["patterns"] if re.search(p, s))
              for name, spec in intents.items()}
    top = max(scores.values()) if scores else 0
    best = [k for k, v in scores.items() if v == top]
    if top == 0:
        return "unrelated", scores
    return (best[0] if len(best) == 1 else "ambiguous"), scores


def grade(probe: dict, reply: str, intents: dict) -> dict:
    graded, rambled = cut_reply(reply)
    if probe["tier"] == "off":
        ok = bool(re.search(probe["expect"], graded.lower()))
        got = probe["intent"] if ok else "miss"
    else:
        got, _ = classify(graded, intents)
        ok = got == probe["intent"]
    return {"id": probe["id"], "tier": probe["tier"], "intent": probe["intent"],
            "prompt": probe["turns"], "reply": graded, "raw": reply,
            "classified": got, "passed": ok, "rambled": rambled}


def run_probes(backend, probes: dict) -> dict:
    intents = probes["intents"]
    results = []
    for p in probes["probes"]:
        users, replies = p["turns"], []
        for i in range(len(users) - 1):  # earlier turns get the model's own replies
            replies.append(cut_reply(backend.reply(turns_for(users[: i + 1], replies)))[0])
        results.append(grade(p, backend.reply(turns_for(users, replies)), intents))
    per_intent: dict = {}
    for r in results:
        if r["tier"] == "off":
            continue
        d = per_intent.setdefault(r["intent"], {"label": intents[r["intent"]].get("label", r["intent"]),
                                                "passed": 0, "total": 0})
        d["total"] += 1
        d["passed"] += int(r["passed"])
    for d in per_intent.values():
        d["rate"] = d["passed"] / d["total"]

    def tally(pred):
        rs = [r for r in results if pred(r)]
        return {"passed": sum(r["passed"] for r in rs), "total": len(rs)}

    return {
        "backend": getattr(backend, "name", type(backend).__name__),
        "decoding": "greedy (temperature 0, top-k 1)",
        "probe_set_version": probes.get("version"),
        "per_intent": per_intent,
        "tier_a": tally(lambda r: r["tier"] == "A"),
        "tier_b": tally(lambda r: r["tier"] == "B"),
        "in_distribution": tally(lambda r: r["tier"] != "off"),
        "off_distribution": tally(lambda r: r["tier"] == "off"),
        "rambled": sum(r["rambled"] for r in results),
        "examples": pick_examples(results),
        "results": results,
    }


def pick_examples(results: list[dict]) -> list[dict]:
    """Three exchanges: a success, an in-distribution miss, an off-topic
    question. Deterministic (first in probe order)."""
    picks = []
    for pred in (lambda r: r["tier"] != "off" and r["passed"],
                 lambda r: r["tier"] != "off" and not r["passed"],
                 lambda r: r["tier"] == "off"):
        r = next((r for r in results if pred(r) and r not in picks), None)
        if r:
            picks.append(r)
    for r in results:  # top up to three if a category was empty
        if len(picks) >= 3:
            break
        if r not in picks:
            picks.append(r)
    return [{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"],
             "passed": r["passed"], "tier": r["tier"]} for r in picks]


def _frac_word(p: int, n: int) -> str:
    if n == 0:
        return "none"
    r = p / n
    if r >= 0.9:
        return "almost all"
    if r >= 0.65:
        return "most"
    if r >= 0.4:
        return "about half"
    if r > 0:
        return "a few"
    return "none"


def expectation_probe(st: dict, pr: dict) -> dict:
    ind, off = pr["in_distribution"], pr["off_distribution"]
    a, b = pr["tier_a"], pr["tier_b"]
    strong = [d["label"] for d in pr["per_intent"].values() if d["rate"] >= 0.99]
    weak = [d["label"] for d in pr["per_intent"].values() if d["rate"] == 0]
    word = _frac_word(ind["passed"], ind["total"])
    parts = [f"On {ind['total']} small-talk questions it answered {word} on topic "
             f"({ind['passed']} of {ind['total']}): {a['passed']} of {a['total']} when "
             f"phrased as in training, {b['passed']} of {b['total']} when reworded."]
    if b["total"] and a["total"] and b["passed"] / b["total"] < a["passed"] / a["total"] - 0.2:
        parts.append("Rewording a question makes it noticeably worse, so it is matching "
                     "familiar phrasing more than understanding.")
    if strong:
        parts.append(f"Reliable here: {', '.join(strong)}.")
    if weak:
        parts.append(f"Never on topic here: {', '.join(weak)}.")
    if off["passed"] == 0:
        parts.append(f"It failed all {off['total']} questions outside its training "
                     "(facts, maths, remembering what you said, instructions), as expected.")
    else:
        parts.append(f"It got {off['passed']} of {off['total']} questions outside its "
                     "training right; treat that as luck or a memorised phrase, not knowledge.")
    total = len(pr["results"])
    if pr["rambled"] > total / 3:
        parts.append(f"In {pr['rambled']} of {total} replies it carried on into an invented "
                     "next turn; a chat front end should cut the reply at the next speaker label.")
    if st["corpus"]["kind"] not in ("tinychat", "dialogue", "instructions"):
        parts.append("It was not trained on conversations, so low scores here are expected.")
    some = [d["label"] for d in pr["per_intent"].values() if 0 < d["rate"] < 0.99]
    can = []
    if strong:
        can.append("reply on topic to " + ", ".join(strong))
    if some:
        can.append("sometimes reply on topic to " + ", ".join(some))
    cannot = ["answer factual questions", "do arithmetic", "remember what you told it",
              "follow instructions"] if off["passed"] == 0 else \
        ["be relied on for facts, maths or memory, even where a probe happened to pass"]
    if weak:
        cannot.append("handle " + ", ".join(weak))
    return {"source": "probe", "summary": " ".join(parts), "can": can, "cannot": cannot}


# --------------------------------------------------------------------------
# assembly and rendering
# --------------------------------------------------------------------------

def build_card(run: dict, probe_result: dict | None = None, kind_override: str = "") -> dict:
    spec = run["spec"]
    a = architecture(spec, run["result"], run["events"])
    prov = paper_provenance(spec)
    name = run["result"].get("name") or run_spec(spec).get("name") \
        or run["events"].get("start", {}).get("name") or run["out_dir"].name
    st = size_and_training(run, a, kind_override)
    expect = expectation_probe(st, probe_result) if probe_result else expectation_heuristic(st)
    if probe_result:
        expect["summary"] = expectation_heuristic(st)["summary"].split(". ")[0] + ". " + expect["summary"]
    sources = [p for p in ("events.jsonl", "result.json") if (run["out_dir"] / p).exists()]
    if run["spec_path"]:
        sp = Path(run["spec_path"])
        sources.append(sp.name if sp.parent.resolve() == run["out_dir"].resolve() else str(sp))
    return {
        "schema": CARD_SCHEMA,
        "name": name,
        "generated_by": "tools/model_card.py",
        "sources": sources,
        "architecture": {k: a.get(k) for k in ("family", "attention", "d", "layers", "heads",
                                              "T", "d_ff", "norm", "activation", "position",
                                              "residual", "optimizer", "window", "sinks")
                         if a.get(k) is not None},
        "what_it_is": what_it_is(a, prov),
        "size_and_training": st,
        "what_to_expect": expect,
        "probe": probe_result,
    }


def render_md(card: dict) -> str:
    w, st, ex = card["what_it_is"], card["size_and_training"], card["what_to_expect"]
    L = [f"# Model card: {card['name']}", "",
         f"_Written by `{card['generated_by']}` from {', '.join(card['sources']) or 'the run directory'}._",
         "", "## What it is", "", w["summary"], ""]
    for c in w["components"]:
        if c["slot"] == "family":
            continue
        tag = " (standard)" if c["default"] else (" (part of the Llama layout)" if c["part_of_family"] else "")
        src = " From the paper." if c["from_paper"] else ""
        L.append(f"- **{c['name']}: {c['label']}**{tag}. {c['plain'][0].upper() + c['plain'][1:]}.{src}")
    if w["paper"]:
        L += ["", w["paper"]["text"]]
    L += ["", "## Size and training", ""]
    L.append(f"- **Size:** {st['params_plain']}.")
    cp = st["corpus"]
    extra = []
    if cp["file"]:
        extra.append(f"`{cp['file']}`")
    if cp["bytes"]:
        extra.append(_bytes(cp["bytes"]))
    if cp["tokens"]:
        extra.append(f"{cp['tokens']:,} training words")
    L.append(f"- **Trained on:** {cp['name']}" + (f" ({', '.join(extra)})" if extra else "") + ".")
    if st["vocab"]:
        L.append(f"- **Vocabulary:** {st['vocab']:,} words; anything else becomes an unknown-word marker.")
    if st["steps"] is not None:
        s = f"- **Training:** {st['steps']:,} steps"
        if st["steps_requested"] and st["steps_requested"] != st["steps"]:
            s += f" of {st['steps_requested']:,} requested"
        if st["early_stopped"]:
            s += ", stopped early when validation loss stopped improving"
        if st["wall_seconds"]:
            s += f", {st['wall_seconds'] / 60:.0f} minutes"
        L.append(s + ".")
    if st["best_val_loss"] is not None:
        s = f"- **Best validation loss:** {st['best_val_loss']:.3f}"
        if st["perplexity"]:
            s += (f". On text it has not seen, its next-word guess is about as uncertain "
                  f"as picking among {_choices(st['perplexity'])} equally likely words")
        L.append(s + ".")
    L += ["", "## What to expect", "", ex["summary"], ""]
    if ex["can"]:
        L.append("It can: " + "; ".join(ex["can"]) + ".")
    if ex["cannot"]:
        L.append("It cannot: " + "; ".join(ex["cannot"]) + ".")
    pr = card.get("probe")
    if pr:
        L += ["", "## Probe results", "",
              f"{len(pr['results'])} fixed prompts (probe set v{pr['probe_set_version']}), "
              f"{pr['decoding']}, via {pr['backend']}.", "",
              "| Topic | On topic |", "|---|---|"]
        for d in pr["per_intent"].values():
            L.append(f"| {d['label']} | {d['passed']}/{d['total']} |")
        o = pr["off_distribution"]
        L.append(f"| outside training (expected to fail) | {o['passed']}/{o['total']} |")
        L += ["", "Example exchanges:", ""]
        for e in pr["examples"]:
            mark = "on topic" if e["passed"] else ("expected failure" if e["tier"] == "off" else "off topic")
            L.append(f"- you: _{' / '.join(e['prompt'])}_  ")
            L.append(f"  model: _{e['reply'] or '(nothing)'}_ ({mark})")
    return "\n".join(L) + "\n"


def sampling_spec(run: dict, tmpdir: str) -> str:
    """A copy of the spec pointed at this out_dir and checkpoint name, so
    `mtstudio sample` loads OUT_DIR/<name>.safetensors wherever the run
    was originally written."""
    if not run["spec"]:
        raise SystemExit("--probe with the mtstudio backend needs a spec (--spec)")
    spec = copy.deepcopy(run_spec(run["spec"]))
    spec.pop("_comment", None)
    spec["out_dir"] = str(run["out_dir"].resolve())
    if run["result"].get("name"):
        spec["name"] = run["result"]["name"]
    ckpt = run["out_dir"] / f"{spec.get('name', 'run')}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    path = os.path.join(tmpdir, "sample_spec.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    return path


def write_card(out_dir: Path, card: dict) -> None:
    with open(out_dir / "card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
        fh.write("\n")
    with open(out_dir / "card.md", "w", encoding="utf-8") as fh:
        fh.write(render_md(card))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--spec", help="spec.json (default OUT_DIR/spec.json)")
    ap.add_argument("--probe", action="store_true", help="run tools/chat_probes.json")
    ap.add_argument("--mtstudio", default=os.environ.get("MTSTUDIO", "build/mtstudio"),
                    help="mtstudio command for the sample backend")
    ap.add_argument("--chat-url", help="use an HTTP /chat backend instead of mtstudio sample")
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--tokens", type=int, default=24, help="max reply tokens per probe")
    ap.add_argument("--corpus-kind", default="",
                    help="override corpus detection: " + ", ".join(k[0] for k in CORPUS_KINDS + [GENERIC_KIND]))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    run = load_run(out_dir, Path(args.spec) if args.spec else None)
    probe_result = None
    if args.probe:
        probes = load_probes(Path(args.probes))
        with tempfile.TemporaryDirectory() as tmp:
            if args.chat_url:
                backend = HttpChatBackend(args.chat_url, tokens=args.tokens)
            else:
                backend = MtstudioSampleBackend(args.mtstudio, sampling_spec(run, tmp),
                                                tokens=args.tokens)
            probe_result = run_probes(backend, probes)
    card = build_card(run, probe_result, args.corpus_kind)
    write_card(out_dir, card)
    print(f"wrote {out_dir / 'card.md'} and {out_dir / 'card.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
